/**
 * An A3M alignment, in the codes AF3's MSA one-hot expects.
 *
 *     const { msa, deletionMatrix } = af3MsaFromA3m(text, { maxSequences: 512 });
 *     featuriseProtein(sequence, { msa, deletionMatrix });
 *
 * This is the whole of AF3's MSA path. It is short because AF3's is short: the
 * data pipeline hands the model aligned rows and deletion counts, and the model
 * takes the first `num_msa` of them. Everything AlphaFold 2 does between those
 * two points - clustering to 508 centres, a separate extra-MSA stack of 1024
 * more rows, Gumbel masking, per-recycle resampling - AF3 does not do at all.
 *
 * 🔴 GAP IS 21, NOT 31. AF3's MSA alphabet is
 * `PROTEIN_TYPES_WITH_UNKNOWN + (GAP,) + NUCLEIC_TYPES_WITH_2_UNKS`, so the gap
 * sits between the amino acids and the nucleotides rather than at the end of a
 * 32-wide one-hot. Putting it at 31 type-checks, folds, and quietly tells the
 * model that every gap is an unknown nucleotide. The table below is transcribed
 * from `_PROTEIN_TO_ID` in AF3's `data/msa_features.py`, aliases included.
 *
 * 🔴 THE DELETION COUNTS STAY RAW. AF2's featuriser squashes them on the way in
 * (`atan(n/3) * 2/pi`); AF3's does not, because its embedder does the squashing
 * itself - `msaFeatures()` emits both a clipped and an arctan channel from the
 * same integer. Passing pre-squashed values here applies the transform twice
 * and costs the model the distinction between one deletion and thirty.
 *
 * 🔴 ROWS ARE TAKEN IN ORDER, NOT SAMPLED. AF3's `truncate_msa_batch` is
 * `jnp.arange(num_msa)` - the first N rows, whatever they are. That is only a
 * good alignment because the pipeline upstream sorted them; MMseqs2 returns
 * hits in decreasing similarity, so the first N are the N most similar. A
 * shuffle here would be a different model input, not a fairer sample.
 */
import { parseA3m } from "../input/a3m.js";

/**
 * AF3's protein MSA alphabet, `_PROTEIN_TO_ID`.
 *
 * The 20 amino acids are in AF2's order, so a code is interchangeable with an
 * AF2 aatype up to 19. Past that they diverge: X is 20 in both, but AF3 puts
 * the gap at 21 where AF2 has no gap code at all.
 */
export const AF3_MSA_CODES = Object.freeze({
  A: 0, R: 1, N: 2, D: 3, C: 4, Q: 5, E: 6, G: 7, H: 8, I: 9,
  L: 10, K: 11, M: 12, F: 13, P: 14, S: 15, T: 16, W: 17, Y: 18, V: 19,
  X: 20,
  "-": 21,
  // The ambiguity codes, resolved as AF3 resolves them: B and Z to the acid
  // rather than the amide, U (selenocysteine) to cysteine, J and O to unknown.
  B: 3, Z: 6, U: 4, J: 20, O: 20,
});

/** The gap, which is neither an amino acid nor a nucleotide. */
export const AF3_MSA_GAP = 21;

/**
 * Turn A3M text into AF3's MSA rows.
 *
 * 🔴 AF3's MSA IS TWO BLOCKS, NOT ONE. A chain carries a `paired_msa` and an
 * `unpaired_msa`, and the model's `msa` is the first followed by the second.
 * The paired block's rows line up ACROSS chains - row s is one species in every
 * chain - which is what lets the model read coevolution between chains; the
 * unpaired block is block-diagonal, each chain's homologs against gaps
 * everywhere else. A merged single alignment can only be the unpaired block.
 *
 * 🔴 AN ABSENT PAIRED BLOCK IS THE QUERY, NOT NOTHING (`Msa.sequences or
 * [query_sequence]`), and AF3 keeps the unpaired block's own query row as well.
 * So a monomer folded against a 32-row A3M reaches the model as 33 rows whose
 * first two are identical. Dropping the duplicate looks tidier, reads better,
 * and is a different model input. `featuriseProtein` writes the first of the
 * two - it is the paired block's only row - and this returns everything after.
 *
 * 🔴 AND THE PROFILE IS OVER THE UNPAIRED BLOCK ALONE. AF3 computes it in the
 * data pipeline, per chain, from the unpaired alignment, before the paired
 * rows are prepended - so `unpairedFrom` is returned rather than left to be
 * guessed from the row count. Averaging the whole array instead double-counts
 * the query in every column.
 *
 * A complex needs no special case here: the rows of a merged complex A3M are
 * already the full token length, with gaps where a chain has no homolog. Which
 * merge produced them - paired or block-diagonal - is the caller's decision,
 * and is what the two arguments distinguish.
 *
 * @param {{paired?: string|null, unpaired?: string|null}|string} alignment A3M
 *   text for each block; a bare string is the unpaired block, which is what a
 *   monomer has.
 * @param {{maxSequences?: number}} [options] rows the MODEL sees, counting the
 *   query row prepended downstream; default 512. AF3's own cap is
 *   `num_msa: 1024`.
 * @returns {{msa: Int32Array[], deletionMatrix: Float32Array[], depth: number,
 *            unpairedFrom: number, length: number}}
 */
export function af3MsaFromA3m(alignment, options = {}) {
  const texts = typeof alignment === "string" ? { unpaired: alignment } : (alignment ?? {});
  const maxSequences = options.maxSequences ?? 512;

  const parse = (text) => {
    if (text === undefined || text === null || text.trim() === "") return null;
    return parseA3m(text);
  };
  const paired = parse(texts.paired);
  const unpaired = parse(texts.unpaired);
  // 🔴 THE PROFILE IS COMPUTED BEFORE DEDUPLICATION, over the chain's FULL
  // unpaired alignment. AF3's features.py calls get_profile_features on every
  // chain and only then runs msa_pairing.deduplicate_unpaired_sequences, so the
  // rows the dedup removes still counted towards the profile. Computing it from
  // what survives is the natural reading and it is a different feature - on a
  // target with good pairing most of the unpaired block is removed, so most of
  // the profile would be missing from it.
  const unpairedProfile = parse(texts.unpairedProfile) ?? unpaired;

  // 🔴 THE PAIRED BLOCK GETS AT MOST HALF THE BUDGET, and the unpaired block
  // gets what is left - `max_paired_sequences = msa_size // 2` in AF3's
  // features.py, then `unpaired_crop = total - paired_crop`. Letting the paired
  // block take the whole cap is the natural way to write this loop and it is
  // wrong in a way only a homo-oligomer shows: a deep paired alignment fills
  // every slot, the unpaired block is dropped entirely, and the profile - which
  // is over the unpaired block - has nothing left to be computed from.
  //
  // 🔴 AND THE PAIRED BLOCK IS ONE ROW TALLER THAN ITS A3M. The pairing keeps
  // row 0 as the query and then emits the species-aligned rows, which begin at
  // the A3M's own query again - the same duplication the unpaired block has. So
  // the block AF3 crops is `depth + 1`, and the rows returned here are all of
  // the A3M's, not all-but-the-first. Skipping one looks right, matches on a
  // monomer (where there is no paired block at all), and is short by a row on
  // every complex.
  const pairedBlock = paired === null ? 1 : paired.depth + 1;
  const pairedCrop = Math.min(pairedBlock, Math.max(1, Math.floor(maxSequences / 2)));
  const unpairedCrop = unpaired === null
    ? 0 : Math.min(unpaired.depth, maxSequences - pairedCrop);

  const msa = [];
  const deletionMatrix = [];
  let length = 0;
  const append = (parsed, from, upTo) => {
    if (parsed === null) return;
    length = parsed.length;
    for (let row = from; row < upTo; row += 1) {
      const aligned = parsed.sequences[row];
      const codes = new Int32Array(parsed.length);
      for (let column = 0; column < parsed.length; column += 1) {
        // An unlisted character cannot reach here - parseA3m rejects anything
        // outside its alphabet - so the fallback is unknown rather than an error.
        codes[column] = AF3_MSA_CODES[aligned[column]] ?? AF3_MSA_CODES.X;
      }
      msa.push(codes);
      deletionMatrix.push(Float32Array.from(parsed.deletionMatrix[row]));
    }
  };

  // The paired block contributes its rows AFTER the first: row zero is the
  // query, and featuriseProtein has already written it from the sequence.
  append(paired, 0, pairedCrop - 1);
  // +1 because unpairedFrom indexes the full array, whose row zero is the query.
  const unpairedFrom = msa.length + 1;
  // 🔴 AND THE UNPAIRED BLOCK SKIPS ITS OWN QUERY WHEN THERE IS A PAIRED ONE.
  // Every A3M repeats the query as its first row, and the merged MSA already
  // opens with it - AF3 reaches the same place by deduplicating, since a paired
  // alignment always contains the query. Keeping it sends the model the query
  // twice and leaves the batch one row taller than AF3's, which is exactly how
  // it was found. With no paired block the unpaired query IS row zero's source
  // and must stay.
  //
  // The crop counts ROWS, not indices, so skipping the query moves the end too:
  // the unpaired block still contributes `unpairedCrop` rows and still spends
  // exactly its share of the budget.
  const unpairedStart = paired === null ? 0 : 1;
  append(unpaired, unpairedStart,
    unpaired === null ? 0 : Math.min(unpaired.depth, unpairedCrop + unpairedStart));

  // The profile's own rows: the whole unpaired block, query included, before
  // any deduplication - see above. Kept separate from `msa` rather than
  // recomputed by the caller, because the two row sets differ only in ways this
  // function knows about.
  const profileMsa = [];
  const profileDeletionMatrix = [];
  {
    const into = (parsed, from, upTo) => {
      if (parsed === null) return;
      for (let row = from; row < upTo; row += 1) {
        const aligned = parsed.sequences[row];
        const codes = new Int32Array(parsed.length);
        for (let column = 0; column < parsed.length; column += 1) {
          codes[column] = AF3_MSA_CODES[aligned[column]] ?? AF3_MSA_CODES.X;
        }
        profileMsa.push(codes);
        profileDeletionMatrix.push(Float32Array.from(parsed.deletionMatrix[row]));
      }
    };
    // 🔴 UNCROPPED, TOO. get_profile_features runs at features.py:543 and the
    // crop only at 576, so the profile sees the whole alignment however few
    // rows the MSA itself is allowed. Cropping it here is invisible on a
    // monomer with a shallow alignment and wrong on every deep one.
    into(unpairedProfile, 0, unpairedProfile === null ? 0 : unpairedProfile.depth);
  }

  return {
    msa,
    deletionMatrix,
    profileMsa,
    profileDeletionMatrix,
    depth: msa.length + 1,
    // With no unpaired rows the profile falls back to the query, which is the
    // single-sequence case and is what AF3 does with an empty MSA.
    unpairedFrom: unpairedCrop === 0 ? 0 : unpairedFrom,
    length,
  };
}
