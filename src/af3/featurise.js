/**
 * AF3's batch for a single protein chain, built from a sequence in JavaScript.
 *
 *     const batch = featuriseProtein("GWSTELEK...");
 *
 * This is the piece that turns "the model runs on the GPU against a batch
 * prepared elsewhere" into "the browser folds a sequence you type". AF3's own
 * featuriser is a 515 MB chemical component dictionary and a tokeniser, and
 * neither is going into a web page; what a protein chain actually reaches of
 * that dictionary is 21 components, baked into reference-conformers.js.
 *
 * Every array here is checked element-for-element against AF3's own batch for
 * 6MRR by tools/oracle/check_af3_featurise.js. Two are checked differently and
 * the difference is the point:
 *
 * 🔴 ref_pos CANNOT MATCH AND DOES NOT NEED TO. AF3 samples a fresh conformer
 * for every residue INSTANCE - fixed bond lengths and angles, random torsions -
 * so the 13 internal glutamates in a 6MRR batch have 13 different side chains.
 * A baked table gives them all the same one. Measured end to end, that moves
 * the trunk's pair representation by relRMS 2.7e-2 and the folded structure by
 * 0.01 A RMSD. The checker holds the bonded geometry to AF3 and lets the
 * torsions go.
 *
 * 🔴 THE MSA IS THE QUERY ALONE. A de novo design has no homologues, and the
 * dump this was built against ran num_msa=1. `msa`, `profile` and
 * `deletionMatrix` are all one row of the query, which is what AF3 produces for
 * a single-sequence input - not a stub. A real MSA changes only these three
 * arrays; nothing else here depends on depth.
 *
 * WHAT IS NOT HERE, deliberately: ligands, nucleic acids, more than one chain,
 * covalent bonds between chains, and templates. Each is a token type this
 * tokeniser does not create, and each would need its own CCD entries.
 */
import { conformerFor, aatypeFor } from "./reference-conformers.js";

const DENSE = 24;
const QUERIES = 32;
const KEYS = 128;
/** AF3's restype alphabet is 31 wide: 20 amino acids, UNK, and the nucleic acids. */
const RESTYPES = 31;

/**
 * A gather in AF3's form: indices into a flattened source, and a mask marking
 * which of them are real.
 */
function gather(count) {
  return { indices: new Int32Array(count), mask: new Float32Array(count), count };
}

/**
 * @param {string} sequence one-letter codes; anything unrecognised becomes UNK
 * @param {{msa?: number[][], deletionMatrix?: number[][]}} [options] extra MSA
 *   rows, each already tokenised to AF3 aatypes and the same length as the
 *   sequence. The query is always row zero and is prepended here.
 */
export function featuriseProtein(sequence, options = {}) {
  const tokens = sequence.length;
  if (tokens === 0) throw new Error("featuriseProtein: empty sequence");
  const subsets = Math.ceil((tokens * DENSE) / QUERIES);

  const aatype = new Int32Array(tokens);
  const refPos = new Float32Array(tokens * DENSE * 3);
  const refMask = new Float32Array(tokens * DENSE);
  const refElement = new Int32Array(tokens * DENSE);
  const refCharge = new Float32Array(tokens * DENSE);
  const refAtomNameChars = new Int32Array(tokens * DENSE * 4);
  const refSpaceUid = new Int32Array(tokens * DENSE);
  const residueIndex = new Int32Array(tokens);
  const tokenIndex = new Int32Array(tokens);
  const asymId = new Int32Array(tokens);
  const entityId = new Int32Array(tokens);
  const symId = new Int32Array(tokens);
  const seqMask = new Float32Array(tokens);

  // The flat index of every real atom, in token-then-slot order. AF3's query
  // layout is this list compacted, so it is built once and read four times.
  const realAtoms = [];
  const pseudoBetaSlot = new Int32Array(tokens).fill(-1);

  for (let token = 0; token < tokens; token += 1) {
    const code = sequence[token];
    aatype[token] = aatypeFor(code);
    residueIndex[token] = token + 1;
    tokenIndex[token] = token + 1;
    asymId[token] = 1;
    entityId[token] = 1;
    symId[token] = 1;
    seqMask[token] = 1;

    const atoms = conformerFor(code, token === tokens - 1);
    for (const [slot, name, element, charge, x, y, z] of atoms) {
      const flat = token * DENSE + slot;
      refMask[flat] = 1;
      refElement[flat] = element;
      refCharge[flat] = charge;
      refPos[flat * 3] = x;
      refPos[flat * 3 + 1] = y;
      refPos[flat * 3 + 2] = z;
      // AF3 stores an atom name as four characters offset by 32, zero-padded.
      for (let character = 0; character < 4; character += 1) {
        refAtomNameChars[flat * 4 + character] =
          character < name.length ? name.charCodeAt(character) - 32 : 0;
      }
      realAtoms.push(flat);
    }
    // 🔴 CB, FALLING BACK TO CA. Glycine has no CB, and taking its slot anyway
    // would read a masked slot sitting at the origin - the pseudo-beta feeds
    // the confidence head's distance bins, so every glycine would come out
    // tens of angstroms from everything.
    const beta = atoms.find((atom) => atom[1] === "CB")
      ?? atoms.find((atom) => atom[1] === "CA");
    if (beta) pseudoBetaSlot[token] = beta[0];
    // 🔴 ONE REFERENCE SPACE PER TOKEN. Every atom of a residue shares its uid,
    // and the atom encoder uses the uid only to decide whether a PAIR of atoms
    // may compare reference coordinates at all - which is what makes the random
    // per-residue orientation harmless.
    for (let slot = 0; slot < DENSE; slot += 1) refSpaceUid[token * DENSE + slot] = token;
  }

  const atomCount = realAtoms.length;

  // token_atoms_to_queries: query slot -> flat token-atom, the compacted list.
  const tokenAtomsToQueries = gather(subsets * QUERIES);
  for (let query = 0; query < atomCount; query += 1) {
    tokenAtomsToQueries.indices[query] = realAtoms[query];
    tokenAtomsToQueries.mask[query] = 1;
  }

  // queries_to_token_atoms: its inverse, laid out over the dense (token, slot)
  // grid, masked exactly where an atom is real.
  const queriesToTokenAtoms = gather(tokens * DENSE);
  for (let query = 0; query < atomCount; query += 1) {
    queriesToTokenAtoms.indices[realAtoms[query]] = query;
    queriesToTokenAtoms.mask[realAtoms[query]] = 1;
  }

  // queries_to_keys: a contiguous 128-wide window per subset of 32 queries,
  // centred on it and SHIFTED IN-BOUNDS at the ends rather than truncated -
  // every subset sees exactly 128 keys.
  //
  // 🔴 THE WINDOW IS CLAMPED AGAINST THE REAL ATOM COUNT, NOT subsets * 32.
  // The query layout is padded out to the dense grid (51 subsets for 574
  // atoms here), so clamping against the padded length would slide the last
  // windows off the end of the molecule and into masked slots.
  const queriesToKeys = gather(subsets * KEYS);
  const tokensToQueries = gather(subsets * QUERIES);
  const tokensToKeys = gather(subsets * KEYS);
  const tokenOfQuery = new Int32Array(atomCount);
  for (let query = 0; query < atomCount; query += 1) {
    tokenOfQuery[query] = (realAtoms[query] / DENSE) | 0;
  }
  const lastStart = Math.max(0, atomCount - KEYS);
  for (let subset = 0; subset < subsets; subset += 1) {
    const start = Math.min(Math.max(subset * QUERIES - (KEYS - QUERIES) / 2, 0), lastStart);
    for (let key = 0; key < KEYS; key += 1) {
      const query = start + key;
      const at = subset * KEYS + key;
      if (query >= atomCount) continue;
      queriesToKeys.indices[at] = query;
      queriesToKeys.mask[at] = 1;
      tokensToKeys.indices[at] = tokenOfQuery[query];
      tokensToKeys.mask[at] = 1;
    }
    for (let slot = 0; slot < QUERIES; slot += 1) {
      const query = subset * QUERIES + slot;
      if (query >= atomCount) continue;
      const at = subset * QUERIES + slot;
      tokensToQueries.indices[at] = tokenOfQuery[query];
      tokensToQueries.mask[at] = 1;
    }
  }

  const tokenAtomsToPseudoBeta = gather(tokens);
  for (let token = 0; token < tokens; token += 1) {
    tokenAtomsToPseudoBeta.indices[token] = token * DENSE + pseudoBetaSlot[token];
    tokenAtomsToPseudoBeta.mask[token] = pseudoBetaSlot[token] >= 0 ? 1 : 0;
  }

  // The MSA. Row zero is the query; anything the caller supplies follows.
  const extra = options.msa ?? [];
  const sequences = 1 + extra.length;
  const msa = new Int32Array(sequences * tokens);
  const msaMask = new Float32Array(sequences * tokens).fill(1);
  const deletionMatrix = new Float32Array(sequences * tokens);
  msa.set(aatype, 0);
  for (let row = 0; row < extra.length; row += 1) {
    msa.set(extra[row], (row + 1) * tokens);
    if (options.deletionMatrix?.[row]) {
      deletionMatrix.set(options.deletionMatrix[row], (row + 1) * tokens);
    }
  }

  // profile: the column-wise restype frequency over the MSA, and deletionMean
  // its mean deletion count. With the query alone the profile is a one-hot.
  const profile = new Float32Array(tokens * RESTYPES);
  const deletionMean = new Float32Array(tokens);
  for (let token = 0; token < tokens; token += 1) {
    for (let row = 0; row < sequences; row += 1) {
      const code = msa[row * tokens + token];
      if (code >= 0 && code < RESTYPES) profile[token * RESTYPES + code] += 1 / sequences;
      deletionMean[token] += deletionMatrix[row * tokens + token] / sequences;
    }
  }

  return {
    sequence, tokens, dense: DENSE, subsets, atomCount, sequences,
    shape: { tokens, dense: DENSE, subsets, queries: QUERIES, keys: KEYS },
    aatype, profile, deletionMean,
    msa, msaMask, deletionMatrix,
    residueIndex, tokenIndex, asymId, entityId, symId, seqMask,
    refPos, refMask, refElement, refCharge, refAtomNameChars, refSpaceUid,
    // AF3 keeps these separate and they are equal for a protein-only chain:
    // every atom the model predicts is one it has a reference conformer for.
    predDenseAtomMask: refMask,
    tokenAtomsToQueries, queriesToKeys, queriesToTokenAtoms,
    tokensToQueries, tokensToKeys, tokenAtomsToPseudoBeta,
    features: { residueIndex, tokenIndex, asymId, entityId, symId },
  };
}
