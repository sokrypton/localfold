/**
 * What an ESMFold2 fold costs, by band, so a bar can move at the fold's speed.
 *
 * 🔴 A BAR NEEDS A COST MODEL AND A STATUS LINE NEEDS THREE WORDS. Reporting
 * every stage by name gave "recycle 0", "trunk 0", "recycle 1", "trunk 1" - and
 * a recycle is two milliseconds against a trunk loop's several seconds, so the
 * line flickered between two stages whose costs differ by a thousand. Reported
 * as exactly that. The recycle is not a phase; it is the seam between two trunk
 * passes.
 *
 * 🔴 THE CONSTANTS ARE MEASURED ON THIS MACHINE, WHICH IS WHAT THEY ARE FOR AND
 * ALSO THEIR LIMIT. They came from timing folds at 40, 150 and 300 residues -
 * `tools/gpu/fold-esmfold2.js` prints the per-stage table - and this machine
 * drifts up to 3.2x between runs, so they are right to about that. That is fine
 * for a BAR, where being approximately right is all that is asked, and it is
 * why the status line shows a percentage and not a time remaining: a number
 * that has to be right to be worth reading should not be built on this.
 *
 * | band | at 40 | at 150 | at 300 | the shape |
 * |---|---|---|---|---|
 * | language model | 2.4 s | 1.9 | 2.3 | flat - it is weight streaming, not n |
 * | trunk, 4 loops | 0.6 | 6.5 | 27.3 | 3.0e-4 * n^2, within 25% at all three |
 * | conditioning | 0.32 | 0.38 | 1.03 | a constant plus a small n^2 |
 * | a sampler step | 34 ms | 72 | 156 | 30 ms + 1.4e-3 * n^2 |
 *
 * 🔴 AND THE LANGUAGE MODEL IS FLAT IN THE SEQUENCE LENGTH, WHICH LOOKS WRONG
 * AND IS NOT. ESM-C's 36 blocks are streamed from a 224 MiB bundle and decoded
 * on the host; at these lengths that dominates its own arithmetic, so the band
 * is about the weights and not about the protein. It stops being flat somewhere
 * above the lengths measured here.
 */

/** ESM-C 600M at int3: the tower the constants here were fitted against. */
export const TOWER_MIB = 223.6;

/** Milliseconds, predicted. Units and time are the same thing in this plan. */
export function esmfold2Plan({ tokens, steps, loops = 4,
                              languageModelMiB = TOWER_MIB }) {
  const squared = tokens * tokens;
  // 🔴 MILLISECONDS THROUGHOUT. The first version wrote the trunk's constant in
  // SECONDS per n^2 and the rest in milliseconds, which put the trunk at 2% of
  // a 300-residue fold where it is 85%. The bar would have crawled through the
  // language model and then jumped.
  // 🔴 THE BAND IS THE TOWER'S BYTES, AND IT WAS A CONSTANT FITTED TO THE ONLY
  // TOWER THERE WAS. A second checkpoint and an off switch arrived after it and
  // were charged 2100 ms each: measured on a 76-mer, ESM-C 600M costs 2145 ms,
  // 300M costs 1355, and no language model at all costs 36. So the bar stood
  // still through two fifths of a fold's predicted time while nothing ran.
  //
  // 🔴 AND IT SCALES BY BYTES, NOT BY LAYERS, which is the same fact the note
  // above records - at these lengths the band is weight streaming and decoding
  // rather than arithmetic. 30 layers against 36 is 0.83 and would predict the
  // 300M tower 30% slow; 129.7 MiB against 223.6 is 0.58, and the measured
  // ratio is 0.63.
  //
  // 🔴 TWO POINTS AND TWO PARAMETERS, SO THE FIT IS EXACT BY CONSTRUCTION and
  // says only that the line passes through both towers this page can load. A
  // second run of the same three folds read 1972, 919 and 63 ms against the
  // fitted 2145, 1355 and 0 - so the constants carry this machine's own drift,
  // which the note at the top of this file puts at up to 3.2x. That is fine for
  // a BAR and is why the line reports a percentage rather than a time.
  //
  // 🔴 AND THE OFF ARM IS NOT FREE, IT IS 63 ms - the shim's zero state still
  // runs, because `shim(0)` is a fixed non-zero vector rather than nothing. It
  // is charged to the band after it, which is 2% out and not worth a term.
  const languageModel = languageModelMiB <= 0 ? 0 : 264 + 8.41 * languageModelMiB;
  const embedder = 150 + 0.003 * squared;
  const trunk = 0.075 * squared * loops;
  const conditioning = 307 + 0.008 * squared;
  const perStep = 32 + 1.38e-3 * squared;
  const sampler = perStep * steps;
  const total = languageModel + embedder + trunk + conditioning + sampler;
  return {
    languageModel, embedder, trunk, conditioning, perStep, sampler, total,
    // Where each band ends, so a partial one can be placed.
    languageModelEnd: languageModel,
    embedderEnd: languageModel + embedder,
    trunkEnd: languageModel + embedder + trunk,
    conditioningEnd: languageModel + embedder + trunk + conditioning,
  };
}

/**
 * The phase names a reader sees. Three, as AF3's line settled on.
 *
 * 🔴 NOT ONE PER STAGE. The fold has nine internal stages and a reader watching
 * a bar has no use for eight of them; what they want to know is whether the
 * slow part has started and how far in it is.
 */
export const ESMFOLD2_PHASES = {
  languageModel: "Language model",
  embedder: "Preparing",
  trunk: "Trunk",
  conditioning: "Preparing",
  sampler: "Folding",
};

/**
 * "Trunk 2/4", the way AF3's line writes its own pass.
 *
 * 🔴 AND NOT THE BLOCK NUMBER, WHICH IS AF3's RULE AND ITS REASON. Its comment:
 * the pairformer is the one stage that already reports - 48 times a pass - so
 * the bar under the line is visibly moving, and a third field would be the
 * "Trunk · pass 1 of 4 · pairformer block 23 of 48" that line was cut down
 * from. A number that changes 96 times sits next to two that barely move and
 * the eye tracks the one part that does not matter. The BAR is where block-by-
 * block belongs; the line says which pass.
 */
export const trunkPhase = (loop, loops) =>
  (loops > 1 ? `${ESMFOLD2_PHASES.trunk} ${loop + 1}/${loops}` : ESMFOLD2_PHASES.trunk);
