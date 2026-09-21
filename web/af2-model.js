/**
 * AlphaFold 2's side of the one arrangement all three graphs now share.
 *
 * 🔴 THREE GRAPHS, THREE ASSEMBLIES, ONE CONVENTION. `predictionFromAf3` is in
 * web/af3-model.js and `predictionFromEsmfold2` in web/esmfold2-model.js, and
 * all three take `(result, about)` and return the object every surface reads.
 * The reason is not tidiness: a runtime that folds WITHOUT a page has to
 * produce exactly what the page produces, and a second field-by-field rebuild
 * is how this project has lost `gpu`, `align`, `position_atoms`, `maps` and
 * `pae_n` - each in silence, each found by somebody whose download had a hole
 * in it.
 *
 * No DOM, no viewer, no downloads: those stay with the caller, which is what
 * makes this importable from a process.
 *
 * 🔴 AND THIS FILE IS WHERE AF2's ORCHESTRATION SHOULD LAND. `foldWithAf2` in
 * web/app.js is a function now rather than the tail of the submit handler,
 * but it is still a PAGE function, so a runtime can build AF2's prediction
 * and cannot produce the `result` to build it from. tools/gpu/fold-af2.js
 * folds headless and does NOT stand in for this: it returns a bench report -
 * timings, checksums, a mean pLDDT - with no pdb and no per-residue arrays.
 *
 * 🔴 AND ITS SEAM IS NOT THE OTHER TWO's, WHICH IS WHY IT IS RECORDED RATHER
 * THAN GUESSED AT NEXT TIME. ESMFold2 split into four clean bands, compute at
 * the ends and display in the middle. AF2's display sits INSIDE the decision
 * it depends on: the cache key is computed, the cache answers, and from that
 * one answer come `resume` (compute), `stem` (page), and `kept` - the
 * previous passes rebuilt as viewer frames (page, but built with
 * `predictionToPdb`, `alignedToFirstPass` and `paeMatrix`). A band cut puts
 * those three on the wrong sides of the line.
 *
 * The shape that works, and the reason for each half:
 *
 *   export function af2FoldKey({sequence, chainLengths, maxMsaSequences,
 *     maxExtraSequences, seed, tolerance, unified, family, alignment,
 *     template})            - ONE definition, exported, because the PAGE needs
 *                             the key BEFORE the fold: it looks its cache up,
 *                             builds `kept` and calls `openBlankFold`. Two
 *                             keys that must agree are two that can drift, and
 *                             a resume against a drifted key continues
 *                             somebody else's fold.
 *   export async function foldAf2Job({..., resume, firstPassLanded,
 *     onStatus, onProgress, onRecycle, onFirstPass})
 *                           - the sweep over models, the ranking, the answer.
 *                             Ten display touchpoints live in `onRecycle` and
 *                             the progress callback and become options; the
 *                             other 190 lines of the loop are compute.
 *                             Returns {alignedRecycles, best, bestIndex,
 *                             final, perModel, template, resumable, seconds}.
 *
 * 🔴 AND IT CANNOT BE VERIFIED WHERE IT IS WRITTEN. Nothing in the node lane
 * exercises an AF2 fold - `npm run test:gpu` cannot load Dawn on this
 * repository's own machine - so a break in the resume path, the five-model
 * sweep or the ranking is invisible until somebody folds AF2 in a browser
 * with a GPU. The gate is a Colab box: the same sequence through
 * `--runtime chrome` and through the runtime, compared field for field, which
 * is what the AF3 branch passed and what the ESMFold2 branch is still owed.
 */
import { confidenceJson, predictionToPdb } from "./prediction-results.js";

/**
 * A fold's RESULT turned into the prediction every surface reads.
 *
 * @param {object} best the winning pass - a structure from one pass beside
 *   another pass's pLDDT would be a file describing nothing that was folded
 * @param {{sequence: string, chains: string[], chainLengths: number[],
 *          alignment: string|null, stem: string, family: string,
 *          recycles: object[], bestPass: number, perModel?: object,
 *          context?: object}} about
 */
export function predictionFromAf2(best, about) {
  const { sequence, chains, chainLengths, alignment, stem, family } = about;
  return {
      stem,
      // The BEST pass, and its own scores with it - a structure from one pass
      // beside another pass's pLDDT would be a file that describes nothing that
      // was ever computed.
      pdb: predictionToPdb(sequence, best.structure, best.confidence.plddt, chainLengths),
      confidence: best.confidence,
      scores: confidenceJson(sequence, best.confidence),
      a3m: alignment,
      chains,
      chainLengths,
      recycles: about.recycles,
      bestPass: about.bestPass,
      contactSource: best.pass,
      // ...every model's own best, ranked, for the archive. Absent for a
      // single-model fold, which has nothing to rank.
      perModel: about.perModel,
      // ...the model that MADE the saved pass, which under a sweep is whichever
      // of the five won rather than the one the row resolved to.
      model: `AlphaFold 2 (${best.family ?? family})`,
      ...(about.context ?? {}),
    };
}
