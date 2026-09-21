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
 * 🔴 AND THIS FILE IS WHERE AF2's ORCHESTRATION SHOULD LAND. It is the only
 * one of the three whose fold still lives inside web/app.js - the other two
 * have `foldAf3` and `foldEsmfold2` behind them - so a runtime can build
 * AF2's prediction today and cannot yet produce the `result` to build it
 * from. tools/gpu/fold-af2.js folds headless but returns a BENCH REPORT:
 * timings, checksums and a mean pLDDT, with no pdb and no per-residue arrays.
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
