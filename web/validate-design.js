/**
 * validate-design.js
 * ------------------
 * Fold a finished design again with a model of the reader's choosing, and lay the result over it.
 *
 * 🔴 THE ONLY NUMBER A DESIGN RUN PRODUCES THAT IS NOT SELF-REPORTED. Protein
 * Hunter optimises a folding model's opinion of its own design - the objective
 * IS that model's confidence - so a high score says the model agrees with
 * itself. RFD4-Proteina is worse off: it generates a structure and scores
 * nothing at all. Re-folding the finished SEQUENCE with another model gives a
 * confidence nothing was optimised against and an RMSD to the backbone the
 * design is supposed to be.
 *
 * 🔴 ASKED FOR, NOT RUN BY EVERY DESIGN (2026-10-27). It ran after each run with a model picked in
 * step 1, which made a run slower whether or not anyone wanted the check and made the picture of
 * it a line of text. It is the page's Refold step now: a design, a model, a button - and the
 * refold drawn over the design.
 */
import { AF3_COUNTS, foldAf3, loadAf3Weights } from "./af3-model.js";
import { describeValidation, selfConsistency } from "../cpu/design/validate.js";
import { foldInputFromPdb, keptTemplates, refoldOverlay } from "../cpu/design/refold.js";

/**
 * The folding models a design can be refolded with, and a hunt can fold with: the AF3-graph
 * families, which share `foldAf3`. 🔴 design.html's two pickers offer exactly these keys
 * (test/design-validate-options.test.js) - an option outside them throws.
 */
export const FOLDERS = { af3: "af3", boltz2: "boltz2", protenix2: "protenix2", openbind0: "openbind0",
  opendde: "opendde" };

/**
 * Refold a design and say how far the result sits from it.
 *
 * @param {object} options
 * @param {string} options.pdb the design
 * @param {string|string[]} [options.chain] the designed chain(s): held out of the fit, and the
 *   RMSD taken over a single one on its own
 * @param {string} options.model a key of FOLDERS
 * @param {object} options.device
 * @param {number} [options.seed]
 * @param {(text: string) => void} [options.status]
 * @param {AbortSignal} [options.signal]
 * @returns {Promise<object>} the refold's confidence, its RMSD to the design, the refold in the
 *   design's frame and letters (`refold`) and how far each residue moved (`deviations` - see
 *   refoldOverlay)
 */
export async function refoldDesign(options) {
  const { pdb, chain, model, device, seed = 0, signal } = options;
  const status = options.status ?? (() => {});
  if (FOLDERS[model] === undefined) {
    throw new Error(`no folding model ${model} (have ${Object.keys(FOLDERS).join(", ")})`);
  }
  const input = foldInputFromPdb(pdb);
  const templates = keptTemplates(pdb, chain);
  status(`refolding with ${model}`);
  const weights = await loadAf3Weights(() => {}, FOLDERS[model]);
  const refolded = await foldAf3({
    ...input,
    templates,
    weights,
    device,
    signal,
    seed,
    calls: AF3_COUNTS.diffusion.preferred,
    onStatus: (text) => status(`refolding with ${model} · ${text}`),
    onProgress: () => {},
  });
  // 🔴 A TEMPLATE THAT DID NOT ARRIVE FOLDS AND SCORES ALL THE SAME, so ask the fold what it covered
  if ((refolded.templateCoverage ?? []).length !== templates.length) {
    throw new Error(`${templates.length} kept chains were templated and the fold took`
      + ` ${(refolded.templateCoverage ?? []).length}`);
  }
  const fitted = refoldOverlay(pdb, refolded.pdb, chain);
  const { rmsd, comparable, residues } = selfConsistency(pdb, fitted.refold,
    Array.isArray(chain) ? (chain.length === 1 ? chain[0] : undefined) : chain);
  const result = {
    model, confidence: refolded.confidence ?? {}, rmsd, comparable, residues,
    templated: templates.map((template) => template.chainId),
    refold: fitted.refold, deviations: fitted.deviations,
  };
  const templated = result.templated.length === 0 ? ""
    : ` (chain${result.templated.length === 1 ? "" : "s"} ${result.templated.join(", ")} templated)`;
  return { ...result, described: `${describeValidation(result)}${templated}` };
}
