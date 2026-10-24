/**
 * validate-design.js
 * ------------------
 * Fold a finished design with a model that had no hand in making it.
 *
 * 🔴 THE ONLY NUMBER A DESIGN RUN PRODUCES THAT IS NOT SELF-REPORTED. Protein
 * Hunter optimises a folding model's opinion of its own design - the objective
 * IS that model's confidence - so a high score says the model agrees with
 * itself. RFD4-Proteina is worse off: it generates a structure and scores
 * nothing at all. Re-folding the finished SEQUENCE with another model gives a
 * confidence nothing was optimised against and an RMSD to the backbone the
 * design is supposed to be.
 *
 * 🔴 AND IT IS ONE PATH FOR BOTH METHODS. This was inside `hunt()`, so the
 * generate path had a `Validate with` control that silently did nothing - which
 * this repository holds to be worse than an absent one. The only thing the two
 * methods disagree about is where the sequence comes from, so that is the
 * argument and everything else is shared.
 */
import { AF3_COUNTS, foldAf3, loadAf3Weights } from "./af3-model.js";
import { describeValidation, selfConsistency } from "../cpu/design/validate.js";
import { THREE_TO_ONE } from "../cpu/design/mpnn/constants.js";

/** The folding models a design can be checked against. */
export const FOLDERS = { af3: "af3", openbind0: "openbind0", opendde: "opendde" };

/**
 * The sequence a structure actually contains, chains joined the way a fold
 * wants them.
 *
 * 🔴 READ OFF THE STRUCTURE RATHER THAN TRACKED ALONGSIDE IT. A generation
 * never had a sequence to begin with - RFD4 produces coordinates and the
 * residue identities come out of its decoder - so the structure is the only
 * place the answer exists. Reading it here also means the validation fold is
 * checking what was actually produced rather than what the page believes was.
 *
 * @param {string} pdb
 * @returns {string} chains joined by ":", which is how this page's folders read
 *   a complex
 */
export function sequenceFromPdb(pdb) {
  const chains = new Map();
  let previous = null;
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM")) continue;
    // Chain, residue number and insertion code together identify a residue; a
    // HETATM is a ligand and has no place in a sequence.
    const key = line.slice(21, 27);
    if (key === previous) continue;
    previous = key;
    const chain = line[21];
    const letter = THREE_TO_ONE[line.slice(17, 20).trim()] ?? "X";
    chains.set(chain, (chains.get(chain) ?? "") + letter);
  }
  return [...chains.values()].join(":");
}

/**
 * Re-fold a design and say how far the result sits from it.
 *
 * @param {object} options
 * @param {string} options.pdb the design
 * @param {string} [options.sequence] what to fold; read from `pdb` when absent
 * @param {string} [options.chain] the designed chain, scored on its own
 * @param {string} options.model a key of FOLDERS
 * @param {object} options.device
 * @param {number} [options.seed]
 * @param {number} [options.steps]
 * @param {(text: string) => void} [options.status]
 * @param {AbortSignal} [options.signal]
 * @returns {Promise<object|null>} null when no model was asked for
 */
export async function validateDesign(options) {
  const { pdb, chain, model, device, seed = 0, steps, signal } = options;
  const status = options.status ?? (() => {});
  if (model === undefined || model === "" || FOLDERS[model] === undefined) return null;

  const sequence = options.sequence ?? sequenceFromPdb(pdb);
  if (sequence.replace(/[:X]/g, "").length === 0) {
    return { model, failed: "the design has no sequence to re-fold" };
  }

  try {
    status(`validating with ${model}`);
    const weights = await loadAf3Weights(() => {}, FOLDERS[model]);
    const refolded = await foldAf3({
      sequence,
      weights,
      device,
      signal,
      // 🔴 A DIFFERENT SEED FROM THE RUN'S. Re-folding with the same one invites
      // the two to share whatever the seed decided, which is the opposite of an
      // independent check.
      seed: seed + 1,
      calls: steps ?? AF3_COUNTS.diffusion.preferred,
      onStatus: (text) => status(`validating with ${model} · ${text}`),
      onProgress: () => {},
    });
    const { rmsd, comparable, residues } = selfConsistency(pdb, refolded.pdb, chain);
    const validation = {
      model, confidence: refolded.confidence ?? {}, rmsd, comparable, residues,
      pdb: refolded.pdb,
    };
    return { ...validation, described: describeValidation(validation) };
  } catch (error) {
    // 🔴 A FAILED VALIDATION DOES NOT FAIL THE RUN. The design is already made;
    // what is missing is the check, and saying so beats throwing the result
    // away.
    const message = error instanceof Error ? error.message : String(error);
    return { model, failed: message, described: `could not validate with ${model}: ${message}` };
  }
}
