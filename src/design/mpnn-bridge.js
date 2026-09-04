/**
 * The join between the two models: an AF3 prediction in, a designed sequence
 * out.
 *
 * 🔴 THE INTERFACE IS A PDB STRING, WHICH IS NOT A COMPROMISE. `foldAf3`
 * already writes one - `toPdb` in src/af3/fold.js, one letter per chain, real
 * residue names - and `structureFromText` in the MPNN port already parses one
 * into exactly the `{X, mask, residueIdx, chainLabels}` that `Model.encode`
 * takes. Reaching into AF3's dense atom layout instead would mean a second
 * implementation of the residue/token distinction that toPdb already got
 * right, and the two would drift.
 *
 * 🔴 THE FIXED CHAINS' OWN SEQUENCE HAS TO BE HANDED TO THE SAMPLER. MPNN
 * decodes autoregressively over the whole structure and reads the amino acid
 * at every position it has already seen; `Model.sample` starts from
 * `opts.S ?? fill(20)`, which is all-X, and never decodes a position the chain
 * mask excludes. So without `S` the target chain is conditioned on as a
 * poly-unknown, and the binder is designed against a target the model cannot
 * read. `structure.S` is the parsed structure's real sequence and is what goes
 * in.
 *
 * The MPNN modules under ./mpnn/ are a mirror of a separate checkout; see
 * SOURCE.md there and tools/sync-mpnn.py.
 */
import { ALPHABET } from "./mpnn/constants.js";
import { Model } from "./mpnn/model.js";
import { structureFromText } from "./mpnn/pdb.js";
import { Weights } from "./mpnn/weights.js";
import { enableAcceleration } from "./mpnn/accel.js";
import { designBias } from "./sample-sequence.js";

/** Where the page serves the mirror from. */
export const DEFAULT_CHECKPOINT = "./web/public/mpnn/solublempnn_v_48_020.mpnn";
export const DEFAULT_KERNELS = "./web/vendor/mpnn/kernels.wasm";

/**
 * Read the checkpoint and install the SIMD kernel.
 *
 * 🔴 THE ACCELERATOR IS BEST EFFORT AND ITS ABSENCE IS NOT AN ERROR. It is a
 * 26 KB WebAssembly module holding one dense kernel; a runtime without SIMD
 * fails to validate it, `enableAcceleration` returns null, and the JS kernel
 * serves every call. The design step is seconds either way against a fold that
 * is seconds - this is not the loop's cost centre.
 *
 * @param {{url?: string, kernels?: string, signal?: AbortSignal,
 *          onProgress?: (received: number, total: number) => void}} [options]
 * @returns {Promise<{model: Model, accelerated: boolean}>}
 */
export async function loadDesigner(options = {}) {
  const weights = await Weights.fetch(options.url ?? DEFAULT_CHECKPOINT, {
    signal: options.signal, onProgress: options.onProgress,
  });
  const accelerator = await enableAcceleration(options.kernels ?? DEFAULT_KERNELS);
  return { model: new Model(weights), accelerated: accelerator !== null };
}

/**
 * Which residues of a parsed structure belong to a chain.
 *
 * @param {{chainIds: string[]}} structure
 * @param {string} chain a PDB chain letter
 * @returns {Float32Array} 1 on that chain, 0 elsewhere
 */
export function chainMaskFor(structure, chain) {
  const mask = new Float32Array(structure.chainIds.length);
  for (let index = 0; index < structure.chainIds.length; index += 1) {
    if (structure.chainIds[index] === chain) mask[index] = 1;
  }
  return mask;
}

/**
 * Redesign one chain of a predicted structure.
 *
 * @param {Model} model
 * @param {object} options
 * @param {string} options.pdb the prediction, as `foldAf3` writes it
 * @param {string} options.chain the chain to design; every other chain is held
 * @param {number} [options.temperature] the reference's 0.1
 * @param {string} [options.omit] letters this cycle may not choose
 * @param {number} [options.alanineBias] added to alanine's logit; negative
 *   discourages it. See alanineBias() in sample-sequence.js.
 * @param {() => number} [options.random] pass `uniformFrom(seed)` to make the
 *   design reproducible
 * @returns {{sequence: string, full: string, score: number,
 *            designed: number, structure: object}} `sequence` is the designed
 *   chain alone, which is what goes back into the fold; `full` is every chain,
 *   colon-separated, in the order the structure lists them.
 */
export function designChain(model, options) {
  const structure = structureFromText(options.pdb, { ligands: false });
  const chainMask = chainMaskFor(structure, options.chain);
  let designed = 0;
  for (const value of chainMask) designed += value;
  if (designed === 0) {
    throw new Error(`the prediction has no chain ${options.chain}`
      + ` (it has ${structure.chainList.join(", ")})`);
  }

  const result = model.sample(model.encode({
    X: structure.X,
    mask: structure.mask,
    residueIdx: structure.residueIdx,
    chainLabels: structure.chainLabels,
  }), {
    batch: 1,
    temperature: options.temperature ?? 0.1,
    rng: options.random ?? Math.random,
    // See the note at the top: the fixed chains are conditioned on their own
    // amino acids, not on the all-X default.
    S: Int32Array.from(structure.S),
    chainMask,
    bias: designBias(structure.chainIds.length, {
      omit: options.omit, alanineBias: options.alanineBias,
    }),
  });

  const letters = [...result.seq[0]];
  const chainLetters = [];
  for (let index = 0; index < chainMask.length; index += 1) {
    if (chainMask[index] === 1) chainLetters.push(letters[index]);
  }
  // 🔴 THE CHAINS COME BACK IN THE STRUCTURE'S ORDER, NOT THE REQUEST'S. The
  // reference splits its designer's output on ":" and indexes it by a chain
  // letter through a fixed A=0, B=1 table, which is right only while the PDB
  // lists chains alphabetically. `chainList` is the order the file actually
  // used, so the caller can line these up itself.
  const byChain = new Map(structure.chainList.map((id) => [id, []]));
  for (let index = 0; index < structure.chainIds.length; index += 1) {
    byChain.get(structure.chainIds[index]).push(letters[index]);
  }
  return {
    sequence: chainLetters.join(""),
    full: structure.chainList.map((id) => byChain.get(id).join("")).join(":"),
    score: result.score[0],
    designed,
    structure,
  };
}

/** MPNN's alphabet, re-exported so callers need not reach into the mirror. */
export { ALPHABET };
