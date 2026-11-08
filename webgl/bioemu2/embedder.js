/**
 * BioEmu2's input embedder in WebGL2: three windowed atom layers and a token
 * mean, giving `s_inputs` (T, 455).
 *
 *     c       = proj_atom_continuous([ref_pos | charge]) + element + four name chars
 *     pr      = windowed atom-pair geometry, then c -> p and a three-layer MLP
 *     q       = c; three times: q = conditionedLayer(q, conditioned on c)
 *     s_inputs = [relu(q) meaned over each token's atoms | res_type | profile
 *                 | deletion_mean | pocket]
 *
 * 🔴 THE HOST HALF IS `cpu/bioemu2/host.js`'s AND IS NOT RE-DERIVED HERE. The
 * atom features, the windowed pair geometry with its MLP, the per-layer window
 * bias and the 455-column assembly are per atom or per atom-window pair, have no
 * GPU in them, and are what `webgpu/bioemu2/` was already running on the host -
 * so they moved to `cpu/` rather than being written twice. What is left for this
 * file is the part that differs between backends: three conditioned windowed
 * atom layers.
 *
 * 🔴 AND THOSE LAYERS ARE THE DENOISER'S OWN, which is the whole saving:
 * `conditionedLayer` and `windowAttention` are gated against the f64 tape
 * reference inside `check-webgl-bioemu2-denoiser.js`, and the only thing new
 * here is that the window BIAS arrives ready-made - the embedder's is computed
 * once on the host, where the denoiser reads `ctx.p` every layer.
 *
 * 🔴 AND THE CONDITIONING IS `c`, HELD FIXED FOR ALL THREE LAYERS, while `q`
 * starts at `c` and moves. The same shape as ESMFold2's atom stack and AF3's,
 * and feeding each layer its own input is a different model that runs.
 */
import { BIOEMU2 } from "../../shared/bioemu2/weights.js";
import {
  atomFeatures, atomPairTail, normalizeWindowPair, sInputs, windowBias,
} from "../../cpu/bioemu2/host.js";
import { conditionedLayer, windowAttention } from "./denoiser.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{features: any, hostWeights: Record<string, Float32Array>,
 *   layers: {weights: any, normZ: Float32Array, projZ: Float32Array}[],
 *   tokens: number, epsilon?: number,
 *   onStage?: (stage: string, values: Float32Array) => void}} call
 *   `hostWeights` is the embedder's host group - the projections and tables
 *   `atomFeatures`, `atomPairTail` and `sInputs` read, under their own names -
 *   and `layers` carries each layer's DEVICE weights beside the two HOST tensors
 *   its window bias needs.
 * @returns {Float32Array} `tokens * 455`, on the host: every reader of
 *   `s_inputs` (the trunk's init, the denoiser's conditioning) takes it as one
 */
export function inputEmbedder(execution, call) {
  const { features: f, hostWeights, layers, tokens, epsilon = 1e-5 } = call;
  const channels = BIOEMU2.atom;
  const heads = BIOEMU2.atomHeads;
  const { c, pr, N: atoms } = atomFeatures(hostWeights, f);
  atomPairTail(hostWeights, c, pr, atoms);
  const prNorm = normalizeWindowPair(pr);
  call.onStage?.("c", c);

  /**
   * 🔴 THE HOST'S WINDOW BIAS IS HEAD-MAJOR AND THIS PORT'S ATTENTION READS
   * HEAD-MINOR. `windowBias` writes `(window, HEAD, query, key)` - which is what
   * the WGSL softmax indexes - while the denoiser's own bias comes out of
   * `pairBias` as `(row, head)`, head LAST, and one kernel reads both. Measured
   * before this permutation: relRMS 0.118 on `s_inputs` with the 71 featuriser
   * columns EXACT, which says the host half agreed and the layers did not.
   */
  const headMinor = (bias) => {
    const windows = atoms / BIOEMU2.windowQueries;
    const keys = BIOEMU2.windowKeys;
    const queries = BIOEMU2.windowQueries;
    const out = new Float32Array(bias.length);
    for (let window = 0; window < windows; window += 1) {
      for (let head = 0; head < heads; head += 1) {
        for (let query = 0; query < queries; query += 1) {
          for (let key = 0; key < keys; key += 1) {
            const from = ((window * heads + head) * queries + query) * keys + key;
            const to = ((window * queries + query) * keys + key) * heads + head;
            out[to] = bias[from];
          }
        }
      }
    }
    return out;
  };

  const conditioning = execution.upload(c, "bioemu2.ie.c");
  const atomPad = execution.upload(Float32Array.from(f.atom_pad_mask), "bioemu2.ie.pad");
  let q = execution.upload(c, "bioemu2.ie.q");
  for (const [index, layer] of layers.entries()) {
    // The bias, once a layer, on the host: LN(pr) through this layer's own
    // projection. `windowBias` takes the normalised pair so the norm runs once.
    const bias = execution.upload(
      headMinor(windowBias(pr, atoms, layer.normZ, layer.projZ, heads, prNorm)),
      `bioemu2.ie.bias.${index}`);
    const next = conditionedLayer(execution, {
      x: q, conditioning, rows: atoms, channels, heads, weights: layer.weights,
      epsilon,
      attend: (input) => windowAttention(execution, {
        x: input, bias, pad: atomPad, atoms, channels, heads,
        weights: layer.weights.attention, epsilon,
      }),
    });
    execution.release(bias);
    execution.release(q);
    q = next;
    call.onStage?.(`layer${index}`, Float32Array.from(execution.read(q)));
  }
  const values = Float32Array.from(execution.read(q));
  execution.release(q);
  execution.release(conditioning);
  execution.release(atomPad);
  return sInputs(values, f, hostWeights["atom_to_token_trans_0/weight"], tokens);
}
