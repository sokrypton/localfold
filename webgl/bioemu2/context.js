/**
 * BioEmu2's static denoiser context in WebGL2: the three tensors a sampler step
 * reads and no step changes.
 *
 *     z  = pairwise_conditioner(concat(z_trunk, rel_pos))   (T^2, 256)
 *     q0 = atom_features(features)                          (N, 128)
 *     p  = the reference-geometry window pair, with the trunk's pair
 *          broadcast into it                                (NW, 32, 128, 16)
 *
 * 🔴 THE SPLIT BETWEEN HOST AND DEVICE IS THE REFERENCE'S, NOT A PREFERENCE.
 * `q0` and `p` are the atom featuriser's, which `cpu/bioemu2/host.js` holds for
 * both backends - a windowed pair of 128 keys a query over 16 channels is small
 * and branchy, and the WGSL port runs it on the host for the same reason. The
 * pairwise conditioner is T^2 x 512 of dense arithmetic and runs here.
 *
 * 🔴 AND `q0` IS THE ATOM CONDITIONING BEFORE `s_to_c`, WHILE THE PAIR IS AFTER
 * IT. The reference copies `c` for q0 and only then adds the trunk single's
 * projection into `c`, so the trunk single reaches the atom STREAM through the
 * token stream and the coupling, and reaches the atom PAIR directly through
 * `atomPairTail`'s c -> p projections. Adding it to q0 as well conforms in shape
 * and is a different model; the order below is `denoiserProgram`'s, line for
 * line, because it is a convention with no safe default.
 *
 * 🔴 AND `z_to_p` IS OF THE TRUNK'S PAIR, NOT OF THE CONDITIONED ONE. It reads
 * `z_trunk` - the input to the conditioner, not its output - and is broadcast
 * into the atom windows. Reading the conditioned pair instead compiles, runs,
 * and feeds the atom stream a pair that has been through three more layers.
 */
import { BIOEMU2 } from "../../shared/bioemu2/weights.js";
import {
  atomFeatures, atomPairTail, addTokenPairToWindows, linearRows, lnRows,
} from "../../cpu/bioemu2/host.js";
import { layerNorm } from "../kernels/layer-norm.js";
import { linear } from "../kernels/linear.js";
import { relativeEncoding } from "./sequence.js";
import { concatSource, sumSource, transitionNoNorm } from "./denoiser.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{features: any, tokens: number, atoms: number, zTrunk: Tensor,
 *   sTrunk: Float32Array, hostWeights: Record<string, Float32Array>,
 *   weights: {relative: {relPos: Tensor, relToken: Tensor, sameEntity: Tensor,
 *     relChain: Tensor}, zToP0Scale: Tensor, zToP0Offset: Tensor, zToP1: Tensor,
 *     pairNormScale: Tensor, pairNormOffset: Tensor, pairProject: Tensor,
 *     transitions: {normScale: Tensor, normOffset: Tensor, fc1: Tensor,
 *       fc2: Tensor, fc3: Tensor}[]}, epsilon?: number}} call
 *   `hostWeights` is the `atom_feature_embedder` group under its own names, and
 *   `sTrunk` the trunk's single as a HOST array, which is where `s_to_c` runs.
 * @returns {{z: Tensor, q0: Tensor, p: Tensor}} the caller's to release
 */
export function staticContext(execution, call) {
  const {
    features: f, tokens, atoms, zTrunk, sTrunk, hostWeights: h, weights: w,
    epsilon = 1e-5,
  } = call;
  const pairs = tokens * tokens;
  const single = BIOEMU2.single;
  const channels = BIOEMU2.pair;
  if (atoms % BIOEMU2.windowQueries !== 0) {
    throw new RangeError(
      `bioemu2 context: ${atoms} atoms is not a whole number of windows of`
      + ` ${BIOEMU2.windowQueries} - the window pair would be ragged`);
  }

  // z_to_p, first and on its own, because the host half below waits on it.
  const forP = layerNorm(execution, {
    x: zTrunk, weight: w.zToP0Scale, bias: w.zToP0Offset,
    rows: pairs, channels, epsilon,
  });
  const zp = linear(execution, {
    x: forP, weight: w.zToP1, rows: pairs,
    inChannels: channels, outChannels: BIOEMU2.atomPair, weightLayout: "out-major",
  });
  execution.release(forP);
  const zpHost = Float32Array.from(execution.read(zp));
  execution.release(zp);

  // --- the pairwise conditioner ------------------------------------------
  const relative = relativeEncoding(execution, {
    features: f, tokens, tables: w.relative,
  });
  const joined = execution.allocate(pairs * 2 * channels, "bioemu2.ctx.cat");
  execution.dispatch({
    source: concatSource(pairs, channels, channels),
    output: joined,
    inputs: { leftTex: zTrunk, rightTex: relative },
    label: `bioemu2.ctx.concat:${tokens}`,
  });
  execution.release(relative);
  const normalised = layerNorm(execution, {
    x: joined, weight: w.pairNormScale, bias: w.pairNormOffset,
    rows: pairs, channels: 2 * channels, epsilon,
  });
  execution.release(joined);
  let z = linear(execution, {
    x: normalised, weight: w.pairProject, rows: pairs,
    inChannels: 2 * channels, outChannels: channels, weightLayout: "out-major",
  });
  execution.release(normalised);
  for (const transition of w.transitions) {
    const inner = layerNorm(execution, {
      x: z, weight: transition.normScale, bias: transition.normOffset,
      rows: pairs, channels, epsilon,
    });
    const update = transitionNoNorm(execution, {
      x: inner, rows: pairs, channels, weights: transition,
    });
    execution.release(inner);
    const next = execution.allocate(pairs * channels, "bioemu2.ctx.z");
    execution.dispatch({
      source: sumSource(pairs * channels),
      output: next,
      inputs: { aTex: z, bTex: update },
      label: `bioemu2.ctx.z-residual:${tokens}`,
    });
    execution.release(update);
    execution.release(z);
    z = next;
  }

  // --- the atom half, on the host ----------------------------------------
  const { c, pr } = atomFeatures(h, f);
  const q0Host = Float32Array.from(c);
  const sToC = linearRows(
    lnRows(sTrunk, tokens, single, h["s_to_c_trans_0/weight"], h["s_to_c_trans_0/bias"]),
    tokens, single, h["s_to_c_trans_1/weight"], BIOEMU2.atom);
  for (let atom = 0; atom < atoms; atom += 1) {
    if (!f.atom_pad_mask[atom]) continue;
    const token = f.token_idx_at_atom[atom];
    for (let k = 0; k < BIOEMU2.atom; k += 1) {
      c[atom * BIOEMU2.atom + k] += sToC[token * BIOEMU2.atom + k];
    }
  }
  addTokenPairToWindows(pr, zpHost, tokens, f);
  atomPairTail(h, c, pr, atoms);

  return {
    z,
    q0: execution.upload(q0Host, "bioemu2.ctx.q0"),
    p: execution.upload(pr instanceof Float32Array ? pr : Float32Array.from(pr),
      "bioemu2.ctx.p"),
  };
}
