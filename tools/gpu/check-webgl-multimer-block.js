/**
 * AF2-MULTIMER's two block kinds in WebGL2 against the WGSL multimer block.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-multimer-block.js [--falsify]
 *
 * 🔴 ONE FLAG SEPARATES THE TWO GRAPHS AND IT MOVES A WHOLE SUB-LAYER.
 * `outerProductMeanFirst` runs the outer product mean BEFORE the MSA row
 * attention, where the monomer runs it after the MSA transition - so on the
 * multimer graph the pair receives the mean of the MSA as the block RECEIVED
 * it, and on the monomer graph the mean of an MSA the block has already updated
 * three times. Same weights, same kernels, different model.
 *
 * 🔴 AND IT HAS TO BE GATED IN BOTH BLOCK KINDS, BECAUSE THE REFERENCE SHIPPED
 * WITH IT WORKING IN ONE. `webgpu/af2/multimer/block.js` says so in its own
 * words at `encodeExtraMsaBlock`: "Only the main evoformer honoured the flag
 * before; this stack read it and did nothing, which left every multimer fold
 * running one of its two stacks in the monomer order." This port writes the
 * branch twice for the same reason the reference does - the two blocks do not
 * share their MSA half - so both are compared here.
 *
 * 🔴 AND THE FLAG IS PROVED NOT TO BE INERT BEFORE ANY AGREEMENT IS BELIEVED.
 * CLAUDE.md's rule: "A control arm that cannot vary the thing under test is not
 * a control." Two `order` arms run this port with the flag off against the same
 * weights and assert the answer MOVES - if it did not, both sides could be
 * ignoring the flag and every residual below would be a monomer comparison
 * wearing a multimer name.
 *
 * The bars are measured floors - one f32 step of jitter on every weight, plus
 * `execution.halfDither` inside every f16 rounding - for the reasons
 * `check-webgl-evoformer-stack.js` sets out: this chain quantises, so its
 * agreement floor is set by how many elements straddle an f16 boundary.
 */
import { EvoformerBlockGpu, encodeExtraMsaBlock } from "../../webgpu/af2/multimer/block.js";
import { WebGpuExecution } from "../../webgpu/runtime/execution.js";
import { setDeviceTuning, setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  evoformerBlock, extraMsaBlock, releaseEvoformerBlockWeights, uploadEvoformerBlockWeights,
} from "../../webgl/af2/evoformer/block.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { generator, syntheticActivations, syntheticBlockWeights } from "./af2-synthetic-block.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/**
 * The WGSL multimer's EXTRA block, which has no class of its own.
 *
 * `ExtraMsaStackGpu` lives in `webgpu/af2/evoformer/stack.js` and calls the
 * MONOMER `encodeExtraMsaBlock`, so it cannot reach the multimer order at all -
 * driving the exported encoder directly is the only way to compare that block,
 * and leaving it out would be leaving out the half the reference got wrong.
 */
async function runWgslExtraBlock(device, input) {
  const execution = new WebGpuExecution(device);
  try {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    const msa = execution.upload("multimer-extra.msa", input.msa, usage);
    const pair = execution.upload("multimer-extra.pair", input.pair, usage);
    const msaMask = execution.upload("multimer-extra.msa-mask", input.msaMask);
    const pairMask = execution.upload("multimer-extra.pair-mask", input.pairMask);
    const encoder = device.createCommandEncoder({ label: "multimer-extra-block" });
    device.pushErrorScope("validation");
    await encodeExtraMsaBlock(
      execution, encoder, input, input.weights, msa, pair, msaMask, pairMask);
    const msaReadback = execution.createReadback("multimer-extra.msa-readback", msa, encoder);
    const pairReadback = execution.createReadback("multimer-extra.pair-readback", pair, encoder);
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
    const [msaOut, pairOut] = await Promise.all([
      execution.mapFloat32(msaReadback), execution.mapFloat32(pairReadback),
    ]);
    return { msa: msaOut, pair: pairOut };
  } finally {
    execution.release();
  }
}

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  // 🔴 `--order=monomer` IS THE BISECT, NOT A MODE. It sets the flag OFF on both
  // sides while still driving the multimer MODULE, which separates "this port
  // gets the multimer order wrong" from "the multimer module differs from the
  // monomer one somewhere else" - the two are indistinguishable from one run.
  const first = option(args, "order", "multimer") !== "monomer";
  const shape = {
    sequences: Number(option(args, "sequences", "4")),
    length: Number(option(args, "length", "6")),
    cM: Number(option(args, "cm", "16")),
    cZ: Number(option(args, "cz", "16")),
    cOuter: Number(option(args, "outer", "8")),
    heads: Number(option(args, "heads", "4")),
    triangleHidden: Number(option(args, "tri-hidden", "16")),
    msaHidden: Number(option(args, "msa-hidden", "32")),
    pairHidden: Number(option(args, "pair-hidden", "32")),
  };
  const { sequences, length, cM, cZ } = shape;
  const mainWeights = syntheticBlockWeights({ ...shape, seed: 404 });
  const extraWeights = syntheticBlockWeights({ ...shape, seed: 505, global: true });
  const activations = syntheticActivations(shape, 606);

  const ULP = 1.2e-7;
  const jitter = generator(8675309);
  const jitterValues = (values) => Float32Array.from(
    values, (v) => v * (1 + ULP * (jitter() * 2 - 1)));
  /** One f32 step on every weight in a set, however deeply nested. */
  const jitterSet = (value) => {
    if (value instanceof Float32Array) return jitterValues(value);
    if (Array.isArray(value)) return value.map(jitterSet);
    if (typeof value === "number") return value;
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, jitterSet(v)]));
  };

  // 🔴 THREE PRECISION AXES, AND TWO OF THEM ARE THE MULTIMER MODULE'S OWN.
  // `setHalfPrecision` is the master switch for the arithmetic a device picks;
  // it does NOT reach `webgpu/af2/multimer/block.js`'s two deliberate f16
  // choices, which are device TUNING: the outer product mean's vector arithmetic
  // ("`opmVectorArithmetic: 'f32'` the control", in its own comment) and the
  // triangle contraction's (`triangleContractArithmetic`, defaulting to f16 with
  // a 32x32 tile). Left alone, this gate compares an f32 chain against an f16
  // one and reads 1e-4 with nothing wrong - which is exactly what it did,
  // through three rounds of finding one more axis.
  //
  // `triangleAccumulatePrecision` is set beside them and moves NOTHING on this
  // M5, to the digit: the reference gates it on `!matrixFits`, and at these
  // widths the matrix path fits, so the f16 accumulators are never reached. It
  // stays because the axis is real on a device where they are, and it is named
  // here so that nobody reads it as one of the three that mattered.
  setDeviceTuning(device, {
    allowHostWeightPacking: true,
    opmVectorArithmetic: "f32",
    triangleContractArithmetic: "f32",
    triangleAccumulatePrecision: "f32",
  });
  setHalfPrecision(device, false);
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const upload = (values, label) => execution.upload(values, label);
  // The control skews half of each MSA row's channels, not all of them: the MSA
  // meets a LayerNorm first in every sub-layer that reads it, and a whole-row
  // affine change is invisible to one.
  const msaTensor = upload(falsify
    ? Float32Array.from(activations.msa, (v, i) => (i % cM < cM / 2 ? v * 1.01 + 0.01 : v))
    : activations.msa, "multimer.msa");
  const pairTensor = upload(activations.pair, "multimer.pair");
  const msaMask = upload(activations.msaMask, "multimer.msa-mask");
  const pairMask = upload(activations.pairMask, "multimer.pair-mask");

  /** One of this port's blocks, with the flag and the dither as parameters. */
  const run = (block, weights, { first, dither }) => {
    const uploaded = uploadEvoformerBlockWeights(execution, weights);
    execution.halfDither = dither;
    const result = block(execution, {
      ...shape, msa: msaTensor, pair: pairTensor, msaMask, pairMask,
      weights: uploaded, outerProductMeanFirst: first,
      // 🔴 THE MULTIMER'S TRANSITIONS KEEP THEIR HIDDEN ROW IN f32 WHERE THE
      // MONOMER'S ROUND IT TO f16. `webgpu/af2/multimer/block.js` passes
      // `"f32"` at the seam where `webgpu/af2/evoformer/block.js` computes
      // `hiddenChannels % 4 === 0 ? "f16" : "f32"`. It is not switchable on
      // either side, so it is a property of the graph.
      transitionPrecision: "f32",
      // 🔴 AND THE MULTIMER'S ATTENTION TARGETS ARE ALL f32 TOO, in its own
      // words: "AND THE MULTIMER'S TARGETS ARE ALL f32, which is the SIMPLER of
      // the two". The monomer's attention stores its normalised input, its
      // projections and its output in f16 with no switch; the multimer's local
      // `encodeAttention` never grew a `projectedStorage` at all. So the two
      // graphs differ in FOUR precision decisions and none of them is a knob.
      msaAttentionPrecision: "f32",
      triangleAttentionPrecision: "f32",
    });
    execution.halfDither = 0;
    const values = {
      msa: Float32Array.from(execution.read(result.msa)),
      pair: Float32Array.from(execution.read(result.pair)),
    };
    execution.release(result.msa);
    execution.release(result.pair);
    releaseEvoformerBlockWeights(execution, uploaded);
    return values;
  };

  const ours = run(evoformerBlock, mainWeights, { first, dither: 0 });
  const oursOtherOrder = run(evoformerBlock, mainWeights, { first: !first, dither: 0 });
  const oursJittered = run(evoformerBlock, jitterSet(mainWeights), { first, dither: 0 });
  const oursDithered = run(evoformerBlock, mainWeights, { first, dither: ULP });
  const oursExtra = run(extraMsaBlock, extraWeights, { first, dither: 0 });
  const oursExtraOtherOrder = run(extraMsaBlock, extraWeights, { first: !first, dither: 0 });
  const oursExtraJittered = run(
    extraMsaBlock, jitterSet(extraWeights), { first, dither: 0 });
  const oursExtraDithered = run(extraMsaBlock, extraWeights, { first, dither: ULP });
  for (const tensor of [msaTensor, pairTensor, msaMask, pairMask]) execution.release(tensor);

  const reference = { ...shape, outerProductMeanFirst: first,
    msa: activations.msa, pair: activations.pair,
    msaMask: activations.msaMask, pairMask: activations.pairMask };
  const theirs = await new EvoformerBlockGpu(device).run(
    { ...reference, weights: mainWeights });
  const theirsExtra = await runWgslExtraBlock(device, { ...reference, weights: extraWeights });

  const override = option(args, "bar", null);
  const rows = [];
  const failures = [];
  const arm = (name, got, want, floor) => {
    const bar = override !== null
      ? Number(override)
      : Number(Math.max(2e-7, floor * 3).toPrecision(3));
    const relative = relativeRms(got, want, name);
    const ok = relative <= bar;
    rows.push({
      arm: name,
      relativeRms: Number(relative.toPrecision(4)),
      oneUlpFloor: Number(floor.toPrecision(3)),
      bar,
      ok,
    });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };
  /** @type {[string, any, any, any, any][]} */
  const COMPARISONS = [
    ["msa", ours, theirs, oursJittered, oursDithered],
    ["pair", ours, theirs, oursJittered, oursDithered],
    ["extra.msa", oursExtra, theirsExtra, oursExtraJittered, oursExtraDithered],
    ["extra.pair", oursExtra, theirsExtra, oursExtraJittered, oursExtraDithered],
  ];
  for (const [label, mine, theirSide, shifted, dithered] of COMPARISONS) {
    const key = label.replace("extra.", "");
    const floor = Math.max(
      relativeRms(shifted[key], mine[key], `floor.jitter.${label}`),
      relativeRms(dithered[key], mine[key], `floor.dither.${label}`));
    arm(`${label} vs WGSL multimer block`, mine[key], theirSide[key], floor);
  }

  // 🔴 THE FLAG MUST MOVE THE ANSWER, OR THE FOUR ARMS ABOVE ARE A MONOMER
  // COMPARISON IN MULTIMER CLOTHES. The pair is where the outer product mean
  // lands, so the pair is where this is asserted; the MSA moves too, through
  // the row attention's pair bias, which is why both are reported.
  const orderMoves = {
    "main pair": Number(relativeRms(
      oursOtherOrder.pair, ours.pair, "order.pair").toPrecision(3)),
    "main msa": Number(relativeRms(
      oursOtherOrder.msa, ours.msa, "order.msa").toPrecision(3)),
    "extra pair": Number(relativeRms(
      oursExtraOtherOrder.pair, oursExtra.pair, "order.extra.pair").toPrecision(3)),
    "extra msa": Number(relativeRms(
      oursExtraOtherOrder.msa, oursExtra.msa, "order.extra.msa").toPrecision(3)),
  };
  const ORDER_FLOOR = 1e-3;
  const inert = Object.entries(orderMoves)
    .filter(([, value]) => value < ORDER_FLOOR)
    .map(([name, value]) => `${name} moved only ${value}`);
  if (inert.length > 0) {
    failures.push(`the order flag is INERT: ${inert.join("; ")}`);
  }

  const red = rows.filter((row) => !row.ok).length;
  return {
    shape,
    falsify,
    order: first ? "multimer (outer product mean first)" : "monomer (bisect)",
    barRule: override !== null ? `--bar=${override}` : "max(2e-7, oneUlpFloor * 3)",
    arms: rows,
    orderMoves,
    orderFloor: ORDER_FLOOR,
    memory: execution.stats(),
    verdict: falsify
      ? (red === rows.length
        ? `ok: all ${rows.length} arms go red on a 1% perturbation of the MSA`
        : `VACUOUS: only ${red} of ${rows.length} arms noticed the MSA moving`)
      : (failures.length === 0
        ? `ok: ${rows.length} arms at their measured floors, and the order flag moves all four`
        : `FAILED: ${failures.join("; ")}`),
  };
}
