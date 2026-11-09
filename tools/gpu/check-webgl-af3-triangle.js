/**
 * AF3's triangle multiplication in WebGL2, against TWO references.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-triangle.js [--falsify]
 *
 * `cpu/af3/trunk/pairformer.js`'s `triangleMultiplication` is the f64 reference
 * and `af3TriangleMultiplication` the WGSL one, both in this page. Both
 * directions, because they contract different axes and the incoming one is
 * where this port's AF2 kernel was once the pair-TRANSPOSE of itself: it
 * measured relRMS 1.3 and the CPU reference it had been checked against shared
 * the mistake, because that reference had been transcribed from the shader.
 *
 * 🔴 FOUR CONVENTIONS, EVERY ONE OF WHICH RETURNS A PLAUSIBLE TENSOR WHEN
 * WRONG:
 *
 *   - the projection split is INTERLEAVED: channel c's two halves are outputs
 *     `2c` and `2c + 1`, not `c` and `c + channels`. Its twin in the transition
 *     is BLOCKED. "Getting either backwards costs no error and no NaN. It
 *     permutes channels and returns a plausible tensor."
 *   - the centre norm normalises over the CHANNEL axis of a channel-major
 *     array. "Doing it over i or j instead still runs and still returns finite
 *     numbers."
 *   - both LayerNorms take AF3's FAST variance.
 *   - and rosettafold3 DIVIDES the product by the sequence length, which looks
 *     scale-invariant and is not: "dividing by L shrinks the variance by L^2 -
 *     4624 at 68 tokens - and once the pair track decays with depth, `var`
 *     stops dominating `var + eps`". The `--over-length` arm runs it, against a
 *     reference told the same dialect.
 *
 * 🔴 AND THE ARMS SWEEP THE LENGTH, because the contraction's k axis is the
 * protein and a kernel that is right at one length can be wrong at another -
 * which is how the WGSL triangle's odd-width bug was found, by a sweep rather
 * than by a fold.
 *
 * 🔴 AND THE LENGTHS ARE CHOSEN SO THAT EVERY PATH RUNS, WHICH 5 AND 11 ALONE
 * DID NOT DO. The WebGL2 contraction picks its kernel by the length: the
 * pair-BLOCKED one (two i and four j into eight render targets) and the
 * coalesced `fetch4` along k both need `L % 4 === 0`, and the transposed
 * incoming path needs it too. At 5 and 11 not one of the three is reachable, so
 * every arm this checker ran took the general one-float-at-a-time loop - the
 * fallback for an odd width - and the three kernels a fold actually uses were
 * gated by nothing.
 *
 * It was caught by a change that reddened 12 and 68 while 5, 11, 16 and 32 all
 * stayed green, which is the wrong way round to find it. So: 5 and 11 for the
 * odd tails, 12 and 68 for the block and the coalesced reads, and 68 because it
 * is the width the trunk folds.
 */
import { webglExecutionOptions } from "./webgl-execution-options.js";
import { triangleMultiplication as cpuTriangle } from "../../cpu/af3/trunk/pairformer.js";
import { af3TriangleWeights } from "../../webgpu/af3/trunk/triangle.js";
import { TriangleMultiplicationIncomingGpu, TriangleMultiplicationOutgoingGpu }
  from "../../webgpu/kernels/triangle/webgpu.js";
import { setDeviceTuning, setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  releaseTriangleWeights, triangleMultiplication, uploadTriangleWeights,
} from "../../webgl/af3/trunk/triangle.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { generator } from "./af2-synthetic-block.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "5e-6"));
  const channels = Number(option(args, "channels", "32"));
  const divideByLength = flag(args, "over-length");
  const lengths = option(args, "lengths", "5,11,12,68").split(",").map(Number);

  const random = generator(27182);
  const noise = (count, spread, centre = 0) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };

  setDeviceTuning(device, { allowHostWeightPacking: true });
  setHalfPrecision(device, false);
  const gl = createWebGL2Context();
  // (the shared option flags reach it, so a kernel path can be switched off and the arm still checked - f32 unless
  // asked, which is what the bar is for: `--half-weights` / `--half-x` for the fold's precisions)
  const execution = new Execution(gl, {
    ...webglExecutionOptions(args), halfWeights: args.includes("--half-weights"), halfActivations: args.includes("--half-x"),
  });
  const arms = [];
  const failures = [];

  const weights = {
    leftNormInputScale: noise(channels, 0.2, 1),
    leftNormInputOffset: noise(channels, 0.1),
    projection: noise(channels * channels * 2, 1 / Math.sqrt(channels)),
    gate: noise(channels * channels * 2, 1 / Math.sqrt(channels)),
    centerNormScale: noise(channels, 0.2, 1),
    centerNormOffset: noise(channels, 0.1),
    outputProjection: noise(channels * channels, 1 / Math.sqrt(channels)),
    gatingLinear: noise(channels * channels, 1 / Math.sqrt(channels)),
  };
  const dialect = divideByLength ? { triangleMulDivideByLength: true } : undefined;

  for (const length of lengths) {
    const pairs = length * length;
    const pair = noise(pairs * channels, 1.2);
    // A couple of masked pairs, since the mask multiplies both operands and a
    // fold's own mask is not all ones.
    const mask = new Float32Array(pairs).fill(1);
    if (pairs > 3) { mask[1] = 0; mask[pairs - 2] = 0; }
    // The control skews half of each pair row's channels: the pair meets a
    // LayerNorm first, and a whole-row affine change is invisible to one.
    const skewed = falsify
      ? Float32Array.from(pair, (v, i) => (i % channels < channels / 2 ? v * 1.01 + 0.01 : v))
      : pair;

    const uploaded = uploadTriangleWeights(execution, weights);
    const pairTensor = execution.upload(skewed, `af3.tri.pair.${length}`);
    const maskTensor = execution.upload(mask, `af3.tri.mask.${length}`);
    for (const direction of ["outgoing", "incoming"]) {
      const update = triangleMultiplication(execution, {
        pair: pairTensor, mask: maskTensor, weights: uploaded,
        length, channels, direction: /** @type {any} */ (direction), divideByLength,
      });
      const ours = Float32Array.from(execution.read(update));
      execution.release(update);

      const expected = cpuTriangle(
        pair, mask, length, channels, direction, weights, dialect);
      // 🔴 THE RUNNER DIRECTLY, NOT `af3TriangleMultiplication`, BECAUSE THAT
      // HELPER CANNOT BE TOLD THE DIALECT. It builds its own `shape` as
      // `{length, cZ, cHidden}` and forwards no divide-by-length flag, which the
      // WGSL shader reads off `shape.triangleMulDivideByLength` - so the
      // `--over-length` arm through the helper compared an rf3 port against a
      // stock-AF3 reference and read 4.5e-3 with nothing wrong. The two
      // references disagreeing by exactly that amount is what said so.
      const Runner = direction === "outgoing"
        ? TriangleMultiplicationOutgoingGpu : TriangleMultiplicationIncomingGpu;
      const theirs = await new Runner(device).run({
        shape: {
          length, cZ: channels, cHidden: channels,
          ...(divideByLength ? { triangleMulDivideByLength: true } : {}),
        },
        epsilon: 1e-5,
        z: pair,
        mask,
        weights: af3TriangleWeights(weights, channels),
      }, { precision: "f32", variance: "fast" });

      for (const [label, want] of [["f64 reference", expected], ["WGSL", theirs.output]]) {
        const name = `${direction} L=${length} vs the ${label}`;
        const relative = relativeRms(ours, want, name);
        const ok = relative <= bar;
        arms.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
        if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
      }
      // The two references against each other: if they disagree, the instrument
      // is wrong and neither number above means anything.
      const between = relativeRms(expected, theirs.output, `between.${direction}.${length}`);
      arms.push({
        arm: `${direction} L=${length}: the two references agree`,
        relativeRms: Number(between.toPrecision(4)), bar, ok: between <= bar,
        falsifiable: false,
        reason: "both references read the unperturbed pair",
      });
      if (!(between <= bar)) {
        failures.push(
          `${direction} L=${length}: the references disagree by ${between.toExponential(3)}`);
      }
    }
    execution.release(pairTensor);
    execution.release(maskTensor);
    releaseTriangleWeights(execution, uploaded);
  }

  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const falsifiable = arms.filter((row) => row.falsifiable !== false);
  const red = falsifiable.filter((row) => !row.ok).length;
  return {
    channels,
    lengths,
    divideByLength,
    bar,
    falsify,
    arms,
    memory: execution.stats(),
    verdict: falsify
      ? (red === falsifiable.length
        ? `ok: all ${falsifiable.length} falsifiable arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${falsifiable.length} falsifiable arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms, both directions at ${lengths.length} lengths,`
          + " both references agreeing"
        : `FAILED: ${failures.join("; ")}`),
  };
}
