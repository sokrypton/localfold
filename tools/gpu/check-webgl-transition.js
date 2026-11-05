/**
 * The transition in WebGL2 against the WGSL one and a CPU reference.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-transition.js [--falsify]
 *
 * Three draws out of kernels that already had their own differentials, so what
 * is actually under test here is the WIRING and the two conventions: that the
 * weights are `[in][out]` (`weight_k * parameters.columns` in the WGSL shader,
 * where the triangle's reference is `[out][in]`), and that the output is the
 * transition's own rather than the input plus it.
 *
 * 🔴 THE ARMS ASK FOR `hiddenPrecision: "f32"`, WHICH IS NOT THE DEFAULT. The
 * shipped Evoformer block keeps the transition's hidden activation in f16 -
 * `hiddenStorage = hiddenChannels % 4 === 0 ? "f16" : "f32"`, not gated by any
 * knob - and `webgl/kernels/transition.js` defaults to matching that.
 * `TransitionGpu`, the standalone class this gate compares against, keeps f32.
 * So the default and this class differ by exactly the rounding, and the last
 * arm MEASURES that rather than hiding it: **1.8e-4 to 2.0e-4**, which is what
 * an f16 hidden activation costs a transition and which is the number a
 * whole-block comparison has to account for.
 *
 * 🔴 BOTH REFERENCES, BECAUSE ONE CANNOT TELL WHICH SIDE IS WRONG. The attention
 * block's cross-check read 1e-3 against WGSL and 2.5e-7 against f64 for the same
 * tensor, and the answer was that the WGSL side had defaulted to f16 arithmetic
 * on an axis that had not been pinned. With only the WGSL arm that reads as a
 * broken kernel; with both it reads as a misconfigured instrument, which is what
 * it was. `precision: "f32"` is pinned here for the same reason.
 */
import { TransitionGpu } from "../../webgpu/kernels/transition.js";
import {
  releaseTransitionWeights, transition, uploadTransitionWeights,
} from "../../webgl/kernels/transition.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function generator(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000;
  };
}
const noise = (count, spread, random, centre = 0) => {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
  return out;
};

/** The transition, from the operation. Weights are `[in][out]`. */
function transitionReference(input) {
  const { activations, weights: w, rows, channels, hiddenChannels } = input;
  const epsilon = input.epsilon ?? 1e-5;
  const out = new Float32Array(rows * channels);
  for (let row = 0; row < rows; row += 1) {
    const at = row * channels;
    let mean = 0;
    for (let c = 0; c < channels; c += 1) mean += activations[at + c];
    mean /= channels;
    let spread = 0;
    for (let c = 0; c < channels; c += 1) spread += (activations[at + c] - mean) ** 2;
    const inverseStd = 1 / Math.sqrt(spread / channels + epsilon);
    const x = new Float64Array(channels);
    for (let c = 0; c < channels; c += 1) {
      x[c] = (activations[at + c] - mean) * inverseStd * w.layerNormScale[c] + w.layerNormOffset[c];
    }
    const hidden = new Float64Array(hiddenChannels);
    for (let h = 0; h < hiddenChannels; h += 1) {
      let sum = w.firstBias[h];
      for (let c = 0; c < channels; c += 1) sum += x[c] * w.firstWeight[c * hiddenChannels + h];
      hidden[h] = Math.max(sum, 0);
    }
    for (let c = 0; c < channels; c += 1) {
      let sum = w.secondBias[c];
      for (let h = 0; h < hiddenChannels; h += 1) {
        sum += hidden[h] * w.secondWeight[h * channels + c];
      }
      out[at + c] = sum;
    }
  }
  return out;
}

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "1e-5"));
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const rows = [];
  const failures = [];
  const halfCost = [];
  const arm = (name, got, want) => {
    const relative = relativeRms(got, want, name);
    const ok = relative <= bar;
    rows.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };

  // AF2's pair transition is channels 128 hidden 512; its MSA transition is
  // 256 and 1024. The small shapes are for the ragged row counts.
  const cases = [
    { rows: 7, channels: 32, hiddenChannels: 128 },
    { rows: 64, channels: 128, hiddenChannels: 512 },
    { rows: 13, channels: 64, hiddenChannels: 64 },
  ];

  for (const shape of cases) {
    const random = generator(shape.rows * 101 + shape.channels + shape.hiddenChannels);
    const input = {
      ...shape,
      activations: noise(shape.rows * shape.channels, 1.5, random),
      weights: {
        layerNormScale: noise(shape.channels, 0.2, random, 1),
        layerNormOffset: noise(shape.channels, 0.1, random),
        firstWeight: noise(shape.channels * shape.hiddenChannels,
          1 / Math.sqrt(shape.channels), random),
        firstBias: noise(shape.hiddenChannels, 0.1, random),
        secondWeight: noise(shape.hiddenChannels * shape.channels,
          1 / Math.sqrt(shape.hiddenChannels), random),
        secondBias: noise(shape.channels, 0.05, random),
      },
    };
    const label = `rows=${shape.rows} c=${shape.channels} h=${shape.hiddenChannels}`;

    const weights = uploadTransitionWeights(execution, input.weights);
    const activations = execution.upload(falsify
      ? Float32Array.from(input.activations,
        (v, i) => (i === (input.activations.length >> 1) ? v * 1.01 + 0.01 : v))
      : input.activations, "activations");
    const output = transition(execution, {
      ...shape, activations, weights, hiddenPrecision: "f32",
    });
    const ours = execution.read(output).slice();
    execution.release(output);

    arm(`vs CPU reference ${label}`, ours, transitionReference(input));
    const wgsl = await new TransitionGpu(device, { precision: "f32" }).run(input);
    arm(`vs WGSL transition ${label}`, ours, wgsl);

    // ...and what the shipped block's f16 hidden activation costs, reported
    // rather than asserted: it is a property of the reference being matched,
    // not of this port.
    const halved = transition(execution, {
      ...shape, activations, weights, hiddenPrecision: "f16",
    });
    halfCost.push({
      shape: label,
      relativeRms: Number(relativeRms(execution.read(halved), ours, "half").toPrecision(3)),
    });
    execution.release(halved);

    releaseTransitionWeights(execution, weights);
    execution.release(activations);
  }

  const stats = execution.stats();
  const red = rows.filter((row) => !row.ok).length;
  return {
    bar,
    falsify,
    arms: rows,
    // What an f16 hidden activation costs - the shipped block's choice.
    f16HiddenCost: halfCost,
    liveTensorsAfter: stats.live,
    verdict: falsify
      ? (red === rows.length
        ? `ok: all ${rows.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${rows.length} arms noticed a 1% perturbation`)
      : (failures.length === 0 && stats.live === 0
        ? `ok: ${rows.length} arms, no tensor left live`
        : `FAILED: ${[...failures, ...(stats.live ? [`${stats.live} tensors leaked`] : [])].join("; ")}`),
  };
}
