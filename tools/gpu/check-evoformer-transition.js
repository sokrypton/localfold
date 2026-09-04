/**
 * AF2's evoformer transition on the GPU against a CPU reference.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-evoformer-transition.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-evoformer-transition.js --rows=37 --channels=131
 *
 * 🔴 AF2 HAS NO OFFICIAL-VALUE GATE ON THIS MACHINE and this is the second
 * hole it fills, after tools/gpu/check-triangle-residual.js. `npm run test:gpu`
 * cannot load Dawn here, and `test/fixtures/evoformer/` is gitignored, so
 * test/evoformer-transition.gpu.test.js cannot run - which leaves the kernel
 * that is 31 ms of AF2's 118 ms block at 512 MSA rows with nothing checking it.
 *
 * The reference is LayerNorm, a linear with ReLU, and a linear - written here
 * rather than imported, because a reference that shares code with the thing it
 * checks tests nothing. It is a differential check, not an oracle one: it says
 * the kernel computes the transition, not that AlphaFold agrees.
 *
 * 🔴 THE SHAPES ARE DELIBERATELY RAGGED. The kernel tiles rows by 16 and
 * columns by 64, so a row count that is a multiple of 16 and a channel count
 * that is a multiple of 64 exercise none of its bounds checks. The defaults are
 * primes.
 */
import { TransitionGpu } from "../../src/evoformer/transition.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function reference({ rows, channels, hiddenChannels, activations, weights, epsilon }) {
  const output = new Float32Array(rows * channels);
  const hidden = new Float32Array(hiddenChannels);
  for (let row = 0; row < rows; row += 1) {
    const base = row * channels;
    let mean = 0;
    for (let c = 0; c < channels; c += 1) mean += activations[base + c];
    mean /= channels;
    let variance = 0;
    for (let c = 0; c < channels; c += 1) {
      const d = activations[base + c] - mean;
      variance += d * d;
    }
    const inverse = 1 / Math.sqrt(variance / channels + epsilon);
    const normalized = new Float32Array(channels);
    for (let c = 0; c < channels; c += 1) {
      normalized[c] = (activations[base + c] - mean) * inverse * weights.layerNormScale[c]
        + weights.layerNormOffset[c];
    }
    for (let h = 0; h < hiddenChannels; h += 1) {
      let total = weights.firstBias[h];
      for (let c = 0; c < channels; c += 1) {
        total += normalized[c] * weights.firstWeight[c * hiddenChannels + h];
      }
      hidden[h] = Math.max(total, 0);
    }
    for (let c = 0; c < channels; c += 1) {
      let total = weights.secondBias[c];
      for (let h = 0; h < hiddenChannels; h += 1) {
        total += hidden[h] * weights.secondWeight[h * channels + c];
      }
      output[base + c] = total;
    }
  }
  return output;
}

export async function main(device, args) {
  const rows = Number(option(args, "rows", "37"));
  const channels = Number(option(args, "channels", "67"));
  const hiddenChannels = Number(option(args, "hidden", "131"));
  const epsilon = 1e-5;

  let state = 24601;
  const random = (count, scale = 1) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      out[i] = (state / 0x7fffffff - 0.5) * scale;
    }
    return out;
  };
  const input = {
    rows, channels, hiddenChannels, epsilon,
    activations: random(rows * channels, 4),
    weights: {
      layerNormScale: random(channels, 0.5).map((v) => v + 1),
      layerNormOffset: random(channels, 0.2),
      firstWeight: random(channels * hiddenChannels, 0.3),
      firstBias: random(hiddenChannels, 0.1),
      secondWeight: random(hiddenChannels * channels, 0.3),
      secondBias: random(channels, 0.1),
    },
  };

  // 🔴 THE PRECISION IS AN AXIS, AND WITHOUT IT THIS CHECKED ONLY HALF THE
  // SHIPPED PATH. A block's transitions run the f16 kernel wherever the device
  // has shader-f16 and the shape is large enough - see chooseLinearKernel - so
  // a checker that always built the f32 one was silent about the kernel that
  // actually folds. Each arm is held to the tolerance its arithmetic implies.
  const precisions = option(args, "precision", "f32,f16").split(",");
  const bounds = { f32: 1e-5, f16: 4e-3 };
  const results = [];
  for (const precision of precisions) {
    if (precision === "f16" && !device.features.has("shader-f16")) continue;
    results.push(await check(device, input, precision, bounds[precision]));
  }
  const failed = results.filter((r) => !r.ok).length;
  if (failed > 0) throw new Error(`${failed} transition precision(s) outside tolerance`);
  return { rows, channels, hiddenChannels, results };
}

async function check(device, input, precision, bound) {
  const { rows, channels, hiddenChannels } = input;
  const { output } = await new TransitionGpu(device, { precision }).run(input);
  const expected = reference(input);
  let error = 0;
  let scale = 0;
  let worst = 0;
  for (let i = 0; i < expected.length; i += 1) {
    error += (output[i] - expected[i]) ** 2;
    scale += expected[i] ** 2;
    worst = Math.max(worst, Math.abs(output[i] - expected[i]));
  }
  const relRms = Math.sqrt(error / scale);
  const ok = relRms <= bound;
  console.log(`${ok ? "PASS" : "FAIL"}\trows ${rows} channels ${channels} hidden ${hiddenChannels}`
    + `\trelRMS ${relRms.toExponential(2)}\tworst ${worst.toExponential(2)}`);
  return { precision, bound, relRms, worst, ok };
}
