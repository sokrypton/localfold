/**
 * The contact map in WebGL2 against the shipped host reference.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-distogram.js [--falsify]
 *
 * 🔴 THE REFERENCE HERE IS NOT A TRANSCRIPTION - IT IS THE PAGE'S OWN HOST
 * IMPLEMENTATION. `distogramContactProbabilities` in shared/heads/distogram.js
 * is what the page ran before the WebGPU kernel replaced it, it keeps f64
 * accumulators, and nothing in it was written for this port. The WebGPU head is
 * an `encode...` function rather than a class, so it cannot be driven
 * standalone; this reference is the second implementation instead, and a better
 * one than a shader transcription because it was written independently.
 *
 * Three conventions it pins, each of which would otherwise be a plausible map:
 * the pair is SYMMETRISED (`half(i, j) + half(j, i)`), the bias is therefore
 * counted TWICE, and a bin counts only when its upper edge is at or below the
 * threshold - so the bin straddling 8 A is excluded.
 *
 * 🔴 AND THE ASYMMETRIC ARM IS THE ONE THAT MATTERS. A random pair tensor is
 * asymmetric, so `pair[i][j] + pair[j][i]` and `2 * pair[i][j]` differ - which
 * is what makes a missing transpose visible. The arms also include a
 * deliberately SYMMETRIC pair, where the two agree, as the control that says
 * the asymmetric arm is doing the work.
 */
import { distogramContactProbabilities } from "../../shared/heads/distogram.js";
import { countedBins, distogramContacts } from "../../webgl/heads/distogram.js";
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
  const bar = Number(option(args, "bar", "1e-6"));
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const rows = [];
  const failures = [];

  const cases = [
    { length: 7, channels: 32, bins: 64, symmetric: false },
    { length: 5, channels: 128, bins: 64, symmetric: false },
    { length: 6, channels: 16, bins: 32, symmetric: false },
    { length: 6, channels: 16, bins: 32, symmetric: true },
  ];

  for (const shape of cases) {
    const { length, channels, bins, symmetric } = shape;
    const random = generator(length * 13 + channels + bins + (symmetric ? 1 : 0));
    const noise = (count, spread) => {
      const out = new Float32Array(count);
      for (let i = 0; i < count; i += 1) out[i] = (random() * 2 - 1) * spread;
      return out;
    };
    const pair = noise(length * length * channels, 1.5);
    if (symmetric) {
      for (let i = 0; i < length; i += 1) {
        for (let j = 0; j < i; j += 1) {
          for (let c = 0; c < channels; c += 1) {
            pair[(j * length + i) * channels + c] = pair[(i * length + j) * channels + c];
          }
        }
      }
    }
    const weights = noise(channels * bins, 1 / Math.sqrt(channels));
    const bias = noise(bins, 0.1);

    const pairTensor = execution.upload(falsify
      ? Float32Array.from(pair, (v, i) => (i < channels / 2 ? v * 1.01 + 0.01 : v))
      : pair, "pair");
    const contacts = distogramContacts(execution, {
      pair: pairTensor,
      weights: execution.upload(weights, "w"),
      bias: execution.upload(bias, "b"),
      length, channels, bins,
    });
    const ours = execution.read(contacts).slice();
    const want = distogramContactProbabilities(pair, weights, bias, length, { channels, bins });

    const label = `L=${length} c=${channels} bins=${bins}${symmetric ? " symmetric" : ""}`;
    const relative = relativeRms(ours, want, label);
    const ok = relative <= bar;
    rows.push({
      arm: label,
      relativeRms: Number(relative.toPrecision(4)),
      countedBins: countedBins(bins),
      bar,
      ok,
    });
    if (!ok) failures.push(`${label}: relRMS ${relative.toExponential(3)} over ${bar}`);

    // ...and a reading a map has to satisfy whatever the weights are.
    let outOfRange = 0;
    for (const value of ours) if (!(value >= 0 && value <= 1)) outOfRange += 1;
    if (outOfRange !== 0) failures.push(`${label}: ${outOfRange} probabilities outside [0, 1]`);

    execution.release(contacts);
    for (const tensor of [...execution.storage.live].filter((t) => t.label !== "pair")) {
      execution.release(tensor);
    }
    execution.release(pairTensor);
  }

  const red = rows.filter((row) => !row.ok).length;
  return {
    bar,
    falsify,
    arms: rows,
    liveTensorsAfter: execution.storage.live.size,
    verdict: falsify
      ? (red === rows.length
        ? `ok: all ${rows.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${rows.length} arms noticed a 1% perturbation`)
      : (failures.length === 0 && execution.storage.live.size === 0
        ? `ok: ${rows.length} arms, every probability in [0, 1], no tensor left live`
        : `FAILED: ${failures.join("; ")}`),
  };
}
