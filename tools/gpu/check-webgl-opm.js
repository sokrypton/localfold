/**
 * The outer product mean in WebGL2 against the WGSL one and a CPU reference -
 * four conventions and a two-stage contraction.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-opm.js [--falsify]
 *
 * 🔴 THIS IS THE OPERATION WITH THE MOST WAYS TO BE PLAUSIBLY WRONG, which is
 * why each one is an assertion rather than a reading of the code. The mask goes
 * on the PROJECTIONS (so the numerator carries `mask[s][i] * mask[s][j]`
 * exactly as the denominator does); `outputWeight` is `[c_left][c_right][cZ]`
 * and `cOuter` is SQUARE, so transposing the two channel axes changes nothing
 * about the tensor's shape; `outputBias` is added INSIDE the division, so it is
 * divided by the mask count too; and the normalisation epsilon is 1e-3 where
 * the LayerNorm's is 1e-5. Three of the four would survive a shape check and
 * return a tensor of the right magnitude.
 *
 * 🔴 THE FALSIFICATION PERTURBS HALF AN MSA ROW, AND BOTH OF THE OBVIOUS
 * CHOICES WERE WRONG FIRST - THE SECOND OF THEM SILENTLY.
 *
 * One element 1% out reaches the output through a LayerNorm over cM, one row of
 * the left and right projections, and then a mean over `sequences`: at 4
 * sequences, 6 residues and cM 16 it moved the output from 1.8e-7 to **9.3e-7**,
 * which is propagation and is still under a 1e-5 bar. Averaging is what this
 * operation IS, so four arms of six went red and two did not.
 *
 * 🔴 SO THE NEXT ATTEMPT PERTURBED A WHOLE ROW, AND **ZERO** OF SIX ARMS
 * NOTICED - A STRICTLY WORSE CONTROL THAT LOOKED STRICTLY STRONGER. LayerNorm
 * is invariant to an affine change of the row it normalises: send `x` to
 * `a * x + b` and the mean goes to `a * mean + b`, the deviations to
 * `a * (x - mean)`, the standard deviation to `a * std`, and
 * `(x - mean) / std` comes out unchanged. `v * 1.01 + 0.01` applied to exactly
 * one row is therefore a NO-OP by construction, and the arms returned to their
 * baseline to the digit. **In any pipeline that starts with a normalisation, a
 * whole-row affine perturbation is not a control at all** - which is worth
 * knowing before writing the next one of these, since every operation in this
 * backend starts with a LayerNorm.
 *
 * Half a row survives it: the row's distribution changes, so the normalisation
 * cannot absorb it, and it is still a small local error rather than a different
 * input.
 *
 * 🔴 AND cOuter IS DELIBERATELY NOT SQUARE-SYMMETRIC IN THE ARMS. If `left` and
 * `right` were fed the same weights, swapping them - or transposing
 * `outputWeight`'s two channel axes - would be invisible. The two projections
 * get independent weights and the arms use `length` values of i and j that
 * differ, so an i/j transpose shows up as it did in the triangle contraction,
 * where exactly that error read relRMS 1.3.
 */
import { OuterProductMeanGpu } from "../../webgpu/kernels/outer-product-mean.js";
import {
  outerProductMean, releaseOuterProductMeanWeights, uploadOuterProductMeanWeights,
} from "../../webgl/kernels/outer-product-mean.js";
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

/** The operation, written from the definition. */
function opmReference(input) {
  const {
    activations, mask, weights: w, sequences, length, cM, cOuter, cZ,
  } = input;
  const layerNormEpsilon = input.layerNormEpsilon ?? 1e-5;
  const normalizationEpsilon = input.normalizationEpsilon ?? 1e-3;
  const rows = sequences * length;

  const left = new Float64Array(rows * cOuter);
  const right = new Float64Array(rows * cOuter);
  for (let row = 0; row < rows; row += 1) {
    const at = row * cM;
    let mean = 0;
    for (let c = 0; c < cM; c += 1) mean += activations[at + c];
    mean /= cM;
    let spread = 0;
    for (let c = 0; c < cM; c += 1) spread += (activations[at + c] - mean) ** 2;
    const inverseStd = 1 / Math.sqrt(spread / cM + layerNormEpsilon);
    const m = new Float64Array(cM);
    for (let c = 0; c < cM; c += 1) {
      m[c] = (activations[at + c] - mean) * inverseStd * w.layerNormScale[c] + w.layerNormOffset[c];
    }
    for (let o = 0; o < cOuter; o += 1) {
      let l = w.leftBias[o];
      let r = w.rightBias[o];
      for (let c = 0; c < cM; c += 1) {
        l += m[c] * w.leftWeight[c * cOuter + o];
        r += m[c] * w.rightWeight[c * cOuter + o];
      }
      left[row * cOuter + o] = l * mask[row];
      right[row * cOuter + o] = r * mask[row];
    }
  }

  const out = new Float32Array(length * length * cZ);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      let count = 0;
      for (let s = 0; s < sequences; s += 1) {
        count += mask[s * length + i] * mask[s * length + j];
      }
      for (let z = 0; z < cZ; z += 1) {
        let acc = 0;
        for (let s = 0; s < sequences; s += 1) {
          for (let cl = 0; cl < cOuter; cl += 1) {
            const l = left[(s * length + i) * cOuter + cl];
            if (l === 0) continue;
            for (let cr = 0; cr < cOuter; cr += 1) {
              acc += l * right[(s * length + j) * cOuter + cr]
                * w.outputWeight[(cl * cOuter + cr) * cZ + z];
            }
          }
        }
        out[(i * length + j) * cZ + z] = (acc + w.outputBias[z]) / (normalizationEpsilon + count);
      }
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
  const arm = (name, got, want) => {
    const relative = relativeRms(got, want, name);
    const ok = relative <= bar;
    rows.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };

  // The CPU reference is O(length^2 * cZ * sequences * cOuter^2), so the arms
  // stay small; AF2's real shape is cM 256, cOuter 32, cZ 128.
  const cases = [
    { sequences: 4, length: 6, cM: 16, cOuter: 8, cZ: 16 },
    { sequences: 6, length: 5, cM: 32, cOuter: 8, cZ: 32 },
    { sequences: 3, length: 8, cM: 16, cOuter: 16, cZ: 16 },
  ];

  for (const shape of cases) {
    const { sequences, length, cM, cOuter, cZ } = shape;
    const random = generator(sequences * 31 + length * 7 + cM + cOuter + cZ);
    const rowCount = sequences * length;
    const input = {
      ...shape,
      activations: noise(rowCount * cM, 1.5, random),
      mask: new Float32Array(rowCount).fill(1),
      weights: {
        layerNormScale: noise(cM, 0.2, random, 1),
        layerNormOffset: noise(cM, 0.1, random),
        leftWeight: noise(cM * cOuter, 1 / Math.sqrt(cM), random),
        leftBias: noise(cOuter, 0.1, random),
        rightWeight: noise(cM * cOuter, 1 / Math.sqrt(cM), random),
        rightBias: noise(cOuter, 0.1, random),
        outputWeight: noise(cOuter * cOuter * cZ, 1 / cOuter, random),
        outputBias: noise(cZ, 0.05, random),
      },
    };
    // A real mask, and NOT a whole-row one: the last residue of the last
    // sequence only, so `count` differs per pair and the denominator is
    // actually exercised. A mask of all ones makes it a constant.
    input.mask[(sequences - 1) * length + length - 1] = 0;
    input.mask[(sequences - 2) * length + 1] = 0;

    const label = `s=${sequences} L=${length} cM=${cM} cO=${cOuter} cZ=${cZ}`;
    const weights = uploadOuterProductMeanWeights(execution, input.weights);
    // HALF an MSA row 1% out, on the WebGL side only. Not one element (below
    // the bar at small depth) and not a whole row (invisible to the LayerNorm
    // that reads it first) - see the header.
    const skewFrom = Math.floor(rowCount / 2) * cM;
    const activations = execution.upload(falsify
      ? Float32Array.from(input.activations, (v, i) => (
        i >= skewFrom && i < skewFrom + cM / 2 ? v * 1.01 + 0.01 : v))
      : input.activations, "activations");
    const mask = execution.upload(input.mask, "mask");

    const output = outerProductMean(execution, { ...shape, activations, mask, weights });
    const ours = execution.read(output).slice();
    execution.release(output);

    arm(`vs CPU reference ${label}`, ours, opmReference(input));
    const wgsl = await new OuterProductMeanGpu(device).run(input);
    arm(`vs WGSL opm ${label}`, ours, wgsl);

    releaseOuterProductMeanWeights(execution, weights);
    execution.release(activations);
    execution.release(mask);
  }

  const stats = execution.stats();
  const red = rows.filter((row) => !row.ok).length;
  return {
    bar,
    falsify,
    arms: rows,
    liveTensorsAfter: stats.live,
    peakMegabytes: Number((stats.peakBytes / 2 ** 20).toFixed(2)),
    verdict: falsify
      ? (red === rows.length
        ? `ok: all ${rows.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${rows.length} arms noticed a 1% perturbation`)
      : (failures.length === 0 && stats.live === 0
        ? `ok: ${rows.length} arms, no tensor left live`
        : `FAILED: ${[...failures, ...(stats.live ? [`${stats.live} tensors leaked`] : [])].join("; ")}`),
  };
}
