/**
 * AF2's outer product mean on the GPU against a CPU reference.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-evoformer-opm.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-evoformer-opm.js --sequences=64 --length=11
 *
 * 🔴 AF2 HAS NO OFFICIAL-VALUE GATE ON THIS MACHINE, and this covers the kernel
 * that is 29 ms of its 176 ms block at 512 MSA rows - `opm.contract`, which is
 * two global reads for every multiply-add and reads at 111 billion a second, so
 * it runs at about 5% of what this part can do.
 *
 * 🔴 IT HAS TWO PATHS AND THIS EXERCISES BOTH. `useOuterFirstContraction` picks
 * between contracting the sequence axis first - which materialises
 * [L, L, c_outer, c_outer] and needs `sequences >= c_outer` and the tensor
 * under a byte limit - and the tiled accumulation otherwise. A check that only
 * ever ran one of them would leave the other unguarded, so --sequences chooses.
 *
 * The reference is written out here rather than imported: a reference sharing
 * code with the thing it checks tests nothing.
 */
import { OuterProductMeanGpu, useOuterFirstContraction }
  from "../../src/evoformer/outer-product-mean.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function reference(input) {
  const { sequences, length, cM, cOuter, cZ, activations, mask, weights } = input;
  const epsilon = input.epsilon ?? 1e-5;
  const left = new Float32Array(sequences * length * cOuter);
  const right = new Float32Array(sequences * length * cOuter);
  for (let s = 0; s < sequences; s += 1) {
    for (let i = 0; i < length; i += 1) {
      const row = (s * length + i) * cM;
      let mean = 0;
      for (let c = 0; c < cM; c += 1) mean += activations[row + c];
      mean /= cM;
      let variance = 0;
      for (let c = 0; c < cM; c += 1) {
        const d = activations[row + c] - mean;
        variance += d * d;
      }
      const inverse = 1 / Math.sqrt(variance / cM + epsilon);
      const normalized = new Float32Array(cM);
      for (let c = 0; c < cM; c += 1) {
        normalized[c] = (activations[row + c] - mean) * inverse * weights.layerNormScale[c]
          + weights.layerNormOffset[c];
      }
      const keep = mask[s * length + i];
      for (let o = 0; o < cOuter; o += 1) {
        let l = weights.leftBias[o];
        let r = weights.rightBias[o];
        for (let c = 0; c < cM; c += 1) {
          l += normalized[c] * weights.leftWeight[c * cOuter + o];
          r += normalized[c] * weights.rightWeight[c * cOuter + o];
        }
        left[(s * length + i) * cOuter + o] = keep * l;
        right[(s * length + i) * cOuter + o] = keep * r;
      }
    }
  }
  const output = new Float32Array(length * length * cZ);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      const outer = new Float32Array(cOuter * cOuter);
      let count = 0;
      for (let s = 0; s < sequences; s += 1) {
        count += mask[s * length + i] * mask[s * length + j];
        for (let a = 0; a < cOuter; a += 1) {
          const l = left[(s * length + i) * cOuter + a];
          for (let b = 0; b < cOuter; b += 1) {
            outer[a * cOuter + b] += l * right[(s * length + j) * cOuter + b];
          }
        }
      }
      for (let z = 0; z < cZ; z += 1) {
        let value = weights.outputBias[z];
        for (let a = 0; a < cOuter; a += 1) {
          for (let b = 0; b < cOuter; b += 1) {
            value += outer[a * cOuter + b] * weights.outputWeight[(a * cOuter + b) * cZ + z];
          }
        }
        output[(i * length + j) * cZ + z] = value / ((input.normalizationEpsilon ?? 1e-3) + count);
      }
    }
  }
  return output;
}

export async function main(device, args) {
  const sequences = Number(option(args, "sequences", "24"));
  const length = Number(option(args, "length", "9"));
  const cM = Number(option(args, "cm", "13"));
  const cOuter = Number(option(args, "outer", "7"));
  const cZ = Number(option(args, "cz", "11"));

  let state = 1337;
  const random = (count, scale = 1) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      out[i] = (state / 0x7fffffff - 0.5) * scale;
    }
    return out;
  };
  // 🔴 A RAGGED MASK, so a padded row that leaked into the product would show.
  const mask = new Float32Array(sequences * length);
  for (let i = 0; i < mask.length; i += 1) mask[i] = (i % 7 === 3) ? 0 : 1;

  const input = {
    sequences, length, cM, cOuter, cZ, epsilon: 1e-5, normalizationEpsilon: 1e-3,
    activations: random(sequences * length * cM, 4), mask,
    weights: {
      layerNormScale: random(cM, 0.5).map((v) => v + 1),
      layerNormOffset: random(cM, 0.2),
      leftWeight: random(cM * cOuter, 0.4), leftBias: random(cOuter, 0.1),
      rightWeight: random(cM * cOuter, 0.4), rightBias: random(cOuter, 0.1),
      outputWeight: random(cOuter * cOuter * cZ, 0.2), outputBias: random(cZ, 0.1),
    },
  };

  const { output } = await new OuterProductMeanGpu(device).run(input);
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
  const path = useOuterFirstContraction(input) ? "outer-first" : "tiled";
  const bound = 1e-5;
  const ok = relRms <= bound;
  console.log(`${ok ? "PASS" : "FAIL"}\t${path}\tsequences ${sequences} length ${length}`
    + `\trelRMS ${relRms.toExponential(2)}\tworst ${worst.toExponential(2)}`);
  if (!ok) throw new Error(`outer product mean relRMS ${relRms.toExponential(2)} exceeds ${bound}`);
  return { path, sequences, length, cM, cOuter, cZ, relRms, worst, ok };
}
