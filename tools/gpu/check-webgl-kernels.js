/**
 * The two WebGL2 kernels against independent CPU references - the gate they
 * have never had.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-kernels.js
 *
 * 🔴 WHY THIS EXISTS: 649 LINES OF KERNEL WITH NO DIFFERENTIAL AT ALL.
 * `webgl/kernels/triangle.js` and `webgl/kernels/attention.js` were written as
 * spikes to answer whether WebGL2 can do this work, run by hand once, and their
 * numbers written into docs/A100.md and docs/DEVELOPING.md. Nothing has run them
 * since. AGENTS.md requires every GPU kernel to have "a deterministic
 * differential test against an independent reference implementation", and these
 * two were the exception - which also means the figures they are quoted for
 * ("1.4x behind on the contraction, 1.55x ahead on attention, computing the
 * same numbers") rest on a comparison no gate can reproduce.
 *
 * The references here are written from the SHADER's own index arithmetic read as
 * algebra, in plain JavaScript, and share no code with the kernels: they are
 * independent in the sense that matters, which is that a wrong fetch in GLSL
 * cannot also be wrong here.
 *
 * 🔴 AND THE LAYOUT ARM IS THE SHARP ONE. `channel-major` and `channel-minor`
 * are the SAME arithmetic over a relaid-out operand - no sum is reordered - so
 * they must agree **bit for bit** and a 1e-6 bar there would pass an off-by-a-
 * group index. The contraction arms sum over k in the same order on both sides,
 * so they are near-exact too; the attention arm is an online softmax against a
 * two-pass one and is the only arm with a real numerical floor.
 *
 * 🔴 AND EVERY BAR IS FALSIFIED - ONE SIDE ONLY, WHICH THE FIRST VERSION GOT
 * WRONG AND THE TOOL CAUGHT ITSELF. `--falsify` perturbs one element by 1% in
 * the operand handed to the KERNEL and leaves the reference's clean, so every
 * arm must go RED. Perturbing the SHARED input instead moves both sides equally
 * and changes nothing: that version reported "only 4 of 18 arms noticed", the 4
 * being the layout arms, which compare two kernel runs and were already
 * one-sided. A falsification that feeds both arms is this repository's
 * "comparing something with itself" trap wearing the clothes of a control.
 */
import { AttentionFlashWebGL2, HEAD_DIM } from "../../webgl/kernels/attention.js";
import { TriangleContractWebGL2 } from "../../webgl/kernels/triangle.js";
import { createWebGL2Context, webgl2Capability } from "../../webgl/index.js";
import { relativeRms } from "./relative-rms.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};
const flag = (args, name) => (args ?? []).includes(`--${name}`);

/** The repository's own deterministic generator, so an arm is reproducible. */
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

const noise = (count, scale, random) => {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) out[i] = (random() * 2 - 1) * scale;
  return out;
};

/**
 * The contraction, from the shader's own indices.
 *
 * outgoing: out[(i,j),h] = sum_k a[(i,k),h] * b[(j,k),h]
 * incoming: out[(i,j),h] = sum_k a[(k,i),h] * b[(k,j),h]
 */
function contractReference(a, b, length, channels, direction) {
  const out = new Float32Array(length * length * channels);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      for (let h = 0; h < channels; h += 1) {
        let sum = 0;
        for (let k = 0; k < length; k += 1) {
          const [left, right] = direction === "outgoing"
            ? [(i * length + k) * channels + h, (j * length + k) * channels + h]
            : [(k * length + i) * channels + h, (k * length + j) * channels + h];
          sum += a[left] * b[right];
        }
        out[(i * length + j) * channels + h] = sum;
      }
    }
  }
  return out;
}

/**
 * Gated attention with an optional pair bias, two-pass softmax.
 *
 * The kernel threads a running maximum and sum through its key loop; this takes
 * the maximum first and normalises after, which is the independent formulation.
 * Output layout is the kernel's `read`: [batch][query][head][channel].
 */
function attentionReference(input) {
  const { query, key, value, gate, mask, pairBias, batch, queries, heads } = input;
  const dim = HEAD_DIM;
  const out = new Float32Array(batch * queries * heads * dim);
  for (let n = 0; n < batch; n += 1) {
    for (let head = 0; head < heads; head += 1) {
      for (let q = 0; q < queries; q += 1) {
        const qBase = ((n * queries + q) * heads + head) * dim;
        const logits = new Float64Array(queries);
        for (let k = 0; k < queries; k += 1) {
          const kBase = ((n * queries + k) * heads + head) * dim;
          let score = 0;
          for (let c = 0; c < dim; c += 1) score += query[qBase + c] * key[kBase + c];
          const maskIndex = input.transpose ? k * batch + n : n * queries + k;
          let logit = score + 1e9 * (mask[maskIndex] - 1);
          if (pairBias !== undefined) logit += pairBias[(head * queries + q) * queries + k];
          logits[k] = Math.min(Math.max(logit, -1e8), 1e8);
        }
        let top = -Infinity;
        for (const logit of logits) top = Math.max(top, logit);
        let total = 0;
        const weights = new Float64Array(queries);
        for (let k = 0; k < queries; k += 1) {
          weights[k] = Math.exp(logits[k] - top);
          total += weights[k];
        }
        for (let c = 0; c < dim; c += 1) {
          let acc = 0;
          for (let k = 0; k < queries; k += 1) {
            acc += weights[k] * value[((n * queries + k) * heads + head) * dim + c];
          }
          out[qBase + c] = (acc / total) * gate[qBase + c];
        }
      }
    }
  }
  return out;
}

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  /** One element 1% out, for the KERNEL's copy alone; identity without the flag. */
  const skew = (values) => (falsify
    ? Float32Array.from(values, (v, i) => (i === (values.length >> 1) ? v * 1.01 + 0.01 : v))
    : values);
  const gl = createWebGL2Context();
  const capability = webgl2Capability(gl);
  const rows = [];
  const failures = [];

  const arm = (name, got, want, bar) => {
    const relative = relativeRms(got, want, name);
    const ok = relative <= bar;
    rows.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
    return relative;
  };

  // --- the contraction -----------------------------------------------------
  const contract = new TriangleContractWebGL2(gl);
  for (const [length, channels] of [[8, 4], [13, 8], [24, 32], [31, 12]]) {
    const random = generator(length * 1000 + channels);
    const a = noise(length * length * channels, 1, random);
    const b = noise(length * length * channels, 1, random);
    for (const direction of ["outgoing", "incoming"]) {
      const got = contract.run({ a: skew(a), b, length, channels, direction });
      const want = contractReference(a, b, length, channels, direction);
      // Same summation order on both sides, so this is rounding alone.
      arm(`triangle ${direction} L=${length} CH=${channels}`, got, want, 2e-6);
    }
    // A relayout reorders no sum: bit-exact or the index is wrong.
    //
    // 🔴 AND ITS FALSIFICATION HAS TO PERTURB ONE SIDE ONLY. Both runs read the
    // same `a`, so a perturbation of the INPUT reaches both arms equally and the
    // difference stays exactly 0 - this arm is a self-comparison and the one
    // kind of arm an input-level falsification cannot touch, which is this
    // repository's "comparing something with itself" trap in miniature. The
    // perturbed copy goes to one run.
    const minor = contract.run({ a, b, length, channels, layout: "channel-minor" });
    const major = contract.run({ a: skew(a), b, length, channels, layout: "channel-major" });
    arm(`triangle layout L=${length} CH=${channels}`, major, minor, 0);
  }

  // --- gated flash attention ----------------------------------------------
  const attention = new AttentionFlashWebGL2(gl);
  for (const [batch, queries, heads] of [[1, 8, 1], [2, 16, 4], [3, 11, 2]]) {
    const random = generator(batch * 100 + queries * 10 + heads);
    const elements = batch * queries * heads * HEAD_DIM;
    const base = {
      query: noise(elements, 0.5, random),
      key: noise(elements, 0.5, random),
      value: noise(elements, 1, random),
      gate: noise(elements, 1, random),
      mask: new Float32Array(batch * queries).fill(1),
      batch, queries, heads,
    };
    // ...a real mask: the last key of every batch row is withheld.
    for (let n = 0; n < batch; n += 1) base.mask[n * queries + queries - 1] = 0;
    const label = `b=${batch} q=${queries} h=${heads}`;
    arm(`attention ${label}`,
      attention.run({ ...base, key: skew(base.key) }), attentionReference(base), 1e-5);

    const pairBias = noise(heads * queries * queries, 2, random);
    const biased = { ...base, pairBias };
    arm(`attention+bias ${label}`,
      attention.run({ ...biased, key: skew(biased.key) }), attentionReference(biased), 1e-5);
  }

  return {
    renderer: capability.renderer,
    maxTextureSize: capability.maxTextureSize,
    pairTokenCeiling: capability.pairTokenCeiling,
    fragmentOutputFloats: capability.fragmentOutputFloats,
    vertexOutputFloats: capability.vertexOutputFloats,
    falsify,
    arms: rows,
    // 🔴 UNDER --falsify EVERY ARM MUST FAIL, so the pass is the inverse.
    verdict: falsify
      ? (failures.length === rows.length
        ? `ok: all ${rows.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${failures.length} of ${rows.length} arms noticed a 1% perturbation`)
      : (failures.length === 0 ? `ok: ${rows.length} arms` : `FAILED: ${failures.join("; ")}`),
  };
}
