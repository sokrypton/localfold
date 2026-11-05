/**
 * Gated attention for every head width this port runs, against a CPU reference
 * and against the kernel it replaces.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-attention.js [--falsify]
 *
 * 🔴 THE HEAD WIDTH SWEEP IS THE POINT. The kernel this one generalises hard
 * coded a 32-channel head and eight render targets - AF2's commonest head and
 * this machine's budget - and neither is a constant: AF3's grid attention runs
 * heads of 16 and of 8, and the WebGL2 specification guarantees only four
 * targets. So the arms run D = 4, 8, 16, 32 and **64**, and 64 is the one that
 * matters: it needs `D/4 = 16` groups where only 8 targets exist, so the kernel
 * draws TWICE and each draw recomputes its own q.k dot products. That path has
 * no other way to be reached on this hardware and is where a `groupBase` error
 * would live - it would leave half the channels holding the other half's
 * answer, which is a plausible tensor of the right shape.
 *
 * 🔴 AND THE SECOND REFERENCE IS THE OLD KERNEL AT ITS OWN CONVENTION. The
 * original applies NO `1/sqrt(head_dim)` scaling - it was correct only for a
 * caller that had pre-scaled its queries, and nothing said so - so the
 * cross-implementation arm passes `scale: 1` to ask the new kernel for the old
 * one's answer. That arm is a regression check against 363 lines that are
 * independently verified (`check-webgl-kernels.js`), and it is the only thing
 * here that would catch the plane/interleave gather being wrong in a way a
 * reference written from the same shader would share.
 */
import { AttentionFlashWebGL2 } from "../../webgl/kernels/attention.js";
import { attentionFlash, attentionFlashPlan } from "../../webgl/kernels/attention-flash.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);

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
const noise = (count, spread, random) => {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) out[i] = (random() * 2 - 1) * spread;
  return out;
};

/**
 * Gated attention, two-pass softmax, from the operation rather than the shader.
 *
 * The kernel threads a running maximum and sum through its key loop; this takes
 * the maximum first and normalises after.
 */
function attentionReference(input) {
  const { query, key, value, gate, mask, pairBias, batch, queries, heads, headDim } = input;
  const scale = input.scale ?? 1 / Math.sqrt(headDim);
  const out = new Float32Array(batch * queries * heads * headDim);
  for (let b = 0; b < batch; b += 1) {
    for (let head = 0; head < heads; head += 1) {
      for (let q = 0; q < queries; q += 1) {
        const qBase = ((b * queries + q) * heads + head) * headDim;
        const logits = new Float64Array(queries);
        for (let k = 0; k < queries; k += 1) {
          const kBase = ((b * queries + k) * heads + head) * headDim;
          let score = 0;
          for (let c = 0; c < headDim; c += 1) score += query[qBase + c] * key[kBase + c];
          score *= scale;
          const maskIndex = input.transpose ? k * batch + b : b * queries + k;
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
        for (let c = 0; c < headDim; c += 1) {
          let acc = 0;
          for (let k = 0; k < queries; k += 1) {
            acc += weights[k] * value[((b * queries + k) * heads + head) * headDim + c];
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
  const skew = (values) => (falsify
    ? Float32Array.from(values, (v, i) => (i === (values.length >> 1) ? v * 1.01 + 0.01 : v))
    : values);

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const rows = [];
  const failures = [];
  const arm = (name, got, want, bar) => {
    const relative = relativeRms(got, want, name);
    const ok = relative <= bar;
    rows.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };

  const plans = {};
  const cases = [
    { batch: 1, queries: 8, heads: 1, headDim: 4 },
    { batch: 2, queries: 12, heads: 2, headDim: 8 },
    { batch: 2, queries: 10, heads: 4, headDim: 16 },
    { batch: 3, queries: 11, heads: 2, headDim: 32 },
    { batch: 2, queries: 9, heads: 2, headDim: 64 },
  ];

  for (const shape of cases) {
    const { batch, queries, heads, headDim } = shape;
    const random = generator(batch * 997 + queries * 31 + heads * 7 + headDim);
    const elements = batch * queries * heads * headDim;
    const base = {
      ...shape,
      query: noise(elements, 0.5, random),
      key: noise(elements, 0.5, random),
      value: noise(elements, 1, random),
      gate: noise(elements, 1, random),
      mask: new Float32Array(batch * queries).fill(1),
    };
    // A real mask: the last key of every batch row is withheld.
    for (let b = 0; b < batch; b += 1) base.mask[b * queries + queries - 1] = 0;

    const upload = (input) => ({
      query: execution.upload(skew(input.query), "q"),
      key: execution.upload(input.key, "k"),
      value: execution.upload(input.value, "v"),
      gate: execution.upload(input.gate, "g"),
      mask: execution.upload(input.mask, "mask"),
      ...(input.pairBias === undefined
        ? {} : { pairBias: execution.upload(input.pairBias, "pair") }),
    });

    const label = `D=${headDim} b=${batch} q=${queries} h=${heads}`;
    plans[`D=${headDim}`] = attentionFlashPlan(headDim, execution.maxDrawBuffers);

    {
      const tensors = upload(base);
      const { output } = attentionFlash(execution, { ...base, ...tensors });
      arm(`attention ${label}`, execution.read(output), attentionReference(base), 1e-6);
      execution.release(output);
      for (const tensor of Object.values(tensors)) execution.release(tensor);
    }

    // ...with a pair bias, which is a differently compiled kernel.
    {
      const biased = { ...base, pairBias: noise(heads * queries * queries, 2, random) };
      const tensors = upload(biased);
      const { output } = attentionFlash(execution, { ...biased, ...tensors });
      arm(`attention+bias ${label}`, execution.read(output), attentionReference(biased), 1e-6);
      execution.release(output);
      for (const tensor of Object.values(tensors)) execution.release(tensor);
    }

    // ...and the transposed mask, which is the only thing that reads `BATCH`.
    {
      const transposed = { ...base, transpose: true, mask: new Float32Array(queries * batch).fill(1) };
      for (let b = 0; b < batch; b += 1) transposed.mask[(queries - 1) * batch + b] = 0;
      const tensors = upload(transposed);
      const { output } = attentionFlash(execution, { ...transposed, ...tensors });
      arm(`attention transposed mask ${label}`, execution.read(output),
        attentionReference(transposed), 1e-6);
      execution.release(output);
      for (const tensor of Object.values(tensors)) execution.release(tensor);
    }
  }

  // --- the kernel it replaces, at its own convention (no 1/sqrt(D) scaling) --
  {
    const shape = { batch: 3, queries: 11, heads: 2, headDim: 32 };
    const { batch, queries, heads, headDim } = shape;
    const random = generator(4242);
    const elements = batch * queries * heads * headDim;
    const input = {
      ...shape,
      query: noise(elements, 0.5, random),
      key: noise(elements, 0.5, random),
      value: noise(elements, 1, random),
      gate: noise(elements, 1, random),
      mask: new Float32Array(batch * queries).fill(1),
      pairBias: noise(heads * queries * queries, 2, random),
    };
    for (let b = 0; b < batch; b += 1) input.mask[b * queries + queries - 1] = 0;
    const old = new AttentionFlashWebGL2(gl).run({ ...input, headDim });
    const tensors = {
      query: execution.upload(skew(input.query), "q"),
      key: execution.upload(input.key, "k"),
      value: execution.upload(input.value, "v"),
      gate: execution.upload(input.gate, "g"),
      mask: execution.upload(input.mask, "mask"),
      pairBias: execution.upload(input.pairBias, "pair"),
    };
    const { output } = attentionFlash(execution, { ...input, ...tensors, scale: 1 });
    arm("against the kernel it replaces, scale 1", execution.read(output), old, 1e-6);
    execution.release(output);
    for (const tensor of Object.values(tensors)) execution.release(tensor);
  }

  const stats = execution.stats();
  const red = rows.filter((row) => !row.ok).length;
  return {
    maxDrawBuffers: execution.maxDrawBuffers,
    plans,
    arms: rows,
    liveTensorsAfter: stats.live,
    programs: stats.programs,
    draws: stats.draws,
    falsify,
    verdict: falsify
      ? (red === rows.length
        ? `ok: all ${rows.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${rows.length} arms noticed a 1% perturbation`)
      : (failures.length === 0 && stats.live === 0
        ? `ok: ${rows.length} arms, no tensor left live`
        : `FAILED: ${[...failures, ...(stats.live ? [`${stats.live} tensors leaked`] : [])].join("; ")}`),
  };
}
