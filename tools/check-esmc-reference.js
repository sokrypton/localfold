// Does src/esmc/tower-reference.js compute ESM-C?
//
//     python3 tools/esmc/dump-esmc-oracle.py --sequence-length 59
//     node tools/check-esmc-reference.js
//
// Against an ORACLE, not against a second reading of the same idea: the dump
// comes from tools/esmc/esmc_forward.py, which agrees with transformers' own
// EsmcForMaskedLM to 2.1e-6 on every hidden state and with the esm package's
// own language_model module to 3.2e-7 on the pair representation.
//
// 🔴 IT CHECKS ONE BLOCK'S INSIDES BEFORE IT CHECKS THE TOWER. A tower that
// disagrees at state 36 says nothing about where; the four matmul INPUTS of
// block 0 separate a wrong LayerNorm from a wrong RoPE from a wrong SwiGLU
// split, and they are the difference between a fix and a bisection.
//
// 🔴 AND THE MANIFEST'S PAIRING IS ASSERTED. A tower is interchangeable between
// ESMFold2 releases and a shim is not, so a bundle whose shim came from a
// different folding model than the dump's is refused rather than measured.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";

import {
  applyRope, esmcAttention, esmcBlock, layerNorm, linear, sequenceIds, shimPair,
  shimSingle, towerStates,
} from "../src/esmc/tower-reference.js";

const ROOT = new URL("..", import.meta.url).pathname;
const bundleDirectory = process.argv[2] ?? join(ROOT, "model-esmc-600m-f32");
const dumpPath = process.argv[3] ?? join(ROOT, "oracle-dumps", "esmc-59.json");

function loadBundle(directory) {
  const manifest = JSON.parse(readFileSync(join(directory, "manifest.json"), "utf8"));
  const files = new Map();
  const weights = {};
  for (const [name, record] of Object.entries(manifest.tensors)) {
    if (record.dtype !== "float32") {
      throw new Error(`${name} is ${record.dtype}; this checker reads the float32 export`);
    }
    if (!files.has(record.file)) {
      files.set(record.file, readFileSync(join(directory, record.file)));
    }
    const buffer = files.get(record.file);
    const count = record.shape.reduce((a, b) => a * b, 1);
    const offset = record.byteOffset ?? 0;
    // A fresh copy: the shard buffer's byteOffset is not guaranteed aligned for
    // a Float32Array view, and a silent misalignment here would look like a
    // wrong kernel later.
    const values = new Float32Array(count);
    for (let i = 0; i < count; i += 1) values[i] = buffer.readFloatLE(offset + i * 4);
    weights[name] = values;
  }
  return { manifest, weights };
}

const relative = (got, want) => {
  let error = 0, total = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i] - want[i];
    error += d * d;
    total += want[i] * want[i];
  }
  return Math.sqrt(error / total);
};

let failures = 0;
function report(label, got, want, bound) {
  if (got.length !== want.length) {
    console.log(`  ${label.padEnd(30)} LENGTH ${got.length} against ${want.length}`);
    failures += 1;
    return;
  }
  const score = relative(got, want);
  const ok = score <= bound;
  if (!ok) failures += 1;
  console.log(`  ${label.padEnd(30)} relRMS ${score.toExponential(2)}`
    + `   bound ${bound.toExponential(0)}   ${ok ? "ok" : "FAILED"}`);
}

const dump = JSON.parse(readFileSync(dumpPath, "utf8"));
const { manifest, weights } = loadBundle(bundleDirectory);

if (manifest.languageModel?.shim !== dump.esmfold2
  || manifest.languageModel?.tower !== dump.esmc) {
  throw new Error(`bundle is ${manifest.languageModel?.tower} + `
    + `${manifest.languageModel?.shim}, dump is ${dump.esmc} + ${dump.esmfold2};`
    + " the shim is per folding model and these are not the same pair");
}

const dims = {
  model: dump.dims.model,
  layers: dump.dims.layers,
  heads: dump.dims.heads,
  ffn: dump.shapes["ffn.fc2_weight"][1],
  pair: dump.shapes.single[1],
  residualScale: dump.dims.residualScale,
};
const ids = Int32Array.from(dump.tokens);
const rows = ids.length;
console.log(`${dump.esmc} + ${dump.esmfold2}: ${rows} tokens, `
  + `${dims.layers} x ${dims.model}, ${dims.heads} heads, ffn ${dims.ffn}`);

// The token ids the reference derives must be the ones the dump recorded, or
// every number below compares two different proteins.
const derived = sequenceIds(dump.sequence);
report("sequenceIds", Float32Array.from(derived), Float32Array.from(dump.tokens), 0);

const embedded = new Float32Array(rows * dims.model);
const table = weights["embed/weights"];
for (let row = 0; row < rows; row += 1) {
  embedded.set(table.subarray(ids[row] * dims.model, (ids[row] + 1) * dims.model),
    row * dims.model);
}
report("embedding lookup", embedded, Float32Array.from(dump.embedded), 1e-6);

console.log("\nblock 0, by the input each matmul receives:");
const attnNormed = layerNorm(embedded.slice(), rows, dims.model,
  weights["blocks/0/attn_norm/scale"], weights["blocks/0/attn_norm/offset"]);
report("attn_norm -> qkv", attnNormed,
  Float32Array.from(dump.block0["attn.layernorm_qkv.weight"]), 1e-5);

const first = esmcBlock(embedded, rows, weights, dims, 0);
report("block 0 output", first, Float32Array.from(dump.block0Output), 1e-5);

console.log("\nthe tower:");
const states = towerStates(ids, weights, dims);
if (states.length !== dims.layers + 1) {
  throw new Error(`${states.length} states, expected ${dims.layers + 1}`);
}
for (const key of Object.keys(dump.states)) {
  report(`state ${key}`, states[Number(key)], Float32Array.from(dump.states[key]), 2e-5);
}

console.log("\nthe shim:");
const inner = states.map((state) => state.subarray(dims.model, state.length - dims.model));
const single = shimSingle(inner, rows - 2, weights, dims);
report("mixed single", single, Float32Array.from(dump.single), 5e-5);
const pair = shimPair(single, rows - 2, weights, dims);
report("pair representation", pair, Float32Array.from(dump.pair), 1e-4);

// 🔴 EVERY ARM PASSING ON THE FIRST RUN IS NOT EVIDENCE UNTIL THE CHECKER HAS
// BEEN SHOWN TO FAIL. Both conventions this file exists to pin - split-halves
// against interleaved RoPE, and QK LayerNorm over d_model against per head -
// produce a tensor of exactly the right shape, so a checker that cannot tell
// them apart reports success for either. These arms rebuild block 0 with the
// WRONG reading of one step and share every other step with the reference, and
// each one must land far outside the bound the real arm passed.
console.log("\nthe conventions this pins, read the other way (each must FAIL):");

function mutatedBlock(x, variant) {
  const { model, heads, residualScale, ffn } = dims;
  const headDim = model / heads;
  const at = (leaf) => weights[`blocks/0/${leaf}`];
  const normed = layerNorm(x.slice(), rows, model,
    at("attn_norm/scale"), at("attn_norm/offset"));
  const projected = linear(normed, rows, model, at("qkv/weights"), 3 * model);
  const query = new Float32Array(rows * model);
  const key = new Float32Array(rows * model);
  const value = new Float32Array(rows * model);
  for (let row = 0; row < rows; row += 1) {
    const source = row * 3 * model;
    query.set(projected.subarray(source, source + model), row * model);
    key.set(projected.subarray(source + model, source + 2 * model), row * model);
    value.set(projected.subarray(source + 2 * model, source + 3 * model), row * model);
  }
  if (variant === "qk-per-head") {
    // The same scale vector, applied per head instead of across the width.
    for (const side of [["q_norm/scale", query], ["k_norm/scale", key]]) {
      const scale = at(side[0]);
      for (let head = 0; head < heads; head += 1) {
        const slice = new Float32Array(rows * headDim);
        for (let row = 0; row < rows; row += 1) {
          for (let d = 0; d < headDim; d += 1) {
            slice[row * headDim + d] = side[1][(row * heads + head) * headDim + d];
          }
        }
        layerNorm(slice, rows, headDim, scale.subarray(head * headDim, (head + 1) * headDim), null);
        for (let row = 0; row < rows; row += 1) {
          for (let d = 0; d < headDim; d += 1) {
            side[1][(row * heads + head) * headDim + d] = slice[row * headDim + d];
          }
        }
      }
    }
  } else {
    layerNorm(query, rows, model, at("q_norm/scale"), null);
    layerNorm(key, rows, model, at("k_norm/scale"), null);
  }
  const rotate = (values) => {
    if (variant === "no-rope") return values;
    if (variant === "rope-interleaved") {
      for (let row = 0; row < rows; row += 1) {
        for (let d = 0; d < headDim; d += 2) {
          const frequency = row / Math.pow(10000, d / headDim);
          const cos = Math.cos(frequency), sin = Math.sin(frequency);
          for (let head = 0; head < heads; head += 1) {
            const slot = (row * heads + head) * headDim + d;
            const a = values[slot], b = values[slot + 1];
            values[slot] = a * cos - b * sin;
            values[slot + 1] = a * sin + b * cos;
          }
        }
      }
      return values;
    }
    return applyRope(values, rows, heads, headDim);
  };
  rotate(query); rotate(key);
  const context = esmcAttention(query, key, value, rows, heads, headDim);
  const attention = linear(context, rows, model, at("attn_out/weights"), model);
  const afterAttention = new Float32Array(rows * model);
  for (let i = 0; i < afterAttention.length; i += 1) {
    afterAttention[i] = x[i] + attention[i] / residualScale;
  }
  const ffnNormed = layerNorm(afterAttention.slice(), rows, model,
    at("ffn_norm/scale"), at("ffn_norm/offset"));
  const hidden = linear(ffnNormed, rows, model, at("fc1/weights"), 2 * ffn);
  const gated = new Float32Array(rows * ffn);
  for (let row = 0; row < rows; row += 1) {
    const source = row * 2 * ffn, destination = row * ffn;
    for (let c = 0; c < ffn; c += 1) {
      const a = hidden[source + c], b = hidden[source + ffn + c];
      // "swiglu-swapped" gates the other half: silu(b) * a.
      gated[destination + c] = variant === "swiglu-swapped"
        ? (b / (1 + Math.exp(-b))) * a
        : (a / (1 + Math.exp(-a))) * b;
    }
  }
  const out = linear(gated, rows, ffn, at("fc2/weights"), model);
  const result = new Float32Array(rows * model);
  for (let i = 0; i < result.length; i += 1) {
    result[i] = afterAttention[i] + out[i] / residualScale;
  }
  return result;
}

const oracleBlock = Float32Array.from(dump.block0Output);
for (const variant of ["rope-interleaved", "no-rope", "qk-per-head", "swiglu-swapped"]) {
  const score = relative(mutatedBlock(embedded, variant), oracleBlock);
  const discriminates = score > 1e-3;
  if (!discriminates) failures += 1;
  console.log(`  ${variant.padEnd(30)} relRMS ${score.toExponential(2)}`
    + `   ${discriminates ? "caught" : "NOT CAUGHT - the check is blind to this"}`);
}

console.log(failures === 0 ? "\nall arms within bound"
  : `\n${failures} arm(s) out of bound`);
process.exit(failures === 0 ? 0 : 1);
