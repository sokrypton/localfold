// One module of ESMFold2's trunk at a time, against its own recorded answer.
//
//     node tools/check-esmfold2-modules.js
//
// 🔴 THE WHOLE-TRUNK CHECK COSTS 77 SECONDS A LOOP AND SAYS "SOMEWHERE IN 24
// BLOCKS". This runs block zero's three sub-modules against the values the
// native model produced for them, which is a second each - so a wrong
// convention is found by looking rather than by bisecting.
//
// 🔴 AND IT SWEEPS THE CONVENTIONS RATHER THAN ASSERTING ONE. ESMFold2 packs
// the triangle's double width BLOCKED and AF3 reads it INTERLEAVED; which half
// is the signal and which the gate, and which of each pair is `a`, are four
// independent binary choices that all conform in shape. Getting one wrong
// returns a plausible tensor, so the arms are enumerated and the one that
// matches is reported.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";

import { transition, triangleMultiplication } from "../src/af3/pairformer-reference.js";

const ROOT = new URL("..", import.meta.url).pathname;
const dump = JSON.parse(readFileSync(
  process.argv[2] ?? join(ROOT, "oracle-dumps", "esmfold2-trunk-40.json"), "utf8"));

const CHANNELS = 256;
const tokens = dump.shapes.pair[1];
const pairs = tokens * tokens;
const mask = new Float32Array(pairs).fill(1);

const relative = (got, want) => {
  let error = 0, total = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i] - want[i];
    error += d * d;
    total += want[i] * want[i];
  }
  return Math.sqrt(error / total);
};

// The raw ESMFold2 tensors for block 0, read straight from the checkpoint by
// tools/esmc/dump-esmfold2-modules.py rather than through the converter - the
// converter is the thing under test.
const raw = JSON.parse(readFileSync(join(ROOT, "oracle-dumps",
  "esmfold2-block0-weights.json"), "utf8"));
const tensor = (name) => Float32Array.from(raw[name]);
const shape = (name) => raw.shapes[name];

/** (out, in) -> (in, out). */
function transpose(values, rows, columns) {
  const out = new Float32Array(rows * columns);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < columns; c += 1) out[c * rows + r] = values[r * columns + c];
  }
  return out;
}

/** Weave two (in, c) halves into (in, 2c): channel ch at 2*ch and 2*ch + 1. */
function interleave(left, right, inner, channels) {
  const out = new Float32Array(inner * channels * 2);
  for (let i = 0; i < inner; i += 1) {
    for (let c = 0; c < channels; c += 1) {
      out[i * channels * 2 + c * 2] = left[i * channels + c];
      out[i * channels * 2 + c * 2 + 1] = right[i * channels + c];
    }
  }
  return out;
}

function half(values, inner, width, which) {
  const out = new Float32Array(inner * width);
  for (let i = 0; i < inner; i += 1) {
    for (let c = 0; c < width; c += 1) out[i * width + c] = values[i * width * 2 + which * width + c];
  }
  return out;
}

function triangleArm(prefix, { signalFirst, swapPair }) {
  const bundleShape = shape(`${prefix}._engine.proj_bundle.weight`);   // (4c, c)
  const bundle = transpose(tensor(`${prefix}._engine.proj_bundle.weight`),
    bundleShape[0], bundleShape[1]);                                   // (c, 4c)
  const signal = half(bundle, CHANNELS, CHANNELS * 2, signalFirst ? 0 : 1);
  const gates = half(bundle, CHANNELS, CHANNELS * 2, signalFirst ? 1 : 0);
  const left = (values) => half(values, CHANNELS, CHANNELS, swapPair ? 1 : 0);
  const right = (values) => half(values, CHANNELS, CHANNELS, swapPair ? 0 : 1);
  const square = (name) => {
    const s = shape(name);
    return transpose(tensor(name), s[0], s[1]);
  };
  return {
    leftNormInputScale: tensor(`${prefix}._engine.norm_start.weight`),
    leftNormInputOffset: tensor(`${prefix}._engine.norm_start.bias`),
    centerNormScale: tensor(`${prefix}._engine.norm_mix.weight`),
    centerNormOffset: tensor(`${prefix}._engine.norm_mix.bias`),
    outputProjection: square(`${prefix}._engine.proj_emit.weight`),
    gatingLinear: square(`${prefix}._engine.proj_gate.weight`),
    projection: interleave(left(signal), right(signal), CHANNELS, CHANNELS),
    gate: interleave(left(gates), right(gates), CHANNELS, CHANNELS),
  };
}

console.log(`${dump.esmfold2}: ${tokens} tokens, block 0's modules\n`);

// The transition first: it has one convention, not four.
{
  const w12Shape = shape("pair_transition.ffn.w12.weight");
  const w3Shape = shape("pair_transition.ffn.w3.weight");
  const weights = {
    inputLayerNormScale: tensor("pair_transition.norm.weight"),
    inputLayerNormOffset: tensor("pair_transition.norm.bias"),
    transition1: transpose(tensor("pair_transition.ffn.w12.weight"), w12Shape[0], w12Shape[1]),
    transition2: transpose(tensor("pair_transition.ffn.w3.weight"), w3Shape[0], w3Shape[1]),
  };
  const input = Float32Array.from(dump.block0.pair_transition.input);
  const recorded = Float32Array.from(dump.block0.pair_transition.output);
  // 🔴 pair_transition RETURNS ITS RESIDUAL AND THE TWO TRIANGLES RETURN THEIR
  // DELTA, IN THE SAME BLOCK. `x + ffn(norm(x))` against `proj_emit(...)`, so
  // the recorded output means a different thing per module and the whole
  // arithmetic reads as wrong (0.90) rather than as offset. rms(output - input)
  // is 1.48 against rms(output) 12.3, which is what said so.
  const want = recorded.map((value, i) => value - input[i]);
  const got = transition(input, pairs, CHANNELS, weights);
  const score = relative(got, want);
  console.log(`  pair_transition            relRMS ${score.toExponential(3)}`
    + `${score < 1e-3 ? "   <-- MATCH" : ""}`);
  // The control: against the recorded output as-is, which is what a converter
  // that missed the residual would be scored against.
  console.log(`  ...against the raw output  relRMS `
    + `${relative(got, recorded).toExponential(3)}   (the residual, not a fault)`);
}

for (const [label, prefix, direction] of [
  ["tri_mul_out -> outgoing", "tri_mul_out", "outgoing"],
  ["tri_mul_out -> incoming", "tri_mul_out", "incoming"],
  ["tri_mul_in  -> incoming", "tri_mul_in", "incoming"],
  ["tri_mul_in  -> outgoing", "tri_mul_in", "outgoing"],
]) {
  const input = Float32Array.from(dump.block0[prefix].input);
  const want = Float32Array.from(dump.block0[prefix].output);
  for (const signalFirst of [true, false]) {
    for (const swapPair of [true, false]) {
      const weights = triangleArm(prefix, { signalFirst, swapPair });
      const got = triangleMultiplication(input, mask, tokens, CHANNELS, direction, weights);
      const score = relative(got, want);
      const note = score < 1e-3 ? "   <-- MATCH" : "";
      console.log(`  ${label}  signal${signalFirst ? "First" : "Second"} `
        + `${swapPair ? "swapped" : "inorder"}  relRMS ${score.toExponential(3)}${note}`);
    }
  }
}
