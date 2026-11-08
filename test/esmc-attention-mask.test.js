// cpu/esmc/tower.js's attention over a packed complex: with `sequenceId`, each chain attends only to itself, so the
// packed run's rows are each chain's run alone - the WebGPU tower's mask (webgpu/esmc/tower.js). Without it the
// reference let a complex's chains attend to each other (PR #7, lhallee).
import test from "node:test";
import assert from "node:assert/strict";
import { esmcAttention } from "../cpu/esmc/tower.js";

test("a packed complex attends chain by chain, exactly as each chain alone", () => {
  const heads = 2, headDim = 4, lengths = [3, 5], rows = 8;
  let seed = 7;
  const draw = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648) - 0.5;
  const tensor = () => Float32Array.from({ length: rows * heads * headDim }, draw);
  const [q, k, v] = [tensor(), tensor(), tensor()];
  const sequenceId = [0, 0, 0, 1, 1, 1, 1, 1];
  const packed = esmcAttention(q, k, v, rows, heads, headDim, sequenceId);
  const width = heads * headDim;
  let start = 0;
  for (const length of lengths) {
    const part = (t) => t.slice(start * width, (start + length) * width);
    const alone = esmcAttention(part(q), part(k), part(v), length, heads, headDim);
    assert.deepEqual(packed.slice(start * width, (start + length) * width), alone);
    start += length;
  }
  const unmasked = esmcAttention(q, k, v, rows, heads, headDim);
  assert.notDeepEqual(unmasked, packed, "the mask changes nothing, so this test reads nothing");
});
