/**
 * A delta bundle read as though it were whole.
 *
 * 🔴 THE THREE CLASSES ARE NOT INTERCHANGEABLE, which is the whole risk here:
 * a tensor that should have been ADDED and is taken whole gives a model built
 * out of differences, and one that should have been taken whole and is added
 * gives the base plus itself. Both load, both fold, and neither says anything.
 */
import { strict as assert } from "node:assert";
import { test } from "node:test";
import { DeltaTensorStore } from "../src/bundles/delta-tensor-store.js";

const store = (tensors, extra = {}) => ({
  manifest: {
    tensors: Object.fromEntries(Object.entries(tensors)
      .map(([name, values]) => [name, { shape: [values.length], dtype: "float32" }])),
    bundle: { model: "base" },
    ...extra,
  },
  open: async() => {},
  shape: (name) => [tensors[name].length],
  tensor: async(name) => Float32Array.from(tensors[name]),
  tensorAsFloat16: async(name) => Float16Array.from(tensors[name]),
  tensorSource: (name) => ({ record: { shape: [tensors[name].length] }, buffer: null, byteOffset: 0 }),
});

const base = () => store({ added: [1, 2, 3], whole: [10, 20], geometry: [7], gone: [0, 0] }, {
  templateEmbedding: { parameters: { attention: { key_w: "gone" } } },
});
const delta = (header) => {
  const inner = store({ added: [0.5, -1, 0.25], whole: [11, 21] });
  inner.manifest.delta = header;
  return inner;
};
const header = { addTo: ["added"], whole: ["whole"], absent: ["gone"], baseModel: "base" };

test("an added tensor is the base plus the delta", async() => {
  const combined = new DeltaTensorStore(base(), delta(header));
  assert.deepEqual([...await combined.tensor("added")], [1.5, 1, 3.25]);
});

test("a whole tensor is the delta's alone, and the base's copy is ignored", async() => {
  const combined = new DeltaTensorStore(base(), delta(header));
  assert.deepEqual([...await combined.tensor("whole")], [11, 21]);
});

test("a tensor the header does not mention comes from the base", async() => {
  const combined = new DeltaTensorStore(base(), delta(header));
  assert.deepEqual([...await combined.tensor("geometry")], [7]);
});

test("an absent tensor is gone from the manifest and from reads", async() => {
  const combined = new DeltaTensorStore(base(), delta(header));
  assert.equal(combined.manifest.tensors.gone, undefined);
  await assert.rejects(() => combined.tensor("gone"), /missing tensor gone/);
});

test("a section whose tensors are all absent is removed, so nothing builds the stage", () => {
  const combined = new DeltaTensorStore(base(), delta(header));
  assert.equal(combined.manifest.templateEmbedding, undefined);
});

test("a section only PARTLY absent is refused - that is a packing fault", () => {
  const wider = base();
  wider.manifest.templateEmbedding.parameters.attention.query_w = "whole";
  assert.throws(() => new DeltaTensorStore(wider, delta(header)), /only partly absent/);
});

test("it offers no tensorSource, which is what routes reads through the reconstruction", () => {
  const combined = new DeltaTensorStore(base(), delta(header));
  assert.equal(combined.tensorSource, undefined);
});

test("a bundle with no delta header is refused rather than read as one", () => {
  assert.throws(() => new DeltaTensorStore(base(), store({ a: [1] })), /carries no `delta` header/);
});
