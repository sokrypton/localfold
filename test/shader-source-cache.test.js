import test from "node:test";
import assert from "node:assert/strict";
import {
  shaderSource, shaderSourceSet, setShaderSourceVerification, shaderSourceVerification,
} from "../src/runtime/shader-source-cache.js";

// A device is only ever a WeakMap key here, so a bare object stands in for one.
const device = () => ({});

test("a source is built once per device and key", () => {
  const one = device();
  let built = 0;
  const build = () => { built += 1; return "@compute fn main() {}"; };
  assert.equal(shaderSource(one, "k", build), "@compute fn main() {}");
  assert.equal(shaderSource(one, "k", build), "@compute fn main() {}");
  assert.equal(built, 1);
});

test("two devices do not share an answer", () => {
  const one = device();
  const two = device();
  assert.equal(shaderSource(one, "k", () => "a"), "a");
  assert.equal(shaderSource(two, "k", () => "b"), "b");
  assert.equal(shaderSource(one, "k", () => "c"), "a");
});

// 🔴 THE MEMO MAKES ComputePipelineCache's COLLISION CHECK VACUOUS - it compares
// the cached source against itself - so the check that a key names everything
// its source depends on has to live here instead. This is that check failing.
test("verification catches a key that does not name its source", () => {
  const one = device();
  const before = shaderSourceVerification();
  setShaderSourceVerification(true);
  try {
    let n = 0;
    const drifting = () => `const N = ${n += 1};`;
    assert.equal(shaderSource(one, "drifts", drifting), "const N = 1;");
    assert.throws(() => shaderSource(one, "drifts", drifting),
      /does not name its source/);
  } finally {
    setShaderSourceVerification(before);
  }
});

test("a set is built once and verified as a whole", () => {
  const one = device();
  let built = 0;
  const build = () => { built += 1; return { a: "x", b: "y" }; };
  assert.deepEqual(shaderSourceSet(one, "s", build), { a: "x", b: "y" });
  assert.deepEqual(shaderSourceSet(one, "s", build), { a: "x", b: "y" });
  assert.equal(built, 1);
  const before = shaderSourceVerification();
  setShaderSourceVerification(true);
  try {
    assert.throws(() => shaderSourceSet(one, "s", () => ({ a: "x", b: "z" })),
      /does not name its sources/);
  } finally {
    setShaderSourceVerification(before);
  }
});
