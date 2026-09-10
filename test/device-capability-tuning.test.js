import test from "node:test";
import assert from "node:assert/strict";
import { matrixCapabilityTuning } from "../src/runtime/device-profile.js";

const device = (...features) => ({ features: new Set(features) });

// 🔴 THE KNOBS THIS LAYER MAY SET, spelled out. A knob added here without a
// measurement is exactly the failure the priors table already had, so the list
// is asserted rather than described in a comment.
const EXPECTED = ["attentionMatrix", "attentionProjectMatrix", "matrixLinear",
  "opmMatrixContract", "stagedMatrixPrefetch"];

const f16Config = { componentType: "f16", resultComponentType: "f32", M: 16, N: 8, K: 8 };

test("a device with the units, shader-f16 and an f16 configuration gets the matrix paths", () => {
  const tuning = matrixCapabilityTuning(
    device("chromium-experimental-subgroup-matrix", "shader-f16"), [f16Config]);
  assert.deepEqual(Object.keys(tuning).sort(), EXPECTED);
  assert.ok(Object.values(tuning).every((value) => value === true));
});

test("no subgroup-matrix feature means no capability tuning at all", () => {
  assert.deepEqual(matrixCapabilityTuning(device("shader-f16"), [f16Config]), {});
});

// The kernels are written in f16; the units alone are not enough to feed them.
test("no shader-f16 means no capability tuning", () => {
  assert.deepEqual(matrixCapabilityTuning(
    device("chromium-experimental-subgroup-matrix"), [f16Config]), {});
});

test("units that offer no f16 configuration mean no capability tuning", () => {
  assert.deepEqual(matrixCapabilityTuning(
    device("chromium-experimental-subgroup-matrix", "shader-f16"),
    [{ componentType: "u8", resultComponentType: "u32", M: 16, N: 16, K: 32 }]), {});
});

test("a device that reports no configurations at all is safe to ask", () => {
  assert.deepEqual(matrixCapabilityTuning(
    device("chromium-experimental-subgroup-matrix", "shader-f16")), {});
  assert.deepEqual(matrixCapabilityTuning(undefined, []), {});
});
