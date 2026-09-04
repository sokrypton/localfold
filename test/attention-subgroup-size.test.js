/**
 * Which devices the subgroup attention kernels may be built for.
 *
 * 🔴 THE FAILURE THIS GUARDS IS FATAL, NOT SLOW. Every subgroup kernel pins
 * `@subgroup_size(32)`, and a device that reports a 16-lane range rejects the
 * pipeline outright - reported from a Pixel as "The subgroup_size attribute
 * (32) is not in the allowed range ([16, 16])" while building
 * block:attention:flash-subgroup-key32. The fold stops there; it does not fall
 * back. Adreno and Mali parts have both subgroup features, so checking for the
 * features alone let them through.
 */
import { describe, expect, it } from "./harness.js";
import {
  allowsAttentionSubgroupSize, buildAttentionFlashKernel, selectAttentionFlashKernel,
  supportsAttentionSubgroups,
} from "../src/evoformer/attention.js";

const deviceWith = (subgroupMinSize, subgroupMaxSize, features = ["subgroups", "subgroup-size-control"]) => ({
  features: new Set(features),
  adapterInfo: subgroupMinSize === undefined && subgroupMaxSize === undefined
    ? {} : { subgroupMinSize, subgroupMaxSize },
  // Generous, so the size range is the only thing under test.
  limits: { maxComputeInvocationsPerWorkgroup: 1024, maxComputeWorkgroupStorageSize: 32_768 },
});

describe("the subgroup attention kernels' size requirement", () => {
  it("no longer picks a subgroup kernel by default, whatever the device allows", () => {
    // The register-resident kernel is faster on every shape measured here; see
    // selectAttentionFlashKernel and tools/gpu/check-attention-variants.js.
    expect(selectAttentionFlashKernel(deviceWith(32, 32), 32, "auto").cacheKey)
      .toBe("attention:flash-registers-32-f32");
  });

  // 🔴 THE PRECISION IS IN THE CACHE KEY BECAUSE IT CHANGES THE SHADER. Two
  // devices in one process - the checkers run both - would otherwise share a
  // compiled pipeline whose staged chunk is the wrong width for one of them.
  it("stages the key and value in f16 only where the device has shader-f16", () => {
    const withF16 = deviceWith(32, 32, ["subgroups", "subgroup-size-control", "shader-f16"]);
    expect(selectAttentionFlashKernel(withF16, 32, "auto").cacheKey)
      .toBe("attention:flash-registers-32-chunk16");
    expect(selectAttentionFlashKernel(deviceWith(32, 32), 32, "auto").cacheKey)
      .toBe("attention:flash-registers-32-f32");
    // ...and an explicit request still wins over what the device offers, which
    // is what lets the differential checker hold the f32 kernel to 1e-5.
    expect(selectAttentionFlashKernel(withF16, 32, "auto", "f32").cacheKey)
      .toBe("attention:flash-registers-32-f32");
  });

  it("accepts a device whose range contains 32", () => {
    expect(allowsAttentionSubgroupSize(deviceWith(32, 32))).toBe(true);
    expect(allowsAttentionSubgroupSize(deviceWith(16, 64))).toBe(true);
    expect(supportsAttentionSubgroups(deviceWith(32, 32))).toBe(true);
  });

  it("refuses the Pixel's [16, 16], where the pipeline would fail to build", () => {
    expect(allowsAttentionSubgroupSize(deviceWith(16, 16))).toBe(false);
    expect(supportsAttentionSubgroups(deviceWith(16, 16))).toBe(false);
  });

  it("refuses a range that is entirely above 32 as well", () => {
    expect(allowsAttentionSubgroupSize(deviceWith(64, 128))).toBe(false);
  });

  it("treats an unreported range as allowed, so working devices keep the path", () => {
    // 🔴 THE FAST PATH IS NOT GIVEN UP ON A MISSING FIELD. A browser that
    // exposes `subgroups` without the sizes has been running these kernels
    // correctly; the range is what the error message quotes, so a browser that
    // can raise that error can also report it.
    expect(allowsAttentionSubgroupSize(deviceWith(undefined, undefined))).toBe(true);
    expect(supportsAttentionSubgroups(deviceWith(undefined, undefined))).toBe(true);
  });

  it("still needs both features and a 32-wide head", () => {
    expect(supportsAttentionSubgroups(deviceWith(32, 32), 64)).toBe(false);
    expect(supportsAttentionSubgroups(deviceWith(32, 32, ["subgroups"]))).toBe(false);
  });
});

describe("building the flash pipeline when the device refuses it", () => {
  const device = {
    features: new Set(["subgroups", "subgroup-size-control"]),
    // 🔴 REPORTS NO RANGE, which is the case introspection cannot catch.
    adapterInfo: {},
    limits: { maxComputeInvocationsPerWorkgroup: 1024, maxComputeWorkgroupStorageSize: 32_768 },
  };

  it("falls back to the register kernel when the subgroup pipeline is rejected", async () => {
    // 🔴 REQUESTED EXPLICITLY, because `auto` no longer picks a subgroup kernel
    // at all - the register one is faster on every shape measured. The fallback
    // still has to work for a caller that names one, and for a head dimension
    // the register kernel cannot take.
    const asked = [];
    const execution = { pipelines: { get: async (key) => {
      asked.push(key);
      if (key.includes("subgroup")) {
        throw new Error("The subgroup_size attribute (32) is not in the allowed range ([16, 16]).");
      }
      return { key };
    } } };
    const built = await buildAttentionFlashKernel(execution, device, 32, "subgroup-key32");
    expect(built.kernel.cacheKey).toBe("attention:flash-registers-32-f32");
    expect(built.kernel.queryTile).toBe(64);
    expect(asked.length).toBe(2);
  });

  it("does not swallow a failure that is not about subgroups", async () => {
    const execution = { pipelines: { get: async () => { throw new Error("out of memory"); } } };
    let message = "no error";
    try {
      await buildAttentionFlashKernel(execution, device, 32, "subgroup-key32");
    } catch (error) {
      message = error.message;
    }
    expect(message).toMatch(/out of memory/);
  });
});
