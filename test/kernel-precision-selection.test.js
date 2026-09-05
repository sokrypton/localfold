/**
 * Every kernel choice that depends on shader-f16 returns a COMPLETE choice.
 *
 * 🔴 THE BUG THIS EXISTS FOR IS SILENT ON THE MACHINE THAT HAS f16. Four
 * kernels now pick between an f32 and an f16 shader, and each pick is a tuple -
 * a tile, the element its arithmetic works in, and the element its weights are
 * stored in - because the dispatch divides by the tile and the packer fills the
 * buffer. `chooseLinearKernel` shipped for an hour with its forced-f16 branch
 * returning two of the three, so the weights were packed f32 and read as f16:
 * half the values at twice the stride, which is a wrong answer and not an
 * error. It was found because a checker's arm did not move when the default
 * did, which is luck.
 *
 * These are pure functions of a device's feature set, so they cost nothing to
 * hold to their contract on the CPU lane - including the branch a device
 * WITHOUT shader-f16 takes, which is the one no GPU checker here can run.
 */
import { describe, expect, it } from "./harness.js";
import {
  chooseLinearKernel, LINEAR_TILE, LINEAR_TILE_WIDE,
} from "../src/evoformer/transition.js";
import {
  selectAttentionProjectKernel, selectAttentionOutputKernel, selectAttentionFlashKernel,
  ATTENTION_PROJECT_TILE, ATTENTION_PROJECT_TILE_F16,
  ATTENTION_OUTPUT_TILE, ATTENTION_OUTPUT_TILE_F16,
} from "../src/evoformer/attention.js";

const withF16 = { features: new Set(["shader-f16"]), limits: {}, adapterInfo: {} };
const withoutF16 = { features: new Set(), limits: {}, adapterInfo: {} };

describe("the linear kernel's choice", () => {
  it("names a tile, an element and a weight format in every branch", () => {
    const shapes = [
      { rows: 59, columns: 384 },       // a structure module's single track
      { rows: 30208, columns: 1024 },   // a deep MSA transition
      { rows: 8192, columns: 1024 },    // one chunk of the same
    ];
    for (const device of [withF16, withoutF16]) {
      for (const requested of ["auto", "f32", "f16"]) {
        if (requested === "f16" && device === withoutF16) continue;
        for (const shape of shapes) {
          const choice = chooseLinearKernel({ ...shape, device, requested });
          expect(typeof choice.precision).toBe("string");
          expect(typeof choice.weightPrecision).toBe("string");
          expect(choice.tile === LINEAR_TILE || choice.tile === LINEAR_TILE_WIDE).toBe(true);
          // 🔴 THE ARITHMETIC AND THE WEIGHTS MOVE TOGETHER OR THE PACKER AND
          // THE SHADER DISAGREE. Nothing here mixes them, and this is what
          // caught the branch that did.
          expect(choice.weightPrecision).toBe(choice.precision);
        }
      }
    }
  });

  it("never asks for f16 on a device that has not got it", () => {
    for (const shape of [{ rows: 30208, columns: 1024 }, { rows: 59, columns: 384 }]) {
      const choice = chooseLinearKernel({ ...shape, device: withoutF16 });
      expect(choice.precision).toBe("f32");
      expect(choice.weightPrecision).toBe("f32");
    }
  });

  it("keeps the narrow tile for f16 and the measured wide/narrow rule for f32", () => {
    // The wide tile halves weight traffic at the cost of workgroups; f16 halves
    // the traffic anyway, so the narrow tile keeps its occupancy as well.
    expect(chooseLinearKernel({ rows: 30208, columns: 1024, device: withF16 }).tile)
      .toBe(LINEAR_TILE);
    expect(chooseLinearKernel({ rows: 30208, columns: 1024, device: withoutF16 }).tile)
      .toBe(LINEAR_TILE_WIDE);
    // ...and a small shape stays f32 either way: there is not enough work to
    // hide the conversion. 24 workgroups measured 0.188 ms against 0.237.
    expect(chooseLinearKernel({ rows: 59, columns: 384, device: withF16 }).precision)
      .toBe("f32");
  });
});

describe("the attention kernels' choices", () => {
  it("give the projection a tile that matches its precision", () => {
    expect(selectAttentionProjectKernel(withF16).tile).toBe(ATTENTION_PROJECT_TILE_F16);
    expect(selectAttentionProjectKernel(withoutF16).tile).toBe(ATTENTION_PROJECT_TILE);
    expect(selectAttentionProjectKernel(withF16, "f32").tile).toBe(ATTENTION_PROJECT_TILE);
    // 🔴 THE TILE IS IN THE CACHE KEY WITH THE PRECISION, because the dispatch
    // divides by it: a key that named only the precision would hand one tile's
    // pipeline to the other's grid and leave rows unprojected.
    for (const device of [withF16, withoutF16]) {
      const chosen = selectAttentionProjectKernel(device);
      expect(chosen.cacheKey.includes(chosen.precision)).toBe(true);
      expect(chosen.cacheKey.includes(String(chosen.tile.rowsPerLane * chosen.tile.lanesY)))
        .toBe(true);
    }
  });

  it("give the output projection the same treatment, residual or not", () => {
    for (const residual of [false, true]) {
      expect(selectAttentionOutputKernel(withF16, residual).tile).toBe(ATTENTION_OUTPUT_TILE_F16);
      expect(selectAttentionOutputKernel(withoutF16, residual).tile).toBe(ATTENTION_OUTPUT_TILE);
    }
    // ...and the residual form gets its own key, or one would serve both.
    expect(selectAttentionOutputKernel(withF16, true).cacheKey
      === selectAttentionOutputKernel(withF16, false).cacheKey).toBe(false);
  });

  it("put the flash kernel's staged precision in its key", () => {
    expect(selectAttentionFlashKernel(withF16, 32, "auto").cacheKey)
      .toBe("attention:flash-registers-32-chunk16");
    expect(selectAttentionFlashKernel(withoutF16, 32, "auto").cacheKey)
      .toBe("attention:flash-registers-32-f32");
  });

  it("refuse f16 on a device without it rather than building a shader it cannot compile", () => {
    expect(() => selectAttentionProjectKernel(withoutF16, "f16")).toThrow();
  });

  it("keys a packed kernel apart from the f32 one it would otherwise share", () => {
    // 🔴 A SHARED KEY HERE IS FOUR BINDINGS OF THE RIGHT LENGTH HOLDING TWICE
    // THE VALUES THEY SHOULD, which nothing validates and nothing throws on.
    const plain = selectAttentionFlashKernel(withF16, 32);
    const packed = selectAttentionFlashKernel(withF16, 32, "auto", "auto",
      { input: "f16", output: "f16" });
    expect(plain.cacheKey).toBe("attention:flash-registers-32-chunk16");
    expect(packed.cacheKey).toBe("attention:flash-registers-32-chunk16-storagef16f16");
    expect(plain.packedStorageSupported).toBe(true);
    // Only the register kernel reads them packed. A head that does not divide
    // into vec4s falls to the portable kernel, which does not - and the caller
    // checks this flag before it allocates, so that device keeps f32 storage.
    expect(selectAttentionFlashKernel(withF16, 30).packedStorageSupported).toBe(false);
    // 🔴 THE SHADER, NOT ONLY THE KEY. An earlier version of this test compared
    // keys alone and passed while `outputStorage` was not threaded into the
    // generator at all: the key differed on the SOURCE storage, so a projection
    // writing f32 into a buffer the flash kernel read as packed went unnoticed
    // until the fold came back NaN. Assert that the storage reaches the WGSL.
    const packedProject = selectAttentionProjectKernel(withF16, "f32", "f16", "f16");
    expect(packedProject.shader.includes("pack2x16float")).toBe(true);
    expect(packedProject.shader.includes("unpack2x16float")).toBe(true);
    expect(selectAttentionProjectKernel(withF16, "f32").shader.includes("2x16float")).toBe(false);
    const packedOutput = selectAttentionOutputKernel(withF16, false, "f32", "f16");
    expect(packedOutput.shader.includes("unpack2x16float")).toBe(true);
    expect(selectAttentionOutputKernel(withF16, false, "f32").shader.includes("2x16float"))
      .toBe(false);
    expect(packed.shader.includes("fn load4")).toBe(true);
    expect(packed.shader.includes("fn store4")).toBe(true);
    expect(plain.shader.includes("2x16float")).toBe(false);
    expect(() => selectAttentionOutputKernel(withoutF16, false, "f16")).toThrow();
    expect(() => chooseLinearKernel({ rows: 30208, columns: 1024, device: withoutF16,
                                      requested: "f16" })).toThrow();
  });
});
