/**
 * The projection kernels' tile is one number, and the residual form is
 * generated rather than patched.
 *
 * 🔴 BOTH OF THESE HAVE A SILENT FAILURE MODE AND NEITHER RAISES. A dispatch
 * that divides by a different tile from the one the shader was generated with
 * leaves rows unprocessed and reads as a speedup - it has happened here once,
 * in the diffusion transformer. And the residual variant of project-output used
 * to be a string replacement on the finished WGSL: when this kernel's writeback
 * was rewritten for register blocking the pattern stopped matching, and
 * String.replace returns the subject unchanged rather than throwing, so AF2's
 * evoformer would have OVERWRITTEN the pair representation where it meant to
 * add to it. Neither the shape nor the types would have moved.
 */
import { createTriangleShaders } from "../src/triangle/shaders.js";
import { describe, expect, it } from "./harness.js";

const NAMES = [
  "layerNormInWeight", "layerNormInBias", "linearAPWeight", "linearAPBias",
  "linearAGWeight", "linearAGBias", "linearBPWeight", "linearBPBias",
  "linearBGWeight", "linearBGBias", "layerNormOutWeight", "layerNormOutBias",
  "linearZWeight", "linearZBias", "linearGWeight", "linearGBias",
];
const offsets = Object.fromEntries(NAMES.map((name, index) => [name, index]));
const shape = { length: 8, cZ: 16, cHidden: 16 };
const build = (tile, residual = false) =>
  createTriangleShaders(shape, "f32", offsets, 1e-5, "outgoing", "two-pass", tile, residual);

describe("triangle projection tile", () => {
  it("reports the tile it generated, so a caller cannot pick its own", () => {
    const shaders = build(undefined);
    expect(shaders.projectTile.rows % 8).toBe(0);
    expect(shaders.projectTile.columns % 8).toBe(0);
  });

  it("puts that tile in the shader, so the dispatch and the kernel agree", () => {
    const shaders = build({ rows: 32, columns: 16 });
    expect(shaders.projectAB).toContain("const TILE_ROWS: u32 = 32u;");
    expect(shaders.projectAB).toContain("const TILE_COLUMNS: u32 = 16u;");
    expect(shaders.projectOutput).toContain("const TILE_ROWS: u32 = 32u;");
    expect(shaders.projectTile.rows).toBe(32);
    expect(shaders.projectTile.columns).toBe(16);
  });

  it("refuses a tile the 8x8 workgroup cannot cover", () => {
    expect(() => build({ rows: 12, columns: 16 })).toThrow();
    expect(() => build({ rows: 32, columns: 20 })).toThrow();
  });

  it("sizes the register block from the tile", () => {
    // Four rows and two columns an invocation at 32x16, and one vec4 a cell:
    // a, b and their two gates share a source and are accumulated together.
    expect(build({ rows: 32, columns: 16 }).projectAB)
      .toContain("var acc: array<vec4<f32>, 8>;");
    expect(build({ rows: 16, columns: 16 }).projectAB)
      .toContain("var acc: array<vec4<f32>, 4>;");
  });

  it("accumulates in the residual form and assigns in the plain one", () => {
    const tile = { rows: 32, columns: 16 };
    const plain = build(tile).projectOutput;
    const residual = build(tile, true).projectOutput;
    expect(plain).toContain("output[row * CZ + out_channel] = projected[at]");
    expect(residual).toContain("output[row * CZ + out_channel] += projected[at]");
  });

  it("changes nothing else between the two forms", () => {
    const tile = { rows: 32, columns: 16 };
    const plain = build(tile).projectOutput.split("\n");
    const residual = build(tile, true).projectOutput.split("\n");
    expect(residual.length).toBe(plain.length);
    const differing = plain.filter((line, index) => line !== residual[index]);
    expect(differing.length).toBe(1);
  });
});
