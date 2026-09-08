/**
 * The batched zero gate's weight layout.
 *
 * 🔴 THE STRIDE IS THE WHOLE RISK HERE. The shader finds a block's weights by
 * multiplying a fixed span by the block index, so a span that disagrees with
 * the packer by one element shifts every block after the first - and the fold
 * still runs, still returns a structure, and returns the WRONG one. That is the
 * exact shape of the OpenDDE dispatch bug in docs/OPENDDE.md, where a size
 * mismatch left two thirds of every row unprocessed and every per-kernel
 * checker passing. A GPU gate cannot catch it cheaply; this can.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { packZeroGateWeights } from "../src/af3/diffusion-transformer-webgpu.js";

const C_COND = 4;
const C = 3;
// 🔴 SPELLED OUT INDEPENDENTLY OF THE SOURCE, ON PURPOSE. Importing the
// packer's own list would make this test agree with any order it happened to
// have; the shader's ZG_* offsets are running sums of exactly this sequence, so
// the sequence is the thing under test.
const NAMES = ["AdaptiveZeroCondWeights", "AdaptiveZeroCondBias",
               "ffwAdaptiveZeroCondWeights", "ffwAdaptiveZeroCondBias",
               "SingleCondLayerNormScale", "SingleCondScaleWeights",
               "SingleCondBias", "SingleCondScaleBias",
               "ffwSingleCondLayerNormScale", "ffwSingleCondScaleWeights",
               "ffwSingleCondBias", "ffwSingleCondScaleBias"];
const LENGTHS = {
  AdaptiveZeroCondWeights: C_COND * C, AdaptiveZeroCondBias: C,
  ffwAdaptiveZeroCondWeights: C_COND * C, ffwAdaptiveZeroCondBias: C,
  SingleCondLayerNormScale: C_COND, SingleCondScaleWeights: C_COND * C,
  SingleCondBias: C_COND * C, SingleCondScaleBias: C,
  ffwSingleCondLayerNormScale: C_COND, ffwSingleCondScaleWeights: C_COND * C,
  ffwSingleCondBias: C_COND * C, ffwSingleCondScaleBias: C,
};
const SPAN = NAMES.reduce((total, name) => total + LENGTHS[name], 0);

/** A block whose every element encodes the block, the tensor and the index. */
function block(at) {
  const out = {};
  for (const [which, name] of NAMES.entries()) {
    out[name] = Float32Array.from({ length: LENGTHS[name] },
      (_, i) => at * 1000 + which * 100 + i);
  }
  return out;
}

test("packs each block at its own stride, tensors in shader order", () => {
  const blocks = [block(0), block(1), block(2)];
  const packed = packZeroGateWeights(blocks, "f32");
  assert.equal(packed.length, blocks.length * SPAN);
  for (const [at, source] of blocks.entries()) {
    let offset = at * SPAN;
    for (const name of NAMES) {
      for (let i = 0; i < LENGTHS[name]; i += 1) {
        assert.equal(packed[offset + i], source[name][i],
          `block ${at} ${name}[${i}] at ${offset + i}`);
      }
      offset += LENGTHS[name];
    }
  }
});

test("the shader's ZG_* offsets land on the tensors they name", () => {
  const packed = packZeroGateWeights([block(0)], "f32");
  const source = block(0);
  const w = C_COND * C;
  const gates = 2 * (w + C);          // both zero gates
  const adalnSpan = C_COND + 2 * w + C;
  // Each of these is a constant the kernel computes the same way. A mismatch
  // here is the kernel reading a neighbouring tensor.
  const at = {
    ZG_FFW: w + C,
    AD_LN: gates,
    AD_SW: gates + C_COND,
    AD_CB: gates + C_COND + w,
    AD_SB: gates + C_COND + 2 * w,
    FA_LN: gates + adalnSpan,
    FA_SW: gates + adalnSpan + C_COND,
    FA_CB: gates + adalnSpan + C_COND + w,
    FA_SB: gates + adalnSpan + C_COND + 2 * w,
  };
  const first = {
    ZG_FFW: "ffwAdaptiveZeroCondWeights", AD_LN: "SingleCondLayerNormScale",
    AD_SW: "SingleCondScaleWeights", AD_CB: "SingleCondBias",
    AD_SB: "SingleCondScaleBias", FA_LN: "ffwSingleCondLayerNormScale",
    FA_SW: "ffwSingleCondScaleWeights", FA_CB: "ffwSingleCondBias",
    FA_SB: "ffwSingleCondScaleBias",
  };
  for (const [name, offset] of Object.entries(at)) {
    assert.equal(packed[offset], source[first[name]][0],
      `${name} should start ${first[name]}`);
  }
});

test("a block's span is what the kernel's stride must be", () => {
  const w = C_COND * C;
  const span = 2 * (w + C) + 2 * (C_COND + 2 * w + C);
  const packed = packZeroGateWeights([block(0), block(1)], "f32");
  assert.equal(packed.length, 2 * span);
  assert.equal(packed[span], block(1).AdaptiveZeroCondWeights[0]);
});

test("refuses a block whose span differs, rather than shifting the rest", () => {
  const blocks = [block(0), block(1)];
  blocks[1].AdaptiveZeroCondWeights = new Float32Array(C_COND * C - 1);
  assert.throws(() => packZeroGateWeights(blocks, "f32"), /block 1 zero-gate/);
});

test("rejects an empty block list instead of packing nothing", () => {
  assert.throws(() => packZeroGateWeights([], "f32"), /no diffusion blocks/);
});

/**
 * 🔴 AND THE SHADER'S OFFSETS MUST POINT AT THE TENSORS THE PACKER WROTE.
 *
 * The packer decides a layout and the kernel navigates it with a set of
 * constants that are running sums of the same lengths, computed independently
 * in `createDiffusionTransformerShaders`. Nothing connects the two but arithmetic
 * agreement, and a mismatch reads a NEIGHBOURING tensor - which is a wrong fold,
 * not a crash. This packs marker values and asks the emitted WGSL where it
 * thinks each tensor starts.
 */
test("the emitted kernel's ZG_* constants index the packer's layout", async () => {
  const mod = await import("../src/af3/diffusion-transformer-webgpu.js");
  const C_COND = 384;
  const C = 768;
  const shape = {
    tokens: 8, channels: C, condChannels: C_COND, pairChannels: 128, heads: 4,
    dimension: 192, factor: 2, samples: 1, tile: 1, splits: 2, outTile: 1,
    outChunk: 384, weightPrecision: "f32", batchedGates: true, lanes: 256,
  };
  const zero = Object.fromEntries(mod.BLOCK_ORDER.map((n) => [n, 0]));
  const source = mod.createDiffusionTransformerShaders(shape, zero).zeroGates;
  assert.equal(typeof source, "string", "batchedGates should emit a zeroGates kernel");
  const constant = (name) => {
    const found = source.match(new RegExp(`const ${name}: u32 = (\\d+)u;`));
    assert.ok(found, `${name} not emitted`);
    return Number(found[1]);
  };

  // The packer's own layout, built from lengths rather than restated numbers.
  const w = C_COND * C;
  const lengths = {
    AdaptiveZeroCondWeights: w, AdaptiveZeroCondBias: C,
    ffwAdaptiveZeroCondWeights: w, ffwAdaptiveZeroCondBias: C,
    SingleCondLayerNormScale: C_COND, SingleCondScaleWeights: w,
    SingleCondBias: w, SingleCondScaleBias: C,
    ffwSingleCondLayerNormScale: C_COND, ffwSingleCondScaleWeights: w,
    ffwSingleCondBias: w, ffwSingleCondScaleBias: C,
  };
  const order = ["AdaptiveZeroCondWeights", "AdaptiveZeroCondBias",
                 "ffwAdaptiveZeroCondWeights", "ffwAdaptiveZeroCondBias",
                 "SingleCondLayerNormScale", "SingleCondScaleWeights",
                 "SingleCondBias", "SingleCondScaleBias",
                 "ffwSingleCondLayerNormScale", "ffwSingleCondScaleWeights",
                 "ffwSingleCondBias", "ffwSingleCondScaleBias"];
  const at = {};
  let running = 0;
  for (const name of order) { at[name] = running; running += lengths[name]; }

  assert.equal(constant("ZG_FFW"), at.ffwAdaptiveZeroCondWeights, "ZG_FFW");
  assert.equal(constant("AD_LN"), at.SingleCondLayerNormScale, "AD_LN");
  assert.equal(constant("AD_SW"), at.SingleCondScaleWeights, "AD_SW");
  assert.equal(constant("AD_CB"), at.SingleCondBias, "AD_CB");
  assert.equal(constant("AD_SB"), at.SingleCondScaleBias, "AD_SB");
  assert.equal(constant("FA_LN"), at.ffwSingleCondLayerNormScale, "FA_LN");
  assert.equal(constant("FA_SW"), at.ffwSingleCondScaleWeights, "FA_SW");
  assert.equal(constant("FA_CB"), at.ffwSingleCondBias, "FA_CB");
  assert.equal(constant("FA_SB"), at.ffwSingleCondScaleBias, "FA_SB");
  // ...and the stride the kernel multiplies by the block index is the span the
  // packer wrote, or every block after the first reads the wrong weights.
  assert.equal(constant("ZG_STRIDE"), running, "ZG_STRIDE must be one block's span");
});
