/**
 * The grid pack's two descriptions of one layout must agree to the element.
 *
 * 🔴 THIS IS THE GATE FOR A BUG THAT ALREADY FIRED AND WAS CAUGHT BY READING.
 * `packGridAttentionWeights` lays the host buffer out; `gridResidentEntries` in
 * pair-track-device-weights.js is a HAND-WRITTEN MIRROR of that layout for the
 * device decoder. A term added to one and not the other does not drop the term:
 * it points the shader's `W_GATE_BIAS` INSIDE the output projection, which is a
 * WRONG answer rather than a missing one, and only on the resident weight path
 * - so a fold on a device without residency is fine and the same fold with it
 * is quietly wrong. docs/ARCHITECTURE.md records it as the second of the three
 * bugs this duplication produced.
 *
 * 🔴 AND IT IS THE OFFSETS THAT MATTER, NOT THE NAMES. Both sides can carry the
 * same set of names in the same order and still disagree, because the offsets
 * are sums of LENGTHS - so the comparison is the number the shader indexes
 * with, per name, plus the total.
 *
 * The layouts are still two descriptions. This makes their disagreement visible
 * so that collapsing them later is a change with a gate under it, rather than a
 * change where the wrong description wins silently.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { packGridAttentionWeights } from "../src/af3/grid-attention-webgpu.js";
import { gridResidentEntries } from "../src/af3/pair-track-device-weights.js";
import { SOURCES } from "../src/runtime/weight-sources.js";

const HEADS = 2;
const DIMENSION = 4;
const WIDTH = HEADS * DIMENSION;
const CHANNELS = 6;

/** The tensors a grid attention block carries, with their element counts. */
const LENGTHS = {
  actNormScale: CHANNELS, actNormOffset: CHANNELS,
  pairBiasProjection: CHANNELS * HEADS,
  qProjection: CHANNELS * WIDTH, kProjection: CHANNELS * WIDTH,
  vProjection: CHANNELS * WIDTH, gatingQuery: CHANNELS * WIDTH,
  outputProjection: WIDTH * CHANNELS,
  gatingQueryBias: WIDTH, outputProjectionBias: CHANNELS,
};

/** A weights object the HOST packer accepts. */
function hostWeights(names) {
  const weights = { heads: HEADS, dimension: DIMENSION };
  for (const name of names) {
    weights[name] = Float32Array.from({ length: LENGTHS[name] }, (_, i) => i + 1);
  }
  return weights;
}

/**
 * The SOURCES map the DEVICE path reads - thunks, never values. `bindable`
 * wants a `count`, a `store.tensorSource` and a `first`.
 */
function deviceSources(names) {
  const sources = {};
  for (const name of names) {
    sources[name] = {
      count: LENGTHS[name], first: 0, tensorName: name,
      store: { tensorSource: () => ({}) },
    };
  }
  return sources;
}

const REQUIRED = ["actNormScale", "actNormOffset", "pairBiasProjection",
                  "qProjection", "kProjection", "vProjection", "gatingQuery",
                  "outputProjection"];
const OPTIONAL = ["gatingQueryBias", "outputProjectionBias"];

/**
 * The device side reports one entry per TENSOR, with the four qkvg projections
 * sharing a base and a destination stride of four; the host side reports one
 * offset for the combined `qkvgProjection` slot. So the comparison is: every
 * plain name at the same offset, and the qkvg base equal to the host's slot.
 */
function compare(names, label) {
  const host = packGridAttentionWeights(hostWeights(names));
  const device = gridResidentEntries(deviceSources(names), WIDTH);
  assert.ok(device !== undefined, `${label}: the device path refused this layout`);

  const byName = new Map(device.entries.map((entry) => [entry.name, entry]));
  for (const name of names) {
    assert.ok(byName.has(name), `${label}: the device layout has no ${name}`);
  }
  assert.equal(device.total, host.data.length,
    `${label}: the two layouts are different SIZES`
    + ` (device ${device.total}, host ${host.data.length})`);

  for (const name of ["actNormScale", "actNormOffset", "pairBiasProjection",
                      "outputProjection", ...OPTIONAL.filter((n) => names.includes(n))]) {
    assert.equal(byName.get(name).offset, host.offsets[name],
      `${label}: ${name} is at ${byName.get(name).offset} on the device`
      + ` and ${host.offsets[name]} on the host`
      + " - the resident path would index another tensor here");
  }
  // The four interleaved projections share the host's single qkvg offset, one
  // lane each.
  const base = host.offsets.qkvgProjection;
  ["qProjection", "kProjection", "vProjection", "gatingQuery"].forEach((name, lane) => {
    assert.equal(byName.get(name).offset, base + lane,
      `${label}: ${name} is lane ${byName.get(name).offset - base}, wanted ${lane}`);
    assert.equal(byName.get(name).destStride, 4, `${label}: ${name} is not interleaved`);
  });
}

describe("the grid pack's host and device layouts agree", () => {
  it("without rosettafold3's two optional biases", () => {
    compare(REQUIRED, "bias-free");
  });

  it("with them, which is the term that fired this bug", () => {
    compare([...REQUIRED, ...OPTIONAL], "rosettafold3");
  });

  it("with only one of them, since they are independent in the loader", () => {
    compare([...REQUIRED, "gatingQueryBias"], "gate bias only");
    compare([...REQUIRED, "outputProjectionBias"], "output bias only");
  });

  it("the device path refuses a layout missing a REQUIRED tensor", () => {
    const short = REQUIRED.filter((name) => name !== "outputProjection");
    assert.equal(gridResidentEntries(deviceSources(short), WIDTH), undefined,
      "a missing required tensor must refuse, not pack a short buffer");
  });
});

/**
 * 🔴 AND THE TRIANGLE IS THE SAME MIRROR, ALSO UNGATED UNTIL NOW.
 * `triangleLayout` in pair-track-device-weights.js describes the destination in
 * `packOrder`'s order, and its own comment says why: "the offsets a compiled
 * shader was given come from that same call, so the two have to agree exactly
 * or the buffer is a plausible tensor in the wrong order." Two layouts, because
 * the matrix path splices four interleaved projections in where the vector path
 * has eight separate ones - so both are compared.
 */
describe("the triangle pack's host and device layouts agree", () => {
  const C = 8;

  /** What `packWeights` wants, at cHidden === cZ === C. */
  function hostTriangle() {
    const sizes = {
      layerNormInWeight: C, layerNormInBias: C,
      linearAPWeight: C * C, linearAPBias: C, linearAGWeight: C * C, linearAGBias: C,
      linearBPWeight: C * C, linearBPBias: C, linearBGWeight: C * C, linearBGBias: C,
      layerNormOutWeight: C, layerNormOutBias: C,
      linearZWeight: C * C, linearZBias: C, linearGWeight: C * C, linearGBias: C,
    };
    const weights = {};
    for (const [name, length] of Object.entries(sizes)) {
      weights[name] = Float32Array.from({ length }, (_, i) => i + 1);
    }
    return weights;
  }

  for (const abLayout of ["blocked", "interleaved"]) {
    it(`the ${abLayout} layout`, async () => {
      const { packWeights } = await import("../src/triangle/weights.js");
      const { triangleLayout } = await import("../src/af3/pair-track-device-weights.js");
      const host = packWeights(hostTriangle(), "f32",
        abLayout === "interleaved"
          ? { abLayout: "interleaved", cHidden: C, cZ: C } : {});
      const device = triangleLayout(C, abLayout);

      // The device side is [[name, spec], ...] in the same order; its offsets
      // are the running sum of `length`.
      let offset = 0;
      const seen = [];
      for (const [name, spec] of device) {
        seen.push(name);
        assert.equal(offset, host.offsets[name],
          `${abLayout}: ${name} is at ${offset} on the device and`
          + ` ${host.offsets[name]} on the host`
          + " - the resident path would index another tensor here");
        offset += spec.length;
      }
      assert.deepEqual(seen, Object.keys(host.offsets),
        `${abLayout}: the two layouts name different tensors, in this order`);
      assert.equal(offset, host.data.length,
        `${abLayout}: the layouts are different SIZES`
        + ` (device ${offset}, host ${host.data.length})`);
    });
  }
});
