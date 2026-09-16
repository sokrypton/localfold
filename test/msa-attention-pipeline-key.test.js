/**
 * Two MSA stacks whose shaders differ must not ask the cache for one key.
 *
 * 🔴 THE COLLISION THIS GATES WAS REPORTED FROM A REAL RUN, and its shape is
 * the one this repository keeps paying for: a pipeline key that does not name
 * its shader. Every kernel in msa-attention-webgpu.js is built from one
 * `common` preamble, so every one of them embeds `HEADS`, `DIMENSION` and
 * `WIDTH` - including `keyMask`, thirteen lines that read only `TOKENS` and
 * `SEQUENCES`. The MSA STACK keyed on the channel widths and not the head
 * shape:
 *
 *     af3-msa:59:128:64:128:0.00001:fast:false:msa:keyMask
 *     line 8 of 35: "const DIMENSION: u32 = 8u;" against "...= 32u;"
 *
 * AlphaFold 3 and RoseTTAFold3 are both `msaChannels` 64 with 8 heads, of
 * dimension 8 and 32 - openbind0 with the first, boltz2 with the second - so
 * the two families produced identical keys for different text. It needs both
 * families in ONE process at the same token count and depth, which is a model
 * switch on the page and never a single-model CLI run: that is why every gate
 * here was green while the page was not.
 *
 * 🔴 AND THE CACHE IS WHAT CAUGHT IT, NOT A FOLD. `ComputePipelineCache`
 * indexes by SOURCE as well as by key, so it refused and named the line rather
 * than handing rosettafold3 AlphaFold 3's kernel. This test is the cheap half
 * of that: it runs on the CPU, with no device and no bundle.
 */
import { describe, it, expect } from "./harness.js";
import { createMsaAttentionShaders, msaAttentionKeyPart }
  from "../src/af3/trunk/msa-attention-webgpu.js";

// The offsets only place weights inside one packed buffer; they are the same
// for every arm here, so any difference below is the SHAPE and not the packing.
const OFFSETS = { actNormScale: 0, actNormOffset: 1, pairNormScale: 2,
                  pairNormOffset: 3, pairLogits: 4, vProjection: 5,
                  gatingQuery: 6, outputProjection: 7 };

const sources = (shape) =>
  createMsaAttentionShaders(shape, OFFSETS, 0.00001, "fast");

// The two that actually collided, read off the shipped bundles' own tensor
// shapes: msa_attention1's gating query is [blocks, msaChannels, heads,
// dimension] - af3 [4, 64, 8, 8] and rosettafold3 [4, 64, 8, 32].
const AF3 = { sequences: 128, tokens: 59, msaChannels: 64, pairChannels: 128,
              heads: 8, dimension: 8 };
const RF3 = { ...AF3, dimension: 32 };

describe("the MSA attention's pipeline key", () => {
  it("separates the two head shapes that collided", () => {
    expect(sources(AF3).keyMask === sources(RF3).keyMask).toBe(false);
    expect(msaAttentionKeyPart(AF3) === msaAttentionKeyPart(RF3)).toBe(false);
  });

  it("names the head shape even in a kernel that does not read it", () => {
    // keyMask uses TOKENS and SEQUENCES only, and still carries DIMENSION -
    // which is exactly why keying on the channels alone was not enough.
    expect(sources(AF3).keyMask.includes("const DIMENSION: u32 = 8u;")).toBe(true);
    expect(sources(RF3).keyMask.includes("const DIMENSION: u32 = 32u;")).toBe(true);
  });

  it("gives different text a different key, over every shape the source embeds", () => {
    const arms = [
      ["af3 / openbind0", AF3],
      ["rosettafold3 / boltz2", RF3],
      ["protenix2-like", { ...AF3, msaChannels: 128 }],
      ["intellifold2-like", { ...AF3, msaChannels: 256, dimension: 32 }],
      ["four heads", { ...AF3, heads: 4 }],
    ];
    // The stack's own key: the channel widths it already named, plus the part
    // under test. A pair that agrees here must agree in the shaders too.
    const key = (s) => `${s.tokens}:${s.sequences}:${s.msaChannels}`
      + `:${s.pairChannels}:${msaAttentionKeyPart(s)}`;
    for (const [nameA, a] of arms) {
      for (const [nameB, b] of arms) {
        if (nameA >= nameB) continue;
        const sameKey = key(a) === key(b);
        for (const kernel of Object.keys(sources(a))) {
          const sameText = sources(a)[kernel] === sources(b)[kernel];
          expect(sameKey && !sameText).toBe(false);
        }
      }
    }
  });
});
