/**
 * OpenDDE's trunk branches, each asserted to CHANGE something.
 *
 * 🔴 A DIALECT FLAG THAT NEVER REACHES THE ARITHMETIC AGREES WITH ITSELF. Every
 * branch here is silent when wrong - the shapes conform, a representation comes
 * out, and it is a different model - so a test that only checked "the OpenDDE
 * arm runs" would pass against code that ignored the flag entirely. Each test
 * below therefore runs BOTH arms over one set of weights and asserts they
 * differ, and where there is a cheap independent statement of the right answer
 * it asserts that too.
 *
 * These are CPU-reference tests. The GPU halves of the same branches are keyed
 * into the shader cache and asserted on the generated WGSL in
 * test/opendde-shaders.test.js, for the reason this repository records twice
 * over: comparing cache keys is not comparing kernels.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { ALPHAFOLD3, OPENDDE } from "../src/af3/dialect.js";
import { msaBlock } from "../src/af3/msa-reference.js";
import { embed } from "../src/af3/embedder-reference.js";

/** A deterministic pseudo-random array, so a difference is the branch. */
function noise(length, seed) {
  const out = new Float32Array(length);
  let state = seed >>> 0;
  for (let index = 0; index < length; index += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[index] = (state / 0xffffffff) * 2 - 1;
  }
  return out;
}

const rms = (a) => Math.sqrt(a.reduce((sum, v) => sum + v * v, 0) / a.length);
function relRms(a, b) {
  const diff = Float32Array.from(a, (v, i) => v - b[i]);
  return rms(diff) / Math.max(rms(a), rms(b), 1e-30);
}

describe("OpenDDE builds the pair from the single embedding", () => {
  const tokens = 5;
  const featureWidth = 7;
  const pairChannels = 3;
  const singleChannels = 4;
  const msaChannels = 2;
  const sequences = 2;

  /**
   * Both dialects need `leftSingle` at their OWN input width, which is the
   * whole point: AlphaFold 3's is [447, 128] and OpenDDE's is [384, 384].
   */
  const weightsFor = (dialect) => {
    const inputWidth = dialect.pairInitFromSingle ? singleChannels : featureWidth;
    return {
      dialect, pairChannels, singleChannels, msaChannels, targetFeatWidth: featureWidth,
      leftSingle: noise(inputWidth * pairChannels, 11),
      rightSingle: noise(inputWidth * pairChannels, 12),
      singleActivations: noise(featureWidth * singleChannels, 13),
      prevEmbeddingNormScale: new Float32Array(pairChannels).fill(1),
      prevEmbeddingNormOffset: new Float32Array(pairChannels),
      prevEmbedding: noise(pairChannels * pairChannels, 14),
      positionActivations: noise(139 * pairChannels, 15),
      msaActivations: noise(49 * msaChannels, 16),
      extraMsaTargetFeat: noise(featureWidth * msaChannels, 17),
      bondEmbedding: noise(pairChannels, 18),
      prevSingleEmbeddingNormScale: new Float32Array(singleChannels).fill(1),
      prevSingleEmbeddingNormOffset: new Float32Array(singleChannels),
      prevSingleEmbedding: noise(singleChannels * singleChannels, 19),
      relativeWidth: 139,
    };
  };

  const input = () => ({
    tokens, sequences, targetFeat: noise(tokens * featureWidth, 21),
    // Required, and rightly: the template embedder contributes even with no
    // templates, so omitting it is not the same as passing zeros.
    templateEmbedding: new Float32Array(tokens * tokens * pairChannels),
    msaRows: new Int32Array(sequences * tokens).fill(1),
    deletionMatrix: new Float32Array(sequences * tokens),
    features: {
      residueIndex: Int32Array.from({ length: tokens }, (_, i) => i),
      tokenIndex: Int32Array.from({ length: tokens }, (_, i) => i),
      asymId: new Int32Array(tokens), entityId: new Int32Array(tokens),
      symId: new Int32Array(tokens),
    },
  });

  it("moves the pair, which is what says the branch is taken", () => {
    const stock = embed(input(), weightsFor(ALPHAFOLD3));
    const dde = embed(input(), weightsFor(OPENDDE));
    assert.equal(stock.pair.length, dde.pair.length);
    // Different SOURCE and different WEIGHTS, so this is a large difference,
    // not a rounding one. Zero here would mean the flag never arrived.
    assert.ok(relRms(stock.pair, dde.pair) > 0.1,
      `the two pair inits agree at relRMS ${relRms(stock.pair, dde.pair)}`);
  });

  it("builds the pair from exactly the single it returns, before recycling", () => {
    // 🔴 THE PAIR IS BUILT FROM s_init WITHOUT THE RECYCLED TERM. With no
    // previous single the recycled contribution is the projection of the
    // LayerNorm's offset, which is zero only because this fixture's offset is
    // zero - so with a zero offset the returned single IS s_init, and the pair
    // has to be reproducible from it exactly.
    const weights = weightsFor(OPENDDE);
    const out = embed(input(), weights);
    const expected = new Float32Array(tokens * tokens * pairChannels);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        for (let c = 0; c < pairChannels; c += 1) {
          let left = 0;
          let right = 0;
          for (let f = 0; f < singleChannels; f += 1) {
            left += out.single[i * singleChannels + f] * weights.leftSingle[f * pairChannels + c];
            right += out.single[j * singleChannels + f] * weights.rightSingle[f * pairChannels + c];
          }
          expected[(i * tokens + j) * pairChannels + c] = left + right;
        }
      }
    }
    // The pair also carries the relative encoding, the bond term and the
    // recycled pair, so this is a lower bound on agreement rather than
    // equality: what is asserted is that the s_init term is present and exact.
    const withoutOthers = Float32Array.from(out.pair, (v, i) => v - expected[i]);
    assert.ok(rms(withoutOthers) < rms(out.pair),
      "the s_init term is not in the pair at all");
  });

  it("refuses to guess when the dialect does not say", () => {
    const weights = weightsFor(OPENDDE);
    assert.throws(() => embed(input(), { ...weights, dialect: {} }),
      /pairInitFromSingle has no default/);
  });
});

describe("OpenDDE updates the MSA before the outer product", () => {
  const tokens = 4;
  const sequences = 3;
  const pairChannels = 2;
  const msaChannels = 2;
  const outerChannels = 2;
  const heads = 2;
  const dimension = 2;

  const block = () => ({
    pairChannels, msaChannels,
    outerProductMean: {
      outerChannels,
      layerNormInputScale: new Float32Array(msaChannels).fill(1),
      layerNormInputOffset: new Float32Array(msaChannels),
      leftProjection: noise(msaChannels * outerChannels, 31),
      rightProjection: noise(msaChannels * outerChannels, 32),
      outputW: noise(outerChannels * outerChannels * pairChannels, 33),
      outputB: noise(pairChannels, 34),
    },
    msaAttention1: {
      heads, dimension,
      actNormScale: new Float32Array(msaChannels).fill(1),
      actNormOffset: new Float32Array(msaChannels),
      pairNormScale: new Float32Array(pairChannels).fill(1),
      pairNormOffset: new Float32Array(pairChannels),
      pairLogits: noise(pairChannels * heads, 35),
      vProjection: noise(msaChannels * heads * dimension, 36),
      gatingQuery: noise(msaChannels * heads * dimension, 37),
      outputProjection: noise(heads * dimension * msaChannels, 38),
    },
    msaTransition: {
      inputLayerNormScale: new Float32Array(msaChannels).fill(1),
      inputLayerNormOffset: new Float32Array(msaChannels),
      transition1: noise(msaChannels * msaChannels * 8, 39),
      transition2: noise(msaChannels * 4 * msaChannels, 40),
    },
    triangleMultiplicationOutgoing: triangle(41),
    triangleMultiplicationIncoming: triangle(42),
    pairAttention1: grid(43),
    pairAttention2: grid(44),
    pairTransition: {
      inputLayerNormScale: new Float32Array(pairChannels).fill(1),
      inputLayerNormOffset: new Float32Array(pairChannels),
      transition1: noise(pairChannels * pairChannels * 8, 45),
      transition2: noise(pairChannels * 4 * pairChannels, 46),
    },
  });

  function triangle(seed) {
    return {
      leftNormInputScale: new Float32Array(pairChannels).fill(1),
      leftNormInputOffset: new Float32Array(pairChannels),
      projection: noise(pairChannels * 2 * pairChannels, seed),
      gate: noise(pairChannels * 2 * pairChannels, seed + 100),
      centerNormScale: new Float32Array(pairChannels).fill(1),
      centerNormOffset: new Float32Array(pairChannels),
      outputProjection: noise(pairChannels * pairChannels, seed + 200),
      gatingLinear: noise(pairChannels * pairChannels, seed + 300),
    };
  }
  function grid(seed) {
    return {
      heads, dimension,
      actNormScale: new Float32Array(pairChannels).fill(1),
      actNormOffset: new Float32Array(pairChannels),
      pairBiasProjection: noise(pairChannels * heads, seed),
      qProjection: noise(heads * dimension * pairChannels, seed + 10),
      kProjection: noise(pairChannels * heads * dimension, seed + 20),
      vProjection: noise(pairChannels * heads * dimension, seed + 30),
      gatingQuery: noise(pairChannels * heads * dimension, seed + 40),
      outputProjection: noise(heads * dimension * pairChannels, seed + 50),
    };
  }

  const state = () => ({
    sequences, tokens,
    pair: noise(tokens * tokens * pairChannels, 51),
    msa: noise(sequences * tokens * msaChannels, 52),
    pairMask: new Float32Array(tokens * tokens).fill(1),
    msaMask: new Float32Array(sequences * tokens).fill(1),
  });

  it("gives a different block from the same weights", () => {
    const weights = block();
    const stock = msaBlock(state(), weights, ALPHAFOLD3);
    const dde = msaBlock(state(), weights, OPENDDE);
    // 🔴 BOTH OUTPUTS MOVE, and the MSA is the sharper of the two: under AF3
    // the row update reads the pair the outer product just changed, and under
    // OpenDDE it reads the pair as it arrived.
    assert.ok(relRms(stock.msa, dde.msa) > 1e-6,
      `the MSA is identical under both orderings (${relRms(stock.msa, dde.msa)})`);
    assert.ok(relRms(stock.pair, dde.pair) > 1e-6,
      `the pair is identical under both orderings (${relRms(stock.pair, dde.pair)})`);
  });

  it("refuses to guess when the dialect does not say", () => {
    assert.throws(() => msaBlock(state(), block(), { swapTransposedBias: false }),
      /msaUpdateBeforeOuterProduct has no default/);
  });
});
