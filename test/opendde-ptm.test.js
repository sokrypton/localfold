/**
 * OpenDDE's PAE distribution reaches pTM, which for a while it did not.
 *
 * 🔴 pTM IS THE HEAD'S ARITHMETIC, NOT ANOTHER HEAD'S OUTPUT. AlphaFold 3's
 * confidence head emits a tm-adjusted PAE beside the expectation; OpenDDE's
 * emits pLDDT, PAE, PDE and experimentally-resolved. That was read here as
 * "OpenDDE has no pTM", and it is the wrong conclusion from the right
 * observation: pTM is a probability-weighted TM term over the PAE BINS, and
 * OpenDDE's PAE is a distribution over the same 64 of them. Upstream
 * (sokrypton/alphafold3) makes exactly this call - `tmscore_adjusted_pae` is a
 * module-level function whose docstring says "OpenDDE has its own confidence
 * head whose outputs still have to reach the same pTM/ipTM the rest of the
 * pipeline reports".
 *
 * What was missing was the DISTRIBUTION: the readout kept the expectation and
 * dropped the logits, so there was nothing left to adjust.
 *
 * No GPU and no bundle: the host readout is the reference the kernel is held
 * against (tools/gpu/check-opendde-confidence.js), and what is checked here is
 * the arithmetic it now carries - the bins, the centres, and whose token count
 * d0 comes from.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { hostPairReadouts } from "../src/af3/confidence/opendde-confidence.js";
import { tmScoreD0 } from "../src/heads/tm-score.js";

const CHANNELS = 8;
const BINS = 64;
const SPIKE = 40;          // the bin the fixture's logits land on

/**
 * Weights that put every pair's PAE on ONE bin.
 *
 * LayerNorm's output sums to zero over the channels, so a projection alone
 * cannot make a row prefer a bin - the offset is what gives it something to
 * multiply. With offset 1 the activation is `normalised + 1`, whose sum is the
 * channel count, and a single loud column is then a spike whatever the pair
 * holds. That is what makes the expected values below closed-form.
 */
function spikedWeights() {
  const pae = new Float32Array(CHANNELS * BINS);
  for (let c = 0; c < CHANNELS; c += 1) pae[c * BINS + SPIKE] = 10;
  return {
    paeBins: BINS, pdeBins: BINS,
    paeLnScale: new Float32Array(CHANNELS).fill(1),
    paeLnOffset: new Float32Array(CHANNELS).fill(1),
    pdeLnScale: new Float32Array(CHANNELS).fill(1),
    pdeLnOffset: new Float32Array(CHANNELS).fill(1),
    pae,
    pde: new Float32Array(CHANNELS * BINS),
  };
}

function fixturePair(tokens) {
  const pair = new Float32Array(tokens * tokens * CHANNELS);
  for (let i = 0; i < pair.length; i += 1) pair[i] = Math.sin(i * 0.37) * 0.5;
  return pair;
}

describe("OpenDDE's tm-adjusted PAE", () => {
  const tokens = 6;
  const centre = (SPIKE + 0.5) * (32 / BINS);

  it("is absent unless a token count is asked for, and present when it is", () => {
    const pair = fixturePair(tokens);
    const weights = spikedWeights();
    assert.equal(hostPairReadouts(pair, tokens, CHANNELS, weights).tm, undefined,
                 "a readout nobody asked a pTM of must not invent one");
    const withTm = hostPairReadouts(pair, tokens, CHANNELS, weights, 100);
    assert.equal(withTm.tm.length, tokens * tokens);
  });

  it("is the TM term of the bin the distribution sits on", () => {
    const out = hostPairReadouts(fixturePair(tokens), tokens, CHANNELS,
                                 spikedWeights(), 100);
    // ...the spike is where the fixture put it, which is what makes the
    // closed form below the right one to compare against.
    assert.ok(Math.abs(out.pae[0] - centre) < 0.05,
              `the fixture's PAE is ${out.pae[0]}, not the spiked bin's ${centre}`);
    const d0 = tmScoreD0(100);
    const want = 1 / (1 + (centre * centre) / (d0 * d0));
    for (let pair = 0; pair < tokens * tokens; pair += 1) {
      assert.ok(Math.abs(out.tm[pair] - want) < 1e-3,
                `pair ${pair}: ${out.tm[pair]} against ${want}`);
    }
  });

  it("takes d0 from the count it is given, not from this head's tokens", () => {
    // 🔴 THE TOKEN COUNT IS THE SCORE'S, NOT THE HEAD'S. OpenDDE's head runs on
    // STRUCTURAL tokens and pTM is reported over the residue tokens; the two
    // part company the moment a ligand or a modified residue expands the
    // structural set, and d0 grows with the count - so the same distribution
    // must score HIGHER over a bigger structure.
    const small = hostPairReadouts(fixturePair(tokens), tokens, CHANNELS,
                                   spikedWeights(), 30);
    const large = hostPairReadouts(fixturePair(tokens), tokens, CHANNELS,
                                   spikedWeights(), 800);
    assert.ok(large.tm[0] > small.tm[0],
              `d0 did not follow the token count: ${small.tm[0]} -> ${large.tm[0]}`);
    for (const value of large.tm) assert.ok(value >= 0 && value <= 1);
  });

  it("reads the pair as given for the PAE and symmetrised for the PDE", () => {
    // 🔴 A SYMMETRIC PAE IS A BUG, AND IT IS ONE FLAG AWAY. The two readouts
    // are the same kernel under one switch - `symmetrise`, which makes the row
    // `z_ij + z_ji` before the LayerNorm - because a DISTANCE error is
    // symmetric and an ALIGNED error is not: PAE(i, j) is the error at i when
    // the prediction is superposed on j, and superposing on the other one is a
    // different question. Setting that flag on both arms, or caching the two
    // pipelines under a key that does not name it, would give a PAE that is
    // exactly its own transpose - which looks like a plausible plot and is the
    // wrong quantity.
    //
    // The spiked weights above cannot see this: they put every pair on one bin
    // by construction, so every matrix they produce is symmetric. This one
    // takes a pair that is nothing like its own transpose and reads the two
    // arms off it.
    const tokens = 5;
    // ...its own weights, not the spike above: a LayerNorm offset of 1 leaves
    // the row's own content a small part of the activation, so the fixture
    // would be nearly symmetric for a reason that has nothing to do with the
    // code under test. Offset zero and a loud projection let the pair speak.
    let seed = 1;
    const noise = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648 - 0.5;
    };
    const weights = {
      paeBins: BINS, pdeBins: BINS,
      paeLnScale: new Float32Array(CHANNELS).fill(1),
      paeLnOffset: new Float32Array(CHANNELS),
      pdeLnScale: new Float32Array(CHANNELS).fill(1),
      pdeLnOffset: new Float32Array(CHANNELS),
      pae: new Float32Array(CHANNELS * BINS).map(noise),
      pde: new Float32Array(CHANNELS * BINS).map(noise),
    };
    // ...and a pair with no structure in it at all, because a smooth one is
    // nearly its own transpose and would need a threshold chosen to fit.
    const pair = new Float32Array(tokens * tokens * CHANNELS);
    for (let i = 0; i < pair.length; i += 1) pair[i] = noise();
    const out = hostPairReadouts(pair, tokens, CHANNELS, weights, 100);
    const worstGap = (matrix) => {
      let worst = 0;
      for (let i = 0; i < tokens; i += 1) {
        for (let j = 0; j < tokens; j += 1) {
          worst = Math.max(worst, Math.abs(matrix[i * tokens + j] - matrix[j * tokens + i]));
        }
      }
      return worst;
    };
    // The control: the fixture has to be asymmetric for the claim to mean
    // anything, and the PDE arm reading the same pair proves the symmetry is
    // the arm's and not the input's.
    assert.ok(worstGap(out.pae) > 2,
              `the PAE is its own transpose to within ${worstGap(out.pae)} -`
              + " the aligned error is not a symmetric quantity");
    assert.equal(worstGap(out.pde), 0,
                 "the PDE is a distance error and must be exactly symmetric");
  });

  it("is not the PAE by another name", () => {
    const out = hostPairReadouts(fixturePair(tokens), tokens, CHANNELS,
                                 spikedWeights(), 100);
    // A readout that wrote the expectation into both would pass every check
    // that only asks whether the field is filled: the PAE is Angstrom over
    // [0, 32] and the TM term is a probability.
    assert.ok(out.pae[0] > 1.5, "the fixture's PAE should be a real distance");
    assert.ok(out.tm[0] < 1, "the TM term should be a probability");
  });
});
