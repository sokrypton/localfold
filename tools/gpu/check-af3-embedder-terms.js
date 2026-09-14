/**
 * AlphaFold 3's z_init, TERM BY TERM, against af3-any-model.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-embedder-terms.js
 *
 * 🔴 STOCK AF3's OWN `z_init_generic` IS THE LEAST EXACT OF THE FIVE - 2.20e-4
 * against openbind0's 3.65e-8, boltz2's 5.05e-8, protenix2's 1.89e-8 and
 * OpenDDE's 4.70e-8, on the model this port was written against first. Its
 * `target_feat` going in is 5.76e-8, so the inputs are right and one of the
 * SUMMANDS is not. The pair init is four terms - left/right of the single, the
 * recycled pair (not zero on pass one: a LayerNorm turns a zero input into its
 * offset), the relative encoding, and the bond embedding - and a whole-tensor
 * residual cannot say which.
 *
 * The dump is `CAPTURE=left_single|right_single|position_activations|
 * bond_embedding|single_activations dump_af3_trunk_taps.py alphafold3`, and it
 * confirms `bond_embedding` is rms 0.0000 on a protein with no ligand.
 */
import { batchFromDump } from "./fold.js";
import { relativeEncoding } from "../../src/af3/embedder-reference.js";
import { linear } from "../../src/af3/pairformer-reference.js";
import { openAf3Store, trunkWeights } from "../../src/af3/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};
const relRms = (ours, expected) => {
  let error = 0, scale = 0;
  for (let i = 0; i < expected.length; i += 1) {
    const d = ours[i] - expected[i];
    error += d * d; scale += expected[i] * expected[i];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
};
const rms = (a) => Math.sqrt(a.reduce((t, v) => t + v * v, 0) / a.length);

export async function main(device, args) {
  const dumpPath = option(args, "oracle", "/oracle-dumps/af3-oracle-emb-alphafold3.json");
  const response = await fetch(dumpPath);
  if (!response.ok) throw new Error(`failed to load ${dumpPath}: ${response.status}`);
  const stages = (await response.json()).stages;
  const of = (name) => {
    const entry = stages[name];
    if (entry === undefined) throw new Error(`the dump has no ${name}`);
    return Float32Array.from(entry.data);
  };

  const batchPath = option(args, "dump", "/oracle-dumps/af3-batch-alphafold3-6mrr.json");
  const batchResponse = await fetch(batchPath);
  if (!batchResponse.ok) throw new Error(`failed to load ${batchPath}`);
  const batch = batchFromDump(await batchResponse.json());

  const store = await openAf3Store(option(args, "model",
    "/model-af3-full-f32/manifest.json"));
  store.prefetch();
  const trunk = await trunkWeights(store);
  const w = trunk.embedder;

  // 🔴 THE ORACLE'S OWN `target_feat` GOES IN, not ours. Ours is exact at
  // 5.76e-8, and using it would fold that residue into every term below.
  const targetFeat = of("target_feat");
  const tokens = batch.tokens;
  const featureWidth = targetFeat.length / tokens;
  const pairChannels = w.pairChannels;

  const arms = [];
  const compare = (label, ours, native) => {
    arms.push({ term: label, relRms: Number(relRms(ours, native).toExponential(2)),
                oursRms: Number(rms(ours).toFixed(4)),
                nativeRms: Number(rms(native).toFixed(4)) });
  };

  compare("single_activations",
    linear(targetFeat, tokens, featureWidth, w.singleChannels, w.singleActivations),
    of("scope.evoformer/single_activations"));
  compare("left_single",
    linear(targetFeat, tokens, featureWidth, pairChannels, w.leftSingle),
    of("scope.evoformer/left_single"));
  compare("right_single",
    linear(targetFeat, tokens, featureWidth, pairChannels, w.rightSingle),
    of("scope.evoformer/right_single"));

  const relative = relativeEncoding(tokens, batch.features);
  compare("position_activations",
    linear(relative, tokens * tokens, w.relativeWidth, pairChannels, w.positionActivations),
    of("scope.evoformer/~_relative_encoding/position_activations"));

  // 🔴 WHICH PAIRS, because the relative encoding is a one-hot: a wrong column
  // for a few (i, j) and a uniformly wrong projection are the same relRMS and
  // different bugs.
  const positioned = linear(relative, tokens * tokens, w.relativeWidth, pairChannels,
                            w.positionActivations);
  const nativePositioned = of("scope.evoformer/~_relative_encoding/position_activations");
  const badPairs = [];
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      let error = 0, scale = 0;
      for (let c = 0; c < pairChannels; c += 1) {
        const at = (i * tokens + j) * pairChannels + c;
        const d = positioned[at] - nativePositioned[at];
        error += d * d; scale += nativePositioned[at] ** 2;
      }
      if (Math.sqrt(error / Math.max(scale, 1e-30)) > 1e-4) {
        badPairs.push({ i, j, d: j - i,
                        rel: Number(Math.sqrt(error / Math.max(scale, 1e-30)).toExponential(2)) });
      }
    }
  }
  // 🔴 IS THE DIFFERENCE A CONSTANT? Every pair wrong by the same relative
  // amount is the signature of ONE one-hot column set differently everywhere -
  // which adds a fixed row of `position_activations` to every pair - and not of
  // a separation-dependent clamp. If it is constant, the row it matches names
  // the column outright.
  const diff0 = new Float32Array(pairChannels);
  for (let c = 0; c < pairChannels; c += 1) {
    diff0[c] = positioned[c] - nativePositioned[c];
  }
  let maxDeviation = 0;
  for (let pairIndex = 0; pairIndex < tokens * tokens; pairIndex += 1) {
    for (let c = 0; c < pairChannels; c += 1) {
      const at = pairIndex * pairChannels + c;
      const d = positioned[at] - nativePositioned[at];
      maxDeviation = Math.max(maxDeviation, Math.abs(d - diff0[c]));
    }
  }
  // ...and which weight ROW it is. `positionActivations` is [relativeWidth,
  // pairChannels], so a column set where it should not be adds exactly its row.
  const rowMatches = [];
  for (let column = 0; column < w.relativeWidth; column += 1) {
    let error = 0, scale = 0;
    for (let c = 0; c < pairChannels; c += 1) {
      const weight = w.positionActivations[column * pairChannels + c];
      const d = Math.abs(diff0[c]) - Math.abs(weight);
      error += d * d; scale += weight * weight;
    }
    const score = Math.sqrt(error / Math.max(scale, 1e-30));
    if (score < 1e-3) rowMatches.push({ column, score: Number(score.toExponential(2)) });
  }
  // 🔴 IS THE REFERENCE'S PROJECTION JUST A GATHER? Its own comment says
  // `one_hot(idx, N) @ W == W[idx]`, so for one pair the native value must be
  // the SUM OF FOUR ROWS of `position_activations` - and if it is not, there is
  // a bias or a scale this port does not apply.
  const f = batch.features;
  const probePairs = [[0, 1], [3, 40], [20, 20]];
  const gatherCheck = probePairs.map(([i, j]) => {
    const clampTo = (v, hi) => Math.min(Math.max(v, 0), hi);
    const sameChain = f.asymId[i] === f.asymId[j];
    const sameEntity = f.entityId[i] === f.entityId[j];
    const posIdx = sameChain ? clampTo(f.residueIndex[i] - f.residueIndex[j] + 32, 64) : 65;
    const sameRes = sameChain && f.residueIndex[i] === f.residueIndex[j];
    const tokIdx = sameRes ? clampTo(f.tokenIndex[i] - f.tokenIndex[j] + 32, 64) : 65;
    const chainIdx = sameEntity
      ? clampTo(f.symId[i] - f.symId[j] + 2, 4) : 5;
    const rows = [posIdx, 66 + tokIdx, ...(sameEntity ? [132] : []), 133 + chainIdx];
    const summed = new Float32Array(pairChannels);
    for (const row of rows) {
      for (let c = 0; c < pairChannels; c += 1) {
        summed[c] += w.positionActivations[row * pairChannels + c];
      }
    }
    const at = (i * tokens + j) * pairChannels;
    const native = nativePositioned.slice(at, at + pairChannels);
    const ours = positioned.slice(at, at + pairChannels);
    return { pair: [i, j], rows,
             gatherVsNative: Number(relRms(summed, native).toExponential(2)),
             oursVsNative: Number(relRms(ours, native).toExponential(2)),
             oursVsGather: Number(relRms(ours, summed).toExponential(2)) };
  });
  // 🔴 IS IT bfloat16? 1.26e-3 is about 2^-10, which is the scale of bf16
  // rounding and NOT of any f32 summation order over four rows. The reference's
  // `_RelativeEncodingProjection` takes `dtype=pair_activations.dtype`, so if
  // the trunk it was dumped from ran the pair in bf16 this term is rounded and
  // every other one - which is computed elsewhere - is not.
  const toBf16 = (x) => {
    const buffer = new ArrayBuffer(4);
    new Float32Array(buffer)[0] = x;
    const bits = new Uint32Array(buffer);
    // round-to-nearest-even on the low 16 bits
    const rounding = 0x7fff + ((bits[0] >>> 16) & 1);
    bits[0] = (bits[0] + rounding) & 0xffff0000;
    return new Float32Array(buffer)[0];
  };
  const roundedWeights = Float32Array.from(w.positionActivations, toBf16);
  const gatherIn = (weights) => {
    const out = new Float32Array(tokens * tokens * pairChannels);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        const clampTo = (v, hi) => Math.min(Math.max(v, 0), hi);
        const sameChain = f.asymId[i] === f.asymId[j];
        const sameEntity = f.entityId[i] === f.entityId[j];
        const posIdx = sameChain ? clampTo(f.residueIndex[i] - f.residueIndex[j] + 32, 64) : 65;
        const sameRes = sameChain && f.residueIndex[i] === f.residueIndex[j];
        const tokIdx = sameRes ? clampTo(f.tokenIndex[i] - f.tokenIndex[j] + 32, 64) : 65;
        const chainIdx = sameEntity ? clampTo(f.symId[i] - f.symId[j] + 2, 4) : 5;
        const rows = [posIdx, 66 + tokIdx, ...(sameEntity ? [132] : []), 133 + chainIdx];
        const at = (i * tokens + j) * pairChannels;
        for (const row of rows) {
          for (let c = 0; c < pairChannels; c += 1) out[at + c] += weights[row * pairChannels + c];
        }
      }
    }
    return out;
  };
  const bf16Weights = relRms(gatherIn(roundedWeights), nativePositioned);
  const bf16Output = relRms(Float32Array.from(gatherIn(w.positionActivations), toBf16),
                            nativePositioned);
  // 🔴 WHICH ROWS IS THE NATIVE VALUE ACTUALLY MADE OF? Same weight (the bundle
  // agrees with the reference's params 404 of 404), same formula, same indices
  // - and still 1.26e-3. Subtract the three rows that are not in doubt and see
  // which of the 139 the remainder IS.
  const rowSearch = [[0, 1], [20, 20], [3, 40]].map(([i, j]) => {
    const clampTo = (v, hi) => Math.min(Math.max(v, 0), hi);
    const sameEntity = f.entityId[i] === f.entityId[j];
    const chainIdx = sameEntity ? clampTo(f.symId[i] - f.symId[j] + 2, 4) : 5;
    const at = (i * tokens + j) * pairChannels;
    const remainder = new Float32Array(pairChannels);
    for (let c = 0; c < pairChannels; c += 1) {
      remainder[c] = nativePositioned[at + c]
        - (sameEntity ? w.positionActivations[132 * pairChannels + c] : 0)
        - w.positionActivations[(133 + chainIdx) * pairChannels + c];
    }
    // ...the remainder should be w_pos[posIdx] + w_token[tokIdx]. Search every
    // (pos, token) pair for the best explanation.
    let best = null;
    for (let a2 = 0; a2 < 66; a2 += 1) {
      for (let b2 = 66; b2 < 132; b2 += 1) {
        let error = 0;
        for (let c = 0; c < pairChannels; c += 1) {
          const d = remainder[c] - w.positionActivations[a2 * pairChannels + c]
            - w.positionActivations[b2 * pairChannels + c];
          error += d * d;
        }
        if (best === null || error < best.error) best = { pos: a2, token: b2 - 66, error };
      }
    }
    const sameChain = f.asymId[i] === f.asymId[j];
    const ourPos = sameChain ? clampTo(f.residueIndex[i] - f.residueIndex[j] + 32, 64) : 65;
    const sameRes = sameChain && f.residueIndex[i] === f.residueIndex[j];
    const ourTok = sameRes ? clampTo(f.tokenIndex[i] - f.tokenIndex[j] + 32, 64) : 65;
    return { pair: [i, j], oursPos: ourPos, oursToken: ourTok,
             bestPos: best.pos, bestToken: best.token,
             residual: Number(Math.sqrt(best.error / pairChannels).toExponential(2)) };
  });
  // ...and the same gather accumulated STEPWISE in bf16, which is a different
  // number from rounding the finished sum.
  const stepwise = new Float32Array(tokens * tokens * pairChannels);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const clampTo = (v, hi) => Math.min(Math.max(v, 0), hi);
      const sameChain = f.asymId[i] === f.asymId[j];
      const sameEntity = f.entityId[i] === f.entityId[j];
      const posIdx = sameChain ? clampTo(f.residueIndex[i] - f.residueIndex[j] + 32, 64) : 65;
      const sameRes = sameChain && f.residueIndex[i] === f.residueIndex[j];
      const tokIdx = sameRes ? clampTo(f.tokenIndex[i] - f.tokenIndex[j] + 32, 64) : 65;
      const chainIdx = sameEntity ? clampTo(f.symId[i] - f.symId[j] + 2, 4) : 5;
      const rows = [posIdx, 66 + tokIdx, ...(sameEntity ? [132] : []), 133 + chainIdx];
      const at = (i * tokens + j) * pairChannels;
      for (let c = 0; c < pairChannels; c += 1) {
        let acc = 0;
        for (const row of rows) {
          acc = toBf16(acc + toBf16(w.positionActivations[row * pairChannels + c]));
        }
        stepwise[at + c] = acc;
      }
    }
  }
  const bf16Stepwise = relRms(stepwise, nativePositioned);
  const worst = arms.reduce((a, b) => (Number(a.relRms) > Number(b.relRms) ? a : b));
  return {
    tokens, featureWidth, pairChannels, relativeWidth: w.relativeWidth,
    bondEmbeddingIsZero: rms(of("scope.evoformer/bond_embedding")) === 0,
    arms, worst, gatherCheck, rowSearch,
    bf16: { weightsRounded: Number(bf16Weights.toExponential(2)),
            outputRounded: Number(bf16Output.toExponential(2)),
            stepwise: Number(bf16Stepwise.toExponential(2)) },
    differenceIsConstant: maxDeviation < 1e-5,
    maxDeviationFromConstant: Number(maxDeviation.toExponential(2)),
    constantRms: Number(rms(diff0).toFixed(6)),
    weightRowsMatchingTheConstant: rowMatches,
    badPairCount: badPairs.length, totalPairs: tokens * tokens,
    badPairsSample: badPairs.slice(0, 12),
    // The separations at which it goes wrong - the clamp is on |j - i|.
    badSeparations: [...new Set(badPairs.map((p) => p.d))].sort((a, b) => a - b),
    // The whole z_init for scale; the terms above must account for it.
    zInitNativeRms: Number(rms(of("tap.z_init_generic")).toFixed(4)),
  };
}
