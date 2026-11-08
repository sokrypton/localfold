/**
 * AF3's trunk embedder in WebGL2, against TWO references.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-embedder.js [--falsify]
 *
 * `cpu/af3/trunk/embedder.js`'s `embed` is the f64 reference and `Af3EmbedderGpu`
 * the WGSL one, both in this page, with the two also held to each other.
 *
 * 🔴 THREE CHAINS, TWO OF THEM ONE ENTITY, because the relative encoding is four
 * one-hots and a single chain lights only one branch of each. The arms give
 * chains 0 and 1 the same entity id with different symmetry ids and chain 2 its
 * own, so every bin is reachable: the chain-relative offset AND its
 * "inter-chain" overflow bin, the residue-relative token offset AND its own,
 * same-entity both ways, and the relative symmetry copy AND its
 * "different-entity" bin.
 *
 * 🔴 AND THE 139 ONE-HOT COLUMNS ARE NEVER MATERIALISED HERE, so this arm is
 * what says the four summed weight rows are the four the reference multiplies.
 * The reference builds the full `tokens^2 x 139` matrix; a mis-mapped bin is a
 * term of the right magnitude in the wrong place, which only a cross-check sees.
 *
 * 🔴 AND THE MSA CODES INCLUDE OUT-OF-RANGE ONES, as the existing WGSL gate's
 * do: `floor(random * 34) - 1` reaches -1 and 32, and the one-hot must simply
 * not fire rather than reading off the end of the matrix.
 *
 * 🔴 AND THE TEMPLATE TERM IS PASSED AS ZEROS TO THE REFERENCE, which is NOT a
 * claim that zeros are right for a fold - "AF3's template embedder contributes
 * even with no templates, so omitting it is not the same as passing zeros", and
 * on four empty slots its output still has std 13.1. It is passed as zeros
 * because the module under test does not own that term: the template stack reads
 * the pair as it is at this point, so the trunk composes the two. Mistaking
 * this for "templates are optional" is the week this project already lost.
 */
import { embed as cpuEmbed } from "../../cpu/af3/trunk/embedder.js";
import { Af3EmbedderGpu } from "../../webgpu/af3/trunk/embedder.js";
import { setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  RELATIVE_WIDTH, embed, releaseEmbedderWeights, tokenIdentity, uploadEmbedderWeights,
} from "../../webgl/af3/trunk/embedder.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { generator } from "./af2-synthetic-block.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "3e-6"));
  const tokens = Number(option(args, "tokens", "9"));
  const sequences = Number(option(args, "sequences", "3"));
  const chains = Number(option(args, "chains", "3"));
  const featureWidth = Number(option(args, "feature-width", "40"));
  const pairChannels = Number(option(args, "pair-channels", "32"));
  const singleChannels = Number(option(args, "single-channels", "32"));
  const msaChannels = Number(option(args, "msa-channels", "16"));
  const msaFeatureWidth = Number(option(args, "msa-width", "34"));
  const pairInitFromSingle = flag(args, "pair-init-from-single");
  const withBonds = flag(args, "bonds");
  // 🔴 AT 35 COLUMNS THE DIALECT IS REQUIRED, and this port refuses without it
  // exactly as the reference does: "this bundle carries an is_paired column and
  // only the dialect says whether the query row is in it - boltz2 says yes,
  // rosettafold3 no". `--msa-width=35` without `--paired-query-row` is
  // rosettafold3, which carries the column and leaves it identically zero.
  const pairedQueryRow = flag(args, "paired-query-row");
  const pairs = tokens * tokens;
  const rows = sequences * tokens;

  const random = generator(223606);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };

  const pairSourceWidth = pairInitFromSingle ? singleChannels : featureWidth;
  const weights = {
    pairChannels, singleChannels, msaChannels, targetFeatWidth: featureWidth,
    relativeWidth: RELATIVE_WIDTH,
    dialect: {
      pairInitFromSingle,
      ...(msaFeatureWidth > 34 ? { msaPairedQueryRow: pairedQueryRow } : {}),
    },
    leftSingle: noise(pairSourceWidth * pairChannels, 1 / Math.sqrt(pairSourceWidth)),
    rightSingle: noise(pairSourceWidth * pairChannels, 1 / Math.sqrt(pairSourceWidth)),
    prevEmbeddingNormScale: noise(pairChannels, 0.2, 1),
    prevEmbeddingNormOffset: noise(pairChannels, 0.1),
    prevEmbedding: noise(pairChannels * pairChannels, 1 / Math.sqrt(pairChannels)),
    positionActivations: noise(RELATIVE_WIDTH * pairChannels, 0.3),
    msaActivations: noise(msaFeatureWidth * msaChannels, 0.3),
    extraMsaTargetFeat: noise(featureWidth * msaChannels, 1 / Math.sqrt(featureWidth)),
    singleActivations: noise(featureWidth * singleChannels, 1 / Math.sqrt(featureWidth)),
    prevSingleEmbeddingNormScale: noise(singleChannels, 0.2, 1),
    prevSingleEmbeddingNormOffset: noise(singleChannels, 0.1),
    prevSingleEmbedding: noise(
      singleChannels * singleChannels, 1 / Math.sqrt(singleChannels)),
    // 🔴 THE WGSL EMBEDDER REQUIRES THIS WEIGHT WHETHER OR NOT THERE ARE BONDS
    // ("embedder weights missing bondEmbedding"), which is how this port's own
    // missing term was found: it had skipped the branch as "zero for a
    // bond-free input", which is true of the CONTRIBUTION and not of the
    // weight. It is a real term for ligands and the `--bonds` arm runs it.
    bondEmbedding: noise(pairChannels, 0.3),
  };

  // Three chains; 0 and 1 share an entity with different symmetry ids.
  const perChain = Math.ceil(tokens / chains);
  const residueIndex = new Int32Array(tokens);
  const tokenIndex = new Int32Array(tokens);
  const asymId = new Int32Array(tokens);
  const entityId = new Int32Array(tokens);
  const symId = new Int32Array(tokens);
  for (let t = 0; t < tokens; t += 1) {
    const chain = Math.min(Math.floor(t / perChain), chains - 1);
    asymId[t] = chain;
    entityId[t] = chain === 1 ? 0 : chain;
    symId[t] = chain === 1 ? 1 : 0;
    residueIndex[t] = t - chain * perChain;
    // 🔴 TWO TOKENS OF ONE RESIDUE, so the residue-relative branch is reached:
    // for a plain protein every token is its own residue and that lookup is
    // the diagonal plus one saturated bin, which tests almost nothing.
    tokenIndex[t] = residueIndex[t];
  }
  if (tokens > 3) {
    residueIndex[2] = residueIndex[1];
    tokenIndex[2] = tokenIndex[1] + 1;
  }

  const msaRows = new Int32Array(rows);
  const deletionMatrix = new Float32Array(rows);
  for (let index = 0; index < rows; index += 1) {
    // ...including out-of-range codes, which must simply not fire.
    msaRows[index] = Math.floor(random() * 34) - 1;
    deletionMatrix[index] = random() * 6;
  }
  // A sparse contact matrix, as the existing WGSL gate builds: a few covalent
  // links inside the second chain.
  const bondMatrix = withBonds ? new Float32Array(pairs) : undefined;
  if (bondMatrix !== undefined) {
    for (let t = perChain; t + 1 < tokens; t += 3) {
      bondMatrix[t * tokens + t + 1] = 1;
      bondMatrix[(t + 1) * tokens + t] = 1;
    }
  }
  const targetFeat = noise(tokens * featureWidth, 1);
  const previousPair = noise(pairs * pairChannels, 0.5);
  const previousSingle = noise(tokens * singleChannels, 0.5);

  setHalfPrecision(device, false);
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const inputTensors = [];
  const upload = (values, label) => {
    const tensor = execution.upload(Float32Array.from(values), label);
    inputTensors.push(tensor);
    return tensor;
  };
  const uploaded = uploadEmbedderWeights(execution, weights);
  // The control skews half of each target row's channels: the target feature
  // reaches the pair, the MSA and the single track, so it is the one input every
  // arm reads - and half a row survives the LayerNorms downstream.
  const targetTensor = upload(falsify
    ? Float32Array.from(targetFeat,
      (v, i) => (i % featureWidth < featureWidth / 2 ? v * 1.01 + 0.01 : v))
    : targetFeat, "af3.embed.target");
  const identity = upload(
    tokenIdentity({ residueIndex, tokenIndex, asymId, entityId, symId }, tokens),
    "af3.embed.identity");
  const ours = embed(execution, {
    targetFeat: targetTensor,
    identity,
    msaCodes: upload(msaRows, "af3.embed.codes"),
    deletions: upload(deletionMatrix, "af3.embed.deletions"),
    previousPair: upload(previousPair, "af3.embed.previous-pair"),
    previousSingle: upload(previousSingle, "af3.embed.previous-single"),
    weights: uploaded, tokens, sequences, pairInitFromSingle,
    ...(msaFeatureWidth > 34 ? { msaPairedQueryRow: pairedQueryRow } : {}),
    ...(bondMatrix === undefined
      ? {} : { bondMatrix: upload(bondMatrix, "af3.embed.bonds") }),
  });
  const failuresEarly = [];
  const mine = {
    pair: Float32Array.from(execution.read(ours.pair)),
    msa: Float32Array.from(execution.read(ours.msa)),
    single: Float32Array.from(execution.read(ours.single)),
  };
  for (const tensor of Object.values(ours)) execution.release(tensor);
  releaseEmbedderWeights(execution, uploaded);

  // 🔴 AND AT 35 COLUMNS THE FLAG MUST MOVE THE ANSWER, or both conventions are
  // one convention twice. The 35th column is 1 on the QUERY ROW alone, so the
  // MSA embedding differs on `tokens` rows of `rows` - and nowhere else, which
  // is why this is measured on the MSA and not on the pair.
  let pairedMoves;
  if (msaFeatureWidth > 34) {
    const other = uploadEmbedderWeights(execution, weights);
    const flipped = embed(execution, {
      targetFeat: targetTensor,
      identity,
      msaCodes: upload(msaRows, "af3.embed.codes.flipped"),
      deletions: upload(deletionMatrix, "af3.embed.deletions.flipped"),
      previousPair: upload(previousPair, "af3.embed.previous-pair.flipped"),
      previousSingle: upload(previousSingle, "af3.embed.previous-single.flipped"),
      weights: other, tokens, sequences, pairInitFromSingle,
      msaPairedQueryRow: !pairedQueryRow,
      ...(bondMatrix === undefined
        ? {} : { bondMatrix: upload(bondMatrix, "af3.embed.bonds.flipped") }),
    });
    pairedMoves = Number(relativeRms(
      Float32Array.from(execution.read(flipped.msa)), mine.msa, "paired.moves")
      .toPrecision(3));
    for (const tensor of Object.values(flipped)) execution.release(tensor);
    releaseEmbedderWeights(execution, other);
    if (!(pairedMoves > 1e-4)) {
      failuresEarly.push(
        `the is_paired flag is INERT: the two conventions agree to ${pairedMoves}`);
    }
  }

  const input = {
    tokens, sequences, targetFeat,
    features: { residueIndex, tokenIndex, asymId, entityId, symId },
    msaRows, deletionMatrix, previousPair, previousSingle,
    ...(bondMatrix === undefined ? {} : { bondMatrix }),
  };
  const expected = cpuEmbed(
    { ...input, templateEmbedding: new Float32Array(pairs * pairChannels) }, weights);
  const theirs = await new Af3EmbedderGpu(device).run(input, weights);

  const arms = [];
  const failures = [...failuresEarly];
  for (const track of ["pair", "msa", "single"]) {
    for (const [label, want] of [["f64 reference", expected], ["WGSL", theirs]]) {
      const name = `${track} vs the ${label}`;
      const relative = relativeRms(mine[track], want[track], name);
      const ok = relative <= bar;
      arms.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
      if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
    }
    const between = relativeRms(expected[track], theirs[track], `between.${track}`);
    arms.push({
      arm: `${track}: the two references agree`,
      relativeRms: Number(between.toPrecision(4)), bar, ok: between <= bar,
      falsifiable: false,
      reason: "both references read the unperturbed target feature",
    });
    if (!(between <= bar)) {
      failures.push(`${track}: the references disagree by ${between.toExponential(3)}`);
    }
  }

  for (const tensor of inputTensors) execution.release(tensor);
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const falsifiable = arms.filter((row) => row.falsifiable !== false);
  const red = falsifiable.filter((row) => !row.ok).length;
  return {
    shape: {
      tokens, sequences, chains, featureWidth, pairChannels, singleChannels,
      msaChannels, msaFeatureWidth, relativeWidth: RELATIVE_WIDTH,
    },
    pairInitFromSingle,
    pairedQueryRow: msaFeatureWidth > 34 ? pairedQueryRow : "no is_paired column",
    pairedMoves,
    bonds: withBonds
      ? bondMatrix.reduce((a, b) => a + b, 0) : "none (the term is zero, the weight is not)",
    sharedEntityChains: [0, 1],
    twoTokensOfOneResidueAt: tokens > 3 ? [1, 2] : null,
    bar,
    falsify,
    arms,
    memory: execution.stats(),
    verdict: falsify
      ? (red === falsifiable.length
        ? `ok: all ${falsifiable.length} falsifiable arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${falsifiable.length} falsifiable arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms over three tracks, both references agreeing`
        : `FAILED: ${failures.join("; ")}`),
  };
}
