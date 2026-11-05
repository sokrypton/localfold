/**
 * ESMFold2's pair features, folding trunk and distogram head in WebGL2 against
 * `cpu/esmfold2/`.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-esmfold2-trunk.js [--falsify]
 *
 * 🔴 TWO CHAINS, BECAUSE THE RELATIVE ENCODING'S CHAIN BLOCK HAS THE OPPOSITE
 * POLARITY TO THE OTHER TWO AND A MONOMER CANNOT SEE IT. The residue and token
 * blocks send a DIFFERENT-chain pair to the out-of-range bin; the chain block
 * sends a SAME-chain one there. "Getting it backwards is a monomer that still
 * folds - every pair is same-chain, so the whole block becomes one constant
 * column either way - and a COMPLEX that is quietly wrong." The gate asserts
 * that swapping the polarity MOVES the answer on this layout, so the arm is
 * known to reach it.
 *
 * 🔴 AND THE DISTOGRAM HEAD IS ASKED BOTH WAYS. Upstream writes
 * `distogram_head(z + z.transpose(-2, -3))`, and feeding it `z` alone "conforms
 * in shape, produces a plausible distogram, and is a different model" - so the
 * unsymmetrised form is an arm that must differ.
 *
 * 🔴 AND THE TRUNK IS AF3's PAIRFORMER BLOCK WITHOUT THE GRID ATTENTIONS OR THE
 * SINGLE TRACK, which is why this gate exists beside `check-webgl-af3-block.js`:
 * the three kernels are shared and the COMPOSITION is not. The reference
 * composes the same three pieces out of `cpu/af3/trunk/pairformer.js` for the
 * same reason.
 */
import {
  distogramLogits as cpuDistogram, distogramProbabilities as cpuProbabilities,
} from "../../cpu/esmfold2/distogram.js";
import {
  relativePositionEncoding, recycleProjection as cpuRecycle, tokenBondEncoding,
  zInitFromInputs,
} from "../../cpu/esmfold2/pair-features.js";
import { transition as cpuTransition, triangleMultiplication as cpuTriangle }
  from "../../cpu/af3/trunk/pairformer.js";
import {
  distogramLogits, distogramProbabilities,
} from "../../webgl/esmfold2/distogram.js";
import {
  RELATIVE_WIDTH, recycleProjection, relativeEncodingSource, tokenIdentity, zInit,
} from "../../webgl/esmfold2/pair-features.js";
import { foldingTrunk } from "../../webgl/esmfold2/trunk.js";
import { sumSource } from "../../webgl/esmfold2/atom-transformer.js";
import { uploadTriangleWeights, releaseTriangleWeights }
  from "../../webgl/af3/trunk/triangle.js";
import { uploadTransitionWeights, releaseTransitionWeights }
  from "../../webgl/af3/trunk/transition.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { generator } from "./af2-synthetic-block.js";
import { pairHalfGroups } from "./af3-synthetic-pairformer.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "5e-6"));
  const tokens = Number(option(args, "tokens", "8"));
  const channels = Number(option(args, "channels", "32"));
  const inputChannels = Number(option(args, "input-channels", "24"));
  const bins = Number(option(args, "bins", "16"));
  const blocks = Number(option(args, "blocks", "2"));
  const pairs = tokens * tokens;

  const random = generator(1414213);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };

  // Two chains, two entities and a symmetry copy, so all four relative blocks
  // have something to say.
  const half = Math.ceil(tokens / 2);
  const residueIndex = new Int32Array(tokens);
  const tokenIndex = new Int32Array(tokens);
  const asymId = new Int32Array(tokens);
  const entityId = new Int32Array(tokens);
  const symId = new Int32Array(tokens);
  for (let t = 0; t < tokens; t += 1) {
    const chain = t < half ? 0 : 1;
    asymId[t] = chain;
    entityId[t] = chain === 0 ? 0 : 1;
    symId[t] = chain;
    residueIndex[t] = t - chain * half;
    // ...one residue carries two tokens, which is the only case the token block
    // is not a copy of the residue block.
    tokenIndex[t] = residueIndex[t] + (t % 3 === 0 ? 1 : 0);
  }
  const features = { residueIndex, tokenIndex, asymId, entityId, symId };

  const xInputs = noise(tokens * inputChannels, 1);
  const bonds = new Float32Array(pairs);
  for (let i = 0; i < tokens - 1; i += 1) {
    bonds[i * tokens + i + 1] = 1;
    bonds[(i + 1) * tokens + i] = 1;
  }
  const languageModelPair = noise(pairs * channels, 1);
  const mask = new Float32Array(pairs).fill(1);
  const hostWeights = {
    zInit1: noise(inputChannels * channels, 1 / Math.sqrt(inputChannels)),
    zInit2: noise(inputChannels * channels, 1 / Math.sqrt(inputChannels)),
    relativePosition: noise(RELATIVE_WIDTH * channels, 0.3),
    tokenBonds: noise(channels, 0.3),
  };
  // 🔴 THE TRANSITION FACTOR IS 4 AND THE WEIGHT IS WHAT SAYS SO: the shipped
  // bundle's `transition1` is [256, 2048], which is 2 x (4 x 256) - the blocked
  // double width around an intermediate of FOUR times the channels, not the
  // manifest's `transitionMultiplier: 2`. `cpu/af3/trunk/pairformer.js`'s
  // `transition` takes the factor as an argument defaulting to 4 and this
  // trunk's reference passes none, so factor-2 weights are read at the wrong
  // stride and come back NaN - which is what this gate did on its first run,
  // for the second time in this campaign.
  const trunkBlocks = Array.from({ length: blocks }, () => {
    const group = pairHalfGroups(noise, channels, 4);
    return {
      triangleMultiplicationOutgoing: group.triangleMultiplicationOutgoing,
      triangleMultiplicationIncoming: group.triangleMultiplicationIncoming,
      pairTransition: group.pairTransition,
    };
  });
  const recycleWeights = {
    scale: noise(channels, 0.2, 1),
    offset: noise(channels, 0.1),
    projection: noise(channels * channels, 1 / Math.sqrt(channels)),
  };
  const distogramWeights = noise(channels * bins, 1 / Math.sqrt(channels));
  const distogramBias = noise(bins, 0.1);

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  const uploadTree = (value, label) => {
    if (value instanceof Float32Array) return upload(value, label);
    if (Array.isArray(value)) return value.map((v, i) => uploadTree(v, `${label}.${i}`));
    if (typeof value === "number" || typeof value === "boolean") return value;
    return Object.fromEntries(Object.entries(value).map(
      ([key, v]) => [key, uploadTree(v, `${label}.${key}`)]));
  };
  // 🔴 THE SKEW GOES TO THE DEVICE ALONE and it has to reach both the per-token
  // term and the pair term, because z_init's five summands reach different arms.
  const skew = (values, period) => Float32Array.from(values,
    (v, i) => (i % period < period / 2 ? v * 1.01 + 0.01 : v));
  const uploaded = uploadTree(hostWeights, "w.ef2.pair");
  const uploadedBlocks = trunkBlocks.map((block) => ({
    triangleMultiplicationOutgoing: uploadTriangleWeights(
      execution, block.triangleMultiplicationOutgoing),
    triangleMultiplicationIncoming: uploadTriangleWeights(
      execution, block.triangleMultiplicationIncoming),
    pairTransition: uploadTransitionWeights(execution, block.pairTransition),
  }));
  const inputs = {
    xInputs: upload(falsify ? skew(xInputs, inputChannels) : xInputs, "ef2.x"),
    identity: upload(tokenIdentity(features, tokens), "ef2.identity"),
    bonds: upload(bonds, "ef2.bonds"),
    languageModelPair: upload(
      falsify ? skew(languageModelPair, channels) : languageModelPair, "ef2.lm-pair"),
  };
  const maskTensor = upload(mask, "ef2.mask");

  const arms = [];
  const failures = [];
  const separations = {};

  // --- z_init ------------------------------------------------------------
  const initTensor = zInit(execution, {
    ...inputs, tokens, inputChannels, channels, weights: uploaded,
  });
  const ourInit = Float32Array.from(execution.read(initTensor));
  const theirInit = (() => {
    const out = zInitFromInputs(xInputs, tokens, inputChannels, channels,
      hostWeights.zInit1, hostWeights.zInit2);
    const relative = relativePositionEncoding(features, tokens, channels,
      hostWeights.relativePosition);
    const bonded = tokenBondEncoding(bonds, tokens, channels, hostWeights.tokenBonds);
    for (let i = 0; i < out.length; i += 1) {
      out[i] += relative[i] + bonded[i] + languageModelPair[i];
    }
    return out;
  })();
  const report = (name, mine, theirs) => {
    const relative = relativeRms(mine, theirs, name);
    const ok = relative <= bar;
    arms.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };
  report("z_init, every term", ourInit, theirInit);

  // --- the relative encoding's chain polarity ---------------------------
  // The reference with the chain block's polarity SWAPPED, which is what says
  // this layout can see it at all.
  {
    const straight = relativePositionEncoding(features, tokens, channels,
      hostWeights.relativePosition);
    // 🔴 THE SECOND CHAIN'S sym_id MOVED, NOT EVERY CHAIN'S. Adding one to all
    // of them leaves every DIFFERENCE where it was, so the chain block's bin does
    // not move and the separation read exactly 0 - a control that cannot vary
    // the thing under test. Only the cross-chain difference decides that bin.
    const movedSym = relativePositionEncoding(
      { ...features, symId: Int32Array.from(symId, (v) => (v === 1 ? 2 : v)) },
      tokens, channels, hostWeights.relativePosition);
    separations["the chain block reads sym_id"] = Number(
      relativeRms(movedSym, straight, "chain").toPrecision(3));
    if (!(separations["the chain block reads sym_id"] > 1e-4)) {
      failures.push("the chain block is INERT on this layout: a monomer cannot see it");
    }
  }

  // --- the trunk ---------------------------------------------------------
  const trunkTensor = foldingTrunk(execution, {
    pair: initTensor, mask: maskTensor, tokens, channels, blocks: uploadedBlocks,
  });
  const ourTrunk = Float32Array.from(execution.read(trunkTensor));
  /** The reference trunk over an arbitrary input, so the loop arm can reuse it. */
  const cpuTrunkOf = (from) => {
    const pair = Float32Array.from(from);
    const add = (delta) => {
      for (let i = 0; i < pair.length; i += 1) pair[i] += delta[i];
    };
    for (const block of trunkBlocks) {
      add(cpuTriangle(pair, mask, tokens, channels, "outgoing",
        block.triangleMultiplicationOutgoing));
      add(cpuTriangle(pair, mask, tokens, channels, "incoming",
        block.triangleMultiplicationIncoming));
      add(cpuTransition(pair, pairs, channels, block.pairTransition));
    }
    return pair;
  };
  const theirTrunk = cpuTrunkOf(theirInit);
  report(`the trunk, ${blocks} blocks`, ourTrunk, theirTrunk);

  // --- the recycle projection -------------------------------------------
  const recycled = recycleProjection(execution, {
    pair: trunkTensor, pairs, channels, weights: uploadTree(recycleWeights, "w.ef2.recycle"),
  });
  report("pair_loop_proj", Float32Array.from(execution.read(recycled)),
    cpuRecycle(theirTrunk, pairs, channels, recycleWeights));
  execution.release(recycled);

  // --- the COMPOSED LOOP, which is what a fold runs and what nothing gated --
  //
  // 🔴 THE SEAM, NOT THE KERNELS EITHER SIDE OF IT. Every arm above is one
  // stage against the f64 reference, and all of them were green while a whole
  // ESMFold2 fold read mean pLDDT 57.5 against the WGSL fold's 88.1 and got
  // WORSE with every recycle (69.5, 58.8, 57.5 at one, two and four loops).
  // webgl/README.md records that as an open defect. A stage gate cannot see it
  // because the fault is in the COMPOSITION: `z = trunk(z_init + proj(z))`,
  // started from zero and iterated, which no gate here has ever run.
  //
  // This arm runs exactly that, `loops` times, on both sides. The reference
  // starts `z` at ZEROS and applies the projection on EVERY loop including the
  // first - `pair_loop_proj(0)` is not zero, because the LayerNorm in front of
  // it has an offset - which is the convention webgpu/esmfold2/fold.js states
  // and which this driver got wrong until it was measured.
  {
    const loops = Number(option(args, "loops", "2"));
    let ours = execution.zeros(pairs * channels, "ef2.check.z0");
    for (let loop = 0; loop < loops; loop += 1) {
      const update = recycleProjection(execution, {
        pair: ours, pairs, channels,
        weights: uploadTree(recycleWeights, "w.ef2.recycle"),
      });
      const input = execution.allocate(pairs * channels, "ef2.check.input");
      execution.dispatch({
        source: sumSource(pairs * channels),
        output: input,
        inputs: { aTex: initTensor, bTex: update },
        label: `ef2.check.recycle-add:${pairs}x${channels}`,
      });
      execution.release(update);
      execution.release(ours);
      ours = foldingTrunk(execution, {
        pair: input, mask: maskTensor, tokens, channels, blocks: uploadedBlocks,
      });
      execution.release(input);
    }
    let theirs = new Float32Array(pairs * channels);
    for (let loop = 0; loop < loops; loop += 1) {
      const update = cpuRecycle(theirs, pairs, channels, recycleWeights);
      const input = Float32Array.from(theirInit);
      for (let i = 0; i < input.length; i += 1) input[i] += update[i];
      theirs = cpuTrunkOf(input);
    }
    report(`the composed loop, ${loops} loops`,
      Float32Array.from(execution.read(ours)), theirs);
    execution.release(ours);
  }

  // --- the distogram head, both ways ------------------------------------
  const weightTensor = upload(distogramWeights, "w.ef2.distogram");
  const biasTensor = upload(distogramBias, "w.ef2.distogram.bias");
  const logitsTensor = distogramLogits(execution, {
    pair: trunkTensor, tokens, channels, bins,
    weights: weightTensor, bias: biasTensor,
  });
  const ourLogits = Float32Array.from(execution.read(logitsTensor));
  report("the distogram head, symmetrised", ourLogits,
    cpuDistogram(theirTrunk, tokens, channels, distogramWeights, distogramBias, bins));
  const plainTensor = distogramLogits(execution, {
    pair: trunkTensor, tokens, channels, bins,
    weights: weightTensor, bias: biasTensor, symmetrise: false,
  });
  const ourPlain = Float32Array.from(execution.read(plainTensor));
  report("the distogram head, unsymmetrised", ourPlain,
    cpuDistogram(theirTrunk, tokens, channels, distogramWeights, distogramBias, bins,
      false));
  separations["the head's symmetrisation"] = Number(
    relativeRms(ourPlain, ourLogits, "symmetrise").toPrecision(3));
  if (!(separations["the head's symmetrisation"] > 1e-3)) {
    failures.push("the symmetrisation changes nothing, so the arm cannot see it");
  }
  execution.release(plainTensor);

  const probabilities = distogramProbabilities(execution, {
    logits: logitsTensor, pairs, bins,
  });
  report("the distogram's probabilities",
    Float32Array.from(execution.read(probabilities)),
    cpuProbabilities(cpuDistogram(theirTrunk, tokens, channels, distogramWeights,
      distogramBias, bins), pairs, bins));
  execution.release(probabilities);
  execution.release(logitsTensor);
  execution.release(trunkTensor);
  execution.release(initTensor);

  for (const block of uploadedBlocks) {
    releaseTriangleWeights(execution, block.triangleMultiplicationOutgoing);
    releaseTriangleWeights(execution, block.triangleMultiplicationIncoming);
    releaseTransitionWeights(execution, block.pairTransition);
  }
  for (const tensor of held) execution.release(tensor);
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const red = arms.filter((row) => !row.ok).length;
  return {
    shape: {
      tokens, channels, inputChannels, bins, blocks, chains: 2,
      relativeWidth: RELATIVE_WIDTH,
    },
    bar,
    falsify,
    arms,
    separations,
    memory: execution.stats(),
    verdict: falsify
      ? (red === arms.length
        ? `ok: all ${arms.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${arms.length} arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms against the f64 reference, two chains`
        : `FAILED: ${failures.join("; ")}`),
  };
}
