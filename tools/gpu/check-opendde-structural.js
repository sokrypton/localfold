/**
 * OpenDDE's structural expander and refiner, against af3-any-model's own.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-opendde-structural.js \
 *       --model=/model-opendde-full-f32/manifest.json
 *
 * 🔴 THE STAGE THAT HAD NO ORACLE, AND THE ONE THE RESIDUAL WAS IN.
 * `check-opendde-expander.js` says it in its own header - "THERE IS NO ORACLE
 * FOR THIS, SO WHAT IS CHECKED IS CONSERVATION AND SHAPE" - so every numeric
 * mistake available here was invisible. Measured against a real native fold of
 * 6MRR, the structural single this stage hands the confidence head is relRMS
 * 9.6% from the reference's, which arrives as pLDDT 3.0% and PAE 10.0% at the
 * page. The head's OWN arithmetic is exact (8.46e-7, check-opendde-confidence-
 * oracle.js) and `s_inputs` into it matches to six digits, which is the control
 * that says the fault is here and not upstream or downstream.
 *
 * It is driven by the REFERENCE's residue-level embeddings, not by a fold, so a
 * difference is this stage's and not the trunk's. The dump is
 * `tools/oracle/dump_af3_opendde_structural.py`.
 *
 * 🔴 AND IT BISECTS THE REFINER BY DEPTH. The four blocks are one stacked
 * tensor, so running 1, 2, 3 and 4 of them and comparing the single each time
 * says which block first diverges - the same trick `--msa-blocks` plays on the
 * trunk. A stage-level number alone would only say "somewhere in four blocks".
 *
 * 🔴 THE REFERENCE PADS TO A BUCKET AND THIS PORT DOES NOT: 160 structural
 * tokens there against 130 here for the same 68 residues. Every comparison is
 * over the first `tokens` rows, and the dump's `in.seqMask` is asserted live
 * across exactly those - a silently short mask would make every relRMS look
 * good by comparing padding with padding.
 */
import { featuriseProtein } from "../../src/af3/featurise/featurise.js";
import { structuralBatch, structuralLayout }
  from "../../src/af3/featurise/structural-tokens.js";
import { expandStructural, structuralAttentionBias, structuralPairFeatures }
  from "../../src/af3/structure/structural-expander-reference.js";
import { Af3PairformerStackGpu } from "../../src/af3/trunk/pairformer-block-webgpu.js";
import {
  af3Dialect, openAf3Store, structuralExpanderWeights, structuralRefinerWeights,
} from "../../src/af3/weights/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** relRMS of `ours` against `theirs`, over the first `rows` rows of `width`. */
function relRms(ours, theirs, rows, width, theirWidth = width) {
  let diff = 0;
  let scale = 0;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const a = ours[row * width + column];
      const b = theirs[row * theirWidth + column];
      diff += (a - b) ** 2;
      scale += b * b;
    }
  }
  return Math.sqrt(diff / Math.max(scale, 1e-30));
}

/** Where the error lives: per row and per channel, which say different things. */
function structureOf(ours, theirs, rows, width, theirWidth = width) {
  const perRow = [];
  for (let row = 0; row < rows; row += 1) {
    let d = 0;
    let s = 0;
    for (let c = 0; c < width; c += 1) {
      const a = ours[row * width + c];
      const b = theirs[row * theirWidth + c];
      d += (a - b) ** 2; s += b * b;
    }
    perRow.push(Math.sqrt(d / Math.max(s, 1e-30)));
  }
  const perChannel = [];
  for (let c = 0; c < width; c += 1) {
    let d = 0;
    let s = 0;
    for (let row = 0; row < rows; row += 1) {
      const a = ours[row * width + c];
      const b = theirs[row * theirWidth + c];
      d += (a - b) ** 2; s += b * b;
    }
    perChannel.push(Math.sqrt(d / Math.max(s, 1e-30)));
  }
  const median = (v) => [...v].sort((x, y) => x - y)[v.length >> 1];
  const worst = (v, n) => v.map((x, i) => [i, x]).sort((a, b) => b[1] - a[1])
    .slice(0, n).map(([i, x]) => ({ at: i, relRms: Number(x.toFixed(4)) }));
  return {
    perRow: { min: Number(Math.min(...perRow).toFixed(4)),
              median: Number(median(perRow).toFixed(4)),
              max: Number(Math.max(...perRow).toFixed(4)) },
    perChannel: { min: Number(Math.min(...perChannel).toFixed(4)),
                  median: Number(median(perChannel).toFixed(4)),
                  max: Number(Math.max(...perChannel).toFixed(4)),
                  worst: worst(perChannel, 8) },
  };
}

export async function main(device, args) {
  const manifest = option(args, "model", "/model-opendde-full-f32/manifest.json");
  const dumpPath = option(args, "dump", "/oracle-dumps/af3-oracle-structural-opendde.json");
  const bound = Number(option(args, "bound", "1e-3"));
  const dump = await (await fetch(dumpPath)).json();
  const stages = dump.stages;
  const at = (name) => {
    const stage = stages[name];
    if (stage === undefined) throw new Error(`the dump has no stage ${name}`);
    return { data: Float32Array.from(stage.data), shape: stage.shape };
  };

  const sequence = option(args, "sequence", dump.sequence);
  if (sequence === undefined) throw new Error("the dump carries no sequence; pass --sequence=");
  const batch = featuriseProtein(sequence, {});
  const layout = structuralLayout(batch);
  const structural = structuralBatch(batch, layout);
  const tokens = layout.tokens;

  const store = await openAf3Store(manifest);
  const weights = await structuralExpanderWeights(store);
  const refinerWeights = await structuralRefinerWeights(store);
  // 🔴 THE BUNDLE'S OWN DIALECT, NOT A DEFAULT. `Af3PairformerStackGpu` reads
  // `swapTransposedBias` and a dozen more off it and RAISES rather than
  // guessing, which is what caught this line being absent.
  const dialect = af3Dialect(store);

  const theirTokens = at("expander.single").shape[0];
  const residueTokens = at("in.single").shape[0];
  if (residueTokens !== batch.tokens) {
    throw new Error(`the dump is ${residueTokens} residue tokens and this`
      + ` featurisation is ${batch.tokens}: the sequences differ`);
  }
  // 🔴 THE MASK MUST BE LIVE ACROSS EVERY ROW COMPARED, or padding is being
  // compared with padding and every number below is meaninglessly good.
  const theirMask = at("in.seqMask").data;
  let live = 0;
  for (let t = 0; t < theirTokens; t += 1) if (theirMask[t] > 0.5) live += 1;
  if (live !== tokens) {
    throw new Error(`the reference has ${live} live structural tokens and this`
      + ` port has ${tokens}; the layouts disagree before any arithmetic`);
  }

  // --- the expander, on the REFERENCE's residue embeddings ------------------
  const embeddings = {
    targetFeat: at("in.targetFeat").data,
    single: at("in.single").data,
    pair: at("in.pair").data,
    asymId: batch.asymId,
  };
  const expanded = expandStructural(layout, embeddings, weights,
    { residueTokens, pairChannels: weights.pairChannels });
  // The same two calls fold.js makes, in the same order; `structuralAttentionBias`
  // takes the PAIR FEATURES and not the structural batch.
  const features = structuralPairFeatures(layout, batch.asymId);
  const bias = structuralAttentionBias(layout, features, weights);

  const results = [];
  const record = (stage, ours, theirName, width) => {
    const theirs = at(theirName);
    const theirWidth = theirs.shape[theirs.shape.length - 1];
    const value = relRms(ours, theirs.data, tokens, width, theirWidth);
    results.push({ stage, relRms: Number(value.toExponential(2)),
                   ...structureOf(ours, theirs.data, tokens, width, theirWidth) });
    return value;
  };
  record("expander.targetFeat", expanded.targetFeat, "expander.targetFeat",
         weights.singleInputChannels);
  record("expander.single", expanded.single, "expander.single", weights.singleChannels);

  // --- the refiner, one block deeper each time ------------------------------
  const pairMask = new Float32Array(tokens * tokens);
  const seqMask = structural.seqMask;
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }
  for (let depth = 1; depth <= refinerWeights.length; depth += 1) {
    const out = await new Af3PairformerStackGpu(device, {}).run(
      { tokens, pair: expanded.pair.slice(), single: expanded.single.slice(),
        pairMask, seqMask },
      refinerWeights.slice(0, depth), dialect,
      { extraPairBias: bias });
    record(`refiner.block${depth - 1}.single`, out.single,
           `refiner.block${depth - 1}.single`, weights.singleChannels);
  }

  const worst = results.reduce((a, b) => (a.relRms > b.relRms ? a : b));
  // 🔴 A GATE THAT CANNOT FAIL IS NOT A GATE. `agrees` alone is a number a
  // reader has to notice; this throws, which is what makes a regression here
  // stop a batch. Watched failing at --bound=1e-5 on a healthy tree (the
  // refiner's fourth block reads 6.0e-5, which is four blocks of f32
  // accumulation and not a defect).
  if (!(worst.relRms <= bound)) {
    throw new Error(`${worst.stage} is ${worst.relRms} against a bound of ${bound}`
      + ` - OpenDDE's structural stage no longer matches af3-any-model's`);
  }
  return {
    model: "opendde", sequence: sequence.length, residueTokens,
    structuralTokens: tokens, referenceTokens: theirTokens,
    bound, stages: results, worst: { stage: worst.stage, relRms: worst.relRms },
    agrees: worst.relRms <= bound,
  };
}
