/**
 * AF3's whole trunk on the GPU, against src/af3/trunk-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-trunk.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-trunk.js --n=256 --blocks=48 --no-check
 *
 * Every stage has its own checker. What this adds is the ORDER - in particular
 * that the template embedding goes in on the part-built pair, after the
 * relative encoding and before the MSA stack. Adding it to the finished pair
 * instead is a natural reading of "add the template embedding" and a different
 * model, and it runs.
 *
 * 🔴 THE PAIR IS COMPARED AGAINST A CONDITIONING ENVELOPE, not a constant. Past
 * a few pairformer blocks a random pair representation is chaotic: two CPU runs
 * differing by 1e-7 at the input diverge to ~6e-4. See tools/gpu/check-af3-block.js.
 */
import { runTrunk } from "../../src/af3/trunk-reference.js";
import { templateEmbedding } from "../../src/af3/template-reference.js";
import { Af3TrunkGpu } from "../../src/af3/trunk-webgpu.js";
import { binEdges as binEdgesOf } from "../../src/af3/trunk-webgpu.js";
import { openAf3Store, trunkWeights } from "../../src/af3/weights.js";

const DIALECT = { swapTransposedBias: false };

function option(args, name, fallback) {
  const prefix = `--${name}=`;
  return args.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

function deterministic(length, seed) {
  let state = seed >>> 0;
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    output[index] = (((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000) * 2 - 1;
  }
  return output;
}

function relativeRms(actual, expected) {
  let error = 0;
  let scale = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const difference = actual[index] - expected[index];
    error += difference * difference;
    scale += expected[index] * expected[index];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
}

function buildInput(tokens, sequences, chains) {
  const perChain = Math.ceil(tokens / chains);
  const residueIndex = new Int32Array(tokens);
  const asymId = new Int32Array(tokens);
  const entityId = new Int32Array(tokens);
  const symId = new Int32Array(tokens);
  for (let t = 0; t < tokens; t += 1) {
    const chain = Math.floor(t / perChain);
    asymId[t] = chain;
    entityId[t] = chain === 1 ? 0 : chain;
    symId[t] = chain === 1 ? 1 : 0;
    residueIndex[t] = t - chain * perChain;
  }
  const seqMask = new Float32Array(tokens);
  for (let t = 0; t < tokens; t += 1) seqMask[t] = t < Math.ceil(tokens * 0.85) ? 1 : 0;
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }
  const msaRows = new Int32Array(sequences * tokens);
  const deletionMatrix = new Float32Array(sequences * tokens);
  const msaMask = new Float32Array(sequences * tokens);
  for (let s = 0; s < sequences; s += 1) {
    for (let t = 0; t < tokens; t += 1) {
      const index = s * tokens + t;
      msaRows[index] = (s * 7 + t * 3) % 32;
      deletionMatrix[index] = ((s * 5 + t) % 9) / 2;
      msaMask[index] = seqMask[t] > 0 && ((s * 7 + t * 3) % 11) < 8 ? 1 : 0;
    }
  }
  return {
    tokens, sequences, templates: 4,
    targetFeat: deterministic(tokens * 447, 11 + tokens),
    features: { residueIndex, tokenIndex: residueIndex, asymId, entityId, symId },
    msaRows, deletionMatrix, msaMask, pairMask, seqMask,
    previousPair: new Float32Array(tokens * tokens * 128),
    previousSingle: new Float32Array(tokens * 384),
  };
}

export async function main(device, args) {
  const tokens = Number(option(args, "n", "24"));
  const sequences = Number(option(args, "sequences", "8"));
  const blocks = Number(option(args, "blocks", "4"));
  const store = await openAf3Store();
  const weights = await trunkWeights(store, blocks, 4);
  const input = buildInput(tokens, sequences, 3);

  // 🔴 THE STAGED WORKGROUP BLOCKS' PRECISION IS AN AXIS HERE TOO. The pair
  // track stages grid attention's key and value and the transition's two blocks
  // in f16 wherever the device has shader-f16, so the assembled trunk cannot be
  // held to the f32 path's number - and raising the one bound would stop the
  // f32 path being checked at the 4.3e-6 it actually reaches.
  const stagedPrecision = option(args, "staged",
    device.features.has("shader-f16") ? "f16" : "f32");
  const staged16 = stagedPrecision === "f16";
  const gpu = await new Af3TrunkGpu(device, { stagedPrecision }).run(input, weights, DIALECT, {
    onStage: (name, ms) => console.log(`  ${name}\t${ms.toFixed(0)} ms`),
  });

  if (args.includes("--no-check")) {
    const total = Object.values(gpu.timings).reduce((sum, value) => sum + value, 0);
    console.log(`trunk\tn=${tokens}\t${blocks} pairformer blocks\t${total.toFixed(0)} ms`);
    return { tokens, blocks, timings: gpu.timings };
  }

  // The reference, with the template embedder wired the same way round.
  const cpu = runTrunk({ ...input,
    templateEmbedding: (pair) => templateEmbedding(
      { pair, pairMask: input.pairMask, tokens, templates: 4, templateOccupied: false },
      weights.template, DIALECT),
  }, weights, DIALECT);

  // The conditioning envelope, as in check-af3-block.js.
  const perturbation = Number(option(args, "perturbation", "1e-7"));
  const perturbed = { ...input, targetFeat: Float32Array.from(input.targetFeat) };
  for (let index = 0; index < perturbed.targetFeat.length; index += 1) {
    perturbed.targetFeat[index] += input.targetFeat[index] * perturbation;
  }
  const control = runTrunk({ ...perturbed,
    templateEmbedding: (pair) => templateEmbedding(
      { pair, pairMask: input.pairMask, tokens, templates: 4, templateOccupied: false },
      weights.template, DIALECT),
  }, weights, DIALECT);
  const envelope = relativeRms(control.pair, cpu.pair);
  // 🔴 THE SOFTMAX HEADS GET THEIR OWN ENVELOPE. contactProbs is a softmax
  // ratio over 64 bins, so it amplifies whatever error reaches the logits by a
  // factor that depends on their spread - comparing it against the PAIR's
  // envelope would be comparing two different sensitivities.
  const contactEnvelope = relativeRms(control.contactProbs, cpu.contactProbs);
  const logitsEnvelope = relativeRms(control.logits, cpu.logits);

  // 🔴 SPLITTING THE CONTACT HEAD'S ERROR IN TWO. contactProbs is a softmax
  // ratio, so it amplifies whatever reaches the logits. Recomputing it on the
  // CPU FROM THE GPU'S OWN LOGITS separates the two possible sources: agreement
  // here means the GPU head's arithmetic is sound and the residual is the
  // trunk's error being amplified; disagreement would mean the head itself.
  const contactFromLogits = (logits) => {
    const breaks = binEdgesOf();
    const spacing = breaks[breaks.length - 1] - breaks[breaks.length - 2];
    const isContact = [];
    for (let bin = 0; bin < 64; bin += 1) {
      const top = bin < 63 ? breaks[bin] : breaks[breaks.length - 1] + spacing;
      isContact.push(top <= 8.0 + 1e-3);
    }
    const output = new Float32Array(tokens * tokens);
    for (let row = 0; row < tokens * tokens; row += 1) {
      const base = row * 64;
      let largest = -Infinity;
      for (let bin = 0; bin < 64; bin += 1) {
        if (logits[base + bin] > largest) largest = logits[base + bin];
      }
      let total = 0;
      let contact = 0;
      for (let bin = 0; bin < 64; bin += 1) {
        const probability = Math.exp(logits[base + bin] - largest);
        total += probability;
        if (isContact[bin]) contact += probability;
      }
      output[row] = input.pairMask[row] * (contact / total);
    }
    return output;
  };
  const headOnly = relativeRms(gpu.contactProbs, contactFromLogits(gpu.logits));
  const amplified = relativeRms(contactFromLogits(gpu.logits), cpu.contactProbs);

  const pairRms = relativeRms(gpu.pair, cpu.pair);
  const singleRms = relativeRms(gpu.single, cpu.single);
  const contactRms = relativeRms(gpu.contactProbs, cpu.contactProbs);
  const logitsRms = relativeRms(gpu.logits, cpu.logits);
  console.log(`trunk\tn=${tokens}, ${sequences} sequences, ${blocks} pairformer blocks`);
  console.log(`pair\trelRMS ${pairRms.toExponential(2)}`
    + `\t(envelope ${envelope.toExponential(2)}, ${(pairRms / Math.max(envelope, 1e-30)).toFixed(1)}x)`);
  console.log(`single\trelRMS ${singleRms.toExponential(2)}`);
  console.log(`contact\trelRMS ${contactRms.toExponential(2)}`
    + `\t(envelope ${contactEnvelope.toExponential(2)},`
    + ` ${(contactRms / Math.max(contactEnvelope, 1e-30)).toFixed(1)}x)`);
  console.log(`  of which the GPU head's own arithmetic: ${headOnly.toExponential(2)}`);
  console.log(`  and the trunk's logits amplified:       ${amplified.toExponential(2)}`);
  console.log(`logits\trelRMS ${logitsRms.toExponential(2)}`
    + `\t(envelope ${logitsEnvelope.toExponential(2)},`
    + ` ${(logitsRms / Math.max(logitsEnvelope, 1e-30)).toFixed(1)}x)`);

  // 🔴 DERIVED FROM THE ARITHMETIC, NOT FROM WHAT PASSES. Forty-eight blocks of
  // f16-staged tiles measure 1.04e-5 where the f32 path measures 6.18e-7, and
  // the contact probabilities - the most sensitive thing the trunk emits - go
  // 9.76e-5 to 1.86e-4, which is a factor of two on a number that is already
  // 19x its own conditioning envelope in f32. 4e-5 keeps a margin over the
  // measurement without leaving room for a bug, which would move this by orders
  // rather than by a factor. The f32 arm keeps 1e-5 and measures 6.18e-7.
  const pairBound = staged16 ? 4e-5 : Math.max(1e-5, envelope * 10);
  if (pairRms > pairBound) {
    throw new Error(`pair relRMS ${pairRms.toExponential(2)} exceeds ${pairBound.toExponential(2)}`);
  }
  const singleBound = staged16 ? 4e-5 : 1e-5;
  if (singleRms > singleBound) {
    throw new Error(`single relRMS ${singleRms.toExponential(2)} exceeds ${singleBound}`);
  }
  // 🔴 THE CONTACT PROBABILITIES DO NOT GET A TIGHT BOUND, AND SETTING ONE
  // WOULD BE A MISTAKE. They are a softmax ratio over 64 bins, and measurement
  // says they amplify the logits they come from by about 235x: the GPU's logits
  // differ from the reference by 4.6e-7 and its contacts by 1.1e-4. That is
  // conditioning, not error - recomputing the contacts on the CPU from the
  // GPU's OWN logits agrees to 3.8e-8, so the head's arithmetic is exact and
  // every bit of the difference arrives from upstream.
  //
  // So the head is bounded on what it can be held to (its own arithmetic), and
  // the contact figure is reported rather than asserted, with only a sanity
  // ceiling. A tolerance tight enough to "catch" 1.1e-4 here would fail on any
  // correct implementation that rounds differently.
  if (headOnly > 1e-5) {
    throw new Error(`the distogram head's own arithmetic is off by `
      + `${headOnly.toExponential(2)}, recomputed from its own logits`);
  }
  if (contactRms > 1e-2) {
    throw new Error(`contact relRMS ${contactRms.toExponential(2)} is past the sanity ceiling`);
  }
  return { tokens, sequences, blocks, pairRms, singleRms, contactRms, logitsRms,
           envelope, contactEnvelope, headOnly,
           timings: gpu.timings };
}
