/**
 * Synthyra's ESMFold2 confidence head in WebGL2 against
 * `cpu/esmfold2/confidence.js`.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-esmfold2-confidence.js [--falsify]
 *
 * 🔴 THE STACK'S ANSWER IS ADDED TO ITS OWN INPUT, which is their line and not a
 * slip: `pair = pair + self.folding_trunk(pair, ...)` where the trunk already
 * returns the UPDATED pair. "A port that read that line as an ordinary residual
 * is short exactly one copy of the input, which is a PAE that looks entirely
 * reasonable and is wrong" - so the gate runs the stack BOTH ways and asserts
 * they differ, because an arm nobody can distinguish is not an arm.
 *
 * 🔴 AND THE pLDDT TABLE IS INDEXED BY THE ATOM'S SLOT WITHIN ITS TOKEN, a
 * running count that resets at each token boundary. The gate's layout gives the
 * tokens DIFFERENT atom counts, so the slot and the global index part company -
 * with equal counts "indexing it by the atom's global position instead" would
 * read the same rows for the first token and the arm would be weak.
 *
 * 🔴 AND THE POOLING MASK IS A BIAS OF -1e9 AND NOT A ZEROED WEIGHT, so a
 * masked token still appears in the softmax at exp(-1e9) = 0 - and the gate
 * masks tokens so the two readings differ.
 *
 * 🔴 AND IT RUNS THE WGSL HEAD BESIDE THIS ONE, ON ONE SET OF WEIGHTS AND ONE
 * SET OF INPUTS, BECAUSE A HEAD AND A REFERENCE THAT SHARE A MISTAKE AGREE
 * PERFECTLY. This backend is gated against `cpu/esmfold2/confidence.js` and the
 * WGSL head against its own reference, so twelve candidate causes of ESMFold2's
 * ~30-point pLDDT divergence between the two backends were eliminated against
 * that pair of references without either moving - and "the two references
 * disagree" was the one possibility no arm here could see. Measured at
 * `--tokens=8 --pair-channels=128` (the widths the WGSL trunk compiles):
 * WebGL2 against the f64 reference **5.17e-08** (pLDDT) and 1.43e-07 (PAE),
 * WGSL against the same reference **1.02e-05** and 4.54e-05, the two heads
 * against each other **1.01e-05** and 4.54e-05. So the heads agree to the WGSL
 * path's own f16/bf16 rounding and NOTHING ELSE: both references are right,
 * every stage is right on synthetic data, and the fold's divergence is three to
 * four orders of magnitude larger than anything this gate can see - it is
 * something the drivers do differently at a fold's real magnitudes.
 *
 * 🔴 AND pTM IS MAXIMISED OVER i, NOT AVERAGED, with `d0` from the LIVE token
 * count clamped at 19. Both are host arithmetic over the PAE logits this port
 * produces, so the gate compares the two SCORES as numbers.
 */
import {
  esmfold2Confidence as cpuConfidence, tmScores,
} from "../../cpu/esmfold2/confidence.js";
import { esmfold2ConfidenceFold } from "../../webgpu/esmfold2/confidence.js";
import {
  esmfold2Confidence, intraTokenSlots,
} from "../../webgl/esmfold2/confidence.js";
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
  const tokens = Number(option(args, "tokens", "6"));
  const pairChannels = Number(option(args, "pair-channels", "16"));
  const singleChannels = Number(option(args, "single-channels", "24"));
  const singleInputs = Number(option(args, "single-inputs", "28"));
  const blocks = Number(option(args, "blocks", "2"));
  const plddtBins = Number(option(args, "plddt-bins", "20"));
  const paeBins = Number(option(args, "pae-bins", "16"));
  const maxAtomsPerToken = Number(option(args, "max-atoms", "5"));
  const distanceBins = Number(option(args, "distance-bins", "16"));
  const pairs = tokens * tokens;

  const random = generator(1732050);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };

  // 🔴 TOKENS WITH DIFFERENT ATOM COUNTS, so the pLDDT table's slot and the
  // atom's global index part company - and a masked token, so the pooling's
  // -1e9 bias has something to do.
  const counts = Array.from({ length: tokens }, (_, token) => 2 + (token % 4));
  const atomToToken = [];
  for (const [token, count] of counts.entries()) {
    for (let i = 0; i < count; i += 1) atomToToken.push(token);
  }
  const atoms = atomToToken.length;
  const atomMask = new Float32Array(atoms).fill(1);
  for (let atom = 0; atom < atoms; atom += 1) if (atom % 6 === 4) atomMask[atom] = 0;
  const tokenMask = new Float32Array(tokens).fill(1);
  if (tokens > 2) tokenMask[tokens - 1] = 0;
  const asymId = Int32Array.from({ length: tokens },
    (_, token) => (token < Math.ceil(tokens / 2) ? 0 : 1));
  const repAtom = Int32Array.from({ length: tokens }, (_, token) => {
    let at = 0;
    for (let t = 0; t < token; t += 1) at += counts[t];
    return at;
  });
  const coordinates = noise(atoms * 3, 15);
  const sInputs = noise(tokens * singleInputs, 1);
  const pair = noise(pairs * pairChannels, 1);
  const pairBias = noise(pairs * pairChannels, 0.5);
  const boundaries = Float32Array.from(
    { length: distanceBins - 1 }, (_, i) => 2 + (i * 50) / (distanceBins - 1));

  const stackBlocks = Array.from({ length: blocks }, () => {
    const group = pairHalfGroups(noise, pairChannels, 4);
    return {
      triangleMultiplicationOutgoing: group.triangleMultiplicationOutgoing,
      triangleMultiplicationIncoming: group.triangleMultiplicationIncoming,
      pairTransition: group.pairTransition,
    };
  });
  const hostWeights = {
    pairChannels, singleChannels, singleInputs, plddtBins, paeBins, maxAtomsPerToken,
    sInputsNormScale: noise(singleInputs, 0.2, 1),
    sInputsNormOffset: noise(singleInputs, 0.1),
    zNormScale: noise(pairChannels, 0.2, 1),
    zNormOffset: noise(pairChannels, 0.1),
    sToZ: noise(singleInputs * pairChannels, 1 / Math.sqrt(singleInputs)),
    sToZTranspose: noise(singleInputs * pairChannels, 1 / Math.sqrt(singleInputs)),
    sToZProdIn1: noise(singleInputs * pairChannels, 1 / Math.sqrt(singleInputs)),
    sToZProdIn2: noise(singleInputs * pairChannels, 1 / Math.sqrt(singleInputs)),
    sToZProdOut: noise(pairChannels * pairChannels, 1 / Math.sqrt(pairChannels)),
    boundaries,
    distanceEmbedding: noise(distanceBins * pairChannels, 0.3),
    poolingAttention: noise(pairChannels, 1 / Math.sqrt(pairChannels)),
    poolingOutput: noise(pairChannels * singleChannels, 1 / Math.sqrt(pairChannels)),
    plddtNormScale: noise(singleChannels, 0.2, 1),
    plddtNormOffset: noise(singleChannels, 0.1),
    plddtWeight: noise(
      maxAtomsPerToken * singleChannels * plddtBins, 1 / Math.sqrt(singleChannels)),
    pae: noise(pairChannels * paeBins, 1 / Math.sqrt(pairChannels)),
    blocks: stackBlocks,
  };

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  const skew = (values, period) => Float32Array.from(values,
    (v, i) => (i % period < period / 2 ? v * 1.01 + 0.01 : v));
  // 🔴 HOISTED BECAUSE BOTH HEADS MUST READ THE SAME NUMBERS. Skewing inside
  // the WebGL2 upload left the WGSL head on the originals, so the two arms
  // against the f64 reference compared an unskewed head with an unskewed
  // reference and read 1.015e-05 under `--falsify` exactly as without it - an
  // arm the flag could not reach, which is this tree's own VACUOUS failure.
  const skewedInputs = falsify ? skew(sInputs, singleInputs) : sInputs;
  const skewedPair = falsify ? skew(pair, pairChannels) : pair;
  const skewedCoordinates = falsify ? skew(coordinates, 3) : coordinates;
  const uploadedWeights = {
    pairChannels, singleChannels, singleInputs, plddtBins, paeBins, maxAtomsPerToken,
    sInputsNormScale: upload(hostWeights.sInputsNormScale, "w.sin-scale"),
    sInputsNormOffset: upload(hostWeights.sInputsNormOffset, "w.sin-offset"),
    zNormScale: upload(hostWeights.zNormScale, "w.z-scale"),
    zNormOffset: upload(hostWeights.zNormOffset, "w.z-offset"),
    sToZ: upload(hostWeights.sToZ, "w.s-to-z"),
    sToZTranspose: upload(hostWeights.sToZTranspose, "w.s-to-z-t"),
    sToZProdIn1: upload(hostWeights.sToZProdIn1, "w.prod-1"),
    sToZProdIn2: upload(hostWeights.sToZProdIn2, "w.prod-2"),
    sToZProdOut: upload(hostWeights.sToZProdOut, "w.prod-out"),
    boundaries: upload(boundaries, "w.boundaries"),
    distanceEmbedding: upload(hostWeights.distanceEmbedding, "w.distance"),
    poolingAttention: upload(hostWeights.poolingAttention, "w.pool-attn"),
    poolingOutput: upload(hostWeights.poolingOutput, "w.pool-out"),
    plddtNormScale: upload(hostWeights.plddtNormScale, "w.plddt-scale"),
    plddtNormOffset: upload(hostWeights.plddtNormOffset, "w.plddt-offset"),
    plddtWeight: upload(hostWeights.plddtWeight, "w.plddt"),
    pae: upload(hostWeights.pae, "w.pae"),
  };
  const uploadedBlocks = stackBlocks.map((block) => ({
    triangleMultiplicationOutgoing: uploadTriangleWeights(
      execution, block.triangleMultiplicationOutgoing),
    triangleMultiplicationIncoming: uploadTriangleWeights(
      execution, block.triangleMultiplicationIncoming),
    pairTransition: uploadTransitionWeights(execution, block.pairTransition),
  }));

  const inputs = {
    tokens, atoms,
    sInputs: upload(skewedInputs, "ef2.s-inputs"),
    pair: upload(skewedPair, "ef2.pair"),
    pairBias: upload(pairBias, "ef2.pair-bias"),
    coordinates: upload(skewedCoordinates, "ef2.coordinates"),
    repAtom: upload(Float32Array.from(repAtom), "ef2.rep-atom"),
    atomToToken: upload(Float32Array.from(atomToToken), "ef2.atom-to-token"),
    tokenMask: upload(tokenMask, "ef2.token-mask"),
    atomMask: upload(atomMask, "ef2.atom-mask"),
    slots: upload(intraTokenSlots(atomToToken, maxAtomsPerToken), "ef2.slots"),
  };

  const arms = [];
  const failures = [];
  const separations = {};
  const out = esmfold2Confidence(execution, {
    ...inputs, weights: uploadedWeights, blocks: uploadedBlocks,
  });
  const mine = {};
  for (const key of ["pair", "single", "plddtLogits", "plddtPerAtom", "plddt",
    "paeLogits", "pae"]) {
    mine[key] = Float32Array.from(execution.read(out[key]));
  }
  for (const tensor of Object.values(out)) execution.release(tensor);

  const theirs = cpuConfidence({
    tokens, atoms, sInputs, pair, pairBias, coordinates,
    repAtom: Int32Array.from(repAtom), atomToToken: Int32Array.from(atomToToken),
    tokenMask, atomMask, asymId,
  }, hostWeights);

  let skipped = null;
  let crossBackend = null;
  const report = (name, ours, them, armBar = bar) => {
    const relative = relativeRms(ours, them, name);
    const ok = relative <= armBar;
    arms.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar: armBar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${armBar}`);
  };
  report("the finished pair", mine.pair, theirs.pair);
  report("the pooled single", mine.single, theirs.single);
  report("the pLDDT logits", mine.plddtLogits, theirs.plddtLogits);
  report("pLDDT per atom", mine.plddtPerAtom, theirs.plddtPerAtom);
  report("pLDDT per token", mine.plddt, theirs.plddt);
  report("the PAE logits", mine.paeLogits, theirs.paeLogits);
  report("PAE", mine.pae, theirs.pae);

  // --- THE WGSL HEAD, on the same weights and the same inputs -------------
  //
  // 🔴 THE ONE COMPARISON NEITHER BACKEND'S GATES MAKE, AND THE ONLY
  // NON-CIRCULAR ONE LEFT. A whole ESMFold2 fold reads mean pLDDT 57.50 on this
  // backend against the WGSL fold's 88.08, with the two STRUCTURES agreeing at
  // TM 0.862 - so the head's score is what differs. But this gate holds the
  // WebGL2 head to `cpu/esmfold2/confidence.js` and the WGSL head is held to its
  // own reference, and a head and a reference that share a mistake agree
  // perfectly: twelve candidate inputs have been eliminated one by one against
  // that pair of references and none of them moved.
  //
  // Running BOTH heads on one set of weights and one set of inputs settles it in
  // one number. If they agree, every stage is right on synthetic data and the
  // divergence is something the drivers do differently at a fold's real
  // magnitudes; if they disagree, the two references disagree and one of them is
  // wrong - which is the possibility no arm here could see.
  //
  // `--no-wgsl` skips it, since it needs a WebGPU device beside the WebGL2 one.
  //
  // 🔴 AND IT RUNS ONLY AT WIDTHS THE WGSL TRUNK WILL COMPILE, WHICH THIS
  // GATE'S DEFAULTS ARE NOT. The WGSL pair transition picks a chunk of its
  // intermediate and falls back to the WHOLE intermediate when its preferred
  // one does not divide it, and the kernel then refuses a chunk that is not a
  // multiple of its 128-lane workgroup - so 16 pair channels (intermediate 64)
  // cannot compile there however right the head is. That refusal is SKIPPED BY
  // NAME and every other error is rethrown, because a swallowed exception is
  // how a WGSL regression would come to read as a skip. The command that runs
  // the arm:
  //
  //     node tools/gpu-chrome.mjs tools/gpu/check-webgl-esmfold2-confidence.js \
  //       --tokens=8 --pair-channels=128 --single-channels=128 --single-inputs=128
  //
  let wgsl = null;
  if (!flag(args, "no-wgsl")) {
    try {
      wgsl = await esmfold2ConfidenceFold(device, {
        tokens,
        atoms,
        sInputs: skewedInputs,
        pair: skewedPair,
        pairBias,
        coordinates: skewedCoordinates,
        repAtom: Int32Array.from(repAtom), atomToToken: Int32Array.from(atomToToken),
        tokenMask, atomMask, asymId,
      }, hostWeights);
    } catch (error) {
      const shape = /is not a multiple of the workgroup|does not divide intermediate/
        .test(String(error?.message ?? ""));
      if (!shape) throw error;
      skipped = `the WGSL head: ${error.message} - run it at --pair-channels=128`;
    }
  }
  if (wgsl) {
    // 🔴 THE WGSL ARMS GET THEIR OWN BAR, WHICH IS THAT PATH'S OWN PRECISION AND
    // NOT A TOLERANCE WIDENED TO PASS. The WGSL head stages in f16 and bf16
    // where this one and the reference are f32 throughout, so it lands at ~1e-5
    // against the SAME f64 reference this backend matches at 5e-8 - the figures
    // below are 1.0e-5 and 4.5e-5, and the WGSL-against-WebGL2 arms read the
    // same, which is the point: the two heads differ by the WGSL path's rounding
    // and by nothing else.
    const wgslBar = 1e-4;
    report("pLDDT per token, WGSL against the f64 reference", wgsl.plddt, theirs.plddt, wgslBar);
    report("PAE, WGSL against the f64 reference", wgsl.pae, theirs.pae, wgslBar);
    // 🔴 AND THE HEAD-TO-HEAD IS REPORTED, NOT GATED, BECAUSE NO INPUT SKEW CAN
    // FALSIFY IT: both heads read the same numbers, so a skew moves both and
    // this figure does not budge. It is not lost - each head is pinned to the
    // f64 reference above, which is the stronger statement, and the two arms
    // bound this difference between them.
    crossBackend = {
      plddt: Number(relativeRms(wgsl.plddt, mine.plddt, "WGSL pLDDT against WebGL2")
        .toPrecision(4)),
      pae: Number(relativeRms(wgsl.pae, mine.pae, "WGSL PAE against WebGL2").toPrecision(4)),
    };
  }

  // pTM and ipTM from both sides' logits, as numbers.
  {
    const ours = tmScores(mine.paeLogits, tokenMask, asymId, tokens, paeBins);
    const worst = Math.max(Math.abs(ours.ptm - theirs.ptm),
      Math.abs(ours.iptm - theirs.iptm));
    // 🔴 THE pTM BAR IS 1e-8 BECAUSE pTM ATTENUATES. It is a bounded statistic -
    // a max over anchors of a masked row mean of probabilities - so a 1%
    // perturbation of the logits reaches it as 7.9e-6, and at a 1e-5 bar the
    // control came back GREEN while every other arm went red ("only 7 of 8").
    // The measured agreement is 2.9e-10, so the bar is the floor with headroom
    // and the control is comfortably red.
    const ok = worst <= 1e-8;
    arms.push({
      arm: "pTM and ipTM off the PAE logits",
      ptm: Number(ours.ptm.toPrecision(6)), iptm: Number(ours.iptm.toPrecision(6)),
      relativeRms: Number(worst.toPrecision(3)), bar: 1e-8, ok,
    });
    if (!ok) failures.push(`pTM differs by ${worst.toExponential(3)}`);
  }

  // 🔴 THE DOUBLE ADD, BOTH WAYS. An ordinary residual is short exactly one copy
  // of the stack's input, and the gate asserts the two differ.
  {
    const single = theirs.pair;
    const ordinary = Float32Array.from(theirs.pair);
    for (let index = 0; index < ordinary.length; index += 1) {
      ordinary[index] -= theirs.initial[index];
    }
    separations["the stack's answer added to its own input"] = Number(
      relativeRms(ordinary, single, "double").toPrecision(3));
    if (!(separations["the stack's answer added to its own input"] > 1e-3)) {
      failures.push("the double add is indistinguishable from an ordinary residual");
    }
  }

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
      tokens, atoms, pairChannels, singleChannels, singleInputs, blocks,
      plddtBins, paeBins, maxAtomsPerToken, distanceBins,
      atomsPerToken: counts,
      paddedAtoms: atoms - atomMask.reduce((a, b) => a + b, 0),
      paddedTokens: tokens - tokenMask.reduce((a, b) => a + b, 0),
      chains: 2,
    },
    bar,
    falsify,
    arms,
    crossBackend,
    skipped,
    separations,
    memory: execution.stats(),
    verdict: falsify
      ? (red === arms.length
        ? `ok: all ${arms.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${arms.length} arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms against the f64 reference${skipped ? ` (skipped ${skipped})` : ""}`
        : `FAILED: ${failures.join("; ")}`),
  };
}
