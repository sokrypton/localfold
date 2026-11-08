/**
 * pLDDT, PAE and pTM in WebGL2 against the WGSL confidence head.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-confidence.js [--falsify]
 *
 * 🔴 THE TWO BIN-CENTRE RULES ARE WHAT THIS GATE IS FOR. pLDDT's centres are
 * `(bin + 0.5) / bins * 100` and PAE's are `breaks[bin] + step / 2` with the
 * last bin CLOSED at `breaks[last] + step / 2` rather than open. Either one
 * wrong gives a confidence map that looks entirely reasonable and a pTM in the
 * right range, which is exactly the class of error a fold's own output cannot
 * reveal.
 *
 * pTM is compared too, through `tmScoresFromTerm` - shared code that both sides
 * use, so what is being compared is the per-pair TERM this port computes on the
 * device, not the reduction. The reduction stays on the host on both sides,
 * because every score over one prediction is the same expectations reduced
 * differently.
 */
import {
  ConfidenceHeadsGpu, computeTmScores, tmScoresFromTerm,
} from "../../webgpu/heads/confidence.js";
import { setDeviceTuning, setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import { tmPerBinFor, tmScoreD0, tmTermFromLogits } from "../../shared/heads/tm-score.js";
import {
  confidenceHeads, releaseConfidenceWeights, uploadConfidenceWeights,
} from "../../webgl/heads/confidence.js";
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
  const bar = Number(option(args, "bar", "1e-5"));
  const length = Number(option(args, "length", "9"));
  const structureChannels = Number(option(args, "structure-channels", "32"));
  const pairChannels = Number(option(args, "pair-channels", "64"));
  const hiddenChannels = Number(option(args, "hidden", "32"));
  const lddtBins = Number(option(args, "lddt-bins", "50"));

  const random = generator(31415);
  const noise = (count, spread, centre = 0) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  const breaks = Float32Array.from({ length: 63 }, (_, index) => index * 0.5);
  const paeBins = breaks.length + 1;
  // The centre rule, restated here on purpose: if this gate imported it from the
  // module under test, the arm could not see that rule being wrong.
  const paeCentresOf = (edges) => {
    const bins = edges.length + 1;
    const step = edges[1] - edges[0];
    const centres = new Float32Array(bins);
    for (let bin = 0; bin < bins - 1; bin += 1) centres[bin] = edges[bin] + step / 2;
    centres[bins - 1] = edges[edges.length - 1] + step / 2;
    return centres;
  };

  const structure = noise(length * structureChannels, 1.5);
  const pair = noise(length * length * pairChannels, 1.5);
  const lddtWeights = {
    normScale: noise(structureChannels, 0.2, 1),
    normOffset: noise(structureChannels, 0.1),
    act0Weight: noise(structureChannels * hiddenChannels, 1 / Math.sqrt(structureChannels)),
    act0Bias: noise(hiddenChannels, 0.1),
    act1Weight: noise(hiddenChannels * hiddenChannels, 1 / Math.sqrt(hiddenChannels)),
    act1Bias: noise(hiddenChannels, 0.1),
    logitsWeight: noise(hiddenChannels * lddtBins, 1 / Math.sqrt(hiddenChannels)),
    logitsBias: noise(lddtBins, 0.1),
  };
  const paeWeights = {
    logitsWeight: noise(pairChannels * paeBins, 1 / Math.sqrt(pairChannels)),
    logitsBias: noise(paeBins, 0.1),
  };

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const weights = uploadConfidenceWeights(execution, lddtWeights, paeWeights);
  // 🔴 THE PERTURBATION HAS TO REACH BOTH INPUTS AND HALF OF EVERY ROW, which
  // the first version of this control did neither of. pLDDT reads the STRUCTURE
  // and PAE and pTM read the PAIR, so skewing only the structure leaves four of
  // seven arms unable to move at all - they came back identical and the inverse
  // verdict said "only 0 of 7". And half of ONE row is too dilute to clear the
  // bar after a LayerNorm and a mean: half of EVERY row is still a local error
  // in shape (a whole row is invisible to the LayerNorm that reads it) and
  // clears it.
  const skew = (values, channels) => Float32Array.from(values,
    (v, i) => (i % channels < channels / 2 ? v * 1.01 + 0.01 : v));
  const structureTensor = execution.upload(
    falsify ? skew(structure, structureChannels) : structure, "structure");
  const pairTensor = execution.upload(falsify ? skew(pair, pairChannels) : pair, "pair");
  const ours = confidenceHeads(execution, {
    structure: structureTensor, pair: pairTensor, length,
    structureChannels, pairChannels, weights, hiddenChannels, lddtBins, breaks,
  });
  const ourScores = tmScoresFromTerm(ours.tmTerm, length);

  setDeviceTuning(device, { allowHostWeightPacking: true });
  setHalfPrecision(device, false);
  const theirs = await new ConfidenceHeadsGpu(device).run(
    structure, pair, length, lddtWeights, paeWeights, breaks);
  // ...and the host's own pTM from the same logits, as a third opinion on the
  // score rather than on the term.
  const hostScores = computeTmScores(theirs.paeLogits, length, breaks);

  const rows = [];
  const failures = [];
  const arm = (name, got, want) => {
    const relative = relativeRms(got, want, name);
    const ok = relative <= bar;
    rows.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };
  arm("plddt", ours.plddt, theirs.plddt);
  arm("predictedAlignedError", ours.predictedAlignedError, theirs.predictedAlignedError);
  // The per-pair TM term against the shared host implementation, computed from
  // the WGSL head's own logits - so this arm is this port's expectation against
  // somebody else's, not against itself.
  arm("tm term", ours.tmTerm, tmTermFromLogits(
    theirs.paeLogits, length,
    tmPerBinFor(paeCentresOf(breaks), tmScoreD0(length))));

  // 🔴 TWO ARMS CANNOT GO RED AND SAY SO, rather than being quietly excluded or
  // having the bar tightened until they do. `maxPredictedAlignedError` is
  // `centres[last]` - a constant of the bin convention, which is exactly what it
  // is here to check, and no input can move it. `meanPlddt` is an average over
  // residues of an arm that IS red, so it is less sensitive by construction. An
  // inverse test that demanded they move would be asking them to be something
  // they are not; one that silently skipped them would hide an arm that has
  // stopped working.
  const scalar = (name, got, want, falsifiable = true) => {
    const relative = Math.abs(got - want) / Math.max(Math.abs(want), 1e-30);
    const ok = relative <= bar;
    rows.push({ arm: name, got: Number(got.toPrecision(8)), want: Number(want.toPrecision(8)),
      relative: Number(relative.toPrecision(3)), bar, ok,
      ...(falsifiable ? {} : { falsifiable: false }) });
    if (!ok) failures.push(`${name}: ${got} against ${want}`);
  };
  scalar("meanPlddt", ours.meanPlddt, theirs.meanPlddt, false);
  scalar("ptm", ourScores.ptm, theirs.ptm);
  scalar("ptm against the host reduction", ourScores.ptm, hostScores.ptm);
  scalar("maxPredictedAlignedError",
    ours.maxPredictedAlignedError, theirs.maxPredictedAlignedError, false);

  // Readings these have to satisfy whatever the weights are.
  let outOfRange = 0;
  for (const value of ours.plddt) if (!(value >= 0 && value <= 100)) outOfRange += 1;
  if (outOfRange !== 0) failures.push(`${outOfRange} pLDDT values outside [0, 100]`);
  if (!(ourScores.ptm >= 0 && ourScores.ptm <= 1)) failures.push(`ptm ${ourScores.ptm} outside [0, 1]`);

  releaseConfidenceWeights(execution, weights);
  execution.release(structureTensor);
  execution.release(pairTensor);

  const falsifiable = rows.filter((row) => row.falsifiable !== false);
  const red = falsifiable.filter((row) => !row.ok).length;
  return {
    shape: { length, structureChannels, pairChannels, hiddenChannels, lddtBins, paeBins },
    bar,
    falsify,
    arms: rows,
    liveTensorsAfter: execution.storage.live.size,
    verdict: falsify
      ? (red === falsifiable.length
        ? `ok: all ${falsifiable.length} falsifiable arms go red on a 1% perturbation `
          + `(${rows.length - falsifiable.length} cannot by construction; see the header)`
        : `VACUOUS: only ${red} of ${falsifiable.length} falsifiable arms noticed a 1% `
          + "perturbation")
      : (failures.length === 0 && execution.storage.live.size === 0
        ? `ok: ${rows.length} arms, pLDDT in [0, 100], pTM in [0, 1], no tensor left live`
        : `FAILED: ${failures.join("; ")}`),
  };
}
