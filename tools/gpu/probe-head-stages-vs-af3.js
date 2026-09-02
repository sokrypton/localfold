/**
 * Where inside the diffusion head does a REAL protein start to disagree?
 *
 *     python3 tools/oracle/dump_af3_trunk.py --blocks 48 --recycles 0 \
 *       --diffusion 4 --float32 --sequence PIAQ...ASK \
 *       --capture '...|atom_transformer_encoder/__call__$|...' \
 *       --capture-args 'diffusion_head/__call__$' --out af3-rings-stages.json
 *     node tools/gpu-chrome.mjs tools/gpu/probe-head-stages-vs-af3.js
 *
 * 🔴 THE DENOISER'S ONLY ORACLE POINT WAS A TWELVE-RESIDUE PEPTIDE, and on it
 * the whole head scores 6.8e-6. On a 59-residue chain the same code scores 2e-2
 * to 6e-2 at EVERY noise level - so the divergence is the MOLECULE, not the
 * schedule, and the twelve-mer is too small to show it: 288 atom slots divide
 * into exactly nine query subsets with nothing left over, no residue pair is
 * more than 32 apart, and there is no ring in the sequence.
 *
 * This runs the CPU reference stage by stage on AF3's own inputs and compares
 * each stage against AF3's own capture of it, so the first number that is large
 * names the module.
 *
 * 🔴 THE CPU REFERENCE, NOT THE GPU. probe-head-cpu-vs-gpu.js already shows the
 * two agree to 5e-7 on this batch, so the GPU is not the question; putting the
 * reference under the oracle keeps the comparison one step long.
 */
import { batchFromDump } from "./fold.js";
import { perAtomConditioning } from "../../src/af3/atom-conditioning-reference.js";
import { atomCrossAttentionEncoder } from "../../src/af3/atom-encoder-reference.js";
import { diffusionHead } from "../../src/af3/diffusion-reference.js";
import { openAf3Store } from "../../src/af3/weights.js";
import { diffusionWeights, atomReference } from "../../src/af3/diffusion-weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const floats = (source) => Float32Array.from(source, (v) => Number(v));

function lastCapture(dump, base) {
  if (dump.outputs[base] !== undefined) return dump.outputs[base];
  return Object.keys(dump.outputs)
    .filter((key) => key.startsWith(`${base}#`))
    .sort((a, b) => Number(a.split("#")[1]) - Number(b.split("#")[1]))
    .map((key) => dump.outputs[key]).pop();
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

const HEAD = "diffuser/~/diffusion_head";

/** Our stage name -> AF3's capture, for the stages that are the same tensor. */
// 🔴 pair_cond_initial_projection AND single_cond_initial_projection ARE NOT
// HERE, and their absence is a finding rather than an omission. Both capture at
// 2.9 and 3.6 relRMS on a batch whose head OUTPUT is exact to 6.8e-6, so
// whatever haiku records under those two names is not the tensor our
// conditioning holds at that point. A row that large next to an exact output
// would read as a second bug and send the next reader after a phantom.
const PAIRED = [
  ["encoder.skipConnection", `${HEAD}/diffusion_atom_transformer_encoder/__call__`],
  ["transformer", `${HEAD}/transformer/__call__`],
  ["after output norm", `${HEAD}/output_norm/__call__`],
];

export async function main(device, args) {
  const dumpPath = option(args, "dump", "/af3-rings-stages.json");
  const step = Number(option(args, "step", "1"));
  const response = await fetch(dumpPath);
  if (!response.ok) throw new Error(`failed to load ${dumpPath}: ${response.status}`);
  const dump = await response.json();
  const batch = batchFromDump(dump);
  const { tokens, dense } = batch;

  const store = await openAf3Store(option(args, "model", "/model-af3-full-f32/manifest.json"));
  const diffusion = await diffusionWeights(store);
  const reference = await atomReference(store);

  const at = (base) => dump.outputs[`${base}#${step}`] ?? dump.outputs[base];
  const noiseLevel = Number(at(`${HEAD}/__call__<1`).data[0]);
  const input = {
    shape: batch.shape,
    conditioning: perAtomConditioning({
      positions: batch.refPos, mask: batch.refMask,
      element: batch.refElement, charge: batch.refCharge,
      atomNameChars: batch.refAtomNameChars,
    }, tokens, dense, reference),
    atomMask: batch.predDenseAtomMask, seqMask: batch.seqMask, features: batch.features,
    targetFeat: floats(lastCapture(dump, "diffuser/evoformer/__call__:target_feat").data),
    refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
    tokenAtomsToQueries: batch.tokenAtomsToQueries,
    queriesToKeys: batch.queriesToKeys,
    queriesToTokenAtoms: batch.queriesToTokenAtoms,
    tokensToQueries: batch.tokensToQueries,
    tokensToKeys: batch.tokensToKeys,
    trunkSingle: floats(lastCapture(dump, "diffuser/evoformer/__call__:single").data),
    trunkPair: floats(lastCapture(dump, "diffuser/evoformer/__call__:pair").data),
    positionsNoisy: floats(at(`${HEAD}/__call__<0`).data),
    noiseLevel,
  };

  const seen = new Map();
  const ours = diffusionHead(input, diffusion, atomCrossAttentionEncoder,
                             (name, value) => seen.set(name, value));

  const rows = [{ stage: "step", step, sigma: Number(noiseLevel.toPrecision(4)) }];
  for (const [name, capture] of PAIRED) {
    const theirs = at(capture);
    if (theirs === undefined || !seen.has(name)) {
      rows.push({ stage: name, relRms: "not captured" });
      continue;
    }
    rows.push({
      stage: name, shape: theirs.shape,
      relRms: Number(relativeRms(seen.get(name), floats(theirs.data)).toPrecision(3)),
    });
  }
  rows.push({
    stage: "head output",
    relRms: Number(relativeRms(ours, floats(at(`${HEAD}/__call__`).data)).toPrecision(3)),
  });
  return rows;
}
