// RFD4-Proteina's trunk input for the native CUDA port, and the CPU reference's answer to it.
//
//   node cuda/rfd4p/export_input.mjs <out dir> [--preset=custom_cath_monomer] [--seed=0]
//        [--step=0] [--recycles=1] [--oracle=<dir>] [--checkpoint=<model.safetensors>]
//
// Writes, in cuda/af3's model.idx/model.bin format:
//   t in/seq       [n, 897]    trunkSeqFeatures   - the noised x_t and every per-token condition
//   t in/cond0     [n, 609]    trunkCondFeatures  - recycleRemaining null (the first cycle's c)
//   t in/cond1     [n, 609]    trunkCondFeatures  - recycleRemaining 1 (every later cycle's c)
//   t in/pair      [n*n, 107]  pairFeatures
//   t in/pairCond  [512]       pairCondFeatures   - one row, broadcast over every pair
//   t in/mask      [n]         the padding mask, 1 where a token is real
//   m meta/n, meta/nRecycle and each feature width
//
// 🔴 THE FEATURES ARE THE PAGE'S OWN, NOT A SECOND COPY. They come from `shared/rfd4p/features.js`
// through `shared/rfd4p/pipeline`'s `buildBatch` - the modules `webgpu/rfd4p/generate.js` calls -
// so the CUDA trunk is handed exactly what the WebGPU trunk is handed. Nothing about a molecule is
// decided here. A C++ twin under `cuda/featurise/` comes later, held to these bytes, as the other
// three ports' featurisers are.
//
// With `--oracle=<dir>` it also writes the CPU reference's answer, stage by stage, for the port's
// `--oracle=` arm:
//   t o/c, o/sInit, o/zInit        the three init stages
//   t o/s<L>, o/z<L>               the single and pair state after layer L
//   t o/bbCa, o/localLatents, o/logits, o/insertion, o/s, o/z   the finished trunk
//
// 🔴 AND THE STAGED LOOP IS CHECKED AGAINST `forwardTrunkCpu` ITSELF. The per-stage values are
// produced here by driving the reference's own exported pieces, which is a second copy of that
// function's twenty-line body and could drift from it. So the final s, z and heads of this loop are
// compared with what `forwardTrunkCpu` returns on the identical input, and the export FAILS if they
// disagree - the duplicate cannot go stale in silence.
import { readFileSync, writeFileSync, mkdirSync, openSync, writeSync, closeSync, renameSync } from "node:fs";
import { buildBatch, decodeTextOrZstd } from "../../shared/rfd4p/pipeline/index.js";
import { getRfd4pPresetById } from "../../shared/rfd4p/presets.js";
import {
  ALLOWED_RESIDUE_TYPES, ATOM37_NAMES, UNIFIED_ATOM37_TOKENS, UNIFIED_RESTYPE_ATOM37_MASK,
} from "../../shared/rfd4p/encoding.js";
import { COND } from "../../shared/rfd4p/features.js";
import { parseSafetensors } from "../../shared/rfd4p/safetensors.js";
import {
  pairCondFeatures, pairFeatures, trunkCondFeatures, trunkSeqFeatures,
} from "../../shared/rfd4p/features.js";
import {
  DEFAULT_SAMPLING, flowEmitMasks, recycleDepths, sampleInitialNoise, sampleSchedule, seededRng,
} from "../../shared/rfd4p/flow.js";
import {
  buildPartialSchedule, partialStepCount, resolvePartialT,
} from "../../shared/rfd4p/partial.js";
import {
  adaptiveLayerNorm, applyNorm, forwardTrunkCpu, layerNorm, linearLayer,
  multiheadAttnAndTransition, pairBiasAttention, pairReprUpdate, swiGluTransition,
} from "../../cpu/rfd4p/reference.js";
import {
  aeMasks, decodeLatentsCpu, decoderPostprocess, decoderSeqFeatures, sequenceForDecoder,
} from "../../cpu/rfd4p/decoder.js";
import { generateRfd4pCpu } from "../../cpu/rfd4p/reference.js";
import { encodeLatentsCpu, encoderSeqFeatures } from "../../cpu/rfd4p/encoder.js";

const args = process.argv.slice(2);
const out = args.find((a) => !a.startsWith("--"));
if (!out) {
  console.error("usage: export_input.mjs <out dir> [--preset=<id>] [--seed=N] [--recycles=N] [--oracle=<dir>]");
  process.exit(1);
}
const option = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const repo = new URL("../..", import.meta.url).pathname;
const hub = `${process.env.HOME}/.cache/huggingface/hub/models--nvidia--RFD4-Proteina/snapshots`;
const presetId = option("preset", "custom_cath_monomer");
const seed = Number(option("seed", "0"));
const step = Number(option("step", "0"));
const nRecycle = Number(option("recycles", "1"));
const oracleDir = option("oracle", "");
const generate = args.includes("--generate");
const stepsOverride = Number(option("steps", "0"));
const recycleOverride = option("recycle", "");
const length = Number(option("length", "0"));
const checkpoint = option("checkpoint", `${hub}/a06e2509c7033b010fc43614e9fd840303588502/model.safetensors`);
const cfg = JSON.parse(readFileSync(option("config", `${repo}model-rfd4p-int8/config.json`), "utf8")).config;
const nn = cfg.nn;

const preset = getRfd4pPresetById(presetId);
if (!preset) { console.error(`no preset ${presetId}`); process.exit(1); }
// 🔴 A PRESET'S STRUCTURE IS A FILE IN THIS CHECKOUT, NOT A FETCH. This used to refuse every
// conditioned preset with "only the pure-spec presets export offline", which was never true: the
// seven examples the page names are `web/examples/*.cif.zst`, shipped here, and the page fetches
// them over its own origin. Reading them off disk is the whole of what that refusal cost, and it
// is what makes the motif, PPI, small-molecule, DNA and partial-diffusion cases exportable at all.
//
// 🔴 AND A PRESET WITHOUT `specMode` IS BUILT FROM ITS STRUCTURE'S ANNOTATIONS, NOT ITS CONTIG.
// web/generate.js has the note: those contigs are written in the ORIGINAL entry's numbering and
// name residues the cropped example does not contain, so building one from the spec fails with
// "Preserved segment 'A52-71' matched no atoms". The structure IS the job.
const fromAnnotations = !length && preset.specMode !== true
  && preset.cifUrl !== null && preset.cifUrl !== undefined;
let inputCif;
if (!length && preset.cifUrl) {
  const bytes = readFileSync(`${repo}${preset.cifUrl.replace(/^\.\//, "")}`);
  inputCif = decodeTextOrZstd(new Uint8Array(bytes));
}
// `--length=N` is one unconditioned protein segment of N residues, for sizing and timing: the trunk's
// cost is a function of the token count and nothing in the conditioning changes a single shape.
const spec = length
  ? { contig: `a1:${length}-${length}P` }
  : { contig: preset.contig, ...(preset.conditions === undefined ? {} : { conditions: preset.conditions }) };
const { batch, n } = fromAnnotations
  ? buildBatch(inputCif, { seed })
  : buildBatch(spec, { seed, ...(inputCif === undefined ? {} : { inputCif }) });

const raw = readFileSync(checkpoint);
const parsedWeights = parseSafetensors(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
const weights = parsedWeights.tensors ?? parsedWeights;

// The sampler's first step, exactly as webgpu/rfd4p/generate.js reaches it.
const field = (key) => batch[key]?.data ?? batch[key];
const pad = field("padding_mask");
const rng = seededRng(seed);
const emit = flowEmitMasks({
  padding_mask: pad,
  is_in_context_conditioning: field("is_in_context_conditioning"),
  supports_local_latent: field("supports_local_latent"),
}, n);
const sched = sampleSchedule({ ...DEFAULT_SAMPLING, nsteps: preset.nsteps ?? DEFAULT_SAMPLING.nsteps });
const x = sampleInitialNoise({
  n, emit, coordCondMask: field("coord_cond_mask"), moleculeIid: field("molecule_iid"),
  rng, centeringScale: DEFAULT_SAMPLING.centering?.scale ?? 0.2,
});
const t = {
  bb_ca: Math.fround(sched.bb_ca.ts[step]),
  local_latents: Math.fround(sched.local_latents.ts[step]),
};
const cache = {};
const f = {
  seq: trunkSeqFeatures(batch, {
    n, xt: x, feats: nn.feats_seq, weights,
    randperm: (len) => rng.randperm(len, step),
    noiseAlignChirality: nn.noise_align_chirality, cache,
  }).data,
  cond0: trunkCondFeatures(batch, { n, t, recycleRemaining: null, feats: nn.feats_cond_seq, weights, cache }).data,
  cond1: trunkCondFeatures(batch, { n, t, recycleRemaining: 1, feats: nn.feats_cond_seq, weights, cache }).data,
  pair: pairFeatures(batch, {
    n, caNm: x.bb_ca, feats: nn.feats_pair_repr,
    dist: { dim: nn.xt_pair_dist_dim, min: nn.xt_pair_dist_min, max: nn.xt_pair_dist_max },
    seqSepDim: nn.seq_sep_dim, cache,
  }).data,
  pairCond: pairCondFeatures(t, nn.feats_pair_cond),
};

const dec = cfg.autoencoder_config.nn_ae.decoder;
const { attn } = aeMasks(batch, n);

// 🔴 THE DECODER'S INPUTS COME FROM THE TRUNK, so when an oracle is wanted the reference trunk runs
// BEFORE the input directory is written: the latent it decodes and the sequence it is conditioned on
// are that trunk's outputs, and both sides must be handed the same ones or the decoder is being
// compared on two different inputs.
let decoderZ = null;
let decoderSeqIds = null;
let trunkRef = null;
if (oracleDir) {
  trunkRef = forwardTrunkCpu(n, f, pad, weights, nn, nRecycle, { insertion: true });
  decoderZ = trunkRef.local_latents;
  decoderSeqIds = Int32Array.from(sequenceForDecoder(batch, n, trunkRef.logits, dec.n_residue_types));
}

/** One model.idx/model.bin directory from `{name: Float32Array}` plus `{name: number}` scalars. */
function writeDir(dir, tensors, meta, ints = {}) {
  mkdirSync(dir, { recursive: true });
  const fd = openSync(`${dir}/model.bin`, "w");
  const lines = Object.entries(meta).map(([key, value]) => `m meta/${key} ${value}`);
  let offset = 0;
  for (const [name, values] of Object.entries(tensors)) {
    const data = values instanceof Float32Array ? values : Float32Array.from(values);
    lines.push(`t ${name} ${offset} ${data.length}`);
    writeSync(fd, Buffer.from(data.buffer, data.byteOffset, data.byteLength));
    offset += data.length;
  }
  for (const [name, values] of Object.entries(ints)) {
    const data = values instanceof Int32Array ? values : Int32Array.from(values);
    lines.push(`i ${name} ${offset} ${data.length}`);
    writeSync(fd, Buffer.from(data.buffer, data.byteOffset, data.byteLength));
    offset += data.length;
  }
  closeSync(fd);
  writeFileSync(`${dir}/model.idx.tmp`, `${lines.join("\n")}\n`);
  renameSync(`${dir}/model.idx.tmp`, `${dir}/model.idx`);
  return offset;
}

// The decoder's own inputs. 🔴 `caNm` IS THE SAMPLE'S CA, NOT THE TRUNK'S OUTPUT - the generator
// decodes the CLEAN ESTIMATE x1, and this exports the step's x_t for the same reason the trunk's
// features do: a parity case wants a fixed, reproducible input, not a converged one.
const floats = writeDir(out, {
  "in/seq": f.seq,
  "in/cond0": f.cond0,
  "in/cond1": f.cond1,
  "in/pair": f.pair,
  "in/pairCond": f.pairCond,
  "in/mask": Float32Array.from(pad, (v) => (v ? 1 : 0)),
  "in/caNm": x.bb_ca,
  "in/attnMask": Float32Array.from(attn, (v) => (v ? 1 : 0)),
  ...(oracleDir ? { "in/zLatent": decoderZ } : {}),
}, {
  n,
  nRecycle,
  seqFeatDim: f.seq.length / n,
  condFeatDim: f.cond0.length / n,
  pairFeatDim: f.pair.length / (n * n),
  pairCondFeatDim: f.pairCond.length,
  decLayers: dec.nlayers,
  decTokenDim: dec.token_dim,
  decCondDim: dec.dim_cond,
  decHeads: dec.nheads,
  decFeatDim: 3 + (cfg.autoencoder_config.nn_ae.latent_z_dim) + 32 + dec.n_residue_types,
}, {
  "in/tokenType": Int32Array.from(field("token_type")),
  ...(oracleDir ? { "in/decSeq": decoderSeqIds } : {}),
});
console.log(`${out}: ${n} tokens, ${(floats * 4 / 1e6).toFixed(1)} MB`);

// ---- the autoencoder ENCODER's inputs, and its own oracle ------------------------------------
//
// 🔴 THE ENCODER IS THE ONLY STAGE WHOSE INPUT IS A REAL STRUCTURE, which is why it could not be
// exported at all until the preset loader learned to read `web/examples/*.cif.zst` off disk. It
// turns clean all-atom coordinates into the local latents a partial-diffusion or inverse-folding
// run starts from, so without it those two presets have no starting point.
//
// 🔴 AND THE ORACLE IS THE DETERMINISTIC ARM. `encodeLatentsCpu` draws `n * 32` normals for its
// reparameterisation out of the SAMPLER's stream; comparing the sampled z would compare two
// Box-Muller implementations rather than the encoder, so the gate takes `deterministic: true`
// (z is the mean) and the sampled arm is covered by the generate loop's replayed noise.
if (args.includes("--encoder")) {
  const encCfg = cfg.autoencoder_config.nn_ae.encoder;
  const encDims = {
    token_type: encCfg.token_type_emb_dim ?? 32,
    x1_a37coors_nm: 37 * 4,
    x1_a37coors_nm_rel: 37 * 4,
    chain_idx_seq: 1,
    incident_bond_type_counts: 37 * 10,
    atom37_element_charge_embedding: 37 * (8 + (encCfg.charge_emb_dim ?? 16)),
  };
  // The offsets the kernel needs, derived from `feats_seq` rather than typed in: a reordered
  // feature list must move the kernel's writes with it, and -1 means the block is absent.
  const encOff = {};
  let o2 = 0;
  encCfg.feats_seq.forEach((fname, idx) => {
    encOff[fname] = o2;
    encOff[`${fname}@`] = idx;
    o2 += encDims[fname];
  });
  const encFdim = o2;
  const atomMask = batch.atom37_mask ? field("atom37_mask") : field("coord_mask");
  const coorsNm = batch.coords_nm
    ? field("coords_nm")
    : Float32Array.from(field("coords"), (v) => Math.fround(v / 10));
  const encFloats = writeDir(`${out}-enc`, {
    "e/coorsNm": Float32Array.from(coorsNm),
    "e/atomMask": Float32Array.from(atomMask, (v) => (v ? 1 : 0)),
    "e/chainIid": Float32Array.from(field("chain_iid")),
    "e/bondCounts": Float32Array.from(field("incident_bond_type_counts")),
    "e/emit": Float32Array.from(aeMasks(batch, n).emit, (v) => (v ? 1 : 0)),
  }, {
    encLayers: encCfg.nlayers,
    encTokenDim: encCfg.token_dim,
    encCondDim: encCfg.dim_cond,
    encHeads: encCfg.nheads,
    encFeatDim: encFdim,
    encTtDim: encCfg.token_type_emb_dim ?? 32,
    encChargeDim: encCfg.charge_emb_dim ?? 16,
    encOffAbs: encOff.x1_a37coors_nm ?? -1,
    encOffRel: encOff.x1_a37coors_nm_rel ?? -1,
    encOffChain: encOff.chain_idx_seq ?? -1,
    encOffBonds: encOff.incident_bond_type_counts ?? -1,
    encOffElCh: encOff.atom37_element_charge_embedding ?? -1,
    encTtCreator: encOff["token_type@"] ?? -1,
    encElChCreator: encOff["atom37_element_charge_embedding@"] ?? -1,
  }, {
    "e/element": Int32Array.from(field("ground_truth_element")),
    "e/charge": Int32Array.from(field("charge")),
  });
  const enc = encodeLatentsCpu(batch, {
    n, weights, cfg: encCfg, deterministic: true,
  });
  writeDir(`${out}-enc-oracle`, {
    "e/feats": Float32Array.from(encoderSeqFeatures(batch, { n, weights, cfg: encCfg }).data),
    "e/mean": enc.mean,
    "e/logScale": enc.logScale,
    "e/zLatent": enc.zLatent,
  }, {});
  console.log(`${out}-enc: ${encFdim} encoder columns, ${(encFloats * 4 / 1e6).toFixed(1)} MB`);
}

// ---- everything a whole generation needs, written once ---------------------------------------
//
// 🔴 ONLY 35 OF THE 897 SEQUENCE COLUMNS CHANGE FROM STEP TO STEP - the three of `xt_bb_ca` and the
// thirty-two of `xt_local_latents`, at offsets 0 and 3 - so the features are exported once and the
// port overwrites those columns on the device. That is the same condition `webgpu/rfd4p/generate.js`
// tests before it takes its own on-device step loop (`caOff === 0 && latOff === 3`), and it holds
// only while there are no coordinate groups to permute and no chirality frames: both are empty for
// an unconditional monomer and this export REFUSES anything else rather than silently freezing a
// feature that should move.
//
// The pair features go out without their distance bins (the first 30 columns), which the port fills
// from the current CA each step.
if (generate) {
  // 🔴 WHAT STILL REFUSES IS A COORDINATE GROUP, and only that. The chirality alignment also moves
  // between steps and used to be refused with it; the port recomputes it on the device now
  // (src/chirality.cuh), which is what lets the small-molecule preset generate at all. A coordinate
  // group's permutation is a `randperm` draw a step and is not done.
  if ((cache.crdUniq?.length ?? 0) > 1) {
    console.error("--generate does not yet cover an input with coordinate groups to permute");
    process.exit(1);
  }
  const alignsChirality = nn.noise_align_chirality && (cache.activeFrames?.length ?? 0) > 0;
  // 🔴 A PARTIAL RUN'S STEP COUNT, SCHEDULE AND RECYCLE DEPTHS ARE ALL DIFFERENT, and they are
  // derived here exactly as `generateRfd4pCpu` derives them rather than guessed from the preset.
  // `partial_trajectory_fraction_steps` runs `round(frac * nsteps)` steps off the TAIL of the full
  // schedule - so a half-trajectory of 60 is 30 steps whose spacing is the last 30 of 60, not the
  // 30-step schedule, which has the right endpoints and the wrong spacing everywhere between.
  // Inverse folding runs the whole schedule (both start times are 0) against a backbone resampled
  // from its forward marginal at every step.
  const baseNsteps = stepsOverride || preset.nsteps || DEFAULT_SAMPLING.nsteps;
  const inverseFolding = preset.inverseFolding === true;
  const frac = inverseFolding ? null : (preset.partialTrajectoryFractionSteps ?? null);
  const partialOpts = {
    nsteps: baseNsteps,
    ...(inverseFolding ? { inverseFolding: true } : {}),
    ...(frac != null ? { partialTrajectoryFractionSteps: frac } : {}),
  };
  const partialTDict = resolvePartialT(partialOpts, DEFAULT_SAMPLING);
  const isPartial = Object.keys(partialTDict).length > 0;
  const steps = isPartial && frac != null ? partialStepCount(frac, baseNsteps) : baseNsteps;
  // 🔴 `fullNsteps` IS baseNsteps ALWAYS, not only when partial - that is what `generateRfd4pCpu`
  // passes, and with it `recycleDepths` takes the schedule's TAIL rather than resampling it.
  const depths = recycleDepths(recycleOverride || preset.recycleSchedule || "none", steps,
    baseNsteps);
  const genSched = isPartial
    ? buildPartialSchedule(partialTDict, steps, DEFAULT_SAMPLING,
      { fullNsteps: frac != null ? baseNsteps : null })
    : sampleSchedule({ ...DEFAULT_SAMPLING, nsteps: steps });
  const pairNoDist = pairFeatures(batch, {
    n, caNm: x.bb_ca, feats: nn.feats_pair_repr,
    dist: { dim: nn.xt_pair_dist_dim, min: nn.xt_pair_dist_min, max: nn.xt_pair_dist_max },
    seqSepDim: nn.seq_sep_dim, cache: {}, skipPairDists: true,
  }).data;

  // The conditioning for every step: it depends on t alone, so sixty rows of it are cheaper to
  // write than a time embedding ported twice.
  const condDim = f.cond0.length / n;
  const condAll = new Float32Array(steps * n * condDim);
  const cond1All = new Float32Array(steps * n * condDim);
  const pairCondAll = new Float32Array(steps * f.pairCond.length);
  for (let s2 = 0; s2 < steps; s2 += 1) {
    const tt2 = {
      bb_ca: Math.fround(genSched.bb_ca.ts[s2]),
      local_latents: Math.fround(genSched.local_latents.ts[s2]),
    };
    condAll.set(trunkCondFeatures(batch, {
      n, t: tt2, recycleRemaining: null, feats: nn.feats_cond_seq, weights, cache: {} }).data,
      s2 * n * condDim);
    cond1All.set(trunkCondFeatures(batch, {
      n, t: tt2, recycleRemaining: 1, feats: nn.feats_cond_seq, weights, cache: {} }).data,
      s2 * n * condDim);
    pairCondAll.set(pairCondFeatures(tt2, nn.feats_pair_cond), s2 * f.pairCond.length);
  }

  // 🔴 THE NOISE IS REPLAYED, NOT REDRAWN. Box-Muller runs through V8's Math.log, which this port
  // has no bit-exact copy of (cuda/featurise/jsmath.h carries sin and cos and not log), and one ulp
  // in one draw diverges a sixty-step trajectory. The draw ORDER is a function of the schedule
  // alone - initial noise, then a draw a step only where the step is stochastic - so the stream is
  // reproducible here without running the model at all.
  const rng2 = seededRng(seed);
  const chunks = [];
  const take = (count) => { const a = rng2.randn(count); chunks.push(a); return a; };
  const offsetMask = (i) => emit.bb_ca[i] && !(field("coord_cond_mask")[i]);
  const groupIds = [...new Set(Array.from({ length: n }, (_, i) => i).filter(offsetMask)
    .map((i) => field("molecule_iid")[i]))].sort((a, b) => a - b);
  const latentDim = cfg.autoencoder_config.nn_ae.latent_z_dim ?? 32;
  // 🔴 THE DRAW ORDER IS THE SPECIFICATION, AND A PARTIAL RUN'S IS NOT THE ORDINARY ONE. Upstream's
  // `corrupt_batch` draws bb_ca's noise, then the latents', then bb_ca's per-molecule offset -
  // where `sample_noise` draws bb_ca, the offset, then the latents. One transposition shifts every
  // subsequent draw of the whole run, which is invisible in any single stage and wrong everywhere.
  if (isPartial) {
    take(n * latentDim);                       // encodeLatentsCpu's reparameterisation
    take(n * 3);                               // corrupt_batch: zero-COM backbone noise
    take(n * latentDim);                       // corrupt_batch: latent noise
    for (let g = 0; g < groupIds.length; g += 1) take(3);
    if (inverseFolding) {
      take(n * 3);                             // the target backbone at ts[0]
      for (let g = 0; g < groupIds.length; g += 1) take(3);
    }
  } else {
    take(n * 3);
    for (let g = 0; g < groupIds.length; g += 1) take(3);
    take(n * latentDim);
  }
  const stepNoise = [];
  for (let s2 = 0; s2 < steps; s2 += 1) {
    for (const dm of ["bb_ca", "local_latents"]) {
      const t2 = Math.fround(genSched[dm].ts[s2]);
      const cfg2 = genSched[dm].step;
      const stochastic = t2 !== 0 && !(t2 > cfg2.t_lim_ode || t2 < cfg2.t_lim_ode_below);
      stepNoise.push(stochastic ? 1 : 0);
      if (stochastic) take(n * (dm === "bb_ca" ? 3 : latentDim));
    }
    // The resampled target comes AFTER both modalities have stepped, and not after the last step,
    // where the clean target is used as it is.
    if (inverseFolding && s2 < steps - 1) {
      take(n * 3);
      for (let g = 0; g < groupIds.length; g += 1) take(3);
    }
  }
  let total = 0;
  for (const c of chunks) total += c.length;
  const noise = new Float32Array(total);
  let at = 0;
  for (const c of chunks) { noise.set(c, at); at += c.length; }

  const gFloats = writeDir(`${out}-gen`, {
    "g/pairNoDist": pairNoDist,
    "g/cond": condAll,
    "g/cond1": cond1All,
    "g/pairCond": pairCondAll,
    "g/noise": noise,
    "g/emitBb": Float32Array.from(emit.bb_ca, (v) => (v ? 1 : 0)),
    "g/emitLl": Float32Array.from(emit.local_latents, (v) => (v ? 1 : 0)),
    // The chirality's CLEAN packed vectors and nothing rotated: the port fits a frame against the
    // noisy CA each step and rotates these, so what is exported is the input, not a step's output.
    ...(alignsChirality ? { "g/chir": Float32Array.from(field("chirality")) } : {}),
  }, {
    steps,
    nGroups: groupIds.length,
    centeringScale: DEFAULT_SAMPLING.centering.scale,
    alignChirality: alignsChirality ? 1 : 0,
    chirOff: alignsChirality ? cache.chirOff : -1,
    chiralFrameWidth: alignsChirality ? batch.chiral_frame_atoms.shape.at(-1) : 0,
    partial: isPartial ? 1 : 0,
    inverseFolding: inverseFolding ? 1 : 0,
    partialTBb: partialTDict.bb_ca ?? 0,
    partialTLl: partialTDict.local_latents ?? 0,
    baseNsteps,
    fullNsteps: frac != null ? baseNsteps : 0,
    distDim: nn.xt_pair_dist_dim,
    distMin: nn.xt_pair_dist_min,
    distMax: nn.xt_pair_dist_max,
  }, {
    "g/depths": Int32Array.from(depths),
    "g/stochastic": Int32Array.from(stepNoise),
    "g/coordCond": Int32Array.from(field("coord_cond_mask"), (v) => (v ? 1 : 0)),
    "g/moleculeIid": Int32Array.from(field("molecule_iid")),
    "g/allowedTypes": Int32Array.from(ALLOWED_RESIDUE_TYPES.flatMap((r) => Array.from(r, (v) => (v ? 1 : 0)))),
    "g/atom37Legal": Int32Array.from(UNIFIED_RESTYPE_ATOM37_MASK.flatMap((r) => Array.from(r, (v) => (v ? 1 : 0)))),
    "g/coordGroup": Int32Array.from(batch[COND.CRDGRP] ? field(COND.CRDGRP) : new Int32Array(n * 37).fill(-1)),
    "g/condSeq": Int32Array.from(field(COND.SEQ)),
    ...(alignsChirality ? {
      "g/chiralFrames": Int32Array.from(field("chiral_frame_atoms")),
      "g/chiralOwners": Int32Array.from(field("chiral_slot_owner")),
    } : {}),
  });
  // 🔴 AND THE WHOLE GENERATION, WHEN IT IS SHORT ENOUGH TO AFFORD. `generateRfd4pCpu` is sixty
  // trunk passes in JavaScript at about a minute each, so the end-to-end comparison is run at a
  // step count somebody can wait for - which is enough, because every mechanism the loop has is
  // exercised by two steps as much as by sixty.
  // 🔴 OPT-IN, BECAUSE SIXTY STEPS OF JS TRUNK IS HOURS. `generateRfd4pCpu` is `depths` trunk
  // passes an actual minute each; at the preset's 60 steps that is a whole afternoon, and running
  // it by accident alongside `--oracle` is how this export went away for two hours. Ask for it with
  // `--gen-oracle` and a step count somebody can wait for.
  if (args.includes("--gen-oracle")) {
    // 🔴 EVERY STEP, NOT ONLY THE LAST. A trajectory comparison that sees the end alone cannot say
    // WHICH step went wrong, and the backbone's first step divides by t = 0.001 - so a difference
    // the trunk makes in the fifth digit arrives at the end a thousand times larger. The per-step
    // record is what separates "the loop is wired wrong" from "the loop amplifies".
    const perStep = {};
    const gen = await generateRfd4pCpu(batch, weights, cfg, {
      nsteps: baseNsteps,
      seed,
      ...(inverseFolding ? { inverseFolding: true } : {}),
      ...(frac != null ? { partialTrajectoryFractionSteps: frac } : {}),
      // 🔴 THE SAME RECYCLE SCHEDULE THE PORT IS HANDED, SAID OUT LOUD. `generateRfd4pCpu`
      // defaults to `medium` and the exporter writes `g/depths` from the PRESET's, so leaving it
      // off ran the reference at six recycles against the port's four - the two sides doing
      // different amounts of work on identical inputs. It read as a wrong trajectory: the noise
      // and the step-0 features were exact to the last bit and the velocity was 8e-2 out.
      recycleSchedule: recycleOverride || preset.recycleSchedule || "none",
      onStep: ({ step: s2, x1, xt }) => {
        perStep[`gen/s${s2}Bb`] = Float32Array.from(xt.bb_ca);
        perStep[`gen/s${s2}Ll`] = Float32Array.from(xt.local_latents);
        perStep[`gen/s${s2}X1Bb`] = Float32Array.from(x1.bb_ca);
        perStep[`gen/s${s2}X1Ll`] = Float32Array.from(x1.local_latents);
      },
    });
    // 🔴 AND THE LOOP'S INPUT, NOT ONLY ITS OUTPUT. `generateRfd4pCpu` draws its own initial noise
    // and builds its own first-step features; the port replays a recorded stream and overwrites two
    // column blocks. Writing both here is what separates "the port's noise is wrong", "the port's
    // features are wrong" and "the port's trunk is wrong" - three answers a final-state comparison
    // gives one number for. A fresh rng reproduces the draw exactly: nothing consumes the stream
    // before `sample_noise`.
    const rng0 = seededRng(seed);
    const x0 = sampleInitialNoise({
      n, emit, coordCondMask: field("coord_cond_mask"), moleculeIid: field("molecule_iid"),
      rng: rng0, centeringScale: DEFAULT_SAMPLING.centering?.scale ?? 0.2,
    });
    const t0 = {
      bb_ca: Math.fround(genSched.bb_ca.ts[0]),
      local_latents: Math.fround(genSched.local_latents.ts[0]),
    };
    const cache0 = {};
    const f0Seq = trunkSeqFeatures(batch, {
      n, xt: x0, feats: nn.feats_seq, weights, randperm: (len) => rng0.randperm(len, 0),
      noiseAlignChirality: nn.noise_align_chirality, cache: cache0,
    }).data;
    const f0Pair = pairFeatures(batch, {
      n, caNm: x0.bb_ca, feats: nn.feats_pair_repr,
      dist: { dim: nn.xt_pair_dist_dim, min: nn.xt_pair_dist_min, max: nn.xt_pair_dist_max },
      seqSepDim: nn.seq_sep_dim, cache: cache0,
    }).data;
    // 🔴 AND THE CONDITIONING, BUILT THE REFERENCE'S WAY - no cache, which is how the generate
    // loop calls it. The block above exports `g/cond` from a call that passes a fresh cache, and
    // "a fresh cache is the same as no cache" is an assumption, not a measurement.
    const f0Cond = trunkCondFeatures(batch, {
      n, t: t0, recycleRemaining: null, feats: nn.feats_cond_seq, weights,
    }).data;
    const f0Cond1 = trunkCondFeatures(batch, {
      n, t: t0, recycleRemaining: 1, feats: nn.feats_cond_seq, weights,
    }).data;
    const f0PairCond = pairCondFeatures(t0, nn.feats_pair_cond);
    writeDir(`${out}-gen-oracle`, {
      "gen/x0Bb": x0.bb_ca,
      "gen/x0Ll": x0.local_latents,
      "gen/f0Seq": Float32Array.from(f0Seq),
      "gen/f0Pair": Float32Array.from(f0Pair),
      "gen/f0Cond": Float32Array.from(f0Cond),
      "gen/f0Cond1": Float32Array.from(f0Cond1),
      "gen/f0PairCond": Float32Array.from(f0PairCond),
      "gen/xBb": gen.x.bb_ca,
      "gen/xLl": gen.x.local_latents,
      "gen/coors": gen.structure ? Float32Array.from(gen.structure.coors ?? []) : new Float32Array(0),
      ...perStep,
    }, {});
    console.log(`${out}-gen-oracle: the reference's own ${steps}-step sample`);
  }

  // The names the PDB needs, beside the tensors: the page's own tables, not a second copy.
  writeFileSync(`${out}-gen/names.txt`,
    `${ATOM37_NAMES.join(" ")}\n${UNIFIED_ATOM37_TOKENS.join(" ")}\n`);
  console.log(`${out}-gen: ${steps} steps, ${(gFloats * 4 / 1e6).toFixed(1)} MB`);
}

if (!oracleDir) process.exit(0);

// ---- the reference's answer, stage by stage -------------------------------------------------
const D = nn.token_dim;
const P = nn.pair_repr_dim;
const C = nn.dim_cond;
const PC = nn.pair_cond_factory_dim ?? 256;
const TRI = nn.tri_mult_c;
const H = nn.nheads;
const NORM = "rmsnorm_torch";
const W = (k) => weights[k].data ?? weights[k];
const stages = {};

const pairMask = new Uint8Array(n * n);
for (let i = 0; i < n; i += 1) {
  if (!pad[i]) continue;
  for (let j = 0; j < n; j += 1) if (pad[j]) pairMask[i * n + j] = 1;
}

const buildCondC = (condFeats) => {
  let c = linearLayer(condFeats, n, condFeats.length / n, W("nn.cond_factory.linear_out.weight"), C);
  c = swiGluTransition(c, pad, n, C, 2, weights, "nn.transition_c_1");
  return swiGluTransition(c, pad, n, C, 2, weights, "nn.transition_c_2");
};
const c0 = buildCondC(f.cond0);
const c1 = nRecycle > 1 ? buildCondC(f.cond1) : c0;
stages["o/c"] = c0;

const sInit = linearLayer(f.seq, n, f.seq.length / n, W("nn.init_repr_factory.linear_out.weight"), D);
for (let i = 0; i < n; i += 1) if (!pad[i]) sInit.fill(0, i * D, (i + 1) * D);
stages["o/sInit"] = sInit;

let zInit = linearLayer(f.pair, n * n, f.pair.length / (n * n),
  W("nn.pair_repr_builder.init_repr_factory.linear_out.weight"), P);
zInit = applyNorm(zInit, n * n, P, NORM, W("nn.pair_repr_builder.init_repr_factory.ln_out.weight"));
for (let r = 0; r < n * n; r += 1) if (!pairMask[r]) zInit.fill(0, r * P, (r + 1) * P);
let zCond = linearLayer(f.pairCond, 1, f.pairCond.length,
  W("nn.pair_repr_builder.cond_factory.linear_out.weight"), PC);
zCond = applyNorm(zCond, 1, PC, NORM, W("nn.pair_repr_builder.cond_factory.ln_out.weight"));
zInit = adaptiveLayerNorm(zInit, zCond, pairMask, n * n, P, 1, PC, NORM, weights, "nn.pair_repr_builder.adaln");
stages["o/zInit"] = zInit;

let s = new Float32Array(n * D);
let z = new Float32Array(n * n * P);
for (let cycle = 0; cycle < Math.max(1, nRecycle); cycle += 1) {
  const sDelta = linearLayer(applyNorm(s, n, D, NORM, W("nn.s_cycle_norm.weight")), n, D,
    W("nn.s_cycle_proj.weight"), D);
  const zDelta = linearLayer(applyNorm(z, n * n, P, NORM, W("nn.z_cycle_norm.weight")), n * n, P,
    W("nn.z_cycle_proj.weight"), P);
  s = sInit.map((v, k) => v + sDelta[k]);
  z = zInit.map((v, k) => v + zDelta[k]);
  const c = cycle === 0 ? c0 : c1;
  for (let i = 0; i < nn.nlayers; i += 1) {
    s = multiheadAttnAndTransition(s, z, c, pad, n, D, P, H, C, true, true, NORM, weights,
      `nn.transformer_layers.${i}`);
    if (nn.update_pair_repr && i < nn.nlayers - 1 && i % (nn.update_pair_repr_every_n ?? 2) === 0) {
      z = pairReprUpdate(s, z, pad, n, D, P, TRI, true, "hadamard", weights, `nn.pair_update_layers.${i}`);
    }
    if (cycle === Math.max(1, nRecycle) - 1) {
      stages[`o/s${i}`] = Float32Array.from(s);
      stages[`o/z${i}`] = Float32Array.from(z);
    }
  }
}

const ref = trunkRef;
stages["o/bbCa"] = ref.bb_ca;
stages["o/localLatents"] = ref.local_latents;
stages["o/logits"] = ref.logits;
stages["o/insertion"] = ref.insertion;
stages["o/s"] = ref.s;
stages["o/z"] = ref.z;

// 🔴 The duplicate loop above must agree with the function it duplicates, or the stages are a
// description of something this port is not being compared against.
const relRms = (a, b) => {
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i += 1) { const d = a[i] - b[i]; num += d * d; den += b[i] * b[i]; }
  return Math.sqrt(num / Math.max(den, 1e-30));
};
for (const [name, mine] of [["s", stages[`o/s${nn.nlayers - 1}`]], ["z", stages[`o/z${nn.nlayers - 1}`]]]) {
  const bar = relRms(mine, name === "s" ? ref.s : ref.z);
  if (!(bar < 1e-6)) {
    console.error(`the staged loop disagrees with forwardTrunkCpu on ${name}: relRMS ${bar.toExponential(2)}`);
    process.exit(1);
  }
}
if (!ref.insertion) { console.error("forwardTrunkCpu returned no insertion - the head did not run"); process.exit(1); }

// No scalars here: `Model::load` refuses a name that is in two of its directories, and the
// input directory beside this one already states n and nRecycle.
// ---- the decoder's answer, stage by stage ----------------------------------------------------
// 🔴 THE SAME DUPLICATE-AND-CHECK AS THE TRUNK'S. The stages come from driving `cpu/rfd4p/decoder.js`'s
// own pieces, which is a second copy of `decodeLatentsCpu`'s body; the postprocessed structure it
// produces is compared against what that function returns, so the copy cannot drift in silence.
const DP = "autoencoder.decoder";
const DW = (k) => weights[`${DP}.${k}`].data ?? weights[`${DP}.${k}`];
const decSeq = decoderSeqIds;
const decFeats = decoderSeqFeatures(batch, { n, caNm: x.bb_ca, z: ref.local_latents, seq: decSeq, weights, cfg: dec });
stages["d/feats"] = decFeats.data;

let dc = new Float32Array(n * dec.dim_cond);
dc = swiGluTransition(dc, attn, n, dec.dim_cond, 2, weights, `${DP}.transition_c_1`);
dc = swiGluTransition(dc, attn, n, dec.dim_cond, 2, weights, `${DP}.transition_c_2`);

let ds = linearLayer(decFeats.data, n, decFeats.dim, DW("init_repr_factory.linear_out.weight"), dec.token_dim);
for (let i = 0; i < n; i += 1) if (!attn[i]) ds.fill(0, i * dec.token_dim, (i + 1) * dec.token_dim);
stages["d/sInit"] = Float32Array.from(ds);
for (let l = 0; l < dec.nlayers; l += 1) {
  if (l === 0) {
    // 🔴 LAYER 0 OPENED UP, because the whole stack's disagreement enters here and relRMS over a
    // finished layer cannot say whether it is the adaLN or the attention.
    const pm = new Uint8Array(n * n);
    for (let i = 0; i < n; i += 1) {
      if (!attn[i]) continue;
      for (let j = 0; j < n; j += 1) if (attn[j]) pm[i * n + j] = 1;
    }
    const ada0 = adaptiveLayerNorm(ds, dc, attn, n, dec.token_dim, n, dec.dim_cond, "layer_norm",
      weights, `${DP}.transformer_layers.0.mhba.adaln`);
    stages["d/ada0"] = Float32Array.from(ada0);
    stages["d/attn0"] = Float32Array.from(pairBiasAttention(ada0, null, pm, n, dec.token_dim,
      dec.pair_repr_dim, dec.nheads, false, dec.use_qkln ?? true, "layer_norm", weights,
      `${DP}.transformer_layers.0.mhba.mha`));
  }
  ds = multiheadAttnAndTransition(ds, null, dc, attn, n, dec.token_dim, dec.pair_repr_dim, dec.nheads,
    dec.dim_cond, false, dec.use_qkln ?? true, "layer_norm", weights, `${DP}.transformer_layers.${l}`);
  stages[`d/s${l}`] = Float32Array.from(ds);
}
const decHead = (name, out) => linearLayer(
  layerNorm(ds, n, dec.token_dim, DW(`${name}.0.weight`), DW(`${name}.0.bias`)),
  n, dec.token_dim, DW(`${name}.1.weight`), out);
const decLogits = decHead("logit_linear", dec.n_residue_types);
const decCoors = decHead("struct_linear", 111);
const decPresence = decHead("atom_presence_linear", 37);
stages["d/logits"] = decLogits;
stages["d/coors"] = decCoors;
stages["d/presence"] = decPresence;

{
  const mine = decoderPostprocess(batch, {
    n, caNm: x.bb_ca, seq: decSeq, logits: decLogits, coorsFlat: decCoors, presence: decPresence,
    nTypes: dec.n_residue_types,
  });
  const theirs = decodeLatentsCpu(batch, {
    n, caNm: x.bb_ca, z: ref.local_latents, seq: decSeq, weights, cfg: dec });
  const bar = relRms(mine.coorsNm, theirs.coorsNm);
  if (!(bar < 1e-6)) {
    console.error(`the staged decoder disagrees with decodeLatentsCpu: relRMS ${bar.toExponential(2)}`);
    process.exit(1);
  }
  stages["d/coorsNm"] = theirs.coorsNm;
  stages["d/atomMask"] = Float32Array.from(theirs.atomMask);
}

const oFloats = writeDir(oracleDir, stages, {});
console.log(`${oracleDir}: ${Object.keys(stages).length} stages, ${(oFloats * 4 / 1e6).toFixed(1)} MB`);
