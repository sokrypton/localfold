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
import { buildBatch } from "../../shared/rfd4p/pipeline/index.js";
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
  adaptiveLayerNorm, applyNorm, forwardTrunkCpu, layerNorm, linearLayer,
  multiheadAttnAndTransition, pairBiasAttention, pairReprUpdate, swiGluTransition,
} from "../../cpu/rfd4p/reference.js";
import {
  aeMasks, decodeLatentsCpu, decoderPostprocess, decoderSeqFeatures, sequenceForDecoder,
} from "../../cpu/rfd4p/decoder.js";

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
const length = Number(option("length", "0"));
const checkpoint = option("checkpoint", `${hub}/a06e2509c7033b010fc43614e9fd840303588502/model.safetensors`);
const cfg = JSON.parse(readFileSync(option("config", `${repo}model-rfd4p-int8/config.json`), "utf8")).config;
const nn = cfg.nn;

const preset = getRfd4pPresetById(presetId);
if (!preset) { console.error(`no preset ${presetId}`); process.exit(1); }
if (preset.cifUrl && !length) {
  console.error(`${presetId} needs a structure; only the pure-spec presets export offline for now`);
  process.exit(1);
}
// `--length=N` is one unconditioned protein segment of N residues, for sizing and timing: the trunk's
// cost is a function of the token count and nothing in the conditioning changes a single shape.
const spec = length
  ? { contig: `a1:${length}-${length}P` }
  : { contig: preset.contig, ...(preset.conditions === undefined ? {} : { conditions: preset.conditions }) };
const { batch, n } = buildBatch(spec, { seed });

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
  if ((cache.crdUniq?.length ?? 0) > 1 || (nn.noise_align_chirality && (cache.activeFrames?.length ?? 0) > 0)) {
    console.error("--generate only covers inputs whose per-step features are the noised state alone");
    process.exit(1);
  }
  const steps = preset.nsteps ?? DEFAULT_SAMPLING.nsteps;
  const depths = recycleDepths(preset.recycleSchedule ?? "none", steps);
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
      bb_ca: Math.fround(sched.bb_ca.ts[s2]),
      local_latents: Math.fround(sched.local_latents.ts[s2]),
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
  take(n * 3);
  const offsetMask = (i) => emit.bb_ca[i] && !(field("coord_cond_mask")[i]);
  const groupIds = [...new Set(Array.from({ length: n }, (_, i) => i).filter(offsetMask)
    .map((i) => field("molecule_iid")[i]))].sort((a, b) => a - b);
  for (let g = 0; g < groupIds.length; g += 1) take(3);
  take(n * 32);
  const stepNoise = [];
  for (let s2 = 0; s2 < steps; s2 += 1) {
    for (const dm of ["bb_ca", "local_latents"]) {
      const t2 = Math.fround(sched[dm].ts[s2]);
      const cfg2 = sched[dm].step;
      const stochastic = t2 !== 0 && !(t2 > cfg2.t_lim_ode || t2 < cfg2.t_lim_ode_below);
      stepNoise.push(stochastic ? 1 : 0);
      if (stochastic) take(n * (dm === "bb_ca" ? 3 : 32));
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
  }, {
    steps,
    nGroups: groupIds.length,
    centeringScale: DEFAULT_SAMPLING.centering.scale,
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
  });
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
