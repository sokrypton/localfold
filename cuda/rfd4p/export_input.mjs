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
import { parseSafetensors } from "../../shared/rfd4p/safetensors.js";
import {
  pairCondFeatures, pairFeatures, trunkCondFeatures, trunkSeqFeatures,
} from "../../shared/rfd4p/features.js";
import {
  DEFAULT_SAMPLING, flowEmitMasks, sampleInitialNoise, sampleSchedule, seededRng,
} from "../../shared/rfd4p/flow.js";
import {
  adaptiveLayerNorm, applyNorm, forwardTrunkCpu, linearLayer, multiheadAttnAndTransition,
  pairReprUpdate, swiGluTransition,
} from "../../cpu/rfd4p/reference.js";

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
const checkpoint = option("checkpoint", `${hub}/a06e2509c7033b010fc43614e9fd840303588502/model.safetensors`);
const cfg = JSON.parse(readFileSync(option("config", `${repo}model-rfd4p-int8/config.json`), "utf8")).config;
const nn = cfg.nn;

const preset = getRfd4pPresetById(presetId);
if (!preset) { console.error(`no preset ${presetId}`); process.exit(1); }
if (preset.cifUrl) {
  console.error(`${presetId} needs a structure; only the pure-spec presets export offline for now`);
  process.exit(1);
}
const { batch, n } = buildBatch(
  { contig: preset.contig, ...(preset.conditions === undefined ? {} : { conditions: preset.conditions }) },
  { seed });

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

/** One model.idx/model.bin directory from `{name: Float32Array}` plus `{name: number}` scalars. */
function writeDir(dir, tensors, meta) {
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
  closeSync(fd);
  writeFileSync(`${dir}/model.idx.tmp`, `${lines.join("\n")}\n`);
  renameSync(`${dir}/model.idx.tmp`, `${dir}/model.idx`);
  return offset;
}

const floats = writeDir(out, {
  "in/seq": f.seq,
  "in/cond0": f.cond0,
  "in/cond1": f.cond1,
  "in/pair": f.pair,
  "in/pairCond": f.pairCond,
  "in/mask": Float32Array.from(pad, (v) => (v ? 1 : 0)),
}, {
  n,
  nRecycle,
  seqFeatDim: f.seq.length / n,
  condFeatDim: f.cond0.length / n,
  pairFeatDim: f.pair.length / (n * n),
  pairCondFeatDim: f.pairCond.length,
});
console.log(`${out}: ${n} tokens, ${(floats * 4 / 1e6).toFixed(1)} MB`);

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

const ref = forwardTrunkCpu(n, f, pad, weights, nn, nRecycle, { insertion: true });
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
const oFloats = writeDir(oracleDir, stages, {});
console.log(`${oracleDir}: ${Object.keys(stages).length} stages, ${(oFloats * 4 / 1e6).toFixed(1)} MB`);
