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
import { ccdBlocksForRawStructure } from "../../shared/rfd4p/pipeline/ccd.js";
import { CONDITION_SPEC_EXAMPLE_KEYS, conditionSpecExamples } from "../../shared/rfd4p/pipeline/spec.js";
import { getTokenStarts } from "../../shared/rfd4p/pipeline/bonds.js";
import { getRfd4pPresetById } from "../../shared/rfd4p/presets.js";
import { parseConditionSpecYaml } from "../../shared/rfd4p/pipeline/spec.js";
import { dirname, isAbsolute, resolve } from "node:path";
import {
  ALLOWED_RESIDUE_TYPES, ATOM37_NAMES, UNIFIED_ATOM37_ENCODING, UNIFIED_ATOM37_TOKENS,
  UNIFIED_RESTYPE_ATOM37_MASK,
} from "../../shared/rfd4p/encoding.js";
import { COND } from "../../shared/rfd4p/features.js";
import { parseSafetensors } from "../../shared/rfd4p/safetensors.js";
import {
  pairCondFeatures, pairFeatures, trunkCondFeatures, trunkSeqFeatures,
} from "../../shared/rfd4p/features.js";
import {
  DEFAULT_RECYCLE_SCHEDULE, DEFAULT_SAMPLING, flowEmitMasks, isStochasticStep, recycleDepths, resolveSampling,
  sampleInitialNoise, sampleSchedule, samplingMeta, seededRng,
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
import { SEQUENCE_PROBABILITY_TOKENS, jobMetricsTags, metricsTagBits } from "../../shared/rfd4p/metrics.js";
import { encodeLatentsCpu, encoderSeqFeatures } from "../../cpu/rfd4p/encoder.js";

const args = process.argv.slice(2);
const out = args.find((a) => !a.startsWith("--"));
if (!out) {
  console.error("usage: export_input.mjs <out dir> [--preset=<id> | --spec=<job.yaml|.json>]"
    + " [--seed=N] [--recycles=N] [--oracle=<dir>] [--generate [--gen-oracle] [--steps=N]]"
    + " [--encoder] [--length=N]");
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

// 🔴 A JOB IS THE PROTOCOL'S OWN ENTRY, NOT A PRESET ID. `--spec=<file>` takes a condition spec -
// the `{contig, conditions, ...}` object RFD4-Proteina's own YAML and JSON jobs are - and treats
// it as a preset with the same fields, so everything below (the structure, the sampler mode, the
// step count) is driven by the file rather than by a name compiled into presets.js. Without it
// this port could only ever run the ten examples.
const specPath = option("spec", "");
const exampleId = option("example", undefined);
const preset = specPath
  ? (() => {
      const text = readFileSync(specPath, "utf8");
      const parsed = parseConditionSpecYaml(text);   // JSON or YAML, key order kept
      // 🔴 A SPEC FILE IS EXAMPLES BY NAME, AS UPSTREAM'S IS (`global_args` folded into each); this
      // port's own flat job is the one-example case. Several examples need `--example=`.
      let examples;
      try {
        examples = conditionSpecExamples(parsed);
      } catch (err) {
        console.error(`${specPath}: ${err.message}`);
        process.exit(1);
      }
      const names = Object.keys(examples);
      if (exampleId !== undefined && !examples[exampleId]) {
        console.error(`no example '${exampleId}' in ${specPath}; it has ${names.join(", ")}`);
        process.exit(1);
      }
      if (exampleId === undefined && names.length > 1) {
        console.error(`${specPath} has ${names.length} examples (${names.join(", ")}); name one with --example=`);
        process.exit(1);
      }
      const name = exampleId ?? names[0];
      const job = examples[name];
      // A spec in upstream's shape carries upstream's four fields and no others: a typo'd one
      // applies nothing, so it is refused (`build_annotated_atom_array_from_condition_spec`).
      const flat = "input" in parsed || "contig" in parsed;
      const unknown = flat ? [] : Object.keys(job).filter((k) => !CONDITION_SPEC_EXAMPLE_KEYS.includes(k)).sort();
      if (unknown.length) {
        console.error(`${name}: unrecognized condition_spec field(s) ${JSON.stringify(unknown)}. `
          + `Recognized fields: ${JSON.stringify([...CONDITION_SPEC_EXAMPLE_KEYS].sort())}.`);
        process.exit(1);
      }
      if (!job.contig && !job.input) {
        console.error(`${specPath} has neither a contig nor an input`);
        process.exit(1);
      }
      // A relative `input` is the spec file's, as upstream resolves one (`_resolve_input_path`).
      const specDir = dirname(resolve(specPath));
      const listed = job.input ?? job.cifUrl ?? job.inputCif ?? job.input_cif;
      const inputs = listed === undefined || listed === null ? null
        : (Array.isArray(listed) ? listed : [listed]).map((p) => (isAbsolute(p) ? p : resolve(specDir, p)));
      return {
        id: name === "job" ? (job.id ?? "job") : name,
        contig: job.contig,
        conditions: job.conditions,
        inputs,
        cifUrl: inputs?.[0] ?? null,
        // A job is a spec unless it says `specMode: false` - see featurise.cpp's note; the
        // file's annotations are kept only under `input_as_annotated_cif`.
        specMode: job.specMode ?? true,
        fromJobFile: true,
        inputAsAnnotatedCif: Boolean(job.input_as_annotated_cif ?? job.inputAsAnnotatedCif ?? false),
        nsteps: job.nsteps,
        recycleSchedule: job.recycleSchedule ?? job.recycle_schedule,
        partialTrajectoryFractionSteps:
          job.partialTrajectoryFractionSteps ?? job.partial_trajectory_fraction_steps,
        inverseFolding: job.inverseFolding ?? job.inverse_folding,
        inverseFoldingMethod: job.inverseFoldingMethod ?? job.inverse_folding_method,
        partialTBackbone: job.partialTBackbone ?? job.partial_t_backbone,
        partialTLatent: job.partialTLatent ?? job.partial_t_latent,
        couplePartialT: job.couplePartialT ?? job.couple_partial_t,
        sampling: job.sampling,
        metrics_tags: job.metrics_tags,
      };
    })()
  : getRfd4pPresetById(presetId);
if (!preset) { console.error(`no preset ${presetId}`); process.exit(1); }
// The sampler: the released one, or the job's `sampling` over it (upstream's config shape).
let SAMPLING;
try {
  SAMPLING = resolveSampling(preset.sampling ?? {});
} catch (err) {
  console.error(err.message);
  process.exit(1);
}
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
const fromAnnotations = !length && preset.cifUrl !== null && preset.cifUrl !== undefined
  && (preset.fromJobFile ? preset.specMode === false : preset.specMode !== true);
// A preset names its structure in the checkout, as the page's preset URLs are; a job file names
// its own, already resolved above.
const inputPaths = length ? [] : (preset.inputs
  ?? (preset.cifUrl ? [preset.cifUrl.startsWith("/") ? preset.cifUrl : `${repo}${preset.cifUrl.replace(/^\.\//, "")}`] : []));
// Bytes, as given: mmCIF or BinaryCIF, plain, gzip or Zstandard - the reader tells them apart.
const inputCif = inputPaths.length ? new Uint8Array(readFileSync(inputPaths[0])) : undefined;
if (fromAnnotations && inputPaths.length > 1) {
  console.error("specMode: false takes one structure as the job; this one names several");
  process.exit(1);
}
// `--length=N` is one unconditioned protein segment of N residues, for sizing and timing: the trunk's
// cost is a function of the token count and nothing in the conditioning changes a single shape.
const spec = length
  ? { contig: `a1:${length}-${length}P` }
  : {
    contig: preset.contig,
    ...(preset.conditions === undefined ? {} : { conditions: preset.conditions }),
    ...(preset.inputAsAnnotatedCif ? { input_as_annotated_cif: true } : {}),
    ...(inputPaths.length ? { input: inputPaths } : {}),
  };
// A raw structure is PREPARED (pipeline/fresh.js), and its non-standard components need their
// dictionary entries - fetched as the page fetches them.
const ccdBlocks = new Map();
if (!fromAnnotations && !preset.inputAsAnnotatedCif) {
  for (const path of inputPaths) {
    for (const [id, block] of await ccdBlocksForRawStructure(readFileSync(path))) ccdBlocks.set(id, block);
  }
}
const built = fromAnnotations
  ? buildBatch(inputCif, { seed })
  : buildBatch(spec, { seed, ccdBlocks, loadInput: (path) => readFileSync(path) });
const { batch, n } = built;
// The per-token naming an input structure supplies, for the writer at the far end.
const outNaming = (() => {
  const atoms = built.atoms;
  // A pure contig has atoms too - `<M>` stubs in the chains and numbering the contig gave them -
  // and the page's writer names a design from them, so they cross for every job.
  if (!atoms) return null;
  const starts = getTokenStarts(atoms, false);
  return {
    resId: starts.map((a) => Number(atoms.res_id[a])),
    chain: starts.map((a) => {
      const cid = String(atoms.chain_id[a] ?? "");
      let packed = 0;
      for (let k = 0; k < cid.length && k < 2; k += 1) packed |= cid.charCodeAt(k) << (8 * k);
      return packed;
    }),
    tables: outputTables(atoms, starts),
  };
})();

// 🔴 WHAT THE PAGE'S WRITER READS OFF THE INPUT ATOMS, carried to the native one - the featuriser's
// `outputTables` writes the same keys and the featurise gate holds them to these bytes. Per token
// from its first atom; per input atom keyed by its slot in the token's OWN atom37 layout (slot 1
// for an atomised token's one atom), because the writer finds an output atom's by NAME.
function outputTables(atoms, starts) {
  const packString = (v, words, into) => {
    const s2 = String(v ?? "");
    for (let w = 0; w < words; w += 1) {
      let packed = 0;
      for (let k = 0; k < 4; k += 1) {
        const at = w * 4 + k;
        if (at < s2.length) packed |= s2.charCodeAt(at) << (8 * k);
      }
      into.push(packed | 0);
    }
  };
  const packDouble = (v, into, at) => {
    const i32 = new Int32Array(new Float64Array([v]).buffer);
    into[at * 2] = i32[0];
    into[at * 2 + 1] = i32[1];
  };
  // `getTokenStarts` returns the starts alone, without the end the native one appends.
  const nTok = starts.length;
  const nAtoms = atoms.atom_name.length;
  const residueType = batch.residue_type?.data ?? batch.residue_type;
  const src = atoms["annotation_src-component_1_atom"];
  const t = {
    "g/inName": new Int32Array(nTok * 37), "g/inElem": new Int32Array(nTok * 37),
    "g/inB": new Int32Array(nTok * 37 * 2), "g/inOcc": new Int32Array(nTok * 37 * 2),
    "g/inCharge": new Int32Array(nTok * 37),
    "g/outAtomize": [], "g/outHetero": [], "g/outEntity": [], "g/outInsCode": [],
    "g/outResName": [], "g/outSrcComp": [],
    // What shared/rfd4p/metrics.js reads off the input, for the native metrics: per token its
    // polymer and protein flags, C_SEC, chain instance, CCD type, C_SEQ and the 28 sequence
    // probabilities (as float bits); per input atom its hotspot flag; and the input's BONDS as
    // pairs of (token * 37 + slot), in the input's order, where both ends have a slot.
    "g/mTok": [], "g/mChainIid": [], "g/mCompType": [], "g/mSeq": [], "g/mSeqProb": [],
    "g/mHot": new Int32Array(nTok * 37),
    // ...and for the motif RMSDs: per token its pn_unit, molecule and molecule entity; every input
    // atom in the input's order (`IN_ATOM_WORDS` each: its slot or -1, flags - indexed, atomised,
    // protein, polymer, coordinate-conditioned - chain type, charge, residue id, coordinates as float bits, then atom,
    // residue, element, chain, insertion code, chain instance and pn_unit as packed text, and its
    // molecule and entity); and the input's bonds as (atom, atom, type) triples.
    "g/mPnUnit": [], "g/mTokMol": [], "g/mInAtom": [], "g/mInBonds": [],
  };
  const slotOf = new Int32Array(nAtoms).fill(-1);
  const tokenOf = new Int32Array(nAtoms).fill(-1);
  const batchMol = batch.molecule_iid?.data ?? batch.molecule_iid;
  const batchEntity = batch.molecule_entity_id?.data ?? batch.molecule_entity_id;
  const bits = new Int32Array(new Float32Array(1).buffer);
  const floatBits = (v) => { new Float32Array(bits.buffer)[0] = v; return bits[0]; };
  for (let tk = 0; tk < nTok; tk += 1) {
    const a0 = starts[tk];
    const atomised = Boolean(atoms.atomize?.[a0]);
    t["g/outAtomize"].push(atomised ? 1 : 0);
    t["g/outHetero"].push(atoms.hetero?.[a0] ? 1 : 0);
    t["g/outEntity"].push(atoms.label_entity_id?.[a0] != null ? Number(atoms.label_entity_id[a0]) : 0);
    packString(atoms.ins_code?.[a0] ?? "", 1, t["g/outInsCode"]);
    packString(atoms.res_name[a0], 2, t["g/outResName"]);
    packString(src?.[a0] ?? ".", 4, t["g/outSrcComp"]);
    t["g/mTok"].push(atoms.is_polymer?.[a0] ? 1 : 0, atoms.is_protein?.[a0] ? 1 : 0,
      Number(atoms["condition_secondary-structure_1_residue"]?.[a0] ?? 0));
    packString(atoms.chain_iid?.[a0] ?? atoms.chain_id[a0], 4, t["g/mChainIid"]);
    packString(atoms.pn_unit_iid?.[a0] ?? atoms.chain_iid?.[a0] ?? atoms.chain_id[a0], 4, t["g/mPnUnit"]);
    // The BATCH's molecule and entity ids (dense, then permuted): what both featurisers agree on.
    t["g/mTokMol"].push(Number(batchMol[tk]), Number(batchEntity[tk]));
    packString(String(atoms.chem_comp_type?.[a0] ?? "").toUpperCase(), 8, t["g/mCompType"]);
    packString(atoms.condition_sequence_1_residue?.[a0] ?? "<M>", 2, t["g/mSeq"]);
    for (const tok of SEQUENCE_PROBABILITY_TOKENS) {
      const v = atoms[`condition_sequence-probability-${tok}_1_residue`]?.[a0];
      t["g/mSeqProb"].push(floatBits(v == null ? NaN : Number(v)));
    }
    const rt = residueType[tk];
    const layout = UNIFIED_ATOM37_ENCODING.token_atoms[UNIFIED_ATOM37_TOKENS[rt]] ?? [];
    for (let ai = starts[tk]; ai < (starts[tk + 1] ?? nAtoms); ai += 1) {
      tokenOf[ai] = tk;
      let slot = -1;
      if (atomised) slot = ai === a0 ? 1 : -1;
      else slot = layout.indexOf(atoms.atom_name[ai]);
      if (slot < 0) continue;
      const at = tk * 37 + slot;
      slotOf[ai] = at;
      t["g/mHot"][at] = atoms.condition_hotspot_1_atom?.[ai] ? 1 : 0;
      const one = [];
      packString(atoms.atom_name[ai], 1, one);
      t["g/inName"][at] = one[0];
      one.length = 0;
      packString(atoms.element?.[ai] ?? "", 1, one);
      t["g/inElem"][at] = one[0];
      t["g/inCharge"][at] = Number(atoms.charge?.[ai] ?? 0);
      packDouble(Number(atoms.b_factor?.[ai]), t["g/inB"], at);
      packDouble(Number(atoms.occupancy?.[ai]), t["g/inOcc"], at);
    }
  }
  const indexed = atoms.condition_index_1_residue ?? atoms.annotations?.condition_index_1_residue;
  for (let ai = 0; ai < nAtoms; ai += 1) {
    const row = t["g/mInAtom"];
    const crd = atoms.condition_coordinate_1_atom ?? atoms.annotations?.condition_coordinate_1_atom;
    const group = (atoms.condition_CoordinateGroup_1_atom ?? atoms.annotations?.condition_CoordinateGroup_1_atom)?.[ai];
    const conditioned = (Array.isArray(crd?.[ai]) && crd[ai].every(Number.isFinite))
      || (group != null && group !== -1 && group !== 0);
    const flags = ((indexed?.[ai] ?? true) ? 1 : 0) | (atoms.atomize?.[ai] ? 2 : 0)
      | (atoms.is_protein?.[ai] ? 4 : 0) | (atoms.is_polymer?.[ai] ? 8 : 0) | (conditioned ? 16 : 0);
    row.push(slotOf[ai], flags, Number(atoms.chain_type?.[ai] ?? -1), Number(atoms.charge?.[ai] ?? 0),
      Number(atoms.res_id[ai]));
    for (let c = 0; c < 3; c += 1) row.push(floatBits(Number(atoms.coord?.[ai]?.[c] ?? NaN)));
    packString(atoms.atom_name[ai], 1, row);
    packString(atoms.res_name[ai], 2, row);
    packString(String(atoms.element?.[ai] ?? "").toUpperCase(), 1, row);
    packString(atoms.chain_id[ai], 1, row);
    packString(atoms.ins_code?.[ai] ?? "", 1, row);
    packString(atoms.chain_iid?.[ai] ?? atoms.chain_id[ai], 4, row);
    packString(atoms.pn_unit_iid?.[ai] ?? atoms.chain_iid?.[ai] ?? atoms.chain_id[ai], 4, row);
    row.push(Number(batchMol[tokenOf[ai]]), Number(batchEntity[tokenOf[ai]]));
  }
  for (const [p, q, type] of atoms.bonds ?? []) t["g/mInBonds"].push(p, q, type ?? 1);
  return Object.fromEntries(Object.entries(t).map(([k, v]) => [k, Int32Array.from(v)]));
}

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
const sched = sampleSchedule({ ...SAMPLING, nsteps: preset.nsteps ?? DEFAULT_SAMPLING.nsteps });
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
// tests before it takes its own on-device step loop (`caOff === 0 && latOff === 3`). The two
// features that also move per step are handled rather than frozen: a chirality frame is
// recomputed on the device each step (`src/chirality.cuh`) and a coordinate group's LABELS are
// permuted from a replayed `randperm` stream (`crdUniq`/`crdPerm`). The rule that matters is that
// a per-step feature is replayed or recomputed and NEVER silently frozen - freezing one is a
// plausible structure generated from a feature the model was supposed to see move.
//
// The pair features go out without their distance bins (the first 30 columns), which the port fills
// from the current CA each step.
if (generate) {
  // 🔴 NOTHING REFUSES HERE ANY MORE, AND THIS COMMENT SAID TWO THINGS DID. The chirality
  // alignment moves between steps and the port recomputes it on the device (src/chirality.cuh),
  // which is what lets the small-molecule preset generate at all; a coordinate group's
  // permutation is a `randperm` draw at the top of every step and is exported beside the noise
  // (`crdUniq`/`crdPerm` below). The paragraph above still states the RULE - a per-step feature
  // is replayed or recomputed, never silently frozen - which is the part worth keeping.
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
  const frac = preset.partialTrajectoryFractionSteps ?? null;
  // 🔴 EVERY FORM `resolve_partial_t` TAKES, not only a fraction: explicit start times and a
  // coupled one are upstream's "advanced interface", and a job naming them was run as an
  // ordinary design here as it was natively.
  const partialOpts = {
    nsteps: baseNsteps,
    ...(inverseFolding ? { inverseFolding: true } : {}),
    ...(preset.inverseFoldingMethod != null ? { inverseFoldingMethod: preset.inverseFoldingMethod } : {}),
    ...(frac != null ? { partialTrajectoryFractionSteps: frac } : {}),
    ...(preset.partialTBackbone != null ? { partialTBackbone: preset.partialTBackbone } : {}),
    ...(preset.partialTLatent != null ? { partialTLatent: preset.partialTLatent } : {}),
    ...(preset.couplePartialT != null ? { couplePartialT: preset.couplePartialT } : {}),
  };
  const partialTDict = resolvePartialT(partialOpts, SAMPLING);
  const isPartial = Object.keys(partialTDict).length > 0;
  const steps = isPartial && frac != null ? partialStepCount(frac, baseNsteps) : baseNsteps;
  // 🔴 `fullNsteps` IS baseNsteps ALWAYS, not only when partial - that is what `generateRfd4pCpu`
  // passes, and with it `recycleDepths` takes the schedule's TAIL rather than resampling it.
  const depths = recycleDepths(recycleOverride || preset.recycleSchedule || DEFAULT_RECYCLE_SCHEDULE, steps,
    baseNsteps);
  const genSched = isPartial
    ? buildPartialSchedule(partialTDict, steps, SAMPLING,
      { fullNsteps: frac != null ? baseNsteps : null })
    : sampleSchedule({ ...SAMPLING, nsteps: steps });
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
  // 🔴 THE LABEL PERMUTATION IS DRAWN AT THE TOP OF EACH STEP, BEFORE THAT STEP'S NOISE, and it
  // consumes RAW uniforms rather than normals - `randperm` takes `next()` once per swap where
  // every other draw here goes through Box-Muller. Modelling it as `randn` would advance the
  // stream by a different amount and move every later draw of the run.
  const crdUniqAll = [...new Set(Array.from(batch[COND.CRDGRP] ? field(COND.CRDGRP) : []))]
    .filter((v) => v !== -1).sort((a, b) => a - b);
  const crdPerm = [];
  const stepNoise = [];
  for (let s2 = 0; s2 < steps; s2 += 1) {
    crdPerm.push(...rng2.randperm(crdUniqAll.length));
    for (const dm of ["bb_ca", "local_latents"]) {
      const t2 = Math.fround(genSched[dm].ts[s2]);
      const stochastic = isStochasticStep(t2, genSched[dm].step);
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
    // 🔴 FLOATS, NOT INTS. Put in the `ints` map these became an Int32Array and every NaN
    // became a 0 - a conditioned coordinate of zero is a real position, so the anchor set
    // would have been every atom at the origin rather than none.
    "g/condCoord": batch[COND.CRD]
      ? Float32Array.from(field(COND.CRD))
      : new Float32Array(n * 37 * 3).fill(NaN),
    "g/condIntCrd": batch[COND.INTCRD]
      ? Float32Array.from(field(COND.INTCRD))
      : new Float32Array(n * 37 * 3).fill(NaN),
    // `coord_to_be_noised`: the centred INPUT position, which is what the geometric insertion
    // mapping measures against - not the generated coordinates.
    "g/coords": Float32Array.from(field("coords")),
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
    nCrdGroups: crdUniqAll.length,
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
    ...samplingMeta(SAMPLING),
    // The job's `metrics_tags`, as shared/rfd4p/metrics.js's bits: which design metrics the
    // native `--metrics=` computes.
    metricsTags: metricsTagBits(jobMetricsTags(preset)),
  }, {
    "g/depths": Int32Array.from(depths),
    "g/stochastic": Int32Array.from(stepNoise),
    "g/coordCond": Int32Array.from(field("coord_cond_mask"), (v) => (v ? 1 : 0)),
    "g/moleculeIid": Int32Array.from(field("molecule_iid")),
    // 🔴 THE CHAIN, SO A MULTI-CHAIN DESIGN COMES OUT AS MORE THAN ONE CHAIN. Without it the
    // port wrote every atom into chain A, which for a binder or a complex is a structure no
    // viewer and no scorer can read correctly.
    "g/chainIid": Int32Array.from(field("chain_iid")),
    // 🔴 AND THE INPUT'S OWN NUMBERING AND CHAIN LETTERS, where there was an input. A motif's
    // residues are deposited under their own numbers; writing the token index renumbers them and
    // the design no longer superposes on the target it was built against. The chain is its
    // CHARACTERS packed little-endian, not an index - `chainName(i)` spells A..Z for chains this
    // port named, and an input with chains H and L would come out A and B.
    ...(outNaming === null ? {} : {
      "g/outResId": Int32Array.from(outNaming.resId),
      "g/outChain": Int32Array.from(outNaming.chain),
      ...outNaming.tables,
    }),
    "g/allowedTypes": Int32Array.from(ALLOWED_RESIDUE_TYPES.flatMap((r) => Array.from(r, (v) => (v ? 1 : 0)))),
    "g/atom37Legal": Int32Array.from(UNIFIED_RESTYPE_ATOM37_MASK.flatMap((r) => Array.from(r, (v) => (v ? 1 : 0)))),
    "g/coordGroup": Int32Array.from(batch[COND.CRDGRP] ? field(COND.CRDGRP) : new Int32Array(n * 37).fill(-1)),
    // Which atoms are indexed, and the INPUT's residue per token - the two things the mapping
    // reads that the generated structure does not carry.
    "g/condIndex": Int32Array.from(field(COND.IDX), (v) => (v ? 1 : 0)),
    ...(crdUniqAll.length ? {
      "g/crdUniq": Int32Array.from(crdUniqAll),
      "g/crdPerm": Int32Array.from(crdPerm),
    } : {}),
    "g/resType": Int32Array.from(field("residue_type")),
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
      ...partialOpts,
      seed,
      // The job's sampler, which the port is handed through `-gen`'s meta.
      sampling: SAMPLING,
      // 🔴 THE SAME RECYCLE SCHEDULE THE PORT IS HANDED, SAID OUT LOUD. `generateRfd4pCpu`
      // defaults to `medium` and the exporter writes `g/depths` from the PRESET's, so leaving it
      // off ran the reference at six recycles against the port's four - the two sides doing
      // different amounts of work on identical inputs. It read as a wrong trajectory: the noise
      // and the step-0 features were exact to the last bit and the velocity was 8e-2 out.
      recycleSchedule: recycleOverride || preset.recycleSchedule || DEFAULT_RECYCLE_SCHEDULE,
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
