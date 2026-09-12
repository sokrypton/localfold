/**
 * Fold a protein with AF3 on the GPU, end to end, and write a PDB.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/fold.js --sequence=GWSTELEK... \
 *       --steps=200 --model=/model-af3-int5/manifest.json
 *     node tools/gpu-chrome.mjs tools/gpu/fold.js --dump=/oracle-dumps/af3-6mrr.json
 *
 * --sequence folds what you type, through src/af3/featurise.js. --dump folds
 * AF3's own batch and reports the disagreement at every point where the two can
 * be compared, which is the only way the trunk can be checked against AF3's.
 *
 * WHAT IS HERE AND NOT IN src/af3/fold.js: argument parsing, the comparison
 * against the dump, and the geometry report. The pipeline itself is shared with
 * the page, because the page has to run what was measured.
 */
import { memorySnapshot, setMemoryBudget } from "../../src/runtime/device-memory.js";import { featuriseProtein } from "../../src/af3/featurise.js";
import { ccdUrl, parseCcdComponent } from "../../src/af3/ccd-component.js";
import { af3ContactClasses } from "../../src/af3/contact-classes.js";
import { CLASS_LIGAND, CLASS_NUCLEIC } from "../../src/heads/contact-threshold.js";
import { af3MsaFromA3m } from "../../src/af3/msa-features.js";
import { mergeRowAlignedChainA3ms } from "../../src/input/chains.js";
import { foldBatch, toPdb, backboneGeometry } from "../../src/af3/fold.js";
import { assertChainGeometry } from "./chain-geometry.js";
import { confidenceWeights, openAf3Store, trunkWeights } from "../../src/af3/weights.js";
import { warmTrunkPipelines } from "../../src/af3/fold.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";
import { Af3DiffusionTransformerGpu } from "../../src/af3/diffusion-transformer-webgpu.js";
import { profileDevice } from "./profile.js";
import { profileBuffers } from "./buffer-profile.js";
import { setDeviceTuning, deviceTuning, DEFAULT_TUNING }
  from "../../src/runtime/device-profile.js";

function option(args, name, fallback) {
  const prefix = `--${name}=`;
  return args.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

const floats = (source) => Float32Array.from(source, (v) => Number(v));
const ints = (source) => Int32Array.from(source, (v) => Number(v));

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

/**
 * AF3's own batch, in the shape src/af3/featurise.js produces, so the fold reads
 * one object either way and the two paths cannot silently diverge in what they
 * supply.
 */
export function batchFromDump(dump) {
  const tokens = dump.tokens;
  const dense = 24;
  // 🔴 PADDED, DELIBERATELY, AND ONLY HERE. The featuriser sizes its subsets
  // from the real atom count; this path does not, because its gathers ARE
  // AF3's - read straight out of the dump, at the dense grid's width - and a
  // subset count that disagreed with the arrays beside it is the failure this
  // repository keeps meeting. The two paths differ in shape and agree to every
  // digit in what they fold, which is what the padding was worth.
  const subsets = Math.ceil((tokens * dense) / 32);
  const raw = (name) => dump.inputs[name].data;
  // 🔴 count IS NOT DECORATION. convert() in atom-encoder-reference.js sizes its
  // output from it, so a gather without one silently produces a zero-length
  // tensor - which reads downstream as a model that runs and folds a 17 A
  // spaghetti rather than as an error.
  const gather = (name) => {
    const indices = ints(raw(`${name}:gather_idxs`));
    return {
    indices, mask: floats(raw(`${name}:gather_mask`)), count: indices.length };
  };
  const refMask = floats(raw("ref_mask"));
  let atomCount = 0;
  for (const value of refMask) atomCount += value;
  return {
    sequence: dump.sequence, tokens, dense, subsets, atomCount,
    shape: { tokens, dense, subsets, queries: 32, keys: 128 },
    aatype: ints(raw("aatype")), profile: floats(raw("profile")),
    deletionMean: floats(raw("deletion_mean")),
    msa: ints(raw("msa")), msaMask: floats(raw("msa_mask")),
    // 🔴 THE ROW COUNT IS A SHAPE, AND THE DUMP IS THE ONLY PLACE IT IS WRITTEN
    // DOWN. AF3 pads its msa array to the crop size and records how many rows
    // are real in `numMsa`, so deriving it from the array's length reads the
    // padding as alignment. Absent, the trunk's shader was built with
    // `const SEQUENCES: u32 = undefinedu;` and the fold died in WGSL parsing -
    // which is a shape bug wearing a compiler error.
    sequences: dump.numMsa ?? 1,
    // ...and the chain identity, which the confidence head's ipTM reduction
    // indexes directly. Absent, it threw inside reduceTmScore AFTER the whole
    // fold had run, which is the most expensive place to discover a missing
    // field.
    asymId: ints(raw("asym_id")),
    deletionMatrix: floats(raw("deletion_matrix")),
    seqMask: floats(raw("seq_mask")),
    refPos: floats(raw("ref_pos")), refMask,
    refElement: ints(raw("ref_element")), refCharge: floats(raw("ref_charge")),
    refAtomNameChars: ints(raw("ref_atom_name_chars")),
    refSpaceUid: ints(raw("ref_space_uid")),
    predDenseAtomMask: floats(raw("pred_dense_atom_mask")),
    tokenAtomsToQueries: gather("token_atoms_to_queries"),
    queriesToKeys: gather("queries_to_keys"),
    queriesToTokenAtoms: gather("queries_to_token_atoms"),
    tokensToQueries: gather("tokens_to_queries"),
    tokensToKeys: gather("tokens_to_keys"),
    tokenAtomsToPseudoBeta: gather("token_atoms_to_pseudo_beta"),
    features: {
      residueIndex: ints(raw("residue_index")), tokenIndex: ints(raw("token_index")),
      asymId: ints(raw("asym_id")), entityId: ints(raw("entity_id")),
      symId: ints(raw("sym_id")),
    },
  };
}

export async function main(device, args) {
  const dumpPath = option(args, "dump", "/oracle-dumps/af3-6mrr.json");
  const steps = Number(option(args, "steps", "50"));
  // Named here rather than inline at the fold, because the guard below reads it.
  const samplerMode = option(args, "mode", "diffusion");
  const blocks = Number(option(args, "blocks", "48"));
  const sequenceArg = option(args, "sequence", "");
  // 🔴 A LIGAND IS THE CASE THE CONTACT MAP'S THRESHOLD IS ABOUT, and until
  // this flag existed there was no way to fold one through AF3 from a shell -
  // so the ligand branches of the featuriser and the distogram head had unit
  // tests and no end-to-end run. Codes are comma-separated CCD names and their
  // geometry comes from the dictionary, over the network. `--kinds` names each
  // colon-joined chain, since the letters cannot say: A, C and G are alanine,
  // cysteine and glycine in a protein and adenine, cytosine and guanine in a
  // nucleic one.
  const ligandCodes = option(args, "ligands", "").split(",").filter((c) => c !== "");
  const chainKinds = option(args, "kinds", "");

  // 🔴 AF3's DIFFUSION SAMPLER NEEDS ITS WHOLE SCHEDULE, AND STOPPING EARLY
  // LOOKS EXACTLY LIKE A BROKEN MODEL. `--steps` sets the discretisation, not a
  // budget: eight steps of the stochastic sampler leaves the walk at high noise
  // and prints an N-CA of 27 A next to an ideal of 1.46, which reads as
  // corrupted weights rather than as the wrong flag. The flow reaches a
  // structure in eight because it is a different walk - see
  // src/af3/diffusion-sampler-webgpu.js - so say so rather than let the
  // geometry report take the blame.
  if (samplerMode === "diffusion" && steps < 50) {
    console.log(`🔴 ${steps} steps of the DIFFUSION sampler will not converge -`
      + " it wants about 200. Add --mode=flow to get a structure in this many,"
      + " or raise --steps. What follows is expected to look like noise.");
  }

  const dump = sequenceArg === "" || args.some((a) => a.startsWith("--dump="))
    ? await (async () => {
        const response = await fetch(dumpPath);
        if (!response.ok) throw new Error(`failed to load ${dumpPath}: ${response.status}`);
        return response.json();
      })()
    : null;

  // 🔴 AN MSA, WHICH THE BROWSER PATH HAS AND THIS DID NOT. --a3m takes one
  // path per CHAIN, comma-separated, and --paired-a3m the same for the paired
  // block; both are merged exactly as web/app.js merges them for AF3, so a
  // fold run here is the fold the page runs. Without this the CLI could only
  // ever reproduce a single-sequence prediction, which is not the case worth
  // debugging on a complex.
  const fetchText = async (path) => {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`failed to load ${path}: ${response.status}`);
    return response.text();
  };
  const chainTexts = async (spec) => (spec === ""
    ? null
    : Promise.all(spec.split(",").map((path) => fetchText(path.trim()))));
  const unpairedTexts = await chainTexts(option(args, "a3m", ""));
  const pairedTexts = await chainTexts(option(args, "paired-a3m", ""));
  const mergeFor = (texts) => (texts === null ? null
    : (texts.length === 1 ? texts[0] : mergeRowAlignedChainA3ms(texts)));
  const rows = unpairedTexts === null && pairedTexts === null
    ? { msa: [], deletionMatrix: [], depth: 1, unpairedFrom: 0 }
    : af3MsaFromA3m({ paired: mergeFor(pairedTexts), unpaired: mergeFor(unpairedTexts) },
                    { maxSequences: Number(option(args, "max-msa", "512")) });

  const ligands = [];
  for (const code of ligandCodes) {
    ligands.push(parseCcdComponent(await (await fetch(ccdUrl(code))).text()));
  }
  const batch = sequenceArg !== ""
    ? featuriseProtein(sequenceArg,
      { msa: rows.msa, deletionMatrix: rows.deletionMatrix, unpairedFrom: rows.unpairedFrom,
        ...(ligands.length === 0 ? {} : { ligands }),
        ...(chainKinds === "" ? {} : { chainKinds: chainKinds.split(",") }) })
    : batchFromDump(dump);
  if (rows.depth > 1) {
    console.log(`MSA ${rows.depth} rows, unpaired block starts at ${rows.unpairedFrom}`);
  }
  if (batch.sequences !== rows.depth) {
    console.log(`🔴 the batch carries ${batch.sequences} MSA rows, not ${rows.depth}`);
  }
  console.log(`${batch.sequence.length} residues, ${batch.tokens} tokens,`
    + ` ${batch.atomCount} atoms, ${batch.subsets} atom subsets,`
    + ` ${blocks} pairformer blocks, ${steps} diffusion steps`);
  console.log(sequenceArg !== ""
    ? "featurised in JavaScript from the sequence"
    : "featurised by AF3, read from the dump");
  // 🔴 A CENSUS, BECAUSE "IT FOLDED" DOES NOT SAY THE LIGAND WAS SEEN AS ONE.
  // A ligand atom and an unknown residue share an aatype, so a class derived
  // from the alphabet alone would call all 31 of ATP's tokens protein - and
  // the fold would still come out, with a protein's contact threshold on every
  // pair. Printed only when there is something to say.
  const classes = af3ContactClasses(batch, batch.tokens);
  const ligandTokens = classes.filter((c) => c === CLASS_LIGAND).length;
  const nucleicTokens = classes.filter((c) => c === CLASS_NUCLEIC).length;
  if (ligandTokens > 0 || nucleicTokens > 0) {
    console.log(`contact classes: ${batch.tokens - ligandTokens - nucleicTokens}`
      + ` polymer, ${ligandTokens} ligand, ${nucleicTokens} nucleic`);
  }

  // --quant=int5:g32:asym[:search] round-trips every learned weight through a
  // storage precision before the fold, so the cost is measured in ANGSTROMS
  // rather than in weight error. --model points at an exported directory
  // instead; the int5 one is packed on disk rather than round-tripped at load.
  const quantSpec = option(args, "quant", "");
  const quant = quantSpec === "" ? null : (() => {
    const [bitsField, groupField, mode, search] = quantSpec.split(":");
    return { bits: Number(bitsField.replace("int", "")),
             group: Number(groupField.replace("g", "")),
             mode: mode ?? "asym", search: search === "search" };
  })();
  if (quant) {
    console.log(`quantised: int${quant.bits} group ${quant.group} ${quant.mode}`
      + `${quant.search ? " with range search" : ""}`
      + `   ${(quant.bits + (quant.mode === "sym" ? 16 : 32) / quant.group).toFixed(2)}`
      + ` bits/weight`);
  }

  // --budget=<MiB> puts a ceiling on the device, which is how a machine too
  // small to keep the weights resident behaves. It is the only way that
  // fallback gets exercised on a Mac.
  const budgetMiB = Number(option(args, "budget", "0"));
  if (budgetMiB > 0) setMemoryBudget(device, budgetMiB * 1024 * 1024);
  const store = await openAf3Store(option(args, "model", "/model-af3-full-f32/manifest.json"),
                                   quant);
  // 🔴 EVERY SHARD AT ONCE, WHICH IS WHAT THE PAGE DOES. `prefetch` is opt-in
  // because a bench that reads four blocks should not pull the whole manifest -
  // but this tool loads a whole model, so a run without it measures a download
  // pattern no user has: shards arrive as tensors are asked for, which leaves
  // most of the connection idle most of the time. Measured on the ESMFold2
  // tool, which had the same hole: a fold 2.25 s -> 1.75.
  store.prefetch();
  // 🔴 THE PAIRFORMER'S SHADERS, WHILE THE SHARDS ARE STILL ARRIVING. A fold's
  // compilation is 0.80 s of AF3's 2.47 and 1.23 of OpenDDE's 2.93, the
  // compiler pool is saturated while it runs, and the weight load in front of
  // it leaves that pool completely idle. Not awaited; see warmTrunkPipelines.
  // `--no-warm` is the arm; see fold-opendde.js.
  if (!args.includes("--no-warm")) {
    void warmTrunkPipelines(device, store, batch.tokens).catch(() => {});
  }
  const weights = {
    trunk: await trunkWeights(store, blocks, 4),
    diffusion: await diffusionWeights(store),
    confidence: await confidenceWeights(store),
    atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };

  // 🔴 --ablate TURNS A DIALECT BRANCH OFF, WHICH IS THE ONLY HONEST WAY TO
  // PRICE ONE. Each of them is silent when wrong: the shapes agree, the fold
  // finishes, and what comes out is a slightly different model. Folding a
  // ported bundle with one convention removed and scoring both against a
  // crystal is the measurement that says whether the convention was worth
  // having - and, if removing it IMPROVES the fold, that it was wrong.
  //
  //     --ablate=maskPaddedKeys              one branch off
  //     --ablate=maskPaddedKeys,symmetriseBonds
  //
  // 🔴 padSingleCondUnknownDna CANNOT BE ABLATED HERE and nothing needs to
  // pretend otherwise: it changes the LayerNorm's WIDTH, so the bundle's own
  // 833-long scale no longer matches and the conditioning throws. That is the
  // structural gate doing its job, and it is why that branch needs no
  // measurement to be trusted.
  //
  // 🔴 AND --enable IS THE OTHER DIRECTION, WHICH IS NOT SYMMETRY FOR ITS OWN
  // SAKE. `swapTransposedBias` is false for OpenBind on the strength of
  // upstream's TRANSPOSED_COLUMN_PAIR_BIAS listing openfold3 and not openbind -
  // a reading of somebody else's table. Turning it ON and scoring the fold is
  // how that reading gets checked against the weights themselves.
  const ablate = option(args, "ablate", "").split(",").filter(Boolean);
  const enable = option(args, "enable", "").split(",").filter(Boolean);
  if (ablate.length + enable.length > 0) {
    const known = Object.keys(weights.trunk.dialect);
    for (const flag of [...ablate, ...enable]) {
      if (!known.includes(flag)) {
        throw new Error(`--ablate/--enable names ${flag}, which is not a dialect `
          + `flag; known: ${known.join(", ")}`);
      }
    }
    const overrides = {
      ...Object.fromEntries(ablate.map((flag) => [flag, false])),
      ...Object.fromEntries(enable.map((flag) => [flag, true])),
    };
    for (const bundle of [weights.trunk, weights.diffusion, weights.confidence,
                          weights.targetFeat]) {
      bundle.dialect = { ...bundle.dialect, ...overrides };
    }
    console.log(`dialect overridden: ${JSON.stringify(overrides)}`);
  }

  // 🔴 A REPEAT FOLD IS THE ONE THE PAGE ACTUALLY SHOWS AFTER THE FIRST. The
  // pipelines and the resident f16 weights are cached for the life of the
  // DEVICE, so a second fold in the same session pays neither - and this file
  // measures a cold process, which is the slowest fold there is. `--folds=2`
  // runs it twice and reports both, so the two can be told apart.
  const folds = Number(option(args, "folds", "1"));
  const keepTrajectory = option(args, "trajectory", "on") !== "off";
  // 🔴 SO THE SPLIT PATH'S COMPILATION COST CAN BE SEEN. Splitting K adds three
  // pipelines a block, and pipeline compilation is most of a cold fold's
  // one-time cost - so the arm that makes a warm fold faster could make a cold
  // one slower, and only a cold/warm pair says which.
  if (option(args, "splitk", null) === "off") {
    setDeviceTuning(device, { diffusionSplitK: null });
  }
  // 🔴 THE CONTROL ARM FOR keepTrunkWeights. On a device whose prior sets it,
  // --keep-weights=off restores the shipped behaviour (give the trunk's
  // weights back every fold) so the two can be measured in one build.
  // 🔴 AND `=on` IS THE ARM A DEVICE WITHOUT THE PRIOR NEEDS. keepTrunkWeights
  // is null everywhere it has not been measured, so on an M2 the only way to
  // find out what it is worth is to force it. The trade is 561 MiB of re-upload
  // and re-PACKING against ~379 MiB of resident memory, and only the upload
  // half is bus - on unified memory the bus is free (probe-bus.js) but the
  // packing is not, so the answer there is not predictable from here.
  // 🔴 FORCE ANY TUNING KNOB, BECAUSE A KNOB NO GATE ENTERS IS A KNOB NOBODY
  // HAS CHECKED. `normSplits` was reachable, wrong and invisible for exactly
  // that reason - the prior sets it to 1, so a whole-fold gate never took the
  // path. Auditing the rest needed a bespoke flag per knob; this is the general
  // one. `--tune=key=value,key=value`, values parsed as JSON so numbers,
  // booleans, null and objects all work:
  //
  //     --tune=singleProjectLanes=128
  //     --tune=attentionQueriesPerLane=2,diffusionLanes=64
  //
  // The test is degeneracy: a knob that only changes HOW something is computed
  // must leave pLDDT and pTM bit-identical.
  // 🔴 REPLAY A REAL DENOISER STEP'S INPUTS THROUGH BOTH SPLIT SETTINGS.
  // check-difftx-splits.js drives the transformer with synthesised noise and
  // passes every one of 106 arms, including the fold's EXACT 24-field shape,
  // while a fold with normSplits=2 diverges on its FIRST denoiser step at
  // relRMS 0.538. The one difference left is the tensor VALUES. This wraps the
  // transformer, lets the fold's first (unchained, host-array) call through,
  // and then re-runs that call's real inputs at normKSplits 1 and 2 on fresh
  // instances. If they differ here it is the data; if they agree, the fault is
  // outside the transformer and the head's plumbing is next.
  if (args.includes("--replay-difftx")) {
    const proto = Af3DiffusionTransformerGpu.prototype;
    const original = proto.run;
    let captured = false;
    proto.run = async function replaying(act, cond, pairCond, mask, tokens, w, opts) {
      const result = await original.call(this, act, cond, pairCond, mask, tokens, w, opts);
      if (!captured && act instanceof Float32Array && cond instanceof Float32Array) {
        captured = true;
        const stat = (v) => {
          let lo = Infinity; let hi = -Infinity; let sum = 0;
          for (const x of v) { if (x < lo) lo = x; if (x > hi) hi = x; sum += x; }
          return { min: Number(lo.toPrecision(4)), max: Number(hi.toPrecision(4)),
                   mean: Number((sum / v.length).toPrecision(4)) };
        };
        const rel = (rawX, rawY) => {
          const x = rawX?.output ?? rawX;
          const y = rawY?.output ?? rawY;
          if (!(x instanceof Float32Array) || !(y instanceof Float32Array)) return NaN;
          let n = 0; let d = 0;
          for (let i = 0; i < x.length; i += 1) { n += (x[i] - y[i]) ** 2; d += y[i] ** 2; }
          return Math.sqrt(n / Math.max(d, 1e-30));
        };
        const at = async (normKSplits) => original.call(
          new Af3DiffusionTransformerGpu(device), act, cond, pairCond, mask, tokens,
          { ...w, normKSplits }, {});
        const one = await at(1);
        const two = await at(2);
        // 🔴 A CHECKSUM, NOT min/max/mean. Those matched across both arms to
        // four figures while the outputs differed, which is suggestive and not
        // proof - two different arrays can share all three.
        // 🔴 UNWRAP. run() resolves to {output, ...}; feeding the object to a
        // length-indexed loop silently compared nothing and reported 0.
        const sums = (raw) => {
          const v = raw?.output ?? raw;
          if (!(v instanceof Float32Array)) return `NOT-AN-ARRAY(${typeof v})`;
          let sum = 0;
          let sumsq = 0;
          for (const x of v) { sum += x; sumsq += x * x; }
          return `sum=${sum.toPrecision(12)} sumsq=${sumsq.toPrecision(12)}`;
        };
        console.log(`REPLAY IN act      ${sums(act)}`);
        console.log(`REPLAY IN cond     ${sums(cond)}`);
        console.log(`REPLAY IN pairCond ${sums(pairCond)}`);
        // 🔴 AND THE FOLD'S OWN CONFIGURATION ON A FRESH INSTANCE. Not
        // `{...w, normKSplits}` - `w` itself, exactly what the fold passed. If
        // this reproduces the fold's own output then instance state is
        // irrelevant; if it produces the OTHER arm's output, the fold's
        // instance is the whole difference.
        const asFold = await original.call(
          new Af3DiffusionTransformerGpu(device), act, cond, pairCond, mask, tokens, w, {});
        const asFoldOut = asFold?.output ?? asFold;
        if (asFoldOut instanceof Float32Array) {
          console.log(`REPLAY fresh instance, the FOLD'S OWN weights: ${sums(asFoldOut)}`);
        }
        console.log(`REPLAY act ${JSON.stringify(stat(act))}`);
        console.log(`REPLAY cond ${JSON.stringify(stat(cond))}`);
        console.log(`REPLAY pairCond ${JSON.stringify(stat(pairCond))}`);
        console.log(`REPLAY tokens=${tokens} maskAllOnes=${mask.every((m) => m === 1)}`);
        // 🔴 ABSOLUTE, NOT RELATIVE. Comparing at(1) to at(2) inside one process
        // says they agree with EACH OTHER and nothing about whether either is
        // right - and in the arm where the device rule is normSplits=2 they can
        // both be wrong together, which is exactly what relRMS 0 would look
        // like. Checksums are comparable ACROSS processes; a relRMS is not.
        console.log(`REPLAY at(normKSplits=1) ${sums(one)}`);
        console.log(`REPLAY at(normKSplits=2) ${sums(two)}`);
        console.log(`REPLAY normKSplits 2 vs 1 on the FOLD'S OWN INPUTS:`
          + ` relRMS ${rel(two, one).toExponential(3)}`);
        // 🔴 AND THE FOLD'S OWN OUTPUT, WHICH IS THE THING THAT DIVERGES. The
        // replay above uses a FRESH instance; the fold uses one that warm()
        // touched during the trunk and that carries a compile memo, a scratch
        // cache and a bind-group cache. If this checksum differs between arms
        // while the replay does not, the fault is that instance state and not
        // the shaders, the shape or the data - all three of which are now
        // eliminated. Compared across two runs, whose inputs are identical
        // because the seed and the trunk are.
        const out = result?.output ?? result;
        if (out instanceof Float32Array) {
          let sum = 0;
          let sumsq = 0;
          for (const v of out) { sum += v; sumsq += v * v; }
          console.log(`REPLAY fold's own transformer output:`
            + ` n=${out.length} sum=${sum.toPrecision(12)} sumsq=${sumsq.toPrecision(12)}`);
        } else {
          console.log(`REPLAY fold's own output is not a host array (${typeof out})`);
        }
      }
      return result;
    };
  }
  const tune = option(args, "tune", "");
  if (tune !== "") {
    const forced = {};
    for (const pair of tune.split(",").filter(Boolean)) {
      const at = pair.indexOf("=");
      if (at < 0) throw new Error(`--tune wants key=value, got ${pair}`);
      const key = pair.slice(0, at);
      const raw = pair.slice(at + 1);
      if (!(key in DEFAULT_TUNING)) {
        throw new Error(`--tune names ${key}, which is not a tuning knob. `
          + `Known: ${Object.keys(DEFAULT_TUNING).sort().join(", ")}`);
      }
      try { forced[key] = JSON.parse(raw); } catch { forced[key] = raw; }
    }
    setDeviceTuning(device, forced);
    console.log(`tuning forced: ${JSON.stringify(forced)}`);
  }
  const keepWeights = option(args, "keep-weights", null);
  if (keepWeights === "off") setDeviceTuning(device, { keepTrunkWeights: null });
  if (keepWeights === "on") setDeviceTuning(device, { keepTrunkWeights: true });
  const keepSampler = option(args, "keep-sampler-weights", null);
  if (keepSampler === "off") setDeviceTuning(device, { keepSamplerWeights: null });
  if (keepSampler === "on") setDeviceTuning(device, { keepSamplerWeights: true });
  // 🔴 THE TWO SPLIT COUNTS THE PRIOR LEAVES AT ONE. attention-output, adaln and
  // ffw-adaln are the three largest labels in a denoiser step and none of them
  // is split; the prior sets attnSplits and normSplits to 1. Those were chosen
  // while the crossover logic was still being debugged and against a target
  // (33x on the diffusion) that turned out to be wrong by 17x, so they are
  // worth re-sweeping. These patch the device's rule in place.
  const attnSplits = option(args, "attn-splits", null);
  const normSplits = option(args, "norm-splits", null);
  // 🔴 THE PER-KERNEL TOKEN TILES, which --tune cannot reach: they live inside
  // the diffusionSplitK object and --tune splits its argument on commas. A tile
  // only moves when the matching K split is on - the split is what pays for it.
  const attnTile = option(args, "attn-tile", null);
  const outTileArg = option(args, "out-tile", null);
  // 🔴 THE CROSSOVER, so the 175 can be re-asked. It was set from a measurement
  // at 240 tokens - "the step is 46.63 ms against 46.68 before, unchanged" -
  // taken before the conditioning hoist and the per-kernel tiles existed, and
  // above it EVERY split disengages at once.
  const crossover = option(args, "crossover", null);
  // 🔴 AND THE WHOLE RULE AS JSON, because the two knobs that carry most of it
  // - `splits` and `tile` - had no flag at all, and the patch below needs a
  // rule to already exist so `--no-prior` could not reach any of them. Pricing
  // what a device with no prior is missing needs exactly that combination.
  //     --split-k='{"splits":16,"tile":4,"crossover":512,"outSplits":4,
  //                 "attnSplits":4,"attnTile":2,"normSplits":4}'
  //     --split-k=off
  const splitKArg = option(args, "split-k", null);
  if (splitKArg !== null) {
    setDeviceTuning(device, { diffusionSplitK:
      splitKArg === "off" ? null : JSON.parse(splitKArg) });
  }
  if (attnSplits !== null || normSplits !== null
      || attnTile !== null || outTileArg !== null || crossover !== null) {
    const rule = deviceTuning(device).diffusionSplitK;
    if (rule === null || rule === undefined) {
      throw new Error("--attn-splits/--norm-splits/--attn-tile/--out-tile need "
        + "a device with a diffusionSplitK rule");
    }
    setDeviceTuning(device, { diffusionSplitK: { ...rule,
      ...(attnSplits === null ? {} : { attnSplits: Number(attnSplits) }),
      ...(normSplits === null ? {} : { normSplits: Number(normSplits) }),
      ...(attnTile === null ? {} : { attnTile: Number(attnTile) }),
      ...(outTileArg === null ? {} : { outTile: Number(outTileArg) }),
      ...(crossover === null ? {} : { crossover: Number(crossover) }) } });
  }
  const window = option(args, "submission-window", null);
  if (window !== null) {
    setDeviceTuning(device, { pairformerSubmissionWindow: Number(window) });
  }
  // 🔴 --profile TIMES THE WHOLE FOLD, WHICH NOTHING ELSE DID. bench-trunk,
  // bench-head and bench-confidence each profile ONE stage against synthesised
  // inputs, so each says where its own time goes and none of them says what
  // share of a fold that stage is. The confidence head in particular had never
  // been placed against the trunk it follows.
  //
  // 🔴 IT IS RESET AT THE LAST FOLD, so `--folds=2 --profile` profiles the
  // WARM one - the pipelines and the resident weights are cached for the life
  // of the device, and a cold process is not the fold the page shows twice.
  const profile = args.includes("--profile") || args.includes("--profile-batched")
    ? profileDevice(device, { batched: args.includes("--profile-batched") })
    : null;
  // 🔴 --buffers ANSWERS THE QUESTION --profile CANNOT. profile.js wraps
  // beginComputePass, which at 68 tokens is 10% of a fold; the other 90% is
  // byte-proportional work outside every compute pass (forcing f32 adds 2.87 s
  // of wall and 4 ms of compute). This times createBuffer, writeBuffer,
  // copyBufferToBuffer, mapAsync and submit, so the remainder gets a name.
  const buffers = args.includes("--buffers") ? profileBuffers(device) : null;
  // 🔴 A WHOLE FOLD DOES NOT FIT IN ONE PROFILE, and the device says so: the
  // query set is capped at 4096 timestamps, which is 2048 passes, and a
  // 200-token trunk alone uses all of them. `--profile-from=<stage>` resets at
  // a stage boundary so the half that was being dropped can be read on its
  // own - `--profile-from=trunk-done` profiles the sampler and the confidence
  // head, which is the half nothing had ever timed inside a real fold.
  const profileFrom = option(args, "profile-from", "");
  const foldSeconds = [];
  let result;
  let trunkStarted = 0;
  let stageAt = 0;
  let lastStage = "";
  const stageMilliseconds = {};
  let diffusionStarted = 0;
  const trajectory = [];
  let lastDenoised = null;
  for (let attempt = 0; attempt < folds; attempt += 1) {
  if (profile !== null && attempt === folds - 1) profile.reset();
  if (buffers !== null && attempt === folds - 1) buffers.reset();
  const started = performance.now();
  trajectory.length = 0;
  lastDenoised = null;

  result = await foldBatch(device, batch, weights, {
    // Omitted, each defaults to what the device supports. See docs/AF3.md.
    stagedPrecision: option(args, "staged", undefined),
    weightPrecision: option(args, "weights", undefined),
    pairWeightPrecision: option(args, "pair-weights", undefined),
    accumulatePrecision: option(args, "accumulate", undefined),
    mode: samplerMode,
    recycles: Number(option(args, "recycles", "0")),
    // 🔴 THE ONLY EARLY STOP A TRUNK-ONLY RECYCLE CAN HAVE. 0 is off; see
    // src/model/feature-convergence.js and docs/AF3.md for what a number here
    // has been measured to mean, which is two inputs' worth and not a corpus.
    recycleTolerance: Number(option(args, "recycle-tolerance", "0")),
    steps, stopAfter: Number(option(args, "truncate", String(steps))),
    seed: Number(option(args, "seed", "20260831")),
    onStage: (name, detail) => {
      // 🔴 EVERY STAGE'S OWN MILLISECONDS, WHICH THIS TOOL PRINTED FOR THE
      // TRUNK AND NOTHING ELSE. A caller's clock attributes the gap between two
      // stages to the EARLIER one, so a fold whose named stages stop at
      // `trunk-done` hides everything after it - which for AF3 at 68 tokens is
      // 2.0 s of a 3.2 s first fold. fold-opendde.js has reported this since
      // the same lesson was learned there.
      const now = performance.now();
      if (stageAt !== 0) {
        stageMilliseconds[lastStage] = (stageMilliseconds[lastStage] ?? 0) + (now - stageAt);
      }
      stageAt = now;
      lastStage = name === "trunk" ? `trunk:${detail.name}` : name;
      if (profile !== null && name === profileFrom && attempt === folds - 1) profile.reset();
      if (name === "target-feat") {
        const theirs = dump?.outputs["diffuser/evoformer/__call__:target_feat"];
        if (theirs) {
          console.log(`target_feat vs AF3  relRMS`
            + ` ${relativeRms(detail.targetFeat, floats(theirs.data)).toExponential(2)}`);
        }
        trunkStarted = performance.now();
      }
      if (name === "trunk") {
        const gpu = memorySnapshot(device);
        console.log(`  ${detail.name.padEnd(12)} ${detail.ms.toFixed(0)} ms`
          + `   gpu ${(gpu.residentBytes / (1024 * 1024)).toFixed(0)} MiB`
          + ` (peak ${(gpu.peakBytes / (1024 * 1024)).toFixed(0)})`);
      }
      if (name === "trunk-done") {
        const gpu = memorySnapshot(device);
        console.log(`trunk done in ${((performance.now() - trunkStarted) / 1000).toFixed(1)} s`
          + `   gpu ${(gpu.residentBytes / (1024 * 1024)).toFixed(0)} MiB`
          + ` (peak ${(gpu.peakBytes / (1024 * 1024)).toFixed(0)})`);
        // 🔴 WHAT IS STILL HELD WHEN THE TRUNK IS DONE. The peak composition
        // shows the pairformer's scratch and the diffusion transformer's
        // resident weights in the same snapshot, though one finishes before
        // the other starts - so either the scratch is not given back, or it is
        // given back to a POOL that goes on holding it. This says which.
        for (const row of gpu.currentByLabel.slice(0, 6)) {
          console.log(`    held ${(row.bytes / (1024 * 1024)).toFixed(1).padStart(8)} MiB`
            + ` x${String(row.count).padEnd(3)} ${row.label}`);
        }
        console.log(`  peak so far ${(gpu.peakBytes / (1024 * 1024)).toFixed(1)} MiB, made of:`);
        for (const row of gpu.peakByLabel.slice(0, 6)) {
          console.log(`    peak ${(row.bytes / (1024 * 1024)).toFixed(1).padStart(8)} MiB`
            + ` x${String(row.count).padEnd(3)} ${row.label}`);
        }
        // Against AF3's own trunk. Only meaningful on AF3's own batch: from a
        // sequence the reference conformers differ, which is worth about
        // 2.7e-2 on pair and 0.01 A of structure.
        // 🔴 A RECYCLED DUMP NAMES ITS CAPTURES PER PASS. With --recycles the
        // evoformer is called once per pass and each capture gets a `#n`, so
        // the bare name matches nothing and the comparison silently does not
        // run - the report simply omits the two lines it exists to print. The
        // LAST pass is the one whose output reaches the diffusion head.
        const lastCapture = (base) => dump?.outputs[base]
          ?? Object.keys(dump?.outputs ?? {})
            .filter((key) => key.startsWith(`${base}#`))
            .sort((a, b) => Number(a.split("#")[1]) - Number(b.split("#")[1]))
            .map((key) => dump.outputs[key]).pop();
        const pair = lastCapture("diffuser/evoformer/__call__:pair");
        const single = lastCapture("diffuser/evoformer/__call__:single");
        if (pair && blocks === 48) {
          console.log(`pair   vs AF3  relRMS`
            + ` ${relativeRms(detail.trunk.pair, floats(pair.data)).toExponential(2)}`);
          console.log(`single vs AF3  relRMS`
            + ` ${relativeRms(detail.trunk.single, floats(single.data)).toExponential(2)}`);
        }
        diffusionStarted = performance.now();
      }
    },
    onStep: ({ step, noiseLevel, denoised, positions }) => {
      lastDenoised = denoised;
      // 🔴 THE TRAJECTORY IS THE TOOL, AND IT IS INSIDE THE TIMED REGION.
      // `foldSeconds` wraps foldBatch, callbacks included, so the two
      // Array.from calls below are charged to the model. --trajectory=off
      // turns them off, which is how a fold's number is separated from this
      // file's contribution to it.
      if (!keepTrajectory) return;
      // Every frame for a short run, every fourth for a long one - the whole
      // trajectory at 200 steps is 200 * 68 * 24 * 3 floats.
      if (steps <= 60 || step % 4 === 0 || step === steps) {
        // 🔴 BOTH TRACKS. `denoised` is the model's running guess and it is
        // already within about 1 A of the final structure at step ONE - the
        // trunk decides the fold and diffusion refines it. `positions` is the
        // actual trajectory, which starts as a cloud thousands of angstroms
        // across. Only the second one looks like folding.
        trajectory.push({ step, noiseLevel,
                          denoised: Array.from(denoised),
                          positions: Array.from(positions) });
      }
      if (step === 1 || step % Math.ceil(steps / 5) === 0 || step === steps) {
        console.log(`  step ${String(step).padStart(3)}/${steps}  sigma`
          + ` ${noiseLevel.toFixed(2).padStart(9)}`);
      }
    },
  });
  const elapsed = (performance.now() - started) / 1000;
  foldSeconds.push(Number(elapsed.toFixed(3)));
  if (folds > 1) console.log(`fold ${attempt + 1} of ${folds}: ${elapsed.toFixed(1)} s`);
  }

  console.log(`diffusion done in ${((performance.now() - diffusionStarted) / 1000).toFixed(1)} s`);
  console.log(`mean pLDDT ${result.meanPlddt.toFixed(1)} over ${result.atoms} atoms`
    + `   pTM ${result.ptm.toFixed(3)}`
    + `   ipTM ${Number.isNaN(result.iptm) ? "n/a (one chain)" : result.iptm.toFixed(3)}`);

  // 🔴 GEOMETRY IS THE CHECK THAT MATTERS HERE, not pLDDT - see the note on
  // backboneGeometry.
  const { nca, cac, caca, worstCaca, gyration, residues } = result.geometry;
  console.log(`backbone  N-CA ${nca.toFixed(2)} A (ideal 1.46)`
    + `   CA-C ${cac.toFixed(2)} A (ideal 1.52)`
    + `   CA-CA ${caca.toFixed(2)} A (ideal 3.80, worst ${worstCaca.toFixed(2)})`);
  console.log(`radius of gyration ${gyration.toFixed(1)} A over ${residues} CA`
    + `   (a compact 68-mer is about 11-12 A)`);
  // 🔴 AND NOW IT IS A GATE. The comment above has said "geometry is the check
  // that matters here" since this tool was written, and nothing failed on it -
  // which is exactly the state fold-af2.js was in when an 825-residue collapse
  // walked through it for a whole campaign. See tools/gpu/chain-geometry.js.
  assertChainGeometry(result.geometry, {
    plddt: result.meanPlddt, doc: "docs/AF3.md and docs/AF2.md",
    allow: args.includes("--allow-broken-geometry"),
  });
  console.log(`total ${foldSeconds[foldSeconds.length - 1].toFixed(1)} s`);
  // 🔴 GROUPED BY THE LABEL'S FIRST WORD, because that is the STAGE. Every
  // pass here is labelled `<stage>.<pass>` - af3-block, difftx, atom, cond,
  // conf - so the prefix answers "which stage is this fold" and the rows under
  // it answer "which kernel in it". The pass count is printed beside each,
  // because Chrome quantises timestamps to ~100 us: a label with one pass is
  // not a measurement, a label with hundreds is.
  if (profile !== null) {
    const rows = await profile.report();
    const stages = new Map();
    let measured = 0;
    for (const row of rows) {
      const stage = row.label.split(".")[0];
      const found = stages.get(stage) ?? { stage, ms: 0, passes: 0 };
      found.ms += row.ms; found.passes += row.passes;
      stages.set(stage, found);
      measured += row.ms;
    }
    const wall = foldSeconds[foldSeconds.length - 1] * 1000;
    const dropped = profile.dropped();
    console.log(`profile${profileFrom === "" ? "" : ` from ${profileFrom}`}:`
      + ` ${measured.toFixed(0)} ms in ${rows.length} labels`
      + ` over ${rows.reduce((n, r) => n + r.passes, 0)} passes`
      + `, of ${wall.toFixed(0)} ms wall (${(100 * measured / wall).toFixed(0)}%)`);
    // 🔴 SAY SO WHEN IT IS A PREFIX. Without this the trunk fills the query set
    // and the sampler reads as free - which is what it did.
    if (dropped > 0) {
      console.log(`  🔴 TRUNCATED: ${dropped} passes found no slot`
        + ` (the device caps this at ${profile.capacityPasses}).`
        + ` These totals are a PREFIX of the fold, not the fold.`
        + ` Use --profile-from=<stage> to profile a later part on its own.`);
    }
    for (const row of [...stages.values()].sort((a, b) => b.ms - a.ms)) {
      console.log(`  ${row.ms.toFixed(0).padStart(6)} ms`
        + ` ${(100 * row.ms / measured).toFixed(1).padStart(5)}%`
        + ` x${String(row.passes).padEnd(5)} ${row.stage}`);
    }
    console.log("  the ten costliest passes:");
    for (const row of rows.slice(0, 10)) {
      console.log(`    ${row.ms.toFixed(0).padStart(6)} ms x${String(row.passes).padEnd(5)} ${row.label}`);
    }
    profile.restore();
  }

  if (buffers !== null) {
    const traffic = buffers.report();
    const wall = foldSeconds[foldSeconds.length - 1] * 1000;
    console.log(`buffer traffic: ${traffic.totalMs.toFixed(0)} ms of ${wall.toFixed(0)} ms wall`
      + ` (${(100 * traffic.totalMs / wall).toFixed(0)}%)`);
    // 🔴 REPORTED APART FROM THE SUM BECAUSE IT IS A UNION, NOT AN ADDEND.
    // Dozens of onSubmittedWorkDone promises are outstanding at once, so this
    // is how much of the fold had at least one wait pending - overlapping the
    // rows above rather than adding to them.
    console.log(`  queue drain: ${traffic.queueWait.unionMs.toFixed(0)} ms`
      + ` (${(100 * traffic.queueWait.unionMs / wall).toFixed(0)}% of wall)`
      + ` over ${traffic.queueWait.calls} onSubmittedWorkDone, union not sum`);
    for (const row of traffic.byKind) {
      console.log(`  ${row.ms.toFixed(0).padStart(6)} ms`
        + ` ${(100 * row.ms / wall).toFixed(1).padStart(5)}%`
        + ` x${String(row.calls).padEnd(6)}`
        + ` ${(row.bytes / (1024 * 1024)).toFixed(1).padStart(9)} MiB  ${row.kind}`);
    }
    console.log("  the twelve costliest, by buffer:");
    for (const row of traffic.rows.slice(0, 12)) {
      console.log(`    ${row.ms.toFixed(0).padStart(6)} ms x${String(row.calls).padEnd(6)}`
        + ` ${(row.bytes / (1024 * 1024)).toFixed(1).padStart(9)} MiB`
        + `  ${row.kind} ${row.label}`);
    }
    buffers.restore();
  }

  // 🔴 THE DENOISED PREDICTION IS NOT THE SAMPLE, and at a coarse schedule they
  // are not close. `positions` is where the sampler's walk ended; `denoised` is
  // what the model predicted on the last call. Reported side by side because
  // the difference between them IS the schedule: with many steps the walk has
  // been pulled onto the prediction and the two agree, and with few it has not.
  const denoisedGeometry = backboneGeometry(batch, lastDenoised);
  console.log(`last denoised  N-CA ${denoisedGeometry.nca.toFixed(2)}`
    + `   CA-C ${denoisedGeometry.cac.toFixed(2)}`
    + `   CA-CA ${denoisedGeometry.caca.toFixed(2)} A`
    + `   gyration ${denoisedGeometry.gyration.toFixed(1)} A`);

  // 🔴 sequence STAYS THE FIRST KEY. tools/score_fold.py finds the result in
  // this log by searching for `{\n  "sequence"`, so reordering the object
  // silently makes every score say "did the run fail?".
  return {
    // Both folds, so a cold process and a warm one can be told apart.
    foldSeconds,
    stageMilliseconds: Object.fromEntries(Object.entries(stageMilliseconds)
      .map(([k, v]) => [k, Math.round(v)]).filter(([, v]) => v > 0)
      .sort((a, b) => b[1] - a[1])),
    sequence: batch.sequence, tokens: batch.tokens, steps,
    denoisedPdb: toPdb(batch, lastDenoised, result.scores.plddt),
    // Per recycle, how far the trunk's single and pair moved from the pass
    // before. A trunk-only recycle produces no coordinates, so this is the only
    // convergence signal there is - see src/model/feature-convergence.js.
    recycleDeltas: result.recycleDeltas?.map((d) => ({
      pass: d.pass,
      pair: Number(d.pair.toExponential(3)),
      single: Number(d.single.toExponential(3)),
    })),
    meanPlddt: result.meanPlddt,
    ptm: result.ptm,
    iptm: Number.isNaN(result.iptm) ? null : result.iptm,
    geometry: { nca: { median: nca }, cac: { median: cac }, caca: { median: caca } },
    gyration, seconds: foldSeconds[foldSeconds.length - 1], pdb: result.pdb, trajectory,
    // What the device is holding at the end, which nothing else reports.
    memory: memorySnapshot(device),
  };
}
