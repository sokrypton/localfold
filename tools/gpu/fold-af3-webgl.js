/**
 * Fold AlphaFold 3 in WebGL2, from a real weight bundle.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/fold-af3-webgl.js
 *     node tools/gpu-chrome.mjs tools/gpu/fold-af3-webgl.js --blocks=48 --steps=200
 *
 * 🔴 THE DEFAULTS ARE A TRUNCATED FOLD, AND THAT IS STATED RATHER THAN HIDDEN.
 * A full AF3 fold is 48 pairformer blocks a pass and 200 denoiser calls, and
 * this backend is about 17x slower than the WGSL one - so the defaults run a
 * PREFIX of the trunk and a short schedule, which is enough to ask whether the
 * composition is right and NOT enough to ask whether the structure is good.
 * `--blocks=48 --steps=200` is the real thing, and the report says which ran.
 *
 * 🔴 AND A CHAIN THAT IS NOT A CHAIN IS THE GATE, NOT pLDDT. `fold-af2.js`
 * learned this the hard way: a collapsed 825-residue fold walked through a whole
 * campaign while pLDDT climbed to 69. So this reports the consecutive CA-CA
 * distances and FAILS when their median is far from 3.8 A - at a step count
 * where the sampler is converged - and reports pLDDT without gating on it.
 *
 * 🔴 AND IT IS THE FIRST THING IN THIS TREE TO READ AN AF3 CHECKPOINT, which is
 * a different test from every stage gate: those drive one stage with NOISE, and
 * "a packing that moved the real model 4e-4 read 6e-7 on noise". A weight read
 * at the wrong stride, a transpose, a null that should have been absent - none
 * of them are visible on uniform noise.
 *
 * `--trunk-only` stops after the trunk and reports its contact map, which is the
 * cheap arm: it costs no denoiser calls and still exercises the embedder, the
 * template stack and both stacks on real weights.
 *
 * 🔴 AND `--sequence=A:B` IS THE ARM THAT REACHES THE CHAIN SCORES. The default
 * is ONE chain, where ipTM is NaN by definition and `chainIptm`/`chainPairIptm`
 * are empty - so a single-chain run says nothing about chain_pair_iptm, chain_ptm
 * or chain_iptm, and this tool reported none of the three until now. They are
 * compared against the WGSL fold's and asserted, which is what makes this a gate
 * for the reduction and not only for the coordinates.
 *
 * 🔴 AND `--ligands=CA` IS THE ARM THAT REACHES THE FRAME MASK. Every token of a
 * protein has a frame, so a protein-only fold cannot tell a masked reduction from
 * an unmasked one however many chains it has - which is how this path kept AF3's
 * anchors wrong while agreeing with the WGSL fold to four decimals. An ion is one
 * token with one atom and no frame: it anchors nothing, its chain_ptm is absent
 * rather than a number, and `framelessTokens` says how many such tokens the fold
 * held. (`--ligands=` fetches the component from the PDB, so that arm needs the
 * network; every other arm here is local.)
 */
import { HttpTensorStore } from "../../shared/bundles/http-tensor-store.js";
import { af3BatchFromA3m } from "../../shared/af3/featurise/batch.js";
import { ccdUrl, parseCcdComponent } from "../../shared/af3/featurise/ccd-component.js";
import { af3Dialect, openAf3Store } from "../../shared/af3/weights/weights.js";
import { foldWeights } from "../../shared/af3/weights/diffusion-weights.js";
import { webglExecution } from "./webgl-execution-options.js";
import { createWebGL2Context } from "../../webgl/index.js";
import {
  foldAf3, prepareFoldPrograms, releaseAf3Weights, SAMPLER_DEFAULTS, uploadAf3Weights,
} from "../../webgl/af3/model/fold.js";
import { AF3_POOL_LIMIT_BYTES, foldMsa } from "../../webgl/af3/model/fold-batch.js";
import { DIFFUSION_DEFAULTS, foldBatch } from "../../webgpu/af3/fold.js";
import { relativeRms } from "./relative-rms.js";

const DEFAULT_SEQUENCE = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** Consecutive CA-CA distances, which is what says a chain is a chain. */
function backboneSpacing(positions, batch) {
  const { tokens, dense } = batch;
  const spacings = [];
  let previous = null;
  for (let token = 0; token < tokens; token += 1) {
    if (!(batch.seqMask[token] > 0)) continue;
    // CA is dense slot 1 for a standard residue; skip a token without one.
    const slot = token * dense + 1;
    if (!(batch.refMask[slot] > 0)) { previous = null; continue; }
    const at = [positions[slot * 3], positions[slot * 3 + 1], positions[slot * 3 + 2]];
    if (previous !== null) {
      spacings.push(Math.hypot(at[0] - previous[0], at[1] - previous[1], at[2] - previous[2]));
    }
    previous = at;
  }
  spacings.sort((a, b) => a - b);
  return spacings;
}

const round4 = (scores) => (scores === undefined ? null
  : Object.fromEntries(Object.entries(scores).map(
    ([chain, value]) => [chain, Number(value.toFixed(4))])));

/**
 * The worst disagreement in each chain-level field, and the KEYS as well as the
 * numbers: a backend reporting a chain the other does not is a difference no
 * comparison of values can show. Such a field comes back as a SENTENCE naming
 * the one-sided chains rather than as a number - 🔴 and not as `Infinity`,
 * which `JSON.stringify` writes as `null`, the same thing this function returns
 * for "there was nothing to compare". A gate whose report cannot tell a missing
 * chain from a missing comparison is a gate that reads green to a reader.
 */
function chainGaps(fields) {
  const out = {};
  for (const [name, ours, theirs] of fields) {
    if (ours === undefined || theirs === undefined) { out[name] = null; continue; }
    const keys = [...new Set([...Object.keys(ours), ...Object.keys(theirs)])];
    if (keys.length === 0) { out[name] = null; continue; }
    const oneSided = [
      ...keys.filter((key) => !(key in theirs)).map((key) => `WebGL2 only ${key}`),
      ...keys.filter((key) => !(key in ours)).map((key) => `WGSL only ${key}`),
    ];
    if (oneSided.length > 0) { out[name] = oneSided.join(", "); continue; }
    let worst = 0;
    for (const key of keys) worst = Math.max(worst, Math.abs(ours[key] - theirs[key]));
    out[name] = Number(worst.toFixed(4));
  }
  return out;
}

export async function main(device, args) {
  const bundle = option(args, "bundle", "/model-af3-int5").replace(/\/$/, "");
  const length = Number(option(args, "length", "0"));
  const sequence = option(args, "sequence",
    length > 0 ? DEFAULT_SEQUENCE.slice(0, length) : DEFAULT_SEQUENCE);
  const blocks = Number(option(args, "blocks", "4"));
  const msaBlocks = Number(option(args, "msa-blocks", "1"));
  const steps = Number(option(args, "steps", "8"));
  const recycles = Number(option(args, "recycles", "0"));
  const templates = Number(option(args, "templates", "4"));
  const seed = Number(option(args, "seed", "42"));
  const trunkOnly = flag(args, "trunk-only");
  // 🔴 `--no-hoist` RECOMPUTES THE PAIR CONDITIONING EVERY STEP, which is the
  // arm that prices the hoist. The two are the same draws on the same inputs -
  // only the single track reads the noise level - so the coordinates must not
  // move, and the gate below asserts they do not.
  const hoist = !flag(args, "no-hoist");
  const reference = !flag(args, "no-reference") && !trunkOnly;
  // 🔴 THE TWO BACKENDS' SAMPLER DEFAULTS ARE HELD TOGETHER HERE, because they
  // are stated twice - once per backend - and a fold compared across them is
  // only a comparison if both walk the same schedule with the same update.
  for (const key of Object.keys(DIFFUSION_DEFAULTS)) {
    if (SAMPLER_DEFAULTS[key] !== DIFFUSION_DEFAULTS[key]) {
      throw new Error(
        `the two backends disagree about the sampler's ${key}:`
        + ` WebGL2 ${SAMPLER_DEFAULTS[key]} against WebGPU ${DIFFUSION_DEFAULTS[key]}`);
    }
  }

  const loadStart = performance.now();
  const store = await openAf3Store(`${bundle}/manifest.json`);
  void HttpTensorStore;
  const dialect = af3Dialect(store);
  const weights = await foldWeights(store, { blocks, msaBlocks });
  const loadMs = Math.round(performance.now() - loadStart);
  if (weights.trunk.dialect.structuralTokens === true) {
    throw new Error(
      "this bundle re-tokenises after the trunk (OpenDDE's structural tokens),"
      + " which this backend has not ported - its expander, refiner and"
      + " confidence head are a second token space");
  }

  // 🔴 A LIGAND IS ONE TOKEN PER HEAVY ATOM AND AN ION IS ONE TOKEN WITH NO
  // FRAME, which is the only way to exercise the anchor mask from here.
  const ligands = [];
  for (const code of option(args, "ligands", "").split(",").filter((c) => c !== "")) {
    ligands.push(parseCcdComponent(await (await fetch(ccdUrl(code))).text()));
  }
  // `--a3m=<path>` folds the alignment's own query with the alignment, drawn a pass at a time as the
  // page draws it (`foldMsa`) - the arm that prices the MSA stack at the depth a real search gives.
  const a3mPath = option(args, "a3m", "");
  const a3m = a3mPath === "" ? null : await (await fetch(a3mPath)).text();
  const folded = a3m === null ? sequence
    : a3m.split("\n").find((line) => line !== "" && !line.startsWith(">") && !line.startsWith("#")).trim();
  // ...and the rows a pass reads (`batch.numMsa`), at the page's own default: its `#max-msa` is 128:256
  const maxMsa = Number(option(args, "max-msa", "128"));
  const { batch } = af3BatchFromA3m(folded, a3m, {
    maxSequences: maxMsa, ...(ligands.length === 0 ? {} : { ligands }) });
  const { tokens, dense } = batch;

  const gl = createWebGL2Context();
  // ...as the page's fold-batch.js builds it: a bounded texture pool (`--no-pool` is the arm without)
  const execution = webglExecution(gl, args ?? [], { pooling: true, poolLimitBytes: AF3_POOL_LIMIT_BYTES });
  const uploadStart = performance.now();
  const uploaded = uploadAf3Weights(execution, weights);
  const uploadMs = Math.round(performance.now() - uploadStart);
  const weightMegabytes = Number((execution.stats().bytes / (1 << 20)).toFixed(1));

  const call = {
    batch, weights: uploaded, hostWeights: weights,
    msa: foldMsa(batch, weights.trunk.dialect, { seed }),
    recycles, steps: trunkOnly ? 0 : steps, seed, templates,
    readTrunk: reference,
    ...(hoist ? {} : { hoistPairConditioning: false }),
  };
  // 🔴 THE PROGRAMS FIRST, AS THE PAGE DOES THEM. `foldAf3Async` prepares and combines; `foldAf3`
  // is the straight-through version the tools call, so without this a tool measured a path no
  // visitor takes - 83 programs where the page links 23. Every group here, since a tool has no
  // GPU waits to spread the rest over.
  const prepareStart = performance.now();
  await execution.prepare((await prepareFoldPrograms(execution, call)).flat());
  const prepareMs = Math.round(performance.now() - prepareStart);
  const stages = [];
  // `--js-profile`: the host's own JavaScript over the fold, by self time (the JS Self-Profiling API; gpu-chrome.mjs
  // serves the page with the Document-Policy it needs)
  const jsProfiler = flag(args, "js-profile") && typeof globalThis.Profiler === "function"
    ? new globalThis.Profiler({ sampleInterval: 1, maxBufferSize: 1_000_000 }) : null;
  const foldStart = performance.now();
  const result = foldAf3(execution, {
    ...call,
    onStage: (name, detail) => stages.push({
      name, ...detail, at: Math.round(performance.now() - foldStart),
    }),
  });
  const foldMs = Math.round(performance.now() - foldStart);
  let jsProfile = null;
  if (jsProfiler !== null) {
    const trace = await jsProfiler.stop();
    const self = new Map();
    const total = new Map();
    const name = (frameId) => {
      const frame = trace.frames[frameId];
      const resource = frame.resourceId === undefined ? "" : trace.resources[frame.resourceId].split("/").slice(-2).join("/");
      return `${frame.name || "(anonymous)"} ${resource}:${frame.line ?? ""}`;
    };
    for (const sample of trace.samples) {
      if (sample.stackId === undefined) { self.set("(idle / native)", (self.get("(idle / native)") ?? 0) + 1); continue; }
      let stack = trace.stacks[sample.stackId];
      self.set(name(stack.frameId), (self.get(name(stack.frameId)) ?? 0) + 1);
      const seen = new Set();
      for (; stack !== undefined; stack = stack.parentId === undefined ? undefined : trace.stacks[stack.parentId]) {
        const key = name(stack.frameId);
        if (seen.has(key)) continue;
        seen.add(key);
        total.set(key, (total.get(key) ?? 0) + 1);
      }
    }
    const top = (map) => [...map.entries()].sort((a, b) => b[1] - a[1]).slice(0, 25)
      .map(([key, count]) => [key, count]);
    jsProfile = { samples: trace.samples.length, self: top(self), total: top(total) };
  }
  const peak = execution.stats();
  // 🔴 AFTER THE WEIGHTS GO BACK, or `live` is 1221 textures of checkpoint and
  // says nothing about whether the FOLD leaked one.
  releaseAf3Weights(execution, uploaded);
  const memory = { ...execution.stats(), peakBytes: peak.peakBytes };

  // --- the WGSL fold of the same weights on the same features ------------
  // 🔴 THE SAME DRAWS, WHICH IS THE ONLY REASON THIS IS A COMPARISON. Both
  // samplers take `normalFrom(seed)` from shared/af3/random.js and consume it in
  // the same order, so the two walks visit the same noise; a second generator
  // would make each fold a different sample from one distribution and leave
  // nothing to compare.
  let coordinates = null;
  let referenceMs = null;
  if (reference) {
    const start = performance.now();
    // 🔴 THE REFERENCE'S PRECISION IS AN ARM, because the DIFFERENCE between
    // the two backends is mostly the WGSL side's f16: this port is f32
    // throughout - there is no f16 in WebGL2 without an extension nobody can
    // rely on - so comparing against the shipped configuration prices the other
    // backend's quantisation as if it were this one's error. `--reference-f32`
    // pins all four of its precision axes and is the arm that separates them.
    const theirs = await foldBatch(device, batch, weights, {
      steps, recycles, seed, templates, trunkLogits: false,
      ...(flag(args, "reference-f32") ? {
        stagedPrecision: "f32", weightPrecision: "f32",
        pairWeightPrecision: "f32", accumulatePrecision: "f32",
      } : {}),
    });
    referenceMs = Math.round(performance.now() - start);
    let worst = 0;
    for (let atom = 0; atom < tokens * dense; atom += 1) {
      if (!(batch.refMask[atom] > 0)) continue;
      worst = Math.max(worst, Math.hypot(
        result.positions[atom * 3] - theirs.positions[atom * 3],
        result.positions[atom * 3 + 1] - theirs.positions[atom * 3 + 1],
        result.positions[atom * 3 + 2] - theirs.positions[atom * 3 + 2]));
    }
    const sample = (values) => Array.from({ length: 6 }, (_, i) => {
      let seen = 0;
      for (let atom = 0; atom < tokens * dense; atom += 1) {
        if (!(batch.refMask[atom] > 0)) continue;
        if (seen === i) return Number(values[atom].toFixed(3));
        seen += 1;
      }
      return null;
    });
    coordinates = {
      referencePrecision: flag(args, "reference-f32") ? "f32" : "the shipped default",
      plddtLengths: [result.plddt.length, theirs.scores?.plddt?.length ?? null],
      oursFirstLive: sample(result.plddt),
      theirsFirstLive: theirs.scores === undefined ? null : sample(theirs.scores.plddt),
      relativeRms: Number(relativeRms(
        result.positions, theirs.positions, "positions").toPrecision(4)),
      worstAtomAngstroms: Number(worst.toFixed(4)),
      theirMeanPlddt: theirs.meanPlddt === undefined
        ? null : Number(theirs.meanPlddt.toFixed(3)),
      theirPtm: theirs.ptm === undefined ? null : Number(theirs.ptm.toFixed(4)),
      // 🔴 AND THE CHAIN SCORES, BECAUSE THEY ARE WHERE A CHANGE TO THE
      // REDUCTION LANDS FIRST. pTM and ipTM come off the GLOBAL term and move by
      // almost nothing when the anchors change - a framed token wins the maximum
      // anyway - so a comparison of those two alone is green while chain_ptm,
      // chain_iptm and chain_pair_iptm are a tenth out. `chainGaps` below is what
      // is asserted.
      chainPtm: [round4(result.chainPtm), round4(theirs.chainPtm)],
      chainIptm: [round4(result.chainIptm), round4(theirs.chainIptm)],
      chainPairIptm: [round4(result.chainPairIptm), round4(theirs.chainPairIptm)],
      chainGaps: chainGaps([
        ["chain_ptm", result.chainPtm, theirs.chainPtm],
        ["chain_iptm", result.chainIptm, theirs.chainIptm],
        ["chain_pair_iptm", result.chainPairIptm, theirs.chainPairIptm],
      ]),
      // 🔴 THE TRUNK'S TWO TRACKS SEPARATELY, because they fail separately: the
      // pair half of a pairformer block never reads the single, so a single-track
      // defect leaves PAE and the coordinates in band and moves pLDDT alone -
      // which is exactly what this comparison first reported.
      trunkSingleRelativeRms: theirs.trunk?.single === undefined ? null
        : Number(relativeRms(result.trunkSingle, theirs.trunk.single, "single").toPrecision(4)),
      trunkPairRelativeRms: theirs.trunk?.pair === undefined ? null
        : Number(relativeRms(result.trunkPair, theirs.trunk.pair, "pair").toPrecision(4)),
      plddtRelativeRms: theirs.scores === undefined ? null : Number(relativeRms(
        result.plddt, theirs.scores.plddt, "plddt").toPrecision(4)),
      paeRelativeRms: theirs.scores === undefined ? null : Number(relativeRms(
        result.pae, theirs.scores.pae, "pae").toPrecision(4)),
    };
  }

  const spacings = trunkOnly ? [] : backboneSpacing(result.positions, batch);
  const median = spacings.length === 0 ? NaN : spacings[spacings.length >> 1];
  // The contact map says the trunk ran: a map of zeros or ones is not a fold.
  let contacts = 0;
  for (let index = 0; index < result.contactProbs.length; index += 1) {
    if (result.contactProbs[index] > 0.5) contacts += 1;
  }
  const failures = [];
  if (!trunkOnly) {
    // 🔴 THE BOND GATE, AT A STEP COUNT THAT CAN MEET IT. Below ~25 steps the
    // sampler is not converged and CA-CA goes out of band on a correct port
    // too, which is docs/AF3.md's own finding - so the assertion is only made
    // where it means something, and the report says when it was skipped.
    if (steps >= 25) {
      if (!(median > 3.4 && median < 4.2)) {
        failures.push(
          `the chain is not a chain: median CA-CA ${median.toFixed(3)} A, not ~3.8`);
      }
    }
    if (!Number.isFinite(result.meanPlddt)) failures.push("pLDDT is not finite");
    if (result.positions.some((value) => !Number.isFinite(value))) {
      failures.push("the coordinates are not all finite");
    }
  }
  // 🔴 A CONTACT MAP OF ZEROS IS A TRUNK THAT DID NOT RUN, AND THE THRESHOLD IS
  // THE FULL TRUNK'S. A one-block prefix of 48 can honestly have nothing over
  // 0.5, so what is asserted on a prefix is that the map VARIES - a constant map
  // is what a dead stack produces - and the contact count is asserted only where
  // the whole trunk ran.
  let lowest = Infinity;
  let highest = -Infinity;
  for (const value of result.contactProbs) {
    lowest = Math.min(lowest, value);
    highest = Math.max(highest, value);
  }
  // 🔴 AND THE ADJACENT PAIRS ARE THE ONES THAT CANNOT BE WRONG. Consecutive
  // residues are ~3.8 A apart in every protein, so a trunk that has learned
  // anything puts |i-j| = 1 well inside the 8 A contact threshold - and the
  // diagonal at a distance of zero is contact 1 by arithmetic. A map whose
  // neighbours are near zero is a trunk producing a plausible-looking tensor
  // that says the chain is not connected.
  let adjacent = 0;
  let adjacentCount = 0;
  let diagonal = 0;
  for (let i = 0; i < tokens; i += 1) {
    if (!(batch.seqMask[i] > 0)) continue;
    diagonal += result.contactProbs[i * tokens + i];
    for (const j of [i - 1, i + 1]) {
      if (j < 0 || j >= tokens || !(batch.seqMask[j] > 0)) continue;
      adjacent += result.contactProbs[i * tokens + j];
      adjacentCount += 1;
    }
  }
  const adjacentMean = adjacentCount === 0 ? NaN : adjacent / adjacentCount;
  const diagonalMean = diagonal / tokens;
  if (!(highest - lowest > 1e-3)) {
    failures.push(`the trunk's contact map is constant at ${highest}`);
  }
  if (weights.trunk.pairformerBlocks.length >= 48) {
    if (!(contacts > 0)) {
      failures.push("the whole trunk's contact map has no contact over 0.5");
    }
    // Consecutive residues are ~3.8 A apart in every protein and the diagonal
    // is a distance of zero, so both are near 1 or the map is not a contact map.
    if (!(adjacentMean > 0.9)) {
      failures.push(
        `the adjacent contacts read ${adjacentMean.toFixed(4)}, not ~1 - a chain`
        + " whose neighbours are not in contact is not a chain");
    }
    if (!(diagonalMean > 0.99)) {
      failures.push(`the diagonal contacts read ${diagonalMean.toFixed(4)}, not 1`);
    }
  }
  if (memory.live !== 0) failures.push(`${memory.live} tensors left live`);
  // 🔴 THE CROSS-BACKEND BAR, FROM MEASUREMENT AND NOT FROM HOPE. Measured on
  // this bundle: worst atom 0.034 A at 8 steps and 0.078 A at 25, with the trunk
  // single at 6e-5 and the pair at 5e-4. Pinning the WGSL side to f32 on all
  // four precision axes tightens the TRUNK by 8x (single 5.9e-5 -> 7.2e-6) and
  // leaves the COORDINATES where they were (0.034 -> 0.032 A) - so the
  // coordinate difference is not the other backend's f16, it is f32
  // accumulation amplified by a walk that divides by t_hat 25 times. The bar is
  // 0.25 A, about three times the worst measured, and a structural difference
  // would be angstroms rather than tenths.
  if (coordinates !== null) {
    if (!(coordinates.worstAtomAngstroms < 0.25)) {
      failures.push(
        `the two backends' coordinates differ by ${coordinates.worstAtomAngstroms} A`
        + " at the worst atom, over a 0.25 A bar");
    }
    for (const [name, value] of [["pLDDT", coordinates.plddtRelativeRms],
      ["PAE", coordinates.paeRelativeRms]]) {
      if (value !== null && !(value < 1e-2)) {
        failures.push(`the two backends' ${name} differ by relRMS ${value}`);
      }
    }
    // 🔴 THE CHAIN BAR IS 0.02, THE SAME NUMBER check-af3-webgl-page.js HOLDS
    // ITS CHAIN FIELDS TO, and it is a measurement rather than a hope: at 61
    // tokens the precision axes alone move ipTM that much between the two
    // backends. A key present on one side alone reads Infinity and fails here.
    for (const [name, gap] of Object.entries(coordinates.chainGaps ?? {})) {
      if (gap === null) continue;
      if (typeof gap !== "number") {
        failures.push(`the two backends' ${name} name different chains: ${gap}`);
      } else if (!(gap < 0.02)) {
        failures.push(`the two backends' ${name} differ by ${gap}`);
      }
    }
  }

  return {
    bundle, model: store.manifest?.model?.name ?? null,
    sequence: sequence.length, tokens, dense, atoms: tokens * dense,
    subsets: batch.subsets, queries: batch.shape.queries, keys: batch.shape.keys,
    configuration: {
      pairformerBlocks: weights.trunk.pairformerBlocks.length,
      msaBlocks: weights.trunk.msaBlocks.length,
      fullTrunk: weights.trunk.pairformerBlocks.length >= 48,
      steps: trunkOnly ? 0 : steps, recycles, templates, seed, hoistPairConditioning: hoist,
      bondGate: !trunkOnly && steps >= 25,
    },
    dialect: Object.fromEntries(
      Object.entries(dialect).filter(([, value]) => value === true)),
    milliseconds: {
      load: loadMs, upload: uploadMs, prepare: prepareMs, fold: foldMs,
      ...(referenceMs === null ? {} : { reference: referenceMs }),
    },
    againstWgsl: coordinates,
    weightMegabytes,
    ...(jsProfile === null ? {} : { jsProfile }),
    // `--profile`'s table: READ ITS RANKING, NOT ITS TOTAL - each kernel is timed by waiting for it
    ...(execution.profile === null ? {} : { profile: execution.profileTable(Number(option(args, "profile-top", "30"))) }),
    caSpacing: spacings.length === 0 ? null : {
      median: Number(median.toFixed(3)),
      lowest: Number(spacings[0].toFixed(3)),
      highest: Number(spacings[spacings.length - 1].toFixed(3)),
      count: spacings.length,
    },
    meanPlddt: Number.isFinite(result.meanPlddt)
      ? Number(result.meanPlddt.toFixed(3)) : null,
    ptm: Number.isFinite(result.ptm) ? Number(result.ptm.toFixed(4)) : null,
    iptm: Number.isFinite(result.iptm) ? Number(result.iptm.toFixed(4)) : null,
    chainPtm: round4(result.chainPtm), chainIptm: round4(result.chainIptm),
    chainPairIptm: round4(result.chainPairIptm),
    // The tokens that cannot anchor, which is what the frame mask changed: 0 for
    // a plain protein, and one per ion or two-atom ligand.
    framelessTokens: result.frame === undefined ? null
      : Array.from(result.frame).reduce((count, live, token) => count
        + (batch.seqMask[token] > 0 && live !== 1 ? 1 : 0), 0),
    contactsOverHalf: contacts,
    contactRange: [Number(lowest.toFixed(4)), Number(highest.toFixed(4))],
    adjacentContactMean: Number(adjacentMean.toFixed(4)),
    diagonalContactMean: Number(diagonalMean.toFixed(4)),
    stages,
    memory,
    verdict: failures.length === 0
      ? `ok: ${tokens} tokens folded in WebGL2`
        + `${weights.trunk.pairformerBlocks.length >= 48 ? "" : " (a trunk PREFIX)"}`
        + `${trunkOnly ? ", trunk only" : ` at ${steps} steps`}`
      : `FAILED: ${failures.join("; ")}`,
  };
}
