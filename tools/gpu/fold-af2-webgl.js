/**
 * Fold AlphaFold 2's monomer in WebGL2, from a real weight bundle, beside the
 * WGSL fold of the same weights.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/fold-af2-webgl.js
 *     node tools/gpu-chrome.mjs tools/gpu/fold-af2-webgl.js --length=40 --no-reference
 *
 * 🔴 THIS IS THE FIRST TIME THE WebGL2 TREE READS A CHECKPOINT, AND THAT IS A
 * DIFFERENT TEST FROM EVERY GATE BEFORE IT. All of them drive one stage with
 * NOISE, and docs/AF2.md records what that misses: "a packing that moved the
 * real model 4e-4 read 6e-7 on noise". Real weights are not uniform - a
 * projection's rows differ in scale by orders of magnitude, a LayerNorm's gains
 * are near one and its offsets are not - so a layout or a transpose that noise
 * forgives is visible here.
 *
 * 🔴 AND IT IS A FOLD, NOT A RESIDUAL. Two things are asserted that no stage
 * gate can ask: that consecutive alpha carbons are 3.8 A apart, which is
 * `fold-af2.js`'s own gate and the one an 825-residue collapse walked through
 * for a whole campaign while pLDDT climbed to 69; and that the coordinates
 * agree with the WGSL fold of the same weights on the same features. pLDDT is
 * reported and is NOT a correctness gate - that is the whole lesson of that
 * campaign.
 *
 * 🔴 THE FEATURES ARE BUILT ON THE HOST, ON PURPOSE. `makeA3mFeatures` is
 * `shared/`, not a backend, and the WebGPU path's device nearest-centre search
 * is an optimisation of it with its own gate (`check-nearest-centres.js`). Both
 * folds here get the SAME feature set, so the comparison is of the two
 * backends and not of two featurisers.
 *
 * `--no-reference` folds in WebGL2 alone, which is what a machine without a
 * WebGPU adapter would do and is how to time this backend without paying for
 * the other one.
 */
import { AlphaFoldMonomerGpu } from "../../webgpu/af2/model/monomer.js";
import { AlphaFoldUnifiedGpu } from "../../webgpu/af2/multimer/model.js";
import { AlphaFoldFixture } from "../../shared/bundles/alphafold-fixture.js";
import { HttpTensorStore } from "../../shared/bundles/http-tensor-store.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import {
  foldAf2, releaseAf2Weights, uploadAf2Weights,
} from "../../webgl/af2/model/fold.js";
import { makeA3mFeatures } from "../../shared/input/a3m-features.js";
import { setDeviceTuning, setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import { relativeRms } from "./relative-rms.js";

const DEFAULT_SEQUENCE = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  // 🔴 THE GRAPH DECIDES SIX THINGS AND THE BUNDLE DECIDES THE GRAPH, which is
  // the order `fold-af2.js` had to learn: typing neither "used to run MULTIMER
  // WEIGHTS ON THE MONOMER GRAPH, which is how the page's own bug reached a
  // visitor". Here `--family=multimer` names both.
  const family = option(args, "family", "monomer");
  if (family !== "monomer" && family !== "multimer") {
    throw new RangeError(`--family is monomer or multimer, not \`${family}\``);
  }
  const graph = family;
  const defaultBundle = family === "multimer" ? "/model-multimer" : "/model-af2-monomer-int5";
  const bundle = option(args, "bundle", defaultBundle).replace(/\/$/, "");
  const length = Number(option(args, "length", "0"));
  const sequence = length > 0 ? DEFAULT_SEQUENCE.slice(0, length) : DEFAULT_SEQUENCE;
  // `--chains=30,29` splits the one residue axis a multimer folds over. One
  // chain is a legal multimer and is the arm that should reproduce a monomer.
  const chains = option(args, "chains", null) === null
    ? [sequence.length]
    : option(args, "chains", "").split(",").map(Number);
  if (chains.reduce((a, b) => a + b, 0) !== sequence.length) {
    throw new RangeError(
      `--chains sums to ${chains.reduce((a, b) => a + b, 0)}, not ${sequence.length}`);
  }
  const recycles = Number(option(args, "recycles", "0"));
  const seed = Number(option(args, "seed", "42"));
  const reference = !flag(args, "no-reference");
  const mainBlocks = Number(option(args, "main-blocks", "0"));
  const floor = reference && !flag(args, "no-floor");
  const extraBlocks = Number(option(args, "extra-blocks", "0"));

  const loadStart = performance.now();
  const store = await HttpTensorStore.open(`${bundle}/manifest.json`);
  const fixture = AlphaFoldFixture.fromStore(store);
  const multimer = graph === "multimer";
  const [embedding, template, templateEmbedding, extraStack, mainStack, structure, confidence,
    geometry, featureTables, paeBreaks] = await Promise.all([
    fixture.embeddingWeights(),
    // ...the same split web/model.js makes: the multimer's embedder runs its own
    // nine-projection template term and has no monomer template stack at all.
    // 🔴 UNDEFINED, NOT NULL. The unified WGSL driver tests `weights.template`
    // for undefined and hands a null straight to `QueryOnlyTemplateGpu`, which
    // reads `.embeddingBias` off it - so a null here is a TypeError inside the
    // reference rather than a skipped stage. `fold-af2.js` passes undefined for
    // the same reason, and this port treats both as absent.
    multimer ? Promise.resolve(undefined) : fixture.templateWeights(),
    multimer ? fixture.templateEmbeddingWeights() : Promise.resolve(undefined),
    fixture.extraStackWeights(),
    fixture.mainStackWeights(),
    fixture.structureWeights(),
    fixture.confidenceWeights(),
    fixture.geometryTables(),
    fixture.queryOnlyFeatureTables(),
    fixture.tensor("confidencePaeBreaks"),
  ]);
  const distogramHead = await fixture.distogramHeadWeights?.() ?? null;
  // 🔴 `--main-blocks=` AND `--extra-blocks=` TRUNCATE BOTH SIDES, which is how
  // a divergence is separated from an accumulation: a fold that agrees at one
  // block and not at forty-eight is an accumulating difference, and one that
  // disagrees at the first block is a defect in the block.
  // 🔴 `--no-template` IS THE ARM THAT PROVES THE TEMPLATE STAGE RAN AT ALL.
  // model_3, model_4 and model_5 are AF2's template-free checkpoints, so the
  // path is real rather than hypothetical - and a template stage that is wired
  // to nothing issues the same draws whether or not its weights are there,
  // which is exactly what a draw count that does not move says.
  const noTemplate = flag(args, "no-template");
  if (noTemplate && multimer) {
    throw new RangeError(
      "--no-template is a monomer arm: the multimer embedder runs whether or not there is a"
      + " template, and dropping it leaves the pair track about 30% out");
  }
  const weights = {
    embedding,
    template: noTemplate ? null : template,
    templateEmbedding,
    extraStack: extraBlocks > 0 ? extraStack.slice(0, extraBlocks) : extraStack,
    mainStack: mainBlocks > 0 ? mainStack.slice(0, mainBlocks) : mainStack,
    structure, lddt: confidence.lddt, pae: confidence.pae, geometry, distogramHead,
  };
  const loadMs = Math.round(performance.now() - loadStart);

  const a3m = `>query\n${sequence}\n`;
  const featureOptions = {
    recycles, randomSeed: seed, maxMsaSequences: 1, maxExtraSequences: 1,
    // `chainAware` is what makes the featuriser emit the asym, entity and
    // symmetry lanes the 73-row relative encoding reads.
    ...(multimer ? { chainAware: true, chainSequences: chains } : {}),
  };
  const features = makeA3mFeatures(a3m, featureTables, featureOptions);

  // --- the WebGL2 fold ----------------------------------------------------
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const uploadStart = performance.now();
  const uploaded = uploadAf2Weights(execution, weights, graph);
  const uploadMs = Math.round(performance.now() - uploadStart);
  const weightMegabytes = Number((execution.stats().bytes / (1 << 20)).toFixed(1));
  const foldStart = performance.now();
  const ours = await foldAf2(execution, {
    featuresByRecycle: features, weights: uploaded, paeBreaks, graph,
  });
  const webglMs = Math.round(performance.now() - foldStart);
  const memory = execution.stats();
  // 🔴 THE FLOOR, BECAUSE "0.07 A APART" MEANS NOTHING ON ITS OWN. The trunk
  // keeps three intermediates in f16 with no switch to reach them - the
  // attention's normalised input, its flash kernel's operands and the
  // transition's hidden row - so both backends quantise, and two f32 values a
  // step apart that straddle an f16 boundary round 5e-4 apart. Over 48 blocks
  // and eight structure iterations that compounds into coordinates.
  //
  // `execution.halfDither` perturbs by one f32 step INSIDE every one of those
  // roundings, which is exactly the difference the two backends' own f32
  // arithmetic represents. Folding twice with and without it gives the spread
  // two CORRECT implementations must be allowed, in angstroms, measured on this
  // fold rather than argued from the kernel widths.
  //
  // 🔴 AND THE DITHER ALONE IS NOT A FLOOR ON EVERY GRAPH, WHICH THE MULTIMER
  // PROVED BY READING EXACTLY ZERO. That graph's attention targets and
  // transitions are all f32, so there is no f16 rounding for a dither to sit
  // inside and `halfDither` moves nothing - a bar of `0 * 3` then fails a fold
  // agreeing to 0.0001 A. So the floor is the LARGER of two measurements: the
  // dither, which prices quantisation where there is any, and one f32 step of
  // jitter on the EMBEDDING weights, which prices f32 accumulation through the
  // whole trunk. The embedder is where both representations come from, so a step
  // there reaches every kernel after it - and it is a few hundred kilobytes
  // rather than the 446 MiB a jitter of every weight would re-upload.
  let dithered;
  let jittered;
  if (floor) {
    execution.halfDither = 1.2e-7;
    dithered = await foldAf2(execution, {
      featuresByRecycle: features, weights: uploaded, paeBreaks, graph,
    });
    execution.halfDither = 0;
    const ULP = 1.2e-7;
    let state = 20260104;
    const next = () => {
      state = (Math.imul(state, 1103515245) + 12345) >>> 0;
      return state / 0x1_0000_0000;
    };
    const jitteredEmbedding = Object.fromEntries(
      Object.entries(weights.embedding).map(([name, values]) => [name,
        Float32Array.from(/** @type {Float32Array} */ (values),
          (v) => v * (1 + ULP * (next() * 2 - 1)))]));
    const shifted = uploadAf2Weights(
      execution, { ...weights, embedding: jitteredEmbedding }, graph);
    jittered = await foldAf2(execution, {
      featuresByRecycle: features, weights: shifted, paeBreaks, graph,
    });
    releaseAf2Weights(execution, shifted);
  }
  releaseAf2Weights(execution, uploaded);
  const liveAfter = execution.stats().live;

  // --- the WGSL fold of the same weights ----------------------------------
  let theirs;
  let webgpuMs;
  if (reference) {
    setDeviceTuning(device, { allowHostWeightPacking: true });
    // f32 throughout, or this compares an f32 backend against an f16 one.
    setHalfPrecision(device, false);
    const referenceStart = performance.now();
    theirs = await new (multimer ? AlphaFoldUnifiedGpu : AlphaFoldMonomerGpu)(device).predict(
      features, weights, paeBreaks, undefined, undefined,
      // The regime the page passes for multimer, verbatim.
      multimer
        ? {
          outerProductMeanFirst: true, positionScale: 20,
          chainAware: true, chainSequences: chains,
        }
        : {});
    webgpuMs = Math.round(performance.now() - referenceStart);
  }

  // --- is it a chain? -----------------------------------------------------
  const atom37 = ours.final.atom37;
  // 🔴 THE CHAIN JUNCTIONS ARE SKIPPED, OR THE METRIC MEASURES NOTHING. Two
  // chains are not bonded to each other, so the step across the boundary is
  // whatever the fold placed them at and would sit in `worst` for ever looking
  // like a broken backbone. `fold-af2.js` makes the same point.
  const breaks = new Set();
  let boundary = 0;
  for (const chain of chains.slice(0, -1)) { boundary += chain; breaks.add(boundary - 1); }
  const distances = [];
  for (let residue = 0; residue + 1 < sequence.length; residue += 1) {
    if (breaks.has(residue)) continue;
    const a = (residue * 37 + 1) * 3;
    const b = ((residue + 1) * 37 + 1) * 3;
    distances.push(Math.hypot(
      atom37[a] - atom37[b], atom37[a + 1] - atom37[b + 1], atom37[a + 2] - atom37[b + 2]));
  }
  const sorted = [...distances].sort((x, y) => x - y);
  const median = sorted[Math.floor(sorted.length / 2)];
  const worst = distances.reduce(
    (far, value) => (Math.abs(value - 3.8) > Math.abs(far - 3.8) ? value : far), 3.8);
  // `fold-af2.js`'s own bands, unchanged: measured healthy median 3.485-3.972
  // and worst 1.69-4.55, measured broken median 1.44-3.41 and worst 0.06 or
  // 7.73-70.45.
  const chainOk = median >= 3.4 && median <= 4.2 && Math.abs(worst - 3.8) <= 2.8;

  const round = (value, places = 4) => Number(value.toFixed(places));
  /** The furthest any one atom moves between two coordinate sets, in angstroms. */
  const worstAtom = (left, right) => Math.max(...Array.from(
    { length: left.length / 3 },
    (_, atom) => Math.hypot(
      left[atom * 3] - right[atom * 3],
      left[atom * 3 + 1] - right[atom * 3 + 1],
      left[atom * 3 + 2] - right[atom * 3 + 2])));
  const ditherFloor = dithered === undefined ? undefined : {
    atom37: round(relativeRms(dithered.final.atom37, atom37, "floor.atom37"), 8),
    worstAtomAngstroms: round(worstAtom(dithered.final.atom37, atom37), 4),
    meanPlddt: round(dithered.final.confidence.meanPlddt, 3),
  };
  const jitterFloor = jittered === undefined ? undefined : {
    atom37: round(relativeRms(jittered.final.atom37, atom37, "floor.jitter.atom37"), 8),
    worstAtomAngstroms: round(worstAtom(jittered.final.atom37, atom37), 4),
    meanPlddt: round(jittered.final.confidence.meanPlddt, 3),
  };
  const floorAngstroms = ditherFloor === undefined ? undefined
    : Math.max(ditherFloor.worstAtomAngstroms, jitterFloor.worstAtomAngstroms);
  const comparison = theirs === undefined ? undefined : {
    atom37: round(relativeRms(atom37, theirs.final.structure.atom37, "atom37"), 8),
    // The number a reader of a structure cares about: how far the two backends
    // put the same atom, in angstroms, at the worst atom.
    worstAtomAngstroms: round(worstAtom(atom37, theirs.final.structure.atom37), 4),
    meanPlddt: {
      webgl: round(ours.final.confidence.meanPlddt, 3),
      webgpu: round(theirs.final.confidence.meanPlddt, 3),
    },
    plddt: round(relativeRms(
      ours.final.confidence.plddt, theirs.final.confidence.plddt, "plddt"), 8),
    predictedAlignedError: round(relativeRms(ours.final.confidence.predictedAlignedError,
      theirs.final.confidence.predictedAlignedError, "pae"), 8),
  };

  return {
    family,
    bundle,
    chains,
    sequence: sequence.length > 24 ? `${sequence.slice(0, 24)}...(${sequence.length})` : sequence,
    length: sequence.length,
    blocks: { extra: weights.extraStack.length, main: weights.mainStack.length },
    templateArm: noTemplate ? "dropped by --no-template" : "present",
    template: multimer
      ? (templateEmbedding === undefined
        ? "MISSING: the multimer graph needs a template embedder"
        : `the multimer embedder, ${templateEmbedding.stack.length} pair blocks, every pass`)
      : (weights.template === null || weights.template === undefined
        ? "absent from this checkpoint" : "the monomer stack, hoisted out of the loop"),
    distogramHead: distogramHead === null ? "absent from this bundle" : "present",
    recycles, passesRun: ours.recycles.length, seed,
    milliseconds: { weightLoad: loadMs, weightUpload: uploadMs, webgl: webglMs, webgpu: webgpuMs },
    weightMegabytes,
    meanPlddt: round(ours.final.confidence.meanPlddt, 3),
    caca: { median: round(median, 3), worst: round(worst, 2), ok: chainOk },
    comparison,
    ditherFloor,
    jitterFloor,
    floorAngstroms,
    atFloor: floorAngstroms === undefined || comparison === undefined ? undefined
      : comparison.worstAtomAngstroms <= floorAngstroms * 3,
    memory: {
      draws: memory.draws,
      programs: memory.programs,
      peakMegabytes: Number((memory.peakBytes / (1 << 20)).toFixed(1)),
      liveTensorsAfter: liveAfter,
    },
    verdict: !chainOk
      ? `FAILED: the fold is not a chain - consecutive CA median ${median.toFixed(3)} A, `
        + `worst ${worst.toFixed(2)} A, against 3.80 expected. pLDDT says `
        + `${ours.final.confidence.meanPlddt.toFixed(2)} and it is not a correctness gate.`
      : liveAfter !== 0
        ? `FAILED: ${liveAfter} tensors left live after the fold`
        : theirs === undefined
          ? `ok: AlphaFold 2 ${family} folds in WebGL2 - ${sequence.length} residues, `
            + `consecutive CA ${median.toFixed(3)} A (no WGSL arm asked for)`
          : floorAngstroms === undefined
            ? `ok: AlphaFold 2 ${family} folds in WebGL2 - ${sequence.length} residues, consecutive CA `
              + `${median.toFixed(3)} A, worst atom ${comparison.worstAtomAngstroms} A from `
              + "the WGSL fold of the same weights (no floor measured)"
            : comparison.worstAtomAngstroms <= floorAngstroms * 3
              ? `ok: AlphaFold 2 ${family} folds in WebGL2 - ${sequence.length} residues, consecutive CA `
                + `${median.toFixed(3)} A, worst atom ${comparison.worstAtomAngstroms} A from the `
                + `WGSL fold against a measured floor of ${round(floorAngstroms, 4)} A`
              : `FAILED: the two backends put an atom ${comparison.worstAtomAngstroms} A apart `
                + `where one f32 step through this fold moves it ${round(floorAngstroms, 4)} A `
                + "- more than rounding explains",
  };
}
