/**
 * An AlphaFold 2 monomer fold, end to end, so a kernel change can be compared
 * against the tree before it.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/fold-af2.js
 *     node tools/gpu-chrome.mjs tools/gpu/fold-af2.js --rows=512 --recycles=1
 *     node tools/gpu-chrome.mjs tools/gpu/fold-af2.js --family=multimer --chains=30,29
 *
 * 🔴 AF2 HAD NO END-TO-END GATE THAT RUNS ON THIS MACHINE, and its kernels have
 * now been rewritten three times. `npm run test:gpu` cannot load Dawn here;
 * tools/gpu/check-evoformer-stack.js is the official-value check and wants
 * test/fixtures/evoformer/model1-query-59-stack, which this checkout does not
 * carry - it has the features and not `stackInputMsa`. So every AF2 kernel
 * change has been gated on per-kernel differential checkers, each of which can
 * only say that ONE kernel still computes its own operation. Nothing said the
 * assembled model still folds.
 *
 * This is not an oracle either - it does not know what AlphaFold would say. It
 * folds deterministically and prints enough to compare two trees: mean pLDDT,
 * pTM, the backbone CA-CA geometry, and a checksum over every coordinate.
 * Run it, stash the change, run it again.
 *
 * 🔴 THE ALIGNMENT IS SYNTHESISED FROM THE QUERY, WHICH IS FINE HERE AND ONLY
 * HERE. Rows are the query with every (i+3)th column dropped to a gap, so the
 * MSA path and its 512-row kernels actually run without fetching anything -
 * this machine is on a metered connection and an MMseqs2 search is 88 s and a
 * download besides. It makes the numbers meaningless as biology and perfectly
 * good as a fingerprint, which is what a regression needs.
 *
 * 🔴 MULTIMER SHARES EVERY KERNEL AND HAD NO GATE AT ALL. src/af2/multimer/block.js
 * builds its blocks from the same attention, transition and outer-product-mean
 * shaders src/af2/evoformer/block.js does, with its own dispatches - so a tile
 * changed in one and not threaded through the other is a bug that only a
 * multimer fold can see. `--family=multimer` runs that path, through
 * AlphaFoldUnifiedGpu and the multimer regime the page passes (outer product
 * mean first, chain-aware, position scale 20), with `--chains` splitting the
 * sequence.
 *
 * 🔴 AND pLDDT IS NOT THE CHECK. docs/AF3.md records a batch with one broken gather
 * folding 17 A of spaghetti at pLDDT 55. Consecutive CA are 3.80 A apart in a
 * real protein and nothing else; `caca` is the number that a wrong kernel
 * cannot fake, so it is printed with its worst outlier.
 */
import { memorySnapshot } from "../../src/runtime/device-memory.js";
import { noteResidencyRefused, setMemoryBudget } from "../../src/runtime/device-memory.js";
import { DEFAULT_TUNING, setDeviceTuning } from "../../src/runtime/device-profile.js";
import { AlphaFoldFixture } from "../../src/bundles/alphafold-fixture.js";
import { HttpTensorStore } from "../../src/bundles/http-tensor-store.js";
import { AlphaFoldMonomerGpu } from "../../src/af2/model/monomer.js";
import { AlphaFoldUnifiedGpu } from "../../src/af2/multimer/model.js";
import { setShaderSourceVerification } from "../../src/runtime/shader-source-cache.js";
import { featureStats, resetFeatureStats } from "../../src/input/a3m-features.js";
import { chainResidues, identityMap, templateSlotAtom37 }
  from "../../src/af3/featurise/template-input.js";
import { superpose } from "./superpose.js";
import { DeltaTensorStore } from "../../src/bundles/delta-tensor-store.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/**
 * The memory report, cut to the rows worth reading.
 *
 * 🔴 `peakByLabel` IS THE ONE TO READ. `byLabel` sums every allocation a label
 * ever made, so a scratch tensor taken and returned once a block reads as
 * forty-eight times its size; it says what CHURNS. `peakByLabel` is what was
 * on the device when it was fullest, and its rows sum to `peakBytes` - it says
 * what to attack.
 */
const trimMemory = (snapshot, rows = 12) => ({
  peakBytes: snapshot.peakBytes,
  peakMiB: Number((snapshot.peakBytes / (1024 * 1024)).toFixed(1)),
  peakByLabel: snapshot.peakByLabel.slice(0, rows).map((entry) => ({
    label: entry.label,
    mib: Number((entry.bytes / (1024 * 1024)).toFixed(2)),
    count: entry.count,
  })),
  peakAccountedMiB: Number((snapshot.peakByLabel.reduce((sum, e) => sum + e.bytes, 0)
    / (1024 * 1024)).toFixed(1)),
  churnByLabel: snapshot.byLabel.slice(0, rows).map((entry) => ({
    label: entry.label,
    mib: Number((entry.bytes / (1024 * 1024)).toFixed(2)),
    count: entry.count,
  })),
});

/** 59 residues with side chains of every length; the shape the benches use. */
const DEFAULT_SEQUENCE = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

/**
 * The progress stream, bucketed into stages by the size of each step.
 *
 * A stage's unit size is its signature - af2Plan gives extra-stack, main-stack,
 * structure and confidence different ones - so consecutive steps of the same
 * size are one stage, and a change of size is a boundary. That is enough to
 * name where a fold's minutes go without the model reporting stage names it
 * does not currently have.
 */
function summariseStages(marks, started) {
  if (marks.length === 0) return undefined;
  const runs = [];
  let previousAt = started;
  let previousCompleted = 0;
  for (const [at, completed] of marks) {
    const units = Math.round(completed - previousCompleted);
    const last = runs[runs.length - 1];
    if (last !== undefined && last.units === units) {
      last.steps += 1;
      last.ms += at - previousAt;
    } else {
      runs.push({ units, steps: 1, ms: at - previousAt });
    }
    previousAt = at;
    previousCompleted = completed;
  }
  return runs.map(({ units, steps, ms }) => ({
    units, steps, seconds: Number((ms / 1000).toFixed(2)),
    msPerStep: Number((ms / steps).toFixed(1)),
  }));
}

export async function main(device, args) {
  // 🔴 A CEILING, SO THE RESIDENCY FALLBACK CAN BE MADE TO FIRE. AF2 keeps its
  // block weights on the device now - 280 MiB of a 387 MiB fold at 59 residues
  // - and drops back to uploading per pass when an allocation would cross the
  // budget. A fallback nothing has ever taken is a fallback nobody has checked;
  // `--budget=200` is small enough to take it and large enough to fold.
  const budgetMiB = Number(option(args, "budget", "0"));
  if (budgetMiB > 0) setMemoryBudget(device, budgetMiB * 1024 * 1024);
  // ...and the arm without residency at all, which is what a device that
  // refused it once gets for the rest of its life. It is the control for
  // `--budget`: without it, a failure under a budget cannot be told apart from
  // a failure the resident weights caused.
  if (args.includes("--no-resident")) noteResidencyRefused(device);
  // 🔴 THE GATE FOR THE SHADER SOURCE MEMO. Memoising a generated source by its
  // pipeline key makes ComputePipelineCache's collision check compare a string
  // with itself, so the check that a key names everything its source depends on
  // moves into the memo - and a check nothing runs is not a check. This flag
  // rebuilds every source on every hit and throws where the two differ, which
  // is the whole-fold version of test/shader-source-cache.test.js.
  if (args.includes("--verify-sources")) setShaderSourceVerification(true);
  // 🔴 THE CONTROL ARM FOR THE DEVICE FEATURISATION, and the way its answer is
  // checked end to end: the nearest-centre search decides the cluster profile,
  // so the two arms must agree on the fold's CHECKSUM and not merely finish.
  const hostFeaturisation = args.includes("--host-features");
  // 🔴 THE TARGET IS READ BEFORE THE SEQUENCE, AND SUPPLIES IT. Deriving the
  // query from the deposition by hand is how the two come to disagree:
  // `chainResidues` counts known HETATM residues (a selenomethionine is a
  // residue) and an ATOM-only reading of the same file does not, which is 261
  // against 255 on 5CAJ chain A. One reader for both sides, the way
  // `fold-opendde.js` does it.
  const targetName = option(args, "target", "");
  const targetChain = option(args, "chain", "A");
  let targetStructure;
  if (targetName !== "") {
    const text = await (await fetch(`/tools/fixtures/${targetName}-crystal.pdb`)).text();
    targetStructure = chainResidues(text, targetChain);
  }
  const sequence = option(args, "sequence",
    targetStructure?.sequence ?? DEFAULT_SEQUENCE);
  const family = option(args, "family", "monomer");
  if (family !== "monomer" && family !== "multimer") {
    throw new RangeError(`unknown family ${family}: expected "monomer" or "multimer"`);
  }
  const chainLengths = option(args, "chains", "").split(",").filter(Boolean).map(Number);
  const rows = Number(option(args, "rows", "128"));
  // 🔴 AF2's TWO STACKS HAVE TWO DEPTHS AND THIS TOOL CONFLATED THEM. The
  // monomer runs an EXTRA-MSA stack and then the main evoformer, and AlphaFold's
  // own monomer preset is 512 clusters against 1024 extra - so a single --rows
  // could express 512/512 or 1024/1024 and not the setting anybody runs.
  const extraRows = Number(option(args, "extra-rows", String(rows)));
  const recycles = Number(option(args, "recycles", "0"));
  const seed = Number(option(args, "seed", "0"));
  // 🔴 `--tune=key=value,...`, THE SAME FLAG tools/gpu/fold.js CARRIES, because
  // AF2 had no way to reach a tuning knob at all - and CLAUDE.md's rule is that
  // a knob no gate enters is a knob nobody has checked. The checksum below is
  // the gate: a knob that only reorders work has to leave it untouched.
  const tune = option(args, "tune", "");
  if (tune !== "") {
    const forced = {};
    for (const pair of tune.split(",").filter(Boolean)) {
      const at = pair.indexOf("=");
      if (at < 0) throw new Error(`--tune wants key=value, got ${pair}`);
      const key = pair.slice(0, at);
      if (!(key in DEFAULT_TUNING)) {
        throw new Error(`--tune names ${key}, which is not a tuning knob. `
          + `Known: ${Object.keys(DEFAULT_TUNING).sort().join(", ")}`);
      }
      const raw = pair.slice(at + 1);
      try { forced[key] = JSON.parse(raw); } catch { forced[key] = raw; }
    }
    setDeviceTuning(device, forced);
  }
  if (!Number.isSafeInteger(rows) || rows < 1) throw new RangeError("rows must be a positive integer");
  if (!Number.isSafeInteger(extraRows) || extraRows < 1) {
    throw new RangeError("extra-rows must be a positive integer");
  }

  // ...the LOCAL bundle, by directory rather than through web/model.js's
  // loadModel: that resolves the monomer family to its remote base, and this
  // machine should not pull 227 MB to run a regression.
  const { MODEL_BUNDLES, loadManifest } = await import("../../src/bundles/manifests/index.js");
  // 🔴 `--bundle=<directory>` READS THE manifest.json BESIDE THE SHARDS, which
  // is the ONLY way to fold a bundle the registry does not name - and AlphaFold
  // 2 ships five models where this repository has published one. It is the
  // arm, not the shipping path: the page reads the manifest MODULE, so a
  // bundle this flag likes can still be one the page cannot load (CLAUDE.md
  // has that trap, and it cost 122 MiB of download before a fold).
  const bundleDirectory = option(args, "bundle", "").replace(/\/$/, "");
  let store = bundleDirectory === ""
    ? await HttpTensorStore.fromManifest(
      MODEL_BUNDLES[family].directory, await loadManifest(family))
    : await HttpTensorStore.open(`${bundleDirectory}/manifest.json`);
  // 🔴 A DELTA BUNDLE IS HALF A MODEL AND SAYS SO. Its manifest carries a
  // `delta` header naming the family it is added to, so this opens that base as
  // well - see src/bundles/delta-tensor-store.js. `--base=` overrides the
  // directory for a base that is not the registry's.
  if (store.manifest.delta !== undefined) {
    const header = store.manifest.delta;
    const baseDirectory = option(args, "base", "").replace(/\/$/, "");
    const base = baseDirectory !== ""
      ? await HttpTensorStore.open(`${baseDirectory}/manifest.json`)
      : await HttpTensorStore.fromManifest(MODEL_BUNDLES[header.baseFamily].directory,
        await loadManifest(header.baseFamily));
    console.log(`[delta] ${header.model ?? "a delta"} on ${header.baseModel}:`
      + ` ${header.addTo.length} added, ${header.whole.length} whole,`
      + ` ${header.absent.length} absent`);
    store = new DeltaTensorStore(base, store);
    // 🔴 AND IT HOST-PACKS, BY CONSTRUCTION RATHER THAN BY SURPRISE. With no
    // `tensorSource` every weight is reconstructed on the host, so the resident
    // descriptors are built from VALUES and the device weight packer refuses
    // them - the right refusal, and the reason this says so out loud rather
    // than leaving a fold looking mysteriously slow. Applying the delta ON the
    // device (planBlockUpload's `accumulate`, gated by
    // tools/gpu/check-delta-upload.js) is what removes the cost.
    console.log("[delta] host weight packing: a reconstructed tensor has no shard");
    setDeviceTuning(device, { allowHostWeightPacking: true });
    // 🔴 AND IT HOST-PACKS, BY CONSTRUCTION RATHER THAN BY SURPRISE. A
    // reconstructed tensor has no shard of its own, so the device weight packer
    // refuses it - the right refusal, and the reason this says so out loud
    // rather than leaving a fold looking mysteriously slow. Applying the delta
    // ON the device (planBlockUpload's `accumulate`, gated by
    // tools/gpu/check-delta-upload.js) is what removes the cost.
  }
  // 🔴 EVERY SHARD AT ONCE, WHICH IS WHAT THE PAGE DOES. `prefetch` is opt-in
  // because a bench that reads four blocks should not pull the whole manifest -
  // but this tool loads a whole model, so a run without it measures a download
  // pattern no user has: shards arrive as tensors are asked for, which leaves
  // most of the connection idle most of the time. Measured on the ESMFold2
  // tool, which had the same hole: a fold 2.25 s -> 1.75.
  store.prefetch();
  const fixture = AlphaFoldFixture.fromStore(store);
  const loadStart = performance.now();
  // ...the same split web/model.js makes: multimer's embedder runs its template
  // track every recycle, the monomer's is the query-only residual.
  const multimer = family === "multimer";
  const [embedding, template, templateEmbedding, extraStack, mainStack, structure, confidence,
         geometry, featureTables, paeBreaks] = await Promise.all([
    fixture.embeddingWeights(),
    multimer ? Promise.resolve(undefined) : fixture.templateWeights(),
    multimer ? fixture.templateEmbeddingWeights() : Promise.resolve(undefined),
    fixture.extraStackWeights(),
    fixture.mainStackWeights(), fixture.structureWeights(), fixture.confidenceWeights(),
    fixture.geometryTables(), fixture.queryOnlyFeatureTables(),
    fixture.tensor("confidencePaeBreaks"),
  ]);
  const weights = {
    embedding, template, templateEmbedding, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry,
  };
  const loadMs = Math.round(performance.now() - loadStart);

  // 🔴 A REAL ALIGNMENT, BECAUSE THE SYNTHETIC ONE HAS TWELVE DISTINCT ROWS.
  // The generator below varies the gap stride by `row % 11`, so rows 1, 12, 23
  // and so on are IDENTICAL: `--rows=128 --extra-rows=128` is 256 rows carrying
  // 12 sequences. That is fine for timing - the work is the same - and useless
  // for anything that depends on what the rows SAY, which is clustering,
  // profiles and deduplication. `--a3m=<path>` folds a real one.
  const a3mPath = option(args, "a3m", "");
  const a3mFromFile = a3mPath === "" ? null
    : await (await fetch(a3mPath.startsWith("/") ? a3mPath : `/${a3mPath}`)).text();

  const lines = [">query", sequence];
  // 🔴 DEEP ENOUGH FOR BOTH CAPS, NOT THE LARGER OF THEM. The clusters are
  // taken first and the extra rows come out of what is left, so a
  // max(512, 1024) = 1024-row alignment at 512 clusters leaves only 512 extra -
  // half the extra stack's work, while the report still says 1024.
  // 🔴 AND EVERY ROW DISTINCT, WHICH IT WAS NOT. The stride used to be
  // `row % 11 + 3`, so rows 1, 12, 23 and so on were IDENTICAL and
  // `--rows=128 --extra-rows=128` was 256 rows carrying **12** sequences. That
  // never mattered while nothing looked at what a row SAID - the work is the
  // same either way, so every timing here stands - but the featuriser drops
  // duplicate rows now, and a degenerate alignment would have collapsed every
  // AF2 gate to a depth-12 fold while still reporting 256.
  //
  // The gap pattern is the row's bit pattern, so two rows agree only if they
  // agree in every bit the columns reach; the assertion below is what says so
  // rather than the argument.
  for (let row = 1; row < rows + extraRows; row += 1) {
    lines.push(`>synthetic${row}`);
    lines.push([...sequence].map((code, column) =>
      (((row >> (column % 24)) & 1) === 1 ? "-" : code)).join(""));
  }
  // 🔴 A GENERATOR THAT SILENTLY REPEATS ITSELF IS THE BUG THIS REPLACES, so
  // it is checked rather than reasoned about. A short sequence cannot express
  // enough bits, and that has to fail loudly rather than quietly shrink the
  // alignment.
  const distinct = new Set(lines.filter((_, index) => index % 2 === 1)).size;
  if (a3mPath === "" && distinct !== rows + extraRows) {
    throw new Error(`the synthetic alignment has ${distinct} distinct rows of `
      + `${rows + extraRows}: a ${sequence.length}-residue sequence cannot express `
      + "enough gap patterns. Use --a3m= with a real alignment.");
  }
  const a3m = a3mFromFile ?? `${lines.join("\n")}\n`;

  const chains = chainLengths.length > 0 ? chainLengths : [sequence.length];
  if (chains.reduce((sum, value) => sum + value, 0) !== sequence.length) {
    throw new RangeError(`--chains sums to ${chains.reduce((a, b) => a + b, 0)}, not ${sequence.length}`);
  }
  // The regime web/app.js passes for multimer, verbatim - it is not defaults,
  // and dropping it once ran multimer WEIGHTS on the monomer graph.
  const regime = multimer
    ? { outerProductMeanFirst: true, positionScale: 20, chainAware: true, chainSequences: chains }
    : {};
  // 🔴 WHERE A FOLD'S TIME GOES BY STAGE, from the progress stream. The block
  // profiler sees one evoformer block; a fold is FOUR extra-MSA blocks at the
  // deeper alignment, then 48 main ones, then the structure module and the
  // heads, and nothing here said what those four cost. `onProgress` fires once
  // per block with a running unit count, and a stage's unit size is its
  // signature - af2Plan gives extra-stack and main-stack different ones - so
  // the deltas name the stage without the model having to report it. Submission
  // is windowed and each window ends on `onSubmittedWorkDone`, so this is
  // GPU-paced to about a window.
  const stageMarks = [];
  const onProgress = ({ completed }) => {
    stageMarks.push([performance.now(), completed]);
  };
  // 🔴 A STRUCTURAL TEMPLATE, WHICH THIS TOOL COULD NOT PASS AND THE MONOMER
  // DRIVER COULD NOT TAKE. `QueryOnlyTemplateGpu` has always accepted one -
  // `input.template` builds the real geometry, its absence writes zeros and the
  // GAP restype, which is a fully masked template - and monomer.js and
  // query-only.js simply never forwarded the field, so no monomer fold could
  // use one. The MULTIMER has forwarded it since it was written.
  //
  // 🔴 AND IT IS atom37, NOT AF3's DENSE 24. `templateSlotAtom37` indexes by
  // atom NAME, so CB is slot 3 for everything that has one - which is what
  // `AF2_ATOM37_MONOMER` means by `pseudoBeta: 3` and `backbone: [2, 1, 0]`.
  // The dense builder indexes by position in each residue's OWN conformer, so
  // handing one to the other reads the wrong atoms with no error at all.
  const templateSpec = option(args, "template", "");
  let templateSlot;
  if (templateSpec !== "") {
    const [path, wantedChain] = templateSpec.split(":");
    const text = await (await fetch(path.startsWith("/") ? path : `/${path}`)).text();
    const structure = chainResidues(text, wantedChain);
    if (structure.residues.length === 0) {
      throw new Error(`--template=${templateSpec} resolved no residues`);
    }
    // 🔴 THE IDENTITY MAP IS ONLY RIGHT WHILE THE SEQUENCES AGREE, and refusing
    // is cheaper than a silently misaligned template: a homolog or a construct
    // with a tag needs a real alignment, which is a different function with a
    // different failure mode.
    if (structure.sequence !== sequence) {
      throw new Error(`--template's chain is ${structure.residues.length} residues `
        + `reading ${structure.sequence.slice(0, 20)}... where the query is `
        + `${sequence.length} `
        + `reading ${sequence.slice(0, 20)}...; this tool maps them residue for `
        + "residue and has no aligner");
    }
    // `length` is not in scope until after the fold; the query's own length is.
    templateSlot = templateSlotAtom37({ structure, tokens: sequence.length,
                                        map: identityMap(structure) });
    // 🔴 `--template-no-sidechains` IS AF2BIND's "nosc", AND IT KEEPS C-BETA.
    // ColabDesign's `rm_target_sc` masks `template_all_atom_mask[..., 5:]`
    // under its own comment "remove sidechains (mask anything beyond CB)", and
    // atom37 slots 0..4 are N, CA, C, CB, O - so CB SURVIVES. That matters for
    // anyone scoring AF2BIND's head: the monomer's pseudo-beta is CB for
    // everything but glycine, so the distogram the head was trained on is
    // CB-based, and a template stripped down to the backbone would be a
    // different feature than the one it saw.
    //
    // It masks rather than moves the atoms, which is what ColabDesign does: the
    // coordinates stay and the mask decides what the geometry reads.
    if (args.includes("--template-no-sidechains")) {
      const slots = 37;
      for (let token = 0; token < sequence.length; token += 1) {
        for (let index = 5; index < slots; index += 1) {
          templateSlot.atomMask[token * slots + index] = 0;
        }
      }
    }
  }

  // The deposited alpha carbons to score against, from the chain already read.
  const truth = targetStructure === undefined ? undefined
    : targetStructure.residues.map((residue) => residue.atoms.get("CA"))
      .filter((point) => point !== undefined);

  resetFeatureStats();
  const started = performance.now();
  const prediction = await new (multimer ? AlphaFoldUnifiedGpu : AlphaFoldMonomerGpu)(device)
    .predictA3m(
      a3m, weights, featureTables,
      { recycles, randomSeed: seed, maxMsaSequences: rows, maxExtraSequences: extraRows, hostFeaturisation,
        template: templateSlot,
        // ...the arm for measuring what deduplication is worth; see
        // planA3mFeatures. Default on, matching AlphaFold's make_msa_features.
        deduplicateMsa: !args.includes("--no-dedupe"),
        chainLengths: chains, ...regime },
      paeBreaks, undefined, onProgress,
    );
  const elapsed = Math.round(performance.now() - started);
  // 🔴 FOLD IT AGAIN, REUSING NOTHING, WHICH IS WHAT A PAGE DOES. The first
  // fold in a process is mostly pipeline compilation and first touch - at 59
  // residues it is 1.14 s of which 0.95 is warm-up - so a single number cannot
  // price weight residency at all, and residency is what a second fold gets.
  // Every repeat is held to the FIRST fold's atoms: same input, same seed, same
  // structure, or the residency returned something else and the clock would
  // have called that a win.
  const repeat = Number(option(args, "repeat", "1"));
  const repeats = [];
  const checksumOf = (atoms) => {
    let sum = 0;
    for (let i = 0; i < atoms.length; i += 1) sum = (sum + Math.round(atoms[i] * 1000)) | 0;
    return sum;
  };
  const firstChecksum = checksumOf(prediction.final.structure.atom37);
  for (let again = 1; again < repeat; again += 1) {
    const at = performance.now();
    const other = await new (multimer ? AlphaFoldUnifiedGpu : AlphaFoldMonomerGpu)(device)
      .predictA3m(
        a3m, weights, featureTables,
        { recycles, randomSeed: seed, maxMsaSequences: rows, maxExtraSequences: extraRows, hostFeaturisation,
          template: templateSlot,
          deduplicateMsa: !args.includes("--no-dedupe"),
          chainLengths: chains, ...regime },
        paeBreaks, undefined, undefined,
      );
    const checksum = checksumOf(other.final.structure.atom37);
    repeats.push({ milliseconds: Math.round(performance.now() - at), checksum,
                   sameFold: checksum === firstChecksum });
    if (checksum !== firstChecksum) {
      throw new Error(`fold ${again + 1} returned checksum ${checksum} where the first`
        + ` returned ${firstChecksum}; the same input at the same seed must fold the same`);
    }
  }
  const final = prediction.final;
  const length = sequence.length;
  const atom37 = final.structure.atom37;

  // 🔴 SCORED AGAINST THE DEPOSITION, WHICH IS THE ONLY THING A TEMPLATE GATE
  // CAN READ. `superpose` fits one chain onto another and returns RMSD and TM;
  // it needs the same number of alpha carbons on both sides, so a target whose
  // file resolves fewer residues than the query has is refused rather than
  // scored against a silent truncation.
  let scored;
  if (truth !== undefined) {
    const modelCa = [];
    for (let residue = 0; residue < length; residue += 1) {
      modelCa.push([atom37[(residue * 37 + 1) * 3], atom37[(residue * 37 + 1) * 3 + 1],
                    atom37[(residue * 37 + 1) * 3 + 2]]);
    }
    if (truth.length !== modelCa.length) {
      throw new Error(`--target=${targetName} resolves ${truth.length} alpha carbons `
        + `and the fold has ${modelCa.length}; scoring them would compare `
        + "different residues");
    }
    scored = superpose(modelCa, truth);
  }

  // Consecutive alpha carbons, which is atom 1 of the 37.
  //
  // 🔴 THE CHAIN JUNCTIONS ARE SKIPPED, OR THE METRIC MEASURES NOTHING. Two
  // chains are not bonded to each other, so the step across the boundary is
  // whatever the fold placed them at - 18.4 A on the first run of this - and it
  // would sit in `worst` for ever looking like a broken backbone.
  const breaks = new Set();
  let boundary = 0;
  for (const chain of chains.slice(0, -1)) { boundary += chain; breaks.add(boundary - 1); }
  const distances = [];
  for (let residue = 0; residue + 1 < length; residue += 1) {
    if (breaks.has(residue)) continue;
    const a = (residue * 37 + 1) * 3;
    const b = ((residue + 1) * 37 + 1) * 3;
    distances.push(Math.hypot(
      atom37[a] - atom37[b], atom37[a + 1] - atom37[b + 1], atom37[a + 2] - atom37[b + 2]));
  }
  const sorted = [...distances].sort((x, y) => x - y);
  const median = sorted[Math.floor(sorted.length / 2)];
  const worst = distances.reduce((far, value) =>
    Math.abs(value - 3.8) > Math.abs(far - 3.8) ? value : far, 3.8);

  // A checksum over every coordinate, so two trees can be compared in one
  // number before anyone looks at the geometry. Scaled and summed as integers,
  // because a float sum of a million terms is not reproducible in itself.
  let checksum = 0;
  for (let index = 0; index < atom37.length; index += 1) {
    checksum = (checksum + Math.round(atom37[index] * 1000)) | 0;
  }

  // 🔴 AND NOW IT IS A GATE, NOT A REPORT. This tool has printed `caca` since it
  // was written and nothing ever asserted on it, so an 825-residue fold whose
  // whole chain had collapsed into a ball two angstroms across - consecutive
  // alpha carbons 0.06 A apart - passed as "the same fold" for the length of a
  // campaign, while pLDDT climbed to 69.31 and pTM to 0.9672 and said it was
  // fine. See docs/AF2.md. Consecutive CA are 3.80 A apart in any real chain;
  // the bands below are wide enough for a bad PREDICTION and far too narrow for
  // a broken one. Measured, healthy: median 3.485 to 3.972, worst 1.69 to 4.55.
  // Measured, broken: median 1.44 to 3.41, worst 0.06 or 7.73 to 70.45.
  const chainOk = median >= 3.4 && median <= 4.2 && Math.abs(worst - 3.8) <= 2.8;
  if (!chainOk && option(args, "allow-broken-geometry", null) === null) {
    throw new Error(
      `the fold is not a chain: consecutive CA median ${median.toFixed(3)} A, worst `
      + `${worst.toFixed(2)} A, against 3.80 expected. pLDDT says `
      + `${final.confidence.meanPlddt.toFixed(2)} and it is not a correctness gate - see `
      + "docs/AF2.md. Pass --allow-broken-geometry to report anyway.");
  }

  const round = (value, places = 4) => Number(value.toFixed(places));
  return {
    sequence: sequence.length > 24 ? `${sequence.slice(0, 24)}...(${length})` : sequence,
    family, chains, length, rows, extraRows, recycles, seed,
    weightLoadMs: loadMs, elapsedMilliseconds: elapsed, repeats,
    // Where "features" in the phase table actually goes; see featureStats.
    featureMilliseconds: Object.fromEntries(Object.entries(featureStats)
      .map(([key, value]) => [key, key === "calls" ? value : Math.round(value)])),
    packBy: Object.fromEntries(Object.entries(globalThis.__pk ?? {}).map(([k,v]) => [k, Math.round(v)]).sort((a,b)=>b[1]-a[1])),
    // What the fold left on the device, and in what - the totals alone cannot
    // say which tensor to attack. See src/runtime/device-memory.js.
    deviceMemory: trimMemory(memorySnapshot(device)),
    meanPlddt: round(final.confidence.meanPlddt, 3),
    ptm: round(final.confidence.ptm, 4),
    ...(final.confidence.iptm === undefined ? {} : { iptm: round(final.confidence.iptm, 4) }),
    caca: { median: round(median, 3), worst: round(worst, 3), ok: chainOk },
    stages: summariseStages(stageMarks, started),
    phases: prediction.stageMilliseconds === undefined ? undefined
      : Object.fromEntries(Object.entries(prediction.stageMilliseconds)
        .map(([name, ms]) => [name, Number((ms / 1000).toFixed(2))])),
    checksum,
    // 🔴 A TARGET SCORE, BECAUSE A TEMPLATE GATE CANNOT BE BUILT ON pLDDT. This
    // tool reported confidence, a checksum and chain geometry and nothing that
    // says whether the fold is the RIGHT one - and the template question is
    // exactly "did it land on the structure it was given". `--target=<name>`
    // scores the alpha carbons against `tools/fixtures/<name>-crystal.pdb`,
    // which is how `fold-opendde.js` has always answered it for the AF3 side.
    // 🔴 UNDER `scored`, THE SHAPE `fold-opendde.js` ALREADY REPORTS, so one
    // gate can read both tools. A second spelling of the same number is how
    // a checker comes to read `undefined` and pass.
    ...(scored === undefined ? {}
      : { scored: { rmsd: round(scored.rmsd, 3), tm: round(scored.tm, 4) } }),
    // The first and last CA, so a difference has somewhere to be looked at.
    firstCa: [0, 1, 2].map((axis) => round(atom37[1 * 3 + axis], 3)),
    lastCa: [0, 1, 2].map((axis) => round(atom37[((length - 1) * 37 + 1) * 3 + axis], 3)),
  };
}
