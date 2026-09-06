// A whole ESMFold2 fold on the GPU: a sequence in, a structure out.
//
//     node tools/gpu-chrome.mjs tools/gpu/fold-esmfold2.js \
//       --sequence=MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQ --sampler=diffusion-15
//
// 🔴 THIS IS THE PORT FOLDING, NOT A DIFFERENTIAL. Every stage is LocalFold's
// own - ESM-C's 36 blocks, the shim's pair, the featuriser, the inputs
// embedder, twenty-four trunk blocks four times over, the conditioning, the
// token transformer, the atom decoder and the sampler - and the only input is
// the sequence. Each stage has been checked against the native model
// separately; what this answers is whether they compose into a structure.
//
// 🔴 AND THE GEOMETRY IS THE GATE, NOT THE COORDINATES. The sampler draws a
// rotation, a translation and a noise vector per step from its own RNG, so this
// cannot reproduce ESMFold2's coordinates and a checker that tried would be
// measuring the RNG. What it reports instead is the CA-CA spacing - 3.8 A is a
// peptide bond, and a port with the arithmetic subtly wrong gives a plausible
// cloud at the wrong scale - and, when a reference structure is supplied, the
// RMSD after superposition.
import { readTensor, readTensorAsFloat16 } from "../../src/reference/dtype.js";
import { GpuBufferAllocator } from "../../src/runtime/allocator.js";
import { EsmcTowerGpu } from "../../src/esmc/tower-webgpu.js";
import { foldEsmfold2, SAMPLER_PRESETS } from "../../src/esmfold2/fold.js";
import {
  atomDecoderWeights, atomEncoderWeights, denoiserWeights, featuriserWeights,
  trunkBlockWeights,
} from "../../src/esmfold2/weights.js";
import { SHIM_PAIR_TENSORS } from "../../src/esmfold2/language-pair-webgpu.js";
import { weightedRigidAlign } from "../../src/esmfold2/sampler-reference.js";
import { ccdUrl, parseCcdComponent } from "../../src/af3/ccd-component.js";
import { toDensePositions } from "../../src/esmfold2/featurise.js";
import { toPdb } from "../../src/af3/fold.js";
import {
  CONTACT_EDGES, contactAngstromsFor,
} from "../../src/esmfold2/distogram-webgpu.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const BLOCK_LEAVES = ["attn_norm/scale", "attn_norm/offset", "qkv/weights",
  "q_norm/scale", "k_norm/scale", "attn_out/weights", "ffn_norm/scale",
  "ffn_norm/offset", "fc1/weights", "fc2/weights"];
/** The four matrices the tower stores at half precision anyway. */
const NARROW_LEAVES = new Set(["qkv/weights", "attn_out/weights",
  "fc1/weights", "fc2/weights"]);
const TOWER_SHARED = ["embed/weights", "final_norm/scale", "lm/combine", "lm/norm/scale",
  "lm/norm/offset", "lm/projection/weights", "lm/downproject/weights",
  "lm/downproject/bias"];

function reader(bundle) {
  const shards = new Map();
  let table;
  // 🔴 CLOSURES, NOT METHODS, BECAUSE `read` IS PASSED AROUND. Every weight
  // helper takes `read` as a bare function; an object method loses `this` the
  // moment it is handed over, and the failure - "cannot read properties of
  // undefined" inside a shard lookup - names neither the bundle nor the tensor.
  const manifest = async () => {
    if (table === undefined) table = await (await fetch(`${bundle}/manifest.json`)).json();
    return table;
  };
  const read = async (name, half = false) => {
    const loaded = await manifest();
    const record = loaded.tensors[name];
    if (record === undefined) throw new Error(`${bundle} has no tensor ${name}`);
    if (!shards.has(record.file)) {
      shards.set(record.file, await (await fetch(`${bundle}/${record.file}`)).arrayBuffer());
    }
    return half
      ? readTensorAsFloat16(record, shards.get(record.file), record.byteOffset ?? 0)
      : readTensor(record, shards.get(record.file), record.byteOffset ?? 0, true);
  };
  return { manifest, read };
}

/** Alpha carbons, by name: 'CA' is [35, 33, 0, 0] under chr(code + 32). */
function alphaCarbons(features) {
  const names = features.refAtomNameChars;
  const out = [];
  for (let atom = 0; atom < features.atoms; atom += 1) {
    if (features.mask[atom] !== 0 && names[atom * 4] === 35 && names[atom * 4 + 1] === 33
        && names[atom * 4 + 2] === 0) out.push(atom);
  }
  return out;
}

export async function main(device, args = []) {
  // 🔴 CHAINS ARE COLON-JOINED AND THEIR KINDS ARE A SEPARATE LIST, because
  // the letters cannot say which is which: `A`, `C` and `G` are alanine,
  // cysteine and glycine in a protein chain and adenine, cytosine and guanine
  // in a nucleic one.
  const sequence = option(args, "sequence",
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQ");
  const kinds = option(args, "kinds", "");
  // 🔴 A LIGAND IS ONE TOKEN PER HEAVY ATOM AND ITS GEOMETRY COMES FROM THE
  // CCD, over the network. `parseCcdComponent` is AF3's reader and the
  // components are the same ones - ESMFold2 reads the same dictionary through
  // RDKit rather than through mmCIF, and takes the ideal conformer for the same
  // reason.
  const ligandCodes = option(args, "ligands", "").split(",").filter((c) => c !== "");
  // 🔴 PRICED AGAINST THE SAMPLER'S OWN SPREAD, NOT AGAINST ZERO. The trunk's
  // two f16 knobs - the transition's staged tiles and the triangle projection's
  // accumulators - are worth 1.09x and 1.17x at 150 tokens, and the question is
  // not what they cost a tensor norm but whether the STRUCTURE notices. Written
  // `staged:accumulate`, or one name for both.
  const trunkPrecision = option(args, "trunk-precision", "");
  const submissionWindow = Number(option(args, "submission-window", "16"));
  const foldBundle = option(args, "bundle", "/model-esmfold2-trunk-f32");
  const towerBundle = option(args, "esmc", "/model-esmc-600m-int3");
  const sampler = option(args, "sampler", "diffusion-15");
  const seed = Number(option(args, "seed", "0"));
  // 🔴 THE REFERENCE IS A DIFFERENT SAMPLE, NOT A DIFFERENT ANSWER. The sampler
  // draws a rotation, a translation and a noise vector per step; this uses its
  // own RNG, so the honest comparison is RMSD after superposition and a few
  // angstroms is what agreement LOOKS like here.
  const reference = option(args, "reference", "");
  // 🔴 THE 8 A CONTACT THRESHOLD IS A RESIDUE-RESIDUE CONVENTION AND A LIGAND
  // TOKEN IS NOT A RESIDUE. `--contact-sweep` keeps the whole distogram - 46
  // MiB at 300 tokens - and reports, per pair KIND, what each threshold would
  // call a contact and how well the head predicts a DISTANCE at all.
  const contactSweep = args.includes("--contact-sweep");
  // 🔴 `lm_mask_pct`, WHICH THIS CHECKPOINT SETS TO ZERO. The knob exists so the
  // port is complete for a checkpoint that does set it, and so the cost of the
  // training-time corruption can be MEASURED rather than guessed at.
  const lmMask = Number(option(args, "lm-mask", "0"));
  if (SAMPLER_PRESETS[sampler] === undefined) {
    throw new Error(`unknown sampler ${sampler}; `
      + `expected one of ${Object.keys(SAMPLER_PRESETS).join(", ")}`);
  }

  const fold = reader(foldBundle);
  const tower = reader(towerBundle);
  const manifest = await fold.manifest();
  const towerTable = await tower.manifest();
  const towerManifest = towerTable.languageModel ?? {};
  if (towerManifest.shim !== manifest.trunk?.source) {
    throw new Error(`the tower's shim is for ${towerManifest.shim} `
      + `and the folding bundle is ${manifest.trunk?.source}; the shim is per model`);
  }
  const M = manifest.trunk;

  // 🔴 THE SHIM'S SINGLE HALF IS NEEDED EVEN THOUGH THE TOWER COMPUTES IT.
  // A non-protein token never reaches the tower, and its hidden state is ZERO
  // rather than absent - so the fold has to evaluate the shim's single half at
  // zero itself. Three tensors, one row of arithmetic, once.
  const shim = {};
  for (const name of [...SHIM_PAIR_TENSORS, "lm/norm/offset", "lm/projection/weights",
                      "lm/downproject/weights", "lm/downproject/bias"]) {
    shim[name] = await tower.read(name);
  }
  const [featuriser, inputsEmbedder, denoiser, encoder, decoder] = await Promise.all([
    featuriserWeights(fold.read),
    atomEncoderWeights(fold.read, "atom", M.atomBlocks),
    denoiserWeights(fold.read, { tokenBlocks: M.tokenBlocks }),
    atomEncoderWeights(fold.read, "diffusionAtomEncoder", M.atomBlocks,
                       { withCoordinates: true }),
    atomDecoderWeights(fold.read, "diffusionAtomDecoder", M.atomBlocks),
  ]);
  denoiser.encoder = encoder;
  denoiser.decoder = decoder;
  const trunkBlocks = [];
  for (let layer = 0; layer < M.blocks; layer += 1) {
    trunkBlocks.push(await trunkBlockWeights(fold.read, layer));
  }

  // 🔴 THE TOWER STREAMS ITS BLOCKS AND THIS MUST NOT DEFEAT THAT. Reading all
  // 36 up front is 2190 MiB of float32 in the tab, which is the thing the
  // streaming design exists to avoid.
  const towerShared = {};
  for (const name of TOWER_SHARED) towerShared[name] = await tower.read(name);
  const allocator = new GpuBufferAllocator(device);
  const runTower = async (ids, sequenceId, onBlock) => {
    const engine = new EsmcTowerGpu(device, allocator);
    const result = await engine.run(ids, {
      rows: ids.length, model: towerManifest.width,
      heads: towerManifest.heads ?? towerManifest.width / 64,
      ffn: towerTable.tensors["blocks/0/fc2/weights"].shape[0],
      layers: towerManifest.layers, pair: M.pairChannels,
      residualScale: towerManifest.residualScale ?? 1,
    }, async (layer) => {
      const weights = {};
      for (const leaf of BLOCK_LEAVES) {
        weights[leaf] = await tower.read(`blocks/${layer}/${leaf}`,
                                         NARROW_LEAVES.has(leaf));
      }
      return weights;
    }, towerShared, { sequenceId, onBlock });
    return result.single;
  };

  const ligands = [];
  for (const code of ligandCodes) {
    const text = await (await fetch(ccdUrl(code))).text();
    ligands.push(parseCcdComponent(text));
  }

  const progress = [];
  const updates = new Map();
  const result = await foldEsmfold2(device, {
    sequence, allocator, seed, sampler,
    entities: (kinds === "" && ligands.length === 0) ? sequence
      : { sequence, ...(kinds === "" ? {} : { chainKinds: kinds.split(",") }),
          ...(ligands.length === 0 ? {} : { ligands }) },
    shape: { ...M, loops: (M.loops ?? 3) + 1 },
    submissionWindow,
    trunk: trunkPrecision === "" ? {} : {
      stagedPrecision: trunkPrecision.split(":")[0],
      accumulatePrecision: trunkPrecision.split(":")[1] ?? trunkPrecision.split(":")[0],
    },
    weights: { featuriser, inputsEmbedder, trunkBlocks, denoiser, shim },
    tower: runTower,
    distogramLogits: contactSweep,
    lmMaskFraction: lmMask,
    onStatus: (label) => { progress.push(label); },
    // 🔴 COUNTED PER PHASE, because "does the trunk report block by block" is a
    // number and not an impression. A bar sampled from the page cannot answer
    // it: 96 block updates inside two seconds are far more frequent than any
    // poll, so the trace shows a handful of values whatever the code does.
    onProgress: () => {
      const phase = progress[progress.length - 1]?.split(" · ")[0] ?? "?";
      updates.set(phase, (updates.get(phase) ?? 0) + 1);
    },
  });

  // ---- what came out.
  const alphas = alphaCarbons(result.features);
  const x = result.coordinates;
  // 🔴 THE SPACING IS WITHIN A CHAIN, NOT ALONG THE ARRAY. Two chains are two
  // molecules and the distance across the break is whatever the sampler placed
  // them at - 22.9 A on the first complex run here, which read as a broken fold
  // and is the metric walking off the end of chain one.
  const chainOfAtom = result.features.asymId;
  const tokenOfAtom = result.features.atomToToken;
  const spacing = [];
  for (let i = 1; i < alphas.length; i += 1) {
    const previous = alphas[i - 1], atom = alphas[i];
    if (chainOfAtom[tokenOfAtom[previous]] !== chainOfAtom[tokenOfAtom[atom]]) continue;
    const a = previous * 3, b = atom * 3;
    spacing.push(Math.hypot(x[b] - x[a], x[b + 1] - x[a + 1], x[b + 2] - x[a + 2]));
  }
  const mean = spacing.length === 0 ? NaN
    : spacing.reduce((t, v) => t + v, 0) / spacing.length;
  // 🔴 A NUCLEIC CHAIN'S GEOMETRY GATE IS THE PHOSPHODIESTER BOND, NOT CA-CA.
  // O3' of one nucleotide to P of the next is about 1.6 A, and it is the same
  // kind of statement 3.8 A is for a peptide: a covalent distance no torsion
  // can change, which a port with the arithmetic subtly wrong gets wrong.
  const named = (atom, text) => {
    const chars = result.features.refAtomNameChars;
    for (let i = 0; i < 4; i += 1) {
      const wanted = i < text.length ? text.charCodeAt(i) - 32 : 0;
      if (chars[atom * 4 + i] !== wanted) return false;
    }
    return true;
  };
  const phosphodiester = [];
  for (let atom = 1; atom < result.atoms; atom += 1) {
    if (!named(atom, "P")) continue;
    for (let back = atom - 1; back >= 0 && back > atom - 30; back -= 1) {
      if (!named(back, "O3'")) continue;
      const token = result.features.atomToToken;
      if (result.features.asymId[token[back]] !== result.features.asymId[token[atom]]) break;
      phosphodiester.push(Math.hypot(x[atom * 3] - x[back * 3],
        x[atom * 3 + 1] - x[back * 3 + 1], x[atom * 3 + 2] - x[back * 3 + 2]));
      break;
    }
  }
  let rmsd;
  if (reference !== "") {
    const dump = await (await fetch(reference)).json();
    const native = Float32Array.from(dump.coordinates);
    if (native.length !== x.length) {
      throw new Error(`${reference} has ${native.length / 3} atoms for ${result.atoms}`);
    }
    const aligned = weightedRigidAlign(x, native, result.features.mask, result.atoms);
    let squared = 0, live = 0;
    for (let atom = 0; atom < result.atoms; atom += 1) {
      if (result.features.mask[atom] === 0) continue;
      live += 1;
      for (let axis = 0; axis < 3; axis += 1) {
        const d = aligned[atom * 3 + axis] - native[atom * 3 + axis];
        squared += d * d;
      }
    }
    rmsd = Math.sqrt(squared / live);
  }
  // 🔴 THE STRUCTURE GOES THROUGH AF3's WRITER, NOT A SECOND ONE. `toPdb`
  // already knows a ligand is HETATM under its component's code, that a
  // nucleotide is " DA" rather than "ALA", that a modified residue takes its
  // own code, and that a complex needs one chain letter per asym id. Permuting
  // the coordinates back into its dense layout is cheaper than any of that.
  const pdb = toPdb(result.features.batch, toDensePositions(result.features, x));

  // 🔴 THE DISTOGRAM AND THE STRUCTURE ARE TWO INDEPENDENT READINGS OF ONE
  // TRUNK, so their agreement is a gate neither can give alone. The distogram
  // head is a single projection off the pair; the coordinates came through the
  // conditioning, twelve token blocks, two atom stacks and a stochastic
  // sampler. If the borrowed bin edges were badly wrong, or either head were
  // mis-wired, the two would not agree - and a fold can be geometrically
  // perfect while being the wrong fold, which CA-CA cannot see.
  let contactPairs = 0;
  let agreement;
  // 🔴 THE REPRESENTATIVE ATOM AND THE PARTNER RULE ARE SHARED, because the
  // threshold sweep below scores the same pairs this does. Two copies of a
  // pseudo-beta rule is two chances to disagree with the distogram's own
  // convention, and a comparison against the wrong representative compares
  // nothing.
  const representative = new Int32Array(result.tokens).fill(-1);
  const { molType, asymId, residueIndex } = result.features;
  const ligand = (t) => molType[t] === 3;
  // ...the same rule the certainty uses, and for the same reason: a separation
  // on the TOKEN index is a rule about a chain's neighbours, and a ligand's
  // atoms are neither.
  const neighbours = (i, j) => asymId[i] === asymId[j]
    && Math.abs(residueIndex[i] - residueIndex[j]) <= 6;
  // ...and the OBSERVED side takes the same per-kind threshold the contact map
  // does, or the two halves of a precision are answering different questions.
  const KIND = ["protein", "nucleic", "nucleic", "ligand"];
  const { residueType } = result.features;
  const cutoff = (i, j) => contactAngstromsFor(molType[i], molType[j],
                                               residueType[i], residueType[j]);
  if (result.contacts !== undefined) {
    for (let atom = 0; atom < result.atoms; atom += 1) {
      if (result.features.mask[atom] === 0) continue;
      const token = result.features.atomToToken[atom];
      // CB where there is one, else CA, else the token's first atom - which is
      // the usual pseudo-beta rule and what a ligand token has anyway.
      if (representative[token] < 0) representative[token] = atom;
      if (named(atom, "CB")) representative[token] = atom;
      else if (named(atom, "CA") && !named(representative[token], "CB")) {
        representative[token] = atom;
      }
    }
    let both = 0, predicted = 0, actual = 0;
    // 🔴 SPLIT BY WHAT THE PAIR IS, because a ligand is ONE TOKEN PER HEAVY
    // ATOM: its atoms are all within a few angstroms of each other, so every
    // one of its k^2 self-pairs is "in contact" and says nothing, and `j >= i +
    // 6` - a rule about a chain's own neighbours - excludes an arbitrary
    // prefix of them rather than a sequence neighbourhood. Counted rather than
    // argued: `kinds` is polymer/polymer, polymer/ligand, ligand/ligand and
    // each ligand's own self-pairs.
    // ...the same six names `tools/calibrate-contact-cutoff.py` reports, so the
    // fold's numbers and the calibration's are read side by side.
    const kinds = {};
    const bucketFor = (i, j) => {
      const name = [KIND[molType[i]], KIND[molType[j]]].sort().join("-");
      if (kinds[name] === undefined) {
        // ...no `cutoff` field: a ligand-protein pair's threshold is the
        // RESIDUE's, so one number here would be whichever pair came first.
        kinds[name] = { predicted: 0, actual: 0, both: 0 };
      }
      return kinds[name];
    };
    for (let i = 0; i < result.tokens; i += 1) {
      for (let j = i + 1; j < result.tokens; j += 1) {
        if (neighbours(i, j)) continue;
        const near = result.contacts[i * result.tokens + j] > 0.5;
        if (near) contactPairs += 1;
        const a = representative[i] * 3, b = representative[j] * 3;
        const close = Math.hypot(x[b] - x[a], x[b + 1] - x[a + 1],
                                 x[b + 2] - x[a + 2]) < cutoff(i, j);
        if (near) predicted += 1;
        if (close) actual += 1;
        if (near && close) both += 1;
        const bucket = bucketFor(i, j);
        if (near) bucket.predicted += 1;
        if (close) bucket.actual += 1;
        if (near && close) bucket.both += 1;
      }
    }
    // ...and how many pairs the separation rule DROPS inside one ligand, which
    // is the other half of the objection: for a polymer those are real
    // neighbours, for a ligand they are an arbitrary third of the molecule.
    let ligandTokens = 0, ligandPairsKept = 0, ligandPairsDropped = 0;
    for (let i = 0; i < result.tokens; i += 1) {
      if (!ligand(i)) continue;
      ligandTokens += 1;
      for (let j = i + 1; j < result.tokens; j += 1) {
        if (!ligand(j) || asymId[i] !== asymId[j]) continue;
        if (neighbours(i, j)) ligandPairsDropped += 1; else ligandPairsKept += 1;
      }
    }
    agreement = { predicted, actual, both,
                  precision: predicted === 0 ? null : both / predicted,
                  recall: actual === 0 ? null : both / actual,
                  kinds, ligandTokens, ligandPairsKept, ligandPairsDropped };
  }

  // 🔴 THE THRESHOLD, SWEPT PER KIND, plus whether the head predicts a DISTANCE
  // for the pairs at all - which is the question a threshold cannot answer. A
  // head that says nothing about protein-ligand pairs gives a flat correlation
  // at every threshold, and one whose threshold is merely wrong does not.
  let sweep;
  if (contactSweep && result.distogram !== undefined) {
    const { logits, bias } = result.distogram;
    const bins = bias.length;
    const width = (CONTACT_EDGES.maximum - CONTACT_EDGES.minimum) / bins;
    const centre = (bin) => CONTACT_EDGES.minimum + (bin + 0.5) * width;
    const thresholds = [4, 5, 6, 8, 10, 12];
    const blank = () => ({ pairs: 0, sums: { x: 0, y: 0, xx: 0, yy: 0, xy: 0 },
      at: Object.fromEntries(thresholds.map((t) =>
        [t, { predicted: 0, actual: 0, both: 0 }])) });
    const buckets = {};
    const probability = new Float64Array(bins);
    for (let i = 0; i < result.tokens; i += 1) {
      for (let j = i + 1; j < result.tokens; j += 1) {
        if (neighbours(i, j)) continue;
        const name = [KIND[molType[i]], KIND[molType[j]]].sort().join("-");
        if (buckets[name] === undefined) buckets[name] = blank();
        const bucket = buckets[name];
        // ...the head symmetrises, so the pair is read once.
        const base = (i * result.tokens + j) * bins;
        let peak = -Infinity;
        for (let bin = 0; bin < bins; bin += 1) {
          const value = logits[base + bin] + bias[bin];
          probability[bin] = value;
          if (value > peak) peak = value;
        }
        let total = 0;
        for (let bin = 0; bin < bins; bin += 1) {
          probability[bin] = Math.exp(probability[bin] - peak);
          total += probability[bin];
        }
        let mode = 0, best = -1;
        for (let bin = 0; bin < bins; bin += 1) {
          probability[bin] /= total;
          if (probability[bin] > best) { best = probability[bin]; mode = bin; }
        }
        const a = representative[i] * 3, b = representative[j] * 3;
        const observed = Math.hypot(x[b] - x[a], x[b + 1] - x[a + 1], x[b + 2] - x[a + 2]);
        const predictedDistance = centre(mode);
        bucket.pairs += 1;
        bucket.sums.x += predictedDistance;
        bucket.sums.y += observed;
        bucket.sums.xx += predictedDistance * predictedDistance;
        bucket.sums.yy += observed * observed;
        bucket.sums.xy += predictedDistance * observed;
        for (const threshold of thresholds) {
          let mass = 0;
          for (let bin = 0; bin < bins; bin += 1) {
            if (centre(bin) < threshold) mass += probability[bin];
          }
          const cell = bucket.at[threshold];
          if (mass > 0.5) cell.predicted += 1;
          if (observed < threshold) cell.actual += 1;
          if (mass > 0.5 && observed < threshold) cell.both += 1;
        }
      }
    }
    sweep = Object.fromEntries(Object.entries(buckets)
      .filter(([, v]) => v.pairs > 0)
      .map(([name, v]) => {
        const n = v.pairs, { x: sx, y: sy, xx, yy, xy } = v.sums;
        const denominator = Math.sqrt((n * xx - sx * sx) * (n * yy - sy * sy));
        return [name, {
          pairs: n,
          meanPredicted: sx / n, meanObserved: sy / n,
          distanceCorrelation: denominator === 0 ? null
            : (n * xy - sx * sy) / denominator,
          at: Object.fromEntries(Object.entries(v.at).map(([t, c]) => [t, {
            ...c, precision: c.predicted === 0 ? null : c.both / c.predicted,
            recall: c.actual === 0 ? null : c.both / c.actual }])),
        }];
      }));
  }

  // 🔴 THE CERTAINTY MIXES INTRA- AND INTER-CHAIN PARTNERS, AND NOTHING HAS
  // EVER CHECKED WHETHER IT SHOULD. The shader excludes only same-chain
  // SEQUENCE neighbours, so a residue on a complex is judged partly on pairs
  // across the interface - and every target in the 11,400-arm sweep that chose
  // its constants was a single protein chain. This recomputes it on the host
  // three ways over the same distogram, which is the comparison that says
  // whether the mixing changes the answer.
  let certaintyByChain;
  if (contactSweep && result.distogram !== undefined) {
    const { logits, bias } = result.distogram;
    const bins = bias.length;
    const width = (CONTACT_EDGES.maximum - CONTACT_EDGES.minimum) / bins;
    const centre = (bin) => CONTACT_EDGES.minimum + (bin + 0.5) * width;
    const RADIUS = 2, SEPARATION = 3, CUTOFF = 12;
    const n = result.tokens;
    const probability = new Float64Array(bins);
    // ...one pass over the pairs, three accumulators per token.
    const arms = ["all", "intra", "inter"];
    const sums = Object.fromEntries(arms.map((a) => [a, new Float64Array(n)]));
    const counts = Object.fromEntries(arms.map((a) => [a, new Float64Array(n)]));
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        if (i === j) continue;
        const sameChain = asymId[i] === asymId[j];
        if (sameChain && Math.abs(residueIndex[i] - residueIndex[j]) <= SEPARATION) continue;
        const base = (i * n + j) * bins;
        let peak = -Infinity;
        for (let bin = 0; bin < bins; bin += 1) {
          const value = logits[base + bin] + bias[bin];
          probability[bin] = value;
          if (value > peak) peak = value;
        }
        let total = 0;
        for (let bin = 0; bin < bins; bin += 1) {
          probability[bin] = Math.exp(probability[bin] - peak);
          total += probability[bin];
        }
        let mode = 0, best = -1;
        for (let bin = 0; bin < bins; bin += 1) {
          probability[bin] /= total;
          if (probability[bin] > best) { best = probability[bin]; mode = bin; }
        }
        if (centre(mode) >= CUTOFF) continue;
        let mass = 0;
        for (let bin = 0; bin < bins; bin += 1) {
          if (Math.abs(centre(bin) - centre(mode)) <= RADIUS) mass += probability[bin];
        }
        for (const arm of arms) {
          if (arm === "intra" && !sameChain) continue;
          if (arm === "inter" && sameChain) continue;
          sums[arm][i] += mass;
          counts[arm][i] += 1;
        }
      }
    }
    const chains = [...new Set(asymId)].sort((a, b) => a - b);
    certaintyByChain = chains.map((chain) => {
      const row = { chain, tokens: 0 };
      for (const arm of arms) {
        let sum = 0, seen = 0;
        for (let i = 0; i < n; i += 1) {
          if (asymId[i] !== chain || counts[arm][i] === 0) continue;
          sum += sums[arm][i] / counts[arm][i];
          seen += 1;
        }
        row[arm] = seen === 0 ? null : Number((sum / seen).toFixed(4));
        if (arm === "all") row.tokens = seen;
      }
      return row;
    });
  }

  return {
    sequence, sampler, seed, trunkPrecision, contactSweep: sweep, certaintyByChain,
    lmMask: result.lmMask,
    // ...what the shipped shader now separates, so the host arm above and the
    // kernel can be compared rather than trusted.
    shippedByChain: result.interfaceCertainty === undefined ? undefined
      : [...new Set(asymId)].sort((a, b) => a - b).map((chain) => {
        const mean = (values, keep) => {
          let sum = 0, seen = 0;
          for (let t = 0; t < result.tokens; t += 1) {
            if (asymId[t] !== chain || !keep(values[t])) continue;
            sum += values[t]; seen += 1;
          }
          return seen === 0 ? null : Number((sum / seen).toFixed(4));
        };
        return { chain,
          within: mean(result.certainty, (v) => v >= 0),
          across: mean(result.interfaceCertainty, (v) => v >= 0) };
      }),
    tokens: result.tokens, atoms: result.atoms, steps: result.steps,
    alphaCarbons: alphas.length,
    caSpacing: spacing.length === 0 ? null
      : { mean, min: Math.min(...spacing), max: Math.max(...spacing) },
    phosphodiester: phosphodiester.length === 0 ? null : {
      bonds: phosphodiester.length,
      mean: phosphodiester.reduce((t, v) => t + v, 0) / phosphodiester.length,
      min: Math.min(...phosphodiester), max: Math.max(...phosphodiester) },
    rmsdToReference: rmsd,
    elapsedSeconds: result.elapsedMilliseconds / 1000,
    timings: result.timings,
    peakMebibytes: result.memory.peakBytes / 1048576,
    stages: [...new Set(progress.map((p) => p.split(" · ")[0]))],
    progressEvents: Object.fromEntries(updates),
    longRangeContacts: contactPairs,
    certainty: result.certainty === undefined ? undefined : {
      mean: [...result.certainty].reduce((t, v) => t + v, 0) / result.certainty.length,
      min: Math.min(...result.certainty), max: Math.max(...result.certainty),
      // ...and split by what the token IS, plus the whole vector, because the
      // question a ligand raises is whether it moves the POLYMER's numbers.
      polymerMean: (() => {
        const v = [...result.certainty].filter((_, t) => result.features.molType[t] !== 3);
        return v.length === 0 ? null : v.reduce((a, b) => a + b, 0) / v.length;
      })(),
      ligandMean: (() => {
        const v = [...result.certainty].filter((_, t) => result.features.molType[t] === 3);
        return v.length === 0 ? null : v.reduce((a, b) => a + b, 0) / v.length;
      })(),
      perToken: [...result.certainty].map((v) => Number(v.toFixed(4))),
    },
    contactAgreement: agreement,
    pdb,
  };
}
