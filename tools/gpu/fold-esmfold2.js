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
import { readTensor } from "../../src/reference/dtype.js";
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

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const BLOCK_LEAVES = ["attn_norm/scale", "attn_norm/offset", "qkv/weights",
  "q_norm/scale", "k_norm/scale", "attn_out/weights", "ffn_norm/scale",
  "ffn_norm/offset", "fc1/weights", "fc2/weights"];
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
  const read = async (name) => {
    const loaded = await manifest();
    const record = loaded.tensors[name];
    if (record === undefined) throw new Error(`${bundle} has no tensor ${name}`);
    if (!shards.has(record.file)) {
      shards.set(record.file, await (await fetch(`${bundle}/${record.file}`)).arrayBuffer());
    }
    return readTensor(record, shards.get(record.file), record.byteOffset ?? 0, true);
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
  const foldBundle = option(args, "bundle", "/model-esmfold2-trunk-f32");
  const towerBundle = option(args, "esmc", "/model-esmc-600m-int3");
  const sampler = option(args, "sampler", "diffusion-15");
  const seed = Number(option(args, "seed", "0"));
  // 🔴 THE REFERENCE IS A DIFFERENT SAMPLE, NOT A DIFFERENT ANSWER. The sampler
  // draws a rotation, a translation and a noise vector per step; this uses its
  // own RNG, so the honest comparison is RMSD after superposition and a few
  // angstroms is what agreement LOOKS like here.
  const reference = option(args, "reference", "");
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
  const runTower = async (ids, sequenceId) => {
    const engine = new EsmcTowerGpu(device, allocator);
    const result = await engine.run(ids, {
      rows: ids.length, model: towerManifest.width,
      heads: towerManifest.heads ?? towerManifest.width / 64,
      ffn: towerTable.tensors["blocks/0/fc2/weights"].shape[0],
      layers: towerManifest.layers, pair: M.pairChannels,
      residualScale: towerManifest.residualScale ?? 1,
    }, async (layer) => {
      const weights = {};
      for (const leaf of BLOCK_LEAVES) weights[leaf] = await tower.read(`blocks/${layer}/${leaf}`);
      return weights;
    }, towerShared, { sequenceId });
    return result.single;
  };

  const ligands = [];
  for (const code of ligandCodes) {
    const text = await (await fetch(ccdUrl(code))).text();
    ligands.push(parseCcdComponent(text));
  }

  const progress = [];
  const result = await foldEsmfold2(device, {
    sequence, allocator, seed, sampler,
    entities: (kinds === "" && ligands.length === 0) ? sequence
      : { sequence, ...(kinds === "" ? {} : { chainKinds: kinds.split(",") }),
          ...(ligands.length === 0 ? {} : { ligands }) },
    shape: { ...M, loops: (M.loops ?? 3) + 1 },
    weights: { featuriser, inputsEmbedder, trunkBlocks, denoiser, shim },
    tower: runTower,
    onProgress: (label) => { progress.push(label); },
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
  if (result.contacts !== undefined) {
    const representative = new Int32Array(result.tokens).fill(-1);
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
    for (let i = 0; i < result.tokens; i += 1) {
      for (let j = i + 6; j < result.tokens; j += 1) {
        const near = result.contacts[i * result.tokens + j] > 0.5;
        if (near) contactPairs += 1;
        const a = representative[i] * 3, b = representative[j] * 3;
        const close = Math.hypot(x[b] - x[a], x[b + 1] - x[a + 1], x[b + 2] - x[a + 2]) < 8;
        if (near) predicted += 1;
        if (close) actual += 1;
        if (near && close) both += 1;
      }
    }
    agreement = { predicted, actual, both,
                  precision: predicted === 0 ? null : both / predicted,
                  recall: actual === 0 ? null : both / actual };
  }

  return {
    sequence, sampler, seed,
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
    stages: progress,
    longRangeContacts: contactPairs,
    contactAgreement: agreement,
    pdb,
  };
}
