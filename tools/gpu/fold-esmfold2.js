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
  const sequence = option(args, "sequence",
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQ");
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

  const shim = {};
  for (const name of SHIM_PAIR_TENSORS) shim[name] = await tower.read(name);
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
  const runTower = async (ids) => {
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
    }, towerShared);
    return result.single;
  };

  const progress = [];
  const result = await foldEsmfold2(device, {
    sequence, allocator, seed, sampler,
    shape: { ...M, loops: (M.loops ?? 3) + 1 },
    weights: { featuriser, inputsEmbedder, trunkBlocks, denoiser, shim },
    tower: runTower,
    onProgress: (label) => { progress.push(label); },
  });

  // ---- what came out.
  const alphas = alphaCarbons(result.features);
  const x = result.coordinates;
  const spacing = [];
  for (let i = 1; i < alphas.length; i += 1) {
    const a = alphas[i - 1] * 3, b = alphas[i] * 3;
    spacing.push(Math.hypot(x[b] - x[a], x[b + 1] - x[a + 1], x[b + 2] - x[a + 2]));
  }
  const mean = spacing.reduce((t, v) => t + v, 0) / spacing.length;
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
  const lines = alphas.map((atom, index) =>
    `ATOM  ${String(index + 1).padStart(5)}  CA  GLY A${String(index + 1).padStart(4)}    `
    + [0, 1, 2].map((axis) => x[atom * 3 + axis].toFixed(3).padStart(8)).join("")
    + "  1.00  0.00           C");

  return {
    sequence, sampler, seed,
    tokens: result.tokens, atoms: result.atoms, steps: result.steps,
    alphaCarbons: alphas.length,
    caSpacing: { mean, min: Math.min(...spacing), max: Math.max(...spacing) },
    rmsdToReference: rmsd,
    elapsedSeconds: result.elapsedMilliseconds / 1000,
    timings: result.timings,
    peakMebibytes: result.memory.peakBytes / 1048576,
    stages: progress,
    pdb: lines.join("\n"),
  };
}
