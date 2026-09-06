// What a trajectory frame should BE: the sampler's state, or the model's guess?
//
//     node tools/gpu-chrome.mjs tools/gpu/probe-esmfold2-trajectory.js
//
// 🔴 A DIFFUSION TRAJECTORY HAS TWO COORDINATE SETS AT EVERY STEP AND ONLY ONE
// OF THEM IS A PICTURE. `coordinates` is what the sampler carries forward - the
// state at the next noise level, which at the top of the schedule is Gaussian
// noise at sigma 411 - and `denoised` is the model's predicted structure at
// that call, EDM preconditioning included. They converge, and at the top they
// are nothing alike.
//
// 🔴 AND THE SAMPLER RE-POSES EVERY STEP, so consecutive frames of EITHER
// differ by a rigid motion far larger than anything the denoiser did. This
// reports the frame-to-frame movement before and after superposition, which is
// what says whether a viewer must fit them.
import { GpuBufferAllocator } from "../../src/runtime/allocator.js";
import { EsmcTowerGpu } from "../../src/esmc/tower-webgpu.js";
import { readTensor, readTensorAsFloat16 } from "../../src/reference/dtype.js";
import { foldEsmfold2 } from "../../src/esmfold2/fold.js";
import { weightedRigidAlign } from "../../src/esmfold2/sampler-reference.js";
import {
  atomDecoderWeights, atomEncoderWeights, denoiserWeights, featuriserWeights,
  trunkBlockWeights,
} from "../../src/esmfold2/weights.js";
import { SHIM_PAIR_TENSORS } from "../../src/esmfold2/language-pair-webgpu.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};
const BLOCK_LEAVES = ["attn_norm/scale", "attn_norm/offset", "qkv/weights",
  "q_norm/scale", "k_norm/scale", "attn_out/weights", "ffn_norm/scale",
  "ffn_norm/offset", "fc1/weights", "fc2/weights"];
const NARROW = new Set(["qkv/weights", "attn_out/weights", "fc1/weights", "fc2/weights"]);
const TOWER_SHARED = ["embed/weights", "final_norm/scale", "lm/combine", "lm/norm/scale",
  "lm/norm/offset", "lm/projection/weights", "lm/downproject/weights",
  "lm/downproject/bias"];

function reader(bundle) {
  const shards = new Map();
  let table;
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

/** Radius of gyration over the live atoms - how big the thing on screen is. */
function radiusOfGyration(x, mask, atoms) {
  let total = 0;
  const centre = [0, 0, 0];
  for (let atom = 0; atom < atoms; atom += 1) {
    if (mask[atom] === 0) continue;
    total += 1;
    for (let axis = 0; axis < 3; axis += 1) centre[axis] += x[atom * 3 + axis];
  }
  for (let axis = 0; axis < 3; axis += 1) centre[axis] /= Math.max(total, 1);
  let squared = 0;
  for (let atom = 0; atom < atoms; atom += 1) {
    if (mask[atom] === 0) continue;
    for (let axis = 0; axis < 3; axis += 1) {
      const d = x[atom * 3 + axis] - centre[axis];
      squared += d * d;
    }
  }
  return Math.sqrt(squared / Math.max(total, 1));
}

const rmsd = (a, b, mask, atoms) => {
  let squared = 0, live = 0;
  for (let atom = 0; atom < atoms; atom += 1) {
    if (mask[atom] === 0) continue;
    live += 1;
    for (let axis = 0; axis < 3; axis += 1) {
      const d = a[atom * 3 + axis] - b[atom * 3 + axis];
      squared += d * d;
    }
  }
  return Math.sqrt(squared / Math.max(live, 1));
};

export async function main(device, args = []) {
  const sequence = option(args, "sequence",
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQ");
  const fold = reader(option(args, "bundle", "/model-esmfold2-int5"));
  const tower = reader(option(args, "esmc", "/model-esmc-600m-int3"));
  const manifest = await fold.manifest();
  const towerTable = await tower.manifest();
  const language = towerTable.languageModel;
  const M = manifest.trunk;

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
  const towerShared = {};
  for (const name of TOWER_SHARED) towerShared[name] = await tower.read(name);
  const allocator = new GpuBufferAllocator(device);

  const frames = [];
  let previous;
  await foldEsmfold2(device, {
    sequence, allocator, seed: 0, sampler: option(args, "sampler", "diffusion-15"),
    shape: { ...M, loops: (M.loops ?? 3) + 1 },
    weights: { featuriser, inputsEmbedder, trunkBlocks, denoiser, shim },
    tower: async (ids, sequenceId) => (await new EsmcTowerGpu(device, allocator).run(ids, {
      rows: ids.length, model: language.width,
      heads: language.heads ?? language.width / 64,
      ffn: towerTable.tensors["blocks/0/fc2/weights"].shape[0],
      layers: language.layers, pair: M.pairChannels,
      residualScale: language.residualScale ?? 1,
    }, async (layer) => {
      const block = {};
      for (const leaf of BLOCK_LEAVES) {
        block[leaf] = await tower.read(`blocks/${layer}/${leaf}`, NARROW.has(leaf));
      }
      return block;
    }, towerShared, { sequenceId })).single,
    onStep: ({ step, coordinates, denoised, features }) => {
      const { mask, atoms } = features;
      const row = {
        step,
        stateRadius: radiusOfGyration(coordinates, mask, atoms),
        denoisedRadius: radiusOfGyration(denoised, mask, atoms),
      };
      if (previous !== undefined) {
        row.denoisedMovedRaw = rmsd(denoised, previous, mask, atoms);
        row.denoisedMovedFitted = rmsd(
          weightedRigidAlign(denoised, previous, mask, atoms), previous, mask, atoms);
      }
      previous = Float32Array.from(denoised);
      frames.push(row);
    },
  });

  console.log("  step   state Rg   denoised Rg   denoised moved (raw / fitted)");
  for (const row of frames) {
    console.log(`  ${String(row.step).padStart(4)}  ${row.stateRadius.toFixed(1).padStart(9)}`
      + `  ${row.denoisedRadius.toFixed(1).padStart(12)}   `
      + (row.denoisedMovedRaw === undefined ? "-"
        : `${row.denoisedMovedRaw.toFixed(1).padStart(6)} / ${row.denoisedMovedFitted.toFixed(2)}`));
  }
  const radii = frames.map((row) => row.stateRadius);
  return {
    sequence, frames,
    stateRadiusRange: [Math.min(...radii), Math.max(...radii)],
    denoisedRadiusRange: [Math.min(...frames.map((r) => r.denoisedRadius)),
                          Math.max(...frames.map((r) => r.denoisedRadius))],
  };
}
