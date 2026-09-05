// Does src/esmc/tower-webgpu.js compute ESM-C's 36 blocks and ESMFold2's mix?
//
//     python3 tools/esmc/dump-esmc-oracle.py --sequence-length 59
//     python3 tools/export_esmc_model.py
//     node tools/gpu-chrome.mjs tools/gpu/check-esmc-tower.js
//
// The states the dump records - 0, 1, 18, 35 and 36 - plus the mixed single
// representation, which is the tensor ESMFold2 actually consumes and the one
// the streaming accumulator exists to produce without materialising 37 states.
//
// 🔴 THE SHARD CACHE EVICTS, BECAUSE THE FLOAT32 BUNDLE IS 2190 MiB. Holding
// every shard a 36-block pass touches would put the whole export in the tab.
// The export writes tensors in block order, so a sequential pass walks the
// shards in order too and a window of three is enough - which is 144 MiB live
// against 2190, the same shape of trade the tower itself makes with weights.
import { EsmcTowerGpu } from "../../src/esmc/tower-webgpu.js";

const WINDOW = 3;

function shardReader(base) {
  const cache = new Map();
  return async (file) => {
    if (cache.has(file)) {
      const buffer = cache.get(file);
      cache.delete(file);
      cache.set(file, buffer);
      return buffer;
    }
    const response = await fetch(`${base}/${file}`);
    if (!response.ok) throw new Error(`${file} answered ${response.status}`);
    const buffer = await response.arrayBuffer();
    cache.set(file, buffer);
    while (cache.size > WINDOW) cache.delete(cache.keys().next().value);
    return buffer;
  };
}

function tensorReader(manifest, read) {
  return async (name) => {
    const record = manifest.tensors[name];
    if (record === undefined) throw new Error(`bundle has no tensor ${name}`);
    if (record.dtype !== "float32") {
      throw new Error(`${name} is ${record.dtype}; this checker reads the float32 export`);
    }
    const count = record.shape.reduce((a, b) => a * b, 1);
    const start = record.byteOffset ?? 0;
    const buffer = await read(record.file);
    if (start + count * 4 > buffer.byteLength) {
      throw new Error(`${name} runs past the end of ${record.file}`);
    }
    return new Float32Array(buffer.slice(start, start + count * 4));
  };
}

const relative = (got, want) => {
  let error = 0, total = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i] - want[i];
    error += d * d;
    total += want[i] * want[i];
  }
  return Math.sqrt(error / total);
};

const BLOCK_LEAVES = ["attn_norm/scale", "attn_norm/offset", "qkv/weights",
  "q_norm/scale", "k_norm/scale", "attn_out/weights", "ffn_norm/scale",
  "ffn_norm/offset", "fc1/weights", "fc2/weights"];

const SHARED = ["embed/weights", "final_norm/scale", "lm/combine", "lm/norm/scale",
  "lm/norm/offset", "lm/projection/weights", "lm/downproject/weights",
  "lm/downproject/bias"];

export async function main(device, args = {}) {
  const bundle = args.bundle ?? "/model-esmc-600m-f32";
  const dumpPath = args.dump ?? "/oracle-dumps/esmc-59.json";
  const bound = Number(args.bound ?? 5e-5);

  const dump = await (await fetch(dumpPath)).json();
  const manifest = await (await fetch(`${bundle}/manifest.json`)).json();
  if (manifest.languageModel?.tower !== dump.esmc
    || manifest.languageModel?.shim !== dump.esmfold2) {
    throw new Error(`bundle is ${manifest.languageModel?.tower} + `
      + `${manifest.languageModel?.shim} and the dump is ${dump.esmc} + `
      + `${dump.esmfold2}; the shim is per folding model`);
  }

  const read = tensorReader(manifest, shardReader(bundle));
  const shape = {
    rows: dump.tokens.length,
    model: dump.dims.model,
    heads: dump.dims.heads,
    ffn: dump.shapes["ffn.fc2_weight"][1],
    layers: dump.dims.layers,
    pair: dump.shapes.single[1],
    residualScale: dump.dims.residualScale,
  };

  const shared = {};
  for (const name of SHARED) shared[name] = await read(name);

  // 🔴 A BLOCK IS READ WHEN THE TOWER ASKS FOR IT. Reading all 36 first is
  // 2190 MiB of float32 in the tab, which is the thing the streaming design
  // exists to avoid - a checker that could not itself stream would be checking
  // the design by contradicting it. The shard window makes this ~144 MiB live.
  const blockWeights = async (layer) => {
    const weights = {};
    for (const leaf of BLOCK_LEAVES) weights[leaf] = await read(`blocks/${layer}/${leaf}`);
    return weights;
  };

  const ids = Int32Array.from(dump.tokens);
  const capture = Object.keys(dump.states).map(Number);
  const tower = new EsmcTowerGpu(device);
  const result = await tower.run(ids, shape, blockWeights, shared, { capture });

  const arms = [];
  for (const key of capture) {
    const got = result.states.get(key);
    if (got === undefined) {
      arms.push({ label: `state ${key}`, relRms: null, within: false,
        note: "not captured" });
      continue;
    }
    arms.push({ label: `state ${key}`,
      relRms: relative(got, Float32Array.from(dump.states[String(key)])) });
  }

  // The mixed single is over the residues only - the dump's shim strips BOS and
  // EOS before mixing, so the tower's rows include them and the comparison must
  // not.
  const pair = shape.pair;
  const interior = result.single.subarray(pair, result.single.length - pair);
  arms.push({ label: "mixed single", relRms: relative(interior, Float32Array.from(dump.single)) });

  for (const arm of arms) arm.within = arm.relRms !== null && arm.relRms <= bound;

  return {
    tower: manifest.languageModel.tower,
    shim: manifest.languageModel.shim,
    shape,
    bound,
    elapsedMilliseconds: result.elapsedMilliseconds,
    memory: result.memory,
    arms,
    ok: arms.every((arm) => arm.within),
  };
}
