// Does src/esmc/block-webgpu.js compute an ESM-C block?
//
//     python3 tools/esmc/dump-esmc-oracle.py --sequence-length 59
//     python3 tools/export_esmc_model.py
//     node tools/gpu-chrome.mjs tools/gpu/check-esmc-block.js
//
// Against the oracle, and against the CPU reference, which are not the same
// question. The oracle says the block computes ESM-C; the reference says the
// GPU and the CPU agree, which is what localises a failure to the shader rather
// than to a convention. `tools/check-esmc-reference.js` has already shown the
// reference discriminates the three conventions that conform in shape.
//
// 🔴 IT FETCHES ONE BLOCK'S SHARDS, NOT THE BUNDLE. The float32 export is
// 2190 MiB and block 0 is 64 of them; pulling the whole thing to check one
// block would make this checker something nobody runs.
import { EsmcBlockGpu } from "../../src/esmc/block-webgpu.js";
import { esmcBlock } from "../../src/esmc/tower-reference.js";

// 🔴 THE SHARD IS FETCHED WHOLE AND SLICED, BECAUSE THE DEV SERVER IGNORES
// Range. `python3 -m http.server` answers a ranged request with 200 and the
// entire file, so a checker that trusted the header would have sliced 4608
// bytes out of the front of a 48 MiB shard and called it a LayerNorm scale -
// a wrong tensor of exactly the right length. Whole shards, cached: block 0
// lives in one or two of them, which is 48-96 MiB rather than the bundle's
// 2190.
const shards = new Map();

async function shard(base, file) {
  if (!shards.has(file)) {
    const response = await fetch(`${base}/${file}`);
    if (!response.ok) throw new Error(`${file} answered ${response.status}`);
    shards.set(file, await response.arrayBuffer());
  }
  return shards.get(file);
}

async function tensor(manifest, name, base) {
  const record = manifest.tensors[name];
  if (record === undefined) throw new Error(`bundle has no tensor ${name}`);
  if (record.dtype !== "float32") {
    throw new Error(`${name} is ${record.dtype}; this checker reads the float32 export`);
  }
  const count = record.shape.reduce((a, b) => a * b, 1);
  const start = record.byteOffset ?? 0;
  const buffer = await shard(base, record.file);
  if (start + count * 4 > buffer.byteLength) {
    throw new Error(`${name} runs past the end of ${record.file}`);
  }
  // A copy, not a view: the shard's byteOffset is not guaranteed to be a
  // multiple of four for a Float32Array view of it.
  return new Float32Array(buffer.slice(start, start + count * 4));
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

export async function main(device, args = {}) {
  const bundle = option(args, "bundle", "/model-esmc-600m-f32");
  const dumpPath = option(args, "dump", "/oracle-dumps/esmc-59.json");
  const bound = Number(option(args, "bound", "2e-5"));

  const dump = await (await fetch(dumpPath)).json();
  const manifest = await (await fetch(`${bundle}/manifest.json`)).json();
  if (manifest.languageModel?.tower !== dump.esmc
    || manifest.languageModel?.shim !== dump.esmfold2) {
    throw new Error(`bundle is ${manifest.languageModel?.tower} + `
      + `${manifest.languageModel?.shim} and the dump is ${dump.esmc} + `
      + `${dump.esmfold2}; the shim is per folding model`);
  }

  const shape = {
    rows: dump.tokens.length,
    model: dump.dims.model,
    heads: dump.dims.heads,
    ffn: dump.shapes["ffn.fc2_weight"][1],
    residualScale: dump.dims.residualScale,
  };

  const leaves = ["attn_norm/scale", "attn_norm/offset", "qkv/weights",
    "q_norm/scale", "k_norm/scale", "attn_out/weights", "ffn_norm/scale",
    "ffn_norm/offset", "fc1/weights", "fc2/weights"];
  const weights = {};
  let bytes = 0;
  for (const leaf of leaves) {
    weights[leaf] = await tensor(manifest, `blocks/0/${leaf}`, bundle);
    bytes += weights[leaf].byteLength;
  }

  const input = Float32Array.from(dump.embedded);
  const oracleOutput = Float32Array.from(dump.block0Output);
  const oracleNormed = Float32Array.from(dump.block0["attn.layernorm_qkv.weight"]);

  const gpu = new EsmcBlockGpu(device);
  const result = await gpu.run(input, shape, weights);

  // The CPU reference on the same weights and the same input, so a divergence
  // can be attributed to the shader rather than to the oracle or the bundle.
  const reference = esmcBlock(input, shape.rows, { ...prefixed(weights) },
    { ...shape, layers: dump.dims.layers }, 0);

  const arms = [
    ["qkv input, vs oracle", result.normed, oracleNormed],
    ["block output, vs oracle", result.output, oracleOutput],
    ["block output, vs CPU reference", result.output, reference],
  ];
  const report = arms.map(([label, got, want]) => ({
    label, relRms: relative(got, want), within: relative(got, want) <= bound,
  }));

  return {
    tower: manifest.languageModel.tower,
    shim: manifest.languageModel.shim,
    shape,
    weightBytes: bytes,
    bound,
    elapsedMilliseconds: result.elapsedMilliseconds,
    arms: report,
    ok: report.every((arm) => arm.within),
  };
}

/** The reference reads `blocks/<layer>/<leaf>`; the checker fetched bare leaves. */
function prefixed(weights) {
  const out = {};
  for (const [leaf, values] of Object.entries(weights)) out[`blocks/0/${leaf}`] = values;
  return out;
}
