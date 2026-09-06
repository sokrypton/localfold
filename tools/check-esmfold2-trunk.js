// Does LocalFold's pairformer arithmetic compute ESMFold2's trunk?
//
//     .venv-esm/bin/python tools/esmc/dump-esmfold2-trunk.py --sequence-length 40
//     python3 tools/export_esmfold2_trunk.py
//     node tools/check-esmfold2-trunk.js
//
// ESMFold2's trunk block is AF3's pairformer block with the two grid attentions
// and the single track removed, so this composes the three pieces that remain
// out of src/af3/pairformer-reference.js and runs 24 of them:
//
//     pair += triangleMultiplication(pair, "outgoing")
//     pair += triangleMultiplication(pair, "incoming")
//     pair += transition(pair)
//
// 🔴 IT COMPOSES THE PIECES RATHER THAN CALLING pairformerBlock, WHICH WOULD
// NEED A SINGLE TRACK THAT DOES NOT EXIST. That block always runs
// singleAttention and a single transition; feeding it a synthesised zero single
// would check arithmetic ESMFold2 never does, and zeroed grid attention weights
// would make it compute 24 blocks of zeros to add. Zeroing is the trick the GPU
// path needs, because encodePairTrack runs a fixed sequence of passes; the CPU
// reference does not need it and should not pretend to.
//
// 🔴 AND IT CHECKS trunk(x) == y PER LOOP, NOT A FOLD. The dump records the
// trunk's INPUT and OUTPUT at each of the four recycles, so this needs neither
// z_init, nor pair_loop_proj, nor the featuriser - each of which is its own
// port with its own way of being wrong.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";

import { readTensor } from "../src/reference/dtype.js";
import { transition, triangleMultiplication } from "../src/af3/pairformer-reference.js";

const ROOT = new URL("..", import.meta.url).pathname;
const bundleDirectory = process.argv[2] ?? join(ROOT, "model-esmfold2-trunk-f32");
const dumpPath = process.argv[3] ?? join(ROOT, "oracle-dumps", "esmfold2-trunk-40.json");
const bound = Number(process.argv[4] ?? 2e-4);

function loadBundle(directory) {
  const manifest = JSON.parse(readFileSync(join(directory, "manifest.json"), "utf8"));
  const shards = new Map();
  const tensors = {};
  for (const [name, record] of Object.entries(manifest.tensors)) {
    if (!shards.has(record.file)) {
      const raw = readFileSync(join(directory, record.file));
      shards.set(record.file,
        raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
    }
    tensors[name] = readTensor(record, shards.get(record.file), record.byteOffset ?? 0, true);
  }
  return { manifest, tensors };
}

const group = (tensors, layer, name, leaves) => {
  const out = {};
  for (const leaf of leaves) {
    const key = `blocks/${layer}/${name}/${leaf}`;
    if (tensors[key] === undefined) throw new Error(`bundle has no ${key}`);
    out[leaf] = tensors[key];
  }
  return out;
};

const TRIANGLE = ["leftNormInputScale", "leftNormInputOffset", "centerNormScale",
  "centerNormOffset", "outputProjection", "gatingLinear", "projection", "gate"];
const TRANSITION = ["inputLayerNormScale", "inputLayerNormOffset",
  "transition1", "transition2"];

const relative = (got, want) => {
  let error = 0, total = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i] - want[i];
    error += d * d;
    total += want[i] * want[i];
  }
  return Math.sqrt(error / total);
};

const dump = JSON.parse(readFileSync(dumpPath, "utf8"));
const { manifest, tensors } = loadBundle(bundleDirectory);
if (manifest.trunk?.source !== dump.esmfold2) {
  throw new Error(`bundle is ${manifest.trunk?.source}, dump is ${dump.esmfold2}`);
}

const tokens = dump.shapes.pair[1];
const channels = manifest.trunk.pairChannels;
const blocks = manifest.trunk.blocks;
if (blocks !== dump.blocks) throw new Error(`${blocks} blocks against the dump's ${dump.blocks}`);
console.log(`${dump.esmfold2}: ${tokens} tokens, ${blocks} blocks x ${channels} channels, `
  + `${dump.loops} loops`);

// A single chain with no padding: every pair is live. The dump was taken the
// same way, which is why this is a constant rather than a feature.
const mask = new Float32Array(tokens * tokens).fill(1);

let failures = 0;
for (const key of Object.keys(dump.intoLoop)) {
  const started = Date.now();
  let pair = Float32Array.from(dump.intoLoop[key]);
  const add = (delta) => {
    for (let i = 0; i < pair.length; i += 1) pair[i] += delta[i];
  };
  for (let layer = 0; layer < blocks; layer += 1) {
    add(triangleMultiplication(pair, mask, tokens, channels, "outgoing",
      group(tensors, layer, "triangleMultiplicationOutgoing", TRIANGLE)));
    add(triangleMultiplication(pair, mask, tokens, channels, "incoming",
      group(tensors, layer, "triangleMultiplicationIncoming", TRIANGLE)));
    add(transition(pair, tokens * tokens, channels,
      group(tensors, layer, "pairTransition", TRANSITION)));
  }
  const want = Float32Array.from(dump.afterLoop[key]);
  const score = relative(pair, want);
  const ok = score <= bound;
  if (!ok) failures += 1;
  console.log(`  loop ${key}: relRMS ${score.toExponential(3)}   bound `
    + `${bound.toExponential(0)}   ${ok ? "ok" : "FAILED"}   (${((Date.now() - started) / 1000).toFixed(1)}s)`);
}

console.log(failures === 0
  ? "\nESMFold2's trunk is AF3's pairformer without the attention or the single track"
  : `\n${failures} loop(s) out of bound`);
process.exit(failures === 0 ? 0 : 1);
