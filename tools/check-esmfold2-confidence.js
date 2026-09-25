/**
 * ESMFold2's confidence head against Synthyra's own, on the host.
 *
 *     node tools/check-esmfold2-confidence.js model-ef2-600-head-f32 \
 *       oracle-dumps/esmfold2-confidence-600.json
 *
 * 🔴 NO GPU AND NO FOLD. The head takes a pair, a single-inputs row and a set
 * of coordinates and returns pLDDT and PAE; all three are in the dump, so this
 * runs on a box with no adapter and says whether the arithmetic is right
 * before a single WGSL line exists. The GPU checker is the next layer and
 * compares against THIS, the way check-opendde-confidence.js compares against
 * `hostPairReadouts`.
 *
 * 🔴 AND THE ORACLE IS THEIR MODULE, NOT A READING OF IT.
 * `tools/oracle/dump_esmfold2_confidence.py` unpacks the `fastplms` sources
 * Synthyra ship base85-zipped inside `fastplms_bundle.py` and imports their
 * `ConfidenceHead`. A reference written from reading their forward pass and
 * checked against a port written from the same reading is two copies of one
 * misunderstanding agreeing perfectly.
 *
 * 🔴 WHAT THE DUMP CANNOT SEE: its `z` and `s_inputs` are seeded normals, so
 * the PAE it produces is near the 16.0 a uniform 64-bin softmax over [0, 32]
 * gives and says nothing about whether a real fold's confidence is
 * informative. Its COORDINATES are a real PDB, because the head buckets
 * rep-atom distances into 128 bins over [2, 52] and a random cloud would put
 * every pair in one column of that embedding. This gates arithmetic; a real
 * trunk's pair is a second arm.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";

import { readTensor } from "../src/weights/dtype.js";
import { esmfold2Confidence } from "../src/esmfold2/confidence-reference.js";

const ROOT = new URL("..", import.meta.url).pathname;
const bundleDir = join(ROOT, process.argv[2] ?? "model-ef2-600-head-f32");
const dumpPath = process.argv[3] ?? join(ROOT, "oracle-dumps", "esmfold2-confidence-600.json");
const bound = Number(process.argv[4] ?? 2e-5);

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

const TRIANGLE = ["leftNormInputScale", "leftNormInputOffset", "centerNormScale",
  "centerNormOffset", "outputProjection", "gatingLinear", "projection", "gate"];
const TRANSITION = ["inputLayerNormScale", "inputLayerNormOffset",
  "transition1", "transition2"];

const relative = (got, want) => {
  let error = 0;
  let total = 0;
  for (let index = 0; index < want.length; index += 1) {
    const delta = got[index] - want[index];
    error += delta * delta;
    total += want[index] * want[index];
  }
  return Math.sqrt(error / Math.max(total, 1e-30));
};

const { manifest, tensors } = loadBundle(bundleDir);
if (manifest.confidence === undefined) {
  console.error(`${bundleDir} carries no confidence head - export it with`
    + " tools/export_esmfold2_trunk.py --confidence");
  process.exit(1);
}
const dump = JSON.parse(readFileSync(dumpPath, "utf8"));
const stage = (name) => {
  const found = dump.stages[name];
  if (found === undefined) throw new Error(`the dump has no ${name}`);
  return Float32Array.from(found.data);
};

const meta = manifest.confidence;
const blocks = [];
for (let layer = 0; layer < meta.blocks; layer += 1) {
  const group = (name, leaves) => Object.fromEntries(leaves.map((leaf) => {
    const key = `confidence/blocks/${layer}/${name}/${leaf}`;
    if (tensors[key] === undefined) throw new Error(`bundle has no ${key}`);
    return [leaf, tensors[key]];
  }));
  blocks.push({
    triangleMultiplicationOutgoing: group("triangleMultiplicationOutgoing", TRIANGLE),
    triangleMultiplicationIncoming: group("triangleMultiplicationIncoming", TRIANGLE),
    pairTransition: group("pairTransition", TRANSITION),
  });
}
const flat = (leaf) => {
  const key = `confidence/${leaf}`;
  if (tensors[key] === undefined) throw new Error(`bundle has no ${key}`);
  return tensors[key];
};
const weights = {
  ...meta, blocks,
  sInputsNormScale: flat("sInputsNorm/scale"), sInputsNormOffset: flat("sInputsNorm/offset"),
  zNormScale: flat("zNorm/scale"), zNormOffset: flat("zNorm/offset"),
  plddtNormScale: flat("plddtNorm/scale"), plddtNormOffset: flat("plddtNorm/offset"),
  sToZ: flat("sToZ"), sToZTranspose: flat("sToZTranspose"),
  sToZProdIn1: flat("sToZProdIn1"), sToZProdIn2: flat("sToZProdIn2"),
  sToZProdOut: flat("sToZProdOut"), distanceEmbedding: flat("distanceEmbedding"),
  boundaries: flat("boundaries"), poolingAttention: flat("poolingAttention"),
  poolingOutput: flat("poolingOutput"), plddtWeight: flat("plddtWeight"),
  pae: flat("pae"),
};

const tokens = dump.tokens;
const atoms = dump.atoms;
const inputs = {
  tokens, atoms,
  sInputs: stage("in.s_inputs"),
  pair: stage("in.z"),
  coordinates: stage("in.x_pred"),
  repAtom: Int32Array.from(stage("in.distogram_atom_idx")),
  atomToToken: Int32Array.from(stage("in.atom_to_token")),
  tokenMask: stage("in.token_attention_mask"),
  atomMask: stage("in.atom_attention_mask"),
};

const started = Date.now();
const got = esmfold2Confidence(inputs, weights);
const seconds = ((Date.now() - started) / 1000).toFixed(1);

const arms = [
  ["out.plddt_logits", got.plddtLogits],
  ["out.plddt_per_atom", got.plddtPerAtom],
  ["out.plddt", got.plddt],
  ["out.plddt_ca", got.plddtCa],
  ["out.pae_logits", got.paeLogits],
  ["out.pae", got.pae],
];
let failures = 0;
const results = [];
for (const [name, mine] of arms) {
  const want = stage(name);
  if (mine.length !== want.length) {
    console.log(`  ${name}: LENGTH ${mine.length} against ${want.length}`);
    failures += 1;
    continue;
  }
  const score = relative(mine, want);
  const ok = score <= bound;
  if (!ok) failures += 1;
  results.push({ stage: name, relRms: Number(score.toExponential(2)) });
  console.log(`  ${name.padEnd(22)} relRMS ${score.toExponential(3)}  ${ok ? "ok" : "FAILS"}`);
}
console.log(`\n${dump.tokens} tokens, ${dump.atoms} atoms, bound ${bound}, ${seconds}s`);
if (failures > 0) {
  console.error(`${failures} arm(s) over the bound`);
  process.exit(1);
}
console.log("the head matches Synthyra's own");
