// Sequence in, ESMFold2's trunk output out. The end-to-end gate.
//
//     .venv-esm/bin/python tools/esmc/dump-esmfold2-trunk.py \
//         --sequence-length 40 --esmc esmc-600m --out oracle-dumps/esmfold2-trunk-40-lm.json
//     node tools/check-esmfold2-fold.js
//
// 🔴 EVERY PER-MODULE CHECK CAN PASS WHILE THE ASSEMBLY IS WRONG, WHICH IS THE
// WHOLE REASON THIS EXISTS. check-esmfold2-featuriser.js says each of z_init's
// terms matches; check-esmfold2-trunk.js says the 24 blocks do. Neither says
// the five terms are SUMMED, that the language model's pair reaches them, that
// the recurrence runs `num_loops + 1` times, or that `z` starts at zero - and
// each of those is a plausible-looking tensor when wrong.
//
// 🔴 AND THE LANGUAGE MODEL'S TERM ONLY EXISTS IN A DUMP TAKEN WITH `--esmc`.
// Without it `lm_z is None` and z_init has four terms, not five - and the
// no-LM path is not "the LM contributing zero": the shim's biases make
// lm_shim(0) non-zero, so substituting zeros is a different model.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";

import { readTensor } from "../src/reference/dtype.js";
import { transition, triangleMultiplication } from "../src/af3/pairformer-reference.js";
import {
  recycleProjection, relativePositionEncoding, tokenBondEncoding, zInitFromInputs,
} from "../src/esmfold2/featuriser-reference.js";
import { inputsEmbedder } from "../src/esmfold2/atom-encoder-reference.js";

const ROOT = new URL("..", import.meta.url).pathname;
const bundleDirectory = process.argv[2] ?? join(ROOT, "model-esmfold2-trunk-f32");
const dumpPath = process.argv[3] ?? join(ROOT, "oracle-dumps", "esmfold2-trunk-40-lm.json");

const TRIANGLE = ["leftNormInputScale", "leftNormInputOffset", "centerNormScale",
  "centerNormOffset", "outputProjection", "gatingLinear", "projection", "gate"];
const TRANSITION = ["inputLayerNormScale", "inputLayerNormOffset",
  "transition1", "transition2"];

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
if (dump.languageModel == null) {
  throw new Error(`${dumpPath} was taken without --esmc, so it has no lm_z term; `
    + "re-dump with --esmc <checkpoint>");
}

const n = dump.shapes.pair[1];
const pairs = n * n;
const channels = manifest.trunk.pairChannels;
const blockCount = manifest.trunk.blocks;
const module = (name) => dump.block0[name];
const argument = (name, key) => Float32Array.from(module(name).arguments[key].values);
// 🔴 THE BOUND IS THE bfloat16 ONE, BECAUSE THE ATOM ATTENTION DOWNCASTS. See
// check-esmfold2-featuriser.js: the same code reads 7.0e-8 against a dump taken
// with --float32-attention and 2.1e-4 against this one, and 24 blocks x 4 loops
// of a contractive stack neither amplifies nor removes it.
const bound = Number(process.argv[4] ?? (dump.float32Attention ? 2e-5 : 2e-3));

console.log(`${dump.esmfold2} + ${dump.languageModel}: ${n} residues, `
  + `${dump.loops} loops x ${blockCount} blocks\n`);

// --- z_init, all five terms.
const feature = (name) => Float32Array.from(dump.features[name].values);
const integers = (name) => Int32Array.from(dump.features[name].values);
const atoms = dump.features.ref_pos.shape[1];
const shape = {
  atoms, tokens: n,
  channels: manifest.trunk.atomChannels, heads: manifest.trunk.atomHeads,
  blocks: manifest.trunk.atomBlocks, tokenChannels: manifest.trunk.tokenChannels,
  hidden: manifest.trunk.atomChannels * 2, windowSize: manifest.trunk.atomWindow,
  // 🔴 f32 EITHER WAY, ON PURPOSE. The native attention downcasts q, k and v to
  // bfloat16; reproducing that would match the dump more closely and compute a
  // WORSE answer, and torch's bf16 accumulation is not reproducible here in any
  // case. The rope TABLE is still bf16, because that one is exact - see
  // src/esmfold2/atom-encoder-reference.js.
  attentionPrecision: "f32",
};
const atomBlocks = [];
for (let layer = 0; layer < shape.blocks; layer += 1) {
  const at = (leaf) => tensors[`atom/blocks/${layer}/${leaf}`];
  atomBlocks.push({
    adaln: at("adaln"), qkv: at("qkv"), attnGate: at("attnGate"),
    attnOut: at("attnOut"), ffnUp: at("ffnUp"), ffnDown: at("ffnDown"),
  });
}
const { tokenAct } = inputsEmbedder({
  refPos: feature("ref_pos"), refCharge: feature("ref_charge"),
  refElement: integers("ref_element"),
  refAtomNameChars: integers("ref_atom_name_chars"),
  refSpaceUid: feature("ref_space_uid"),
  atomToToken: integers("atom_to_token"),
  mask: feature("atom_attention_mask"),
}, shape, {
  atomLinear: tensors["atom/linear"], atomNormScale: tensors["atom/norm/scale"],
  atomNormOffset: tensors["atom/norm/offset"], atomToToken: tensors["atom/toToken"],
  blocks: atomBlocks,
});

// s_inputs = [pooled | aatype | profile | deletion_mean].
const embedderArgs = module("inputs_embedder").arguments;
const singleInputs = manifest.trunk.singleInputs;
const classes = embedderArgs.aatype.shape[2];
const aatype = Float32Array.from(embedderArgs.aatype.values);
const profile = Float32Array.from(embedderArgs.profile.values);
const deletion = Float32Array.from(embedderArgs.deletion_mean.values);
const sInputs = new Float32Array(n * singleInputs);
for (let token = 0; token < n; token += 1) {
  const to = token * singleInputs;
  for (let c = 0; c < shape.tokenChannels; c += 1) {
    sInputs[to + c] = tokenAct[token * shape.tokenChannels + c];
  }
  for (let c = 0; c < classes; c += 1) {
    sInputs[to + shape.tokenChannels + c] = aatype[token * classes + c];
    sInputs[to + shape.tokenChannels + classes + c] = profile[token * classes + c];
  }
  sInputs[to + singleInputs - 1] = deletion[token];
}

const zInit = zInitFromInputs(sInputs, n, singleInputs, channels,
  tensors["featuriser/zInit1"], tensors["featuriser/zInit2"]);
const relPos = relativePositionEncoding({
  residueIndex: integers("residue_index"), asymId: integers("asym_id"),
  symId: integers("sym_id"), entityId: integers("entity_id"),
  tokenIndex: integers("token_index"),
}, n, channels, tensors["featuriser/relPos"]);
const bonds = tokenBondEncoding(
  Float32Array.from(dump.features.token_bonds.values), n, channels,
  tensors["featuriser/tokenBonds"]);
// 🔴 THE LANGUAGE MODEL'S PAIR IS TAKEN FROM THE DUMP, NOT RECOMPUTED. The
// shim and the 36-block tower have their own oracle and their own checkers
// (tools/gpu/check-esmc-tower.js against oracle-dumps/esmc-59.json); running
// them again here would make one failure look like two and would not say which.
const lmPair = Float32Array.from(module("language_model").output);
for (let i = 0; i < zInit.length; i += 1) {
  zInit[i] += relPos[i] + bonds[i] + lmPair[i];
}

// The first loop's trunk input IS z_init + pair_loop_proj(0), so this is the
// assembly's own gate before a single block runs.
const recycle = {
  scale: tensors["recycle/norm/scale"], offset: tensors["recycle/norm/offset"],
  projection: tensors["recycle/projection"],
};
const zero = new Float32Array(pairs * channels);
// 🔴 HOISTED, because the first version called it INSIDE the map's callback -
// a full pair-sized projection per element, 409,600 times. It is the same
// mistake as putting a query inside a loop and it does not look like one when
// the expression reads as arithmetic.
const atZero = recycleProjection(zero, pairs, channels, recycle);
const firstInput = zInit.map((value, i) => value + atZero[i]);
let failures = 0;
const report = (label, score, limit = bound) => {
  const ok = score <= limit;
  if (!ok) failures += 1;
  console.log(`  ${label.padEnd(30)} relRMS ${score.toExponential(3)}   bound `
    + `${limit.toExponential(0)}   ${ok ? "ok" : "FAILED"}`);
};
report("z_init + pair_loop_proj(0)", relative(firstInput, Float32Array.from(dump.intoLoop["0"])));

// --- the recurrence.
const group = (layer, name, leaves) => {
  const out = {};
  for (const leaf of leaves) out[leaf] = tensors[`blocks/${layer}/${name}/${leaf}`];
  return out;
};
const mask = new Float32Array(pairs).fill(1);
let pair = new Float32Array(pairs * channels);
for (let loop = 0; loop < dump.loops; loop += 1) {
  const started = Date.now();
  const projected = recycleProjection(pair, pairs, channels, recycle);
  pair = zInit.map((value, i) => value + projected[i]);
  report(`  loop ${loop}, into the trunk`,
    relative(pair, Float32Array.from(dump.intoLoop[String(loop)])));
  const add = (delta) => { for (let i = 0; i < pair.length; i += 1) pair[i] += delta[i]; };
  for (let layer = 0; layer < blockCount; layer += 1) {
    add(triangleMultiplication(pair, mask, n, channels, "outgoing",
      group(layer, "triangleMultiplicationOutgoing", TRIANGLE)));
    add(triangleMultiplication(pair, mask, n, channels, "incoming",
      group(layer, "triangleMultiplicationIncoming", TRIANGLE)));
    add(transition(pair, pairs, channels, group(layer, "pairTransition", TRANSITION)));
  }
  report(`  loop ${loop}, out of the trunk`,
    relative(pair, Float32Array.from(dump.afterLoop[String(loop)])));
  console.log(`     (${((Date.now() - started) / 1000).toFixed(1)}s)`);
}

console.log(failures === 0
  ? "\nLocalFold computes ESMFold2's trunk from a sequence"
  : `\n${failures} stage(s) out of bound`);
process.exit(failures === 0 ? 0 : 1);
