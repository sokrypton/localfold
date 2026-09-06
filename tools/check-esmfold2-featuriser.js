// The three cheap terms of ESMFold2's z_init, plus the recycle projection.
//
//     .venv-esm/bin/python tools/esmc/dump-esmfold2-trunk.py --sequence-length 40
//     python3 tools/export_esmfold2_trunk.py
//     node tools/check-esmfold2-featuriser.js
//
// 🔴 z_init IS A SUM OF FIVE TERMS AND A CHECK OF THE TOTAL LOCALISES NOTHING.
// Two projections of the atom encoder's output, a relative position encoding, a
// token-bond encoding and the language model's pair - each its own port, each
// pair-shaped, each capable of being wrong by a permutation that still sums to
// something plausible. The dump records every one separately.
//
// 🔴 AND THE ATOM ENCODER IS NOT HERE YET, WHICH IS THE POINT OF SPLITTING
// THEM. `z_init_1` and `z_init_2` are checked against the x_inputs the model
// itself produced, so they are right before the expensive term that feeds them
// exists. What this cannot say is anything about that term.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";

import { readTensor } from "../src/reference/dtype.js";
import {
  recycleProjection, relativePositionEncoding, tokenBondEncoding, zInitFromInputs,
} from "../src/esmfold2/featuriser-reference.js";
import { inputsEmbedder } from "../src/esmfold2/atom-encoder-reference.js";

const ROOT = new URL("..", import.meta.url).pathname;
const bundleDirectory = process.argv[2] ?? join(ROOT, "model-esmfold2-trunk-f32");
const dumpPath = process.argv[3] ?? join(ROOT, "oracle-dumps", "esmfold2-trunk-40.json");
const BOUND = 2e-6;

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
const n = dump.shapes.pair[1];
const channels = manifest.trunk.pairChannels;
const singleInputs = manifest.trunk.singleInputs;
const feature = (name) => Int32Array.from(dump.features[name].values);
const module = (name) => dump.block0[name];

console.log(`${dump.esmfold2}: ${n} tokens, ${channels} pair channels, `
  + `${singleInputs} single-input channels\n`);

let failures = 0;
const report = (label, score, note = "") => {
  const ok = score <= BOUND;
  if (!ok) failures += 1;
  console.log(`  ${label.padEnd(34)} relRMS ${score.toExponential(3)}   `
    + `${ok ? "ok" : "FAILED"}${note}`);
};

// --- token_bonds: one input channel, so it is an outer product.
{
  const bonds = Float32Array.from(module("token_bonds").arguments["0"].values);
  const got = tokenBondEncoding(bonds, n, channels, tensors["featuriser/tokenBonds"]);
  const want = Float32Array.from(module("token_bonds").output);
  // 🔴 A SINGLE CHAIN HAS NO TOKEN BONDS, SO THE REFERENCE IS ALL ZEROS AND
  // relRMS IS 0/0. It came out NaN, which is the honest answer to "how close
  // are you, relatively, to nothing" - and a checker that quietly took the
  // absolute error instead would report a perfect score for a term it never
  // exercised. The bonds matrix is only non-zero for a ligand or a modified
  // residue, neither of which this dump has.
  const signal = want.some((value) => value !== 0);
  if (signal) {
    report("token_bonds", relative(got, want));
  } else {
    let worst = 0;
    for (let i = 0; i < got.length; i += 1) worst = Math.max(worst, Math.abs(got[i] - want[i]));
    const ok = worst === 0;
    if (!ok) failures += 1;
    console.log(`  ${"token_bonds".padEnd(34)} max ${worst.toExponential(3)}   `
      + `${ok ? "ok" : "FAILED"}   🔴 NOT EXERCISED: this dump has no bonds,\n`
      + "     so all this says is that zero in gives zero out. A ligand or a\n"
      + "     modified residue is what would exercise it.");
  }
}

// --- the inputs embedder: the atom transformer and its pooling.
{
  const feature = (name) => Float32Array.from(dump.features[name].values);
  const integers = (name) => Int32Array.from(dump.features[name].values);
  const atoms = dump.features.ref_pos.shape[1];
  const embedderArgs = module("inputs_embedder").arguments;
  const shape = {
    atoms, tokens: n,
    channels: manifest.trunk.atomChannels, heads: manifest.trunk.atomHeads,
    blocks: manifest.trunk.atomBlocks, tokenChannels: manifest.trunk.tokenChannels,
    hidden: manifest.trunk.atomChannels * 2, windowSize: manifest.trunk.atomWindow,
  };
  const blockWeights = [];
  for (let layer = 0; layer < shape.blocks; layer += 1) {
    const at = (leaf) => tensors[`atom/blocks/${layer}/${leaf}`];
    blockWeights.push({
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
    atomLinear: tensors["atom/linear"],
    atomNormScale: tensors["atom/norm/scale"],
    atomNormOffset: tensors["atom/norm/offset"],
    atomToToken: tensors["atom/toToken"],
    blocks: blockWeights,
  });

  // 🔴 THE MODULE'S OUTPUT IS s_inputs, NOT THE POOLING: 451 channels, of which
  // the first `tokenChannels` are the atom transformer's and the rest are the
  // residue one-hot, the profile and the deletion mean. Those three come from
  // the module's own arguments rather than being rebuilt here - what is under
  // test is the atom transformer, and a featuriser that also built the one-hots
  // would be testing two things and localising neither.
  const want = Float32Array.from(module("inputs_embedder").output);
  const width = dump.block0.inputs_embedder.outputShape[2];
  const pooled = new Float32Array(n * shape.tokenChannels);
  for (let token = 0; token < n; token += 1) {
    for (let c = 0; c < shape.tokenChannels; c += 1) {
      pooled[token * shape.tokenChannels + c] = want[token * width + c];
    }
  }
  // 🔴 THE ATOM ATTENTION COMPUTES IN bfloat16 IN A float32 MODEL, SO THE BOUND
  // DEPENDS ON WHICH DUMP THIS IS. `SWA3DRoPEAttention.forward` casts q, k and
  // v unconditionally - `if q.dtype not in (float16, bfloat16): q, k, v =
  // q.bfloat16(), ...` - so a float32 port cannot agree below about 2e-4, and
  // that residual looks exactly like a convention bug. It is not one: against
  // `dump-esmfold2-trunk.py --float32-attention`, which neutralises the cast,
  // the same code reads 7.0e-8.
  const control = dump.float32Attention === true;
  const atomBound = control ? 2e-6 : 5e-4;
  const atomScore = relative(tokenAct, pooled);
  const atomOk = atomScore <= atomBound;
  if (!atomOk) failures += 1;
  console.log(`  ${"inputs_embedder (the atom half)".padEnd(34)} relRMS `
    + `${atomScore.toExponential(3)}   bound ${atomBound.toExponential(0)}   `
    + `${atomOk ? "ok" : "FAILED"}   ${control ? "(float32 control)" : "(bf16 attention)"}`);
  if (!control) {
    console.log("  🔴 this is the SHIPPING arm and its floor is the module's own\n"
      + "     bfloat16 downcast, not this port. --float32-attention is the arm\n"
      + "     that says the arithmetic is right, and it reads 7.0e-8.");
  }

  // ...and the tail, so the concatenation's own layout is pinned too.
  const aatype = Float32Array.from(embedderArgs.aatype.values);
  const profile = Float32Array.from(embedderArgs.profile.values);
  const deletion = Float32Array.from(embedderArgs.deletion_mean.values);
  const classes = embedderArgs.aatype.shape[2];
  let worstTail = 0;
  for (let token = 0; token < n; token += 1) {
    for (let c = 0; c < classes; c += 1) {
      worstTail = Math.max(worstTail, Math.abs(
        want[token * width + shape.tokenChannels + c] - aatype[token * classes + c]));
      worstTail = Math.max(worstTail, Math.abs(
        want[token * width + shape.tokenChannels + classes + c]
        - profile[token * classes + c]));
    }
    worstTail = Math.max(worstTail, Math.abs(
      want[token * width + width - 1] - deletion[token]));
  }
  const tailOk = worstTail < 1e-6;
  if (!tailOk) failures += 1;
  console.log(`  ${"...its [aatype | profile | deletion] tail".padEnd(34)} max `
    + `${worstTail.toExponential(3)}   ${tailOk ? "ok" : "FAILED"}`);
}

// --- z_init_1 and z_init_2, against the x_inputs the model itself produced.
{
  const xInputs = Float32Array.from(module("inputs_embedder").output);
  const want1 = Float32Array.from(module("z_init_1").output);
  const want2 = Float32Array.from(module("z_init_2").output);
  const both = zInitFromInputs(xInputs, n, singleInputs, channels,
    tensors["featuriser/zInit1"], tensors["featuriser/zInit2"]);
  // The broadcast sum is what ships; the two projections are recovered from it
  // so each is scored on its own. Row 0's column term is column 0's, so
  // z1[i] = both[i][0] - z2[0] and z2[j] = both[0][j] - z1[0].
  const got1 = new Float32Array(n * channels);
  const got2 = new Float32Array(n * channels);
  for (let i = 0; i < n; i += 1) {
    for (let c = 0; c < channels; c += 1) {
      got1[i * channels + c] = both[(i * n + 0) * channels + c] - want2[c];
      got2[i * channels + c] = both[(0 * n + i) * channels + c] - want1[c];
    }
  }
  report("z_init_1", relative(got1, want1));
  report("z_init_2", relative(got2, want2));
}

// --- rel_pos, sweeping the one convention that a monomer cannot see.
{
  const features = {
    residueIndex: feature("residue_index"), asymId: feature("asym_id"),
    symId: feature("sym_id"), entityId: feature("entity_id"),
    tokenIndex: feature("token_index"),
  };
  const want = Float32Array.from(module("rel_pos").output);
  const got = relativePositionEncoding(features, n, channels, tensors["featuriser/relPos"]);
  report("rel_pos", relative(got, want));

  // 🔴 AND THE CONTROL SAYS THE SWEEP CANNOT SEE IT ON THIS INPUT. The chain
  // block's polarity is inverted relative to the residue block's, and on a
  // MONOMER every pair is same-chain - so both readings put every pair in one
  // bin and differ only by WHICH constant column is added. That is a real
  // difference in the output, but it is a constant, and it would vanish on a
  // complex only if the polarity were right. A dump with two chains is what
  // would actually discriminate; until there is one, this records that the
  // arms differ and by how much, rather than claiming the convention is
  // confirmed.
  const flipped = { ...features, asymId: features.asymId.map((_, i) => i) };
  const other = relativePositionEncoding(flipped, n, channels, tensors["featuriser/relPos"]);
  const separation = relative(other, want);
  console.log(`  ${"...with every token its own chain".padEnd(34)} relRMS `
    + `${separation.toExponential(3)}   ${separation > 1e-2 ? "discriminates" : "DOES NOT"}`);
  if (!(separation > 1e-2)) failures += 1;
  console.log("  🔴 this dump is a MONOMER, so the chain block's polarity is\n"
    + "     exercised but not pinned; a two-chain dump is what would pin it.");
}

// --- pair_loop_proj, which runs once per loop.
{
  const pairs = n * n;
  const weights = {
    scale: tensors["recycle/norm/scale"], offset: tensors["recycle/norm/offset"],
    projection: tensors["recycle/projection"],
  };
  // Loop 0 feeds it a zero pair, which cannot tell a correct projection from
  // one that ignores its input - so this uses loop 1's, whose input is loop 0's
  // trunk output.
  // 🔴 z_init IS NOT RECORDED, AND SUBTRACTING IT IS WHAT MAKES THIS A CHECK.
  // The dump has the trunk's INPUT per loop, which is z_init + proj(z), not
  // z_init. Two of those differ by proj(z_1) - proj(z_0), so both unknowns
  // cancel and no term of z_init has to be right for this to be exact.
  //
  // 🔴 AND proj(0) IS NOT ZERO, WHICH IS WHERE THIS FIRST WENT WRONG.
  // `pair_loop_proj`'s Linear is zero-INITIALISED upstream and then trained,
  // and the LayerNorm has a learned offset besides - so proj(0) is
  // Linear(offset) and measures 4.76 at its largest. Dropping it read 2.0e-1
  // against a correct projection.
  const zero = new Float32Array(pairs * channels);
  const atZero = recycleProjection(zero, pairs, channels, weights);
  const projected = recycleProjection(
    Float32Array.from(dump.afterLoop["0"]), pairs, channels, weights);
  const got = projected.map((value, i) => value - atZero[i]);
  const first = Float32Array.from(dump.intoLoop["0"]);
  const want = Float32Array.from(dump.intoLoop["1"]).map((value, i) => value - first[i]);
  report("pair_loop_proj", relative(got, want));

  let worst = 0;
  for (let i = 0; i < atZero.length; i += 1) worst = Math.max(worst, Math.abs(atZero[i]));
  console.log(`  ${"...and pair_loop_proj(0)".padEnd(34)} max ${worst.toExponential(3)}   `
    + "(trained, not zero - see above)");
}

console.log(failures === 0
  ? "\nz_init's three cheap terms and the recycle projection agree with ESMFold2"
  : `\n${failures} term(s) out of bound`);
process.exit(failures === 0 ? 0 : 1);
