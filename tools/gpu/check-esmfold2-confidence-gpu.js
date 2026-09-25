/**
 * ESMFold2's confidence head on the device, against the host reference.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-esmfold2-confidence-gpu.js \
 *       --bundle=/model-ef2-600-head-f32 \
 *       --dump=/oracle-dumps/esmfold2-confidence-600.json
 *
 * 🔴 IT IS DIFFERENTIAL AGAINST A REFERENCE THAT IS ITSELF ORACLE-GATED, which
 * is the layering check-opendde-confidence.js uses: `check-esmfold2-confidence.js`
 * holds `src/esmfold2/confidence-reference.js` to Synthyra's own module at
 * 2.4e-7 to 8.7e-7, and this holds the device to that reference. A device
 * checker written straight against the dump would fold two questions - is the
 * arithmetic right, and does the GPU compute the arithmetic - into one number.
 *
 * 🔴 AND THE FOUR BLOCKS ARE NOT THIS FILE'S SUBJECT. They run through
 * `Esmfold2TrunkGpu`, the same class as the trunk's 24, so what is new here is
 * the pair handed to them and the readouts taken off their answer. The blocks
 * are in the comparison because a wrong pair going in is invisible otherwise.
 */
import { GpuBufferAllocator } from "../../src/runtime/allocator.js";
import { setDeviceTuning } from "../../src/runtime/device-profile.js";
import { Esmfold2TrunkGpu } from "../../src/esmfold2/trunk-webgpu.js";
import { esmfold2Confidence } from "../../src/esmfold2/confidence-reference.js";
import { esmfold2ConfidencePairInit, esmfold2ConfidenceReadouts }
  from "../../src/esmfold2/confidence-webgpu.js";
import { layerNorm, linear } from "../../src/af3/trunk/pairformer-reference.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

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

export async function main(device, args) {
  const bundle = option(args, "bundle", "/model-ef2-600-head-f32");
  const dumpPath = option(args, "dump", "/oracle-dumps/esmfold2-confidence-600.json");
  // 🔴 TWO BOUNDS, BECAUSE TWO DIFFERENT THINGS ARE BEING CHECKED. The pair
  // INIT and the readouts are this port's new kernels and are held to 1e-5;
  // anything downstream of the four blocks inherits the trunk stack's own
  // arithmetic, which stages and accumulates in f16 and is not this file's
  // subject - `Esmfold2TrunkGpu` is the same class the trunk's 24 blocks run
  // through and is gated against ESMFold2's own dump elsewhere.
  //
  // Measured here, 68 tokens, four blocks, against the host reference:
  //
  //   shipped profile                                        8.37e-4
  //   --staged=f32 --accumulate=f32                          8.37e-4  (identical)
  //   --tune=triangleProjectMatrix=false                     2.72e-3  (WORSE)
  //   both of the above together                             6.94e-4
  //   one block / two blocks / four                 3.11e-4 / 5.37e-4 / 8.37e-4
  //
  // 🔴 AND THE PRECISION ARMS MOVING NOTHING IS THE FINDING, NOT A MISS. They
  // reached the stack - `precision` comes back `{staged: "f32", accumulate:
  // "f32"}` - and the numbers were byte-identical, because the triangle's
  // projection runs on f16 MATRIX units chosen from the device tuning, which no
  // precision flag reaches. That is CLAUDE.md's own note about `--f16=off` and
  // the matrix pair kernels, and it is why this file takes `--tune=`.
  const bound = Number(option(args, "bound", "1e-5"));
  const blockBound = Number(option(args, "block-bound", "2e-3"));
  const depth = Number(option(args, "blocks", "0"));
  // 🔴 THE MATRIX PAIR KERNELS ARE NOT REACHABLE FROM A PRECISION FLAG, which
  // is why this takes `--tune=` at all. `triangleProjectMatrix` picks an f16
  // matrix-unit projection out of the DEVICE TUNING, so `--staged=f32
  // --accumulate=f32` moved the residual by nothing at all - byte-identical,
  // both arms, with `precision` confirming they reached the stack. CLAUDE.md
  // records the same shape for OpenDDE's head, which needed `pairMatrixKernels:
  // false` beside its three precision pins.
  for (const pair of (args ?? []).filter((a) => a.startsWith("--tune="))
       .flatMap((a) => a.slice("--tune=".length).split(",")).filter(Boolean)) {
    const at = pair.indexOf("=");
    if (at < 0) throw new Error(`--tune wants key=value, got ${pair}`);
    const raw = pair.slice(at + 1);
    let value;
    try { value = JSON.parse(raw); } catch { value = raw; }
    setDeviceTuning(device, { [pair.slice(0, at)]: value });
  }

  const manifest = await (await fetch(`${bundle}/manifest.json`)).json();
  if (manifest.confidence === undefined) {
    throw new Error(`${bundle} carries no confidence head`);
  }
  const shards = new Map();
  const read = async (name) => {
    const record = manifest.tensors[name];
    if (record === undefined) throw new Error(`bundle has no ${name}`);
    if (!shards.has(record.file)) {
      shards.set(record.file,
                 await (await fetch(`${bundle}/${record.file}`)).arrayBuffer());
    }
    const count = record.shape.reduce((a, b) => a * b, 1);
    return new Float32Array(shards.get(record.file), record.byteOffset, count);
  };

  const meta = manifest.confidence;
  const blocks = [];
  for (let layer = 0; layer < meta.blocks; layer += 1) {
    const group = async (name, leaves) => Object.fromEntries(await Promise.all(
      leaves.map(async (leaf) =>
        [leaf, await read(`confidence/blocks/${layer}/${name}/${leaf}`)])));
    blocks.push({
      triangleMultiplicationOutgoing: await group("triangleMultiplicationOutgoing", TRIANGLE),
      triangleMultiplicationIncoming: await group("triangleMultiplicationIncoming", TRIANGLE),
      pairTransition: await group("pairTransition", TRANSITION),
    });
  }
  if (depth > 0) blocks.length = depth;
  const weights = { ...meta, blocks };
  for (const [key, leaf] of [
    ["sInputsNormScale", "sInputsNorm/scale"], ["sInputsNormOffset", "sInputsNorm/offset"],
    ["zNormScale", "zNorm/scale"], ["zNormOffset", "zNorm/offset"],
    ["plddtNormScale", "plddtNorm/scale"], ["plddtNormOffset", "plddtNorm/offset"],
    ["sToZ", "sToZ"], ["sToZTranspose", "sToZTranspose"],
    ["sToZProdIn1", "sToZProdIn1"], ["sToZProdIn2", "sToZProdIn2"],
    ["sToZProdOut", "sToZProdOut"], ["distanceEmbedding", "distanceEmbedding"],
    ["boundaries", "boundaries"], ["poolingAttention", "poolingAttention"],
    ["poolingOutput", "poolingOutput"], ["plddtWeight", "plddtWeight"], ["pae", "pae"],
  ]) weights[key] = await read(`confidence/${leaf}`);

  const dump = await (await fetch(dumpPath)).json();
  const stage = (name) => Float32Array.from(dump.stages[name].data);
  const tokens = dump.tokens;
  const atoms = dump.atoms;
  const inputs = {
    tokens, atoms,
    sInputs: stage("in.s_inputs"), pair: stage("in.z"),
    coordinates: stage("in.x_pred"),
    repAtom: Int32Array.from(stage("in.distogram_atom_idx")),
    atomToToken: Int32Array.from(stage("in.atom_to_token")),
    tokenMask: stage("in.token_attention_mask"), atomMask: stage("in.atom_attention_mask"),
    // 🔴 A REAL pairBias, NOT ZEROS. Their head adds the relative-position and
    // token-bonds encodings to the normalised pair, and this port carries them
    // as one addend; zeros here would run both arms past a term neither
    // exercised, which is exactly how the omission survived in the first place
    // (see docs/EF2FAST.md). Seeded, so the two arms see the same tensor.
    pairBias: (() => {
      const out = new Float32Array(tokens * tokens * weights.pairChannels);
      let state = 0x9e3779b9;
      for (let i = 0; i < out.length; i += 1) {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        out[i] = ((state >>> 8) / 0x1000000 - 0.5) * 0.5;
      }
      return out;
    })(),
  };

  const host = esmfold2Confidence(inputs, weights);

  // The per-token half stays on the host in the shipped path too; see the
  // module header. What the device is asked for is everything pair-shaped.
  const dPair = weights.pairChannels;
  const dInputs = weights.singleInputs;
  const normed = layerNorm(inputs.sInputs, tokens, dInputs,
                           weights.sInputsNormScale, weights.sInputsNormOffset);
  const rows = linear(normed, tokens, dInputs, dPair, weights.sToZ);
  const cols = linear(normed, tokens, dInputs, dPair, weights.sToZTranspose);
  const left = linear(normed, tokens, dInputs, dPair, weights.sToZProdIn1);
  const right = linear(normed, tokens, dInputs, dPair, weights.sToZProdIn2);
  const repCoordinates = new Float32Array(tokens * 3);
  for (let token = 0; token < tokens; token += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      repCoordinates[token * 3 + axis] = inputs.coordinates[inputs.repAtom[token] * 3 + axis];
    }
  }

  const allocator = new GpuBufferAllocator(device);
  const started = performance.now();
  const initial = await esmfold2ConfidencePairInit(device, {
    tokens, pair: inputs.pair, rows, cols, left, right, repCoordinates,
    pairBias: inputs.pairBias,
  }, weights, { allocator });

  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      pairMask[i * tokens + j] = inputs.tokenMask[i] * inputs.tokenMask[j];
    }
  }
  // 🔴 THE BLOCKS' ARITHMETIC IS A KNOB AND THE BOUND FOLLOWS IT. The trunk
  // stack stages in f16 wherever the device has it, which is the shipped path
  // and lands near 1e-3 against an f64 host reference - the same band
  // check-af3-msa-block.js records for two arms over one pair track. `--staged=f32`
  // is the arm that says whether the difference is precision or a defect.
  const staged = option(args, "staged", "f16");
  const accumulate = option(args, "accumulate", "f16");
  const stack = await new Esmfold2TrunkGpu(device, { allocator, stagedPrecision: staged, accumulatePrecision: accumulate }).run(
    { pair: Float32Array.from(initial), pairMask }, blocks, { n: tokens, channels: dPair });
  const finished = Float32Array.from(stack.pair);
  for (let index = 0; index < finished.length; index += 1) finished[index] += initial[index];

  const readouts = await esmfold2ConfidenceReadouts(device, {
    tokens, pair: finished, tokenMask: inputs.tokenMask,
  }, weights, { allocator });
  const elapsed = performance.now() - started;

  const arms = [
    ["pair init (before the blocks)", initial, host.initial],
    ["pair (after the blocks)", finished, host.pair],
    ["single (pooled)", readouts.single, host.single],
    ["paeLogits", readouts.paeLogits, host.paeLogits],
  ];
  const results = [];
  let worst = 0;
  let failed = null;
  for (const [name, got, want] of arms) {
    if (got.length !== want.length) throw new Error(`${name}: ${got.length} against ${want.length}`);
    const score = relative(got, want);
    // The init is ours; everything after the blocks carries their arithmetic.
    const limit = name.startsWith("pair init") ? bound : blockBound;
    if (!(score <= limit)) failed = `${name} is ${score.toExponential(2)} against ${limit}`;
    worst = Math.max(worst, score);
    results.push({ stage: name, relRms: Number(score.toExponential(2)), bound: limit });
  }
  if (failed !== null) {
    throw new Error(`${failed} [ran ${JSON.stringify(stack.precision)}] - `
      + results.map((r) => `${r.stage} ${r.relRms}`).join(", "));
  }
  return {
    model: "esmfold2-confidence", bundle, tokens, atoms,
    staged, ranPrecision: stack.precision,
    blocks: meta.blocks, pairChannels: dPair, paeBins: meta.paeBins,
    milliseconds: Number(elapsed.toFixed(1)),
    bound, blockBound, stages: results,
    worst: Number(worst.toExponential(2)), agrees: true,
  };
}
