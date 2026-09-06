// Does src/esmfold2/trunk-webgpu.js compute ESMFold2's folding trunk?
//
//     .venv-esm/bin/python tools/esmc/dump-esmfold2-trunk.py --sequence-length 40
//     python3 tools/export_esmfold2_trunk.py
//     node tools/gpu-chrome.mjs tools/gpu/check-esmfold2-trunk-gpu.js
//
// Two things at once, and the second is the one that needed writing.
//
// 🔴 THE ORACLE ARM says the GPU computes what the native model computed, per
// recycle, against the values dump-esmfold2-trunk.py recorded going into and
// coming out of the trunk. The CPU reference already passes this at 1.4e-6
// (tools/check-esmfold2-trunk.js) - so a GPU failure here is the kernels or
// the sequence, not the weight conversion.
//
// 🔴 THE SKIP ARM says the shipped path skips the grid attention and NOTHING
// ELSE. ESMFold2's block is AF3's with the two grid attentions removed, and the
// cheap way to get that is to leave the passes in with a zeroed output
// projection - an attention that adds zero is the identity, so a zeroed AF3
// block already IS this block. That is pure waste (`grid.attend` is 34.6% of an
// AF3 trunk's GPU time at 700 tokens, and this trunk runs 24 blocks four times
// over), so the shipped path drops the passes instead. Dropping passes from a
// track whose five updates each read the pair as the last one left it is
// exactly the kind of change that returns a plausible tensor, so the two arms
// are run over the same weights at the same shapes and compared BIT FOR BIT.
//
// 🔴 AND THE ZEROED WEIGHTS ARE SYNTHESISED HERE, NOT SHIPPED. Five c x c
// matrices a block over 24 blocks is 37.7 MiB of zeros; a bundle carrying them
// would be paying to say something this checker says for free.
import { readTensor } from "../../src/reference/dtype.js";
import { Esmfold2TrunkGpu } from "../../src/esmfold2/trunk-webgpu.js";

const TRIANGLE = ["leftNormInputScale", "leftNormInputOffset", "centerNormScale",
  "centerNormOffset", "outputProjection", "gatingLinear", "projection", "gate"];
const TRANSITION = ["inputLayerNormScale", "inputLayerNormOffset",
  "transition1", "transition2"];

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const relative = (got, want) => {
  let error = 0, total = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i] - want[i];
    error += d * d;
    total += want[i] * want[i];
  }
  return Math.sqrt(error / total);
};

/** How many elements of two arrays differ, exactly. */
const differing = (a, b) => {
  let count = 0;
  for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) count += 1;
  return count;
};

/** A grid attention that adds zero: the identity, spelled as an AF3 module. */
function zeroAttention(channels, heads) {
  const zeros = (n) => new Float32Array(n);
  const scale = new Float32Array(channels).fill(1);
  return {
    actNormScale: scale, actNormOffset: zeros(channels),
    pairBiasProjection: zeros(channels * heads),
    qProjection: zeros(channels * channels),
    kProjection: zeros(channels * channels),
    vProjection: zeros(channels * channels),
    gatingQuery: zeros(channels * channels),
    // 🔴 THE OUTPUT PROJECTION IS THE ONE THAT HAS TO BE ZERO. The rest could
    // be anything: with it zero the module adds zero whatever the attention
    // computed, which is what makes the two arms comparable at all.
    outputProjection: zeros(channels * channels),
    heads, dimension: channels / heads,
  };
}

export async function main(device, args = []) {
  const bundle = option(args, "bundle", "/model-esmfold2-trunk-f32");
  const dumpPath = option(args, "dump", "/oracle-dumps/esmfold2-trunk-40.json");
  const bound = Number(option(args, "bound", "2e-4"));
  // The zeroed arm costs a full second trunk pass per loop, so it runs on one
  // loop by default: what it is testing is a property of the ENCODING, which
  // does not vary with the input.
  const zeroedLoops = Number(option(args, "zeroed-loops", "1"));
  // 🔴 PRECISION IS AN AXIS, AND EACH ARM IS HELD TO ITS OWN ARITHMETIC. The
  // shipped defaults stage the triangle's tiles and hold its projection
  // accumulators in f16 - worth 1.55x on that kernel - and this trunk runs 24
  // blocks four times over, so whatever they round compounds further here than
  // in an AF3 trunk. Raising one bound to cover both arms would stop the f32
  // path being checked at all; see CLAUDE.md.
  // Written `staged:accumulate`, or one name for both. They are separate
  // kernels - the transition stages its tiles, the triangle's projection holds
  // eight vec4 of accumulators - and bundling them under one word hides which
  // of the two is paying for the error.
  const precisions = option(args, "precision", "f32,default").split(",");

  const dump = await (await fetch(dumpPath)).json();
  const manifest = await (await fetch(`${bundle}/manifest.json`)).json();
  if (manifest.trunk?.source !== dump.esmfold2) {
    throw new Error(`bundle is ${manifest.trunk?.source} and the dump is ${dump.esmfold2}`);
  }
  const n = dump.shapes.pair[1];
  const channels = manifest.trunk.pairChannels;
  const blockCount = manifest.trunk.blocks;
  if (blockCount !== dump.blocks) {
    throw new Error(`${blockCount} blocks against the dump's ${dump.blocks}`);
  }

  const shards = new Map();
  const read = async (name) => {
    const record = manifest.tensors[name];
    if (record === undefined) throw new Error(`bundle has no tensor ${name}`);
    if (!shards.has(record.file)) {
      shards.set(record.file, await (await fetch(`${bundle}/${record.file}`)).arrayBuffer());
    }
    return readTensor(record, shards.get(record.file), record.byteOffset ?? 0, true);
  };
  const group = async (layer, name, leaves) => {
    const out = {};
    for (const leaf of leaves) out[leaf] = await read(`blocks/${layer}/${name}/${leaf}`);
    return out;
  };

  const heads = manifest.trunk.zeroedAttentionHeads;
  const blocks = [];
  for (let layer = 0; layer < blockCount; layer += 1) {
    blocks.push({
      triangleMultiplicationOutgoing:
        await group(layer, "triangleMultiplicationOutgoing", TRIANGLE),
      triangleMultiplicationIncoming:
        await group(layer, "triangleMultiplicationIncoming", TRIANGLE),
      pairTransition: await group(layer, "pairTransition", TRANSITION),
      // Read by the zeroed arm alone; the shipped one never looks.
      pairAttention1: zeroAttention(channels, heads),
      pairAttention2: zeroAttention(channels, heads),
    });
  }

  // A single chain with no padding: every pair is live, which is how the dump
  // was taken. That is why this is a constant and not a feature.
  const pairMask = new Float32Array(n * n).fill(1);

  const arms = [];
  let failures = 0;
  const keys = Object.keys(dump.intoLoop);
  for (const precision of precisions) {
    // The f32 arm is the one that says the kernels compute the trunk; the f16
    // arm is what the page would run, held to the bound its own rounding
    // implies rather than to the other's.
    // 🔴 `default` MEANS OVERRIDE NOTHING, which is the arm that matters most
    // and the one an explicit list cannot express: naming the shipped settings
    // here would make this checker agree with itself when the stack's default
    // moved. It passes no options and lets the stack choose.
    const [staged, accumulate] = precision === "default" ? [undefined, undefined]
      : (precision.includes(":") ? precision.split(":") : [precision, precision]);
    const limit = (staged === "f32" && accumulate === "f32")
      ? bound : Number(option(args, "f16-bound", "1e-3"));
    const settings = staged === undefined
      ? {} : { stagedPrecision: staged, accumulatePrecision: accumulate };
    const stack = new Esmfold2TrunkGpu(device, settings);
    for (const key of keys) {
      const pair = Float32Array.from(dump.intoLoop[key]);
      const want = Float32Array.from(dump.afterLoop[key]);
      const skipped = await stack.run({ pair, pairMask }, blocks, { n, channels });
      const score = relative(skipped.pair, want);
      if (!(score <= limit)) failures += 1;
      const arm = {
        precision, loop: Number(key), relRms: score, within: score <= limit, bound: limit,
        milliseconds: Number(skipped.elapsedMilliseconds.toFixed(1)),
        peakMiB: Number((skipped.memory.peakBytes / 2 ** 20).toFixed(1)),
      };

      if (Number(key) < zeroedLoops) {
        const zeroed = await stack.run(
          { pair: Float32Array.from(dump.intoLoop[key]), pairMask }, blocks,
          { n, channels, gridAttention: true });
        arm.zeroed = {
          differingElements: differing(skipped.pair, zeroed.pair),
          milliseconds: Number(zeroed.elapsedMilliseconds.toFixed(1)),
          relRms: relative(zeroed.pair, want),
        };
        // 🔴 BIT-IDENTICAL, NOT "CLOSE". The two arms run the same kernels
        // over the same weights; the only difference is passes that provably
        // add zero. Anything but zero differing elements is the skip having
        // moved something else - which is what a tolerance would hide.
        if (arm.zeroed.differingElements !== 0) failures += 1;
        arm.zeroed.speedup = Number((arm.zeroed.milliseconds / arm.milliseconds).toFixed(2));
      }
      arms.push(arm);
    }
  }

  return {
    bundle, dump: dumpPath, tokens: n, blocks: blockCount, channels,
    loops: keys.length, bound, precisions, arms, failures,
    ok: failures === 0,
  };
}
