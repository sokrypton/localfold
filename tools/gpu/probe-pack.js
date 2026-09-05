/**
 * What the diffusion transformer's FIRST call actually spends, and on what.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-pack.js
 *
 * 🔴 docs/AF3.md CALLS THAT CALL "THE LARGEST SINGLE ITEM LEFT" AND ATTRIBUTES
 * IT TO "the one-off f16 conversion of the transformer's weights". That is a
 * THIRD of it. Measured over all 24 blocks of /model-af3-int5/, which pack to
 * 378.2 MiB:
 *
 *     cold, f16 (decode + convert + concatenate)   494 ms
 *     warm, f16 (convert + concatenate)            244
 *     warm, f32 (concatenate alone)                 66
 *
 * so the int5 DECODE is 250 ms, the f16 conversion 178, and the concatenation
 * 66. The decode is the store binding a block lazily: the first read of each of
 * its ~40 tensors decodes that tensor out of the shard, and it is paid whatever
 * precision the buffer ends up in. Dropping to f32 weights would save 178 ms of
 * 494 and cost 378 MiB more on the device, which is not a trade worth making.
 *
 * 🔴 AND ALL OF IT IS HOST WORK THAT DEPENDS ONLY ON THE WEIGHTS. Nothing in it
 * reads the trunk, and a trunk pass leaves the host idle - `bench-trunk.js`
 * reports 9.4 ms of encoding against 2948 of waiting - so in principle these
 * 494 ms could be hidden behind it entirely. What stops it is memory, not
 * ordering: `src/af3/fold.js` releases the trunk's ~350 MiB of resident weights
 * BEFORE the sampler makes the transformer's 378 MiB resident, precisely so the
 * two never coexist. Packing during the trunk means holding those 378 MiB
 * somewhere - on the host if the upload is deferred, on the device if it is not
 * - and either way it is 378 MiB bought with 494 ms. Measured here so that the
 * trade can be made deliberately rather than discovered.
 */
import { openAf3Store } from "../../src/af3/weights.js";
import { diffusionWeights } from "../../src/af3/diffusion-weights.js";
import { packBlockWeights } from "../../src/af3/diffusion-transformer-webgpu.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  void device;
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const weights = await diffusionWeights(store);
  const blocks = weights.transformer.superBlocks.flatMap((group) => group.blocks);

  // 🔴 COLD THEN WARM, ON DIFFERENT BLOCKS AND THEN THE SAME ONE. The first
  // pack of a block pays the decode AND the conversion; a second pack of the
  // same block pays the conversion alone, because the store's memo now holds
  // the decoded tensors.
  const coldStart = performance.now();
  let bytes = 0;
  for (const block of blocks) bytes += packBlockWeights(block, "f16").data.byteLength;
  const coldMs = performance.now() - coldStart;

  const warmStart = performance.now();
  for (const block of blocks) packBlockWeights(block, "f16");
  const warmMs = performance.now() - warmStart;

  const wideStart = performance.now();
  for (const block of blocks) packBlockWeights(block, "f32");
  const wideMs = performance.now() - wideStart;

  return {
    nativeFloat16: typeof globalThis.Float16Array === "function",
    blocks: blocks.length,
    packedMiB: Number((bytes / (1024 * 1024)).toFixed(1)),
    coldMs: Number(coldMs.toFixed(0)),
    warmMs: Number(warmMs.toFixed(0)),
    wideMs: Number(wideMs.toFixed(0)),
    decodeMs: Number((coldMs - warmMs).toFixed(0)),
  };
}
