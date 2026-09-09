/**
 * Does the ESMFold2 trunk's DEVICE-decoded pair track compute the host one's
 * answer?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-esmfold2-trunk-pack.js \
 *       --bundle=/model-esmfold2-int5 --length=40
 *
 * 🔴 THIS TRUNK SHARES `packPairTrackWeights` WITH THREE AF3 STACKS AND WAS THE
 * ONE CALLER STILL BUILDING ALL FIVE ON THE HOST - 828 ms of `trunk 0` against
 * 32 ms for the same block warm, out of a 2.7 s first fold. Wiring it to
 * `residentPairTrackOnDevice` is four lines and moves four interleaves and two
 * transposes onto the GPU, which is exactly the kind of change that returns a
 * plausible tensor when it is wrong.
 *
 * The bar is ZERO differing elements, not a tolerance: the decoder is
 * bit-identical to the host one (check-quantised-upload.js) and the destination
 * layout is the same, so any difference at all is a mapping and not arithmetic.
 *
 * 🔴 AND IT RUNS THE TWO ARMS IN ONE PROCESS OVER ONE INPUT, because this box
 * drifts up to 3.2x between runs and, more to the point, a second process would
 * be a second random pair representation.
 */
import { Esmfold2TrunkGpu } from "../../src/esmfold2/trunk-webgpu.js";
import { blockUploadStats } from "../../src/runtime/quantised-upload.js";
import { reader } from "./fold-esmfold2.js";
import { trunkBlockWeights } from "../../src/esmfold2/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const bundle = option(args, "bundle", "/model-esmfold2-int5");
  const n = Number(option(args, "length", "40"));
  const blockCount = Number(option(args, "blocks", "4"));
  const { read } = reader(bundle);
  const blocks = [];
  for (let layer = 0; layer < blockCount; layer += 1) {
    // eslint-disable-next-line no-await-in-loop
    blocks.push(await trunkBlockWeights(read, layer));
  }

  // A deterministic pair representation: the arms must see the same one.
  const pairs = n * n;
  const channels = 256;
  const pair = new Float32Array(pairs * channels);
  let seed = 12345;
  for (let i = 0; i < pair.length; i += 1) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    pair[i] = (seed / 0x7fffffff) * 2 - 1;
  }
  const pairMask = new Float32Array(pairs).fill(1);

  const run = async (devicePairTrack) => {
    const trunk = new Esmfold2TrunkGpu(device, { devicePairTrack });
    const out = await trunk.run({ pair: pair.slice(), pairMask }, blocks, { n, channels });
    return out.pair;
  };
  const host = await run(false);
  // 🔴 A GATE THAT CANNOT FAIL IS NOT A GATE. If `residentPairTrackOnDevice`
  // refused - a source shape it does not read, a codec the planner declines -
  // both arms would be the host packer and `differing` would be zero for the
  // wrong reason. So the device arm has to be seen decoding.
  const decodesBefore = blockUploadStats.calls;
  const gpu = await run(true);
  const decodes = blockUploadStats.calls - decodesBefore;

  let differing = 0;
  let worst = 0;
  for (let i = 0; i < host.length; i += 1) {
    if (host[i] === gpu[i]) continue;
    differing += 1;
    worst = Math.max(worst, Math.abs(host[i] - gpu[i]));
  }
  const finite = [...gpu].every((v) => Number.isFinite(v));
  const ok = differing === 0 && finite && decodes >= 3 * blockCount;
  if (!ok) {
    throw new Error(`the trunk's device pair track differs: ${differing} elements, worst `
      + `${worst}, finite ${finite}, ${decodes} device decodes over ${blockCount} blocks `
      + "(three a block: two triangles and the transition)");
  }
  return { bundle, n, channels, blocks: blockCount, elements: host.length,
           differing, worst, finite, decodes, ok };
}
