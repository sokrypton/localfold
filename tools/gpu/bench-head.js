/**
 * Where one denoiser call's time goes.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-head.js
 *     node tools/gpu-chrome.mjs tools/gpu/bench-head.js --tokens=120 --calls=4
 *
 * 🔴 NO TRUNK, NO MSA, NO CONFIDENCE HEAD, AND A SYNTHETIC target_feat. A
 * sampler calls the head up to 200 times and everything else once, so the head
 * is the whole optimisation target - and running a real trunk first cost 800 ms
 * and pulled half the checkpoint over the wire before the first timed call. The
 * trunk's single and pair are deterministic noise of the right shape here: the
 * denoiser's COST does not depend on their contents, only their dimensions, and
 * this measures cost. Anything measuring accuracy belongs in a probe with an
 * oracle behind it.
 *
 * 🔴 THE FIRST CALL IS NOT THE NUMBER. It compiles every pipeline and warms
 * every buffer pool, and on a 59-token chain it runs about 40% long.
 *
 * 🔴 AND `steady` IS A MEDIAN, NOT A MEAN, over the calls after it. Run-to-run
 * spread here is around ten milliseconds a stage, which is wide enough that a
 * mean of two calls once reported a REMOVED pass as costing negative time -
 * so single-run deltas below about 15 ms mean nothing and a bisect built on
 * them will confidently name the wrong pass.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { perAtomConditioning } from "../../src/af3/atom-conditioning-reference.js";
import { Af3DiffusionHeadGpu } from "../../src/af3/diffusion-head-webgpu.js";
import { normalFrom } from "../../src/af3/fold.js";
import { openAf3Store } from "../../src/af3/weights.js";
import { diffusionWeights, atomReference } from "../../src/af3/diffusion-weights.js";
import { profileDevice } from "./profile.js";
import { deviceTuning, setDeviceTuning } from "../../src/runtime/device-profile.js";
import { ALPHAFOLD3 } from "../../src/af3/dialect.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** A residue mix with side chains of every length, repeated to the asked size. */
const ALPHABET = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

export async function main(device, args) {
  const tokens = Number(option(args, "tokens", "59"));
  // 🔴 THE TOKEN TILE, FORCED. The shipped rule pins it to 1 below 175 tokens,
  // and the question this flag exists to ask is whether that is still right now
  // that the weight traffic, not the workgroup count, looks like the binding
  // constraint - a tile of 1 gives one workgroup per TOKEN, and each of them
  // re-reads the projection's whole weight slice.
  // `--splitk=8/4` is splits/tile; `--splitk=off` disables it.
  const splitArg = option(args, "splitk", null);
  if (splitArg !== null) {
    // 🔴 MERGED OVER THIS DEVICE'S OWN SETTING, NOT WRITTEN OVER IT. The prior
    // carries six fields - splits, tile, crossover, outSplits, attnSplits,
    // attnTile, normSplits - and replacing the object with two of them moves
    // five knobs while claiming to sweep one, which is this repository's
    // recorded way of drawing the wrong conclusion from a clean-looking arm.
    const current = deviceTuning(device).diffusionSplitK ?? {};
    setDeviceTuning(device, {
      diffusionSplitK: splitArg === "off" ? null
        : (() => {
          const [splits, t] = splitArg.split("/").map(Number);
          return { ...current, splits,
                   ...(Number.isFinite(t) ? { tile: t } : {}), crossover: 1e9 };
        })(),
    });
  }
  // `--atomtile=1` forces it, `--atomtile=off` restores the shipped rule.
  const lanesArg = option(args, "lanes", null);
  if (lanesArg !== null) setDeviceTuning(device, { diffusionLanes: Number(lanesArg) });
  const outchunkArg = option(args, "outchunk", null);
  const atomArg = option(args, "atomtile", null);
  if (atomArg !== null) {
    setDeviceTuning(device, {
      atomRowTile: atomArg === "off" ? null
        : { below: Number(atomArg), atOrAbove: Number(atomArg), crossover: 0 },
    });
  }
  const splitNormArg = option(args, "normsplit", null);
  if (splitNormArg !== null) {
    setDeviceTuning(device, { diffusionNormSplit: splitNormArg !== "off" });
  }
  const normArg = option(args, "normsg", null);
  if (normArg !== null) {
    setDeviceTuning(device, { diffusionNormSubgroups: normArg !== "off" });
  }
  const tileArg = option(args, "tile", null);
  if (tileArg !== null) {
    const t = Number(tileArg);
    setDeviceTuning(device, { diffusionTokenTile: { below: t, atOrAbove: t, crossover: 0 } });
  }
  const calls = Number(option(args, "calls", "9"));
  // 🔴 SIXTEEN WAS NOT ENOUGH TO SEE THE ATOM BLOCKS. They are 106 ms of a
  // 261 ms call at 150 tokens and most of their passes fell off the end of the
  // list, so the two stages that are 41% of a denoiser call were invisible to
  // the only tool that profiles it.
  const passCount = Number(option(args, "passes", "16"));
  const sequence = Array.from({ length: tokens },
    (_, index) => ALPHABET[index % ALPHABET.length]).join("");

  const batch = featuriseProtein(sequence, {});
  const { dense } = batch;
  const store = await openAf3Store(option(args, "model", "/model-af3-full-f32/manifest.json"));
  const loadStart = performance.now();
  const weights = await diffusionWeights(store);
  const reference = await atomReference(store);
  const loadMs = Math.round(performance.now() - loadStart);

  const noise = normalFrom(11);
  const fill = (length) => {
    const out = new Float32Array(length);
    for (let index = 0; index < length; index += 1) out[index] = noise();
    return out;
  };

  const input = {
    shape: batch.shape,
    dialect: ALPHAFOLD3,
    conditioning: perAtomConditioning({
      positions: batch.refPos, mask: batch.refMask, element: batch.refElement,
      charge: batch.refCharge, atomNameChars: batch.refAtomNameChars,
    }, tokens, dense, reference),
    atomMask: batch.predDenseAtomMask, seqMask: batch.seqMask, features: batch.features,
    targetFeat: fill(tokens * 447),
    refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
    tokenAtomsToQueries: batch.tokenAtomsToQueries, queriesToKeys: batch.queriesToKeys,
    queriesToTokenAtoms: batch.queriesToTokenAtoms,
    tokensToQueries: batch.tokensToQueries, tokensToKeys: batch.tokensToKeys,
    trunkSingle: fill(tokens * 384), trunkPair: fill(tokens * tokens * 128),
    positionsNoisy: fill(tokens * dense * 3),
    noiseLevel: 16,
  };

  // --profile times every labelled compute pass; see tools/gpu/profile.js.
  const profile = args.includes("--profile") ? profileDevice(device) : null;
  // 🔴 THE DENOISER'S WEIGHT PRECISION, WHICH DEFAULTS TO f32 WHERE THE TRUNK'S
  // DEFAULTS TO f16. pairformer-block-webgpu.js takes f16 whenever the device
  // has it; diffusion-transformer-webgpu.js takes f32 unless a caller asks
  // otherwise, and no caller does. These kernels are weight-bandwidth-bound -
  // one multiply-add per weight loaded - so that is twice the bytes on the
  // exact passes that dominate a step.
  const weightPrecision = option(args, "weights", null);
  const head = new Af3DiffusionHeadGpu(device,
    weightPrecision === null ? undefined : { weightPrecision });
  const rows = [];
  for (let call = 0; call < calls; call += 1) {
    // ...the last call only, so pipeline compilation is out of the numbers.
    if (profile !== null && call === calls - 1) profile.reset();
    const started = performance.now();
    const out = await head.run(input, weights);
    rows.push({
      call, whole: Math.round(performance.now() - started),
      ...Object.fromEntries(Object.entries(out.timings).map(([k, v]) => [k, Math.round(v)])),
    });
  }
  const gpuPasses = profile === null ? undefined : await profile.report();
  // ...and how much of the last call the GPU spent doing nothing. See
  // profile.js's summary(): the sum of the passes cannot see the gaps.
  const gpuSummary = profile === null ? undefined : await profile.summary();
  profile?.restore();
  const after = rows.slice(1);
  const mean = (pick) => {
    const values = after.map(pick).sort((a, b) => a - b);
    return values[Math.floor(values.length / 2)];
  };
  const spread = (pick) => {
    const values = after.map(pick).sort((a, b) => a - b);
    return `${values[0]}-${values[values.length - 1]}`;
  };
  return {
    tokens, atoms: tokens * dense, subsets: batch.shape.subsets, weightLoadMs: loadMs,
    ...(gpuPasses === undefined ? {} : { gpuSummary, gpuPasses: gpuPasses.slice(0, passCount) }),
    rows,
    range: {
      whole: spread((r) => r.whole),
      atomEncoder: spread((r) => r["atom-encoder"]),
      transformer: spread((r) => r.transformer),
    },
    // Every stage the head reports, in the order it reports them, so a stage
    // added to the head shows up here without editing this file.
    steady: Object.fromEntries(Object.keys(rows[0])
      .filter((key) => key !== "call")
      .map((key) => [key, mean((row) => row[key])])),
  };
}
