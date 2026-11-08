/**
 * AF3's 24-block diffusion transformer in WebGL2 against
 * `cpu/af3/diffusion/diffusion.js`.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-diffusion-transformer.js [--falsify]
 *
 * 🔴 THIS IS 198 M OF THE DIFFUSION HEAD'S 203 M PARAMETERS, so it is the
 * largest single thing in the model and the one most worth a differential.
 *
 * 🔴 THE BLOCKS ARE NESTED SIX BY FOUR AND THE PAIR LOGITS FOLLOW THAT NESTING.
 * "The LayerNorm over the pair conditioning is computed ONCE and shared, but
 * each of the six SUPER-BLOCKS then projects it to its own four blocks' worth
 * of head biases. So there are six projections, not one and not twenty-four, and
 * a flat reading of the stack indexes the wrong weights for every block after
 * the fourth." The arms run TWO super-blocks of TWO, which is the smallest shape
 * in which a flat reading goes wrong - with one super-block it cannot.
 *
 * 🔴 AND THE CONDITIONING IS WIDER THAN THE ACTIVATION, which is the case the
 * atom block's reference cannot be asked about: 384 against 768 in the real
 * model, and here 24 against 32. "Assuming square runs on the atom stack and
 * reads the diffusion weights at the wrong stride."
 *
 * 🔴 AND THE SCALE IS THE PER-HEAD DIMENSION, TAKEN AFTER THE DIVISION BY THE
 * HEAD COUNT: "using the full 768 instead of 48 is a factor of four on every
 * logit, which softmax turns into a much flatter attention". The arms use a head
 * count greater than one, without which the two scales coincide.
 *
 * 🔴 AND boltz2's UP-GATE IS GATED ON THE CONDITIONING AS WELL AS ON THE WEIGHT,
 * because "boltz2's plain Transition - the one the diffusion conditioning's four
 * transitions use - has NO up-gate, and only its adaLN-conditioned
 * ConditionedTransitionBlock does". Both the conditioned and the unconditioned
 * transition are arms, and the up-gate arm asserts it moves the answer.
 */
import {
  conditionedTransition as cpuTransition, diffusionTransformer as cpuTransformer,
} from "../../cpu/af3/diffusion/diffusion.js";
import {
  conditionedTransition, diffusionTransformer,
} from "../../webgl/af3/diffusion/transformer.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { generator } from "./af2-synthetic-block.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "5e-6"));
  const tokens = Number(option(args, "tokens", "6"));
  const channels = Number(option(args, "channels", "32"));
  const condChannels = Number(option(args, "cond-channels", "24"));
  const pairChannels = Number(option(args, "pair-channels", "16"));
  const heads = Number(option(args, "heads", "4"));
  const dimension = Number(option(args, "dimension", "8"));
  const superBlocks = Number(option(args, "super-blocks", "2"));
  const perSuper = Number(option(args, "per-super", "2"));
  const transitionFactor = Number(option(args, "factor", "2"));
  const upGate = flag(args, "up-gate");
  const width = heads * dimension;
  const pairs = tokens * tokens;

  const random = generator(2236067);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  const adaptiveGroup = (prefix) => ({
    [`${prefix}SingleCondLayerNormScale`]: noise(condChannels, 0.2, 1),
    [`${prefix}SingleCondScaleWeights`]: noise(
      condChannels * channels, 1 / Math.sqrt(condChannels)),
    [`${prefix}SingleCondScaleBias`]: noise(channels, 0.1),
    [`${prefix}SingleCondBias`]: noise(
      condChannels * channels, 0.2 / Math.sqrt(condChannels)),
  });
  const intermediate = channels * transitionFactor;
  const blockWeights = () => ({
    ...adaptiveGroup(""),
    ...adaptiveGroup("ffw"),
    qProjection: noise(channels * width, 1 / Math.sqrt(channels)),
    qBias: noise(width, 0.1),
    kProjection: noise(channels * width, 1 / Math.sqrt(channels)),
    vProjection: noise(channels * width, 1 / Math.sqrt(channels)),
    gatingQuery: noise(channels * width, 1 / Math.sqrt(channels)),
    Transition2: noise(width * channels, 0.3 / Math.sqrt(width)),
    AdaptiveZeroCondWeights: noise(condChannels * channels, 1 / Math.sqrt(condChannels)),
    AdaptiveZeroCondBias: noise(channels, 0.1),
    ffwTransition1: noise(channels * intermediate * 2, 1 / Math.sqrt(channels)),
    ffwTransition2: noise(intermediate * channels, 0.3 / Math.sqrt(intermediate)),
    ffwAdaptiveZeroCondWeights: noise(
      condChannels * channels, 1 / Math.sqrt(condChannels)),
    ffwAdaptiveZeroCondBias: noise(channels, 0.1),
    ...(upGate
      ? { ffwAToB: noise(channels * intermediate, 1 / Math.sqrt(channels)) } : {}),
  });
  const weights = {
    channels, condChannels, pairChannels, heads, dimension,
    blocksPerSuperBlock: perSuper, transitionFactor,
    pairInputLayerNormScale: noise(pairChannels, 0.2, 1),
    superBlocks: Array.from({ length: superBlocks }, () => ({
      pairLogitsProjection: noise(
        pairChannels * perSuper * heads, 1 / Math.sqrt(pairChannels)),
      blocks: Array.from({ length: perSuper }, blockWeights),
    })),
  };

  const act = noise(tokens * channels, 1);
  const cond = noise(tokens * condChannels, 1);
  const pairCond = noise(pairs * pairChannels, 1);
  const mask = new Float32Array(tokens).fill(1);
  if (tokens > 2) mask[tokens - 1] = 0;

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  const uploadTree = (value, label) => {
    if (value instanceof Float32Array) return upload(value, label);
    if (Array.isArray(value)) return value.map((v, i) => uploadTree(v, `${label}.${i}`));
    if (typeof value === "number") return value;
    return Object.fromEntries(Object.entries(value).map(
      ([key, v]) => [key, uploadTree(v, `${label}.${key}`)]));
  };
  // The control skews half of each row's channels: the activation's first stop
  // is an unscaled LayerNorm, and a whole-row affine change is invisible to one.
  const actTensor = upload(falsify
    ? Float32Array.from(act, (v, i) => (i % channels < channels / 2 ? v * 1.01 + 0.01 : v))
    : act, "af3.difftx.act");
  const condTensor = upload(cond, "af3.difftx.cond");
  const pairTensor = upload(pairCond, "af3.difftx.pair");
  const maskTensor = upload(mask, "af3.difftx.mask");
  const uploaded = uploadTree(weights, "w.af3.difftx");

  const arms = [];
  const failures = [];
  const out = diffusionTransformer(execution, {
    act: actTensor, cond: condTensor, pairCond: pairTensor, mask: maskTensor,
    tokens, weights: uploaded,
  });
  const mine = Float32Array.from(execution.read(out));
  execution.release(out);
  const expected = cpuTransformer(act, cond, pairCond, mask, tokens, weights);
  const relative = relativeRms(mine, expected, "transformer");
  const ok = relative <= bar;
  arms.push({ arm: `the transformer, ${superBlocks} x ${perSuper} blocks`,
    relativeRms: Number(relative.toPrecision(4)), bar, ok });
  if (!ok) failures.push(`transformer: relRMS ${relative.toExponential(3)} over ${bar}`);

  // The two transition forms on their own, because the unconditioned one is what
  // the diffusion CONDITIONING uses and the transformer never exercises it.
  for (const conditioned of [true, false]) {
    const name = conditioned ? "conditioned transition" : "unconditioned transition";
    const blockOnly = weights.superBlocks[0].blocks[0];
    const plain = conditioned ? blockOnly : {
      ...blockOnly,
      ffwLayerNormScale: noise(channels, 0.2, 1),
      ffwLayerNormOffset: noise(channels, 0.1),
    };
    const uploadedPlain = uploadTree(plain, `w.af3.difftx.${name}`);
    const result = conditionedTransition(execution, {
      x: actTensor, ...(conditioned ? { cond: condTensor } : {}),
      rows: tokens, channels, factor: transitionFactor,
      weights: uploadedPlain, prefix: "", condChannels,
    });
    const mineTransition = Float32Array.from(execution.read(result));
    execution.release(result);
    const expectedTransition = cpuTransition(
      falsify
        ? Float32Array.from(act,
          (v, i) => (i % channels < channels / 2 ? v * 1.01 + 0.01 : v))
        : act,
      conditioned ? cond : null, tokens, channels, transitionFactor, plain, "",
      condChannels);
    // ...the reference reads the SKEWED activation here, so this arm compares
    // arithmetic rather than the control; the control is the transformer's.
    const relativeTransition = relativeRms(mineTransition, expectedTransition, name);
    const okTransition = relativeTransition <= bar;
    arms.push({
      arm: name, relativeRms: Number(relativeTransition.toPrecision(4)), bar,
      ok: okTransition, falsifiable: false,
      reason: "the reference is handed the same skewed activation",
    });
    if (!okTransition) {
      failures.push(`${name}: relRMS ${relativeTransition.toExponential(3)} over ${bar}`);
    }
  }

  for (const tensor of held) execution.release(tensor);
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const falsifiable = arms.filter((row) => row.falsifiable !== false);
  const red = falsifiable.filter((row) => !row.ok).length;
  return {
    shape: {
      tokens, channels, condChannels, pairChannels, heads, dimension,
      superBlocks, perSuper, transitionFactor,
    },
    upGate,
    maskedTokens: tokens - mask.reduce((a, b) => a + b, 0),
    bar,
    falsify,
    arms,
    memory: execution.stats(),
    verdict: falsify
      ? (red === falsifiable.length
        ? `ok: all ${falsifiable.length} falsifiable arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${falsifiable.length} falsifiable arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms against the f64 reference`
        : `FAILED: ${failures.join("; ")}`),
  };
}
