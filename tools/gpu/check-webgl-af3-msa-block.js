/**
 * One AF3 MSA block in WebGL2, against TWO references.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-msa-block.js [--falsify]
 *
 * `cpu/af3/trunk/msa.js`'s `msaBlock` is the f64 reference and `Af3MsaStackGpu`
 * the WGSL one, both in this page, with the two also held to each other.
 *
 * 🔴 THE TWO CROSSING LINES ARE WHAT THIS GATE IS FOR, AND BOTH ORDERS ARE
 * ARMS. "AF3 takes the outer product of the MSA as it arrived and then updates
 * the MSA against the pair that outer product just changed; OpenDDE and Boltz-2
 * update the MSA first and take the outer product of the result. Same modules,
 * same weights, different function." Neither side has a default for it - the CPU
 * reference throws, the WGSL stack throws, and so does this port.
 *
 * 🔴 AND THE ATTENTION'S WEIGHTS ARE SHARED BY EVERY SEQUENCE, so the arms need
 * MORE THAN ONE SEQUENCE or the sharing is untested - and a RAGGED MSA MASK, or
 * the key mask's depth-maximum is untested too. The existing WGSL gate makes
 * exactly those two points; this one plants a token that only the second
 * sequence covers, so a port using each sequence's own mask row lands somewhere
 * else.
 *
 * 🔴 AND boltz2's OUTER PRODUCT MEAN DIVIDES BEFORE IT ADDS THE BIAS, which is
 * "worth exactly `(1 - 1/n) * b` - a per-channel CONSTANT on every pair - so it
 * is invisible at MSA depth 1". The `--opm-bias-after-norm` arm runs it, at a
 * depth where it is worth something.
 */
import { msaBlock as cpuMsaBlock } from "../../cpu/af3/trunk/msa.js";
import { Af3MsaStackGpu } from "../../webgpu/af3/trunk/msa-stack.js";
import { setDeviceTuning, setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  msaBlock, releaseMsaBlockWeights, uploadMsaBlockWeights,
} from "../../webgl/af3/trunk/msa-block.js";
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
  const bar = Number(option(args, "bar", "2e-5"));
  const tokens = Number(option(args, "tokens", "6"));
  const sequences = Number(option(args, "sequences", "4"));
  const count = Number(option(args, "blocks", "2"));
  const pairChannels = Number(option(args, "pair-channels", "32"));
  const msaChannels = Number(option(args, "msa-channels", "32"));
  const outerChannels = 8;
  const msaHeads = 4;
  const msaDimension = 8;
  const pairHeads = 4;
  const pairDimension = 8;
  const opmBiasAfterNorm = flag(args, "opm-bias-after-norm");
  const pairs = tokens * tokens;
  const rows = sequences * tokens;

  const random = generator(141421);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  const triangleGroup = () => ({
    leftNormInputScale: noise(pairChannels, 0.2, 1),
    leftNormInputOffset: noise(pairChannels, 0.1),
    projection: noise(pairChannels * pairChannels * 2, 0.4 / Math.sqrt(pairChannels)),
    gate: noise(pairChannels * pairChannels * 2, 0.4 / Math.sqrt(pairChannels)),
    centerNormScale: noise(pairChannels, 0.2, 1),
    centerNormOffset: noise(pairChannels, 0.1),
    outputProjection: noise(pairChannels * pairChannels, 0.4 / Math.sqrt(pairChannels)),
    gatingLinear: noise(pairChannels * pairChannels, 0.4 / Math.sqrt(pairChannels)),
  });
  const gridGroup = () => {
    const width = pairHeads * pairDimension;
    return {
      heads: pairHeads, dimension: pairDimension,
      actNormScale: noise(pairChannels, 0.2, 1),
      actNormOffset: noise(pairChannels, 0.1),
      pairBiasProjection: noise(pairChannels * pairHeads, 1 / Math.sqrt(pairChannels)),
      qProjection: noise(width * pairChannels, 1 / Math.sqrt(pairChannels)),
      kProjection: noise(width * pairChannels, 1 / Math.sqrt(pairChannels)),
      vProjection: noise(pairChannels * width, 1 / Math.sqrt(pairChannels)),
      gatingQuery: noise(width * pairChannels, 1 / Math.sqrt(pairChannels)),
      outputProjection: noise(width * pairChannels, 0.4 / Math.sqrt(width)),
    };
  };
  const transitionGroup = (channels, factor) => ({
    inputLayerNormScale: noise(channels, 0.2, 1),
    inputLayerNormOffset: noise(channels, 0.1),
    transition1: noise(channels * channels * factor * 2, 1 / Math.sqrt(channels)),
    transition2: noise(channels * factor * channels, 0.4 / Math.sqrt(channels * factor)),
  });
  const blockWeights = () => {
    const width = msaHeads * msaDimension;
    return {
      pairChannels, msaChannels,
      outerProductMean: {
        outerChannels,
        layerNormInputScale: noise(msaChannels, 0.2, 1),
        layerNormInputOffset: noise(msaChannels, 0.1),
        leftProjection: noise(msaChannels * outerChannels, 1 / Math.sqrt(msaChannels)),
        rightProjection: noise(msaChannels * outerChannels, 1 / Math.sqrt(msaChannels)),
        outputW: noise(outerChannels * outerChannels * pairChannels, 1 / outerChannels),
        outputB: noise(pairChannels, 0.1),
      },
      msaAttention1: {
        heads: msaHeads, dimension: msaDimension,
        actNormScale: noise(msaChannels, 0.2, 1),
        actNormOffset: noise(msaChannels, 0.1),
        pairNormScale: noise(pairChannels, 0.2, 1),
        pairNormOffset: noise(pairChannels, 0.1),
        pairLogits: noise(pairChannels * msaHeads, 1 / Math.sqrt(pairChannels)),
        vProjection: noise(msaChannels * width, 1 / Math.sqrt(msaChannels)),
        gatingQuery: noise(msaChannels * width, 1 / Math.sqrt(msaChannels)),
        outputProjection: noise(width * msaChannels, 0.4 / Math.sqrt(width)),
      },
      msaTransition: transitionGroup(msaChannels, 4),
      triangleMultiplicationOutgoing: triangleGroup(),
      triangleMultiplicationIncoming: triangleGroup(),
      pairAttention1: gridGroup(),
      pairAttention2: gridGroup(),
      pairTransition: transitionGroup(pairChannels, 4),
    };
  };
  const blocks = Array.from({ length: count }, blockWeights);

  const sequence = new Float32Array(tokens);
  for (let i = 0; i < tokens; i += 1) sequence[i] = i < Math.ceil(tokens * 0.8) ? 1 : 0;
  const pairMask = new Float32Array(pairs);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = sequence[i] * sequence[j];
  }
  // 🔴 A RAGGED MSA MASK WITH A TOKEN ONLY THE SECOND SEQUENCE COVERS. The key
  // mask is the MAXIMUM over the depth, so that token stays attendable by every
  // sequence - and a port using each sequence's own row would mask it for the
  // others.
  const msaMask = new Float32Array(rows);
  for (let s = 0; s < sequences; s += 1) {
    for (let t = 0; t < tokens; t += 1) {
      msaMask[s * tokens + t] = sequence[t] > 0 && (s + t) % 3 !== 0 ? 1 : 0;
    }
  }
  const onlySecond = 1 % tokens;
  for (let s = 0; s < sequences; s += 1) msaMask[s * tokens + onlySecond] = s === 1 ? 1 : 0;

  const msa = noise(rows * msaChannels, 1);
  const pair = noise(pairs * pairChannels, 1);

  setDeviceTuning(device, { allowHostWeightPacking: true });
  setHalfPrecision(device, false);
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  // The control skews half of each MSA row's channels: the MSA meets a LayerNorm
  // in every sub-layer that reads it, and a whole-row affine change is
  // invisible to one.
  const msaTensor = execution.upload(falsify
    ? Float32Array.from(msa, (v, i) => (i % msaChannels < msaChannels / 2 ? v * 1.01 + 0.01 : v))
    : msa, "af3.msa-block.msa");
  const pairTensor = execution.upload(pair, "af3.msa-block.pair");
  const msaMaskTensor = execution.upload(msaMask, "af3.msa-block.msa-mask");
  const pairMaskTensor = execution.upload(pairMask, "af3.msa-block.pair-mask");

  const arms = [];
  const failures = [];
  for (const msaUpdateBeforeOuterProduct of [false, true]) {
    const order = msaUpdateBeforeOuterProduct
      ? "msa first (OpenDDE, Boltz-2)" : "outer product first (AF3)";
    const uploaded = blocks.map((block) => uploadMsaBlockWeights(execution, block));
    let state = { msa: msaTensor, pair: pairTensor };
    let own = false;
    for (const weights of uploaded) {
      const next = msaBlock(execution, {
        msa: state.msa, pair: state.pair,
        msaMask: msaMaskTensor, pairMask: pairMaskTensor,
        sequences, tokens, weights, msaUpdateBeforeOuterProduct, opmBiasAfterNorm,
      });
      if (own) {
        execution.release(state.msa);
        execution.release(state.pair);
      }
      state = next;
      own = true;
    }
    const ours = {
      msa: Float32Array.from(execution.read(state.msa)),
      pair: Float32Array.from(execution.read(state.pair)),
    };
    execution.release(state.msa);
    execution.release(state.pair);
    for (const weights of uploaded) releaseMsaBlockWeights(execution, weights);

    const dialect = {
      swapTransposedBias: false,
      msaUpdateBeforeOuterProduct,
      ...(opmBiasAfterNorm ? { opmBiasAfterNorm: true } : {}),
    };
    let expected = { msa, pair };
    for (const weights of blocks) {
      expected = cpuMsaBlock(
        { ...expected, msaMask, pairMask, sequences, tokens }, weights, dialect);
    }
    const theirs = await new Af3MsaStackGpu(device, {
      stagedPrecision: "f32", accumulatePrecision: "f32", weightPrecision: "f32",
    }).run({ msa, pair, msaMask, pairMask, sequences, tokens }, blocks, dialect);

    for (const track of ["msa", "pair"]) {
      for (const [label, want] of [["f64 reference", expected], ["WGSL", theirs]]) {
        const name = `${order}, ${track} vs the ${label}`;
        const relative = relativeRms(ours[track], want[track], name);
        const ok = relative <= bar;
        arms.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
        if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
      }
      const between = relativeRms(expected[track], theirs[track], `between.${order}.${track}`);
      arms.push({
        arm: `${order}, ${track}: the two references agree`,
        relativeRms: Number(between.toPrecision(4)), bar, ok: between <= bar,
        falsifiable: false,
        reason: "both references read the unperturbed MSA",
      });
      if (!(between <= bar)) {
        failures.push(
          `${order}, ${track}: the references disagree by ${between.toExponential(3)}`);
      }
    }
  }

  // 🔴 AND THE ORDER MUST MOVE THE ANSWER, or both arms above are one arm twice.
  const orderMoves = {};
  for (const track of ["msa", "pair"]) {
    const first = arms.find((a) => a.arm.startsWith("outer product first")
      && a.arm.includes(`${track} vs the f64`));
    const second = arms.find((a) => a.arm.startsWith("msa first")
      && a.arm.includes(`${track} vs the f64`));
    orderMoves[track] = { first: first?.relativeRms, second: second?.relativeRms };
  }

  for (const tensor of [msaTensor, pairTensor, msaMaskTensor, pairMaskTensor]) {
    execution.release(tensor);
  }
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const falsifiable = arms.filter((row) => row.falsifiable !== false);
  const red = falsifiable.filter((row) => !row.ok).length;
  return {
    shape: {
      tokens, sequences, blocks: count, pairChannels, msaChannels, outerChannels,
      msaHeads, msaDimension, pairHeads, pairDimension,
    },
    opmBiasAfterNorm,
    tokenCoveredOnlyBySequence1: onlySecond,
    maskedMsaRows: rows - msaMask.reduce((a, b) => a + b, 0),
    bar,
    falsify,
    arms,
    orderMoves,
    memory: execution.stats(),
    verdict: falsify
      ? (red === falsifiable.length
        ? `ok: all ${falsifiable.length} falsifiable arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${falsifiable.length} falsifiable arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms, both orders, both tracks, both references agreeing`
        : `FAILED: ${failures.join("; ")}`),
  };
}
