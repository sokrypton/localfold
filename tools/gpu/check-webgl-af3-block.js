/**
 * One AF3 pairformer block in WebGL2, against TWO references.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-block.js [--falsify]
 *
 * `cpu/af3/trunk/pairformer.js`'s `pairformerBlock` is the f64 reference and
 * `Af3PairformerStackGpu` the WGSL one, both in this page, with the two also
 * held to each other.
 *
 * 🔴 THIS IS THE COMPOSITION GATE, AND THE PER-OPERATION ONES CANNOT STAND IN
 * FOR IT. The existing WGSL gate says what it pins: "the ORDER, the residual
 * chain, and the fact that the single track reads the pair after all five pair
 * updates rather than before. Every one of those is a reordering that still
 * runs and still returns a pair representation." Two more, specific to a port:
 *
 *   - each update reads the RUNNING activation, not the block input. "AF3
 *     threads them sequentially where chai-1 sums parallel deltas" - so a block
 *     that summed seven deltas from one input would be a different model with
 *     the same weights.
 *   - the pair logits are broadcast over heads from a one-per-head projection,
 *     which needs a relayout - and a texel of that relayout straddles two head
 *     planes whenever the pair count is not a multiple of four. That is the
 *     shape of bug the triangle's operands kernel already had.
 *
 * Both tracks are compared, because an error confined to the single track would
 * hide behind a pair that agrees: the single track reads the pair, so a wrong
 * pair reaches it, but a wrong single never reaches the pair.
 *
 * 🔴 TWO BLOCKS, NOT ONE, for the reason the AF2 stack gate gives: with one
 * block the loop is untested and a driver that re-ran the first block's input
 * would agree with a one-block reference.
 */
import { pairformerBlock as cpuBlock } from "../../cpu/af3/trunk/pairformer.js";
import { Af3PairformerStackGpu } from "../../webgpu/af3/trunk/pairformer-block.js";
import { setDeviceTuning, setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  pairformerBlock, releasePairformerBlockWeights, uploadPairformerBlockWeights,
} from "../../webgl/af3/trunk/pairformer-block.js";
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
  const count = Number(option(args, "blocks", "2"));
  const pairChannels = Number(option(args, "pair-channels", "32"));
  // 🔴 THE WIDTHS ARE NOT FREE: THE WGSL REFERENCE REFUSES SOME OF THEM. Its
  // transition chunks the intermediate axis against a 128-wide workgroup and
  // throws "chunk 192 is not a multiple of the workgroup 128" - so a single
  // track of 48 channels widened by 4 is not a shape it will run, however
  // legal it is arithmetically. 32 x 4 = 128 is, and so is the real AF3's
  // 384 x 4. A synthetic shape has to be one BOTH sides accept or the gate
  // measures nothing.
  // ...and the single attention's width must be a multiple of 64 ("width 32 is
  // not a multiple of 64 * 1 splits"), so 4 heads of 16 over a 64-channel single
  // track is the smallest shape that clears both.
  const singleChannels = Number(option(args, "single-channels", "64"));
  const pairHeads = 4;
  const pairDimension = 8;
  const singleHeads = 4;
  const singleDimension = 16;
  const swapTransposedBias = flag(args, "swap-bias");
  const pairs = tokens * tokens;

  const random = generator(577215);
  const noise = (count_, spread, centre = 0) => {
    const out = new Float32Array(count_);
    for (let i = 0; i < count_; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
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
    const width = singleHeads * singleDimension;
    return {
      pairChannels, singleChannels,
      triangleMultiplicationOutgoing: triangleGroup(),
      triangleMultiplicationIncoming: triangleGroup(),
      pairAttention1: gridGroup(),
      pairAttention2: gridGroup(),
      pairTransition: transitionGroup(pairChannels, 4),
      singlePairLogitsNormScale: noise(pairChannels, 0.2, 1),
      singlePairLogitsNormOffset: noise(pairChannels, 0.1),
      singlePairLogitsProjection: noise(
        pairChannels * singleHeads, 1 / Math.sqrt(pairChannels)),
      singleAttention: {
        heads: singleHeads, dimension: singleDimension,
        layerNormScale: noise(singleChannels, 0.2, 1),
        layerNormOffset: noise(singleChannels, 0.1),
        qProjection: noise(singleChannels * width, 1 / Math.sqrt(singleChannels)),
        // The one genuine bias in the whole block.
        qBias: noise(width, 0.1),
        kProjection: noise(singleChannels * width, 1 / Math.sqrt(singleChannels)),
        vProjection: noise(singleChannels * width, 1 / Math.sqrt(singleChannels)),
        gatingQuery: noise(singleChannels * width, 1 / Math.sqrt(singleChannels)),
        outputProjection: noise(width * singleChannels, 0.4 / Math.sqrt(width)),
      },
      singleTransition: transitionGroup(singleChannels, 4),
    };
  };
  const blocks = Array.from({ length: count }, blockWeights);

  // A masked tail, as the WGSL gate uses: the last fifth of the tokens are
  // padding, so the pair mask is zero on two of its four quadrants.
  const sequence = new Float32Array(tokens);
  for (let i = 0; i < tokens; i += 1) sequence[i] = i < Math.ceil(tokens * 0.8) ? 1 : 0;
  const pairMask = new Float32Array(pairs);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = sequence[i] * sequence[j];
  }
  const pair = noise(pairs * pairChannels, 1);
  const single = noise(tokens * singleChannels, 1);

  setDeviceTuning(device, { allowHostWeightPacking: true });
  setHalfPrecision(device, false);
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  // The control skews half of each pair row's channels: the pair's first stop in
  // every sub-layer is a LayerNorm, and a whole-row affine change is invisible
  // to one.
  const pairTensor = execution.upload(falsify
    ? Float32Array.from(pair,
      (v, i) => (i % pairChannels < pairChannels / 2 ? v * 1.01 + 0.01 : v))
    : pair, "af3.block.pair");
  const singleTensor = execution.upload(single, "af3.block.single");
  const pairMaskTensor = execution.upload(pairMask, "af3.block.pair-mask");
  const seqMaskTensor = execution.upload(sequence, "af3.block.seq-mask");

  let state = { pair: pairTensor, single: singleTensor };
  let own = false;
  const uploaded = blocks.map((block) => uploadPairformerBlockWeights(execution, block));
  for (const weights of uploaded) {
    const next = pairformerBlock(execution, {
      pair: state.pair, single: state.single,
      pairMask: pairMaskTensor, seqMask: seqMaskTensor,
      tokens, weights, swapTransposedBias,
    });
    if (own) {
      execution.release(state.pair);
      execution.release(state.single);
    }
    state = next;
    own = true;
  }
  const ours = {
    pair: Float32Array.from(execution.read(state.pair)),
    single: Float32Array.from(execution.read(state.single)),
  };
  execution.release(state.pair);
  execution.release(state.single);
  for (const weights of uploaded) releasePairformerBlockWeights(execution, weights);
  for (const tensor of [pairTensor, singleTensor, pairMaskTensor, seqMaskTensor]) {
    execution.release(tensor);
  }

  const dialect = { swapTransposedBias };
  let expected = { pair, single };
  for (const weights of blocks) {
    expected = cpuBlock(
      { ...expected, pairMask, seqMask: sequence, tokens }, weights, dialect);
  }
  const theirs = await new Af3PairformerStackGpu(device, {
    stagedPrecision: "f32", accumulatePrecision: "f32", weightPrecision: "f32",
  }).run({ pair, single, pairMask, seqMask: sequence, tokens }, blocks, dialect);

  const arms = [];
  const failures = [];
  for (const track of ["pair", "single"]) {
    for (const [label, want] of [["f64 reference", expected], ["WGSL", theirs]]) {
      const name = `${track} vs the ${label}`;
      const relative = relativeRms(ours[track], want[track], name);
      const ok = relative <= bar;
      arms.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
      if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
    }
    const between = relativeRms(expected[track], theirs[track], `between.${track}`);
    arms.push({
      arm: `${track}: the two references agree`,
      relativeRms: Number(between.toPrecision(4)), bar, ok: between <= bar,
      falsifiable: false,
      reason: "both references read the unperturbed pair",
    });
    if (!(between <= bar)) {
      failures.push(`${track}: the references disagree by ${between.toExponential(3)}`);
    }
  }

  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const falsifiable = arms.filter((row) => row.falsifiable !== false);
  const red = falsifiable.filter((row) => !row.ok).length;
  return {
    shape: {
      tokens, blocks: count, pairChannels, singleChannels,
      pairHeads, pairDimension, singleHeads, singleDimension,
    },
    swapTransposedBias,
    maskedTokens: tokens - sequence.reduce((a, b) => a + b, 0),
    bar,
    falsify,
    arms,
    memory: execution.stats(),
    verdict: falsify
      ? (red === falsifiable.length
        ? `ok: all ${falsifiable.length} falsifiable arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${falsifiable.length} falsifiable arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms over ${count} blocks, both tracks, both references agreeing`
        : `FAILED: ${failures.join("; ")}`),
  };
}
