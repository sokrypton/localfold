/**
 * Where a real-shape AF3 denoiser call spends its time in WebGL2.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/profile-webgl-af3-head.js
 *
 * 🔴 A STAGE CLOCK ON A SYNCHRONOUS API IS A REAL CLOCK. The WebGL2 execution
 * reads back with `readPixels`, so a `performance.now()` around a stage measures
 * that stage - unlike `webgpu/`'s head, whose "stage() timers measure when the
 * HOST returns, so the stage that awaits a readback absorbs everything queued
 * before it". There is no queue here to absorb.
 *
 * It exists because a two-step fold of a 59-residue chain did not finish in ten
 * minutes and the stage list could not say which draw was eating it.
 */
import { af3BatchFromA3m } from "../../shared/af3/featurise/batch.js";
import { openAf3Store } from "../../shared/af3/weights/weights.js";
import { foldWeights } from "../../shared/af3/weights/diffusion-weights.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { diffusionHead } from "../../webgl/af3/diffusion/head.js";
import { perAtomConditioning } from "../../webgl/af3/diffusion/atom-conditioning.js";
import { RELATIVE_WIDTH, tokenIdentity } from "../../webgl/af3/trunk/embedder.js";
import { dialectFlags, uploadAf3Weights } from "../../webgl/af3/model/fold.js";

const DEFAULT_SEQUENCE = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const length = Number(option(args, "length", "0"));
  const sequence = length > 0 ? DEFAULT_SEQUENCE.slice(0, length) : DEFAULT_SEQUENCE;
  const store = await openAf3Store(
    `${option(args, "bundle", "/model-af3-int5")}/manifest.json`);
  // One pairformer block: this measures the HEAD, and the trunk's tensors only
  // have to exist and be the right shape.
  const weights = await foldWeights(store, { blocks: 1, msaBlocks: 1 });
  const { batch } = af3BatchFromA3m(sequence, null, {});
  const { tokens, dense } = batch;
  const flags = dialectFlags(weights.trunk.dialect);

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const uploaded = uploadAf3Weights(execution, weights);
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(
      values instanceof Float32Array ? values : Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  const uploadGather = (g, name) => ({
    indices: upload(Float32Array.from(g.indices), `${name}.indices`),
    mask: upload(g.mask, `${name}.mask`),
  });
  const pairChannels = weights.trunk.embedder.pairChannels;
  const singleChannels = weights.trunk.embedder.singleChannels;
  const refMask = upload(batch.refMask, "ref-mask");
  const refPos = upload(batch.refPos, "ref-pos");
  const input = {
    conditioning: perAtomConditioning(execution, {
      positions: refPos, mask: refMask,
      element: upload(batch.refElement, "ref-element"),
      charge: upload(batch.refCharge, "ref-charge"),
      atomNameChars: upload(batch.refAtomNameChars, "ref-names"),
      weights: uploaded.atomReference, tokens, dense, rawRefCharge: flags.rawRefCharge,
    }),
    atomMask: refMask,
    seqMask: upload(batch.seqMask, "seq-mask"),
    refPos,
    refSpaceUid: upload(batch.refSpaceUid, "ref-uid"),
    // The trunk's outputs stand in as noise of the right shape: this measures
    // draws, not a fold.
    trunkSingle: upload(
      Float32Array.from({ length: tokens * singleChannels },
        (_, i) => Math.sin(i * 0.7)), "trunk-single"),
    trunkPair: upload(
      Float32Array.from({ length: tokens * tokens * pairChannels },
        (_, i) => Math.sin(i * 0.3)), "trunk-pair"),
    targetFeat: upload(
      Float32Array.from(
        { length: tokens * weights.diffusion.conditioning.targetFeatWidth },
        (_, i) => Math.cos(i * 0.5)), "target-feat"),
    identity: upload(tokenIdentity(batch.features, tokens), "identity"),
    gathers: {
      tokenAtomsToQueries: uploadGather(batch.tokenAtomsToQueries, "ta2q"),
      queriesToKeys: uploadGather(batch.queriesToKeys, "q2k"),
      queriesToTokenAtoms: uploadGather(batch.queriesToTokenAtoms, "q2ta"),
      tokensToQueries: uploadGather(batch.tokensToQueries, "t2q"),
      tokensToKeys: uploadGather(batch.tokensToKeys, "t2k"),
    },
    shape: batch.shape,
    weights: uploaded.diffusion,
    relativeWidth: RELATIVE_WIDTH,
    fourierWeight: uploaded.hostFourier.weight,
    fourierBias: uploaded.hostFourier.bias,
    padUnknownDna: flags.padUnknownDna,
    maskPaddedKeys: flags.maskPaddedKeys,
    keysUidFromQueriesLayout: flags.keysUidFromQueriesLayout,
    keyMaskedAtomAttention: flags.keyMaskedAtomAttention,
    chainedAtomLayerNorm: flags.chainedAtomLayerNorm,
    diffusionNoResidual: flags.diffusionNoResidual,
    pairNormPerBlock: flags.pairNormPerBlock,
    maskAtomActPerBlock: flags.maskAtomActPerBlock,
    preTrunkQuery: flags.preTrunkQuery,
  };

  const marks = [];
  let last = performance.now();
  const start = last;
  const out = diffusionHead(execution, {
    ...input,
    positionsNoisy: upload(new Float32Array(tokens * dense * 3), "noisy"),
    noiseLevel: 16,
    onStage: (stage, tensor) => {
      // Reading the tensor forces every draw queued for it, so the mark is the
      // stage's own cost and not a queue's.
      execution.read(tensor);
      const now = performance.now();
      marks.push({ stage, milliseconds: Math.round(now - last) });
      last = now;
    },
  });
  execution.read(out);
  const total = Math.round(performance.now() - start);
  execution.release(out);

  return {
    sequence: sequence.length, tokens, dense, atoms: tokens * dense,
    shape: batch.shape,
    totalMilliseconds: total,
    stages: marks.sort((a, b) => b.milliseconds - a.milliseconds),
    memory: execution.stats(),
    verdict: `one denoiser call at ${tokens} tokens: ${total} ms`,
  };
}
