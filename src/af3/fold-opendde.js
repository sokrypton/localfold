/**
 * OpenDDE's fold: the trunk on residues, everything after it on structural
 * tokens.
 *
 *     residues -> trunk -> expander -> refiner -> diffusion -> residues
 *
 * 🔴 THE TWO TOKEN SPACES ARE THE WHOLE OF WHAT MAKES THIS A SEPARATE DRIVER.
 * `foldBatch` runs one batch end to end because every other model here does;
 * OpenDDE re-tokenises between the trunk and the diffusion, so the batch the
 * sampler sees is not the batch the trunk saw and the coordinates have to come
 * back. Everything else - the trunk, the atom encoder, the denoiser, the
 * samplers, the PDB writer - is the AlphaFold 3 code unchanged, which is what
 * the widths being derived from the weights bought.
 */
import { Af3TrunkGpu } from "./trunk-webgpu.js";
import { Af3PairformerStackGpu } from "./pairformer-block-webgpu.js";
import { Af3StructuralExpanderGpu } from "./structural-expander-webgpu.js";
import { structuralPairFeatures } from "./structural-expander-reference.js";
import { structuralBatch, structuralLayout } from "./structural-tokens.js";
import { buildTargetFeat, normalFrom } from "./fold.js";
import { perAtomConditioning } from "./atom-conditioning-reference.js";
import { diffusionConditioning } from "./diffusion-reference.js";
import { sampleOnGpu, flowOnGpu } from "./diffusion-sampler-webgpu.js";
import { Af3DiffusionHeadGpu } from "./diffusion-head-webgpu.js";
import { af3ContactClasses } from "./contact-classes.js";

/**
 * The diffusion's coordinates, scattered back onto the residue layout.
 *
 * 🔴 THE STRUCTURE EVERY CONSUMER READS IS THE RESIDUE ONE. The PDB writer, the
 * geometry checks and anything comparing against a deposition all index
 * (residue token, atom slot); handing them 91 structural tokens where they
 * expect 47 residues is where the writer stops. `residueAtomGather` is the
 * inverse of the regrouping and was built with it.
 */
export function structuralToResidue(positions, layout, residueTokens, dense) {
  const out = new Float32Array(residueTokens * dense * 3);
  for (let index = 0; index < residueTokens * dense; index += 1) {
    const from = layout.residueAtomGather[index];
    if (from < 0) continue;
    out[index * 3] = positions[from * 3];
    out[index * 3 + 1] = positions[from * 3 + 1];
    out[index * 3 + 2] = positions[from * 3 + 2];
  }
  return out;
}

/**
 * @param {GPUDevice} device
 * @param {object} batch the RESIDUE batch, from featuriseProtein
 * @param {object} weights trunk, targetFeat, expander, refiner, diffusion
 */
export async function foldOpendde(device, batch, weights, options = {}) {
  const dialect = weights.trunk.dialect;
  const tokens = batch.tokens;
  const dense = batch.dense;
  const recycles = options.recycles ?? 0;
  const steps = options.steps ?? 200;
  const stage = async (name, work) => {
    options.onStageStart?.(name);
    const started = performance.now();
    const value = await work();
    options.onStage?.(name, performance.now() - started);
    return value;
  };

  // --- the trunk, on residues, exactly as AlphaFold 3 runs it ---
  const targetFeat = await buildTargetFeat(batch, weights.targetFeat, device);
  const seqMask = batch.seqMask;
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }
  const trunkGpu = new Af3TrunkGpu(device, {});
  const pairChannels = weights.trunk.embedder.pairChannels;
  const singleChannels = weights.trunk.embedder.singleChannels;
  let previousPair = new Float32Array(tokens * tokens * pairChannels);
  let previousSingle = new Float32Array(tokens * singleChannels);
  let trunk;
  for (let pass = 0; pass <= recycles; pass += 1) {
    trunk = await stage(`trunk ${pass + 1}/${recycles + 1}`, () => trunkGpu.run({
      tokens, sequences: batch.sequences ?? 1, templates: 4, targetFeat,
      features: batch.features, msaRows: batch.msa, deletionMatrix: batch.deletionMatrix,
      msaMask: batch.msaMask, bondMatrix: batch.bondMatrix, pairMask, seqMask,
      previousPair, previousSingle,
      contactClasses: af3ContactClasses(batch, tokens),
    }, weights.trunk, dialect, {}));
    previousPair = trunk.pair;
    previousSingle = trunk.single;
  }

  // --- the structural tokens, and everything after them ---
  const layout = structuralLayout(batch);
  const structural = structuralBatch(batch, layout);
  const features = structuralPairFeatures(layout, batch.asymId);
  const expanded = await stage("expand", () =>
    new Af3StructuralExpanderGpu(device).run(
      layout, { single: trunk.single, pair: trunk.pair, targetFeat, asymId: batch.asymId },
      weights.expander, features, tokens));

  // The target_feat of a structural token: its parent's, plus a role embedding.
  const inputWidth = weights.expander.singleInputChannels;
  const structuralTargetFeat = new Float32Array(layout.tokens * inputWidth);
  for (let i = 0; i < layout.tokens; i += 1) {
    const from = layout.parent[i] * inputWidth;
    const role = layout.role[i] * inputWidth;
    for (let c = 0; c < inputWidth; c += 1) {
      structuralTargetFeat[i * inputWidth + c] =
        targetFeat[from + c] + weights.expander.singleInputRoleEmbedding[role + c];
    }
  }

  // 🔴 THE REFINER IS THE PAIRFORMER, AND IT TAKES THE EXPANDER'S BIAS. Four
  // blocks at a third shape - 8 single heads of 48, pair transition factor 2 -
  // every one of which is derived from its own weights.
  const structTokens = layout.tokens;
  const structSeqMask = structural.seqMask;
  const structPairMask = new Float32Array(structTokens * structTokens);
  for (let i = 0; i < structTokens; i += 1) {
    for (let j = 0; j < structTokens; j += 1) {
      structPairMask[i * structTokens + j] = structSeqMask[i] * structSeqMask[j];
    }
  }
  const attentionBias = new Float32Array(structTokens * structTokens);
  for (let at = 0; at < attentionBias.length; at += 1) {
    attentionBias[at] = weights.expander.attnBiasSameParent[0] * features.sameParent[at]
      + weights.expander.attnBiasSameResidueTwin[0] * features.twin[at]
      + weights.expander.attnBiasPrevBbChain[0] * features.prevBackbone[at]
      + weights.expander.attnBiasNextBbChain[0] * features.nextBackbone[at]
      + weights.expander.attnBiasRolePairType[features.rolePairType[at]];
  }
  const refined = await stage("refine", () =>
    new Af3PairformerStackGpu(device, {}).run(
      { tokens: structTokens, pair: expanded.pair, single: expanded.single,
        pairMask: structPairMask, seqMask: structSeqMask },
      weights.refiner, dialect, { extraPairBias: attentionBias }));

  // --- the diffusion, on the structural batch ---
  const conditioning = perAtomConditioning({
    positions: structural.refPos, mask: structural.refMask,
    element: structural.refElement, charge: structural.refCharge,
    atomNameChars: structural.refAtomNameChars,
  }, structTokens, dense, weights.atomReference);
  // 🔴 THE PAIR CONDITIONING IS COMPUTED HERE, ONCE, BECAUSE ITS ARITHMETIC IS
  // OpenDDE's. It compresses the trunk pair and the relative encoding
  // SEPARATELY to the pair width and concatenates those, where AlphaFold 3
  // concatenates the trunk pair with the RAW relative features - and the trunk
  // pair reaching it is 384 wide rather than 128. It does not depend on the
  // noise level (which is why the head caches it across the sampler's steps
  // anyway), so the reference computes it for the fold and the head is handed
  // the result.
  const pairConditioning = await stage("pair-conditioning", async () =>
    diffusionConditioning({
      tokens: structTokens, trunkSingle: refined.single, trunkPair: refined.pair,
      targetFeat: structuralTargetFeat, noiseLevel: 1, features: structural.features,
      dialect: weights.diffusion.dialect,
    }, weights.diffusion.conditioning).pair);

  const headInput = {
    shape: structural.shape, dialect: weights.diffusion.dialect,
    conditioning, atomMask: structural.predDenseAtomMask, seqMask: structSeqMask,
    features: structural.features, targetFeat: structuralTargetFeat,
    refPos: structural.refPos, refSpaceUid: structural.refSpaceUid,
    tokenAtomsToQueries: structural.tokenAtomsToQueries,
    queriesToKeys: structural.queriesToKeys,
    queriesToTokenAtoms: structural.queriesToTokenAtoms,
    tokensToQueries: structural.tokensToQueries,
    tokensToKeys: structural.tokensToKeys,
    trunkSingle: refined.single, trunkPair: refined.pair,
    pairConditioning,
  };
  const head = steps > 0 ? new Af3DiffusionHeadGpu(device, {}) : undefined;
  const sampled = await stage("sample", () => (options.mode === "flow"
    ? flowOnGpu(device, headInput, weights.diffusion, {
        cycles: steps, head, normal: normalFrom(options.seed ?? 20260831),
        onStep: options.onStep, ...(options.schedule ?? {}),
      })
    : sampleOnGpu(device, headInput, weights.diffusion, {
        steps, head, normal: normalFrom(options.seed ?? 20260831),
        onStep: options.onStep, ...(options.schedule ?? {}),
      })));
  head?.dispose();

  // 🔴 EVERY STAGE IS CHECKED FINITE, because a NaN anywhere in this chain
  // reaches the coordinates and the geometry then reports null for everything -
  // which says a fold failed but not where.
  const finiteOf = (a) => a.every(Number.isFinite);
  const stages = {
    trunkSingle: finiteOf(trunk.single), trunkPair: finiteOf(trunk.pair),
    expandedSingle: finiteOf(expanded.single), expandedPair: finiteOf(expanded.pair),
    refinedSingle: finiteOf(refined.single), refinedPair: finiteOf(refined.pair),
    sampled: finiteOf(sampled),
  };

  return {
    stages,
    positions: structuralToResidue(sampled, layout, tokens, dense),
    structuralPositions: sampled,
    structuralTokens: structTokens,
    layout, structural, trunk,
    contactProbs: trunk.contactProbs, binEdges: trunk.binEdges,
  };
}
