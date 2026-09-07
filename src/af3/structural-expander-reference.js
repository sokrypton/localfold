/**
 * OpenDDE's StructuralTokenExpander: residue representations onto structural
 * tokens.
 *
 * Everything the diffusion runs on comes from here. The trunk produced a single
 * and a pair over RESIDUE tokens; this gathers them onto the structural tokens
 * (see structural-tokens.js), adds learned role embeddings, projects the pair
 * through a matrix chosen per ROLE PAIR, adds five learned pair biases from the
 * structural relationships, and returns a scalar attention bias the refiner and
 * the diffusion both read.
 *
 * 🔴 THE PAIR PROJECTION IS FORTY-NINE MATRICES, SELECTED PER PAIR.
 * `pair_block_proj` is [7 * 7, c_z, c_z] and the pair (i, j) is projected
 * through the one its two ROLES name - a backbone-to-sidechain pair does not
 * see the matrix a sidechain-to-backbone pair does, which is why there are 49
 * and not 28. At 384 channels that tensor alone is 28.9 MiB.
 *
 * 🔴 AND THE FIVE PAIR BIASES ARE EMBEDDINGS OF BOOLEANS, so each is a
 * [2, c_z] table indexed by 0 or 1 - which means the FALSE row is added too and
 * is not zero. Treating a false feature as "no contribution" drops a trained
 * vector from every pair that does not have the relationship, which is most of
 * them.
 */
import { layerNorm, linear } from "./pairformer-reference.js";
import { NO_TWIN, ROLES } from "./structural-tokens.js";

const BACKBONE_ROLES = new Set([1, 3, 5]);
const SIDECHAIN_ROLE = 2;
const BASE_ROLES = new Set([4, 6]);
/** role_pair_type: bb-bb 0, bb-sc 1, sc-bb 2, sc-sc 3, bb-base 4, base-bb 5, base-base 6, else 7. */
export const ROLE_PAIR_TYPES = 8;

/**
 * The five boolean pair features and the role-pair type, per structural pair.
 *
 * Returned flat, one entry per (i, j), because every consumer indexes them that
 * way and building six n^2 arrays once beats recomputing the predicates in the
 * pair loop, the bias loop and the attention-bias loop.
 */
export function structuralPairFeatures(layout, asymIdOfResidue) {
  const n = layout.tokens;
  const { parent, role, prevParent, nextParent } = layout;
  const sameParent = new Uint8Array(n * n);
  const twin = new Uint8Array(n * n);
  const prevBackbone = new Uint8Array(n * n);
  const nextBackbone = new Uint8Array(n * n);
  const rolePairType = new Uint8Array(n * n);

  const isBackbone = Array.from({ length: n }, (_, i) => BACKBONE_ROLES.has(role[i]));
  const isSidechain = Array.from({ length: n }, (_, i) => role[i] === SIDECHAIN_ROLE);
  const isBase = Array.from({ length: n }, (_, i) => BASE_ROLES.has(role[i]));

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const at = i * n + j;
      const same = parent[i] === parent[j];
      sameParent[at] = same ? 1 : 0;
      const sameChain = asymIdOfResidue[parent[i]] === asymIdOfResidue[parent[j]];
      twin[at] = same && ((isBackbone[i] && (isSidechain[j] || isBase[j]))
        || (isBackbone[j] && (isSidechain[i] || isBase[i]))) ? 1 : 0;
      prevBackbone[at] = isBackbone[i] && isBackbone[j] && sameChain
        && prevParent[i] === parent[j] ? 1 : 0;
      nextBackbone[at] = isBackbone[i] && isBackbone[j] && sameChain
        && nextParent[i] === parent[j] ? 1 : 0;
      // 🔴 THE ORDER OF THESE TESTS IS THE DEFINITION. Upstream writes them as
      // successive `where`s, so a later one overrides an earlier - and the pairs
      // they can both match (a backbone with a backbone, say) are disjoint here
      // anyway. What is NOT disjoint is the default: anything involving a
      // role-0 atom token falls through to 7.
      let type = 7;
      if (isBackbone[i] && isBackbone[j]) type = 0;
      else if (isBackbone[i] && isSidechain[j]) type = 1;
      else if (isSidechain[i] && isBackbone[j]) type = 2;
      else if (isSidechain[i] && isSidechain[j]) type = 3;
      else if (isBackbone[i] && isBase[j]) type = 4;
      else if (isBase[i] && isBackbone[j]) type = 5;
      else if (isBase[i] && isBase[j]) type = 6;
      rolePairType[at] = type;
    }
  }
  return { sameParent, twin, prevBackbone, nextBackbone, rolePairType };
}

/** The single track: the parent's representation, a split MLP, and a role. */
export function expandSingle(layout, singleInputs, single, weights) {
  const n = layout.tokens;
  const cSingleInputs = weights.singleInputRoleEmbedding.length / ROLES;
  const cSingle = weights.singleRoleEmbedding.length / ROLES;

  const inputsStruct = new Float32Array(n * cSingleInputs);
  for (let i = 0; i < n; i += 1) {
    const from = layout.parent[i] * cSingleInputs;
    const roleBase = layout.role[i] * cSingleInputs;
    for (let c = 0; c < cSingleInputs; c += 1) {
      inputsStruct[i * cSingleInputs + c] =
        singleInputs[from + c] + weights.singleInputRoleEmbedding[roleBase + c];
    }
  }

  // The parent's single, gathered, then `x + mlp(x) + role`.
  const parentSingle = new Float32Array(n * cSingle);
  for (let i = 0; i < n; i += 1) {
    const from = layout.parent[i] * cSingle;
    for (let c = 0; c < cSingle; c += 1) parentSingle[i * cSingle + c] = single[from + c];
  }
  const normalised = layerNorm(parentSingle, n, cSingle,
                              weights.singleSplitNormScale, weights.singleSplitNormOffset);
  const hidden = linear(normalised, n, cSingle, 2 * cSingle, weights.singleSplit1);
  for (let index = 0; index < hidden.length; index += 1) {
    const value = hidden[index];
    hidden[index] = value / (1 + Math.exp(-value));   // silu
  }
  const projected = linear(hidden, n, 2 * cSingle, cSingle, weights.singleSplit2);

  const singleStruct = new Float32Array(n * cSingle);
  for (let i = 0; i < n; i += 1) {
    const roleBase = layout.role[i] * cSingle;
    for (let c = 0; c < cSingle; c += 1) {
      singleStruct[i * cSingle + c] = parentSingle[i * cSingle + c]
        + projected[i * cSingle + c] + weights.singleRoleEmbedding[roleBase + c];
    }
  }
  return { inputsStruct, singleStruct };
}

/** The pair track: gather, the per-role-pair projection, and five biases. */
export function expandPair(layout, pair, residueTokens, features, weights, channels) {
  const n = layout.tokens;
  const out = new Float32Array(n * n * channels);
  const { parent, role } = layout;

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const at = (i * n + j) * channels;
      const from = (parent[i] * residueTokens + parent[j]) * channels;
      const matrix = (role[i] * ROLES + role[j]) * channels * channels;
      const pairType = features.rolePairType[i * n + j];
      const sameParent = features.sameParent[i * n + j];
      const twin = features.twin[i * n + j];
      const prev = features.prevBackbone[i * n + j];
      const next = features.nextBackbone[i * n + j];
      for (let d = 0; d < channels; d += 1) {
        let total = pair[from + d];
        for (let c = 0; c < channels; c += 1) {
          total += pair[from + c] * weights.pairBlockProj[matrix + c * channels + d];
        }
        // Each of these is an embedding of a BOOLEAN, so the false row counts.
        total += weights.sameParentEmbedding[sameParent * channels + d]
          + weights.sameResidueTwinEmbedding[twin * channels + d]
          + weights.prevBbChainEmbedding[prev * channels + d]
          + weights.nextBbChainEmbedding[next * channels + d]
          + weights.rolePairTypeEmbedding[pairType * channels + d];
        out[at + d] = total;
      }
    }
  }
  return out;
}

/** The scalar attention bias the refiner and the diffusion both read. */
export function structuralAttentionBias(layout, features, weights) {
  const n = layout.tokens;
  const bias = new Float32Array(n * n);
  for (let at = 0; at < n * n; at += 1) {
    bias[at] = weights.attnBiasSameParent[0] * features.sameParent[at]
      + weights.attnBiasSameResidueTwin[0] * features.twin[at]
      + weights.attnBiasPrevBbChain[0] * features.prevBackbone[at]
      + weights.attnBiasNextBbChain[0] * features.nextBackbone[at]
      + weights.attnBiasRolePairType[features.rolePairType[at]];
  }
  return bias;
}

/** The whole expander: everything the diffusion and the refiner run on. */
export function expandStructural(layout, embeddings, weights, shape) {
  const features = structuralPairFeatures(layout, embeddings.asymId);
  const { inputsStruct, singleStruct } = expandSingle(
    layout, embeddings.targetFeat, embeddings.single, weights);
  const pairStruct = expandPair(
    layout, embeddings.pair, shape.residueTokens, features, weights, shape.pairChannels);
  return {
    targetFeat: inputsStruct, single: singleStruct, pair: pairStruct,
    attentionBias: structuralAttentionBias(layout, features, weights),
    features,
  };
}
export { NO_TWIN };
