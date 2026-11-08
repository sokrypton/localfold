/**
 * How ESMFold2's pair representation starts, and how it comes back each loop.
 *
 *     z_init = z_init_1(x)[i] + z_init_2(x)[j]
 *            + rel_pos(residue_index, asym_id, sym_id, entity_id, token_index)
 *            + token_bonds(bonds)
 *            + lm_z
 *
 *     z = 0;  repeat n_loops + 1:  z = folding_trunk(z_init + pair_loop_proj(z))
 *
 * 🔴 THE LOOP RUNS `num_loops + 1` TIMES AND `z` STARTS AT ZERO. "The config
 * says 3 and the model runs four; reading that as three is a silent quarter less
 * trunk, which still folds." `pair_loop_proj`'s Linear is zero-initialised
 * upstream, so the first iteration is exactly `folding_trunk(z_init)`.
 *
 * 🔴 THE CHAIN BLOCK'S POLARITY IS THE OPPOSITE OF THE OTHER TWO, IN THE SAME
 * FUNCTION. The residue and token blocks send a DIFFERENT-chain pair to the
 * out-of-range bin; the chain block sends a SAME-chain one there. "Both are 'the
 * pair does not have a meaningful offset on this axis' and they are written the
 * two opposite ways round. Getting it backwards is a monomer that still folds -
 * every pair is same-chain, so the whole block becomes one constant column
 * either way - and a COMPLEX that is quietly wrong."
 *
 * 🔴 AND A ONE-HOT TIMES A MATRIX IS A ROW LOOKUP. At 300 tokens the one-hot
 * alone is 90,000 x 139 floats and only four of the 139 are ever non-zero, three
 * of them exactly 1 - so this adds four weight rows a pair in one draw. The same
 * decision `af3/trunk/embedder.js` records for AF3's 139.
 *
 * 🔴 AND z_init PROJECTS THEN BROADCASTS, NEVER THE OTHER WAY. The two are the
 * same arithmetic and differ by a factor of n: at 300 tokens it is 600
 * projections of 451 channels rather than 90,000 of them.
 */
import { ONE_TARGET, PRELUDE, memoSource } from "../runtime/execution.js";
import { layerNorm } from "../kernels/layer-norm.js";
import { linear } from "../kernels/linear.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

export const RESIDX_BINS = 32;
export const CHAIN_BINS = 2;
/** 66 + 66 + 1 + 6 = 139, in that concatenation order. */
export const RELATIVE_WIDTH = (2 * RESIDX_BINS + 2) * 2 + 1 + (2 * CHAIN_BINS + 2);

/**
 * The five per-token lanes the relative encoding reads, interleaved.
 *
 * One tensor rather than five, because every one of them is read per PAIR and a
 * fragment that fetches five textures to answer one pair is five bindings where
 * one will do.
 */
export function tokenIdentity(features, tokens) {
  const out = new Float32Array(tokens * 8);
  for (let token = 0; token < tokens; token += 1) {
    out[token * 8] = features.residueIndex[token];
    out[token * 8 + 1] = features.tokenIndex[token];
    out[token * 8 + 2] = features.asymId[token];
    out[token * 8 + 3] = features.entityId[token];
    out[token * 8 + 4] = features.symId[token];
  }
  return out;
}

/**
 * `out[i][j][c] = sum of the four blocks' weight rows`.
 *
 * The bins are computed here rather than uploaded, because they are a function
 * of five integers a token and the alternative is an `n^2 * 4` tensor.
 */
const relativeEncodingSourceUncached = (tokens, channels) => `${PRELUDE}${ONE_TARGET}
const int L = ${tokens};
const int C4 = ${channels / 4};
const int ELEMENTS = ${tokens * tokens * channels};
const int RESIDX = ${RESIDX_BINS};
const int CHAINS = ${CHAIN_BINS};
const int RESIDUE_WIDTH = ${2 * RESIDX_BINS + 2};
uniform sampler2D identityTex;
uniform sampler2D weightTex;

int clipBin(int value, int high) { return min(max(value, 0), high); }

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;

  int residueI = int(fetch1(identityTex, i * 8));
  int residueJ = int(fetch1(identityTex, j * 8));
  int tokenI = int(fetch1(identityTex, i * 8 + 1));
  int tokenJ = int(fetch1(identityTex, j * 8 + 1));
  bool sameChain = fetch1(identityTex, i * 8 + 2) == fetch1(identityTex, j * 8 + 2);
  bool sameEntity = fetch1(identityTex, i * 8 + 3) == fetch1(identityTex, j * 8 + 3);
  int symI = int(fetch1(identityTex, i * 8 + 4));
  int symJ = int(fetch1(identityTex, j * 8 + 4));
  bool sameResidue = residueI == residueJ;

  int residueBin = sameChain
    ? clipBin(residueI - residueJ + RESIDX, 2 * RESIDX)
    : 2 * RESIDX + 1;
  int tokenBin = (sameChain && sameResidue)
    ? clipBin(tokenI - tokenJ + RESIDX, 2 * RESIDX)
    : 2 * RESIDX + 1;
  // ...and the chain block the other way round. See the note at the top.
  int chainBin = sameChain
    ? 2 * CHAINS + 1
    : clipBin(symI - symJ + CHAINS, 2 * CHAINS);

  int tokenBase = RESIDUE_WIDTH;
  int entityBase = tokenBase + RESIDUE_WIDTH;
  int chainBase = entityBase + 1;
  vec4 total = fetch4(weightTex, residueBin * C4 + group)
    + fetch4(weightTex, (tokenBase + tokenBin) * C4 + group)
    + fetch4(weightTex, (chainBase + chainBin) * C4 + group);
  // same_entity is a float in [0, 1] and not a one-hot, so it SCALES its row.
  // (No backticks in a GLSL comment: this is inside a JS template literal and
  // one would end it. Fifth time in this tree.)
  if (sameEntity) total += fetch4(weightTex, entityBase * C4 + group);
  result = total;
}`;

/** `out[i][j][c] = rows[i][c] + columns[j][c]`, z_init's broadcast. */
const broadcastSumSourceUncached = (tokens, channels) => `${PRELUDE}${ONE_TARGET}
const int L = ${tokens};
const int C4 = ${channels / 4};
const int ELEMENTS = ${tokens * tokens * channels};
uniform sampler2D rowsTex;
uniform sampler2D columnsTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;
  result = fetch4(rowsTex, i * C4 + group) + fetch4(columnsTex, j * C4 + group);
}`;

/** `out[pair][c] += bonds[pair] * weight[c]`, one input channel. */
const tokenBondSourceUncached = (tokens, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${tokens * tokens * channels};
uniform sampler2D baseTex;
uniform sampler2D bondTex;
uniform sampler2D weightTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  result = fetch4(baseTex, texel) + fetch1(bondTex, pair) * fetch4(weightTex, group);
}`;

/**
 * `z_init`, every term of it.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{xInputs: Tensor, identity: Tensor, bonds?: Tensor, languageModelPair?: Tensor,
 *   tokens: number, inputChannels: number, channels: number, weights: any,
 *   epsilon?: number, onStage?: (stage: string, tensor: Tensor) => void}} call
 * @returns {Tensor} `tokens^2 * channels`
 */
export function zInit(execution, call) {
  const {
    xInputs, identity, tokens, inputChannels, channels, weights: w,
  } = call;
  // 🔴 THE TWO PER-TOKEN PROJECTIONS ARE OPTIONAL, AND THE SHIPPED CHECKPOINT
  // HAS THEM. (An earlier version of this comment said it did not, from a
  // listing that collapsed `featuriser/zInit1` and `zInit2` out of view - they
  // are both [451, 256] in `model-esmfold2-int5`. A false claim in a comment is
  // worse than no comment, so: they are there.) They stay optional because
  // absent is not zero to a kernel - a missing sampler is refused by name - so a
  // bundle without them skips the draw rather than binding zeros, and the pair
  // then starts from the relative encoding, the bonds and the language model's
  // own pair. Allocated zero either way, which `Storage.allocate` guarantees.
  let pair = execution.allocate(tokens * tokens * channels, "ef2.z-init");
  if (w.zInit1 !== undefined && w.zInit2 !== undefined) {
    const rows = linear(execution, {
      x: xInputs, weight: w.zInit1, rows: tokens,
      inChannels: inputChannels, outChannels: channels, weightLayout: "in-major",
    });
    const columns = linear(execution, {
      x: xInputs, weight: w.zInit2, rows: tokens,
      inChannels: inputChannels, outChannels: channels, weightLayout: "in-major",
    });
    execution.release(pair);
    pair = execution.allocate(tokens * tokens * channels, "ef2.z-init");
    execution.dispatch({
      source: broadcastSumSource(tokens, channels),
      output: pair,
      inputs: { rowsTex: rows, columnsTex: columns },
      label: `ef2.z-init.broadcast:${tokens}x${channels}`,
    });
    execution.release(rows);
    execution.release(columns);
  }
  call.onStage?.("z-init.broadcast", pair);

  const relative = execution.allocate(tokens * tokens * channels, "ef2.relative");
  execution.dispatch({
    source: relativeEncodingSource(tokens, channels),
    output: relative,
    inputs: { identityTex: identity, weightTex: w.relativePosition },
    label: `ef2.relative:${tokens}x${channels}`,
  });
  call.onStage?.("z-init.relative", relative);
  const summed = execution.allocate(tokens * tokens * channels, "ef2.z-init");
  execution.dispatch({
    source: addSource(tokens * tokens * channels),
    output: summed,
    inputs: { aTex: pair, bTex: relative },
    label: `ef2.z-init.relative-add:${tokens}x${channels}`,
  });
  execution.release(relative);
  execution.release(pair);
  pair = summed;

  // 🔴 THE BOND TERM IS PRESENT ONLY WHERE THE JOB DECLARED ONE, and absent is
  // not zero to the kernel: a missing sampler is refused by name, so the draw is
  // skipped rather than bound to zeros.
  if (call.bonds !== undefined) {
    const bonded = execution.allocate(tokens * tokens * channels, "ef2.z-init");
    execution.dispatch({
      source: tokenBondSource(tokens, channels),
      output: bonded,
      inputs: { baseTex: pair, bondTex: call.bonds, weightTex: w.tokenBonds },
      label: `ef2.z-init.bonds:${tokens}x${channels}`,
    });
    execution.release(pair);
    pair = bonded;
  }
  if (call.languageModelPair !== undefined) {
    const withPair = execution.allocate(tokens * tokens * channels, "ef2.z-init");
    execution.dispatch({
      source: addSource(tokens * tokens * channels),
      output: withPair,
      inputs: { aTex: pair, bTex: call.languageModelPair },
      label: `ef2.z-init.lm-pair:${tokens}x${channels}`,
    });
    execution.release(pair);
    pair = withPair;
  }
  pair.label = "ef2.z-init";
  return pair;
}

/** `out = a + b`, local rather than imported so this file's draws are its own. */
const addSourceUncached = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D aTex;
uniform sampler2D bTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(aTex, texel) + fetch4(bTex, texel);
}`;

/**
 * `pair_loop_proj`: LayerNorm then a square projection, once per loop.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @returns {Tensor} `pairs * channels`, the update the next loop adds to z_init
 */
export function recycleProjection(execution, { pair, pairs, channels, weights, epsilon = 1e-5 }) {
  const normalised = layerNorm(execution, {
    x: pair, weight: weights.scale, bias: weights.offset,
    rows: pairs, channels, epsilon,
  });
  const projected = linear(execution, {
    x: normalised, weight: weights.projection, rows: pairs,
    inChannels: channels, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(normalised);
  projected.label = "ef2.pair-loop-proj";
  return projected;
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const relativeEncodingSource = memoSource(relativeEncodingSourceUncached);
export const broadcastSumSource = memoSource(broadcastSumSourceUncached);
export const tokenBondSource = memoSource(tokenBondSourceUncached);
export const addSource = memoSource(addSourceUncached);
