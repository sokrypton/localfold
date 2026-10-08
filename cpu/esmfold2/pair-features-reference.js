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
 * 🔴 THE LOOP RUNS `num_loops + 1` TIMES AND `z` STARTS AT ZERO. The config
 * says 3 and the model runs four; reading that as three is a silent quarter
 * less trunk, which still folds. `pair_loop_proj`'s Linear is zero-initialised
 * upstream, so the first iteration is exactly `folding_trunk(z_init)`.
 *
 * 🔴 AND THIS IS THE **EXPERIMENTAL** MODEL'S RECURRENCE. `esm`'s `model.py`
 * defines a different one for the non-experimental ESMFold2 - a diagonal
 * state-space update, `z = a * z + linear(norm(z_inject), b)`, with a learned
 * decay, a readout and a second trunk as a coda. Both files define a
 * `folding_trunk` and a `z_init`; neither names the other. See CLAUDE.md.
 */

/** LayerNorm over the last axis, with scale and offset. */
function layerNorm(input, rows, channels, scale, offset, epsilon = 1e-5) {
  const out = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    const base = row * channels;
    let mean = 0;
    for (let c = 0; c < channels; c += 1) mean += input[base + c];
    mean /= channels;
    let variance = 0;
    for (let c = 0; c < channels; c += 1) {
      const d = input[base + c] - mean;
      variance += d * d;
    }
    const scaleBy = 1 / Math.sqrt(variance / channels + epsilon);
    for (let c = 0; c < channels; c += 1) {
      out[base + c] = (input[base + c] - mean) * scaleBy * scale[c] + offset[c];
    }
  }
  return out;
}

/** rows x inChannels against (inChannels, outChannels). */
function linear(input, rows, inChannels, outChannels, weights) {
  const out = new Float32Array(rows * outChannels);
  for (let row = 0; row < rows; row += 1) {
    const inBase = row * inChannels;
    const outBase = row * outChannels;
    for (let c = 0; c < inChannels; c += 1) {
      const value = input[inBase + c];
      if (value === 0) continue;
      const weightBase = c * outChannels;
      for (let o = 0; o < outChannels; o += 1) out[outBase + o] += value * weights[weightBase + o];
    }
  }
  return out;
}

/**
 * The 139 relative-position features, as bin indices rather than a one-hot.
 *
 * 🔴 THE CHAIN BLOCK'S POLARITY IS THE OPPOSITE OF THE OTHER TWO, IN THE SAME
 * FUNCTION. The residue and token blocks send a DIFFERENT-chain pair to the
 * out-of-range bin; the chain block sends a SAME-chain one there:
 *
 *     dij_residue = where(same_chain, dij, 2 * r + 1)
 *     dij_chain   = where(same_chain, 2 * c + 1, dij)
 *
 * Both are "the pair does not have a meaningful offset on this axis" and they
 * are written the two opposite ways round. Getting it backwards is a monomer
 * that still folds - every pair is same-chain, so the whole block becomes one
 * constant column either way - and a COMPLEX that is quietly wrong. That is
 * why check-esmfold2-featuriser.js sweeps it rather than trusting this note.
 *
 * @returns {Int32Array} four bin indices per pair: residue, token, sameEntity, chain
 */
export function relativePositionBins(features, n, residxBins = 32, chainBins = 2) {
  const { residueIndex, asymId, symId, entityId, tokenIndex } = features;
  const bins = new Int32Array(n * n * 4);
  const clip = (value, high) => Math.min(Math.max(value, 0), high);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const at = (i * n + j) * 4;
      const sameChain = asymId[i] === asymId[j];
      const sameResidue = residueIndex[i] === residueIndex[j];
      bins[at] = sameChain
        ? clip(residueIndex[i] - residueIndex[j] + residxBins, 2 * residxBins)
        : 2 * residxBins + 1;
      bins[at + 1] = (sameChain && sameResidue)
        ? clip(tokenIndex[i] - tokenIndex[j] + residxBins, 2 * residxBins)
        : 2 * residxBins + 1;
      bins[at + 2] = entityId[i] === entityId[j] ? 1 : 0;
      // ...and this one the other way round. See the note above.
      bins[at + 3] = sameChain
        ? 2 * chainBins + 1
        : clip(symId[i] - symId[j] + chainBins, 2 * chainBins);
    }
  }
  return bins;
}

/**
 * `rel_pos`: one-hot the four blocks, concatenate, project.
 *
 * 🔴 A ONE-HOT TIMES A MATRIX IS A ROW LOOKUP, AND WRITING IT AS A MATMUL HERE
 * WOULD BE 139x THE WORK FOR THE SAME ANSWER. At 300 tokens the one-hot alone
 * is 90,000 x 139 floats. Only four of those 139 are ever non-zero and three of
 * them are 1, so this adds four weight rows per pair.
 */
export function relativePositionEncoding(features, n, channels, weights,
                                         residxBins = 32, chainBins = 2) {
  const bins = relativePositionBins(features, n, residxBins, chainBins);
  const residueWidth = 2 * residxBins + 2;
  // The concatenation order: residue, token, sameEntity, chain.
  const tokenBase = residueWidth;
  const entityBase = tokenBase + residueWidth;
  const chainBase = entityBase + 1;
  const out = new Float32Array(n * n * channels);
  for (let pair = 0; pair < n * n; pair += 1) {
    const at = pair * 4;
    const outBase = pair * channels;
    const rows = [bins[at], tokenBase + bins[at + 1], chainBase + bins[at + 3]];
    for (const row of rows) {
      const weightBase = row * channels;
      for (let c = 0; c < channels; c += 1) out[outBase + c] += weights[weightBase + c];
    }
    // ...`same_entity` is a float in [0, 1], not a one-hot, so it scales.
    if (bins[at + 2] !== 0) {
      const weightBase = entityBase * channels;
      for (let c = 0; c < channels; c += 1) out[outBase + c] += weights[weightBase + c];
    }
  }
  return out;
}

/** `token_bonds`: one input channel, so the projection is an outer product. */
export function tokenBondEncoding(bonds, n, channels, weights) {
  const out = new Float32Array(n * n * channels);
  for (let pair = 0; pair < n * n; pair += 1) {
    const value = bonds[pair];
    if (value === 0) continue;
    const base = pair * channels;
    for (let c = 0; c < channels; c += 1) out[base + c] = value * weights[c];
  }
  return out;
}

/**
 * `z_init_1(x)[i] + z_init_2(x)[j]`: a per-token projection broadcast two ways.
 *
 * 🔴 PROJECT THEN BROADCAST, NEVER BROADCAST THEN PROJECT. The two are the same
 * arithmetic and differ by a factor of n in cost: at 300 tokens this is 600
 * projections of 451 channels rather than 90,000 of them.
 */
export function zInitFromInputs(xInputs, n, inputChannels, channels, first, second) {
  const rows = linear(xInputs, n, inputChannels, channels, first);
  const columns = linear(xInputs, n, inputChannels, channels, second);
  const out = new Float32Array(n * n * channels);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const base = (i * n + j) * channels;
      for (let c = 0; c < channels; c += 1) {
        out[base + c] = rows[i * channels + c] + columns[j * channels + c];
      }
    }
  }
  return out;
}

/** `pair_loop_proj`: LayerNorm then a square projection, once per loop. */
export function recycleProjection(pair, pairs, channels, weights, epsilon = 1e-5) {
  const normalised = layerNorm(pair, pairs, channels, weights.scale, weights.offset, epsilon);
  return linear(normalised, pairs, channels, channels, weights.projection);
}

export { layerNorm, linear };
