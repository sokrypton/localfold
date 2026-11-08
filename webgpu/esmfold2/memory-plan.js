/**
 * What an ESMFold2 fold holds on the device at its fullest, and where it can give some back.
 *
 * The pair tensor P is `tokens^2 * channels * 4` bytes and it decides everything: 88 MiB at 300 tokens, 1.4 GiB at 1200.
 * The weights are small and constant beside it. So a fold fits a card according to how many P it holds at once, and
 * every stage holds them one at a time now:
 *
 *   track      the pair, the contraction's output, a and b for one channel group (as few as there is room for), two row chunks
 *   prep       the language pair and z_init
 *   recycle    z_init beside the track, unless it waits on the host
 *   head       the initial pair beside the track, unless it waits on the host
 *
 * Below `COMPACT_MIN_BYTES` the track keeps its whole tensors, because a tensor that small is not worth the dispatches.
 */

import { MOST_SEQUENTIAL_GROUPS, SEQUENTIAL_CHUNK_BYTES, fewestSequentialGroups } from "../af3/trunk/sequential-groups.js";

const MIB = 1024 * 1024;

/** Weights resident at the same time, measured at 600 tokens: 170 MiB of trunk beside the denoiser's 200 and change. */
export const FOLD_WEIGHT_BYTES = 480 * MIB;
/** The same weights on a device with no shader-f16, which reads them as f32. Measured on an A10 at 600 and 1200 tokens. */
export const FOLD_WEIGHT_BYTES_F32 = 720 * MIB;
/** The sequential track's own row chunks, two of them. */
export const TRACK_CHUNK_BYTES = 2 * SEQUENTIAL_CHUNK_BYTES;
/** A pair this size or larger takes the sequential track; see gridChunkWorthIt in webgpu/af3/trunk/pair-track.js. */
export const COMPACT_MIN_BYTES = 32 * MIB;
/** What one fold may not leave free: the allocator's sums do not count a destroyed buffer the driver has not yet freed. */
export const FOLD_MARGIN_BYTES = 256 * MIB;

export const pairBytes = (tokens, channels = 256) => tokens * tokens * channels * 4;

/**
 * The most a fold holds at once, in bytes. `spill`: z_init and the head's initial pair wait on the host. `groups`: the
 * track's channel groups; a and b for one are 2/groups of a pair tensor, and at the most groups this is the least a fold
 * can hold, which is what decides whether it fits.
 */
export function esmfold2PeakBytes({ tokens, channels = 256, spill = false, weightBytes = FOLD_WEIGHT_BYTES,
                                    groups = MOST_SEQUENTIAL_GROUPS }) {
  const P = pairBytes(tokens, channels);
  // Below the gate the whole-tensor track: the pair, a, b, the contraction's output and the normalised input.
  const track = P < COMPACT_MIN_BYTES ? 5 * P : (2 + 2 / groups) * P + TRACK_CHUNK_BYTES;
  const beside = spill ? 0 : P;
  return track + beside + weightBytes;
}

/**
 * Whether z_init and the head's initial pair should wait on the host, given what the device holds and may hold.
 *
 * `residentBytes` is mostly this fold's own weights by the time this is asked, so it stands in for the plan's figure
 * and is not added to it; whichever is the larger is what the fold will have on the card beside its tensors.
 */
export function esmfold2Spills({ tokens, channels = 256, residentBytes = 0, budgetBytes,
                                 weightBytes = FOLD_WEIGHT_BYTES }) {
  if (budgetBytes === undefined || budgetBytes === null) return false;
  const peak = esmfold2PeakBytes({ tokens, channels, weightBytes: Math.max(residentBytes, weightBytes) });
  return peak + FOLD_MARGIN_BYTES > budgetBytes;
}

/** The channel groups a fold will take, from what it will have resident when the track starts. See sequential-groups.js. */
export function esmfold2Groups({ tokens, channels = 256, spill = false, weightBytes = FOLD_WEIGHT_BYTES, budgetBytes, maxBufferBytes }) {
  const P = pairBytes(tokens, channels);
  return fewestSequentialGroups({ pairBytes: P, residentBytes: weightBytes + (spill ? P : 2 * P), budgetBytes, maxBufferBytes });
}

/**
 * The longest fold any device can bind. Every pair-sized tensor is several buffers past one binding's reach and is bound
 * a chunk at a time, but the contraction's output is read across all its channels at once, from two buffers.
 */
export const esmfold2TokenCeiling = ({ channels = 256, maxBindingBytes = 2147483644 } = {}) =>
  Math.floor(Math.sqrt(2 * maxBindingBytes / (channels * 4)));

/** The longest fold this budget holds, in tokens. */
export function esmfold2MaxTokens({ budgetBytes, channels = 256, maxBindingBytes = 2147483644,
                                    weightBytes = FOLD_WEIGHT_BYTES }) {
  const available = budgetBytes - weightBytes - TRACK_CHUNK_BYTES - FOLD_MARGIN_BYTES;
  const tokens = Math.floor(Math.sqrt(Math.max(0, available) / (2 + 2 / MOST_SEQUENTIAL_GROUPS) / (channels * 4)));
  return Math.max(0, Math.min(tokens, esmfold2TokenCeiling({ channels, maxBindingBytes })));
}
