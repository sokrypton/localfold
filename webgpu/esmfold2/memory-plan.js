/**
 * What an ESMFold2 fold holds on the device at its fullest, and where it can give some back.
 *
 * The pair tensor P is `tokens^2 * channels * 4` bytes and it decides everything: 88 MiB at 300 tokens, 1.4 GiB at 1200.
 * The weights are small and constant beside it. So a fold fits a card according to how many P it holds at once, and
 * every stage holds them one at a time now:
 *
 *   track      the pair, the contraction's output, a and b for one channel group, two row chunks
 *   prep       the language pair and z_init
 *   recycle    z_init beside the track, unless it waits on the host
 *   head       the initial pair beside the track, unless it waits on the host
 *
 * Below `COMPACT_MIN_BYTES` the track keeps its whole tensors, because a tensor that small is not worth the dispatches.
 */

const MIB = 1024 * 1024;

/** Weights resident at the same time, measured at 600 tokens: 170 MiB of trunk beside the denoiser's 200 and change. */
export const FOLD_WEIGHT_BYTES = 480 * MIB;
/** The sequential track's own row chunks, two of them. */
export const TRACK_CHUNK_BYTES = 128 * MIB;
/** A pair this size or larger takes the sequential track; see gridChunkWorthIt in webgpu/af3/trunk/pair-track.js. */
export const COMPACT_MIN_BYTES = 128 * MIB;
/** What one fold may not leave free: the allocator's sums do not count a destroyed buffer the driver has not yet freed. */
export const FOLD_MARGIN_BYTES = 256 * MIB;

/** a and b for one of the sequential track's channel groups, in pair tensors. */
const AB_PAIRS = 2 / 8;

export const pairBytes = (tokens, channels = 256) => tokens * tokens * channels * 4;

/** The most a fold holds at once, in bytes. `spill`: z_init and the head's initial pair wait on the host. */
export function esmfold2PeakBytes({ tokens, channels = 256, spill = false }) {
  const P = pairBytes(tokens, channels);
  // Below the gate the whole-tensor track: the pair, a, b, the contraction's output and the normalised input.
  const track = P < COMPACT_MIN_BYTES ? 5 * P : (2 + AB_PAIRS) * P + TRACK_CHUNK_BYTES;
  const beside = spill ? 0 : P;
  return track + beside + FOLD_WEIGHT_BYTES;
}

/** Whether z_init and the head's initial pair should wait on the host, given what the device holds and may hold. */
export function esmfold2Spills({ tokens, channels = 256, residentBytes = 0, budgetBytes }) {
  if (budgetBytes === undefined || budgetBytes === null) return false;
  return residentBytes + esmfold2PeakBytes({ tokens, channels }) + FOLD_MARGIN_BYTES > budgetBytes;
}

/** The longest fold this budget holds, in tokens. The ceiling is the largest tensor a storage binding can name. */
export function esmfold2MaxTokens({ budgetBytes, channels = 256, maxBindingBytes = 2147483644 }) {
  const available = budgetBytes - FOLD_WEIGHT_BYTES - TRACK_CHUNK_BYTES - FOLD_MARGIN_BYTES;
  const tokens = Math.floor(Math.sqrt(available / (2 + AB_PAIRS) / (channels * 4)));
  const ceiling = Math.floor(Math.sqrt(maxBindingBytes / (channels * 4)));
  return Math.max(0, Math.min(tokens, ceiling));
}
