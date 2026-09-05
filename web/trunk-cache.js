/**
 * Whether the last fold's trunk is worth keeping on the host.
 *
 * 🔴 IT IS A CONVENIENCE AND IT WAS UNBOUNDED. `trunkCache` holds the reusable
 * trunk so that changing the sampler, its step count or the seed re-folds
 * without re-running the trunk - which is most of a fold, so on a 58-mer it is
 * the difference between three seconds and one. What it holds is `trunk.pair`,
 * and that is `tokens^2 x 128` float32: 3.4 MB at 58 residues, 1.2 GB at 1530.
 *
 * 🔴 AND HOLDING 1.2 GB OF JS HEAP HAS VISIBLE CONSEQUENCES ELSEWHERE. After a
 * 1530-residue fold, switching the viewer to cartoon refused with "Cartoon
 * needs about 21 MB, -712 MB free - staying in tube". That message is py2Dmol's
 * and it is correct: it reads `jsHeapSizeLimit - usedJSHeapSize`, and the
 * figure is NEGATIVE because the tab is genuinely over Chrome's heap limit.
 * The cache is a large part of why it is still over it once the fold is done.
 *
 * So the trade is taken by size. A fold small enough for the cache to be cheap
 * keeps it; one large enough to cost the page its heap does not - and a fold
 * that took ten minutes is not one anybody re-samples casually, which is the
 * case the cache exists for.
 */

/**
 * An eighth of the heap, when the browser will say how big the heap is.
 *
 * 🔴 THE SAME NUMBER py2Dmol's CARTOON CHECK READS, on purpose: the two are
 * competing for one heap, and a cache sized against a constant while the viewer
 * sizes itself against the limit is how one starves the other. `performance
 * .memory` is Chrome-only, so the fixed cap below is what every other browser
 * gets. On a typical desktop Chrome the two agree closely - a 4 GB limit gives
 * 512 MiB - which is the point at which they were chosen.
 */
export const TRUNK_CACHE_FALLBACK_BYTES = 512 * 1024 * 1024;
export const TRUNK_CACHE_HEAP_SHARE = 1 / 8;

/** How much host memory a reusable trunk holds. */
export function trunkCacheBytes(reusable) {
  let bytes = 0;
  const add = (value) => {
    if (ArrayBuffer.isView(value)) bytes += value.byteLength;
  };
  add(reusable?.targetFeat);
  // ...every array the trunk carries, not `pair` alone. It is the biggest by a
  // long way and it is not the only one, and a helper that names its fields
  // silently stops counting the day one is added.
  for (const value of Object.values(reusable?.trunk ?? {})) add(value);
  return bytes;
}

/**
 * The most a trunk may hold and still be worth caching, in bytes.
 *
 * @param {{memory?: {jsHeapSizeLimit?: number}}} [perf] injectable for tests
 */
export function trunkCacheLimit(perf) {
  const limit = perf?.memory?.jsHeapSizeLimit;
  if (typeof limit !== "number" || !Number.isFinite(limit) || limit <= 0) {
    return TRUNK_CACHE_FALLBACK_BYTES;
  }
  return limit * TRUNK_CACHE_HEAP_SHARE;
}

/**
 * Whether to keep this trunk, and what to say about it if not.
 *
 * Returns `{ keep, bytes, limit }` rather than deciding, so the caller can log
 * the refusal - a re-fold that silently re-runs a ten-minute trunk is worse
 * than one that says why.
 */
export function shouldCacheTrunk(reusable, perf) {
  const bytes = trunkCacheBytes(reusable);
  const limit = trunkCacheLimit(perf);
  return { keep: bytes <= limit, bytes, limit };
}
