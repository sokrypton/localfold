/**
 * The WGSL a fold generates and then throws away.
 *
 * 🔴 A PIPELINE CACHE HIT STILL PAYS FOR ITS SOURCE. `ComputePipelineCache.get`
 * takes finished WGSL and discards it whenever the pipeline already exists, and
 * a block asks for the same pipelines on every block of every recycle. Measured
 * on this box with `pipelineCacheStats`: a 59-residue AF2 fold generates
 * **26.9 MiB of WGSL for 2,795 cache hits** against 93 misses, and a
 * 236-residue fold with one recycle generates 53.2 MiB. None of it reaches
 * `createShaderModule`, which is why `shaderSourceMiB` - which counts what
 * does - read 0.65 MiB and looked fine.
 *
 * The sources are pure functions of a shape and a set of storage choices, and
 * the pipeline key already names those in full, so the key is the cache key.
 *
 * 🔴 AND THAT IS THE WHOLE RISK. `ComputePipelineCache` compares the source it
 * is handed against the one it compiled and throws on a mismatch, which is the
 * check that has caught a key not naming everything its shader depends on (see
 * the project-ab collision in docs/AF2.md). Memoising by the same key makes
 * that comparison vacuous - it compares a string with itself. So the check
 * moves here: `setShaderSourceVerification(true)` rebuilds on every hit and
 * throws when the rebuild differs, and test/shader-source-cache.test.js turns
 * it on so the rule is gated rather than remembered.
 */
export const shaderSourceStats = { hits: 0, misses: 0, ms: 0, bytes: 0, hitBytes: 0 };

let verifying = false;

/** Rebuild every cached source and compare it. Off in a fold, on in the tests. */
export function setShaderSourceVerification(enabled) {
  verifying = enabled === true;
}

/** Whether sources are being verified - for a test that wants to restore it. */
export const shaderSourceVerification = () => verifying;

const CACHES = new WeakMap();

/**
 * The memoised source for `key`, built by `build` at most once per device.
 *
 * Per DEVICE rather than global: a source generator reads the device's matrix
 * configuration and tuning, and two devices in one process - which every
 * differential checker makes - must not share an answer that depends on them.
 * The key still has to name everything, because one device sweeping a knob is
 * the common case and it is the same device throughout.
 *
 * @param {GPUDevice} device
 * @param {string} key names everything the source depends on
 * @param {() => string} build
 * @returns {string}
 */
export function shaderSource(device, key, build) {
  let cache = CACHES.get(device);
  if (cache === undefined) {
    cache = new Map();
    CACHES.set(device, cache);
  }
  const cached = cache.get(key);
  if (cached !== undefined) {
    shaderSourceStats.hits += 1;
    shaderSourceStats.hitBytes += cached.length;
    if (verifying) {
      const rebuilt = build();
      if (rebuilt !== cached) {
        throw new Error(`shader source cache key ${key} does not name its source: `
          + `the rebuild differs by ${Math.abs(rebuilt.length - cached.length)} characters`);
      }
    }
    return cached;
  }
  const started = performance.now();
  const built = build();
  shaderSourceStats.ms += performance.now() - started;
  shaderSourceStats.misses += 1;
  shaderSourceStats.bytes += built.length;
  cache.set(key, built);
  return built;
}

/**
 * The memoised source SET for `key` - several sources from one generator.
 *
 * The triangle multiplication and the transition each build four to seven
 * sources in one call that shares its parsing and its offsets, so caching them
 * one at a time would call the generator once per source and save nothing.
 *
 * @template T
 * @param {GPUDevice} device
 * @param {string} key
 * @param {() => T} build
 * @returns {T}
 */
export function shaderSourceSet(device, key, build) {
  let cache = CACHES.get(device);
  if (cache === undefined) {
    cache = new Map();
    CACHES.set(device, cache);
  }
  const marked = `set:${key}`;
  const cached = cache.get(marked);
  if (cached !== undefined) {
    shaderSourceStats.hits += 1;
    if (verifying) {
      const rebuilt = build();
      if (JSON.stringify(rebuilt) !== JSON.stringify(cached)) {
        throw new Error(`shader source set key ${key} does not name its sources`);
      }
    }
    return cached;
  }
  const started = performance.now();
  const built = build();
  shaderSourceStats.ms += performance.now() - started;
  shaderSourceStats.misses += 1;
  cache.set(marked, built);
  return built;
}
