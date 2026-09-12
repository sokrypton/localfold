/**
 * What the pipeline cache was handed and threw away.
 *
 * 🔴 A CACHE HIT STILL PAYS FOR ITS SOURCE. `get` takes the finished WGSL, and
 * a block asks for the same pipelines on every block of every recycle, so every
 * generated source after the first is built and discarded. Nothing measured it:
 * `shaderSourceMiB` in probe-compiles.js counts what reaches
 * `createShaderModule`, which is exactly the sources that were NOT wasted.
 */
/**
 * The WGSL extensions this repository emits, and the feature each one needs.
 *
 * 🔴 A SHADER THAT ENABLES ONE WITHOUT THE FEATURE FAILS AS A WGSL PARSE ERROR
 * NAMING NO KEY, which cost a whole search: OpenDDE and ESMFold2 both did it and
 * neither folded on a stock Chrome. See docs/A100.md.
 */
export const EXTENSION_FEATURES = [
  ["f16", "shader-f16"],
  ["subgroups", "subgroups"],
  ["subgroup_size_control", "subgroup-size-control"],
  ["chromium_experimental_subgroup_matrix", "chromium-experimental-subgroup-matrix"],
];

export const pipelineCacheStats = {
  hits: 0, misses: 0, shared: 0, hitSourceBytes: 0, byKey: new Map(),
};

export class ComputePipelineCache {
  device;
  #pipelines = new Map

  ();

  /**
   * The pipelines already built, by the WGSL and entry point they were built
   * from.
   *
   * 🔴 TWO KEYS CAN NAME THE SAME SHADER, AND 78 OF OpenDDE's 269 DO. Measured
   * with `distinctSources` in probe-compiles.js: 269 modules made from 191
   * distinct texts, so 29% of the compiles reproduce a pipeline that already
   * exists. The keys differ for good reasons - they carry a token count, a
   * direction, a geometry - and two different shapes can still generate
   * character-for-character the same kernel.
   *
   * The source itself is the key, not a hash of it: a hash collision here would
   * hand a caller somebody else's kernel, and this cache exists to make that
   * impossible rather than unlikely. The texts are held by the source memo in
   * src/runtime/shader-source-cache.js anyway, so this retains nothing new.
   *
   * 🔴 SAFE ONLY BECAUSE THE LAYOUT IS `auto` AND DERIVED FROM THE SOURCE. Two
   * pipelines built from identical WGSL with the same entry point have the same
   * bind group layouts by construction. The label differs and is cosmetic -
   * profile.js times labelled compute PASSES, not pipelines.
   */
  #byContent = new Map();

  constructor(device) {
    this.device = device;
  }

  get(key, code, entryPoint = "main") {
    const cached = this.#pipelines.get(key);
    if (cached !== undefined) {
      pipelineCacheStats.hits += 1;
      const wasted = typeof code === "string" ? code.length : 0;
      pipelineCacheStats.hitSourceBytes += wasted;
      const row = pipelineCacheStats.byKey.get(key)
        ?? pipelineCacheStats.byKey.set(key, { hits: 0, bytes: 0 }).get(key);
      row.hits += 1;
      row.bytes += wasted;
      if (cached.code !== code || cached.entryPoint !== entryPoint) {
        // 🔴 A COLLISION IS A KEY THAT DOES NOT NAME ITS SHADER, AND THE NEXT
        // QUESTION IS ALWAYS "WHICH PART". Saying only the key leaves that to a
        // bisect; the first differing line usually names the constant that
        // moved - a weight offset, a tile, a channel count - and is what turns
        // this from an afternoon into a minute.
        const was = cached.code.split("\n");
        const now = String(code).split("\n");
        let at = 0;
        while (at < Math.max(was.length, now.length) && was[at] === now[at]) at += 1;
        const detail = cached.entryPoint !== entryPoint
          ? `entry point ${cached.entryPoint} against ${entryPoint}`
          : `line ${at + 1} of ${now.length}: ${JSON.stringify((was[at] ?? "").trim().slice(0, 90))}`
            + ` against ${JSON.stringify((now[at] ?? "").trim().slice(0, 90))}`;
        throw new Error(`WebGPU pipeline cache key collision for ${key} - ${detail}`);
      }
      return cached.pipeline;
    }
    const content = `${entryPoint}\u0000${code}`;
    const shared = this.#byContent.get(content);
    if (shared !== undefined) {
      pipelineCacheStats.shared += 1;
      this.#pipelines.set(key, { code, entryPoint, pipeline: shared });
      return shared;
    }
    pipelineCacheStats.misses += 1;
    // 🔴 A SHADER THAT ASKS FOR f16 ON A DEVICE WITHOUT IT FAILS AS A WGSL PARSE
    // ERROR WITH NO KEY, WHICH IS HOURS. Dawn reports "extension 'f16' is not
    // allowed in the current environment" against an uncaptured device error,
    // and nothing in it says which of the twelve shader factories emitted the
    // `enable f16`. It is a real shipped state and not a hypothetical: a stock
    // Chrome has no `shader-f16` on ANY NVIDIA GPU (see docs/A100.md), and two
    // of the four models were reaching here with it. Name the key instead.
    // 🔴 EVERY EXTENSION, NOT JUST f16, AND NOT ONLY THE FIRST LINE. The first
    // version of this read `code.startsWith("enable f16;")`, which is true of
    // the sources that happened to put f16 first and silently false of any that
    // do not - a guard with a false negative built in. WGSL requires every
    // `enable` before the first declaration, so scanning the leading directives
    // is exact, and the three extensions this repository emits each map to a
    // feature a device may not have.
    for (const [extension, feature] of EXTENSION_FEATURES) {
      if (!new RegExp(`(^|\\n)\\s*enable\\s+${extension}\\s*;`).test(code)) continue;
      if (this.device.features.has(feature)) continue;
      throw new Error(`${key} enables ${extension} on a device without `
        + `${feature}. The caller must gate on the feature - see `
        + "halfPrecisionAvailable and deviceProfile in src/runtime/device-profile.js. "
        + "A stock Chrome has neither shader-f16 nor the subgroup matrix units on "
        + "NVIDIA; see docs/A100.md.");
    }
    const pipeline = this.device.createComputePipelineAsync({
        label: key,
        layout: "auto",
        compute: {
          module: this.device.createShaderModule({ label: `${key}.wgsl`, code }),
          entryPoint,
        },
      });
    this.#pipelines.set(key, { code, entryPoint, pipeline });
    this.#byContent.set(content, pipeline);
    return pipeline;
  }

  get size() {
    return this.#pipelines.size;
  }
}

const DEVICE_PIPELINE_CACHES = new WeakMap();

/**
 * Returns the pipeline cache owned by a device.
 *
 * AlphaFold executes the same kernels in every block and recycle. Keeping this
 * cache at device lifetime avoids asking the browser to recreate identical
 * compute pipelines whenever a short-lived operator/execution object is made.
 */
export function pipelineCacheForDevice(device) {
  let cache = DEVICE_PIPELINE_CACHES.get(device);
  if (cache === undefined) {
    cache = new ComputePipelineCache(device);
    DEVICE_PIPELINE_CACHES.set(device, cache);
  }
  return cache;
}
