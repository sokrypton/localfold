/**
 * What the pipeline cache was handed and threw away.
 *
 * 🔴 A CACHE HIT STILL PAYS FOR ITS SOURCE. `get` takes the finished WGSL, and
 * a block asks for the same pipelines on every block of every recycle, so every
 * generated source after the first is built and discarded. Nothing measured it:
 * `shaderSourceMiB` in probe-compiles.js counts what reaches
 * `createShaderModule`, which is exactly the sources that were NOT wasted.
 */
export const pipelineCacheStats = { hits: 0, misses: 0, hitSourceBytes: 0, byKey: new Map() };

export class ComputePipelineCache {
  device;
  #pipelines = new Map

  ();

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
        throw new Error(`WebGPU pipeline cache key collision for ${key}`);
      }
      return cached.pipeline;
    }
    pipelineCacheStats.misses += 1;
    const pipeline = this.device.createComputePipelineAsync({
        label: key,
        layout: "auto",
        compute: {
          module: this.device.createShaderModule({ label: `${key}.wgsl`, code }),
          entryPoint,
        },
      });
    this.#pipelines.set(key, { code, entryPoint, pipeline });
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
