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
      if (cached.code !== code || cached.entryPoint !== entryPoint) {
        throw new Error(`WebGPU pipeline cache key collision for ${key}`);
      }
      return cached.pipeline;
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
