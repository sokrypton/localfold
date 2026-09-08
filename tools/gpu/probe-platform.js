/** What this browser offers a numerical workload, beside WebGPU itself. */
export async function main(device) {
  const has = (o, k) => { try { return typeof o?.[k] !== "undefined"; } catch { return false; } };
  const wasmFeature = (bytes) => { try { return WebAssembly.validate(new Uint8Array(bytes)); }
    catch { return false; } };
  // A minimal module using v128 - validates only where SIMD is supported.
  const simd = wasmFeature([0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,
    253,15,253,98,11]);
  const threads = wasmFeature([0,97,115,109,1,0,0,0,5,4,1,3,1,1,10,11,1,9,0,65,0,254,16,2,0,26,11]);
  return {
    hardwareConcurrency: navigator.hardwareConcurrency,
    // Host-side arithmetic
    wasmSimd128: simd,
    wasmThreadsAtomics: threads,
    sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    crossOriginIsolated: globalThis.crossOriginIsolated ?? null,
    // 🔴 NATIVE HALF PRECISION IN JAVASCRIPT. src/runtime/float16.js hand-rolls
    // the conversion; where this exists the engine does it.
    float16Array: typeof Float16Array !== "undefined",
    mathF16round: typeof Math.f16round === "function",
    // Moving work off the critical path
    worker: typeof Worker !== "undefined",
    offscreenCanvas: typeof OffscreenCanvas !== "undefined",
    // Getting weights in
    compressionStreams: typeof DecompressionStream !== "undefined",
    decompressionFormats: (() => {
      const out = [];
      for (const f of ["gzip", "deflate", "deflate-raw"]) {
        try { new DecompressionStream(f); out.push(f); } catch { /* unsupported */ }
      }
      return out;
    })(),
    cacheStorage: typeof caches !== "undefined",
    storageEstimate: has(navigator, "storage"),
    // Other compute backends
    webnn: typeof navigator.ml !== "undefined",
    webgpuFeatures: [...device.features].sort(),
  };
}
