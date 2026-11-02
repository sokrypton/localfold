/**
 * The CPU fold from the page's side: the same callbacks as `foldEsmfold2`, with the
 * work in web/esmfold2-cpu-worker.js.
 */
import { predictionAbortError } from "../webgpu/runtime/abort.js";

let worker;
let workerFamily;

/** End the worker, which is what stops a fold and gives back its 2 to 4 GiB. */
export function releaseCpuFold() {
  worker?.terminate();
  worker = undefined;
  workerFamily = undefined;
}

/**
 * @param {object} options `family`, `entities`, `sequence`, `loops`, `languageModel`,
 *   `sampler`, `seed`, `threads`, `signal`, and `onBatch`, `onStatus`, `onProgress`,
 *   `onContacts`, `onStep`, `onDownload`
 * @returns {Promise<object>} what `foldEsmfold2` returns, less its GPU-only reuse state
 */
export function foldEsmfold2OnCpu(options) {
  // One model to a worker: a backend's weights never leave its memory, and another
  // model's reads back from the shard cache in seconds.
  if (workerFamily !== options.family) releaseCpuFold();
  worker ??= new Worker(new URL("./esmfold2-cpu-worker.js", import.meta.url), { type: "module" });
  workerFamily = options.family;
  const running = worker;
  return new Promise((resolve, reject) => {
    let features;
    const finish = (settle, value) => {
      options.signal?.removeEventListener("abort", onAbort);
      running.onmessage = null;
      running.onerror = null;
      settle(value);
    };
    const onAbort = () => {
      releaseCpuFold();
      finish(reject, predictionAbortError());
    };
    if (options.signal?.aborted) { onAbort(); return; }
    options.signal?.addEventListener("abort", onAbort, { once: true });
    running.onerror = (event) => {
      releaseCpuFold();
      finish(reject, new Error(event.message || "The CPU fold's worker failed to start."));
    };
    running.onmessage = ({ data }) => {
      switch (data.type) {
        case "download": options.onDownload?.(data); break;
        case "batch": options.onBatch?.(data.batch); break;
        case "status": options.onStatus?.(data.text); break;
        case "progress": options.onProgress?.(data.fraction); break;
        case "contacts": void options.onContacts?.(data.contacts); break;
        case "step":
          features = data.features ?? features;
          void options.onStep?.({ step: data.step, total: data.total, coordinates: data.coordinates,
                                  denoised: data.denoised, features });
          break;
        case "result": finish(resolve, data.result); break;
        case "error":
          console.error("CPU fold failed", data.stack);
          // A wasm trap or an allocation failure leaves the worker's memory in no
          // state to fold again; a fresh one reloads from the shard cache.
          releaseCpuFold();
          finish(reject, new Error(data.message));
          break;
        default:
      }
    };
    running.postMessage({
      type: "fold", family: options.family, entities: options.entities, sequence: options.sequence,
      loops: options.loops, languageModel: options.languageModel, sampler: options.sampler,
      seed: options.seed, threads: options.threads,
    });
  });
}
