/**
 * The CPU fold's thread. A CPU fold blocks whatever it runs on for minutes, so it
 * runs here, and stopping it is ending this worker.
 *
 * The weights stay until the worker is ended, so a second fold of the same model
 * pays nothing to load. One model to a worker: web/esmfold2-cpu.js ends this one to
 * change model, because a backend's weights never leave its memory.
 */
import { CpuBackend } from "../cpu/runtime/backend.js";
import { setLinearBackend } from "../cpu/runtime/linear-backend.js";
import { foldEsmfold2Cpu } from "./esmfold2-cpu-fold.js";
import { loadEsmfold2CpuWeights } from "../cpu/esmfold2/load.js";

let held;

async function weightsFor(family, threads) {
  if (held !== undefined) return held.loaded;
  const backend = await CpuBackend.create({ threads });
  setLinearBackend(backend);
  const loaded = await loadEsmfold2CpuWeights(family, backend, (progress) => {
    self.postMessage({ type: "download", ...progress });
  });
  held = { family, backend, loaded };
  return loaded;
}

/** A typed array's own buffer may be moved; one over the backend's memory may not. */
const movable = (value) => (ArrayBuffer.isView(value) && value.buffer instanceof ArrayBuffer ? [value.buffer] : []);

self.onmessage = async ({ data }) => {
  if (data.type !== "fold") return;
  try {
    const loaded = await weightsFor(data.family, data.threads);
    const result = await foldEsmfold2Cpu({
      entities: data.entities, sequence: data.sequence,
      shape: { ...loaded.shape, loops: data.loops },
      weights: loaded.weights, confidenceWeights: loaded.confidenceWeights, language: loaded.language,
      languageModel: data.languageModel, sampler: data.sampler, seed: data.seed,
      onBatch: (batch) => self.postMessage({ type: "batch", batch }),
      onStatus: (text) => self.postMessage({ type: "status", text }),
      onProgress: (fraction) => self.postMessage({ type: "progress", fraction }),
      onContacts: (contacts) => { self.postMessage({ type: "contacts", contacts }); },
      onStep: ({ step, total, coordinates, denoised, features }) => {
        self.postMessage({ type: "step", step, total, coordinates, denoised,
                           features: step === 0 ? features : undefined });
      },
    });
    self.postMessage({ type: "result", result }, [...movable(result.coordinates), ...movable(result.contacts)]);
  } catch (error) {
    self.postMessage({ type: "error", message: error?.message ?? String(error), stack: error?.stack });
  }
};
