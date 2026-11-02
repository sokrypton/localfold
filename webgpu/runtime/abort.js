/** Create the consistent error used when a local prediction is stopped. */
export function predictionAbortError() {
  const error = new Error("Prediction stopped");
  error.name = "AbortError";
  return error;
}

/**
 * Stop at a boundary where no JavaScript-owned GPU resources are in flight.
 * WebGPU commands that were already submitted cannot be preempted, so callers
 * invoke this before and after the awaited boundaries that drain those commands.
 */
export function throwIfAborted(signal) {
  if (signal === undefined) return;
  if (signal === null || typeof signal !== "object" || typeof signal.aborted !== "boolean") {
    throw new TypeError("signal must be an AbortSignal");
  }
  if (signal.aborted) throw predictionAbortError();
}

/** True for both LocalFold's error and AbortErrors produced by fetch. */
export function isAbortError(error) {
  return error !== null && typeof error === "object" && error.name === "AbortError";
}

/**
 * Await a promise but reject immediately if the AbortSignal aborts.
 * This prevents the host from waiting out in-flight GPU queues or readbacks
 * once the user requests a stop.
 *
 * @template T
 * @param {Promise<T>} promise
 * @param {AbortSignal} [signal]
 * @returns {Promise<T>}
 */
export async function withAbort(promise, signal) {
  if (signal === undefined) return promise;
  throwIfAborted(signal);
  let onAbort;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        onAbort = () => reject(predictionAbortError());
        signal.addEventListener("abort", onAbort, { once: true });
      }),
    ]);
  } finally {
    if (onAbort !== undefined) {
      signal.removeEventListener("abort", onAbort);
    }
  }
}

