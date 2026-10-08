/**
 * Deferred WebGPU validation for the block loops.
 *
 * 🔴 `await device.popErrorScope()` INSIDE A BLOCK LOOP COSTS A ROUND TRIP PER
 * BLOCK. The Evoformer submits 48 of them per recycle, so awaiting each scope
 * put a host-device synchronisation between every pair of blocks and undid the
 * pipelining the loop is written for. Removing the scopes recovered the speed
 * and gave up the only check that says a block was malformed.
 *
 * The pop itself is synchronous - only its *result* is a promise - so a scope
 * can be pushed before encoding and popped after submitting without waiting for
 * either. Collecting those promises and reading them once, at a boundary that
 * already synchronises, keeps the diagnostic and costs nothing per block.
 *
 * The push must come BEFORE encoding: most validation errors are raised as
 * commands are encoded, not when the buffer is submitted, so a scope opened
 * after `encode...()` covers the half that rarely fails.
 */
export class DeferredValidation {
  #device;
  #label;
  #pending = [];

  /**
   * @param {GPUDevice} device
   * @param {string} label names the stack in any error message
   */
  constructor(device, label) {
    this.#device = device;
    this.#label = label;
  }

  /** Open a scope around the work that is about to be encoded and submitted. */
  begin() {
    this.#device.pushErrorScope("validation");
  }

  /**
   * Close the scope opened by the matching `begin()`, without waiting for it.
   * @param {string} where identifies the block, e.g. `block 7`
   */
  end(where) {
    // ...resolved to a message rather than rejected. These promises are held
    // across the rest of the loop, and a rejection nobody is awaiting yet is an
    // unhandled rejection the moment the loop throws for any other reason -
    // an abort, most often - which would report the wrong failure.
    this.#pending.push(this.#device.popErrorScope().then(
      (error) => (error === null ? null : `${where}: ${error.message}`),
    ));
  }

  /** Throw if any scope closed over a validation error. */
  async settle() {
    const pending = this.#pending;
    this.#pending = [];
    const failures = (await Promise.all(pending)).filter((message) => message !== null);
    if (failures.length > 0) {
      throw new Error(`WebGPU ${this.#label} validation failed: ${failures.join("; ")}`);
    }
  }
}
