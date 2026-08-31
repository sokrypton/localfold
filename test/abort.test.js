import assert from "node:assert/strict";
import { describe, expect, it } from "./harness.js";

import { AlphaFoldMonomerGpu } from "../src/model/monomer.js";
import { AlphaFoldQueryOnlyGpu } from "../src/model/query-only.js";
import { isAbortError, predictionAbortError, throwIfAborted, withAbort } from "../src/runtime/abort.js";
import { StructureCoreGpu } from "../src/structure/core.js";
import { StructureModuleGpu } from "../src/structure/module.js";

describe("prediction abort and cancellation", () => {
  it("creates errors with the standard AbortError identity", () => {
    const error = predictionAbortError();
    expect(error.name).toBe("AbortError");
    expect(error.message).toBe("Prediction stopped");
    expect(isAbortError(error)).toBe(true);
    expect(isAbortError(new Error("unrelated"))).toBe(false);
  });

  it("validates and observes an AbortSignal in throwIfAborted", () => {
    assert.doesNotThrow(() => throwIfAborted(undefined));
    expect(() => throwIfAborted({})).toThrow(/signal must be an AbortSignal/);
    const controller = new AbortController();
    assert.doesNotThrow(() => throwIfAborted(controller.signal));
    controller.abort();
    expect(() => throwIfAborted(controller.signal)).toThrow({ name: "AbortError" });
  });

  it("races in-flight promises against the abort signal with withAbort", async() => {
    expect(await withAbort(Promise.resolve(42), undefined)).toBe(42);
    const controller = new AbortController();
    expect(await withAbort(Promise.resolve("done"), controller.signal)).toBe("done");

    // Pre-aborted
    controller.abort();
    await assert.rejects(withAbort(new Promise(() => {}), controller.signal), { name: "AbortError" });

    // Aborted mid-flight
    const liveController = new AbortController();
    let hungSettled = false;
    const hungPromise = new Promise((resolve) => {
      setTimeout(() => { hungSettled = true; resolve("late"); }, 500);
    });
    const abortPromise = withAbort(hungPromise, liveController.signal);
    setTimeout(() => liveController.abort(), 10);
    await assert.rejects(abortPromise, { name: "AbortError" });
    expect(hungSettled).toBe(false);
  });

  it("stops model and structure public runs before dispatch when already aborted", async() => {
    const controller = new AbortController();
    controller.abort();
    const signal = controller.signal;
    const features = [{ aatype: new Int32Array(1), seqMask: new Float32Array([1]) }];
    await assert.rejects(
      new AlphaFoldQueryOnlyGpu({}).predict(features, {}, undefined, undefined, undefined, { signal }),
      { name: "AbortError" },
    );
    await assert.rejects(
      new AlphaFoldMonomerGpu({}).predict(features, {}, undefined, undefined, undefined, { signal }),
      { name: "AbortError" },
    );
    await assert.rejects(new StructureCoreGpu({}).run({ signal }), { name: "AbortError" });
    await assert.rejects(new StructureModuleGpu({}).run({ signal }), { name: "AbortError" });
  });
});

