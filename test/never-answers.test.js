/**
 * What the page does about a call into the browser that never comes back, and about the few that refuse outright: a GPU
 * request, a service worker that failed to install, a lock. None of them says anything at all, so each needs a deadline or
 * a way past, and the time is short here only because `setTimeout` is told so.
 */
import { strict as assert } from "node:assert";
import { afterEach, beforeEach, describe, it } from "node:test";
import { ModelDownloadError } from "../shared/bundles/http-tensor-store.js";

const realTimeout = globalThis.setTimeout;
const never = () => new Promise(() => {});
const stored = new Map();

beforeEach(() => {
  stored.clear();
  // A deadline here is seconds, and everything else in these tests is a tick.
  globalThis.setTimeout = (run, milliseconds, ...rest) => realTimeout(run, milliseconds >= 1000 ? 1 : milliseconds, ...rest);
});
afterEach(() => { globalThis.setTimeout = realTimeout; });

globalThis.sessionStorage = {
  getItem: (key) => stored.get(key) ?? null,
  setItem: (key, value) => stored.set(key, value),
  removeItem: (key) => stored.delete(key),
};
globalThis.window = { crossOriginIsolated: false };
globalThis.document = { title: "LocalFold", visibilityState: "visible", addEventListener() {}, removeEventListener() {} };
globalThis.window.addEventListener = () => {};
globalThis.window.removeEventListener = () => {};
const browser = { hardwareConcurrency: 8, gpu: { requestAdapter: async () => null }, deviceMemory: 16 };
Object.defineProperty(globalThis, "navigator", { value: browser, configurable: true });

/** The module as a new page load finds it, with what it reads from sessionStorage at the top. */
const freshSettings = (load) => import(new URL(`../web/settings-panel.js?${load}`, import.meta.url).href);

function fakeAdapter(requestDevice) {
  return { features: new Set(), limits: {}, info: { vendor: "acme", architecture: "test", description: "fake card" }, requestDevice };
}

const fakeDevice = () => ({
  label: "fake", features: new Set(), limits: { maxBufferSize: 2 ** 31 }, lost: never(),
  queue: { submit() {}, onSubmittedWorkDone: async () => {}, writeBuffer() {} }, addEventListener() {},
  destroyed: false, destroy() { this.destroyed = true; }, pushErrorScope() {}, popErrorScope: async () => null,
  createBuffer: () => { throw new Error("the occupancy probe was not stubbed"); },
});

describe("a GPU that never answers", () => {
  it("stops waiting for an adapter, and asks again", async () => {
    const { getDevice, planningGpu, releaseDevice } = await import("../web/model.js");
    browser.gpu.requestAdapter = never;
    assert.equal(await planningGpu(), undefined);
    await assert.rejects(getDevice(), /no usable GPU/);
    const device = fakeDevice();
    browser.gpu.requestAdapter = async () => fakeAdapter(async () => device);
    assert.equal(await getDevice(), device);
    await releaseDevice();
  });

  it("stops waiting for a device, and gives back one that comes after all", async () => {
    const { getDevice, releaseDevice } = await import("../web/model.js");
    let answer;
    browser.gpu.requestAdapter = async () => fakeAdapter(() => new Promise((resolve) => { answer = resolve; }));
    await assert.rejects(getDevice(), /not accepting work/);
    const late = fakeDevice();
    answer(late);
    await new Promise((resolve) => { realTimeout(resolve, 5); });
    assert.equal(late.destroyed, true);
    await releaseDevice();
  });
});

describe("the worker that threads need", () => {
  it("is given up on when it never starts, and the line counts the one thread that leaves", async () => {
    browser.serviceWorker = { register: async () => ({}), ready: never() };
    const { cpuThreadsOnceIsolated, setIsolated } = await freshSettings("never-ready");
    assert.equal(cpuThreadsOnceIsolated(), 4);
    await assert.rejects(setIsolated(true), /did not start/);
    assert.equal(cpuThreadsOnceIsolated(), 1);
    assert.equal(await setIsolated(true), false);
  });

  it("is asked once: a page that came back without its headers does not reload for them again", async () => {
    browser.serviceWorker = { register: async () => ({}), ready: Promise.resolve({}) };
    const first = await freshSettings("first-load");
    assert.equal(await first.setIsolated(true), true);
    const second = await freshSettings("second-load");
    assert.equal(second.cpuThreadsOnceIsolated(), 1);
    assert.equal(await second.setIsolated(true), false);
  });

  it("is asked again by a page it did give the headers to", async () => {
    browser.serviceWorker = { register: async () => ({}), ready: Promise.resolve({}) };
    const first = await freshSettings("isolated-first");
    assert.equal(await first.setIsolated(true), true);
    globalThis.window.crossOriginIsolated = true;
    const second = await freshSettings("isolated-second");
    assert.equal(second.cpuThreadsOnceIsolated(), 4);
    globalThis.window.crossOriginIsolated = false;
  });
});

describe("a browser that refuses the fold's lock", () => {
  it("does not take the fold with it", async () => {
    browser.locks = { request: () => Promise.reject(new DOMException("no", "SecurityError")) };
    const { beginFold } = await import("../web/fold-guard.js");
    const end = await beginFold({ signal: new AbortController().signal, onWaiting: () => assert.fail("nothing to wait for") });
    end();
  });

  it("still lets Stop end the wait for another tab", async () => {
    const controller = new AbortController();
    browser.locks = {
      request: (_, options, granted) => {
        if (options.ifAvailable) return Promise.resolve(granted(null));
        return new Promise((_, reject) => {
          options.signal.addEventListener("abort", () => reject(new DOMException("stopped", "AbortError")));
        });
      },
    };
    const { beginFold } = await import("../web/fold-guard.js");
    let waited = false;
    const fold = beginFold({ signal: controller.signal, onWaiting: () => { waited = true; } });
    await new Promise((resolve) => { realTimeout(resolve, 1); });
    assert.equal(waited, true);
    controller.abort();
    await assert.rejects(fold, { name: "AbortError" });
  });
});

describe("a CPU fold's worker", () => {
  const started = [];
  globalThis.Worker = class {
    constructor() { started.push(this); }

    postMessage() {}

    terminate() {}
  };
  const failing = async (error) => {
    const { foldEsmfold2OnCpu, releaseCpuFold } = await import("../web/esmfold2-cpu.js");
    releaseCpuFold();
    const fold = foldEsmfold2OnCpu({ family: "ef2-fast-300m" });
    started.at(-1).onmessage({ data: { type: "error", stack: "", ...error } });
    return fold.then(() => assert.fail("the fold resolved"), (caught) => caught);
  };

  it("keeps what kind of failure a failed download is, across the worker", async () => {
    const caught = await failing({ message: "failed to load tensor x: Failed to fetch", downloadFailed: "Failed to fetch" });
    assert.ok(caught instanceof ModelDownloadError);
    assert.equal(caught.reason, "Failed to fetch");
  });

  it("does not call anything else one", async () => {
    const caught = await failing({ message: "something else" });
    assert.equal(caught instanceof ModelDownloadError, false);
    assert.equal(caught.message, "something else");
  });
});
