import assert from "node:assert/strict";
import { afterEach, describe, it } from "./harness.js";

// The module asks once a page, so each test imports a page of its own.
const freshPage = async (name) => (await import(`../web/chrome-benchmark.js?${name}`)).settleBrowserBenchmark;

describe("Chrome's benchmark, before the first device", () => {
  afterEach(() => { delete globalThis.Summarizer; });

  it("asks once a page, however many devices follow", async () => {
    let asked = 0;
    globalThis.Summarizer = { availability: async () => { asked += 1; return "downloadable"; } };
    const settle = await freshPage("once");
    await settle();
    await settle();
    assert.equal(asked, 1);
  });

  it("waits for the answer", async () => {
    let answer;
    globalThis.Summarizer = { availability: () => new Promise((done) => { answer = done; }) };
    const settle = await freshPage("waits");
    let settled = false;
    const waiting = settle().then(() => { settled = true; });
    await new Promise((done) => setTimeout(done, 20));
    assert.equal(settled, false);
    answer("downloadable");
    await waiting;
    assert.equal(settled, true);
  });

  it("has nothing to wait for in a browser without the API", async () => {
    const settle = await freshPage("absent");
    await settle();
  });

  it("is not held up by an API that refuses, or that throws where it is called", async () => {
    const held = console.warn;
    console.warn = () => {};
    try {
      globalThis.Summarizer = { availability: () => Promise.reject(new Error("blocked by policy")) };
      await (await freshPage("refuses"))();
      globalThis.Summarizer = { availability: () => { throw new Error("not in this context"); } };
      await (await freshPage("throws"))();
    } finally {
      console.warn = held;
    }
  });

  it("is not held up by an API that never answers", async () => {
    globalThis.Summarizer = { availability: () => new Promise(() => {}) };
    const settle = await freshPage("never");
    await settle(10);
  });
});
