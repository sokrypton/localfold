import { afterEach, describe, expect, it, vi } from "./harness.js";
import { HttpTensorStore } from "../src/reference/http-tensor-store.js";

afterEach(() => vi.unstubAllGlobals());

describe("HttpTensorStore", () => {
  it("bounds concurrent tensor downloads", async() => {
    const tensors = Object.fromEntries(Array.from({ length: 20 }, (_, index) => [`tensor${index}`, {
      file: `tensor${index}.f32.bin`, dtype: "float32", shape: [1],
    }]));
    let active = 0; let maximum = 0;
    vi.stubGlobal("fetch", vi.fn(async(input) => {
      const url = String(input);
      if (url.endsWith("manifest.json")) return new Response(JSON.stringify({ tensors }));
      active += 1; maximum = Math.max(maximum, active);
      await new Promise((resolve) => setTimeout(resolve, 2));
      active -= 1;
      return new Response(Float32Array.of(1));
    }));
    const progress = [];
    const store = await HttpTensorStore.open(new URL("https://example.test/model/manifest.json"), (value) => {
      progress.push({ loadedBytes: value.loadedBytes, loadedTensors: value.loadedTensors });
    });
    await Promise.all(Object.keys(tensors).map((name) => store.tensor(name)));
    expect(maximum).toBe(8);
    expect(progress.at(-1)).toEqual({ loadedBytes: 80, loadedTensors: 20 });
  });

  it("retries transient tensor responses", async() => {
    let attempts = 0;
    vi.stubGlobal("fetch", vi.fn(async(input) => {
      if (String(input).endsWith("manifest.json")) return new Response(JSON.stringify({
        tensors: { value: { file: "value.f32.bin", dtype: "float32", shape: [1] } },
      }));
      attempts += 1;
      return attempts === 1 ? new Response(null, { status: 503 }) : new Response(Float32Array.of(7));
    }));
    const store = await HttpTensorStore.open(new URL("https://example.test/model/manifest.json"));
    expect(Array.from(await store.tensor("value"))).toEqual([7]);
    expect(attempts).toBe(2);
  });

  it("downloads a shared shard once and returns tensor views at byte offsets", async() => {
    let downloads = 0;
    const shard = new Float32Array([1, 2, 3, 4]);
    vi.stubGlobal("fetch", vi.fn(async(input) => {
      if (String(input).endsWith("manifest.json")) return new Response(JSON.stringify({ tensors: {
        first: { file: "weights.bin", dtype: "float32", shape: [2], byteOffset: 0 },
        second: { file: "weights.bin", dtype: "float32", shape: [2], byteOffset: 8 },
      } }));
      downloads += 1;
      // CDNs may report the compressed transfer length while fetch exposes decoded bytes.
      return new Response(shard, { headers: { "content-encoding": "gzip", "content-length": "2" } });
    }));
    const store = await HttpTensorStore.open(new URL("https://example.test/model/manifest.json"));
    const [first, second] = await Promise.all([store.tensor("first"), store.tensor("second")]);
    expect(Array.from(first)).toEqual([1, 2]);
    expect(Array.from(second)).toEqual([3, 4]);
    expect(downloads).toBe(1);
  });

  it("loads tensors directly from manifest object with fromManifest without fetching manifest URL", async() => {
    const shard = new Float32Array([10, 20]);
    vi.stubGlobal("fetch", vi.fn(async(input) => {
      const url = String(input);
      if (url.endsWith("manifest.json")) throw new Error("should not fetch manifest.json");
      return new Response(shard);
    }));
    const manifest = {
      formatVersion: 1,
      tensors: {
        item: { file: "weights.bin", dtype: "float32", shape: [2], byteOffset: 0 },
      },
    };
    const store = await HttpTensorStore.fromManifest("https://example.test/model/", manifest);
    const item = await store.tensor("item");
    expect(Array.from(item)).toEqual([10, 20]);
  });
});

describe("prefetch ordering", () => {
  // 🔴 THE LAST SHARD TO START DECIDES WHEN THE LOAD ENDS, and the shards are
  // not evenly sized: a shard is at least one whole tensor, so AF3's stacked
  // single-transition weights make a 40.5 MiB shard against a 7.9 MiB median.
  // Started last it runs on alone after the other connections have nothing
  // left to do; started first the small ones fill in around it.
  it("asks for the biggest shard first", async () => {
    const asked = [];
    const bytes = { "small.bin": 16, "mid.bin": 1024, "huge.bin": 16384 };
    const manifest = {
      tensors: {
        small: { file: "small.bin", dtype: "float32", shape: [4] },
        huge: { file: "huge.bin", dtype: "float32", shape: [4096] },
        middling: { file: "mid.bin", dtype: "float32", shape: [256] },
      },
    };
    const original = globalThis.fetch;
    // A complete response for each, so every promise settles and nothing leaks
    // past the end of the test.
    globalThis.fetch = (url) => {
      const file = String(url).split("/").pop().split("?")[0];
      asked.push(file);
      return Promise.resolve(new Response(new ArrayBuffer(bytes[file] ?? 0), { status: 200 }));
    };
    try {
      const store = await HttpTensorStore.fromManifest("./m/", manifest, undefined, "");
      asked.length = 0;                       // ignore anything the open itself did
      store.prefetch();
      await store.tensor("huge");             // let the queue drain
      await store.tensor("small");
      await store.tensor("middling");
    } finally {
      globalThis.fetch = original;
    }
    expect(asked.slice(0, 3)).toEqual(["huge.bin", "mid.bin", "small.bin"]);
  });
});
