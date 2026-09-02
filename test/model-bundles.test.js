/**
 * Where a model's shards are fetched from.
 *
 * 🔴 THIS IS ONE STRING AND IT DECIDES WHETHER 150 MB ARRIVES. Shard URLs are
 * resolved with `new URL(file, base)`, which is unforgiving in exactly one way:
 * a base without a trailing slash loses its last segment, so
 * ".../resolve/abc123" puts the shard beside `abc123` instead of inside it and
 * the fold dies on a 404 naming a path that looks almost right.
 */
import { describe, expect, it } from "./harness.js";
import { MODEL_BUNDLES, bundleBaseUrl } from "../src/reference/manifests/index.js";

const shardUrl = (family, file) =>
  new URL(file, new URL(bundleBaseUrl(family), "https://localfold.org/index.html")).href;

describe("where a bundle's shards come from", () => {
  it("gives every family a base that ends in a slash", () => {
    for (const family of Object.keys(MODEL_BUNDLES)) {
      expect(bundleBaseUrl(family).endsWith("/")).toBe(true);
    }
  });

  it("resolves a shard inside the base, not beside it", () => {
    for (const family of Object.keys(MODEL_BUNDLES)) {
      const url = shardUrl(family, "weights.0.bin");
      expect(url.endsWith("/weights.0.bin")).toBe(true);
      // The base has to survive whole: a lost last segment is the failure this
      // is here to catch, and it looks like a working URL.
      const base = bundleBaseUrl(family).replace(/^\.\//, "");
      const tail = base.replace(/\/$/, "").split("/").pop();
      expect(url.includes(`/${tail}/weights.0.bin`)).toBe(true);
    }
  });

  it("takes a remote over the directory beside the page", () => {
    const before = MODEL_BUNDLES.af3.remote;
    try {
      MODEL_BUNDLES.af3.remote = "https://huggingface.co/o/r/resolve/abc123";
      expect(bundleBaseUrl("af3")).toBe("https://huggingface.co/o/r/resolve/abc123/");
      expect(shardUrl("af3", "af3.0.bin"))
        .toBe("https://huggingface.co/o/r/resolve/abc123/af3.0.bin");
    } finally {
      if (before === undefined) delete MODEL_BUNDLES.af3.remote;
      else MODEL_BUNDLES.af3.remote = before;
    }
  });

  it("names a family it does not have", () => {
    expect(() => bundleBaseUrl("nonesuch")).toThrow(/unknown model family/);
  });
});
