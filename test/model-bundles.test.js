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
import { MODEL_BUNDLES, bundleBaseUrl } from "../src/bundles/manifests/index.js";

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

  /**
   * 🔴 `noTemplateEmbedder` IS A CLAIM ABOUT THE WEIGHTS AND THE WEIGHTS CAN
   * SETTLE IT. AlphaFold 2's models 3, 4 and 5 carry no template embedder, and
   * the page refuses a template under them on the strength of that flag - so a
   * flag that disagrees with the bundle is a page that either drops a template
   * silently or refuses one it could have used. The delta's own manifest lists
   * every tensor the checkpoint does not have; this holds the two together.
   */
  it("flags a template-free model exactly when its bundle has no template tensors", async() => {
    for (const [family, bundle] of Object.entries(MODEL_BUNDLES)) {
      if (bundle.delta === undefined) continue;
      const base = MODEL_BUNDLES[bundle.delta.base];
      const baseManifest = (await base.load()).MANIFEST;
      const templateTensors = Object.values(baseManifest.templateEmbedding?.parameters ?? {})
        .flatMap((leaves) => Object.values(leaves));
      expect(templateTensors.length > 0).toBe(true);
      const absent = new Set((await bundle.load()).MANIFEST.delta.absent);
      const none = templateTensors.every((name) => absent.has(name));
      const some = templateTensors.some((name) => absent.has(name));
      // Partly absent is a packing fault, not a model - the reader refuses it
      // too, and this says so where the bundles are described.
      expect(some).toBe(none);
      expect(bundle.noTemplateEmbedder === true).toBe(none);
    }
  });
});
