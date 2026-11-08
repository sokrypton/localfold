import { describe, expect, it } from "./harness.js";
import { shardCacheName, staleShardCaches } from "../shared/bundles/http-tensor-store.js";

const manifest = (model, bytes, tensors) => ({
  bundle: { model, bytes },
  tensors: Object.fromEntries(Array.from({ length: tensors }, (_, i) => [`t${i}`, {}])),
});

const HUB = "https://huggingface.co/sokrypton/localfold/resolve";
const ESMC_600 = `${HUB}/71ece35/esmc-600m-int3/`;
const ESMC_300 = `${HUB}/71ece35/esmc-300m-int3/`;
const PTM_URL = `${HUB}/aaaaaaa/model-1-ptm-int5/`;
const MULTIMER_URL = `${HUB}/aaaaaaa/model-1-multimer-int5/`;

const esmc600 = manifest("esmc", undefined, 374);
const esmc300 = manifest("esmc", undefined, 314);
const ptm = manifest("model_1_ptm", 102055906, 335);
const multimer = manifest("model_1_multimer_v3", 102126424, 354);

describe("which cached shard sets a bundle supersedes", () => {
  it("keeps another model's cache, which is the whole point", () => {
    const names = [shardCacheName(ptm, PTM_URL), shardCacheName(multimer, MULTIMER_URL)];
    expect(staleShardCaches(names, multimer, MULTIMER_URL)).toEqual([]);
    expect(staleShardCaches(names, ptm, PTM_URL)).toEqual([]);
  });

  it("keeps a bundle sharing its model name, as both ESM-C towers do", () => {
    // The regression this exists for. Both manifests name their model "esmc", so
    // each swept the other and every switch between 300M and 600M downloaded.
    const names = [shardCacheName(esmc600, ESMC_600), shardCacheName(esmc300, ESMC_300)];
    expect(staleShardCaches(names, esmc300, ESMC_300)).toEqual([]);
    expect(staleShardCaches(names, esmc600, ESMC_600)).toEqual([]);
  });

  it("gives two bundles with the same size and tensor count different names", () => {
    // ef2-fast-300m and ef2-fast-600m report the same model, bytes and tensors.
    const trunk = manifest("esmfold2-trunk", 134034256, 953);
    expect(shardCacheName(trunk, `${HUB}/7f265c2/ef2-fast-300m-int5/`)
      === shardCacheName(trunk, `${HUB}/7f265c2/ef2-fast-600m-int5/`)).toEqual(false);
  });

  it("drops a previous export of the same bundle", () => {
    const stale = shardCacheName(manifest("model_1_ptm", 99000000, 335), PTM_URL);
    const names = [stale, shardCacheName(ptm, PTM_URL), shardCacheName(multimer, MULTIMER_URL)];
    expect(staleShardCaches(names, ptm, PTM_URL)).toEqual([stale]);
  });

  it("drops one whose tensor count changed, not only its size", () => {
    const stale = shardCacheName(manifest("model_1_multimer_v3", 102126424, 370), MULTIMER_URL);
    expect(staleShardCaches([stale, shardCacheName(multimer, MULTIMER_URL)], multimer, MULTIMER_URL))
      .toEqual([stale]);
  });

  it("drops the same export at an older revision, whose shard URLs are dead", () => {
    const stale = shardCacheName(ptm, `${HUB}/0000000/model-1-ptm-int5/`);
    expect(staleShardCaches([stale, shardCacheName(ptm, PTM_URL)], ptm, PTM_URL)).toEqual([stale]);
  });

  it("does not let one bundle claim another whose directory extends it", () => {
    const longer = shardCacheName(manifest("m", 1, 1), `${HUB}/aaaaaaa/model-int5-2/`);
    expect(staleShardCaches([longer], manifest("m", 2, 2), `${HUB}/aaaaaaa/model-int5/`)).toEqual([]);
  });

  it("names the same bundle from its base and from its manifest alike", () => {
    expect(staleShardCaches([shardCacheName(ptm, PTM_URL)], ptm, `${PTM_URL}manifest.json`).length)
      .toEqual(1);
  });

  it("drops caches named the old way, which nothing can read now", () => {
    const old = "localfold-model-esmc-0-314";
    expect(staleShardCaches([old], esmc300, ESMC_300)).toEqual([old]);
  });

  it("ignores caches that are not ours", () => {
    expect(staleShardCaches(["some-other-app-v1"], ptm, PTM_URL)).toEqual([]);
  });
});
