import { describe, expect, it } from "./harness.js";
import { staleShardCaches } from "../src/reference/http-tensor-store.js";

const manifest = (model, bytes, tensors) => ({
  bundle: { model, bytes },
  tensors: Object.fromEntries(Array.from({ length: tensors }, (_, i) => [`t${i}`, {}])),
});

const PTM = "localfold-model-model_1_ptm-102055906-335";
const MULTIMER = "localfold-model-model_1_multimer_v3-102126424-354";

describe("which cached shard sets a manifest supersedes", () => {
  it("keeps another model's cache, which is the whole point", () => {
    // 🔴 THE REGRESSION THIS EXISTS FOR. The sweep used to drop every cache
    // that was not the one being opened, so loading one family deleted the
    // other's 97 MiB and every switch paid a full cold download.
    expect(staleShardCaches([PTM, MULTIMER], manifest("model_1_multimer_v3", 102126424, 354)))
      .toEqual([]);
    expect(staleShardCaches([PTM, MULTIMER], manifest("model_1_ptm", 102055906, 335)))
      .toEqual([]);
  });

  it("drops a previous export of the same model", () => {
    const stale = "localfold-model-model_1_ptm-99000000-335";
    expect(staleShardCaches([stale, PTM, MULTIMER], manifest("model_1_ptm", 102055906, 335)))
      .toEqual([stale]);
  });

  it("drops one whose tensor count changed, not only its size", () => {
    const stale = "localfold-model-model_1_multimer_v3-102126424-370";
    expect(staleShardCaches([stale, MULTIMER], manifest("model_1_multimer_v3", 102126424, 354)))
      .toEqual([stale]);
  });

  it("does not let one model claim another whose name extends it", () => {
    // `model_1` is a prefix of `model_1_multimer_v3`; the trailing "-" is what
    // keeps the first from sweeping the second.
    const short = "localfold-model-model_1-1-1";
    expect(staleShardCaches([short, MULTIMER], manifest("model_1_multimer_v3", 102126424, 354)))
      .toEqual([]);
    expect(staleShardCaches([short, MULTIMER], manifest("model_1", 2, 1))).toEqual([short]);
  });

  it("ignores caches that are not ours", () => {
    expect(staleShardCaches(["some-other-app-v1", PTM], manifest("model_1_ptm", 1, 1)))
      .toEqual([PTM]);
    expect(staleShardCaches(["some-other-app-v1"], manifest("model_1_ptm", 1, 1))).toEqual([]);
  });
});
