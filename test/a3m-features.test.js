import { readFile } from "node:fs/promises";
import { describe, expect, it } from "./harness.js";
import { makeA3mFeatures } from "../src/input/a3m-features.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";

describe("A3M model feature preprocessing", () => {
  it("threads physical-chain offsets into monomer-model A3M features", async() => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open("test/fixtures/evoformer/model1-query-59-stack/manifest.json"));
    const features = makeA3mFeatures(">query\nACDEF\n>hit\nAC-EF\n", await fixture.queryOnlyFeatureTables(), {
      recycles: 0, chainLengths: [2, 3], maxMsaSequences: 2, maxExtraSequences: 0,
    })[0];
    expect(Array.from(features.residueIndex)).toEqual([0, 1, 202, 203, 204]);
  });

  it("clusters the uploaded 8,076-row alignment into model-1 tensors", async() => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open("test/fixtures/evoformer/model1-query-59-stack/manifest.json"));
    const result = makeA3mFeatures(await readFile("test.a3m", "utf8"), await fixture.queryOnlyFeatureTables(), {
      recycles: 0, randomSeed: 0,
    });
    const features = result[0];
    expect(features.msaSequences).toBe(508);
    expect(features.extraSequences).toBe(1024);
    expect(features.msaFeatures.length).toBe(508 * 59 * 49);
    expect(features.extraMsa.length).toBe(1024 * 59);
    expect(features.msaMask.every((value) => value === 1)).toBe(true);
    expect(features.extraMsaMask.every((value) => value === 1)).toBe(true);
  }, 30_000);

  it("subsamples MSA depth to user-specified cluster and extra limits", async() => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open("test/fixtures/evoformer/model1-query-59-stack/manifest.json"));
    const text = await readFile("test.a3m", "utf8");
    const tables = await fixture.queryOnlyFeatureTables();
    const f256 = makeA3mFeatures(text, tables, { recycles: 0, maxMsaSequences: 256, maxExtraSequences: 512 })[0];
    expect(f256.msaSequences).toBe(256);
    expect(f256.extraSequences).toBe(512);
    expect(f256.msaFeatures.length).toBe(256 * 59 * 49);
    expect(f256.extraMsa.length).toBe(512 * 59);

    const f64 = makeA3mFeatures(text, tables, { recycles: 0, maxMsaSequences: 64, maxExtraSequences: 128 })[0];
    expect(f64.msaSequences).toBe(64);
    expect(f64.extraSequences).toBe(128);
    expect(f64.msaFeatures.length).toBe(64 * 59 * 49);
    expect(f64.extraMsa.length).toBe(128 * 59);
  }, 30_000);
});
