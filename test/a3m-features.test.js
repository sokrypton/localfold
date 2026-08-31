import { readFile } from "node:fs/promises";
import { describe, expect, it } from "./harness.js";
import { makeA3mFeatures, makeComplexA3mFeatures } from "../src/input/a3m-features.js";
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


describe("complex features built per chain", () => {
  const tables = {
    atom37ToAtom14: new Float32Array(37 * 21),
    atom37Mask: new Float32Array(37 * 21).fill(1),
  };
  const chainA = ">q\nACDEFG\n>h1\nACDEFH\n>h2\nSCDEFG\n>h3\nACDWFG\n>h4\nAQDEFG\n";
  const chainB = ">q\nMKLP\n>k1\nMKLQ\n>k2\nWKLP\n";

  it("gives every copy the full cluster budget", () => {
    const [features] = makeComplexA3mFeatures([chainA, chainA], tables, { recycles: 0 });
    // 5 rows in chainA, so each copy keeps all 5 - not 5 shared between them.
    expect(features.msaSequences).toBe(5);
    expect(features.msaFeatures.length).toBe(5 * 12 * 49);
  });

  it("breaks the symmetry between identical copies", () => {
    const [features] = makeComplexA3mFeatures([chainA, chainA], tables, { recycles: 0 });
    const width = 12, span = 6, CH = 49;
    let identical = 0;
    for (let row = 0; row < features.msaSequences; row += 1) {
      let same = true;
      for (let i = 0; i < span && same; i += 1) {
        for (let c = 0; c < CH; c += 1) {
          if (features.msaFeatures[(row * width + i) * CH + c]
            !== features.msaFeatures[(row * width + span + i) * CH + c]) { same = false; break; }
        }
      }
      if (same) identical += 1;
    }
    // The query row may coincide; the sampled rows must not all line up.
    expect(identical).toBeLessThan(features.msaSequences);
  });

  it("is reproducible for a given seed", () => {
    const a = makeComplexA3mFeatures([chainA, chainA], tables, { recycles: 0, randomSeed: 7 })[0];
    const b = makeComplexA3mFeatures([chainA, chainA], tables, { recycles: 0, randomSeed: 7 })[0];
    expect(Array.from(a.msaFeatures)).toEqual(Array.from(b.msaFeatures));
  });

  it("changes with the seed", () => {
    const a = makeComplexA3mFeatures([chainA, chainA], tables, { recycles: 0, randomSeed: 1 })[0];
    const b = makeComplexA3mFeatures([chainA, chainA], tables, { recycles: 0, randomSeed: 2 })[0];
    const identical = a.msaFeatures.every((value, index) => value === b.msaFeatures[index]);
    expect(identical).toBe(false);
  });

  it("pads the shallower chain with gaps rather than truncating the deeper one", () => {
    const [features] = makeComplexA3mFeatures([chainA, chainB], tables, { recycles: 0 });
    expect(features.msaSequences).toBe(5);
    const width = 10, CH = 49, GAP = 21;
    // chainB has 3 rows, so rows 3 and 4 are all gap across its 4 residues.
    for (const row of [3, 4]) {
      for (let i = 6; i < 10; i += 1) {
        expect(features.msaFeatures[(row * width + i) * CH + GAP]).toBe(1);
      }
    }
  });

  it("carries the chain-break offsets in the residue index", () => {
    const [features] = makeComplexA3mFeatures([chainA, chainB], tables, { recycles: 0 });
    expect(features.residueIndex[0]).toBe(0);
    expect(features.residueIndex[5]).toBe(5);
    expect(features.residueIndex[6]).toBe(6 + 200);
  });

  it("is what makeA3mFeatures returns for an array", () => {
    const direct = makeComplexA3mFeatures([chainA, chainB], tables, { recycles: 0, randomSeed: 3 })[0];
    const viaDispatch = makeA3mFeatures([chainA, chainB], tables, { recycles: 0, randomSeed: 3 })[0];
    expect(Array.from(viaDispatch.msaFeatures)).toEqual(Array.from(direct.msaFeatures));
  });
});
