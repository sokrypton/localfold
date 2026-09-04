import { describe, expect, it } from "./harness.js";
import { MAX_ALANINE, objectiveOf, runDesign, withChain } from "../src/design/hunter-loop.js";

/**
 * A stub pair standing in for a GPU fold and an MPNN design.
 *
 * The loop takes both as arguments precisely so this file can exist: every
 * schedule below is checked without a device, which matters because
 * `npm run test:gpu` cannot load Dawn on this machine. What the stubs record -
 * the omissions and bias each cycle was given, and the pdb each design saw -
 * is the loop's contract with the two models it drives.
 */
function stubs({ scores = [], sequences = [] } = {}) {
  const folds = [];
  const designs = [];
  return {
    folds,
    designs,
    fold: async (complex, context) => {
      folds.push({ complex, ...context });
      const score = scores[folds.length - 1] ?? 0.5;
      return {
        pdb: `PDB-${folds.length - 1}`,
        confidence: {
          meanPlddt: score * 100,
          ptm: score,
          iptm: score,
          chainPairIptm: { "1|2": score },
        },
      };
    },
    design: async (pdb, context) => {
      designs.push({ pdb, ...context });
      return { sequence: sequences[designs.length - 1] ?? "GGGGG", score: -1.5 };
    },
  };
}

const collect = async (options) => {
  const records = [];
  for await (const record of runDesign(options)) records.push(record);
  return records;
};

const twoChains = (extra = {}) => ({
  chains: ["", "TARGETSEQ"], chainIndex: 0, length: 5, percentX: 100, seed: 1, ...extra,
});

describe("runDesign", () => {
  it("folds once more than it designs: cycle 0 has no design step", async () => {
    const stub = stubs();
    const records = await collect({ ...twoChains(), cycles: 3, ...stub });
    expect(records).toHaveLength(4);
    expect(stub.folds).toHaveLength(4);
    expect(stub.designs).toHaveLength(3);
    expect(records.map((record) => record.cycle)).toEqual([0, 1, 2, 3]);
  });

  it("starts from a mostly-unknown sequence and folds it in place", async () => {
    const stub = stubs();
    const [first] = await collect({ ...twoChains(), cycles: 1, ...stub });
    expect(first.sequence).toBe("XXXXX");
    expect(first.complex).toBe("XXXXX:TARGETSEQ");
    expect(stub.folds[0].complex).toBe("XXXXX:TARGETSEQ");
  });

  it("keeps a starting sequence it is given instead of drawing one", async () => {
    const stub = stubs();
    const [first] = await collect({
      ...twoChains({ chains: ["MKVLW", "TARGETSEQ"] }), cycles: 0, ...stub,
    });
    expect(first.sequence).toBe("MKVLW");
  });

  // The loop's whole point: each design reads the structure the previous fold
  // produced, not the one before it and not the starting sequence.
  it("designs off the previous cycle's structure", async () => {
    const stub = stubs();
    await collect({ ...twoChains(), cycles: 3, ...stub });
    expect(stub.designs.map((design) => design.pdb)).toEqual(["PDB-0", "PDB-1", "PDB-2"]);
  });

  it("omits proline on the first design cycle only", async () => {
    const stub = stubs();
    await collect({ ...twoChains(), cycles: 3, ...stub });
    expect([...stub.designs[0].omit].sort().join("")).toBe("CP");
    expect(stub.designs[1].omit).toBe("C");
    expect(stub.designs[2].omit).toBe("C");
  });

  it("ramps the alanine bias only when it is switched on", async () => {
    const off = stubs();
    await collect({ ...twoChains(), cycles: 3, ...off });
    expect(off.designs.map((design) => design.alanineBias)).toEqual([0, 0, 0]);

    const on = stubs();
    await collect({ ...twoChains(), cycles: 3, alanineBias: true, ...on });
    const ramp = on.designs.map((design) => Number(design.alanineBias.toFixed(4)));
    expect(ramp).toEqual([-0.5, -0.3, -0.1]);
  });

  it("marks exactly one cycle best, and it is the highest scoring one", async () => {
    const stub = stubs({ scores: [0.2, 0.9, 0.4, 0.5] });
    const records = await collect({ ...twoChains(), cycles: 3, ...stub });
    expect(records.filter((record) => record.best)).toHaveLength(1);
    expect(records.find((record) => record.best).cycle).toBe(1);
  });

  // Cycle 0's sequence is the mostly-X draw the run started from, so naming it
  // best would report the question as the answer. Found by a real run: a
  // 20-mer of pure X folded to pLDDT 72.9 and won.
  it("never names cycle 0 best, however well it folds", async () => {
    const records = await collect({
      ...twoChains(), cycles: 2, ...stubs({ scores: [0.99, 0.2, 0.3] }),
    });
    expect(records[0].best).toBe(false);
    expect(records.find((record) => record.best).cycle).toBe(2);
  });

  it("names nothing best when every design cycle is over the ceiling", async () => {
    const records = await collect({
      ...twoChains(), cycles: 2, ...stubs({ sequences: ["AAAAA", "AAAAG"] }),
    });
    expect(records.some((record) => record.best)).toBe(false);
  });

  // The guard is on winning, not on continuing: a poly-alanine cycle still
  // runs, is still designed off, and is still shown - it just cannot be the
  // answer, however well it scores.
  it("refuses a cycle over the alanine ceiling as the answer, but still runs it", async () => {
    const stub = stubs({
      scores: [0.2, 0.99, 0.4],
      sequences: ["AAAAA", "GGKLW"],
    });
    const records = await collect({ ...twoChains(), cycles: 2, ...stub });
    expect(records[1].sequence).toBe("AAAAA");
    expect(records[1].alanine).toBeGreaterThan(MAX_ALANINE);
    expect(records[1].best).toBe(false);
    // ...and the run went on to design off it.
    expect(stub.designs[1].pdb).toBe("PDB-1");
    expect(records.find((record) => record.best).cycle).toBe(2);
  });

  it("selects on the designed chain's interface for a complex", async () => {
    const records = await collect({ ...twoChains(), cycles: 1, ...stubs({ scores: [0.3, 0.8] }) });
    expect(records[0].objective).toBe("iptm");
    expect(records[0].score).toBeCloseTo(0.3, 6);
  });

  // Protein Hunter has no monomer path - it reads pair_chains_iptm and takes
  // 0.0 with one chain, so every cycle would tie. pLDDT is what a monomer has.
  it("selects on pLDDT for a monomer, where there is no interface", async () => {
    const stub = stubs();
    stub.fold = async (complex, context) => {
      stub.folds.push({ complex, ...context });
      const value = [0.4, 0.7][stub.folds.length - 1];
      return {
        pdb: `PDB-${stub.folds.length - 1}`,
        confidence: { meanPlddt: value * 100, ptm: value, iptm: Number.NaN, chainPairIptm: {} },
      };
    };
    const records = await collect({
      chains: [""], chainIndex: 0, length: 5, percentX: 100, seed: 1, cycles: 1, ...stub,
    });
    expect(records[0].objective).toBe("plddt");
    expect(records[0].score).toBeCloseTo(0.4, 6);
    expect(records[1].best).toBe(true);
  });

  it("is reproducible from a seed and different without one", async () => {
    const one = await collect({ ...twoChains({ seed: 5, percentX: 40 }), cycles: 0, ...stubs() });
    const two = await collect({ ...twoChains({ seed: 5, percentX: 40 }), cycles: 0, ...stubs() });
    const other = await collect({ ...twoChains({ seed: 6, percentX: 40 }), cycles: 0, ...stubs() });
    expect(one[0].sequence).toBe(two[0].sequence);
    expect(one[0].sequence === other[0].sequence).toBe(false);
  });

  it("draws a length from the range when it is given neither length nor sequence", async () => {
    const records = await collect({
      chains: ["", "TARGET"], minLength: 8, maxLength: 12, percentX: 100, seed: 2,
      cycles: 0, ...stubs(),
    });
    expect(records[0].sequence.length >= 8 && records[0].sequence.length <= 12).toBe(true);
  });

  it("stops at an abort, keeping the cycles it had already yielded", async () => {
    const controller = new AbortController();
    const stub = stubs();
    const records = [];
    let thrown = null;
    try {
      for await (const record of runDesign({
        ...twoChains(), cycles: 5, signal: controller.signal, ...stub,
      })) {
        records.push(record);
        if (records.length === 2) controller.abort();
      }
    } catch (error) {
      thrown = error;
    }
    expect(thrown?.name).toBe("AbortError");
    expect(records).toHaveLength(2);
    // The abort landed before the third fold was asked for.
    expect(stub.folds).toHaveLength(2);
  });

  it("rejects a chain index outside the complex", async () => {
    let thrown = null;
    try {
      await collect({ chains: ["MK"], chainIndex: 3, cycles: 0, ...stubs() });
    } catch (error) {
      thrown = error;
    }
    expect(thrown instanceof RangeError).toBe(true);
  });
});

describe("withChain", () => {
  it("replaces one chain and joins with colons", () => {
    expect(withChain(["AAA", "BBB", "CCC"], 1, "ZZ")).toBe("AAA:ZZ:CCC");
  });

  it("leaves a single chain uncolonised", () => {
    expect(withChain(["AAA"], 0, "ZZ")).toBe("ZZ");
  });
});

describe("objectiveOf", () => {
  // chainPairTmScores writes `${asymId}|${asymId}` and toPdb turns asym n into
  // letter n-1, so chain index 0 is asym "1". Matching on letters found no
  // interface at all and silently fell through to pLDDT.
  it("reads the pair keys as asym ids, not chain letters", () => {
    const result = objectiveOf({ meanPlddt: 90, iptm: 0.1, chainPairIptm: { "1|2": 0.7 } }, 0);
    expect(result).toEqual({ value: 0.7, objective: "iptm" });
  });

  it("averages only the interfaces the designed chain is in", () => {
    const pairs = { "1|2": 0.2, "1|3": 0.4, "2|3": 0.9 };
    expect(objectiveOf({ meanPlddt: 90, iptm: 0.5, chainPairIptm: pairs }, 0).value)
      .toBeCloseTo(0.3, 6);
    expect(objectiveOf({ meanPlddt: 90, iptm: 0.5, chainPairIptm: pairs }, 2).value)
      .toBeCloseTo(0.65, 6);
  });

  it("falls back to pLDDT when there is no interface at all", () => {
    expect(objectiveOf({ meanPlddt: 82, iptm: Number.NaN, chainPairIptm: {} }, 0))
      .toEqual({ value: 0.82, objective: "plddt" });
  });
});
