import { describe, expect, it } from "./harness.js";
import {
  buildFoldArchive, fullDataJson, jobRequestJson, msasFromArchive,
  summaryConfidencesJson, tokenIdentifiers,
} from "../web/fold-archive.js";
import { readZip, writeZip } from "../web/zip.js";
import { mergeSearchedChains } from "../src/input/mmseqs2-api.js";

// Two residues per chain, one atom each, so the atom arrays are checkable by eye.
function pdb(chains) {
  const lines = [];
  let serial = 1;
  chains.forEach((length, chain) => {
    const id = "AB"[chain];
    for (let within = 0; within < length; within += 1) {
      lines.push(`ATOM  ${String(serial).padStart(5)}  CA  ALA ${id}`
        + `${String(within + 1).padStart(4)}    `
        + `${(serial).toFixed(3).padStart(8)}${(0).toFixed(3).padStart(8)}`
        + `${(0).toFixed(3).padStart(8)}  1.00${(70 + serial).toFixed(2).padStart(6)}`
        + "           C");
      serial += 1;
    }
  });
  lines.push("END");
  return `${lines.join("\n")}\n`;
}

const square = (tokens, value) =>
  Float32Array.from({ length: tokens * tokens }, (_, index) => value(
    Math.floor(index / tokens), index % tokens));

function prediction(chainLengths = [2, 2]) {
  const tokens = chainLengths.reduce((total, length) => total + length, 0);
  return {
    pdb: pdb(chainLengths),
    chainLengths,
    confidence: {
      plddt: Float32Array.from({ length: tokens }, (_, index) => (index === 0 ? 30 : 88)),
      meanPlddt: 74.5,
      ptm: 0.8123,
      iptm: 0.5567,
      chainPairIptm: { "0|1": 0.5567 },
      predictedAlignedError: square(tokens, (i, j) => Math.abs(i - j) + 0.456),
      maxPredictedAlignedError: 31.5,
      contactProbs: square(tokens, (i, j) => (i === j ? 1 : 0.2)),
    },
  };
}

describe("the fold archive", () => {
  const entities = [
    { type: "protein", value: "ACDE", copies: 2, template: { kind: "pdb", source: "1abc" } },
    { type: "ligand", value: "hem", copies: 1 },
  ];

  it("writes the members the AlphaFold 3 server writes", () => {
    const files = buildFoldArchive({
      stem: "fold_test", model: "AlphaFold 3", settings: { seed: 7 },
      entities, prediction: prediction(),
      msas: { unpaired: ["u-a", "u-b"], paired: ["p-a", "p-b"] },
    });
    expect([...files.keys()].sort()).toEqual([
      "README.md",
      "fold_test_full_data_0.json",
      "fold_test_job_request.json",
      "fold_test_model_0.pdb",
      "fold_test_summary_confidences_0.json",
      "msas/fold_test_paired_msa_chains_a.a3m",
      "msas/fold_test_paired_msa_chains_b.a3m",
      "msas/fold_test_unpaired_msa_chains_a.a3m",
      "msas/fold_test_unpaired_msa_chains_b.a3m",
    ]);
  });

  // 🔴 A COUNT, NOT REPEATED ENTRIES. expandEntities turns two copies into two
  // chains because that is what the model is handed; the request that produced
  // the job says `count: 2` on one entry, and a file listing the sequence twice
  // describes a different job.
  it("keeps copies as a count in the request", () => {
    const request = JSON.parse(jobRequestJson({ name: "j", seed: 7, entities }));
    expect(request[0].sequences).toHaveLength(2);
    expect(request[0].sequences[0].proteinChain.count).toBe(2);
    expect(request[0].sequences[0].proteinChain.useStructureTemplate).toBe(true);
    expect(request[0].sequences[1].ligand.ligand).toBe("HEM");
    expect(request[0].modelSeeds).toEqual(["7"]);
    expect(request[0].dialect).toBe("alphafoldserver");
  });

  it("nests the contacts exactly as it nests the PAE", () => {
    const data = JSON.parse(fullDataJson({
      ...prediction(), tokenChainIds: ["A", "A", "B", "B"],
      tokenResIds: [1, 2, 1, 2], confidence: prediction().confidence,
    }));
    expect(data.contact_probs).toHaveLength(4);
    expect(data.contact_probs[0]).toEqual([1, 0.2, 0.2, 0.2]);
    // ...and both rounded the way the server rounds, which is also what keeps
    // a float32 0.2 from being written as 0.20000000298023224.
    expect(data.pae[0]).toEqual([0.46, 1.46, 2.46, 3.46]);
    expect(data.atom_plddts).toEqual([71, 72, 73, 74]);
    expect(data.atom_chain_ids).toEqual(["A", "A", "B", "B"]);
  });

  // 🔴 AN UNSCORED INTERFACE IS null, NOT 0. Zero reads as "the model looked and
  // was sure it is bad"; the pair map simply omits a pair it could not score.
  it("writes the chain-pair matrix with the diagonal absent", () => {
    const summary = JSON.parse(summaryConfidencesJson({
      confidence: prediction().confidence, chainLengths: [2, 2],
      tokenChainIds: ["A", "A", "B", "B"],
    }));
    expect(summary.chain_pair_iptm).toEqual([[null, 0.56], [0.56, null]]);
    expect(summary.iptm).toBe(0.56);
    expect(summary.ptm).toBe(0.81);
    // One of four residues below pLDDT 50.
    expect(summary.fraction_disordered).toBe(0.25);
    // 🔴 A CLASH IS A CLAIM ABOUT GEOMETRY WE DO NOT MAKE. Every other field the
    // server writes is computed here from what the model produced; `has_clash`
    // would have to be invented, and a zero reads as "checked, and clean".
    expect("has_clash" in summary).toBe(false);
  });

  // 🔴 A MINIMUM OVER ORDERED PAIRS, WHICH IS WHY IT IS NOT SYMMETRIC. The
  // server's own file has [[0.76, 0.83], [0.82, 0.76]] - the diagonal is real
  // and the two off-diagonal entries differ - so a symmetric matrix here would
  // have been a plausible-looking wrong reading of the format.
  it("takes the chain-pair PAE minimum over ordered pairs", () => {
    const tokens = 4;
    // Chain A is tokens 0-1, chain B is 2-3. Make A->B closer than B->A.
    const pae = new Float32Array(tokens * tokens).fill(9);
    for (let i = 0; i < tokens; i += 1) pae[i * tokens + i] = 0.5;
    pae[0 * tokens + 2] = 1.25;
    pae[2 * tokens + 0] = 3.75;
    const summary = JSON.parse(summaryConfidencesJson({
      confidence: { predictedAlignedError: pae },
      chainLengths: [2, 2], tokenChainIds: ["A", "A", "B", "B"],
    }));
    expect(summary.chain_pair_pae_min).toEqual([[0.5, 1.25], [3.75, 0.5]]);
  });

  it("writes each chain's own pTM and its ipTM against the rest", () => {
    const summary = JSON.parse(summaryConfidencesJson({
      confidence: {
        ...prediction().confidence,
        chainPtm: { 0: 0.9012, 1: 0.4011 },
        chainIptm: { 0: 0.5567, 1: 0.5567 },
      },
      chainLengths: [2, 2], tokenChainIds: ["A", "A", "B", "B"],
    }));
    expect(summary.chain_ptm).toEqual([0.9, 0.4]);
    expect(summary.chain_iptm).toEqual([0.56, 0.56]);
  });

  it("refuses to infer a token layout it cannot know", () => {
    // A ligand makes AF3's token count exceed the residues, and numbering the
    // extra tokens as polymer would put them on the last chain.
    expect(() => tokenIdentifiers([2, 2], 9)).toThrow();
    expect(tokenIdentifiers([2, 2], 9, {
      chainIds: Array(9).fill("A"), resIds: Array(9).fill(1),
    }).chainIds).toHaveLength(9);
  });

  it("refuses to name more chains than it can spell", () => {
    expect(() => buildFoldArchive({
      stem: "x", entities, prediction: prediction(Array(27).fill(1)),
    })).toThrow();
  });
});

describe("the alignment round trip", () => {
  // Two chains whose paired and unpaired blocks are DIFFERENT, which is the
  // only way the test can tell them apart on the way back.
  const chains = ["ACDEFG", "MKVLAA"];
  const unpaired = [
    ">query\nACDEFG\n>u1\nACDEFA\n>u2\nACDEFC\n",
    ">query\nMKVLAA\n>u3\nMKVLAC\n",
  ];
  const paired = new Map([
    [chains[0], ">query\nACDEFG\n>p1 OX=9\nACDEFF\n"],
    [chains[1], ">query\nMKVLAA\n>p1 OX=9\nMKVLAF\n"],
  ]);

  it("comes back out of the archive as the same blocks that went in", async () => {
    const files = buildFoldArchive({
      stem: "fold_rt", model: "AlphaFold 3", settings: { seed: 1 },
      entities: chains.map((value) => ({ type: "protein", value, copies: 1 })),
      prediction: prediction([6, 6]),
      msas: { unpaired, paired: chains.map((chain) => paired.get(chain)) },
    });
    const restored = msasFromArchive(await readZip(await writeZip(files)));
    expect(restored.chainA3ms).toEqual(unpaired);
    expect(restored.pairedA3ms.get(0)).toBe(paired.get(chains[0]));
    expect(restored.pairedA3ms.get(1)).toBe(paired.get(chains[1]));
  });

  // 🔴 THE ASSERTION THE WHOLE FEATURE EXISTS FOR. Before this, an uploaded
  // alignment was recorded as unpaired in its entirety - AlphaFold 3 reads the
  // paired block first and takes its profile over the unpaired one alone, so a
  // fold restored from its own download was silently a different fold. Merging
  // the restored per-chain files through the SAME function the search path uses
  // is what makes the two routes agree by construction rather than by promise.
  it("merges back to the blocks the search itself would have produced", async () => {
    const fromSearch = mergeSearchedChains({
      sequences: chains, chainA3ms: unpaired, pairedA3ms: paired, model: "af3",
    });
    const files = buildFoldArchive({
      stem: "fold_rt", model: "AlphaFold 3", settings: { seed: 1 },
      entities: chains.map((value) => ({ type: "protein", value, copies: 1 })),
      prediction: prediction([6, 6]),
      msas: { unpaired, paired: chains.map((chain) => paired.get(chain)) },
    });
    const restored = msasFromArchive(await readZip(await writeZip(files)));
    const fromArchive = mergeSearchedChains({
      sequences: chains,
      chainA3ms: restored.chainA3ms,
      pairedA3ms: new Map(chains.map((chain, index) =>
        [chain, restored.pairedA3ms.get(index)])),
      model: "af3",
    });
    expect(fromArchive.blocks.paired).toBe(fromSearch.blocks.paired);
    expect(fromArchive.blocks.unpaired).toBe(fromSearch.blocks.unpaired);
    expect(fromArchive.a3m).toBe(fromSearch.a3m);
    // ...and the paired block is genuinely present, or the comparison above
    // would be two identical nothings.
    expect(fromSearch.blocks.paired).toContain("ACDEFF");
  });
});
