import { describe, expect, it, vi } from "./harness.js";
import {
  extractMmseqs2A3m, generateMmseqs2ComplexMsa, generateMmseqs2Msa,
  generateMmseqs2PairedMsa, readTarFiles,
} from "../src/input/mmseqs2-api.js";
import { parseA3m } from "../src/input/a3m.js";
import { mergeChainA3ms, mergeRowAlignedChainA3ms, mergeUnpairedChainA3ms }
  from "../src/input/chains.js";

function tar(files) {
  const chunks= [];
  for (const [name, value] of Object.entries(files)) {
    const header = new Uint8Array(512); const encodedName = new TextEncoder().encode(name);
    header.set(encodedName, 0);
    header.set(new TextEncoder().encode(value.length.toString(8).padStart(11, "0") + "\0"), 124);
    header[156] = 48;
    const data = new TextEncoder().encode(value); const padded = new Uint8Array(Math.ceil(data.length / 512) * 512); padded.set(data);
    chunks.push(header, padded);
  }
  chunks.push(new Uint8Array(1024));
  const output = new Uint8Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
  let offset = 0; for (const chunk of chunks) { output.set(chunk, offset); offset += chunk.length; }
  return output;
}

const resultTar = tar({
  "uniref.a3m": ">101\nACDE\n>hit1\nAC-E\n\0",
  "bfd.mgnify30.metaeuk30.smag30.a3m": ">101\nACDE\n>env1\nAcCDE\n\0",
});

describe("MMseqs2 API", () => {
  it("reads files and combines the ColabFold UniRef and environmental A3Ms", () => {
    expect([...readTarFiles(resultTar).keys()]).toEqual(["uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"]);
    expect(extractMmseqs2A3m(resultTar)).toBe(">101\nACDE\n>hit1\nAC-E\n>101\nACDE\n>env1\nAcCDE\n");
  });

  it("submits, polls, downloads, and validates a generated MSA", async () => {
    const requests= [];
    const responses = [
      new Response(JSON.stringify({ status: "PENDING", id: "ticket-1" })),
      new Response(JSON.stringify({ status: "RUNNING" })),
      new Response(JSON.stringify({ status: "COMPLETE" })),
      new Response(new Uint8Array([1, 2, 3])),
    ];
    const fetchImplementation = vi.fn(async (input, init) => {
      requests.push({ url: String(input), ...(init === undefined ? {} : { init }) });
      return responses.shift() ;
    });
    const phases= [];
    const result = await generateMmseqs2Msa("ACDE", {
      fetchImplementation, wait: async () => {}, decompress: async () => resultTar,
      onProgress: (progress) => phases.push(progress.phase),
    });
    expect(result.ticket).toBe("ticket-1"); expect(result.depth).toBe(4);
    expect(requests.map((entry) => entry.url)).toEqual([
      "https://api.colabfold.com/ticket/msa", "https://api.colabfold.com/ticket/ticket-1",
      "https://api.colabfold.com/ticket/ticket-1", "https://api.colabfold.com/result/download/ticket-1",
    ]);
    expect(String(requests[0] .init?.body)).toContain("mode=env");
    expect(phases).toEqual(["submitting", "queued", "running", "downloading", "complete"]);
  });

  it("pairs both copies when asked", async() => {
    const responses = [
      new Response(JSON.stringify({ status: "COMPLETE", id: "ticket-homo" })),
      new Response(new Uint8Array([1, 2, 3])),
    ];
    const fetchImplementation = vi.fn(async() => responses.shift());
    const result = await generateMmseqs2ComplexMsa(["ACDE", "ACDE"], {
      fetchImplementation, wait: async() => {}, decompress: async() => resultTar,
      model: "multimer",
    });
    expect(fetchImplementation.mock.calls.length).toBe(2);
    expect(result.tickets).toEqual(["ticket-homo"]);
    const parsed = parseA3m(result.a3m);
    expect(parsed.query).toBe("ACDEACDE");
    // ...one row carrying the homolog in BOTH copies, not two gap-padded rows.
    // Same protein twice means row s is one organism, so the pairing is exact.
    expect(parsed.sequences).toContain("AC-EAC-E");
    expect(parsed.sequences.includes("AC-E----")).toBe(false);
    expect(parsed.sequences.includes("----AC-E")).toBe(false);
  });

  it("gives each model the merge that model wants", async() => {
    // 🔴 THE MAPPING IS PINNED, NOT THE CONTENT. All three merges return a
    // valid A3M of the right width whose first row is the query, so handing a
    // model the wrong one produces a fold rather than an error - which is how a
    // homodimer came to be folded against a doubled alignment. Comparing the
    // result against the merge functions themselves catches a swapped mapping
    // even on an input where two of them happen to agree, which a homodimer is.
    const run = async(model) => {
      const responses = [
        new Response(JSON.stringify({ status: "COMPLETE", id: "ticket-homo" })),
        new Response(new Uint8Array([1, 2, 3])),
      ];
      return generateMmseqs2ComplexMsa(["ACDE", "ACDE"], {
        fetchImplementation: async() => responses.shift(),
        wait: async() => {}, decompress: async() => resultTar, model,
      });
    };
    for (const [model, merge] of [
      ["monomer", mergeUnpairedChainA3ms],
      ["multimer", mergeChainA3ms],
      ["af3", mergeRowAlignedChainA3ms],
    ]) {
      const result = await run(model);
      expect(result.a3m).toBe(merge(result.chainA3ms));
    }
  });

  it("distinguishes the three merges on a heteromer, which is where they differ", () => {
    // A homomer cannot tell multimer from AF3: with one entity, dense-within-
    // entity and dense-throughout are the same thing. Two entities separate
    // them, and separate both from the monomer's block-diagonal form.
    const a = ">q\nACDE\n>h\nAC-E\n";
    const b = ">q\nWYWY\n>k\nW-WY\n";
    // The monomer gives every homolog a row to itself.
    expect(parseA3m(mergeUnpairedChainA3ms([a, b])).sequences).toContain("AC-E----");
    // Multimer block-diagonalises BETWEEN entities, so a heteromer looks the
    // same - copies of ONE sequence are what it would have made dense.
    expect(parseA3m(mergeChainA3ms([a, b])).sequences).toContain("AC-E----");
    // AF3 is dense regardless of entity: one row carries both chains' hits.
    const af3 = parseA3m(mergeRowAlignedChainA3ms([a, b])).sequences;
    expect(af3).toContain("AC-EW-WY");
    expect(af3.includes("AC-E----")).toBe(false);
  });

  it("pairs distinct sequences through ticket/pair, and stacks the blocks", async() => {
    // 🔴 THE PAIRED ROWS MUST LINE UP ACROSS CHAINS AND SIT ON TOP. pair.a3m
    // holds every query's alignment NUL-separated, row r of one being the same
    // species as row r of another - so the merged result must carry one row
    // with BOTH chains' paired hits, above the block-diagonal unpaired rows.
    const pairTar = tar({
      "pair.a3m": ">101\nACDE\n>hit_sp1\nAC-E\n\0>102\nWYWY\n>hit_sp1\nW-WY\n\0",
    });
    // Each chain's unpaired block holds its paired hit AGAIN - which is what a
    // real search returns, both blocks coming from the same databases - and one
    // hit found only unpaired. AlphaFold drops the first and keeps the second.
    const chainTar = (query, hit, only) => tar({
      "uniref.a3m": `>101\n${query}\n>u1\n${hit}\n`
        + (only === undefined ? "" : `>u2\n${only}\n`) + "\0",
      "bfd.mgnify30.metaeuk30.smag30.a3m": `>101\n${query}\n\0`,
    });
    const posted = [];
    let downloads = 0;
    const result = await generateMmseqs2ComplexMsa(["ACDE", "WYWY"], {
      fetchImplementation: async(url, init) => {
        const href = String(url);
        if (init?.method === "POST") {
          posted.push({ href, mode: new URLSearchParams(String(init.body)).get("mode") });
          return new Response(JSON.stringify({ status: "COMPLETE", id: `t${posted.length}` }));
        }
        return new Response(new Uint8Array([1, 2, 3]));
      },
      wait: async() => {},
      // The two chain searches come first, then the pair job.
      decompress: async() => {
        downloads += 1;
        if (downloads === 1) return chainTar("ACDE", "AC-E", "ACD-");
        if (downloads === 2) return chainTar("WYWY", "W-WY", "WY-Y");
        return pairTar;
      },
      model: "multimer",
    });

    // One pair submission, with the pairing mode, alongside the two searches.
    expect(posted.filter((p) => p.href.endsWith("ticket/pair")).length).toBe(1);
    expect(posted.find((p) => p.href.endsWith("ticket/pair")).mode).toBe("pairgreedy");
    expect(result.pairedDepth).toBe(2);

    const rows = parseA3m(result.a3m).sequences;
    expect(rows[0]).toBe("ACDEWYWY");
    // The paired row carries both chains' hits at once...
    expect(rows[1]).toBe("AC-EW-WY");
    // ...and the unpaired rows follow it, block-diagonal between entities.
    expect(rows).toContain("ACD-----");
    expect(rows).toContain("----WY-Y");
    // 🔴 AND THE ROWS THE PAIRED BLOCK ALREADY HAS ARE GONE FROM THE UNPAIRED
    // ONE. AlphaFold deduplicates per chain before merging
    // (msa_pairing.deduplicate_unpaired_sequences), and it is not tidiness: the
    // blocks share one row budget, so a duplicate evicts a sequence that
    // carried something. Both chains' unpaired blocks held their paired hit.
    expect(rows.includes("AC-E----")).toBe(false);
    expect(rows.includes("----W-WY")).toBe(false);
    // The query appears once, not once per block.
    expect(rows.filter((row) => row === "ACDEWYWY").length).toBe(1);
    // AF3 gets them apart, because its profile is over the unpaired block only.
    expect(parseA3m(result.blocks.paired).sequences).toContain("AC-EW-WY");
    expect(parseA3m(result.blocks.unpaired).sequences.includes("AC-EW-WY")).toBe(false);
    // 🔴 AND THE PROFILE'S BLOCK STILL HAS THE DEDUPLICATED ROWS. AF3 computes
    // the profile at features.py:543 and deduplicates at :559, so the rows the
    // dedup removes still counted towards it. Handing the model the shrunken
    // block for both is a different feature, not a tidier one.
    expect(parseA3m(result.blocks.unpairedProfile).sequences).toContain("AC-E----");
    expect(parseA3m(result.blocks.unpairedProfile).sequences).toContain("----W-WY");
  });

  it("does not pair a homomer, where one search already speaks for every copy", async() => {
    // AlphaFold decides this the same way: feature_processing sets
    // pair_msa_sequences from `not _is_homomer_or_monomer`. A second search here
    // would cost a round trip to learn nothing.
    const posted = [];
    const result = await generateMmseqs2ComplexMsa(["ACDE", "ACDE"], {
      fetchImplementation: async(url, init) => {
        if (init?.method === "POST") {
          posted.push(String(url));
          return new Response(JSON.stringify({ status: "COMPLETE", id: "t" }));
        }
        return new Response(new Uint8Array([1, 2, 3]));
      },
      wait: async() => {}, decompress: async() => resultTar, model: "multimer",
    });
    expect(posted.some((href) => href.endsWith("ticket/pair"))).toBe(false);
    expect(result.blocks.paired).toBe(null);
    expect(result.pairedDepth).toBe(0);
  });

  it("never pairs for the AF2 monomer, whatever the complex", async() => {
    // The monomer has no chain input at all - the +200 offset stands in for one
    // - so a row spanning two chains would claim their residues coevolved.
    const posted = [];
    // Each chain's unpaired block holds its paired hit AGAIN - which is what a
    // real search returns, both blocks coming from the same databases - and one
    // hit found only unpaired. AlphaFold drops the first and keeps the second.
    const chainTar = (query, hit, only) => tar({
      "uniref.a3m": `>101\n${query}\n>u1\n${hit}\n`
        + (only === undefined ? "" : `>u2\n${only}\n`) + "\0",
      "bfd.mgnify30.metaeuk30.smag30.a3m": `>101\n${query}\n\0`,
    });
    let downloads = 0;
    const result = await generateMmseqs2ComplexMsa(["ACDE", "WYWY"], {
      fetchImplementation: async(url, init) => {
        if (init?.method === "POST") {
          posted.push(String(url));
          return new Response(JSON.stringify({ status: "COMPLETE", id: "t" }));
        }
        return new Response(new Uint8Array([1, 2, 3]));
      },
      wait: async() => {},
      decompress: async() => (downloads++ === 0 ? chainTar("ACDE", "AC-E") : chainTar("WYWY", "W-WY")),
      model: "monomer",
    });
    expect(posted.some((href) => href.endsWith("ticket/pair"))).toBe(false);
    expect(result.blocks.paired).toBe(null);
  });

  it("refuses paired alignments whose depths disagree", async() => {
    // Equal depth is what makes them paired: row r must exist in every entity
    // for "row r is one species" to mean anything. A ragged set still merges
    // into a valid alignment and would silently pair different organisms.
    const ragged = tar({
      "pair.a3m": ">101\nACDE\n>a\nAC-E\n>b\nA--E\n\0>102\nWYWY\n>a\nW-WY\n\0",
    });
    let message = "no error";
    try {
      await generateMmseqs2PairedMsa(["ACDE", "WYWY"], {
        fetchImplementation: async(url, init) => (init?.method === "POST"
          ? new Response(JSON.stringify({ status: "COMPLETE", id: "t" }))
          : new Response(new Uint8Array([1, 2, 3]))),
        wait: async() => {}, decompress: async() => ragged,
      });
    } catch (error) { message = error.message; }
    expect(message).toContain("aligned row counts");
  });

  it("refuses a model it has no merge for", async() => {
    let message = "no error";
    try {
      const responses = [
        new Response(JSON.stringify({ status: "COMPLETE", id: "t" })),
        new Response(new Uint8Array([1, 2, 3])),
      ];
      await generateMmseqs2ComplexMsa(["ACDE", "ACDE"], {
        fetchImplementation: async() => responses.shift(),
        wait: async() => {}, decompress: async() => resultTar, model: "af2",
      });
    } catch (error) { message = error.message; }
    expect(message).toContain("unknown model af2");
  });

  it("is block-diagonal by default", async() => {
    const responses = [
      new Response(JSON.stringify({ status: "COMPLETE", id: "ticket-homo" })),
      new Response(new Uint8Array([1, 2, 3])),
    ];
    const fetchImplementation = vi.fn(async() => responses.shift());
    const result = await generateMmseqs2ComplexMsa(["ACDE", "ACDE"], {
      fetchImplementation, wait: async() => {}, decompress: async() => resultTar,
    });
    const parsed = parseA3m(result.a3m);
    expect(parsed.sequences).toContain("AC-E----");
    expect(parsed.sequences).toContain("----AC-E");
    expect(parsed.sequences.includes("AC-EAC-E")).toBe(false);
  });
});
