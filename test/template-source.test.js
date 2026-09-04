import { describe, expect, it } from "./harness.js";
import {
  buildTemplate, describeCoverage, fetchStructure, mapToQuery, parseSource,
} from "../web/template-source.js";
import { chainResidues } from "../src/af3/template-input.js";
import { GAP_AATYPE } from "../src/af3/template-input.js";

function line(serial, name, resName, chain, resSeq, x, y, z, bFactor) {
  const out = new Array(60).fill(" ");
  const put = (start, text) => {
    for (let index = 0; index < text.length; index += 1) out[start + index] = text[index];
  };
  put(0, "ATOM  ");
  put(11 - String(serial).length, String(serial));
  put(12, name.padEnd(4));
  put(20 - resName.length, resName);
  put(21, chain);
  put(26 - String(resSeq).length, String(resSeq));
  put(30, x.toFixed(3).padStart(8));
  put(38, y.toFixed(3).padStart(8));
  put(46, z.toFixed(3).padStart(8));
  put(54, "  1.00");
  return out.join("") + bFactor.toFixed(2).padStart(6);
}

/** Three alanines on chain A, with per-residue pLDDT in the B-factor. */
const structure = (confidences = [90, 90, 90], chain = "A") => `${confidences
  .flatMap((bFactor, residue) => ["N", "CA", "C", "O", "CB"].map((name, slot) =>
    line(residue * 5 + slot + 1, name, "ALA", chain, residue + 1,
         residue * 3.8 + slot * 0.1, slot, 0, bFactor)))
  .join("\n")}\nTER\nEND\n`;

describe("parseSource", () => {
  it("reads a PDB entry, with or without a chain", () => {
    expect(parseSource("1abc")).toEqual({ id: "1ABC", chain: undefined, kind: "pdb" });
    expect(parseSource("1abc_B")).toEqual({ id: "1ABC", chain: "B", kind: "pdb" });
    expect(parseSource("1abc:B")).toEqual({ id: "1ABC", chain: "B", kind: "pdb" });
  });

  // Four characters is an entry and anything else an accession, which is the
  // rule ../mpnn's fetchPDB uses and is right often enough not to ask.
  it("reads anything else as a UniProt accession", () => {
    expect(parseSource("P00533").kind).toBe("afdb");
    expect(parseSource(" p00533 ").id).toBe("P00533");
  });
});

describe("fetchStructure", () => {
  const responses = (byUrl) => async (url) => ({
    ok: byUrl[url] !== undefined,
    status: byUrl[url] === undefined ? 404 : 200,
    text: async () => byUrl[url],
  });

  // 🔴 `.pdb` FIRST, unlike ../mpnn's fetchPDB. This reader is fixed-column, so
  // an mmCIF it cannot parse would be a template of nothing.
  it("asks the RCSB for the legacy format before the mmCIF", () => {
    const asked = [];
    return fetchStructure("1abc", {
      fetch: async (url) => {
        asked.push(url);
        return { ok: url.endsWith(".pdb"), status: 404, text: async () => "PDB" };
      },
    }).then((result) => {
      expect(asked[0].endsWith("1ABC.pdb")).toBe(true);
      expect(result.kind).toBe("pdb");
    });
  });

  it("falls back to the mmCIF, and says which it got", async () => {
    const asked = [];
    await fetchStructure("1abc", {
      fetch: async (url) => {
        asked.push(url);
        return { ok: url.endsWith(".cif"), status: 404, text: async () => "CIF" };
      },
    });
    expect(asked).toHaveLength(2);
  });

  it("goes to AlphaFold DB for an accession", async () => {
    const result = await fetchStructure("P00533", {
      fetch: responses({
        "https://alphafold.ebi.ac.uk/files/AF-P00533-F1-model_v4.pdb": "AFDB",
      }),
    });
    expect(result.text).toBe("AFDB");
    expect(result.kind).toBe("afdb");
  });

  it("reports the last failure rather than an empty template", async () => {
    let thrown = null;
    try {
      await fetchStructure("1abc", { fetch: responses({}) });
    } catch (error) {
      thrown = error;
    }
    expect(thrown?.message).toContain("404");
  });
});

describe("mapToQuery", () => {
  const parsed = chainResidues(structure(), "A");

  it("takes the structure's own sequence when the chain has none", () => {
    const result = mapToQuery(parsed, "");
    expect(result.sequence).toBe("AAA");
    expect(result.identical).toBe(true);
    expect(result.map.size).toBe(3);
  });

  it("maps residue for residue when the two already agree", () => {
    expect(mapToQuery(parsed, "AAA").identical).toBe(true);
  });

  // 🔴 correspondence() SHORT-CIRCUITS TO THE IDENTITY ON EQUAL LENGTHS, which
  // is right for a point mutation and wrong here: a template the same length as
  // the query is not the same protein.
  it("aligns when they differ, even at the same length", () => {
    const result = mapToQuery(chainResidues(structure([90, 90, 90]), "A"), "GGG");
    expect(result.identical).toBe(false);
  });
});

describe("buildTemplate", () => {
  it("covers the chain and reports what it used", () => {
    const built = buildTemplate({ text: structure(), tokens: 3 });
    expect(built.coverage.residues).toBe(3);
    expect(built.coverage.of).toBe(3);
    expect(built.coverage.atoms).toBe(15);
    expect(built.slot.aatype[0] === GAP_AATYPE).toBe(false);
  });

  it("places a chain at its offset in a complex", () => {
    const built = buildTemplate({ text: structure(), tokens: 8, offset: 5 });
    expect(built.slot.aatype[0]).toBe(GAP_AATYPE);
    expect(built.slot.aatype[5] === GAP_AATYPE).toBe(false);
  });

  // AlphaFold DB has every residue and no way to say "I did not see this", so
  // a disordered tail at pLDDT 30 would be handed over as geometry.
  it("drops residues below the pLDDT floor and says how many", () => {
    const built = buildTemplate({
      text: structure([95, 30, 95]), tokens: 3, minConfidence: 70,
    });
    expect(built.coverage.residues).toBe(2);
    expect(built.coverage.dropped).toBe(1);
    expect(built.slot.aatype[1]).toBe(GAP_AATYPE);
  });

  // 🔴 AN EMPTY SLOT IS NOT NEUTRAL. The aatype features are read whether or
  // not there is geometry, so a slot covering nothing is the GAP token
  // everywhere - a claim, not an absence.
  it("refuses a template that covers nothing", () => {
    expect(() => buildTemplate({
      text: structure([20, 20, 20]), tokens: 3, minConfidence: 70,
    })).toThrow(/covers none of the chain/);
    expect(() => buildTemplate({ text: structure(), chain: "Z", tokens: 3 }))
      .toThrow(/no chain Z/);
  });

  it("carries the spanning flag onto the slot", () => {
    expect(buildTemplate({ text: structure(), tokens: 3, spanChains: true })
      .slot.spanChains).toBe(true);
  });
});

describe("describeCoverage", () => {
  it("says what was used and what was not", () => {
    const text = describeCoverage({
      chain: "A", residues: 90, of: 120, atoms: 700, dropped: 5, aligned: true,
    });
    expect(text).toContain("90 of 120");
    expect(text).toContain("75%");
    expect(text).toContain("5 below the pLDDT floor");
    expect(text).toContain("aligned");
  });
});
