import { describe, expect, it } from "./harness.js";
import {
  buildTemplate, describeCoverage, fetchStructure, mapToQuery, parseSource,
  residuesFromCif,
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

  // 🔴 THE VERSION COMES FROM AlphaFold DB, NOT FROM US. The version lives in
  // the filename and it moves: this built `..._v4.pdb` directly until the v6
  // release retired v4, and then every AlphaFold DB template was a 404 naming a
  // URL nobody had chosen. Asked for, it cannot go stale.
  it("asks AlphaFold DB which file it is serving", async () => {
    const asked = [];
    const result = await fetchStructure("P00533", {
      fetch: async (url) => {
        asked.push(url);
        if (url.includes("/api/prediction/")) {
          return { ok: true, status: 200,
            json: async () => [{ pdbUrl: "https://example/AF-P00533-F1-model_v9.pdb" }] };
        }
        return { ok: true, status: 200, text: async () => "AFDB" };
      },
    });
    expect(asked[0]).toContain("/api/prediction/P00533");
    expect(asked[1]).toContain("model_v9.pdb");
    expect(result.text).toBe("AFDB");
    expect(result.kind).toBe("afdb");
  });

  it("says so when AlphaFold DB holds nothing for an accession", async () => {
    let thrown = null;
    try {
      await fetchStructure("P00533", {
        fetch: async () => ({ ok: true, status: 200, json: async () => [] }),
      });
    } catch (error) {
      thrown = error;
    }
    expect(thrown?.message).toContain("no structure for P00533");
  });

  // 🔴 THE KIND IS THE CALLER'S TO NAME. Counting characters is right most of
  // the time and silently wrong the rest, which is why the page stopped doing
  // it - a four-character accession went to the RCSB and a typo'd PDB id to
  // AlphaFold DB, both without saying so.
  it("honours an explicit kind over the four-character guess", async () => {
    const asked = [];
    await fetchStructure("1abc", {
      kind: "afdb",
      fetch: async (url) => {
        asked.push(url);
        return url.includes("/api/prediction/")
          ? { ok: true, status: 200, json: async () => [{ pdbUrl: "https://example/x.pdb" }] }
          : { ok: true, status: 200, text: async () => "AFDB" };
      },
    });
    expect(asked[0]).toContain("/api/prediction/1ABC");
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

describe("heteroatoms in a template", () => {
  const withWaters = `${structure().trim()}\n`
    + `${line(99, "O", "HOH", "A", 900, 9, 9, 9, 30).replace(/^ATOM  /, "HETATM")}\n`
    + `${line(100, "SE", "MSE", "A", 4, 12, 0, 0, 40).replace(/^ATOM  /, "HETATM")}\n`
    + `${line(101, "CA", "MSE", "A", 4, 13, 0, 0, 40).replace(/^ATOM  /, "HETATM")}\nEND\n`;

  /**
   * 🔴 A WATER IS NOT A RESIDUE, AND A PDB PUTS IT ON A CHAIN. Reading every
   * HETATM as one made 1qys chain A ninety-NINE residues instead of
   * ninety-two: eight waters, each an X with one atom, each contributing a
   * pseudo-beta position to the distogram and a row to the alignment. Found by
   * comparing the PDB reader against the mmCIF one on the same entry.
   */
  it("drops waters", () => {
    const parsed = chainResidues(withWaters, "A");
    expect(parsed.residues.some((residue) => residue.number === "900")).toBe(false);
  });

  // Selenomethionine is how a great many structures were phased; dropping
  // every heteroatom takes a real methionine out of the middle of a chain.
  it("keeps selenomethionine, as M", () => {
    const parsed = chainResidues(withWaters, "A");
    const mse = parsed.residues.find((residue) => residue.number === "4");
    expect(mse?.code).toBe("M");
    expect(parsed.sequence).toBe("AAAM");
  });
});

describe("mmCIF templates", () => {
  // The MMseqs2 template endpoint hands over mmCIF where the RCSB and
  // AlphaFold DB hand over PDB, so buildTemplate sniffs rather than being told.
  const cif = `data_TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
ATOM 1 N ALA A 1 0.0 1.0 0.0 1.0
ATOM 2 CA ALA A 1 0.0 0.0 0.0 1.0
ATOM 3 C ALA A 1 1.0 0.0 0.0 1.0
ATOM 4 CB ALA A 1 0.0 -1.0 1.0 1.0
ATOM 5 N ALA A 2 4.0 1.0 0.0 1.0
ATOM 6 CA ALA A 2 4.0 0.0 0.0 1.0
ATOM 7 C ALA A 2 5.0 0.0 0.0 1.0
ATOM 8 CB ALA A 2 4.0 -1.0 1.0 1.0
HETATM 9 O HOH A 900 9.0 9.0 9.0 1.0
#
`;

  it("reads a chain, and still drops the water", () => {
    const parsed = residuesFromCif(cif, "A");
    expect(parsed.residues).toHaveLength(2);
    expect(parsed.sequence).toBe("AA");
  });

  it("is picked by buildTemplate without being told the format", () => {
    const built = buildTemplate({ text: cif, chain: "A", tokens: 2 });
    expect(built.coverage.residues).toBe(2);
    expect(built.coverage.atoms).toBe(8);
  });
});
