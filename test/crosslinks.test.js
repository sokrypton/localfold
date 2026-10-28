/**
 * Crosslinks (af3x's `crosslinks`): the row, the job JSON, and the ligand and bonds a row expands into - a
 * crosslinker bridge (shared/chem/crosslinkers.js, generated from af3-any-model) bonded to its two residues.
 */
import { describe, expect, it } from "./harness.js";
import { jobFromJson, jobInputJson } from "../web/job-json.js";
import { entitiesProblem, entityProblem, expandEntities, parseCrosslink } from "../web/entities.js";
import { CROSSLINKERS, CROSSLINKER_COMPONENTS, crosslinker } from "../shared/chem/crosslinkers.js";
import { parseCcdComponent } from "../shared/af3/featurise/ccd-component.js";
import { af3BatchFromA3m } from "../shared/af3/featurise/batch.js";
import { dialectFor, featuriserDialect } from "../shared/af3/dialect.js";
import { toPdb } from "../shared/af3/structure/pdb.js";

const SEQ_A = "MKTAYIAKQRQISFVKSHFSRQ";   // M1 K2 T3 A4 Y5 K8 S13 K16
const SEQ_B = "GSKELAAKLGKEAYEK";         // G1 S2 K3 K8 G10 K11 Y14
const row = (type, value) => ({ type, value, copies: 1, modifications: [] });
const proteins = () => [row("protein", SEQ_A), row("protein", SEQ_B)];
const job = (crosslinks) => JSON.stringify({
  dialect: "alphafold3", version: 4, name: "xl", modelSeeds: [1],
  sequences: [{ protein: { id: "A", sequence: SEQ_A } }, { protein: { id: "B", sequence: SEQ_B } }],
  crosslinks });

describe("the crosslink row", () => {
  it("parses a crosslinker and two residues", () => {
    expect(parseCrosslink("sulfo-SDA: a16 - B10")).toEqual({
      name: "sulfo-SDA", from: { chain: "A", residue: 16 }, to: { chain: "B", residue: 10 } });
    expect(parseCrosslink("DSSO A2 - B3")).toBe(null);
    expect(parseCrosslink("DSSO: A2:NZ - B3")).toBe(null);
  });
  it("names the crosslinker and refuses one it does not know", () => {
    expect(crosslinker("bs(peg)5").code).toBe("XL-BSPEG5");
    expect(entityProblem(row("crosslink", "EDC: A2 - B3"))).toMatch(/Unknown crosslinker EDC/);
    expect(entityProblem(row("crosslink", "DSS: A2 - A2"))).toMatch(/two different residues/);
  });
  it("allows one crosslink per residue", () => {
    expect(entitiesProblem([...proteins(), row("crosslink", "DSS: A2 - B3"), row("crosslink", "DSSO: A8 - B3")]))
      .toMatch(/B3 carries two crosslinks/);
  });
  it("has a component for every crosslinker", () => {
    for (const xl of Object.values(CROSSLINKERS)) {
      const names = parseCcdComponent(CROSSLINKER_COMPONENTS[xl.code]).atoms.map((a) => a.name);
      for (const atom of xl.attach) expect(names.includes(atom)).toBe(true);
    }
  });
});

describe("a crosslink in a job", () => {
  it("round-trips through the rows and the open dialect", () => {
    const crosslinks = [{ name: "DSSO", residue_pairs: [[["A", 2], ["B", 3]], [["A", 8], ["B", 11]]] },
                        { name: "sulfo-SDA", residue_pairs: [[["A", 16], ["B", 10]]] }];
    const read = jobFromJson(job(crosslinks));
    expect(read.entities.filter((e) => e.type === "crosslink").map((e) => e.value))
      .toEqual(["DSSO: A2 - B3", "DSSO: A8 - B11", "sulfo-SDA: A16 - B10"]);
    expect(JSON.parse(jobInputJson({ name: "xl", seed: 1, entities: read.entities })).crosslinks).toEqual(crosslinks);
  });
  it("drops a pair on a residue already crosslinked, and says so", () => {
    const read = jobFromJson(job([{ name: "DSS", residue_pairs: [[["A", 2], ["B", 3]], [["A", 2], ["B", 8]]] }]));
    expect(read.entities.filter((e) => e.type === "crosslink").length).toBe(1);
    expect(read.notes.join(" ")).toMatch(/dropped 1 crosslink .*DSS A2-B8/);
  });
  it("refuses a crosslinker or a chain it does not know", () => {
    expect(() => jobFromJson(job([{ name: "EDC", residue_pairs: [] }]))).toThrow(/unknown crosslinker/);
    expect(() => jobFromJson(job([{ name: "DSS", residue_pairs: [[["A", 2], ["Q", 3]]] }]))).toThrow(/no chain "Q"/);
  });
});

describe("a crosslink expanded", () => {
  it("is a ligand after every other one, bonded to its two residues", () => {
    const out = expandEntities([...proteins(), row("ligand", "ATP"), row("crosslink", "DSSO: A2 - B3"),
                                row("crosslink", "SDA: A1 - B10")]);
    expect(out.ligandCodes).toEqual(["ATP", "XL-DSSO", "XL-SDA"]);
    const [d1, d2] = CROSSLINKERS.DSSO.attach;
    const [s1, s2] = CROSSLINKERS.SDA.attach;
    expect(out.bonds).toEqual([
      { from: { asym: 0, residue: 2, atom: "NZ" }, to: { asym: 3, residue: 1, atom: d1 } },
      { from: { asym: 1, residue: 3, atom: "NZ" }, to: { asym: 3, residue: 1, atom: d2 } },
      { from: { asym: 0, residue: 1, atom: "N" }, to: { asym: 4, residue: 1, atom: s1 } },   // M1: the N-terminus
      { from: { asym: 1, residue: 10, atom: "CA" }, to: { asym: 4, residue: 1, atom: s2 } },  // a glycine's diazirine end
    ]);
  });
  it("refuses an NHS end on a residue it cannot acylate", () => {
    expect(() => expandEntities([...proteins(), row("crosslink", "DSS: A4 - B3")])).toThrow(/NHS ester .* not ALA/);
  });
  it("featurises: the bridge atomised, bonded to both residues' tokens", () => {
    const out = expandEntities([...proteins(), row("crosslink", "DSSO: A2 - B3")]);
    const { batch } = af3BatchFromA3m(out.sequence, null, {
      chainKinds: out.chainKinds, bonds: out.bonds,
      ligands: out.ligandCodes.map((code) => parseCcdComponent(CROSSLINKER_COMPONENTS[code])),
      ...featuriserDialect(dialectFor("af3")) });
    const n = batch.asymId.length;
    const heavy = parseCcdComponent(CROSSLINKER_COMPONENTS["XL-DSSO"]).atoms.filter((a) => a.element !== "H").length;
    expect(n).toBe(SEQ_A.length + SEQ_B.length + heavy);
    const bonded = [];
    for (let p = 0; p < n * n; p += 1) if (batch.bondMatrix[p] > 0 && p % n > Math.floor(p / n)) bonded.push([Math.floor(p / n), p % n]);
    // residue A2 (token 1) and B3 (token 22+2) each bonded to a bridge token, and the bridge's own bonds
    expect(bonded.some(([i, j]) => i === 1 && j >= SEQ_A.length + SEQ_B.length)).toBe(true);
    expect(bonded.some(([i, j]) => i === SEQ_A.length + 2 && j >= SEQ_A.length + SEQ_B.length)).toBe(true);
  });
  it("writes a PDB whose bridge records keep their columns", () => {
    const out = expandEntities([...proteins(), row("crosslink", "DSSO: A2 - B3")]);
    const { batch } = af3BatchFromA3m(out.sequence, null, {
      chainKinds: out.chainKinds, bonds: out.bonds,
      ligands: out.ligandCodes.map((code) => parseCcdComponent(CROSSLINKER_COMPONENTS[code])),
      ...featuriserDialect(dialectFor("af3")) });
    const pdb = toPdb(batch, new Float32Array(batch.asymId.length * (batch.refMask.length / batch.asymId.length) * 3), null);
    const het = pdb.split("\n").filter((line) => line.startsWith("HETATM"));
    expect(het.length > 0).toBe(true);
    // XL-DSSO cut to its three columns, the chain letter where a reader looks for it
    for (const line of het) { expect(line.slice(17, 20)).toBe("XL-"); expect(line.slice(21, 22)).toBe("C"); }
  });
});
