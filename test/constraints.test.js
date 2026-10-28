/**
 * Pocket and contact constraints: the rows, the job JSON, and the features boltz2 and chai-1 read, held against
 * af3-any-model's own featurisation of the same jobs (tools/fixtures/constraints/af3-any-model-features.json, its
 * non-default entries). The device side - the term each site adds - is tools/gpu/check-constraints.js.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "./harness.js";
import { jobFromJson, jobInputJson } from "../web/job-json.js";
import { entityProblem, expandEntities, parsePocket, parseContact } from "../web/entities.js";
import { mergeJobAlignments } from "../shared/input/chains.js";
import { smilesComponent } from "../shared/chem/component.js";
import { parseCcdComponent, ligandChain } from "../shared/af3/featurise/ccd-component.js";
import { af3BatchFromA3m } from "../shared/af3/featurise/batch.js";
import { applyConstraints, chai1RestraintTable } from "../shared/af3/featurise/constraints.js";
import { dialectFor, featuriserDialect } from "../shared/af3/dialect.js";

const fixture = (path) => readFileSync(new URL(`../tools/fixtures/${path}`, import.meta.url), "utf8");
const ccd = (code) => parseCcdComponent(fixture(`ccd/${code}.cif`));
const REFERENCE = JSON.parse(fixture("constraints/af3-any-model-features.json")).cases;

async function batchFor(name, model) {
  const job = jobFromJson(fixture(`constraints/${name}.json`));
  const request = expandEntities(job.entities);
  const merged = job.alignments === undefined ? null
    : mergeJobAlignments(job.alignments, request.chains, request.chainKinds);
  const ligands = [];
  for (const entry of request.ligandCodes) {
    ligands.push(typeof entry === "string" ? ccd(entry) : entry.codes ? ligandChain(entry.codes.map(ccd))
      : await smilesComponent(entry.smiles, { code: entry.code ?? "LIG" }));
  }
  return af3BatchFromA3m(request.sequence, merged?.alignment ?? null, {
    chainKinds: request.chainKinds, ligands,
    ...(merged === null ? {} : { msaColumnKinds: merged.msaColumnKinds }),
    modifications: request.modifications.map((m) => ({ chain: m.chain, position: m.position, ...ccd(m.code) })),
    ...featuriserDialect(dialectFor(model)), constraints: request.constraints, modelName: model,
  }).batch;
}

describe("constraint rows", () => {
  it("parses a pocket and a distance, the distance defaulting to 6 A", () => {
    expect(parsePocket("b: A9, a12 ,C1:O1 @ 7.5")).toEqual({
      binder: "B", maxDistance: 7.5,
      contacts: [{ chain: "A", residue: 9, atom: null }, { chain: "A", residue: 12, atom: null },
                 { chain: "C", residue: 1, atom: "O1" }] });
    expect(parsePocket("B: A9").maxDistance).toBe(6);
    expect(parseContact("A41 - B1:PG @ 8")).toEqual({
      token1: { chain: "A", residue: 41, atom: null }, token2: { chain: "B", residue: 1, atom: "PG" }, maxDistance: 8 });
  });
  it("refuses what it cannot read", () => {
    expect(parsePocket("A9, A12")).toBe(null);
    expect(parsePocket("B: A9 @ 0")).toBe(null);
    expect(parseContact("A10 - B44 @ -1")).toBe(null);
    expect(entityProblem({ type: "contact", value: "A10 - A10", copies: 1 })).toBe("A contact joins two different tokens");
    expect(entityProblem({ type: "pocket", value: "B: A9 @ 6", copies: 1 })).toBe(null);
  });
});

describe("constraints in a job", () => {
  it("round-trips through the rows and the open dialect", () => {
    const job = jobFromJson(fixture("constraints/pocket_contact.json"));
    expect(job.entities.filter((e) => e.type === "pocket").map((e) => e.value)).toEqual(["B: A9, A12, A13, A16, A30 @ 6"]);
    expect(job.entities.filter((e) => e.type === "contact").map((e) => e.value)).toEqual(["A41 - B1:PG @ 8"]);
    const written = JSON.parse(jobInputJson({ name: "x", seed: 1, entities: job.entities }));
    expect(written.constraints).toEqual(JSON.parse(fixture("constraints/pocket_contact.json")).constraints);
  });
  it("refuses force, which nothing here implements", () => {
    const raw = JSON.parse(fixture("constraints/contact.json"));
    raw.constraints[0].contact.force = true;
    expect(() => jobFromJson(JSON.stringify(raw))).toThrow(/force/);
  });
  it("is refused by a model trained on neither", () => {
    expect(() => applyConstraints({}, [{ kind: "contact", maxDistance: 8 }], null, "protenix2")).toThrow(/not trained/);
  });
});

describe("constraint features against af3-any-model", () => {
  for (const [key, ref] of Object.entries(REFERENCE)) {
    const [model, name] = key.split("/");
    it(key, async () => {
      const batch = await batchFor(name, model);
      const n = batch.asymId.length;
      expect(n).toBe(ref.tokens);
      if (model === "boltz2") {
        const marked = [];
        for (let p = 0; p < n * n; p += 1) {
          if (batch.contactConditioning[p] !== ref.base) {
            marked.push([Math.floor(p / n), p % n, batch.contactConditioning[p], batch.contactThreshold[p]]);
          }
        }
        expect(marked).toEqual(ref.marked);
      } else {
        const listed = (plane) => {
          const out = [];
          for (let p = 0; p < n * n; p += 1) if (plane[p] !== -1) out.push([Math.floor(p / n), p % n, plane[p]]);
          return out;
        };
        expect(listed(batch.tokenDistanceRestraint)).toEqual(ref.contact);
        expect(listed(batch.tokenPairPocketRestraint)).toEqual(ref.pocket);
      }
    });
  }
  it("chai-1 refuses an atom end, as af3-any-model does", async () => {
    let refused = null;
    try { await batchFor("contact_ligand", "chai1"); } catch (error) { refused = error; }
    expect(refused?.message ?? "").toMatch(/not atoms/);
  });
});

describe("the constraint tables", () => {
  it("chai-1's no-restraint pair is row 0 and adds nothing", () => {
    const contact = Float32Array.from([-1, 8, -1, -1]);
    const pocket = Float32Array.from([-1, -1, 6, -1]);
    const t = chai1RestraintTable(contact, pocket, new Float32Array(14 * 2).fill(1));
    expect(Array.from(t.index)).toEqual([0, 1, 2, 0]);
    expect(Array.from(t.table.subarray(0, 2))).toEqual([0, 0]);
  });
});
