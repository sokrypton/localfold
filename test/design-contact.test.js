import { describe, expect, it } from "./harness.js";
import { CONTACT_ANGSTROMS, POCKET_ANGSTROMS, contactOf, contactProblem, hotspotPocket } from "../cpu/design/contact.js";
import { af3BatchFromA3m } from "../shared/af3/featurise/batch.js";
import { dialectFor, featuriserDialect } from "../shared/af3/dialect.js";
import { CONTACT_CLASSES } from "../shared/af3/featurise/constraints.js";
import { CHAIN_IDS } from "../webgpu/design/hunter-loop.js";

/** A backbone at `x`, for one residue of `chain` numbered `residue`. */
const residue = (chain, number, x) => [
  `ATOM      1  N   ALA ${chain}${String(number).padStart(4)}    ${(x - 1).toFixed(3).padStart(8)}`
    + "   0.000   0.000  1.00  0.00           N",
  `ATOM      2  CA  ALA ${chain}${String(number).padStart(4)}    ${x.toFixed(3).padStart(8)}`
    + "   0.000   0.000  1.00  0.00           C",
  `ATOM      3  C   ALA ${chain}${String(number).padStart(4)}    ${(x + 1).toFixed(3).padStart(8)}`
    + "   0.000   0.000  1.00  0.00           C",
  `ATOM      4  O   ALA ${chain}${String(number).padStart(4)}    ${(x + 2).toFixed(3).padStart(8)}`
    + "   0.000   0.000  1.00  0.00           O",
].join("\n");

const pdb = (...residues) => `${residues.join("\n")}\nEND\n`;

describe("is the designed chain touching what it was designed against", () => {
  // 🔴 THE CASE A FOLD'S OWN CONFIDENCE CANNOT CATCH: the binder is 50 A away,
  // and ipTM is a number the model computes about its own prediction - it can be
  // confident about a chain that is nowhere near the target. The geometry is the
  // only thing that says so.
  it("sees a binder that drifted off", () => {
    const far = pdb(residue("A", 1, 0), residue("B", 1, 50));
    expect(contactOf(far, { chain: "A" }).contacts).toBe(0);
    expect(contactProblem(far, { chain: "A" })).toContain("touches nothing");
    expect(contactProblem(far, { chain: "A" })).toContain("50.0 A");

    const close = pdb(residue("A", 1, 0), residue("B", 1, 5));
    expect(contactOf(close, { chain: "A" }).contacts).toBe(1);
    expect(contactProblem(close, { chain: "A" })).toBe(null);
  });

  it("measures to the convention's cutoff", () => {
    expect(CONTACT_ANGSTROMS).toBe(8);
    // just inside, and just outside
    expect(contactOf(pdb(residue("A", 1, 0), residue("B", 1, 7.9)), { chain: "A" }).contacts).toBe(1);
    expect(contactOf(pdb(residue("A", 1, 0), residue("B", 1, 8.1)), { chain: "A" }).contacts).toBe(0);
    // ...and a caller may ask for a tighter one
    expect(contactOf(pdb(residue("A", 1, 0), residue("B", 1, 7)),
      { chain: "A", angstroms: 5 }).contacts).toBe(0);
  });

  // 🔴 AN EPITOPE IS "TOUCH THESE RESIDUES", not "touch the target somewhere",
  // and the difference is the whole point of naming hotspots: a binder can sit
  // firmly on the opposite face and pass the weaker check.
  it("asks for the named residues, not merely the chain", () => {
    const wrongFace = pdb(residue("A", 1, 0),
      residue("B", 10, 5), residue("B", 20, 60));
    // it touches chain B...
    expect(contactProblem(wrongFace, { chain: "A" })).toBe(null);
    // ...but not residue 20, which is the one asked for
    const problem = contactProblem(wrongFace, { chain: "A", residues: { B: [20] } });
    expect(problem).toContain("0/1 hotspots");
    // and residue 10 it does touch
    expect(contactProblem(wrongFace, { chain: "A", residues: { B: [10] } })).toBe(null);
    // 🔴 and naming none is no hotspots, not "touch nothing"
    expect(contactProblem(wrongFace, { chain: "A", residues: {} })).toBe(null);
    expect(contactProblem(wrongFace, { chain: "A", residues: { B: [] } })).toBe(null);
  });

  // 🔴 A HOTSPOT THE PREDICTION DOES NOT HAVE IS REPORTED, NOT DROPPED. Asking
  // for four and silently checking three is how an epitope filter stops meaning
  // anything.
  it("counts hotspots that are not in the prediction as untouched", () => {
    const only10 = pdb(residue("A", 1, 0), residue("B", 10, 5));
    const result = contactOf(only10, { chain: "A", residues: { B: [10, 999] } });
    expect(result.wanted).toBe(2);
    expect(result.touched).toEqual(["B:10"]);
    expect(result.missing).toEqual(["B:999"]);
    expect(contactProblem(only10, { chain: "A", residues: { B: [10, 999] }, minimum: 2 }))
      .toContain("1/2 hotspots");
    // one is enough by default
    expect(contactProblem(only10, { chain: "A", residues: { B: [10, 999] } })).toBe(null);
  });

  // 🔴 A MONOMER DESIGN HAS NOTHING TO TOUCH, and this check must have no
  // opinion about it rather than vetoing every cycle of an unconditional run.
  it("says nothing about a single chain", () => {
    const alone = pdb(residue("A", 1, 0), residue("A", 2, 4));
    expect(contactProblem(alone, { chain: "A" })).toBe(null);
    expect(contactOf(alone, { chain: "A" }).nearest).toBe(Infinity);
  });
});

describe("hotspots as a pocket constraint", () => {
  // 🔴 ProteinHunter hands its hotspots to boltz2 as a pocket, conditioning the fold rather than
  // only rejecting it afterwards; the same here, for a folder that reads one
  it("names the designed chain the binder and each hotspot a contact", () => {
    expect(hotspotPocket(undefined, 0, CHAIN_IDS)).toEqual([]);
    expect(hotspotPocket({ B: [] }, 0, CHAIN_IDS)).toEqual([]);
    expect(hotspotPocket({ B: [2, 4], C: [1] }, 0, CHAIN_IDS)).toEqual([{ kind: "pocket", binder: 0,
      contacts: [{ asym: 1, residue: 2 }, { asym: 1, residue: 4 }, { asym: 2, residue: 1 }],
      maxDistance: POCKET_ANGSTROMS }]);
    expect(() => hotspotPocket({ A: [1] }, 0, CHAIN_IDS)).toThrow(/is the designed chain/);
    expect(() => hotspotPocket({ "?": [1] }, 0, CHAIN_IDS)).toThrow(/not a chain/);
  });

  it("reaches boltz2's features on exactly the binder x hotspot pairs", () => {
    const { batch } = af3BatchFromA3m("XXXX:ACDEFG", null, {
      ...featuriserDialect(dialectFor("boltz2")), modelName: "boltz2",
      constraints: hotspotPocket({ B: [2, 5] }, 0, CHAIN_IDS),
    });
    const n = batch.asymId.length;
    const at = (i, j) => batch.contactConditioning[i * n + j];
    // binder tokens 0..3, chain B residue 2 is token 5 and residue 5 is token 8
    expect(at(0, 5)).toBe(CONTACT_CLASSES["BINDER>POCKET"]);
    expect(at(8, 3)).toBe(CONTACT_CLASSES["POCKET>BINDER"]);
    expect(at(0, 6)).toBe(CONTACT_CLASSES.UNSELECTED);
    expect(batch.contactThreshold[0 * n + 5]).toBe(POCKET_ANGSTROMS);
    // a hotspot past the chain's end is refused, not dropped
    expect(() => af3BatchFromA3m("XXXX:ACDEFG", null, {
      ...featuriserDialect(dialectFor("boltz2")), modelName: "boltz2",
      constraints: hotspotPocket({ B: [9] }, 0, CHAIN_IDS),
    })).toThrow(/names no token/);
  });
});
