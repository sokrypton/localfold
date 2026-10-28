import { describe, expect, it } from "./harness.js";
import { featuriseProtein } from "../shared/af3/featurise/featurise.js";
import { pdbSlots, toPdb } from "../shared/af3/structure/pdb.js";
import { startFromPdb } from "../shared/af3/structure/warm-start.js";

// 🔴 A PREVIOUS FOLD, LAID OUT FOR A MUTATED SEQUENCE (2026-10-28): Protein Hunter's warm start.
// Every atom of an unchanged residue comes back where it was; a mutated residue keeps its backbone
// and CB and puts the atoms it did not have on its C-alpha.

/** Coordinates that name their own slot, so a copied atom says where it came from. */
function folded(batch) {
  const positions = new Float32Array(batch.tokens * batch.dense * 3);
  for (let slot = 0; slot < batch.tokens * batch.dense; slot += 1) positions.set([slot, slot % 7, 1], slot * 3);
  return toPdb(batch, positions);
}
const byName = (batch, positions) => new Map(pdbSlots(batch).map(({ slot, atomName, resSeq }) =>
  [`${resSeq}:${atomName}`, [...positions.slice(slot * 3, slot * 3 + 3)]]));

describe("a warm start from the previous cycle", () => {
  const before = featuriseProtein("AGKVL");
  const pdb = folded(before);
  const old = new Map(pdb.split("\n").filter((line) => line.startsWith("ATOM")).map((line) =>
    [`${Number(line.slice(22, 26))}:${line.slice(12, 16).trim()}`,
      [Number(line.slice(30, 38)), Number(line.slice(38, 46)), Number(line.slice(46, 54))]]));

  it("copies every atom of an unchanged residue", () => {
    const after = featuriseProtein("AGKVL");
    const start = startFromPdb(after, pdb);
    expect(start.placed).toBe(0);
    for (const [key, at] of byName(after, start.positions)) expect(at).toEqual(old.get(key));
  });

  it("keeps a mutated residue's backbone and CB and puts the rest on its C-alpha", () => {
    const after = featuriseProtein("AGRVL");   // K3 -> R3
    const start = startFromPdb(after, pdb);
    const now = byName(after, start.positions);
    for (const atom of ["N", "CA", "C", "O", "CB"]) expect(now.get(`3:${atom}`)).toEqual(old.get(`3:${atom}`));
    // arginine's guanidinium was never there: it starts on the C-alpha
    for (const atom of ["NE", "CZ", "NH1", "NH2"]) expect(now.get(`3:${atom}`)).toEqual(old.get("3:CA"));
    // ...and K's CG, CD, CE do not leak into R's slots of the same name: they are a different residue's
    expect(now.get("3:CG")).toEqual(old.get("3:CA"));
    expect(start.kept).toBe(5);
    expect(now.get("4:CG1")).toEqual(old.get("4:CG1"));
  });

  it("refuses a previous fold of a different job", () => {
    expect(() => startFromPdb(featuriseProtein("AGKVLA"), pdb)).toThrow(/no residue 6 of chain A/);
  });
});
