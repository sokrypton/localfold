import { describe, expect, it } from "./harness.js";
import { residueIdentity, widenRelativePositionWeight } from "../src/multimer/input-embedder.js";
import { residueIndexWithChainBreaks } from "../src/input/chains.js";

const MAX_RELATIVE = 32;
const MAX_RELATIVE_CHAIN = 2;

/** The 65-row lookup the pair shader used before the graph was widened. */
function monomerRows(residueIndex, i, j) {
  const raw = residueIndex[i] - residueIndex[j] + MAX_RELATIVE;
  return [Math.max(0, Math.min(raw, 2 * MAX_RELATIVE))];
}

/** A mirror of the widened shader: which weight rows a pair (i, j) reads. */
function multimerRows(identity, length, i, j) {
  const at = (lane, residue) => identity[lane * length + residue];
  const asymSame = at(1, i) === at(1, j);
  const entitySame = at(2, i) === at(2, j);
  const raw = at(0, i) - at(0, j) + MAX_RELATIVE;
  const clipped = Math.max(0, Math.min(raw, 2 * MAX_RELATIVE));
  const offsetRow = asymSame ? clipped : 2 * MAX_RELATIVE + 1;
  const entityRow = 2 * MAX_RELATIVE + 2;
  const delta = at(3, i) - at(3, j);
  const clippedChain = Math.max(0, Math.min(delta + MAX_RELATIVE_CHAIN, 2 * MAX_RELATIVE_CHAIN));
  const chainRow = entitySame ? clippedChain : 2 * MAX_RELATIVE_CHAIN + 1;
  const rows = [offsetRow];
  if (entitySame) rows.push(entityRow);
  rows.push(entityRow + 1 + chainRow);
  return rows;
}

describe("widening the relative-position table", () => {
  it("takes a monomer's 65 rows to 73, leaving the originals in place", () => {
    const weight = Float32Array.from({ length: 65 * 4 }, (_, index) => index + 1);
    const widened = widenRelativePositionWeight(weight, 4);
    expect(widened.length).toBe(73 * 4);
    for (let index = 0; index < 65 * 4; index += 1) expect(widened[index]).toBe(weight[index]);
    for (let index = 65 * 4; index < 73 * 4; index += 1) expect(widened[index]).toBe(0);
  });

  it("passes a multimer's 73 rows through untouched", () => {
    const weight = new Float32Array(73 * 4).fill(2);
    expect(widenRelativePositionWeight(weight, 4)).toBe(weight);
  });

  it("refuses a table that is neither", () => {
    expect(() => widenRelativePositionWeight(new Float32Array(70 * 4), 4)).toThrow(/70 rows/);
  });
});

describe("the widened relative encoding", () => {
  it("reduces to the monomer lookup for a single chain", () => {
    // 🔴 THE CLAIM THE WHOLE SUPERSET RESTS ON. One chain must read the same
    // offset row it always did, plus rows that are zero in widened monomer
    // weights - so the fold is unchanged without a monomer branch anywhere.
    const length = 40;
    const residueIndex = residueIndexWithChainBreaks(length, undefined);
    const identity = residueIdentity({ length, residueIndex });
    for (let i = 0; i < length; i += 1) for (let j = 0; j < length; j += 1) {
      const [offsetRow, ...extra] = multimerRows(identity, length, i, j);
      expect(offsetRow).toBe(monomerRows(residueIndex, i, j)[0]);
      // the extra rows are 66 (entity-same) and 69 (zero symmetry delta),
      // both inside the zero-padded region of a converted monomer table
      expect(extra).toEqual([66, 69]);
      for (const row of extra) expect(row).toBeGreaterThan(64);
    }
  });

  it("sends cross-chain pairs to the dedicated bin, not to a clipped offset", () => {
    const length = 6;
    const residueIndex = residueIndexWithChainBreaks(length, [3, 3]);
    const identity = residueIdentity({
      length, residueIndex,
      asymId: Float32Array.from([0, 0, 0, 1, 1, 1]),
      entityId: Float32Array.from([0, 0, 0, 0, 0, 0]),
      symId: Float32Array.from([0, 0, 0, 1, 1, 1]),
    });
    // within a chain: an ordinary offset row
    expect(multimerRows(identity, length, 0, 1)[0]).toBe(MAX_RELATIVE - 1);
    // across chains: row 65, the "different chain entirely" bin
    expect(multimerRows(identity, length, 0, 4)[0]).toBe(2 * MAX_RELATIVE + 1);
  });

  it("reads the symmetry delta for copies of one entity", () => {
    const length = 4;
    const identity = residueIdentity({
      length, residueIndex: Float32Array.from([0, 1, 2, 3]),
      asymId: Float32Array.from([0, 0, 1, 1]),
      entityId: Float32Array.from([0, 0, 0, 0]),
      symId: Float32Array.from([0, 0, 1, 1]),
    });
    // same entity, symmetry delta -1 -> clipped 1 -> row 67 + 1
    expect(multimerRows(identity, length, 0, 2).at(-1)).toBe(68);
    // same entity, delta 0 -> row 67 + 2
    expect(multimerRows(identity, length, 0, 1).at(-1)).toBe(69);
  });

  it("drops the entity row and takes the last chain bin for different entities", () => {
    const length = 4;
    const identity = residueIdentity({
      length, residueIndex: Float32Array.from([0, 1, 2, 3]),
      asymId: Float32Array.from([0, 0, 1, 1]),
      entityId: Float32Array.from([0, 0, 1, 1]),
      symId: Float32Array.from([0, 0, 0, 0]),
    });
    const rows = multimerRows(identity, length, 0, 2);
    expect(rows.includes(66)).toBe(false);
    expect(rows.at(-1)).toBe(67 + 2 * MAX_RELATIVE_CHAIN + 1);
  });

  it("never indexes past the 73 rows the table has", () => {
    const length = 8;
    const identity = residueIdentity({
      length, residueIndex: residueIndexWithChainBreaks(length, [2, 2, 2, 2]),
      asymId: Float32Array.from([0, 0, 1, 1, 2, 2, 3, 3]),
      entityId: Float32Array.from([0, 0, 0, 0, 1, 1, 1, 1]),
      symId: Float32Array.from([0, 0, 1, 1, 0, 0, 1, 1]),
    });
    for (let i = 0; i < length; i += 1) for (let j = 0; j < length; j += 1) {
      for (const row of multimerRows(identity, length, i, j)) {
        expect(row).toBeGreaterThan(-1);
        expect(row).toBeLessThan(73);
      }
    }
  });
});

describe("the residue identity buffer", () => {
  it("is all zeros beyond the residue index for a monomer", () => {
    const identity = residueIdentity({ length: 3, residueIndex: Float32Array.from([0, 1, 2]) });
    expect(Array.from(identity)).toEqual([0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it("lays the three chain lanes after the index", () => {
    const identity = residueIdentity({
      length: 2, residueIndex: Float32Array.from([0, 1]),
      asymId: Float32Array.from([0, 1]),
      entityId: Float32Array.from([0, 0]),
      symId: Float32Array.from([0, 1]),
    });
    expect(Array.from(identity)).toEqual([0, 1, 0, 1, 0, 0, 0, 1]);
  });

  it("refuses a lane that is not one value per residue", () => {
    expect(() => residueIdentity({
      length: 3, residueIndex: Float32Array.from([0, 1, 2]), asymId: Float32Array.from([0]),
    })).toThrow(/one value per residue/);
  });
});
