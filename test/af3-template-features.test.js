import { describe, expect, it } from "./harness.js";
import {
  AF2_ATOM37, AF3_DENSE, BACKBONE_SLOTS, DGRAM_BINS, NUM_DENSE, PSEUDO_BETA_SLOT,
  backboneFrames, coverageOf, distogram, multichainMaskFor, pseudoBeta,
  templateGeometry,
} from "../src/af3/template-features.js";

/**
 * These are the geometry features the template embedder used to refuse to
 * compute. `tools/oracle/check_af3_template_geometry.js` holds them to AF3
 * itself and is the real check; this file exists because that one needs a
 * 470 KB dump and a CPU jax run, and the three things below are exactly the
 * ones that were wrong or nearly wrong while everything else agreed.
 */

/** A residue's dense atoms: N, CA, C, O, CB in slots 0..4. */
function residue(atoms) {
  const positions = new Float32Array(NUM_DENSE * 3);
  const mask = new Float32Array(NUM_DENSE);
  for (const [slot, point] of Object.entries(atoms)) {
    positions[slot * 3] = point[0];
    positions[slot * 3 + 1] = point[1];
    positions[slot * 3 + 2] = point[2];
    mask[slot] = 1;
  }
  return { positions, mask };
}

function template(residues, aatypes) {
  const tokens = residues.length;
  const atomPositions = new Float32Array(tokens * NUM_DENSE * 3);
  const atomMask = new Float32Array(tokens * NUM_DENSE);
  residues.forEach((one, index) => {
    atomPositions.set(one.positions, index * NUM_DENSE * 3);
    atomMask.set(one.mask, index * NUM_DENSE);
  });
  return { aatype: Int32Array.from(aatypes), atomPositions, atomMask };
}

/** An axis-aligned residue at `origin`: CA there, C on +x, N on +y. */
const aligned = (origin, cb) => residue({
  0: [origin[0], origin[1] + 1, origin[2]],
  1: origin,
  2: [origin[0] + 1, origin[1], origin[2]],
  ...(cb ? { 4: cb } : {}),
});

describe("the constant tables", () => {
  it("gives every amino acid CB and glycine CA", () => {
    expect(PSEUDO_BETA_SLOT).toHaveLength(31);
    for (let code = 0; code < 20; code += 1) {
      expect(PSEUDO_BETA_SLOT[code]).toBe(code === 7 ? 1 : 4);
    }
  });

  // 🔴 (C, CA, N), NOT (N, CA, C). AF3 reads the SIDE-CHAIN frame table's
  // group 0, whose convention is reversed, and undoes it by naming its locals
  // `c, b, a`. Read the other way the frame is reflected through its own CA.
  it("orders the backbone frame as (C, CA, N)", () => {
    expect(BACKBONE_SLOTS).toHaveLength(31);
    for (let code = 0; code < 20; code += 1) expect(BACKBONE_SLOTS[code]).toEqual([2, 1, 0]);
  });
});

describe("pseudoBeta", () => {
  it("takes CB, and CA for glycine", () => {
    const one = aligned([0, 0, 0], [5, 5, 5]);
    const beta = pseudoBeta(Int32Array.from([0]), one.positions, one.mask, 1);
    expect([...beta.positions]).toEqual([5, 5, 5]);
    const glycine = pseudoBeta(Int32Array.from([7]), one.positions, one.mask, 1);
    expect([...glycine.positions]).toEqual([0, 0, 0]);
  });

  it("is unmasked where the residue has no such atom", () => {
    const noCb = aligned([0, 0, 0]);
    expect(pseudoBeta(Int32Array.from([0]), noCb.positions, noCb.mask, 1).mask[0]).toBe(0);
    expect(pseudoBeta(Int32Array.from([7]), noCb.positions, noCb.mask, 1).mask[0]).toBe(1);
  });
});

describe("distogram", () => {
  const binOf = (row) => {
    for (let bin = 0; bin < DGRAM_BINS; bin += 1) if (row[bin] === 1) return bin;
    return -1;
  };
  const at = (distance) => {
    const positions = Float32Array.from([0, 0, 0, distance, 0, 0]);
    return binOf(distogram(positions, 2).subarray(DGRAM_BINS, 2 * DGRAM_BINS));
  };

  // 🔴 THE USUAL CLAMPED BUCKETISATION IS WRONG HERE. AF3 builds this as
  // `(d2 > lower) * (d2 < upper)`, so a distance below the FIRST break fails
  // every test and the row is all zero - it does not fall into bin 0. Writing
  // the usual clamp puts every close contact, which is every pair the model
  // cares most about, one bin too high.
  it("puts a pair closer than 3.25 A in no bin at all", () => {
    expect(at(0)).toBe(-1);
    expect(at(3)).toBe(-1);
    expect(at(3.24)).toBe(-1);
    expect(at(3.3)).toBe(0);
  });

  it("catches everything past the last break in the last bin", () => {
    expect(at(51)).toBe(DGRAM_BINS - 1);
    expect(at(500)).toBe(DGRAM_BINS - 1);
  });

  it("walks the bins with distance, one-hot throughout", () => {
    let previous = -1;
    for (let distance = 3.3; distance < 50; distance += 1.5) {
      const bin = at(distance);
      expect(bin >= previous).toBe(true);
      previous = bin;
    }
    const rows = distogram(Float32Array.from([0, 0, 0, 10, 0, 0]), 2);
    for (let pair = 0; pair < 4; pair += 1) {
      let total = 0;
      for (let bin = 0; bin < DGRAM_BINS; bin += 1) total += rows[pair * DGRAM_BINS + bin];
      // The diagonal is distance zero, which is in no bin; the off-diagonal is.
      expect(total).toBe(pair === 0 || pair === 3 ? 0 : 1);
    }
  });
});

describe("backboneFrames", () => {
  it("puts CA at the origin of its own frame and C on the +x axis", () => {
    const one = aligned([7, -2, 3]);
    const frames = backboneFrames(Int32Array.from([0]), one.positions, one.mask, 1);
    expect([...frames.translations]).toEqual([7, -2, 3]);
    // The first column is e1, along C - CA, which here is +x.
    expect(frames.rotations[0]).toBeCloseTo(1, 6);
    expect(frames.rotations[3]).toBeCloseTo(0, 6);
    expect(frames.mask[0]).toBe(1);
  });

  it("has no frame where one of the three atoms is missing", () => {
    const partial = residue({ 1: [0, 0, 0], 2: [1, 0, 0] });   // CA and C, no N
    expect(backboneFrames(Int32Array.from([0]), partial.positions, partial.mask, 1)
      .mask[0]).toBe(0);
  });
});

describe("templateGeometry", () => {
  const ones = (tokens) => new Float32Array(tokens * tokens).fill(1);

  /**
   * 🔴 THE ONE THAT WAS ACTUALLY WRONG. `unit_vector[i][j]` is
   * `R_i^-1 (t_j - t_i)` - the FRAME is the row and the POINT is the column,
   * because AF3 writes `rigid[:, None].inverse().apply_to_point(points)` and
   * the broadcast puts frames on axis 0. Written the other way it is the exact
   * transpose: still unit length, still smooth, still masked right. Measured
   * against AF3, the transposed reading scored relRMS 1.5 and the correct one
   * 1.3e-7, with the distogram and both masks bit-exact either way.
   */
  it("expresses j in i's frame, not i in j's", () => {
    // Two residues whose frames are the identity, 10 A apart along +x.
    const geometry = templateGeometry(
      template([aligned([0, 0, 0], [0, 0, 0]), aligned([10, 0, 0], [10, 0, 0])], [0, 0]),
      ones(2), 2);
    const uv = (i, j) => [...geometry.unitVector.slice((i * 2 + j) * 3, (i * 2 + j) * 3 + 3)];
    // Seen from residue 0, residue 1 is in the +x direction.
    expect(uv(0, 1)[0]).toBeCloseTo(1, 5);
    // ...and seen from residue 1, residue 0 is in -x. Transposed, these swap.
    expect(uv(1, 0)[0]).toBeCloseTo(-1, 5);
  });

  it("leaves the diagonal finite rather than dividing by zero", () => {
    const geometry = templateGeometry(
      template([aligned([0, 0, 0], [0, 0, 0])], [0]), ones(1), 1);
    for (const value of geometry.unitVector) expect(Number.isFinite(value)).toBe(true);
  });

  // A template covers ONE chain, so a distance between two chains of the query
  // is not something it knows - and unmasked it is a real number computed from
  // two structures that were never in the same frame.
  it("masks every feature across a chain boundary", () => {
    const chains = Float32Array.from([1, 0, 0, 1]);
    const geometry = templateGeometry(
      template([aligned([0, 0, 0], [0, 0, 0]), aligned([10, 0, 0], [10, 0, 0])], [0, 0]),
      chains, 2);
    expect(geometry.pseudoBetaMask2d[1]).toBe(0);
    expect(geometry.backboneMask2d[1]).toBe(0);
    expect(geometry.unitVector[3]).toBe(0);
    for (let bin = 0; bin < DGRAM_BINS; bin += 1) {
      expect(geometry.distogram[DGRAM_BINS + bin]).toBe(0);
    }
    // ...and the intra-chain pairs are untouched.
    expect(geometry.pseudoBetaMask2d[0]).toBe(1);
  });

  // 🔴 AN UNRESOLVED ATOM'S COORDINATES REACH THE DISTOGRAM THROUGH THE GATHER
  // EVEN THOUGH ITS MASK IS ZERO - the mask only zeroes the OUTPUT, and a bin
  // computed from a stale coordinate is still a bin. AF3 multiplies the
  // positions by the mask before anything reads them.
  it("zeroes an unresolved atom's coordinates before reading them", () => {
    const stale = template([aligned([0, 0, 0], [0, 0, 0]), aligned([10, 0, 0])], [0, 0]);
    // Residue 1 has no CB, but leave a stale coordinate in that slot.
    stale.atomPositions[(NUM_DENSE + 4) * 3] = 999;
    const geometry = templateGeometry(stale, ones(2), 2);
    // Its pseudo-beta is unmasked, so nothing it touched can be nonzero.
    expect(geometry.pseudoBetaMask2d[1]).toBe(0);
    for (let bin = 0; bin < DGRAM_BINS; bin += 1) {
      expect(geometry.distogram[DGRAM_BINS + bin]).toBe(0);
    }
  });
});

describe("multichainMaskFor", () => {
  // Two chains of two tokens each.
  const asymId = Int32Array.from([1, 1, 2, 2]);
  const covered = Float32Array.from([1, 1, 1, 1]);
  const at = (mask, i, j) => mask[i * 4 + j];

  /**
   * 🔴 AF3 MASKS ACROSS CHAINS FOR A REASON ABOUT PROVENANCE. Its `Template`
   * is one protein chain and a complex's chains are templated by SEPARATE
   * searches, so a cross-chain distance is computed from two structures that
   * were never in one frame. Measured on a two-chain query with a template on
   * each chain: masking as AF3 does reproduces it to relRMS 5.5e-7, and
   * leaving the cross-chain pairs open scores 1.09.
   */
  it("closes cross-chain pairs by default", () => {
    const mask = multichainMaskFor(asymId, 4, { coverage: covered });
    expect(at(mask, 0, 1)).toBe(1);
    expect(at(mask, 2, 3)).toBe(1);
    expect(at(mask, 0, 2)).toBe(0);
    expect(at(mask, 3, 1)).toBe(0);
  });

  // ...but when both chains came from ONE file they ARE in one frame, and the
  // cross-chain distances are the interface geometry a binder method wants.
  it("opens them when one structure covered both chains", () => {
    const mask = multichainMaskFor(asymId, 4, { coverage: covered, spanChains: true });
    expect(at(mask, 0, 2)).toBe(1);
    expect(at(mask, 3, 1)).toBe(1);
  });

  // A pair with one end outside the template is still two frames apart, so
  // spanning opens only what the slot actually covers at BOTH ends.
  it("opens only the pairs it covers at both ends", () => {
    const partial = Float32Array.from([1, 1, 1, 0]);
    const mask = multichainMaskFor(asymId, 4, { coverage: partial, spanChains: true });
    expect(at(mask, 0, 2)).toBe(1);
    expect(at(mask, 0, 3)).toBe(0);
    // ...and an intra-chain pair stays open whether or not it is covered.
    expect(at(mask, 2, 3)).toBe(1);
  });

  it("cannot span without knowing what it covers", () => {
    const mask = multichainMaskFor(asymId, 4, { spanChains: true });
    expect(at(mask, 0, 2)).toBe(0);
  });
});

describe("coverageOf", () => {
  it("reads coverage off the atom mask, so the two cannot disagree", () => {
    const template = { atomMask: new Float32Array(3 * NUM_DENSE) };
    template.atomMask[0 * NUM_DENSE + 1] = 1;
    template.atomMask[2 * NUM_DENSE + 4] = 1;
    expect([...coverageOf(template, 3)]).toEqual([1, 0, 1]);
  });
});

describe("the AF2 atom37 layout", () => {
  /** A residue in atom37: N=0, CA=1, C=2, CB=3, O=4. */
  const atom37 = (origin, cb) => {
    const positions = new Float32Array(37 * 3);
    const mask = new Float32Array(37);
    const put = (slot, point) => {
      positions[slot * 3] = point[0];
      positions[slot * 3 + 1] = point[1];
      positions[slot * 3 + 2] = point[2];
      mask[slot] = 1;
    };
    put(0, [origin[0], origin[1] + 1, origin[2]]);
    put(1, origin);
    put(2, [origin[0] + 1, origin[1], origin[2]]);
    if (cb) put(3, cb);
    return { positions, mask };
  };

  // 🔴 CB IS SLOT 3 HERE AND SLOT 4 IN AF3'S DENSE LAYOUT, because atom37 is
  // N, CA, C, CB, O and the dense one is N, CA, C, O, CB. Reading the wrong
  // one takes the backbone oxygen as the pseudo-beta: a real coordinate, about
  // a bond length from the right answer, which moves a short-range distogram
  // bin and nothing at long range.
  it("takes CB from slot 3, not slot 4", () => {
    const one = atom37([0, 0, 0], [5, 5, 5]);
    const beta = pseudoBeta(Int32Array.from([0]), one.positions, one.mask, 1, AF2_ATOM37);
    expect([...beta.positions]).toEqual([5, 5, 5]);
    expect(beta.mask[0]).toBe(1);
  });

  it("still gives glycine CA", () => {
    const one = atom37([1, 2, 3], [5, 5, 5]);
    const beta = pseudoBeta(Int32Array.from([7]), one.positions, one.mask, 1, AF2_ATOM37);
    expect([...beta.positions]).toEqual([1, 2, 3]);
  });

  // The frame is `from_two_vectors(C - CA, N - CA)` translated to CA in BOTH
  // models - AF3 through the side-chain table with its reversed convention,
  // AF2 through make_transform_from_reference(N, CA, C). Same slots, too.
  it("builds the same frame as AF3 does, from the same three slots", () => {
    const one = atom37([7, -2, 3]);
    const frames = backboneFrames(
      Int32Array.from([0]), one.positions, one.mask, 1, AF2_ATOM37);
    expect([...frames.translations]).toEqual([7, -2, 3]);
    expect(frames.rotations[0]).toBeCloseTo(1, 6);
    expect(frames.mask[0]).toBe(1);
    expect(AF2_ATOM37.backbone).toEqual(AF3_DENSE.backbone);
  });

  it("reads coverage over 37 slots rather than 24", () => {
    const template = { atomMask: new Float32Array(2 * 37) };
    template.atomMask[37 + 3] = 1;
    expect([...coverageOf(template, 2, AF2_ATOM37)]).toEqual([0, 1]);
  });
});
