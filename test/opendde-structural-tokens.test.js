/**
 * The structural tokeniser on the inputs it was never folded with.
 *
 * 🔴 LIGANDS, NUCLEIC CHAINS AND COMPLEXES REACH THE TOKENISER BY
 * CONSTRUCTION, WHICH IS NOT THE SAME AS BEING TESTED. Every OpenDDE number in
 * docs/OPENDDE.md is one protein chain: 6MRR and 1QYS, MSA depth 1. The
 * regrouping's branches for a ligand's lone atoms, a nucleotide's
 * backbone/base split and a second chain's adjacency have run only in the
 * sense that nothing threw.
 *
 * They are testable without a GPU and without weights, because the tokeniser
 * is a REGROUPING: the atoms do not change, only the (token, slot) they sit
 * in. So the statements worth making are conservation laws - every real atom
 * appears exactly once, no atom changes its identity, no residue's halves get
 * different reference spaces - plus the four things a shape cannot see: which
 * role a token takes, which atom is its centre, whether a twin was recorded,
 * and whether chain adjacency stops at a chain boundary.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { featuriseProtein } from "../src/af3/featurise.js";
import {
  structuralLayout, structuralBatch, structuralToResidue, atomNameAt,
  ROLE_ATOM, ROLE_PROTEIN_BB, ROLE_PROTEIN_SC,
  ROLE_DNA_BB, ROLE_DNA_BASE, ROLE_RNA_BB, ROLE_RNA_BASE, NO_TWIN,
} from "../src/af3/structural-tokens.js";

/** Glycerol, as the featuriser wants a ligand: heavy atoms and their bonds. */
const GLYCEROL = {
  code: "GOL",
  atoms: [
    { name: "C1", element: 6, charge: 0, x: 1.2, y: 0.1, z: -0.3 },
    { name: "O1", element: 8, charge: 0, x: 2.4, y: 0.9, z: -0.1 },
    { name: "C2", element: 6, charge: 0, x: 0.0, y: 0.9, z: 0.2 },
    { name: "O2", element: 8, charge: 0, x: 0.1, y: 1.3, z: 1.6 },
    { name: "C3", element: 6, charge: 0, x: -1.3, y: 0.1, z: -0.1 },
    { name: "O3", element: 8, charge: 0, x: -2.4, y: 0.9, z: 0.3 },
  ],
  bonds: [
    { from: 0, to: 1 }, { from: 0, to: 2 }, { from: 2, to: 3 },
    { from: 2, to: 4 }, { from: 4, to: 5 },
  ],
};

/** The residue batch's live (token, slot) slots, as flat indices. */
function liveSlots(batch) {
  const live = [];
  for (let index = 0; index < batch.tokens * batch.dense; index += 1) {
    if (batch.refMask[index]) live.push(index);
  }
  return live;
}

/**
 * Every conservation law, over one batch. Asserted for each input shape below
 * rather than once, because the branch a shape takes is the thing under test.
 */
function assertConserved(name, batch) {
  const layout = structuralLayout(batch);
  const structural = structuralBatch(batch, layout);

  // 1. The regrouping is a bijection: every live residue slot is claimed by
  //    exactly one structural slot, and no structural slot claims two.
  const claims = new Map();
  for (let token = 0; token < layout.tokens; token += 1) {
    layout.sources[token].forEach((slot, at) => {
      const source = layout.parent[token] * batch.dense + slot;
      assert.equal(claims.has(source), false,
        `${name}: residue slot ${source} claimed twice`);
      claims.set(source, token * batch.dense + at);
    });
  }
  const live = liveSlots(batch);
  assert.equal(claims.size, live.length,
    `${name}: ${claims.size} atoms regrouped, ${live.length} live`);
  for (const source of live) {
    assert.ok(claims.has(source), `${name}: live atom ${source} lost`);
  }

  // 2. `residueAtomGather` is that same map, since it is what puts the
  //    diffusion's coordinates back. Two derivations of one fact is exactly
  //    the shape of failure this repository keeps meeting, so it is checked.
  for (const [source, target] of claims) {
    assert.equal(layout.residueAtomGather[source], target,
      `${name}: gather disagrees with sources at ${source}`);
  }

  // 3. No atom changes its identity in the move.
  for (const [source, target] of claims) {
    assert.equal(structural.refMask[target], batch.refMask[source], `${name}: mask`);
    assert.equal(structural.refElement[target], batch.refElement[source], `${name}: element`);
    assert.equal(structural.refCharge[target], batch.refCharge[source], `${name}: charge`);
    for (let axis = 0; axis < 3; axis += 1) {
      assert.equal(structural.refPos[target * 3 + axis], batch.refPos[source * 3 + axis],
        `${name}: position`);
    }
    for (let c = 0; c < 4; c += 1) {
      assert.equal(structural.refAtomNameChars[target * 4 + c],
        batch.refAtomNameChars[source * 4 + c], `${name}: name`);
    }
    // The residue's space, not the token's: a side chain must be allowed to
    // compare reference coordinates with its own backbone.
    assert.equal(structural.refSpaceUid[target], batch.refSpaceUid[source],
      `${name}: reference space`);
  }

  // 4. A twin is the other half of one residue, and shares its space.
  for (let token = 0; token < layout.tokens; token += 1) {
    const twin = layout.twin[token];
    if (twin === NO_TWIN) continue;
    assert.equal(layout.twin[twin], token, `${name}: twin not mutual`);
    assert.equal(layout.parent[twin], layout.parent[token], `${name}: twin of another residue`);
    assert.equal(structural.refSpaceUid[token * batch.dense],
      structural.refSpaceUid[twin * batch.dense], `${name}: twins in different spaces`);
  }

  // 5. Chain adjacency stops at a chain boundary.
  const chainOf = (token) => batch.chainOfResidue?.[batch.residueOfToken[token]] ?? 0;
  for (let token = 0; token < layout.tokens; token += 1) {
    for (const neighbour of [layout.prevParent[token], layout.nextParent[token]]) {
      if (neighbour < 0) continue;
      assert.equal(chainOf(neighbour), chainOf(layout.parent[token]),
        `${name}: adjacency crosses a chain`);
    }
  }

  // 6. The round trip through the residue layout is the identity on live
  //    atoms, because that is what every consumer downstream indexes.
  const back = structuralToResidue(structural.refPos, layout, batch.tokens, batch.dense);
  for (const source of live) {
    for (let axis = 0; axis < 3; axis += 1) {
      assert.equal(back[source * 3 + axis], batch.refPos[source * 3 + axis],
        `${name}: round trip at ${source}`);
    }
  }

  // 7. The centre is a real atom of the token it centres.
  for (let token = 0; token < layout.tokens; token += 1) {
    const slot = layout.pseudoBetaSlot[token];
    assert.ok(slot < layout.sources[token].length, `${name}: centre outside the token`);
    assert.equal(structural.refMask[token * batch.dense + slot], 1,
      `${name}: centre is not a real atom`);
  }

  return { layout, structural };
}

/** The role and centre name of every structural token of one residue token. */
function tokensOfResidue(batch, layout, structural, residueToken) {
  const out = [];
  for (let token = 0; token < layout.tokens; token += 1) {
    if (layout.parent[token] !== residueToken) continue;
    out.push({
      token,
      role: layout.role[token],
      centre: atomNameAt(structural, token, layout.pseudoBetaSlot[token]),
      atoms: layout.sources[token].map((_, at) => atomNameAt(structural, token, at)),
      twin: layout.twin[token],
    });
  }
  return out;
}

describe("a ligand goes through as one role-0 token per atom", () => {
  const batch = featuriseProtein("GAWSTLAK", { ligands: [GLYCEROL] });

  it("conserves every atom", () => {
    assertConserved("ligand", batch);
  });

  it("gives each ligand atom its own token, its own centre and no twin", () => {
    const { layout, structural } = assertConserved("ligand", batch);
    const span = batch.ligandSpans[0];
    assert.equal(span.count, GLYCEROL.atoms.length);
    const seen = [];
    for (let token = 0; token < layout.tokens; token += 1) {
      const parent = layout.parent[token];
      if (parent < span.from || parent >= span.from + span.count) continue;
      assert.equal(layout.role[token], ROLE_ATOM, "a ligand atom is role 0");
      assert.equal(layout.twin[token], NO_TWIN, "ligand atoms are not twins");
      assert.equal(layout.sources[token].length, 1, "one atom to a token");
      assert.equal(layout.pseudoBetaSlot[token], 0, "and it is its own centre");
      seen.push(atomNameAt(structural, token, 0));
    }
    assert.deepEqual(seen, GLYCEROL.atoms.map((atom) => atom.name));
  });

  it("carries the ligand's bonds onto the structural tokens", () => {
    const { layout, structural } = assertConserved("ligand", batch);
    const structuralOf = new Map();
    for (let token = 0; token < layout.tokens; token += 1) {
      if (layout.role[token] === ROLE_ATOM) structuralOf.set(layout.parent[token], token);
    }
    const span = batch.ligandSpans[0];
    for (const bond of GLYCEROL.bonds) {
      const from = structuralOf.get(span.from + bond.from);
      const to = structuralOf.get(span.from + bond.to);
      assert.equal(structural.bondMatrix[from * layout.tokens + to], 1,
        `bond ${bond.from}-${bond.to} lost`);
    }
  });
});

describe("a nucleic chain splits backbone from base", () => {
  const dna = featuriseProtein("ACGT", { chainKinds: ["dna"] });
  const rna = featuriseProtein("ACGU", { chainKinds: ["rna"] });

  it("conserves every atom of DNA and of RNA", () => {
    assertConserved("dna", dna);
    assertConserved("rna", rna);
  });

  it("takes the nucleic roles, not the protein ones", () => {
    for (const [name, batch, bb, base] of [
      ["dna", dna, ROLE_DNA_BB, ROLE_DNA_BASE],
      ["rna", rna, ROLE_RNA_BB, ROLE_RNA_BASE],
    ]) {
      const { layout } = assertConserved(name, batch);
      assert.equal(layout.tokens, 2 * batch.tokens, `${name}: every nucleotide splits`);
      for (let token = 0; token < layout.tokens; token += 1) {
        assert.ok(layout.role[token] === bb || layout.role[token] === base,
          `${name}: role ${layout.role[token]} is not nucleic`);
        assert.notEqual(layout.twin[token], NO_TWIN, `${name}: a half without its twin`);
      }
    }
  });

  it("puts the phosphate with the backbone and the ring with the base", () => {
    const { layout, structural } = assertConserved("dna", dna);
    const [backbone, base] = tokensOfResidue(dna, layout, structural, 0);
    assert.equal(backbone.role, ROLE_DNA_BB);
    assert.equal(base.role, ROLE_DNA_BASE);
    assert.ok(backbone.atoms.includes("P"), "P belongs to the backbone");
    assert.ok(backbone.atoms.includes("C1'"), "C1' belongs to the backbone");
    assert.equal(base.atoms.includes("P"), false, "no phosphate in the base");
    assert.ok(base.atoms.includes("N9"), "adenine's N9 belongs to the base");
  });

  /**
   * 🔴 THE ONE THING A PROTEIN TEST CANNOT SEE. A purine carries an N1 as well
   * as an N9 - in its six-membered ring - so a single ["N1", "N9", ...]
   * preference silently centres every A, G, DA and DG on the wrong ring.
   */
  it("centres a purine on N9 and a pyrimidine on N1", () => {
    const expected = { dna: ["N9", "N1", "N9", "N1"], rna: ["N9", "N1", "N9", "N1"] };
    for (const [name, batch] of [["dna", dna], ["rna", rna]]) {
      const { layout, structural } = assertConserved(name, batch);
      const centres = [];
      for (let residue = 0; residue < batch.tokens; residue += 1) {
        const [, base] = tokensOfResidue(batch, layout, structural, residue);
        centres.push(base.centre);
      }
      assert.deepEqual(centres, expected[name], `${name}: base centres`);
    }
  });

  it("centres a nucleic backbone on C4'", () => {
    const { layout, structural } = assertConserved("dna", dna);
    for (let residue = 0; residue < dna.tokens; residue += 1) {
      const [backbone] = tokensOfResidue(dna, layout, structural, residue);
      assert.equal(backbone.centre, "C4'");
    }
  });
});

describe("a complex keeps its chains apart", () => {
  const complex = featuriseProtein("GAWSTLAK:ACGT",
    { chainKinds: ["protein", "dna"] });

  it("conserves every atom across both chains", () => {
    assertConserved("complex", complex);
  });

  it("tokenises each chain by its own kind", () => {
    const { layout } = assertConserved("complex", complex);
    const proteinRoles = new Set([ROLE_PROTEIN_BB, ROLE_PROTEIN_SC]);
    const dnaRoles = new Set([ROLE_DNA_BB, ROLE_DNA_BASE]);
    for (let token = 0; token < layout.tokens; token += 1) {
      const chain = complex.chainOfResidue[complex.residueOfToken[layout.parent[token]]];
      const roles = chain === 0 ? proteinRoles : dnaRoles;
      assert.ok(roles.has(layout.role[token]),
        `chain ${chain} took role ${layout.role[token]}`);
    }
  });

  it("does not make the last residue of one chain adjacent to the first of the next", () => {
    const { layout } = assertConserved("complex", complex);
    const boundary = 8;  // the protein chain's length, in residue tokens
    for (let token = 0; token < layout.tokens; token += 1) {
      if (layout.parent[token] === boundary - 1) {
        assert.equal(layout.nextParent[token], -1, "the protein's last residue has no next");
      }
      if (layout.parent[token] === boundary) {
        assert.equal(layout.prevParent[token], -1, "the DNA's first residue has no previous");
      }
    }
  });
});

describe("a protein residue splits only when both halves are real", () => {
  const batch = featuriseProtein("GAWSTLAK");
  const { layout, structural } = assertConserved("protein", batch);

  it("leaves glycine whole, as one backbone token", () => {
    const glycine = tokensOfResidue(batch, layout, structural, 0);
    assert.equal(glycine.length, 1);
    assert.equal(glycine[0].role, ROLE_PROTEIN_BB);
    assert.equal(glycine[0].twin, NO_TWIN);
    assert.equal(glycine[0].centre, "CA");
    assert.deepEqual(new Set(glycine[0].atoms), new Set(["N", "CA", "C", "O"]));
  });

  it("splits alanine into a backbone and a CB-centred sidechain", () => {
    const alanine = tokensOfResidue(batch, layout, structural, 1);
    assert.equal(alanine.length, 2);
    assert.deepEqual(alanine.map((one) => one.role), [ROLE_PROTEIN_BB, ROLE_PROTEIN_SC]);
    assert.equal(alanine[0].centre, "CA");
    assert.equal(alanine[1].centre, "CB");
    assert.deepEqual(alanine[1].atoms, ["CB"]);
    assert.equal(alanine[0].twin, alanine[1].token);
  });

  it("keeps the terminal OXT with the backbone", () => {
    const last = tokensOfResidue(batch, layout, structural, batch.tokens - 1);
    assert.ok(last[0].atoms.includes("OXT"), "OXT is a backbone atom");
  });
});

describe("a modified residue atomises like a ligand and stays in its chain", () => {
  /** Phosphoserine's heavy atoms, as the CCD gives them. */
  const SEP = {
    chain: 0, position: 3, code: "SEP",
    atoms: [
      { name: "N", element: 7, charge: 0, x: 1.0, y: 0.0, z: 0.0 },
      { name: "CA", element: 6, charge: 0, x: 2.0, y: 0.5, z: 0.0 },
      { name: "CB", element: 6, charge: 0, x: 3.0, y: 0.0, z: 1.0 },
      { name: "OG", element: 8, charge: 0, x: 4.0, y: 0.5, z: 1.0 },
      { name: "P", element: 15, charge: 0, x: 5.0, y: 0.0, z: 2.0 },
      { name: "O1P", element: 8, charge: -1, x: 6.0, y: 0.5, z: 2.0 },
      { name: "O2P", element: 8, charge: -1, x: 5.0, y: -1.4, z: 2.0 },
      { name: "O3P", element: 8, charge: 0, x: 5.0, y: 0.5, z: 3.3 },
      { name: "C", element: 6, charge: 0, x: 2.5, y: 1.9, z: 0.0 },
      { name: "O", element: 8, charge: 0, x: 3.7, y: 2.1, z: 0.0 },
      { name: "OXT", element: 8, charge: 0, x: 1.7, y: 2.9, z: 0.0 },
    ],
    bonds: [],
  };
  const batch = featuriseProtein("GAWSTLAK", { modifications: [SEP] });

  it("conserves every atom", () => {
    assertConserved("modified", batch);
  });

  it("gives each of its atoms a role-0 token in the protein chain", () => {
    const { layout } = assertConserved("modified", batch);
    const span = batch.modifiedSpans[0];
    let counted = 0;
    for (let token = 0; token < layout.tokens; token += 1) {
      const parent = layout.parent[token];
      if (parent < span.from || parent >= span.from + span.count) continue;
      counted += 1;
      assert.equal(layout.role[token], ROLE_ATOM, "a modified residue's atom is role 0");
      assert.equal(layout.twin[token], NO_TWIN, "and has no twin");
      // ...but it is still in the chain, so it keeps its neighbours.
      assert.notEqual(layout.prevParent[token], -1, "a mid-chain atom has a previous");
      assert.notEqual(layout.nextParent[token], -1, "and a next");
    }
    assert.equal(counted, span.count);
  });
});
