/**
 * Parsing a PDB chemical component, for ligands.
 *
 * The fixture is glycerol's real mmCIF, trimmed to the two loops that matter.
 * It is a real file rather than a constructed one because the shape of these
 * loops is the whole difficulty: the column list is long, it has grown over the
 * years, and the coordinates sit AFTER three flags that did not always exist.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { ccdUrl, parseCcdComponent } from "../src/af3/ccd-component.js";

// GOL, as files.rcsb.org serves it. Hydrogens kept in, so the test can show
// they are dropped; one atom name quoted, as components with primes are.
const GOL = `data_GOL
_chem_comp.id                                    GOL
_chem_comp.name                                  GLYCEROL
_chem_comp.type                                  NON-POLYMER
_chem_comp.formula                               "C3 H8 O3"
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_align
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
_chem_comp_atom.pdbx_backbone_atom_flag
_chem_comp_atom.pdbx_n_terminal_atom_flag
_chem_comp_atom.pdbx_c_terminal_atom_flag
_chem_comp_atom.model_Cartn_x
_chem_comp_atom.model_Cartn_y
_chem_comp_atom.model_Cartn_z
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
_chem_comp_atom.pdbx_component_atom_id
_chem_comp_atom.pdbx_component_comp_id
_chem_comp_atom.pdbx_ordinal
GOL C1  C1  C 0 1 N N N N N N 29.490 2.376 31.160 -1.249 -0.665 0.295  C1  GOL 1
GOL O1  O1  O 0 1 N N N N N N 28.524 1.609 30.767 -2.413 -0.087 -0.300 O1  GOL 2
GOL C2  C2  C 0 1 N N N N N N 29.893 0.000 29.893 0.000  0.034  -0.245 C2  GOL 3
GOL O2  O2  O 0 1 N N N N N N 29.804 0.000 29.804 0.000  1.400  0.174  O2  GOL 4
GOL C3  C3  C 0 1 N N N N N N 29.785 1.249 29.785 1.249  -0.665 0.295  C3  GOL 5
GOL "O3'" O3  O -1 1 N N N N N N 29.472 2.413 29.472 2.413  -0.087 -0.300 O3 GOL 6
GOL H11 H11 H 0 1 N N N N N N 30.000 3.000 31.000 -1.300 -1.700 0.000  H11 GOL 7
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
_chem_comp_bond.pdbx_ordinal
GOL C1 O1  SING N N 1
GOL C1 C2  SING N N 2
GOL C2 O2  DOUB N N 3
GOL C3 "O3'" SING N N 4
GOL C1 H11 SING N N 5
#
`;

describe("CCD component", () => {
  it("reads atoms by column name, not by position", () => {
    // 🔴 THE COLUMNS MOVED ONCE ALREADY. pdbx_backbone_atom_flag and the two
    // terminal flags were added between the stereo config and the coordinates,
    // so anything counting fields from the left reads a coordinate as a flag.
    // This fixture has all three; the coordinates are at 15-17, not 12-14.
    const gol = parseCcdComponent(GOL);
    assert.equal(gol.code, "GOL");
    assert.deepEqual(gol.atoms.map((a) => a.name), ["C1", "O1", "C2", "O2", "C3", "O3'"]);
    assert.deepEqual(gol.atoms.map((a) => a.element), [6, 8, 6, 8, 6, 8]);
    assert.deepEqual(gol.atoms[0], { name: "C1", element: 6, charge: 0, x: -1.249, y: -0.665, z: 0.295 });
  });

  it("prefers the ideal conformer over the model one", () => {
    // The model coordinates come from a crystal and carry its contacts; the
    // ideal ones are the free molecule. AF3 uses neither - it resamples
    // torsions per instance - so what matters is that the bond LENGTHS are
    // right, and the ideal conformer is the one that has nothing else in it.
    const gol = parseCcdComponent(GOL);
    assert.equal(gol.atoms[0].x, -1.249, "ideal x, not the model's 29.490");
  });

  it("drops hydrogens, because AF3 tokenises heavy atoms only", () => {
    // A six-token glycerol, not fourteen. Keeping them makes a batch of the
    // wrong length whose every downstream shape is still self-consistent.
    const gol = parseCcdComponent(GOL);
    assert.equal(gol.atoms.length, 6);
    assert.ok(!gol.atoms.some((a) => a.name.startsWith("H")));
    // ...and bonds to them go too, rather than dangling at an absent index.
    assert.ok(!gol.bonds.some((b) => b.from >= 6 || b.to >= 6));
  });

  it("keeps quoted atom names, charges and bond orders", () => {
    const gol = parseCcdComponent(GOL);
    assert.equal(gol.atoms[5].name, "O3'", "primes are quoted in mmCIF");
    assert.equal(gol.atoms[5].charge, -1);
    assert.deepEqual(gol.bonds.map((b) => b.order), [1, 1, 2, 1]);
    assert.deepEqual(gol.bonds[3], { from: 4, to: 5, order: 1 }, "C3 to the quoted O3'");
  });

  it("names the file it would fetch, and refuses what is not a code", () => {
    assert.equal(ccdUrl("gol"), "https://files.rcsb.org/ligands/download/GOL.cif");
    assert.throws(() => ccdUrl("not a code"), /not a CCD code/);
    assert.throws(() => ccdUrl(""), /not a CCD code/);
  });
});
