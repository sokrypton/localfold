/**
 * The CCD's UNK component as a reference conformer: N CA C O CB CG (and OXT at a chain's end), its ideal coordinates
 * (_chem_comp_atom.pdbx_model_Cartn_*_ideal, files.rcsb.org/ligands/view/UNK.cif), the dense slots in the CCD's order.
 *
 * AlphaFold 3 builds an unknown amino acid as GLY (chemical_components.get_all_atoms_in_entry: "The CCD version of
 * 'UNK' is weird. It has a CB and a CG atom"); OpenFold3, so OpenBind-0, keeps the CCD's UNK - six atoms where AF3 has
 * four, the pseudo-beta its CB. The dialect's `ccdUnknownResidue` selects this (af3-any-model b84aa69: unk_residues
 * 10.6 -> 1.2 A). AF3 samples a fresh conformer per instance; ref_pos is a reported floor wherever it is compared.
 * Each atom is [slot, name, element, charge, x, y, z], as reference-conformers.js.
 */
export const UNK_CCD_CONFORMER = Object.freeze({
  aatype: 20,
  internal: [[0, "N", 7, 0, 1.827, -0.498, -1.278], [1, "CA", 6, 0, 2.145, -1.162, -2.517], [2, "C", 6, 0, 1.576, -0.429, -3.719],
             [3, "O", 8, 0, 0.85, 0.554, -3.635], [4, "CB", 6, 0, 1.719, -2.634, -2.491], [5, "CG", 6, 0, 0.211, -2.855, -2.469]],
  cTerminal: [[0, "N", 7, 0, 1.827, -0.498, -1.278], [1, "CA", 6, 0, 2.145, -1.162, -2.517], [2, "C", 6, 0, 1.576, -0.429, -3.719],
              [3, "O", 8, 0, 0.85, 0.554, -3.635], [4, "CB", 6, 0, 1.719, -2.634, -2.491], [5, "CG", 6, 0, 0.211, -2.855, -2.469],
              [6, "OXT", 8, 0, 1.959, -0.964, -4.903]],
});
