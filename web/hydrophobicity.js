/**
 * Hydropathy, for the parts of the page py2Dmol does not draw.
 *
 * THE VIEWER DOES NOT NEED THIS. `setSidechainColor("hydrophobicity")` resolves
 * Kyte & Doolittle inside py2Dmol at draw time, which is what keeps it correct
 * across frames. What is here serves the legend and the mutation menu - two
 * pieces of ordinary DOM that have to agree with the picture beside them.
 *
 * COLOURS COME FROM py2Dmol WHEN IT IS LOADED, read through
 * `py2Dmol.hydrophobicityBands`, so the swatch and the side chain cannot drift
 * apart. The table below is the fallback for the moment before the vendor
 * script has run, and for tests that have no viewer at all - and it is checked
 * against py2Dmol's in test/hydrophobicity.test.js.
 */

/** Kyte & Doolittle 1982, J Mol Biol 157:105-132. The scale everyone means. */
export const KYTE_DOOLITTLE = {
  I: 4.5, V: 4.2, L: 3.8, F: 2.8, C: 2.5, M: 1.9, A: 1.8, G: -0.4, T: -0.7,
  S: -0.8, W: -0.9, Y: -1.3, P: -1.6, H: -3.2, E: -3.5, Q: -3.5, D: -3.5,
  N: -3.5, K: -3.9, R: -4.5, X: 0,
};

/**
 * Five buckets, in py2Dmol's own shape: `{ min, hex, label }`.
 *
 * A gradient over twenty residues reads as twenty slightly different colours,
 * which is a picture you cannot name anything in. Buckets you can point at.
 */
export const FALLBACK_BANDS = [
  { min: 3.0, hex: "#f2994a", label: "very hydrophobic" },
  { min: 1.0, hex: "#f2c94c", label: "hydrophobic" },
  { min: -1.0, hex: "#cfd8d4", label: "neutral" },
  { min: -3.0, hex: "#56b9dc", label: "hydrophilic" },
  { min: -Infinity, hex: "#187bd1", label: "very hydrophilic" },
];

/** The bands the viewer is actually drawing with, or ours if it is not loaded. */
export function hydrophobicityBands() {
  const theirs = globalThis.py2Dmol?.hydrophobicityBands;
  return Array.isArray(theirs) && theirs.length > 0 ? theirs : FALLBACK_BANDS;
}

/** What the legend labels, when no viewer has loaded to ask. */
export const HYDROPHOBICITY_BANDS = FALLBACK_BANDS;

/**
 * The colour py2Dmol would give this residue's side chain.
 *
 * @param {string} code one-letter amino acid
 * @returns {string} a #rrggbb
 */
export function hydropathyColor(code) {
  const value = KYTE_DOOLITTLE[code] ?? 0;
  const bands = hydrophobicityBands();
  const band = bands.find((candidate) => value >= candidate.min);
  return (band ?? bands[bands.length - 1]).hex;
}

/** The band a residue falls in - its colour, and the words for it. */
export function hydropathyBand(code) {
  const value = KYTE_DOOLITTLE[code] ?? 0;
  const bands = hydrophobicityBands();
  return bands.find((candidate) => value >= candidate.min) ?? bands[bands.length - 1];
}

/**
 * The twenty, most hydrophobic first.
 *
 * IN THE ORDER THE COLOURS ARE IN, which is the point: the menu's cells are
 * painted by hydropathy, so laying them out by anything else scatters the ramp
 * and the reader has to hunt. Sorted, the strip IS the scale - orange at one
 * end, blue at the other - and "something a bit less hydrophobic than this" is
 * the cell next door rather than somewhere across the grid.
 *
 * Ties are broken alphabetically so the order is fixed. Four residues share
 * -3.5 (D, E, N, Q) and a wobbling menu is one you cannot learn.
 */
export const RESIDUES_BY_HYDROPATHY = Object.keys(KYTE_DOOLITTLE)
  .filter((code) => code !== "X")
  .sort((a, b) => KYTE_DOOLITTLE[b] - KYTE_DOOLITTLE[a] || a.localeCompare(b));
