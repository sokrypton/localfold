/**
 * The gate that says a fold is a chain, shared by every fold tool.
 *
 * 🔴 IT EXISTS BECAUSE FOUR TOOLS COMPUTED THIS AND ONE ASSERTED ON IT.
 * `fold-af2.js` had printed `caca` since it was written and nothing failed on
 * it, so an 825-residue fold whose whole chain had collapsed into a ball two
 * angstroms across - consecutive alpha carbons 0.06 A apart - passed as "the
 * same fold" for the length of a kernel campaign, while pLDDT climbed to 69.31
 * and pTM to 0.9672 and said it was fine. See docs/AF2.md.
 *
 * `fold.js` prints the same three distances under a comment reading "GEOMETRY
 * IS THE CHECK THAT MATTERS HERE, not pLDDT"; `fold-opendde.js` opens "the
 * geometry is the gate before the fold is"; `fold-esmfold2.js` reports the
 * spacing instead of the RNG for the same reason. None of the three gated.
 * **A number a tool prints is not a gate until something fails on it.**
 *
 * 🔴 AND THE RULE IS ONE RULE, IN ONE PLACE. Three copies of a band is three
 * bands, and this repository already has the version of that mistake where a
 * kernel's shape was resolved twice and the two answers differed.
 */

/**
 * The bands, measured on AF2 folds that were and were not chains.
 *
 * Consecutive alpha carbons are 3.80 A apart in any real protein - it is a
 * covalent geometry, not a prediction - so these are wide enough for a bad
 * PREDICTION and far too narrow for a broken one.
 *
 *     healthy   median 3.485 to 3.972   worst 1.69 to 4.55
 *     broken    median 1.44 to 3.41     worst 0.06, or 7.73 to 70.45
 */
export const CHAIN_GEOMETRY_BANDS = {
  medianLow: 3.4, medianHigh: 4.2, worstFrom38: 2.8,
};

/**
 * @param {{caca: number, worstCaca: number}} geometry medians in angstroms;
 *   `backboneGeometry` in src/af3/fold.js returns exactly this shape.
 * @param {{label?: string, plddt?: number, allow?: boolean, doc?: string}} options
 * @returns {{ok: boolean, reason: string|null, skipped: boolean}}
 */
export function chainGeometryVerdict(geometry, options = {}) {
  const { caca, worstCaca } = geometry;
  const bands = { ...CHAIN_GEOMETRY_BANDS, ...(options.bands ?? {}) };
  // 🔴 NO ALPHA CARBONS IS NOT A FAILURE. A ligand-only or nucleic-only fold
  // has no CA-CA distance to have an opinion about, and NaN > x is false in
  // both directions - so a gate written as a comparison would pass it silently
  // either way. Say so instead.
  if (!Number.isFinite(caca) || !Number.isFinite(worstCaca)) {
    return { ok: true, skipped: true, reason: null };
  }
  const ok = caca >= bands.medianLow && caca <= bands.medianHigh
    && Math.abs(worstCaca - 3.8) <= bands.worstFrom38;
  if (ok) return { ok: true, skipped: false, reason: null };
  return {
    ok: false,
    skipped: false,
    reason: `the fold is not a chain: consecutive CA median ${caca.toFixed(3)} A, `
      + `worst ${worstCaca.toFixed(2)} A, against 3.80 expected`
      + (Number.isFinite(options.plddt)
        ? `. pLDDT says ${options.plddt.toFixed(2)} and it is not a correctness gate`
        : "")
      + ` - see ${options.doc ?? "docs/AF2.md"}. Pass --allow-broken-geometry to `
      + "report anyway.",
  };
}

/** The same verdict, thrown. `allow` is the tool's --allow-broken-geometry. */
export function assertChainGeometry(geometry, options = {}) {
  const verdict = chainGeometryVerdict(geometry, options);
  if (!verdict.ok && options.allow !== true) throw new Error(verdict.reason);
  return verdict;
}

/**
 * The median and the worst of a list of consecutive-CA distances, for a tool
 * that has the distances rather than `backboneGeometry`'s summary.
 */
export function chainGeometryOf(spacings) {
  const values = [...spacings].filter(Number.isFinite);
  if (values.length === 0) return { caca: NaN, worstCaca: NaN };
  const sorted = [...values].sort((a, b) => a - b);
  return {
    caca: sorted[Math.floor(sorted.length / 2)],
    worstCaca: values.reduce(
      (far, value) => (Math.abs(value - 3.8) > Math.abs(far - 3.8) ? value : far), 3.8),
  };
}
