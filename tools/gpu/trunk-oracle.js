/**
 * af3-any-model's own trunk seams, compared against ours on the same batch.
 *
 * 🔴 EVERY OTHER TRUNK GATE HERE COMPARES THE GPU AGAINST THIS PORT'S OWN CPU
 * REFERENCE, so the two agree and neither is held to the model. That is how
 * boltz2's `target_feat` sat at relRMS 1.00e+0 while `check-af3-trunk` read
 * 2.74e-5. `tools/oracle/dump_af3_trunk_taps.py` records the reference's z at
 * each seam - `z_init_generic`, `z_after_template`, `z_after_msa`,
 * `trunk_in_single`, `trunk_out_pair` - plus the final single and pair, and
 * this compares them.
 *
 * Extracted from tools/gpu/fold.js because fold-opendde.js needed the same
 * thing and OpenDDE is the family that had no trunk oracle at all. A copy is
 * how the page and the CLI came to build different batches.
 *
 * 🔴 AND IT COUNTS WHAT IT COMPARED. The original returned silently when a
 * label was absent from the dump or undefined on our side, so a renamed seam
 * reported NOTHING and read exactly like agreement. `summary()` names every
 * label that was offered and never matched, and the caller fails on it.
 */

/** Load a dump, or null when no `--trunk-oracle=` was given. */
export async function loadTrunkOracle(path) {
  if (path === "") return null;
  const response = await fetch(path);
  if (!response.ok) throw new Error(`failed to load ${path}: ${response.status}`);
  const dump = await response.json();
  if (dump?.stages === undefined) throw new Error(`${path} carries no stages`);
  return dump;
}

/**
 * A comparator over one dump.
 * @param {?object} dump from loadTrunkOracle
 * @param {number} [bound] relRMS above which a seam is called wrong
 */
export function trunkOracleComparer(dump, bound = 1e-2) {
  const compared = [];
  const missing = [];
  const compare = (label, ours) => {
    const entry = dump?.stages?.[label];
    if (dump === null) return;
    if (entry === undefined) { missing.push(`${label} (not in the dump)`); return; }
    if (ours === undefined) { missing.push(`${label} (we produced nothing)`); return; }
    const expected = Float32Array.from(entry.data);
    if (expected.length !== ours.length) {
      console.log(`  native ${label}\tLENGTH ${ours.length} vs ${expected.length}`);
      compared.push({ label, relRms: Number.POSITIVE_INFINITY, lengthMismatch: true });
      return;
    }
    let error = 0, scale = 0;
    for (let i = 0; i < expected.length; i += 1) {
      const d = ours[i] - expected[i];
      error += d * d; scale += expected[i] * expected[i];
    }
    const relRms = Math.sqrt(error / Math.max(scale, 1e-30));
    const mine = Math.sqrt(ours.reduce((t, v) => t + v * v, 0) / ours.length);
    console.log(`  native ${label}\t${relRms.toExponential(2)}`
      + `\tours rms ${mine.toFixed(4)}\tnative rms ${entry.rms.toFixed(4)}`);
    compared.push({ label, relRms, oursRms: mine, nativeRms: entry.rms });
  };
  const summary = () => ({
    bound,
    compared,
    // A label the dump has and we never offered is the interesting absence: it
    // means this tool does not reach that seam, not that the seam agrees.
    offeredButUnmatched: missing,
    notCompared: Object.keys(dump?.stages ?? {})
      .filter((label) => !compared.some((c) => c.label === label)),
    worst: compared.reduce((w, c) => (w === null || c.relRms > w.relRms ? c : w), null),
    agrees: compared.length > 0 && compared.every((c) => c.relRms <= bound),
  });
  return { compare, summary };
}
