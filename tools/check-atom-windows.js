/**
 * Our atom key window against af3-any-model's own, for every dumped model.
 *
 *     node tools/check-atom-windows.js
 *
 * 🔴 THE ONE QUESTION NO OTHER GATE HERE ASKS. `check-af3-denoise.js` takes the
 * reference's atom windows out of the dump precisely so it can compare score
 * models rather than featurisers - which means the WINDOW itself has never been
 * compared against anything. A whole convention lived in that gap: three
 * families CLAMP the window and mask what falls outside where AlphaFold 3
 * SLIDES it bodily in bounds, and two of this port's shipped models were
 * sliding it. af3-any-model measured its own version of that at 6MRR
 * 0.767 -> 0.737 on opendde.
 *
 * 🔴 AND A FOLD'S RMSD CANNOT SETTLE IT. The change moved opendde's 6MRR by
 * 0.026 A, which is inside a band this repository has documented at 1 A across
 * seeds - so the fold says nothing and the gather says everything. This
 * compares INTEGERS.
 *
 * 🔴 IT COMPARES ONLY WHERE EITHER SIDE CALLS THE SLOT REAL. A masked slot's
 * index is arbitrary on both sides - the reference clamps into range, this port
 * leaves zero - so comparing indices there reports a difference that computes
 * nothing. The mask is compared everywhere; the index only where both agree the
 * slot is live.
 *
 * 🔴 AND THEIR SUBSET AXIS IS THE DENSE GRID WHERE OURS IS THE REAL ATOMS - 51
 * against 18 on a 68-mer, because `subsets` counts real atoms here and
 * `tokens * dense` there. The first 18 are the same 18; their tail is padding
 * and is not compared.
 */
import { readFileSync } from "node:fs";
import { featuriseProtein } from "../src/af3/featurise/featurise.js";
import { dialectFor, DIALECTS } from "../src/af3/dialect.js";
import { chiralCentres } from "../src/af3/featurise/template-features.js";

const DUMPS = {
  alphafold3: "af3-batch-alphafold3-6mrr.json",
  openbind0: "af3-batch-openbind0-6mrr.json",
  opendde: "af3-batch-opendde-6mrr.json",
  boltz2: "af3-batch-boltz2-6mrr.json",
  protenix2: "af3-batch-protenix2-6mrr.json",
  intellifold2: "af3-batch-intellifold2-6mrr.json",
  rosettafold3: "af3-batch-rosettafold3-6mrr.json",
};

const flat = (value) =>
  (Array.isArray(value?.data) ? value.data : Array.isArray(value) ? value.flat(9) : []);

function compare(sequence, dump, options) {
  const batch = featuriseProtein(sequence, options);
  const ours = batch.queriesToKeys;
  const theirIdx = flat(dump.inputs["queries_to_keys:gather_idxs"]);
  const theirMask = flat(dump.inputs["queries_to_keys:gather_mask"]);
  const keys = ours.indices.length / batch.shape.subsets;
  let maskDiff = 0;
  let indexDiff = 0;
  let live = 0;
  for (let at = 0; at < ours.indices.length; at += 1) {
    const mine = ours.mask[at] > 0.5;
    const theirs = theirMask[at] > 0.5;
    if (mine !== theirs) { maskDiff += 1; continue; }
    if (!mine) continue;
    live += 1;
    if (ours.indices[at] !== theirIdx[at]) indexDiff += 1;
  }
  return { keys, subsets: batch.shape.subsets, maskDiff, indexDiff, live };
}

let missing = 0;
let wrong = 0;   // eslint-disable-line prefer-const
const rows = [];
for (const [model, file] of Object.entries(DUMPS)) {
  let dump;
  try { dump = JSON.parse(readFileSync(new URL(`../oracle-dumps/${file}`, import.meta.url))); }
  catch { missing += 1; rows.push({ model, note: "no dump" }); continue; }
  const dialect = dialectFor(model);
  const shared = {
    symmetriseBonds: dialect.symmetriseBonds,
    centreRefConformers: dialect.centreRefConformers,
    qblockAtomKeys: dialect.qblockAtomKeys,
    ...(dialect.dropTerminalAtoms === true ? { terminalAtoms: false } : {}),
  };
  // Both arms, always: the SHIPPED one has to be exact and the other has to be
  // worse, or the flag is decoration. See the note on differential gates.
  const shipped = compare(dump.sequence, dump,
                          { ...shared, paddedAtomKeys: dialect.paddedAtomKeys });
  const other = compare(dump.sequence, dump,
                        { ...shared, paddedAtomKeys: !dialect.paddedAtomKeys });
  const exact = shipped.maskDiff === 0 && shipped.indexDiff === 0;
  if (!exact) wrong += 1;
  rows.push({ model, paddedAtomKeys: dialect.paddedAtomKeys, ...shipped, exact,
              otherArm: { maskDiff: other.maskDiff, indexDiff: other.indexDiff } });
}

for (const row of rows) {
  if (row.note) { console.log(`${row.model.padEnd(14)} ${row.note}`); continue; }
  console.log(`${row.model.padEnd(14)} paddedAtomKeys=${String(row.paddedAtomKeys).padEnd(5)}`
    + ` mask ${String(row.maskDiff).padStart(4)}  index ${String(row.indexDiff).padStart(4)}`
    + ` of ${row.live} live   ${row.exact ? "EXACT" : "DIFFERS"}`
    + `   (other arm ${row.otherArm.maskDiff}/${row.otherArm.indexDiff})`);
}
console.log(`\n${rows.length - missing - wrong} exact, ${wrong} differing,`
  + ` ${missing} without a dump`);

// 🔴 AND RoseTTAFold3's CHIRAL CENTRES, WHICH LIVE IN THE SAME DUMP. They are
// the only reflection-asymmetric signal the network has and on a plain protein
// they are not a no-op: 6MRR carries 213 of them. Checked here rather than in
// their own tool because the question is the same one - does this port's
// featuriser build what the reference's does - and the dump is already open.
{
  const dump = (() => {
    try {
      return JSON.parse(readFileSync(
        new URL("../oracle-dumps/af3-batch-rosettafold3-6mrr.json", import.meta.url)));
    } catch { return null; }
  })();
  if (dump === null) {
    console.log("\nchiral centres: no rosettafold3 dump");
  } else {
    const dialect = dialectFor("rosettafold3");
    const batch = featuriseProtein(dump.sequence, {
      symmetriseBonds: dialect.symmetriseBonds,
      centreRefConformers: dialect.centreRefConformers,
      qblockAtomKeys: dialect.qblockAtomKeys,
      ...(dialect.dropTerminalAtoms === true ? { terminalAtoms: false } : {}),
      paddedAtomKeys: dialect.paddedAtomKeys,
    });
    const ours = chiralCentres(batch.aatype, batch.predDenseAtomMask, batch.tokens);
    const theirCentres = flat(dump.inputs.chiral_centers).map(Number);
    const theirAngles = flat(dump.inputs.chiral_angles).map(Number);
    let differing = Math.abs(ours.count - theirAngles.length);
    for (let at = 0; at < Math.min(ours.count, theirAngles.length); at += 1) {
      for (let j = 0; j < 4; j += 1) {
        if (ours.centers[at * 4 + j] !== theirCentres[at * 4 + j]) differing += 1;
      }
      if (Math.abs(ours.angles[at] - theirAngles[at]) > 1e-6) differing += 1;
    }
    console.log(`chiral centres  ours ${ours.count} against ${theirAngles.length},`
      + ` ${differing} elements differing   ${differing === 0 ? "EXACT" : "DIFFERS"}`);
    if (differing !== 0) wrong += 1;
  }
}
if (Object.keys(DIALECTS).some((m) => DUMPS[m] === undefined)) {
  console.log("🔴 a dialect has no entry here, so its window is unchecked:",
    Object.keys(DIALECTS).filter((m) => DUMPS[m] === undefined).join(", "));
}
