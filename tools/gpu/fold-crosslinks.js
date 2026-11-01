/**
 * Crosslinks folded: the page's path for a `crosslink` row (web/entities.js expandEntities: a built-in bridge ligand
 * bonded to its two residues) end to end on WebGPU, and whether the fold honours it.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/fold-crosslinks.js [--model=/model-boltz2-int5/manifest.json] [--seed=1]
 *
 * A report, not a gate. Per crosslink: each residue-bridge bond's length in the folded structure (a covalent bond is ~1.3-1.5 A; an
 * unhonoured one is wherever the sampler left it), and the two residues' CA-CA distance against the same job folded
 * without its crosslinks - which a spacer arm bounds (DSSO's ~10 A, a diazirine's shorter).
 */
import { expandEntities } from "../../web/entities.js";
import { CROSSLINKER_COMPONENTS } from "../../shared/chem/crosslinkers.js";
import { parseCcdComponent } from "../../shared/af3/featurise/ccd-component.js";
import { af3BatchFromA3m } from "../../shared/af3/featurise/batch.js";
import { dialectFor, featuriserDialect } from "../../shared/af3/dialect.js";
import { toPdb } from "../../shared/af3/structure/pdb.js";
import { foldBatch } from "../../webgpu/af3/fold.js";
import { confidenceWeights, openAf3Store, trunkDepths, trunkWeights } from "../../shared/af3/weights/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights } from "../../shared/af3/weights/diffusion-weights.js";

const option = (args, name, fallback) =>
  args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const row = (type, value) => ({ type, value, copies: 1, modifications: [] });

// barnase:barstar-sized chains with lysines either side; the crosslinks name residues the NHS ends can acylate
const A = "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR";
const B = "KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDCLTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGCDITIILS";
const CROSSLINKS = ["DSSO: A27 - B21", "DSSO: A62 - B1", "sulfo-SDA: A49 - B35"];

function atomsOf(pdb) {
  const at = new Map();
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    const key = `${line.slice(21, 22).trim()}${Number(line.slice(22, 26))}:${line.slice(12, 16).trim()}`;
    at.set(key, [Number(line.slice(30, 38)), Number(line.slice(38, 46)), Number(line.slice(46, 54))]);
  }
  return at;
}
const distance = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
const letter = (asym) => String.fromCharCode(65 + asym);

export async function main(device, args) {
  const store = await openAf3Store(option(args, "model", "/model-boltz2-int5/manifest.json"));
  store.prefetch();
  const depths = trunkDepths(store);
  const weights = {
    trunk: await trunkWeights(store, depths.pairformerBlocks, depths.msaBlocks, { allowPrefix: true }),
    diffusion: await diffusionWeights(store), confidence: await confidenceWeights(store),
    atomReference: await atomReference(store), targetFeat: await targetFeatureWeights(store),
  };
  const dialect = weights.trunk.dialect ?? dialectFor("boltz2");
  const seed = Number(option(args, "seed", "1"));
  const fold = async (rows) => {
    const out = expandEntities([row("protein", A), row("protein", B), ...rows.map((v) => row("crosslink", v))]);
    const { batch } = af3BatchFromA3m(out.sequence, null, {
      chainKinds: out.chainKinds, ...(out.bonds === undefined ? {} : { bonds: out.bonds }),
      ligands: out.ligandCodes.map((code) => parseCcdComponent(CROSSLINKER_COMPONENTS[code])),
      ...featuriserDialect(dialect) });
    const result = await foldBatch(device, batch, weights, { recycles: 3, seed });
    return { out, atoms: atomsOf(toPdb(batch, result.positions, result.scores?.plddt)), iptm: result.iptm };
  };
  const linked = await fold(CROSSLINKS);
  const free = await fold([]);
  const report = [];
  const bonds = linked.out.bonds;
  for (let k = 0; k < CROSSLINKS.length; k += 1) {
    const [r1, r2] = [bonds[2 * k], bonds[2 * k + 1]];
    const length = (bond) => distance(linked.atoms.get(`${letter(bond.from.asym)}${bond.from.residue}:${bond.from.atom}`),
                                      linked.atoms.get(`${letter(bond.to.asym)}${bond.to.residue}:${bond.to.atom}`));
    const ca = (atoms) => distance(atoms.get(`${letter(r1.from.asym)}${r1.from.residue}:CA`),
                                   atoms.get(`${letter(r2.from.asym)}${r2.from.residue}:CA`));
    const line = { crosslink: CROSSLINKS[k], bond1: length(r1), bond2: length(r2),
                   caWith: ca(linked.atoms), caWithout: ca(free.atoms) };
    report.push(line);
    console.log(`${line.crosslink}: residue-bridge bonds ${line.bond1.toFixed(2)} / ${line.bond2.toFixed(2)} A;`
      + ` CA-CA ${line.caWith.toFixed(1)} A with, ${line.caWithout.toFixed(1)} A without`);
  }
  console.log(`ipTM ${linked.iptm?.toFixed(3)} with, ${free.iptm?.toFixed(3)} without`);
  // a report, not a gate: whether the model closes every bond is the model's behaviour (af3x notes that crosslinks
  // which cannot all be satisfied get broken), and a single no-MSA sample is not a fair test of it. Measured
  // 2026-10-28, Boltz-2 int5, seed 1: 3 of 6 bonds under 2 A, every CA-CA pulled in (32.3/32.7/20.3 A ->
  // 20.3/11.6/13.0 A), ipTM 0.10 -> 0.23 - and CUDA's int8 fold of the same job (cuda/af3/README.md) alike.
  const formed = report.reduce((n, r) => n + (r.bond1 < 2.0) + (r.bond2 < 2.0), 0);
  console.log(`${formed} of ${2 * report.length} residue-bridge bonds under 2 A`);
  return { report, formed };
}
