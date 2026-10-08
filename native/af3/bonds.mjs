// Bond lengths of a native fold, by class (mainchain, side chain, peptide, nucleic, ligand), with
// ideals from the reference conformers and - for ligands, nucleotides and modified residues - the
// CCD's own bond list. tools/gpu/bond-geometry.js does the scoring.
//   node native/af3/bonds.mjs <fold.pdb> [CODE,CODE...]
import { readFileSync } from "node:fs";
const repo = new URL("../..", import.meta.url).pathname.replace(/\/$/, "");
const { bondGeometry, bondReport } = await import(`${repo}/tools/gpu/bond-geometry.js`);
const { ccdUrl, parseCcdComponent } = await import(`${repo}/shared/af3/featurise/ccd-component.js`);
const [pdbPath, codes = ""] = process.argv.slice(2);
const conformers = JSON.parse(readFileSync(`${repo}/tools/oracle/reference-conformers.json`, "utf8"));
const components = new Map();
for (const code of codes.split(",").filter(Boolean)) {
  components.set(code, parseCcdComponent(await (await fetch(ccdUrl(code))).text()));
}
const result = bondGeometry(readFileSync(pdbPath, "utf8"), conformers, { components });
console.log(bondReport(result));
