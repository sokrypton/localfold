/**
 * What featuriseBioemu2 needs from outside a job: the CCD components it names,
 * each SMILES ligand's template, and each protein chain's alignment.
 *
 * One copy for the page (web/bioemu2-model.js) and the fold tool
 * (tools/gpu/fold-bioemu2.js), so the two cannot hand the featuriser different
 * inputs for the same job.
 */
import { ccdUrl } from "../af3/featurise/ccd-component.js";
import { STANDARD_COMPONENTS } from "../af3/featurise/standard-components.js";
import { smilesComponent } from "../chem/component.js";

/**
 * Every component a job names, fetched once: `local(code)` first (a URL or null -
 * the tools read tools/fixtures/ccd), then the PDB's own server.
 * @returns {Promise<(code: string) => string>} the featuriser's `ccd`
 */
export async function fetchComponents(job, { local = () => null, signal } = {}) {
  const codes = new Set();
  for (const chain of job.chains) {
    for (const m of chain.modifications ?? []) codes.add(m.code);
    for (const c of chain.ccdCodes ?? []) codes.add(c);
    if (chain.kind === "protein" && /X/.test(chain.sequence)) codes.add("UNK");
    if (chain.kind === "rna" && /N/.test(chain.sequence)) codes.add("N");
    if (chain.kind === "dna" && /N/.test(chain.sequence)) codes.add("DN");
  }
  const texts = new Map();
  await Promise.all([...codes].filter((code) => STANDARD_COMPONENTS[code] === undefined).map(async (code) => {
    const nearby = local(code);
    let response = nearby ? await fetch(nearby, { signal }) : null;
    if (response === null || !response.ok) response = await fetch(ccdUrl(code), { signal });
    if (!response.ok) throw new Error(`No chemical component ${code} at the PDB (${response.status})`);
    texts.set(code, await response.text());
  }));
  return (code) => texts.get(code);
}

const SYMBOLS = ["H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG", "AL", "SI", "P", "S", "CL", "AR",
  "K", "CA", "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR", "RB",
  "SR", "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN", "SB", "TE", "I"];

/**
 * Each SMILES ligand's template, from this port's own conformer (shared/chem): its atoms in SMILES order, named
 * element-and-counter, centred on their mean. BioEmu2 itself draws an unseeded RDKit conformer named by canonical
 * rank, so no port reproduces that draw; see shared/bioemu2/featurise.js.
 */
export async function smilesTemplates(job) {
  const out = {};
  for (const chain of job.chains) {
    if (chain.smiles == null) continue;
    const component = await smilesComponent(chain.smiles);
    const counts = {};
    const names = component.atoms.map((a) => {
      const symbol = SYMBOLS[a.element - 1] ?? "X";
      counts[symbol] = (counts[symbol] ?? 0) + 1;
      return `${symbol}${counts[symbol]}`;
    });
    const n = component.atoms.length;
    const mean = [0, 0, 0];
    for (const a of component.atoms) { mean[0] += a.x / n; mean[1] += a.y / n; mean[2] += a.z / n; }
    const coords = new Float32Array(n * 3);
    component.atoms.forEach((a, i) => {
      coords[i * 3] = a.x - mean[0]; coords[i * 3 + 1] = a.y - mean[1]; coords[i * 3 + 2] = a.z - mean[2];
    });
    out[chain.id] = { names, element: component.atoms.map((a) => a.element),
                      charge: component.atoms.map((a) => a.charge ?? 0), coords,
                      bonds: component.bonds.map((b) => [b.from, b.to]) };
  }
  return out;
}

/**
 * A merged, row-aligned A3M over several protein chains, split back into one A3M a chain.
 *
 * The page's search, paste and upload paths all end in ONE text over the concatenated chains; BioEmu2 reads each
 * chain's own alignment (Boltz-1's rows lined up by index), so the columns go back to their chains: a match column
 * (upper case or '-') advances the chain's position, an insertion (lower case) belongs to the residue before it.
 */
export function splitA3mByChains(a3m, lengths) {
  const out = lengths.map(() => []);
  let header = null;
  for (const raw of a3m.split(/\r?\n/)) {
    const line = raw.trim();
    if (line === "") continue;
    if (line.startsWith(">")) { header = line; continue; }
    if (line.startsWith("#")) continue;
    const pieces = lengths.map(() => "");
    let chain = 0;
    let column = 0;
    for (const c of line) {
      const insertion = c >= "a" && c <= "z";
      if (!insertion) {
        while (chain < lengths.length - 1 && column >= lengths[chain]) { chain += 1; column = 0; }
        column += 1;
      }
      pieces[chain] += c;
    }
    pieces.forEach((piece, at) => out[at].push(`${header ?? ">row"}\n${piece}`));
  }
  return out.map((rows) => `${rows.join("\n")}\n`);
}
