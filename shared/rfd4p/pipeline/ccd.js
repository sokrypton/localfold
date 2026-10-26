/**
 * Chemical Component Dictionary entries fetched per component from RCSB, for a raw structure's
 * non-standard residues and ligands (plain RCSB downloads carry no usable `_chem_comp_atom`).
 * AtomWorks reads the same tables from its bundled CCD; here only the components a structure uses
 * are fetched (~2-20 KB each,
 * `files.rcsb.org/ligands/view/<ID>.cif`, served with `Access-Control-Allow-Origin: *`).
 */
import { CCD_TEMPLATES, WATER_LIKE_CCDS } from './ccd-templates.js';
import { mergeCcdIntoBlock, parseCifBlock, readStructureBlock } from './cif.js';

export { mergeCcdIntoBlock };

export const CCD_URL = (id) => `https://files.rcsb.org/ligands/view/${encodeURIComponent(id)}.cif`;
const CACHE_NAME = 'rfd4p-ccd-v1';
const memory = new Map();

/**
 * The components a RAW structure's preparation (fresh.js) needs a dictionary entry for: every
 * residue it names outside the generated templates (ccd-templates.js) - the standard residues and
 * the ions - that is not water. Upstream reads the whole CCD for these; the page fetches each.
 */
export function componentsNeedingCcd(resNames) {
  const out = new Set();
  for (const name of resNames) {
    if (!name || CCD_TEMPLATES[name] || WATER_LIKE_CCDS.includes(name)) continue;
    if (/^[A-Z0-9]{1,5}$/i.test(name)) out.add(name);
  }
  return [...out].sort();
}

async function fetchText(id, fetchFn) {
  const url = CCD_URL(id);
  const caches = globalThis.caches;
  let cache = null;
  try {
    cache = caches ? await caches.open(CACHE_NAME) : null;
    const hit = cache && (await cache.match(url));
    if (hit) return await hit.text();
  } catch {
    cache = null; // storage unavailable (private window, Node): fetch every time
  }
  const resp = await fetchFn(url);
  if (!resp.ok) throw new Error(`${id}: HTTP ${resp.status}`);
  const text = await resp.text();
  try {
    if (cache) await cache.put(url, new Response(text, { headers: { 'Content-Type': 'chemical/x-cif' } }));
  } catch {
    // best effort
  }
  return text;
}

/**
 * Fetch CCD entries; resolves to `{ blocks: Map<id, cifBlock>, failed: string[] }`. Never throws: a component
 * that cannot be fetched keeps the old behaviour (no template), and is listed in `failed`.
 */
export async function fetchCcdComponents(ids, { fetchFn = globalThis.fetch?.bind(globalThis) } = {}) {
  const blocks = new Map();
  const failed = [];
  await Promise.all(ids.map(async (id) => {
    try {
      if (!memory.has(id)) memory.set(id, fetchText(id, fetchFn).then(parseCifBlock));
      const block = await memory.get(id);
      if (!block?.chem_comp_atom) throw new Error(`${id}: no _chem_comp_atom`);
      blocks.set(id, block);
    } catch (err) {
      memory.delete(id);
      failed.push(id);
      console.warn('CCD fetch failed:', err?.message ?? err);
    }
  }));
  // In code order, not completion order: the entries are merged into the structure's own
  // tables row by row, and the bonds they carry are read in that order.
  const ordered = new Map([...blocks.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
  return { blocks: ordered, failed: failed.sort() };
}

/**
 * Every dictionary entry a raw structure's preparation needs (`componentsNeedingCcd`), fetched.
 * 🔴 A COMPONENT THAT CANNOT BE FETCHED IS A REFUSAL, NOT A STRUCTURE PREPARED WITHOUT IT: its
 * missing atoms, its charges and its atom order would all be silently wrong.
 *
 * @param {string|Uint8Array} input mmCIF text or bytes
 * @returns {Promise<Map<string, object>>}
 */
export async function ccdBlocksForRawStructure(input, { fetchFn } = {}) {
  const site = readStructureBlock(input).block.atom_site ?? {};
  const names = [...(site.label_comp_id ?? site.auth_comp_id ?? [])];
  const ids = componentsNeedingCcd(new Set(names));
  if (!ids.length) return new Map();
  const { blocks, failed } = await fetchCcdComponents(ids, fetchFn ? { fetchFn } : {});
  if (failed.length) {
    throw new Error(`could not fetch the Chemical Component Dictionary entr${failed.length === 1 ? 'y' : 'ies'} for `
      + `${failed.join(', ')} from RCSB, which preparing this structure needs`);
  }
  return blocks;
}
