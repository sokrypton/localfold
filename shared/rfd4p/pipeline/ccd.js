/**
 * Chemical Component Dictionary entries fetched per component from RCSB, for structures whose nonstandard
 * residues or ligands come without `_chem_comp_bond` (plain RCSB downloads). AtomWorks reads the same tables from
 * its bundled CCD; here only the components a structure uses are fetched (~2-20 KB each,
 * `files.rcsb.org/ligands/view/<ID>.cif`, served with `Access-Control-Allow-Origin: *`).
 */
import { STD_RESIDUE_BONDS, catRows as rows, mergeCcdIntoBlock, parseCifBlock } from './cif.js';

export { mergeCcdIntoBlock };

export const CCD_URL = (id) => `https://files.rcsb.org/ligands/view/${encodeURIComponent(id)}.cif`;
const CACHE_NAME = 'rfd4p-ccd-v1';
const memory = new Map();

/** Components in `resNames` with neither a built-in bond template nor `_chem_comp_bond` rows in `cifBlock`. */
export function componentsNeedingCcd(resNames, cifBlock) {
  const have = new Set(Object.keys(STD_RESIDUE_BONDS));
  const ccb = rows(cifBlock?.chem_comp_bond);
  for (let r = 0; r < ccb.n; r += 1) have.add(String(ccb.get('comp_id', r)));
  const out = new Set();
  for (const name of resNames) {
    if (!name || have.has(name) || /^\d+-\d+$/.test(name) || name === '<M>' || name === 'UNK') continue;
    if (/^[A-Z0-9]{1,5}$/i.test(name)) out.add(name);
  }
  return [...out];
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
  return { blocks, failed };
}
