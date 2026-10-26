/**
 * A raw structure, PREPARED as upstream prepares a condition spec's `input:` when it is not an
 * annotated CIF (`load_condition_spec_inputs(..., as_annotated_cif=False)`: atomworks' `parse`
 * with RCSB_WITH_METALS and no assembly).
 *
 * 🔴 THE MODEL SEES WHAT THIS DOES, SO IT IS NOT A CLEAN-UP. Upstream completes every polymer to
 * its sequence - unmodeled residues and truncated side chains added without coordinates - so a
 * contig can name a residue the crystal never resolved and the model is told which atoms are
 * missing rather than not told they exist. It picks one alternate conformer per residue without
 * clashes (seeded, so the same file always loads the same atoms), drops waters, crystallization
 * aids that nothing is bonded to, and hydrogens, turns MSE into MET, puts each residue's atoms
 * in the dictionary's order and takes their formal charges from it. This port loaded a raw file
 * as written: on 6MRR the batch lacked 23 heavy atoms upstream adds and called eight partial
 * residues resolved, and 1BRS's chain A could not be named from residue 1 at all.
 *
 * It works on the mmCIF BLOCK and hands back a block: the `_atom_site` rows rewritten, every other
 * category kept, so the reader that follows (cif.js) derives the chemistry exactly as it does for
 * any other file. tools/check-rfd4p-load.py holds the result to atomworks atom for atom.
 */
import { CCD_TEMPLATES, WATER_LIKE_CCDS } from './ccd-templates.js';
import { PyRandom } from './pyrandom.js';

/** atomworks `CRYSTALLIZATION_AIDS`, which `RCSB_WITH_METALS` removes. */
export const CRYSTALLIZATION_AIDS = Object.freeze([
  'ACT', 'CD', 'DMS', 'EDO', 'EPE', 'FMT', 'GOL', 'IOD', 'MES', 'MPD', 'PEG', 'PG4', 'PGE', 'PO4', 'SO4', 'TRS',
]);

const ALTLOC_DEFAULTS = new Set(['.', '?', ' ', '']);
const unset = (v) => v === undefined || v === null || v === '.' || v === '?' || v === '';
const col = (cat, key) => {
  const v = cat?.[key];
  return v === undefined ? [] : Array.isArray(v) ? v : [v];
};

/** A component's template: the generated standard set, else the file's or a fetched `_chem_comp_atom`. */
function templateLookup(block, ccdBlocks) {
  const fromCat = new Map();
  const add = (cat) => {
    const ids = col(cat, 'comp_id');
    // The first source to describe a component describes all of it.
    const taken = new Set(fromCat.keys());
    for (let r = 0; r < ids.length; r += 1) {
      const id = ids[r];
      if (CCD_TEMPLATES[id] || taken.has(id)) continue;
      const element = String(col(cat, 'type_symbol')[r] ?? '').toUpperCase();
      if (!fromCat.has(id)) fromCat.set(id, { type: '', atoms: [] });
      const charge = col(cat, 'charge')[r];
      fromCat.get(id).atoms.push([
        col(cat, 'atom_id')[r], col(cat, 'alt_atom_id')[r] ?? col(cat, 'atom_id')[r], element,
        unset(charge) ? 0 : Number.parseInt(charge, 10) || 0,
        col(cat, 'pdbx_leaving_atom_flag')[r] === 'Y' ? 1 : 0,
        col(cat, 'pdbx_aromatic_flag')[r] === 'Y' ? 1 : 0,
      ]);
    }
  };
  for (const b of ccdBlocks?.values?.() ?? []) add(b.chem_comp_atom);
  add(block.chem_comp_atom);
  const types = new Map();
  const ids = col(block.chem_comp, 'id');
  ids.forEach((id, r) => types.set(id, String(col(block.chem_comp, 'type')[r] ?? '').toUpperCase()));
  for (const b of ccdBlocks?.values?.() ?? []) {
    col(b.chem_comp, 'id').forEach((id, r) => types.set(id, String(col(b.chem_comp, 'type')[r] ?? '').toUpperCase()));
  }
  return (code) => {
    const t = CCD_TEMPLATES[code] ?? fromCat.get(code);
    if (!t) return null;
    return { type: t.type || types.get(code) || '', atoms: t.atoms };
  };
}

/** The polymer-bond atoms of a component, by its chemical type (`get_polymerization_atoms`). */
function polymerizationAtoms(type) {
  if (/PEPTIDE/.test(type)) return ['C', 'N'];
  if (/DNA|RNA/.test(type)) return ["O3'", 'P'];
  return [null, null];
}

/** `_validate_label_seq_ids`: the refusals atomworks raises before it reads a CIF's atoms. */
function validateLabelSeqIds(site, block) {
  const asym = col(site, 'label_asym_id');
  const seq = col(site, 'label_seq_id');
  const ent = col(site, 'label_entity_id');
  if (!asym.length || !seq.length) return;
  const byAsym = new Map();
  asym.forEach((a, i) => {
    const e = byAsym.get(a) ?? { missing: false, present: false, entities: new Set() };
    if (seq[i] === '.' || seq[i] === '?') e.missing = true; else e.present = true;
    if (ent.length) e.entities.add(ent[i]);
    byAsym.set(a, e);
  });
  for (const [a, e] of [...byAsym.entries()].sort(([x], [y]) => (x < y ? -1 : x > y ? 1 : 0))) {
    if (e.missing && e.present) {
      throw new Error(`CIF label_asym_id '${a}' mixes defined and missing label_seq_id values. Polymer members, `
        + 'including noncanonical monomers, require label_seq_id; standalone non-polymers must use a separate label_asym_id.');
    }
    if (ent.length && e.entities.size !== 1) throw new Error(`CIF label_asym_id '${a}' maps to multiple label_entity_id values.`);
  }
  const types = new Map(col(block.entity, 'id').map((id, r) => [id, String(col(block.entity, 'type')[r] ?? '').trim().toLowerCase()]));
  if (!ent.length || !types.size) return;
  for (let i = 0; i < asym.length; i += 1) {
    if ((seq[i] === '.' || seq[i] === '?') && types.get(ent[i]) === 'polymer') {
      throw new Error(`CIF polymer label_asym_id '${asym[i]}' has missing label_seq_id values. Every polymer member, `
        + 'including noncanonical monomers, must have a sequence ID.');
    }
  }
}

/**
 * `select_altlocs_clash_aware(seed=0)`: one alternate location per residue, avoiding clashes and
 * giving neighbours that share letters the same one. `rows` are the first model's atom rows.
 */
function selectAltlocsClashAware(rows, block, lookup) {
  const isAlt = rows.map((r) => !ALTLOC_DEFAULTS.has(r.alt));
  if (!isAlt.some(Boolean)) return rows;
  // Residues with alternates, keyed (chain, res_id) as atomworks keys them at load.
  const groups = new Map();
  rows.forEach((r, i) => {
    if (!isAlt[i]) return;
    const key = `${r.chain}\u0000${r.seq}`;
    if (!groups.has(key)) groups.set(key, { key, chain: r.chain, seq: r.seq, letters: new Map() });
    const g = groups.get(key);
    if (!g.letters.has(r.alt)) g.letters.set(r.alt, []);
    g.letters.get(r.alt).push(i);
  });
  const residues = [...groups.values()];

  // `_parse_struct_conn_bonds`: both directions, a non-polymer's '.' sequence id read as its
  // author number, as its atoms' residue number is.
  const sc = block.struct_conn;
  const scBonds = new Set();
  if (sc) {
    const n = col(sc, 'ptnr1_label_asym_id').length;
    for (let k = 0; k < n; k += 1) {
      const side = (p) => {
        let s = col(sc, `ptnr${p}_label_seq_id`)[k];
        if (s === '.') s = col(sc, `ptnr${p}_auth_seq_id`)[k] ?? s;
        return `${col(sc, `ptnr${p}_label_asym_id`)[k]}\u0000${s}\u0000${col(sc, `ptnr${p}_label_atom_id`)[k]}`;
      };
      scBonds.add(`${side(1)}\u0001${side(2)}`);
      scBonds.add(`${side(2)}\u0001${side(1)}`);
    }
  }
  const bonded = (i, j) => {
    const a = rows[i];
    const b = rows[j];
    const [outA, inA] = polymerizationAtoms(lookup(a.comp)?.type ?? '');
    const [outB, inB] = polymerizationAtoms(lookup(b.comp)?.type ?? '');
    if ((a.atom === outA && b.atom === inB) || (b.atom === outB && a.atom === inA)) return true;
    return scBonds.has(`${a.chain}\u0000${a.seq}\u0000${a.atom}\u0001${b.chain}\u0000${b.seq}\u0000${b.atom}`);
  };

  // `_build_constraints`: every pair of alternate atoms of different residues within 5 A links the
  // residues as a preference; within 1.5 A and not bonded, the two letters clash.
  const idx = [];
  for (const g of residues) for (const [letter, list] of g.letters) for (const i of list) idx.push({ i, g, letter });
  const valid = idx.filter(({ i }) => rows[i].xyz.every(Number.isFinite));
  const cell = 5;
  const grid = new Map();
  const cellKey = (c) => c.map((x) => Math.floor(x / cell)).join(',');
  valid.forEach((v, k) => {
    const key = cellKey(rows[v.i].xyz);
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key).push(k);
  });
  const clashes = new Map(); // `${g}|${letter}` -> Set(`${g}|${letter}`)
  const clashList = [];
  const prefs = [];
  const prefSeen = new Set();
  const seenClash = new Set();
  const order = new Map(residues.map((g, k) => [g, k]));
  valid.forEach((v, k) => {
    const c = rows[v.i].xyz;
    const base = c.map((x) => Math.floor(x / cell));
    for (let dx = -1; dx <= 1; dx += 1) for (let dy = -1; dy <= 1; dy += 1) for (let dz = -1; dz <= 1; dz += 1) {
      const list = grid.get(`${base[0] + dx},${base[1] + dy},${base[2] + dz}`);
      if (!list) continue;
      for (const m of list) {
        if (m <= k) continue;
        const w = valid[m];
        if (w.g === v.g) continue;
        const d = Math.hypot(c[0] - rows[w.i].xyz[0], c[1] - rows[w.i].xyz[1], c[2] - rows[w.i].xyz[2]);
        if (d <= 1.5 && !bonded(v.i, w.i)) {
          const e1 = `${order.get(v.g)}|${v.letter}|${order.get(w.g)}|${w.letter}`;
          const e2 = `${order.get(w.g)}|${w.letter}|${order.get(v.g)}|${v.letter}`;
          if (!seenClash.has(e2) && !seenClash.has(e1)) {
            seenClash.add(e1);
            clashList.push([v.g, v.letter, w.g, w.letter]);
          }
        }
        if (d <= 5) {
          // Residue keys compare as (chain, res_id) tuples upstream: chain text, then the number.
          const less = v.g.chain < w.g.chain || (v.g.chain === w.g.chain && v.g.seq < w.g.seq);
          const [a, b] = less ? [v.g, w.g] : [w.g, v.g];
          const pk = `${order.get(a)}|${order.get(b)}`;
          if (!prefSeen.has(pk)) { prefSeen.add(pk); prefs.push([a, b]); }
        }
      }
    }
  });
  const prefPairs = new Set();
  for (const [a, b] of prefs) { prefPairs.add(`${order.get(a)}|${order.get(b)}`); prefPairs.add(`${order.get(b)}|${order.get(a)}`); }
  for (const [gi, li, gj, lj] of clashList) {
    if (prefPairs.has(`${order.get(gi)}|${order.get(gj)}`)) continue;
    const a = `${order.get(gi)}|${li}`;
    const b = `${order.get(gj)}|${lj}`;
    if (!clashes.has(a)) clashes.set(a, new Set());
    if (!clashes.has(b)) clashes.set(b, new Set());
    clashes.get(a).add(b);
    clashes.get(b).add(a);
  }

  // `_solve_greedy`, drawing from `random.Random(0)` in upstream's order.
  const rng = new PyRandom(0);
  const options = new Map(residues.map((g) => [g, [...g.letters.keys()]]));
  for (const letters of options.values()) rng.shuffle(letters);
  const parent = new Map(residues.map((g) => [g, g]));
  const find = (x) => {
    while (parent.get(x) !== x) { parent.set(x, parent.get(parent.get(x))); x = parent.get(x); }
    return x;
  };
  for (const [a, b] of prefs) parent.set(find(a), find(b));
  const groupMap = new Map();
  for (const g of residues) {
    const root = find(g);
    if (!groupMap.has(root)) groupMap.set(root, []);
    groupMap.get(root).push(g);
  }
  const unions = [...groupMap.values()];
  rng.shuffle(unions);
  const assignment = new Map();
  const countClashes = (g, letter) => {
    let n = 0;
    for (const other of clashes.get(`${order.get(g)}|${letter}`) ?? []) {
      const [gk, l] = other.split('|');
      if (assignment.get(residues[Number(gk)]) === l) n += 1;
    }
    return n;
  };
  const minBy = (list, score) => {
    let best = list[0];
    let bestScore = score(best);
    for (const x of list.slice(1)) {
      const s = score(x);
      const lt = Array.isArray(s) ? (s[0] < bestScore[0] || (s[0] === bestScore[0] && s[1] < bestScore[1])) : s < bestScore;
      if (lt) { best = x; bestScore = s; }
    }
    return best;
  };
  for (const group of unions) {
    rng.shuffle(group);
    if (group.length === 1) {
      const g = group[0];
      assignment.set(g, minBy(options.get(g), (l) => countClashes(g, l)));
    } else {
      const all = new Set();
      for (const g of group) for (const l of options.get(g)) all.add(l);
      const candidates = [...all].sort();
      rng.shuffle(candidates);
      const best = minBy(candidates, (l) => {
        let supporters = 0;
        let total = 0;
        for (const g of group) if (options.get(g).includes(l)) { supporters += 1; total += countClashes(g, l); }
        return [-supporters, total];
      });
      for (const g of group) if (options.get(g).includes(best)) assignment.set(g, best);
    }
  }
  let unresolved = 0;
  for (const [g, l] of assignment) unresolved += countClashes(g, l);
  if (unresolved > 0) {
    throw new Error(`No fully clash-free altloc assignment found for ${residues.length} residues. `
      + `${unresolved / 2} unresolved clashes remain.`);
  }
  const keep = rows.map((_, i) => !isAlt[i]);
  for (const [g, l] of assignment) for (const i of g.letters.get(l)) keep[i] = true;
  return rows.filter((_, i) => keep[i]);
}

/**
 * Prepare a raw structure's block. Returns a NEW block whose `_atom_site` is the prepared atoms
 * (one model, no alternate locations, no hydrogens, explicit charges, missing atoms with `?`
 * coordinates and zero occupancy); every other category is the input's.
 *
 * @param {object} block a parsed mmCIF block (cif.js `parseCifBlock`)
 * @param {{ccdBlocks?: Map<string, object>}} [opts] fetched dictionary entries for components the
 *   generated templates and the file do not describe
 */
export function prepareFreshBlock(block, { ccdBlocks } = {}) {
  const site = block.atom_site;
  if (!site) throw new Error('CIF block does not contain _atom_site category');
  validateLabelSeqIds(site, block);
  const lookup = templateLookup(block, ccdBlocks);

  const get = (k, r, fallback = '') => {
    const v = col(site, k)[r];
    return v === undefined ? fallback : v;
  };
  const nRows = col(site, 'label_atom_id').length || col(site, 'auth_atom_id').length;
  const models = col(site, 'pdbx_PDB_model_num');
  let rows = [];
  for (let r = 0; r < nRows; r += 1) {
    if (models.length && models[r] !== models[0]) continue;
    const seqText = get('label_seq_id', r, '.');
    const auth = get('auth_seq_id', r, '?');
    rows.push({
      group: get('group_PDB', r, 'ATOM'),
      element: String(get('type_symbol', r, 'C')).toUpperCase(),
      atom: get('label_atom_id', r) || get('auth_atom_id', r),
      alt: get('label_alt_id', r, '.'),
      comp: get('label_comp_id', r) || get('auth_comp_id', r),
      chain: get('label_asym_id', r) || get('auth_asym_id', r, 'A'),
      entity: get('label_entity_id', r, '?'),
      // The residue number at load, as biotite reads it: the label number, or the author number
      // where there is none (a non-polymer's '.').
      seq: Number.parseInt(unset(seqText) ? auth : seqText, 10),
      labelled: !unset(seqText),
      auth,
      xyz: ['Cartn_x', 'Cartn_y', 'Cartn_z'].map((k) => {
        const v = get(k, r, '?');
        return unset(v) ? NaN : Number.parseFloat(v);
      }),
      occupancy: get('occupancy', r, '1'),
      b: get('B_iso_or_equiv', r, '?'),
      charge: get('pdbx_formal_charge', r, '?'),
    });
  }
  if (col(site, 'label_alt_id').length) rows = selectAltlocsClashAware(rows, block, lookup);
  // Hydrogens stay until the link products are built (links.js), which count them, as upstream's
  // do (`hydrogen_policy="keep"`); the reader drops them after that.
  rows = rows.filter((r) => !WATER_LIKE_CCDS.includes(r.comp));

  // Which chains are polymers, and of what kind (`build_chain_info`).
  const entityPolyType = new Map(col(block.entity_poly, 'entity_id').map((id, r) => [id, String(col(block.entity_poly, 'type')[r] ?? '').toLowerCase()]));
  const entityType = new Map(col(block.entity, 'id').map((id, r) => [id, String(col(block.entity, 'type')[r] ?? '').trim().toLowerCase()]));
  const chainOrder = [];
  const chainRows = new Map();
  for (const r of rows) {
    if (!chainRows.has(r.chain)) { chainRows.set(r.chain, []); chainOrder.push(r.chain); }
    chainRows.get(r.chain).push(r);
  }
  const polySeq = new Map();
  const ps = block.entity_poly_seq;
  col(ps, 'entity_id').forEach((e, k) => {
    if (!polySeq.has(e)) polySeq.set(e, []);
    polySeq.get(e).push({ num: Number.parseInt(col(ps, 'num')[k], 10), mon: col(ps, 'mon_id')[k] });
  });

  const out = [];
  for (const chain of chainOrder) {
    const list = chainRows.get(chain);
    const entity = list[0].entity;
    const polyType = entityPolyType.get(entity) ?? '';
    const isPolymer = entityPolyType.has(entity) || entityType.get(entity) === 'polymer'
      || (!entityType.has(entity) && list[0].labelled);

    // Residues as they are modeled, in file order.
    const residues = [];
    for (const r of list) {
      const last = residues.at(-1);
      if (last && last.seq === r.seq && last.comp === r.comp) last.atoms.push(r);
      else residues.push({ seq: r.seq, comp: r.comp, auth: r.auth, group: r.group, atoms: [r] });
    }
    // The sequence: modeled residues plus the polymer's unmodeled ones (`_enrich_with_poly_seq`).
    let sequence = residues.map((res) => ({ res }));
    if (isPolymer && polySeq.has(entity)) {
      const have = new Set(residues.map((res) => res.seq));
      const missing = polySeq.get(entity).filter((p) => !have.has(p.num));
      if (missing.length) {
        sequence = [...residues.map((res) => ({ res, num: res.seq })), ...missing.map((p) => ({ num: p.num, mon: p.mon }))]
          .sort((a, b) => a.num - b.num);
      }
    }
    sequence.forEach((s, k) => {
      const res = s.res;
      let comp = res ? res.comp : s.mon;
      if (comp === 'MSE') comp = 'MET';
      const template = lookup(comp);
      const hasPrev = isPolymer && k > 0;
      const hasNext = isPolymer && k < sequence.length - 1;
      // A leaving atom at a polymer bond to a neighbour (an OXT, an OP3, the H2 they take with
      // them) is added like any other and leaves with the bond, when links.js builds it.
      const resId = res ? res.seq : s.num;
      const auth = res ? res.auth : String(s.num);
      const group = res ? (res.comp === 'MSE' ? 'ATOM' : res.group) : 'ATOM';
      const emit = (atom, element, xyz, occupancy, b, charge, src) => out.push({
        group, element, atom, comp, chain, entity, seq: isPolymer ? resId : -1, auth, xyz, occupancy, b, charge, src,
      });
      // An atom of no element cannot carry a covalent link, and a polymer residue is linked to
      // its neighbours: atomworks refuses it there (`_validate_link_valence`). That is an
      // annotated file's expandable-segment sentinel read as a raw structure.
      const unknown = (hasPrev || hasNext) && res?.atoms.find((a) => a.element === 'X');
      if (unknown) {
        throw new Error(`Unresolved link valence at ${chain}/${res.seq}/${res.comp}/${unknown.atom} (element=X, `
          + 'charge=0): a raw structure cannot carry an atom of no element inside a polymer - load an annotated '
          + 'CIF with input_as_annotated_cif: true');
      }
      if (!template) {
        if (!res) return;
        for (const a of res.atoms) {
          emit(a.atom, a.element, a.xyz, a.occupancy, a.b, unset(a.charge) ? '0' : a.charge, a);
        }
        return;
      }
      // Standard names (an alternate name maps to the template's), template order, then any atom
      // the template does not name in file order.
      const byName = new Map();
      const extra = [];
      for (const a of res?.atoms ?? []) {
        let name = a.atom;
        if (res.comp === 'MSE' && name === 'SE') name = 'SD';
        const t = template.atoms.find((x) => x[0] === name) ?? template.atoms.find((x) => x[1] === name);
        if (t) { if (!byName.has(t[0])) byName.set(t[0], a); } else extra.push(a);
      }
      for (const [name, , element, charge] of template.atoms) {
        const a = byName.get(name);
        if (a) {
          const el = res.comp === 'MSE' && a.atom === 'SE' ? 'S' : a.element;
          emit(name, el, a.xyz, a.occupancy, a.b, String(charge), a);
        } else {
          emit(name, element, [NaN, NaN, NaN], '0', '?', String(charge), null);
        }
      }
      for (const a of extra) {
        emit(a.atom, a.element, a.xyz, a.occupancy, a.b, unset(a.charge) ? '0' : a.charge, a);
      }
    });
  }

  // Crystallization aids go, unless covalently joined to something that stays (`remove_components`).
  const aids = new Set(CRYSTALLIZATION_AIDS);
  let prepared = out;
  if (out.some((a) => aids.has(a.comp))) {
    const resKey = (a) => `${a.chain}\u0000${a.seq === -1 ? Number.parseInt(a.auth, 10) : a.seq}`;
    const parentOf = new Map();
    const find = (x) => { while (parentOf.get(x) !== x) x = parentOf.get(x); return x; };
    for (const a of out) parentOf.set(resKey(a), resKey(a));
    const sc = block.struct_conn;
    const types = col(sc, 'conn_type_id');
    for (let k = 0; k < types.length; k += 1) {
      if (String(types[k]).toLowerCase() !== 'covale') continue;
      const key = (p) => {
        const s = col(sc, `ptnr${p}_label_seq_id`)[k];
        return `${col(sc, `ptnr${p}_label_asym_id`)[k]}\u0000${s === '.' ? col(sc, `ptnr${p}_auth_seq_id`)[k] : s}`;
      };
      const a = key(1);
      const b = key(2);
      if (parentOf.has(a) && parentOf.has(b)) parentOf.set(find(a), find(b));
    }
    const keepRoot = new Set();
    for (const a of out) if (!aids.has(a.comp)) keepRoot.add(find(resKey(a)));
    prepared = out.filter((a) => !aids.has(a.comp) || keepRoot.has(find(resKey(a))));
  }

  const fmt = (x) => (Number.isFinite(x) ? String(x) : '?');
  const atomSite = {
    group_PDB: [], id: [], type_symbol: [], label_atom_id: [], label_alt_id: [], label_comp_id: [],
    label_asym_id: [], label_entity_id: [], label_seq_id: [], pdbx_PDB_ins_code: [], Cartn_x: [], Cartn_y: [],
    Cartn_z: [], occupancy: [], B_iso_or_equiv: [], pdbx_formal_charge: [], auth_seq_id: [], auth_comp_id: [],
    auth_asym_id: [], auth_atom_id: [], pdbx_PDB_model_num: [],
  };
  prepared.forEach((a, k) => {
    atomSite.group_PDB.push(a.group);
    atomSite.id.push(String(k + 1));
    atomSite.type_symbol.push(a.element);
    atomSite.label_atom_id.push(a.atom);
    atomSite.label_alt_id.push('.');
    atomSite.label_comp_id.push(a.comp);
    atomSite.label_asym_id.push(a.chain);
    atomSite.label_entity_id.push(a.entity);
    atomSite.label_seq_id.push(a.seq === -1 ? '.' : String(a.seq));
    atomSite.pdbx_PDB_ins_code.push('?');
    atomSite.Cartn_x.push(fmt(a.xyz[0]));
    atomSite.Cartn_y.push(fmt(a.xyz[1]));
    atomSite.Cartn_z.push(fmt(a.xyz[2]));
    atomSite.occupancy.push(a.occupancy);
    atomSite.B_iso_or_equiv.push(a.b);
    atomSite.pdbx_formal_charge.push(a.charge);
    atomSite.auth_seq_id.push(a.auth);
    atomSite.auth_comp_id.push(a.comp);
    atomSite.auth_asym_id.push(a.chain);
    atomSite.auth_atom_id.push(a.atom);
    atomSite.pdbx_PDB_model_num.push('1');
  });
  return { ...block, atom_site: atomSite };
}
