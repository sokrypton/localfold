/**
 * BioEmu2's steering potentials' tables, built on the host.
 *
 * A transcription of af3-any-model's pipeline/bioemu2_steering.tables (bioemu2
 * @7846769's denoising/steering/{feats, geometry_topology}.py and the
 * `prepare()` half of potentials.py), keyed as the vendor's feats are, on
 * featuriseBioemu2's topology. The potentials themselves run on the device, in
 * each backend's sampler.
 *
 * 🔴 THE BOND-GEOMETRY REFERENCE IS OTHER PEOPLE'S DATA AND IS NOT SHIPPED.
 * `stereo_chemistry.yaml` is CCP4 MON_LIB via OpenStructure (LGPL-3.0), so it
 * is fetched from BioEmu2's repository at the commit this port follows - as
 * the colab port fetches it - and handed in as text. `STEREO_URL` is where.
 *
 * 🔴 AND THE LIGAND HALF (Tier 2: RDKit bounds, chiral centres, stereo bonds,
 * planar double bonds) IS NOT HERE YET. The vendor builds it from Boltz-1's CCD
 * pickle of RDKit molecules - a 346 MB file of Python objects - with RDKit's
 * bounds matrix and CIP ranks. `ligandTables` is the seam; without it a fold
 * with a ligand steers its polymers only, and says so.
 */
import { BioEmu2Graph, PROTEIN, DNA, RNA, NUCLEIC_TO_CCD, TOKEN_NAMES, CANONICAL_PROTEIN } from "./graph.js";

export const STEREO_COMMIT = "7846769";
export const STEREO_URL = `https://raw.githubusercontent.com/microsoft/bioemu2/${STEREO_COMMIT}/src/bioemu2/assets/stereo_chemistry.yaml`;
export const VDW_RADII = Object.freeze([
  1.2, 1.4, 2.2, 1.9, 1.8, 1.7, 1.6, 1.55, 1.5, 1.54,
  2.4, 2.2, 2.1, 2.1, 1.95, 1.8, 1.8, 1.88, 2.8, 2.4,
  2.3, 2.15, 2.05, 2.05, 2.05, 2.05, 2.0, 2.0, 2.0, 2.1,
  2.1, 2.1, 2.05, 1.9, 1.9, 2.02, 2.9, 2.55, 2.4, 2.3,
  2.15, 2.1, 2.05, 2.05, 2.0, 2.05, 2.1, 2.2, 2.2, 2.25,
  2.2, 2.1, 2.1, 2.16, 3.0, 2.7, 2.5, 2.48, 2.47, 2.45,
  2.43, 2.42, 2.4, 2.38, 2.37, 2.35, 2.33, 2.32, 2.3, 2.28,
  2.27, 2.25, 2.2, 2.1, 2.05, 2.0, 2.0, 2.05, 2.1, 2.05,
  2.2, 2.3, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.4,
  2.0, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
]);
const BOND_FALLBACK = [1.1, 2.2, 0.03];
const ANGLE_FALLBACK = [60.0, 180.0, 3.0];
const DISULFIDE_ANGLE = [104.2, 2.1];
const PEPTIDE_CONTEXTS = ["PEPTIDE", "GLY", "PRO_TRANS", "PRO_CIS"];
const NUCLEIC_ACID_LINK_KEY = "NA";

/** `min(atoms, reversed(atoms))`, as a tuple comparison: the table's key for either direction. */
function canonical(atoms) {
  const reversed = [...atoms].reverse();
  for (let i = 0; i < atoms.length; i += 1) {
    if (atoms[i] < reversed[i]) return atoms.join("_");
    if (atoms[i] > reversed[i]) return reversed.join("_");
  }
  return atoms.join("_");
}

/**
 * stereo_chemistry.yaml -> {bonds, angles, link_bonds, link_angles}, each
 * {residue: {canonical key: [mean, sigma]}}. The file is two levels of mapping
 * over `atom_atom: [mean, sigma]` lists, which is all this reads.
 */
export function parseStereoTables(text) {
  const root = {};
  const stack = [{ indent: -1, node: root }];
  let pendingList = null;
  for (const raw of text.split(/\r?\n/)) {
    if (!raw.trim() || raw.trimStart().startsWith("#")) continue;
    const indent = raw.length - raw.trimStart().length;
    const line = raw.trim();
    if (line.startsWith("- ")) {
      pendingList.push(Number(line.slice(2)));
      continue;
    }
    while (stack.length > 1 && stack[stack.length - 1].indent >= indent) stack.pop();
    const colon = line.lastIndexOf(":");
    // (a key is quoted only when it starts with a quote; C1'_C2'_H2' ends with one and is not)
    let keyText = line.slice(0, colon);
    if (/^['"]/.test(keyText)) keyText = keyText.slice(1, -1);
    const parent = stack[stack.length - 1].node;
    const rest = line.slice(colon + 1).trim();
    if (rest === "") {
      // a mapping or, if '- ' lines follow at a deeper indent, a list: decided by the first child
      const node = {};
      parent[keyText] = node;
      stack.push({ indent, node, key: keyText, parent });
      pendingList = [];
      // a list is attached lazily: when the next line is '- ', the node becomes the list
      Object.defineProperty(node, "__list", { value: pendingList, enumerable: false });
    } else {
      parent[keyText] = rest;
    }
  }
  const lists = (node) => {
    for (const [k, v] of Object.entries(node)) {
      if (v && typeof v === "object") {
        if (v.__list.length && Object.keys(v).length === 0) node[k] = v.__list;
        else lists(v);
      }
    }
  };
  lists(root);
  const table = (data) => Object.fromEntries(Object.entries(data).map(([res, entries]) => [res,
    Object.fromEntries(Object.entries(entries).map(([k, [m, s]]) => [canonical(k.split("_")), [m, s]]))]));
  return {
    bonds: table(root.monlib.bond_data), angles: table(root.monlib.angle_data),
    link_bonds: table(root.monlib_link.bond_data), link_angles: table(root.monlib_link.angle_data),
  };
}

const lookup = (table, residue, atoms) => table[residue]?.[canonical(atoms)] ?? null;
const peptideLinkKey = (following) => (following === "GLY" ? "GLY" : following === "PRO" ? "PRO_TRANS" : "PEPTIDE");

function pooledOf(table, names, peptide) {
  const ids = peptide ? PEPTIDE_CONTEXTS : CANONICAL_PROTEIN.map((t) => TOKEN_NAMES[t]);
  const entries = ids.map((i) => lookup(table, i, names)).filter((r) => r !== null);
  if (entries.length === 0) return null;
  return [Math.min(...entries.map((e) => e[0])), Math.max(...entries.map((e) => e[0])), Math.max(...entries.map((e) => e[1]))];
}

function bondReference(g, i, j, tables, pooled) {
  const names = [g.labels[i][1], g.labels[j][1]];
  const same = g.sameResidue(i, j);
  const following = g.peptideFollowingResidue(i, j);
  if (g.ordinaryX.has(i) || g.ordinaryX.has(j)) {
    if (same || following !== null) {
      const k = `${canonical(names)}|${following !== null}`;
      if (!pooled.has(k)) pooled.set(k, pooledOf(following !== null ? tables.link_bonds : tables.bonds, names, following !== null));
      return pooled.get(k) ?? BOND_FALLBACK;
    }
    return BOND_FALLBACK;
  }
  let ref;
  if (following !== null) ref = lookup(tables.link_bonds, peptideLinkKey(following), names);
  else if (g.isNaLinkBond(i, j)) ref = lookup(tables.link_bonds, NUCLEIC_ACID_LINK_KEY, names);
  else if (same) ref = lookup(tables.bonds, g.labels[i][0], names);
  else if (g.isDisulfide(i, j)) ref = lookup(tables.link_bonds, "PEPTIDE", names);
  else ref = null;
  return ref !== null ? [ref[0], ref[0], ref[1]] : BOND_FALLBACK;
}

function angleReference(g, oi, c, ok, tables, pooled) {
  const atoms = [oi, c, ok];
  const names = atoms.map((a) => g.labels[a][1]);
  const same = g.sameResidue(oi, c) && g.sameResidue(c, ok);
  const following = g.peptideAngleFollowingResidue(oi, c, ok);
  const peptide = following !== null;
  if (atoms.some((a) => g.ordinaryX.has(a))) {
    if (same || peptide) {
      const k = `${canonical(names)}|${peptide}`;
      if (!pooled.has(k)) pooled.set(k, pooledOf(peptide ? tables.link_angles : tables.angles, names, peptide));
      return pooled.get(k) ?? ANGLE_FALLBACK;
    }
    return ANGLE_FALLBACK;
  }
  let ref;
  if (following !== null) ref = lookup(tables.link_angles, peptideLinkKey(following), names);
  else if (g.isNaLinkAngle(oi, c, ok)) ref = lookup(tables.link_angles, NUCLEIC_ACID_LINK_KEY, names);
  else if (same) ref = lookup(tables.angles, g.labels[c][0], names);
  else if (canonical(names) === "CB_SG_SG" && [oi, ok].some((o) => g.isDisulfide(c, o))) ref = DISULFIDE_ANGLE;
  else ref = null;
  return ref !== null ? [ref[0], ref[0], ref[1]] : ANGLE_FALLBACK;
}

/** compute_geometry_bounds, in float32 as the vendor's torch arithmetic. */
export function geometryBounds(reference, tolerance, angle = false) {
  const [lo, hi, sigma] = angle ? reference.map((v) => (v * Math.PI) / 180) : reference;
  const width = Math.fround(Math.fround(sigma) * Math.fround(tolerance));
  const lower = Math.fround(Math.fround(lo) - width);
  const upper = Math.fround(Math.fround(hi) + width);
  return [Math.max(0, lower), Math.min(angle ? Math.PI : Infinity, upper)];
}

/** The vendor's index layout: (k, n) rows of a list of k-tuples - `np.array(x).reshape(-1, k).T`. */
function rows(tuples, k) {
  return Array.from({ length: k }, (_, r) => Int32Array.from(tuples, (t) => t[r]));
}
const rowsOf = (k, tuples) => rows(tuples, k);

/** build_polymer_steric_graph's endpoint mask. */
function polymerStericEligible(g, topology) {
  return topology.residueType.map((rtype, a) => {
    const mol = topology.molType[a];
    const protein = mol === PROTEIN && (CANONICAL_PROTEIN.includes(rtype) || g.ordinaryX.has(a));
    const nucleic = (mol === DNA || mol === RNA) && rtype in NUCLEIC_TO_CCD;
    return protein || nucleic;
  });
}

/**
 * Every potential's prepared arrays, keyed as the vendor's feats are.
 *
 * @param {object} topology featuriseBioemu2's
 * @param {object} stereo parseStereoTables(the fetched yaml)
 * @param {{asymId: Int32Array, entityId: Int32Array}} tokens the token features
 * @param {{sigmaTolerance?: number, ligandTables?: object}} options
 */
export function steeringTables(topology, stereo, tokens, options = {}) {
  const { sigmaTolerance = 10.0, ligandTables = null } = options;
  const n = topology.names.length;
  const nPad = topology.nPad;
  const g = new BioEmu2Graph(topology);
  const chain = new Int32Array(nPad);
  const pad = new Uint8Array(nPad);
  const radius = new Float32Array(nPad);
  for (let a = 0; a < n; a += 1) {
    chain[a] = topology.atomAsym[a];
    pad[a] = 1;
    radius[a] = [0.0, ...VDW_RADII][Math.min(topology.element[a], 118)];
  }
  const nChains = Math.max(...tokens.asymId) + 1;
  const sizes = new Int32Array(nChains);
  for (let a = 0; a < nPad; a += 1) if (pad[a]) sizes[chain[a]] += 1;
  const out = { atom_pad_mask: pad, atom_chain_id: chain, atom_vdw_radius: radius, chain_sizes: sizes };
  // SymmetricChainCOMPotential: chain pairs of one entity
  const pairs = [...new Map(Array.from(tokens.asymId, (a, t) => [`${a},${tokens.entityId[t]}`, [a, tokens.entityId[t]]])).values()]
    .sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  const sym = [];
  for (let x = 0; x < pairs.length; x += 1) {
    for (let y = x + 1; y < pairs.length; y += 1) if (pairs[x][1] === pairs[y][1]) sym.push([pairs[x][0], pairs[y][0]]);
  }
  out.symmetric_chain_index = rows(sym, 2);
  // InterChainVDWOverlapPotential
  const excl = g.nonbondedExclusions();
  const cross = excl.filter(([a, b]) => chain[a] !== chain[b]);
  out.interchain_steric_atom_mask = Uint8Array.from(pad, (v, a) => (v && sizes[chain[a]] > 1 ? 1 : 0));
  out.interchain_bonded_pair_key = cross.map(([a, b]) => a * nPad + b);
  // IntraChainVDWOverlapPotential / PolymerLocalClashPotential
  const eligibleAtoms = polymerStericEligible(g, topology);
  const eligible = new Uint8Array(nPad);
  for (let a = 0; a < n; a += 1) eligible[a] = eligibleAtoms[a] ? 1 : 0;
  out.polymer_steric_atom_mask = eligible;
  out.polymer_bonded_pair_key = excl.map(([a, b]) => a * nPad + b);
  out.polymer_local_clash_index = rowsOf(2, excl.filter(([a, b]) => !g.bondSet.has(`${a},${b}`)
    && eligible[a] && eligible[b] && topology.residueIdx[a] !== topology.residueIdx[b]));
  // BondLengthPotential / BondAnglePotential: references (lowest mean, highest mean, sigma) and the bounds at
  // `sigmaTolerance`, the default potentials'
  let pooled = new Map();
  out.bond_ref = g.bonds.map(([i, j]) => bondReference(g, i, j, stereo, pooled));
  pooled = new Map();
  out.angle_ref = g.angles.map(([i, k, c]) => angleReference(g, i, c, k, stereo, pooled));
  const bl = out.bond_ref.map((r) => geometryBounds(r, sigmaTolerance));
  out.bond_index = rows(g.bonds, 2);
  out.bond_lower = Float32Array.from(bl, (b) => b[0]);
  out.bond_upper = Float32Array.from(bl, (b) => b[1]);
  const al = out.angle_ref.map((r) => geometryBounds(r, sigmaTolerance, true));
  out.angle_index = rows(g.angles, 3);
  out.angle_lower = Float32Array.from(al, (a) => a[0]);
  out.angle_upper = Float32Array.from(al, (a) => a[1]);
  out.graph_bonds = g.bonds;
  // Tier 2
  Object.assign(out, ligandTables ?? emptyLigandTables());
  out.ligandSteered = ligandTables !== null || Object.keys(topology.ligandCcd).length === 0;
  return out;
}

export function emptyLigandTables() {
  return {
    rdkit_bounds_index: [], rdkit_bounds_bond_mask: [], rdkit_bounds_angle_mask: [], rdkit_upper_bounds: [],
    rdkit_lower_bounds: [], chiral_atom_index: [], chiral_atom_orientations: [], stereo_bond_index: [],
    stereo_bond_orientations: [], planar_bond_index: [],
  };
}
