/**
 * motif.js
 * --------
 * Coordinates a design is held to: the loaded structure's ligands, kept DNA strands and the motif residues inside
 * the designed chain, as the entries `hunt({motif})` inpaints into every design fold (cpu/af3/diffusion/inpaint.js) -
 * and the AME terms a held motif is judged by. The page's half of task-bench.mjs's fixed-target and AME guides.
 *
 * 🔴 TOKENS ARE THE FOLD'S, NOT THE STRUCTURE'S. A polymer chain is one token a residue, laid out in the job's chain
 * order with the designed chain at its own (run's) length, so a held residue's token moves with every chain before
 * it; a ligand is its whole chain (a negative token: asymId `chains + 1 + j`), its atoms found by name.
 *
 * 🔴 EVERY ENTRY IS OPTIONAL, because a deposited structure names atoms the fold does not build - a leaving atom, a 5'
 * phosphate, an OXT - and those are dropped and counted by `motifSlots`, never guessed at. A name the fold has TWICE
 * is still refused there.
 */
import { weightedRigidAlign } from "../af3/diffusion/rigid-align.js";

/** A loaded structure's residues in order: `pick(chain, residue, name)` decides which, heavy atoms only. */
function residues(atoms, pick) {
  const out = [];
  const index = new Map();
  for (let i = 0; i < atoms.chain_id.length; i += 1) {
    const chain = String(atoms.chain_id[i]);
    const residue = Number(atoms.res_id[i]);
    const name = String(atoms.res_name[i]).trim().toUpperCase();
    if (!pick(chain, residue, name)) continue;
    if (String(atoms.element[i] ?? "").toUpperCase() === "H") continue;
    const xyz = atoms.coord[i];
    if (xyz === undefined || ![0, 1, 2].every((a) => Number.isFinite(xyz[a]))) continue;
    const key = `${chain}:${residue}:${name}`;
    if (!index.has(key)) { index.set(key, out.length); out.push({ chain, residue, name, atoms: [] }); }
    out[index.get(key)].atoms.push([String(atoms.atom_name[i]).trim(), xyz[0], xyz[1], xyz[2]]);
  }
  return out;
}

/**
 * What a job holds in place, from a loaded structure.
 *
 * @param {object} input
 * @param {{chain_id: ArrayLike, res_id: ArrayLike, res_name: ArrayLike, atom_name: ArrayLike, element: ArrayLike,
 *   coord: ArrayLike<number[]>}} input.atoms the loaded structure
 * @param {string[]} input.chains the job's chains (`chainsOf`), the designed one empty or pre-shaped
 * @param {number} input.chainIndex the designed chain
 * @param {string[]} [input.ligandCodes] the job's ligands, each its own chain after the polymers
 * @param {object[]} [input.structureParts] `chainsOf`'s parts copied from the structure
 * @returns {{atoms: (length: number) => object[], motif: {position: number, res: string, atoms: Array}[],
 *   strands: number, ligands: number}} `atoms` is hunt's motif callback; `motif` the held residues inside the
 *   designed chain with their target atoms, which the AME criterion scores
 */
export function heldCoordinates({ atoms, chains, chainIndex, ligandCodes = [], structureParts = [] }) {
  const partResidues = (part) => residues(atoms, (chain, residue) => chain === part.chain
    && residue >= part.first && residue <= part.last);
  // the motif: held structure residues inside the designed chain, at their positions in it
  const motif = [];
  const strands = [];
  for (const part of structureParts) {
    if (!part.held) continue;
    const found = partResidues(part);
    if (part.index === chainIndex) {
      found.forEach((r, k) => motif.push({ position: part.at + k, res: r.name, atoms: r.atoms }));
    } else if (part.polymer === "dna" || part.polymer === "rna") {
      found.forEach((r, k) => strands.push({ index: part.index, position: part.at + k, atoms: r.atoms }));
    }
  }
  // each ligand code's n-th copy in the structure is the job's n-th ligand of that code
  const seen = new Map();
  const ligands = ligandCodes.map((code, j) => {
    const copy = seen.get(code) ?? 0;
    seen.set(code, copy + 1);
    const found = residues(atoms, (chain, residue, name) => name === code)[copy];
    if (found === undefined) throw new Error(`ligand ${code} (copy ${copy + 1}) is not in the loaded structure, so its coordinates cannot be held`);
    return { asym: chains.length + 1 + j, atoms: found.atoms };
  });
  const atomsFor = (length) => {
    const sizes = chains.map((sequence, i) => (i === chainIndex && sequence.length === 0 ? length : sequence.length));
    const offsets = sizes.map((_, i) => sizes.slice(0, i).reduce((a, b) => a + b, 0));
    return [
      ...motif.flatMap((r) => r.atoms.map(([atom, x, y, z]) => ({ token: offsets[chainIndex] + r.position, atom, xyz: [x, y, z], optional: true }))),
      ...strands.flatMap((r) => r.atoms.map(([atom, x, y, z]) => ({ token: offsets[r.index] + r.position, atom, xyz: [x, y, z], optional: true }))),
      ...ligands.flatMap((l) => l.atoms.map(([atom, x, y, z]) => ({ token: -l.asym, atom, xyz: [x, y, z], optional: true }))),
    ];
  };
  return { atoms: atomsFor, motif, strands: new Set(strands.map((s) => s.index)).size, ligands: ligands.length };
}

/** A PDB's heavy atoms: the designed chain's by residue number and name, and everyone else's. */
function allAtoms(pdb, chain) {
  const binder = new Map();
  const other = [];
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    if (line.slice(76, 78).trim() === "H") continue;
    const xyz = [+line.slice(30, 38), +line.slice(38, 46), +line.slice(46, 54)];
    if (line[21] === chain) binder.set(`${+line.slice(22, 26)}:${line.slice(12, 16).trim()}`, xyz);
    else other.push(xyz);
  }
  return { binder, other };
}

/** RMSD of two equal-length point lists after superposing the first onto the second. */
export function selfRmsd(a, b) {
  const n = a.length;
  const x = new Float32Array(n * 3);
  const y = new Float32Array(n * 3);
  a.forEach((p, i) => x.set(p, i * 3));
  b.forEach((p, i) => y.set(p, i * 3));
  const fitted = weightedRigidAlign(x, y, new Float32Array(n).fill(1), n);
  let sum = 0;
  for (let i = 0; i < n * 3; i += 1) sum += (fitted[i] - y[i]) ** 2;
  return Math.sqrt(sum / n);
}

/**
 * The AME terms (the paper's `ame` criterion): the refold's motif atoms at `positions` against the motif's target
 * coordinates, one superposition over all of them; and designed-chain atoms within 1.5 A of another chain's in the
 * DESIGN (the paper's clash term is on the generated structure).
 *
 * @param {{res: string, atoms: Array<[string, number, number, number]>}[]} motif
 * @param {number[]} positions each motif residue's 0-based position in the designed chain
 * @param {string} designPdb
 * @param {string} refoldPdb
 * @param {string} [chain="A"]
 * @returns {{motifRmsd: number, clashes: number}}
 */
export function motifTerms(motif, positions, designPdb, refoldPdb, chain = "A") {
  const refold = allAtoms(refoldPdb, chain);
  const want = [];
  const got = [];
  motif.forEach((m, k) => {
    for (const [name, x, y, z] of m.atoms) {
      const at = refold.binder.get(`${positions[k] + 1}:${name}`);
      // a name the fold does not build (an OXT, a leaving atom) was never guided and is not scored
      if (at === undefined) continue;
      want.push([x, y, z]);
      got.push(at);
    }
  });
  if (got.length < 3) throw new Error(`a motif of ${got.length} scorable atoms`);
  const design = allAtoms(designPdb, chain);
  let clashes = 0;
  for (const p of design.binder.values()) {
    for (const q of design.other) if ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2 < 2.25) clashes += 1;
  }
  return { motifRmsd: selfRmsd(got, want), clashes };
}

/**
 * The motif found unindexed: the assignment of its residues to the fold's residues of the same types (within 12 A of
 * another chain) whose atoms best fit it - a depth-first search pruned by each pair's centroid distance (within 3 A
 * of the motif's), the full RMSD at each leaf. Null when some type has no such residue.
 *
 * @returns {{rmsd: number, positions: number[]}|null}
 */
export function motifMatch(motif, pdb, chain = "A") {
  const found = new Map();
  const other = [];
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    if (line.slice(76, 78).trim() === "H") continue;
    const xyz = [+line.slice(30, 38), +line.slice(38, 46), +line.slice(46, 54)];
    if (line[21] !== chain) { other.push(xyz); continue; }
    const n = +line.slice(22, 26);
    if (!found.has(n)) found.set(n, { name: line.slice(17, 20).trim(), atoms: new Map() });
    found.get(n).atoms.set(line.slice(12, 16).trim(), xyz);
  }
  const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
  const mean = (ps) => [0, 1, 2].map((c) => ps.reduce((s, p) => s + p[c], 0) / ps.length);
  // only the names every candidate must carry: what the fold builds for that residue type
  const names = motif.map((m) => m.atoms.map(([a]) => a));
  const target = motif.map((m) => m.atoms.map(([, x, y, z]) => [x, y, z]));
  const want = target.map(mean);
  const candidates = motif.map((m, k) => {
    const out = [];
    for (const [n, r] of found) {
      if (r.name !== m.res) continue;
      const kept = names[k].filter((a) => r.atoms.has(a));
      if (kept.length < Math.min(3, names[k].length)) continue;
      const tip = kept.map((a) => r.atoms.get(a));
      if (!tip.some((p) => other.some((q) => d2(p, q) < 144))) continue;
      out.push({ position: n - 1, kept, centre: mean(tip), atoms: r.atoms });
    }
    return out;
  });
  if (candidates.some((c) => c.length === 0)) return null;
  let best = null;
  let leaves = 0;
  const chosen = [];
  const search = (k) => {
    if (leaves > 20000) return;
    if (k === motif.length) {
      leaves += 1;
      const got = [];
      const wanted = [];
      chosen.forEach((c, j) => names[j].forEach((a, i) => {
        if (!c.atoms.has(a)) return;
        got.push(c.atoms.get(a));
        wanted.push(target[j][i]);
      }));
      const rmsd = selfRmsd(got, wanted);
      if (best === null || rmsd < best.rmsd) best = { rmsd, positions: chosen.map((c) => c.position) };
      return;
    }
    for (const c of candidates[k]) {
      if (chosen.some((o) => o.position === c.position)) continue;
      if (chosen.some((o, j) => Math.abs(Math.sqrt(d2(o.centre, c.centre)) - Math.sqrt(d2(want[j], want[k]))) > 3)) continue;
      chosen.push(c);
      search(k + 1);
      chosen.pop();
    }
  };
  search(0);
  return best;
}
