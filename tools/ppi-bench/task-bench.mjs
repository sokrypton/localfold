// Protein Hunter on RFD4-Proteina's DNA- and small-molecule-binder benchmarks (tasks.json, from prep-tasks.py),
// every fold on the native CUDA port - native-bench.mjs's loop with the target given by an AlphaFold 3 job (DNA
// by sequence, a ligand by SMILES or CCD code), MPNN that sees it (NA-MPNN for DNA, LigandMPNN for a ligand), and
// each task's own pass criteria (the paper's S8.2):
//   dna: min interface PAE <= 1.5 and binder backbone RMSD <= 8 A after aligning all DNA atoms, design to refold
//   sm:  protein-ligand ipTM >= 0.80, min interface PAE <= 1.5, protein backbone RMSD <= 1.5 A and ligand RMSD
//        <= 5 A after protein-backbone alignment
// The interface terms are the paper's oracle's: chain_pair_* averaged both ways, min (PAE) or mean (ipTM) over the
// binder's partners.
// usage: node task-bench.mjs <out.jsonl> <config.json> <target>:<attempt> ...
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { createHash } from "node:crypto";

const run = promisify(execFile);
const HERE = path.dirname(new URL(import.meta.url).pathname);
const ROOT = path.resolve(HERE, "../..");
const { runDesign } = await import(`${ROOT}/webgpu/design/hunter-loop.js`);
const { designChain } = await import(`${ROOT}/cpu/design/mpnn-bridge.js`);
const { Weights } = await import(`${ROOT}/cpu/design/mpnn/weights.js`);
const { Model } = await import(`${ROOT}/cpu/design/mpnn/model.js`);
const { enableAcceleration } = await import(`${ROOT}/cpu/design/mpnn/accel.js`);
const { contactProblem } = await import(`${ROOT}/cpu/design/contact.js`);
const { weightedRigidAlign } = await import(`${ROOT}/cpu/af3/diffusion/rigid-align.js`);

const [out, configPath, ...jobs] = process.argv.slice(2);
const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
const tasks = JSON.parse(fs.readFileSync(path.join(HERE, "tasks.json"), "utf8"));
const done = new Set(fs.existsSync(out)
  ? fs.readFileSync(out, "utf8").split("\n").filter(Boolean).map((l) => { const d = JSON.parse(l); return `${d.target}:${d.attempt}`; })
  : []);
const bytes = (file) => { const b = fs.readFileSync(file); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
if ((await enableAcceleration(bytes(`${ROOT}/web/vendor/mpnn/kernels.wasm`))) === null) throw new Error("MPNN's accelerator did not load");
const checkpoint = { dna: "na_mpnn_design.mpnn", sm: "ligandmpnn_v_32_020_25.mpnn", ame: "ligandmpnn_v_32_020_25.mpnn" };
const mpnns = {};
const mpnnFor = (task) => (mpnns[task] ??= new Model(Weights.fromArrayBuffer(bytes(`${ROOT}/web/public/mpnn/${config.mpnn?.[task] ?? checkpoint[task]}`))));
const work = fs.mkdtempSync(path.join(os.tmpdir(), "task-native-"));

function mulberry(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const seedOf = (text) => createHash("sha1").update(text).digest().readUInt32LE(0);
const LETTERS = "ABCDEFGHIJ";

/** One native fold of [binder, ...the task's chains] from an AF3 job; the structure and the paper's terms. */
async function fold(t, binder, { steps, recycles, seed, motif, msa, model }) {
  const file = path.join(work, `f${process.pid}-${Date.now()}.pdb`);
  const job = file.replace(".pdb", ".json");
  // `msa`: sequences handed to AF3 as the binder's unpaired alignment (mpnnMsa); none, a single-sequence fold
  const alignment = msa === undefined ? {} : { unpairedMsa: [`>query\n${binder}`, ...msa.map((m, k) => `>mpnn${k}\n${m}`)].join("\n") + "\n", pairedMsa: "" };
  const sequences = [{ protein: { id: "A", sequence: binder, ...alignment } }, ...t.chains.map((c, i) => {
    const id = LETTERS[i + 1];
    if (c.kind === "dna") return { dna: { id, sequence: c.sequence } };
    return { ligand: c.ccd !== undefined ? { id, ccdCodes: [c.ccd] } : { id, smiles: c.smiles } };
  })];
  fs.writeFileSync(job, JSON.stringify({ dialect: "alphafold3", version: 2, name: "j", modelSeeds: [seed], sequences }));
  // `motif`: a design fold inpainted with the task's catalytic motif (cuda/af3 --motif; never a check or the evaluation)
  const guide = motif === undefined ? [] : [`--motif=${motif}`, ...(config.motifUntil === undefined ? [] : [`--motif-until=${config.motifUntil}`])];
  // `model`: another AF3-lineage checkpoint (cuda/af3/fold --model=, its own standalone process - one resident server)
  await run(`${ROOT}/cuda/af3/fold`, [file, `--job=${job}`, ...(model === undefined || model === "af3" ? [] : [`--model=${model}`]), "--", `--steps=${steps}`, `--recycles=${recycles}`, `--seed=${seed}`, ...guide],
    { maxBuffer: 1 << 26 });
  const pdb = fs.readFileSync(file, "utf8");
  const s = JSON.parse(fs.readFileSync(file.replace(".pdb", "_summary_confidences.json"), "utf8"));
  for (const f of fs.readdirSync(work)) if (f.startsWith(path.basename(file, ".pdb"))) fs.rmSync(path.join(work, f));
  const k = s.chain_pair_pae_min.length;
  const both = (m, j) => (m[0][j] + m[j][0]) / 2;
  let ipae = Infinity, iptm = 0;
  for (let j = 1; j < k; j += 1) { ipae = Math.min(ipae, both(s.chain_pair_pae_min, j)); iptm += both(s.chain_pair_iptm, j) / (k - 1); }
  const chainPairIptm = {};
  for (let i = 0; i < k; i += 1) for (let j = i + 1; j < k; j += 1) chainPairIptm[`${i + 1}|${j + 1}`] = Math.max(s.chain_pair_iptm[i][j], s.chain_pair_iptm[j][i]);
  return { pdb, binderPtm: s.chain_ptm[0], ipae, iptm, confidence: { iptm: s.iptm, ptm: s.ptm, meanPlddt: s.mean_plddt, chainPairIptm } };
}

/** One ESMFold2 fold (cuda/ef2, a fresh process - 1.2 s with its start-up, so no second resident server) of the same
 * job, its terms read as AF3's are: the binder's chain pTM, ipTM and interface PAE from the chain-level summary
 * cuda/ef2 writes since 2026-10-30. */
async function foldEf2(t, binder, { seed }) {
  const file = path.join(work, `e${process.pid}-${Date.now()}.pdb`);
  const job = file.replace(".pdb", ".json");
  const sequences = [{ protein: { id: "A", sequence: binder } }, ...t.chains.map((c, i) => {
    const id = LETTERS[i + 1];
    if (c.kind === "dna") return { dna: { id, sequence: c.sequence } };
    return { ligand: c.ccd !== undefined ? { id, ccdCodes: [c.ccd] } : { id, smiles: c.smiles } };
  })];
  fs.writeFileSync(job, JSON.stringify({ dialect: "alphafold3", version: 2, name: "j", modelSeeds: [seed], sequences }));
  await run(`${ROOT}/cuda/ef2/fold`, [file, `--job=${job}`, "--", `--seed=${seed}`], { maxBuffer: 1 << 26 });
  const pdb = fs.readFileSync(file, "utf8");
  const s = JSON.parse(fs.readFileSync(file.replace(".pdb", "_summary_confidences.json"), "utf8"));
  for (const f of fs.readdirSync(work)) if (f.startsWith(path.basename(file, ".pdb"))) fs.rmSync(path.join(work, f));
  if (s.chain_pair_iptm === undefined) throw new Error("cuda/ef2 wrote no chain-level summary - rebuild it");
  const k = s.chain_pair_pae_min.length;
  const both = (m, j) => (m[0][j] + m[j][0]) / 2;
  let ipae = Infinity, iptm = 0;
  for (let j = 1; j < k; j += 1) { ipae = Math.min(ipae, both(s.chain_pair_pae_min, j)); iptm += both(s.chain_pair_iptm, j) / (k - 1); }
  const chainPairIptm = {};
  for (let i = 0; i < k; i += 1) for (let j = i + 1; j < k; j += 1) chainPairIptm[`${i + 1}|${j + 1}`] = Math.max(s.chain_pair_iptm[i][j], s.chain_pair_iptm[j][i]);
  return { pdb, binderPtm: s.chain_ptm[0], ipae, iptm, confidence: { iptm: s.iptm, ptm: s.ptm, meanPlddt: s.mean_plddt, chainPairIptm } };
}

/** Atoms by chain:residue:name: the binder's backbone, and every atom of the other chains. */
function atoms(pdb) {
  const binder = new Map(), rest = new Map();
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    const name = line.slice(12, 16).trim(), chain = line[21];
    const key = `${chain}:${line.slice(22, 26).trim()}:${name}`;
    const xyz = [+line.slice(30, 38), +line.slice(38, 46), +line.slice(46, 54)];
    if (chain === "A") { if (["N", "CA", "C", "O"].includes(name)) binder.set(key, xyz); } else rest.set(key, xyz);
  }
  return { binder, rest };
}
/** RMSD of `measure` after fitting `fit` (design onto refold; both maps keyed alike). */
function fittedRmsd(fitD, fitR, measureD, measureR) {
  const fk = [...fitD.keys()].filter((k) => fitR.has(k)), mk = [...measureD.keys()].filter((k) => measureR.has(k));
  const keys = [...fk, ...mk], n = keys.length;
  const x = new Float32Array(n * 3), y = new Float32Array(n * 3), w = new Float32Array(n);
  keys.forEach((k, i) => { const fit = i < fk.length; x.set((fit ? fitR : measureR).get(k), i * 3); y.set((fit ? fitD : measureD).get(k), i * 3); w[i] = fit ? 1 : 0; });
  const fitted = weightedRigidAlign(x, y, w, n);
  let sum = 0;
  for (let i = fk.length; i < n; i += 1) for (let a = 0; a < 3; a += 1) sum += (fitted[i * 3 + a] - y[i * 3 + a]) ** 2;
  return Math.sqrt(sum / mk.length);
}
/** Every heavy atom of the binder by residue number and name, and the other chains' heavy atoms. */
function allAtoms(pdb) {
  const binder = new Map(), other = [];
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    if (line.slice(76, 78).trim() === "H") continue;
    const xyz = [+line.slice(30, 38), +line.slice(38, 46), +line.slice(46, 54)];
    if (line[21] === "A") binder.set(`${+line.slice(22, 26)}:${line.slice(12, 16).trim()}`, xyz); else other.push(xyz);
  }
  return { binder, other };
}
/** RMSD of two equal-length point lists after superposing the first onto the second. */
function selfRmsd(a, b) {
  const n = a.length, x = new Float32Array(n * 3), y = new Float32Array(n * 3), w = new Float32Array(n).fill(1);
  a.forEach((p, i) => x.set(p, i * 3)); b.forEach((p, i) => y.set(p, i * 3));
  const fitted = weightedRigidAlign(x, y, w, n);
  let sum = 0;
  for (let i = 0; i < n * 3; i += 1) sum += (fitted[i] - y[i]) ** 2;
  return Math.sqrt(sum / n);
}
/** The AME terms (the paper's `ame` criterion): the refold's motif atoms at the design's positions against the task's
 * conditioned coordinates, one superposition over all of them; and polymer-ligand pairs under 1.5 A in the DESIGN
 * (its clash metric is on the generated structure). */
function motifTerms(t, positions, designPdb, refoldPdb) {
  const f = allAtoms(refoldPdb), want = [], got = [];
  t.motif.forEach((m, k) => {
    for (const [name, x, y, z] of m.atoms) {
      const at = f.binder.get(`${positions[k] + 1}:${name}`);
      if (at === undefined) throw new Error(`refold has no ${m.res} ${name} at residue ${positions[k] + 1}`);
      want.push([x, y, z]); got.push(at);
    }
  });
  const d = allAtoms(designPdb);
  let clashes = 0;
  for (const p of d.binder.values()) for (const q of d.other) if ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2 < 2.25) clashes += 1;
  return { motifRmsd: selfRmsd(got, want), clashes };
}
/** AME, unindexed: the assignment of the motif's residues to the fold's residues of the same types (within 12 A of a
 * ligand) whose atoms best fit the motif - a depth-first search pruned by each pair's tip-centroid distance (within
 * 3 A of the motif's), the full RMSD at each leaf. Null when no residue of some type is near a ligand. */
function motifMatch(t, pdb) {
  const res = new Map(), lig = [];
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    if (line.slice(76, 78).trim() === "H") continue;
    const xyz = [+line.slice(30, 38), +line.slice(38, 46), +line.slice(46, 54)];
    if (line[21] !== "A") { lig.push(xyz); continue; }
    const n = +line.slice(22, 26);
    if (!res.has(n)) res.set(n, { name: line.slice(17, 20), atoms: new Map() });
    res.get(n).atoms.set(line.slice(12, 16).trim(), xyz);
  }
  const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
  const mean = (ps) => [0, 1, 2].map((c) => ps.reduce((s, p) => s + p[c], 0) / ps.length);
  const want = t.motif.map((m) => mean(m.atoms.map(([, x, y, z]) => [x, y, z])));
  const cands = t.motif.map((m) => {
    const out = [];
    for (const [n, r] of res) {
      if (r.name !== m.res || !m.atoms.every(([a]) => r.atoms.has(a))) continue;
      const tip = m.atoms.map(([a]) => r.atoms.get(a));
      if (!tip.some((p) => lig.some((q) => d2(p, q) < 144))) continue;
      out.push({ position: n - 1, tip, centre: mean(tip) });
    }
    return out;
  });
  if (cands.some((c) => c.length === 0)) return null;
  let best = null, leaves = 0;
  const chosen = [];
  const dfs = (k) => {
    if (leaves > 20000) return;
    if (k === t.motif.length) {
      leaves += 1;
      const got = chosen.flatMap((c) => c.tip), want2 = t.motif.flatMap((m) => m.atoms.map(([, x, y, z]) => [x, y, z]));
      const rmsd = selfRmsd(got, want2);
      if (best === null || rmsd < best.rmsd) best = { rmsd, positions: chosen.map((c) => c.position) };
      return;
    }
    for (const c of cands[k]) {
      if (chosen.some((o) => o.position === c.position)) continue;
      if (chosen.some((o, j) => Math.abs(Math.sqrt(d2(o.centre, c.centre)) - Math.sqrt(d2(want[j], want[k]))) > 3)) continue;
      chosen.push(c); dfs(k + 1); chosen.pop();
    }
  };
  dfs(0);
  return best;
}
/** The largest ligand-chain centroid distance, design to refold, after fitting the binder's backbone (the `disco` term). */
function ligandCentroid(designPdb, refoldPdb) {
  const parse = (pdb) => {
    const bb = new Map(), lig = new Map();
    for (const line of pdb.split("\n")) {
      if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
      const name = line.slice(12, 16).trim(), chain = line[21];
      const xyz = [+line.slice(30, 38), +line.slice(38, 46), +line.slice(46, 54)];
      if (chain === "A") { if (["N", "CA", "C", "O"].includes(name)) bb.set(`${line.slice(22, 26).trim()}:${name}`, xyz); }
      else if (line.slice(76, 78).trim() !== "H") { if (!lig.has(chain)) lig.set(chain, []); lig.get(chain).push(xyz); }
    }
    return { bb, lig };
  };
  const d = parse(designPdb), f = parse(refoldPdb);
  const keys = [...d.bb.keys()].filter((k) => f.bb.has(k));
  const chains = [...d.lig.keys()];
  const n = keys.length + chains.length;
  const x = new Float32Array(n * 3), y = new Float32Array(n * 3), w = new Float32Array(n);
  const mean = (ps) => [0, 1, 2].map((c) => ps.reduce((s, p) => s + p[c], 0) / ps.length);
  keys.forEach((k, i) => { x.set(f.bb.get(k), i * 3); y.set(d.bb.get(k), i * 3); w[i] = 1; });
  chains.forEach((c, j) => { x.set(mean(f.lig.get(c)), (keys.length + j) * 3); y.set(mean(d.lig.get(c)), (keys.length + j) * 3); });
  const fitted = weightedRigidAlign(x, y, w, n);
  let worst = 0;
  for (let j = keys.length; j < n; j += 1) worst = Math.max(worst, Math.hypot(fitted[j * 3] - y[j * 3], fitted[j * 3 + 1] - y[j * 3 + 1], fitted[j * 3 + 2] - y[j * 3 + 2]));
  return worst;
}
function verdict(task, designPdb, r, t, positions) {
  const d = atoms(designPdb), f = atoms(r.pdb);
  if (task === "ame") {
    const rmsd = fittedRmsd(d.binder, f.binder, d.binder, f.binder);
    const m = motifTerms(t, positions, designPdb, r.pdb);
    return { rmsd, ...m, pass: r.ipae < 1.5 && rmsd <= 2 && m.motifRmsd <= 1.5 && m.clashes === 0 };
  }
  if (task === "dna") {
    const rmsd = fittedRmsd(d.rest, f.rest, d.binder, f.binder);
    return { rmsd, pass: r.ipae <= 1.5 && rmsd <= 8 };
  }
  const rmsd = fittedRmsd(d.binder, f.binder, d.binder, f.binder);
  const ligandRmsd = fittedRmsd(d.binder, f.binder, d.rest, f.rest);
  // `criterion: "disco"`: the paper's other small-molecule criterion judges the attempt - binder backbone RMSD <= 2 A and
  // every ligand chain's centroid within 2 A of the design's after the backbone fit (disco-score.py's terms); no confidence
  const centroid = ligandCentroid(designPdb, r.pdb);
  const disco = rmsd <= 2 && centroid <= 2;
  return { rmsd, ligandRmsd, centroid, disco,
    pass: config.criterion === "disco" ? disco : r.iptm >= 0.8 && r.ipae <= 1.5 && rmsd <= 1.5 && ligandRmsd <= 5 };
}

/** `minFootprint`: a design must touch at least that many residues of the other chains (heavy atoms within 4.5 A) -
 * on a 10-bp duplex a binder that grips one end or three bases moves 15-40 A on refolding, one that spans both
 * strands holds its pose. */
function footprintProblem(pdb) {
  if (config.minFootprint === undefined) return null;
  const binder = [], other = new Map();
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    if (line.slice(76, 78).trim() === "H") continue;
    const xyz = [+line.slice(30, 38), +line.slice(38, 46), +line.slice(46, 54)];
    if (line[21] === "A") binder.push(xyz);
    else { const k = line.slice(21, 26); if (!other.has(k)) other.set(k, []); other.get(k).push(xyz); }
  }
  let touched = 0;
  for (const atoms of other.values()) {
    if (atoms.some((a) => binder.some((b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2 < 4.5 ** 2))) touched += 1;
  }
  return touched >= config.minFootprint ? null : `footprint ${touched} residues, needs ${config.minFootprint}`;
}

/** `dnaFrame`: the duplex as AF3 folds it from its sequence (beside a 5-residue stub, the binder slot filled), once a
 * target - nothing the benchmark's "unfixed" input does not give - every heavy atom by chain, residue and name. */
const dnaFrames = {};
async function dnaFrame(name, t) {
  if (dnaFrames[name] !== undefined) return dnaFrames[name];
  const r = await fold(t, "GGGGG", { steps: 200, recycles: 5, seed: 11 });
  const atomsOf = [];
  for (const line of r.pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    if (line[21] === "A" || line.slice(76, 78).trim() === "H") continue;
    atomsOf.push({ chain: "BCDEFGHIJ".indexOf(line[21]), residue: +line.slice(22, 26), atom: line.slice(12, 16).trim(),
      xyz: [+line.slice(30, 38), +line.slice(38, 46), +line.slice(46, 54)] });
  }
  return (dnaFrames[name] = atomsOf);
}

/** AME: each ligand chain's atom names as AF3 builds it (a stub fold, once a target) - the input conditions atoms
 * AF3's CCD component may not have (FBP's O1, a leaving atom), and those cannot be guided. Only the PROTEIN motif atoms
 * are scored, so a ligand atom left out weakens the guide and changes nothing judged; the count is on the row. */
const ligandNames = {};
async function ligandAtomNames(name, t) {
  if (ligandNames[name] !== undefined) return ligandNames[name];
  const r = await fold(t, "GGGGG", { steps: 10, recycles: 0, seed: 11 });
  const names = t.chains.map(() => new Set());
  for (const line of r.pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    const k = "BCDEFGHIJ".indexOf(line[21]);
    if (k >= 0) names[k].add(line.slice(12, 16).trim());
  }
  return (ligandNames[name] = names);
}

/** Each ligand chain's atoms in token order, as AF3 builds them: [{name, element}] (the stub fold's PDB, in order). */
const ligandOrder = {};
async function ligandAtomsInOrder(name, t) {
  if (ligandOrder[name] !== undefined) return ligandOrder[name];
  const r = await fold(t, "GGGGG", { steps: 10, recycles: 0, seed: 11 });
  const chains = t.chains.map(() => []);
  for (const line of r.pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    const k = "BCDEFGHIJ".indexOf(line[21]);
    if (k >= 0 && line.slice(76, 78).trim() !== "H") chains[k].push({ name: line.slice(12, 16).trim(), element: line.slice(76, 78).trim() });
  }
  return (ligandOrder[name] = chains);
}

// "F-<key>": a small-molecule target with its ligand FIXED (the paper's studio_figure_2c_fixed) - the same task,
// the file's conformer inpainted into every design fold
function taskOf(name) {
  const t = tasks[name] ?? (name.startsWith("F-") && tasks[name.slice(2)]?.task === "sm" ? tasks[name.slice(2)] : undefined);
  if (t === undefined) throw new Error(`no task ${name}`);
  return t;
}

async function attempt(name, index) {
  const t = taskOf(name);
  const fixedLigand = name.startsWith("F-") && t.task === "sm";
  const cfg = config;
  const rng = mulberry(seedOf(`${cfg.pair ?? cfg.tag}:${name}:${index}`));
  // `lengths`: one of a list (the paper's 150/200/250 for a small molecule); `length`: a range; else the file's range
  const [lo, hi] = cfg.length?.[t.task] ?? t.binder;
  const draw = rng();
  const length = cfg.lengths?.[t.task] !== undefined ? cfg.lengths[t.task][Math.floor(draw * cfg.lengths[t.task].length)]
    : lo + Math.floor(draw * (hi - lo + 1));
  const evalSeed = 1 + Math.floor(rng() * 2 ** 30);
  const seed = 1000 * index + 7;
  const mpnn = mpnnFor(t.task);
  const assess = async (binder, designPdb, foldSeed, positions) => {
    const r = await fold(t, binder, { steps: 200, recycles: 5, seed: foldSeed });
    return { ...r, ...verdict(t.task, designPdb, r, t, positions) };
  };
  // `check.seeds`: a candidate passes the internal check only if it passes at that many seeds (the first that
  // fails ends it) - a small molecule's interface PAE sits on the 1.5 line, and one passing seed is 64% likely to
  // mean a passing evaluation
  const checkAt = async (entry, s0) => {
    let result;
    for (let k = 0; k < (cfg.check?.seeds ?? 1); k += 1) {
      result = await assess(entry.sequence, entry.pdb, s0 + 104729 * k, entry.positions);
      result.seedsPassed = k + (result.pass ? 1 : 0);
      if (!result.pass) break;
    }
    return result;
  };
  // the design-time stop: confident about the terms the task is judged on
  const confidenceOnly = t.task === "dna" || t.task === "ame";     // (neither criterion reads ipTM)
  // (AME: and the design's own motif within `stop.motif`, unclashed)
  const motifOk = (e, bound) => t.task !== "ame" || bound === undefined || (e.motifRmsd <= bound && e.clashes === 0);
  const looksDone = (e) => cfg.stop !== undefined && !e.guided && e.ipae <= cfg.stop.ipae && (confidenceOnly || e.iptm >= cfg.stop.iptm) && motifOk(e, cfg.stop.motif);
  // 🔴 THE FOOTPRINT SCREENS WHAT IS CHECKED, NOT WHAT THE LOOP DESIGNS FROM: as a per-cycle veto it failed most
  // docked folds, and a docked cycle was followed by a docked one 2 times in 68 against 92 in 290 without it
  const promisingOf = (e) => !e.guided && e.ipae <= cfg.check.ipae && (confidenceOnly || e.iptm >= cfg.check.iptm) && footprintProblem(e.pdb) === null
    && motifOk(e, cfg.check.motif);
  const records = [], eligible = [], checked = [];
  let unguided = 0;     // (AME: conditioned ligand atoms AF3 does not build, left out of the guide)
  let runs = 0, foldSeconds = 0;
  const started = Date.now();
  for (let restart = 0; restart < (cfg.restarts ?? 1); restart += 1) {
    runs += 1;
    const controller = new AbortController();
    const mine = [];
    // AME: the motif's residues at `motifGap`-spaced random positions (unindexed - the designer places them), held
    // through every MPNN draw, and every design fold inpainted with the motif and the ligands' conditioned atoms
    let positions, motif, start = "";
    // `guideCycles`: only cycles before it are inpainted; the rest fold free, and only a free cycle can be checked -
    // a guided fold's motif agreement is the guide's, not the sequence's
    const guidedAt = (cycle) => cfg.guideCycles === undefined || cycle < cfg.guideCycles;
    // `mpnnMsa` {depth, temperature}: a design fold's alignment, MPNN's own draws off the previous cycle's backbone
    // (the motif held) at a higher temperature - `depth` of them after cycle 0, falling linearly to none at the last
    // cycle, so the run leans on the alignment early and ends single-sequence, as it is judged
    const mpnnMsaAt = (context) => {
      if (cfg.mpnnMsa === undefined || context.previous === undefined) return undefined;
      const n = Math.round(cfg.mpnnMsa.depth * (1 - context.cycle / Math.max(1, cfg.cycles - 1)));
      if (n <= 0) return undefined;
      const random = mulberry(seedOf(`msa:${name}:${index}:${restart}:${context.cycle}`));
      return Array.from({ length: n }, () => designChain(mpnn, { pdb: context.previous, chain: "A",
        temperature: cfg.mpnnMsa.temperature, omit: cfg.omit?.[t.task] ?? "C", random, kind: "protein", hold: positions }).sequence);
    };
    if (t.task === "ame") {
      const gap = cfg.motifGap ?? 4, edge = cfg.motifEdge ?? 8;
      for (let tries = 0; ; tries += 1) {
        if (tries > 10000) throw new Error(`no placement of ${t.motif.length} residues ${gap} apart in ${length}`);
        positions = t.motif.map(() => edge + Math.floor(rng() * (length - 2 * edge)));
        const sorted = [...positions].sort((a, b) => a - b);
        if (sorted.every((p, i) => i === 0 || p - sorted[i - 1] >= gap)) break;
      }
      const one = { ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E", GLY: "G", HIS: "H", ILE: "I",
        LEU: "L", LYS: "K", MET: "M", PHE: "F", PRO: "P", SER: "S", THR: "T", TRP: "W", TYR: "Y", VAL: "V" };
      const letters = Array(length).fill("X");
      t.motif.forEach((m, k) => { letters[positions[k]] = one[m.res]; });
      start = letters.join("");
      motif = path.join(work, `motif-${process.pid}-${name}-${index}-${restart}.txt`);
      const built = await ligandAtomNames(name, t);
      unguided = t.chains.reduce((n, c, i) => n + (c.conditioned ?? []).filter(([a]) => !built[i].has(a)).length, 0);
      fs.writeFileSync(motif, [
        ...t.motif.flatMap((m, k) => m.atoms.map(([a, x, y, z]) => `${positions[k]} ${a} ${x} ${y} ${z}`)),
        ...t.chains.flatMap((c, i) => (c.conditioned ?? []).filter(([a]) => built[i].has(a)).map(([a, x, y, z]) => `${-(i + 2)} ${a} ${x} ${y} ${z}`)),
      ].join("\n") + "\n");
    } else if (fixedLigand) {
      unguided = 0;
      // the conformer by token: a SMILES chain's atoms are its tokens in SMILES order, checked element by element
      // against what AF3 built (a mismatch is refused, never guessed past); a CCD chain's by atom name
      const built = await ligandAtomsInOrder(name, t);
      const lines = [];
      t.chains.forEach((c, i) => {
        if (c.conformer !== undefined) {
          const got = built[i].map((a) => a.element.toUpperCase()).join(" "), want = c.conformer.map(([e]) => e.toUpperCase()).join(" ");
          if (got !== want) throw new Error(`${name} chain ${i}: AF3 built ${got} where the conformer is ${want}`);
          c.conformer.forEach(([, x, y, z], k) => lines.push(`${-(i + 2)} ${built[i][k].name} ${x} ${y} ${z}`));
        } else {
          const names = new Set(built[i].map((a) => a.name)), used = new Set();
          for (const [n, x, y, z] of c.conformerNames ?? []) {
            if (names.has(n) && !used.has(n)) { used.add(n); lines.push(`${-(i + 2)} ${n} ${x} ${y} ${z}`); } else unguided += 1;
          }
        }
      });
      // (under three atoms - an ion, two coppers - there is no shape to hold, and a fit needs three: unguided)
      if (lines.length >= 3) {
        motif = path.join(work, `motif-${process.pid}-${name}-${index}-${restart}.txt`);
        fs.writeFileSync(motif, lines.join("\n") + "\n");
      } else unguided += lines.length;
    } else if (t.task === "dna" && t.dnaAtoms !== undefined) {
      // DNA FIXED (the paper's dna_fixed, an "F-" target): the given duplex inpainted into every design fold, token by
      // token - only the atoms AF3 builds (the stub fold's names; a 5' phosphate may be absent); the check and the
      // evaluation fold it free, as for the unfixed set
      const offsets = [length];
      for (const c of t.chains) offsets.push(offsets.at(-1) + c.sequence.length);
      const builtDna = new Set((await dnaFrame(name, t)).map((a) => `${a.chain}:${a.residue}:${a.atom}`));
      const kept = t.dnaAtoms.filter(([i, r, n]) => builtDna.has(`${i}:${r}:${n}`));
      unguided = t.dnaAtoms.length - kept.length;
      motif = path.join(work, `motif-${process.pid}-${name}-${index}-${restart}.txt`);
      fs.writeFileSync(motif, kept.map(([i, r, n, x, y, z]) => `${offsets[i] + r - 1} ${n} ${x} ${y} ${z}`).join("\n") + "\n");
    } else if (t.task === "dna" && cfg.dnaFrame) {
      // 🔴 THE DUPLEX HELD IN ITS OWN FOLDED SHAPE: every design fold inpaints it (cuda/af3 --motif), each nucleotide a
      // token after the binder's `length`, so the binder is designed against B-form DNA rather than whatever duplex
      // this fold's noise made; the check and the evaluation fold it free
      const offsets = [length];
      for (const c of t.chains) offsets.push(offsets.at(-1) + c.sequence.length);
      motif = path.join(work, `motif-${process.pid}-${name}-${index}-${restart}.txt`);
      fs.writeFileSync(motif, (await dnaFrame(name, t)).map((a) => `${offsets[a.chain] + a.residue - 1} ${a.atom} ${a.xyz.join(" ")}`).join("\n") + "\n");
    }
    let lastDocked = null, undockedRun = 0;
    const statesOf = new Map();     // (a design fold's structure -> the other models' structures of its sequence)
    try {
      const stream = runDesign({
        chains: [start], chainIndex: 0, cycles: cfg.cycles, length, run: restart,
        ...(positions === undefined ? {} : { hold: positions }),
        seed: seed + 101 * restart, signal: controller.signal, kind: "protein", beam: cfg.beam ?? 1,
        ...(cfg.temperature === undefined ? {} : { temperature: cfg.temperature }),
        // `omit`: letters MPNN may never write, per task (a DNA binder without D/E is basic) - the loop's own default is "C"
        ...(cfg.omit?.[t.task] === undefined ? {} : { omit: cfg.omit[t.task] }),
        veto: ({ pdb }) => contactProblem(pdb, { chain: "A" }),
        fold: async (sequence, context) => {
          const t0 = Date.now();
          // `contactTries`: a fold that leaves the binder off the target is folded again with the next seed, up to
          // that many times - so MPNN designs from a docked backbone rather than from a binder floating free
          let r, docked = false;
          for (let k = 0; k < (cfg.contactTries ?? 1); k += 1) {
            const s = seed + 101 * restart + 13 * context.cycle + 7919 * k;
            // `designEngine`: "ef2" designs with ESMFold2 (the check and the evaluation stay AF3)
            // `ef2Cycles`: ESMFold2 for cycles before that one, AF3 after - ESMFold2 docks a DNA binder on every cycle
            // where AF3 docks ~5%, and AF3 then refines a docked start into one AF3 believes
            const ef2 = cfg.designEngine === "ef2" || (cfg.ef2Cycles !== undefined && context.cycle < cfg.ef2Cycles);
            r = ef2 ? await foldEf2(t, sequence, { seed: s })
              : await fold(t, sequence, { steps: cfg.steps, recycles: cfg.recycles, seed: s,
                motif: guidedAt(context.cycle) ? motif : undefined, msa: mpnnMsaAt(context),
                // `rotate`: the design folds' model by cycle (Protein Hunter's rotation: a design has to satisfy several
                // models); the check, the probe and the evaluation stay AF3
                model: cfg.rotate?.[context.cycle % cfg.rotate.length] });
            if (contactProblem(r.pdb, { chain: "A" }) === null) { docked = true; break; }
          }
          // `states`: the same sequence folded by each of these models too (guided alike), and MPNN designs the next
          // cycle from all the structures at once, every position tied across them (mpnn-bridge.js `states`) - the
          // models AVERAGED at the designer, where `rotate` alternates them; this cycle's own terms stay AF3's
          if (cfg.states !== undefined) {
            const others = [];
            for (const other of cfg.states) {
              const q = await fold(t, sequence.split(":")[0], { steps: cfg.steps, recycles: cfg.recycles,
                seed: seed + 101 * restart + 13 * context.cycle + 31337, motif: guidedAt(context.cycle) ? motif : undefined, model: other });
              others.push(q.pdb);
            }
            statesOf.set(r.pdb, others);
          }
          foldSeconds += (Date.now() - t0) / 1000;
          // `stick`: a cycle that will not dock hands the NEXT design the restart's last docked structure instead of
          // its own - MPNN redesigning an undocked binder writes a soluble protein that never comes back. The cycle
          // itself can never be chosen (its terms are infinite; its structure is not its sequence's)
          if (docked) lastDocked = r;
          else if (cfg.stick && lastDocked !== null) {
            return { pdb: lastDocked.pdb, confidence: lastDocked.confidence, terms: { binderPtm: 0, ipae: Infinity, iptm: 0, stuck: true } };
          }
          // AME: the design fold's own motif agreement and clashes (meaningful once `motifUntil` releases the guide for
          // the last steps - a fold guided to the end holds the motif exactly), which choose what is checked
          let own = t.task === "ame" ? motifTerms(t, positions, r.pdb, r.pdb) : {};
          // `motifMatch` (free cycles): the motif re-found among the fold's own residues, and the held positions moved
          // there - AF3 builds the pocket from whichever residues it likes, and those are the ones to keep
          if (t.task === "ame" && cfg.motifMatch && !guidedAt(context.cycle)) {
            const m = motifMatch(t, r.pdb);
            if (m !== null && m.rmsd < own.motifRmsd) {
              positions.splice(0, positions.length, ...m.positions);
              own = motifTerms(t, positions, r.pdb, r.pdb);
            }
          }
          own = t.task === "ame" ? { ...own, positions: [...positions] } : own;
          // `probe` (AME): every guided design fold's sequence folded again FREE at the design settings, the motif
          // matched unindexed in it - the probe's interface, motif, positions and structure are what is judged and
          // checked; MPNN still designs from the guided backbone
          if (t.task === "ame" && cfg.probe && guidedAt(context.cycle)) {
            const q = await fold(t, sequence, { steps: cfg.steps, recycles: cfg.recycles, seed: seed + 101 * restart + 13 * context.cycle + 4999 });
            const m = motifMatch(t, q.pdb);
            const at = m !== null ? m.positions : [...positions];
            own = { ...motifTerms(t, at, q.pdb, q.pdb), positions: at, probePdb: q.pdb, probed: true };
            return { pdb: r.pdb, confidence: r.confidence, terms: { binderPtm: q.binderPtm, ipae: q.ipae, iptm: q.iptm, ...own } };
          }
          return { pdb: r.pdb, confidence: r.confidence, terms: { binderPtm: r.binderPtm, ipae: r.ipae, iptm: r.iptm, ...own } };
        },
        design: (pdb, context) => designChain(mpnn, { pdb, chain: context.chain, temperature: context.temperature,
          ...(statesOf.has(pdb) ? { states: statesOf.get(pdb).map((other) => ({ pdb: other })) } : {}),
          omit: context.omit, alanineBias: context.alanineBias, random: context.random, hold: context.hold, kind: context.kind,
          // `residueBias`: per task, MPNN logits added per letter (a DNA binder pushed toward K/R)
          ...(cfg.residueBias?.[t.task] === undefined ? {} : { residueBias: cfg.residueBias[t.task] }) }),
      });
      for await (const record of stream) {
        const terms = record.folded.terms;
        records.push({ restart, cycle: record.cycle, score: record.score, vetoed: record.vetoed ?? null,
          alanine: record.alanine, ...terms, probePdb: undefined });
        // `giveUpStuck`: a restart whose last N design cycles all failed to dock is abandoned for a fresh one -
        // on DNA most restarts undock after cycle 0 and never come back, and each stuck cycle still costs its
        // `contactTries` folds
        if (record.cycle > 0) undockedRun = (record.vetoed ?? null) !== null || terms.stuck ? undockedRun + 1 : 0;
        if (cfg.giveUpStuck !== undefined && undockedRun >= cfg.giveUpStuck) controller.abort();
        // a restart that has not found the interface by `giveUp.cycle` is abandoned for a fresh one
        if (cfg.giveUp !== undefined && record.cycle >= cfg.giveUp.cycle && terms.ipae > cfg.giveUp.ipae) controller.abort();
        if (record.cycle > 0 && record.alanine <= 0.2 && (record.vetoed ?? null) === null) {
          const entry = { ...terms, restart, cycle: record.cycle, sequence: record.sequence, pdb: terms.probePdb ?? record.pdb, score: record.score,
            positions: terms.positions ?? positions,
            guided: !terms.probed && motif !== undefined && guidedAt(record.cycle) && cfg.guideCycles !== undefined };
          eligible.push(entry); mine.push(entry);
          if (looksDone(entry)) controller.abort();
        }
      }
    } catch (error) {
      if (!controller.signal.aborted) throw error;
    }
    if (cfg.check !== undefined) {
      const promising = mine.filter(promisingOf).sort((a, b) => a.ipae - b.ipae).slice(0, cfg.check.top);
      for (const entry of promising) {
        const result = await checkAt(entry, seed * 7 + entry.cycle + 1000 * restart);
        checked.push({ entry, result });
        if (result.pass) break;
      }
      if (checked.some((c) => c.result.pass)) break;
    } else if (eligible.some(looksDone)) {
      break;
    }
  }
  // `climb` (AME): when no restart produced a passing check, hill-climb on the graded quantity itself - from the
  // `from` free (probe) structures nearest the motif, `gens` generations of `draws` MPNN sequences (the matched
  // residues held), each folded FREE and the motif matched again; a generation's best (motif RMSD, then interface
  // PAE) replaces the current one when it is closer. Every structure reached is a candidate for the check.
  let climbed = 0;
  if (t.task === "ame" && cfg.climb !== undefined && !checked.some((c) => c.result.pass)) {
    const score = (e) => e.motifRmsd + (e.clashes > 0 ? 5 : 0) + Math.max(0, e.ipae - 1.5);
    const starts = eligible.filter((e) => !e.guided && e.motifRmsd !== undefined).sort((a, b) => score(a) - score(b)).slice(0, cfg.climb.from);
    for (const [n, start] of starts.entries()) {
      let current = start;
      for (let g = 0; g < cfg.climb.gens; g += 1) {
        let bestDraw = null;
        for (let k = 0; k < cfg.climb.draws; k += 1) {
          const random = mulberry(seedOf(`climb:${name}:${index}:${n}:${g}:${k}`));
          const designed = designChain(mpnn, { pdb: current.pdb, chain: "A", temperature: cfg.climb.temperature ?? 0.1,
            omit: cfg.omit?.[t.task] ?? "C", random, kind: "protein", hold: current.positions });
          const r = await fold(t, designed.sequence, { steps: cfg.steps, recycles: cfg.recycles, seed: seed + 70000 + 1000 * n + 37 * g + k });
          climbed += 1;
          const m = motifMatch(t, r.pdb);
          const at = m !== null && m.rmsd < motifTerms(t, current.positions, r.pdb, r.pdb).motifRmsd ? m.positions : current.positions;
          const e = { ipae: r.ipae, iptm: r.iptm, binderPtm: r.binderPtm, restart: -2, cycle: 10 * n + g, sequence: designed.sequence,
            pdb: r.pdb, positions: [...at], ...motifTerms(t, at, r.pdb, r.pdb) };
          records.push({ restart: -2, cycle: e.cycle, climb: k, ipae: e.ipae, motifRmsd: e.motifRmsd, clashes: e.clashes });
          const alanine = [...designed.sequence].filter((x) => x === "A").length / designed.sequence.length;
          if (alanine > 0.2 || contactProblem(r.pdb, { chain: "A" }) !== null) continue;
          eligible.push(e);
          if (bestDraw === null || score(e) < score(bestDraw)) bestDraw = e;
        }
        if (bestDraw !== null && score(bestDraw) < score(current)) current = bestDraw;
      }
    }
    for (const entry of eligible.filter((e) => e.restart === -2 && promisingOf(e)).sort((a, b) => score(a) - score(b)).slice(0, cfg.climb.top ?? 3)) {
      const result = await checkAt(entry, seed * 7 + 555 + checked.length);
      checked.push({ entry, result });
      if (result.pass) break;
    }
  }
  // `polish`: when no restart produced a passing check, redesign around the NEAREST misses instead of from scratch -
  // `draws` MPNN sequences on each of the `from` closest checked candidates' design backbones, each folded at the
  // design settings, and the best `top` of them (by design iPAE, under the check's thresholds) checked in turn
  const margin = (r) => Math.max(r.ipae / 1.5, t.task === "dna" ? r.rmsd / 8
    : t.task === "ame" ? Math.max(r.rmsd / 2, r.motifRmsd / 1.5, r.clashes > 0 ? 2 : 0)
    : Math.max(r.rmsd / 1.5, r.ligandRmsd / 5, 0.8 / Math.max(r.iptm, 1e-3)));
  let polished = 0;
  if (cfg.polish !== undefined && !checked.some((c) => c.result.pass)) {
    const near = [...checked].sort((a, b) => margin(a.result) - margin(b.result)).slice(0, cfg.polish.from);
    const pool = [];
    for (const [n, c] of near.entries()) {
      for (let k = 0; k < cfg.polish.draws; k += 1) {
        const random = mulberry(seedOf(`polish:${name}:${index}:${n}:${k}`));
        const designed = designChain(mpnn, { pdb: c.entry.pdb, chain: "A", temperature: cfg.polish.temperature ?? 0.1,
          omit: cfg.omit?.[t.task] ?? "C", random, kind: "protein", hold: c.entry.positions });
        const r = await fold(t, designed.sequence, { steps: cfg.steps, recycles: cfg.recycles, seed: seed + 50000 + 97 * n + k });
        polished += 1;
        records.push({ restart: -1, cycle: n, polish: k, ipae: r.ipae, iptm: r.iptm, binderPtm: r.binderPtm });
        const alanine = [...designed.sequence].filter((x) => x === "A").length / designed.sequence.length;
        if (alanine <= 0.2 && (contactProblem(r.pdb, { chain: "A" }) ?? footprintProblem(r.pdb)) === null) pool.push({ ipae: r.ipae, iptm: r.iptm, binderPtm: r.binderPtm, restart: -1, cycle: n, sequence: designed.sequence, pdb: r.pdb, positions: c.entry.positions,
          ...(t.task === "ame" ? motifTerms(t, c.entry.positions, r.pdb, r.pdb) : {}) });
      }
    }
    for (const entry of pool.filter(promisingOf).sort((a, b) => a.ipae - b.ipae).slice(0, cfg.polish.top)) {
      const result = await checkAt(entry, seed * 7 + 777 + checked.length);
      checked.push({ entry, result });
      eligible.push(entry);
      if (result.pass) break;
    }
  }
  const chosen = checked.find((c) => c.result.pass) ?? null;
  // (no passing check: the lowest interface PAE - for AME the nearest to passing, a guided design never)
  const fallbackKey = (e) => t.task !== "ame" ? e.ipae
    : e.guided ? Infinity : (e.motifRmsd ?? 99) / 1.5 + (e.clashes > 0 ? 5 : 0) + Math.max(0, e.ipae - 1.5);
  const best = chosen !== null ? chosen.entry
    : eligible.length > 0 ? eligible.reduce((a, b) => (fallbackKey(b) < fallbackKey(a) ? b : a)) : null;
  const row = { target: name, task: t.task, attempt: index, length, config: cfg.tag, runs, polished, climbed, unguided, records, foldSeconds,
    checks: checked.map((c) => ({ restart: c.entry.restart, cycle: c.entry.cycle, ipae: c.result.ipae, iptm: c.result.iptm,
      rmsd: c.result.rmsd, ligandRmsd: c.result.ligandRmsd, motifRmsd: c.result.motifRmsd, clashes: c.result.clashes,
      pass: c.result.pass, seedsPassed: c.result.seedsPassed })) };
  if (best !== null) {
    // 🔴 THE EVALUATION IS ALWAYS THE NATIVE AF3 at the benchmark's settings, with its own seed
    const final = await assess(best.sequence, best.pdb, evalSeed, best.positions);
    Object.assign(row, { binder: best.sequence, bestCycle: best.cycle, bestRestart: best.restart,
      binderPtm: final.binderPtm, ipae: final.ipae, iptm: final.iptm, rmsd: final.rmsd, ligandRmsd: final.ligandRmsd,
      motifRmsd: final.motifRmsd, clashes: final.clashes, positions: best.positions,
      pass: final.pass, designPdb: best.pdb, refoldPdb: final.pdb });
  } else {
    row.pass = false;
  }
  row.wall = (Date.now() - started) / 1000;
  return row;
}

// --rescore <rows.jsonl>: every submitted design evaluated again at its own evaluation seed (the attempt's seeded
// draw) - for rows whose evaluation ran on a server that ignored the job's seed (fixed 2026-10-30; see af3.cu)
async function rescore(rowsPath) {
  for (const line of fs.readFileSync(rowsPath, "utf8").split("\n").filter(Boolean)) {
    const d = JSON.parse(line);
    if (done.has(`${d.target}:${d.attempt}`) || d.error) continue;
    const t = taskOf(d.target);
    const rng = mulberry(seedOf(`${config.pair ?? config.tag}:${d.target}:${d.attempt}`));
    rng();
    const evalSeed = 1 + Math.floor(rng() * 2 ** 30);
    const row = { target: d.target, task: d.task, attempt: d.attempt, config: d.config, rescored: true,
      served: { ipae: d.ipae, iptm: d.iptm, rmsd: d.rmsd, ligandRmsd: d.ligandRmsd, pass: d.pass } };
    if (d.binder) {
      // `evalModel`: CROSS-EVALUATION - the same designs judged by a model that took no part in them (Protenix-2,
      // Boltz-2, ...), same settings, criterion and seed; AF3 when unset
      const r = await fold(t, d.binder, { steps: 200, recycles: 5, seed: evalSeed, model: config.evalModel });
      Object.assign(row, { binder: d.binder, binderPtm: r.binderPtm, ipae: r.ipae, iptm: r.iptm, evalModel: config.evalModel ?? "af3",
        ...verdict(t.task, d.designPdb, r, t, d.positions) });
    } else {
      row.pass = false;
    }
    fs.appendFileSync(out, JSON.stringify(row) + "\n");
    console.log(d.target, d.attempt, d.pass, "->", row.pass, row.ipae?.toFixed?.(2), row.rmsd?.toFixed?.(2));
  }
}

try {
  if (jobs[0] === "--rescore") { await rescore(jobs[1]); jobs.length = 0; }
  for (const job of jobs) {
    if (done.has(job)) continue;
    const [name, index] = job.split(":");
    let row;
    try {
      row = await attempt(name, Number(index));
    } catch (error) {
      row = { target: name, attempt: Number(index), config: config.tag, error: String(error?.stack ?? error).slice(0, 2000) };
    }
    fs.appendFileSync(out, JSON.stringify(row) + "\n");
    console.log(name, index, row.pass, row.iptm?.toFixed?.(2), row.ipae?.toFixed?.(2), row.rmsd?.toFixed?.(2), row.ligandRmsd?.toFixed?.(2), row.wall?.toFixed?.(0), row.error?.slice(0, 160) ?? "");
  }
} finally {
  fs.rmSync(work, { recursive: true, force: true });
}
