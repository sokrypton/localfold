/**
 * protocol.js
 * -----------
 * The design protocols that beat RFD4-Proteina on its own benchmarks, as data, and the per-fold terms and pass
 * criteria they are judged by - the page's half of tools/ppi-bench (docs/DESIGN-TODO.md, "Toward every reported
 * number").
 *
 * 🔴 ONE LOOP, THREE RECIPES. Every winning run is the same search around a Protein Hunter hunt - restarts that stop
 * early on a confident cycle, the most promising cycles refolded at the evaluation's settings as a check, and (ligand
 * and DNA) a polish of the nearest misses - and differs only in switches the job's own contents decide: a ligand in
 * the job means the small-molecule recipe, a DNA chain the DNA one, protein alone the PPI one. `protocolFor` reads
 * them off the job, so nothing here asks a reader which benchmark they are running.
 *
 * 🔴 THE NUMBERS ARE THE MEASURED RUNS', NAMED BY TAG (NC, TS7, TD15), and a value no run chose is not in a recipe.
 * Where a switch was never isolated (polish, `stick`, the footprint) the recipe carries it because the winning run
 * did, not because it was shown to matter - DESIGN-TODO says which.
 *
 * 🔴 THE CRITERIA ARE THE PAPER'S (RFD4 `configs/metrics/pass_criteria/all_pass_criteria.yaml`), applied to the
 * page's own refold: a check that passes here is a check at the evaluator's thresholds, read off the page's AF3.
 */
import { weightedRigidAlign } from "../af3/diffusion/rigid-align.js";

/** Which confidence number each criterion reads beside the interface PAE (DNA's reads none). */
const CONFIDENCE = { ppi: "binderPtm", sm: "iptm", dna: null };

/**
 * The winning recipes. `stop` ends a restart on a design cycle this confident; `check` refolds a restart's `top`
 * most promising cycles at 200 steps and 5 recycles (at `seeds` seeds, all of which must pass); `polish` redraws
 * `draws` sequences on each of the `from` nearest checked misses and checks the `top` of them.
 */
export const PROTOCOLS = {
  // NC: 37.9% against RFD4-Proteina + Search's 28.1% (ppi_backbone)
  ppi: {
    tag: "NC", criterion: "ppi", cycles: 6, steps: 100, recycles: 3, restarts: 3,
    stop: { confidence: 0.85, ipae: 1.2 },
    check: { top: 2, confidence: 0.8, ipae: 2.5 },
  },
  // TS7: 62.3% against 46.6% (studio_figure_2c_unfixed); temperature 0.2 measured as moving nothing, kept as run
  sm: {
    tag: "TS7", criterion: "sm", cycles: 6, steps: 100, recycles: 3, restarts: 10, temperature: 0.2,
    designer: "ligand",
    stop: { confidence: 0.85, ipae: 1.2 },
    check: { top: 3, confidence: 0.8, ipae: 2.5, seeds: 2 },
    polish: { from: 2, draws: 8, top: 3, temperature: 0.2 },
  },
  // TD15: 8.3% against 4.8% (dna_unfixed). LigandMPNN designs the binder (TD19 is measuring NA-MPNN against it)
  dna: {
    tag: "TD15", criterion: "dna", cycles: 6, steps: 100, recycles: 3, restarts: 10, omit: "CDE",
    designer: "ligand", foldTries: 5, stick: true, minFootprint: 8,
    stop: { confidence: 0.85, ipae: 1.2 },
    check: { top: 3, confidence: 0.8, ipae: 2.5 },
    polish: { from: 2, draws: 8, top: 3 },
  },
};

/**
 * The recipe for a job, from what it carries besides the designed chain.
 *
 * @param {{ligands?: number, nucleic?: number, partners?: number, designs?: string}} contents ligands, DNA/RNA
 *   chains and other chains of any kind beside the designed one, and what the designed chain is made of
 * @returns {{protocol: object|null, why: string}} null with the reason where no recipe applies
 */
export function protocolFor(contents = {}) {
  if ((contents.designs ?? "protein") !== "protein") {
    return { protocol: null, why: "the designed chain is DNA or RNA, and every recipe designs a protein" };
  }
  if ((contents.ligands ?? 0) > 0 && (contents.nucleic ?? 0) > 0) {
    return { protocol: null, why: "a ligand and a DNA chain together: no benchmark measured that combination" };
  }
  if ((contents.ligands ?? 0) > 0) return { protocol: PROTOCOLS.sm, why: "a ligand is present" };
  if ((contents.nucleic ?? 0) > 0) return { protocol: PROTOCOLS.dna, why: "a DNA or RNA chain is present" };
  if ((contents.partners ?? 0) > 0) return { protocol: PROTOCOLS.ppi, why: "the target is protein" };
  return { protocol: null, why: "there is nothing to bind: a search judges an interface" };
}

/**
 * A fold's interface terms for the designed chain: the lowest interface PAE against any other chain (each way,
 * averaged), the designed chain's own pTM, and its ipTM averaged over the other chains.
 *
 * @param {{batch: {asymId: ArrayLike<number>}, confidence: {predictedAlignedError: ArrayLike<number>,
 *   chainPtm: Record<string, number>, chainPairIptm: Record<string, number>}}} fold foldAf3's result
 * @param {number} chainIndex which chain is designed (its asymId is the chainIndex-th distinct one)
 * @returns {{ipae: number, binderPtm: number, iptm: number}}
 */
export function interfaceTerms(fold, chainIndex = 0) {
  const asym = Array.from(fold.batch.asymId);
  const chains = [...new Set(asym)];
  const binder = chains[chainIndex];
  if (binder === undefined) throw new RangeError(`no chain ${chainIndex} among ${chains.length}`);
  const others = chains.filter((c) => c !== binder);
  if (others.length === 0) throw new Error("interface terms need a second chain");
  const n = asym.length;
  const pae = fold.confidence.predictedAlignedError;
  const minPae = (a, b) => {
    let m = Infinity;
    for (let i = 0; i < n; i += 1) if (asym[i] === a) for (let j = 0; j < n; j += 1) if (asym[j] === b) m = Math.min(m, pae[i * n + j]);
    return m;
  };
  const ipae = Math.min(...others.map((c) => (minPae(binder, c) + minPae(c, binder)) / 2));
  const pair = (c) => fold.confidence.chainPairIptm[`${Math.min(binder, c)}|${Math.max(binder, c)}`];
  const iptm = others.reduce((sum, c) => sum + pair(c), 0) / others.length;
  return { ipae, binderPtm: fold.confidence.chainPtm[String(binder)], iptm };
}

/** Is a cycle with these terms confident enough for `bound` ({confidence, ipae}) under `criterion`? */
export function confident(terms, bound, criterion) {
  const key = CONFIDENCE[criterion];
  if (key === undefined) throw new Error(`no criterion ${criterion}`);
  return terms.ipae <= bound.ipae && (key === null || terms[key] >= bound.confidence);
}

/** Heavy atoms of a PDB by chain:residue:name, split into the designed chain's backbone and everything else. */
function atomsOf(pdb, chain, restBackboneOnly) {
  const binder = new Map();
  const rest = new Map();
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    if (line.slice(76, 78).trim() === "H") continue;
    const name = line.slice(12, 16).trim();
    const backbone = line.startsWith("ATOM") && ["N", "CA", "C", "O"].includes(name);
    const key = `${line[21]}:${line.slice(22, 26).trim()}:${name}`;
    const xyz = [+line.slice(30, 38), +line.slice(38, 46), +line.slice(46, 54)];
    if (line[21] === chain) { if (backbone) binder.set(key, xyz); } else if (backbone || !restBackboneOnly) rest.set(key, xyz);
  }
  return { binder, rest };
}

/** The RMSD of `measure` (design against refold) once the refold's `fit` atoms are superposed on the design's. */
function fittedRmsd(fitDesign, fitRefold, measureDesign, measureRefold) {
  const fit = [...fitDesign.keys()].filter((k) => fitRefold.has(k));
  const measure = [...measureDesign.keys()].filter((k) => measureRefold.has(k));
  if (fit.length < 3 || measure.length === 0) throw new Error(`an RMSD over ${fit.length} fitted and ${measure.length} measured atoms`);
  const keys = [...fit, ...measure];
  const n = keys.length;
  const x = new Float32Array(n * 3);
  const y = new Float32Array(n * 3);
  const w = new Float32Array(n);
  keys.forEach((k, i) => {
    const fitted = i < fit.length;
    x.set((fitted ? fitRefold : measureRefold).get(k), i * 3);
    y.set((fitted ? fitDesign : measureDesign).get(k), i * 3);
    w[i] = fitted ? 1 : 0;
  });
  const moved = weightedRigidAlign(x, y, w, n);
  let sum = 0;
  for (let i = fit.length; i < n; i += 1) for (let a = 0; a < 3; a += 1) sum += (moved[i * 3 + a] - y[i * 3 + a]) ** 2;
  return Math.sqrt(sum / measure.length);
}

/**
 * Does a refold of a design pass the paper's criterion?
 *
 * ppi: binder pTM >= 0.8, interface PAE <= 1.5, binder backbone RMSD <= 2.5 after fitting the target's backbone.
 * sm: ipTM >= 0.8, interface PAE <= 1.5, binder backbone RMSD <= 1.5 (self-fitted), ligand RMSD <= 5 after the
 * binder's fit. dna: interface PAE <= 1.5, binder backbone RMSD <= 8 after fitting the DNA.
 *
 * @param {"ppi"|"sm"|"dna"} criterion
 * @param {string} designPdb
 * @param {string} refoldPdb
 * @param {{ipae: number, binderPtm: number, iptm: number}} terms the refold's interfaceTerms
 * @param {string} [chain="A"] the designed chain's letter
 * @returns {{pass: boolean, margin: number, rmsd: number, ligandRmsd?: number}} `margin` is the largest of the
 *   ratios to each threshold - under 1 passes - which orders near misses
 */
export function verdict(criterion, designPdb, refoldPdb, terms, chain = "A") {
  const d = atomsOf(designPdb, chain, criterion === "ppi");
  const r = atomsOf(refoldPdb, chain, criterion === "ppi");
  if (criterion === "ppi") {
    const rmsd = fittedRmsd(d.rest, r.rest, d.binder, r.binder);
    const margin = Math.max(terms.ipae / 1.5, rmsd / 2.5, 0.8 / Math.max(terms.binderPtm, 1e-3));
    return { rmsd, margin, pass: terms.binderPtm >= 0.8 && terms.ipae <= 1.5 && rmsd <= 2.5 };
  }
  if (criterion === "dna") {
    const rmsd = fittedRmsd(d.rest, r.rest, d.binder, r.binder);
    return { rmsd, margin: Math.max(terms.ipae / 1.5, rmsd / 8), pass: terms.ipae <= 1.5 && rmsd <= 8 };
  }
  if (criterion === "sm") {
    const rmsd = fittedRmsd(d.binder, r.binder, d.binder, r.binder);
    const ligandRmsd = fittedRmsd(d.binder, r.binder, d.rest, r.rest);
    const margin = Math.max(terms.ipae / 1.5, rmsd / 1.5, ligandRmsd / 5, 0.8 / Math.max(terms.iptm, 1e-3));
    return { rmsd, ligandRmsd, margin, pass: terms.iptm >= 0.8 && terms.ipae <= 1.5 && rmsd <= 1.5 && ligandRmsd <= 5 };
  }
  throw new Error(`no criterion ${criterion}`);
}

/**
 * How many residues of the other chains the designed chain touches (any heavy atoms within 4.5 A) - on a 10-bp
 * duplex a binder gripping one end moves 15-40 A on refolding and one spanning both strands holds (TD9's
 * `minFootprint`, which screens what is CHECKED and never what the loop designs from).
 *
 * @param {string} pdb
 * @param {string} [chain="A"]
 * @returns {number}
 */
export function footprint(pdb, chain = "A") {
  const binder = [];
  const other = new Map();
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    if (line.slice(76, 78).trim() === "H") continue;
    const xyz = [+line.slice(30, 38), +line.slice(38, 46), +line.slice(46, 54)];
    if (line[21] === chain) binder.push(xyz);
    else {
      const key = line.slice(21, 26);
      if (!other.has(key)) other.set(key, []);
      other.get(key).push(xyz);
    }
  }
  let touched = 0;
  for (const atoms of other.values()) {
    if (atoms.some((a) => binder.some((b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2 < 4.5 ** 2))) touched += 1;
  }
  return touched;
}
