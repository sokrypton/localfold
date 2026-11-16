/**
 * ProteinEBM (jproney/ProteinEBM, MIT; Roney et al., bioRxiv 2025.12.09.693073): what every backend shares - the
 * checkpoint's dimensions, its diffuser constants, the inputs a chain becomes, and the host-side pieces of the network
 * (the relative-position class of a residue pair, the Fourier time row).
 *
 * The model reads a CA trace and a sequence. Its energy is E(r, t) = sum_i |W_r a_i|^2 and its score the gradient
 * -dE/dr, so a backend runs a forward AND a backward pass; see webgpu/proteinebm/network.js.
 */
export { bundleReader } from "../bioemu2/weights.js";

/** base_pretrain.yaml's model and diffuser (all four released checkpoints share them). */
export const PROTEINEBM = Object.freeze({
  tokenS: 256, tokenZ: 128, single: 512, layers: 16, heads: 8, dimFourier: 256, relPosMax: 32, chainMax: 1,
  minB: 0.1, maxB: 20.0, coordinateScaling: 0.1,
});

/** AlphaFold's residue order, which protein_utils.restype_order is; anything else is 20 (unknown). */
export const RESTYPES = "ARNDCQEGHILKMFPSTWYV";
const THREE = { ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E", GLY: "G", HIS: "H", ILE: "I",
                LEU: "L", LYS: "K", MET: "M", PHE: "F", PRO: "P", SER: "S", THR: "T", TRP: "W", TYR: "Y", VAL: "V" };

export function aatypeOf(sequence) {
  return Int32Array.from(sequence.toUpperCase(), (c) => { const k = RESTYPES.indexOf(c); return k < 0 ? 20 : k; });
}

/**
 * The relative-position encoder's class for residues (i, j): d_residue = clip(ri - rj + 32, 0, 64), or 65 across
 * chains; d_chain = clip(|ci - cj|, 0, 1). Encoded as d_residue * 2 + d_chain (132 classes).
 */
export function relposClass(ri, rj, ci = 0, cj = 0) {
  const r = PROTEINEBM.relPosMax;
  const dres = ci === cj ? Math.min(Math.max(ri - rj + r, 0), 2 * r) : 2 * r + 1;
  const dchain = Math.min(Math.abs(ci - cj), PROTEINEBM.chainMax);
  return dres * 2 + dchain;
}

/**
 * fourier_to_single(LN(cos(2 pi (w t + b)))) - the time level's 512-vector, which the single conditioner adds to every
 * residue. `host` = {fourier: {w, b, nw, nb, out}} from the bundle.
 */
export function fourierRow({ fourier }, t) {
  const n = fourier.b.length;
  const x = new Float64Array(n);
  for (let k = 0; k < n; k += 1) x[k] = Math.cos(2 * Math.PI * (fourier.w[k] * t + fourier.b[k]));
  let mean = 0;
  for (const v of x) mean += v;
  mean /= n;
  let q = 0;
  for (const v of x) q += (v - mean) * (v - mean);
  const rstd = 1 / Math.sqrt(q / n + 1e-5);
  for (let k = 0; k < n; k += 1) x[k] = (x[k] - mean) * rstd * fourier.nw[k] + fourier.nb[k];
  const width = fourier.out.length / n;
  const row = new Float64Array(width);
  for (let o = 0; o < width; o += 1) {
    let s = 0;
    for (let k = 0; k < n; k += 1) s += fourier.out[o * n + k] * x[k];
    row[o] = s;
  }
  return row;
}

/** The R3 diffuser (r3_diffuser.py): b(t), its square root g(t), and the marginal integral. */
export const diffuser = {
  b: (t) => PROTEINEBM.minB + t * (PROTEINEBM.maxB - PROTEINEBM.minB),
  g: (t) => Math.sqrt(diffuser.b(t)),
  marginal: (t) => t * PROTEINEBM.minB + 0.5 * t * t * (PROTEINEBM.maxB - PROTEINEBM.minB),
};

/**
 * One chain's CA trace out of a PDB, as residues_to_features reads it: the first model, every amino-acid residue
 * with a CA (the first altloc), numbered by the file's own residue numbers. `chain` picks one (default the first).
 */
export function caTraceFromPdb(text, chain = null) {
  const residues = [];
  const seen = new Set();
  let pick = chain;
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith("ENDMDL")) break;
    const record = line.slice(0, 6);
    if (record !== "ATOM  " && record !== "HETATM") continue;
    const resName = line.slice(17, 20).trim();
    if (record === "HETATM" && resName !== "MSE") continue;
    if (line.slice(12, 16).trim() !== "CA") continue;
    const id = line[21];
    pick ??= id;
    if (id !== pick) continue;
    const key = `${line.slice(22, 27)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    residues.push({ name: resName, number: parseInt(line.slice(22, 26), 10),
                    xyz: [parseFloat(line.slice(30, 38)), parseFloat(line.slice(38, 46)), parseFloat(line.slice(46, 54))] });
  }
  const n = residues.length;
  const ca = new Float32Array(n * 3);
  residues.forEach((r, i) => ca.set(r.xyz, i * 3));
  // (a non-standard residue is unknown, as residues_to_features's restype_3to1 makes it - an MSE too)
  const sequence = residues.map((r) => THREE[r.name] ?? "X").join("");
  return { chain: pick, sequence, aatype: aatypeOf(sequence), residueIdx: Int32Array.from(residues, (r) => r.number), ca };
}

const ONE_TO_THREE = Object.fromEntries(Object.entries(THREE).map(([three, one]) => [one, three]));

/** A CA-only PDB of a trace (`residueIdx` numbers it; 1..N when omitted). */
export function caPdb(pos, sequence, residueIdx = null, chain = "A") {
  const f = (v) => v.toFixed(3).padStart(8);
  const lines = [];
  for (let i = 0; i < pos.length / 3; i += 1) {
    const number = residueIdx ? residueIdx[i] : i + 1;
    lines.push(`ATOM  ${String(i + 1).padStart(5)}  CA  ${ONE_TO_THREE[sequence[i]] ?? "UNK"} ${chain}`
      + `${String(number).padStart(4)}    ${f(pos[i * 3])}${f(pos[i * 3 + 1])}${f(pos[i * 3 + 2])}  1.00  0.00           C`);
  }
  lines.push("TER", "END");
  return `${lines.join("\n")}\n`;
}

/** Centre a trace (in place) on its mean; returns it. */
export function centre(r) {
  const n = r.length / 3;
  for (let c = 0; c < 3; c += 1) {
    let m = 0;
    for (let i = 0; i < n; i += 1) m += r[i * 3 + c];
    m /= n;
    for (let i = 0; i < n; i += 1) r[i * 3 + c] -= m;
  }
  return r;
}
