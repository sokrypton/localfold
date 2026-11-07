/**
 * The page's fold (web/af3-model.js `foldAf3`) on its WebGL2 backend against its WebGPU one: the
 * same sequence, seed and schedule through `backend: "webgl2"` and through `foldBatch`, compared
 * on what a refold reports - Cα RMSD between the two structures, mean pLDDT, pTM, ipTM and per-chain
 * pTM - with the local bundle (no download).
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-webgl-page.js [--family=openbind0] [--steps=10]
 *       [--sequence=AAAA:BBBB]
 */
import { MODEL_BUNDLES } from "../../shared/bundles/manifests/index.js";
import { foldAf3, loadAf3Weights } from "../../web/af3-model.js";
import { foldBatch } from "../../webgpu/af3/fold.js";

const option = (args, name, fallback) =>
  (args.find((a) => a.startsWith(`--${name}=`)) ?? `--${name}=${fallback}`).slice(name.length + 3);

function caOf(pdb) {
  return pdb.split("\n").filter((line) => line.startsWith("ATOM") && line.slice(12, 16).trim() === "CA")
    .map((line) => [Number(line.slice(30, 38)), Number(line.slice(38, 46)), Number(line.slice(46, 54))]);
}

/** Cα RMSD after the optimal superposition (Kabsch, by power iteration on the quaternion matrix). */
function rmsd(a, b) {
  const n = a.length;
  const centre = (p) => p.reduce((s, x) => [s[0] + x[0] / n, s[1] + x[1] / n, s[2] + x[2] / n], [0, 0, 0]);
  const ca = centre(a);
  const cb = centre(b);
  const A = a.map((p) => [p[0] - ca[0], p[1] - ca[1], p[2] - ca[2]]);
  const B = b.map((p) => [p[0] - cb[0], p[1] - cb[1], p[2] - cb[2]]);
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  let e0 = 0;
  for (let i = 0; i < n; i += 1) {
    for (let r = 0; r < 3; r += 1) for (let c = 0; c < 3; c += 1) S[r][c] += A[i][r] * B[i][c];
    e0 += A[i][0] ** 2 + A[i][1] ** 2 + A[i][2] ** 2 + B[i][0] ** 2 + B[i][1] ** 2 + B[i][2] ** 2;
  }
  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = S;
  const K = [
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
    [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
  ];
  // The largest eigenvalue of K, shifted so it is also the largest in magnitude.
  const shift = e0;
  let v = [1, 0, 0, 0];
  let lambda = 0;
  for (let it = 0; it < 500; it += 1) {
    const w = K.map((row, r) => row.reduce((s, x, c) => s + x * v[c], 0) + shift * v[r]);
    const norm = Math.hypot(...w);
    v = w.map((x) => x / norm);
    lambda = norm - shift;
  }
  return Math.sqrt(Math.max(0, (e0 - 2 * lambda) / n));
}

export async function main(device, args) {
  const family = option(args, "family", "openbind0");
  const steps = Number(option(args, "steps", "10"));
  const sequence = option(args, "sequence", "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRV:MKVLLTGAGSRAAELVRKAGH");
  MODEL_BUNDLES[family].remote = undefined;
  MODEL_BUNDLES[family].directory = `/${MODEL_BUNDLES[family].directory.replace(/^\.\//, "")}`;
  const weights = await loadAf3Weights(() => {}, family);
  const run = async (backend) => {
    const t0 = performance.now();
    let frames = 0;
    const out = await foldAf3({
      sequence, calls: steps, recycles: 0, seed: 7, weights, device, backend,
      signal: new AbortController().signal,
      onStatus: () => {}, onProgress: () => {}, onFrame: () => { frames += 1; },
    });
    return { out, seconds: Number(((performance.now() - t0) / 1000).toFixed(1)), frames };
  };
  const gl = await run("webgl2");
  const gpu = await run(undefined);
  const c = (r) => r.out.confidence;
  const round = (x) => (typeof x === "number" ? Number(x.toFixed(4)) : x);
  const per = (o) => Object.fromEntries(Object.entries(o ?? {}).map(([k, v]) => [k, round(v)]));
  const caRmsd = rmsd(caOf(gl.out.pdb), caOf(gpu.out.pdb));
  const meanPaeGap = (() => {
    const a = c(gl).predictedAlignedError;
    const b = c(gpu).predictedAlignedError;
    let s = 0;
    for (let i = 0; i < a.length; i += 1) s += Math.abs(a[i] - b[i]);
    return s / a.length;
  })();
  const report = {
    family, steps, tokens: gl.out.batch.tokens,
    seconds: { webgl2: gl.seconds, webgpu: gpu.seconds }, frames: { webgl2: gl.frames, webgpu: gpu.frames },
    caRmsdA: round(caRmsd),
    meanPlddt: [round(c(gl).meanPlddt), round(c(gpu).meanPlddt)],
    ptm: [round(c(gl).ptm), round(c(gpu).ptm)],
    iptm: [round(c(gl).iptm), round(c(gpu).iptm)],
    chainPtm: [per(c(gl).chainPtm), per(c(gpu).chainPtm)],
    // 🔴 chain_iptm AND chain_pair_iptm TOO, BECAUSE THEY ARE THE FIELDS A CHANGE TO THE SCORES
    // LANDS IN FIRST. This compared chain_ptm alone, and chain_ptm is a chain pair's score on the
    // DIAGONAL - so the interface term's d0 (the pair's own chains, not every token) and the
    // mean-of-pairs chain_iptm were both outside what any cross-backend gate read. With two chains
    // chain_iptm equals the one pair score; pass `--sequence=A:B:C` for the arm that separates them.
    chainIptm: [per(c(gl).chainIptm), per(c(gpu).chainIptm)],
    chainPairIptm: [per(c(gl).chainPairIptm), per(c(gpu).chainPairIptm)],
    meanAbsPaeGapA: round(meanPaeGap),
    contactProbs: [gl.out.contactProbs.length, gpu.out.contactProbs.length],
  };
  // `--f32=1`: WebGPU's fold of the same batch with every precision axis f32, to tell this
  // backend's error from the shipped WebGPU fold's f16.
  if (option(args, "f32", "0") === "1") {
    const f32 = await foldBatch(device, gl.out.batch, weights, { steps, recycles: 0, seed: 7,
      stagedPrecision: "f32", weightPrecision: "f32", pairWeightPrecision: "f32", accumulatePrecision: "f32" });
    report.webgpuF32 = { meanPlddt: round(f32.meanPlddt), ptm: round(f32.ptm), iptm: round(f32.iptm) };
  }
  // pTM and ipTM to 0.02: on AF3 at 61 tokens the precision alone moves ipTM that much (WebGPU f32
  // 0.304, shipped f16 0.311, WebGL2 0.321 with the structures 0.02 A apart). Every chain field is
  // held to the same 0.02 and to the same KEYS - a backend reporting a chain the other does not is
  // a difference the numbers cannot show.
  const chainsAgree = (a, b) => {
    const keys = [...new Set([...Object.keys(a ?? {}), ...Object.keys(b ?? {})])];
    return keys.every((k) => Math.abs((a?.[k] ?? NaN) - (b?.[k] ?? NaN)) < 0.02);
  };
  report.ok = caRmsd < 0.25 && Math.abs(c(gl).meanPlddt - c(gpu).meanPlddt) < 1
    && Math.abs(c(gl).ptm - c(gpu).ptm) < 0.02 && Math.abs(c(gl).iptm - c(gpu).iptm) < 0.02 && meanPaeGap < 0.25
    && chainsAgree(c(gl).chainPtm, c(gpu).chainPtm) && chainsAgree(c(gl).chainIptm, c(gpu).chainIptm)
    && chainsAgree(c(gl).chainPairIptm, c(gpu).chainPairIptm);
  return report;
}
