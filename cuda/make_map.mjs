// A native port's weights as a MAP onto a published bundle, so a machine loads the bundle as it is (its
// codes decoded on the device, cuda/af3/src/common.cuh's loadBundle) and no float32 file is written.
//
//   node --js-float16array cuda/make_map.mjs <bundle dir> <exported weights dir> <out .map>
//
// The exported weights are what the port's exporter wrote from that same bundle (cuda/af3/export-model.mjs
// --weights-only, ...): every tensor of it is found in the decoded bundle as a whole tensor or a contiguous
// slice of one (a block of a stacked tensor) - or is all zeros (a placeholder the page's loader makes) -
// and written as
//   b <native name> <bundle tensor> <first element> <length>
//   z <native name> <length>
// and the two derivations the page's loader makes, verified bit for bit against the export:
//   p <native name> <length> o 1 <length> 0 1                         all ones (a LayerNorm scale whose
//                                                                      per-block scales were folded away)
//   p <native name> <length> x <rank> <dims> <dst> <dst strides> <scale> <off> <strides> <proj> <off> <strides>
//      a per-block pair LayerNorm scale folded into its sibling projection (scale[c] * proj[c, h], one
//      float32 multiply), per block or packed [C, BLOCKS, HEADS] (shared/af3/weights/diffusion-weights.js)
// with every metadata line (`m ...`) copied as it is. A tensor found nowhere is an error, never a guess:
// the port's exporter then does arithmetic the map cannot carry.
import { readFileSync, writeFileSync, openSync, readSync } from "node:fs";
import { readTensor } from "../shared/weights/dtype.js";

const [bundleDir, weightsDir, out] = process.argv.slice(2);
if (!out) { console.error("usage: make_map.mjs <bundle dir> <exported weights dir> <out .map>"); process.exit(1); }
const manifest = JSON.parse(readFileSync(`${bundleDir}/manifest.json`, "utf8"));
const shards = new Map();
const decoded = [];
for (const [name, record] of Object.entries(manifest.tensors)) {
  if (!shards.has(record.file)) {
    const b = readFileSync(`${bundleDir}/${record.file}`);
    shards.set(record.file, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  }
  decoded.push([name, readTensor(record, shards.get(record.file), record.byteOffset ?? 0, true)]);
}
// the fold candidates: every (scale [...lead, C], projection [...lead, C, H]) sibling pair, as parts
const shapeOf = (name) => manifest.tensors[name].shape;
const valuesOf = new Map(decoded);
const folds = [];
for (const scaleName of Object.keys(manifest.tensors)) {
  if (!scaleName.endsWith("/pair_input_layer_norm/scale")) continue;
  const projName = scaleName.replace(/pair_input_layer_norm\/scale$/, "pair_logits_projection/weights");
  if (!manifest.tensors[projName]) continue;
  const ss = shapeOf(scaleName), ps = shapeOf(projName);
  const C = ss.at(-1), H = ps.at(-1);
  if (ps.length !== ss.length + 1 || ss.some((d, k) => ps[k] !== d)) continue;
  const lead = ss.slice(0, -1), blocks = lead.reduce((a, b) => a * b, 1);
  for (let k = 0; k < blocks; ++k)              // one block's [C, H]
    folds.push({ n: C * H, dims: [C, H], dst: [H, 1], s: [scaleName, k * C, [1, 0]], p: [projName, k * C * H, [H, 1]] });
  if (lead.length >= 1) {                         // [C, B, H] over the last lead axis, per outer index
    const B = lead.at(-1), outer = blocks / B;
    for (let g = 0; g < outer; ++g)
      folds.push({ n: C * B * H, dims: [C, B, H], dst: [B * H, H, 1],
                   s: [scaleName, g * B * C, [1, C, 0]], p: [projName, g * B * C * H, [H, C * H, 1]] });
  }
}
const evalFold = (f) => {
  const out = new Float32Array(f.n), S = valuesOf.get(f.s[0]), P = valuesOf.get(f.p[0]);
  const [d0, d1, d2 = 1] = f.dims;
  for (let a = 0; a < d0; ++a) for (let b = 0; b < d1; ++b) for (let c = 0; c < d2; ++c) {
    const i = [a, b, c].slice(0, f.dims.length), dot = (st, o) => i.reduce((x, v, k) => x + v * st[k], o);
    out[dot(f.dst, 0)] = Math.fround(S[dot(f.s[2], f.s[1])] * P[dot(f.p[2], f.p[1])]);
  }
  return out;
};
const lines = readFileSync(`${weightsDir}/model.idx`, "utf8").split("\n").filter(Boolean);
const binFd = openSync(`${weightsDir}/model.bin`, "r");     // (read a tensor at a time: a file may pass 2 GiB)
const outLines = [];
let slices = 0, zeros = 0, derived = 0, consts = 0; const missing = [];
for (const line of lines) {
  if (!line.startsWith("t ")) { outLines.push(line); continue; }
  const [, name, offset, length] = line.split(" ");
  // a dialect's array-valued convention (protenix2's emptyTemplateRestypeColumns) is the FEATURISER's, which
  // the input exporter applies; the port reads only the dialect's flags (the `m` lines), so it is not a weight
  if (name.startsWith("trunk.dialect.")) continue;
  const n = Number(length), t = new Float32Array(n);
  readSync(binFd, new Uint8Array(t.buffer), 0, n * 4, Number(offset) * 4);
  let found = null;
  const probe = Array.from({ length: Math.min(8, n) }, (_, k) => Math.floor((k * n) / Math.min(8, n)));
  search: for (const [source, v] of decoded) {
    if (v.length % n) continue;
    for (let first = 0; first + n <= v.length; first += n) {
      if (probe.some((k) => v[first + k] !== t[k])) continue;
      let same = true;
      for (let i = 0; same && i < n; ++i) if (v[first + i] !== t[i]) same = false;
      if (same) { found = [source, first]; break search; }
    }
  }
  // stock AlphaFold 3's Fourier noise embedding is a CONSTANT of its source (frozen from a fixed seed), which
  // LocalFold's exporter bakes into the bundle and DeepMind's own af3.bin.zst does not carry: written as its
  // values (`c`, 9 significant digits round-trip a float32) so one map reads the bundle and that blob alike.
  // Ported models trained their own and carry it, so theirs stay `b` slices.
  if (found && manifest.model?.name === "alphafold3" && /\/fourier_embedding_(weight|bias)$/.test(found[0])) {
    outLines.push(`c ${name} ${n} ${Array.from(t, (x) => x.toPrecision(9)).join(" ")}`); ++consts; continue;
  }
  if (found) { outLines.push(`b ${name} ${found[0]} ${found[1]} ${n}`); ++slices; continue; }
  if (t.every((x) => x === 0)) { outLines.push(`z ${name} ${n}`); ++zeros; continue; }
  if (t.every((x) => x === 1)) { outLines.push(`p ${name} ${n} o 1 ${n} 0 1`); ++derived; continue; }
  const fold = folds.find((f) => f.n === n && evalFold(f).every((x, i) => x === t[i]));
  if (fold) {
    const r = fold.dims.length;
    outLines.push(`p ${name} ${n} x ${r} ${fold.dims.join(" ")} 0 ${fold.dst.join(" ")} ` +
                  `${fold.s[0]} ${fold.s[1]} ${fold.s[2].join(" ")} ${fold.p[0]} ${fold.p[1]} ${fold.p[2].join(" ")}`);
    ++derived; continue;
  }
  missing.push(`${name} [${n}]${t.every((x) => x === 1) ? " (all ones)" : ""}`);
}
if (missing.length) {
  console.error(`${missing.length} tensors in no tensor of ${bundleDir} (the exporter derives them):\n  ${missing.join("\n  ")}`);
  process.exit(1);
}
writeFileSync(out, outLines.join("\n") + "\n");
console.log(`${out}: ${slices} slices, ${zeros} zero tensors, ${derived} derived, ${consts} constants, ${outLines.length - slices - zeros - derived - consts} metadata lines`);
