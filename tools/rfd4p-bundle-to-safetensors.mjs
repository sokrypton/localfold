/**
 * The published int8 bundle written back out as a float32 `.safetensors` - what
 * `cuda/rfd4p/export_input.mjs --checkpoint=` and `export_weights.mjs` read.
 *
 *   node tools/rfd4p-bundle-to-safetensors.mjs [--bundle=model-rfd4p-int8] [--out=<file>]
 *     [--prefix=nn.,autoencoder.]
 *
 * 🔴 THIS IS NOT THE VENDOR CHECKPOINT AND MUST NEVER BE CALLED ONE. Every int8 tensor comes
 * back as `code * scale` at block 64, so what it writes is the bundle's weights DEQUANTISED -
 * the f32 checkpoint rounded once and unrounded, not recovered. An oracle built from this file
 * is an oracle for THE BUNDLE, which is the point: `metal/rfd4p` reads these same shards, so
 * the quantisation is on both sides of the comparison and cancels, and a residual is a port
 * difference rather than a weight difference. It is the same argument `tools/dump-bioemu2-webgpu.mjs`
 * makes for holding a Metal port to the page's own int5 bundle.
 *
 * 🔴 AND IT EXISTS BECAUSE THIS MACHINE HAS NO f32 CHECKPOINT. `export_input.mjs` defaults to a
 * 1.7 GB Hugging Face URL; an Apple box with the page's 448 MB bundle already on disk should not
 * have to fetch the vendor weights to produce a reference for the weights it actually runs.
 *
 * Written shard by shard in manifest order: one shard is mapped at a time and one tensor decoded
 * at a time, so the peak is a shard plus a tensor rather than the whole 1.9 GB of float32.
 */
import { closeSync, mkdirSync, openSync, readFileSync, renameSync, writeSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { readShardedTensor } from "../shared/rfd4p/safetensors.js";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const option = (name, fallback) => {
  const hit = process.argv.slice(2).find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};
const bundle = resolve(ROOT, option("bundle", "model-rfd4p-int8"));
const out = resolve(ROOT, option("out", `${bundle}/model.safetensors`));
const prefixes = option("prefix", "").split(",").filter(Boolean);

const manifest = JSON.parse(readFileSync(`${bundle}/manifest.json`, "utf8"));
const tensors = manifest.tensors ?? {};
// Manifest order within each shard, shards in the manifest's own order: the file is written in one
// forward pass, so the layout follows the read order rather than needing the whole set in memory.
const shards = (manifest.shards ?? []).map((s) => s.file);
const byShard = new Map(shards.map((f) => [f, []]));
for (const [name, rec] of Object.entries(tensors)) {
  if (prefixes.length && !prefixes.some((p) => name.startsWith(p))) continue;
  const list = byShard.get(rec.file);
  if (!list) throw new Error(`${name}: shard ${rec.file} is not in the manifest's shard list`);
  list.push([name, rec]);
}

// The header first, because safetensors states every tensor's byte range up front.
const header = {};
let at = 0;
for (const file of shards) {
  for (const [name, rec] of byShard.get(file)) {
    const elements = rec.shape.reduce((p, v) => p * v, 1);
    header[name] = { dtype: "F32", shape: rec.shape, data_offsets: [at, at + elements * 4] };
    at += elements * 4;
  }
}
const headerBytes = Buffer.from(JSON.stringify(header), "utf8");
// safetensors pads the header to 8 bytes; the length prefix is a little-endian u64.
const pad = (8 - (headerBytes.length % 8)) % 8;
const length = Buffer.alloc(8);
length.writeBigUInt64LE(BigInt(headerBytes.length + pad));

mkdirSync(dirname(out), { recursive: true });
const tmp = `${out}.writing`;
const fd = openSync(tmp, "w");
writeSync(fd, length);
writeSync(fd, headerBytes);
if (pad) writeSync(fd, Buffer.alloc(pad, 0x20));
let written = 0;
let count = 0;
for (const file of shards) {
  const list = byShard.get(file);
  if (!list.length) continue;
  const raw = readFileSync(`${bundle}/${file}`);
  const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  for (const [name, rec] of list) {
    const data = readShardedTensor(rec, buf, true);
    const elements = rec.shape.reduce((p, v) => p * v, 1);
    if (data.length !== elements) throw new Error(`${name}: decoded ${data.length} of ${elements}`);
    writeSync(fd, Buffer.from(data.buffer, data.byteOffset, data.byteLength));
    written += data.byteLength;
    count += 1;
  }
}
closeSync(fd);
renameSync(tmp, out);
if (written !== at) throw new Error(`the header names ${at} bytes of tensor data, ${written} were written`);
console.log(`${out}: ${count} tensors, ${(written / 1e9).toFixed(2)} GB float32 (dequantised from ${manifest.bundle?.format ?? "?"})`);
