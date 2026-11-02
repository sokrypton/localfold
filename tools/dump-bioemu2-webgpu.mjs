/**
 * webgpu/bioemu2's stages, written as a cuda/featurise input directory - what metal/bioemu2 reads with --oracle=.
 *
 *     node tools/dump-bioemu2-webgpu.mjs --out=<dir> [--job=/tools/fixtures/bioemu2/6mrr.json]
 *       [--model=/model-bioemu2-int5] [--sigmas=160,16,1,0.05] [--mode=structure]
 *
 * Runs tools/gpu/dump-bioemu2-stages.js in Chrome (tools/gpu-chrome.mjs) and writes what it hands back.
 *
 * 🔴 WHY WEBGPU AND NOT THE COLAB ORACLE. tools/oracle/dump_bioemu2.py is the reference everything else is held to,
 * but it needs af3-any-model's colab environment and the vendor checkpoint, which no Apple box has. This is the next
 * reference down and a real one: a SEPARATE implementation (WGSL, its own kernels, its own featuriser) of the same
 * network reading THE SAME int5 bundle - so the quantisation cancels and a residual is a port difference, not a
 * weight difference. webgpu/bioemu2 is itself held to the colab dumps (tools/gpu/check-bioemu2-network.js), so the
 * absolute claim transfers; what this settles is "do the two ports agree", the question a new port has.
 *
 * 🔴 THE DENOISER'S INPUTS ARE DUMPED, NOT DRAWN TWICE: x at each sigma is written beside the output, so the Metal
 * side reads the same coordinates - AGENTS.md wants a differential test deterministic, and two ports agreeing on a
 * sampler's RNG is a different question.
 */
import { spawn } from "node:child_process";
import { writeFile, mkdir } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SENTINEL = "---BE2DUMP---";
const out = process.argv.find((a) => a.startsWith("--out="))?.slice(6);
if (!out) { console.error("--out=<dir>"); process.exit(1); }
const passed = process.argv.slice(2).filter((a) => !a.startsWith("--out="));

const child = spawn(process.execPath, [resolve(ROOT, "tools/gpu-chrome.mjs"), "tools/gpu/dump-bioemu2-stages.js", ...passed],
                    { cwd: ROOT, stdio: ["ignore", "pipe", "inherit"],
                      env: { ...process.env, LOCALFOLD_GPU_TIMEOUT_MS: process.env.LOCALFOLD_GPU_TIMEOUT_MS ?? "1800000" } });
const chunks = [];
child.stdout.on("data", (c) => chunks.push(c));
const code = await new Promise((res) => child.on("exit", res));
const text = Buffer.concat(chunks).toString("utf8");
const a = text.indexOf(SENTINEL), b = text.lastIndexOf(SENTINEL);
for (const line of text.slice(0, a < 0 ? text.length : a).split("\n")) if (line.trim()) console.log(line);
if (code !== 0 || a < 0 || b <= a) { console.error(`the page handed back no dump (exit ${code})`); process.exit(1); }
const [header, b64] = text.slice(a + SENTINEL.length, b).trim().split("\n");
const { idx } = JSON.parse(header);
const bin = Buffer.from(b64, "base64");
const words = idx.filter((l) => l[0] !== "m").reduce((n, l) => Math.max(n, Number(l.split(" ")[2]) + Number(l.split(" ")[3])), 0);
if (words * 4 !== bin.length) { console.error(`the index names ${words * 4} bytes, the payload is ${bin.length}`); process.exit(1); }
await mkdir(out, { recursive: true });
await writeFile(resolve(out, "model.bin"), bin);
await writeFile(resolve(out, "model.idx"), `${idx.join("\n")}\n`);
console.log(`-> ${out} (${(bin.length / 1e6).toFixed(1)} MB, ${idx.length} entries)`);
