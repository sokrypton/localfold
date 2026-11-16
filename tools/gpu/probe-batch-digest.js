/**
 * A digest of the featurised AF3 batch, field by field - to compare two BROWSERS, not two models.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-batch-digest.js --model=/model-openbind0-int5/manifest.json
 *     LOCALFOLD_GPU_FIREFOX=1 node tools/gpu-chrome.mjs tools/gpu/probe-batch-digest.js --model=...
 *
 * The batch is built on the HOST, so it is the JavaScript engine's: V8 and SpiderMonkey differ in Math's last
 * place and in what they ship. Every typed field is hashed over its bytes (FNV-1a) with its length and sum.
 */
import { af3BatchFromA3m } from "../../shared/af3/featurise/batch.js";
import { dialectFor, featuriserDialect } from "../../shared/af3/dialect.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function digest(value) {
  const bytes = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  let hash = 0x811c9dc5;
  for (let index = 0; index < bytes.length; index += 1) hash = Math.imul(hash ^ bytes[index], 0x01000193) >>> 0;
  let sum = 0;
  for (let index = 0; index < value.length; index += 1) sum += Number(value[index]);
  return `${value.constructor.name}[${value.length}] ${hash.toString(16).padStart(8, "0")} sum ${sum}`;
}

function walk(value, path, out, depth = 0) {
  if (value === null || value === undefined || depth > 4) return;
  if (ArrayBuffer.isView(value)) { out[path] = digest(value); return; }
  if (Array.isArray(value)) {
    if (value.length > 0 && typeof value[0] === "number") out[path] = `Array[${value.length}] sum ${value.reduce((a, b) => a + b, 0)}`;
    else if (value.length <= 64) value.forEach((item, index) => walk(item, `${path}[${index}]`, out, depth + 1));
    return;
  }
  if (typeof value === "object") { for (const key of Object.keys(value).sort()) walk(value[key], path === "" ? key : `${path}.${key}`, out, depth + 1); return; }
  if (typeof value === "number" || typeof value === "string" || typeof value === "boolean") out[path] = String(value).slice(0, 60);
}

export async function main(_device, args) {
  const path = option(args, "model", "/model-af3-int5/manifest.json");
  const name = (await (await fetch(path)).json())?.model?.name;
  const sequence = option(args, "sequence", "GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK");
  const built = af3BatchFromA3m(sequence, option(args, "a3m", "") === "" ? null
    : await (await fetch(option(args, "a3m", ""))).text(), {
    maxSequences: 512, seed: 20260831, ...featuriserDialect(dialectFor(name)),
  });
  const fields = {};
  walk(built.batch, "", fields);
  return { model: name, engine: navigator.userAgent.includes("Firefox") ? "SpiderMonkey" : "V8", count: Object.keys(fields).length, fields };
}
