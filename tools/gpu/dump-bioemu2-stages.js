// webgpu/bioemu2's stages, handed back as one base64 blob for tools/dump-bioemu2-webgpu.mjs to write as a
// cuda/featurise input directory - the oracle metal/bioemu2 reads with --oracle=.
//
//     node tools/gpu-chrome.mjs tools/gpu/dump-bioemu2-stages.js --job=/tools/fixtures/bioemu2/6mrr.json \
//       [--model=/model-bioemu2-int5] [--sigmas=160,16,1,0.05] [--mode=structure]
//
// 🔴 THIS RUNS IN CHROME AND NOT IN NODE, THOUGH webgpu/node.js EXISTS. Dawn's node binding on this box has no
// `device.tick` and its `mapAsync` never settles, so a node run exits 0 having written nothing - which is why
// nothing else in the tree imports createNodeDevice either. A page cannot write files, so the dump comes back
// through gpu-chrome's own result channel (a string is printed raw) between sentinels.
//
// 🔴 THE STAGES ARE CHAINED, each from this port's own previous one, as cuda/bioemu2's --oracle arm compares them:
// the first stage that disagrees is the one to read and a later one inherits it.
import { Bioemu2Gpu } from "../../webgpu/bioemu2/network.js";
import { bundleReader } from "../../shared/bioemu2/weights.js";
import { featuriseBioemu2, bioemu2JobFromAf3Json } from "../../shared/bioemu2/featurise.js";
import { fetchComponents, smilesTemplates } from "../../shared/bioemu2/inputs.js";

const option = (args, name, fallback) => (args ?? []).find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

export const SENTINEL = "---BE2DUMP---";

/** A deterministic draw, so both ports read the same coordinates: a 64-bit LCG into a Box-Muller normal. */
function draw(n, seed) {
  let s = BigInt(seed) * 6364136223846793005n + 1442695040888963407n;
  const next = () => {
    s = (s * 6364136223846793005n + 1442695040888963407n) & 0xffffffffffffffffn;
    return Number(s >> 11n) / 9007199254740992;
  };
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    const u = Math.max(next(), 1e-12);
    out[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
  }
  return out;
}

export async function main(device, args) {
  const jobUrl = option(args, "job", "/tools/fixtures/bioemu2/6mrr.json");
  const model = option(args, "model", "/model-bioemu2-int5");
  const mode = option(args, "mode", "structure");
  const sigmas = option(args, "sigmas", "160,16,1,0.05").split(",").map(Number);
  // --half=off --trunk=f32: check-bioemu2-network.js's float32 arm (float32 weights decoded from the same codes, the
  // pair track in float32) - the one a float32 port is held to; the default is the shipped arm
  const half = option(args, "half", "on") !== "off";
  const trunkPrecision = option(args, "trunk", "device");

  const job = bioemu2JobFromAf3Json(await (await fetch(jobUrl)).json());
  const ccd = await fetchComponents(job, { local: (code) => `/tools/fixtures/ccd/${code}.cif` });
  const { features, atomKeys } = featuriseBioemu2(job, {
    ccd, mode, contacts: null, smilesTemplates: await smilesTemplates(job) });

  const read = bundleReader(model);
  await read.prefetch();
  const net = new Bioemu2Gpu(device, read, { half, trunkPrecision });
  const g = net.upload(features);
  const realAtoms = features.atom_mask.reduce((s, v) => s + v, 0);

  // the index's lines and the flat float32 payload, as metal/core/model.cpp's loadInput reads them
  const lines = [];
  const parts = [];
  let words = 0;
  const meta = (name, value) => lines.push(`m ${name} ${value}`);
  const put = (name, array) => {
    const data = array instanceof Float32Array ? array : Float32Array.from(array);
    lines.push(`f ${name} ${words} ${data.length}`);
    parts.push(data);
    words += data.length;
  };

  meta("o/tokens", g.T);
  meta("o/half", half ? 1 : 0);
  meta("o/trunk_f32", trunkPrecision === "f32" ? 1 : 0);
  meta("o/real_atoms", realAtoms);

  const sIn = await net.inputEmbedder(g);
  put("o/s_inputs", sIn);

  const trunk = await net.trunkProgram(g, sIn);
  put("o/z_init", await net.ops.read(trunk.zInit));
  put("o/s_init", trunk.sInit);
  for (let k = 1; k <= 4; k += 1) {
    await trunk.pass();
    put(`o/pass${k}.single`, await net.ops.read(trunk.s));
    put(`o/pass${k}.pair`, await net.ops.read(trunk.z));
  }
  const sTrunk = await net.ops.read(trunk.s);
  const zHost = await net.ops.read(trunk.z);
  const zTrunk = net.ops.buffer("dump.z", zHost.length, zHost);

  const { logits } = await net.distogram(g, zTrunk);
  put("o/distogram", logits);

  const dn = await net.denoiserProgram(g, sIn, sTrunk, zTrunk);
  put("o/ctx.z", await net.ops.read(dn.context.z));
  put("o/ctx.q0", dn.context.q0);
  put("o/ctx.p", dn.context.p);
  for (let k = 0; k < sigmas.length; k += 1) {
    const x = draw(atomKeys.length * 3, 1000 + k);
    const out = await dn.denoise(x, sigmas[k]);
    meta(`o/denoise${k}.sigma`, sigmas[k]);
    put(`o/denoise${k}.x`, x);
    put(`o/denoise${k}.out`, out.subarray(0, realAtoms * 3));
  }

  // one buffer, then base64 in chunks (btoa over a 150 MB string at once throws)
  const flat = new Float32Array(words);
  let at = 0;
  for (const p of parts) { flat.set(p, at); at += p.length; }
  const bytes = new Uint8Array(flat.buffer, flat.byteOffset, flat.byteLength);
  let b64 = "";
  for (let i = 0; i < bytes.length; i += 32766) {
    let s = "";
    const end = Math.min(i + 32766, bytes.length);
    for (let k = i; k < end; k += 1) s += String.fromCharCode(bytes[k]);
    b64 += btoa(s);     // (32766 = 3 x 10922: a chunk that is not a whole number of 3-byte groups ends in `=`
  }                     //  padding, and a decoder stops at the first - 32768 here wrote 38 MB and read back 32 KB)
  console.log(`dumped ${lines.length} entries, ${(bytes.length / 1e6).toFixed(1)} MB (${g.T} tokens, ${realAtoms} atoms)`);
  return `${SENTINEL}\n${JSON.stringify({ idx: lines, tokens: g.T, atoms: realAtoms })}\n${b64}\n${SENTINEL}`;
}
