/**
 * Which WGSL constructs this browser's COMPILER accepts - beyond what `device.features` says.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-wgsl-support.js
 *     LOCALFOLD_GPU_FIREFOX=1 node tools/gpu-chrome.mjs tools/gpu/probe-wgsl-support.js
 *
 * Chrome compiles WGSL with Tint and Firefox with naga, and the two do not accept the same text: a kernel written
 * and gated in Chrome can be refused in Firefox at pipeline creation, on a device that reports every feature it
 * asks for (issue #10 - AlphaFold 2's attention, BioEmu2's GEMM). Each row is one construct this repository's
 * kernels use, compiled alone; a refusal prints the compiler's own message.
 */
const CASES = {
  "bitcast<vec4<f16>>(vec2<u32>)": ["enable f16;", "let v: vec4<f16> = bitcast<vec4<f16>>(vec2<u32>(u32(o[0]), 1u)); o[0] = f32(v.w);"],
  "bitcast<vec2<f16>>(u32)": ["enable f16;", "let v: vec2<f16> = bitcast<vec2<f16>>(u32(o[0])); o[0] = f32(v.y);"],
  "bitcast<u32>(vec2<f16>)": ["enable f16;", "let v: u32 = bitcast<u32>(vec2<f16>(f16(o[0]), 1.0h)); o[0] = f32(v);"],
  "bitcast<vec2<u32>>(vec4<f16>)": ["enable f16;", "let v: vec2<u32> = bitcast<vec2<u32>>(vec4<f16>(f16(o[0]))); o[0] = f32(v.y);"],
  "bitcast<f32>(vec2<f16>)": ["enable f16;", "let v: f32 = bitcast<f32>(vec2<f16>(f16(o[0]), 1.0h)); o[0] = v;"],
  "unpack2x16float / pack2x16float": ["", "let v = unpack2x16float(u32(o[0])); o[0] = f32(pack2x16float(v));"],
  "f16 arithmetic": ["enable f16;", "let v = f16(o[0]) * 2.0h; o[0] = f32(v);"],
  "subgroupAdd (subgroups)": ["enable subgroups;", "o[0] = subgroupAdd(o[0]);"],
  "const array indexed by a let": ["", "const T = array<f32, 4>(1.0, 2.0, 3.0, 4.0); let i = u32(o[0]); var t = T; o[0] = t[i];"],
  "dot of vec4<f16>": ["enable f16;", "let a = vec4<f16>(f16(o[0])); o[0] = f32(dot(a, a));"],
  "fma": ["", "o[0] = fma(o[0], 2.0, 1.0);"],
  "arrayLength": ["", "o[0] = f32(arrayLength(&o));"],
  "switch": ["", "switch (u32(o[0])) { case 0u: { o[0] = 1.0; } default: { o[0] = 2.0; } }"],
};

export async function main(device) {
  const rows = [];
  for (const [name, [header, body]] of Object.entries(CASES)) {
    const feature = header.includes("f16") ? "shader-f16" : header.includes("subgroups") ? "subgroups" : null;
    if (feature !== null && !device.features.has(feature)) { rows.push({ construct: name, accepted: null, note: `device lacks ${feature}` }); continue; }
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ code: `${header}
@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1) fn main() {
  ${body}
}` });
    const info = await module.getCompilationInfo();
    await device.popErrorScope();
    const errors = info.messages.filter((message) => message.type === "error").map((message) => message.message.split("\n").slice(0, 3).join(" ").slice(0, 200));
    rows.push({ construct: name, accepted: errors.length === 0, ...(errors.length === 0 ? {} : { message: errors[0] }) });
    console.log(`${errors.length === 0 ? "ok     " : "REFUSED"}  ${name}${errors.length === 0 ? "" : `  -  ${errors[0]}`}`);
  }
  return { rows, refused: rows.filter((row) => row.accepted === false).map((row) => row.construct) };
}
