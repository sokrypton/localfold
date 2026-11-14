/**
 * WGSL this repository writes, rewritten for a compiler that refuses part of it.
 *
 * 🔴 FIREFOX'S COMPILER (naga) REFUSES EVERY BITCAST THAT CHANGES THE ELEMENT COUNT ACROSS f16 - two halves in a
 * u32, four in a vec2<u32> - which Chrome's (Tint) takes and which the packed-f16 kernels use to read and write
 * their operands: AlphaFold 2's attention, BioEmu2's and RFD4-Proteina's GEMMs (issue #10; the message is "the
 * type ... is expected to be `vec2<f16>`, but got `f32`"). `tools/gpu/probe-wgsl-support.js` lists what a browser
 * takes; webgpu/runtime/device.js asks once per device and `f16WidthBitcastAvailable` answers.
 *
 * The rewrite is the same halves by the two builtins every compiler has: `unpack2x16float` widens the two halves
 * of a word exactly, and `pack2x16float` of two values that ARE halves stores them exactly. Only these three
 * spellings are rewritten; a bitcast between types of one width (f32 and u32) is left alone.
 */
import { recordWgslSupport } from "./device-profile.js";

/**
 * Ask this device's compiler, once, by compiling the construct - and record the answer for
 * `f16WidthBitcastAvailable`. Typed and read at `.w`: naga takes the bitcast alone and gives it TWO elements, so
 * only a use that needs four refuses.
 */
export async function recordCompilerSupport(device) {
  if (!device.features.has("shader-f16")) return;
  let accepted = true;
  try {
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ label: "wgsl-support.f16-width-bitcast", code:
      "enable f16;\n@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n"
      + "@compute @workgroup_size(1) fn main() {\n"
      + "  let v: vec4<f16> = bitcast<vec4<f16>>(vec2<u32>(u32(o[0]), 1u));\n  o[0] = f32(v.w);\n}\n" });
    const info = await module.getCompilationInfo();
    const refused = await device.popErrorScope();
    accepted = refused === null && !info.messages.some((message) => message.type === "error");
  } catch { accepted = false; }
  recordWgslSupport(device, { f16WidthBitcast: accepted });
}

const HELPERS = `
fn lf_h4_from_words(w: vec2<u32>) -> vec4<f16> {
  return vec4<f16>(vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)));
}
fn lf_word_from_h2(v: vec2<f16>) -> u32 { return pack2x16float(vec2<f32>(v)); }
`;

/** The index of the parenthesis closing the one that opens at `open`. */
function closing(code, open) {
  let depth = 0;
  for (let index = open; index < code.length; index += 1) {
    if (code[index] === "(") depth += 1;
    else if (code[index] === ")") { depth -= 1; if (depth === 0) return index; }
  }
  throw new Error("unbalanced parentheses in a shader's bitcast");
}

function replaceCalls(code, head, rewrite) {
  let out = "";
  let from = 0;
  for (;;) {
    const at = code.indexOf(head, from);
    if (at < 0) return out + code.slice(from);
    const open = at + head.length - 1;
    const close = closing(code, open);
    // the argument may hold another of the same call
    out += code.slice(from, at) + rewrite(replaceCalls(code.slice(open + 1, close), head, rewrite));
    from = close + 1;
  }
}

export function withoutF16WidthBitcasts(code) {
  if (!code.includes("bitcast<")) return code;
  let out = replaceCalls(code, "bitcast<vec4<f16>>(", (argument) => `lf_h4_from_words(${argument})`);
  out = replaceCalls(out, "bitcast<vec2<f16>>(", (argument) => `vec2<f16>(unpack2x16float(${argument}))`);
  // (only where the argument is spelled as two halves: `bitcast<u32>(x)` of an f32 is a bitcast of one width)
  out = replaceCalls(out, "bitcast<u32>(", (argument) => (/^\s*vec2<f16>\(/.test(argument)
    ? `lf_word_from_h2(${argument})` : `bitcast<u32>(${argument})`));
  if (out === code) return code;
  if (/bitcast<(vec2<u32>|f32)>\(\s*vec[24]<f16>/.test(out)) {
    throw new Error("a shader bitcasts f16 vectors in a spelling withoutF16WidthBitcasts does not rewrite");
  }
  // after the directives, which WGSL wants first
  const lines = out.split("\n");
  let last = -1;
  for (let index = 0; index < lines.length; index += 1) {
    if (/^\s*(enable|requires|diagnostic)\b/.test(lines[index])) last = index;
    else if (lines[index].trim() !== "" && !lines[index].trim().startsWith("//")) break;
  }
  lines.splice(last + 1, 0, HELPERS);
  return lines.join("\n");
}
