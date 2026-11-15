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
import { recordWgslSupport, f16WidthBitcastAvailable, loopsUnrolledFor } from "./device-profile.js";

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
  // (the refusal is also how naga is recognised, and naga's loops are what `withUnrolledLoops` is for)
  recordWgslSupport(device, { f16WidthBitcast: accepted, unrollLoops: !accepted, naga: !accepted });
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

/**
 * A shader with its constant-count loops unrolled, for a compiler whose loops are slow.
 *
 * 🔴 EVERY LOOP ITERATION COSTS IN FIREFOX, WHATEVER IS IN IT. tools/gpu/probe-loop-cost.js, Firefox 160 against
 * Chrome 157 on an M2, 1536 iterations a lane: a loop of pure arithmetic 60.9 ms against 23.9, a storage read
 * 52.7 against 14.6, the denoiser's dense inner loop 61.7 against 18.6 - and the same bodies written eight to an
 * iteration 16.1 against 14.0, 15.7 against 15.6 and 28.2 against 22.5. So it is the loop and not the reads (naga
 * guards every loop it emits against running for ever, and the guard is in the iteration). Chrome is no faster
 * unrolled and its dense loop is slower (22.5 against 18.6), so this is applied for that compiler alone.
 *
 * What is unrolled: an innermost `for (var i = A; i < B; i += S)` (or `i++`) whose three terms are integer
 * literals or module constants, whose body is short, neither assigns `i` nor leaves early, and has no barrier. The body is repeated with
 * `let i` bound to each value - `factor` to an iteration, and the remainder after the loop as straight code -
 * so the arithmetic and its order are the loop's own. Comments are dropped first (a brace in one would be counted).
 * 🔴 A loop whose start or bound is NOT a constant (a lane-strided one, a count from a uniform - 88 of AlphaFold 2's
 * 115) was unrolled too, behind a test for eight remaining, and bought nothing: an AF2 block 23.1 -> 25.9 ms, its fold
 * 1.15 -> 1.15 s, AlphaFold 3 and ESMFold2 unmoved. Not kept.
 */
/** The longest loop body, in characters, that is unrolled. */
const UNROLLED_BODY_LIMIT = 800;

export function withUnrolledLoops(source, factor = 8) {
  if (!source.includes("for")) return source;
  const code = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
  // module constants that are integers, by name; one declared twice (shadowed in a function) is not trusted
  const constants = new Map();
  const declared = new Map();
  for (const match of code.matchAll(/\b(?:const|let|var|override)\s+(\w+)\b/g)) {
    declared.set(match[1], (declared.get(match[1]) ?? 0) + 1);
  }
  const integer = (text) => {
    const expression = text.trim().replace(/\b([A-Za-z_]\w*)\b/g, (name) => {
      const known = constants.get(name);
      return known === undefined ? "?" : `${known.value}${known.unsigned ? "u" : ""}`;
    });
    if (!/^[\du\s+\-*()]+$/.test(expression) || !/\d/.test(expression)) return null;
    const unsigned = /\du/.test(expression);
    if (unsigned !== !/\d(?!u|\d)/.test(expression)) return null;   // every literal one type
    let value;
    try { value = Function(`"use strict"; return (${expression.replace(/u/g, "")});`)(); } catch { return null; }
    return Number.isSafeInteger(value) && value >= 0 ? { value, unsigned } : null;
  };
  const pending = [...code.matchAll(/^[ \t]*const\s+(\w+)\s*(?::\s*(u32|i32)\s*)?=\s*([^;]+);/gm)];
  for (let round = 0, grew = true; grew && round < 8; round += 1) {
    grew = false;
    for (const [, name, type, text] of pending) {
      if (constants.has(name) || declared.get(name) !== 1) continue;
      const found = integer(text);
      if (found === null) continue;
      constants.set(name, { value: found.value, unsigned: type === undefined ? found.unsigned : type === "u32" });
      grew = true;
    }
  }

  const head = /for\s*\(\s*var\s+(\w+)\s*(?::\s*(u32|i32)\s*)?=\s*([^;]+);\s*\1\s*<\s*([^;]+);\s*\1\s*(?:\+=\s*([^;{]+?)|\+\+)\s*\)\s*\{/g;
  const loops = [];
  for (const match of code.matchAll(head)) {
    let depth = 1;
    let end = match.index + match[0].length;
    for (; end < code.length && depth > 0; end += 1) {
      if (code[end] === "{") depth += 1; else if (code[end] === "}") depth -= 1;
    }
    if (depth !== 0) continue;
    loops.push({ match, bodyStart: match.index + match[0].length, end });
  }
  let out = code;
  for (const { match, bodyStart, end } of loops.reverse()) {
    const [, name, type, startText, boundText, stepText] = match;
    const body = code.slice(bodyStart, end - 1);
    // 🔴 ONLY A SMALL BODY, AND NEVER ONE THAT STAGES. The guard is a few instructions an iteration, so it is the
    // whole cost of a three-instruction body and none of a tile's: the triangle contraction's loop over k (a staging
    // pass, two barriers and a 4x4 tile of accumulators an iteration) went 2.7 ms to 16 unrolled eight times.
    if (/\b(for|while|loop|break|continue|return|discard|workgroupBarrier)\b/.test(body)
      || body.length > UNROLLED_BODY_LIMIT) continue;
    if (new RegExp(`(^|[^\\w.])${name}\\s*(=(?!=)|\\+=|-=|\\*=|\\+\\+|--)|&\\s*${name}\\b`).test(body)) continue;
    const start = integer(startText), bound = integer(boundText), step = integer(stepText ?? (type === "i32" ? "1" : "1u"));
    if (start === null || bound === null || step === null || step.value === 0) continue;
    const unsigned = type === undefined ? start.unsigned : type === "u32";
    if (bound.unsigned !== unsigned || (stepText !== undefined && step.unsigned !== unsigned)
      || start.unsigned !== unsigned) continue;
    const trips = Math.max(0, Math.ceil((bound.value - start.value) / step.value));
    if (trips < 2) continue;
    const suffix = unsigned ? "u" : "i";
    const inner = `${name}_lf`;
    if (code.includes(inner)) continue;
    const copy = (value) => `{ let ${name} = ${value};${body}}`;
    const whole = trips > factor ? Math.floor(trips / factor) : 0;
    const parts = [];
    if (whole > 0) {
      const copies = [];
      for (let k = 0; k < factor; k += 1) {
        copies.push(copy(k === 0 ? inner : `${inner} + ${k * step.value}${suffix}`));
      }
      parts.push(`for (var ${inner} = ${start.value}${suffix}; ${inner} < ${start.value + whole * factor * step.value}${suffix};`
        + ` ${inner} += ${factor * step.value}${suffix}) {\n${copies.join("\n")}\n}`);
    }
    for (let trip = whole * factor; trip < trips; trip += 1) parts.push(copy(`${start.value + trip * step.value}${suffix}`));
    out = out.slice(0, match.index) + parts.join("\n") + out.slice(end);
  }
  return out;
}

/** Bytes of a workgroup array's element, by its WGSL type; a type not here is left alone. */
const ELEMENT_BYTES = new Map([
  ["f32", 4], ["u32", 4], ["i32", 4], ["f16", 2],
  ["vec2<f32>", 8], ["vec3<f32>", 16], ["vec4<f32>", 16], ["vec2<u32>", 8], ["vec3<u32>", 16], ["vec4<u32>", 16],
  ["vec2<i32>", 8], ["vec3<i32>", 16], ["vec4<i32>", 16], ["vec2<f16>", 4], ["vec3<f16>", 8], ["vec4<f16>", 8],
]);

/** The largest workgroup array, in bytes, that Firefox starts a workgroup with cheaply. */
export const WORKGROUP_ARRAY_BYTES = 8192;

/** The index of the bracket closing the one that opens at `open`. */
function closingBracket(code, open) {
  let depth = 0;
  for (let index = open; index < code.length; index += 1) {
    if (code[index] === "[") depth += 1;
    else if (code[index] === "]") { depth -= 1; if (depth === 0) return index; }
  }
  return -1;
}

/**
 * A shader with every workgroup array over 8 KiB cut into parts of at most 8 KiB.
 *
 * 🔴 IN FIREFOX A WORKGROUP ARRAY OVER 8192 BYTES COSTS EVERY WORKGROUP ~3 ns A BYTE BEFORE IT RUNS, AND ONE OF
 * 8192 COSTS NOTHING. tools/gpu/probe-workgroup-init.js, 4096 workgroups touching 64 floats of an array declared at
 * 2048 floats: 0.98 ms; at 2052: 36.2; at 4096: 108; at 8192: 428 (Chrome: 0.40, -, 0.47, 1.8). WGSL zeroes workgroup
 * memory at entry and past that size it is done slowly. It is the ARRAY's size and not the kernel's total - two
 * arrays of 2048 floats are 2.0 ms - and nesting does not help (rows of 2048 in one variable are 108 again). So
 * ESMFold2's hidden LayerNorm, whose tile is 8320 bytes, was 108 ms of a 284 ms trunk pass where Chrome's is 3, and
 * OpenDDE at 384 channels has twenty-two such kernels.
 *
 * Each such `var<workgroup> name: array<T, N>` becomes parts `name_p0..` and two functions, `name_get(i)` and
 * `name_set(i, v)`, that pick the part; every `name[e]` in the text is rewritten to one of them. An array used in a
 * way this does not rewrite (a component assigned, a pointer taken) is left whole.
 */
export function withSmallWorkgroupArrays(source) {
  if (!source.includes("var<workgroup>")) return source;
  let code = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
  const constants = new Map();
  for (let round = 0; round < 6; round += 1) {
    for (const [, name, text] of code.matchAll(/\bconst\s+(\w+)\s*(?::\s*\w+\s*)?=\s*([^;]+);/g)) {
      const expression = text.replace(/\b([A-Za-z_]\w*)\b/g, (word) => (constants.has(word) ? String(constants.get(word)) : word))
        .replace(/(\d)u\b/g, "$1");
      if (!/^[\d\s+\-*()]+$/.test(expression) || !/\d/.test(expression)) continue;
      try { constants.set(name, Function(`"use strict"; return (${expression});`)()); } catch { /* not an integer */ }
    }
  }
  const count = (text) => {
    const expression = text.replace(/\b([A-Za-z_]\w*)\b/g, (word) => (constants.has(word) ? String(constants.get(word)) : word))
      .replace(/(\d)u\b/g, "$1");
    if (!/^[\d\s+\-*()]+$/.test(expression) || !/\d/.test(expression)) return null;
    try { return Function(`"use strict"; return (${expression});`)(); } catch { return null; }
  };
  for (const declaration of [...code.matchAll(/var<workgroup>\s+(\w+)\s*:\s*array<\s*([\w<>]+?)\s*,\s*([^;]+?)>\s*;/g)]) {
    const [whole, name, type, countText] = declaration;
    const bytes = ELEMENT_BYTES.get(type);
    const elements = count(countText);
    if (bytes === undefined || !Number.isSafeInteger(elements) || elements * bytes <= WORKGROUP_ARRAY_BYTES) continue;
    const part = Math.floor(WORKGROUP_ARRAY_BYTES / bytes);
    const parts = Math.ceil(elements / part);
    // every use, last first, so an index that itself reads the array is rewritten inside the outer one
    let text = code;
    let ok = !new RegExp(`&\\s*${name}\\b`).test(text);
    const uses = [...text.matchAll(new RegExp(`(?<![\\w.])${name}\\s*\\[`, "g"))].map((m) => m.index).reverse();
    const bare = [...text.matchAll(new RegExp(`(?<![\\w.])${name}\\b(?!\\s*[\\[:])`, "g"))];
    if (bare.length > 0) ok = false;
    for (const at of uses) {
      if (!ok) break;
      const open = text.indexOf("[", at);
      const close = closingBracket(text, open);
      if (close < 0) { ok = false; break; }
      const index = text.slice(open + 1, close);
      const rest = text.slice(close + 1);
      const assigned = /^\s*(\+|-|\*|\/)?=(?!=)/.exec(rest);
      if (/^\s*\.\s*\w+\s*(\+|-|\*|\/)?=(?!=)/.test(rest)) { ok = false; break; }
      if (assigned === null) {
        text = `${text.slice(0, at)}${name}_get(u32(${index}))${rest}`;
        continue;
      }
      const end = rest.indexOf(";");
      if (end < 0) { ok = false; break; }
      const value = rest.slice(assigned[0].length, end).trim();
      const stored = assigned[1] === undefined ? value : `${name}_get(u32(${index})) ${assigned[1]} (${value})`;
      text = `${text.slice(0, at)}${name}_set(u32(${index}), ${stored})${rest.slice(end)}`;
    }
    if (!ok) continue;
    const lines = [];
    for (let k = 0; k < parts; k += 1) {
      lines.push(`var<workgroup> ${name}_p${k}: array<${type}, ${Math.min(part, elements - k * part)}>;`);
    }
    const pick = (use) => {
      const out = [];
      for (let k = 0; k < parts - 1; k += 1) out.push(`  if (i < ${(k + 1) * part}u) { ${use(k, k === 0 ? "i" : `i - ${k * part}u`)} }`);
      out.push(`  ${use(parts - 1, `i - ${(parts - 1) * part}u`)}`);
      return out.join("\n");
    };
    lines.push(`fn ${name}_get(i: u32) -> ${type} {\n${pick((k, at) => `return ${name}_p${k}[${at}];`)}\n}`);
    lines.push(`fn ${name}_set(i: u32, v: ${type}) {\n${pick((k, at) => `${name}_p${k}[${at}] = v; return;`)}\n}`);
    code = text.replace(whole, lines.join("\n"));
  }
  return code;
}

/**
 * The text this device's compiler is handed for a shader this repository wrote: both rewrites above, each only where
 * the device wants it. webgpu/runtime/pipeline-cache.js compiles through it, and so must anything that compiles a
 * kernel to TIME it (webgpu/af3/trunk/grid-attend-race.js), or it times a kernel the fold does not run.
 */
export function compilerText(device, source) {
  const text = f16WidthBitcastAvailable(device) ? source : withoutF16WidthBitcasts(source);
  // (the loop rewrite stands for both of naga's costs: the arrays first, then the loops over what is left)
  return loopsUnrolledFor(device) ? withUnrolledLoops(withSmallWorkgroupArrays(text)) : text;
}
