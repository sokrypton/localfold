// The rewrite a compiler without the f16 width bitcasts is given (webgpu/runtime/wgsl-compat.js): Firefox's naga
// refuses them, and three models' packed-f16 kernels use them (issue #10).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { withoutF16WidthBitcasts } from "../webgpu/runtime/wgsl-compat.js";

test("the three width-changing spellings are rewritten, nested and with indexed arguments", () => {
  const out = withoutF16WidthBitcasts(`enable f16;
fn f() {
  let a = bitcast<vec4<f16>>(key[i + g(1u)]);
  let b = vec4<f16>(bitcast<vec2<f16>>(u.x), bitcast<vec2<f16>>(b[bOff + 2u * s]));
  o[0] = vec2<u32>(bitcast<u32>(vec2<f16>(v.xy)), bitcast<u32>(vec2<f16>(v.zw)));
}`);
  assert.ok(!/bitcast<vec[24]<f16>>/.test(out));
  assert.ok(!/bitcast<u32>\(\s*vec2<f16>/.test(out));
  assert.ok(out.includes("lf_h4_from_words(key[i + g(1u)])"));
  assert.ok(out.includes("vec2<f16>(unpack2x16float(b[bOff + 2u * s]))"));
  assert.ok(out.includes("lf_word_from_h2(vec2<f16>(v.zw))"));
  // the helpers follow the directive, which WGSL wants first
  assert.ok(out.indexOf("enable f16;") < out.indexOf("fn lf_h4_from_words"));
  assert.ok(out.indexOf("fn lf_h4_from_words") < out.indexOf("fn f()"));
});

test("a bitcast of one width is left alone, and so is a shader with none", () => {
  const same = "fn g() { let c = bitcast<u32>(x); let d = bitcast<f32>(c); }";
  assert.equal(withoutF16WidthBitcasts(same), same);
  assert.equal(withoutF16WidthBitcasts("fn h() {}"), "fn h() {}");
});

test("no kernel source spells a width-changing f16 bitcast the rewrite does not cover", () => {
  const files = [];
  const walk = (directory) => {
    for (const name of readdirSync(directory)) {
      const path = join(directory, name);
      if (statSync(path).isDirectory()) walk(path);
      else if (name.endsWith(".js")) files.push(path);
    }
  };
  walk(new URL("../webgpu", import.meta.url).pathname);
  const uncovered = [];
  for (const file of files) {
    if (file.endsWith("wgsl-compat.js")) continue;
    const text = readFileSync(file, "utf8");
    // vec2<u32> or f32 built straight from f16 vectors: a fourth and fifth spelling naga refuses too
    for (const match of text.matchAll(/bitcast<(vec2<u32>|f32)>\(\s*vec[24]<f16>/g)) uncovered.push(`${file}: ${match[0]}`);
  }
  assert.deepEqual(uncovered, []);
});

// withUnrolledLoops: Firefox's compiler pays for every loop iteration (tools/gpu/probe-loop-cost.js), so the short
// constant-count loops are unrolled for it.
const { withUnrolledLoops } = await import("../webgpu/runtime/wgsl-compat.js");

test("a short constant-count loop is unrolled eight to an iteration, the remainder as straight code", () => {
  const out = withUnrolledLoops(`const K: u32 = 19u;
const C: u32 = 4u;
fn f() {
  var acc = 0.0;
  for (var i = 0u; i < K; i += 1u) {
    acc += wt[i] * w[base + i * C];   // one read each
  }
}`);
  // sixteen in the loop, three after it, and the loop variable is the body's own `let`
  assert.match(out, /for \(var i_lf = 0u; i_lf < 16u; i_lf \+= 8u\)/);
  assert.equal(out.match(/acc \+= wt\[i\] \* w\[base \+ i \* C\];/g).length, 8 + 3);
  for (const value of ["i_lf", "i_lf + 7u", "16u", "17u", "18u"]) assert.ok(out.includes(`{ let i = ${value};`), value);
  assert.ok(!out.includes("19u;\n{ let i") && !out.includes("let i = 19u"));
  assert.ok(!out.includes("// one read each"));
});

test("a stride, a start and an i++ are carried, and a loop of eight or fewer has no loop left", () => {
  const strided = withUnrolledLoops("fn f() { for (var d = 2u; d < 12u; d += 4u) { s += x[d]; } }");
  assert.ok(!/for\s*\(/.test(strided));
  for (const value of ["2u", "6u", "10u"]) assert.ok(strided.includes(`{ let d = ${value};`), value);
  const signed = withUnrolledLoops("fn f() { for (var t = 0; t < 3; t++) { s += x[t]; } }");
  for (const value of ["0i", "1i", "2i"]) assert.ok(signed.includes(`{ let t = ${value};`), value);
});

test("only the innermost loop is unrolled, and a constant built from constants is resolved", () => {
  const out = withUnrolledLoops(`const A: u32 = 4u;
const B: u32 = A * 6u;
fn f() {
  for (var chunk = 0u; chunk < 48u; chunk += B) {
    for (var i = 0u; i < B; i += 1u) { acc += x[chunk + i]; }
  }
}`);
  assert.match(out, /for \(var chunk = 0u; chunk < 48u; chunk \+= B\)/);
  assert.match(out, /for \(var i_lf = 0u; i_lf < 24u; i_lf \+= 8u\)/);
});

test("a loop that stages, leaves early, moves its own counter or has a runtime count is left as written", () => {
  for (const body of [
    "for (var i = local; i < 384u; i += 256u) { wt[i] = x[i]; }",                       // a lane-strided start
    "for (var i = 0u; i < n; i += 1u) { s += x[i]; }",                                    // a runtime bound
    "for (var i = 0u; i < 64u; i += 1u) { workgroupBarrier(); s += x[i]; }",              // a barrier
    "for (var i = 0u; i < 64u; i += 1u) { if (x[i] < 0.0) { break; } s += x[i]; }",       // leaves early
    "for (var i = 0u; i < 64u; i += 1u) { s += x[i]; i += 1u; }",                         // moves its counter
  ]) {
    const code = `fn f() { ${body} }`;
    assert.equal(withUnrolledLoops(code), code, body);
  }
  // a name declared twice is not a constant to trust
  const shadowed = "const N: u32 = 8u;\nfn f() { let N = k; for (var i = 0u; i < N; i += 1u) { s += x[i]; } }";
  assert.equal(withUnrolledLoops(shadowed), shadowed);
  // ...and a long body is a tile, where the iteration's own cost is nothing
  const long = `fn f() { for (var i = 0u; i < 64u; i += 1u) { ${"s += x[i] * y[i] + z[i];\n".repeat(60)} } }`;
  assert.equal(withUnrolledLoops(long), long);
});

// withSmallWorkgroupArrays: in Firefox a workgroup array over 8 KiB costs every workgroup before it runs
// (tools/gpu/probe-workgroup-init.js), so such an array is cut into parts behind a getter and a setter.
const { withSmallWorkgroupArrays, WORKGROUP_ARRAY_BYTES } = await import("../webgpu/runtime/wgsl-compat.js");

test("a workgroup array over 8 KiB is cut into parts, and every use goes through its getter or setter", () => {
  const out = withSmallWorkgroupArrays(`const ROWS: u32 = 8u;
const CH: u32 = 260u;
var<workgroup> tile: array<f32, ROWS * CH>;   // 8320 bytes
var<workgroup> small: array<f32, 64>;
fn main() {
  tile[slot * CH + c] = source[c];
  tile[i] += small[tile_index[i]];
  let value = tile[slot * CH + c] - tile[idx[tile[0u] > 0.0]];
  small[lane] = value;
}`);
  assert.ok(out.includes("var<workgroup> tile_p0: array<f32, 2048>;"));
  assert.ok(out.includes("var<workgroup> tile_p1: array<f32, 32>;"));
  assert.ok(!/var<workgroup> tile:/.test(out));
  assert.ok(out.includes("tile_set(u32(slot * CH + c), source[c]);"));
  assert.ok(out.includes("tile_set(u32(i), tile_get(u32(i)) + (small[tile_index[i]]));"));
  assert.ok(out.includes("let value = tile_get(u32(slot * CH + c)) - tile_get(u32(idx[tile_get(u32(0u)) > 0.0]));"));
  // a name the array's is only the start of, and the small array, are untouched
  assert.ok(out.includes("small[tile_index[i]]") && out.includes("small[lane] = value;"));
  assert.ok(!/(?<![\w.])tile\s*\[/.test(out));
  assert.match(out, /fn tile_get\(i: u32\) -> f32 \{\n  if \(i < 2048u\) \{ return tile_p0\[i\]; \}\n  return tile_p1\[i - 2048u\];\n\}/);
});

test("the parts follow the element: 8 KiB of vec4<f16> is 1024 elements, and three parts chain", () => {
  const out = withSmallWorkgroupArrays("var<workgroup> gated: array<vec4<f16>, 3072>;\nfn f() { gated[k] = v; let x = gated[k].x; }");
  for (const part of ["gated_p0: array<vec4<f16>, 1024>", "gated_p1: array<vec4<f16>, 1024>", "gated_p2: array<vec4<f16>, 1024>"]) {
    assert.ok(out.includes(part), part);
  }
  assert.ok(out.includes("if (i < 2048u) { return gated_p1[i - 1024u]; }"));
  assert.ok(out.includes("gated_set(u32(k), v);") && out.includes("let x = gated_get(u32(k)).x;"));
});

test("an array at the limit, one of an unknown size and one used in a way that is not rewritten are left whole", () => {
  assert.equal(WORKGROUP_ARRAY_BYTES, 8192);
  for (const code of [
    "var<workgroup> tile: array<f32, 2048>;\nfn f() { tile[i] = 1.0; }",
    "var<workgroup> tile: array<f32, N>;\nfn f() { tile[i] = 1.0; }",
    "var<workgroup> tile: array<vec4<f32>, 1024>;\nfn f() { tile[i].x = 1.0; }",
    "var<workgroup> tile: array<f32, 4096>;\nfn f() { g(&tile); }",
    "var<workgroup> tile: array<Thing, 4096>;\nfn f() { tile[i] = t; }",
  ]) assert.equal(withSmallWorkgroupArrays(code), code, code);
});
