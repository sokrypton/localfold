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
