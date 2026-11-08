import test from "node:test";
import assert from "node:assert/strict";
import {
  assembleParts, part, partFromSource, sized, targetDeclarations, ONE_TARGET, PRELUDE,
} from "../webgl/runtime/execution.js";
import { ipaSoftmaxSource, transformPointsSource } from "../webgl/af2/structure/ipa.js";
import { atom37Source } from "../webgl/af2/structure/geometry.js";
import { contractSource } from "../webgl/kernels/triangle-contract.js";

const add = part({ ELEMENTS: 8 }, { uniforms: ["sampler2D aTex", "sampler2D bTex"], body: "  result = vec4(1.0);" });
const scale = part({ ELEMENTS: 8, C4: 2 }, { uniforms: ["sampler2D aTex", "float scale"], helpers: "float scaleHelper() { return scale; }", body: "  result = vec4(scaleHelper());" });

test("a part on its own is one main, with its sizes declared", () => {
  assert.match(add.code, /^uniform highp int ELEMENTS;$/m);
  assert.match(add.code, /^void main\(\) \{\n {2}result = vec4\(1\.0\);\n\}$/m);
  assert.doesNotMatch(add.code, /KERNEL/);
});

test("parts together: shared uniforms once, a function each, main choosing by KERNEL", () => {
  const code = assembleParts([add.part, scale.part]);
  assert.equal(code.match(/^uniform sampler2D aTex;$/gm).length, 1);
  assert.equal(code.match(/^uniform highp int ELEMENTS;$/gm).length, 1);
  assert.match(code, /^uniform highp int KERNEL;$/m);
  assert.match(code, /^void kernel0\(\) \{/m);
  assert.match(code, /^void kernel1\(\) \{/m);
  assert.match(code, /if \(KERNEL == 0\) kernel0\(\);\n {2}else if \(KERNEL == 1\) kernel1\(\);/);
  assert.match(code, /float scaleHelper\(\)/);
});

test("a uniform declared two ways cannot join", () => {
  const other = part({ ELEMENTS: 8 }, { uniforms: ["usampler2D aTex"], body: "  result = vec4(0.0);" });
  assert.throws(() => assembleParts([add.part, other.part]), /aTex is declared both/);
});

// `partFromSource`: a finished kernel read back as a part, which is how the 150-odd kernels in
// this tree that were never written as parts come to share a program.
const finished = (declarations, body) => sized({ ROWS: 4 },
  `${PRELUDE}${ONE_TARGET}${declarations}\nvoid main() {\n${body}\n}`);

test("a finished kernel becomes a part, its own names renamed apart", () => {
  const piece = partFromSource(finished(
    "uniform sampler2D xTex;\nconst int CH4 = 32;\nfloat twice(float v) { return v + v; }",
    "  result = vec4(twice(fetch4(xTex, linearTexel() * CH4).x));").code);
  // `sized` injects its own after the version line, so a read-back part declares them first.
  assert.deepEqual(piece.uniforms, ["highp int ROWS", "sampler2D xTex"]);
  // The const and the helper keep module scope and lose their names, which is what lets a second
  // kernel declare `CH4` too; the uniform keeps its, which is what lets the two share a unit.
  const [, prefix] = piece.helpers.match(/const int (p\w+_)CH4 = 32;/);
  assert.match(piece.helpers, new RegExp(`float ${prefix}twice\\(float v\\)`));
  assert.match(piece.body, new RegExp(`${prefix}twice\\(fetch4\\(xTex, linearTexel\\(\\) \\* ${prefix}CH4\\)`));
});

test("two finished kernels that both declare CH4 share one program", () => {
  const ours = (width, body) => partFromSource(finished(
    `uniform sampler2D xTex;\nconst int CH4 = ${width};`, body).code);
  const code = assembleParts([
    ours(32, "  result = vec4(float(CH4));"),
    ours(8, "  result = vec4(float(CH4) * 2.0);"),
  ]);
  assert.equal(code.match(/^uniform sampler2D xTex;$/gm).length, 1);
  assert.equal(code.match(/^const int p\w+_CH4 = (32|8);$/gm).length, 2);
});

test("a field is not a name: `.x` survives a rename", () => {
  const piece = partFromSource(finished("const int x = 3;",
    "  vec4 v = vec4(1.0);\n  result = vec4(v.x * float(x));").code);
  assert.match(piece.body, /v\.x \* float\(p\w+_x\)/);
});

test("a kernel the runtime will not read as a part is refused", () => {
  // Several render targets: a part writes `result`.
  assert.equal(partFromSource(`${PRELUDE}${targetDeclarations(2)}
void main() {
  result0 = vec4(0.0);
  result1 = vec4(1.0);
}`), null);
  // A struct at module scope, which is neither a uniform, a constant nor a function.
  assert.equal(partFromSource(finished("struct Frame { vec3 t; };",
    "  result = vec4(0.0);").code), null);
  // Something that is not one of PRELUDE's kernels at all.
  assert.equal(partFromSource("#version 300 es\nvoid main() {}"), null);
});

// 🔴 AND THE RENAME IS CHECKED AGAINST REAL KERNELS, NOT ONLY MADE-UP ONES. Stripping the digest
// prefix back out of the assembled program must give the kernel it came from, line for line: that
// is what says the conversion is a RENAME and not a rewrite. Four kernels from three models, two
// of them with helper functions and one a 2048-element table.
test("a real kernel, converted and stripped back, is the kernel it came from", () => {
  const sources = [
    transformPointsSource(59, 12, 4),
    contractSource(59, 128, "outgoing"),
    atom37Source(59),
    ipaSoftmaxSource(59, 12),
  ];
  for (const source of sources) {
    // A `sized` kernel hands back {code, sizes}; the older ones are the text itself.
    const code = typeof source === "string" ? source : source.code;
    const piece = partFromSource(code);
    assert.notEqual(piece, null);
    // Comments excepted: parsing the declarations takes them out, which no compiler can see.
    const lines = (text) => text.replace(/p[0-9a-z]{4,}_/g, "")
      .split("\n").map((line) => line.trim())
      .filter((line) => line !== "" && !line.startsWith("//")).sort();
    assert.deepEqual(lines(assembleParts([piece])), lines(code));
  }
});
