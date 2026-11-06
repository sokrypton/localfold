import test from "node:test";
import assert from "node:assert/strict";
import {
  assembleParts, part, partFromSource, sized, targetDeclarations, ONE_TARGET, PRELUDE,
} from "../webgl/runtime/execution.js";

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

test("a finished kernel becomes a part, its widths moved into the body", () => {
  const piece = partFromSource(finished("uniform sampler2D xTex;\nconst int CH4 = 32;",
    "  result = fetch4(xTex, linearTexel() * CH4);").code);
  // `sized` injects its own after the version line, so a read-back part declares them first.
  assert.deepEqual(piece.uniforms, ["highp int ROWS", "sampler2D xTex"]);
  // The const is a LOCAL one now: still compile-time, and no longer a name two parts share.
  assert.match(piece.body, /^ {2}const int CH4 = 32;$/m);
  assert.match(piece.body, /result = fetch4\(xTex, linearTexel\(\) \* CH4\);/);
  // And it assembles beside another part without either losing a declaration.
  const code = assembleParts([add.part, piece]);
  assert.equal(code.match(/^uniform highp int ROWS;$/gm).length, 1);
  assert.match(code, /^void kernel1\(\) \{\n {2}const int CH4 = 32;$/m);
});

test("a kernel the runtime will not read as a part says which rule it broke", () => {
  // A helper would have to be renamed: parts share one namespace.
  assert.equal(partFromSource(finished("float twice(float v) { return v + v; }",
    "  result = vec4(twice(1.0));").code), null);
  // Several render targets: a part writes `result`.
  assert.equal(partFromSource(`${PRELUDE}${targetDeclarations(2)}
void main() {
  result0 = vec4(0.0);
  result1 = vec4(1.0);
}`), null);
  // A uniform array whose length is one of the consts, which moving the const would take away.
  assert.equal(partFromSource(finished("const int BINS = 4;\nuniform float edges[BINS];",
    "  result = vec4(edges[0]);").code), null);
  // Something that is not one of PRELUDE's kernels at all.
  assert.equal(partFromSource("#version 300 es\nvoid main() {}"), null);
});
