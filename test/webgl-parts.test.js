import test from "node:test";
import assert from "node:assert/strict";
import { assembleParts, part } from "../webgl/runtime/execution.js";

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
