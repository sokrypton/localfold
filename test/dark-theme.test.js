/**
 * The dark theme's two promises: every token is defined in both themes, and the generated
 * sheet is what the vendored one generates. Re-vendoring py2Dmol without re-running
 * tools/dark-theme.py leaves white boxes on a dark page and nothing else says so.
 */
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";

const source = (path) => readFileSync(new URL(`../${path}`, import.meta.url), "utf8");
const css = source("web/localfold.css");

/** The custom properties a `selector { ... }` block sets, by name. */
function propertiesOf(selector) {
  const at = css.indexOf(`${selector} {`);
  assert.notEqual(at, -1, `${selector} is missing`);
  const body = css.slice(at, css.indexOf("}", at));
  return new Set([...body.matchAll(/(--lf-[a-z0-9-]+)\s*:/g)].map((match) => match[1]));
}

describe("the dark theme", () => {
  it("defines every token in both themes", () => {
    const light = propertiesOf(":root");
    const dark = propertiesOf(':root[data-theme="dark"]');
    assert.deepEqual([...light].sort(), [...dark].sort());
  });

  it("uses only tokens that exist", () => {
    const defined = propertiesOf(":root");
    for (const file of [css, source("web/localfold-dark.css"), source("index.html")]) {
      for (const [, name] of file.matchAll(/var\((--lf-[a-z0-9-]+)\)/g)) {
        assert.ok(defined.has(name), `${name} is used and never defined`);
      }
    }
  });

  it("sets the theme before the stylesheet can paint the wrong one", () => {
    const page = source("index.html");
    assert.ok(page.indexOf("localfold.theme") < page.indexOf("<body"));
  });

  it("is generated from the py2Dmol sheet this tree vendors", () => {
    const vendored = source("web/vendor/py2Dmol.app.css");
    const generated = source("web/localfold-dark.css");
    // Every white surface py2Dmol hard-codes is in the generated sheet under its own selector.
    const bare = [...vendored.replace(/\/\*[\s\S]*?\*\//g, "").matchAll(/([^{}]+)\{[^{}]*background:\s*#(?:fff|ffffff)\s*;/g)]
      .map((match) => match[1].trim().split(",")[0].trim().replace(/\s+/g, " "));
    assert.ok(bare.length > 5);
    for (const selector of bare) {
      if (selector.startsWith("@")) continue;
      assert.ok(generated.includes(selector), `${selector} has no dark rule: run tools/dark-theme.py`);
    }
  });
});
