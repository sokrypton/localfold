import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "./harness.js";

const read = (name) => readFileSync(fileURLToPath(new URL(name, import.meta.url)), "utf8");

/**
 * 🔴 THE ROLE COLOURS LIVE IN TWO PLACES AND HAVE TO AGREE, which is why this
 * exists. The legend's swatches are CSS classes; the 3D view reaches a WebGL
 * renderer that cannot read a CSS class, so `web/design.js` writes the same
 * values out as literals. The whole point of the role colours is that a residue
 * picked in one view is recognisable in the other - if the two drift, the page
 * is actively lying about what is being decided, and nothing else would catch
 * it because both halves keep working.
 *
 * Neither file can be imported here: one is CSS and the other reaches for
 * `document` while loading. So both are read as text, which is what
 * test/designers.test.js does with the checkpoint list for the same reason.
 */
describe("the design page's residue-role colours", () => {
  const css = read("../web/localfold.css");
  const js = read("../web/design.js");

  const fromCss = Object.fromEntries(
    [...css.matchAll(/^\.role-([a-z]+)\s*\{\s*background:\s*(#[0-9a-fA-F]{3,6});/gm)]
      .map(([, role, colour]) => [role, colour.toLowerCase()]),
  );
  const fromJs = Object.fromEntries(
    [...js.matchAll(/^\s{2}([a-z]+):\s*"(#[0-9a-fA-F]{3,6})",$/gm)]
      .map(([, role, colour]) => [role, colour.toLowerCase()]),
  );

  it("found both lists, so the patterns above still match the files", () => {
    expect(Object.keys(fromCss).length).toBe(5);
    expect(Object.keys(fromJs).length).toBe(5);
  });

  it("names the same five roles in the stylesheet and in the renderer", () => {
    expect(Object.keys(fromCss).sort()).toEqual(["fixed", "free", "motif", "target", "tied"]);
    expect(Object.keys(fromJs).sort()).toEqual(Object.keys(fromCss).sort());
  });

  it("gives every role the same colour in both", () => {
    // 🔴 `#fff` and `#ffffff` are the same colour and must not fail this; the
    // free role is written short in the stylesheet and long in the renderer.
    const expand = (hex) => (hex.length === 4
      ? `#${hex[1]}${hex[1]}${hex[2]}${hex[2]}${hex[3]}${hex[3]}` : hex);
    for (const [role, colour] of Object.entries(fromCss)) {
      expect(expand(fromJs[role])).toBe(expand(colour));
    }
  });

  it("is named by the legend the page actually shows", () => {
    const html = read("../design.html");
    for (const role of Object.keys(fromCss)) expect(html).toContain(`role-${role}`);
  });
});
