import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "./harness.js";
import { CHAIN_COLOURS, LIGAND_COLOUR } from "../web/contig-ui.js";

const read = (name) => readFileSync(fileURLToPath(new URL(name, import.meta.url)), "utf8");

/**
 * 🔴 THE COLOURS HAVE ONE SOURCE NOW, AND THIS IS WHAT KEEPS IT THAT WAY.
 *
 * This file used to compare two hand-written lists - `.role-*` rules in the
 * stylesheet against a `ROLE_COLOURS` literal in `web/design.js` - because the
 * legend was CSS classes and the 3D view reaches a WebGL renderer that cannot
 * read a CSS class. That duplication is gone: the palette lives in
 * `web/contig-ui.js`, the span rows and the legend take it through a
 * `--chain-colour` custom property, and the viewer takes it through the same
 * export. One list, so there is nothing left to drift.
 *
 * What is worth gating is that nobody writes the second copy again, which is
 * how the original bug class would come back. A page that hard-codes a swatch
 * colour is a page whose legend can disagree with its own picture.
 */
describe("the design page's colours come from one place", () => {
  const css = read("../web/localfold.css");
  const js = read("../web/design.js");

  it("has no per-role background rules left in the stylesheet", () => {
    // 🔴 `.role-free { background: #fff }` and its four siblings are what the
    // renderer's literal had to be kept in step with. The swatch takes
    // `var(--chain-colour)` now and the colour is set from the job.
    expect(/^\.role-(free|fixed|tied|motif|target)\s*\{/m.test(css)).toBe(false);
    expect(css).toContain("--chain-colour");
  });

  it("has no second copy of the palette in the page", () => {
    // Any six-digit literal in design.js that is one of the chain colours would
    // be a hard-coded duplicate of the palette.
    for (const colour of [...CHAIN_COLOURS, LIGAND_COLOUR]) {
      expect(js.includes(colour)).toBe(false);
    }
  });

  it("builds the legend rather than writing it out", () => {
    const html = read("../design.html");
    // The one element the page builds into; no `role-free`/`role-target`
    // classes, which were the hard-coded five.
    expect(html).toContain('id="role-list"');
    expect(/role-(free|fixed|tied|motif|target)/.test(html)).toBe(false);
  });

  // 🔴 AND THE PALETTE HAS TO BE READABLE AS COLOURS, because it is written
  // straight into a style property and an invalid one is dropped silently by
  // the CSS parser - a swatch that is simply absent rather than wrong.
  it("is all well-formed hex", () => {
    for (const colour of [...CHAIN_COLOURS, LIGAND_COLOUR]) {
      expect(/^#[0-9a-f]{6}$/.test(colour)).toBe(true);
    }
    expect(new Set(CHAIN_COLOURS).size).toBe(CHAIN_COLOURS.length);
  });
});
