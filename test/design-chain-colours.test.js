import { describe, expect, it } from "./harness.js";
import {
  CHAIN_COLOURS, chainColour, chainOfSpans, chainsOf, shade,
} from "../web/contig-ui.js";

/** A span list, written the short way the page's own rows are. */
const design = (min, max = min) => ({ kind: "generated", min, max });
const breaks = () => ({ kind: "break" });
const ligand = (code) => ({ kind: "ligand", code });
const sequence = (text) => ({ kind: "sequence", text });

describe("one colour per chain, on the row and in the structure", () => {
  // 🔴 THE TWO WALKS MUST AGREE OR THE INDICATOR IS WORSE THAN NOTHING. The row
  // takes its colour from `chainOfSpans` and the viewer paints `chainsOf`'s
  // chains; if they counted chains differently the swatch would name one chain
  // and the structure another, and a reader would have no way to tell. Nothing
  // else compares them, because they are used from different modules.
  it("counts the same chains as chainsOf", () => {
    for (const spans of [
      [design(40)],
      [design(40), breaks(), sequence("ACDEFGHIK")],
      [sequence("ACDEFGHIK"), breaks(), design(40), breaks(), sequence("MNPQRST")],
      [design(40), ligand("SAM")],
      [design(40), breaks(), ligand("SAM"), breaks(), sequence("ACDEFGHIK")],
    ]) {
      const { chains } = chainsOf(spans, { sequenceOf: () => null });
      const indices = chainOfSpans(spans).filter((at) => at !== null);
      const distinct = new Set(indices);
      expect(distinct.size).toBe(chains.length);
      // ...and they are 0..n-1, so the colour of chain k is chainColour(k) on
      // both sides rather than merely "some consistent mapping".
      expect([...distinct].sort((a, b) => a - b)).toEqual([...chains.keys()]);
    }
  });

  // 🔴 A LIGAND AND A BREAK BELONG TO NO CHAIN, so they take no chain colour.
  // Giving a break the next chain's colour would announce a chain before its
  // first span exists, and `chainsOf` does not count one until a part arrives.
  it("gives a break and a ligand no chain", () => {
    const spans = [design(40), breaks(), ligand("SAM"), sequence("ACDEF")];
    expect(chainOfSpans(spans)).toEqual([0, null, null, 1]);
  });

  // 🔴 AN EMPTY OR DOUBLED BREAK MUST NOT SKIP A COLOUR. `chainsOf` starts a
  // chain when a part arrives, so a leading break is not a chain and two in a
  // row are not two - and a walk that counted breaks instead of parts would
  // shift every colour after them by one.
  it("does not advance on a break with nothing in front of it", () => {
    expect(chainOfSpans([breaks(), design(40)])).toEqual([null, 0]);
    expect(chainOfSpans([design(40), breaks(), breaks(), design(20)]))
      .toEqual([0, null, null, 1]);
  });

  // The palette wraps rather than running out or returning undefined, which a
  // 7-chain job would otherwise paint `undefined` and the CSS would drop.
  it("wraps past the end of the palette", () => {
    expect(chainColour(CHAIN_COLOURS.length)).toBe(CHAIN_COLOURS[0]);
    for (let at = 0; at < 30; at += 1) {
      expect(/^#[0-9a-f]{6}$/.test(chainColour(at))).toBe(true);
    }
  });

  // 🔴 A ROLE IS A TREATMENT OF THE CHAIN'S OWN COLOUR, so the blend must keep
  // the hue recognisable at the ends and be exact at both.
  it("blends toward a colour without losing the hue", () => {
    expect(shade("#2563eb", "#ffffff", 0)).toBe("#2563eb");
    expect(shade("#2563eb", "#ffffff", 1)).toBe("#ffffff");
    const faded = shade("#2563eb", "#ffffff", 0.6);
    const [r, g, b] = [1, 3, 5].map((at) => Number.parseInt(faded.slice(at, at + 2), 16));
    // still blue: the blue channel leads, and it is lighter than it was
    expect(b > r && b > g).toBe(true);
    expect(r > 0x25).toBe(true);
  });
});
