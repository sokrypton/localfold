import { readFileSync } from "node:fs";
import { describe, expect, it } from "./harness.js";
import { buildBatch, decodeTextOrZstd } from "../shared/rfd4p/pipeline/index.js";

/**
 * Upstream's own contig checks (`inference_utils/contig_spec.py`), which the page's pipeline did
 * not have: each of these jobs was built silently into a structure its author did not write. The
 * native featuriser carries the same checks and cuda/rfd4p/gate.py's `refusals` arm holds it to
 * the same sentences.
 */
describe("the contig checks upstream makes", () => {
  const repo = new URL("..", import.meta.url).pathname;
  const ppi = decodeTextOrZstd(new Uint8Array(readFileSync(`${repo}web/examples/ppi_pdl1.cif.zst`)));
  const refuses = (spec, words, inputCif = ppi) => {
    let message = "";
    try {
      buildBatch(spec, { seed: 0, ...(inputCif ? { inputCif } : {}) });
    } catch (e) {
      message = e.message;
    }
    expect(message.includes(words) ? words : `no "${words}" in "${message}"`).toBe(words);
  };
  it("refuses overlapping preserved segments", () => refuses({ contig: "B2-10,B5-20" }, "overlap"));
  it("refuses a range past its chain", () => refuses({ contig: "B2-400" }, "runs outside chain"));
  it("refuses a duplicated segment name", () =>
    refuses({ contig: "a:5-5,a:6-6" }, "Duplicate expandable-segment name", null));
  it("refuses an input with nothing preserved", () =>
    refuses({ contig: "50-50" }, "no preserved segments"));
  it("refuses a suffix that disagrees with its anchor", () =>
    refuses({ contig: "B2-40,5-5D" }, "disagrees"));
  it("still takes a pure contig with no input", () => {
    expect(buildBatch({ contig: "50-50" }, { seed: 0 }).n).toBe(50);
  });
});

/**
 * And upstream's VALUE checks (`condition_spec._apply_conditions`), the same jobs the native
 * featuriser's refusals arm holds it to.
 */
describe("the condition values upstream refuses", () => {
  const repo = new URL("..", import.meta.url).pathname;
  const ppi = decodeTextOrZstd(new Uint8Array(readFileSync(`${repo}web/examples/ppi_pdl1.cif.zst`)));
  const refuses = (conditions, words) => {
    let message = "";
    try {
      buildBatch({ contig: "B2-40", conditions }, { seed: 0, inputCif: ppi });
    } catch (e) {
      message = e.message;
    }
    expect(message.toLowerCase().includes(words.toLowerCase()) ? words : `no "${words}" in "${message}"`).toBe(words);
  };
  const B = [{ select: "sel('B')" }];
  it("a literal residue name for C_SEQ", () => refuses({ C_SEQ: { ALA: B } }, "expected boolean"));
  it("a centroid of four numbers", () => refuses({ C_CENTROID: { "1,2,3,4": B } }, "exactly 3 components"));
  it("a four-level CATH code", () => refuses({ C_CATH: { "3.40.50.20": B } }, "Unknown CAT label"));
  it("a direct C_INTCRD", () => refuses({ C_INTCRD: { true: B } }, "saved payload"));
  it("coordinate group 0", () => refuses({ C_CRDGRP: { 0: B } }, "integer ID from 1"));
  it("a group on anything but C_IDX false", () =>
    refuses({ C_HOT: { true: [{ select: "sel('B')", group: "g" }] } }, "only meaningful for C_IDX"));
  it("an enum integer that is no member", () => refuses({ C_SEC: { 7: B } }, "Invalid enum value"));
  it("a two-body condition", () => refuses({ C_BP: { PAIR: B } }, "2-body condition"));
  it("still takes upstream's own label shapes", () => {
    for (const label of ["3.40.50", "3.40.50.x", "3.40.x", "3.x.x", "x.x.x"]) {
      expect(buildBatch({ contig: "B2-40", conditions: { C_CATH: { [label]: B } } },
        { seed: 0, inputCif: ppi }).n).toBe(39);
    }
  });
});
