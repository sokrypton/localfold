/**
 * What the saved session records, and what its README promises.
 *
 * 🔴 THE STORE ITSELF IS NOT TESTED HERE, because IndexedDB does not exist in
 * node and a fake one would assert against the fake. `web/fold-session.js`'s
 * I/O is gated by `tools/fold-in-page.py --session`, in a real browser against
 * a real quota. What is testable without one is every decision taken before a
 * byte is written - the row, and the three states the README can be in - and
 * those are the ones that were wrong when this was written.
 */
import { describe, expect, it } from "./harness.js";
import { jobMeta } from "../web/fold-session.js";
import { buildFoldArchive } from "../web/fold-archive.js";

function prediction(chainLengths = [4, 4]) {
  const tokens = chainLengths.reduce((total, length) => total + length, 0);
  return {
    pdb: "END\n",
    chainLengths,
    confidence: {
      meanPlddt: 91.08,
      ptm: 0.84,
      predictedAlignedError: Float32Array.from({ length: tokens * tokens }, () => 1),
      contactProbs: Float32Array.from({ length: tokens * tokens }, () => 0.5),
    },
  };
}

/**
 * 🔴 THE SESSION ITSELF IS py2Dmol'S AND IS NOT REBUILT HERE. What is ours is
 * the `localfold` key beside it - the job - because py2Dmol's frames carry
 * coordinates and maps and nothing that says which model ran against what.
 */
describe("the job behind a saved session", () => {
  it("counts residues across every chain", () => {
    const meta = jobMeta({ stem: "af3_1", model: "AlphaFold 3",
      prediction: prediction([76]), sequence: "M" });
    expect(meta.residues).toBe(76);
    expect(meta.confidence.meanPlddt).toBe(91.08);
    expect(meta.confidence.ptm).toBe(0.84);
  });

  it("adds up a complex", () => {
    const meta = jobMeta({ stem: "af3_1", model: "AlphaFold 3",
      prediction: prediction([58, 76]), sequence: "A:B" });
    expect(meta.residues).toBe(134);
    expect(meta.chainLengths).toEqual([58, 76]);
  });

  /**
   * 🔴 ABSENT, NOT ZERO. EF2-fast carries no `confidence` object at all - on
   * purpose, since an object of zeros reads as the model's opinion - and a row
   * defaulting pLDDT to 0 would offer to restore "a fold scoring 0.0" rather
   * than one with no score. Same mistake the archive was taught not to make in
   * its B-factor column.
   */
  it("leaves a scoreless model's score absent", () => {
    const meta = jobMeta({ stem: "ef2_1", model: "ESMFold2",
      prediction: { chainLengths: [40], pdb: "END\n" }, sequence: "M" });
    // 🔴 THE WHOLE OBJECT IS ABSENT, not an object of undefined fields.
    // EF2-fast stores no `confidence` at all - on purpose, since an object of
    // zeros reads as the model's opinion - and `updateScoresCard` hides its
    // box outright when handed undefined, which is the right answer for a
    // model with no confidence head. A shell of undefined keys would instead
    // draw the card with dashes in it, claiming the fold was scored and the
    // numbers were lost.
    expect(meta.confidence).toBe(undefined);
    expect(meta.residues).toBe(40);
    // ...and the structure still travels, so its PDB button still works.
    expect(meta.pdb).toBe("END\n");
  });

  /**
   * 🔴 THE MATRICES ARE NOT IN THIS RECORD, AND THAT IS THE POINT. py2Dmol's
   * session already carries the PAE on `frame.pae` and the contact map in
   * `frame.maps`, so a float copy here writes every pair twice - and n^2 is
   * the term that grows fastest with chain length. `matricesFromFrames` in
   * web/app.js reads them back; what stays here is the summary, which is not
   * in any frame and is a handful of numbers.
   */
  /**
   * 🔴 THE ALIGNMENT TRAVELS, AND IT IS THE ONE FIELD ALLOWED TO GO. Without it
   * a restored fold can be looked at and not REPRODUCED - a re-search finds
   * different hits - so it is stored; it is also the only field bounded by what
   * a public server returned rather than by the fold, so `rememberSession`
   * drops it and saves again rather than losing the session to one deep MSA.
   */
  it("carries the alignment when it was given one", () => {
    const meta = jobMeta({ stem: "af3_1", model: "AlphaFold 3",
      prediction: prediction([4]), sequence: "ACDE",
      msaOrigin: "MMseqs2 search at api.colabfold.com",
      msas: { unpaired: [">q\nACDE\n"], paired: [undefined] } });
    expect(meta.msas.unpaired[0]).toBe(">q\nACDE\n");
    expect(meta.msaOrigin).toBe("MMseqs2 search at api.colabfold.com");
  });

  // ...and absent stays absent, so a session saved without room for one
  // restores into the archive's omitted state rather than an empty `msas/`.
  it("leaves the alignment absent when there was none", () => {
    const meta = jobMeta({ stem: "ef2_1", model: "ESMFold2",
      prediction: prediction([4]), sequence: "ACDE" });
    expect(meta.msas).toBe(undefined);
  });

  it("keeps the summary and not the matrices", () => {
    const meta = jobMeta({ stem: "af3_1", model: "AlphaFold 3",
      prediction: prediction([4]), sequence: "ACDE" });
    expect(meta.confidence.meanPlddt).toBe(91.08);
    expect(meta.confidence.ptm).toBe(0.84);
    expect(meta.confidence.predictedAlignedError).toBe(undefined);
    expect(meta.confidence.contactProbs).toBe(undefined);
    expect(meta.confidence.plddt).toBe(undefined);
  });

  /**
   * 🔴 AND NOTHING IN THE RECORD MAY BE A TYPED ARRAY, because it is gzipped
   * through JSON: a Float32Array survives structuredClone and does NOT survive
   * JSON.stringify - it returns as `{"0":1.2,...}`, an object with numeric
   * keys that every reader treats as a matrix of undefined. The matrices were
   * the only typed arrays that ever reached here; this is what catches the
   * next one.
   */
  it("holds nothing that JSON would turn into an object of indices", () => {
    const meta = jobMeta({ stem: "af3_1", model: "AlphaFold 3",
      prediction: prediction([4]), sequence: "ACDE" });
    const offenders = [];
    const walk = (value, path) => {
      if (ArrayBuffer.isView(value)) { offenders.push(path); return; }
      if (Array.isArray(value)) value.forEach((v, i) => walk(v, `${path}[${i}]`));
      else if (value && typeof value === "object") {
        for (const [key, v] of Object.entries(value)) walk(v, `${path}.${key}`);
      }
    };
    walk(meta, "meta");
    // ...named, so a failure says which field rather than that one exists.
    expect(offenders).toEqual([]);
    // ...and the whole record survives the trip it actually takes.
    expect(JSON.parse(JSON.stringify(meta)).confidence.ptm).toBe(0.84);
  });
});

/**
 * 🔴 THE README'S THIRD STATE. It had two - "here is `msas/`" when the model
 * takes an alignment, "there is no `msas/`" when it does not - both keyed on
 * `msaOrigin`, which answers *does this model take one* and NOT *does this
 * archive carry one*. A saved session takes one and carries none, so the first
 * branch wrote a paragraph telling the reader to drop the zip on the upload
 * box to reproduce the fold, describing a directory not in the file.
 */
describe("what the README says about an alignment", () => {
  const base = {
    stem: "fold_test", model: "AlphaFold 3", settings: { seed: 0 },
    entities: [{ type: "protein", value: "ACDE", copies: 1 }],
    prediction: prediction([4]),
  };
  const readmeOf = (extra) => buildFoldArchive({ ...base, ...extra }).get("README.md");

  it("describes msas/ when the archive carries one", () => {
    const text = readmeOf({
      msaOrigin: "MMseqs2 search at api.colabfold.com",
      msas: { unpaired: ["a-sequence"] },
    });
    expect(text.includes("`msas/` holds one alignment per chain")).toBe(true);
    expect(text.includes("not in this archive")).toBe(false);
  });

  it("says there is none when the model takes none", () => {
    const text = readmeOf({ msaOrigin: undefined });
    expect(text.includes("folds from the sequence alone")).toBe(true);
    expect(text.includes("`msas/` holds one alignment per chain")).toBe(false);
  });

  it("says the alignment was left out when it was", () => {
    const text = readmeOf({
      msaOrigin: "MMseqs2 search at api.colabfold.com",
      msas: {},
      alignmentOmitted: true,
    });
    // The line under "What ran" still names the origin - the fold DID use one.
    expect(text.includes("- alignment: MMseqs2 search at api.colabfold.com"
      + " (not in this archive)")).toBe(true);
    expect(text.includes("`msas/` holds one alignment per chain")).toBe(false);
    expect(text.includes("folds from the sequence alone")).toBe(false);
    // 🔴 AND IT SAYS WHAT THAT COSTS. "Smaller" is not the point; "a re-search
    // may find different hits, so this will not reproduce" is.
    expect(text.includes("may find different hits")).toBe(true);
  });

  /**
   * 🔴 THE FOURTH STATE, AND EVERY SINGLE-SEQUENCE ARCHIVE GOT IT WRONG. The
   * "here is `msas/`" branch was reached by any model with an alignment
   * CONTROL, so a fold that deliberately used none described a directory that
   * was not in the file and told the reader to drop the zip back "to fold again
   * with exactly these alignments". Whether the archive carries an alignment is
   * a different question from whether the model takes one.
   */
  it("says a single-sequence fold ran on the sequence alone", () => {
    const text = readmeOf({ msaOrigin: "none (single sequence)", msas: {} });
    expect(text.includes("ran on the sequence alone")).toBe(true);
    expect(text.includes("`msas/` holds one alignment per chain")).toBe(false);
    // ...and NOT the omitted caveat: there was nothing to omit, and saying so
    // sends the reader looking for an archive that carries it.
    expect(text.includes("may find different hits")).toBe(false);
    // ...while the line under "What ran" still names what the fold was given.
    expect(text.includes("- alignment: none (single sequence)")).toBe(true);
  });

  /**
   * The state that must not exist: an archive with no `msas/` member whose
   * README tells the reader to drop it on the upload box to reproduce the
   * fold. This is the assertion that would have caught the bug.
   */
  it("never promises msas/ in an archive that has none", () => {
    for (const extra of [
      { msaOrigin: "MMseqs2 search at api.colabfold.com", msas: {}, alignmentOmitted: true },
      { msaOrigin: undefined, msas: {} },
      // ...and the one that was wrong: a model that HAS the control, on a fold
      // that used none.
      { msaOrigin: "none (single sequence)", msas: {} },
      // ...and a caller asking for the alignment branch while passing no a3m,
      // which is how the bug reached every such archive: the flag said one
      // thing and the file held another. What was WRITTEN decides now.
      { msaOrigin: "MMseqs2 search at api.colabfold.com", msas: {} },
    ]) {
      const files = buildFoldArchive({ ...base, ...extra });
      const hasMsas = [...files.keys()].some((name) => name.startsWith("msas/"));
      expect(hasMsas).toBe(false);
      expect(files.get("README.md").includes("`msas/` holds")).toBe(false);
    }
  });
});
