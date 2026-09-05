/**
 * Which model the page folds with, and what that name is allowed to mean.
 *
 * 🔴 TWO BUNDLES BUILD AlphaFold 3's GRAPH NOW, and `=== "af3"` stopped meaning
 * what its call sites meant by it the moment the second arrived. Three of them
 * were CAPABILITY checks - ligands, modified residues, nucleic chains - and
 * every one refused those inputs under OpenBind-0-0 with a message naming a
 * capability the model actually has, because OpenBind-0 runs the same featuriser
 * and the same token layout. It is the parameters that differ, not the graph.
 *
 * These are read out of the source rather than run, because the alternative is
 * a browser: web/app.js is a DOM module with top-level side effects and cannot
 * be imported here at all. The end-to-end proof is tools/model-terms.py and a
 * ligand fold driven through the page.
 */
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";

import { AF3_FAMILIES, MODEL_BUNDLES } from "../src/reference/manifests/index.js";

const app = readFileSync(new URL("../web/app.js", import.meta.url), "utf8");
const page = readFileSync(new URL("../index.html", import.meta.url), "utf8");

describe("the AlphaFold 3 families", () => {
  it("lists every bundle that builds that graph, and only those", () => {
    assert.deepEqual([...AF3_FAMILIES].sort(), ["af3", "openbind0"]);
    for (const family of AF3_FAMILIES) {
      assert.ok(family in MODEL_BUNDLES, `${family} has no bundle`);
    }
  });

  it("offers each of them in the model row", () => {
    for (const family of Object.keys(MODEL_BUNDLES)) {
      assert.match(page, new RegExp(`<option value="${family}"`),
        `index.html does not offer ${family}`);
    }
  });
});

describe("what needs an AlphaFold 3 graph", () => {
  // 🔴 THE FAULT THIS PINS. `choice !== "af3"` here read as "is this DeepMind's
  // checkpoint", and the reported symptom was
  // "Ligands need AlphaFold 3; the model is set to openbind0" - a refusal to
  // fold something the selected model handles perfectly well.
  for (const what of ["ligandCount", "modificationCount", "nucleicCount"]) {
    it(`tests the family, not the name, for ${what}`, () => {
      // The condition runs to the end of the line, and it CONTAINS brackets -
      // `!isAf3Family(choice)` - so a `[^)]+` class stops inside the call it is
      // meant to find and reports a correct guard as missing.
      const guard = new RegExp(`if \\(${what} > 0 && (.+)\\) \\{`);
      const found = app.match(guard);
      assert.ok(found !== null, `no guard found for ${what}`);
      assert.match(found[1], /isAf3Family\(/,
        `${what} is gated on something other than isAf3Family: ${found[1]}`);
      assert.doesNotMatch(found[1], /"af3"/,
        `${what} still compares against the literal "af3"`);
    });
  }
});

describe("what a fold is called", () => {
  it("gives every model its own stem, so two folds are told apart", () => {
    // 🔴 THE OBJECT NAME IS THE ONLY PLACE THE MODEL SHOWS ON SCREEN. Both
    // AF3-graph bundles used to produce `af3_N` and both AlphaFold 2 models
    // `prediction_N`, so a page holding one of each showed two objects with the
    // same prefix - and the stem becomes the archive's file names too, so a
    // downloaded `af3_1_model_0.pdb` could have come from either.
    const table = app.match(/const MODEL_STEMS = \{([^}]*)\}/s);
    assert.ok(table !== null, "MODEL_STEMS is not where this test expects");
    const stems = {};
    for (const [, key, value] of table[1].matchAll(/(\w+):\s*"([^"]+)"/g)) {
      stems[key] = value;
    }
    // Every family the page offers has one...
    for (const family of Object.keys(MODEL_BUNDLES)) {
      assert.ok(family in stems, `${family} has no stem`);
    }
    // ...and no two share it, which is the whole point.
    const used = Object.values(stems);
    assert.equal(new Set(used).size, used.length,
      `two models share a stem: ${used.join(", ")}`);
  });

  it("uses it on both fold paths, not just the AlphaFold 3 one", () => {
    const uses = [...app.matchAll(/MODEL_STEMS\[family\]/g)];
    assert.equal(uses.length, 2,
      "MODEL_STEMS should name the object on the AF3 path and the AF2 path");
  });

  it("still lets a supplied FASTA header win", () => {
    // A name somebody typed beats a generated one; the model prefix is the
    // fallback, not an override.
    assert.match(app, /safeJobName\(header\)\s*\n\s*:\s*`\$\{MODEL_STEMS/);
  });
});

describe("the trunk cache", () => {
  it("keys on the model, or one model's trunk is denoised by the other", () => {
    // 🔴 THE FAULT THIS PINS, AND IT SHIPPED FOR AN HOUR. The cached trunk is a
    // pair and single representation, and those have the same shapes whichever
    // parameters produced them - so with the family missing from the key, a
    // fold with OpenBind followed by a fold with AlphaFold 3 on the same
    // sequence handed AF3's diffusion head OpenBind-0's trunk. Reproduced in the
    // page: 32 residues came back at pLDDT 41.5 with the status line reading
    // "trunk reused", against 83.3 once the key was fixed. Nothing errors; the
    // chain simply comes apart.
    const key = app.slice(app.indexOf("const trunkKey = JSON.stringify({"),
                          app.indexOf("const cached = trunkCache?.key"));
    assert.ok(key.length > 0, "the trunk key is not where this test expects");
    const fields = key.replace(/\/\/[^\n]*/g, "");
    assert.match(fields, /\bfamily\b/,
      "the trunk cache key does not include the model family");
  });
});

describe("?model= in the URL", () => {
  it("does not resolve of3 to openbind0", () => {
    // 🔴 THEY ARE DIFFERENT MODELS. OpenFold3's preview-2 and its v0.5.0
    // release differ in forward conventions - see src/af3/dialect.js - so
    // quietly accepting one name for the other hands somebody a model they did
    // not ask for, which is the whole class of error this port guards against.
    const aliases = app.match(/const MODEL_ALIASES = \{([^}]*)\}/s);
    assert.ok(aliases !== null, "MODEL_ALIASES is not where this test expects");
    assert.doesNotMatch(aliases[1], /\bof3\b/);
    // `openbind` resolves to this release rather than standing for the next.
    assert.match(aliases[1], /openbind:\s*"openbind0"/);
    assert.doesNotMatch(aliases[1], /openfold3/);
    assert.match(aliases[1], /ob:\s*"openbind0"/);
  });

  it("cannot accept the licence terms on somebody's behalf", () => {
    // The gate reads localStorage and the dialog's own result; nothing in it
    // may read the URL, or a link could agree for the person who opened it.
    const gate = app.slice(app.indexOf("async function agreeModelTerms"),
                           app.indexOf("function termsAccepted"));
    assert.ok(gate.length > 0, "agreeModelTerms is not where this test expects");
    assert.doesNotMatch(gate, /searchParams|location|URL\(/);
  });
});

describe("the licence dialog", () => {
  it("offers a way past AlphaFold 3's terms, not only a way through them", () => {
    assert.match(page, /id="model-terms-switch"[^>]*>|value="openbind0"/);
    assert.match(page, /id="model-terms-accept"/);
  });

  it("does not call the parameters closed source, which they are not", () => {
    // 🔴 THE CODE IS OPENLY LICENSED; THE PARAMETERS ARE USE-RESTRICTED. A
    // dialog asking somebody to accept a licence must not misstate it.
    // 🔴 THE RENDERED COPY, NOT THE COMMENTARY. The markup carries a comment
    // explaining why "not open source" was removed, and the first version of
    // this test failed on its own explanation - which is the assertion working,
    // and the wrong text to assert on.
    const dialog = page.slice(page.indexOf('id="model-terms"'),
                              page.indexOf("</dialog>")).replace(/<!--[\s\S]*?-->/g, "");
    assert.doesNotMatch(dialog, /not open source/i);
    assert.match(dialog, /not available for\s+commercial use/i);
    // 🔴 AND NOT NARROWER THAN THE TERMS EITHER. "Academic use only" is the
    // obvious short phrase and it is wrong twice over: DeepMind's terms cover
    // non-profits, research institutes, journalism and government bodies too,
    // and they exclude a researcher employed by a commercial organisation.
    // "Not available for commercial use" is the short form that stays true;
    // the linked terms carry the detail this dialog has no room for.
    assert.doesNotMatch(dialog, /academic use only/i);
  });
});
