/**
 * Which model the page folds with, and what that name is allowed to mean.
 *
 * 🔴 TWO BUNDLES BUILD AlphaFold 3's GRAPH NOW, and `=== "af3"` stopped meaning
 * what its call sites meant by it the moment the second arrived. Three of them
 * were CAPABILITY checks - ligands, modified residues, nucleic chains - and
 * every one refused those inputs under OpenBind with a message naming a
 * capability the model actually has, because OpenBind runs the same featuriser
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
    assert.deepEqual([...AF3_FAMILIES].sort(), ["af3", "openbind"]);
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
  // "Ligands need AlphaFold 3; the model is set to openbind" - a refusal to
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

describe("the trunk cache", () => {
  it("keys on the model, or one model's trunk is denoised by the other", () => {
    // 🔴 THE FAULT THIS PINS, AND IT SHIPPED FOR AN HOUR. The cached trunk is a
    // pair and single representation, and those have the same shapes whichever
    // parameters produced them - so with the family missing from the key, a
    // fold with OpenBind followed by a fold with AlphaFold 3 on the same
    // sequence handed AF3's diffusion head OpenBind's trunk. Reproduced in the
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
  it("does not resolve of3 to openbind", () => {
    // 🔴 THEY ARE DIFFERENT MODELS. OpenFold3's preview-2 and its v0.5.0
    // release differ in forward conventions - see src/af3/dialect.js - so
    // quietly accepting one name for the other hands somebody a model they did
    // not ask for, which is the whole class of error this port guards against.
    const aliases = app.match(/const MODEL_ALIASES = \{([^}]*)\}/s);
    assert.ok(aliases !== null, "MODEL_ALIASES is not where this test expects");
    assert.doesNotMatch(aliases[1], /\bof3\b/);
    assert.doesNotMatch(aliases[1], /openfold3/);
    assert.match(aliases[1], /ob:\s*"openbind"/);
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
    assert.match(page, /id="model-terms-switch"[^>]*>|value="openbind"/);
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
  });
});
