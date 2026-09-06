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

import { AF3_FAMILIES, FOLDING_FAMILIES, MODEL_BUNDLES }
  from "../src/reference/manifests/index.js";

const app = readFileSync(new URL("../web/app.js", import.meta.url), "utf8");
const page = readFileSync(new URL("../index.html", import.meta.url), "utf8");

describe("the AlphaFold 3 families", () => {
  it("lists every bundle that builds that graph, and only those", () => {
    assert.deepEqual([...AF3_FAMILIES].sort(), ["af3", "openbind0"]);
    for (const family of AF3_FAMILIES) {
      assert.ok(family in MODEL_BUNDLES, `${family} has no bundle`);
    }
  });

  // 🔴 THE FOLDING FAMILIES, NOT EVERY BUNDLE. ESM-C is a bundle - the loader,
  // the shard cache and the site build all reach it by name - and it is a
  // language model, so offering it in the model picker would be offering it as
  // a structure predictor.
  // 🔴 AND REACHABLE IS NOT THE SAME AS "IN THE MODEL ROW", SINCE EF2-fast SHIPS
  // AS TWO CHECKPOINTS UNDER ONE ENTRY. They differ only in the language model
  // they were trained against, so the PLM row picks between them and the model
  // row shows one name - see PLM_FAMILIES in web/app.js. What must stay true is
  // that every folding family is reachable from SOMEWHERE, or it is a bundle
  // the site publishes and nobody can fold with.
  // 🔴 ONE READER OF THE MODEL ROW, OR TWO ANSWERS. EF2-fast's checkpoints share
  // an <option> and the PLM row picks between them, so `chosenFamily()` is the
  // resolution and anything reading `model-family` directly gets the other
  // answer. `modelFamily()` did, which is what runFold preloads and folds with -
  // so the page folded the 600M pair while every label, stem and archive field
  // said 300M, and the two settings produced byte-identical structures. Nothing
  // errors; the fold is simply of a model the page did not report.
  it("resolves the family in one place, so the fold matches its label", () => {
    const body = app.slice(app.indexOf("const modelFamily = "));
    const head = body.slice(0, body.indexOf("\n};"));
    assert.ok(head.includes("chosenFamily()"),
              "modelFamily does not resolve through chosenFamily");
    assert.ok(!/getElementById\("model-family"\)\?\.value/.test(head),
              "modelFamily reads the model row directly, which skips the PLM row");
    // ...and runFold folds with what modelFamily returned, not a second read.
    const run = app.slice(app.indexOf("let family = modelFamily("));
    const preload = run.slice(0, run.indexOf("startModelPreload("));
    assert.ok(!preload.includes('getElementById("model-family")'),
              "runFold re-reads the model row between resolving and preloading");
  });

  it("makes each of them reachable from a control", () => {
    const app = readFileSync(new URL("../web/app.js", import.meta.url), "utf8");
    const variants = app.slice(app.indexOf("const PLM_FAMILIES = {"));
    const table = variants.slice(0, variants.indexOf("};"));
    for (const family of FOLDING_FAMILIES) {
      const inRow = new RegExp(`<option value="${family}"`).test(page);
      const inVariants = table.includes(`"${family}"`);
      assert.ok(inRow || inVariants,
        `${family} is offered by no control in index.html or PLM_FAMILIES`);
    }
  });

  // ...and every family a control names has to exist, which is the other
  // direction: a dropdown offering a family with no bundle fails at load.
  it("names no family it does not have", () => {
    const app = readFileSync(new URL("../web/app.js", import.meta.url), "utf8");
    const variants = app.slice(app.indexOf("const PLM_FAMILIES = {"));
    for (const [, family] of variants.slice(0, variants.indexOf("};"))
        .matchAll(/"(ef2-[\w-]+)"/g)) {
      assert.ok(FOLDING_FAMILIES.includes(family),
        `PLM_FAMILIES names ${family}, which is not a folding family`);
    }
  });
});

describe("what needs an AlphaFold 3 graph", () => {
  // 🔴 THE FAULT THIS PINS. `choice !== "af3"` here read as "is this DeepMind's
  // checkpoint", and the reported symptom was
  // "Ligands need AlphaFold 3; the model is set to openbind0" - a refusal to
  // fold something the selected model handles perfectly well.
  // 🔴 THE GUARD IS `supportsAllAtom`, NOT `isAf3Family`, AND THE DIFFERENCE IS
  // THE POINT. ESMFold2 runs a different GRAPH and the same all-atom
  // representation, so a guard asking "is this AlphaFold 3" refuses a ligand
  // under a model that has ligand tokens. Templates are the one thing that is
  // still an AF3 question, and they are tested separately below.
  for (const what of ["ligandCount", "modificationCount", "nucleicCount"]) {
    it(`tests the family, not the name, for ${what}`, () => {
      // The condition runs to the end of the line, and it CONTAINS brackets -
      // `!isAf3Family(choice)` - so a `[^)]+` class stops inside the call it is
      // meant to find and reports a correct guard as missing.
      const guard = new RegExp(`if \\(${what} > 0 && (.+)\\) \\{`);
      const found = app.match(guard);
      assert.ok(found !== null, `no guard found for ${what}`);
      assert.match(found[1], /supportsAllAtom\(/,
        `${what} is gated on something other than supportsAllAtom: ${found[1]}`);
      assert.doesNotMatch(found[1], /"af3"/,
        `${what} still compares against the literal "af3"`);
    });
  }

  // 🔴 AND A TEMPLATE STILL IS AN AlphaFold 3 QUESTION. ESMFold2 has no template
  // module at all - z_init has five terms and none of them is one - so a
  // template set on an entity row would be fetched, aligned and dropped.
  it("keeps templates on isAf3Family, which is the one that is still true", () => {
    const found = app.match(/if \(templateCount > 0 && (.+)\) \{/);
    assert.ok(found !== null, "no guard found for templateCount");
    assert.match(found[1], /isAf3Family\(/,
      `templateCount is gated on something else: ${found[1]}`);
  });
});

describe("what a fold is called", () => {
  it("does not download a language model for a fold with no protein", () => {
    // 🔴 ESM-C IS HANDED PROTEIN TOKENS ONLY - `protein_mask = (mol_type == 0)
    // & token_mask` - so a ligand, DNA or RNA input has no row for it. The fold
    // already skips the CALL at `lm.ids.length === 0`; what it could not skip
    // was the 224 MiB download, which `towerStore.prefetch()` starts the moment
    // the model is chosen. This pins the decision, because nothing else would
    // notice it silently reverting to always-fetch.
    assert.ok(app.includes("needsLanguageModel"), "the decision is not made");
    const preload = app.slice(app.indexOf("function startModelPreload"));
    const decision = preload.slice(0, preload.indexOf("loadEsmfold2Weights"));
    assert.ok(decision.includes('entity.type === "protein"'),
              "the decision does not read the entities");
    // ...and an empty list is a page nobody has typed into yet, where the
    // likeliest next thing is a protein - so it fetches.
    assert.ok(decision.includes("typed.length === 0"), "an empty list must fetch");
    // ...matched as two facts rather than one literal, because the call also
    // carries the family now and a one-line pattern breaks on the wrapping.
    const call = preload.slice(preload.indexOf("loadEsmfold2Weights"));
    assert.ok(call.slice(0, 200).includes("languageModel: needsLanguageModel"),
              "the decision does not reach the loader");
    assert.ok(call.slice(0, 200).includes("family"),
              "the loader is not told which checkpoint to load");
  });

  it("gives every model its own stem, so two folds are told apart", () => {
    // 🔴 THE OBJECT NAME IS THE ONLY PLACE THE MODEL SHOWS ON SCREEN. Both
    // AF3-graph bundles used to produce `af3_N` and both AlphaFold 2 models
    // `prediction_N`, so a page holding one of each showed two objects with the
    // same prefix - and the stem becomes the archive's file names too, so a
    // downloaded `af3_1_model_0.pdb` could have come from either.
    const table = app.match(/const MODEL_STEMS = \{([^}]*)\}/s);
    assert.ok(table !== null, "MODEL_STEMS is not where this test expects");
    const stems = {};
    // ...a key is quoted when it is not a bare identifier, which `ef2-fast-600m`
    // is not - and an unquoted-only pattern silently sees a table with one
    // family missing rather than failing to parse.
    for (const [, key, value] of table[1].matchAll(/"?([\w-]+)"?:\s*"([^"]+)"/g)) {
      stems[key] = value;
    }
    // Every family the page offers has one...
    for (const family of FOLDING_FAMILIES) {
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
