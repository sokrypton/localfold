import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { AF3_COUNTS, OPENDDE_COUNTS, OPENDDE_SAMPLER_MODE } from "../web/af3-model.js";

/**
 * The page's sampler select and the step-count tables must name the same modes.
 *
 * 🔴 BECAUSE THE TWO LISTS ARE WRITTEN IN DIFFERENT FILES AND ONE OF THEM WENT
 * STALE. `AF3_COUNTS` lives in web/af3-model.js and the `<option>`s live in
 * index.html, and `web/app.js` subscripts the table with the select's value.
 * One of its two readings had a `?? table.flow` fallback and the other did not,
 * so removing a mode from one file and not the other is a "cannot read
 * properties of undefined" in the middle of starting a fold - which is this
 * repository's stale-allow-list trap, twice recorded and now gated.
 *
 * It runs both directions: no option without a row (the crash), and no row
 * without an option (a dead measurement nobody can select).
 */
const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");

/** The option values of one `<select>`, by id. */
function optionsOf(id) {
  const open = html.indexOf(`<select id="${id}"`);
  assert.notEqual(open, -1, `index.html has no <select id="${id}">`);
  const close = html.indexOf("</select>", open);
  assert.notEqual(close, -1, `<select id="${id}"> is never closed`);
  const values = [...html.slice(open, close).matchAll(/<option value="([^"]*)"/g)]
    .map((match) => match[1]);
  // 🔴 A RULE THAT STOPS MATCHING PASSES BY FINDING NOTHING - the same guard
  // test/folded-grid-guard.test.js carries.
  assert.ok(values.length >= 2, `#${id} parsed to ${values.length} options`);
  return values;
}

test("the sampler select and the count tables name the same modes", async (t) => {
  const modes = optionsOf("af3-mode");

  await t.test("every option has an AF3_COUNTS row", () => {
    for (const mode of modes) {
      assert.ok(AF3_COUNTS[mode] !== undefined,
                `#af3-mode offers "${mode}" and AF3_COUNTS has no row for it`);
    }
  });

  await t.test("every AF3_COUNTS row is an option", () => {
    for (const mode of Object.keys(AF3_COUNTS)) {
      assert.ok(modes.includes(mode),
                `AF3_COUNTS has "${mode}" and #af3-mode does not offer it`);
    }
  });

  await t.test("OpenDDE's narrower table is a subset of the options", () => {
    for (const mode of Object.keys(OPENDDE_COUNTS)) {
      assert.ok(modes.includes(mode), `OPENDDE_COUNTS has "${mode}" and the select does not`);
    }
    // The value the page FORCES for that family must be one it can serve.
    assert.ok(OPENDDE_COUNTS[OPENDDE_SAMPLER_MODE] !== undefined);
    assert.ok(AF3_COUNTS[OPENDDE_SAMPLER_MODE] !== undefined);
  });

  // 🔴 AND THE DEFAULT IS DIFFUSION, asserted rather than assumed: the page
  // shipped Flow as its default and every doc that said so went stale when it
  // changed. `selected` in the markup is the only thing that decides it.
  await t.test("the marked-up default is diffusion", () => {
    const open = html.indexOf('<select id="af3-mode"');
    const close = html.indexOf("</select>", open);
    const selected = [...html.slice(open, close).matchAll(/<option value="([^"]*)"[^>]*selected/g)]
      .map((match) => match[1]);
    assert.deepEqual(selected, ["diffusion"]);
  });

  // ...and "ode" specifically is gone from the page while the STEP survives for
  // the CLI, which is the decision docs/AF3.md records.
  await t.test("ode is not offered on the page", () => {
    assert.ok(!modes.includes("ode"));
    assert.equal(AF3_COUNTS.ode, undefined);
  });
});
