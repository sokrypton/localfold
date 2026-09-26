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

test("AlphaFold 3's short schedule is its own and nobody else's", async (t) => {
  const { ALPHAFOLD3_COUNTS, ALPHAFOLD3_SHORT_SCHEDULE, countsForFamily,
          diffusionScheduleFor } = await import("../web/af3-model.js");
  const modes = optionsOf("af3-mode");

  await t.test("its table offers the same modes as the page", () => {
    assert.deepEqual(Object.keys(ALPHAFOLD3_COUNTS).sort(), Object.keys(AF3_COUNTS).sort());
    for (const mode of modes) assert.ok(ALPHAFOLD3_COUNTS[mode] !== undefined, mode);
  });

  await t.test("its preferred count takes the short schedule", () => {
    const preferred = ALPHAFOLD3_COUNTS.diffusion.preferred;
    assert.ok(preferred < ALPHAFOLD3_SHORT_SCHEDULE.below);
    assert.deepEqual(diffusionScheduleFor("af3", "diffusion", preferred),
                     { sigmaMax: ALPHAFOLD3_SHORT_SCHEDULE.sigmaMax });
  });

  await t.test("the model's own schedule everywhere else", () => {
    assert.equal(diffusionScheduleFor("af3", "diffusion", 25), undefined);
    assert.equal(diffusionScheduleFor("af3", "diffusion", 200), undefined);
    assert.equal(diffusionScheduleFor("af3", "flow", 16), undefined);
    for (const family of ["boltz2", "protenix2", "openbind0", "intellifold2",
                          "rosettafold3", "opendde"]) {
      assert.equal(diffusionScheduleFor(family, "diffusion", 20), undefined, family);
    }
    assert.equal(countsForFamily("boltz2"), AF3_COUNTS);
    assert.equal(countsForFamily("opendde"), OPENDDE_COUNTS);
    assert.equal(countsForFamily("af3"), ALPHAFOLD3_COUNTS);
  });

  // 🔴 ONE READING OF THE TABLE. The dial and the fold each chose a table by
  // family, and a second family-specific table is exactly where two copies of
  // that choice would disagree - the dial offering 20 and the fold defaulting
  // to 25, or the reverse.
  await t.test("web/app.js reads the table through countsForFamily only", () => {
    const app = readFileSync(new URL("../web/app.js", import.meta.url), "utf8");
    assert.equal((app.match(/countsForFamily\(/g) ?? []).length, 2);
    assert.ok(!/[^_]AF3_COUNTS\[|OPENDDE_COUNTS\[/.test(app));
    assert.ok(/schedule: diffusionScheduleFor\(/.test(app));
  });
});
