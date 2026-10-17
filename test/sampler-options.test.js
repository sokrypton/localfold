import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as af3Model from "../web/af3-model.js";

/**
 * ONE SAMPLER AND ONE DIAL. Flow was removed (docs/AF3.md, "Flow removed") and every AF3-lineage family folds the
 * exponential update on the geometric schedule at twenty-five steps (docs/AF3.md, "The default sampler") - so the
 * page has no sampler select, one step table with one row, and sends nothing but a step count. A select, a second
 * table or a schedule left behind is a control that does nothing, or a second copy of a choice that can disagree.
 */
const read = (path) => readFileSync(new URL(path, import.meta.url), "utf8");
const html = read("../index.html"), app = read("../web/app.js");

test("one sampler: no select on the page, one row in the table", () => {
  assert.ok(!html.includes('id="af3-mode"'), "index.html still offers a sampler select");
  assert.ok(!/af3-mode(?!l)/.test(app), "web/app.js still reads the sampler select");
  assert.deepEqual(Object.keys(af3Model.AF3_COUNTS), ["diffusion"]);
});

test("one dial for every family, preferring twenty-five", () => {
  assert.equal(af3Model.AF3_COUNTS.diffusion.preferred, 25);
  assert.ok(af3Model.AF3_COUNTS.diffusion.values.includes(25));
  for (const gone of ["OPENDDE_COUNTS", "ALPHAFOLD3_COUNTS", "SHORT_SCHEDULES", "countsForFamily",
                      "diffusionScheduleFor"]) {
    assert.equal(af3Model[gone], undefined, `web/af3-model.js still exports ${gone}`);
  }
});

test("the page and the worker send steps, never a schedule start", () => {
  assert.ok(!/sigmaMax/.test(app), "web/app.js still resolves a schedule start");
  assert.ok(!/sigma-max|sigmaMax/.test(read("../python/localfold/worker.py")), "python/localfold/worker.py still passes a schedule start");
});

test("the fold's default is the exponential update on the geometric schedule", () => {
  // both ports: the page's foldBatch and the native binary the Colab backend runs
  assert.match(read("../webgpu/af3/fold.js"),
               /DIFFUSION_DEFAULTS = Object\.freeze\(\{ update: "exponential", schedule: "geometric" \}\)/);
  assert.match(read("../cuda/af3/src/sampler.cuh"), /bool geometric = true, exponential = true, churn = true/);
});
