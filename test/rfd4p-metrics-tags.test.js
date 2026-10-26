// The tags the page scores a design under: `design` and what upstream's validation sets give a
// task conditioned like this one (`conditioningMetricsTags`), read off the batch - a motif's RMSD
// where there is a motif, and nowhere else, because upstream's raises on a design with none.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { buildBatch, decodeTextOrZstd } from "../shared/rfd4p/pipeline/index.js";
import { conditioningMetricsTags } from "../shared/rfd4p/metrics.js";

const root = new URL("..", import.meta.url).pathname;

function tagsFor(jobPath) {
  const path = resolve(root, jobPath);
  const job = JSON.parse(readFileSync(path, "utf8"));
  const inputCif = decodeTextOrZstd(new Uint8Array(readFileSync(resolve(dirname(path), job.input))));
  const spec = { contig: job.contig, ...(job.conditions ? { conditions: job.conditions } : {}) };
  return conditioningMetricsTags(buildBatch(spec, { seed: 0, inputCif }).atoms);
}

test("a design is tagged by the conditioning it carries", () => {
  assert.deepEqual(tagsFor("cuda/rfd4p/featurise/gate-groups-5o45.json"),
    ["3d_coordinate_conditioning", "internal_coordinate_conditioning"]);
  assert.deepEqual(tagsFor("cuda/rfd4p/featurise/gate-metrics.json"),
    ["3d_coordinate_conditioning", "internal_coordinate_conditioning", "hotspot_conditioning"]);
  assert.deepEqual(tagsFor("cuda/rfd4p/featurise/gate-partial-1brs.json"), []);
});
