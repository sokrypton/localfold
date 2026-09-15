import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { compareBaseline } from "../tools/gate-folds.mjs";
import { FOLDS } from "../tools/gate-folds.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

/**
 * 🔴 THE GATE THAT CATCHES A DRIFTED FIGURE, CHECKED WITHOUT A GPU.
 *
 * `tools/gate-baseline.json` exists because docs/A100.md recorded the stock
 * gate's four figures on 2026-09-12 and two of them were wrong three days
 * later - AF3 85.8348 against 83.0999, OpenDDE 92.0489 against 92.0382 - while
 * CLAUDE.md carried a third value for one of them. Bisected, each move is one
 * correctness fix whose own commit message states the new number. Nothing was
 * broken; the figures simply lived in prose and nothing re-ran them.
 *
 * A check whose only proof of working is a twenty-minute fold is a check nobody
 * falsifies, so the comparison is a pure function and these run in `npm test`.
 */
const stored = {
  adapter: "nvidia / ampere",
  recorded: "2026-09-15",
  arms: { portable: { AF2: '"checksum": -1287025', AF3: '"meanPlddt": 83.1295' } },
};

test("an unchanged run does not fail and says so", () => {
  const { enforced, lines } = compareBaseline({
    arm: "portable", adapter: "nvidia / ampere", stored,
    measured: { AF2: '"checksum": -1287025', AF3: '"meanPlddt": 83.1295' },
  });
  assert.equal(enforced, 0);
  assert.match(lines.join("\n"), /2 of 2 signatures unchanged/);
});

test("a drifted signature FAILS on the adapter that recorded it", () => {
  const { enforced, lines } = compareBaseline({
    arm: "portable", adapter: "nvidia / ampere", stored,
    measured: { AF2: '"checksum": -1287025', AF3: '"meanPlddt": 85.8348' },
  });
  assert.equal(enforced, 1, "a moved figure must make the gate exit non-zero");
  const text = lines.join("\n");
  assert.match(text, /83\.1295\s+->\s+.*85\.8348/, "it must print both values");
  assert.match(text, /npm run test:portable -- --write-baseline/, "and how to re-record");
});

/**
 * 🔴 AND NOT ON ANOTHER MACHINE, because a signature does not travel: the A100
 * folds `fold-af2.js` at -1287025 and an M2 at -1282976 over the same code and
 * the same input, and both are correct. A gate that went red on that is a gate
 * everyone turns off.
 */
test("the same drift on another adapter is reported and not enforced", () => {
  const { enforced, lines } = compareBaseline({
    arm: "portable", adapter: "apple / metal-3", stored,
    measured: { AF2: '"checksum": -1282976' },
  });
  assert.equal(enforced, 0);
  assert.match(lines.join("\n"), /reported and NOT failed/);
  assert.match(lines.join("\n"), /apple \/ metal-3/);
});

test("a model missing from the baseline is not a drift", () => {
  const { enforced } = compareBaseline({
    arm: "portable", adapter: "nvidia / ampere", stored,
    measured: { AF2: '"checksum": -1287025', rosettafold3: '"meanPlddt": 81.5' },
  });
  assert.equal(enforced, 0, "a model recorded later must not fail the older baseline");
});

test("an arm with no baseline asks for one instead of failing", () => {
  const { enforced, lines } = compareBaseline({
    arm: "stock", adapter: "nvidia / ampere", stored, measured: { AF2: "x" },
  });
  assert.equal(enforced, 0);
  assert.match(lines.join("\n"), /no baseline for "stock"/);
});

/**
 * 🔴 AND THE RECORDED FILE MUST COVER THE FOLDS THAT EXIST, or adding a model
 * to `FOLDS` quietly leaves it outside every baseline - which is this
 * repository's own recurring shape: a list in one place and a list in another.
 */
test("the shipped baseline covers every arm and every fold", () => {
  const shipped = JSON.parse(readFileSync(join(ROOT, "tools/gate-baseline.json"), "utf8"));
  assert.ok(shipped.adapter, "the baseline must name the adapter that produced it");
  assert.deepEqual(Object.keys(shipped.arms).sort(), ["portable", "spec-floor", "stock"]);
  const names = FOLDS.map(([name]) => name);
  for (const [arm, signatures] of Object.entries(shipped.arms)) {
    assert.deepEqual(Object.keys(signatures).sort(), [...names].sort(),
      `"${arm}" does not cover every fold in FOLDS`);
  }
});
