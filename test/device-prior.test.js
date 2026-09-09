/**
 * The device profile's prior lookup, and the switch that turns it off.
 *
 * 🔴 `PRIORS` HAS TWO ENTRIES AND EVERY OTHER GPU IN THE WORLD TAKES THE
 * DEFAULTS, which is one M2's answers and costs 1.5x on AF2 and ESMFold2 - see
 * docs/PERF.md. `ignoreDevicePrior` is how a machine that HAS a prior measures
 * what the machines that do not are getting, and its keep-list is how a sweep
 * asks what one knob of that gap is worth.
 *
 * 🔴 AND THE KEEP-LIST IS TESTED BECAUSE THE SWEEP THAT USES IT FAILED SILENTLY
 * ONCE. The driver held its arms in a bash array called `GROUPS`, which is a
 * read-only built-in holding the user's group ids: the assignment did nothing,
 * every arm ran the bare baseline, and twelve identical timings came back. That
 * one was loud enough to notice. A keep-list that quietly kept nothing would
 * not be - every arm would simply read as "this knob is worth zero", which is a
 * plausible answer and the wrong one.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import {
  DEFAULT_TUNING, deviceTuning, ignoreDevicePrior, recordAdapter, setDeviceTuning,
} from "../src/runtime/device-profile.js";

/** Enough of a GPUDevice for the profile: it only ever uses it as a map key. */
const fakeDevice = () => ({ features: new Set(), limits: {} });
const asAmpere = (device) => recordAdapter(device, {
  info: { vendor: "nvidia", architecture: "ampere", subgroupMatrixConfigs: [] },
});

describe("the device prior", () => {
  it("gives a recognised architecture its measured answers", () => {
    const device = asAmpere(fakeDevice());
    const tuning = deviceTuning(device);
    assert.equal(tuning.gridAttendMatrix, true);
    assert.equal(tuning.stagedMatrixBlock, "64x128x16x1x8");
    // ...and the default is not that, or this test would pass on nothing.
    assert.equal(DEFAULT_TUNING.gridAttendMatrix, null);
    assert.equal(DEFAULT_TUNING.stagedMatrixBlock, null);
  });

  it("gives an unknown architecture the defaults", () => {
    const device = recordAdapter(fakeDevice(), {
      info: { vendor: "acme", architecture: "unheard-of", subgroupMatrixConfigs: [] },
    });
    assert.equal(deviceTuning(device).gridAttendMatrix, null);
    assert.equal(deviceTuning(device).stagedMatrixBlock, null);
  });

  it("ignoreDevicePrior makes a recognised device answer as an unknown one", () => {
    const device = asAmpere(fakeDevice());
    assert.equal(deviceTuning(device).gridAttendMatrix, true);
    ignoreDevicePrior(device);
    assert.equal(deviceTuning(device).gridAttendMatrix, null);
    assert.equal(deviceTuning(device).stagedMatrixBlock, null);
  });

  it("keeps exactly the named knobs, at the PRIOR's values", () => {
    const device = asAmpere(fakeDevice());
    ignoreDevicePrior(device, ["gridAttendMatrix", "gridAttendMatrixTile"]);
    const tuning = deviceTuning(device);
    // 🔴 THE VALUE COMES FROM THE PRIOR AND IS NEVER SPELLED BY THE CALLER,
    // which is the whole point: `diffusionTokenTile` is an object and `--tune`
    // splits its argument on commas, so a sweep could not write it at all.
    assert.equal(tuning.gridAttendMatrix, true);
    assert.equal(tuning.gridAttendMatrixTile, "4x32");
    // ...and everything else is back to the default.
    assert.equal(tuning.stagedMatrixBlock, null);
    assert.equal(tuning.pairTransitionSplit, null);
  });

  it("keeps an object-valued knob whole", () => {
    const device = asAmpere(fakeDevice());
    ignoreDevicePrior(device, ["diffusionTokenTile"]);
    const kept = deviceTuning(device).diffusionTokenTile;
    assert.equal(typeof kept, "object");
    assert.equal(kept.crossover, 175);
    assert.equal(deviceTuning(device).atomRowTile, DEFAULT_TUNING.atomRowTile);
  });

  it("names that are not in the prior keep nothing and do not throw", () => {
    const device = asAmpere(fakeDevice());
    ignoreDevicePrior(device, ["notAKnob"]);
    assert.equal(deviceTuning(device).gridAttendMatrix, null);
  });

  it("an explicit override still beats both", () => {
    const device = asAmpere(fakeDevice());
    ignoreDevicePrior(device);
    setDeviceTuning(device, { gridAttendMatrix: true });
    assert.equal(deviceTuning(device).gridAttendMatrix, true);
  });
});
