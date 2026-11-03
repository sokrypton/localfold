/**
 * Where a fold is sent, from the figures a page can read before it downloads anything.
 *
 * The cases are the machines the plan exists for: an 8 GiB card with other programs on it, a rented card past the point a
 * buffer can be bound, no GPU, a software GPU, and a card whose driver has just reset.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  cpuFoldSeconds, cpuPeakBytes, describePlan, gpuFoldSeconds, measuredFoldSeconds, noteFold, planEsmfold2, planMeasuredModel,
  readFoldRates, roughDuration,
} from "../web/compute-plan.js";
import { cardBudgetBytes } from "../webgpu/runtime/device-memory.js";
import { esmfold2MaxTokens, esmfold2PeakBytes, esmfold2Spills, FOLD_WEIGHT_BYTES, pairBytes }
  from "../webgpu/esmfold2/memory-plan.js";

const GIB = 1024 ** 3;
const MIB = 1024 ** 2;
const rtx3070 = {
  name: "RTX 3070", budgetBytes: cardBudgetBytes(8 * GIB), maxBindingBytes: 2147483644, halfWeights: true,
};
const a10 = { name: "A10", budgetBytes: cardBudgetBytes(24 * GIB), maxBindingBytes: 2147483644, halfWeights: false };
const machine = { gpu: rtx3070, threads: 8, ramBytes: 32 * GIB };

function fakeStorage() {
  const map = new Map();
  return { getItem: (key) => map.get(key) ?? null, setItem: (key, value) => map.set(key, value) };
}

describe("the memory plan against what was measured", () => {
  it("reads a fold's peak on an RTX 3070 to within 2%", () => {
    // The allocator's own peak from a page fold: 1000 residues whole, 1200 and 1350 with the spill.
    for (const [tokens, spill, measuredMiB] of [[1000, false, 3787], [1200, true, 3792], [1350, true, 4646]]) {
      const planned = esmfold2PeakBytes({ tokens, spill }) / MIB;
      assert.ok(Math.abs(planned / measuredMiB - 1) < 0.02, `${tokens}: planned ${planned} against ${measuredMiB} MiB`);
    }
  });

  it("does not count a fold's own weights twice in deciding to spill", () => {
    const budgetBytes = rtx3070.budgetBytes;
    assert.equal(esmfold2Spills({ tokens: 1000, residentBytes: FOLD_WEIGHT_BYTES, budgetBytes }), false);
    assert.equal(esmfold2Spills({ tokens: 1200, residentBytes: FOLD_WEIGHT_BYTES, budgetBytes }), true);
  });

  it("agrees with the largest fold it says a budget holds", () => {
    const tokens = esmfold2MaxTokens({ budgetBytes: rtx3070.budgetBytes });
    assert.equal(planEsmfold2({ tokens }, machine).where, "gpu");
    assert.equal(planEsmfold2({ tokens: tokens + 1 }, machine).where, "over");
  });

  it("stops at the longest fold two halves of a tensor can bind, whatever the card", () => {
    const large = { ...a10, budgetBytes: 80 * GIB };
    assert.equal(esmfold2MaxTokens({ budgetBytes: large.budgetBytes }), 2047);
    assert.equal(planEsmfold2({ tokens: 2047 }, { ...machine, gpu: large }).where, "gpu");
    const plan = planEsmfold2({ tokens: 2048 }, { ...machine, gpu: large });
    assert.equal(plan.where, "over");
    assert.equal(plan.why, "binding");
  });

  it("spills past one binding, however much the budget has room for", () => {
    const large = { ...a10, budgetBytes: 80 * GIB };
    assert.equal(planEsmfold2({ tokens: 1448 }, { ...machine, gpu: large }).spills, false);
    assert.equal(planEsmfold2({ tokens: 1449 }, { ...machine, gpu: large }).spills, true);
  });
});

describe("where a fold is sent", () => {
  it("takes the GPU when the fold fits, and says when it spills", () => {
    const small = planEsmfold2({ tokens: 600 }, machine);
    assert.equal(small.where, "gpu");
    assert.equal(small.spills, false);
    const tight = planEsmfold2({ tokens: 1200 }, machine);
    assert.equal(tight.where, "gpu");
    assert.equal(tight.spills, true);
    assert.ok(tight.needBytes < planEsmfold2({ tokens: 1200 }, { ...machine, gpu: a10 }).needBytes);
  });

  it("asks when the fold is too big for the card, with what the CPU would cost", () => {
    const plan = planEsmfold2({ tokens: 1400 }, { ...machine, ramBytes: 64 * GIB });
    assert.equal(plan.where, "over");
    assert.equal(plan.why, "memory");
    assert.ok(plan.cpuSeconds > plan.seconds);
    assert.match(describePlan(plan).text, /Needs \d\.\d of 4\.8 GiB · CPU about/);
    assert.equal(describePlan(plan).tone, "warn");
  });

  it("offers no CPU a fold would not fit in", () => {
    const plan = planEsmfold2({ tokens: 1400 }, machine);
    assert.equal(plan.where, "over");
    assert.equal(plan.cpuFits, false);
    assert.match(describePlan(plan).text, /^Needs \d\.\d of 4\.8 GiB$/);
  });

  it("counts a device without shader-f16 as holding wider weights", () => {
    const wide = planEsmfold2({ tokens: 600 }, { ...machine, gpu: a10 });
    assert.ok(wide.needBytes > planEsmfold2({ tokens: 600 }, machine).needBytes);
  });

  it("folds on the CPU with no GPU, with a software GPU, with one refusing work, and when asked", () => {
    for (const [override, why] of [
      [{ gpu: undefined }, "no-gpu"],
      [{ gpu: { ...rtx3070, software: true } }, "software"],
      [{ gpu: { ...rtx3070, refusing: true } }, "gpu-refusing"],
      [{ preference: "cpu" }, "chosen"],
    ]) {
      const plan = planEsmfold2({ tokens: 300 }, { ...machine, ...override });
      assert.equal(plan.where, "cpu", why);
      assert.equal(plan.why, why);
      assert.match(describePlan(plan).text, /^8 CPU threads · about/);
      assert.ok(describePlan(plan).title.length > 0, why);
    }
  });

  it("says so when no part of the machine can hold it", () => {
    const plan = planEsmfold2({ tokens: 1500 }, { ...machine, gpu: undefined, ramBytes: 8 * GIB });
    assert.equal(plan.where, "none");
    assert.equal(describePlan(plan).tone, "warn");
    assert.ok(cpuPeakBytes({ tokens: 1500 }) > 8 * GIB * 0.7);
  });

  it("plans on the CPU where the browser will not say how much RAM there is", () => {
    assert.equal(planEsmfold2({ tokens: 300 }, { gpu: undefined, threads: 2 }).where, "cpu");
  });

  it("reads memory with no budget, which is a ceiling the reader has lifted", () => {
    const plan = planEsmfold2({ tokens: 1400 }, { ...machine, gpu: { ...rtx3070, budgetBytes: undefined } });
    assert.equal(plan.where, "gpu");
    assert.equal(plan.spills, false);
    assert.match(describePlan(plan).text, /^\d\.\d GiB · about/);
  });
});

describe("how long it will take", () => {
  it("follows what an A10 measured, to within 15%", () => {
    for (const [tokens, seconds] of [[300, 7.4], [600, 27.3], [800, 49.1], [1000, 78.4], [1200, 120.6]]) {
      const planned = gpuFoldSeconds({ tokens });
      assert.ok(Math.abs(planned / seconds - 1) < 0.15, `${tokens}: planned ${planned} against ${seconds} s`);
    }
  });

  it("follows what eight CPU threads measured, and what fewer cost", () => {
    assert.ok(Math.abs(cpuFoldSeconds({ tokens: 76, threads: 8 }) - 54) < 1);
    assert.ok(Math.abs(cpuFoldSeconds({ tokens: 300, threads: 8 }) / 670 - 1) < 0.1);
    assert.ok(Math.abs(cpuFoldSeconds({ tokens: 76, threads: 1 }) / 135 - 1) < 0.05);
    assert.ok(Math.abs(cpuFoldSeconds({ tokens: 76, threads: 4 }) / 68 - 1) < 0.05);
  });

  it("charges fewer loops for less of the trunk", () => {
    assert.ok(gpuFoldSeconds({ tokens: 600, loops: 1 }) < gpuFoldSeconds({ tokens: 600, loops: 4 }) / 2);
  });

  it("is written as roughly as it is known", () => {
    assert.equal(roughDuration(2), "about 2 s");
    assert.equal(roughDuration(27), "about 25 s");
    assert.equal(roughDuration(120), "about 2 min");
    assert.equal(roughDuration(5400), "about 1.5 h");
    assert.equal(roughDuration(12 * 3600), "about 12 h");
  });

  it("learns a card's rate from a fold that ran, and ignores one too short to say", () => {
    const storage = fakeStorage();
    noteFold({ where: "gpu", card: "RTX 3070", tokens: 600, loops: 4, seconds: 2 }, storage);
    assert.deepEqual(readFoldRates(storage).cards, {});
    const reference = gpuFoldSeconds({ tokens: 600, loops: 4 });
    noteFold({ where: "gpu", card: "RTX 3070", tokens: 600, loops: 4, seconds: reference * 2 }, storage);
    const rates = readFoldRates(storage);
    assert.ok(Math.abs(rates.cards["RTX 3070"] - 2) < 1e-9);
    const plan = planEsmfold2({ tokens: 600 }, { ...machine, rates });
    assert.ok(Math.abs(plan.seconds / reference - 2) < 1e-9);
    noteFold({ where: "gpu", card: "RTX 3070", tokens: 600, loops: 4, seconds: reference }, storage);
    assert.ok(Math.abs(readFoldRates(storage).cards["RTX 3070"] - 1.5) < 1e-9);
  });

  it("keeps the CPU's rate apart from any card's", () => {
    const storage = fakeStorage();
    noteFold({ where: "cpu", tokens: 300, loops: 4, threads: 8, seconds: cpuFoldSeconds({ tokens: 300, threads: 8 }) * 3 },
             storage);
    const rates = readFoldRates(storage);
    assert.ok(Math.abs(rates.cpu - 3) < 1e-9);
    assert.deepEqual(rates.cards, {});
  });

  it("survives a store that holds nonsense", () => {
    const storage = fakeStorage();
    storage.setItem("localfold.fold-rates", "not json");
    assert.deepEqual(readFoldRates(storage), { cards: {} });
  });
});

describe("the pair tensor", () => {
  it("is the figure the rest is built on", () => {
    assert.equal(pairBytes(1000), 1000 * 1000 * 256 * 4);
  });
});

describe("a CPU fold's memory", () => {
  it("reads what the process reached, to within 2%", () => {
    for (const [tokens, measuredGib] of [[76, 2.83], [300, 4.93], [600, 9.97]]) {
      const planned = cpuPeakBytes({ tokens }) / GIB;
      assert.ok(Math.abs(planned / measuredGib - 1) < 0.02, `${tokens}: planned ${planned} against ${measuredGib} GiB`);
    }
  });
});

describe("the other models", () => {
  const onCard = { gpu: rtx3070 };

  it("a size that was measured reads as it was measured", () => {
    assert.equal(planMeasuredModel("opendde", { tokens: 600 }, onCard).needBytes, 4846 * MIB);
    assert.equal(planMeasuredModel("boltz2", { tokens: 300 }, onCard).needBytes, 1003 * MIB);
    assert.equal(measuredFoldSeconds("mol1", { tokens: 300 }), 24);
  });

  it("past the last size the last line goes on, and more residues never cost less", () => {
    for (const family of ["boltz2", "opendde", "monomer", "multimer", "bioemu2", "intellifold2"]) {
      let needed = 0;
      let taken = 0;
      for (let tokens = 76; tokens <= 2000; tokens += 50) {
        const plan = planMeasuredModel(family, { tokens }, { gpu: a10 });
        assert.ok(plan.needBytes >= needed, `${family} at ${tokens}`);
        assert.ok(plan.seconds >= taken, `${family} at ${tokens}`);
        needed = plan.needBytes;
        taken = plan.seconds;
      }
    }
  });

  it("AlphaFold 3 reads as OpenBind-0, whose weights are not the reader's to accept", () => {
    assert.deepEqual(planMeasuredModel("af3", { tokens: 400 }, onCard), planMeasuredModel("openbind0", { tokens: 400 }, onCard));
  });

  it("a model nobody measured has no line", () => {
    assert.equal(planMeasuredModel("not-a-model", { tokens: 400 }, onCard), undefined);
    assert.equal(measuredFoldSeconds("not-a-model", { tokens: 400 }), undefined);
  });

  it("what does not fit says so, and the rest say what they will hold and take", () => {
    const fits = describePlan(planMeasuredModel("boltz2", { tokens: 300 }, onCard));
    assert.equal(fits.tone, "plain");
    assert.match(fits.text, /^1\.0 of 4\.8 GiB · about \d+ s$/);
    const over = planMeasuredModel("opendde", { tokens: 700 }, onCard);
    assert.equal(over.over, true);
    assert.equal(describePlan(over).tone, "warn");
    assert.match(describePlan(over).text, /^Needs .* of 4\.8 GiB$/);
  });

  it("every way of having no GPU says it folds on one only", () => {
    const cases = [{ preference: "cpu", gpu: rtx3070 }, {}, { gpu: { ...rtx3070, refusing: true } }, { gpu: { ...rtx3070, software: true } }];
    for (const machine of cases) {
      const shown = describePlan(planMeasuredModel("boltz2", { tokens: 300 }, machine));
      assert.equal(shown.text, "Folds on a GPU only");
      assert.equal(shown.tone, "warn");
      assert.ok(shown.title.length > 0);
    }
  });

  it("five models in one press take five times as long, in the memory of one", () => {
    const one = planMeasuredModel("monomer", { tokens: 300 }, onCard);
    const five = planMeasuredModel("monomer", { tokens: 300, models: 5 }, onCard);
    assert.equal(five.seconds, one.seconds * 5);
    assert.equal(five.needBytes, one.needBytes);
  });

  it("a fold of any model teaches the card's rate, for all of them and for ESMFold2", () => {
    const storage = fakeStorage();
    noteFold({ where: "gpu", card: rtx3070.name, family: "opendde", tokens: 300, loops: 4, seconds: 72 }, storage);
    const rates = readFoldRates(storage);
    assert.equal(rates.cards[rtx3070.name], 2);
    assert.equal(planMeasuredModel("boltz2", { tokens: 300 }, { gpu: rtx3070, rates }).seconds, 22);
    assert.equal(planEsmfold2({ tokens: 300 }, { gpu: rtx3070, rates }).seconds, 2 * gpuFoldSeconds({ tokens: 300 }));
  });

  it("a model nobody measured teaches nothing", () => {
    const storage = fakeStorage();
    noteFold({ where: "gpu", card: rtx3070.name, family: "not-a-model", tokens: 300, loops: 4, seconds: 72 }, storage);
    assert.deepEqual(readFoldRates(storage).cards, {});
  });
});
