import { expect, test } from "@playwright/test";

// The kernel checks live on dev.html: the folding page is one sequence box and
// a button now, and has no diagnostic to drive.
test("runs the triangle kernel in a browser WebGPU implementation", async({ page }) => {
  await page.goto("/dev.html?autorun=0&length=5&cz=7&hidden=6&precision=f32");
  const available = await page.evaluate(async() => {
    if (navigator.gpu === undefined) return false;
    return (await navigator.gpu.requestAdapter()) !== null;
  });
  test.skip(!available, "the installed browser has no usable WebGPU adapter");
  await page.getByRole("button", { name: "Run kernel" }).click();
  await expect(page.locator("#status")).toHaveAttribute("data-state", "passed");
  const metrics = await page.evaluate(() => window.__LOCALFOLD_RESULT__);
  expect(metrics?.meanAbsoluteError).toBeLessThan(1e-5);
  expect(metrics?.maxAbsoluteError).toBeLessThan(1e-4);
});

test("runs fp16 inputs and weights when the browser exposes shader-f16", async({ page }) => {
  await page.goto("/dev.html?autorun=0&length=8&cz=8&hidden=8&precision=f16");
  const hasF16 = await page.evaluate(async() => {
    const adapter = await navigator.gpu?.requestAdapter();
    return adapter?.features.has("shader-f16") ?? false;
  });
  test.skip(!hasF16, "the browser adapter does not expose shader-f16");
  await page.getByRole("button", { name: "Run kernel" }).click();
  await expect(page.locator("#status")).toHaveAttribute("data-state", "passed");
  const metrics = await page.evaluate(() => window.__LOCALFOLD_RESULT__);
  expect(metrics?.meanAbsoluteError).toBeLessThan(1e-3);
  expect(metrics?.maxAbsoluteError).toBeLessThan(1e-2);
});

test("shows the single-sequence folding page", async({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /AlphaFold2/ })).toBeVisible();
  await expect(page.locator("#sequence-length")).toHaveText("59 residues");
  await expect(page.getByRole("button", { name: "Fold" })).toBeVisible();
  await expect(page.locator("#results")).toBeHidden();
  // ...the residue count follows the box, which is the only live wiring the
  // page has before a GPU is involved.
  await page.locator("#sequence").fill("ACDEFGHIKL");
  await expect(page.locator("#sequence-length")).toHaveText("10 residues");
});

// THE MUTATION PANEL, DRIVEN AS A READER DRIVES IT.
//
// Built by web/mutate.js; needs no GPU, no weights and no fold - the viewer only
// supplies WHICH residue was clicked, and that rule is covered in
// test/mutate.test.js against a stubbed pick. What is left is the part only a
// browser has: where it sits, what it reports, and - the reason it stopped
// being a popup - what does NOT close it.
test("docks a mutation panel under the viewer and stages a pick", async({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async() => {
    const { createMutationPanel } = await import("/web/mutate.js");
    const host = document.getElementById("mutate-host");
    const viewer = document.getElementById("viewer");
    const picks = [];
    const panel = createMutationPanel(host, (i, r) => picks.push([i, r]));
    const before = { hidden: panel.element.hidden };

    panel.show(41, "W", "TRP 42");
    const box = viewer.getBoundingClientRect();
    const at = panel.element.getBoundingClientRect();
    const opened = {
      hidden: panel.element.hidden,
      title: panel.element.querySelector(".mutate-title").textContent,
      label: panel.element.getAttribute("aria-label"),
      cells: panel.element.querySelectorAll(".mutate-cell").length,
      current: [...panel.element.querySelectorAll(".mutate-cell.is-current")]
        .map((cell) => cell.dataset.residue),
      // ORDERED BY HYDROPATHY, so the row reads as the scale it is coloured by
      order: [...panel.element.querySelectorAll(".mutate-cell")].map((cell) => cell.dataset.residue).join(""),
      // ...each painted the colour its side chain takes in the viewer
      coloured: [...panel.element.querySelectorAll(".mutate-cell")]
        .filter((cell) => /^#[0-9a-f]{6}$/.test(cell.style.getPropertyValue("--cell"))).length,
      isoleucine: panel.element.querySelector('.mutate-cell[data-residue="I"]')
        .style.getPropertyValue("--cell"),
      arginine: panel.element.querySelector('.mutate-cell[data-residue="R"]')
        .style.getPropertyValue("--cell"),
      // DOCKED: directly beneath the viewer and the same width, not floating
      // over it. Within a pixel, because the two borders overlap by one.
      belowViewer: Math.abs(at.top - box.bottom) <= 2,
      sameWidth: Math.abs(at.width - box.width) <= 2,
      outsideViewer: !viewer.contains(panel.element),
    };

    panel.element.querySelector('.mutate-cell[data-residue="A"]').click();
    const afterPick = {
      picks: [...picks],
      hidden: panel.element.hidden,
      staged: panel.staged,
      marked: [...panel.element.querySelectorAll(".mutate-cell.is-current")]
        .map((cell) => cell.dataset.residue),
    };

    // 🔴 THE WHOLE REASON IT IS NOT A POPUP. Dragging the structure is a press
    // outside the panel, and the popup treated that as a dismissal - so looking
    // at the residue you had just clicked closed the menu for changing it.
    viewer.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    document.body.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    const afterRotate = { hidden: panel.element.hidden, staged: panel.staged };

    // ...and it survives the viewer being emptied, which every fold does. It is
    // a sibling, so there is nothing to re-attach.
    viewer.replaceChildren();
    const afterRebuild = { connected: panel.element.isConnected, hidden: panel.element.hidden };

    // ONE FOLD BUTTON ON THE PAGE, and it is not this one. The panel had a
    // duplicate; it went, along with the habit of hiding itself when pressed.
    const noFoldButton = panel.element.querySelector(".mutate-fold") === null;

    panel.show(7, "G", "GLY 8");
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    const afterEscape = { picks: picks.length, hidden: panel.element.hidden };

    return { before, opened, afterPick, afterRotate, afterRebuild, noFoldButton, afterEscape };
  });

  expect(result.before.hidden).toBe(true);
  expect(result.opened.hidden).toBe(false);
  expect(result.opened.title).toBe("TRP 42 →");
  expect(result.opened.label).toBe("Mutate TRP 42");
  expect(result.opened.cells).toBe(20);
  expect(result.opened.order).toBe("IVLFCMAGTSWYPHDENQKR");   // 4.5 down to -4.5
  expect(result.opened.current).toEqual(["W"]);
  expect(result.opened.coloured).toBe(20);
  expect(result.opened.isoleucine).toBe("#f2994a");   // 4.5, very hydrophobic
  expect(result.opened.arginine).toBe("#187bd1");     // -4.5, very hydrophilic
  expect(result.opened.belowViewer).toBe(true);
  expect(result.opened.sameWidth).toBe(true);
  expect(result.opened.outsideViewer).toBe(true);
  // a pick stages and LEAVES the panel up, so Fold beside it stays reachable
  expect(result.afterPick.picks).toEqual([[41, "A"]]);
  expect(result.afterPick.hidden).toBe(false);
  expect(result.afterPick.staged).toBe("A");
  expect(result.afterPick.marked).toEqual(["A"]);
  // rotating the structure must not dismiss it
  expect(result.afterRotate).toEqual({ hidden: false, staged: "A" });
  expect(result.afterRebuild).toEqual({ connected: true, hidden: false });
  expect(result.noFoldButton).toBe(true);
  expect(result.afterEscape).toEqual({ picks: 1, hidden: true });
});
