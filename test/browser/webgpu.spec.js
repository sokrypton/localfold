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
