import { describe, expect, it } from "./harness.js";
import { readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";

/**
 * EVERY SHIPPING MODULE PARSES.
 *
 * 🔴 THIS EXISTS BECAUSE A BROKEN MODULE WAS COMMITTED AND PUSHED. A comment in
 * src/triangle/shaders.js said `var<workgroup>` with backticks around it, and
 * the comment lives inside a JS template literal holding WGSL - so the
 * backticks closed the literal and the file stopped parsing. The whole suite
 * stayed green, because no test outside the GPU ones imports that file: the
 * shaders are strings handed to a device, so nothing on a machine without one
 * had ever needed to load the module. The site built, deployed, and the Fold
 * button silently did nothing.
 *
 * Importing every module under src/ costs a second and closes that gap for all
 * of them, not just the one that failed. It asserts nothing about behaviour -
 * only that the file is loadable, which is the property that was missing.
 */
const ROOT = resolve(import.meta.dirname, "..");
const SOURCE = join(ROOT, "src");

function javascriptFiles(directory) {
  const found = [];
  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    if (statSync(path).isDirectory()) found.push(...javascriptFiles(path));
    else if (entry.endsWith(".js")) found.push(path);
  }
  return found.sort();
}

const modules = javascriptFiles(SOURCE).map((path) => relative(ROOT, path));

describe("every module under src/ loads", () => {
  it("finds modules to check at all, so an empty sweep cannot pass", () => {
    expect(modules.length > 20).toBe(true);
  });

  it.each(modules)("imports %s", async(path) => {
    // ...a fresh URL each time so a module that threw once is not served from
    // the loader's cache as a success on a later run in the same process.
    await import(`${pathToFileURL(join(ROOT, path)).href}?parse-check`);
  });
});
