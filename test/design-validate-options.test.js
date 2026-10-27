import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "./harness.js";
import { FOLDERS } from "../web/validate-design.js";

const read = (name) => readFileSync(fileURLToPath(new URL(name, import.meta.url)), "utf8");

/** A select's option values and its selected one, from design.html. */
function picker(id) {
  const html = read("../design.html");
  const block = html.slice(html.indexOf(`<select id="${id}"`));
  const body = block.slice(0, block.indexOf("</select>"));
  return {
    options: [...body.matchAll(/<option value="([^"]*)"/g)].map(([, value]) => value),
    chosen: /<option value="([^"]*)"[^>]*\bselected\b/.exec(body)?.[1],
  };
}

describe("the design page's model pickers", () => {
  // 🔴 AN OPTION THAT IS NOT IN `FOLDERS` CANNOT FOLD. The refold picker had two such options,
  // Boltz-2 and Protenix-2, for as long as it existed and they validated nothing; the hunt's "Folds
  // with" offered the same two and `hunt` threw "no folder boltz2" on them.
  it("the refold offers exactly the models that can refold", () => {
    const { options, chosen } = picker("refold-model");
    expect(options.slice().sort()).toEqual(Object.keys(FOLDERS).sort());
    expect(Object.keys(FOLDERS).includes(chosen)).toBe(true);
  });

  it("the hunt folds only with models it has", () => {
    const { options, chosen } = picker("folders");
    expect(options.every((value) => FOLDERS[value] !== undefined)).toBe(true);
    expect(FOLDERS[chosen] !== undefined).toBe(true);
  });
});
