import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "./harness.js";
import { FOLDERS } from "../web/validate-design.js";

const read = (name) => readFileSync(fileURLToPath(new URL(name, import.meta.url)), "utf8");

describe("the validation model picker", () => {
  // 🔴 AN OPTION THAT IS NOT IN `FOLDERS` VALIDATES NOTHING, SILENTLY.
  // `validateDesign` returns null for a model it does not know, so the run
  // finishes with no number and no complaint - and the page had two such
  // options, Boltz-2 and Protenix-2, for as long as the control existed. A
  // control that silently does nothing is worse than an absent one.
  it("offers exactly the models that can validate", () => {
    const html = read("../design.html");
    const block = html.slice(html.indexOf('<select id="validate"'));
    const options = [...block.slice(0, block.indexOf("</select>"))
      .matchAll(/<option value="([^"]*)"/g)].map(([, value]) => value);
    // "" is "nothing", which is a real choice rather than a model.
    expect(options[0]).toBe("");
    expect(options.slice(1).sort()).toEqual(Object.keys(FOLDERS).sort());
  });

  // A default nobody can honour would be the same bug on first load.
  it("defaults to a model that can validate", () => {
    const html = read("../design.html");
    const block = html.slice(html.indexOf('<select id="validate"'));
    const chosen = /<option value="([^"]*)"[^>]*\bselected\b/.exec(
      block.slice(0, block.indexOf("</select>")));
    expect(chosen !== null).toBe(true);
    expect(Object.keys(FOLDERS).includes(chosen[1])).toBe(true);
  });
});
