import { describe, expect, it } from "./harness.js";
import { readFileSync } from "node:fs";
import { parseSimpleYaml } from "../shared/bioemu2/potentials-yaml.js";
import { potentialSpecs, stepParameters, guidanceInterval } from "../shared/bioemu2/potentials.js";

/**
 * A BioEmu2 `--potentials` file reads as af3-any-model's load_potentials reads it:
 * tools/fixtures/bioemu2/potentials/reference.json is that port's own reading
 * (its kinds, constructor arguments, every parameter at seven times and the
 * guidance interval at three), or the error it raised.
 */
const fixture = (name) => readFileSync(new URL(`../tools/fixtures/bioemu2/potentials/${name}`, import.meta.url), "utf8");
const reference = JSON.parse(fixture("reference.json"));

describe("BioEmu2 potentials files", () => {
  for (const [name, want] of Object.entries(reference)) {
    it(`reads ${name} as the reference does`, () => {
      const read = () => potentialSpecs(parseSimpleYaml(fixture(name)));
      if (want.error) {
        expect(read).toThrow();
        return;
      }
      const specs = read();
      expect(specs.map(([kind]) => kind)).toEqual(want.kinds);
      expect(specs.map(([, , args]) => args)).toEqual(want.args);
      for (const [t, params] of Object.entries(want.params)) {
        const got = stepParameters(Number(t), specs);
        for (const [k, values] of Object.entries(params)) {
          expect(got[k].map((v) => (Number.isNaN(v) ? null : v))).toEqual(values);
        }
      }
      for (const [t, interval] of Object.entries(want.interval)) expect(guidanceInterval(Number(t), specs)).toEqual(interval);
    });
  }

  it("refuses what it does not read rather than half-reading it", () => {
    expect(() => parseSimpleYaml("potentials:\n\t- x: 1\n")).toThrow();
    expect(() => parseSimpleYaml("a: &anchor 1\n")).toThrow();
    expect(() => potentialSpecs(parseSimpleYaml("potentials: []\n"))).toThrow();
    expect(() => potentialSpecs(parseSimpleYaml("other: 1\n"))).toThrow();
  });
});
