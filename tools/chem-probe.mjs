// The page's chemistry over SMILES on stdin, one line each - cuda/featurise/chem_probe.cpp's twin, for tools/check-native-featuriser.py.
import { readFileSync } from "node:fs";
import { smilesComponent } from "../shared/chem/component.js";
for (const line of readFileSync(0, "utf8").split("\n")) {
  if (!line) continue;
  try {
    const c = await smilesComponent(line, { code: "LIG" });
    console.log("OK" + c.atoms.map((a) => ` ${a.name}:${a.element}:${a.charge}:${a.x}:${a.y}:${a.z}`).join("") + " |"
      + c.bonds.map((b) => ` ${b.from}-${b.to}:${b.order}`).join(""));
  } catch (e) { console.log("ERR " + e.message); }
}
