import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "./harness.js";
import { chainMaskFor, designChain } from "../src/design/mpnn-bridge.js";
import { DESIGNER_NAMES, DESIGNERS } from "../src/design/designers.js";
import { Model, sequenceToString } from "../src/design/mpnn/model.js";
import { NA_ALPHABET } from "../src/design/mpnn/na.js";
import { ALPHABET } from "../src/design/mpnn/constants.js";
import { Weights } from "../src/design/mpnn/weights.js";
import { structureFromText } from "../src/design/mpnn/pdb.js";
import { enableAcceleration } from "../src/design/mpnn/accel.js";
import { uniformFrom } from "../src/af3/fold.js";

const path = (name) => fileURLToPath(new URL(name, import.meta.url));

/**
 * A two-chain structure, from a one-chain crystal.
 *
 * 1qys (Top7) is already in the repository and is 92 residues of a real fold.
 * The second chain is that chain again, relabelled and moved 30 A along x -
 * far enough that no neighbour graph bridges the two, which is what makes it
 * usable as a FIXED chain whose letters must come back untouched. The point of
 * the fixture is the chain bookkeeping, not the biology.
 */
const CHAIN_RESIDUES = 40;

function twoChainPdb() {
  const lines = readFileSync(path("../1qys-crystal.pdb"), "utf8").split("\n");
  // ...the first 40 residues of it. The whole 92 make every assertion below
  // read identically and cost the CPU suite twenty seconds instead of five;
  // what is under test is the chain bookkeeping, which 40 residues exercise.
  const atoms = lines.filter((line) => line.startsWith("ATOM")
    && Number(line.slice(22, 26)) <= CHAIN_RESIDUES);
  const moved = atoms.map((line) => {
    const x = Number(line.slice(30, 38)) + 30;
    return `${line.slice(0, 21)}B${line.slice(22, 30)}`
      + `${x.toFixed(3).padStart(8)}${line.slice(38)}`;
  });
  return `${[...atoms, "TER", ...moved, "TER", "END"].join("\n")}\n`;
}

const pdb = twoChainPdb();

/**
 * The mirrored SIMD kernel, installed into `ops.linear` for this file.
 *
 * 🔴 IT IS ALSO THE ONLY THING THAT EXERCISES web/vendor/mpnn/kernels.wasm.
 * The page loads it by URL and falls back silently when it cannot, which means
 * a truncated or stale copy of it would never be noticed - so this asserts it
 * loaded, and every design below then runs through it. It halves the file's
 * runtime as a side effect.
 */
const accelerated = await enableAcceleration(
  readFileSync(path("../web/vendor/mpnn/kernels.wasm")).buffer,
);

/**
 * 🔴 READ, NOT FETCHED. `Weights.fetch` goes through the global fetch, which
 * node will not point at a file on disk. The page's loader and this one differ
 * only in how the bytes arrive.
 */
function designer(name = "soluble") {
  const bytes = readFileSync(path(`../web/public/mpnn/${DESIGNERS[name].file}`));
  return new Model(Weights.fromArrayBuffer(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
}

describe("the mirrored kernels", () => {
  it("load and install", () => {
    expect(accelerated === null).toBe(false);
  });
});

describe("chainMaskFor", () => {
  it("selects one chain's residues and no others", () => {
    const structure = structureFromText(pdb, { ligands: false });
    const mask = chainMaskFor(structure, "A");
    expect(structure.chainList).toEqual(["A", "B"]);
    let selected = 0;
    for (let index = 0; index < mask.length; index += 1) {
      if (mask[index] === 1) {
        selected += 1;
        expect(structure.chainIds[index]).toBe("A");
      }
    }
    expect(selected).toBe(mask.length / 2);
  });

  it("selects nothing for a chain the structure does not have", () => {
    const structure = structureFromText(pdb, { ligands: false });
    expect([...chainMaskFor(structure, "Z")].some((value) => value === 1)).toBe(false);
  });
});

describe("designChain", () => {
  const model = designer();
  const native = structureFromText(pdb, { ligands: false });
  const nativeA = native.sequence.slice(0, native.sequence.length / 2);
  const nativeB = native.sequence.slice(native.sequence.length / 2);

  it("returns a sequence the length of the designed chain", () => {
    const result = designChain(model, { pdb, chain: "A", random: uniformFrom(1) });
    expect(result.sequence).toHaveLength(nativeA.length);
    expect(result.designed).toBe(nativeA.length);
  });

  // 🔴 THE FIXED CHAIN COMING BACK UNCHANGED IS THE `S` CHECK. `Model.sample`
  // starts from all-X unless it is handed the structure's own sequence, and it
  // never decodes a position the chain mask excludes - so without `S` chain B
  // would come back as 92 X's and, worse, chain A would have been designed
  // against a target the model could not read. Nothing else in the loop would
  // have complained.
  it("leaves every chain it was not asked to design exactly as it found it", () => {
    const result = designChain(model, { pdb, chain: "A", random: uniformFrom(2) });
    const chains = result.full.split(":");
    expect(chains).toHaveLength(2);
    expect(chains[1]).toBe(nativeB);
    expect(chains[0]).toBe(result.sequence);
  });

  it("designs the other chain when asked for it", () => {
    const result = designChain(model, { pdb, chain: "B", random: uniformFrom(3) });
    expect(result.full.split(":")[0]).toBe(nativeA);
  });

  it("never draws X, which is an input letter and not a design choice", () => {
    const result = designChain(model, { pdb, chain: "A", random: uniformFrom(4) });
    expect(result.sequence.includes("X")).toBe(false);
  });

  it("honours the omissions it is given", () => {
    const result = designChain(model, {
      pdb, chain: "A", omit: "CPW", temperature: 0.3, random: uniformFrom(5),
    });
    expect(/[CPW]/.test(result.sequence)).toBe(false);
  });

  // A large negative bias is the same mechanism as an omission, so it is
  // checkable the same way - which is what makes the ramp meaningful.
  it("suppresses alanine when the bias is strongly negative", () => {
    const biased = designChain(model, {
      pdb, chain: "A", alanineBias: -1e6, temperature: 0.3, random: uniformFrom(6),
    });
    expect(biased.sequence.includes("A")).toBe(false);
  });

  it("is reproducible from a seeded generator", () => {
    const first = designChain(model, { pdb, chain: "A", temperature: 0.5, random: uniformFrom(9) });
    const again = designChain(model, { pdb, chain: "A", temperature: 0.5, random: uniformFrom(9) });
    expect(again.sequence).toBe(first.sequence);
  });

  it("names the chains it does have when asked for one it does not", () => {
    expect(() => designChain(model, { pdb, chain: "Z" })).toThrow(/no chain Z.*A, B/s);
  });
});

/**
 * 🔴 EVERY FAMILY THE PAGE SHIPS, ON THE SAME PROTEIN FIXTURE.
 *
 * The picker can reach four checkpoints and three of them were unreachable
 * from any test. Two things differ per family and both are silent when wrong:
 *
 *   * the PARSE. LigandMPNN wants heteroatoms kept, NA-MPNN wants nucleic
 *     residues promoted to positions, and handing a model the wrong reading
 *     produces a plausible answer against an incomplete structure.
 *   * the ALPHABET. NA-MPNN's is 33 letters, `Model.sample` indexes the bias
 *     as `bias[position * V + letter]` with V from the CHECKPOINT, and
 *     `result.seq` is built against the 21-letter alphabet whatever the model
 *     is - while NA_ALPHABET's protein letters are the SAME 21 IN A DIFFERENT
 *     ORDER (`ARNDCQEGHILKMFPSTWYVX` against `ACDEFGHIKLMNPQRSTVWYX`).
 *
 * 🔴 SO "IS IT MADE OF AMINO ACIDS" PROVES NOTHING HERE, WHICH IS HOW THIS
 * TEST FIRST PASSED WITHOUT MEANING IT. Both failures produce a valid protein
 * sequence of the right length. Measured on this fixture: the correct reading
 * gives `SKKITVTIKSKDKTKTITYEV...`, the 21-letter reading of the same tokens
 * gives `SNNLTYTLNSNENTNTLTWHY...`, and a 21-wide bias gives
 * `SARVTVTITEADTTRTLTAE` and then fourteen alanines. Every one of those is
 * twenty amino acids at the right length. What discriminates is the TOKENS
 * and the alphabet they were read in, which is why designChain returns both.
 */
describe.each(DESIGNER_NAMES)("designChain under %s", (name) => {
  const native = structureFromText(pdb, { ligands: false });
  const nativeB = native.sequence.slice(native.sequence.length / 2);

  it("designs a protein sequence and holds the other chain", () => {
    const result = designChain(designer(name), {
      pdb, chain: "A", temperature: 0.2, random: uniformFrom(21),
    });
    // ...the chain's real length, which is not CHAIN_RESIDUES: 1qys is
    // numbered from 4, so its first 40 residue NUMBERS are 37 residues.
    expect(result.sequence).toHaveLength(nativeB.length);
    expect(/^[ACDEFGHIKLMNPQRSTVWY]+$/.test(result.sequence)).toBe(true);
    expect(result.full.split(":")[1]).toBe(nativeB);
  });

  it("reads its tokens in the alphabet its checkpoint was trained on", () => {
    const model = designer(name);
    const result = designChain(model, {
      pdb, chain: "A", temperature: 0.2, random: uniformFrom(21),
    });
    // The bias must have been the width the checkpoint indexes it at, or the
    // sample above read across position boundaries.
    expect(result.alphabet).toHaveLength(model.numLetters);
    expect(result.alphabet).toBe(model.isNA ? NA_ALPHABET : ALPHABET);
    // ...and the letters handed back are those tokens under THAT alphabet.
    const expected = [...result.tokens].slice(0, nativeB.length)
      .map((token) => result.alphabet[token]).join("");
    expect(result.sequence).toBe(expected);
    // For NA-MPNN the wrong reading is a different protein, not a broken one -
    // so the test that the branch exists is that the two disagree.
    if (model.isNA) {
      expect(result.sequence === sequenceToString(result.tokens).slice(0, nativeB.length))
        .toBe(false);
    }
  });
});
