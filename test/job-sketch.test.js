import test from "node:test";
import assert from "node:assert/strict";
import { ball, bridge, pieceLength, sketchJob } from "../web/job-sketch.js";

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const steps = (points) => points.slice(1).map((p, i) => dist(p, points[i]));
const xyz = (a) => [a.x, a.y, a.z];

/** A straight C-alpha trace of `n` residues on chain `chain`, numbered from `first`. */
function strand(chain, first, n, origin = [0, 0, 0]) {
  return Array.from({ length: n }, (_, i) => ({ chainId: chain, resId: first + i, resName: "ALA",
    atomName: "CA", element: "C", x: origin[0] + 3.8 * i, y: origin[1], z: origin[2], isHetero: false }));
}

test("a ball's steps are C-alpha steps, near enough to read as a chain", () => {
  for (const n of [8, 30, 100]) {
    const s = steps(ball(n, [0, 0, 0], [0, 0, 1]));
    assert.equal(s.length, n - 1);
    for (const step of s) assert.ok(step > 3.4 && step < 4.0, `n ${n}: step ${step}`);
  }
});

test("a bridge joins its two ends in n + 1 even 3.8 A steps", () => {
  const from = [0, 0, 0];
  const to = [10, 0, 0];
  const points = bridge(12, from, to, [0, 1, 0]);
  assert.equal(points.length, 12);
  for (const s of steps([from, ...points, to])) assert.ok(Math.abs(s - 3.8) < 0.15, `step ${s}`);
  // bowed the way it was told, not through the structure
  assert.ok(Math.max(...points.map((p) => p[1])) > 5);
});

test("a gap already longer than the linker is drawn straight", () => {
  const points = bridge(2, [0, 0, 0], [30, 0, 0], [0, 1, 0]);
  assert.ok(points.every((p) => Math.abs(p[1]) < 1e-9));
});

test("a range is sketched at its midpoint, a sequence at its length", () => {
  assert.equal(pieceLength({ kind: "generated", min: 50, max: 120 }), 85);
  assert.equal(pieceLength({ kind: "generated", min: 100, max: "" }), 100);
  assert.equal(pieceLength({ kind: "sequence", value: "ACDEF GH" }), 7);
  assert.equal(pieceLength({ kind: "segment", chain: "A", first: 1, last: 9 }), 0);
});

test("a de novo job is one ghost chain with nothing real", () => {
  const { atoms, ghosts } = sketchJob([], new Map(), [{ kind: "generated", min: 100, max: 100 }]);
  assert.equal(atoms.length, 100);
  assert.equal(ghosts.size, 1);
  assert.deepEqual([...ghosts.values()], [{ span: 0, chain: 0 }]);
});

test("a motif's linker runs from the end of one kept piece to the start of the next", () => {
  const structure = [...strand("A", 1, 10), ...strand("A", 30, 10, [34.2, 10, 0])];
  const chainOf = new Map([["A", "A"]]);
  const spans = [
    { kind: "generated", min: 6, max: 6 },
    { kind: "segment", chain: "A", first: 1, last: 10 },
    { kind: "generated", min: 8, max: 8 },
    { kind: "segment", chain: "A", first: 30, last: 39 },
  ];
  const { atoms, ghosts } = sketchJob(structure, chainOf, spans);
  assert.equal(ghosts.size, 2);
  const [nTerm, linker] = [...ghosts.keys()];
  assert.ok(![...ghosts.keys()].includes("A"), "a ghost never shares a letter with the structure");
  const of = (letter) => atoms.filter((a) => a.chainId === letter).map(xyz);
  const linkerPoints = of(linker);
  assert.equal(linkerPoints.length, 8);
  assert.ok(Math.abs(dist(linkerPoints[0], xyz(structure[9])) - 3.8) < 0.15);
  assert.ok(Math.abs(dist(linkerPoints[7], xyz(structure[10])) - 3.8) < 0.15);
  // the N-terminal piece ENDS beside residue A1
  const head = of(nTerm);
  assert.equal(head.length, 6);
  assert.ok(dist(head[5], xyz(structure[0])) < 4.5, `${dist(head[5], xyz(structure[0]))}`);
});

test("a binder chain is drawn beside the target's hotspots", () => {
  const structure = strand("B", 1, 40);
  const spans = [{ kind: "generated", min: 60, max: 60 }, { kind: "break" },
    { kind: "segment", chain: "B", first: 1, last: 40 }];
  const hot = [{ chain: "B", resId: 35 }];
  const { atoms, ghosts } = sketchJob(structure, new Map([["B", "B"]]), spans, { hotspots: hot });
  const letter = [...ghosts.keys()][0];
  const points = atoms.filter((a) => a.chainId === letter).map(xyz);
  const centre = points.reduce((s, p) => [s[0] + p[0], s[1] + p[1], s[2] + p[2]], [0, 0, 0]).map((v) => v / points.length);
  const hotspot = xyz(structure[34]);
  const far = xyz(structure[0]);
  assert.ok(dist(centre, hotspot) < dist(centre, far), "nearer the hotspot than the target's far end");
});

test("a pocket's design chain is drawn round its ligand", () => {
  const ligand = Array.from({ length: 6 }, (_, i) => ({ chainId: "B", resId: 1, resName: "SAM", atomName: `C${i}`,
    element: "C", x: 50 + i, y: 20, z: -10, isHetero: true }));
  const spans = [{ kind: "ligand", value: "SAM" }, { kind: "break" }, { kind: "generated", min: 120, max: 120 }];
  const { atoms, ghosts } = sketchJob(ligand, new Map([["B", "B"]]), spans);
  const letter = [...ghosts.keys()][0];
  const points = atoms.filter((a) => a.chainId === letter).map(xyz);
  const centre = points.reduce((s, p) => [s[0] + p[0], s[1] + p[1], s[2] + p[2]], [0, 0, 0]).map((v) => v / points.length);
  assert.ok(dist(centre, [52.5, 20, -10]) < 3, `centre ${centre}`);
});

// 🔴 A DNA OR RNA PIECE IS SKETCHED AS NUCLEOTIDES, which py2Dmol draws as nucleic: it was GLY
// C-alphas, a protein trace, whatever "Made of" said. A designed piece has no letters yet and is the
// dictionary's unknown nucleotide - "D" alone, which `"ACGT".includes("")` once gave it, is no residue.
test("a nucleic piece is a strand of C4's, named by its bases", () => {
  const spans = [{ kind: "generated", min: 4, max: 4, polymer: "dna" }, { kind: "break" },
    { kind: "sequence", value: "ACGU", polymer: "rna" }, { kind: "break" }, { kind: "generated", min: 3, max: 3 }];
  const { atoms, ghosts } = sketchJob([], new Map(), spans);
  const [dna, rna, protein] = [...ghosts.keys()].map((letter) => atoms.filter((a) => a.chainId === letter));
  assert.deepEqual(dna.map((a) => `${a.resName}:${a.atomName}`), Array(4).fill("DN:C4'"));
  assert.deepEqual(rna.map((a) => a.resName), ["A", "C", "G", "U"]);
  assert.deepEqual(protein.map((a) => `${a.resName}:${a.atomName}`), Array(3).fill("GLY:CA"));
  // ...a strand's step, not a C-alpha's (6.5 A along the ball's arc, so a little less as a chord)
  const step = dist(xyz(dna[0]), xyz(dna[1]));
  assert.ok(step > 5.5 && step <= 6.5, `step ${step}`);
});
