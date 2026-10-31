import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { combineStates, designChain } from "../cpu/design/mpnn-bridge.js";
import { DESIGNERS } from "../cpu/design/designers.js";
import { Model } from "../cpu/design/mpnn/model.js";
import { Weights } from "../cpu/design/mpnn/weights.js";
import { enableAcceleration } from "../cpu/design/mpnn/accel.js";
import { uniformFrom } from "../webgpu/af3/fold.js";

const path = (name) => fileURLToPath(new URL(name, import.meta.url));
const wasm = readFileSync(path("../web/vendor/mpnn/kernels.wasm"));
await enableAcceleration(wasm.buffer.slice(wasm.byteOffset, wasm.byteOffset + wasm.byteLength));
const bytes = readFileSync(path(`../web/public/mpnn/${DESIGNERS.soluble.file}`));
const model = new Model(Weights.fromArrayBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));

// Top7's first 30 residues as the designed chain A, and residues 61-90 as a partner B
const atoms = readFileSync(path("../tools/fixtures/1qys-crystal.pdb"), "utf8").split("\n").filter((l) => l.startsWith("ATOM"));
const residue = (l) => Number(l.slice(22, 26));
const complex = (shift = 0) => [
  ...atoms.filter((l) => residue(l) <= 30),
  ...atoms.filter((l) => residue(l) > 60 && residue(l) <= 90).map((l) => `${l.slice(0, 21)}B${l.slice(22)}`),
].map((l) => (shift === 0 ? l : `${l.slice(0, 30)}${(Number(l.slice(30, 38)) + shift).toFixed(3).padStart(8)}${l.slice(38)}`))
  .join("\n") + "\nEND\n";
/** The same complex with chain A's residues' coordinates handed to each other in a scrambled order: another state. */
function scrambled(seed) {
  const random = uniformFrom(seed);
  const lines = complex().split("\n");
  const own = [...new Set(lines.filter((l) => l[21] === "A").map(residue))];
  const to = new Map(own.map((r) => [r, own[Math.floor(random() * own.length)]]));
  const ca = new Map(lines.filter((l) => l[21] === "A" && l.slice(12, 16).trim() === "CA").map((l) => [residue(l), l.slice(30, 54)]));
  return lines.map((l) => (l[21] !== "A" || !l.startsWith("ATOM") ? l
    : `${l.slice(0, 30)}${ca.get(to.get(residue(l)))}${l.slice(54)}`.replace(/ CA /, " CA ")))
    .join("\n");
}

test("states are laid side by side: the first as it is, the rest moved 1000 A and renamed", () => {
  const { text, letters } = combineStates([complex(), complex()], "A");
  assert.deepEqual(letters, ["A", "Z"]);
  const first = text.split("\n").find((l) => l.startsWith("ATOM") && l[21] === "A");
  const twin = text.split("\n").find((l) => l.startsWith("ATOM") && l[21] === "Z");
  assert.ok(Math.abs(Number(twin.slice(30, 38)) - Number(first.slice(30, 38)) - 1000) < 1e-3);
  assert.ok(text.split("\n").some((l) => l[21] === "Y"), "the partner B renamed too");
});

test("a tied design writes one sequence: every twin position takes its first state's letter, holds in both", () => {
  const out = designChain(model, { pdb: complex(), chain: "A", random: uniformFrom(4), hold: [0, 5],
    states: [{ pdb: scrambled(1) }] });
  const length = new Set(complex().split("\n").filter((l) => l[21] === "A").map(residue)).size;
  assert.equal(out.sequence.length, length);
  const [a, , z] = out.full.split(":");
  assert.equal(a, z, "the twin chain carries the same letters");
  // held positions keep the crystal's own residue (S at both the first state's and its twin's)
  const single = designChain(model, { pdb: complex(), chain: "A", random: uniformFrom(4), hold: [0, 5] });
  assert.equal(out.sequence[0], single.sequence[0]);
  assert.equal(out.sequence[5], single.sequence[5]);
});

test("a state's weight is what it contributes: at zero weight changing it changes nothing, at half it does", () => {
  const design = (other, weight) => designChain(model, { pdb: complex(), chain: "A", random: uniformFrom(9), weight: 1 - weight,
    temperature: 0.1, states: [{ pdb: other, weight }] }).sequence;
  assert.equal(design(scrambled(1), 0), design(scrambled(2), 0));
  assert.notEqual(design(scrambled(1), 0.5), design(scrambled(2), 0.5));
});

test("history: a design hands back its raw logits, and a logit bias steers the next one", () => {
  const first = designChain(model, { pdb: complex(), chain: "A", random: uniformFrom(3) });
  const V = first.logits.length / first.sequence.length;
  assert.equal(first.logits.length, first.sequence.length * V);
  assert.ok(first.logits.every(Number.isFinite));
  // a bias of +50 on tryptophan everywhere writes tryptophan everywhere (cysteine stays omitted)
  const W = "ACDEFGHIKLMNPQRSTVWYX".indexOf("W");
  const toward = new Float32Array(first.logits.length);
  for (let i = 0; i < first.sequence.length; i += 1) toward[i * V + W] = 50;
  assert.match(designChain(model, { pdb: complex(), chain: "A", random: uniformFrom(3), logitBias: toward }).sequence, /^W+$/);
  // ...and the same holds across tied states: the group carries the bias once, not once per state
  const tied = designChain(model, { pdb: complex(), chain: "A", random: uniformFrom(3), logitBias: toward,
    states: [{ pdb: scrambled(1) }] });
  assert.match(tied.sequence, /^W+$/);
  assert.throws(() => designChain(model, { pdb: complex(), chain: "A", logitBias: new Float32Array(3) }), /logit bias of 3/);
});
