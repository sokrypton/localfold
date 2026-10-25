// The same matrix through `output.js`'s solver - the reference half of the gate's `assign` arm.
import { linearSumAssignmentMinimize, linearSumAssignmentMaximize } from "../../../shared/rfd4p/output.js";

const [seed, nR, nC, mode, infEvery] = process.argv.slice(2);
let s = Number(seed) >>> 0;
const next = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return (s >>> 8) / 16777216; };
const rows = Number(nR);
const cols = Number(nC);
const every = Number(infEvery ?? 0);
const m = new Float64Array(rows * cols);
for (let i = 0; i < m.length; i += 1) {
  const v = next() * 20 - 10;
  m[i] = every && i % every === 0 ? Infinity : v;
}
const f = mode === "x" ? linearSumAssignmentMaximize : linearSumAssignmentMinimize;
const { rowInd, colInd } = f(m, rows, cols);
console.log(`${[...rowInd].map((r, k) => `${r}:${colInd[k]}`).join(" ")} `);
