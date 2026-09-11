/**
 * Every shader reached through a FOLDED grid guards its index.
 *
 * 🔴 THIS IS THE 160-RESIDUE AF2 RACE, AND IT WAS HUNTED AS A MISSING BARRIER.
 * `linearGrid` rounds twice - elements up to a whole workgroup, then workgroups
 * up to a whole row of GRID_WIDTH - so once the y fold engages the dispatch is a
 * MULTIPLE of 32,768 workgroups and almost never the count that was wanted. The
 * shaders below all run under it, and one of them had no bounds check:
 * `base[index] += update[index]`, with 917,504 invocations past the end.
 *
 * WGSL leaves an out-of-bounds write either discarded or clamped into the
 * buffer, and different backends pick differently: Vulkan discards through
 * robustBufferAccess, Metal takes an explicit index clamp. Clamped, every
 * excess invocation read-modify-writes ONE address - a different answer every
 * pass, on one machine and not the other, which is exactly what the
 * specification permits. An A100 agreeing with itself proved nothing.
 *
 * So the rule is structural rather than numeric: a folded index is guarded on
 * the next line, always, and this fails when the next one is not. docs/AF2.md
 * has the audit that missed it - it asked whether the shader reads `id.y`, which
 * this one does.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

import { WebGpuExecution } from "../src/runtime/execution.js";

// Every way this repository folds a dispatch into y. `linearGrid` folds an
// INVOCATION index with the workgroup size in the stride - 64 everywhere but the
// diffusion transformer, which templates it - and `rowGrid` folds a WORKGROUP
// index with no stride at all. The optional paren is not cosmetic either: a tile
// grid writes `let base_row = (group.x + group.y * GRID_WIDTH) * ROWS;`.
//
// 🔴 THIS PATTERN WAS TOO NARROW TWICE WHILE IT WAS BEING WRITTEN - first
// matching only the invocation convention, then matching 23 of the 48 workgroup
// folds because of that paren. A scan that silently covers half the code is the
// exact failure this file exists to prevent, which is what the floor below is for.
const FOLDED = /let\s+(\w+)\s*=\s*\(?\s*(\w+)\.x\s*\+\s*\2\.y\s*\*\s*[^;]*GRID_WIDTH[^;]*;/;

const sourceFiles = (directory) => readdirSync(directory).flatMap((entry) => {
  const path = join(directory, entry);
  if (statSync(path).isDirectory()) return sourceFiles(path);
  return path.endsWith(".js") ? [path] : [];
});

describe("a folded linear grid", () => {
  // Called off the prototype: it reads no instance state, and a device is the
  // one thing the CPU suite cannot have.
  const linearGrid = (elements) => WebGpuExecution.prototype.linearGrid.call(null, elements);

  it("covers every element it was asked for", () => {
    for (const elements of [1, 63, 64, 65, 445_568, 2_097_152, 3_276_800, 87_120_000]) {
      const [x, y] = linearGrid(elements);
      assert.ok(x * y * 64 >= elements, `${elements} is not covered by ${x}x${y}`);
      assert.ok(x <= 65_535 && y <= 65_535, `${elements} needs ${x}x${y}, over the dispatch limit`);
    }
  });

  // The pair tensor is `L * L * 128` elements, so `2 * L * L` workgroups - which
  // crosses GRID_WIDTH at exactly L = 128. That is the boundary the race was
  // reported on: 128 residues passes, 160 does not.
  const overrun = (length) => {
    const elements = length * length * 128;
    const [x, y] = linearGrid(elements);
    return x * y * 64 - elements;
  };

  it("dispatches exactly as many invocations as there are elements, up to 128 residues", () => {
    for (const length of [59, 80, 100, 128]) {
      assert.equal(overrun(length), 0, `${length} residues should need no guard`);
    }
  });

  it("OVERRUNS past 128 residues, which is why the guard has to be there", () => {
    assert.equal(overrun(129), 2_064_256);
    assert.equal(overrun(160), 917_504);
    assert.equal(overrun(200), 1_171_456);
    assert.equal(overrun(400), 491_520);
    assert.equal(overrun(825), 960_384);
  });
});

describe("every shader that folds its index", () => {
  const guarded = [];
  const unguarded = [];
  for (const path of sourceFiles("src")) {
    const lines = readFileSync(path, "utf8").split("\n");
    for (const [at, line] of lines.entries()) {
      const match = FOLDED.exec(line);
      if (match === null) continue;
      // 🔴 THE BOUND IS OFTEN ON A DERIVED NAME, NOT ON THE INDEX. A tile grid
      // reads `let tile = group.x + group.y * GRID_WIDTH; let first = tile * ROWS;
      // if (first >= PAIRS)`, and shaders like that look unguarded to a rule that
      // only watches the index itself. So follow the derivation: a compare against
      // anything computed from the folded index counts, and what fails is reaching
      // a buffer subscript without having compared any of them.
      const tainted = new Set([match[1]]);
      const touches = (text) => [...tainted].some((n) => new RegExp(`\\b${n}\\b`).test(text));
      let bounded = false;
      let settled = false;
      for (const line of lines.slice(at + 1, at + 12)) {
        if (settled) break;
        for (const statement of line.split(";")) {
          const text = statement.trim();
          if (text === "" || text.startsWith("//")) continue;
          if (!touches(text)) continue;
          // 🔴 AND WITHIN ONE STATEMENT THE ORDER STILL DECIDES. The confidence
          // head writes `if (index < arrayLength(&source)) { output[index] = ... }`
          // - guard and subscript on one line - so asking "does this subscript?"
          // before "does it compare?" calls a correct shader broken.
          const subscript = text.search(
            new RegExp(`\\[[^\\]]*\\b(${[...tainted].join("|")})\\b`));
          const compare = text.search(/(>=|<)/);
          if (compare >= 0 && (subscript < 0 || compare < subscript)) {
            bounded = true; settled = true; break;
          }
          if (subscript >= 0) { settled = true; break; }
          const assigned = /^(?:let|var)\s+(\w+)\s*=/.exec(text);
          if (assigned !== null) tainted.add(assigned[1]);
        }
      }
      (bounded ? guarded : unguarded).push(`${path}:${at + 1}`);
    }
  }

  it("checks its bounds before it indexes anything", () => {
    assert.deepEqual(unguarded, []);
  });

  // 🔴 A GATE THAT CANNOT FAIL IS NOT A GATE. If the regex stops matching the
  // shaders - a rename, a reformat - the check above passes by finding nothing.
  it("found all three folding conventions, not just one", () => {
    // 122 today: 79 invocation folds, 41 workgroup folds, 2 templated-lane ones.
    assert.ok(guarded.length >= 120, `only ${guarded.length} folded indices found`);
  });
});
