import test from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

/**
 * 🔴 EVERY DOCUMENT IS IN CLAUDE.md's INDEX, BECAUSE THE INDEX IS HOW ANY OF
 * THEM IS FOUND.
 *
 * `docs/RUNNING.md` was not. It is the USER-facing half - running the site from
 * a checkout, the Linux/NVIDIA flags, using the kernels as a library - and
 * README.md links it twice, so it was reachable from outside the repository and
 * invisible from inside it. A document nobody working here can find is a
 * document that goes stale, and this file's neighbours are five hundred
 * measurements that only mean anything while they are current.
 *
 * The rule is deliberately weak: it asks that the FILE be named, not that the
 * description be good. A gate cannot read prose, and one that pretended to
 * would be the kind of gate this repository keeps finding cannot fail.
 */
test("every docs/ file is named in CLAUDE.md's index", () => {
  const index = readFileSync(join(ROOT, "CLAUDE.md"), "utf8");
  const table = index.slice(index.indexOf("| doc | what is in it |"));
  assert.ok(table.length > 0, "CLAUDE.md has no `| doc | what is in it |` table");

  const documents = readdirSync(join(ROOT, "docs")).filter((name) => name.endsWith(".md"));
  assert.ok(documents.length >= 10, `only ${documents.length} documents walked`);

  const missing = documents.filter((name) => !table.includes(`docs/${name}`));
  assert.deepEqual(missing, [],
    `${missing.length} document(s) in docs/ that CLAUDE.md's index does not name`);
});

/**
 * 🔴 AND THE INDEX NAMES NOTHING THAT IS GONE, which is the same rule read
 * backwards and the one that catches a rename. A row pointing at a file that
 * moved is worse than a missing row: it reads as an answer.
 */
test("CLAUDE.md's index names no document that does not exist", () => {
  const index = readFileSync(join(ROOT, "CLAUDE.md"), "utf8");
  const table = index.slice(index.indexOf("| doc | what is in it |"));
  const documents = new Set(readdirSync(join(ROOT, "docs")));
  const named = [...table.matchAll(/^\|\s*`docs\/([A-Za-z0-9_-]+\.md)`/gm)].map((m) => m[1]);
  assert.ok(named.length >= 10, `only ${named.length} rows in the index`);
  const gone = named.filter((name) => !documents.has(name));
  assert.deepEqual(gone, [], `${gone.length} index row(s) name a document that is gone`);
});
