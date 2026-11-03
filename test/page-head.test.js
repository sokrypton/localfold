// The head waits for what it asks, so nothing from another host may ask it to: a network that drops that host answers nothing at all,
// and the page stays blank until the browser gives up. tools/check-robustness.py holds the real thing, a-library-host-that-never-answers.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");

test("a script from another host does not hold the page up", () => {
  const scripts = html.match(/<script\b[^>]*\bsrc="https?:[^>]*>/g);
  assert.ok(scripts !== null, "the page loads none, so this reads nothing");
  for (const script of scripts) assert.match(script, /\b(async|defer)\b/, script);
});

test("a stylesheet from another host does not hold the page up", () => {
  const sheets = html.match(/<link\b[^>]*\bhref="https?:[^>]*>/g);
  assert.ok(sheets !== null, "the page loads none, so this reads nothing");
  for (const sheet of sheets) {
    assert.match(sheet, /\bmedia="print"/, sheet);
    assert.match(sheet, /\bonload="this\.media = 'all'"/, sheet);
  }
});
