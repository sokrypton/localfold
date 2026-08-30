// The test vocabulary this suite uses, over node's built-in runner.
//
// WHY A SHIM RATHER THAN A REWRITE. Removing the npm toolchain meant leaving
// vitest, and the mechanical translation of 150 assertions into assert calls is
// exactly the edit that quietly changes a tolerance - which AGENTS.md forbids
// for a reason. The matcher surface here is small and closed (this file is the
// whole of it), so the tests keep the assertions they were verified with and
// only their import line moved.
//
// Run them with:  node --test --test-concurrency=1 test/*.test.js
import { after, afterEach, before, beforeEach, describe as nodeDescribe, it as nodeIt } from "node:test";
import assert from "node:assert/strict";

export { after as afterAll, before as beforeAll, beforeEach, afterEach };

/** vitest's `%s`/`$property` title interpolation, for `it.each`. */
function title(template, row) {
  return template
    .replace(/\$([A-Za-z0-9_]+)/g, (match, key) => (row?.[key] !== undefined ? String(row[key]) : match))
    .replace(/%[sdi]/, () => (typeof row === "object" ? JSON.stringify(row) : String(row)));
}

function each(register) {
  return (rows) => (template, body) => {
    for (const row of rows) {
      const args = Array.isArray(row) ? row : [row];
      register(title(template, row), () => body(...args));
    }
  };
}

export const describe = Object.assign(
  (name, body) => nodeDescribe(name, body),
  {
    skipIf: (condition) => (name, body) => nodeDescribe(name, { skip: condition }, body),
    skip: (name, body) => nodeDescribe(name, { skip: true }, body),
    each: each((name, body) => nodeDescribe(name, body)),
  },
);

export const it = Object.assign(
  (name, body) => nodeIt(name, body),
  {
    skipIf: (condition) => (name, body) => nodeIt(name, { skip: condition }, body),
    skip: (name, body) => nodeIt(name, { skip: true }, body),
    each: each((name, body) => nodeIt(name, body)),
  },
);

/**
 * The matchers this suite actually uses, and no others - an unknown matcher
 * should be a missing-property TypeError here rather than a silent pass.
 */
export function expect(actual) {
  return {
    toBe: (expected) => assert.strictEqual(actual, expected),
    toEqual: (expected) => assert.deepStrictEqual(actual, expected),
    toBeLessThan: (bound) => assert.ok(actual < bound, `expected ${actual} < ${bound}`),
    toBeGreaterThan: (bound) => assert.ok(actual > bound, `expected ${actual} > ${bound}`),
    toBeCloseTo: (expected, digits = 2) => {
      const tolerance = 10 ** -digits / 2;
      assert.ok(Math.abs(actual - expected) < tolerance,
        `expected ${actual} within ${tolerance} of ${expected}`);
    },
    toHaveLength: (length) => assert.strictEqual(actual.length, length),
    toContain: (value) => assert.ok(
      typeof actual === "string" ? actual.includes(value) : Array.from(actual).includes(value),
      `expected ${typeof actual === "string" ? JSON.stringify(actual) : actual} to contain ${JSON.stringify(value)}`,
    ),
    toMatch: (pattern) => assert.match(actual, pattern),
    toThrow: (pattern) => assert.throws(actual, pattern),
  };
}

// --- the one piece of vitest's mocking this suite reaches for ---------------
//
// http-tensor-store.test.js stubs `fetch`. Nothing else mocks anything, so this
// is a global stash and a restore, not a mocking framework.
const stubbed = new Map();

export const vi = {
  fn: (implementation = () => undefined) => {
    const calls = [];
    const mock = (...args) => { calls.push(args); return implementation(...args); };
    mock.mock = { calls };
    return mock;
  },
  stubGlobal: (name, value) => {
    if (!stubbed.has(name)) stubbed.set(name, { present: name in globalThis, value: globalThis[name] });
    globalThis[name] = value;
  },
  unstubAllGlobals: () => {
    for (const [name, previous] of stubbed) {
      if (previous.present) globalThis[name] = previous.value;
      else delete globalThis[name];
    }
    stubbed.clear();
  },
};
