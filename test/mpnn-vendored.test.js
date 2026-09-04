import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "./harness.js";

/**
 * The MPNN modules under src/design/mpnn/ are a mirror of a separate checkout,
 * not code that belongs to this repository.
 *
 * 🔴 A COPY NOBODY CHECKS IS A FORK. The failure this guards against is
 * ordinary and quiet: someone fixes a bug in the mirror because that is the
 * file the stack trace named, the upstream checkout never hears about it, and
 * the next `python3 tools/sync-mpnn.py` silently reverts the fix. So the sync
 * tool can verify itself, and this runs it.
 *
 * It SKIPS rather than fails when ../mpnn is not present: a clone of this
 * repository alone is a valid checkout and the mirror is complete in it. The
 * check only means anything where both trees exist.
 */
const root = fileURLToPath(new URL("..", import.meta.url));
const upstream = fileURLToPath(new URL("../../mpnn/mpnn/model.js", import.meta.url));

describe.skipIf(!existsSync(upstream))("the vendored MPNN mirror", () => {
  it("matches the upstream checkout", () => {
    const result = spawnSync("python3", ["tools/sync-mpnn.py", "--check"],
                             { cwd: root, encoding: "utf8" });
    expect(`${result.stdout}${result.stderr}`.trim().length > 0).toBe(true);
    if (result.status !== 0) {
      throw new Error(`src/design/mpnn/ has drifted from ../mpnn:\n${result.stdout}`);
    }
  });
});
