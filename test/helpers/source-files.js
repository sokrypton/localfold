import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";

/**
 * Every source file under a directory - and NOTHING an editor wrote.
 *
 * 🔴 ONE WALKER, BECAUSE FOUR LEARNED THIS SEPARATELY AND A FIFTH DID NOT.
 * A jupyter-lab running against this checkout writes
 * `<dir>/.ipynb_checkpoints/<name>-checkpoint.js` every time a file is saved:
 * stale copies whose imports are whatever that file said when it was last
 * snapshotted. They are gitignored, they never reach the repository, and
 * deleting them is futile - they come back with the next save.
 *
 * What matters is that nothing READS them, and four things have had to learn
 * that one at a time:
 *
 *   test/modules-parse.test.js      imported one and failed the whole CPU suite
 *                                   on a file nobody wrote
 *   test/no-import-cycles.test.js   fed two of them into the dependency graph
 *   test/helpers/af3-source.js      would have handed one back by basename
 *   tools/build_site.py             walked them into the DEPLOY check, which
 *                                   failed on advice pointing at .gitignore
 *
 * Every one of those is the same rule discovered again. It lives here now, so
 * the fifth walker inherits it instead of rediscovering it. (build_site.py is
 * Python and keeps its own copy with a comment pointing here - a shared rule
 * cannot cross that boundary, which is worth knowing rather than pretending.)
 *
 * @param {string} dir absolute path to walk
 * @param {{extensions?: string[]}} [options] default: .js and .mjs
 * @returns {string[]} absolute paths
 */
export function sourceFiles(dir, options = {}) {
  const extensions = options.extensions ?? [".js", ".mjs"];
  const out = [];
  const walk = (at) => {
    for (const entry of readdirSync(at)) {
      // A dotted directory is an editor's or a tool's, never source; and
      // node_modules is somebody else's whole tree.
      if (entry.startsWith(".") || entry === "node_modules") continue;
      const path = join(at, entry);
      if (statSync(path).isDirectory()) walk(path);
      else if (extensions.some((suffix) => entry.endsWith(suffix))) out.push(path);
    }
  };
  walk(dir);
  return out;
}
