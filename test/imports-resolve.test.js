import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { sourceFiles } from "./helpers/source-files.js";
import { dirname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Every relative import in this repository resolves to a file that exists.
 *
 * 🔴 BECAUSE THERE IS NO BUNDLER AND NO TYPE CHECKER IN THE GATE PATH. The page
 * and `tools/gpu-chrome.mjs` both serve `src/` over HTTP as ES modules, so a
 * path that resolves nowhere is a 404 at RUNTIME, in whichever lane happens to
 * import it - and `npm test` imports perhaps a third of these files. A rename
 * or a directory move that misses one import is therefore invisible until some
 * GPU gate runs the one module that names it, which may be never:
 * docs/PARITY.md records nineteen of twenty-one AF3 checkers not running on
 * this box at all.
 *
 * `tsc` is advisory here (1075 errors on this tree, per CLAUDE.md) so it is not
 * the gate. This is: it is cheap, it is total, and it fails on exactly one
 * thing.
 */
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const AREAS = ["src", "tools", "test", "web"];
const SKIP = new Set(["node_modules", ".git", ".ipynb_checkpoints"]);

/**
 * The source with comments removed.
 *
 * 🔴 BECAUSE THIS FILE'S OWN FIRST RUN FAILED ON A COMMENT. Doc comments here
 * quote imports as examples - `await import("./ops.js")` in
 * test/module-references.test.js - and a scanner that reads them reports a
 * module that was never meant to exist. A gate whose first finding is its own
 * false positive teaches the reader to ignore it.
 */
function withoutComments(text) {
  return text.replace(/\/\*[\s\S]*?\*\//g, "")
    .split("\n").filter((line) => !/^\s*\/\//.test(line)).join("\n");
}

/** Static `import ... from "x"`, `export ... from "x"` and dynamic `import("x")`. */
function specifiersOf(source) {
  const text = withoutComments(source);
  const found = [];
  const patterns = [
    /(?:^|\n)\s*import\s+[^;]*?from\s*["']([^"']+)["']/g,
    /(?:^|\n)\s*export\s+[^;]*?from\s*["']([^"']+)["']/g,
    /(?:^|\n)\s*import\s*["']([^"']+)["']/g,
    /\bimport\s*\(\s*["']([^"']+)["']\s*\)/g,
  ];
  for (const pattern of patterns) {
    for (const match of text.matchAll(pattern)) found.push(match[1]);
  }
  return found;
}

test("every relative import resolves to a file that exists", () => {
  const files = AREAS.flatMap((area) => sourceFiles(join(ROOT, area)));
  // 🔴 A RULE THAT STOPS MATCHING PASSES BY FINDING NOTHING.
  assert.ok(files.length >= 150, `walked ${files.length} files`);

  const broken = [];
  let checked = 0;
  for (const file of files) {
    for (const spec of specifiersOf(readFileSync(file, "utf8"))) {
      // Bare specifiers are node's or the CDN's problem, not this rule's;
      // absolute ones are served from the repo root by tools/serve.py.
      const target = spec.startsWith(".") ? resolve(dirname(file), spec)
        : spec.startsWith("/") ? join(ROOT, spec)
          : null;
      if (target === null) continue;
      checked += 1;
      if (!existsSync(target)) {
        broken.push(`${file.slice(ROOT.length + 1)} -> ${spec}`);
      }
    }
  }
  assert.ok(checked >= 400, `only ${checked} relative imports found`);
  assert.deepEqual(broken, [], `${broken.length} imports resolve nowhere`);
});

/**
 * 🔴 AND THE SAME RULE FOR PATHS NAMED IN PROSE, because the check above is
 * blind to them and that blindness cost 129 stale references in one afternoon.
 * The `src/af3` reorganisation rewrote import SPECIFIERS and `.md` files and
 * left every `src/af3/<file>.js` written inside a code comment pointing at a
 * path that no longer existed - in `dialect.js`, in `transition-webgpu.js`, in
 * 75 files. A comment naming a module the reader cannot open is the
 * "doc that disagrees with itself" failure this repository keeps paying for,
 * and it is mechanically checkable.
 *
 * Two exclusions, both deliberate:
 *   - `web/vendor/` is somebody else's code, and its references to THEIR tests
 *     (test/tmalign_*.mjs) are correct in their repository and not ours to edit.
 *   - a path preceded by `/` is part of a longer one - `../py2Dmol/src/app/
 *     session.js` is a file in a DIFFERENT checkout, and the first version of
 *     this rule reported three of those as missing modules.
 */
test("every module path named in a comment exists", () => {
  const files = AREAS.flatMap((area) => sourceFiles(join(ROOT, area)))
    .filter((file) => !file.includes(`${sep}vendor${sep}`));
  assert.ok(files.length >= 150, `walked ${files.length} files`);

  const broken = [];
  let named = 0;
  const PATH = /(?<![/\w-])((?:src|tools|test|web)\/[A-Za-z0-9_./-]+\.m?js)\b/g;
  for (const file of files) {
    for (const match of readFileSync(file, "utf8").matchAll(PATH)) {
      const named_path = match[1];
      // An editor artefact is named here precisely BECAUSE it should not exist
      // - see test/modules-parse.test.js, which skips hidden directories.
      if (named_path.includes(".ipynb_checkpoints")) continue;
      named += 1;
      if (!existsSync(join(ROOT, named_path))) {
        broken.push(`${file.slice(ROOT.length + 1)} names ${named_path}`);
      }
    }
  }
  assert.ok(named >= 100, `only ${named} module paths named in prose`);
  assert.deepEqual(broken, [], `${broken.length} comments name a module that does not exist`);
});

/**
 * 🔴 AND THE SAME RULE FOR PYTHON, YAML AND HTML, because the two rules above
 * read only `.js` and `.mjs` - and the `src/` reorganisation broke
 * **tools/write_manifest_module.py** without a single gate noticing.
 *
 * That file carries a `"module"` WRITE TARGET per family, each naming a file
 * under the old `src/reference/manifests` directory. After that directory became `src/bundles/` it
 * would have written thirteen manifest modules into a directory nothing loads,
 * silently recreating the old tree while the live one went stale - and it is
 * the tool you run straight after `hf upload`, so the first time anyone noticed
 * would have been a published bundle the page could not find.
 *
 * Forty-three paths across thirteen non-JS files were stale, most of them prose
 * but not all. The lesson is the one the comment-path rule above already
 * records, one file type further out: **a path is a path whatever the file
 * extension is.**
 *
 * 🔴 AND MARKDOWN IS IN HERE, WHICH IS THE LAST FILE TYPE AND THE ONE THAT
 * MATTERS MOST. The docs are how anything in this repository is FOUND -
 * CLAUDE.md's tool table is the index - so a doc naming a module that moved is
 * worse than code doing it: code fails, a doc just misleads. The reorganisation
 * left 41 stale paths across nine documents, including
 * docs/DEVELOPING.md's note on the WebGL2 modules, which are kept
 * deliberately and which a reader would then have failed to find.
 *
 * py2Dmol's own tree is excluded by prefix - `src/align/`, `src/app/`,
 * `src/core/`, `src/io/`, `src/panels/` are a DIFFERENT checkout's paths, named
 * here on purpose, and the first version of this rule reported them as missing.
 */
test("every src/ path named in python, yaml, html or markdown exists", () => {
  // Other projects' own `src/` trees, named here on purpose and not ours to
  // resolve: py2Dmol's SIX directories, the AlphaFold 3 reference checkout,
  // Dawn's C++, and the two upstreams `tools/sync-*.py` vendor from.
  //
  // 🔴 THE CARTOON DIRECTORY IS py2Dmol's SIXTH AND IT ARRIVED WITH A VENDOR
  // BUMP. `tools/fold-in-page.py` cites py2Dmol/src/cartoon/paintgl.js as the
  // provenance of the ribbon cache it counts, and that file has never existed
  // in this repository on any branch - it is upstream's. The gate caught it on
  // a merge, which is the whole point of reading a bare `src/...` as a claim
  // about OUR tree: a vendor bump is exactly when somebody else's path arrives
  // written as though it were ours.
  //
  // 🔴 AND THIS COMMENT HAD TO BE WRITTEN WITH THE PROJECT IN FRONT OF IT,
  // because the sibling rule above reads any BARE src-rooted module path as
  // ours and went red on this very paragraph. A path preceded by `/` is
  // another checkout's - which is the convention that rule's own header
  // documents, and the accurate way to write it.
  const FOREIGN = ["src/align/", "src/app/", "src/cartoon/", "src/core/", "src/io/",
                   "src/panels/", "src/alphafold3/", "src/dawn/", "src/mpnn/",
                   "src/py2Dmol/"];
  // Its own walker: the shared one above is deliberately JS-only, and widening
  // it made the comment rule read python it has no exclusions for.
  const walkAny = (dir, out = []) => {
    for (const entry of readdirSync(dir)) {
      if (entry.startsWith(".") && entry !== ".github") continue;
      if (entry === "node_modules" || entry === "__pycache__") continue;
      const path = join(dir, entry);
      if (statSync(path).isDirectory()) walkAny(path, out);
      else if (/\.(py|ya?ml|html|sh|md)$/.test(entry)) out.push(path);
    }
    return out;
  };
  const files = [];
  for (const area of ["tools", ".github", "docs"]) {
    const root = join(ROOT, area);
    if (existsSync(root)) walkAny(root, files);
  }
  for (const loose of ["index.html", "dev.html", "package.json",
                       "CLAUDE.md", "AGENTS.md", "README.md", "HANDOFF.md"]) {
    if (existsSync(join(ROOT, loose))) files.push(join(ROOT, loose));
  }
  assert.ok(files.length >= 20, `walked ${files.length} non-JS files`);

  // 🔴 ONE DOCUMENT MAY NAME A PATH THAT MOVED, AND IT IS THE ONE THAT RECORDS
  // THE MOVE. docs/ARCHITECTURE.md's whole subject is the reorganisation -
  // "`src/evoformer/` WAS TWO THINGS", "after `src/reference/` became
  // `src/bundles/`" - and rewriting those sentences to the new names would
  // delete the history the document exists to keep. Every other file names the
  // tree as it is.
  //
  // 🔴 AND THE ALLOWANCE RETIRES ITSELF, both ways: a FORMER path that comes
  // BACK into existence fails (the allowance is no longer needed and would now
  // hide a real one), and a FORMER path that HISTORY no longer names fails too
  // (a list nobody prunes is a list that stops meaning anything). Same rule as
  // ALLOWED in test/no-import-cycles.test.js.
  const HISTORY = new Set(["docs/ARCHITECTURE.md"]);
  // Matched as a PREFIX, because a narrative names both the directory and a
  // file inside it - `src/reference/manifests` and, quoting the line that
  // broke, `/src/reference/manifests/index.js`.
  const FORMER = ["src/reference", "src/model", "src/evoformer", "src/triangle"];
  const formerUsed = new Set();
  const former = (path) => FORMER.find(
    (prefix) => path === prefix || path.startsWith(`${prefix}/`));

  const broken = [];
  let named = 0;
  // 🔴 THE LEADING SLASH IS OPTIONAL, AND LEAVING IT OUT COST TWO LIVE BUGS.
  // The first version's lookbehind was `(?<![/\w-])`, written to stop the rule
  // matching the tail of a longer path - and a leading `/` is exactly how the
  // BROWSER names a module, which is the form every CDP tool and every
  // `await import(...)` inside an evaluated string uses. So the one dialect the
  // gate could not see was the one the page speaks, and it missed both of
  // `tools/fold-in-page.py`'s: `/src/reference/manifests/index.js`, whose
  // rewrite of the `remote:` line simply stopped matching so `--local-weights`
  // silently fetched from Hugging Face instead, and
  // `/src/reference/http-tensor-store.js`, whose import sat inside a `try` that
  // prints `'unavailable: ' + error.message` - a fallback, reporting a moved
  // file as a missing measurement.
  const PATH = /(?<![\w.-])\/?(src\/[A-Za-z0-9_./-]+\.m?js)\b/g;
  // 🔴 AND A DIRECTORY IS A PATH TOO. `src/reference/manifests` without a file
  // on the end is how six documents and two tools name the manifest modules,
  // and `src/reference/` has not existed since the reorganisation. A rule that
  // requires a `.js` suffix reads those as prose.
  const DIRECTORY = /(?<![\w.-])\/?(src\/[a-z0-9-]+(?:\/[a-z0-9-]+)*)\/?(?![\w./-])/g;
  for (const file of files) {
    const text = readFileSync(file, "utf8");
    const relative = file.slice(ROOT.length + 1);
    for (const match of text.matchAll(PATH)) {
      const named_path = match[1];
      if (FOREIGN.some((prefix) => named_path.startsWith(prefix))) continue;
      if (HISTORY.has(relative) && former(named_path)) {
        formerUsed.add(former(named_path));
        continue;
      }
      named += 1;
      if (!existsSync(join(ROOT, named_path))) {
        broken.push(`${file.slice(ROOT.length + 1)} names ${named_path}`);
      }
    }
    for (const match of text.matchAll(DIRECTORY)) {
      const named_path = match[1];
      if (FOREIGN.some((prefix) => `${named_path}/`.startsWith(prefix))) continue;
      if (HISTORY.has(relative) && former(named_path)) {
        formerUsed.add(former(named_path));
        continue;
      }
      named += 1;
      if (!existsSync(join(ROOT, named_path))) {
        broken.push(`${relative} names directory ${named_path}`);
      }
    }
  }
  assert.ok(named >= 150, `only ${named} src paths named outside JS`);
  assert.deepEqual(broken, [], `${broken.length} non-JS files name a module that is gone`);

  const revived = FORMER.filter((path) => existsSync(join(ROOT, path)));
  assert.deepEqual(revived, [],
    `FORMER names ${revived.join(", ")}, which exist again - drop the allowance`);
  const unused = FORMER.filter((path) => !formerUsed.has(path));
  assert.deepEqual(unused, [],
    `FORMER names ${unused.join(", ")}, which no HISTORY file mentions - prune it`);
});
