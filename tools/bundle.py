"""Emit one self-contained HTML file that runs from file://.

    python3 tools/bundle.py                     # -> localfold-local.html
    python3 tools/bundle.py --out somewhere.html

WHY THIS EXISTS, AND WHY IT IS NOT THE NORMAL PATH.

Served over http:// the page needs no build at all - single.html loads
web/main.js as a module and the browser resolves the rest. Opened as a FILE it
cannot: the origin is null, and Chrome allows cross-origin module loads only for
http, https, data and extension schemes. A module graph is simply unreachable
there, whatever the file layout.

So this collapses the graph into one classic script, which file:// does allow.
It is a second artifact, not a stage in front of the first: `python3 -m
http.server` still runs the checkout as written, and this file is what you make
when you want to hand somebody a demo that needs nothing installed at all.

WHAT IT DOES NOT SOLVE. fetch() is unavailable on file:// too, so the 355 MiB of
weights cannot be loaded from a URL no matter how the scripts arrive. The page
asks for the model folder through a file picker instead - see
DirectoryTensorStore - because a file the user hands the page is a grant rather
than a request, and grants are honoured on any origin.

HOW THE COLLAPSE WORKS. The graph has no cycles (asserted below), so the modules
can be emitted in topological order, each in its own function scope, each
returning its exports into a registry. `import` becomes a destructuring read of
an already-built entry; `export` becomes an assignment on the way out. Nothing
is renamed and nothing is minified: a stack trace from the bundle still names
the function you wrote, and the WGSL inside it is untouched.
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRY = "web/main.js"
STYLE = "web/style.css"
VENDOR = "web/vendor/py2Dmol.embed.min.js"
# 🔴 THE OFFLINE BUNDLE IS THE SINGLE-SEQUENCE PAGE, not the front page. A
# file:// bundle exists so the whole thing runs with no network at all, and
# index.html is the MSA page - which can search a remote server for an
# alignment, and is therefore the one page that has a reason to be online.
PAGE = "single.html"

# `import { a, b as c } from "./x.js";`  - the brace body may span lines.
IMPORT = re.compile(r'^import\s*\{([^}]*)\}\s*from\s*"([^"]+)"\s*;?[ \t]*$', re.M | re.S)
# ...and the forms this project does not use. Caught so they fail loudly rather
# than silently surviving into the bundle as syntax the wrapper cannot hold.
UNSUPPORTED = re.compile(r'^import\s+(?!\{)|^export\s+default\b|^export\s*\*', re.M)

REEXPORT = re.compile(r'^export\s*\{([^}]*)\}\s*from\s*"([^"]+)"\s*;?[ \t]*$', re.M | re.S)
EXPORT_LIST = re.compile(r'^export\s*\{([^}]*)\}\s*;?[ \t]*$', re.M | re.S)
EXPORT_DECL = re.compile(r'^export\s+(async\s+function|function|class|const|let|var)\s+([A-Za-z0-9_$]+)', re.M)


def resolve(importer: str, specifier: str) -> str:
    """A relative specifier, as a repo-root-relative module id."""
    target = (ROOT / importer).parent / specifier
    return target.resolve().relative_to(ROOT).as_posix()


def imports_of(source: str, module: str):
    """Every module this one needs - `export {...} from` is a dependency too."""
    return [resolve(module, match.group(2))
            for pattern in (IMPORT, REEXPORT)
            for match in pattern.finditer(source)]


def order_from(entry: str):
    """Every module the entry needs, dependencies first. Raises on a cycle."""
    sources, order, state = {}, [], {}

    def visit(module: str, stack):
        if state.get(module) == "done":
            return
        if state.get(module) == "open":
            cycle = " -> ".join(stack[stack.index(module):] + [module])
            raise SystemExit(f"import cycle, which this bundler cannot order: {cycle}")
        state[module] = "open"
        source = (ROOT / module).read_text()
        bad = UNSUPPORTED.search(source)
        if bad:
            line = source[:bad.start()].count("\n") + 1
            raise SystemExit(f"{module}:{line}: unsupported module syntax for the offline bundle:"
                             f" {source[bad.start():bad.end() + 40].splitlines()[0]!r}")
        sources[module] = source
        for dependency in imports_of(source, module):
            visit(dependency, stack + [module])
        state[module] = "done"
        order.append(module)

    visit(entry, [])
    return order, sources


def transform(module: str, source: str) -> str:
    """One module, as the body of a function that returns its exports."""
    exported = []

    def on_import(match):
        bindings = " ".join(match.group(1).split())
        # `a as b` is `a: b` once it is a destructuring pattern.
        bindings = re.sub(r"([A-Za-z0-9_$]+)\s+as\s+([A-Za-z0-9_$]+)", r"\1: \2", bindings)
        return f'const {{ {bindings} }} = __afRequire("{resolve(module, match.group(2))}");'

    def on_reexport(match):
        names = [name.strip() for name in match.group(1).split(",") if name.strip()]
        source_id = resolve(module, match.group(2))
        lines = []
        for name in names:
            local, _, alias = (part.strip() for part in name.partition(" as "))
            lines.append(f'__afExports["{alias or local}"] = __afRequire("{source_id}")["{local}"];')
        return "\n".join(lines)

    def on_export_list(match):
        for name in (part.strip() for part in match.group(1).split(",")):
            if not name:
                continue
            local, _, alias = (part.strip() for part in name.partition(" as "))
            exported.append((alias or local, local))
        return ""

    def on_export_decl(match):
        exported.append((match.group(2), match.group(2)))
        return f"{match.group(1)} {match.group(2)}"

    body = REEXPORT.sub(on_reexport, source)
    body = IMPORT.sub(on_import, body)
    body = EXPORT_LIST.sub(on_export_list, body)
    body = EXPORT_DECL.sub(on_export_decl, body)

    assignments = "".join(f'\n  __afExports["{name}"] = {local};' for name, local in exported)
    return (f'// ---- {module} ----\n'
            f'__afModules["{module}"] = (function () {{\n'
            f'  var __afExports = {{}};\n'
            f'{body}{assignments}\n'
            f'  return __afExports;\n'
            f'}})();\n')


def build(out: Path) -> int:
    order, sources = order_from(ENTRY)
    modules = "\n".join(transform(module, sources[module]) for module in order)
    script = ('(function () {\n"use strict";\n'
              'var __afModules = {};\n'
              'function __afRequire(id) {\n'
              '  var value = __afModules[id];\n'
              '  if (value === undefined) throw new Error("module not bundled: " + id);\n'
              '  return value;\n'
              '}\n\n'
              f'{modules}'
              '})();\n')

    page = (ROOT / PAGE).read_text()
    style = (ROOT / STYLE).read_text()
    vendor = (ROOT / VENDOR).read_text()

    page = page.replace('<link rel="stylesheet" href="./web/style.css" />',
                        f"<style>\n{style}\n</style>")
    page = re.sub(r'[ \t]*<!-- py2Dmol first.*?-->\n', "", page, flags=re.S)
    page = page.replace('<script src="./web/vendor/py2Dmol.embed.min.js"></script>',
                        f"<script>\n{vendor}\n</script>")
    page = page.replace('<script type="module" src="./web/main.js"></script>',
                        f"<script>\n{script}\n</script>")
    assert "<script" in page and "src=" not in page.split("<body")[1], \
        "the offline page still references an external file"

    out.write_text(page)
    print(f"{out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}"
          f"  {len(order)} modules, {out.stat().st_size / 1024 / 1024:.1f} MiB")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "localfold-local.html"),
                        help="where to write the self-contained page")
    raise SystemExit(build(Path(parser.parse_args().out).resolve()))
