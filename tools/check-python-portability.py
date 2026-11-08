"""Does every python tool here run on the OLDEST python3 a developer has?

    python3 tools/check-python-portability.py

🔴 BECAUSE `python3` IS WHOEVER'S PYTHON IT IS, AND ON A MAC IT IS APPLE'S 3.9.
Every python gate in package.json is spelled `python3 <tool>`, and the one in
.github/workflows/pages.yml is too, so the version is whatever the machine
hands over - 3.12 on this box's own install, 3.11 in the two workflows that pin
it, and **3.9.6** as `/usr/bin/python3` on the M2. A tool that needs 3.10 does
not fail there with a version number: `tools/build_site.py` died with
`TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` out of a
`def` line, which reads as a bug in the tool rather than a bug in the
invitation to run it, and `npm run test:site` was simply broken on that Mac.

Two rules, both of them about code that RUNS at import time:

  * a `|` union in an annotation needs `from __future__ import annotations`,
    which makes the annotation a string and costs nothing at runtime. Five
    tools here already carry it; three did not.
  * nothing may use syntax newer than FLOOR, which `ast.parse` is asked
    directly rather than by finding a second interpreter to try.

It cannot see a union built OUTSIDE an annotation (`Row = int | None`), which
the future import does not fix either - that one is a rewrite to `Optional`.
Exits non-zero naming every file and line.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

FLOOR = (3, 9)
ROOT = pathlib.Path(__file__).resolve().parent.parent


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return out.stdout.split()


def annotations_of(node: ast.AST) -> list[ast.expr]:
    """Every annotation the module body EVALUATES as it is imported."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        spots = [node.returns]
        spots += [a.annotation for a in
                  args.posonlyargs + args.args + args.kwonlyargs]
        spots += [a.annotation for a in (args.vararg, args.kwarg) if a]
        return [s for s in spots if s is not None]
    if isinstance(node, ast.AnnAssign):
        return [node.annotation]
    return []


def union_lines(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        for spot in annotations_of(node):
            if any(isinstance(x, ast.BinOp) and isinstance(x.op, ast.BitOr)
                   for x in ast.walk(spot)):
                lines.append(spot.lineno)
    return sorted(set(lines))


def main() -> int:
    broken = []
    for name in tracked():
        source = (ROOT / name).read_text()
        try:
            # The floor's own parser: a match statement or a 3.10 `except*` is a
            # SyntaxError here and nothing else in this file would notice it.
            ast.parse(source, filename=name, feature_version=FLOOR)
        except SyntaxError as error:
            broken.append(f"{name}:{error.lineno}: {error.msg}"
                          f" - syntax newer than python {FLOOR[0]}.{FLOOR[1]}")
            continue
        tree = ast.parse(source, filename=name)
        if any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
               and any(a.name == "annotations" for a in n.names)
               for n in ast.walk(tree)):
            continue
        for line in union_lines(tree):
            broken.append(f"{name}:{line}: a `|` union in an annotation,"
                          " without `from __future__ import annotations`")
    for row in broken:
        print(row)
    print(f"{len(tracked())} python files, {len(broken)} that python"
          f" {FLOOR[0]}.{FLOOR[1]} could not run")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
