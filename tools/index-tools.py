#!/usr/bin/env python3
"""Every tool in this repository, with the one line it says about itself.

    python3 tools/index-tools.py            # check docs/TOOLS.md is current
    python3 tools/index-tools.py --write    # regenerate it

🔴 WHY THIS EXISTS: 44 TOOLS WERE NAMED BY NOTHING AT ALL. CLAUDE.md's table is
the curated view - it answers "which tool answers this question" and is
deliberately not exhaustive - and `docs/` covers what a campaign touched. Between
them they left a third of `tools/gpu/` unmentioned, including working
differential checkers such as `check-af3-opm.js` and `check-chiral-gradient.js`.
A tool nobody can find is a tool nobody runs, which is docs/PARITY.md's whole
complaint from the other end.

🔴 AND THE INDEX IS DERIVED, NEVER WRITTEN. A hand-maintained list of three
hundred files is a list that goes stale, and this repository's own recurring
lesson is that a doc disagreeing with itself costs more than a missing one. Each
entry is the FIRST LINE OF THE FILE'S OWN HEADER - its Python docstring, its
leading `/** */` or `//` block, or an HTML `<title>` - so an index entry cannot
drift from the file: changing one changes the other or the gate goes red.

The gate is therefore two rules, and both are about accounting rather than
prose:

  1. every tool has a header line saying what it is, and
  2. `docs/TOOLS.md` lists exactly the tools that exist, with exactly those
     lines.

Adding a tool without a header fails rule 1. Adding, deleting or renaming one
without `--write` fails rule 2.

`tools/fixtures/` is excluded: it is data, not tools.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "docs" / "TOOLS.md"
EXTENSIONS = (".py", ".js", ".mjs", ".html")
EXCLUDE_PREFIX = ("tools/fixtures/",)

# The groups, in the order they appear in the index. A file lands in the first
# group whose prefix it starts with; `tools/` itself is the catch-all and is
# last for that reason.
GROUPS = [
    ("tools/gpu/", "`tools/gpu/` - the WebGPU lane",
     "Each exports `async function main(device, args)` and is run as\n"
     "`node tools/gpu-chrome.mjs tools/gpu/<module>.js [--flags]`."),
    ("tools/oracle/", "`tools/oracle/` - the reference side",
     "These run against a checkout of the reference implementation, not against\n"
     "this port. They need its Python environment; see CLAUDE.md."),
    ("tools/esmc/", "`tools/esmc/` - ESM-C and ESMFold2 export",
     "The PyTorch side of the ESMFold2 port: fetching the checkpoint, probing it,\n"
     "and turning it into a bundle."),
    ("tools/", "`tools/` - everything else",
     "CPU checkers, the export and quantisation pipeline, the page drivers that\n"
     "go through CDP, and the deploy."),
]


def tracked_tools() -> list[str]:
    listing = subprocess.run(
        ["git", "ls-files", "tools"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    return sorted(
        path for path in listing
        if path
        and os.path.splitext(path)[1] in EXTENSIONS
        and not path.startswith(EXCLUDE_PREFIX)
    )


def header_line(path: Path) -> str | None:
    """The first line of the file's own header, or None if it has no header.

    🔴 THE FIRST LINE, NOT A SUMMARY. Nothing here paraphrases: an entry that
    said something the file does not is exactly the drift this gate exists to
    stop.
    """
    text = path.read_text(encoding="utf8", errors="replace")
    suffix = path.suffix

    if suffix == ".py":
        # The docstring, past a shebang and a `from __future__` line.
        match = re.search(
            r'^\s*(?:#![^\n]*\n)?(?:from __future__[^\n]*\n)?\s*("""|\'\'\')(.*?)\1',
            text, re.S,
        )
        return _first(match.group(2)) if match else None

    if suffix == ".html":
        match = re.search(r"<title>(.*?)</title>", text, re.S)
        return _first(match.group(1)) if match else None

    # .js and .mjs: a leading /** */ block, else a run of leading // lines.
    match = re.match(r"\s*/\*\*(.*?)\*/", text, re.S)
    if match:
        return _first(match.group(1), strip="*")
    match = re.match(r"((?:[ \t]*//[^\n]*\n)+)", text)
    if match:
        return _first(match.group(1), strip="/")
    return None


# A header's opening sentence is usually WRAPPED across two or three source
# lines, so taking the first line alone truncates a third of them mid-clause.
# Read on to the end of the sentence instead, and no further.
SENTENCE_LINES = 4


def _first(block: str, strip: str = "") -> str | None:
    collected: list[str] = []
    for line in block.splitlines():
        cleaned = line.strip()
        if strip:
            cleaned = cleaned.lstrip(strip).strip()
        if not cleaned:
            if collected:
                break          # a blank line ends the opening paragraph
            continue
        collected.append(cleaned)
        if cleaned.endswith((".", "?", "!", ":")) or len(collected) == SENTENCE_LINES:
            break
    if not collected:
        return None
    return " ".join(collected)


def group_of(path: str) -> int:
    for index, (prefix, _, _) in enumerate(GROUPS):
        if path.startswith(prefix):
            return index
    raise AssertionError(f"{path} is under no group")


def render(entries: dict[str, str]) -> str:
    out = [
        "# Every tool, and the one line it says about itself",
        "",
        "🔴 **THIS FILE IS GENERATED BY `python3 tools/index-tools.py --write`, AND",
        "`npm run test:tools` FAILS WHEN IT IS STALE.** Every line below is the first",
        "line of the named file's own header, copied rather than written, so an entry",
        "cannot drift from the thing it describes. To change an entry, change the",
        "file's header and regenerate.",
        "",
        "This is the exhaustive list. **CLAUDE.md's table is the one to read first** -",
        "it is organised by the question a tool answers and says which arms are traps,",
        "which is what you want when you know what you are asking. Come here when you",
        "do not know whether a tool already exists.",
        "",
        f"{len(entries)} tools. `tools/fixtures/` is data and is not listed.",
        "",
    ]
    for index, (prefix, title, blurb) in enumerate(GROUPS):
        rows = sorted(path for path in entries if group_of(path) == index)
        if not rows:
            continue
        out.append(f"## {title} ({len(rows)})")
        out.append("")
        out.append(blurb)
        out.append("")
        for path in rows:
            name = path[len(prefix):]
            out.append(f"- **`{name}`** - {entries[path]}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="regenerate docs/TOOLS.md instead of checking it")
    arguments = parser.parse_args()

    paths = tracked_tools()
    entries: dict[str, str] = {}
    headless: list[str] = []
    for path in paths:
        line = header_line(ROOT / path)
        if line is None:
            headless.append(path)
        else:
            entries[path] = line

    if headless:
        print(f"🔴 {len(headless)} tool(s) with no header line saying what they are:")
        for path in headless:
            print(f"   {path}")
        print("\nAdd a docstring, a /** */ block or a <title>. The index is derived from it.")
        return 1

    rendered = render(entries)
    if arguments.write:
        INDEX.write_text(rendered, encoding="utf8")
        print(f"wrote {INDEX.relative_to(ROOT)}: {len(entries)} tools")
        return 0

    if not INDEX.exists():
        print(f"🔴 {INDEX.relative_to(ROOT)} does not exist. Run --write.")
        return 1
    current = INDEX.read_text(encoding="utf8")
    if current != rendered:
        print(f"🔴 {INDEX.relative_to(ROOT)} is stale. Run:")
        print("      python3 tools/index-tools.py --write")
        import difflib
        diff = list(difflib.unified_diff(
            current.splitlines(), rendered.splitlines(),
            "docs/TOOLS.md", "generated", lineterm="", n=1,
        ))
        for line in diff[:60]:
            print("   " + line)
        if len(diff) > 60:
            print(f"   ... {len(diff) - 60} more diff lines")
        return 1
    print(f"docs/TOOLS.md is current: {len(entries)} tools, every one with a header line")
    return 0


if __name__ == "__main__":
    sys.exit(main())
