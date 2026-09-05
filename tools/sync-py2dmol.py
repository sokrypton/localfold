"""Mirror py2Dmol's built bundles from its own checkout into this one.

    python3 tools/sync-py2dmol.py                  # from ../py2Dmol
    python3 tools/sync-py2dmol.py --from ~/src/py2Dmol
    python3 tools/sync-py2dmol.py --check          # is the mirror in sync?

WHY A COPY AND NOT A DEPENDENCY. py2Dmol is a sibling checkout of classic
`<script>` files with no module system, and this site deploys as static files a
browser loads by relative path. There is nothing in the serving path that could
fetch it.

🔴 THE FOUR FILES WERE COPIED BY HAND AND NOTHING RECORDED FROM WHERE. That is
the failure this exists to end, and it is the same one `tools/sync-mpnn.py`
already ends for the MPNN port: a copy without a record is a fork, and a fork
nobody declared gets edited in place and silently diverges. Before this tool the
vendored bundles named no upstream commit, so "is this current?" could only be
answered by diffing 800 KB of minified JavaScript against a build.

🔴 AND TWO OF THE FOUR ARE BUILT, NOT COPIED. `py2Dmol.embed.min.js` and
`py2Dmol.full.min.js` are bundles that upstream's own `tools/bundle.py build`
writes into `py2Dmol/resources/bundles/`; the other two are source files that
ship as they are. So this runs the upstream build first rather than trusting
whatever happens to be sitting in that directory, which may be from any commit
or from none.

🔴 AND `full` AND `embed` ARE BOTH NEEDED, which is easy to get wrong because
one is nearly a superset. index.html loads `full` (the website plus the embed
API); single.html and proteinhunter.html load `embed`. Syncing only the larger
one leaves the two smaller pages on a stale viewer.
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "web" / "vendor"

# vendored name -> path within the upstream checkout, and whether the upstream
# build has to run first to produce it.
FILES = {
    "py2Dmol.app.css": ("src/app/style.css", False),
    "py2Dmol.align.js": ("src/align/align.js", False),
    "py2Dmol.embed.min.js": ("py2Dmol/resources/bundles/py2Dmol.embed.min.js", True),
    "py2Dmol.full.min.js": ("py2Dmol/resources/bundles/py2Dmol.full.min.js", True),
}

# Which page loads which, so SOURCE.md says why each file is here.
LOADED_BY = {
    "py2Dmol.app.css": "index.html, single.html, proteinhunter.html",
    "py2Dmol.align.js": "index.html (TM-align; upstream cannot bundle it)",
    "py2Dmol.embed.min.js": "single.html, proteinhunter.html",
    "py2Dmol.full.min.js": "index.html",
}

SOURCE = VENDOR / "SOURCE.md"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def upstream_commit(root: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def upstream_dirty(root: Path) -> bool:
    try:
        out = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() != ""
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def build(root: Path) -> None:
    """Run upstream's own bundler, so the mirror is of a commit and not a mood."""
    subprocess.run([sys.executable, "tools/bundle.py", "build"],
                   cwd=str(root), check=True, capture_output=True)


def source_markdown(root: Path, commit: str, hashes: dict) -> str:
    lines = [
        "# Vendored from the py2Dmol checkout",
        "",
        "🔴 **DO NOT EDIT THESE FILES.** They are a mirror. Change them",
        "upstream and re-run `python3 tools/sync-py2dmol.py`;",
        "`python3 tools/sync-py2dmol.py --check` says whether they have drifted.",
        "",
        f"- upstream: `{root}`",
        f"- commit: `{commit}`",
        "",
        "Two of these are built by upstream's `tools/bundle.py build` and two",
        "are source files that ship as they are. `full` is the website plus the",
        "embed API and `embed` is the embed API alone - both are needed,",
        "because index.html loads one and the other two pages load the other.",
        "",
    ]
    for name in FILES:
        lines.append(f"- `{name}` — `{hashes[name]}` — {LOADED_BY[name]}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="source", default=str(ROOT.parent / "py2Dmol"))
    parser.add_argument("--check", action="store_true",
                        help="report whether the mirror matches upstream,"
                             " changing nothing")
    parser.add_argument("--no-build", action="store_true",
                        help="trust the bundles already in the checkout")
    arguments = parser.parse_args()

    root = Path(arguments.source).expanduser().resolve()
    if not root.is_dir():
        print(f"no py2Dmol checkout at {root}")
        return 1

    if not arguments.no_build:
        build(root)

    missing = [name for name, (rel, _) in FILES.items() if not (root / rel).is_file()]
    if missing:
        print("upstream is missing: " + ", ".join(missing))
        return 1

    commit = upstream_commit(root)
    if upstream_dirty(root):
        # 🔴 A DIRTY UPSTREAM MAKES THE STAMP A LIE. The commit recorded would
        # name a tree that does not contain what was copied, which is worse than
        # no stamp at all - it looks authoritative.
        print(f"WARNING: {root} has uncommitted changes; the recorded commit"
              f" {commit[:12]} does not describe what was copied")

    changed = []
    for name, (rel, _) in FILES.items():
        source = root / rel
        target = VENDOR / name
        if not target.exists() or digest(source) != digest(target):
            changed.append(name)
            if not arguments.check:
                shutil.copyfile(source, target)

    hashes = {name: digest(root / rel) for name, (rel, _) in FILES.items()}
    if arguments.check:
        for name in changed:
            print(f"  {name}: mirror differs from upstream")
        stamp = SOURCE.read_text() if SOURCE.exists() else ""
        if stamp != source_markdown(root, commit, hashes):
            print("  SOURCE.md is not what this run would write")
            changed.append("SOURCE.md")
        print(f"{len(changed)} file(s) out of date" if changed
              else f"in sync with {commit[:12]}")
        return 1 if changed else 0

    SOURCE.write_text(source_markdown(root, commit, hashes))
    for name in FILES:
        print(f"  {name:<26} {(VENDOR / name).stat().st_size:>8} bytes"
              f"  {hashes[name]}{'  (updated)' if name in changed else ''}")
    print(f"mirrored py2Dmol {commit[:12]} — {len(changed)} file(s) changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
