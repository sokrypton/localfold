"""Mirror the MPNN port from its own checkout into this one.

    python3 tools/sync-mpnn.py                 # from ../mpnn
    python3 tools/sync-mpnn.py --from ~/src/mpnn
    python3 tools/sync-mpnn.py --check         # is the mirror in sync?

WHY A COPY AND NOT A DEPENDENCY.

`../mpnn` is a sibling checkout, not something a browser can resolve at
runtime, and this site deploys as static files that a browser loads by relative
path. There is no bundler here to inline it and no package manager in the
serving path to fetch it, so the modules have to live under src/.

🔴 A COPY WITHOUT A RECORD IS A FORK, and a fork nobody declared is the kind
that gets edited in place and then silently diverges. So this tool exists
instead of a one-off `cp`: it stamps the upstream commit into
src/design/mpnn/SOURCE.md, and `--check` re-runs the copy into a scratch
directory and diffs, which is what test/mpnn-vendored.test.js asks for. Edit
upstream and re-sync; never edit the mirror.

🔴 THE CLOSURE, NOT THE DIRECTORY. Five of the fifteen upstream modules
(c6d, potts, search, trmrf, trmrfaccel) implement the Potts and trMRF features,
which this page does not use and which pull in nothing else. Copying them would
ship dead code that build_site.py would then have to keep loading. The list
below is the import closure of what mpnn-bridge.js actually reaches, and
`--check` recomputes it, so a new upstream import that widens the closure is a
failure here rather than a 404 in the browser.
"""
import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES = ROOT / "src" / "design" / "mpnn"
WASM = ROOT / "web" / "vendor" / "mpnn"
PUBLIC = ROOT / "web" / "public" / "mpnn"

# What mpnn-bridge.js imports. Everything else is reached from these.
ENTRY = ["model.js", "pdb.js", "weights.js", "accel.js", "constants.js"]

# 🔴 ONE CHECKPOINT PER FAMILY, NOT ALL SIXTEEN. Upstream ships 66 MB across
# five families at four noise levels each; Pages publishes at most a gigabyte
# and the AF3 parameters are already most of it. These four are 16 MB, they are
# fetched on demand rather than on load, and they are all at noise 0.2 - which
# is the level Protein Hunter's own pipeline.py uses and the level a design
# method wants: enough backbone noise that the model does not read a predicted
# structure as exact. The membrane models are the family left out; nothing on
# this page can say a residue is in a bilayer.
#
# See DESIGNERS in src/design/designers.js, which is the same list and is what
# the page reads. The two are held together by test/designers.test.js.
CHECKPOINTS = [
    "solublempnn_v_48_020.mpnn",
    "proteinmpnn_v_48_020.mpnn",
    "ligandmpnn_v_32_020_25.mpnn",
    "na_mpnn_design.mpnn",
]

IMPORT = re.compile(r'from\s+"\./([^"]+)"')


def closure(source: Path) -> list[str]:
    """Every module reachable from ENTRY, by relative import."""
    seen: set[str] = set()
    stack = list(ENTRY)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        path = source / name
        if not path.exists():
            raise SystemExit(f"{path} does not exist")
        seen.add(name)
        stack.extend(IMPORT.findall(path.read_text(encoding="utf-8")))
    return sorted(seen)


def upstream_commit(source: Path) -> str:
    result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def source_note(source: Path, commit: str, names: list[str]) -> str:
    lines = [
        "# Vendored from the mpnn checkout",
        "",
        "🔴 **DO NOT EDIT THESE FILES.** They are a mirror. Change them",
        "upstream and re-run `python3 tools/sync-mpnn.py`;",
        "`test/mpnn-vendored.test.js` fails if the mirror and the upstream",
        "checkout disagree.",
        "",
        f"- upstream: `{source}`",
        f"- commit: `{commit}`",
        "",
        "The modules below are the import closure of what",
        "`src/design/mpnn-bridge.js` reaches. Upstream also ships `c6d.js`,",
        "`potts.js`, `search.js`, `trmrf.js` and `trmrfaccel.js`, which serve",
        "the Potts and trMRF features and are not mirrored.",
        "",
    ]
    for name in names:
        digest = hashlib.sha256((source / name).read_bytes()).hexdigest()[:16]
        lines.append(f"- `{name}` — `{digest}`")
    return "\n".join(lines) + "\n"


def sync(source: Path, check: bool) -> int:
    modules = source / "mpnn"
    names = closure(modules)
    commit = upstream_commit(source)

    wanted = {MODULES / name: (modules / name).read_bytes() for name in names}
    wanted[MODULES / "SOURCE.md"] = source_note(modules, commit, names).encode("utf-8")
    wanted[WASM / "kernels.wasm"] = (source / "wasm" / "kernels.wasm").read_bytes()
    for name in CHECKPOINTS:
        wanted[PUBLIC / name] = (source / "weights" / name).read_bytes()

    if check:
        stale = [path for path, data in wanted.items()
                 if not path.exists() or path.read_bytes() != data]
        extra = [path for path in MODULES.glob("*.js")
                 if path not in wanted]
        for path in stale:
            print(f"stale: {path.relative_to(ROOT)}")
        for path in extra:
            print(f"not upstream: {path.relative_to(ROOT)}")
        if stale or extra:
            print("\nrun: python3 tools/sync-mpnn.py")
            return 1
        print(f"in sync with {source} @ {commit[:7]} ({len(names)} modules)")
        return 0

    for path, data in wanted.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for path in MODULES.glob("*.js"):
        if path not in wanted:
            path.unlink()
    print(f"mirrored {len(names)} modules + {len(CHECKPOINTS)} checkpoint(s)"
          f" from {source} @ {commit[:7]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="source", default=str(ROOT.parent / "mpnn"),
                        help="the mpnn checkout to mirror (default ../mpnn)")
    parser.add_argument("--check", action="store_true",
                        help="report whether the mirror is in sync, change nothing")
    arguments = parser.parse_args()
    source = Path(arguments.source).expanduser().resolve()
    if not (source / "mpnn" / "model.js").exists():
        print(f"{source} does not look like an mpnn checkout", file=sys.stderr)
        return 1
    return sync(source, arguments.check)


if __name__ == "__main__":
    sys.exit(main())
