"""Assemble the deployable site into dist/.

    python3 tools/build_site.py            # dist/, without model parameters
    python3 tools/build_site.py --model    # ...and the model/ directory too

WHY THIS EXISTS, AND WHY IT IS NOT A BUNDLER.

The page loads src/**/*.js as written - plain ES modules, resolved by the
browser through the same relative paths they have in the checkout. So there is
nothing to compile, and the only real question a "build" answers here is WHICH
FILES a public site should contain: not test/, not tools/, and not the 946 KB
test.a3m or the fixtures under test/fixtures, which together dwarf the app.

That makes this a copy with an allow-list, and the allow-list is the point. It
is written out below rather than derived, because a derived rule ("everything
but test/") silently ships the next directory somebody adds.

THE LAYOUT IS PRESERVED, exactly. The pages at the top, web/ and src/ beside
them, because index.html says ./web/main.js and main.js says ../src/model/... -
flattening any of that would mean rewriting import paths, and rewriting import
paths is the build step this repository just got rid of.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist"

# WHAT A PUBLIC SITE CONTAINS. Files are copied as-is; directories are copied
# whole, minus the ignore patterns below.
FILES = [".nojekyll", "index.html", "single.html", "dev.html"]
DIRECTORIES = ["web", "src"]

# ...and never these, wherever they appear.
IGNORE = shutil.ignore_patterns("*.pyc", "__pycache__", ".DS_Store", "*.map")


def build(include_model: bool) -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for name in FILES:
        source = ROOT / name
        if not source.exists():
            print(f"missing {name}", file=sys.stderr)
            return 1
        shutil.copy2(source, OUT / name)

    for name in DIRECTORIES:
        source = ROOT / name
        if not source.is_dir():
            print(f"missing {name}/", file=sys.stderr)
            return 1
        shutil.copytree(source, OUT / name, ignore=IGNORE)

    # THE PARAMETERS ARE OPT-IN, and they are the whole reason the workflow has
    # a repository variable: GitHub Pages is public even when its source
    # repository is private, so a model directory that happens to be lying
    # around in the checkout must not publish itself.
    if include_model:
        model = ROOT / "model"
        if not model.is_dir():
            print("--model was given but model/ does not exist;"
                  " run `node tools/export-web-model.js <manifest>` first", file=sys.stderr)
            return 1
        # ...the .bin shards only. A served page reads those through fetch, and
        # the base64 scripts beside them are a third larger and exist purely for
        # file:// - shipping both would put a 356 MiB site at 830 MiB, most of a
        # GitHub Pages allowance spent on bytes nothing on that site can use.
        shutil.copytree(model, OUT / "model",
                        ignore=shutil.ignore_patterns("*.pyc", "__pycache__", ".DS_Store",
                                                      "*.map", "weights-*.js", "manifest.js"))

    total = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    count = sum(1 for path in OUT.rglob("*") if path.is_file())
    print(f"dist/  {count} files, {total / 1024 / 1024:.1f} MiB"
          f"{'' if include_model else '  (no model parameters)'}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="store_true",
                        help="include the exported model/ directory")
    raise SystemExit(build(parser.parse_args().model))
