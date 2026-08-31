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
import hashlib
import json
import re
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



# WHY THE BUILD RESOLVES THE MODULE GRAPH.
#
# 🔴 A DEPLOY ONCE 404'd ON TWO MODULES THAT EVERY CHECKOUT HAD. .gitignore said
# `model/` for the exported weights, and an unanchored pattern matches a
# directory of that name at ANY depth - so src/model/ was silently untracked,
# the files existed locally, the site built, and the published page failed to
# load web/app.js with no error anyone would see. Copying is not enough: a
# build has to answer "does what I just assembled actually load", and for ES
# modules that means every relative specifier resolving to a file in dist/.
IMPORT = re.compile(
    r"""(?:^|[\s;}])(?:import|export)\s+(?:[^'"]*?\sfrom\s+)?["']([^"']+)["']""",
    re.MULTILINE,
)
SCRIPT_SRC = re.compile(r"""<script[^>]+src=["']([^"']+)["']""", re.IGNORECASE)


# The manifest is a checked-in JS module, so nothing regenerates it when the
# shards are re-exported. These are the bytes per element it describes.
DTYPE_BYTES = {"int8": 1, "float16": 2, "float32": 4}


def default_manifest() -> dict:
    """The manifest the site actually loads, read out of its JS module."""
    text = (ROOT / "src" / "reference" / "manifest.js").read_text(encoding="utf-8")
    return json.loads(text.split("=", 1)[1].rsplit(";", 1)[0].strip())


def manifest_mismatches(model: Path) -> list[str]:
    """Every way the checked-in manifest disagrees with the shards being shipped.

    🔴 THE MANIFEST IS NO LONGER DERIVED FROM THE WEIGHTS. It used to be read
    from model/manifest.json, which the exporter wrote beside the shards it had
    just produced, so the two could not disagree. It is now a committed JS
    module the exporter FALLS BACK to, which inverts the dependency: re-export
    the weights and the manifest keeps describing the previous ones. Nothing
    fails - every offset still lands inside a file of about the right size - and
    the page loads tensors sliced at the wrong byte and folds to noise.

    So the offsets are checked against the shards here, where the two are
    packaged together and a mismatch can still stop the deploy.
    """
    problems = []
    manifest = default_manifest()
    tensors = manifest.get("tensors")
    if not tensors:
        return ["src/reference/manifest.js has no tensor table"]

    # ...the manifest may also still exist beside the weights. When it does it
    # was written by the exporter, so it is the authority and any difference
    # means the committed copy is stale.
    on_disk = model / "manifest.json"
    if on_disk.is_file():
        exported = json.loads(on_disk.read_text(encoding="utf-8"))
        if exported.get("tensors") != tensors:
            problems.append(
                "src/reference/manifest.js disagrees with model/manifest.json;"
                " re-run tools/export-js-weights.py")

    # ...the strongest check available, and the one that actually catches the
    # failure this guards: a manifest describing a PREVIOUS export sits at
    # plausible offsets inside shards of plausible size, so only the bytes
    # themselves distinguish it from a current one.
    digests = manifest.get("shardDigests")
    if not digests:
        problems.append("src/reference/manifest.js has no shardDigests table")
    else:
        for shard, expected in sorted(digests.items()):
            path = model / shard
            if not path.is_file():
                continue  # reported below, with the tensors that name it
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                problems.append(
                    f"{shard}: sha256 {actual[:16]}... but the manifest records"
                    f" {expected[:16]}...; the manifest describes different weights")

    required: dict[str, int] = {}
    for name, tensor in tensors.items():
        dtype = tensor.get("dtype")
        if dtype not in DTYPE_BYTES:
            problems.append(f"{name}: unknown dtype {dtype!r}")
            continue
        elements = 1
        for extent in tensor["shape"]:
            elements *= extent
        end = tensor["byteOffset"] + elements * DTYPE_BYTES[dtype]
        if dtype == "int8":
            # ...int8 tensors carry a float16 scale per block, stored separately.
            block = tensor["block"]
            blocks = -(-elements // block)
            end = max(end, tensor["scaleOffset"] + blocks * 2)
        shard = tensor["file"]
        required[shard] = max(required.get(shard, 0), end)

    for shard, extent in sorted(required.items()):
        path = model / shard
        if not path.is_file():
            problems.append(f"{shard}: named by the manifest but not in model/")
        elif path.stat().st_size < extent:
            problems.append(
                f"{shard}: {path.stat().st_size} bytes, but the manifest reads to {extent}")
    return problems


def unresolved_imports(root: Path) -> list[str]:
    """Every relative import under root that does not point at a file."""
    problems = []
    sources = list(root.rglob("*.js")) + list(root.glob("*.html"))
    for path in sources:
        text = path.read_text(encoding="utf-8", errors="replace")
        specifiers = IMPORT.findall(text)
        if path.suffix == ".html":
            specifiers += SCRIPT_SRC.findall(text)
        for specifier in specifiers:
            if specifier.startswith(("http://", "https://", "//", "data:")):
                continue
            target = specifier.split("?", 1)[0].split("#", 1)[0]
            if not target.startswith("."):
                # bare or absolute: absolute is resolved against the site root
                if not target.startswith("/"):
                    continue
                resolved = root / target.lstrip("/")
            else:
                resolved = (path.parent / target).resolve()
            if not resolved.is_file():
                problems.append(f"{path.relative_to(root)} -> {specifier}")
    return sorted(problems)


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
        mismatches = manifest_mismatches(model)
        if mismatches:
            print("the checked-in manifest does not describe model/:", file=sys.stderr)
            for mismatch in mismatches:
                print(f"  {mismatch}", file=sys.stderr)
            print("the site would load tensors at the wrong offsets and fold to noise;"
                  " regenerate src/reference/manifest.js from the current export",
                  file=sys.stderr)
            return 1
        shutil.copytree(model, OUT / "model",
                        ignore=shutil.ignore_patterns("*.pyc", "__pycache__", ".DS_Store",
                                                      "*.map", "weights-*.js", "manifest.js"))

    problems = unresolved_imports(OUT)
    if problems:
        print(f"{len(problems)} import(s) do not resolve inside dist/:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print("the site would load in a checkout and 404 once deployed;"
              " check .gitignore is not swallowing a source directory", file=sys.stderr)
        return 1

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
