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
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
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

# ...imported rather than restated: tools/write_manifest_module.py owns the
# Python-side description of a bundle, and a second copy here would be one more
# place to forget when a model is added.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from write_manifest_module import BUNDLES  # noqa: E402


def build_commit() -> str:
    """The commit this build came from: the CI one, or the checkout's HEAD."""
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def compiled_manifest(module: Path) -> dict:
    """The manifest the site actually loads, read out of its generated module."""
    text = module.read_text(encoding="utf-8")
    return json.loads(text.split("=", 1)[1].rsplit(";", 1)[0].strip())


def registry_mismatches() -> list[str]:
    """The Python and JS descriptions of the model bundles, checked against each other.

    🔴 A FAMILY IN ONE REGISTRY AND NOT THE OTHER is a page that offers a model
    it cannot load, or a manifest nobody regenerates. They are two files because
    one is read by a build and the other by a browser; they are checked here so
    that being two files cannot mean being two answers.
    """
    index = (ROOT / "src" / "reference" / "manifests" / "index.js").read_text(encoding="utf-8")
    in_js = set(re.findall(r"^  (\w+): \{$", index, re.MULTILINE))
    in_py = set(BUNDLES)
    problems = []
    for family in sorted(in_py - in_js):
        problems.append(f"{family}: in tools/write_manifest_module.py but not in"
                        " src/reference/manifests/index.js")
    for family in sorted(in_js - in_py):
        problems.append(f"{family}: in src/reference/manifests/index.js but not in"
                        " tools/write_manifest_module.py")
    for family in sorted(in_py & in_js):
        module = ROOT / BUNDLES[family]["module"]
        if not module.is_file():
            problems.append(f"{family}: {BUNDLES[family]['module']} does not exist;"
                            f" run python3 tools/write_manifest_module.py {family}")
    return problems


def manifest_mismatches(model: Path, module: Path) -> list[str]:
    """Every way a compiled-in manifest disagrees with the shards being shipped.

    🔴 THE MANIFEST IS NOT DERIVED FROM THE WEIGHTS AT LOAD TIME. It is a
    committed JS module, which inverts the dependency: re-export the shards and
    the module keeps describing the previous ones. Nothing fails - every offset
    still lands inside a file of about the right size - and the page loads
    tensors sliced at the wrong byte and folds to noise.

    So the offsets are checked against the shards here, where the two are
    packaged together and a mismatch can still stop the deploy. This is the
    whole reason a compiled-in manifest is safe to keep.
    """
    problems = []
    manifest = compiled_manifest(module)
    relative = module.relative_to(ROOT)
    tensors = manifest.get("tensors")
    if not tensors:
        return [f"{relative} has no tensor table"]

    # ...the manifest may also still exist beside the weights. When it does it
    # was written by the exporter, so it is the authority and any difference
    # means the committed copy is stale.
    on_disk = model / "manifest.json"
    if on_disk.is_file():
        exported = json.loads(on_disk.read_text(encoding="utf-8"))
        if exported.get("tensors") != tensors:
            problems.append(
                f"{relative} disagrees with {model.name}/manifest.json;"
                " re-run tools/write_manifest_module.py")

    # ...the strongest check available, and the one that actually catches the
    # failure this guards: a manifest describing a PREVIOUS export sits at
    # plausible offsets inside shards of plausible size, so only the bytes
    # themselves distinguish it from a current one.
    digests = manifest.get("shardDigests")
    if not digests:
        problems.append(f"{relative} has no shardDigests table")
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
            problems.append(f"{shard}: named by the manifest but not in {model.name}/")
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

    # 🔴 WHAT THE SITE IS SERVING, ANSWERABLE FROM OUTSIDE IT. "Is it live?"
    # used to be answered by eye - fetch a file, squint at its bytes - and got
    # the wrong answer for an hour, because this repository is a FORK and a fork
    # does not run its workflows on push. Deploys only ever happened when
    # someone dispatched one by hand, and nothing said so.
    #
    # This stamp makes deployment machine-checkable: tools/deploy.py polls it
    # until it reports the commit that was pushed, so "live" is a fact rather
    # than an impression.
    (OUT / "build.json").write_text(json.dumps({
        "commit": build_commit(),
        "builtAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }) + "\n", encoding="utf-8")

    # THE PARAMETERS ARE OPT-IN, and they are the whole reason the workflow has
    # a repository variable: GitHub Pages is public even when its source
    # repository is private, so a model directory that happens to be lying
    # around in the checkout must not publish itself.
    if include_model:
        # ...EVERY FAMILY THE REGISTRY KNOWS, on the same terms. The Pages
        # workflow unpacks each release bundle into dist/ itself, so this path
        # is for a local build; either way a bundle that is present is checked
        # and one that is absent is simply not shipped.
        problems = registry_mismatches()
        if problems:
            print("the model registries disagree:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        shipped = 0
        for family, bundle in sorted(BUNDLES.items()):
            model = ROOT / bundle["export"]
            if not model.is_dir():
                continue
            # ...the .bin shards only. A served page reads those through fetch,
            # and the base64 scripts beside them are a third larger and exist
            # purely for file:// - shipping both would put a 356 MiB site at
            # 830 MiB, most of a GitHub Pages allowance spent on bytes nothing
            # on that site can use.
            mismatches = manifest_mismatches(model, ROOT / bundle["module"])
            if mismatches:
                print(f"{bundle['module']} does not describe {bundle['export']}/:",
                      file=sys.stderr)
                for mismatch in mismatches:
                    print(f"  {mismatch}", file=sys.stderr)
                print("the site would load tensors at the wrong offsets and fold to"
                      f" noise; run python3 tools/write_manifest_module.py {family}",
                      file=sys.stderr)
                return 1
            shutil.copytree(model, OUT / bundle["export"],
                            ignore=shutil.ignore_patterns("*.pyc", "__pycache__", ".DS_Store", "*.map",
                                                          "weights-*.js", "manifest.js",
                                                          "manifest.json"))
            shipped += 1
        if shipped == 0:
            print("--model was given but no export directory exists;"
                  " run `node tools/export-web-model.js <manifest>` first", file=sys.stderr)
            return 1

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
