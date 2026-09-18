"""Assemble the deployable site into dist/.

    python3 tools/build_site.py            # dist/, without model parameters
    python3 tools/build_site.py --model    # ...and the model/ directory too

WHY THIS EXISTS, AND WHY IT IS NOT A BUNDLER.

The page loads src/**/*.js as written - plain ES modules, resolved by the
browser through the same relative paths they have in the checkout. So there is
nothing to compile, and the only real question a "build" answers here is WHICH
FILES a public site should contain: not test/, not tools/, and not the 946 KB
tools/fixtures/test.a3m or the fixtures under test/fixtures, which together dwarf the app.

That makes this a copy with an allow-list, and the allow-list is the point. It
is written out below rather than derived, because a derived rule ("everything
but test/") silently ships the next directory somebody adds.

THE LAYOUT IS PRESERVED, exactly. The pages at the top, web/ and src/ beside
them, because index.html says ./web/app.js and app.js says ../src/af2/model/... -
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
# The domain the published site answers to. See the CNAME note in build().
DOMAIN = "localfold.org"

# WHAT A PUBLIC SITE CONTAINS. Files are copied as-is; directories are copied
# whole, minus the ignore patterns below.
FILES = [".nojekyll", "index.html", "dev.html"]
# 🔴 HELD BACK UNTIL THEY ARE FIXED AND CHECKED, NOT DELETED. `single.html` and
# `proteinhunter.html` are out of the repository and out of the site while the
# model row they were built against moves under them; the files stay on disk and
# gitignored, and every tool that touches them SKIPS a page it cannot find
# rather than dropping its probe - so putting them back is one line here and the
# gates come back with them.
OPTIONAL = ["single.html", "proteinhunter.html"]
DIRECTORIES = ["web", "src"]

# ...and never these, wherever they appear.
# 🔴 `.ipynb_checkpoints` IS IN HERE BECAUSE THE SITE WAS SHIPPING THEM. A
# jupyter-lab running against this checkout writes
# `<dir>/.ipynb_checkpoints/<name>-checkpoint.js` beside every file it saves,
# and `copytree` took them: dist/src carried 188 files against src/'s 180 -
# eight stale snapshots of real modules, published. Nothing imports them, so
# nothing broke; they were simply somebody's editor state on a public site.
#
# The unresolved-import CHECK below learned to skip them the same day and this
# did not, which is the lesson: a rule applied to the inspection and not to the
# COPY leaves the artefact in the artefact.
IGNORE = shutil.ignore_patterns("*.pyc", "__pycache__", ".DS_Store", "*.map",
                                ".ipynb_checkpoints")



# WHY THE BUILD RESOLVES THE MODULE GRAPH.
#
# 🔴 A DEPLOY ONCE 404'd ON TWO MODULES THAT EVERY CHECKOUT HAD. .gitignore said
# `model/` for the exported weights, and an unanchored pattern matches a
# directory of that name at ANY depth - so the `model/` directory under `src/`
# was silently untracked, the files existed locally, the site built, and the
# published page failed to load web/app.js with no error anyone would see.
# Copying is not enough: a build has to answer "does what I just assembled
# actually load", and for ES modules that means every relative specifier
# resolving to a file in dist/.
IMPORT = re.compile(
    r"""(?:^|[\s;}])(?:import|export)\s+(?:[^'"]*?\sfrom\s+)?["']([^"']+)["']""",
    re.MULTILINE,
)
SCRIPT_SRC = re.compile(r"""<script[^>]+src=["']([^"']+)["']""", re.IGNORECASE)


# The manifest is a checked-in JS module, so nothing regenerates it when the
# shards are re-exported. These are the bytes per element it describes.
DTYPE_BYTES = {"int8": 1, "float16": 2, "float32": 4}
# 🔴 A PACKED WIDTH IS NOT A WHOLE NUMBER OF BYTES, which is why none of them is
# in the table above. Thirty-two five-bit codes are exactly 160 bits, so a group
# is exactly 20 bytes and no group straddles another - see tools/quantize_af3.py,
# where that is the reason group 32 was chosen - and thirty-two three-bit codes
# are 12. The reader may take two bytes for a code ending on the final one, so
# there is one trailing byte of slack. `manifest_mismatches` derives the span
# from the width in the dtype's name rather than knowing one of them.

# ...imported rather than restated: tools/write_manifest_module.py owns the
# Python-side description of a bundle, and a second copy here would be one more
# place to forget when a model is added.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from write_manifest_module import BUNDLES  # noqa: E402

# Model names whose parameters carry redistribution terms, and the terms they
# carry. Publishing one is permitted here - the maintainers hold an academic
# licence for it and the page gates the download behind acceptance - but only
# deliberately: set LOCALFOLD_ACCEPT_MODEL_TERMS to the names being published.
#
# 🔴 THIS IS A GATE, NOT A JUDGEMENT. It exists because ONE COMMAND FLAG chooses
# which checkpoint fills a bundle directory, so the difference between weights
# that may be served and weights that may not is invisible in the tree. The
# manifest records which is in there; this makes something read it before the
# bytes go somewhere public.
RESTRICTED_TERMS = {
    "alphafold3": "DeepMind's AF3 Weights Terms of Use and Prohibited Use Policy",
}
ACCEPTED_TERMS = frozenset(
    name.strip() for name in os.environ.get("LOCALFOLD_ACCEPT_MODEL_TERMS", "").split(",")
    if name.strip()
)


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
    index = (ROOT / "src" / "bundles" / "manifests" / "index.js").read_text(encoding="utf-8")
    # ...a key is quoted when it is not a bare identifier, which
    # `ef2-fast-600m` is not. Matching only unquoted keys made this check
    # report a family as MISSING from the file it is defined in.
    in_js = set(re.findall(r'^  "?([\w-]+)"?: \{$', index, re.MULTILINE))
    in_py = set(BUNDLES)
    problems = []
    for family in sorted(in_py - in_js):
        problems.append(f"{family}: in tools/write_manifest_module.py but not in"
                        " src/bundles/manifests/index.js")
    for family in sorted(in_js - in_py):
        problems.append(f"{family}: in src/bundles/manifests/index.js but not in"
                        " tools/write_manifest_module.py")
    for family in sorted(in_py & in_js):
        module = ROOT / BUNDLES[family]["module"]
        if not module.is_file():
            problems.append(f"{family}: {BUNDLES[family]['module']} does not exist;"
                            f" run python3 tools/write_manifest_module.py {family}")
    return problems


def unpublished_families() -> set[str]:
    """The families whose shards are not published ANYWHERE yet.

    🔴 `remote: null` IS NOT `remote: "..."` AND IS NOT A LOCAL BUNDLE EITHER.
    A model that is ported but whose weights have not been uploaded has neither,
    and `remote_families` - which matches a quoted URL - reads it as local and
    would publish 612 MiB of it into a Pages allowance of one gigabyte, from
    whatever export directory happens to be lying around in the checkout. It is
    skipped instead, and the skip is PRINTED, because a bundle silently missing
    from a site is the failure mode this whole file exists to stop.
    """
    index = (ROOT / "src" / "bundles" / "manifests" / "index.js").read_text(encoding="utf-8")
    families = set()
    family = None
    for line in index.splitlines():
        opened = re.match(r'^  "?([\w-]+)"?: \{$', line)
        if opened:
            family = opened.group(1)
        elif family is not None and re.match(r"^\s*remote:\s*null\s*,\s*$", line):
            families.add(family)
        elif line == "  },":
            family = None
    return families


def remote_families() -> set[str]:
    """The families whose shards are fetched from somewhere else.

    🔴 A BUNDLE WITH A `remote` MUST NOT BE PUBLISHED HERE. GitHub Pages caps a
    published site at a gigabyte and the weights are most of it - AF2 monomer is
    227 MB and AF3 150 MB before a third model exists - so a page meaning to
    offer five of them keeps its parameters elsewhere. Shipping them anyway
    would spend the allowance twice: once on the artefact and once on a copy no
    page fetches, because the browser resolves shards against the remote.

    Read out of index.js rather than duplicated here, for the reason
    registry_mismatches gives: two files may not mean two answers.
    """
    index = (ROOT / "src" / "bundles" / "manifests" / "index.js").read_text(encoding="utf-8")
    families = set()
    family = None
    for line in index.splitlines():
        # 🔴 A KEY IS QUOTED WHEN IT IS NOT A BARE IDENTIFIER, and three of the
        # four EF2 families are. Matching only unquoted keys left this reading
        # `remote:` lines as belonging to no family at all - so bundles that ARE
        # hosted were counted as local and would have been published a second
        # time, spending the Pages allowance on a copy no page fetches. The same
        # pattern was wrong in registry_mismatches and in two JavaScript tests.
        opened = re.match(r'^  "?([\w-]+)"?: \{$', line)
        if opened:
            family = opened.group(1)
        elif family is not None and re.match(r"^\s*remote:\s*[\"']", line):
            families.add(family)
        elif line == "  },":
            family = None
    return families


def unreachable_offers() -> list[str]:
    """Families the PAGE offers whose shards no visitor can fetch.

    🔴 A MODEL IN `index.html` WITH NO `remote:` IS A 404 FOR EVERY VISITOR, AND
    NOTHING SAID SO. `bundleBaseUrl` is `bundle.remote ?? bundle.directory`, and
    the directory - `./model-intellifold2-int5/` - is a LOCAL export that this
    build deliberately never copies, because GitHub Pages caps a site at a
    gigabyte and IntelliFold-2 alone is 612 MiB. So a family wired into the
    picker before its bundle is published resolves to a path that is not on the
    site: the option appears, the visitor chooses it, and the fold dies fetching
    shard zero.

    It has been latent rather than shipped only because the deploy is behind:
    the live commit does not carry those two options and HEAD does. That is the
    worst shape for a defect - correct in the tree, correct on the site, broken
    the moment the two meet - and it is CLAUDE.md's own "a bundle the CLI likes
    can be one the PAGE cannot load", one turn further out.

    Local directories ARE legitimate for a developer, which is why this asks
    what the OPTION list offers rather than what the registry contains.
    """
    offered = set(re.findall(r'<option value="([\w-]+)"',
                             (ROOT / "index.html").read_text(encoding="utf-8")))
    hosted = remote_families()
    index = (ROOT / "src" / "bundles" / "manifests" / "index.js").read_text(encoding="utf-8")
    known = set(re.findall(r'^  "?([\w-]+)"?: \{$', index, re.MULTILINE))
    # An <option> that is not a model family at all - a sampler, a preset - is
    # not this check's business.
    return sorted(family for family in offered & known if family not in hosted)


def unreachable_model_numbers() -> list[str]:
    """AlphaFold 2 model numbers whose bundle no visitor can fetch.

    🔴 THE SAME TRAP AS `unreachable_offers`, ONE CONTROL FURTHER IN. AF2's five
    models are picked by a NUMBER beside the model row rather than by five
    entries in it, and `chosenFamily` resolves "monomer" plus "3" to the family
    `monomer-3`. So an unpublished delta bundle is not an `<option value=
    "monomer-3">` anywhere for that check to find - it is an option reading "3",
    and dropping the family list alone would leave the number on screen and the
    fold dying on shard zero.
    """
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    block = re.search(r'<select id="af2Model">(.*?)</select>', page, re.DOTALL)
    if block is None:
        return []
    hosted = remote_families()
    numbers = re.findall(r'<option value="(\d+)"', block.group(1))
    return [number for number in numbers
            if number != "1" and f"monomer-{number}" not in hosted]


def restricted_terms(module: Path) -> str | None:
    """The restricted licence this bundle's weights carry, if unaccepted.

    Read from what the artefact SAYS IT IS - the manifest's model.name - and
    not from the directory it sits in, because one command flag chooses which
    checkpoint of a lineage fills a bundle directory and the tree looks
    identical either way.
    """
    named = (compiled_manifest(module).get("model") or {}).get("name")
    if named in RESTRICTED_TERMS and named not in ACCEPTED_TERMS:
        return named
    return None


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
    #
    # 🔴 EVERY KEY, NOT THE TENSOR TABLE. This compared `tensors` alone, and a
    # manifest is more than its tensors: `distogramHead` carries AlphaFold 2's
    # bin breaks, and those were wrong in both AF2 manifests for a long time
    # with this check looking straight past them. Anything the exporter wrote
    # and the module copied can drift, so the comparison is over the whole
    # object. `shardDigests` is the one exception - the writer computes it from
    # the shards and the exporter does not write it at all.
    on_disk = model / "manifest.json"
    if on_disk.is_file():
        exported = json.loads(on_disk.read_text(encoding="utf-8"))
        keys = (set(exported) | set(manifest)) - {"shardDigests"}
        differing = sorted(key for key in keys
                           if exported.get(key) != manifest.get(key))
        if differing:
            problems.append(
                f"{relative} disagrees with {model.name}/manifest.json on"
                f" {', '.join(differing)}; re-run tools/write_manifest_module.py")

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
        # 🔴 ANY PACKED WIDTH, NOT int5 ALONE. This tested for the literal string
        # because int5 was the only packed dtype when it was written, and the
        # ESM-C bundle ships int3 - so a correct manifest was rejected with
        # "unknown dtype 'int3'" and advice to regenerate it, which would have
        # produced the identical file. src/weights/dtype.js has decoded int1
        # through int7 the whole time; this is the second place that knew about
        # one width, after `BYTES` in that same file.
        packed = re.fullmatch(r"int([1-7])", dtype or "")
        if dtype not in DTYPE_BYTES and packed is None:
            problems.append(f"{name}: unknown dtype {dtype!r}")
            continue
        elements = 1
        for extent in tensor["shape"]:
            elements *= extent
        if packed is not None:
            # ...packed groups plus the slack byte, then a float16 scale AND a
            # float16 zero point per group: these are asymmetric, so there are
            # two tables after the codes and not one.
            bits = int(packed.group(1))
            block = tensor["block"]
            if (block * bits) % 8:
                problems.append(f"{name}: {dtype} at group {block} does not pack"
                                " into whole bytes")
                continue
            blocks = -(-elements // block)
            end = tensor["byteOffset"] + blocks * (block * bits // 8) + 1
            end = max(end, tensor["zeroOffset"] + blocks * 2)
        else:
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
    # 🔴 AN EDITOR'S SNAPSHOT IS NOT A SOURCE FILE, AND THIS WALKED THEM INTO
    # THE DEPLOY CHECK. A jupyter-lab running against this checkout writes
    # `<dir>/.ipynb_checkpoints/<name>-checkpoint.js` whenever a file is saved -
    # stale copies whose imports are whatever that file said when it was last
    # snapshotted. After the src/ reorganisation those paths stopped resolving
    # and this function failed the whole build on two files nobody wrote, with
    # advice ("check .gitignore is not swallowing a source directory") that
    # points away from the cause. They ARE gitignored; they simply are not
    # source. The JS side shares one walker for this rule -
    # test/helpers/source-files.js - and this is the Python copy, which cannot
    # import it. If that rule changes, change it here too.
    sources = [p for p in root.rglob("*.js")
               if not any(part.startswith(".") for part in p.parts)]
    sources += list(root.glob("*.html"))
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
    for name in OPTIONAL:
        source = ROOT / name
        if source.exists():
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

    # ...EVERY FAMILY THE REGISTRY KNOWS, on the same terms, WHETHER OR NOT ANY
    # PARAMETERS ARE BEING PUBLISHED.
    #
    # 🔴 CHECKING IS NOT PUBLISHING, AND CONFLATING THEM MADE THIS GATE DEAD.
    # The registry check and the manifest check used to sit inside `if
    # include_model:`, which was sound while bundles were published from here.
    # Every bundle is hosted on Hugging Face now and the Pages workflow runs
    # `build_site.py` with no --model at all - so the one check that says the
    # compiled manifest still describes the shipped weights had stopped running
    # on every deploy. What a bundle's manifest SAYS is shipped with the page
    # regardless of where its shards live; only the COPY is opt-in.
    problems = registry_mismatches()
    if problems:
        print("the model registries disagree:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    remote = remote_families()
    unpublished = unpublished_families()
    for family, bundle in sorted(BUNDLES.items()):
        model = ROOT / bundle["export"]
        if not model.is_dir():
            continue
        # 🔴 A REMOTE BUNDLE IS CHECKED TOO, BECAUSE ITS MODULE STILL SHIPS.
        # Only its SHARDS live elsewhere. The publish path skipped these before
        # the check, so the two AF2 families - the only ones with an exporter
        # manifest to compare against at all - were the two never compared.
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

    # THE PARAMETERS ARE OPT-IN, and they are the whole reason the workflow has
    # a repository variable: GitHub Pages is public even when its source
    # repository is private, so a model directory that happens to be lying
    # around in the checkout must not publish itself.
    if include_model:
        shipped = 0
        for family, bundle in sorted(BUNDLES.items()):
            model = ROOT / bundle["export"]
            if not model.is_dir():
                continue
            if family in remote:
                print(f"{bundle['export']}/ is hosted remotely; not publishing it")
                continue
            if family in unpublished:
                print(f"{bundle['export']}/ has no remote yet; not publishing it")
                continue
            # ...the .bin shards only. A served page reads those through fetch,
            # and the base64 scripts beside them are a third larger and exist
            # purely for file:// - shipping both would put a 356 MiB site at
            # 830 MiB, most of a GitHub Pages allowance spent on bytes nothing
            # on that site can use.
            restricted = restricted_terms(ROOT / bundle["module"])
            if restricted is not None:
                print(f"{bundle['export']}/ holds {restricted} parameters, which"
                      f" carry {RESTRICTED_TERMS[restricted]}.", file=sys.stderr)
                print("Publishing them is a deliberate act, so it is opt-in:"
                      f" set LOCALFOLD_ACCEPT_MODEL_TERMS={restricted} to confirm"
                      " the licence covers this deployment, or re-export the"
                      " bundle from a checkpoint that carries no such terms.",
                      file=sys.stderr)
                return 1

            shutil.copytree(model, OUT / bundle["export"],
                            ignore=shutil.ignore_patterns("*.pyc", "__pycache__", ".DS_Store", "*.map",
                                                          "weights-*.js", "manifest.js",
                                                          "manifest.json"))
            shipped += 1
        # 🔴 "NOTHING WAS PUBLISHED" IS NOT "NOTHING EXISTS", NOW THAT EVERY
        # BUNDLE IS HOSTED. --model asked for parameters and skipped every one
        # as remote, which is the correct outcome and used to exit 1 saying no
        # export directory exists - a message naming a cause that is not the
        # cause, with the directories sitting right there.
        if shipped == 0 and not any((ROOT / bundle["export"]).is_dir()
                                    for bundle in BUNDLES.values()):
            print("--model was given but no export directory exists;"
                  " run `node tools/export-web-model.js <manifest>` first", file=sys.stderr)
            return 1

    # 🔴 THE PUBLISHED TREE MUST BE THE SOURCE TREE, FILE FOR FILE. dist/src
    # carried 188 .js against src/'s 180 and nobody noticed: eight
    # `.ipynb_checkpoints` snapshots, copied straight onto a public site by
    # `copytree`. Nothing imported them so nothing broke, and that is exactly
    # why it needs asserting rather than watching - a stray file in the deploy
    # has no symptom until it is somebody's stale code on the internet.
    published = {p.relative_to(OUT / "src") for p in (OUT / "src").rglob("*.js")}
    authored = {p.relative_to(ROOT / "src") for p in (ROOT / "src").rglob("*.js")
                if not any(part.startswith(".") for part in p.parts)}
    if published != authored:
        extra = sorted(str(p) for p in published - authored)
        missing = sorted(str(p) for p in authored - published)
        for p in extra:
            print(f"dist/src carries {p}, which is not a source file", file=sys.stderr)
        for p in missing:
            print(f"dist/src is missing {p}", file=sys.stderr)
        return 1

    # 🔴 AND THE PAGE MUST NOT OFFER A MODEL THE SITE CANNOT SERVE, so the BUILD
    # takes the option out rather than a person remembering to. See
    # unreachable_offers: every other check here asks whether what was copied is
    # right, and this one asks whether what was NOT copied is still advertised.
    #
    # Dropped from dist/ and never from the checkout: a developer with the local
    # export in place is exactly who those options are for, and `tools/serve.py`
    # serves them. Publishing the bundle and re-pinning its `remote` makes the
    # option come back with no edit here or in index.html - the registry decides,
    # which is the point.
    #
    # 🔴 AND IT SAYS SO EVERY BUILD, because a silent drop is the same bug one
    # step quieter: a model that vanishes from the picker with no line of output
    # looks exactly like a model nobody ported.
    unreachable = unreachable_offers()
    if unreachable:
        page = (OUT / "index.html").read_text(encoding="utf-8")
        for family in unreachable:
            pattern = re.compile(rf'[ \t]*<option value="{re.escape(family)}"[^>]*>'
                                 r'[^<]*</option>\n?')
            page, count = pattern.subn("", page)
            if count != 1:
                print(f"expected one <option> for {family}, matched {count}",
                      file=sys.stderr)
                return 1
        (OUT / "index.html").write_text(page, encoding="utf-8")
        print(f"dropped {len(unreachable)} model option(s) the site cannot serve:"
              f" {', '.join(unreachable)}")
        print("  their `remote` is null, so bundleBaseUrl would fall back to a local"
              " directory this build does not publish.")
        print("  Publish the bundle and re-pin the remote (docs/HOSTING.md) and the"
              " option returns by itself.")

    # ...and the same for AlphaFold 2's model NUMBER, which is a control rather
    # than a family list and so invisible to the check above.
    numbers = unreachable_model_numbers()
    if numbers:
        page = (OUT / "index.html").read_text(encoding="utf-8")
        block = re.search(r'<select id="af2Model">(.*?)</select>', page, re.DOTALL)
        if block is None:
            print("index.html has no af2Model select to trim", file=sys.stderr)
            return 1
        trimmed = block.group(1)
        for number in numbers:
            pattern = re.compile(rf'[ \t]*<option value="{number}"[^>]*>[^<]*</option>\n?')
            trimmed, count = pattern.subn("", trimmed)
            if count != 1:
                print(f"expected one af2Model <option> for {number}, matched {count}",
                      file=sys.stderr)
                return 1
        page = page[:block.start(1)] + trimmed + page[block.end(1):]
        (OUT / "index.html").write_text(page, encoding="utf-8")
        print(f"dropped AlphaFold 2 model number(s) the site cannot serve: {', '.join(numbers)}")
        print("  their delta bundles have no `remote`; publish them and re-pin it"
              " (docs/HOSTING.md) and the numbers return by themselves.")

    problems = unresolved_imports(OUT)
    if problems:
        print(f"{len(problems)} import(s) do not resolve inside dist/:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print("the site would load in a checkout and 404 once deployed;"
              " check .gitignore is not swallowing a source directory", file=sys.stderr)
        return 1

    # 🔴 THE CUSTOM DOMAIN LIVES IN THE ARTIFACT, NOT ONLY IN THE SETTINGS.
    # This site publishes through upload-pages-artifact rather than a branch, and
    # for that route the CNAME file is what binds the domain to the repository -
    # a deploy without one can drop the setting, after which Pages serves 404 to
    # a domain whose DNS is perfectly correct. That failure reads as a DNS
    # problem and is not one: the request reaches GitHub, which does not know
    # whose site to answer with.
    (OUT / "CNAME").write_text(f"{DOMAIN}\n")

    total = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    count = sum(1 for path in OUT.rglob("*") if path.is_file())
    print(f"dist/  {count} files, {total / 1024 / 1024:.1f} MiB"
          f"{'' if include_model else '  (no model parameters)'}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="store_true",
                        help="include the exported model/ directory")
    # 🔴 SO THE WORKFLOW CAN ASK THE REGISTRY RATHER THAN REPEAT IT. The Pages
    # job unpacks each release bundle into dist/ ITSELF, after this script has
    # run, so a family skipped here is still published unless the job skips it
    # too - and a second list of which models are remote is a second answer.
    # Exits 0 when the family is hosted remotely, which is what `if` wants.
    parser.add_argument("--is-remote", metavar="FAMILY", default=None,
                        help="exit 0 if FAMILY's shards are fetched from elsewhere")
    arguments = parser.parse_args()
    if arguments.is_remote is not None:
        raise SystemExit(0 if arguments.is_remote in remote_families() else 1)
    raise SystemExit(build(arguments.model))
