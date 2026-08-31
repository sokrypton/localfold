"""Compile an export's manifest into the JS module the page loads.

    python3 tools/write_manifest_module.py monomer
    python3 tools/write_manifest_module.py multimer
    python3 tools/write_manifest_module.py --all

WHY THE MANIFEST IS COMPILED IN RATHER THAN FETCHED. The page needs the tensor
table before it can ask for a single weight, so fetching it costs a round trip
before the first byte of a 97 MiB download - and it is a request that can fail
on its own, which is exactly how the multimer path used to break: a site with no
multimer bundle 404ed on model-multimer/manifest.json and the fold died there.
A module cannot 404. It ships with the code that reads it.

WHY THAT IS DANGEROUS, AND WHAT PAYS FOR IT. Compiling it in INVERTS THE
DEPENDENCY: the manifest no longer comes from the weights, so re-export the
shards and the committed module keeps describing the previous ones. Nothing
crashes - every offset still lands inside a file of roughly the right size - and
the page loads tensors sliced at the wrong byte and folds to noise.

The sha256 of each shard is therefore written into the module, and
tools/build_site.py refuses to package an export whose bytes do not match. That
turns a silent wrong answer into a build failure, which is the only reason
compiling the manifest in is safe. Every module this writes carries digests;
one without them is rejected rather than trusted.

Run this after any re-export, and commit the module it writes.
"""
import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 🔴 THE ONE PLACE A MODEL BUNDLE IS DESCRIBED, on the Python side. Its twin is
# MODEL_BUNDLES in src/reference/manifests/index.js, and the two are checked
# against each other by tools/build_site.py - a family added to one and not the
# other is a build failure rather than a page that half works.
BUNDLES = {
    "monomer": {
        "export": "model",
        "module": "src/reference/manifests/monomer.js",
        "model": "model_1_ptm",
    },
    "multimer": {
        "export": "model-multimer",
        "module": "src/reference/manifests/multimer.js",
        "model": "model_1_multimer_v3",
    },
}


def read_module(module: pathlib.Path) -> dict:
    """The manifest out of a generated module, which is one JSON object literal."""
    text = module.read_text(encoding="utf-8")
    return json.loads(text.split("=", 1)[1].rsplit(";", 1)[0].strip())


def shard_digests(export: pathlib.Path, manifest: dict) -> dict[str, str]:
    """sha256 per shard the tensor table names, in a stable order."""
    shards = sorted({tensor["file"] for tensor in manifest["tensors"].values()})
    digests = {}
    for shard in shards:
        path = export / shard
        if not path.is_file():
            raise SystemExit(f"{export}/{shard} is named by the manifest but is not there")
        digests[shard] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def write(family: str) -> int:
    bundle = BUNDLES[family]
    export = ROOT / bundle["export"]
    module = ROOT / bundle["module"]
    source = export / "manifest.json"
    # 🔴 TWO PLACES A TENSOR TABLE CAN COME FROM, and the exporter's wins. The
    # multimer exporter writes manifest.json beside its shards, so that is the
    # authority. The monomer's does not - its table has only ever existed as the
    # committed module - so there the module is re-read and only the digests are
    # recomputed from the shards on disk. Refreshing digests against a table
    # that did not change is still worth doing: it is what proves the committed
    # table describes the shards about to be packaged.
    if source.is_file():
        manifest = json.loads(source.read_text(encoding="utf-8"))
        origin = f"{bundle['export']}/manifest.json"
    elif module.is_file():
        manifest = read_module(module)
        origin = f"{bundle['module']} (no exporter manifest; digests refreshed)"
    else:
        print(f"neither {bundle['export']}/manifest.json nor {bundle['module']} exists;"
              " export the weights first", file=sys.stderr)
        return 1
    if not manifest.get("tensors"):
        print(f"{origin} has no tensor table", file=sys.stderr)
        return 1

    # ...recomputed rather than copied. The exporter does not write digests, and
    # a digest carried over from a previous run would defeat the check it exists
    # for.
    manifest["shardDigests"] = shard_digests(export, manifest)

    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(
        f"/**\n"
        f" * The {bundle['model']} tensor table, compiled in rather than fetched.\n"
        f" *\n"
        f" * GENERATED - do not edit. Re-export the weights, then:\n"
        f" *   python3 tools/write_manifest_module.py {family}\n"
        f" *\n"
        f" * tools/build_site.py checks shardDigests against the shards being\n"
        f" * packaged and fails the build if they disagree, which is what makes a\n"
        f" * committed copy of a derived artefact safe to keep.\n"
        f" */\n"
        f"export const MANIFEST = {json.dumps(manifest)};\n",
        encoding="utf-8",
    )
    tensors = len(manifest["tensors"])
    shards = len(manifest["shardDigests"])
    size = module.stat().st_size
    print(f"{bundle['module']}  {tensors} tensors, {shards} shards,"
          f" {size / 1024:.0f} KiB  <- {origin}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("family", nargs="?", choices=sorted(BUNDLES))
    parser.add_argument("--all", action="store_true",
                        help="write every bundle whose export is present")
    arguments = parser.parse_args()
    if arguments.all:
        present = [name for name in sorted(BUNDLES)
                   if (ROOT / BUNDLES[name]["export"]).is_dir()]
        if not present:
            print("no export directory is present; nothing to write", file=sys.stderr)
            return 1
        return max(write(name) for name in present)
    if arguments.family is None:
        parser.error("name a family, or pass --all")
    return write(arguments.family)


if __name__ == "__main__":
    raise SystemExit(main())
