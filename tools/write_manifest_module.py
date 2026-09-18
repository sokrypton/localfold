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
# MODEL_BUNDLES in src/bundles/manifests/index.js, and the two are checked
# against each other by tools/build_site.py - a family added to one and not the
# other is a build failure rather than a page that half works.
BUNDLES = {
    "monomer": {
        "export": "model",
        "module": "src/bundles/manifests/monomer.js",
        "model": "model_1_ptm",
    },
    # model_2_ptm as a delta on model_1_ptm - 43 MiB against 97. The module is
    # the delta's own tensor table; the base's comes from "monomer".
    "monomer-2": {
        "export": "model-mono-2-delta",
        "module": "src/bundles/manifests/monomer-2.js",
        "model": "model_2_ptm",
    },
    # model_3_ptm as a delta on model_1_ptm - 43 MiB against 97. The module is
    # the delta's own tensor table; the base's comes from "monomer".
    "monomer-3": {
        "export": "model-mono-3-delta",
        "module": "src/bundles/manifests/monomer-3.js",
        "model": "model_3_ptm",
    },
    # model_4_ptm as a delta on model_1_ptm - 43 MiB against 97. The module is
    # the delta's own tensor table; the base's comes from "monomer".
    "monomer-4": {
        "export": "model-mono-4-delta",
        "module": "src/bundles/manifests/monomer-4.js",
        "model": "model_4_ptm",
    },
    # model_5_ptm as a delta on model_1_ptm - 43 MiB against 97. The module is
    # the delta's own tensor table; the base's comes from "monomer".
    "monomer-5": {
        "export": "model-mono-5-delta",
        "module": "src/bundles/manifests/monomer-5.js",
        "model": "model_5_ptm",
    },
    "multimer": {
        "export": "model-multimer",
        "module": "src/bundles/manifests/multimer.js",
        "model": "model_1_multimer_v3",
    },
    # model_2_multimer_v3 as a delta on model_1 - 44 MiB against 74.
    "multimer-2": {
        "export": "model-multi-2-delta",
        "module": "src/bundles/manifests/multimer-2.js",
        "model": "model_2_multimer_v3",
    },
    # model_3_multimer_v3 as a delta on model_1 - 44 MiB against 74.
    "multimer-3": {
        "export": "model-multi-3-delta",
        "module": "src/bundles/manifests/multimer-3.js",
        "model": "model_3_multimer_v3",
    },
    # model_4_multimer_v3 as a delta on model_1 - 44 MiB against 74.
    "multimer-4": {
        "export": "model-multi-4-delta",
        "module": "src/bundles/manifests/multimer-4.js",
        "model": "model_4_multimer_v3",
    },
    # model_5_multimer_v3 as a delta on model_1 - 44 MiB against 74.
    "multimer-5": {
        "export": "model-multi-5-delta",
        "module": "src/bundles/manifests/multimer-5.js",
        "model": "model_5_multimer_v3",
    },
    # The whole AF3 diffuser - trunk, diffusion head and confidence head - at
    # int5, written by tools/export_af3_model.py and packed by
    # tools/quantize_af3.py. 265 MiB, and it folds.
    #
    # 🔴 THESE ARE DEEPMIND'S PARAMETERS, NOT OPENFOLD3'S. The manifest says so
    # (model.name is "alphafold3") and build_site.py refuses to publish them
    # without LOCALFOLD_ACCEPT_MODEL_TERMS=alphafold3.
    "af3": {
        "export": "model-af3-int5",
        "module": "src/bundles/manifests/af3.js",
        "model": "alphafold3",
    },
    # ...and the Apache-2.0 one, which is the same graph and the same 265 MiB.
    # OpenBind-0 is OpenFold3's v0.5.0 release; `--model openbind` exports it and
    # the manifest's model.name carries the dialect into the page, so the two
    # bundles cannot be confused for one another at load time. Nothing gates
    # this one: its weights carry no prohibited-use policy.
    "openbind0": {
        "export": "model-openbind0-int5",
        "module": "src/bundles/manifests/openbind0.js",
        "model": "openbind0",
    },
    # OpenDDE, whole: the trunk, the distogram head, the structural-token
    # expander and refiner, the diffusion, and OpenDDE's own confidence head.
    # 481 tensors, 655.8 M parameters - which is upstream's published
    # parameter_count exactly - at int5 group 32, 473 MiB in TWELVE shards.
    #
    # 🔴 TWELVE, AND SIZED ON THE PACKED BYTES. The float32 export's 48 MiB
    # shard cap is discarded; what decides the count is the largest TENSOR,
    # because a tensor is indivisible. OpenDDE has three 216 MiB float32
    # tensors, 40.5 MiB apiece at int5, so no sharding beats 40.5 MiB and
    # twelve is the fewest that reaches it. AlphaFold 3's bundle has the same
    # 40.5 MiB cap from two such tensors and is 8 shards only because it is
    # 265 MiB rather than 473. See quantize_af3.py's shard_count.
    #
    # 🔴 IT IS A TRUNK AND NOT A FOLD, which MODEL_BUNDLES records as
    # `foldingModel: false`. OpenDDE expands each residue into about two
    # structural tokens between the trunk and the diffusion and runs the
    # diffusion and its own confidence head on that expanded set, so the parts
    # that make COORDINATES are a second token space rather than a branch. The
    # distogram is what this bundle answers with. See docs/OPENDDE.md.
    "opendde": {
        "export": "model-opendde-int5",
        "module": "src/bundles/manifests/opendde.js",
        "model": "opendde",
    },
    # Boltz-2 (MIT), whole: trunk, diffusion and its own confidence head, at
    # int5 group 32. 364 MiB in eight shards, and the strongest model in the
    # reference's table - it folds 6MRR at RMSD 0.542 A / TM 0.972 against
    # AlphaFold 3's 0.657, with a seed spread of 0.07 A where protenix2's is
    # 1.03. See docs/AF3.md.
    #
    # 🔴 ITS SHAPE IS NOT AlphaFold 3's, and three constants in this port were
    # typed to AF3's: 64 pairformer blocks against 48, an 8-block confidence
    # stack against 4, and a 35-column MSA feature against 34. All three are
    # read off the weights now.
    "boltz2": {
        "export": "model-boltz2-int5",
        "module": "src/bundles/manifests/boltz2.js",
        "model": "boltz2",
    },
    # Protenix-v2, whole, at int5 group 32. 334 MiB in eight shards.
    #
    # 🔴 ITS SAMPLER HAS A ONE-ANGSTROM SEED SPREAD ON 6MRR - 0.607 to 1.642 A
    # over three seeds, where boltz2 spans 0.07 - so a single-seed comparison of
    # this model means nothing. That is a property of the model and not of the
    # port; docs/HOSTING.md has the table and the retraction it cost.
    "protenix2": {
        "export": "model-protenix2-int5",
        "module": "src/bundles/manifests/protenix2.js",
        "model": "protenix2",
    },
    # IntelliFold-2, whole, at int5 group 32. 612 MiB in twelve shards.
    #
    # 🔴 THE SMALLEST DIALECT IN THE FAMILY AND THE WIDEST BUNDLE. Its module
    # tree is stock AlphaFold 3's - the nine-projection template embedder, AF3's
    # MSA stack and confidence head - and only two conventions differ; what it
    # does not share is its SHAPES, with a 512-channel trunk pair and a
    # 256-channel template stack against AF3's 128 and 64. See docs/AF3.md.
    "intellifold2": {
        "export": "model-intellifold2-int5",
        "module": "src/bundles/manifests/intellifold2.js",
        "model": "intellifold2",
    },
    # RoseTTAFold3, whole, at int5 group 32. 266 MiB in six shards.
    #
    # 🔴 ITS TRUNK IS AT THE FAMILY'S BAND AND ITS FOLD IS NOT A CHAIN YET, so
    # this entry exists for the CLI and the gates rather than for the page -
    # MODEL_BUNDLES gates it behind LOCALFOLD_INCLUDE_ROSETTAFOLD3_MODEL and
    # nothing sets that. See docs/AF3.md for what is left.
    "rosettafold3": {
        "export": "model-rosettafold3-int5",
        "module": "src/bundles/manifests/rosettafold3.js",
        "model": "rosettafold3",
    },
    # ESMFold2-Experimental-Fast's folding half: the trunk, the inputs embedder
    # and the whole structure head, at int5 group 32. 122 MiB.
    #
    # 🔴 IT IS HALF A MODEL AND CANNOT FOLD ALONE. The language model is a
    # SEPARATE 224 MiB bundle with its own licence and its own exporter, and the
    # shim that joins them is per folding model - which is why the ESM-C
    # manifest carries both names. MODEL_BUNDLES.esmfold2 names `esmc` as its
    # companion so a page cannot load one without the other.
    # 🔴 THE KEY IS THE CHECKPOINT'S NAME, NOT THE ARCHITECTURE'S: "esmfold2"
    # alone reads as ESM's released ESMFold2-Fast, which folds from ESM-C 6B.
    # The bundle DIRECTORY keeps its older name, as openbind0's does - a path is
    # not the model's name, and renaming it would move 366 MiB for nothing.
    "ef2-fast-600m": {
        "export": "model-esmfold2-int5",
        "module": "src/bundles/manifests/esmfold2.js",
        "model": "esmfold2-trunk",
    },
    # ESM-C 600M at int3 group 128, plus the shim that turns its 37 hidden
    # states into ESMFold2's pair term. Three bits, because the structural
    # damage was measured against the SAMPLER's own seed spread rather than
    # against zero - see docs/EF2FAST.md.
    "esmc": {
        "export": "model-esmc-600m-int3",
        "module": "src/bundles/manifests/esmc.js",
        "model": "esmc",
    },
    # The same folding model against ESM-C 300M - a SEPARATE checkpoint, since
    # its shim is trained for 30 layers x 960 against the 600M's 36 x 1152.
    "ef2-fast-300m": {
        "export": "model-ef2-fast-300m-int5",
        "module": "src/bundles/manifests/ef2-fast-300m.js",
        "model": "esmfold2-trunk",
    },
    "esmc-300m": {
        "export": "model-esmc-300m-int3",
        "module": "src/bundles/manifests/esmc-300m.js",
        "model": "esmc",
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
