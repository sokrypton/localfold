#!/usr/bin/env python3
"""Write a LocalFold float32 monomer directory from AF2 params_model_N_ptm.npz.

    python3 tools/export_monomer_model.py ~/params/ptm/params_model_2_ptm.npz \
        --out model-monomer-2.f32
    python3 tools/quantize_model.py --source model-monomer-2.f32 --out model-monomer-2

🔴 ALPHAFOLD 2 IS FIVE MODELS AND THIS REPOSITORY COULD ONLY BUILD ONE. The
multimer had `export_multimer_model.py` and could be built from any of its five
checkpoints in a second; the monomer's `model/` came out of
`capture_alphafold_single_sequence.py`, which runs the official model in a JAX
environment and intercepts tensors, through `export-web-model.js` over fixture
files that are not in the repository. So "fold with model_3_ptm" needed a
machine with jax, haiku and the alphafold package, and nobody had one. This
needs numpy and the npz.

🔴 THE NAMES COME FROM THE SHIPPED MANIFEST, NOT FROM A COUNTER. The multimer
exporter numbers its tensors as it walks the parameter tree, which is correct
and makes a rebuilt model_1 only ACCIDENTALLY comparable with the shipped one -
any reordering in the walk renames every tensor. Here each tensor is written
under the name `src/bundles/manifests/monomer.js` already gives it, so a rebuilt
model_1_ptm is name-identical to the bundle that ships and the two can be folded
against each other. That is the gate this exporter is checked by: rebuilt
model_1, quantised the way the shipped bundle is, must fold to the SAME
checksum as `model/`.

🔴 AND THAT CHECKSUM MOVED WHEN THE BASE DID. It was **-1287025** while
`model/` was int8 symmetric block 64 (tools/quantize_model.py) and is
**-1309830** now that it is int5 asymmetric group 32 (tools/quantize_af3.py,
73 MiB against 98, measured free on both a single-sequence fold and 5CAJ with
an alignment). A figure recorded here before 2026-09-18 is the int8 base's.

🔴 AND COVERAGE IS ASSERTED IN BOTH DIRECTIONS. A weight the reference names and
the npz does not have is a missing tensor; a name written twice is a walk that
visited a module twice. Either produces a bundle that loads, folds, and is
wrong - the failure mode this repository keeps meeting - so both are refused
here rather than reported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_multimer_params import load_params  # noqa: E402
from export_multimer_model import ShardWriter  # noqa: E402

A = "alphafold/alphafold_iteration/"
SECTION_SCOPES = {
    "evoformerStack": A + "evoformer/evoformer_iteration/",
    "extraMsaStack": A + "evoformer/extra_msa_stack/",
    "embedding": A + "evoformer/",
    "templateEmbedding": A + "evoformer/template_embedding/",
    "structureModule": A + "structure_module/",
}
CONFIDENCE_SCOPES = {
    "predictedLddt": A + "predicted_lddt_head/",
    "predictedAlignedError": A + "predicted_aligned_error_head/",
}
DISTOGRAM = A + "distogram_head/half_logits"


def reference_manifest() -> dict:
    """The shipped monomer tensor table, read from its generated module."""
    source = (Path(__file__).resolve().parent.parent
              / "src" / "bundles" / "manifests" / "monomer.js").read_text()
    marker = source.index("MANIFEST")
    return json.loads(source[source.index("=", marker) + 1:].rsplit(";", 1)[0].strip())


def export(params_path: Path, borrow_dir: Path, out_dir: Path) -> int:
    params = load_params(params_path)
    reference = reference_manifest()
    model = params_path.stem.replace("params_", "")

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(out_dir)
    manifest: dict = {
        "formatVersion": 1,
        "source": f"AlphaFold monomer parameters from {params_path.name}",
        "model": {"name": model.replace("_ptm", ""), "recycles": 3},
        "bundle": {"purpose": "browser-inference", "model": model,
                   "encoding": "float32-le"},
    }
    missing: list[str] = []
    written: set[str] = set()
    structural: set[str] = set()

    def put(name: str, values: np.ndarray) -> str:
        if name in written:
            raise SystemExit(f"{name} written twice - the walk visited a module twice")
        written.add(name)
        return writer.add(name, values)

    for section, scope in SECTION_SCOPES.items():
        table = reference[section]["parameters"]
        # 🔴 ALL OF A SECTION ABSENT IS A MODEL, SOME OF IT ABSENT IS A BUG.
        # model_3, model_4 and model_5 carry no template embedder at all - 67
        # tensors, none of them - and that is what those checkpoints ARE, not a
        # failed export. So a section with nothing in the npz is omitted from
        # the manifest and reported as a fact; a section with SOME of its
        # modules missing still fails, because that is a name this exporter got
        # wrong. Writing an empty `parameters: {}` instead would be the worst of
        # both: `templateWeights` reads it as a table and dies in a gather.
        if all(params.get(scope + module) is None for module in table):
            print(f"  no {section} in this checkpoint - omitted "
                  f"({len(set().union(*(set(v.values()) for v in table.values())))} tensors)")
            continue
        parameters: dict = {}
        for module, leaves in table.items():
            source = params.get(scope + module)
            if source is None:
                missing.append(f"{section}/{module}")
                continue
            entry = {}
            for leaf, name in leaves.items():
                if leaf not in source:
                    missing.append(f"{section}/{module}//{leaf}")
                    continue
                entry[leaf] = put(name, source[leaf])
                if section == "structureModule":
                    structural.add(name)
            parameters[module] = entry
        manifest[section] = {key: value for key, value in reference[section].items()
                             if key != "parameters"}
        manifest[section]["parameters"] = parameters

    confidence: dict = {}
    for head, modules in reference["confidenceHeads"]["parameters"].items():
        entry = {}
        for module, leaves in modules.items():
            source = params.get(CONFIDENCE_SCOPES[head] + module)
            if source is None:
                missing.append(f"confidenceHeads/{head}/{module}")
                continue
            entry[module] = {leaf: put(name, source[leaf]) for leaf, name in leaves.items()}
        confidence[head] = entry
    manifest["confidenceHeads"] = {"parameters": confidence}

    # 🔴 THE DISTOGRAM HEAD IS NOT UNDER A SECTION - it is two tensors named
    # straight in the manifest root, added by tools/add_distogram_head.py after
    # the fact, and a walk over the sections above misses both. The page's
    # contact overlay is what reads them.
    head = reference.get("distogramHead")
    if head is not None:
        source = params.get(DISTOGRAM)
        if source is None:
            missing.append("distogramHead")
        else:
            for leaf in ("weights", "bias"):
                put(head[leaf], source[leaf])
        manifest["distogramHead"] = head

    # The chemistry tables and the PAE bin edges: residue_constants, not learned
    # parameters, and identical in every model. Borrowed from an existing export
    # exactly as export_multimer_model.py borrows them.
    borrowed = list(reference["residueGeometry"]["tensors"])
    if "confidencePaeBreaks" in reference["tensors"]:
        borrowed.append("confidencePaeBreaks")
    table = json.loads((borrow_dir / "manifest.json").read_text())["tensors"]
    for name in borrowed:
        record = table[name]
        if record["dtype"] != "float32":
            raise SystemExit(f"{name} is {record['dtype']} in {borrow_dir}; this reads float32")
        shard = (borrow_dir / record["file"]).read_bytes()
        values = np.frombuffer(shard, dtype="<f4", count=int(np.prod(record["shape"])),
                               offset=record["byteOffset"]).reshape(record["shape"])
        put(name, values)
    manifest["residueGeometry"] = reference["residueGeometry"]

    # 🔴 THE KEEP-LIST IS DECLARED, SO EITHER QUANTISER CAN READ THIS EXPORT.
    # tools/quantize_model.py DERIVES it from the sections (structure module,
    # geometry tables, PAE bin edges); tools/quantize_af3.py - the asymmetric
    # one, which is what a sub-byte codec needs - only honours an explicit
    # `float32Tensors`. Writing it here means an AF2 export can be packed either
    # way rather than only the way it was written for, and the two agree about
    # which tensors must not be rounded.
    keep = set(structural)
    for name in manifest["residueGeometry"]["tensors"]:
        keep.add(name)
    if "confidencePaeBreaks" in written:
        keep.add("confidencePaeBreaks")
    # 🔴 AND THE DISTOGRAM HEAD, WHICH IS 33 KB AND IS THE CONTACT MAP. It is
    # 128x64 plus a bias - not worth a codec at any bit width, and the one head
    # whose output the page DRAWS rather than reports. test/manifest.test.js
    # pins it, and caught it missing from this list the first time an AF2
    # export was packed by the asymmetric quantiser.
    if head is not None:
        keep.update(head[leaf] for leaf in ("weights", "bias"))
    manifest["float32Tensors"] = sorted(keep & written)
    writer.close()
    manifest["tensors"] = writer.records
    (out_dir / "manifest.json").write_text(json.dumps(manifest))

    total = sum(int(np.prod(r["shape"])) for r in writer.records.values())
    print(f"{len(writer.records)} tensors, {total:,} elements, {writer.index} shards")
    print(f"wrote {out_dir}/manifest.json")
    # ...and a section this checkpoint does not have owes no tensors.
    absent = set()
    for section in SECTION_SCOPES:
        if section in manifest:
            continue
        for leaves in reference[section]["parameters"].values():
            absent.update(leaves.values())
    unwritten = set(reference["tensors"]) - written - absent
    if unwritten:
        print(f"\n🔴 {len(unwritten)} tensors the shipped manifest names and this did not write:",
              file=sys.stderr)
        for name in sorted(unwritten)[:20]:
            print(f"    {name}", file=sys.stderr)
        return 1
    if missing:
        print(f"\n🔴 {len(missing)} parameters had no source:", file=sys.stderr)
        for name in missing[:20]:
            print(f"    {name}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("params", type=Path, help="params_model_N_ptm.npz")
    parser.add_argument("--borrow", type=Path, default=Path("model"),
                        help="existing export, for the residue-geometry tables")
    parser.add_argument("--out", type=Path, default=Path("model-monomer.f32"))
    args = parser.parse_args()
    return export(args.params, args.borrow, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
