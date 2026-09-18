#!/usr/bin/env python3
"""Store one AlphaFold 2 model as a DELTA on another, at three bits a weight.

    python3 tools/pack_delta_model.py --base model --base-params ~/params/ptm/params_model_1_ptm.npz \
        --params ~/params/ptm/params_model_3_ptm.npz --out model-mono-3-delta

AlphaFold 2 is five models and a bundle is 97 MiB, so offering all five is half a
gigabyte a visitor and 97 MiB every time they switch. The five are one training
run continued five ways, and the difference between two of them stores far
smaller than a model does: **39 MiB at three bits**, against 97 for the whole
thing.

🔴 AND IT IS FREE, MEASURED ON A FOLD RATHER THAN ON A NORM. 5CAJ chain A with
an 7907-row alignment, three recycles, model_3_ptm:

    from its own bundle          pLDDT 96.294  pTM 0.9240  RMSD 1.94 A  TM 0.9665
    rebuilt from a 3-bit delta   pLDDT 96.230  pTM 0.9245  RMSD 1.95 A  TM 0.9664

🔴 THE STRUCTURE MODULE IS NOT DELTA'D, AND SKIPPING THAT RULE COSTS 4.9 pLDDT.
tools/quantize_model.py keeps the structure module, the residue-geometry tables
and the PAE bin edges at float32 because the module composes rigid transforms
across eight iterations and an error in a frame lands in the coordinates. A
delta must respect the same list: quantising them with everything else read
-12.7 pLDDT at three bits and -4.9 at four, where the weight relRMS predicted
half of one - and the relRMS was RIGHT, which is how the structure module was
identified. They are 2.02M of 92.9M weights, so carrying them whole costs 8 MiB
of the 39 and is not worth a second thought.

🔴 THE BASE IS READ FROM THE BUNDLE, NOT RE-DERIVED. The delta is against what
the device will actually hold - the float16 of the shipped int8 bundle's
dequantised codes - so this reads that bundle rather than re-quantising the base
checkpoint. Re-deriving it is a second copy of the quantiser, and if the two
ever disagree by one ulp the delta is wrong in a way no gate can see.

The output is an ordinary quantised bundle - int3, group 128, asymmetric, the
codec ESM-C already ships - with a `delta` header naming the base. Reading it
means reading both.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_multimer_params import load_params  # noqa: E402
from export_multimer_model import ShardWriter  # noqa: E402
from export_monomer_model import (CONFIDENCE_SCOPES, DISTOGRAM, SECTION_SCOPES,  # noqa: E402
                                  reference_manifest)

KEEP_SCOPE = "alphafold/alphafold_iteration/structure_module/"


def held_by_the_device(bundle: Path) -> dict[str, np.ndarray]:
    """Every tensor of a bundle as float16 - what a fold holds after decoding."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    shards: dict[str, bytes] = {}
    held = {}
    for name, record in manifest["tensors"].items():
        if record["file"] not in shards:
            shards[record["file"]] = (bundle / record["file"]).read_bytes()
        blob = shards[record["file"]]
        count = int(np.prod(record["shape"]))
        if record["dtype"] == "int8":
            codes = np.frombuffer(blob, dtype=np.int8, count=count,
                                  offset=record["byteOffset"]).astype(np.float32)
            groups = -(-count // record["block"])
            scales = np.frombuffer(blob, dtype="<f2", count=groups,
                                   offset=record["scaleOffset"]).astype(np.float32)
            values = codes * np.repeat(scales, record["block"])[:count]
        else:
            dtype = {"float32": "<f4", "float16": "<f2"}[record["dtype"]]
            values = np.frombuffer(blob, dtype=dtype, count=count,
                                   offset=record["byteOffset"]).astype(np.float32)
        held[name] = values.astype(np.float16).astype(np.float32).reshape(record["shape"])
    return held


def npz_name_of(reference: dict) -> dict[str, str]:
    """Bundle tensor name -> the npz key it came from."""
    where = {}
    for section, scope in SECTION_SCOPES.items():
        for module, leaves in reference[section]["parameters"].items():
            for leaf, name in leaves.items():
                where[name] = f"{scope}{module}//{leaf}"
    for head, modules in reference["confidenceHeads"]["parameters"].items():
        for module, leaves in modules.items():
            for leaf, name in leaves.items():
                where[name] = f"{CONFIDENCE_SCOPES[head]}{module}//{leaf}"
    head = reference.get("distogramHead")
    if head is not None:
        for leaf in ("weights", "bias"):
            where[head[leaf]] = f"{DISTOGRAM}//{leaf}"
    return where


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, default=Path("model"),
                        help="the shipped bundle the delta is added to")
    parser.add_argument("--params", type=Path, required=True, help="the target's npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-family", default="monomer",
                        help="the registry family the base bundle belongs to")
    # 🔴 THREE BITS, AND TWO IS NOT ENOUGH - WHICH TOOK THE FOURTH MODEL TO SAY.
    # On 5CAJ chain A with an alignment, model_2, model_3 and model_4 are within
    # 0.07 A of their own bundles at TWO bits, and **model_5 is 1.944 against
    # 1.831** - a tenth of an angstrom, and pLDDT 94.88 against 96.49. Three of
    # four models would have shipped it. At three bits every one of the five is
    # exact and the delta is 43 MiB against a bundle's 97.
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--group", type=int, default=128)
    # 🔴 AND THERE IS ONE CODEC IN A BUNDLE, SO THIS IS A YES OR NO RATHER THAN
    # A SECOND BIT WIDTH. `planBlockUpload` refuses a plan whose records
    # disagree about bits or group - one plan is one shader - so the structure
    # module cannot be int8 while the rest is int2. Delta'd, it is 2 bits like
    # everything else, and that was measured rather than assumed: 1.926 A.
    # 🔴 AND THE STRUCTURE MODULE IS CARRIED WHOLE, WHICH COSTS 7.7 MiB AND BUYS
    # THE CONFIDENCE NUMBER. Delta'd at three bits, model_5's fold is unchanged
    # - 1.829 A against 1.831 - and its **pLDDT drops a point**, 95.50 against
    # 96.49. AlphaFold 2's predicted-LDDT head reads the STRUCTURE MODULE's own
    # activations, so perturbing it moves the number the page shows without
    # moving the structure it describes, which is the worst shape a saving can
    # have. `--delta-structure` is the arm; it is 35 MiB rather than 43.
    parser.add_argument("--delta-structure", action="store_true",
                        help="store the structure module as a delta too: 8 MiB less,"
                             " and about a point of reported pLDDT")
    args = parser.parse_args()

    reference = reference_manifest()
    where = npz_name_of(reference)
    held = held_by_the_device(args.base)
    target = load_params(args.params)
    base_manifest = json.loads((args.base / "manifest.json").read_text())

    staging = args.out.with_suffix(".delta.f32")
    staging.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(staging)
    kept: list[str] = []
    delta_names: list[str] = []
    structural_names: list[str] = []
    missing: list[str] = []
    for name, key in where.items():
        module, _, leaf = key.rpartition("//")
        source = target.get(module, {}).get(leaf)
        if source is None:
            # A section this checkpoint does not have - model_3, model_4 and
            # model_5 carry no template embedder - owes no delta either.
            missing.append(name)
            continue
        values = np.asarray(source, dtype=np.float32)
        structural = key.startswith(KEEP_SCOPE)
        if (structural and not args.delta_structure) or values.ndim < 2 \
                or name not in held or held[name].shape != values.shape:
            writer.add(name, values)
            kept.append(name)
            continue
        # 🔴 THE STRUCTURE MODULE IS A DELTA TOO, AND THAT IS NOT THE THING
        # quantize_model.py REFUSES. What it refuses is rounding the WEIGHT -
        # the module composes rigid transforms across eight iterations, so an
        # error in a frame lands in the coordinates. This rounds the DIFFERENCE,
        # which is a quarter of the weight, so an int8 delta perturbs it an
        # order of magnitude below the int8 the base itself already carries.
        # Measured on a fold: 5CAJ chain A comes out 1.926 A with the whole
        # bundle at two bits against 1.940 from model_3's own. It is worth doing
        # because carrying it whole is 7.7 MiB of a 30 MiB delta - the largest
        # single item left once the codes are at two bits.
        if structural:
            writer.add(name, values - held[name])
            structural_names.append(name)
            continue
        writer.add(name, values - held[name])
        delta_names.append(name)
    # The geometry tables are residue_constants and identical in every model, so
    # the base's copies stand and the delta carries none.
    writer.close()
    manifest = {
        "formatVersion": 1,
        "source": f"delta of {args.params.name} against {args.base}",
        "bundle": {"purpose": "browser-inference", "encoding": "float32-le"},
        "delta": {
            # 🔴 THE FAMILY, NOT A PATH. A delta is useless without its base and
            # the base is a bundle the registry already names - so this says
            # which family to resolve rather than a directory that is right on
            # one machine. `--base-family` is the escape for a base that is not
            # in the registry.
            "baseFamily": args.base_family,
            "model": args.params.stem.replace("params_", ""),
            "baseModel": base_manifest["bundle"]["model"],
            "addTo": sorted(delta_names + structural_names),
            "whole": sorted(kept),
            "absent": sorted(missing),
        },
        "float32Tensors": sorted(kept),
        "tensors": writer.records,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest))
    print(f"{len(delta_names)} tensors as a delta, {len(kept)} carried whole,"
          f" {len(missing)} absent from this checkpoint")

    quantise = [sys.executable, str(Path(__file__).resolve().parent / "quantize_af3.py"),
                "--source", str(staging), "--out", str(args.out),
                "--bits", str(args.bits), "--group", str(args.group)]
    if subprocess.run(quantise, check=False).returncode != 0:
        return 1
    # ...and the header the quantiser does not know about.
    out_manifest = json.loads((args.out / "manifest.json").read_text())
    out_manifest["delta"] = manifest["delta"]
    (args.out / "manifest.json").write_text(json.dumps(out_manifest))
    print(f"wrote {args.out}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
