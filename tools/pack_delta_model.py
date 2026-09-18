#!/usr/bin/env python3
"""Store one AlphaFold 2 model as a DELTA on another, at three bits a weight.

    python3 tools/pack_delta_model.py --base model --base-params ~/params/ptm/params_model_1_ptm.npz \
        --params ~/params/ptm/params_model_3_ptm.npz --out model-mono-3-delta

AlphaFold 2 is five models and a bundle is 97 MiB, so offering all five is half a
gigabyte a visitor and 97 MiB every time they switch. The five are one training
run continued five ways, and the difference between two of them stores far
smaller than a model does: **39 MiB at three bits**, against 97 for the whole
thing.

🔴 THREE BITS, AND TWO IS NOT ENOUGH - WHICH A WELL-DETERMINED FOLD SAYS IS FINE
AND A SINGLE-SEQUENCE ONE REFUSES. Two bits was measured, chosen, shipped and
taken back inside an hour, and the reason is the one this repository keeps
meeting: an alignment PINS the answer, so a coarser delta hardly moves it, and
without one the weights are all there is. The 59-residue gate sequence, no
alignment, pLDDT against each model's own bundle:

    model    its own bundle    3-bit delta    2-bit delta
    model_2      62.435          63.549         61.959
    model_3      58.468          58.539         51.248
    model_4      61.866          58.818         37.923   <- twenty-four points
    model_5      63.941          64.705         57.630

On 5CAJ chain A with a 7907-row alignment the SAME two-bit bundles are within a
tenth of an angstrom (RMSD / pLDDT):

    model    its own bundle    via a 2-bit delta    via a 3-bit delta
    model_2  1.891 / 96.182    1.895 / 95.964       1.897 / 96.133
    model_3  1.940 / 96.294    1.950 / 95.819       1.950 / 96.230
    model_4  1.983 / 96.418    2.049 / 96.287       1.981 / 96.437
    model_5  1.831 / 96.485    1.944 / 94.880       1.825 / 96.327

Two bits is 24 MiB and three is 43. Even on the determined fold the difference
is a systematic shift rather than scatter - four seeds through both arms give
**+0.094 +/- 0.020 A with 4 of 4 moving the same way** against a seed band of
0.163 - and the single-sequence table above is what makes it a refusal rather
than a trade. `--bits 2` is still there for anyone who wants the bytes and folds
with an alignment every time.

🔴 SO THE REPORTED pLDDT READS LOW BY A DIFFERENT AMOUNT PER MODEL, AND THAT IS
THE THING TO KNOW WHEN RANKING THE FIVE: model_2 -0.22, model_3 -0.47, model_4
-0.13, model_5 **-1.61**. On this target that is enough to move model_5 from the
most confident of the four to the least. `--bits 3` is the setting where every
bias is inside the seed band (-0.06, -0.05, +0.02, -0.16), at 43 MiB.

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
import re
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
        packed = re.fullmatch(r"int([1-9])", record["dtype"])
        if record["dtype"] == "int8":
            codes = np.frombuffer(blob, dtype=np.int8, count=count,
                                  offset=record["byteOffset"]).astype(np.float32)
            groups = -(-count // record["block"])
            scales = np.frombuffer(blob, dtype="<f2", count=groups,
                                   offset=record["scaleOffset"]).astype(np.float32)
            values = codes * np.repeat(scales, record["block"])[:count]
        elif packed is not None:
            # 🔴 THE BASE NEED NOT BE int8, AND SAYING SO IS WHY THIS READS THE
            # BUNDLE RATHER THAN RE-QUANTISING THE CHECKPOINT. AlphaFold 2's
            # monomer ships int8 symmetric today and int5 ASYMMETRIC is 73 MiB
            # against 98 at the same fold, so a delta has to be able to sit on
            # either - and it is only correct if it is subtracted from exactly
            # what the device will hold.
            bits, group = int(packed.group(1)), record["block"]
            raw = np.frombuffer(blob, dtype=np.uint8, offset=record["byteOffset"],
                                count=(count * bits + 7) // 8)
            spread = np.unpackbits(raw, bitorder="little")[:count * bits].reshape(count, bits)
            codes = (spread * (1 << np.arange(bits))).sum(1).astype(np.float32)
            groups = -(-count // group)
            scales = np.frombuffer(blob, dtype="<f2", count=groups,
                                   offset=record["scaleOffset"]).astype(np.float32)
            zeros = np.frombuffer(blob, dtype="<f2", count=groups,
                                  offset=record["zeroOffset"]).astype(np.float32)
            values = (codes * np.repeat(scales, group)[:count]
                      + np.repeat(zeros, group)[:count])
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
    # 🔴 THREE BITS. TWO WAS TRIED, MEASURED ON A WELL-DETERMINED FOLD, SHIPPED,
    # AND TAKEN BACK - see the header. On 5CAJ with a 7907-row alignment two
    # bits costs a tenth of an angstrom; on the SAME models folding a 59-residue
    # sequence with NO alignment it costs model_4 twenty-four points of pLDDT.
    # An alignment pins the answer and the weights are all there is without one,
    # which is where a coarser delta lands.
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
    # 🔴 EIGHT, WHICH IS WHERE THE LINK STOPS PAYING - MEASURED, NOT INHERITED.
    # A delta is a SWITCH, so its download is what a reader waits on between two
    # models, and a shard is one HTTP stream. Timed against the twelve-shard
    # bundle already on Hugging Face, three interleaved passes, median MB/s by
    # concurrency: 1 -> 27.6, 2 -> 38.2, 4 -> 50.1, 6 -> 86.7, 8 -> 91.4,
    # 12 -> 77.5 with a spread of 62. It climbs steeply to six and is flat after,
    # so a 43 MiB delta is 1.2 s in two shards and 0.5 s in eight. The first
    # version of this comment priced two shards at a tenth of a second from
    # docs/HOSTING.md's 56 and 78 MB/s; this link gives 27.6 on one stream, so
    # the parallelism is worth more here than that note implies.
    parser.add_argument("--shards", type=int, default=8)
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
        # 🔴 AND WHAT IT COSTS IS THE CONFIDENCE NUMBER RATHER THAN THE
        # STRUCTURE, which is the worst shape a saving can have: AF2's
        # predicted-LDDT head reads the structure module's own activations, so
        # model_5 delta'd here folds to 1.829 A against its bundle's 1.831 -
        # exact - and reports 95.50 against 96.49. It is 7.7 MiB of the 24, and
        # `--whole-structure` keeps it for anyone who would rather have the
        # number than the bytes.
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
                "--bits", str(args.bits), "--group", str(args.group),
                "--shards", str(args.shards)]
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
