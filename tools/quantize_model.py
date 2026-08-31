"""Rewrite an exported model/ directory with quantised weight shards.

    python3 tools/quantize_model.py --source model.f32-backup            # int8, the default
    python3 tools/quantize_model.py --source model.f32-backup --format f16
    python3 tools/quantize_model.py --source model.f32-backup --out model16

WHY THIS EXISTS.

The browser page used to round every weight at runtime, on every fold, into a
freshly allocated 371 MiB tree. That cost about six seconds a fold in garbage
collection and bought nothing, because the rounded values went straight back
into a Float32Array - four bytes a weight before and after. Precision is a
STORAGE decision, so it belongs in storage.

WHAT THE FORMATS COST, measured end to end: a four-pass fold of the 59-residue
reference sequence, against 63.4 pLDDT / 0.421 pTM at float16.

    float32              355.3 MiB   1.0x     -
    float16              181.5 MiB   2.0x     63.4    baseline
    int8  block 64        97.3 MiB   3.7x     63.3    -0.1
    int6  block 32        78.3 MiB   4.5x     62.9    -0.5
    int5  block 32        67.5 MiB   5.3x     55.6    -7.8
    int4  block 32        56.6 MiB   6.3x     52.8   -10.6

int8 is the point this ships at: byte-aligned, so there is no bit-packing, and
the loss is inside the run-to-run noise. Below six bits the model falls off a
cliff, and it is not a gentle one.

🔴 SYMMETRIC, AND ONLY BECAUSE EIGHT BITS IS ENOUGH. A block's weights are not
centred on zero - the midpoint sits a median 17% of the half-range away, 42% at
the ninetieth percentile - so a symmetric fit wastes range AND leaves a
systematic bias in every block. Bias is the part that matters: zero-mean
rounding noise averages out across 48 Evoformer blocks, a mean shift compounds.
At five bits that bias is worth 6.1 pLDDT (55.6 symmetric against 61.7 with a
zero point). At eight bits the step is small enough that it measures 0.1, which
is noise, and a zero point would cost two more bytes per block for nothing.
If this ever drops below six bits, add one.

🔴 THE STRUCTURE MODULE AND THE GEOMETRY TABLES STAY FLOAT32.

The structure module composes rigid transforms across eight iterations, and an
error in a frame is carried into the next one and lands in the coordinates.
AlphaFold's own capture records it as float32 (`structureModule.dtype`), and the
geometry tables are not learned weights at all - they are the residue-constants
literals, where rounding an ideal atom position moves an atom by construction.
They are 2.2% of the parameters, so keeping them is nearly free.

WHAT IS PRESERVED: which tensor lives in which shard, and the shard count. Only
the encoding and the offsets change, so the eight-way split survives.
"""
import argparse
import json
import shutil
import sys
from math import prod
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BLOCK = 64


def collect_names(node, known, into):
    """Every tensor id reachable under a manifest section, however nested."""
    if isinstance(node, str):
        if node in known:
            into.add(node)
    elif isinstance(node, dict):
        for value in node.values():
            collect_names(value, known, into)
    elif isinstance(node, list):
        for value in node:
            collect_names(value, known, into)


def float32_names(manifest):
    """The tensors that must not be quantised. See the note at the top."""
    known = set(manifest["tensors"])
    keep = set()
    for section in ("structureModule", "residueGeometry"):
        collect_names(manifest.get(section, {}), known, keep)
    # ...and the PAE bin edges, which are not weights either: they are the
    # boundaries the confidence head's expectation is taken against, and nudging
    # one moves every reported error by a fraction of a bin.
    for name in manifest["tensors"]:
        if name.startswith("geometry") or name == "confidencePaeBreaks":
            keep.add(name)
    # ...and whatever the export itself named. The rules above are the AF2
    # graph's, written when there was one graph; AF3 has no structure module and
    # no residue-geometry tables, and the tensors it must not quantise (every
    # LayerNorm scale and offset) are only identifiable to the exporter that
    # laid them out. An export that declares nothing is unaffected.
    for name in manifest.get("float32Tensors", []):
        if name not in known:
            print(f"float32Tensors names {name!r}, which is not in this export",
                  file=sys.stderr)
            raise SystemExit(1)
        keep.add(name)
    return keep


def quantise_int8(values, block):
    """Symmetric per-block int8, with the scale kept as float16.

    Returns the codes and the scales. The scale is rounded to float16 BEFORE the
    codes are computed, so that the encoder divides by exactly the number the
    decoder will multiply by - computing codes against an f32 scale and then
    storing an f16 one puts a second rounding between them that nothing corrects.
    """
    padded = -values.size % block
    grid = np.concatenate([values, np.zeros(padded, np.float32)]).reshape(-1, block)
    amax = np.abs(grid).max(axis=1, keepdims=True)
    scale = (amax / 127.0).astype("<f2").astype(np.float32)
    safe = np.where(scale == 0, 1.0, scale)
    codes = np.clip(np.rint(grid / safe), -127, 127).astype(np.int8)
    return codes.reshape(-1)[:values.size], scale.reshape(-1).astype("<f2")


def span(entry):
    """How many bytes a written tensor occupies from its byteOffset."""
    elements = prod(entry["shape"])
    if entry["dtype"] == "int8":
        return (entry["scaleOffset"] - entry["byteOffset"]) + -(-elements // entry["block"]) * 2
    return elements * {"float32": 4, "float16": 2}[entry["dtype"]]


def convert(source_dir, out_dir, fmt):
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())
    tensors = manifest["tensors"]
    if any(record["dtype"] != "float32" for record in tensors.values()):
        print("source must be a float32 export; point --source at the originals", file=sys.stderr)
        return 1
    keep_f32 = float32_names(manifest)

    by_shard = {}
    for name, record in tensors.items():
        by_shard.setdefault(record["file"], []).append((record["byteOffset"], name))
    for entries in by_shard.values():
        entries.sort()

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"int8": 0, "float16": 0, "float32": 0}
    written = {}

    for shard, entries in sorted(by_shard.items()):
        source = np.fromfile(source_dir / shard, dtype="<f4")
        target = shard.replace(".f32.bin", ".bin")
        chunks = []
        offset = 0
        content_end = 0

        def emit(raw, align=4):
            # 🔴 PADDING BETWEEN TENSORS, NEVER AFTER THE LAST ONE. A reader
            # works out how long a shard should be as the furthest tensor end in
            # it, so trailing padding makes the file longer than anything
            # accounts for - and the streaming reader treats a shard that
            # overruns its expected length as a corrupt download. `content_end`
            # is where the last real byte went; the file is cut there.
            nonlocal offset, content_end
            chunks.append(raw)
            offset += len(raw)
            content_end = offset
            padding = -offset % align
            if padding:
                chunks.append(b"\0" * padding)
                offset += padding

        for byte_offset, name in entries:
            record = tensors[name]
            elements = prod(record["shape"])
            values = source[byte_offset // 4: byte_offset // 4 + elements]
            if values.size != elements:
                print(f"{name}: shard {shard} has {values.size} of {elements}", file=sys.stderr)
                return 1
            entry = {"file": target, "shape": record["shape"], "byteOffset": offset}
            if name in keep_f32:
                entry["dtype"] = "float32"
                emit(values.astype("<f4", copy=False).tobytes())
            elif fmt == "f16":
                entry["dtype"] = "float16"
                emit(values.astype("<f2").tobytes())
            else:
                codes, scales = quantise_int8(values, BLOCK)
                entry["dtype"] = "int8"
                entry["block"] = BLOCK
                # ...CODES FIRST, THEN SCALES, and the scale offset is recorded
                # rather than derived. Codes are one byte each and scales are
                # two, so where the scales begin depends on a padding rule; a
                # reader that recomputed it would be repeating this file's
                # arithmetic and would break the day the rule changed.
                emit(codes.tobytes(), align=2)
                entry["scaleOffset"] = offset
                emit(scales.tobytes())
            counts[entry["dtype"]] += 1
            written[name] = entry
        payload = b"".join(chunks)[:content_end]
        # ...and the invariant that made the cut necessary, checked rather than
        # trusted: every tensor must end at or before the file does, and the
        # last one must end exactly at it.
        ends = [written[n]["byteOffset"] + span(written[n]) for _, n in entries]
        if max(ends) != len(payload):
            print(f"{target}: last tensor ends at {max(ends)} but the shard is"
                  f" {len(payload)} bytes", file=sys.stderr)
            return 1
        (out_dir / target).write_bytes(payload)

    manifest["tensors"] = written
    bundle = manifest.setdefault("bundle", {})
    bundle["encoding"] = "mixed-le"
    bundle["encodingNote"] = (
        f"{'int8 with float16 per-block scales' if fmt == 'int8' else 'float16'}-le,"
        " except the structure module, the residue geometry tables and the PAE"
        " bin edges, which stay float32-le"
    )
    if fmt == "int8":
        bundle["quantisation"] = {"scheme": "symmetric-per-block", "bits": 8, "block": BLOCK}
    bundle["bytes"] = sum(path.stat().st_size for path in out_dir.glob("*.bin"))
    bundle["shards"] = len(by_shard)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"{out_dir.name}/  " + ", ".join(f"{n} {k}" for k, n in counts.items() if n)
          + f"   {bundle['bytes'] / 1048576:.1f} MiB across {len(by_shard)} shards")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=str(ROOT / "model"))
    parser.add_argument("--format", choices=("int8", "f16"), default="int8")
    parser.add_argument("--out", default=None,
                        help="destination; defaults to a staging directory beside --source")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    if args.out is None:
        staging = source.parent / f"{source.name}.{args.format}-staging"
        if staging.exists():
            shutil.rmtree(staging)
        code = convert(source, staging, args.format)
        if code != 0:
            raise SystemExit(code)
        print(f"written to {staging} - verify, then replace model/ with it")
        raise SystemExit(0)
    raise SystemExit(convert(source, Path(args.out).resolve(), args.format))
