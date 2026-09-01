"""Rewrite an exported AF3 model directory as int5, asymmetric, group 32.

    python3 tools/quantize_af3.py --source model-af3-full-f32 --out model-af3-int5

WHY int5 AND NOT int4. A scheme costs its codes PLUS its metadata, and at the
group sizes low precision needs, the metadata is most of the difference. int4
with a scale and a zero point every 16 weights is 6 bits a weight - the same
budget as this, which is 1.6x more accurate. See tools/analyse_quantisation.py.

WHAT IT COSTS, measured by folding 6MRR and comparing to the crystal structure
rather than to pLDDT (tools/score_fold.py):

    scheme                 bits/w     MiB    x   RMSD     TM  pLDDT  CA-CA
    float32                 32.00  1405.3  1.0   0.69  0.950   81.6   3.75
    int8  g64 sym            8.25   362.3  3.9   0.70  0.949   81.7   3.76
    int6  g64 asym           6.50   285.5  4.9   0.72  0.947   81.9   3.73
    int5  g32 asym           6.00   263.5  5.3   0.66  0.953   81.2   3.74  <-
    int4  g32 asym           5.00   219.6  6.4   1.21  0.896   76.2   3.48
    int4  g32 asym+search    5.00   219.6  6.4   0.76  0.942   82.0   3.66

0.66 against float32's 0.69 is the spread between diffusion seeds, so int5 is
free. int4 is where the model notices, and it needs a per-group range search to
be worth having at all.

🔴 ASYMMETRIC, WITH A ZERO POINT. A group of weights is not centred on zero, so
a symmetric fit spends half its codes on a range that holds nothing AND leaves a
systematic bias - and bias is the part that matters, because zero-mean rounding
noise averages out across 48 blocks while a mean shift compounds. At eight bits
the step is small enough not to care; at five it is worth 1.28x.

🔴 32 CODES OF 5 BITS IS EXACTLY 160 BITS, so a group occupies exactly 20 bytes
and no group ever straddles another. That is the whole reason group 32 is the
convenient size here rather than a tuned one - at group 24 or 48 the packing
needs a case for values split across the boundary, and that case is where a
packer goes wrong silently.

🔴 NORMS, OFFSETS AND BIASES STAY FLOAT32. They are 0.09% of the parameters and
the worst thing to group-quantise: a 128-wide LayerNorm scale is four groups, so
four scales carry the whole tensor, and that tensor's job is to set the scale of
everything after it.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GROUP = 32
BITS = 5
KEEP_FLOAT32 = re.compile(r"/(scale|offset|bias)$|_bias$|_weight$|/output_b$")


def quantise(values, group=GROUP, bits=BITS):
    """Asymmetric per-group codes, plus a float16 scale and zero point each.

    Returns (codes uint8 in [0, 2**bits), scales float16, zeros float16).
    """
    levels = 2 ** bits - 1
    padded = (-len(values)) % group
    grid = np.concatenate([values, np.zeros(padded, np.float32)]).reshape(-1, group)

    low = grid.min(1, keepdims=True).astype(np.float32)
    high = grid.max(1, keepdims=True).astype(np.float32)
    # 🔴 THE SCALE AND ZERO ARE ROUNDED TO float16 BEFORE THE CODES ARE CHOSEN.
    # Quantising against a scale the reader will not see puts a second, silent
    # error on top of the first, and it is the reader's value that decides what
    # the weight becomes.
    zeros = low.astype(np.float16)
    scales = ((high - low) / levels).astype(np.float16)
    safe = scales.astype(np.float32)
    safe[safe == 0] = 1.0
    codes = np.clip(np.rint((grid - zeros.astype(np.float32)) / safe), 0, levels).astype(np.uint8)
    return codes, scales.reshape(-1), zeros.reshape(-1)


def pack(codes):
    """32 five-bit codes into 20 bytes, least significant bit first.

    One trailing byte of slack, so a reader may always take two bytes for a
    code that ends on the final one without walking off the buffer.
    """
    groups, group = codes.shape
    bits = np.unpackbits(codes[:, :, None], axis=2, count=BITS, bitorder="little")
    stream = bits.reshape(groups, group * BITS)
    return np.packbits(stream, axis=1, bitorder="little").astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="model-af3-full-f32")
    parser.add_argument("--out", default="model-af3-int5")
    arguments = parser.parse_args()

    source = ROOT / arguments.source
    out = ROOT / arguments.out
    manifest = json.loads((source / "manifest.json").read_text())
    tensors = manifest["tensors"]

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shards = {}
    for record in tensors.values():
        shards.setdefault(record["file"], []).append(record)

    # 🔴 THE SHARDS ARE RENAMED, BECAUSE THEY ARE NO LONGER WHAT THEY SAY.
    # The float32 export writes weights-NN.f32.bin and this used to reuse that
    # name for its output, so a 265 MiB int5 bundle shipped 26 files each
    # claiming to be float32. Nothing reads the extension - the manifest carries
    # the dtype and names the file - so it was a lie that cost nothing and
    # misinformed everyone who looked.
    renamed = {name: name.replace(".f32.bin", f".int{BITS}.bin")
               for name in shards}

    kept = quantised = 0
    kept_bytes = quantised_bytes = source_bytes = 0
    for filename, records in sorted(shards.items()):
        blob = (source / filename).read_bytes()
        pieces = []
        cursor = 0
        for record in sorted(records, key=lambda r: r.get("byteOffset", 0)):
            count = int(np.prod(record["shape"]))
            start = record.get("byteOffset", 0)
            values = np.frombuffer(blob, dtype="<f4", count=count, offset=start)
            source_bytes += count * 4

            # Every tensor restarts on a four-byte boundary, as the f32 export does.
            pad = (-cursor) % 4
            if pad:
                pieces.append(b"\x00" * pad)
                cursor += pad
            record["byteOffset"] = cursor

            name = next(n for n, r in tensors.items() if r is record)
            record["file"] = renamed[filename]
            if KEEP_FLOAT32.search(name):
                payload = np.ascontiguousarray(values, dtype="<f4").tobytes()
                record["dtype"] = "float32"
                kept += 1
                kept_bytes += len(payload)
            else:
                codes, scales, zeros = quantise(values)
                packed = pack(codes).tobytes() + b"\x00"
                scale_pad = (-len(packed)) % 4
                record["dtype"] = f"int{BITS}"
                record["block"] = GROUP
                record["scaleOffset"] = cursor + len(packed) + scale_pad
                record["zeroOffset"] = record["scaleOffset"] + scales.nbytes
                payload = (packed + b"\x00" * scale_pad
                           + scales.astype("<f2").tobytes()
                           + zeros.astype("<f2").tobytes())
                quantised += 1
                quantised_bytes += len(payload)
            pieces.append(payload)
            cursor += len(payload)
        (out / renamed[filename]).write_bytes(b"".join(pieces))

    manifest["quantisation"] = {
        "scheme": "asymmetric-per-group", "bits": BITS, "group": GROUP,
        "scaleDtype": "float16", "zeroDtype": "float16",
    }
    (out / "manifest.json").write_text(json.dumps(manifest))

    total = kept_bytes + quantised_bytes
    print(f"{quantised} tensors quantised to int{BITS} group {GROUP} asymmetric")
    print(f"{kept} tensors kept float32 ({kept_bytes / 2**20:.1f} MiB - norms and biases)")
    print(f"{source_bytes / 2**20:.1f} MiB -> {total / 2**20:.1f} MiB"
          f"   {source_bytes / total:.2f}x")


if __name__ == "__main__":
    main()
