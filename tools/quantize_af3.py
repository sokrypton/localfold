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
# Defaults, not constants: --bits and --group set them. int5 group 32 is the
# scheme the AF3 and OpenBind bundles ship under and the one the table above
# prices; ESM-C measured int3 at group 128 landing on float32's median crystal
# RMSD, which is 303 MiB against 491 for the same pair.
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


def pack(codes, bits=BITS):
    """`group` codes of `bits` bits each, least significant bit first.

    One trailing byte of slack, so a reader may always take two bytes for a
    code that ends on the final one without walking off the buffer - which is
    why the width is capped at nine in src/reference/dtype.js.

    🔴 THE GROUP MUST PACK INTO WHOLE BYTES. 32 codes of 5 bits is exactly 20,
    of 3 bits exactly 12, of 6 bits exactly 24 - but 32 of 7 is 28 exactly too,
    while 24 of 5 is 15 and a group of 24 would need a case for a code split
    across the boundary. The caller is checked rather than trusted.
    """
    groups, group = codes.shape
    if (group * bits) % 8:
        raise SystemExit("group %d of %d bits does not pack into whole bytes"
                         % (group, bits))
    unpacked = np.unpackbits(codes[:, :, None], axis=2, count=bits, bitorder="little")
    stream = unpacked.reshape(groups, group * bits)
    return np.packbits(stream, axis=1, bitorder="little").astype(np.uint8)


# 🔴 A SHARD COUNT IS A MULTIPLE OF THE CONNECTION COUNT, AND THE SIZES ARE
# BALANCED. Measured against Hugging Face at eight connections, longest first:
# `af3-int5`'s EIGHT shards spent 4.5-5.3 s of a 9 s download with a connection
# idle, because eight shards on eight connections is one shard each - the first
# to finish has nothing else to do and the load ends when the slowest single
# shard does. `esmc-600m-int3`'s fifty-four spent 0.6-1.7 s, at the same
# throughput and 1.8 s of request overhead against 0.27 (a shard request costs a
# measured 271 ms, the 307 to cdn.hf.co included).
#
# So the tail comes from IMBALANCE, not from count: n shards of equal size on n
# connections finish together and leave no tail at all. What costs is a ragged
# last round. This picks a multiple of CONNECTIONS near a 16 MiB target and
# packs longest-first into the emptiest bin, which is the standard makespan
# heuristic and the same reasoning HttpTensorStore.prefetch already applies to
# the ORDER it starts them in.
CONNECTIONS = 8
SHARD_TARGET = 16 * 1024 * 1024


def shard_count(total_bytes, connections=CONNECTIONS, target=SHARD_TARGET,
                override=None):
    """How many files to write, as a multiple of `connections`.

    🔴 THE COUNT IS SIZED ON THE PACKED BYTES, NEVER ON THE FLOAT32 EXPORT.
    `export_af3_model.py` caps a shard at 48 MiB of float32 and this discards
    that layout entirely - an int5 bundle inheriting a float32 one is how a
    265 MiB bundle ended up in 26 pieces chosen for 1405 MiB.

    🔴 AND `override` IS NOT FREE, WHICH IS WHY IT HAS TO BE ASKED FOR. A count
    that is not a multiple of `connections` leaves the last round ragged: 12
    shards on 8 connections is one full round and then four, with four
    connections idle for the whole of it. The floor is a different constraint
    again - a tensor is indivisible, and no sharding beats the largest one - so
    below `total / largest tensor` the count buys nothing and above
    `connections` it costs a round. See pack_shards.
    """
    if override is not None:
        return max(1, min(64, int(override)))
    rounds = max(1, round(total_bytes / target / connections))
    return min(64, connections * rounds)


def pack_shards(payloads, count):
    """Assign each (index, nbytes) to a bin, longest first into the emptiest.

    🔴 A TENSOR IS INDIVISIBLE, so a bundle with one very large tensor cannot be
    balanced better than that tensor - AF3's stacked single-transition weights
    are 40.5 MiB against a 7.9 MiB median. Greedy longest-first is what there is;
    it is optimal to within 4/3 of the best possible makespan.
    """
    bins = [[] for _ in range(count)]
    loads = [0] * count
    for index, nbytes in sorted(payloads, key=lambda p: -p[1]):
        at = loads.index(min(loads))
        bins[at].append(index)
        loads[at] += nbytes
    return [b for b in bins if b], loads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="model-af3-full-f32")
    parser.add_argument("--out", default="model-af3-int5")
    parser.add_argument("--bits", type=int, default=BITS)
    parser.add_argument("--shards", type=int, default=None,
                        help="force the shard count; sized on the PACKED bytes "
                             "either way. Not a multiple of 8 leaves a ragged round.")
    parser.add_argument("--group", type=int, default=GROUP)
    arguments = parser.parse_args()
    bits, group = arguments.bits, arguments.group
    if not 1 <= bits <= 7:
        raise SystemExit("--bits must be 1..7; eight and above are not packed")

    source = ROOT / arguments.source
    out = ROOT / arguments.out
    manifest = json.loads((source / "manifest.json").read_text())
    tensors = manifest["tensors"]
    # 🔴 THE MANIFEST'S OWN float32 LIST WAS BEING IGNORED HERE, AND HONOURED BY
    # tools/quantize_model.py. Every AF3 export writes `float32Tensors` and all
    # 150 of its entries already match KEEP_FLOAT32, so this changed nothing for
    # AF3 and looked like it worked - but a bundle whose float32 tensors are not
    # named `/scale`, `/offset` or `/bias` had no way to say so. ESM-C's
    # embedding table is one: it is a `weights` by name and it is the input to
    # all 36 blocks, which the AF3 port measured at corr 0.19 against native
    # when it was left as raw codes.
    named_float32 = set(manifest.get("float32Tensors", []))
    unknown = named_float32 - set(tensors)
    if unknown:
        raise SystemExit("float32Tensors names tensors not in this export: "
                         + ", ".join(sorted(unknown)))

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shards = {}
    for record in tensors.values():
        shards.setdefault(record["file"], []).append(record)

    # 🔴 THE OUTPUT SHARDS ARE NOT THE INPUT'S. This used to write one file per
    # source shard, so an int3 bundle inherited a layout chosen for float32 -
    # `export_esmc_model.py` caps a shard at 48 MiB and quantisation shrinks each
    # eightfold, which left ESM-C 600M as 54 files of 4.1 MiB. The sizes are only
    # known HERE, after packing, so this is the one place that can choose. See
    # shard_count and pack_shards.
    #
    # 🔴 AND THE BYTES ARE UNCHANGED, WHICH IS WHAT MAKES IT SAFE. Quantisation
    # is deterministic and only the grouping into files moves, so a fold before
    # and after must be identical - which is the check to run, not a tolerance.
    kept = quantised = 0
    kept_bytes = quantised_bytes = source_bytes = 0
    built = []                      # (name, record, payload) in manifest order
    for filename, records in sorted(shards.items()):
        blob = (source / filename).read_bytes()
        for record in sorted(records, key=lambda r: r.get("byteOffset", 0)):
            count = int(np.prod(record["shape"]))
            start = record.get("byteOffset", 0)
            values = np.frombuffer(blob, dtype="<f4", count=count, offset=start)
            source_bytes += count * 4
            name = next(n for n, r in tensors.items() if r is record)
            if KEEP_FLOAT32.search(name) or name in named_float32:
                payload = np.ascontiguousarray(values, dtype="<f4").tobytes()
                record["dtype"] = "float32"
                record.pop("block", None)
                record.pop("scaleOffset", None)
                record.pop("zeroOffset", None)
                kept += 1
                kept_bytes += len(payload)
            else:
                codes, scales, zeros = quantise(values, group, bits)
                packed = pack(codes, bits).tobytes() + b"\x00"
                scale_pad = (-len(packed)) % 4
                record["dtype"] = f"int{bits}"
                record["block"] = group
                # ...offsets WITHIN the payload; the shard offset is added below,
                # once packing has decided where this payload starts.
                record["scaleOffset"] = len(packed) + scale_pad
                record["zeroOffset"] = record["scaleOffset"] + scales.nbytes
                payload = (packed + b"\x00" * scale_pad
                           + scales.astype("<f2").tobytes()
                           + zeros.astype("<f2").tobytes())
                quantised += 1
                quantised_bytes += len(payload)
            built.append((name, record, payload))
        del blob

    total = sum(len(p) for _, _, p in built)
    bins, loads = pack_shards([(i, len(p)) for i, (_, _, p) in enumerate(built)],
                              shard_count(total, override=arguments.shards))
    for shard, members in enumerate(bins):
        pieces = []
        cursor = 0
        # ...in manifest order inside the file, so a reader stepping through it
        # walks forwards; the packing chose WHICH file, not the order within.
        for index in sorted(members):
            _, record, payload = built[index]
            pad = (-cursor) % 4          # every tensor restarts four-byte aligned
            if pad:
                pieces.append(b"\x00" * pad)
                cursor += pad
            record["file"] = f"weights-{shard:02d}.int{bits}.bin"
            record["byteOffset"] = cursor
            if "scaleOffset" in record:
                record["scaleOffset"] += cursor
                record["zeroOffset"] += cursor
            pieces.append(payload)
            cursor += len(payload)
        (out / f"weights-{shard:02d}.int{bits}.bin").write_bytes(b"".join(pieces))

    manifest["quantisation"] = {
        "scheme": "asymmetric-per-group", "bits": bits, "group": group,
        "scaleDtype": "float16", "zeroDtype": "float16",
    }
    (out / "manifest.json").write_text(json.dumps(manifest))

    total = kept_bytes + quantised_bytes
    print(f"{quantised} tensors quantised to int{bits} group {group} asymmetric")
    print(f"{kept} tensors kept float32 ({kept_bytes / 2**20:.1f} MiB - norms and biases)")
    print(f"{source_bytes / 2**20:.1f} MiB -> {total / 2**20:.1f} MiB"
          f"   {source_bytes / total:.2f}x")


if __name__ == "__main__":
    main()
