#!/usr/bin/env python3
"""This tree's int8 RFD4-Proteina trunk against bios-labs' MLX fp16 trunk, tensor by tensor.

    python3 tools/compare-rfd4p-mlx-weights.py <ours.safetensors> <theirs_fp16.safetensors>

🔴 WHY THIS IS WORTH A TOOL. `bios-labs/RFD4-Proteina-MLX-fp16` is an independent Apple-silicon port of
the same model - its own kernels, its own runtime (MLX), its own conversion script - built from the SAME
source checkpoint revision this tree pins (`a06e2509...`, stated in its `weights/MANIFEST.json`) and
cast fp32 -> fp16 with, in its own words, "no transposition, renaming, retraining, pruning or
quantization below fp16". That makes it the nearest thing to a lossless copy of the trunk weights that
can be fetched without the original 3.5 GB checkpoint, and so the one reference against which this
tree's INT8 bundle can be priced: a disagreement here is this port's quantisation, not a port
difference, because no inference runs on either side.

What it answers, which nothing here could answer before: how much the int8 bundle costs at the weight
level, per tensor, and whether any tensor is an outlier rather than the average being fine. The fold
gates measure the end of that chain; this measures its start.

🔴 AND IT SAMPLES RATHER THAN READING 178 M PARAMETERS IN PYTHON. Up to `--sample` evenly spaced
elements per tensor (20,000 by default), which for a relRMS over tensors of 10^3 to 10^7 elements is far
more than enough - the estimate's own standard error is below 1 % at that count - and keeps the whole
comparison inside a few seconds without numpy, which the system python3 here does not have.

🔴 WHAT IT MEASURED THE FIRST TIME, and the reading that was nearly wrong. All **558** of their trunk
tensors match ours by name AND shape - none missing, none reshaped - which is an independent check on
this tree's weight export that nothing else here provides. Then, partitioned by what our bundle actually
stores:

  our float32 (285 tensors)  median relRMS **2.06e-4**  (1.47e-4 to 2.61e-4)  -> THEIR fp16 cast
  our int8    (273 tensors)  median relRMS **6.07e-3**  (5.75e-3 to 9.66e-3)  -> OUR quantisation

The float32 group is the control and it lands where fp16's eleven significant bits say it must (~2.8e-4
RMS at most), in a tight band, so it is pure rounding and not a layout or scaling difference. Against
that, this bundle's int8 costs **about 30x more than fp16** per quantised trunk tensor, and spread
evenly - the worst (`transformer_layers.13.transition.linear_out`, 9.66e-3) is only 1.6x the median, so
there is no single bad tensor to fix, just a precision floor. Read UNPARTITIONED the whole set gives a
median of 2.5e-4, which looks like a flattering int8 figure and is really the 285 float32 tensors
holding the average down; that reading was taken and corrected, and it is why this tool wants the
manifest.

Names are matched after stripping this tree's own prefix (`w/`), since the bundle stores a trunk tensor
as `w/nn.*` and theirs as `nn.*`; a tensor present on one side only is reported rather than skipped,
because a missing name is the interesting case.
"""
import json
import mmap
import struct
import sys

FMT = {"F16": ("e", 2), "F32": ("f", 4), "BF16": (None, 2)}


def open_safetensors(path):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    n = struct.unpack_from("<Q", mm, 0)[0]
    header = json.loads(mm[8:8 + n])
    base = 8 + n
    return mm, {k: v for k, v in header.items() if k != "__metadata__"}, base


def read_sampled(mm, base, entry, want):
    """`want` evenly spaced elements of one tensor, as floats."""
    dtype = entry["dtype"]
    if dtype not in FMT or FMT[dtype][0] is None:
        return None
    code, width = FMT[dtype]
    start, end = entry["data_offsets"]
    count = (end - start) // width
    if count == 0:
        return []
    step = max(1, count // want)
    out = []
    at = base + start
    for i in range(0, count, step):
        out.append(struct.unpack_from("<" + code, mm, at + i * width)[0])
    return out


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__.strip().splitlines()[2].strip())
    ours_path, theirs_path = sys.argv[1], sys.argv[2]
    want = 20000
    for a in sys.argv[3:]:
        if a.startswith("--sample="):
            want = int(a[len("--sample="):])

    # 🔴 PARTITIONED BY WHAT OUR BUNDLE ACTUALLY STORES, or the number means nothing. `model-rfd4p-int8`
    # is MIXED - 285 of the 558 trunk tensors are float32 and 273 are int8 - so one average over all of
    # them answers two different questions at once:
    #   * where OURS is float32, our value is exact and theirs is a cast, so the difference is THEIR fp16
    #     rounding and nothing of ours (fp16 carries 11 significant bits, so its own relRMS is ~2.8e-4);
    #   * where OURS is int8, our quantisation dominates and the number is the price of this bundle.
    # Read together they gave a median of 2.5e-4 that looked like a flattering int8 figure and was really
    # the fp16 group holding the average down.
    codec = {}
    mpath = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else None
    if mpath:
        man = json.load(open(mpath))
        tt = man.get("tensors") or man
        for k, v in tt.items():
            if isinstance(v, dict):
                codec[k] = v.get("codec") or v.get("dtype")

    om, oh, ob = open_safetensors(ours_path)
    tm, th, tb = open_safetensors(theirs_path)
    # this tree prefixes a weight with `w/`; theirs are bare `nn.*`
    ours_by_bare = {}
    for k in oh:
        ours_by_bare.setdefault(k[2:] if k.startswith("w/") else k, k)

    both, only_theirs, shape_diff, rows, degenerate = 0, [], [], [], []
    for name, entry in sorted(th.items()):
        ok = ours_by_bare.get(name)
        if ok is None:
            only_theirs.append(name)
            continue
        oe = oh[ok]
        if list(oe["shape"]) != list(entry["shape"]):
            shape_diff.append(f"{name}: ours {oe['shape']} theirs {entry['shape']}")
            continue
        a = read_sampled(om, ob, oe, want)
        b = read_sampled(tm, tb, entry, want)
        if a is None or b is None or len(a) != len(b):
            shape_diff.append(f"{name}: unreadable or sampled differently")
            continue
        num = sum((x - y) * (x - y) for x, y in zip(a, b))
        den = sum(y * y for y in b)
        # 🔴 A ZERO DENOMINATOR IS REPORTED, NOT SCORED 0.0. `rel = ... if den > 0 else 0.0` hands a
        # PERFECT score to a tensor that was never compared - an empty one, or one whose sampled values
        # are all zero (a zeroed bias is ordinary) - which is this tree's own documented trap, "an
        # unguarded relRMS over a non-array returns exactly 0". Checked against the checkpoint this was
        # run on: 0 of 558 tensors are empty and 0 are all-zero, so no reported number changes; the guard
        # is for the next checkpoint, where silence would read as agreement.
        if den <= 0:
            degenerate.append(f"{name}: denominator 0 over {len(a)} sampled "
                              f"({'empty' if not a else 'all-zero'}) - NOT scored")
            continue
        rel = (num / den) ** 0.5
        both += 1
        rows.append((rel, name, len(a)))

    rows.sort(reverse=True)
    groups = {}
    for rel, name, n in rows:
        groups.setdefault(codec.get(name, "unknown"), []).append((rel, name))
    print(f"ours   {ours_path}: {len(oh)} tensors")
    print(f"theirs {theirs_path}: {len(th)} tensors")
    print(f"matched by name and shape: {both}")
    if only_theirs:
        print(f"  !! {len(only_theirs)} of theirs not found on our side: {', '.join(only_theirs[:6])}")
    if shape_diff:
        print(f"  !! {len(shape_diff)} shape mismatches: {shape_diff[0]}")
    if degenerate:
        print(f"  !! {len(degenerate)} not comparable (a zero denominator is not a score):")
        for d in degenerate[:5]:
            print(f"       {d}")
    if not rows:
        return 1
    for name, label in (("float32", "our F32 vs their fp16 -> THEIR conversion error"),
                        ("int8", "our INT8 vs their fp16 -> OUR quantisation"),
                        ("unknown", "codec unknown (no manifest passed)")):
        g = groups.get(name)
        if not g:
            continue
        vals = sorted(r[0] for r in g)
        print(f"{name:8} {len(g):3} tensors  median {vals[len(vals)//2]:.4e}  "
              f"min {vals[0]:.4e}  max {vals[-1]:.4e}   ({label})")
    worst = groups.get("int8") or rows
    print("  worst five of the int8 group:" if groups.get("int8") else "  worst five:")
    for rel, name in (worst[:5] if groups.get("int8") else [(r[0], r[1]) for r in rows[:5]]):
        print(f"    {rel:.4e}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
