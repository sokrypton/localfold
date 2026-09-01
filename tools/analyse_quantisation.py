"""What does each quantisation scheme cost AF3's weights, per bit of storage?

    python3 tools/analyse_quantisation.py
    python3 tools/analyse_quantisation.py --tensors 12 --bits 4,5,6,8

WHY BITS PER WEIGHT AND NOT "int4 against int8". A scheme's cost is its codes
PLUS its metadata, and at small group sizes the metadata is most of the
difference. int4 with a scale and a zero point every 16 weights is 6 bits a
weight - the same budget as int5 with a group of 32, which is 1.6x more
accurate. Comparing "int4" to "int5" without saying the group size compares
nothing.

WHAT THIS FOUND, on the six biggest tensors (67% of the parameters):

  bits/w  scheme                          relative RMS error
    4.25  int4 asym g128                        0.1317
    4.43  int4 asym g128 + 1 outlier            0.1092
    4.97  int4 asym g128 + 4 outliers           0.0793
    5.19  int4 asym g64  + 2 outliers           0.0771
    5.25  int5 asym g128                        0.0631   <- int4 stops winning
    6.00  int5 asym g32                         0.0431
    6.50  int6 asym g64                         0.0258
    8.25  int8 asym g128                        0.0076
          AF3's own bfloat16                    0.0039

THREE THINGS WORTH KNOWING.

The zero point is the single biggest win at low precision - 1.28x at four bits -
because a group of weights is not centred on zero and a symmetric fit spends
half its codes on a range that holds nothing.

SEARCHING THE RANGE RATHER THAN TAKING IT FROM THE EXTREMES BUYS ONLY 3-4%.
That is worth knowing because it sounds like it should be the whole game: one
outlier does stretch a group's 16 levels across a span the other 31 weights
never visit. It just turns out the extremes are usually close to the best
choice, and where they are not, the error saved is small.

PULLING THE OUTLIERS OUT ENTIRELY DOES WORK, where clipping them does not.
Storing the 1-4 most extreme weights of each group in float16 and quantising
the rest puts int4 back on the frontier below about 5.2 bits a weight.

NF4 - levels at the quantiles of a gaussian - measured WORSE than plain
asymmetric here (0.0949 against 0.0898 at group 32). AF3's weight groups are
not gaussian enough for it, and its absmax scaling gives up the zero point.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "model-af3-full-f32"


def load(tensors, name):
    record = tensors[name]
    return np.fromfile(MODEL / record["file"], dtype="<f4",
                       count=int(np.prod(record["shape"])),
                       offset=record.get("byteOffset", 0))


def grouped(weights, group, cap, rng):
    """A sample of complete groups - the tail past the last full group is a
    rounding detail, not a scheme difference."""
    usable = len(weights) // group * group
    rows = weights[:usable].reshape(-1, group).astype(np.float64)
    if len(rows) > cap:
        rows = rows[rng.choice(len(rows), cap, replace=False)]
    return rows


def relative(reconstructed, original):
    return float(np.sqrt(((reconstructed - original) ** 2).sum() / (original ** 2).sum()))


def symmetric(rows, bits, clip=1.0):
    levels = 2 ** (bits - 1) - 1
    scale = np.abs(rows).max(1, keepdims=True) * clip / levels
    scale[scale == 0] = 1
    return np.clip(np.round(rows / scale), -levels - 1, levels) * scale


def asymmetric(rows, bits, clip=1.0):
    levels = 2 ** bits - 1
    middle = (rows.max(1, keepdims=True) + rows.min(1, keepdims=True)) / 2
    half = (rows.max(1, keepdims=True) - rows.min(1, keepdims=True)) / 2 * clip
    low = middle - half
    scale = 2 * half / levels
    scale[scale == 0] = 1
    return np.clip(np.round((rows - low) / scale), 0, levels) * scale + low


def searched(rows, bits, quantiser, grid):
    """Each group keeps whichever range minimises its own error."""
    best = None
    output = np.empty_like(rows)
    for clip in grid:
        candidate = quantiser(rows, bits, clip)
        error = ((candidate - rows) ** 2).sum(1)
        if best is None:
            best = error.copy()
            output[:] = candidate
        else:
            better = error < best
            best[better] = error[better]
            output[better] = candidate[better]
    return output


def with_outliers(rows, bits, count):
    """The `count` most extreme weights of each group kept in float16, and
    replaced by the median so they do not set the range for the rest."""
    output = rows.copy()
    order = np.argsort(-np.abs(rows - rows.mean(1, keepdims=True)), axis=1)[:, :count]
    index = np.arange(len(rows))[:, None]
    kept = rows[index, order].copy()
    output[index, order] = np.median(rows, axis=1, keepdims=True)
    quantised = asymmetric(output, bits)
    quantised[index, order] = kept
    return quantised


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensors", type=int, default=6,
                        help="how many of the biggest tensors to average over")
    parser.add_argument("--bits", default="4,5,6,8")
    parser.add_argument("--cap", type=int, default=20000,
                        help="groups sampled per tensor")
    arguments = parser.parse_args()

    tensors = json.loads((MODEL / "manifest.json").read_text())["tensors"]
    biggest = [name for _, name in sorted(
        ((int(np.prod(record["shape"])), name) for name, record in tensors.items()),
        reverse=True)[:arguments.tensors]]
    rng = np.random.default_rng(0)
    samples = {name: {group: grouped(load(tensors, name), group, arguments.cap, rng)
                      for group in (16, 32, 64, 128)} for name in biggest}
    grid = np.round(np.arange(0.5, 1.0001, 0.02), 3)

    rows = []
    for bits in [int(value) for value in arguments.bits.split(",")]:
        for group in (16, 32, 64, 128):
            def average(function):
                return float(np.mean([function(samples[name][group]) for name in biggest]))
            rows.append((bits + 16 / group,
                         average(lambda x: relative(symmetric(x, bits), x)),
                         f"int{bits} sym g{group}"))
            rows.append((bits + 32 / group,
                         average(lambda x: relative(asymmetric(x, bits), x)),
                         f"int{bits} asym g{group}"))
            rows.append((bits + 32 / group,
                         average(lambda x: relative(searched(x, bits, asymmetric, grid), x)),
                         f"int{bits} asym+search g{group}"))
            if bits == 4:
                for count in (1, 2, 4):
                    extra = count * (16 + np.log2(group)) / group
                    rows.append((bits + 32 / group + extra,
                                 average(lambda x: relative(with_outliers(x, bits, count), x)),
                                 f"int{bits} asym g{group} + {count} outlier"
                                 f"{'s' if count > 1 else ''}"))

    rows.sort()
    print(f"averaged over the {len(biggest)} biggest tensors, "
          f"{arguments.cap} groups sampled from each\n")
    print(f"{'bits/weight':>11s} {'rel error':>10s}  scheme")
    best = float("inf")
    for bits_per_weight, error, label in rows:
        marker = " *" if error < best else "  "
        best = min(best, error)
        print(f"{bits_per_weight:11.2f} {error:10.4f}  {label:34s}{marker}")
    print("\n * marks the frontier: nothing cheaper is also more accurate.")
    print("   AF3's own bfloat16 is 0.0039, for scale.")


if __name__ == "__main__":
    main()
