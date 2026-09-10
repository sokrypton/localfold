"""Ask every tuning knob whether it does anything at all.

    python3 tools/audit-knobs.py --tool=fold-af2
    python3 tools/audit-knobs.py --tool=fold --args=--model=/model-af3-int5/manifest.json

🔴 THREE DEAD KNOBS TURNED UP IN ONE DAY AND ALL THREE BY ACCIDENT.
`matrixLinear: false` fell through into the matrix path because the gate tested
only null and undefined, so every arm measured with it "off" was measured on;
`attentionProjectMatrix` was set from a feature list that says the units exist
without saying they are 8x8, and crashed a whole model; and
`pairTransitionChunkBytes` was placed in an options object that its only reader
never looked at, so every AF3 arm was measured at the default. Three found while
looking at something else implies more findable by searching.

This searches. For each knob it runs the same workload with the knob set to a
value DIFFERENT from the one this device resolves today, and compares two
digests from probe-compiles.js: every shader compiled, by label and content, and
every dispatch, by label and grid. A knob that changes neither compiled a
different kernel nowhere and launched a different grid nowhere.

🔴 THAT IS "SUSPICIOUS", NOT "DEAD", AND THE DIFFERENCE MATTERS. A knob can be
alive and still move neither digest: `keepTrunkWeights` decides whether buffers
are released, `batchComputePasses` decides how dispatches are grouped into
passes, `allowHostWeightPacking` decides where a decode happens. The output
names them so a reader can tell the two apart - this tool says where to LOOK,
and only reading the code says which it is.

The workload matters too: a knob for the diffusion sampler cannot move anything
in an AF2 fold. Run it per tool and read each column against what that tool
exercises.
"""
import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Knobs whose value is an object or carries a comma: `--tune` splits its
# argument on commas, so these cannot be written on a command line at all. They
# are listed rather than skipped silently - an unaudited knob is exactly what
# this tool exists to find.
# 🔴 A KNOB WHOSE VALUE IS A PRECISION TAKES A STRING, AND GUESSING `true` FOR
# IT FAILS THE RUN RATHER THAN AUDITING IT. Two did on the first pass -
# opmContractPrecision and stagedMatrixResult - which is the tool reporting its
# own blind spot rather than a finding, so they are named here instead.
PRECISIONS = ("opmContractPrecision", "stagedMatrixResult", "opmMatrixOutput")


def alternatives(name, value):
    """Values worth trying for a knob currently resolved to `value`."""
    if name in PRECISIONS and not isinstance(value, bool):
        return ["f16", "f32"] if name != "opmMatrixOutput" else [not bool(value)]
    if isinstance(value, bool):
        return [not value]
    if value is None:
        return [True]
    if isinstance(value, (int, float)):
        if value == 0:
            return [1]
        return [value * 2, max(1, int(value // 2))]
    if isinstance(value, str) and "," not in value:
        return None            # a string tile: no safe alternative to guess
    return None                # object-valued: --tune cannot express it


def run(tool, extra, tune):
    args = ["node", "tools/gpu-chrome.mjs", "tools/gpu/probe-compiles.js", "--tool=" + tool]
    args += [a for a in extra if a]
    if tune is not None:
        args.append("--tune=" + tune)
    done = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=1800)
    text = "\n".join(l for l in done.stdout.split("\n") if not l.startswith("[gpu-chrome]"))
    at = text.find("\n{")
    body = text[at + 1:] if at >= 0 else text[text.find("{"):]
    try:
        return json.loads(body)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", default="fold-af2")
    parser.add_argument("--args", default="", help="extra flags for the tool, space separated")
    parser.add_argument("--tuning", default=None,
                        help="a JSON file of this device's resolved tuning, from probe-tuning.js")
    parser.add_argument("--only", default="", help="audit one knob")
    args = parser.parse_args()
    extra = args.args.split()

    with open(args.tuning) as handle:
        tuning = json.load(handle)

    base = run(args.tool, extra, None)
    if base is None:
        print("the baseline run produced no result", file=sys.stderr)
        return 1
    print(f"baseline  shaders {base['shaderDigest']} ({base['shaderCount']})"
          f"  dispatch {base['dispatchDigest']} ({base['dispatchShapes']})")

    unexpressible, suspicious, live, failed = [], [], [], []
    for name in sorted(tuning):
        if args.only and name != args.only:
            continue
        values = alternatives(name, tuning[name])
        if values is None:
            unexpressible.append(name)
            continue
        moved = False
        for value in values:
            got = run(args.tool, extra, f"{name}={json.dumps(value)}")
            if got is None:
                failed.append((name, value))
                continue
            if (got["shaderDigest"] != base["shaderDigest"]
                    or got["dispatchDigest"] != base["dispatchDigest"]):
                moved = True
                break
        (live if moved else suspicious).append(name)
        print(f"{'moved   ' if moved else 'NO EFFECT'} {name} = {tuning[name]!r} -> {values}")

    print("\n--- summary ---")
    print(f"moved a shader or a grid ({len(live)}): {', '.join(live)}")
    print(f"🔴 changed NOTHING ({len(suspicious)}): {', '.join(suspicious)}")
    print(f"not expressible through --tune ({len(unexpressible)}): {', '.join(unexpressible)}")
    if failed:
        print(f"the run failed ({len(failed)}): {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
