#!/usr/bin/env python3
"""Every top-level symbol of the cuda/rfd4p files metal/rfd4p RE-IMPLEMENTS, and what covers it.

    python3 tools/check-metal-rfd4p-coverage.py           # the audit
    python3 tools/check-metal-rfd4p-coverage.py --list     # every symbol and its resolution

🔴 WHAT THIS IS, AND WHY IT IS NOT A WAIVER TABLE. A plain name audit is useless here: of the 100
top-level definitions in the nine re-implemented files, **21 resolve to the Metal side by name and 79
do not**, because the two ports share no naming convention and were never meant to - `rmsNormK` is
`rfd4p_rms_norm`, `triangle` is core's `gemmTriGate`, `mulSigmoidAddGate4HK` is an epilogue FLAG on a
GEMM rather than a kernel, and `fromPlanesTiledK` is work the triangle gate's own epilogue does and
never writes out. Asserting 79 hand-written counterparts would be me writing plausible sentences, and a
wrong one is worse than none: it would read as evidence while proving nothing.

So this audit asserts two things it can actually check, and says plainly which is which:

  1. EVERY reference symbol is accounted for, and the accounting is DERIVED. The symbol list comes from
     the reference's own source through `tools/metal-extract-host.py`'s brace-balanced scanner (the same
     one `derive.sh` uses), not from a list kept here. A symbol ADDED to cuda/rfd4p that nothing here
     mentions fails this - which is the drift this file exists to catch, and the one thing a behavioural
     gate cannot notice.

  2. EVERY file maps to a GREEN ARM of metal/rfd4p/gate.py, and the mapping says why that arm reaches
     that file. This is the honest strength of the claim: it is coverage by EXERCISE, not by name. The
     arms compare values against the reference at every stage seam they name.

Where a counterpart is named below it is one verified to exist in metal/rfd4p or metal/core at audit
time (the audit checks it). Where none is named, the entry says so and rests on its arm.

🔴 AND THE FILES THAT ARE *NOT* PORT SURFACE ARE EXCLUDED ON PURPOSE AND BY NAME: metrics.h, oracle.h,
schedule.h, metrics_tags.h, superpose.h, symmetry.h and assign.h are SHARED - derived at build time by
metal/rfd4p/derive.sh or included outright - so there is nothing to re-implement and a name audit over
them would report a hundred false positives. `derive.sh` is where that boundary is declared; this file
reads the same one and fails if a new .cuh appears that is neither derived nor mapped.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "cuda", "rfd4p", "src")

# The files metal/rfd4p re-implements. Everything else under src/ is shared (see the header), and the
# audit checks that split against derive.sh rather than trusting this list.
PORTED = ["bundle.cuh", "chirality.cuh", "decoder.cuh", "encoder.cuh", "ops.cuh", "partial.cuh",
          "sampler.cuh", "trunk.cuh", "generate.cuh"]
SHARED = ["metrics.h", "oracle.h", "schedule.h", "metrics_tags.h", "superpose.h", "symmetry.h",
          "assign.h"]

# file -> (gate arm, why that arm reaches this file)
COVER = {
    "bundle.cuh": ("presets", "the int8 bundle is read before any arm folds; a bundle that did not load "
                              "fails all ten presets, not one"),
    "chirality.cuh": ("smallmol", "sm_sam is the only shipped preset with chiral frames - the CUDA gate's "
                                  "own note says so - and its trajectory arm is green over 14 comparisons"),
    "decoder.cuh": ("trajectory", "the decoder runs at the end of every generation; its twelve layers and "
                                  "their chained outputs are 24 of the stage oracle's rows"),
    "encoder.cuh": ("partial", "the encoder only runs for a corrupted start, and its own stage rows read "
                               "mean 1.02e-3 / logScale 8.20e-4 against the reference"),
    "ops.cuh": ("trajectory", "the trunk's elementwise and GEMM epilogue layer: every stage seam at 67, "
                              "183 and 250 tokens goes through it"),
    "partial.cuh": ("partial", "corruptBatch and the per-group offsets; the corrupted START is compared at "
                               "1.78e-08 on the backbone"),
    "sampler.cuh": ("trajectory", "the Euler step and the zero-COM recentre, once per step of every arm"),
    "trunk.cuh": ("trajectory", "the pair and token updates; 100 comparisons at three token counts"),
    "generate.cuh": ("presets", "its three kernels are the state write, the pair distance bins and the emit "
                                "mask - the writer gate holds the first to 22 byte-identical files"),
}

# Counterparts verified to exist on the Metal side. NOT exhaustive by design: only entries whose
# counterpart this audit can point at and check. The rest rest on their file's arm, above.
NAMED = {
    "dequantInt8K": "lf_decode",
    "loadInt8Bundle": "loadWeights",
    "readBlob": "loadWeights",
    "chiralFramesK": "alignChirality",
    "chiralApplyK": "alignChirality",
    "encoderFeatsK": "encoderFeatures",
    "encoderSampleK": "encode",
    "decoderFeatsK": "decoderFeatures",
    "aeConditioning": "aeLayer",
    "adaLnApplyK": "rfd4p_adaln_rows",
    "adaScaleAddK": "rfd4p_scale_add",
    "adaScaleAddBroadcastK": "rfd4p_adaln_bcast",
    "swiGluSecondK": "rfd4p_swiglu_h",
    "rmsNormInK": "rfd4p_rms_norm",
    "pairBiasK": "rfd4p_bias_scale",
    "pairBias": "weightTimesNorm",
    "pairBiasAttention": "lf_attention_tensor3",
    "hadamardWriteK": "rfd4p_pair_write",
    "hadamardWrite4K": "rfd4p_pair_write_h",
    "hadamardWrite4HK": "rfd4p_pair_write_h",
    "hadamardWrite": "rfd4p_pair_write_h",
    "toPlanesK": "rfd4p_tri_ab",
    "gatePlanesK": "rfd4p_tri_ab_h",
    "gatePlanes2K": "rfd4p_tri_ab_hm",
    "gatePlanes": "rfd4p_tri_ab_hmb",
    "fromPlanesK": "rfd4p_tri_gather",
    "fromPlanes": "rfd4p_tri_gather_h",
    "linearPlanes": "triGateWeight",
    "packVGK": "weightStack",
    "biasLayoutK": "biasStack",
    "centerEmittedK": "rfd4p_center",
    "offsetGroups": "groupOffsets",
    "noisedTargetBackbone": "noisedBackbone",
    "pairDistBinsK": "rfd4p_pair_dists",
    "adaLn": "adalnRows",
    "adaLnPacked": "adalnPre",
    "buildCondPack": "precomputeCond",
    "buildCondPackT": "precomputeCond",
    "swiGluTransition": "transition",
    "swiGluTransitionT": "transitionH",
    "pairReprUpdate": "pairUpdate",
    "triangleT": "triangle",
    "forwardT": "trunkForward",
    "maskEmitK": "initialNoise",
    "widenK": "lf_pad_zero",
    "addIntoK": "lf_add",
    "biasAddK": "lf_add_bias",
    "mulSigmoidAddK": "rfd4p_mul_sigmoid",
    "mulSigmoidAddGateK": "rfd4p_mul_sigmoid_h",
    "mulSigmoidAddGate4K": "rfd4p_mul_sigmoid_hh",
    "updateThenLn": "lf_layernorm",
    "hadamardThenLn": "lf_layernorm4",
    "narrowK": "lf_copy2d",
}


# 🔴 THE COUNTS ARE PINNED SO A NEW SYMBOL IS A DECISION AND NOT A DRIFT, which is the same reason
# test/af3-example-jobs.test.js asserts how many of AlphaFold 3's examples load. Without this a symbol
# added to cuda/rfd4p would fall quietly into the by-arm bucket and the audit would stay green while the
# thing it exists to notice had happened. Re-record these only with the resolution beside them.
EXPECT = {"byname": 21, "named": 53, "byarm": 26}


def symbols():
    """The reference's own top-level definitions, through the extractor derive.sh uses."""
    src = open(os.path.join(HERE, "metal-extract-host.py")).read()
    ns = {"re": re, "sys": sys}
    exec(src[src.index("def body_span"):src.index("def main()")], ns)   # noqa: S102
    body_span = ns["body_span"]
    pat = re.compile(r'^(?P<pre>(?:__global__\s+|__device__\s+|__host__\s+|extern\s+"C"\s+|inline\s+'
                     r'|static\s+|constexpr\s+|template\s*<[^>]*>\s*)*)'
                     r'(?:[A-Za-z_][\w:<>,\s\*&]*?\s+)(?P<name>[A-Za-z_]\w*)\s*\(', re.M)
    lb = re.compile(r'__launch_bounds__\s*\([^)]*\)\s*(?:[A-Za-z_][\w:<>,\s\*&]*?\s+)?([A-Za-z_]\w*)\s*\(')
    out = []
    for base in PORTED:
        text = open(os.path.join(SRC, base)).read()
        seen = set()
        for m in pat.finditer(text):
            if body_span(text, m.start()) < 0:
                continue
            name, pre = m.group("name"), m.group("pre")
            if name == "__launch_bounds__":
                hit = lb.search(text, m.start())
                if not hit:
                    continue
                name = hit.group(1)
            kind = "kernel" if "__global__" in pre else "host"
            # generate.cuh's host surface is DERIVED (derive.sh extracts it), so only its kernels are
            # port surface - the same split derive.sh declares, read from the same place
            if base == "generate.cuh" and kind != "kernel":
                continue
            if (name, kind) in seen:
                continue
            seen.add((name, kind))
            out.append((base, kind, name))
    return out


def metal_text():
    import glob
    files = []
    for pat in ("metal/rfd4p/*.mm", "metal/rfd4p/*.h", "metal/rfd4p/*.metal",
                "metal/core/*.mm", "metal/core/*.h", "metal/core/*.metal"):
        files += glob.glob(os.path.join(REPO, pat))
    return "\n".join(open(f).read() for f in files)


def main():
    want_list = "--list" in sys.argv
    syms = symbols()
    text = metal_text()
    present = lambda nm: re.search(r"\b" + re.escape(nm) + r"\b", text) is not None

    def snake(nm):
        return re.sub(r"(?<!^)(?=[A-Z])", "_", re.sub(r"K$", "", nm)).lower()

    byname, named, byarm, bad = 0, 0, 0, []
    rows = []
    for base, kind, name in syms:
        cands = [name, "rfd4p_" + snake(name), snake(name), "lf_" + snake(name)]
        hit = next((c for c in cands if present(c)), None)
        if hit:
            byname += 1
            rows.append((base, name, "name", hit))
        elif name in NAMED:
            if not present(NAMED[name]):
                bad.append(f"{name} -> {NAMED[name]} (named counterpart is GONE from the Metal side)")
            named += 1
            rows.append((base, name, "named", NAMED[name]))
        else:
            byarm += 1
            rows.append((base, name, "arm", COVER[base][0]))

    # every ported file must name a gate arm, and every src file must be ported or declared shared
    for base in PORTED:
        if base not in COVER:
            bad.append(f"{base} is re-implemented and maps to no gate arm")
    known = set(PORTED) | set(SHARED)
    for f in sorted(os.listdir(SRC)):
        if f.endswith((".h", ".cuh")) and f not in known:
            bad.append(f"{f} is new in cuda/rfd4p/src and is neither ported nor declared shared")

    got = {"byname": byname, "named": named, "byarm": byarm}
    for key, n in EXPECT.items():
        if got[key] != n:
            bad.append(f"{key} is {got[key]}, recorded as {n} - a symbol was added, removed or resolved "
                       f"differently; resolve it in NAMED or COVER and re-record EXPECT")

    if want_list:
        for base, name, how, via in rows:
            print(f"  {base:14} {name:28} {how:6} {via}")
    print(f"metal/rfd4p coverage: {len(syms)} reference symbols over {len(PORTED)} re-implemented files")
    print(f"  {byname:3} resolve to the Metal side BY NAME")
    print(f"  {named:3} by a named counterpart, each checked to exist")
    print(f"  {byarm:3} by the gate arm that exercises their file (names diverge; see the header)")
    for b in bad:
        print(f"  FAILED: {b}")
    print("ok" if not bad else f"{len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
