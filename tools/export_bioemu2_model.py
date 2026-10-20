"""Export BioEmu2 (af3-any-model's bioemu2.bin.zst) as a float32 bundle the page reads.

    python3 tools/export_bioemu2_model.py --blob ~/ported/bioemu2/bioemu2.bin.zst --out model-bioemu2-f32
    python3 tools/quantize_af3.py --source model-bioemu2-f32 --out model-bioemu2-int5

The blob is af3-any-model's conversion of microsoft/bioemu2's bioemu2-preview
checkpoint: one Haiku scope, `diffuser/~/bioemu2`, whose names are the vendor's
state-dict keys with '.' -> '/', torch (out, in) layouts kept, the 4 MSA, 48
PairMixer and 16 dual-stream layers STACKED on a leading axis. Here:

  * every stack is UNSTACKED into `<stack>/<layer>/<rest>`, because a reader
    takes one contiguous span per tensor and a block wants its own;
  * the PAIR TRACK - every tri_mul_out, tri_mul_in and pair transition, the
    PairMixer's 48 and the MSA module's 4 - is written in AF3's layout under
    `pairmixer/<i>/` and `msa/<i>/`, the names webgpu/af3/trunk/pair-track.js
    packs (shared/esmfold2/weights.js trunkBlockWeights reads them). BioEmu2's
    triangle multiplication is Boltz-2's TriangleMultiplicationFull, which is
    AF3's with [a | b] blocked where AF3 interleaves, and the incoming einsum
    reading its operands swapped - tools/esmc/esmfold2_trunk_weights.py's two
    conversions, for the same reason (see its notes). Its transition is a
    SwiGLU with fc1 the gated half, i.e. AF3's blocked [gate | value];
  * everything else keeps the vendor name and the TORCH layout, (out, in):
    webgpu/bioemu2's GEMM reads a weight row by output channel.

float32Tensors names what quantize_af3.py must leave alone: every vector (norms,
biases, the query biases), every table read by GATHER rather than by product
(the element, atom-name, relative-position and method embeddings), the frozen
Fourier projections and the damped-harmonic time constants (taus, quality
factors, the dt gate) - a random projection or a time constant is the thing
whose exact values carry the signal - and anything under 4096 numbers.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "esmc"))
from export_af3_model import ShardWriter, read_blob  # noqa: E402

SCOPE = "diffuser/~/bioemu2"
STACKS = ("sequence_embedder/msa_module/layers", "sequence_embedder/pairmixer_module/layers",
          "structure_module/layers")
PAIR_CHANNELS = 256


def interleave(left, right):
    out = np.empty((left.shape[0], left.shape[1] * 2), np.float32)
    out[:, 0::2] = left
    out[:, 1::2] = right
    return out


def triangle(p, swap):
    """A Boltz-2 tri_mul as AF3's triangleMultiplication wants it."""
    c = PAIR_CHANNELS
    t = lambda name: np.ascontiguousarray(p[name].astype(np.float32).T)
    p_in, g_in = t("p_in/weight"), t("g_in/weight")          # (c, 2c): [a | b]
    first, second = (c, 0) if swap else (0, c)
    halves = lambda w: (w[:, first:first + c], w[:, second:second + c])
    return {
        "leftNormInputScale": p["norm_in/weight"], "leftNormInputOffset": p["norm_in/bias"],
        "centerNormScale": p["norm_out/weight"], "centerNormOffset": p["norm_out/bias"],
        "outputProjection": t("p_out/weight"), "gatingLinear": t("g_out/weight"),
        "projection": interleave(*halves(p_in)), "gate": interleave(*halves(g_in)),
    }


def transition(p):
    """SwiGLU fc3(silu(fc1 LN x) * fc2 LN x) as AF3's blocked [gate | value]."""
    w12 = np.concatenate([p["fc1/weight"], p["fc2/weight"]], 0).astype(np.float32).T
    return {"inputLayerNormScale": p["norm/weight"], "inputLayerNormOffset": p["norm/bias"],
            "transition1": np.ascontiguousarray(w12),
            "transition2": np.ascontiguousarray(p["fc3/weight"].astype(np.float32).T)}


def sub(params, prefix):
    n = len(prefix) + 1
    return {k[n:]: v for k, v in params.items() if k.startswith(prefix + "/")}


def keep_float32(name, array):
    if array.ndim <= 1 or array.size < 4096:
        return True
    return bool(re.search(r"embed_element|embed_atom_name_chars|embed_rel_|method_conditioning|fourier|"
                          r"dt_embed/|dt_gate|encoding_un", name))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--blob", default=str(Path.home() / "ported/bioemu2/bioemu2.bin.zst"))
    ap.add_argument("--out", default="model-bioemu2-f32")
    a = ap.parse_args()
    params = {name: np.asarray(arr) for scope, name, arr in read_blob(a.blob) if scope == SCOPE}
    if not params:
        raise SystemExit(f"{a.blob}: no {SCOPE} scope")
    out = {}
    pair_track_consumed = set()
    # the pair track, in AF3's layout
    for stack, tag, count in (("sequence_embedder/pairmixer_module/layers", "pairmixer", 48),
                              ("sequence_embedder/msa_module/layers", "msa", 4)):
        p = sub(params, stack)
        tz = "transition_z" if tag == "pairmixer" else "z_transition"
        for layer in range(count):
            one = {k: v[layer] for k, v in p.items()}
            for group, values in (("triangleMultiplicationOutgoing", triangle(sub(one, "tri_mul_out"), False)),
                                  ("triangleMultiplicationIncoming", triangle(sub(one, "tri_mul_in"), True)),
                                  ("pairTransition", transition(sub(one, tz)))):
                for leaf, v in values.items():
                    out[f"{tag}/{layer}/{group}/{leaf}"] = v
        for k in p:
            if k.startswith(("tri_mul_out/", "tri_mul_in/", tz + "/")):
                pair_track_consumed.add(f"{stack}/{k}")
    # everything else, unstacked, in the vendor's names and layouts
    for name, value in params.items():
        if name in pair_track_consumed:
            continue
        stack = next((s for s in STACKS if name.startswith(s + "/")), None)
        if stack is None:
            out[name] = value
            continue
        rest = name[len(stack) + 1:]
        for layer in range(value.shape[0]):
            out[f"{stack}/{layer}/{rest}"] = value[layer]
    # the vendor's checkpoint drops nothing here: sequence_embedder/uncond_transition is the classifier-free
    # branch's, which no inference path reads, and is kept so the bundle IS the checkpoint
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(outdir)
    keep = []
    for name in sorted(out):
        v = np.ascontiguousarray(out[name], dtype=np.float32)
        writer.add(name, v)
        if keep_float32(name, v):
            keep.append(name)
    writer.close()
    total = sum(int(np.prod(r["shape"])) for r in writer.records.values())
    manifest = {
        "formatVersion": 1,
        "source": f"BioEmu2 parameters from {Path(a.blob).name}",
        "model": {"name": "bioemu2", "recycles": 3},
        "bundle": {"purpose": "browser-inference", "model": "bioemu2", "encoding": "float32-le"},
        "bioemu2": {"pairChannels": 256, "singleChannels": 384, "tokenChannels": 768, "atomChannels": 128,
                    "atomPairChannels": 16, "msaChannels": 128, "pairmixerBlocks": 48, "msaBlocks": 4,
                    "denoiserLayers": 16, "tokenHeads": 16, "atomHeads": 4, "msaHeads": 8,
                    "inputEmbedderLayers": 3, "sInputsWidth": 455, "trunkPasses": 4, "distogramBins": 64,
                    "layout": {"pairTrack": "af3 (interleaved triangle, blocked transition)",
                               "other": "torch (out, in)"}},
        "coverage": {"included": [SCOPE], "excludedScopes": []},
        "float32Tensors": keep,
        "tensors": writer.records,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{outdir}/  {len(writer.records)} tensors, {total / 1e6:.1f} M parameters, "
          f"{writer.index} shards, {len(keep)} kept float32")


if __name__ == "__main__":
    main()
