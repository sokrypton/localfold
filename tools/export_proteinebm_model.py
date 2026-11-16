"""Export a ProteinEBM checkpoint (jproney/ProteinEBM, MIT) as a float32 bundle the page reads.

    python3 tools/export_proteinebm_model.py \
        --checkpoint ~/Documents/GitHub/ProteinEBM/weights/model_1_frozen_1m_md.pt --out model-proteinebm-1-f32

The checkpoint is a PyTorch Lightning file; the model is `state_dict` under `model.`, every weight float32 in torch's
(out, in) layout, which is what webgpu/proteinebm reads (its GEMM takes a weight row by output channel). Here:

  * `model.` is stripped and '.' -> '/', so `token_transformer/layers/3/adaln/s_scale/weight`;
  * `token_transformer/layers/<i>/output_projection/0/*` is DROPPED: it is the same nn.Linear registered twice
    (`output_projection_linear` wrapped in a Sequential), and the checkpoint stores it twice - asserted equal;
  * `s_to_a_linear/*` is DROPPED: built in __init__ and read by no forward pass;
  * the optimizer, scheduler and Lightning state are not model weights and are not read.

float32Tensors names what a quantiser must leave alone: every vector, the embedding tables read by gather (sequence,
contact), the 3-wide coordinate embeddings and output heads (the energy is the squared norm of a 3-vector head, and
the score is its gradient: these carry the signal), the relative-position table and the frozen Fourier projection.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_af3_model import ShardWriter  # noqa: E402

KEEP_FLOAT32 = ("sequence_embedding/", "contact_embedding/", "noisy_coord_embedding/", "self_conditioning_embedding/",
                "r_update_proj", "sidechain_proj/", "rel_pos/", "single_conditioner/fourier_embed/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()

    import torch
    state = torch.load(a.checkpoint, weights_only=False, map_location="cpu")["state_dict"]
    params = {k[len("model."):].replace(".", "/"): v.float().numpy() for k, v in state.items() if k.startswith("model.")}
    out = {}
    for name, value in params.items():
        if name.startswith("s_to_a_linear/"):
            continue
        # (the LAYER's own: a transition's output_projection/0 is that block's only copy)
        if name.startswith("token_transformer/") and name.split("/")[3:5] == ["output_projection", "0"]:
            twin = name.replace("/output_projection/0/", "/output_projection_linear/")
            assert np.array_equal(value, params[twin]), name
            continue
        out[name] = value
    layers = 1 + max(int(n.split("/")[2]) for n in out if n.startswith("token_transformer/layers/"))

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(outdir)
    keep = []
    for name in sorted(out):
        v = np.ascontiguousarray(out[name], dtype=np.float32)
        writer.add(name, v)
        if v.ndim < 2 or v.size < 4096 or name.startswith(KEEP_FLOAT32):
            keep.append(name)
    writer.close()
    total = sum(int(np.prod(r["shape"])) for r in writer.records.values())
    manifest = {
        "formatVersion": 1,
        "source": f"ProteinEBM parameters from {Path(a.checkpoint).name} (jproney/ProteinEBM, MIT)",
        "model": {"name": "proteinebm"},
        "bundle": {"purpose": "browser-inference", "model": "proteinebm", "encoding": "float32-le"},
        "proteinebm": {"tokenS": 256, "tokenZ": 128, "dimFourier": 256, "single": 512, "layers": layers,
                       "heads": 8, "conditioningTransitions": 2, "relPosMax": 32, "chainMax": 1,
                       "diffuser": {"minB": 0.1, "maxB": 20.0, "coordinateScaling": 0.1},
                       "layout": "torch (out, in)"},
        "float32Tensors": keep,
        "tensors": writer.records,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{outdir}/  {len(writer.records)} tensors, {total / 1e6:.1f} M parameters, "
          f"{writer.index} shards, {len(keep)} kept float32")


if __name__ == "__main__":
    main()
