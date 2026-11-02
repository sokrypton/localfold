"""The full ESMFold2's MSA encoder: biohub's own module against a port's, on the port's own inputs.

    EF2_DETERMINISTIC=1 EF2_SAVE_ZINIT=zi.f32 EF2_SAVE_MSA=msa.bin EF2_SAVE_INJECT=out.f32 \\
        metal/ef2/fold out.pdb --model=ef2 --sequence=... --a3m=... --weights-dir=<dir>
    ~/.cache/localfold/venv-esm/bin/python tools/esmc/probe-esmfold2-msa-encoder.py \\
        --checkpoint <biohub/ESMFold2 at 8fc3ff4710>/model.safetensors --zinit zi.f32 --msa msa.bin --port out.f32

`esm` 3.4.1's MSAEncoder (esm/models/esmfold2/model.py) built at the checkpoint's widths and loaded from its
`msa_encoder.*` tensors, fed the port's z_init and s_inputs (EF2_SAVE_ZINIT and its ".s") and the rows, deletions and
column mask the port's encoder read (EF2_SAVE_MSA) - so the comparison is the encoder alone, in float32 on the CPU.
EF2_DETERMINISTIC=1 takes the alignment's first rows and masks no column, which is what makes the rows reproducible.
The deletion feature is (pi / 2) atan(d / 3), the vendor's transform (cuda/ef2/README.md).
"""
import argparse
import math
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from safetensors_read import SafeTensors  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--zinit", required=True)
    parser.add_argument("--msa", required=True)
    parser.add_argument("--port", required=True)
    a = parser.parse_args()

    from esm.models.esmfold2.model import MSAEncoder

    ckpt = SafeTensors(a.checkpoint)
    state = {k[len("msa_encoder."):]: torch.from_numpy(np.array(ckpt[k], np.float32))
             for k in ckpt.keys() if k.startswith("msa_encoder.")}
    d_msa, d_inputs = state["embed.weight"].shape[0], state["project_inputs.weight"].shape[1]
    d_pair = state["blocks.0.outer_product_mean.Wout.weight"].shape[0]
    n_layers = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("blocks."))
    enc = MSAEncoder(d_msa=d_msa, d_pair=d_pair, d_inputs=d_inputs, n_layers=n_layers).float().eval()
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if missing or unexpected:
        sys.exit(f"the checkpoint does not fit the module: missing {missing}, unexpected {unexpected}")

    raw = open(a.msa, "rb").read()
    S, T = np.frombuffer(raw[:8], np.int32)
    n = int(S) * int(T)
    rows = np.frombuffer(raw[8:8 + 4 * n], np.int32).reshape(S, T)
    dels = np.frombuffer(raw[8 + 4 * n:8 + 8 * n], np.float32).reshape(S, T)
    mask = np.frombuffer(raw[8 + 8 * n:8 + 12 * n], np.float32).reshape(S, T)
    zi = np.fromfile(a.zinit, np.float32).reshape(1, T, T, d_pair)
    s_in = np.fromfile(a.zinit + ".s", np.float32).reshape(1, T, d_inputs)

    # [B, L, M, ...], as the module's forward says its inputs arrive
    onehot = np.eye(33, dtype=np.float32)[rows] * mask[..., None]
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x))
    with torch.no_grad():
        out = enc(x_pair=t(zi), x_inputs=t(s_in),
                  msa_oh=t(onehot.transpose(1, 0, 2)[None]),
                  has_deletion=t((dels > 0).astype(np.float32).T[None]),
                  deletion_value=t((math.pi / 2 * np.arctan(dels / 3)).astype(np.float32).T[None]),
                  msa_attention_mask=t(mask.T[None]))
    ref = out[0].numpy()
    port = np.fromfile(a.port, np.float32).reshape(ref.shape)
    rel = np.sqrt(((port - ref) ** 2).mean() / (ref ** 2).mean())
    print(f"MSA encoder, {S} rows x {T} tokens: relRMS {rel:.3e} (reference max |x| {np.abs(ref).max():.4g},"
          f" the port's {np.abs(port).max():.4g})")


main()
