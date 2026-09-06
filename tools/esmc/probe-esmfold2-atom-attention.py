"""Inside ONE atom attention, so a 2.4e-3 residual can be localised.

    .venv-esm/bin/python tools/esmc/probe-esmfold2-atom-attention.py

🔴 THE MODULE'S OUTPUT ALONE SAYS "SOMEWHERE IN A PROJECTION, A NORM, A
ROTATION, A MASKED SOFTMAX AND A GATE". Every one of those was verified by
reading against two independent sources and by sweeping nine convention arms,
and the residual did not move - which is exactly when the next step is to
capture the intermediates rather than to read again.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

SEQUENCE = ('MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYN'
            'IQKESTLHLVLRLRGGMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGT')


def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    from esm.models.esmfold2 import EsmFold2ExperimentalModel
    from esm.models.esmfold2.protein_utils import prepare_protein_features
    from esm.models.esmfold2 import layers as L

    sequence = SEQUENCE[:40]
    model = EsmFold2ExperimentalModel.from_pretrained(
        str(ROOT / 'esmfold2-fast-600m'), load_esmc=False).eval()

    attention = model.inputs_embedder.atom_attention_encoder.atom_transformer.blocks[0].attn
    captured = {}
    original = attention.forward

    def instrumented(x, attention_params):
        cos, sin = attention_params[0], attention_params[1]
        B, N = x.shape[:2]
        qkv = attention.Wqkv(x)
        qkv = qkv.view(B, N, 3, attention.n_heads, attention.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)
        captured['qkv_raw'] = torch.stack([q, k, v]).clone()
        q, k = L.qk_norm(q), L.qk_norm(k)
        captured['qk_normed'] = torch.stack([q, k]).clone()
        q = L.apply_rotary_emb_3d(q, cos, sin)
        k = L.apply_rotary_emb_3d(k, cos, sin)
        captured['qk_roped'] = torch.stack([q, k]).clone()
        captured['cos'] = cos.clone()
        captured['sin'] = sin.clone()
        captured['input'] = x.clone()
        out = original(x, attention_params)
        captured['output'] = out.clone()
        return out

    attention.forward = instrumented
    features = prepare_protein_features(sequence)
    model(**features, num_diffusion_samples=1, seed=0)
    attention.forward = original

    if not captured:
        raise SystemExit('the attention never ran')
    # 🔴 THE DTYPE IS PART OF THE ANSWER, so it is recorded rather than cast
    # away. This module downcasts q, k and v to bfloat16 whatever the model was
    # loaded as, and a probe that silently promoted everything to float32 would
    # hide the one thing it exists to find.
    for name, t in captured.items():
        print('  %-12s %-8s %s' % (name, str(t.dtype).replace('torch.', ''), list(t.shape)))
    payload = {name: {'shape': list(t.shape), 'dtype': str(t.dtype).replace('torch.', ''),
                      'values': np.asarray(t.detach().cpu().float(),
                                           np.float32).reshape(-1).tolist()}
               for name, t in captured.items()}
    out = ROOT / 'oracle-dumps' / 'esmfold2-atom-attention-40.json'
    out.write_text(json.dumps(payload))
    print('wrote %s  (%.1f MB): %s' % (out, out.stat().st_size / 1e6,
                                       ', '.join('%s%s' % (k, v['shape'])
                                                 for k, v in payload.items())))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
