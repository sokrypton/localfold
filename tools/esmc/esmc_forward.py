"""ESM-C's tower, in torch, from the raw checkpoint - and ESMFold2's shim on top.

This is the thing the compression study measures. It is written from the block
layout ../alphafold3/converters/esmc.py documents rather than by importing the
`esm` package, so that a quantisation arm and its reference do not share a
forward pass; `probe-esmc-compression.py --oracle` checks it against
transformers' own ESMC when that is installed.

    h  = LN(x; attn_norm) @ qkv        -> [q | k | v]
    q,k = LN(q; q_ln), LN(k; k_ln)     # over the FULL d_model, not per head
    q,k = RoPE(q, k)                   # head_dim 64, base 10000, split halves
    x  = x + out_proj(attn(q,k,v)) / residual_scale
    x  = x + fc2(swiglu(fc1(LN(x; ffn_norm)))) / residual_scale

🔴 residual_scale IS sqrt(n_layers / 36), WHICH IS EXACTLY 1 AT 600M. ESM-C 600M
has 36 layers, so every division is by one and a port that dropped the term
would still agree here and diverge on the 6B tower. It is kept, and derived.

🔴 THE LAST HIDDEN STATE IS POST THE FINAL LayerNorm AND THE OTHER 36 ARE NOT.
ESMFold2 mixes all 37 - embedding plus 36 blocks - and returning the pre-norm
value for the last reads corr 0.909 against native where every other layer is
>= 0.9998 (the AF3 port's own measurement).
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from safetensors_read import SafeTensors  # noqa: E402

# ESM-C's alphabet in id order. 0/1/2/32 are BOS/PAD/EOS/MASK.
VOCAB = ('<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B '
         'U Z O . - | <mask>').split()
TOKEN = {t: i for i, t in enumerate(VOCAB)}
BOS, PAD, EOS, UNK, MASK = 0, 1, 2, TOKEN['<unk>'], 32

# Every block tensor that is a matrix, in the order a block uses them. The
# 1-D ones (the four LayerNorms) are deliberately absent: they are 0.04% of the
# tower and the worst thing to group-quantise, which is the same rule
# tools/quantize_af3.py's KEEP_FLOAT32 applies to AF3.
BLOCK_MATRICES = (
    'attn.layernorm_qkv.weight',
    'attn.out_proj.weight',
    'ffn.fc1_weight',
    'ffn.fc2_weight',
)


def sequence_ids(sequence):
    """One-letter sequence -> token ids, with BOS and EOS attached."""
    body = [TOKEN.get(c.upper(), UNK) for c in sequence]
    return np.array([BOS] + body + [EOS], np.int64)


class Checkpoint:
    """The raw ESM-C safetensors, read lazily and handed out as numpy."""

    def __init__(self, path):
        self.file = SafeTensors(pathlib.Path(path) / 'model.safetensors')
        blocks = {k.split('.')[3] for k in self.file.keys()
                  if k.startswith('esmc.transformer.blocks.')}
        self.n_layers = len(blocks)
        self.d_model = self.file.shape('esmc.embed.weight')[1]
        self.vocab = self.file.shape('esmc.embed.weight')[0]
        self.n_heads = self.d_model // 64      # RotaryEmbedding(d_model // n_heads)
        self.residual_scale = math.sqrt(self.n_layers / 36.0)

    def block_key(self, layer, leaf):
        return 'esmc.transformer.blocks.%d.%s' % (layer, leaf)

    def matrices(self):
        """-> [(key, shape)] for every tensor a quantisation scheme would touch."""
        out = []
        for layer in range(self.n_layers):
            for leaf in BLOCK_MATRICES:
                key = self.block_key(layer, leaf)
                out.append((key, self.file.shape(key)))
        return out

    def __getitem__(self, key):
        return np.asarray(self.file[key], np.float32)

    def close(self):
        self.file.close()


def _layer_norm(x, scale, offset=None, eps=1e-5):
    y = torch.nn.functional.layer_norm(x, (x.shape[-1],), eps=eps)
    y = y * scale
    return y if offset is None else y + offset


def _rope(x, positions, base=10000.0):
    """x is (L, heads, head_dim); split-halves rotation, as ESM-C applies it."""
    d = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32,
                                       device=x.device) / d))
    angle = positions[:, None] * inv[None, :]
    cos, sin = torch.cos(angle)[:, None], torch.sin(angle)[:, None]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Tower:
    """ESM-C's 36 blocks, with each block's weights supplied by a callback.

    The callback is what a quantisation arm replaces: `weights(key)` returns the
    matrix that arm would have the GPU read, so an arm and the reference differ
    in exactly one place and nothing else.
    """

    def __init__(self, checkpoint, weights=None, device='cpu', dtype=torch.float32):
        self.ck = checkpoint
        self.device = device
        self.dtype = dtype
        self._supply = weights or (lambda key: checkpoint[key])
        self._cache = {}

    def _t(self, key, quantisable=True):
        if key not in self._cache:
            array = self._supply(key) if quantisable else self.ck[key]
            self._cache[key] = torch.as_tensor(np.array(array, copy=False, order='C'),
                                               dtype=self.dtype,
                                               device=self.device)
        return self._cache[key]

    def block(self, x, layer, positions, record=None):
        """One ESM-C block. `record` collects each matmul's INPUT, which is
        what a calibrated quantiser needs and what nothing else here wants.

        Kept as ONE implementation rather than a second copy inside the
        calibration script: a calibrated arm and its reference must differ in
        the weights and in nothing else, and two block bodies is exactly how
        they stop doing that.
        """
        ck = self.ck
        b = lambda leaf, q=True: self._t(ck.block_key(layer, leaf), q)
        heads, head_dim = ck.n_heads, ck.d_model // ck.n_heads
        h = _layer_norm(x, b('attn.layernorm_qkv.layer_norm_weight', False),
                        b('attn.layernorm_qkv.layer_norm_bias', False))
        if record is not None:
            record['attn.layernorm_qkv.weight'] = h
        h = h @ b('attn.layernorm_qkv.weight').T
        q, k, v = h.split(ck.d_model, dim=-1)
        q = _layer_norm(q, b('attn.q_ln.weight', False))
        k = _layer_norm(k, b('attn.k_ln.weight', False))
        q = _rope(q.reshape(*q.shape[:-1], heads, head_dim), positions)
        k = _rope(k.reshape(*k.shape[:-1], heads, head_dim), positions)
        v = v.reshape(*v.shape[:-1], heads, head_dim)
        context = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(-3, -2), k.transpose(-3, -2), v.transpose(-3, -2))
        context = context.transpose(-3, -2).reshape(*x.shape[:-1], ck.d_model)
        if record is not None:
            record['attn.out_proj.weight'] = context
        x = x + (context @ b('attn.out_proj.weight').T) / ck.residual_scale
        h = _layer_norm(x, b('ffn.layer_norm_weight', False),
                        b('ffn.layer_norm_bias', False))
        if record is not None:
            record['ffn.fc1_weight'] = h
        h = h @ b('ffn.fc1_weight').T
        gate, value = h.split(h.shape[-1] // 2, dim=-1)
        hidden = torch.nn.functional.silu(gate) * value
        if record is not None:
            record['ffn.fc2_weight'] = hidden
        return x + (hidden @ b('ffn.fc2_weight').T) / ck.residual_scale

    def embed(self, ids):
        table = self._t('esmc.embed.weight', quantisable=False)
        return table[torch.as_tensor(ids, dtype=torch.long, device=self.device)]

    def final_norm(self, x):
        return _layer_norm(x, self._t('esmc.transformer.norm.weight', False))

    def positions(self, length):
        return torch.arange(length, dtype=torch.float32, device=self.device)

    def hidden_states(self, ids):
        """-> (n_layers + 1, L, d_model), every state ESMFold2 is allowed to mix."""
        x = self.embed(ids)
        positions = self.positions(x.shape[-2])
        states = [x]
        for layer in range(self.ck.n_layers):
            x = self.block(x, layer, positions)
            states.append(x)
        states[-1] = self.final_norm(states[-1])
        return torch.stack(states)

    def logits(self, ids):
        """The masked-LM head, which is what makes a forward pass falsifiable."""
        x = self.hidden_states(ids)[-1]
        w0 = self._t('lm_head.0.weight', False)
        x = torch.nn.functional.gelu(x @ w0.T + self._t('lm_head.0.bias', False))
        x = _layer_norm(x, self._t('lm_head.2.weight', False),
                        self._t('lm_head.2.bias', False))
        return x @ self._t('lm_head.3.weight', False).T + self._t('lm_head.3.bias', False)


class Shim:
    """ESMFold2's LanguageModelShim: 37 hidden states -> its pair representation.

    🔴 THE LAYER MIX IS A CONSTANT, SO THE 37 STATES NEVER HAVE TO EXIST AT
    ONCE. `single` is sum_k combine[k] * LN(h_k) @ projection, and both the norm
    and the projection are shared across k, so a running accumulator of
    (tokens, 256) replaces a (37, tokens, 1152) tensor - 4.5x smaller at any
    length, and it means the tower can be streamed a block at a time on a device
    that could not hold its states. `accumulate` is that form; `single` is the
    same arithmetic done all at once, kept because the two agreeing is what says
    the streaming form is right.
    """

    def __init__(self, path, device='cpu'):
        f = SafeTensors(pathlib.Path(path) / 'model.safetensors')
        g = lambda leaf: torch.as_tensor(
            np.array(f['language_model.' + leaf], np.float32), device=device)
        self.combine = torch.softmax(g('base_z_combine'), 0)
        self.norm_scale = g('base_z_linear.0.weight')
        self.norm_offset = g('base_z_linear.0.bias')
        self.projection = g('base_z_linear.1.weight')          # (256, 1152)
        self.down_w = g('base_z_mlp.0.downproject.weight')     # (256, 256)
        self.down_b = g('base_z_mlp.0.downproject.bias')
        self.mlp1_w = g('base_z_mlp.0.output_mlp.0.weight')    # (256, 512)
        self.mlp1_b = g('base_z_mlp.0.output_mlp.0.bias')
        self.mlp2_w = g('base_z_mlp.0.output_mlp.2.weight')    # (256, 256)
        self.mlp2_b = g('base_z_mlp.0.output_mlp.2.bias')
        self.out_scale = g('base_z_mlp.1.weight')
        self.out_offset = g('base_z_mlp.1.bias')
        f.close()

    def single(self, states):
        """(37, L, d_model) -> (L, 256), the mixed and downprojected single."""
        x = _layer_norm(states, self.norm_scale, self.norm_offset)
        x = x @ self.projection.T
        x = torch.einsum('k,kld->ld', self.combine.to(x.dtype), x)
        return x @ self.down_w.T + self.down_b

    def accumulate(self, state, layer, into=None):
        """One state's contribution to the mix, so the tower can be streamed.

        `finish` applies the downprojection once at the end; it is outside the
        sum because it is affine and its bias must not be added 37 times.
        """
        x = _layer_norm(state, self.norm_scale, self.norm_offset)
        term = self.combine[layer].to(state.dtype) * (x @ self.projection.T)
        return term if into is None else into + term

    def finish(self, mixed):
        return mixed @ self.down_w.T + self.down_b

    def pair(self, single):
        """(L, 256) -> (L, L, 256). The outer product carries a product AND a
        difference, so the pair representation sees magnitude and direction.

        🔴 THE DOWNPROJECTION IS ON THE SINGLE, NOT ON THE PAIR. Both are
        256-wide and only the concatenation's 512 says which way round they go -
        `output_mlp.0` is (256, 512) and `downproject` is (256, 256), so a
        reading that puts downproject after the outer product does not typecheck
        and a reading that puts output_mlp.0 there would.
        """
        z = torch.cat([single[:, None] * single[None, :],
                       single[:, None] - single[None, :]], -1)
        z = torch.nn.functional.gelu(z @ self.mlp1_w.T + self.mlp1_b)
        z = z @ self.mlp2_w.T + self.mlp2_b
        return _layer_norm(z, self.out_scale, self.out_offset)


def relative_rms(candidate, reference):
    candidate = candidate.to(torch.float64)
    reference = reference.to(torch.float64)
    return float(torch.sqrt(((candidate - reference) ** 2).sum()
                            / (reference ** 2).sum()))
