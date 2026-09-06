"""ESMFold2's folding trunk, in the layout LocalFold's AF3 pairformer wants.

ESMFold2's trunk block is AF3's pairformer block with the two grid attentions
and the single track removed, so it needs no new kernels - only its weights in
AF3's shapes, with the attention zeroed. ../alphafold3 measured that identity at
corr 1.00000000, relerr 4.5e-07.

🔴 THE TRIANGLE'S DOUBLE WIDTH IS INTERLEAVED AND THE TRANSITION'S IS BLOCKED,
IN THE SAME BLOCK. src/af3/pairformer-reference.js spells both out because
getting either backwards costs no error and no NaN - it permutes channels and
returns a plausible tensor. ESMFold2 stores BOTH blocked:

    proj_bundle  (4h, c) -> [signal(2h) | gates(2h)], each -> (left h | right h)
    ffn.w12      (2h, c) -> [gate(h) | value(h)]

so the triangle's two halves have to be woven - channel `ch` lands at outputs
`ch * 2` and `ch * 2 + 1` - and the transition's must be left alone.

🔴 AND TORCH STORES (out, in) WHILE THIS GRAPH READS (in, out). `linear` in the
reference takes `transposed = false` by default and indexes `weights[c * out]`.
Every projection below is transposed on the way through; the norms are vectors
and are not.
"""
from __future__ import annotations

import numpy as np

# One tri-mul's tensors, and one transition's, as the reference names them.
TRIANGLE = {
    'leftNormInputScale': ('_engine.norm_start.weight', 'vector'),
    'leftNormInputOffset': ('_engine.norm_start.bias', 'vector'),
    'centerNormScale': ('_engine.norm_mix.weight', 'vector'),
    'centerNormOffset': ('_engine.norm_mix.bias', 'vector'),
    'outputProjection': ('_engine.proj_emit.weight', 'transpose'),
    'gatingLinear': ('_engine.proj_gate.weight', 'transpose'),
}


def interleave(left, right):
    """(in, c) and (in, c) -> (in, 2c) with channel ch at 2*ch and 2*ch + 1."""
    if left.shape != right.shape:
        raise ValueError('%s against %s' % (left.shape, right.shape))
    inner, channels = left.shape
    out = np.empty((inner, channels * 2), np.float32)
    out[:, 0::2] = left
    out[:, 1::2] = right
    return out


def triangle_weights(get, prefix, channels, swap_pair=False):
    """One tri_mul_out/tri_mul_in as AF3's triangleMultiplication wants it.

    🔴 THE INCOMING DIRECTION WANTS ITS TWO HALVES SWAPPED AND THE OUTGOING ONE
    DOES NOT. ESMFold2 runs both directions through ONE engine and distinguishes
    them by which half of proj_bundle is the left operand, where AF3 keeps the
    halves fixed and changes the einsum. So a converter that treats the two
    modules alike - which is what reading the shapes suggests, since they are
    identical - gets the incoming one exactly backwards. Swept, not read:
    tools/check-esmfold2-modules.js scores in-order at 3.24e-1 and swapped at
    2.86e-7 for tri_mul_in, and the reverse for tri_mul_out.
    """
    out = {}
    for name, (leaf, kind) in TRIANGLE.items():
        values = np.asarray(get('%s.%s' % (prefix, leaf)), np.float32)
        out[name] = values if kind == 'vector' else np.ascontiguousarray(values.T)

    bundle = np.asarray(get('%s._engine.proj_bundle.weight' % prefix), np.float32)
    if bundle.shape[0] != 4 * channels:
        raise ValueError('proj_bundle is %s; expected (%d, %d)'
                         % (bundle.shape, 4 * channels, channels))
    bundle = np.ascontiguousarray(bundle.T)          # (c, 4h)
    signal, gates = bundle[:, :2 * channels], bundle[:, 2 * channels:]
    first, second = (channels, 0) if swap_pair else (0, channels)
    halves = lambda w: (w[:, first:first + channels], w[:, second:second + channels])
    out['projection'] = interleave(*halves(signal))
    out['gate'] = interleave(*halves(gates))
    return out


def transition_weights(get, prefix):
    """pair_transition as AF3's transition wants it: BLOCKED, gate half first."""
    w12 = np.ascontiguousarray(np.asarray(get('%s.ffn.w12.weight' % prefix), np.float32).T)
    w3 = np.ascontiguousarray(np.asarray(get('%s.ffn.w3.weight' % prefix), np.float32).T)
    return {
        'inputLayerNormScale': np.asarray(get('%s.norm.weight' % prefix), np.float32),
        'inputLayerNormOffset': np.asarray(get('%s.norm.bias' % prefix), np.float32),
        'transition1': w12,
        'transition2': w3,
    }


def zero_attention(channels, heads):
    """A grid attention that contributes nothing, and a single track that is absent.

    🔴 ZEROING RATHER THAN BRANCHING IS WHAT MAKES THIS NEED NO GRAPH CODE. An
    attention whose OUTPUT projection is zero adds zero, so `pair += attention`
    is the identity and the block becomes ESMFold2's. It is also pure waste on
    the GPU - grid.attend is the largest kernel in an AF3 trunk - so the shipped
    path should skip the pass and be checked bit-identical against this.
    """
    value = channels // heads
    zeros = lambda *shape: np.zeros(shape, np.float32)
    return {
        'queryNormScale': np.ones(channels, np.float32),
        'queryNormOffset': zeros(channels),
        'queryProjection': zeros(channels, channels),
        'keyProjection': zeros(channels, channels),
        'valueProjection': zeros(channels, channels),
        'gatingProjection': zeros(channels, channels),
        'outputProjection': zeros(channels, channels),
        'bias': zeros(heads, 1),
        'heads': heads,
        'valueChannels': value,
    }


def trunk_block(get, layer, channels, heads, prefix='folding_trunk'):
    """One ESMFold2 trunk block as an AF3 pairformer block with no attention."""
    at = '%s.blocks.%d' % (prefix, layer)
    return {
        'pairChannels': channels,
        'singleChannels': 0,
        'triangleMultiplicationOutgoing':
            triangle_weights(get, '%s.tri_mul_out' % at, channels),
        'triangleMultiplicationIncoming':
            triangle_weights(get, '%s.tri_mul_in' % at, channels, swap_pair=True),
        'pairTransition': transition_weights(get, '%s.pair_transition' % at),
        'pairAttention': zero_attention(channels, heads),
    }
