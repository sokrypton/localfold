"""ESMFold2's folding trunk -> a LocalFold bundle, in AF3's pairformer layout.

    python3 tools/export_esmfold2_trunk.py --esmfold2 esmfold2-fast-600m \
        --out model-esmfold2-trunk-f32

The trunk needs no new kernels: it is AF3's pairformer block with the grid
attention and the single track removed, and ../alphafold3 measured an AF3
PairFormerIteration with zeroed attention against ESMFold2's pair-only block at
corr 1.00000000, relerr 4.5e-07. What it needs is its weights in AF3's shapes -
see tools/esmc/esmfold2_trunk_weights.py for the two double-width conventions
that differ inside the same block.

🔴 THE ZEROED ATTENTION IS NOT EXPORTED. It is 5 matrices of c x c a block, 24
blocks, 37.7 MiB of zeros - and a bundle that ships them would be paying for a
pass the shipped path should skip. The checker synthesises them; a bundle that
needed them would be admitting the graph has not been told.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'esmc'))

from safetensors_read import SafeTensors                    # noqa: E402
from esmfold2_trunk_weights import trunk_block              # noqa: E402
from export_esmc_model import ShardWriter                   # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esmfold2', default='esmfold2-fast-600m')
    parser.add_argument('--out', default='model-esmfold2-trunk-f32')
    parser.add_argument('--heads', type=int, default=8,
                        help="the zeroed attention's head count, from the config")
    arguments = parser.parse_args()

    source = SafeTensors(ROOT / arguments.esmfold2 / 'model.safetensors')
    get = lambda name: source[name]
    layers = len({k.split('.')[2] for k in source.keys()
                  if k.startswith('folding_trunk.blocks.')})
    channels = source.shape('folding_trunk.blocks.0.pair_transition.norm.weight')[0]

    out_dir = ROOT / arguments.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob('*.f32.bin'):
        stale.unlink()
    writer = ShardWriter(out_dir)

    for layer in range(layers):
        block = trunk_block(get, layer, channels, arguments.heads)
        for group, values in block.items():
            if not isinstance(values, dict):
                continue
            if group == 'pairAttention':
                continue                      # synthesised, never shipped
            for leaf, array in values.items():
                writer.add('blocks/%d/%s/%s' % (layer, group, leaf), array)
    # The recycle projection, which is the trunk's one divergence from AF3.
    writer.add('recycle/norm/scale', np.asarray(get('pair_loop_proj.0.weight'), np.float32))
    writer.add('recycle/norm/offset', np.asarray(get('pair_loop_proj.0.bias'), np.float32))
    writer.add('recycle/projection', np.ascontiguousarray(
        np.asarray(get('pair_loop_proj.1.weight'), np.float32).T))
    # 🔴 EVERYTHING THAT BUILDS z_init EXCEPT THE ATOM ENCODER AND THE LANGUAGE
    # MODEL. z_init is a sum of five terms and these are the three cheap ones;
    # the shim's pair term is already in the ESM-C bundle, and the atom
    # encoder's is its own port. Exporting them separately is what lets the
    # featuriser be checked before the expensive half of it exists.
    transposed = lambda name: np.ascontiguousarray(
        np.asarray(get(name), np.float32).T)
    writer.add('featuriser/relPos', transposed('rel_pos.embed.weight'))
    writer.add('featuriser/tokenBonds', transposed('token_bonds.weight'))
    writer.add('featuriser/zInit1', transposed('z_init_1.weight'))
    writer.add('featuriser/zInit2', transposed('z_init_2.weight'))

    # The distogram head: two tensors, and the trunk's only output today.
    writer.add('distogram/weights', transposed('distogram_head.weight'))
    writer.add('distogram/bias', np.asarray(get('distogram_head.bias'), np.float32))

    # 🔴 THE INPUTS EMBEDDER, WHICH IS NOT AF3'S ATOM ENCODER. Sliding-window
    # self-attention over atoms with a 3D rotary embedding built from the
    # reference conformer - see src/esmfold2/atom-encoder-reference.js. Reusing
    # AF3's windowed pair-biased encoder would have been the obvious wrong move;
    # nothing in the shapes says they differ.
    atom = 'inputs_embedder.atom_attention_encoder'
    writer.add('atom/linear', transposed('%s.atom_linear.weight' % atom))
    writer.add('atom/norm/scale', np.asarray(get('%s.atom_norm.weight' % atom), np.float32))
    writer.add('atom/norm/offset', np.asarray(get('%s.atom_norm.bias' % atom), np.float32))
    writer.add('atom/toToken', transposed('%s.atom_to_token_linear.weight' % atom))
    atom_blocks = len({k.split('.')[5] for k in source.keys()
                       if k.startswith('%s.atom_transformer.blocks.' % atom)})
    for layer in range(atom_blocks):
        at = '%s.atom_transformer.blocks.%d' % (atom, layer)
        for leaf, name in (('adaln_modulation.1', 'adaln'), ('attn.Wqkv', 'qkv'),
                           ('attn.gate_proj', 'attnGate'), ('attn.out_proj', 'attnOut'),
                           ('ffn.w_up', 'ffnUp'), ('ffn.w_down', 'ffnDown')):
            writer.add('atom/blocks/%d/%s' % (layer, name),
                       transposed('%s.%s.weight' % (at, leaf)))
    writer.close()

    parameters = sum(int(np.prod(r['shape'])) for r in writer.records.values())
    manifest = {
        'formatVersion': 1,
        'source': 'ESMFold2 folding trunk from %s' % arguments.esmfold2,
        'model': {'name': 'esmfold2-trunk', 'recycles': 0},
        'bundle': {'purpose': 'browser-inference', 'model': 'esmfold2-trunk',
                   'encoding': 'float32-le'},
        'trunk': {'source': arguments.esmfold2, 'blocks': layers,
                  'pairChannels': int(channels), 'zeroedAttentionHeads': arguments.heads,
                  # 🔴 THE LAYOUT, IN THE ARTEFACT. A bundle that outlives a
                  # change to its exporter decodes cleanly into the wrong thing -
                  # which is exactly what the stale int5 ESM-C bundle did.
                  'singleInputs': int(source.shape('z_init_1.weight')[1]),
                  'atomChannels': int(source.shape(
                      '%s.atom_linear.weight' % 'inputs_embedder.atom_attention_encoder')[0]),
                  'atomBlocks': atom_blocks,
                  'atomHeads': 4,
                  'atomWindow': 128,
                  'distogramBins': int(source.shape('distogram_head.weight')[0]),
                  'tokenChannels': int(source.shape(
                      '%s.atom_to_token_linear.weight'
                      % 'inputs_embedder.atom_attention_encoder')[0]),
                  'relativeFeatures': int(source.shape('rel_pos.embed.weight')[1]),
                  'weightLayout': 'af3-pairformer-in-out',
                  'triangleDoubleWidth': 'interleaved',
                  'transitionDoubleWidth': 'blocked-gate-first'},
        'tensors': writer.records,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    written = sum(p.stat().st_size for p in out_dir.glob('*.f32.bin'))
    print('%s: %d blocks x %d channels, %d tensors, %.1fM parameters, %.0f MiB'
          % (out_dir.name, layers, channels, len(writer.records),
             parameters / 1e6, written / 2 ** 20))
    source.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
