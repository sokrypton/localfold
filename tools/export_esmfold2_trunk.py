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
