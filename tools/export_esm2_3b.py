#!/usr/bin/env python3
"""ESM2 3B (the tower Chai-1's tokens read) as a bundle the native port reads, its matrices kept int8 - from
af3-any-model's own quantisation (lm/esm2.bin.zst), not a second one.

  python3 tools/export_esm2_3b.py --tower <af3-any-model's esm2.unpacked dir> --out model-esm2-3b-int8

36 blocks x 2560, 40 heads of 64, FFN 10240; 2.8 B parameters, 2.7 GB as int8 codes. As tools/export_esmc6b.py
does for ESM-C 6B, each matrix [in, out] with a float32 scale per OUTPUT channel (esm.py `_deq`) is written
TRANSPOSED - [out, in], one float16 scale a row - so the codes are af3-any-model's byte for byte. q, k and v are
three matrices there and ONE here, stacked by rows (`blocks/<n>/qkv/weightsT` [3C, C]): a row is an output
channel and keeps its own scale, so the stack is exact. Everything else is float32: the embedding, the norms
and every bias (q|k|v concatenated as the rows are).

ESM2 against ESM-C, which the native tower code already runs (alphafold3/model/esm.py): q/k/v biased and no
q/k norms; a GELU feed-forward with biases (fc1 is not [gate | value]); no residual scale; the embedding
scaled by 1 - 0.15 * 0.8 (token dropout's inference constant); and only the LAST state read, after the final
LayerNorm (which has an offset).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
from export_esmc6b import MixedShards                       # noqa: E402

VECTORS = ('attn_norm/scale', 'attn_norm/offset', 'ffn_norm/scale', 'ffn_norm/offset')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tower', required=True, help="af3-any-model's esm2.unpacked directory (its .npy cache)")
    parser.add_argument('--out', default='model-esm2-3b-int8')
    arguments = parser.parse_args()
    tower = pathlib.Path(arguments.tower)
    load = lambda key: np.load(tower / (key.replace('/', '__') + '.npy'), mmap_mode='r')
    out_dir = ROOT / arguments.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(out_dir.glob('weights-*.bin')):
        stale.unlink()
    writer = MixedShards(out_dir)

    embed = load('embed/weights')
    layers, width = load('blocks/q/weights').shape[0], embed.shape[1]
    writer.float32('embed/weights', embed)
    writer.float32('final_norm/scale', load('final_norm/scale'))
    writer.float32('final_norm/offset', load('final_norm/offset'))
    worst = 0.0

    def matrix(name, codes_in_out, scale):
        nonlocal worst
        scale = np.asarray(scale, np.float32)
        half = scale.astype(np.float16)
        worst = max(worst, float(np.max(np.abs(half.astype(np.float32) / scale - 1))))
        writer.int8_rows(name, np.asarray(codes_in_out).T, half)

    for layer in range(layers):
        at = 'blocks/%d/' % layer
        for leaf in VECTORS:
            writer.float32(at + leaf, load('blocks/' + leaf)[layer])
        qkv = np.concatenate([np.asarray(load('blocks/%s/weights' % m)[layer]) for m in 'qkv'], axis=1)   # [in, 3C]
        matrix(at + 'qkv/weightsT', qkv,
               np.concatenate([load('blocks/%s/weights__q_scale' % m)[layer] for m in 'qkv']))
        writer.float32(at + 'qkv/bias', np.concatenate([load('blocks/%s/bias' % m)[layer] for m in 'qkv']))
        for m in ('attn_out', 'fc1', 'fc2'):
            matrix(at + m + '/weightsT', load('blocks/%s/weights' % m)[layer], load('blocks/%s/weights__q_scale' % m)[layer])
            writer.float32(at + m + '/bias', load('blocks/%s/bias' % m)[layer])
        if layer % 10 == 0:
            print('  layer %d' % layer, flush=True)
    writer.close()

    ffn = int(load('blocks/fc1/weights').shape[2])
    manifest = {
        'formatVersion': 1,
        'source': 'ESM2 3B from af3-any-model esm2.bin.zst (int8, per-output-channel scales)',
        'model': {'name': 'esm2', 'recycles': 0},
        'bundle': {'purpose': 'native-inference', 'model': 'esm2', 'encoding': 'mixed'},
        'languageModel': {'tower': 'esm2-3b', 'layers': int(layers), 'width': int(width), 'heads': int(width // 64),
                          'ffn': ffn, 'residualScale': 1.0, 'embedScale': 1.0 - 0.15 * 0.8,
                          'transposedMatrices': 1},
        'weightLayout': 'blocks/*/*/weightsT are [out, in]; qkv rows are q | k | v',
        'tensors': writer.records,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    size = sum(p.stat().st_size for p in out_dir.glob('weights-*.bin'))
    print('%s: %d layers x %d, %d tensors, %.2f GB; worst float16 scale rounding %.1e'
          % (out_dir.name, layers, width, len(writer.records), size / 1e9, worst))
    return 0


if __name__ == '__main__':
    sys.exit(main())
