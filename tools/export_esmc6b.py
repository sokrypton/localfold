#!/usr/bin/env python3
"""ESM-C 6B (the tower of the released ESMFold2 and ESMFold2-Fast) as a bundle the native port reads, with the
tower's matrices kept int8 - from af3-any-model's own quantisation, not a second one.

  python3 tools/export_esmc6b.py --tower <esmc.unpacked dir> --esmfold2 <dir with ESMFold2's model.safetensors>
                                 --out model-esmc-6b-int8

The tower is 6.35 B parameters: 25.4 GB as float32 and 12.7 GB as float16, which no T4 holds and which
`tools/export_esmc_model.py` would write in full from biohub/ESMC-6B's 26 GB. af3-any-model publishes it int8
(`lm/esmc.bin.zst`, 5.5 GB; its loader leaves an unpacked .npy cache, `esmc.unpacked/`): each block matrix
[in, out] as int8 codes with one float32 scale per OUTPUT channel (alphafold3/model/esm.py `load`, `_deq`). The
page's int8 codec is a float16 scale per contiguous block of elements, so each matrix is written TRANSPOSED -
[out, in], `block` = in - which makes that per-output-channel scale exactly a per-block one: the codes are
af3-any-model's byte for byte, the scales rounded once to float16. The names carry the transpose
(`blocks/<n>/<matrix>/weightsT`), and the native port keeps them resident as codes (MODEL resident int8) and
expands one layer at a time to float16 for its GEMMs (native/ef2/src/esmc.cuh).

Everything else is float32: the embedding, the norms, and ESMFold2's shim (`language_model.*`, from the folding
checkpoint - a shim is per model, the tower is shared). The residual scale is the architecture's: every branch
divided by sqrt(n_layers / 36) (esm.py `_dims_from`), 1.4907 at 80 layers.
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
from export_esmc_model import SHIM_TENSORS                  # noqa: E402

SHARD_LIMIT = 512 << 20
MATRICES = ('qkv', 'attn_out', 'fc1', 'fc2')
VECTORS = ('attn_norm/scale', 'attn_norm/offset', 'q_norm/scale', 'k_norm/scale', 'ffn_norm/scale', 'ffn_norm/offset')


class MixedShards:
    """float32 and int8 (codes, then a float16 scale per block) records in shards of at most SHARD_LIMIT."""

    def __init__(self, out_dir):
        self.out_dir, self.records, self.index, self.handle, self.offset = out_dir, {}, 0, None, 0

    def _file(self):
        return 'weights-%03d.bin' % self.index

    def _room(self, nbytes):
        if self.handle is None or (self.offset + nbytes > SHARD_LIMIT and self.offset > 0):
            if self.handle is not None:
                self.handle.close()
                self.index += 1
            self.handle, self.offset = (self.out_dir / self._file()).open('wb'), 0

    def _write(self, data):
        pad = (-self.offset) % 16
        if pad:
            self.handle.write(b'\0' * pad)
            self.offset += pad
        at = self.offset
        self.handle.write(data)
        self.offset += len(data)
        return at

    def float32(self, name, values):
        values = np.ascontiguousarray(values, '<f4')
        self._room(values.nbytes + 16)
        at = self._write(values.tobytes())
        self.records[name] = {'file': self._file(), 'shape': list(values.shape), 'byteOffset': at, 'dtype': 'float32'}

    def int8_rows(self, name, codes, scales):
        """codes [rows, cols] int8, scales [rows]: one float16 scale per row (block = cols)."""
        codes = np.ascontiguousarray(codes, np.int8)
        half = np.ascontiguousarray(scales, '<f2')
        self._room(codes.nbytes + half.nbytes + 32)
        at = self._write(codes.tobytes())
        scale_at = self._write(half.tobytes())
        self.records[name] = {'file': self._file(), 'shape': list(codes.shape), 'byteOffset': at, 'dtype': 'int8',
                              'block': int(codes.shape[1]), 'scaleOffset': scale_at}

    def close(self):
        if self.handle is not None:
            self.handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tower', required=True, help="af3-any-model's esmc.unpacked directory (its .npy cache)")
    parser.add_argument('--esmfold2', required=True, help="the folding checkpoint's directory, for its shim")
    parser.add_argument('--out', default='model-esmc-6b-int8')
    arguments = parser.parse_args()
    tower = pathlib.Path(arguments.tower)
    load = lambda key: np.load(tower / (key.replace('/', '__') + '.npy'), mmap_mode='r')
    out_dir = ROOT / arguments.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(out_dir.glob('weights-*.bin')):
        stale.unlink()
    writer = MixedShards(out_dir)

    embed = load('embed/weights')
    layers, width = load('blocks/qkv/weights').shape[0], embed.shape[1]
    writer.float32('embed/weights', embed)
    writer.float32('final_norm/scale', load('final_norm/scale'))
    worst = 0.0
    for layer in range(layers):
        for leaf in VECTORS:
            writer.float32('blocks/%d/%s' % (layer, leaf), load('blocks/' + leaf)[layer])
        for matrix in MATRICES:
            codes = np.asarray(load('blocks/%s/weights' % matrix)[layer])                  # [in, out]
            scale = np.asarray(load('blocks/%s/weights__q_scale' % matrix)[layer], np.float32)   # [out]
            half = scale.astype(np.float16)
            worst = max(worst, float(np.max(np.abs(half.astype(np.float32) / scale - 1))))
            writer.int8_rows('blocks/%d/%s/weightsT' % (layer, matrix), codes.T, half)
        if layer % 10 == 0:
            print('  layer %d' % layer, flush=True)
    fold = SafeTensors(pathlib.Path(arguments.esmfold2) / 'model.safetensors')
    for leaf, name in SHIM_TENSORS:
        values = np.asarray(fold['language_model.' + leaf], np.float32)
        if name in ('lm/projection/weights', 'lm/downproject/weights', 'lm/pair_mlp_1/weights', 'lm/pair_mlp_2/weights'):
            values = np.ascontiguousarray(values.T)              # (inner-major, as export_esmc_model.py writes them)
        writer.float32(name, values)
    fold.close()
    writer.close()

    residual_scale = float(np.sqrt(layers / 36.0))
    mix = int(np.asarray(writer.records['lm/combine']['shape'])[0])
    if mix != layers + 1:
        raise SystemExit('the shim mixes %d states and the tower has %d layers: a shim is per tower' % (mix, layers))
    manifest = {
        'formatVersion': 1,
        'source': 'ESM-C tower from af3-any-model esmc.bin.zst (int8, per-output-channel scales), ESMFold2 shim from %s'
                  % pathlib.Path(arguments.esmfold2).name,
        'model': {'name': 'esmc', 'recycles': 0},
        'bundle': {'purpose': 'native-inference', 'model': 'esmc', 'encoding': 'mixed'},
        'languageModel': {'tower': 'esmc-6b', 'shim': pathlib.Path(arguments.esmfold2).name,
                          'layers': int(layers), 'width': int(width), 'heads': int(width // 64),
                          'residualScale': residual_scale, 'mixEntries': mix,
                          'transposedMatrices': 1},
        'weightLayout': 'inner-major; blocks/*/*/weightsT are [out, in]',
        'tensors': writer.records,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    size = sum(p.stat().st_size for p in out_dir.glob('weights-*.bin'))
    print('%s: %d layers x %d, %d tensors, %.2f GB; worst float16 scale rounding %.1e'
          % (out_dir.name, layers, width, len(writer.records), size / 1e9, worst))
    return 0


if __name__ == '__main__':
    sys.exit(main())
