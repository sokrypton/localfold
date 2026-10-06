#!/usr/bin/env python3
"""ESM2 3B (the tower Chai-1's tokens read) as a bundle the native port reads, its matrices kept int8 - from
af3-any-model's own quantisation (lm/esm2.bin.zst), not a second one.

  python3 tools/export_esm2_3b.py --tower <af3-any-model's esm2.unpacked dir> --out model-esm2-3b-int8
  ~/venv_ef2/bin/python tools/export_esm2_3b.py --hf <facebook/esm2_t36_3B_UR50D dir> --out model-esm2-3b-f32 \
      --check model-esm2-3b-int8
  python3 tools/quantize_af3.py --source model-esm2-3b-f32 --out model-esm2-3b-int3 --bits 3 --group 128

--hf writes the same tensors, every one float32, from the ORIGINAL weights (Hugging Face's EsmForMaskedLM
checkpoint) - the quantiser's source, for an int3 bundle at ESM-C's group 128 rather than one requantised from
af3-any-model's int8. --check holds each matrix to the int8 bundle's dequantised values, so a wrong mapping
fails here rather than as a quietly worse language model.

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


def from_hf(hf_dir, out, check):
    """The original checkpoint as a float32 bundle in this tool's layout (see the header)."""
    import torch
    state = {}
    for part in sorted(pathlib.Path(hf_dir).glob('pytorch_model-*.bin')):
        state.update(torch.load(part, map_location='cpu', weights_only=True))
    get = lambda k: state[k].float().numpy()
    layers = 1 + max(int(k.split('.')[3]) for k in state if k.startswith('esm.encoder.layer.'))
    out_dir = ROOT / out
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(out_dir.glob('weights-*.bin')):
        stale.unlink()
    writer = MixedShards(out_dir)
    tensors = {'embed/weights': get('esm.embeddings.word_embeddings.weight'),
               'final_norm/scale': get('esm.encoder.emb_layer_norm_after.weight'),
               'final_norm/offset': get('esm.encoder.emb_layer_norm_after.bias')}
    for layer in range(layers):
        h, at = 'esm.encoder.layer.%d.' % layer, 'blocks/%d/' % layer
        tensors[at + 'attn_norm/scale'] = get(h + 'attention.LayerNorm.weight')
        tensors[at + 'attn_norm/offset'] = get(h + 'attention.LayerNorm.bias')
        tensors[at + 'ffn_norm/scale'] = get(h + 'LayerNorm.weight')
        tensors[at + 'ffn_norm/offset'] = get(h + 'LayerNorm.bias')
        qkv = ['attention.self.%s' % m for m in ('query', 'key', 'value')]       # torch Linear weights are [out, in]
        tensors[at + 'qkv/weightsT'] = np.concatenate([get(h + m + '.weight') for m in qkv])
        tensors[at + 'qkv/bias'] = np.concatenate([get(h + m + '.bias') for m in qkv])
        for ours, theirs in (('attn_out', 'attention.output.dense'), ('fc1', 'intermediate.dense'), ('fc2', 'output.dense')):
            tensors[at + ours + '/weightsT'] = get(h + theirs + '.weight')
            tensors[at + ours + '/bias'] = get(h + theirs + '.bias')
    if check:
        reference = json.loads((ROOT / check / 'manifest.json').read_text())['tensors']
        worst, worst_name = 0.0, None
        for name, values in tensors.items():
            r = reference[name]
            if list(values.shape) != r['shape']:
                raise SystemExit('%s: %s here, %s in %s' % (name, list(values.shape), r['shape'], check))
            raw = (ROOT / check / r['file']).read_bytes()
            n = int(np.prod(r['shape']))
            if r['dtype'] == 'float32':
                theirs = np.frombuffer(raw, '<f4', n, r['byteOffset'])
            else:
                codes = np.frombuffer(raw, np.int8, n, r['byteOffset']).astype(np.float32)
                scale = np.frombuffer(raw, '<f2', n // r['block'], r['scaleOffset']).astype(np.float32)
                theirs = codes * np.repeat(scale, r['block'])
            rel = float(np.sqrt(np.mean((values.ravel() - theirs) ** 2) / np.mean(theirs ** 2)))
            if rel > worst:
                worst, worst_name = rel, name
            # (int8 with one scale a ROW is coarse on fc2's 10240-wide rows: 2.8e-2 there; a wrong tensor is ~1.4)
            if rel > 1e-1:
                raise SystemExit('%s: relRMS %.2e against %s - not the same tensor' % (name, rel, check))
        print('every tensor within relRMS %.2e of %s (its int8 rounding; worst %s)' % (worst, check, worst_name))
    for name, values in tensors.items():
        writer.float32(name, values)
    writer.close()
    width = tensors['embed/weights'].shape[1]
    manifest = {
        'formatVersion': 1,
        'source': 'ESM2 3B from facebook/esm2_t36_3B_UR50D (float32), the quantiser\'s source',
        'model': {'name': 'esm2', 'recycles': 0},
        'bundle': {'purpose': 'native-inference', 'model': 'esm2', 'encoding': 'float32'},
        'languageModel': {'tower': 'esm2-3b', 'layers': int(layers), 'width': int(width), 'heads': int(width // 64),
                          'ffn': int(tensors['blocks/0/fc1/weightsT'].shape[0]), 'residualScale': 1.0,
                          'embedScale': 1.0 - 0.15 * 0.8, 'transposedMatrices': 1},
        'weightLayout': 'blocks/*/*/weightsT are [out, in]; qkv rows are q | k | v',
        # 🔴 THE TOKEN EMBEDDING STAYS float32 (338 KB), as af3-any-model's int8 keeps it: at int3 group 128 its
        # rare rows go first, and X - every modified residue in Chai-1's ESM2 sequence - took a phosphoserine
        # job's pLDDT 82.5 -> 77.4 and pTM 0.75 -> 0.65 with the structure unmoved
        'float32Tensors': ['embed/weights'],
        'tensors': writer.records,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('%s: %d layers x %d, %d tensors' % (out_dir.name, layers, width, len(writer.records)))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tower', help="af3-any-model's esm2.unpacked directory (its .npy cache)")
    parser.add_argument('--hf', help="facebook/esm2_t36_3B_UR50D's directory: a float32 bundle from it")
    parser.add_argument('--check', help='an int8 bundle each --hf tensor must agree with')
    parser.add_argument('--out', default='model-esm2-3b-int8')
    arguments = parser.parse_args()
    if arguments.hf:
        return from_hf(arguments.hf, arguments.out, arguments.check)
    if not arguments.tower:
        parser.error('--tower or --hf')
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
