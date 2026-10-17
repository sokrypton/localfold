"""ESM-C's tower and ESMFold2's shim -> the bundle format the browser reads.

    python3 tools/export_esmc_model.py --esmc esmc-600m \
        --esmfold2 esmfold2-fast-600m --out model-esmc-600m-f32

Writes `manifest.json` plus `weights-NN.f32.bin` shards, which is exactly what
`tools/quantize_af3.py` consumes and `shared/bundles/http-tensor-store.js` fetches
- so the int5 packing, the sharding limits and the shard cache all work on this
without knowing what ESM-C is.

🔴 THE SHIM SHIPS WITH THE TOWER, IN THE SAME BUNDLE, NAMED AFTER THE FOLDING
MODEL IT CAME FROM. Every ESMFold2 release trains its own `language_model.*` and
they share only the tower; ../alphafold3 fed one variant another's shim and read
corr 0.026 against native where the variant's own reads 0.999998 - 8.798 A
against 0.812 on 6MRR. Putting them in one artifact with the folding model's
name recorded in the manifest is what makes that pairing checkable after the
fact rather than assumed.

🔴 AND THE NORMS AND THE EMBEDDING ARE NOT QUANTISABLE, WHICH THE MANIFEST SAYS.
`tools/quantize_af3.py` decides by NAME (`KEEP_FLOAT32`), so the export uses
names that match its rule: anything ending `/scale`, `/offset` or `/bias` stays
float32. The 64 x 1152 embedding is named as a weight and is 0.01% of the tower,
but it is the INPUT to all 36 blocks - the AF3 port measured leaving ESM-C's
embedding as raw int8 codes at corr 0.19 against native, with layer 0 already
at 0.89 - so it is listed explicitly.

🔴 AND lm_head IS LEFT OUT. It is 1.4 M parameters of masked-language-model head
that ESMFold2 never reads; `--include-lm-head` keeps it for anyone checking the
tower against a masked-recovery number rather than against a fold.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'esmc'))

from safetensors_read import SafeTensors          # noqa: E402

SHARD_LIMIT = 48 * 1024 * 1024

# 🔴 THE PROJECTIONS ARE TRANSPOSED ON THE WAY OUT, INTO (inner, columns).
# torch stores a Linear as (out_features, in_features) and computes x @ W.T;
# webgpu/kernels/transition.js's tiled kernel - the one CLAUDE.md measures at
# 1140-1550 GFLOP/s - indexes `weights[k * columns + column]`, which is
# (in_features, out_features). Transposing here is one pass at export time;
# transposing per fold would be 573 M elements a fold on the main thread, and
# writing a second kernel to read the other layout would duplicate the fastest
# dense projection in the repository.
TRANSPOSED = ("qkv/weights", "attn_out/weights", "fc1/weights", "fc2/weights",
              "lm/projection/weights", "lm/downproject/weights",
              "lm/pair_mlp_1/weights", "lm/pair_mlp_2/weights")

# The tower, per block. The name on the right is what the browser asks for; the
# suffix decides whether tools/quantize_af3.py will touch it.
BLOCK_TENSORS = (
    ('attn.layernorm_qkv.layer_norm_weight', 'attn_norm/scale'),
    ('attn.layernorm_qkv.layer_norm_bias', 'attn_norm/offset'),
    ('attn.layernorm_qkv.weight', 'qkv/weights'),
    ('attn.q_ln.weight', 'q_norm/scale'),
    ('attn.k_ln.weight', 'k_norm/scale'),
    ('attn.out_proj.weight', 'attn_out/weights'),
    ('ffn.layer_norm_weight', 'ffn_norm/scale'),
    ('ffn.layer_norm_bias', 'ffn_norm/offset'),
    ('ffn.fc1_weight', 'fc1/weights'),
    ('ffn.fc2_weight', 'fc2/weights'),
)

# ESMFold2's LanguageModelShim, which turns the 37 states into a pair rep.
SHIM_TENSORS = (
    ('base_z_combine', 'lm/combine'),
    ('base_z_linear.0.weight', 'lm/norm/scale'),
    ('base_z_linear.0.bias', 'lm/norm/offset'),
    ('base_z_linear.1.weight', 'lm/projection/weights'),
    ('base_z_mlp.0.downproject.weight', 'lm/downproject/weights'),
    ('base_z_mlp.0.downproject.bias', 'lm/downproject/bias'),
    ('base_z_mlp.0.output_mlp.0.weight', 'lm/pair_mlp_1/weights'),
    ('base_z_mlp.0.output_mlp.0.bias', 'lm/pair_mlp_1/bias'),
    ('base_z_mlp.0.output_mlp.2.weight', 'lm/pair_mlp_2/weights'),
    ('base_z_mlp.0.output_mlp.2.bias', 'lm/pair_mlp_2/bias'),
    ('base_z_mlp.1.weight', 'lm/pair_norm/scale'),
    ('base_z_mlp.1.bias', 'lm/pair_norm/offset'),
)


class PublishedEsmc:
    """biohub's PUBLISHED layout (ESMC-6B, and any `esm` >= 3.4 save) read under the fused names below.

    A sharded checkpoint (`model.safetensors.index.json`) naming each projection on its own - `esmc.layers.N.self_attn.
    q_proj.weight`, `mlp.gate_proj.weight` - where this exporter reads TransformerEngine's fused ones. `esm`'s
    checkpoint_layout.py is the conversion, and it is exact: a fused weight is the ROW-WISE concatenation of its parts,
    q|k|v and gate|up in that order (EsmcLayerNormMLP splits fc1 as `x1, x2 = chunk(2)`, `silu(x1) * x2`).
    """

    ROOT = {'esmc.embed.weight': 'esmc.embed_tokens.weight', 'esmc.transformer.norm.weight': 'esmc.norm.weight'}
    BLOCK = {'attn.layernorm_qkv.layer_norm_weight': 'input_layernorm.weight',
             'attn.layernorm_qkv.layer_norm_bias': 'input_layernorm.bias',
             'attn.q_ln.weight': 'self_attn.q_norm.weight', 'attn.k_ln.weight': 'self_attn.k_norm.weight',
             'attn.out_proj.weight': 'self_attn.o_proj.weight',
             'ffn.layer_norm_weight': 'post_attention_layernorm.weight',
             'ffn.layer_norm_bias': 'post_attention_layernorm.bias', 'ffn.fc2_weight': 'mlp.down_proj.weight'}
    FUSED = {'attn.layernorm_qkv.weight': ('self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight'),
             'ffn.fc1_weight': ('mlp.gate_proj.weight', 'mlp.up_proj.weight')}

    def __init__(self, directory):
        sharded = directory / 'model.safetensors.index.json'
        if sharded.exists():
            index = json.loads(sharded.read_text())['weight_map']
        else:                                   # (one file: ESMC-300M-1500000's, for instance)
            with SafeTensors(directory / 'model.safetensors') as one:
                index = {k: 'model.safetensors' for k in one.keys()}
        self.files = {f: SafeTensors(directory / f) for f in sorted(set(index.values()))}
        self.where = index
        self.layers = len({k.split('.')[2] for k in index if k.startswith('esmc.layers.')})

    def _raw(self, name):
        return self.files[self.where[name]][name]

    def _published(self, name):
        if name in self.ROOT:
            return [self.ROOT[name]]
        parts = name.split('.')
        if name.startswith('esmc.transformer.blocks.'):
            leaf, base = '.'.join(parts[4:]), 'esmc.layers.%s.' % parts[3]
            if leaf in self.FUSED:
                return [base + x for x in self.FUSED[leaf]]
            if leaf in self.BLOCK:
                return [base + self.BLOCK[leaf]]
        raise KeyError(name)

    def keys(self):
        names = list(self.ROOT)
        for layer in range(self.layers):
            names += ['esmc.transformer.blocks.%d.%s' % (layer, leaf) for leaf in list(self.BLOCK) + list(self.FUSED)]
        return names

    def shape(self, name):
        shapes = [self.files[self.where[p]].shape(p) for p in self._published(name)]
        return (sum(x[0] for x in shapes),) + tuple(shapes[0][1:])

    def __getitem__(self, name):
        parts = [self._raw(p) for p in self._published(name)]
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    def close(self):
        for f in self.files.values():
            f.close()


class ShardWriter:
    """Lay tensors into float32 shards, four-byte aligned, none over the limit."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.records = {}
        self.chunks = []
        self.offset = 0
        self.index = 0

    def _flush(self):
        if not self.chunks:
            return
        path = self.out_dir / ('weights-%02d.f32.bin' % self.index)
        with path.open('wb') as handle:
            for chunk in self.chunks:
                handle.write(chunk)
        self.chunks, self.offset, self.index = [], 0, self.index + 1

    def add(self, name, values):
        flat = np.ascontiguousarray(values, dtype='<f4').reshape(-1)
        if self.offset + flat.nbytes > SHARD_LIMIT and self.chunks:
            self._flush()
        self.records[name] = {
            'file': 'weights-%02d.f32.bin' % self.index,
            'shape': list(np.asarray(values).shape),
            'byteOffset': self.offset,
            'dtype': 'float32',
        }
        self.chunks.append(flat.tobytes())
        self.offset += flat.nbytes

    def close(self):
        self._flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esmc', default='esmc-600m')
    parser.add_argument('--esmfold2', default='esmfold2-fast-600m')
    parser.add_argument('--out', default='model-esmc-600m-f32')
    parser.add_argument('--include-lm-head', action='store_true')
    parser.add_argument('--residual-scale', type=float, default=None,
                        help='the blocks\' residual divisor (default sqrt(layers / 36), as the model computes it)')
    arguments = parser.parse_args()

    tower_dir = ROOT / arguments.esmc          # (an absolute path stays itself)
    fold_dir = ROOT / arguments.esmfold2
    out_dir = ROOT / arguments.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob('*.f32.bin'):
        stale.unlink()

    def single_file_published():
        with SafeTensors(tower_dir / 'model.safetensors') as probe:
            return 'esmc.embed_tokens.weight' in probe
    published = (tower_dir / 'model.safetensors.index.json').exists() or single_file_published()
    tower = PublishedEsmc(tower_dir) if published else SafeTensors(tower_dir / 'model.safetensors')
    fold = SafeTensors(fold_dir / 'model.safetensors')
    writer = ShardWriter(out_dir)

    layers = len({k.split('.')[3] for k in tower.keys()
                  if k.startswith('esmc.transformer.blocks.')})
    d_model = tower.shape('esmc.embed.weight')[1]
    # 🔴 ESM-C's HEAD DIMENSION IS 64 AT EVERY WIDTH, which is the only reason
    # the head count is derivable: no tensor's shape carries it, and 1152
    # channels reads just as well as 16 heads of 72.
    heads = d_model // 64
    # 🔴 AND THE RESIDUAL SCALE IS NOT A TENSOR: every ESM-C block divides both
    # branches by sqrt(n_layers / 36) (esm's EsmcTransformerStack, scale_residue
    # always on; tools/esmc/esmc_forward.py derives it the same way) - 1 at 600M
    # (36 layers), 1.4907 at 6B (80). 🔴 This said 1.0 for every checkpoint, so the
    # 300M bundle (30 layers, where the rule gives 0.9129) was exported with 1.0;
    # --residual-scale=1 reproduces that export.
    residual_scale = (arguments.residual_scale if arguments.residual_scale is not None
                      else math.sqrt(layers / 36))

    def add(name, values):
        """Transposed where the GPU wants (inner, columns); verbatim otherwise."""
        values = np.asarray(values, np.float32)
        leaf = name.split('/', 2)[2] if name.startswith('blocks/') else name
        if leaf in TRANSPOSED:
            if values.ndim != 2:
                raise SystemExit('%s is %dd; only a matrix can be transposed'
                                 % (name, values.ndim))
            values = np.ascontiguousarray(values.T)
        writer.add(name, values)

    add('embed/weights', np.asarray(tower['esmc.embed.weight'], np.float32))
    add('final_norm/scale',
        np.asarray(tower['esmc.transformer.norm.weight'], np.float32))
    for layer in range(layers):
        for leaf, name in BLOCK_TENSORS:
            source = 'esmc.transformer.blocks.%d.%s' % (layer, leaf)
            add('blocks/%d/%s' % (layer, name),
                np.asarray(tower[source], np.float32))
    if arguments.include_lm_head:
        for leaf, name in (('lm_head.0.weight', 'lm_head/dense/weights'),
                           ('lm_head.0.bias', 'lm_head/dense/bias'),
                           ('lm_head.2.weight', 'lm_head/norm/scale'),
                           ('lm_head.2.bias', 'lm_head/norm/offset'),
                           ('lm_head.3.weight', 'lm_head/decoder/weights'),
                           ('lm_head.3.bias', 'lm_head/decoder/bias')):
            add(name, np.asarray(tower[leaf], np.float32))

    for leaf, name in SHIM_TENSORS:
        add(name, np.asarray(fold['language_model.' + leaf], np.float32))
    writer.close()

    parameters = sum(int(np.prod(r['shape'])) for r in writer.records.values())
    quantisable = sum(int(np.prod(r['shape'])) for name, r in writer.records.items()
                      if not name.endswith(('/scale', '/offset', '/bias'))
                      and int(np.prod(r['shape'])) >= 1 << 16)
    manifest = {
        'formatVersion': 1,
        'source': 'ESM-C tower from %s, ESMFold2 shim from %s'
                  % (arguments.esmc, arguments.esmfold2),
        'model': {'name': 'esmc', 'recycles': 0},
        'bundle': {'purpose': 'browser-inference', 'model': 'esmc',
                   'encoding': 'float32-le'},
        # 🔴 THE PAIRING, IN THE ARTEFACT. A tower is interchangeable between
        # releases and a shim is not, so which folding model this shim came
        # from has to survive the export.
        # 🔴 THE HEAD COUNT AND THE RESIDUAL SCALE BELONG HERE TOO. They are
        # not derivable from any tensor's shape - 1152 channels is 18 heads of
        # 64 and would read just as well as 16 of 72 - so a loader without them
        # has to guess, and the wrong guess builds an attention kernel that
        # compiles and mixes the wrong channels together.
        'languageModel': {'tower': arguments.esmc,
                          'shim': arguments.esmfold2,
                          'layers': layers, 'width': d_model,
                          'heads': heads, 'residualScale': residual_scale,
                          'mixEntries': layers + 1},
        'weightLayout': 'inner-major',
        # 🔴 NAMED, BECAUSE THE QUANTISER'S RULE IS ABOUT SUFFIXES AND THESE
        # ARE NOT ABOUT THEIR SUFFIXES. `embed/weights` is the input to all 36
        # blocks; `lm/combine` is 37 numbers that become a softmax over the
        # whole tower's output, and at 37 elements a group-32 scheme would fit
        # it in two groups.
        'float32Tensors': ['embed/weights', 'lm/combine'],
        'tensors': writer.records,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    written = sum(p.stat().st_size for p in out_dir.glob('*.f32.bin'))
    print('%s: %d tensors, %.1fM parameters (%.1fM quantisable), '
          '%.0f MiB float32 across %d shards'
          % (out_dir.name, len(writer.records), parameters / 1e6,
             quantisable / 1e6, written / 2 ** 20, writer.index))
    tower.close()
    fold.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
