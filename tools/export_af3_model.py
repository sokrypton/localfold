#!/usr/bin/env python3
"""Write a LocalFold float32 model directory from AF3-lineage parameters.

    python3 tools/export_af3_model.py                     # the trunk, from AF3
    python3 tools/export_af3_model.py --model openbind0   # ...from OpenBind-0
    python3 tools/export_af3_model.py --include diffuser  # everything

The output is the shape tools/quantize_model.py consumes - manifest.json plus
weights-NN.f32.bin shards, every tensor float32 - so the int8 packing, the
sharding limits and the digest bookkeeping stay in the tooling that already
produces the shipped monomer and multimer models.

WHERE THE WEIGHTS COME FROM. A ColabDesign2 blob: one zstd stream of (scope,
name, array) records, which is what its converters emit for all seven
AF3-lineage checkpoints (OpenFold3, Boltz-2, Chai-1, Protenix, RF3, IntelliFold,
OpenDDE). Reading the blob rather than any one checkpoint format is the whole
point - a second model becomes a different --blob, not a second exporter.

🔴 WHICH CHECKPOINT FILLED THE BUNDLE IS RECORDED IN IT. AF3's own parameters
carry DeepMind's Weights Terms of Use; OpenFold3's are Apache 2.0; both build
the same graph and both land in `model-af3/`, so the directory cannot say which
is inside. The manifest's model.name does, and tools/build_site.py reads it and
requires LOCALFOLD_ACCEPT_MODEL_TERMS before publishing a restricted one. The
page gates the download behind licence acceptance for the same reason.

🔴 THE TENSOR NAMES ARE THE HAIKU PATHS, UNCHANGED. The AF2 exports rename every
tensor to `stack_haiku_0042` and keep the real structure in nested manifest
sections; that made sense when the manifest had to describe a graph the loader
already knew. Here the names ARE the description, they match the oracle's module
paths character for character, and a tensor that is wrong can be found by
grepping for it in both places. 213 names cost 27 KiB in the compiled manifest.

🔴 ONLY THE TRUNK BY DEFAULT. The diffusion head is 203 M of the 368 M
parameters and the confidence head another 13 M, and neither has a graph to run
on yet. Exporting them would triple the download to ship bytes nothing reads.
--include widens the export when that changes; the manifest records what was
taken so a bundle can never quietly be half a model.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COLABDESIGN2 = os.path.expanduser("~/Documents/GitHub/ColabDesign2")
# One blob per model, each obtained from source and checked against the graph's
# own jax.eval_shape table before use.
#
# 🔴 openbind IS DOWNLOADED, NOT CONVERTED HERE. It is OpenFold3's v0.5.0
# release, published already converted at
# huggingface.co/sokrypton/af3-any-model, and `read_blob` above reads that file
# directly - so adding it needed no torch, no 2.3 GB checkpoint and no second
# checkout:
#
#     mkdir -p ~/af3_ported && cd ~/af3_ported
#     curl -sSLO https://huggingface.co/sokrypton/af3-any-model/resolve/\
#       bc9038fdabff2a06968f85e91838fe49c936d68f/openbind.bin.zst
#
# 🔴 AND THE REVISION IS PINNED, for the reason the shard URLs are: a blob
# fetched from a moving branch can change under a bundle that did not, and the
# resulting failure names neither half. `openbind.shapes.json` beside it is the
# conversion's own coverage record - it reports 406 arrays and 0 missing, and
# names `of3-ob-174k.pt` as the source.
#
# 🔴 openbind0 IS NOT openfold3, WHATEVER THE LINEAGE SUGGESTS. They are two
# releases of one project with different forward conventions - see
# src/af3/dialect.js - so they are two entries, and the manifest's model.name
# is what carries the difference into the page.
#
# 🔴 AND THE RELEASE NUMBER IS PART OF THE NAME. Upstream's announcement calls
# this model OpenBind-0, and their own registry's bare `openbind` is a name a
# LATER release would also answer to - which is exactly how `openfold3` came to
# mean two models with different forward conventions. A bundle named openbind0
# cannot be mistaken for OpenBind-1 by a loader that has never heard of it.
# `openbind` is kept as an alias, because that is the name upstream publishes
# the blob under.
BLOBS = {
    "alphafold3": "~/af3_official_weights/af3.bin.zst",
    "openfold3": "~/af3_converted_cd2/of3_ported_weights.bin.zst",
    "openbind0": "~/af3_ported/openbind.bin.zst",
}

# The trunk: the evoformer stacks, the conditioning that builds their inputs
# (including the atom transformer encoder, which produces 384 of target_feat's
# 447 columns), and the distogram head that reads the pair out.
TRUNK = ("diffuser/evoformer", "diffuser/distogram_head")

# 🔴 SIZED IN FLOAT32 BYTES, FOR AN INT8 RESULT, exactly as the multimer export
# is: quantize_model.py maps shard to shard, so a 48 MiB float32 shard lands at
# about 12 MiB - the size the shipped monomer's shards are, and the size that
# measured best against making four times as many requests for the same bytes.
#
# 🔴 AF3 IS PACKED TO int5, NOT int8, SO ITS SHARDS LAND NEARER 10 MiB. That is
# why the shipped af3-int5 bundle clusters there rather than at 12: the limit is
# calibrated for a quantiser it is no longer paired with. Harmless, and worth
# expressing per target dtype whenever this is next re-exported.
#
# 🔴 AND A SHARD IS AT LEAST ONE WHOLE TENSOR, which is what `and self.chunks`
# below means: a tensor over the limit gets a file to itself rather than being
# split, because a reader takes it from one contiguous span in one file. AF3's
# single-transition weights are stacked over 48 blocks and land at 40.5 MiB
# against a 7.9 MiB median - the two outliers in that bundle are one tensor
# each. Evening them out means letting a tensor span shards, which is a manifest
# and reader change; measured, it is not worth it - splitting the largest across
# four ranges saved 7%, and fetching biggest-first saved nothing at all, because
# a cold load is bytes over bandwidth long before it is a tail.
SHARD_LIMIT = 48 * 1024 * 1024


class ShardWriter:
    """Lay tensors into float32 shards, four-byte aligned, none over the limit."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.records: dict[str, dict] = {}
        self.chunks: list[bytes] = []
        self.offset = 0
        self.index = 0

    def _flush(self) -> None:
        if not self.chunks:
            return
        path = self.out_dir / f"weights-{self.index:02d}.f32.bin"
        with path.open("wb") as handle:
            for chunk in self.chunks:
                handle.write(chunk)
        self.chunks, self.offset, self.index = [], 0, self.index + 1

    def add(self, name: str, values: np.ndarray) -> None:
        flat = np.ascontiguousarray(values, dtype="<f4").reshape(-1)
        if self.offset + flat.nbytes > SHARD_LIMIT and self.chunks:
            self._flush()
        self.records[name] = {
            "file": f"weights-{self.index:02d}.f32.bin",
            "shape": list(values.shape),
            "byteOffset": self.offset,
            "dtype": "float32",
        }
        self.chunks.append(flat.tobytes())
        self.offset += flat.nbytes

    def close(self) -> None:
        self._flush()


def decode(data: bytes, dtype: str, length: int, offset: int, shape) -> np.ndarray:
    """One record's buffer as an array.

    🔴 AlphaFold 3's OWN BLOB IS bfloat16, WHICH numpy HAS NO DTYPE FOR. It is
    the top sixteen bits of a float32 and nothing else - same exponent, seven
    mantissa bits - so widening is a shift, not a conversion, and it needs no
    ml_dtypes. Every ported blob this reads is float32; discovering that AF3's
    is not was the first thing this reader did.
    """
    if dtype == "bfloat16":
        raw = np.frombuffer(data, dtype="<u2", count=length // 2, offset=offset)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    numpy_dtype = np.dtype(dtype)
    return np.frombuffer(data, dtype=numpy_dtype, count=length // numpy_dtype.itemsize,
                         offset=offset).reshape(shape)


def read_blob(path: str):
    """(scope, name, array) records out of a haiku parameter blob.

    🔴 READ HERE RATHER THAN THROUGH A CHECKOUT, because the format is a wire
    format and this is thirty lines of it. It is AlphaFold 3's own
    (`src/alphafold3/model/params.py`, `encode_record`): a `<5i` header giving
    the lengths of the scope, the name, the dtype string, the shape and the
    buffer, then those five fields back to back, repeated to the end of the
    stream, the whole thing zstd-compressed when the name says `.zst`. Both
    producers this project reads write exactly that - ColabDesign2's converters
    and the `converters/` package in a sokrypton/alphafold3 checkout - so
    importing either one to parse it made the exporter depend on a clone at a
    hardcoded path for a struct.unpack.

    🔴 AND THAT MATTERS FOR A PORTED MODEL SPECIFICALLY. `openbind.bin.zst` is
    published on Hugging Face already converted; needing a torch environment and
    a 2 GB checkpoint to read a file that has been downloaded is the difference
    between "add a model" and "set up a machine".
    """
    path = os.path.expanduser(str(path))
    if path.endswith(".zst"):
        try:
            import zstandard
        except ImportError as error:
            raise SystemExit(
                f"reading {path} needs zstandard: pip install zstandard") from error
        with open(path, "rb") as handle:
            data = zstandard.ZstdDecompressor().stream_reader(handle).read()
    else:
        with open(path, "rb") as handle:
            data = handle.read()

    header = struct.Struct("<5i")
    records = []
    at = 0
    while at < len(data):
        if len(data) - at < header.size:
            raise SystemExit(f"{path}: {len(data) - at} trailing bytes, "
                             f"too few for a {header.size}-byte record header")
        scope_len, name_len, dtype_len, rank, buffer_len = header.unpack_from(data, at)
        at += header.size
        take = lambda n: data[at:at + n]  # noqa: E731
        scope = take(scope_len).decode("utf-8"); at += scope_len
        name = take(name_len).decode("utf-8"); at += name_len
        dtype = take(dtype_len).decode("utf-8"); at += dtype_len
        shape = struct.unpack_from(f"<{rank}i", data, at); at += rank * 4
        values = decode(data, dtype, buffer_len, at, shape)
        at += buffer_len
        # 🔴 THE IDENTIFIER RECORD IS NOT A PARAMETER. A blob written by
        # `converters/common.py` opens with `__meta__/__identifier__`, 64 bytes
        # naming the model; AF3's own carries the same. Handing it to the
        # exporter's name matching would have it looking for a tensor.
        if scope == "__meta__":
            continue
        records.append((scope, name, np.asarray(values)))
    return records


# Where a checkpoint keeps its Fourier noise embedding, when it keeps one.
FOURIER_SCOPE = "diffuser/~/diffusion_head"
FOURIER_LEAVES = ("fourier_embedding_weight", "fourier_embedding_bias")


def stock_fourier_constants():
    """AF3's frozen Fourier weight and bias, as arrays.

    🔴 STOCK AF3 KEEPS THESE IN ITS SOURCE, NOT IN ITS CHECKPOINT. DeepMind
    generated them once from a fixed seed and froze them "to future proof
    against changes in jax rng generation", so a stock export has no tensor for
    them while every PORTED model of the lineage trained its own and carries it
    under the names above. Baking the constants in under those same names is
    what lets one loader read either: the alternative is a table compiled into
    the page that silently applies somebody else's random projection the moment
    the weights change.
    """
    if COLABDESIGN2 not in sys.path:
        sys.path.insert(0, COLABDESIGN2)
    from colabdesign2.af3.alphafold3.model.network import (  # noqa: E402
        noise_level_embeddings,
    )
    return (np.asarray(noise_level_embeddings._WEIGHT, dtype=np.float32),
            np.asarray(noise_level_embeddings._BIAS, dtype=np.float32))


def export(blob_path: str, out_dir: Path, include: tuple[str, ...],
           model: str) -> int:
    records = read_blob(blob_path)
    # ...synthesised only when the checkpoint has none, so a trained embedding
    # is never overwritten by the constants.
    if not any(scope == FOURIER_SCOPE and name in FOURIER_LEAVES
               for scope, name, _ in records):
        weight, bias = stock_fourier_constants()
        records = list(records) + [
            (FOURIER_SCOPE, FOURIER_LEAVES[0], weight),
            (FOURIER_SCOPE, FOURIER_LEAVES[1], bias),
        ]
    wanted = [(scope, name, array) for scope, name, array in records
              if scope.startswith(include)]
    if not wanted:
        print(f"nothing in the blob starts with any of {include}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(out_dir)
    # ...sorted, so the byte layout is a function of the contents and re-running
    # the export twice produces identical shards. The digests in the compiled
    # manifest are checked at build time; a layout that depended on dict order
    # would fail that check for no reason.
    keep_float32 = []
    for scope, name, array in sorted(wanted):
        tensor = f"{scope}/{name}"
        writer.add(tensor, array)
        # LayerNorm scales and offsets, and every bias. They are 0.14% of the
        # trunk's parameters - about half a mebibyte - and they are where
        # per-block int8 is worst: a 128-element vector is two blocks, so two
        # scales carry the whole tensor, and a norm's job is to set the scale of
        # everything downstream of it. Free to keep, so keep them.
        # ...and the Fourier embedding, which is 512 numbers and a RANDOM
        # PROJECTION: quantising it would perturb the very thing whose spread
        # makes the embedding informative, to save half a kilobyte.
        if name in ("offset", "scale", "bias") or name in FOURIER_LEAVES:
            keep_float32.append(tensor)
    writer.close()

    excluded = sorted({scope.split("/")[1] for scope, _, _ in records
                       if not scope.startswith(include) and "/" in scope})
    parameters = sum(int(np.prod(record["shape"])) for record in writer.records.values())
    manifest = {
        "formatVersion": 1,
        "source": f"AF3-lineage parameters from {Path(blob_path).name}",
        "model": {"name": model, "recycles": 0},
        "bundle": {"purpose": "browser-inference", "model": model,
                   "encoding": "float32-le"},
        # 🔴 WHAT WAS LEFT OUT, IN THE ARTEFACT ITSELF. A bundle carrying only the
        # trunk is correct today and wrong the moment something asks it for the
        # diffusion head, and the difference has to be visible without going back
        # to the exporter's default argument.
        "coverage": {"included": list(include), "excludedScopes": excluded},
        "float32Tensors": keep_float32,
        "tensors": writer.records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    written = sum(path.stat().st_size for path in out_dir.glob("*.f32.bin"))
    print(f"{out_dir.name}/  {len(writer.records)} tensors,"
          f" {parameters / 1e6:.1f} M parameters,"
          f" {written / 1048576:.0f} MiB float32 across {writer.index} shards"
          f"  ({len(keep_float32)} kept float32)")
    if excluded:
        print(f"  not exported: {', '.join(excluded)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="alphafold3", choices=sorted(BLOBS),
                        help="which checkpoint of the lineage to export")
    parser.add_argument("--blob", default=None,
                        help="an AF3-lineage parameter blob (default: the model's)")
    parser.add_argument("--out", default=None,
                        help="default: model-af3-f32, or model-<model>-f32")
    parser.add_argument("--include", nargs="+", default=list(TRUNK),
                        help="scope prefixes to export (default: the trunk)")
    parser.add_argument("--colabdesign2", default=None,
                        help="where ColabDesign2 is checked out")
    arguments = parser.parse_args()
    if arguments.colabdesign2:
        global COLABDESIGN2
        COLABDESIGN2 = os.path.expanduser(arguments.colabdesign2)
    blob = arguments.blob or BLOBS[arguments.model]
    # One bundle directory for the AF3 graph, whichever checkpoint of the
    # lineage fills it. Which one that was is recorded in the manifest's
    # model.name, and tools/build_site.py reads it there before publishing -
    # see RESTRICTED_TERMS, which is why the name has to be in the artefact
    # rather than only in the directory it landed in.
    out = arguments.out or str(ROOT / "model-af3-f32")
    return export(os.path.expanduser(blob), Path(out),
                  tuple(arguments.include), arguments.model)


if __name__ == "__main__":
    raise SystemExit(main())
