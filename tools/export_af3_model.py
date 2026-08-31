#!/usr/bin/env python3
"""Write a LocalFold float32 model directory from AF3-lineage parameters.

    python3 tools/export_af3_model.py                     # the trunk, from AF3
    python3 tools/export_af3_model.py --model openfold3   # ...from OpenFold3
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
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COLABDESIGN2 = os.path.expanduser("~/Documents/GitHub/ColabDesign2")
# One blob per model, each obtained from source and checked against the graph's
# own jax.eval_shape table before use. The OpenFold3 one is ColabDesign2's
# converter's output - the alphafold3 repo ships an older converter under a
# different name, and its output is not what this reads.
BLOBS = {
    "alphafold3": "~/af3_official_weights/af3.bin.zst",
    "openfold3": "~/af3_converted_cd2/of3_ported_weights.bin.zst",
}

# The trunk: the evoformer stacks, the conditioning that builds their inputs
# (including the atom transformer encoder, which produces 384 of target_feat's
# 447 columns), and the distogram head that reads the pair out.
TRUNK = ("diffuser/evoformer", "diffuser/distogram_head")

# 🔴 SIZED IN FLOAT32 BYTES, FOR AN INT8 RESULT, exactly as the multimer export
# is: quantize_model.py maps shard to shard, so a 48 MiB float32 shard lands at
# about 12 MiB - the size the shipped monomer's shards are, and the size that
# measured best against making four times as many requests for the same bytes.
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


def read_blob(path: str):
    """(scope, name, array) records, through ColabDesign2's reader.

    Imported here rather than at the top so that --help works on a machine that
    has never seen ColabDesign2, and so the failure names the thing that is
    missing instead of dying on an import line.
    """
    if COLABDESIGN2 not in sys.path:
        sys.path.insert(0, COLABDESIGN2)
    try:
        from colabdesign2.af3.converters import read_blob as read
    except ImportError as error:
        raise SystemExit(
            f"cannot import ColabDesign2's blob reader from {COLABDESIGN2}: {error}\n"
            "It is where the AF3-lineage converters live; clone it or pass"
            " --colabdesign2.") from error
    return read(path)


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
