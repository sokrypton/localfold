#!/usr/bin/env python3
"""Write a LocalFold float32 model directory from AF2-multimer parameters.

The output is the shape tools/quantize_model.py consumes - manifest.json plus
weights-NN.f32.bin shards, every tensor float32 - so the int8 packing, the
sharding limits and the digest bookkeeping stay in the tooling that already
produces the shipped monomer model rather than being reimplemented here.

WHAT COMES FROM WHERE.

    evoformer / extra-MSA / embedding / structure / confidence
        the multimer checkpoint, through tools/convert_multimer_params.py

    residue geometry
        the existing monomer export. These are residue_constants tables -
        default frames, atom14 layouts, masks - chemistry rather than learned
        parameters, and identical in both models.

    template embedder
        🔴 NOT EXPORTED. Multimer's is architecturally different from monomer's
        and no reshape maps one to the other, so a multimer fold runs
        template-free, as ColabDesign2's merge does. The manifest records the
        omission rather than quietly borrowing the monomer's, which would load
        and be wrong.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_multimer_params import (  # noqa: E402
    EVO, convert_multimer_params, load_params,
)

A = "alphafold/alphafold_iteration/"
# 🔴 SIZED IN FLOAT32 BYTES, FOR AN INT8 RESULT. quantize_model.py maps shard to
# shard, so whatever is laid out here divides by about four on the way out.
# Cutting at 13 MiB - the size the monomer's shipped shards are - gave 25 shards
# of 4 MiB rather than 8 of 12, which is three times the requests for the same
# bytes. Four times the target, so the quantised shards land where they should.
SHARD_LIMIT = 48 * 1024 * 1024

SECTION_SCOPES = {
    "evoformerStack": A + "evoformer/evoformer_iteration/",
    "extraMsaStack": A + "evoformer/extra_msa_stack/",
    "embedding": A + "evoformer/",
    "structureModule": A + "structure_module/",
}
SECTION_PREFIX = {
    "evoformerStack": "stack_haiku",
    "extraMsaStack": "extra_stack_haiku",
    "embedding": "embedding_haiku",
    "structureModule": "structure_haiku",
    "confidenceHeads": "confidence_haiku",
    "templateEmbedding": "template_haiku",
}
CONFIDENCE = {
    "predictedLddt": {
        "act_0": A + "predicted_lddt_head/act_0",
        "act_1": A + "predicted_lddt_head/act_1",
        "input_layer_norm": A + "predicted_lddt_head/input_layer_norm",
        "logits": A + "predicted_lddt_head/logits",
    },
    "predictedAlignedError": {"logits": A + "predicted_aligned_error_head/logits"},
}


class ShardWriter:
    """Lay tensors into float32 shards, four-byte aligned, none over the limit."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.records: dict[str, dict] = {}
        self.chunks: list[np.ndarray] = []
        self.offset = 0
        self.index = 0

    def _flush(self) -> None:
        if not self.chunks:
            return
        path = self.out_dir / f"weights-{self.index:02d}.f32.bin"
        with path.open("wb") as handle:
            for chunk in self.chunks:
                handle.write(chunk.tobytes())
        self.chunks, self.offset, self.index = [], 0, self.index + 1

    def add(self, name: str, values: np.ndarray) -> str:
        flat = np.ascontiguousarray(values, dtype="<f4").reshape(-1)
        if self.offset + flat.nbytes > SHARD_LIMIT and self.chunks:
            self._flush()
        self.records[name] = {
            "file": f"weights-{self.index:02d}.f32.bin",
            "shape": list(values.shape),
            "byteOffset": self.offset,
            "dtype": "float32",
        }
        self.chunks.append(flat)
        self.offset += flat.nbytes
        return name

    def close(self) -> None:
        self._flush()


def export(params_path: Path, monomer_dir: Path, out_dir: Path) -> int:
    raw = load_params(params_path)
    params = dict(raw)
    params.update(convert_multimer_params(raw))

    reference = json.loads((Path(__file__).resolve().parent.parent
                            / "src" / "reference" / "manifest.js").read_text()
                           .split("=", 1)[1].rsplit(";", 1)[0].strip())

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(out_dir)
    manifest: dict = {
        "formatVersion": 1,
        "source": f"AlphaFold multimer parameters from {params_path.name}",
        "model": {"name": "model_1_multimer_v3", "recycles": 3},
        "bundle": {"purpose": "browser-inference", "model": "model_1_multimer_v3",
                   "encoding": "float32-le"},
    }
    counters = {prefix: 0 for prefix in SECTION_PREFIX.values()}

    def identifier(section: str) -> str:
        prefix = SECTION_PREFIX[section]
        name = f"{prefix}_{counters[prefix]:04d}"
        counters[prefix] += 1
        return name

    missing: list[str] = []
    for section, scope in SECTION_SCOPES.items():
        wanted = reference[section].get("parameters", {})
        parameters: dict = {}
        for module in sorted(wanted):
            source = params.get(scope + module)
            if source is None:
                missing.append(f"{section}/{module}")
                continue
            parameters[module] = {
                leaf: writer.add(identifier(section), source[leaf]) for leaf in sorted(source)
            }
        manifest[section] = {key: value for key, value in reference[section].items()
                             if key != "parameters"}
        manifest[section]["parameters"] = parameters

    confidence: dict = {}
    for head, modules in CONFIDENCE.items():
        entry = {}
        for leaf_name, path in modules.items():
            source = params.get(path)
            if source is None:
                missing.append(f"confidenceHeads/{leaf_name}")
                continue
            entry[leaf_name] = {leaf: writer.add(identifier("confidenceHeads"), source[leaf])
                                for leaf in sorted(source)}
        confidence[head] = entry
    manifest["confidenceHeads"] = {"parameters": confidence}

    # ...the chemistry tables and the PAE bin edges, straight from the monomer
    # export. They are residue_constants, not learned parameters.
    borrowed = list(reference["residueGeometry"]["tensors"]) + ["confidencePaeBreaks"]
    monomer_manifest = json.loads((monomer_dir / "manifest.json").read_text()) \
        if (monomer_dir / "manifest.json").is_file() else reference
    for name in borrowed:
        record = monomer_manifest["tensors"][name]
        # 🔴 THESE ARE READ AS FLOAT32, so say so rather than assume it. The
        # monomer model/ has no manifest.json any more, so this falls back to
        # the compiled-in one - which describes the INT8 export. Every borrowed
        # tensor happens to be in its float32 keep-list (the geometry tables and
        # the PAE bin edges are never quantised), but that is luck: quantise one
        # of them later and this would read the codes as floats and produce
        # silent nonsense in the geometry.
        if record["dtype"] != "float32":
            raise SystemExit(
                f"{name} is {record['dtype']} in the monomer export; this reads float32. "
                "Point --monomer at a float32 export, or widen it here first.")
        shard = (monomer_dir / record["file"]).read_bytes()
        values = np.frombuffer(shard, dtype="<f4",
                               count=int(np.prod(record["shape"])),
                               offset=record["byteOffset"]).reshape(record["shape"])
        writer.add(name, values)
    manifest["residueGeometry"] = reference["residueGeometry"]

    # 🔴 THE TEMPLATE EMBEDDER IS NOT OPTIONAL, whatever its name suggests.
    # Multimer's config has template.enabled TRUE, and the embedding wrapper adds
    # its output to the pair unconditionally - masking every template off does
    # not zero that term, because the embedder's biases and layer norms still
    # produce one. Omitting it put the pair track 30% out from the first block.
    #
    # It IS architecturally different from the monomer's, so nothing is
    # converted here: the modules are exported under their own names, and the
    # graph reads them as multimer's own shapes.
    # ...`params` carries BOTH forms of the template stack's triangle
    # multiplication - the checkpoint's fused one and the split one
    # convert_multimer_params produced - so the fused originals are dropped
    # rather than shipped twice.
    superseded = ("/projection", "/gate", "/left_norm_input", "/center_norm")
    template = {}
    for module in sorted(params):
        if "template" not in module:
            continue
        if "triangle_multiplication" in module and module.endswith(superseded):
            continue
        leaf = module.split("evoformer/")[-1]
        template[leaf] = {name: writer.add(identifier("templateEmbedding"), values)
                          for name, values in sorted(params[module].items())}
    manifest["templateEmbedding"] = {
        "parameterFormat": "haiku",
        "implementation": "AF2-multimer TemplateEmbedding",
        "pairStackBlocks": 2,
        "parameters": template,
    }

    writer.close()
    manifest["tensors"] = writer.records
    (out_dir / "manifest.json").write_text(json.dumps(manifest))

    total = sum(int(np.prod(r["shape"])) for r in writer.records.values())
    print(f"{len(writer.records)} tensors, {total:,} elements, {writer.index} shards")
    print(f"wrote {out_dir}/manifest.json")
    if missing:
        print(f"\n🔴 {len(missing)} parameters had no source:", file=sys.stderr)
        for name in missing:
            print(f"    {name}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("params", type=Path, help="params_model_N_multimer_v3.npz")
    parser.add_argument("--monomer", type=Path, default=Path("model"),
                        help="existing export, for the residue-geometry tables")
    parser.add_argument("--out", type=Path, default=Path("model-multimer.f32"))
    args = parser.parse_args()
    return export(args.params, args.monomer, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
