#!/usr/bin/env python3
"""
Export an RFD4-Proteina checkpoint (`model.safetensors` + `config.json`) into a
`../localfold`-style sharded bundle (`manifest.json` + `shard_XXX.bin`) with
symmetric per-block INT8 quantization (`block=64` + `float16` scale per block).

Mirrors `../localfold/tools/quantize_model.py` and `../localfold/tools/export_bioemu2_model.py`:
  * Every 2D weight matrix inside `nn.transformer_layers.*`, `nn.pair_update_layers.*`,
    `autoencoder.decoder.transformer_layers.*`, and `autoencoder.encoder.transformer_layers.*`
    with >= 4096 elements is quantized to symmetric `int8` (block=64) with a `float16` scale
    rounded to float16 BEFORE dividing (`codes = clip(rint(w / f16(amax / 127)), -127, 127)`).
  * Every 1D vector (norms, biases), small tensor (< 4096 elements), feature/conditioning
    projection, and output readout head stays `float32`.
  * Autoencoder encoder tensors (`autoencoder.encoder.*`) are packed into dedicated
    `encoder_shard_XXX.bin` files so standard inference (Trunk + Decoder = 303 MiB) does not
    have to download the 127 MiB Encoder shards unless Partial Diffusion / Inverse Folding is used.

Usage:
  python3 tools/export_rfd4p_model.py \
    --input ./weights/model.safetensors \
    --config ./model-rfd4p-int8/config.json \
    --output ./model-rfd4p-int8 \
    --format int8 \
    --shard-mb 32
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
from math import prod
from pathlib import Path
import numpy as np

BLOCK = 64

QUANTIZE_LAYER_RE = re.compile(
    r"^(nn\.transformer_layers\.|nn\.pair_update_layers\.|"
    r"autoencoder\.decoder\.transformer_layers\.|"
    r"autoencoder\.encoder\.transformer_layers\.)"
)


def parse_safetensors_header(f) -> tuple[int, dict, dict]:
    header_len_bytes = f.read(8)
    if len(header_len_bytes) < 8:
        raise ValueError("Truncated safetensors header length.")
    (header_len,) = struct.unpack("<Q", header_len_bytes)
    header_json = json.loads(f.read(header_len).decode("utf-8"))
    metadata = header_json.pop("__metadata__", {})
    return 8 + header_len, metadata, header_json


def read_safetensor_f32(f, data_offset: int, entry: dict) -> np.ndarray:
    dtype_str = entry["dtype"]
    start, end = entry["data_offsets"]
    f.seek(data_offset + start)
    raw = f.read(end - start)
    if dtype_str == "F32":
        return np.frombuffer(raw, dtype="<f4").copy()
    if dtype_str == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if dtype_str == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2")
        u32 = u16.astype(np.uint32) << 16
        return u32.view(np.float32).copy()
    if dtype_str == "I64":
        return np.frombuffer(raw, dtype="<i8").astype(np.float32)
    if dtype_str == "I32":
        return np.frombuffer(raw, dtype="<i4").astype(np.float32)
    raise ValueError(f"Unsupported safetensors dtype: {dtype_str}")


def keep_float32(name: str, shape: list[int]) -> bool:
    """Keep 1D vectors, small tensors (<4096), stem/conditioning projections, and readout heads in float32."""
    if len(shape) != 2 or prod(shape) < 4096:
        return True
    if not QUANTIZE_LAYER_RE.search(name):
        return True
    if "norm" in name:
        return True
    return False


def quantise_int8(values: np.ndarray, block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-block int8 with float16 scale (matching ../localfold/tools/quantize_model.py)."""
    flat = values.reshape(-1).astype(np.float32, copy=False)
    padded = -flat.size % block
    grid = np.concatenate([flat, np.zeros(padded, np.float32)]).reshape(-1, block) if padded else flat.reshape(-1, block)
    amax = np.abs(grid).max(axis=1, keepdims=True)
    scale = (amax / 127.0).astype("<f2").astype(np.float32)
    safe = np.where(scale == 0, 1.0, scale)
    codes = np.clip(np.rint(grid / safe), -127, 127).astype(np.int8)
    return codes.reshape(-1)[: flat.size], scale.reshape(-1).astype("<f2")


def span(entry: dict) -> int:
    elements = prod(entry["shape"])
    if entry["dtype"] == "int8":
        return (entry["scaleOffset"] - entry["byteOffset"]) + -(-elements // entry["block"]) * 2
    return elements * {"float32": 4, "float16": 2}[entry["dtype"]]


def export_bundle(
    input_path: Path,
    config_path: Path | None,
    output_dir: Path,
    fmt: str = "int8",
    shard_mb: int = 32,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_max_bytes = max(1, shard_mb) * 1024 * 1024

    if config_path and config_path.exists():
        shutil.copyfile(config_path, output_dir / "config.json")

    with input_path.open("rb") as f:
        data_offset, metadata, tensors_meta = parse_safetensors_header(f)
        all_items = sorted(tensors_meta.items(), key=lambda kv: kv[1]["data_offsets"][0])

        # Partition into core inference (Trunk + Decoder) and optional Encoder
        core_items = [(k, v) for k, v in all_items if not k.startswith("autoencoder.encoder.")]
        enc_items = [(k, v) for k, v in all_items if k.startswith("autoencoder.encoder.")]

        written: dict[str, dict] = {}
        shards_meta: list[dict] = []
        counts = {"int8": 0, "float16": 0, "float32": 0}

        def write_group(items: list[tuple[str, dict]], prefix: str, role: str) -> None:
            shard_idx = 0
            chunks: list[bytes] = []
            shard_names: list[str] = []
            offset = 0
            content_end = 0

            def emit(raw: bytes, align: int = 4) -> None:
                nonlocal offset, content_end
                chunks.append(raw)
                offset += len(raw)
                content_end = offset
                padding = -offset % align
                if padding:
                    chunks.append(b"\0" * padding)
                    offset += padding

            def flush_shard() -> None:
                nonlocal shard_idx, chunks, shard_names, offset, content_end
                if not shard_names:
                    return
                shard_file = f"{prefix}_{shard_idx:03d}.bin"
                payload = b"".join(chunks)[:content_end]
                for tname in shard_names:
                    written[tname]["file"] = shard_file
                ends = [written[tname]["byteOffset"] + span(written[tname]) for tname in shard_names]
                if max(ends) != len(payload):
                    raise RuntimeError(
                        f"{shard_file}: last tensor ends at {max(ends)} but shard is {len(payload)} bytes"
                    )
                (output_dir / shard_file).write_bytes(payload)
                shards_meta.append({"file": shard_file, "bytes": len(payload), "role": role})
                shard_idx += 1
                chunks = []
                shard_names = []
                offset = 0
                content_end = 0

            for name, entry_meta in items:
                shape = list(entry_meta["shape"])
                values = read_safetensor_f32(f, data_offset, entry_meta)
                elements = prod(shape)
                if values.size != elements:
                    raise RuntimeError(f"{name}: expected {elements} elements, got {values.size}")

                is_f32 = keep_float32(name, shape) or fmt == "f32"
                est_bytes = elements * 4 if is_f32 else (elements * 2 if fmt == "f16" else elements + -(-elements // BLOCK) * 2)
                if offset > 0 and offset + est_bytes > shard_max_bytes:
                    flush_shard()

                rec: dict = {"file": "", "shape": shape, "byteOffset": offset}
                if is_f32:
                    rec["dtype"] = "float32"
                    emit(values.astype("<f4", copy=False).tobytes(), align=4)
                elif fmt == "f16":
                    rec["dtype"] = "float16"
                    emit(values.astype("<f2").tobytes(), align=4)
                else:
                    codes, scales = quantise_int8(values, BLOCK)
                    rec["dtype"] = "int8"
                    rec["block"] = BLOCK
                    emit(codes.tobytes(), align=2)
                    rec["scaleOffset"] = offset
                    emit(scales.tobytes(), align=4)

                counts[rec["dtype"]] += 1
                written[name] = rec
                shard_names.append(name)

            flush_shard()

        write_group(core_items, "shard", "core")
        write_group(enc_items, "encoder_shard", "encoder")

    total_bytes = sum(s["bytes"] for s in shards_meta)
    core_bytes = sum(s["bytes"] for s in shards_meta if s["role"] == "core")
    enc_bytes = sum(s["bytes"] for s in shards_meta if s["role"] == "encoder")

    manifest = {
        "format": "rfd4p-sharded-v1",
        "bundle": {
            "encoding": "mixed-le",
            "format": fmt,
            "quantisation": {"scheme": "symmetric-per-block", "bits": 8, "block": BLOCK} if fmt == "int8" else None,
            "bytes": total_bytes,
            "coreBytes": core_bytes,
            "encoderBytes": enc_bytes,
            "shards": len(shards_meta),
        },
        "shards": shards_meta,
        "metadata": metadata,
        "tensors": written,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"{output_dir}/  "
        + ", ".join(f"{n} {k}" for k, n in counts.items() if n)
        + f"  | core={core_bytes / 1048576:.1f} MiB, encoder={enc_bytes / 1048576:.1f} MiB,"
        + f" total={total_bytes / 1048576:.1f} MiB across {len(shards_meta)} shards"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("./model-rfd4p-int8/model.safetensors"))
    parser.add_argument("--config", type=Path, default=Path("./model-rfd4p-int8/config.json"))
    parser.add_argument("--output", type=Path, default=Path("./model-rfd4p-int8"))
    parser.add_argument("--format", choices=("int8", "f16", "f32"), default="int8")
    parser.add_argument("--shard-mb", type=int, default=32)
    args = parser.parse_args()
    export_bundle(args.input, args.config, args.output, args.format, args.shard_mb)


if __name__ == "__main__":
    main()
