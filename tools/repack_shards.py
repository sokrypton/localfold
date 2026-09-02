"""Lay an exported bundle into a few even shards instead of many uneven ones.

    python3 tools/repack_shards.py model-af3-int5 --files 8 --out model-af3-int5.packed
    python3 tools/repack_shards.py model-af3-int5 --files 8 --in-place

🔴 THIS MOVES BYTES, IT DOES NOT MAKE THEM. Nothing here quantises, re-orders
within a tensor, or touches a weight: a tensor's span is copied whole and its
three offsets - byteOffset, scaleOffset and, for int5, zeroOffset - all shift by
the same amount, because the span from byteOffset already covers the codes, the
padding and the scales. Every tensor is then read back from the new file and
compared with the original, byte for byte, before the manifest is written. A
repack that changed a weight would be a different model wearing the same name.

🔴 WHY FEWER FILES. The split dates from a fork whose weights were a single
gigabyte, and that reason is gone - the bundles are 97 to 265 MiB. What the
split still buys is parallel fetch of INDIVIDUALLY CACHEABLE pieces: a whole
shard comes back as a 200 and the Cache API can store it, where the ranges of
one big file come back as 206 and it cannot. Measured on a fast link, four
connections moved 27.5 MB/s against 14 serial, and eight moved 26.3 - so four
already saturate it and twenty-six buy nothing over eight. What twenty-six DO
buy is a mess: the af3 bundle ran from 0.0 to 40.5 MiB.

🔴 AND A SHARD IS STILL AT LEAST ONE WHOLE TENSOR. The largest is 40.5 MiB,
stacked over 48 blocks, so eight files land at 40.5, 40.5 and six of 30.6
rather than eight of 33.1. Evening that out means letting a tensor span files,
which is a reader change in the code where an off-by-one is a different protein;
measured, the tail it would save is about 3% of a cold load.
"""
import argparse
import json
import pathlib
import shutil
import sys

WIDTHS = {"float32": 4, "float16": 2, "int8": 1, "int5": 1}
ALIGNMENT = 4          # the exporter aligns every tensor, and the readers rely on it


def tensor_byte_length(record: dict) -> int:
    """The bytes one tensor occupies from its byteOffset; mirrors dtype.js."""
    elements = 1
    for size in record["shape"]:
        elements *= size
    if record["dtype"] not in ("int8", "int5"):
        return elements * WIDTHS[record["dtype"]]
    groups = -(-elements // record["block"])
    trailing = groups * 4 if record["dtype"] == "int5" else groups * 2
    return (record["scaleOffset"] - record.get("byteOffset", 0)) + trailing


def pack(spans: list[tuple[str, int]], files: int) -> list[list[str]]:
    """Longest-first into the emptiest bin, which is the standard makespan answer."""
    bins: list[list[str]] = [[] for _ in range(files)]
    totals = [0] * files
    for name, size in sorted(spans, key=lambda entry: -entry[1]):
        target = min(range(files), key=lambda index: totals[index])
        bins[target].append(name)
        totals[target] += -(-size // ALIGNMENT) * ALIGNMENT
    return [group for group in bins if group]


def read_compiled_manifest(module: pathlib.Path) -> dict:
    """The MANIFEST object out of src/reference/manifests/<family>.js."""
    text = module.read_text("utf-8")
    start = text.index("{", text.index("export const MANIFEST"))
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:index + 1])
    raise SystemExit(f"could not read MANIFEST out of {module}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", help="the export directory, e.g. model-af3-int5")
    parser.add_argument("--files", type=int, default=8)
    parser.add_argument("--manifest", default=None,
                        help="the compiled module to read, when the bundle has no manifest.json")
    parser.add_argument("--out", default=None, help="where to write; default <bundle>.packed")
    parser.add_argument("--in-place", action="store_true",
                        help="replace the bundle once every tensor has been verified")
    arguments = parser.parse_args()

    source = pathlib.Path(arguments.bundle).resolve()
    # 🔴 NOT EVERY BUNDLE KEEPS A manifest.json BESIDE ITS SHARDS. The monomer's
    # lives only as the compiled module the page imports, because that is the
    # copy the browser reads; the export directory holds shards and nothing
    # else. Falling back to the module means the tool works on either shape.
    manifest_path = source / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text("utf-8"))
    elif arguments.manifest is not None:
        manifest = read_compiled_manifest(pathlib.Path(arguments.manifest))
    else:
        print(f"{source}/manifest.json does not exist; pass --manifest with the"
              " compiled module for this bundle", file=sys.stderr)
        return 1
    records = manifest["tensors"]
    suffix = next(iter(records.values()))["file"].split(".", 1)[1]

    # Read every original shard once; a bundle is 265 MiB at most and this is a
    # one-shot tool, so holding them is simpler than seeking.
    originals = {name: (source / name).read_bytes()
                 for name in sorted({r["file"] for r in records.values()})}

    spans = [(name, tensor_byte_length(record)) for name, record in records.items()]
    groups = pack(spans, arguments.files)

    out = pathlib.Path(arguments.out) if arguments.out else source.with_suffix(source.suffix + ".packed")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rewritten: dict[str, dict] = {}
    for index, names in enumerate(groups):
        file = f"weights-{index:02d}.{suffix}"
        blob = bytearray()
        for name in names:
            record = dict(records[name])
            old_start = record.get("byteOffset", 0)
            length = tensor_byte_length(record)
            while len(blob) % ALIGNMENT:
                blob.append(0)
            new_start = len(blob)
            blob += originals[record["file"]][old_start:old_start + length]
            delta = new_start - old_start
            record["file"] = file
            record["byteOffset"] = new_start
            for key in ("scaleOffset", "zeroOffset"):
                if key in record:
                    record[key] += delta
            rewritten[name] = record
        (out / file).write_bytes(bytes(blob))

    # 🔴 VERIFIED BEFORE THE MANIFEST IS WRITTEN, tensor by tensor, against the
    # bytes that were there before. This is the whole warrant for the tool.
    packed = {name: (out / name).read_bytes()
              for name in sorted({r["file"] for r in rewritten.values()})}
    for name, record in rewritten.items():
        old = records[name]
        length = tensor_byte_length(old)
        before = originals[old["file"]][old.get("byteOffset", 0):old.get("byteOffset", 0) + length]
        after = packed[record["file"]][record["byteOffset"]:record["byteOffset"] + length]
        if before != after:
            print(f"{name}: {len(before)} bytes before, {len(after)} after, and they differ",
                  file=sys.stderr)
            return 1
        for key in ("scaleOffset", "zeroOffset"):
            if key in old and (record[key] - record["byteOffset"]) != (old[key] - old.get("byteOffset", 0)):
                print(f"{name}: {key} moved relative to the tensor", file=sys.stderr)
                return 1

    manifest["tensors"] = rewritten
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    sizes = sorted(((out / f).stat().st_size for f in packed), reverse=True)
    print(f"{source.name}: {len(originals)} shards -> {len(packed)},"
          f" {len(records)} tensors verified byte for byte")
    print("  " + ", ".join(f"{size / 1048576:.1f}" for size in sizes) + " MiB")

    if arguments.in_place:
        for stale in source.glob(f"*.{suffix}"):
            stale.unlink()
        for file in packed:
            shutil.move(str(out / file), source / file)
        shutil.move(str(out / "manifest.json"), source / "manifest.json")
        shutil.rmtree(out)
        print(f"  replaced {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
