"""Is every shard a manifest names actually at the remote, at the right length?

    python3 tools/check_remote_bundle.py af3
    python3 tools/check_remote_bundle.py af3 --revision <sha>

🔴 A PARTIAL UPLOAD LOOKS LIKE A WORKING ONE UNTIL A FOLD READS IT. The page
compiles the manifest in and fetches shards by name, so a shard that is missing
or short does not fail at load - it fails deep inside a dequantisation, as
"<file> has an invalid byte length", a message that names neither which half is
wrong nor why. Three separate hours have gone into that already.

This asks the remote for the LENGTH of every shard, without downloading any of
them, and compares against what the manifest says the tensors need. HEAD is
enough: Hugging Face answers it with the real content-length, and for a file
stored with LFS or xet it does so after following the redirect to the CDN.

Exits non-zero on the first thing that would fold to noise.
"""
import argparse
import json
import pathlib
import re
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent

# int8 and int5 carry their codes, then padding, then per-block scales, so a
# tensor's span is measured from where it starts rather than from its shape.
WIDTHS = {"float32": 4, "float16": 2, "int8": 1, "int5": 1}
INT5_GROUP_BYTES = 20


def tensor_byte_length(record: dict) -> int:
    """The bytes one tensor occupies, matching src/reference/dtype.js."""
    width = WIDTHS[record["dtype"]]
    elements = 1
    for size in record["shape"]:
        elements *= size
    if record["dtype"] not in ("int8", "int5"):
        return elements * width
    block = record["block"]
    groups = -(-elements // block)
    trailing = groups * 4 if record["dtype"] == "int5" else groups * 2
    return (record["scaleOffset"] - record.get("byteOffset", 0)) + trailing


def bundle_base(family: str) -> str:
    """The `remote` (or `directory`) the JS registry gives this family."""
    index = (ROOT / "src" / "reference" / "manifests" / "index.js").read_text("utf-8")
    current = None
    fields: dict[str, dict[str, str]] = {}
    for line in index.splitlines():
        opened = re.match(r"^  (\w+): \{$", line)
        if opened:
            current = opened.group(1)
            fields[current] = {}
        elif current is not None:
            found = re.match(r'^\s*(remote|directory):\s*"([^"]+)"', line)
            if found:
                fields[current][found.group(1)] = found.group(2)
            elif line == "  },":
                current = None
    if family not in fields:
        raise SystemExit(f"no bundle named {family} in src/reference/manifests/index.js")
    entry = fields[family]
    base = entry.get("remote") or entry["directory"]
    return base if base.endswith("/") else base + "/"


def compiled_manifest(family: str) -> dict:
    """The manifest as the page compiles it in, read out of the JS module."""
    module = ROOT / "src" / "reference" / "manifests" / f"{family}.js"
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


def content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("content-length")
            return int(length) if length is not None else None
    except Exception as error:                                   # noqa: BLE001
        print(f"  {url}\n    {type(error).__name__}: {error}", file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family")
    parser.add_argument("--revision", default=None,
                        help="check this commit instead of whatever the registry pins")
    arguments = parser.parse_args()

    base = bundle_base(arguments.family)
    if arguments.revision is not None:
        base = re.sub(r"/resolve/[^/]+/", f"/resolve/{arguments.revision}/", base)
    if base.startswith("."):
        print(f"{arguments.family} has no remote; nothing to check", file=sys.stderr)
        return 0

    manifest = compiled_manifest(arguments.family)
    # The furthest byte any tensor reads from each shard, which is the length
    # that shard must be AT LEAST.
    needed: dict[str, int] = {}
    for record in manifest["tensors"].values():
        end = record.get("byteOffset", 0) + tensor_byte_length(record)
        needed[record["file"]] = max(needed.get(record["file"], 0), end)

    print(f"{arguments.family}: {len(manifest['tensors'])} tensors in"
          f" {len(needed)} shards\n{base}")
    problems = 0
    total = 0
    for shard in sorted(needed):
        length = content_length(base + shard)
        want = needed[shard]
        if length is None:
            print(f"  MISSING  {shard}")
            problems += 1
        elif length < want:
            print(f"  SHORT    {shard}  {length} bytes, needs {want}")
            problems += 1
        else:
            total += length
            print(f"  ok       {shard}  {length / 1048576:.1f} MiB")
    if problems:
        print(f"\n{problems} shard(s) would fold to noise", file=sys.stderr)
        return 1
    print(f"\nall {len(needed)} shards present, {total / 1048576:.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
