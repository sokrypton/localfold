"""Re-encode the exported model shards as classic scripts, for file:// pages.

    python3 tools/export-js-weights.py                 # model/ -> model/*.js
    python3 tools/export-js-weights.py --model other/  # ...from somewhere else

WHY. A page opened as a file cannot fetch(), so it cannot read the .bin shards
sitting beside it however they are spelled. It CAN load a classic
<script src>, because script tags were never subject to the same-origin read
rule, and it CAN fetch() a data: URL, because that carries its own bytes rather
than asking an origin for them. Putting each shard's bytes into a script as a
base64 data: URL therefore reaches a file page by the only two doors it has.

WHAT IT COSTS. Base64 is four characters per three bytes, so this writes about
a third more than it reads - the 182 MiB of float16 shards become roughly
242 MiB of script. That is the price of the doors, and it is why the SERVED site keeps
using the binaries: HttpTensorStore reads those directly and pays nothing.
These files exist for the offline page alone.

The .bin shards are left exactly as they are, float16 and float32 tensors
alike. Nothing here re-quantises, re-orders or otherwise touches a weight - the
manifest it copies carries each tensor's dtype, and the store on the other end
reads it. This is a transport encoding, and `--check` re-decodes every shard to
prove the bytes round-trip.
"""
import argparse
import base64
import json
import pathlib
import sys

# One script per shard, so a load can report progress eight times rather than
# once and can free each base64 string as soon as it is decoded.
TEMPLATE = ('window.__afWeights = window.__afWeights || {{ shards: {{}} }};\n'
            'window.__afWeights.shards["{name}"] =\n'
            '"data:application/octet-stream;base64,{payload}";\n')

MANIFEST_TEMPLATE = ('window.__afWeights = window.__afWeights || {{ shards: {{}} }};\n'
                     'window.__afWeights.manifest = {manifest};\n'
                     'window.__afWeights.scripts = {scripts};\n')


def build(model: pathlib.Path, check: bool) -> int:
    manifest_path = model / "manifest.json"
    if not manifest_path.is_file():
        print(f"no manifest at {manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())
    shards = sorted({record["file"] for record in manifest["tensors"].values()})

    scripts = {}
    written = 0
    for shard in shards:
        source = model / shard
        if not source.is_file():
            print(f"missing shard {source}", file=sys.stderr)
            return 1
        raw = source.read_bytes()
        payload = base64.b64encode(raw).decode("ascii")
        if check and base64.b64decode(payload) != raw:
            print(f"{shard} did not round-trip", file=sys.stderr)
            return 1
        # weights-00.bin -> weights-00.js
        name = shard.split(".")[0] + ".js"
        (model / name).write_text(TEMPLATE.format(name=shard, payload=payload))
        scripts[shard] = name
        written += len(payload)
        print(f"  {shard}  {len(raw) / 1024 / 1024:6.1f} MiB"
              f" -> {name}  {len(payload) / 1024 / 1024:6.1f} MiB")

    (model / "manifest.js").write_text(MANIFEST_TEMPLATE.format(
        manifest=json.dumps(manifest, separators=(",", ":")),
        scripts=json.dumps(scripts, separators=(",", ":")),
    ))
    print(f"{len(shards)} shards -> {written / 1024 / 1024:.1f} MiB of script"
          f"{' (bytes verified)' if check else ''}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="model", help="the exported model directory")
    parser.add_argument("--check", action="store_true",
                        help="re-decode every shard and compare, before writing")
    arguments = parser.parse_args()
    raise SystemExit(build(pathlib.Path(arguments.model).resolve(), arguments.check))
