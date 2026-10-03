#!/usr/bin/env python3
"""Fetch published model bundles for the native ports, from the `remote:` URLs the page itself uses.

    python3 native/fetch_bundles.py af3 ef2-fast-600m esmc [--into=<repo>] [--suffix=-remote]

Each name is a family key of src/bundles/manifests/index.js; its bundle (manifest.json and every shard
the manifest names) lands in the family's `directory` under the repository, so the native wrappers find
it where they look (model-af3-int5/, model-esmfold2-int5/, ...). --suffix appends to that directory
name, for keeping a fetched copy beside a local export. A file already present at its size is kept.
"""
import json
import os
import re
import sys
import urllib.request


def families(index_js):
    """family key -> (directory, remote), parsed from the registry's object literal."""
    text = open(index_js).read()
    out = {}
    for m in re.finditer(r'^  "?([A-Za-z0-9-]+)"?: \{(.*?)^  \},', text, re.S | re.M):
        body = m.group(2)
        d = re.search(r'directory:\s*"([^"]+)"', body)
        r = re.search(r'remote:\s*"([^"]+)"', body)
        if d:
            out[m.group(1)] = (d.group(1), r.group(1) if r else None)
    return out


def fetch(url, path):
    tmp = path + ".part"
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, path)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    repo = opts.get("into", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    known = families(os.path.join(repo, "src", "bundles", "manifests", "index.js"))
    if not args:
        print("families:", " ".join(sorted(known)))
        return
    for name in args:
        if name not in known:
            sys.exit(f"no family {name!r} in src/bundles/manifests/index.js")
        directory, remote = known[name]
        if not remote:
            sys.exit(f"{name} has no published remote")
        dest = os.path.normpath(os.path.join(repo, directory.rstrip("/") + opts.get("suffix", "")))
        os.makedirs(dest, exist_ok=True)
        # 🔴 THE MANIFEST LAST: it is what says a bundle is here (the native wrappers and the CUDA worker
        # check for it), so it is written only once every shard it names is - an interrupted download
        # leaves no manifest, rather than one naming shards that never came. Every shard lands through a
        # `.part` renamed when whole, so a shard that exists is a whole one and is not fetched again.
        manifest_path = os.path.join(dest, "manifest.json")
        fetch(remote + "manifest.json", manifest_path + ".fetching")
        manifest = json.load(open(manifest_path + ".fetching"))
        files = sorted({record["file"] for record in manifest["tensors"].values()})
        for k, f in enumerate(files):
            path = os.path.join(dest, f)
            if not os.path.exists(path):
                fetch(remote + f, path)
            print(f"  {name}: {f} ({k + 1}/{len(files)})", flush=True)
        os.replace(manifest_path + ".fetching", manifest_path)
        print(f"{name} -> {dest}")


if __name__ == "__main__":
    main()
