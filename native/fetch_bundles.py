#!/usr/bin/env python3
"""Fetch published model bundles for the native ports, from the `remote:` URLs the page itself uses.

    python3 native/fetch_bundles.py af3 ef2-fast-600m esmc [--into=<repo>] [--suffix=-remote]

Each name is a family key of src/bundles/manifests/index.js; its bundle (manifest.json and every shard
the manifest names) lands in the family's `directory` under the repository, so the native wrappers find
it where they look (model-af3-int5/, model-esmfold2-int5/, ...). --suffix appends to that directory
name, for keeping a fetched copy beside a local export. A file already present at its size is kept.
"""
import concurrent.futures
import fcntl
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
        # ...and ONE FETCH A BUNDLE AT A TIME: the notebook prefetches the default model's while the ports
        # compile, and a fold arriving meanwhile must wait for that download rather than race it into the
        # same `.part` files
        with open(os.path.join(dest, ".fetch.lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if os.path.exists(manifest_path):
                print(f"{name} -> {dest} (already here)")
                continue
            fetch(remote + "manifest.json", manifest_path + ".fetching")
            manifest = json.load(open(manifest_path + ".fetching"))
            files = sorted({record["file"] for record in manifest["tensors"].values()})
            # the shards eight at a time: one connection to Hugging Face is ~21 MB/s from here, eight 112
            # (AF3's 277 MB in 2.5 s against 13.5)
            missing = [f for f in files if not os.path.exists(os.path.join(dest, f))]
            with concurrent.futures.ThreadPoolExecutor(8) as pool:
                pending = {pool.submit(fetch, remote + f, os.path.join(dest, f)): f for f in missing}
                done = len(files) - len(missing)
                for future in concurrent.futures.as_completed(pending):
                    future.result()
                    done += 1
                    print(f"  {name}: {pending[future]} ({done}/{len(files)})", flush=True)
            os.replace(manifest_path + ".fetching", manifest_path)
        print(f"{name} -> {dest}")


if __name__ == "__main__":
    main()
