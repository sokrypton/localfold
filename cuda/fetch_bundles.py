#!/usr/bin/env python3
"""Fetch published model bundles for the native ports, from the `remote:` URLs the page itself uses.

    python3 cuda/fetch_bundles.py af3 ef2-fast-600m esmc [--into=<repo>] [--suffix=-remote]

    python3 cuda/fetch_bundles.py --af3-any-model chai1 esm2

--af3-any-model fetches af3-any-model's own published blobs instead (huggingface.co/sokrypton/af3-any-model, int8,
pinned below), the weights the CUDA ports fold the AF3 lineage with - one *.bin.zst each, into af3am-<name>/,
which cuda/af3 reads as published (common.cuh's blob reader). AlphaFold 3's own (af3) are hosted there for
academic, non-commercial use under Google DeepMind's AF3 terms, and fetched only once those terms are accepted:
LOCALFOLD_ACCEPT_MODEL_TERMS=alphafold3, or a yes at the prompt.

Each name is a family key of shared/bundles/manifests/index.js; its bundle (manifest.json and every shard
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


# af3-any-model's int8 blobs, at the commit that added AlphaFold 3's (after chai1's structure token-pair weights)
AF3_ANY_MODEL = "https://huggingface.co/sokrypton/af3-any-model/resolve/98f787eacd6e49cdba88e044622773a8702f03fe/"
BLOBS = {
    "af3": "alphafold3/af3.int8.bin.zst",     # (Google DeepMind's: academic, non-commercial use, terms first)
    "chai1": "chai1/chai1.int8.bin.zst",
    "boltz2": "boltz2/boltz2.int8.bin.zst",
    "protenix2": "protenix/protenix2.int8.bin.zst",
    "intellifold2": "intellifold2/intellifold2.int8.bin.zst",
    "rosettafold3": "rosettafold3/rosettafold3.int8.bin.zst",
    "openbind0": "openfold3/openbind0.int8.bin.zst",
    "opendde": "opendde/opendde.int8.bin.zst",
    "esm2": "lm/esm2.bin.zst",          # (chai-1's ESM2 3B: int8 as published, its only form)
}


def fetch_ranged(url, path, name, parts=8):
    """One large file as `parts` byte ranges at once (one connection is ~21 MB/s from here), then joined."""
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:
        size = int(resp.headers["Content-Length"])
    step = -(-size // parts)
    ranges = [(k * step, min(size, (k + 1) * step) - 1) for k in range(parts) if k * step < size]

    def piece(k):
        lo, hi = ranges[k]
        req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (lo, hi)})
        with urllib.request.urlopen(req) as resp, open("%s.part%d" % (path, k), "wb") as f:
            while chunk := resp.read(1 << 22):
                f.write(chunk)
        if os.path.getsize("%s.part%d" % (path, k)) != hi - lo + 1:
            raise RuntimeError("%s: range %d came back short" % (url, k))
        return k
    with concurrent.futures.ThreadPoolExecutor(parts) as pool:
        for done, _ in enumerate(pool.map(piece, range(len(ranges))), 1):
            print(f"  {name}: {os.path.basename(path)} ({done}/{len(ranges)})", flush=True)
    with open(path + ".part", "wb") as out:
        for k in range(len(ranges)):
            with open("%s.part%d" % (path, k), "rb") as f:
                while chunk := f.read(1 << 24):
                    out.write(chunk)
            os.remove("%s.part%d" % (path, k))
    if os.path.getsize(path + ".part") != size:
        raise RuntimeError("%s: %d bytes, not %d" % (url, os.path.getsize(path + ".part"), size))
    os.replace(path + ".part", path)


AF3_TERMS = ("AlphaFold 3's parameters are Google DeepMind's, for academic, non-commercial use only, under the\n"
             "AlphaFold 3 Model Parameters Terms of Use and Prohibited Use Policy:\n"
             "  https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md\n"
             "  https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md")


def accepted_af3_terms():
    """The user's acceptance of DeepMind's AF3 terms: LOCALFOLD_ACCEPT_MODEL_TERMS naming alphafold3 (what
    tools/build_site.py reads too, and what the CUDA worker sets once the page's terms dialog has been
    accepted), or a yes at a prompt."""
    named = {n.strip() for n in os.environ.get("LOCALFOLD_ACCEPT_MODEL_TERMS", "").split(",")}
    if "alphafold3" in named:
        return True
    print(AF3_TERMS, file=sys.stderr)
    if not sys.stdin.isatty():
        return False
    return input("Do you accept these terms? [y/N] ").strip().lower() in ("y", "yes")


def fetch_blobs(names, repo):
    for name in names:
        if name not in BLOBS:
            sys.exit(f"af3-any-model has no blob named {name!r} here: {', '.join(BLOBS)}")
        if name == "af3" and not accepted_af3_terms():
            sys.exit("AlphaFold 3's parameters need its terms accepted first: set LOCALFOLD_ACCEPT_MODEL_TERMS=alphafold3")
        dest = os.path.join(repo, "af3am-" + name)
        os.makedirs(dest, exist_ok=True)
        path = os.path.join(dest, os.path.basename(BLOBS[name]))
        with open(os.path.join(dest, ".fetch.lock"), "w") as lock:     # (one fetch a blob at a time)
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not os.path.exists(path):
                fetch_ranged(AF3_ANY_MODEL + BLOBS[name], path, name)
        print(f"{name} -> {dest}")


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
    if "--af3-any-model" in sys.argv[1:]:
        return fetch_blobs(args, repo)
    known = families(os.path.join(repo, "shared", "bundles", "manifests", "index.js"))
    if not args:
        print("families:", " ".join(sorted(known)))
        return
    for name in args:
        if name not in known:
            sys.exit(f"no family {name!r} in shared/bundles/manifests/index.js")
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
