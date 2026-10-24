"""Does our copy of RFD4-Proteina's tensor manifest still describe the shards being served?

    python3 tools/check-rfd4p-manifest.py            # compare against the pinned commit and upstream HEAD
    python3 tools/check-rfd4p-manifest.py --write    # re-copy from upstream HEAD and move the pin

🔴 WHY THIS IS A GATE AND NOT A CURIOSITY. The ~420 MB of INT8 shards are NOT in
this repository: they are fetched from `sokrypton/rfd4-proteina-webgpu`'s Pages
URL at run time (see `model-rfd4p-int8/SOURCE.md`). What IS here is `manifest.json`,
which records the SHARD AND BYTE OFFSET of every one of the 1,257 tensors - so
the file we keep and the files we fetch have to describe each other.

If that port re-exports its weights, the shards it serves change and our
manifest still names the old offsets. Every tensor would then be read from the
wrong place in a file that downloads perfectly, with the right length and no
error anywhere. That is not a hypothetical failure mode in this repository: it
is what `model-boltz2-int5/` is doing right now against its own manifest module,
which `tools/build_site.py` refuses to build over.

🔴 AND IT CANNOT BE CHECKED LOCALLY, which is why this reaches the network: the
whole point is whether a file HERE agrees with files THERE. Offline it says so
and exits 0, because a missing network is not a drift.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "sokrypton/rfd4-proteina-webgpu"
# The files we keep a copy of, as paths inside that repository.
# 🔴 OUR PATH -> UPSTREAM'S. The local directory was flattened and renamed when
# the hosting decision was made (the shards are served by the port, never kept
# here), and upstream's own layout is NOT ours to change - so the two sides are
# written out as a mapping rather than assumed equal.
COPIED = {
    "model-rfd4p-int8/config.json": "weights/config.json",
    "model-rfd4p-int8/manifest.json": "weights/int8/manifest.json",
}
SOURCE = os.path.join(ROOT, "model-rfd4p-int8/SOURCE.md")


def digest(data):
    return hashlib.sha256(data).hexdigest()[:16]


def pinned_commit():
    """The commit `model-rfd4p-int8/SOURCE.md` says our copy came from."""
    text = open(SOURCE).read()
    found = re.search(r"^- commit: `([0-9a-f]{7,40})`", text, re.M)
    if found is None:
        sys.exit(f"🔴 {SOURCE} does not record a commit. It is the only thing making drift detectable.")
    return found.group(1)


def upstream(path, ref):
    """One file from that repository at `ref`, or None if the network is not there."""
    url = f"https://raw.githubusercontent.com/{REPO}/{ref}/{path}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        sys.exit(f"🔴 {path} is not at {REPO}@{ref[:8]} ({error.code})."
                 " Either the pin names a commit that no longer has it, or the layout moved.")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def head_commit():
    probe = subprocess.run(["gh", "api", f"repos/{REPO}/commits/HEAD", "--jq", ".sha"],
                           capture_output=True, text=True)
    return probe.stdout.strip() if probe.returncode == 0 and probe.stdout.strip() else None


def main():
    write = "--write" in sys.argv
    pin = pinned_commit()
    head = head_commit()
    print(f"pinned at {pin[:8]}" + (f", upstream HEAD {head[:8]}" if head else ", upstream HEAD unknown"))

    reference = head if (write and head) else pin
    drifted, missing = [], []
    for path, theirPath in COPIED.items():
        local_path = os.path.join(ROOT, path)
        if not os.path.exists(local_path):
            missing.append(path)
            continue
        local = open(local_path, "rb").read()
        # 🔴 FETCHED BY UPSTREAM'S PATH, COMPARED AGAINST OURS. The local
        # directory was renamed and flattened; theirs was not, and is not ours
        # to change. Asking for our own path here would 404 and read as "the
        # file is gone from upstream" rather than as a bug in this script.
        theirs = upstream(theirPath, reference)
        if theirs is None:
            print("offline, so nothing was compared - a missing network is not a drift")
            return 0
        if local == theirs:
            print(f"  ok      {path} ({digest(local)})")
        elif write:
            open(local_path, "wb").write(theirs)
            print(f"  updated {path} ({digest(local)} -> {digest(theirs)})")
        else:
            drifted.append(path)
            print(f"  DRIFTED {path}: ours {digest(local)}, {reference[:8]}'s {digest(theirs)}")

    if missing:
        print("🔴 not in this checkout: " + ", ".join(missing))
        return 1

    if write:
        if head and head != pin:
            open(SOURCE, "w").write(open(SOURCE).read().replace(f"`{pin}`", f"`{head}`"))
            print(f"moved the pin to {head[:8]}")
        return 0

    # 🔴 A MOVED UPSTREAM IS A WARNING; A DIFFERING FILE IS A FAILURE. Upstream
    # committing anything at all does not touch the weights, so HEAD moving on
    # its own means nothing. What matters is whether THESE files changed.
    if drifted:
        print(f"\n🔴 {len(drifted)} file(s) differ from {REPO}@{reference[:8]}."
              "\nThe shards fetched at run time are that port's CURRENT ones, so a manifest that"
              "\ndisagrees reads every tensor from the wrong offset and says nothing."
              "\nRe-copy with --write, and re-run the rfd4p tests after.")
        return 1

    if head and head != pin:
        print(f"\nupstream has moved to {head[:8]} but the copied files are unchanged;"
              " --write moves the pin.")
    print("\nour copy describes the shards being served")
    return 0


if __name__ == "__main__":
    sys.exit(main())
