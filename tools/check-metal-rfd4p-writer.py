#!/usr/bin/env python3
"""The Metal port's structure writer and design metrics against the PAGE's, on the same sample.

    python3 tools/check-metal-rfd4p-writer.py [--bundle=<dir>] [--work=<dir>] [--presets=a,b] [--steps=6]

🔴 EVERY OTHER ARM THIS PORT HAS SCORES A STRUCTURE AS GEOMETRY, AND THE WRITER IS WRONG WHERE GEOMETRY CANNOT
LOOK. metal/rfd4p kept its own records-and-writers pair that nothing compared with anything: it named every atom
by its atom37 SLOT rather than the residue's own atom order, left out the PDB's HEADER/TITLE and every TER,
dropped the two mmCIF annotation columns, printed coordinates through `snprintf` where the reference rounds with
`toFixed`, carried no source component, and did not align the design to the input's frame at all - so a motif
design came out in the sampler's centred frame and superposed on nothing. Ten green gate arms and a correct
CA-CA spacing throughout. This is cuda/rfd4p/gate.py's `writer` and `metrics` arms, pointed at the Metal binary.

`--dump-sample=` writes what `processSampleToAtoms` takes; check-writer.mjs runs the page's writer on it and
requires the SAME TEXT, and design-metrics.mjs its metrics, value for value, with the pre-oracle verdict.

🔴 AND THE CENTRING OFFSET IS REPORTED, NOT FAILED. `shared/rfd4p/output.js` adds `inputAtoms.center_shift` back
after aligning - the centroid the featuriser subtracted when it featurised the conditioning - and no native port
can: the featuriser exports `g/coords` and `g/condCoord` already centred (centroids 1e-7) and the offset with
them. So each case is compared BOTH ways, and what is asserted is that removing the shift makes the two files
identical - which says the writer agrees about everything that is this port's to get right, and names the one
thing that is not. A case whose shift is zero is held to the page's text outright.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BINARY = REPO / "metal" / "rfd4p" / "localfold-rfd4p"
# Every tag a ported metric gates on, so the metrics arm computes all of them; the writer reads none of it.
METRICS_TAGS = ["design", "structure_prediction", "small_molecule_only", "hotspot_conditioning",
                "partial_diffusion", "insertion_rmsd_filter", "motif_rmsd_filter", "ligand_clash_filter"]
# 🔴 AND THE TAGS THAT NAME A MOTIF ONLY WHERE THE PRESET HAS ONE, as gate.py's own note says: a motif RMSD on a
# design with no coordinate-conditioned atom raises upstream, and here.
MOTIF_TAGS = {
    "ppi_pdl1": ["3d_coordinate_conditioning"],
    "motif_idx_1prw": ["3d_coordinate_conditioning"],
    "motif_unidx_1prw": ["unindexed", "3d_coordinate_conditioning"],
    "dna_7m5w": ["3d_coordinate_conditioning"],
}
PRESETS = ["uncond_short", "ppi_pdl1", "motif_idx_1prw", "motif_unidx_1prw", "sm_sam", "dna_7m5w"]


def option(name, fallback):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a[len(name) + 3:]
    return fallback


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=REPO, **kw)


def main():
    bundle = Path(option("bundle", REPO / "model-rfd4p-int8"))
    work = Path(option("work", "/tmp/metal-rfd4p-writer"))
    steps = int(option("steps", "6"))
    presets = [p for p in option("presets", ",".join(PRESETS)).split(",") if p]
    work.mkdir(parents=True, exist_ok=True)
    if not BINARY.exists():
        print(f"no {BINARY.relative_to(REPO)} - build it with `bash metal/build.sh rfd4p`")
        return 2

    writer_bad, metrics_bad, shifted = [], [], []
    for pid in presets:
        r = run(["node", REPO / "cuda/rfd4p/featurise/preset-spec.mjs", pid])
        if r.returncode != 0:
            writer_bad.append(f"{pid}: {r.stderr.strip()[-120:]}")
            continue
        spec = json.loads(r.stdout)
        spec["nsteps"] = steps
        spec["metrics_tags"] = METRICS_TAGS + MOTIF_TAGS.get(pid, [])
        job = work / f"{pid}.json"
        job.write_text(json.dumps(spec))
        dump = work / f"dump-{pid}"
        for ext in ("cif", "pdb"):
            out = work / f"{pid}.{ext}"
            g = run([BINARY, f"--job={job}", f"--bundle={bundle}", f"--work-dir={work}/w-{pid}",
                     f"--out={out}", "--seed=0", f"--dump-sample={dump}",
                     f"--metrics={dump}.metrics.json"])
            if g.returncode != 0:
                writer_bad.append(f"{pid}.{ext}: {g.stderr.strip()[-160:] or g.stdout.strip()[-160:]}")
                continue
            w = run(["node", REPO / "cuda/rfd4p/check-writer.mjs", f"--spec={job}", "--seed=0", dump, out])
            if w.returncode == 0:
                continue
            # The one allowed difference: the centring offset no native port is given. Removing it must make
            # the two files identical - anything left over is this port's.
            s = run(["node", REPO / "tools/rfd4p-writer-unshifted.mjs", f"--spec={job}", "--seed=0", dump, out])
            if s.returncode == 0:
                shifted.append(f"{pid}.{ext}")
            else:
                writer_bad.append(f"{pid}.{ext}: {s.stdout.strip().splitlines()[0] if s.stdout else w.stdout.strip().splitlines()[0]}")
        # the metrics, native against the page's on the same sample: every value and the verdict
        native = Path(f"{dump}.metrics.json")
        if not native.exists():
            metrics_bad.append(f"{pid}: no native metrics")
            continue
        m = run(["node", REPO / "cuda/rfd4p/design-metrics.mjs", f"--spec={job}", "--seed=0", dump])
        if m.returncode != 0:
            metrics_bad.append(f"{pid}: {m.stderr.strip()[-160:]}")
            continue
        a, b = json.loads(native.read_text()), json.loads(m.stdout)
        for k in sorted(set(a["metrics"]) | set(b["metrics"])):
            x, y = a["metrics"].get(k), b["metrics"].get(k)
            if x == y:
                continue
            # 🔴 A SUPERPOSED RMSD INHERITS THE CENTRING OFFSET, so it is reported at its size rather than
            # asserted equal: the motif reference is in the input's frame and the design is a shift away.
            if isinstance(x, float) and isinstance(y, float) and abs(x - y) <= 1e-6 * max(1.0, abs(y)):
                shifted.append(f"{pid}:{k}")
                continue
            metrics_bad.append(f"{pid}: {k} metal {x} page {y}")
        if a["preOracle"] != b["preOracle"]:
            metrics_bad.append(f"{pid}: the pre-oracle verdict differs")

    n = len(presets)
    print(f"writer   {n * 2} files against the page's text   "
          f"{'ok ' if not writer_bad else 'DIFFERS: ' + '; '.join(writer_bad[:4])}")
    print(f"metrics  {n} designs against the page's metrics  "
          f"{'ok ' if not metrics_bad else 'DIFFERS: ' + '; '.join(metrics_bad[:4])}")
    if shifted:
        print(f"         {len(shifted)} identical only once the unexported centring offset is removed: "
              f"{', '.join(shifted[:8])}")
    return 1 if writer_bad or metrics_bad else 0


sys.exit(main())
