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

🔴 AND THE CENTRING OFFSET IS ASSERTED NOW, NOT REPORTED. `shared/rfd4p/output.js` adds
`inputAtoms.center_shift` back after aligning - the centroid the featuriser subtracted when it featurised the
conditioning - and for a while no native port could, because the featuriser computed it and wrote it nowhere.
It exports it (`meta/centerShift*`), metal/rfd4p adds it back, and every case is held to the page's text
outright. The unshifted comparison is kept as a DIAGNOSIS of failure rather than a second way to pass: if a
file differs and removing the shift makes it identical, the offset is not reaching the writer, and the tool
says so instead of reporting a near-miss as expected.

🔴 AND A LAST-PLACE DOUBLE IS NAMED FOR WHAT IT IS. Two motif RMSDs differ at 1.3e-16 and 2.3e-16 relative -
one bit of a double, C++ against V8 inside a superposition - and a gate that calls that "the centring offset"
is a gate that will hide the offset coming back. They are reported separately, with the relative size.
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
    "yaml_5o45_binder": ["3d_coordinate_conditioning", "internal_coordinate_conditioning"],
    "partial_diffusion_1prw": ["3d_coordinate_conditioning"],
    "inverse_folding_1prw": ["3d_coordinate_conditioning"],
}
# 🔴 EVERY CASE cuda/rfd4p/gate.py's WRITER ARM USES, not the six this started with. The four that were
# missing are the ones that reach what the others cannot: `yaml_5o45_binder` is the only INTERNAL-coordinate
# conditioning case (its own tag, and the only one whose motif reference comes from a per-group fit),
# `partial_diffusion_1prw` and `inverse_folding_1prw` start from a corrupted input rather than from noise and
# so exercise the encoder and `corruptBatch`'s draw order end to end, and `custom_cath_monomer` is a contig
# with no input structure at all - no anchors, so no alignment and no centring offset.
# 🔴 AND THE UNINDEXED ONE TWICE, at 6 steps and at 60, because at 6 no mapping can be grafted and the
# insertion head's predicted mapping goes unexercised - gate.py's own note, and this port's `--no-insertion`
# arm is the other half of it.
PRESETS = ["uncond_short", "ppi_pdl1", "motif_idx_1prw", "motif_unidx_1prw", "sm_sam", "dna_7m5w",
           "yaml_5o45_binder", "partial_diffusion_1prw", "inverse_folding_1prw", "custom_cath_monomer"]
CONVERGED = [("motif_unidx_1prw", 60)]


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

    writer_bad, metrics_bad, lastplace = [], [], []
    cases = [(pid, steps) for pid in presets] + [c for c in CONVERGED if c[0] in presets]
    for pid, steps in cases:
        r = run(["node", REPO / "cuda/rfd4p/featurise/preset-spec.mjs", pid])
        if r.returncode != 0:
            writer_bad.append(f"{pid}: {r.stderr.strip()[-120:]}")
            continue
        spec = json.loads(r.stdout)
        spec["nsteps"] = steps
        spec["metrics_tags"] = METRICS_TAGS + MOTIF_TAGS.get(pid, [])
        tag = pid if steps == int(option("steps", "6")) else f"{pid}-{steps}"
        job = work / f"{tag}.json"
        job.write_text(json.dumps(spec))
        dump = work / f"dump-{tag}"
        for ext in ("cif", "pdb"):
            out = work / f"{tag}.{ext}"
            g = run([BINARY, f"--job={job}", f"--bundle={bundle}", f"--work-dir={work}/w-{tag}",
                     f"--out={out}", "--seed=0", f"--dump-sample={dump}",
                     f"--metrics={dump}.metrics.json"])
            if g.returncode != 0:
                writer_bad.append(f"{tag}.{ext}: {g.stderr.strip()[-160:] or g.stdout.strip()[-160:]}")
                continue
            w = run(["node", REPO / "cuda/rfd4p/check-writer.mjs", f"--spec={job}", "--seed=0", dump, out])
            if w.returncode == 0:
                continue
            # It differs. Say WHY: if removing the centring offset from the page's side makes the two identical,
            # the offset is not reaching this writer - `meta/centerShift*` absent from the export, or the
            # add-back not running - which is a different repair from a wrong record or a wrong column.
            d = run(["node", REPO / "tools/rfd4p-writer-unshifted.mjs", f"--spec={job}", "--seed=0", dump, out])
            why = (" - and identical with the centring offset removed, so the offset is not being added back:"
                   " check meta/centerShift* in the export" if d.returncode == 0 else "")
            writer_bad.append(f"{tag}.{ext}: {w.stdout.strip().splitlines()[0] if w.stdout else 'failed'}{why}")
        # the metrics, native against the page's on the same sample: every value and the verdict
        native = Path(f"{dump}.metrics.json")
        if not native.exists():
            metrics_bad.append(f"{tag}: no native metrics")
            continue
        m = run(["node", REPO / "cuda/rfd4p/design-metrics.mjs", f"--spec={job}", "--seed=0", dump])
        if m.returncode != 0:
            metrics_bad.append(f"{tag}: {m.stderr.strip()[-160:]}")
            continue
        a, b = json.loads(native.read_text()), json.loads(m.stdout)
        for k in sorted(set(a["metrics"]) | set(b["metrics"])):
            x, y = a["metrics"].get(k), b["metrics"].get(k)
            if x == y:
                continue
            # 🔴 A LAST-PLACE DOUBLE IS NOT A DIFFERENCE IN THE PORT. A superposed RMSD goes through an eigen
            # solve, and C++ and V8 disagree in the final bit; reported with its size rather than asserted
            # equal, and the bound is tight enough (1e-12) that anything structural still fails.
            if isinstance(x, float) and isinstance(y, float) and abs(x - y) <= 1e-12 * max(1e-30, abs(y)):
                lastplace.append(f"{tag}:{k} ({abs(x - y) / max(1e-30, abs(y)):.1e})")
                continue
            metrics_bad.append(f"{tag}: {k} metal {x} page {y}")
        if a["preOracle"] != b["preOracle"]:
            metrics_bad.append(f"{tag}: the pre-oracle verdict differs")

    n = len(cases)
    print(f"writer   {n * 2} files against the page's text   "
          f"{'ok ' if not writer_bad else 'DIFFERS: ' + '; '.join(writer_bad[:4])}")
    print(f"metrics  {n} designs against the page's metrics  "
          f"{'ok ' if not metrics_bad else 'DIFFERS: ' + '; '.join(metrics_bad[:4])}")
    if lastplace:
        print(f"         {len(lastplace)} agreeing to the last place of a double rather than exactly: "
              f"{', '.join(lastplace[:6])}")
    return 1 if writer_bad or metrics_bad else 0


sys.exit(main())
