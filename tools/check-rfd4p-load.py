#!/usr/bin/env python3
"""Hold the page's structure LOADER to upstream's, atom for atom: `load_condition_spec_inputs`.

    ~/venv_rfd4p/bin/python tools/check-rfd4p-load.py [--structures DIR] [--verbose]

A condition spec's `input:` is loaded by atomworks with one of two presets - `ANNOTATED_CIF`
(`input_as_annotated_cif: true`: the file as written, its conditions kept) or `RCSB_WITH_METALS`
(the default: a raw structure PREPARED - waters and crystallization aids removed, MSE as MET, an
alternate location chosen per residue without clashes, unmodeled polymer residues and missing
heavy atoms added without coordinates, metal coordination bonds kept). Everything after the
load is gated elsewhere against the page; this asks whether the page loads what upstream loads.

Compared per heavy atom (upstream's hydrogens never reach the batch): chain, residue number and
name, insertion code, atom name, element, formal charge, occupancy and coordinates (NaN where
upstream has none), over the shipped examples and five raw RCSB entries (tools/fixtures/rcsb:
missing residues, truncated side chains, 603 alternate-location atoms, waters, ions, DNA, a
modified-residue macrocycle) in both modes, plus every `.cif` in `--structures`.

🔴 AND THE AUTHORED LINKS' PRODUCTS: 5O45's macrocycle closes through a non-standard residue
(CCS CE - PHE N) whose dictionary entry has no leaving atom for that bond, and atomworks' link
chemistry (`resolve_link_chemistry`) lowers the carboxylate's double bond and puts -1 on OZ1 -
which this port did not do until links.js, and which only a comparison of bonds as well as atoms
could see. tools/fixtures/rcsb-links holds ten of upstream's own link examples (the PDB entries
link_chemistry.py cites: a restored P=O, a lowered C=C, a moved C=O, a neutralised amide, a
carbonyl addition, an oxo lowered on S, a dropped terminal N, a selenolate, and two products with
no legal valence that both sides refuse), read raw - and a raw load is held to upstream on the
native featuriser too (`--unannotated`), the one load that builds link products.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from rfd4p_ccd import ccd_entries

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = ["5o45_cropped", "dna_7m5w", "motif_idx_1prw", "motif_unidx_1prw", "ppi_pdl1", "sm_sam",
            "uncond_short"]
FIELDS = ["chain_id", "res_id", "res_name", "ins_code", "atom_name", "element", "charge", "occupancy"]
RAW = REPO / "tools" / "fixtures" / "rcsb"
NATIVE = REPO / "cuda" / "rfd4p" / "featurise" / "rfd4p-featurise"
# Upstream's own link examples (the PDB entries link_chemistry.py cites), one per repair the port
# makes and two it refuses: read raw only, the one load that builds link products.
LINKS = REPO / "tools" / "fixtures" / "rcsb-links"
EXPECTED: dict[str, list[str]] = {}


def upstream_atoms(path: Path, annotated: bool | str) -> dict:
    """`annotated`: True / False for a condition spec's `input:`, "direct" for a CIF handed to the
    generator itself (`DesignProblemDataset`, which takes the file's biological assembly 1)."""
    import numpy as np
    from rfproteina.inference_utils.condition_spec import load_condition_spec_inputs
    from rfproteina.inference_utils.datasets import DesignProblemDataset

    try:
        if annotated == "direct":
            a = DesignProblemDataset._load_from_path(None, path)["atom_array"]
        else:
            a = load_condition_spec_inputs([str(path)], as_annotated_cif=annotated)
    except Exception as e:  # noqa: BLE001 - a refusal is an answer
        return {"raise": f"{type(e).__name__}: {e}"}
    a = a[a.element != "H"]
    out = {k: [str(v) if k in ("chain_id", "res_name", "ins_code", "atom_name", "element") else float(v)
               for v in getattr(a, k)] for k in FIELDS}
    out["coord"] = [[None if math.isnan(x) else round(float(x), 3) for x in c] for c in a.coord]
    out["bonds"] = sorted([min(int(i), int(j)), max(int(i), int(j)), int(t)] for i, j, t in a.bonds.as_array()) \
        if a.bonds is not None else []
    return out


def page_atoms(cases: list[tuple[str, Path, bool]], ccd: dict[str, str]) -> dict:
    script = """
import { readFileSync } from "node:fs";
import { parseCifBlock, parseCifToAtomArray } from "./shared/rfd4p/pipeline/cif.js";
import { loadConditionSpecInput, loadDirectInput } from "./shared/rfd4p/pipeline/spec.js";
const { cases, ccd } = JSON.parse(readFileSync(0, "utf8"));
const ccdBlocks = new Map(Object.entries(ccd).map(([id, text]) => [id, parseCifBlock(text)]));
const out = {};
for (const [key, path, annotated] of cases) {
  try {
    const a = annotated === "direct"
      ? loadDirectInput(readFileSync(path), { ccdBlocks })
      : loadConditionSpecInput(readFileSync(path), { asAnnotatedCif: annotated, ccdBlocks });
    const keep = [...Array(a.nAtoms).keys()].filter((i) => a.element[i] !== "H");
    const pick = (k) => keep.map((i) => a[k][i]);
    out[key] = {
      chain_id: pick("chain_id"), res_id: pick("res_id"), res_name: pick("res_name"),
      ins_code: pick("ins_code"), atom_name: pick("atom_name"), element: pick("element"),
      charge: pick("charge"), occupancy: pick("occupancy"),
      coord: keep.map((i) => a.coord[i].map((x) => (Number.isFinite(x) ? Math.round(x * 1000) / 1000 : null))),
      bonds: (() => {
        const at = new Map(keep.map((g, k) => [g, k]));
        return (a.bonds ?? []).filter(([i, j]) => at.has(i) && at.has(j))
          .map(([i, j, t]) => [Math.min(at.get(i), at.get(j)), Math.max(at.get(i), at.get(j)), t])
          .sort((x, y) => x[0] - y[0] || x[1] - y[1] || x[2] - y[2]);
      })(),
    };
  } catch (e) { out[key] = { raise: e.message }; }
}
process.stdout.write(JSON.stringify(out));
"""
    r = subprocess.run(["node", "--input-type=module", "-e", script], cwd=REPO, capture_output=True, text=True,
                       input=json.dumps({"cases": [[k, str(p), a] for k, p, a in cases], "ccd": ccd}))
    if r.returncode != 0:
        sys.exit(f"node failed: {r.stderr[-2000:]}")
    return json.loads(r.stdout)


def native_atoms(path: Path, ccd_file: Path) -> dict:
    """The native featuriser's load of a RAW structure (`--unannotated`, a job's load), its
    dictionary entries the same ones the page is handed."""
    r = subprocess.run([NATIVE, "x", f"--dump-cif={path}", "--unannotated", f"--ccd={ccd_file}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {"raise": r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "refused"}
    rows = [line.split("\t")[1:] for line in r.stdout.splitlines() if line.startswith("ua\t")]
    bonds = [[int(x) for x in line.split("\t")[1:]] for line in r.stdout.splitlines() if line.startswith("ub\t")]
    num = lambda v: None if v in ("nan", "-nan") else round(float(v), 3)
    return {
        "res_name": [w[0] for w in rows], "atom_name": [w[1] for w in rows],
        "coord": [[num(w[2]), num(w[3]), num(w[4])] for w in rows],
        "chain_id": [w[5] for w in rows], "element": [w[6] for w in rows], "charge": [int(w[7]) for w in rows],
        "res_id": [int(w[8]) for w in rows], "ins_code": [w[9] for w in rows], "occupancy": [float(w[10]) for w in rows],
        "bonds": sorted([min(i, j), max(i, j), t] for i, j, t in bonds),
    }


def differences(up: dict, page: dict) -> list[str]:
    if "raise" in up or "raise" in page:
        if "raise" in up and "raise" in page:
            return []
        return [f"upstream {'refuses: ' + up['raise'] if 'raise' in up else 'loads'}, page "
                f"{'refuses: ' + page['raise'] if 'raise' in page else 'loads'}"]
    nu, np_ = len(up["atom_name"]), len(page["atom_name"])
    if nu != np_:
        # Say where the two first part, which is what names the step.
        for i in range(min(nu, np_)):
            if (up["chain_id"][i], up["res_id"][i], up["atom_name"][i]) != (
                    page["chain_id"][i], page["res_id"][i], page["atom_name"][i]):
                break
        else:
            i = min(nu, np_)
        def at(side, i):
            return "end" if i >= len(side["atom_name"]) else (
                f"{side['chain_id'][i]}/{side['res_name'][i]}{side['res_id'][i]:g}/{side['atom_name'][i]}")
        return [f"{nu} heavy atoms upstream, {np_} here; first apart at #{i}: upstream {at(up, i)}, "
                f"page {at(page, i)}"]
    out = []
    for k in FIELDS:
        bad = [i for i in range(nu) if (up[k][i] != page[k][i] and not (
            isinstance(up[k][i], float) and isinstance(page[k][i], (int, float)) and up[k][i] == float(page[k][i])))]
        if bad:
            i = bad[0]
            out.append(f"{k}: {len(bad)} of {nu} (first #{i} {up['chain_id'][i]}/{up['res_name'][i]}"
                       f"{up['res_id'][i]:g}/{up['atom_name'][i]}: upstream {up[k][i]!r}, page {page[k][i]!r})")
    # 🔴 AND THE BONDS, which every featurised bond and path length is built from: compared as a
    # set of (atom, atom, type). This checker compared atoms alone, so a polymer bond the page
    # never made (52 to 52A, an insertion code) loaded "exact".
    ub, pb = {tuple(b) for b in up["bonds"]}, {tuple(b) for b in page["bonds"]}
    if ub != pb:
        def name(i):
            return f"{up['chain_id'][i]}/{up['res_name'][i]}{up['res_id'][i]:g}/{up['atom_name'][i]}"
        only_up, only_page = sorted(ub - pb), sorted(pb - ub)
        out.append(f"bonds: {len(only_up)} only upstream {[(name(b[0]), name(b[1]), b[2]) for b in only_up[:2]]}, "
                   f"{len(only_page)} only here {[(name(b[0]), name(b[1]), b[2]) for b in only_page[:2]]}")
    bad = [i for i in range(nu) if any(
        (a is None) != (b is None) or (a is not None and abs(a - b) > 1.5e-3)
        for a, b in zip(up["coord"][i], page["coord"][i]))]
    if bad:
        i = bad[0]
        out.append(f"coord: {len(bad)} of {nu} (first #{i} {up['chain_id'][i]}/{up['res_name'][i]}"
                   f"{up['res_id'][i]:g}/{up['atom_name'][i]}: upstream {up['coord'][i]}, page {page['coord'][i]})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", help="a directory of raw .cif files to load in both modes")
    ap.add_argument("--only")
    ap.add_argument("--no-native", action="store_true", help="the page alone")
    args = ap.parse_args()
    import zstandard

    with tempfile.TemporaryDirectory() as tmp:
        cases = []
        for name in EXAMPLES:
            raw = (REPO / "web" / "examples" / f"{name}.cif.zst").read_bytes()
            path = Path(tmp) / f"{name}.cif"
            path.write_bytes(zstandard.ZstdDecompressor().decompressobj().decompress(raw))
            cases += [(f"{name} annotated", path, True), (f"{name} fresh", path, False),
                      (f"{name} direct", path, "direct")]
        import gzip

        for packed in sorted(RAW.glob("*.cif.zst")):
            path = Path(tmp) / packed.name.removesuffix(".zst")
            path.write_bytes(zstandard.ZstdDecompressor().decompressobj().decompress(packed.read_bytes()))
            cases += [(f"{path.name.removesuffix('.cif')} annotated", path, True),
                      (f"{path.name.removesuffix('.cif')} fresh", path, False),
                      (f"{path.name.removesuffix('.cif')} direct", path, "direct")]
        # The other formats upstream takes: gzip, and RCSB's own BinaryCIF (its encoder, not one
        # written here) - each must load exactly as the mmCIF does on both sides.
        gz = Path(tmp) / "1brs.cif.gz"
        gz.write_bytes(gzip.compress((Path(tmp) / "1brs.cif").read_bytes()))
        cases += [("1brs gzip fresh", gz, False)]
        for packed in sorted(RAW.glob("*.bcif.zst")):
            path = Path(tmp) / packed.name.removesuffix(".zst")
            path.write_bytes(zstandard.ZstdDecompressor().decompressobj().decompress(packed.read_bytes()))
            stem = path.name.removesuffix(".bcif")
            cases += [(f"{stem} bcif annotated", path, True), (f"{stem} bcif fresh", path, False)]
        for packed in sorted(LINKS.glob("*.cif.zst")):
            path = Path(tmp) / f"link-{packed.name.removesuffix('.zst')}"
            path.write_bytes(zstandard.ZstdDecompressor().decompressobj().decompress(packed.read_bytes()))
            cases += [(f"{packed.name.removesuffix('.cif.zst')} link fresh", path, False)]
        if args.structures:
            for path in sorted(Path(args.structures).glob("*.cif")):
                cases += [(f"{path.stem} annotated", path, True), (f"{path.stem} fresh", path, False)]
        if args.only:
            cases = [c for c in cases if args.only in c[0]]
        ccd = ccd_entries(sorted({p for _, p, _ in cases}))
        page = page_atoms(cases, ccd)
        # The native featuriser reads the same entries, as one dictionary file.
        ccd_file = Path(tmp) / "ccd.cif"
        ccd_file.write_text("".join(ccd.values()))
        with_native = NATIVE.exists() and not args.no_native
        bad = 0
        for key, path, annotated in cases:
            up = upstream_atoms(path, annotated)
            sides = {"page": page[key]}
            # A raw structure's load is the one that builds link products; the native port's
            # annotated and direct loads are held to the page by gate.py's `cif` arm.
            if with_native and annotated is False:
                sides["native"] = native_atoms(path, ccd_file)
            wrong = {}
            for side, got in sides.items():
                d = differences(up, got)
                if d != EXPECTED.get(key, []):
                    wrong[side] = d
            bad += bool(wrong)
            print(f"  {key:32s} {'exact' if not wrong else 'DIFFERS'}")
            for side, d in wrong.items():
                for line in d:
                    print(f"      {side}: {line}")
    print(f"{len(cases)} loads, {bad} differing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
