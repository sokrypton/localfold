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

🔴 ONE DIFFERENCE IS EXPECTED AND ASSERTED, NOT EXCUSED: 5O45's macrocycle closes through a
non-standard residue (CCS CE - PHE N) whose dictionary entry has no leaving atom for that bond,
and atomworks' link chemistry (`resolve_link_chemistry`, 1,247 lines of RDKit valence rules)
lowers the carboxylate's double bond and puts -1 on OZ1. That is not ported; the entry below
requires EXACTLY that difference, so the day it is ported - or anything else moves - this fails.
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
EXPECTED = {
    "5o45 fresh": ["charge: 1 of 1171 (first #1162 B/CCS13/OZ1: upstream -1.0, page 0)"],
}


def upstream_atoms(path: Path, annotated: bool) -> dict:
    import numpy as np
    from rfproteina.inference_utils.condition_spec import load_condition_spec_inputs

    try:
        a = load_condition_spec_inputs([str(path)], as_annotated_cif=annotated)
    except Exception as e:  # noqa: BLE001 - a refusal is an answer
        return {"raise": f"{type(e).__name__}: {e}"}
    a = a[a.element != "H"]
    out = {k: [str(v) if k in ("chain_id", "res_name", "ins_code", "atom_name", "element") else float(v)
               for v in getattr(a, k)] for k in FIELDS}
    out["coord"] = [[None if math.isnan(x) else round(float(x), 3) for x in c] for c in a.coord]
    return out


def page_atoms(cases: list[tuple[str, Path, bool]], ccd: dict[str, str]) -> dict:
    script = """
import { readFileSync } from "node:fs";
import { parseCifBlock } from "./shared/rfd4p/pipeline/cif.js";
import { loadConditionSpecInput } from "./shared/rfd4p/pipeline/spec.js";
const { cases, ccd } = JSON.parse(readFileSync(0, "utf8"));
const ccdBlocks = new Map(Object.entries(ccd).map(([id, text]) => [id, parseCifBlock(text)]));
const out = {};
for (const [key, path, annotated] of cases) {
  try {
    const a = loadConditionSpecInput(readFileSync(path), { asAnnotatedCif: annotated, ccdBlocks });
    const keep = [...Array(a.nAtoms).keys()].filter((i) => a.element[i] !== "H");
    const pick = (k) => keep.map((i) => a[k][i]);
    out[key] = {
      chain_id: pick("chain_id"), res_id: pick("res_id"), res_name: pick("res_name"),
      ins_code: pick("ins_code"), atom_name: pick("atom_name"), element: pick("element"),
      charge: pick("charge"), occupancy: pick("occupancy"),
      coord: keep.map((i) => a.coord[i].map((x) => (Number.isFinite(x) ? Math.round(x * 1000) / 1000 : null))),
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
    args = ap.parse_args()
    import zstandard

    with tempfile.TemporaryDirectory() as tmp:
        cases = []
        for name in EXAMPLES:
            raw = (REPO / "web" / "examples" / f"{name}.cif.zst").read_bytes()
            path = Path(tmp) / f"{name}.cif"
            path.write_bytes(zstandard.ZstdDecompressor().decompressobj().decompress(raw))
            cases += [(f"{name} annotated", path, True), (f"{name} fresh", path, False)]
        import gzip

        for packed in sorted(RAW.glob("*.cif.zst")):
            path = Path(tmp) / packed.name.removesuffix(".zst")
            path.write_bytes(zstandard.ZstdDecompressor().decompressobj().decompress(packed.read_bytes()))
            cases += [(f"{path.name.removesuffix('.cif')} annotated", path, True),
                      (f"{path.name.removesuffix('.cif')} fresh", path, False)]
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
        if args.structures:
            for path in sorted(Path(args.structures).glob("*.cif")):
                cases += [(f"{path.stem} annotated", path, True), (f"{path.stem} fresh", path, False)]
        if args.only:
            cases = [c for c in cases if args.only in c[0]]
        page = page_atoms(cases, ccd_entries(sorted({p for _, p, _ in cases})))
        bad = 0
        for key, path, annotated in cases:
            d = differences(upstream_atoms(path, annotated), page[key])
            expected = EXPECTED.get(key, [])
            ok = d == expected
            bad += not ok
            print(f"  {key:32s} {('exact' if not d else 'the expected difference') if ok else 'DIFFERS'}")
            for line in d:
                print(f"      {line}")
    print(f"{len(cases)} loads, {bad} differing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
