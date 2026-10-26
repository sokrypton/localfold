#!/usr/bin/env python3
"""Hold both ports' selection language to UPSTREAM's, atom for atom, on the shipped structures.

    ~/venv_rfd4p/bin/python tools/check-rfd4p-query.py [--binary cuda/rfd4p/featurise/rfd4p-featurise]

Each query in the corpus is evaluated three ways over each of the seven `web/examples/*.cif.zst`
structures: by atomworks itself (`rfproteina.utils.mask_spec.resolve_mask`, the function every
condition spec's selection goes through upstream), by the page's `evalSelection`
(shared/rfd4p/pipeline/query.js) and by the native featuriser's (`--dump-cif= --select=`). The
three masks must be identical, and a query upstream refuses must be refused by both ports.

🔴 AN ORACLE, NOT A READING. The evaluator this replaced was written from a reading of the
language and was wrong in ways no reading caught - Python's precedence (`&` binds tighter than a
comparison), chained comparisons, trailing text, names that are not names - and the native one
took a union of `sel(...)` and nothing else. The corpus is chosen to exercise every node type,
every derived annotation and every refusal; atomworks decides what each one means.

The structures are loaded upstream as a job loads them (`load_condition_spec_inputs`, annotated
CIF), which gives the same atoms in the same order as both readers here and the instance ids
(`transformation_id`, `*_iid`) that a bare parse lacks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from rfd4p_ccd import ccd_entries

REPO = Path(__file__).resolve().parent.parent

STRUCTURES = ["5o45_cropped", "dna_7m5w", "motif_idx_1prw", "motif_unidx_1prw", "ppi_pdl1",
              "sm_sam", "uncond_short"]
# Raw RCSB entries: ions, waters, DNA, modified residues - what the shipped examples lack.
RAW = ["rcsb/1brs", "rcsb/1prw", "rcsb/5o45", "rcsb/7m5w"]

DERIVED = [
    "is_protein", "is_nucleic_acid", "is_dna", "is_rna", "is_polymer", "is_ligand",
    "is_standard_aa", "is_standard_or_unknown_aa", "is_standard_unknown_or_masked_aa",
    "is_standard_rna", "is_standard_dna", "is_standard_or_unknown_rna",
    "is_standard_or_unknown_dna", "is_standard_unknown_or_masked_rna",
    "is_standard_unknown_or_masked_dna", "is_protein_backbone", "is_protein_sidechain",
    "is_rna_backbone", "is_dna_backbone", "is_nucleic_acid_backbone", "is_pyrimidine",
    "is_purine", "is_tip_atom", "is_metal", "is_carbohydrate", "is_res_start", "is_chain_start",
    "is_token_start", "res_has_tip_atom",
]

CORPUS = [
    # The path DSL.
    "sel('A')", "sel('B')", "sel('*/*/*/CA')", "sel('A/*/[54,56]/*')", "sel('B/*/[30-20]/*')",
    "sel('B/*/[5-10, 40, 60-62]/*')", "sel('B/*/[ 5 , 7 ]/*')", "sel('B/*/7')", "sel('B/*/0007')",
    "sel('*/*/*/[N,CA,C,O]')", "sel('A/*/*/*/1')", "sel('A/*/*/*/2')", "sel('[A,B]/*/*/CA')",
    "sel('A/*/5, B/*/7')", "sel('A/ALA')", "sel('*/*/*/[C1\\',C2\\']')", "sel(\"*/*/*/C1'\")",
    "sel('')", "sel('*')", "sel('A//7')",
    "sel('A/*/1-3')", "sel('A/*/[]/*')", "sel('A/*/[1,2/*')", "sel('A/*/x/*')",
    # Attributes, comparisons and Python's precedence.
    "chain_id == 'A'", "chain_id != 'A'", "res_name in ['ALA', 'GLY', 'LYS']",
    "res_name not in ('ALA', 'GLY')", "res_id >= 50", "2 <= res_id <= 60", "res_id < 10 or res_id > 100",
    "atom_name == 'CA' and chain_id == 'B'", "not (atom_name == 'CA')", "~(res_id > 5)",
    "(chain_id == 'A') & (atom_name == 'CA')", "(res_id > 5) | (atom_name == 'N')",
    "res_id >= 2 & chain_id == 'A'", "element == 'C'", "hetero", "~hetero", "b_factor > 30",
    "occupancy < 1", "x > 0", "z <= -5.5", "res_id == '5'", "res_id == 5.0", "res_id in [5, 6, '7']",
    "res_id != res_id", "x == x", "chain_id == chain_id", "-res_id < -50", "+res_id > 50",
    "True", "False", "1", "0", "'A'", "''", "None", "[1, 2]",
    # The derived annotations, alone and in combination.
    *DERIVED,
    "is_protein_backbone & sel('B')", "is_tip_atom | is_metal", "res_name == 'CA'", "sel('*/CA')",
    "is_ligand & ~is_metal", "res_name in ['HOH', 'CA', 'DA']", "within_chain_res_idx < 3",
    "res_min_occupancy < 1", "token_min_occupancy == 1", "token_id < 5",
    # Functions.
    "has_bonds()", "~has_bonds()", "has_nan_coord()", "~has_nan_coord()",
    "sel('A/*/*/*') and not has_nan_coord()", "sa_mask('C_CRD')", "sa_mask('C_SEQ')",
    "sa_mask('C_IDX')", "sa_mask('C_HB')", "sa_mask('C_SEC')", "sa_mask('C_CRDGRP')",
    "sa_mask('condition_coordinate_1_atom')", "sa_mask('C_CATH')",
    "apply_and_spread('residue', atom_name == 'CA', 'any')",
    "apply_and_spread('residue', b_factor, 'max') > 40",
    "apply_and_spread('chain', is_protein, 'all')", "apply_and_spread('system', res_id, 'min') == res_id",
    "apply_and_spread('token', occupancy, 'mean') < 1",
    # Refusals: every one of these raises upstream.
    "foo", "A", "ALA", "res_name == 'ALA' garbage", "sel('A', 'B')", "sel(5)", "frobnicate()",
    "has_bonds(1)", "res_id + 1 > 2", "res_id is 5", "chain_id.lower()", "x[0]", "res_id >",
    "apply_and_spread('atom', hetero, 'any')", "apply_and_spread('residue', hetero, 'median')",
    "res_name < 5", "sa_mask('C_BP')",
]


def upstream_masks(cifs: dict[str, Path]) -> dict:
    from rfproteina.inference_utils.condition_spec import load_condition_spec_inputs
    from rfproteina.utils.mask_spec import resolve_mask

    out = {}
    for name, path in cifs.items():
        arr = load_condition_spec_inputs([str(path)], as_annotated_cif=True)
        out[name] = {}
        for q in CORPUS:
            try:
                m = resolve_mask(arr.copy(), q)
                out[name][q] = "".join("1" if v else "0" for v in m)
            except Exception as e:  # noqa: BLE001 - every refusal is a refusal
                out[name][q] = f"raise:{type(e).__name__}"
    return out


def js_masks(cifs: dict[str, Path]) -> dict:
    script = """
import { readFileSync } from "node:fs";
import { parseCifBlock, parseCifToAtomArray } from "./shared/rfd4p/pipeline/cif.js";
import { evalSelection } from "./shared/rfd4p/pipeline/spec.js";
const { cifs, corpus, ccd } = JSON.parse(readFileSync(0, "utf8"));
const ccdBlocks = new Map(Object.entries(ccd).map(([id, text]) => [id, parseCifBlock(text)]));
const out = {};
for (const [name, path] of Object.entries(cifs)) {
  const atoms = parseCifToAtomArray(readFileSync(path, "utf8"), { ccdBlocks });
  Object.assign(atoms, atoms.annotations);
  out[name] = {};
  for (const q of corpus) {
    try { out[name][q] = evalSelection(atoms, q).map((v) => (v ? "1" : "0")).join(""); }
    catch (e) { out[name][q] = `raise:${e.constructor.name}:${e.message}`; }
  }
}
process.stdout.write(JSON.stringify(out));
"""
    r = subprocess.run(["node", "--input-type=module", "-e", script], cwd=REPO, capture_output=True,
                       text=True, input=json.dumps({"cifs": {k: str(v) for k, v in cifs.items()},
                                                    "corpus": CORPUS, "ccd": ccd_entries(list(cifs.values()))}))
    if r.returncode != 0:
        sys.exit(f"node failed: {r.stderr[-2000:]}")
    return json.loads(r.stdout)


def native_masks(cifs: dict[str, Path], binary: Path) -> dict:
    out = {}
    for name, path in cifs.items():
        out[name] = {}
        for q in CORPUS:
            r = subprocess.run([str(binary), "x", f"--dump-cif={path}", f"--select={q}"],
                               capture_output=True, text=True)
            line = next((ln for ln in r.stdout.splitlines() if ln.startswith("select\t")), None)
            if r.returncode != 0 or line is None:
                out[name][q] = f"raise:{r.stderr.strip()[:200]}"
            else:
                out[name][q] = "".join(line.split("\t")[2:])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(REPO / "cuda/rfd4p/featurise/rfd4p-featurise"))
    ap.add_argument("--no-native", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    import zstandard

    with tempfile.TemporaryDirectory() as tmp:
        cifs = {}
        for name in STRUCTURES + RAW:
            src = (REPO / "tools" / "fixtures" / f"{name}.cif.zst") if name.startswith("rcsb/") else (
                REPO / "web" / "examples" / f"{name}.cif.zst")
            path = Path(tmp) / f"{name.replace('/', '_')}.cif"
            path.write_bytes(zstandard.ZstdDecompressor().decompressobj().decompress(src.read_bytes()))
            cifs[name] = path
        up = upstream_masks(cifs)
        sides = {"page": js_masks(cifs)}
        if not args.no_native:
            sides["native"] = native_masks(cifs, Path(args.binary))

    bad = 0
    for side, masks in sides.items():
        wrong = []
        for name in STRUCTURES + RAW:
            for q in CORPUS:
                want, have = up[name][q], masks[name][q]
                if want.startswith("raise") and have.startswith("raise"):
                    continue
                if want != have:
                    wrong.append((name, q, want, have))
        bad += len(wrong)
        total = len(STRUCTURES + RAW) * len(CORPUS)
        print(f"{side}: {total - len(wrong)} of {total} agree with upstream"
              f"{'' if not wrong else f', {len(wrong)} differ'}")
        seen = set()
        for name, q, want, have in wrong:
            if q in seen and not args.verbose:
                continue
            seen.add(q)
            def show(m):
                return m if m.startswith("raise") else f"{m.count('1')}/{len(m)}"
            print(f"  {q!r} on {name}: upstream {show(want)}, {side} {show(have)[:160]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
