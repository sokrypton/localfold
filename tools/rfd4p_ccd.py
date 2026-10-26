#!/usr/bin/env python3
"""The dictionary entries the RFD4 page would FETCH for a structure, written from the CCD atomworks reads.

    from rfd4p_ccd import ccd_entries      # tools/check-rfd4p-load.py, tools/check-rfd4p-query.py

A raw RCSB file carries no `_chem_comp_atom` or `_chem_comp_bond`, and the page fetches each
component outside its generated templates from RCSB (shared/rfd4p/pipeline/ccd.js). An oracle
must hand the page the same entries without the network, so they are written here from biotite's
bundled CCD - the one atomworks itself reads - as the mmCIF text RCSB would serve.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def ccd_entries(paths: list[Path]) -> dict[str, str]:
    """`{component: mmCIF text}` for every component these files name outside the generated
    templates: `_chem_comp`, `_chem_comp_atom` and `_chem_comp_bond`, as RCSB serves them."""
    import re

    import numpy as np
    from biotite.structure.info.ccd import get_ccd

    have = set(re.findall(r'^  "([^"]+)":', (REPO / "shared/rfd4p/pipeline/ccd-templates.js").read_text(), re.M))
    names = set()
    for path in paths:
        try:
            text = path.read_text()
        except UnicodeDecodeError:   # gzip or BinaryCIF: their components are their mmCIF twin's
            continue
        names |= set(re.findall(r"^(?:ATOM|HETATM)\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)", text, re.M))
    ccd = get_ccd()
    comp = ccd["chem_comp"]
    comp_type = dict(zip(comp["id"].as_array(str), comp["type"].as_array(str), strict=True))

    def loop(category: str, cols: list[str], name: str) -> list[str]:
        cat = ccd[category]
        rows = np.nonzero(cat["comp_id"].as_array(str) == name)[0]
        arrays = {c: cat[c].as_array(str) for c in cols}
        quote = lambda v: f'"{v}"' if "'" in v else f"'{v}'"
        return ["loop_", *(f"_{category}.{c}" for c in cols),
                *(" ".join(quote(arrays[c][i]) for c in cols) for i in rows)]

    out = {}
    for name in sorted(names - have):
        if name not in comp_type:
            continue
        lines = [f"data_{name}", f"_chem_comp.id {name}", f"_chem_comp.type '{comp_type[name]}'"]
        lines += loop("chem_comp_atom", ["comp_id", "atom_id", "alt_atom_id", "type_symbol", "charge",
                                         "pdbx_leaving_atom_flag", "pdbx_aromatic_flag"], name)
        lines += loop("chem_comp_bond", ["comp_id", "atom_id_1", "atom_id_2", "value_order",
                                         "pdbx_aromatic_flag"], name)
        out[name] = "\n".join(lines) + "\n"
    return out
