# Benchmark targets (RFD4-Proteina repo data/benchmarks/ppi/ppi_all_atom, BSD-3) -> targets.json
import json, math, os, subprocess, sys
SRC = sys.argv[1]
THREE = dict(ALA="A",ARG="R",ASN="N",ASP="D",CYS="C",GLN="Q",GLU="E",GLY="G",HIS="H",ILE="I",LEU="L",LYS="K",
             MET="M",PHE="F",PRO="P",SER="S",THR="T",TRP="W",TYR="Y",VAL="V",MSE="M")
out = {}
for f in sorted(os.listdir(SRC)):
    name = f.replace(".cif.zst", "")
    text = subprocess.run(["zstd", "-dc", os.path.join(SRC, f)], capture_output=True, text=True).stdout.split("\n")
    cols, rows, i = [], [], 0
    while i < len(text):
        if text[i].startswith("loop_") and text[i+1].startswith("_atom_site."):
            i += 1
            while text[i].startswith("_atom_site."):
                cols.append(text[i].strip().split(".", 1)[1]); i += 1
            while i < len(text) and text[i].strip() and not text[i].startswith(("loop_", "#", "_")):
                rows.append(text[i].split()); i += 1
            break
        i += 1
    c = {k: j for j, k in enumerate(cols)}
    binder, chains, hot = None, {}, set()
    for r in rows:
        asym = r[c["label_asym_id"]]
        if r[c["label_comp_id"]] != "UNK" and r[c["label_comp_id"]] not in THREE and "-" in r[c["label_comp_id"]]:
            binder = (int(r[c["annotation_expsegmin_1_residue"]]), int(r[c["annotation_expsegmax_1_residue"]]), asym); continue
        if r[c["label_atom_id"]] == "X":
            binder = (int(r[c["annotation_expsegmin_1_residue"]]), int(r[c["annotation_expsegmax_1_residue"]]), asym); continue
        x, y, z = (float(r[c[k]]) for k in ("Cartn_x", "Cartn_y", "Cartn_z"))
        seq = int(r[c["label_seq_id"]]); res = r[c["label_comp_id"]]
        ch = chains.setdefault(asym, {})
        ch.setdefault(seq, {"name": res, "atoms": []})
        hk = next((k for k in c if k.endswith("hotspot_1_atom")), None)
        if hk is not None and r[c[hk]] == "True": hot.add((asym, seq))
        if math.isnan(x) or float(r[c["occupancy"]]) == 0: continue
        ch[seq]["atoms"].append((r[c["label_atom_id"]], r[c["type_symbol"]], x, y, z))
    assert binder is not None, name
    # fold order: binder first (A), then the target chains in file order, relettered B, C, ...
    letters = {}
    pdb, serial, seqs = [], 1, []
    for k, asym in enumerate(chains):
        letter = "BCDEFG"[k]; letters[asym] = letter
        s = ""
        for num in sorted(chains[asym]):
            res = chains[asym][num]
            s += THREE.get(res["name"], "X")
            for (atom, el, x, y, z) in res["atoms"]:
                an = atom if len(atom) == 4 else " " + atom
                pdb.append("ATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s" % (serial, an, res["name"], letter, num, x, y, z, el))
                serial += 1
        seqs.append(s)
    pdb.append("END")
    out[name] = {"binder": [binder[0], binder[1]], "sequences": seqs, "pdb": "\n".join(pdb) + "\n",
                 "hotspots": sorted([letters[a], n] for a, n in hot if a in letters),
                 "lengths": [len(s) for s in seqs]}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "targets.json"), "w"))
for k, v in out.items(): print(k, v["binder"], v["lengths"], len(v["hotspots"]), "hotspots")
