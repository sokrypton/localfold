"""Score a modified residue's own bonds in a predicted mmCIF.

The same rule the other two references were measured with: the residue's atoms
selected by their residue number, bonds taken where the CCD's ideal geometry
puts two of them within 1.95 A, and the ratio reported per bond. The ideals
come from the dictionary rather than from the prediction.
"""
import sys, math, urllib.request

CIF, RESNUM, CODE = sys.argv[1], int(sys.argv[2]), sys.argv[3]

def atoms_of(text):
    cols, rows, inloop = [], [], False
    for line in text.split("\n"):
        if line.startswith("_atom_site."):
            cols.append(line.strip().split(".")[1]); inloop = True; continue
        if inloop:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                rows.append(line.split())
            elif rows:
                break
    i = {c: k for k, c in enumerate(cols)}
    return i, rows

# the dictionary's ideal geometry for the component
url = "https://files.rcsb.org/ligands/download/%s.cif" % CODE
ideal = {}
text = urllib.request.urlopen(url, timeout=60).read().decode()
ci, crows = atoms_of(text) if "_atom_site." in text else (None, [])
# chem_comp_atom, not atom_site, for a ligand definition
cols, rows, inloop = [], [], False
for line in text.split("\n"):
    if line.startswith("_chem_comp_atom."):
        cols.append(line.strip().split(".")[1]); inloop = True; continue
    if inloop:
        if line.startswith("#") or line.strip() == "":
            if rows: break
            continue
        parts = line.split()
        if len(parts) >= len(cols): rows.append(parts)
j = {c: k for k, c in enumerate(cols)}
for r in rows:
    name = r[j["atom_id"]].strip('"')
    if name.startswith("H"): continue
    try:
        ideal[name] = (float(r[j["pdbx_model_Cartn_x_ideal"]]),
                       float(r[j["pdbx_model_Cartn_y_ideal"]]),
                       float(r[j["pdbx_model_Cartn_z_ideal"]]))
    except (KeyError, ValueError):
        pass

i, rows = atoms_of(open(CIF).read())
got = {}
for r in rows:
    if int(r[i["label_seq_id"]]) != RESNUM: continue
    nm = r[i["label_atom_id"]].strip('"')
    if nm.startswith("H"): continue
    got[nm] = (float(r[i["Cartn_x"]]), float(r[i["Cartn_y"]]), float(r[i["Cartn_z"]]))

d = lambda a, b: math.dist(a, b)
names = sorted(set(ideal) & set(got))
print("residue %d (%s): %d atoms matched -> %s" % (RESNUM, CODE, len(names), ",".join(names)))
ratios, errs = [], []
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        na, nb = names[a], names[b]
        ii = d(ideal[na], ideal[nb])
        if not (0.9 < ii < 1.95): continue
        gg = d(got[na], got[nb])
        ratios.append(gg / ii); errs.append(gg - ii)
        print("   %-9s ideal %.3f  predicted %.3f  ratio %.3f" % (na + "-" + nb, ii, gg, gg / ii))
if ratios:
    print("   %d bonds   mean ratio %.3f   bond rms %.4f A"
          % (len(ratios), sum(ratios) / len(ratios),
             math.sqrt(sum(e * e for e in errs) / len(errs))))
