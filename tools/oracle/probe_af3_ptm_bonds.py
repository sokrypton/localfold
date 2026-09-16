"""Does the REFERENCE inflate a modified residue too, or only this port?

🔴 THE FIRST VERSION OF THIS PROBE WAS WRONG AND ITS ANSWER LOOKED LIKE ONE.
It searched EVERY token for an atom called "CB" and paired it with any "OG"
whose ref_pos distance fell in a 0.9-2.0 A window - and ref_pos is each
residue's OWN local frame, so a cross-residue distance is not a distance at all
and lands in that window by accident. It reported CB-OG at 16.321 A on a
reference fold whose N-CA, CA-CB, CA-C and C-O were all within 3%, which is not
a molecule anyone would build; the tell was the ideal itself reading 1.571 where
the dictionary says 1.428.

The modified residue's atoms are selected by `residue_index` here, which is what
actually identifies them: an atomised residue is one token per atom and they all
carry the same index.
"""
import os, sys
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
for e in ("/home/ubuntu/alphafold3/src", "/home/ubuntu/alphafold3",
          "/home/ubuntu/alphafold3/dev/oracles"):
    sys.path.insert(0, e)
import numpy as np
import fold_check
from alphafold3.common import folding_input

MODEL = os.environ.get("MODEL", "boltz2")
SEQ = "ACSEFGHIKLWY"
AT = int(os.environ.get("AT", "3"))

chains = [folding_input.ProteinChain(
    id="A", sequence=SEQ, ptms=[("SEP", AT)], unpaired_msa="", paired_msa="",
    templates=[])]
out, batch = fold_check.fold(MODEL, SEQ, seed=1, chains=chains)
pos = np.asarray(out["diffusion_samples"]["atom_positions"])[0]
mask = np.asarray(batch["ref_mask"])
ref = np.asarray(batch["ref_pos"])
names = np.asarray(batch["ref_atom_name_chars"])
resid = np.asarray(batch["residue_index"])
nm = lambda t, s: "".join(chr(c + 32) for c in names[t, s] if c > 0).strip()

T, A = mask.shape
# every live slot belonging to the modified residue, by residue_index
slots = [(t, s) for t in range(T) for s in range(A)
         if mask[t, s] and int(resid[t]) == AT]
print("model", MODEL, "tokens", T, " slots on residue", AT, ":", len(slots),
      "->", ",".join(nm(t, s) for t, s in slots))
rows, ratios = [], []
for i in range(len(slots)):
    for j in range(i + 1, len(slots)):
        (ta, sa), (tb, sb) = slots[i], slots[j]
        ideal = float(np.linalg.norm(ref[ta, sa] - ref[tb, sb]))
        if not (0.9 < ideal < 1.95):
            continue                      # a bond, by the dictionary's own geometry
        seen = float(np.linalg.norm(pos[ta, sa] - pos[tb, sb]))
        rows.append((nm(ta, sa) + "-" + nm(tb, sb), ideal, seen, seen / ideal))
        ratios.append(seen / ideal)
for label, ideal, seen, r in rows:
    print("   %-9s ideal %.3f  predicted %.3f  ratio %.3f" % (label, ideal, seen, r))
if ratios:
    err = [s - i for _, i, s, _ in rows]
    print("   %d bonds   mean ratio %.3f   bond rms %.4f A"
          % (len(rows), float(np.mean(ratios)),
             float(np.sqrt(np.mean(np.square(err))))))
