"""AlphaFold 3's own fraction_disordered and has_clash on a PDB this port wrote - the oracle for the
two terms af3 adds to its ranking score (src/scores.cuh). Run it with af3-any-model's environment,
which carries AF3's compiled mkdssp:

    ~/.venv-lfjax/bin/python native/af3/scores_oracle.py out.pdb [more.pdb ...]

Prints, per file: fraction_disordered, has_clash, the per-residue windowed rASA (--rasa) and DSSP's
raw accessibility (--acc).
"""
import sys

import gemmi
from alphafold3 import structure
from alphafold3.model import confidences


def load(path):
    st = gemmi.read_structure(path)
    # a modified residue is HETATM inside a polymer chain; gemmi would split it off as a ligand, where
    # AF3's structure keeps it in the chain - mark the residues of any chain with ATOM records as ATOM
    for ch in st[0]:
        if any(r.het_flag == "A" for r in ch):
            for r in ch:
                r.het_flag = "A"
    st.setup_entities()
    for ent in st.entities:       # the polymers' full sequences, which AF3's reader requires
        if ent.entity_type == gemmi.EntityType.Polymer:
            sub = ent.subchains[0]
            ent.full_sequence = [r.name for ch in st[0] for r in ch if r.subchain == sub]
    st.assign_label_seq_id()
    doc = st.make_mmcif_document()
    return structure.from_mmcif(doc.as_string())


def main():
    show_rasa, show_acc = "--rasa" in sys.argv, "--acc" in sys.argv
    for path in [a for a in sys.argv[1:] if not a.startswith("--")]:
        s = load(path)
        print(f"{path}: fraction_disordered {confidences.fraction_disordered(s):.6f}  "
              f"has_clash {int(confidences.has_clash(s))}")
        if show_acc:     # DSSP's raw accessibility per residue, each protein chain alone
            from alphafold3.cpp import mkdssp
            p = s.filter_to_entity_type(protein=True)
            for chain_id in p.chains:
                c = p.filter(chain_id=chain_id).rename_chain_ids(new_id_by_old_id={chain_id: "A"})
                out, go = [], False
                for row in mkdssp.get_dssp(c.to_mmcif(), calculate_surface_accessibility=True).splitlines():
                    if go and row[13:14] != "!":
                        out.append(row[34:38].strip())
                    go = go or row.startswith("  #  RESIDUE")
                print(chain_id, " ".join(out))
        if show_rasa:
            p = s.filter_to_entity_type(protein=True)
            for chain_id in p.chains:
                c = p.filter(chain_id=chain_id).rename_chain_ids(new_id_by_old_id={chain_id: "A"})
                r = confidences.windowed_solvent_accessible_area(c.to_mmcif())
                print(chain_id, " ".join(f"{v:.4f}" for v in r))


if __name__ == "__main__":
    main()
