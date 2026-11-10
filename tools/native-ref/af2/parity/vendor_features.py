"""ColabFold's own merged AlphaFold 2 input for a case, before any random draw.

    ~/venv_colabfold/bin/python cuda/af2/parity/vendor_features.py <case.json> <out.npz> [<merged.a3m>]

The case names the chains (sequences, in fold order), each chain's unpaired A3M and (for a complex)
its row-aligned paired A3M, and the model type. ColabFold's `generate_input_feature` - the function
`colabfold_batch` itself calls - builds the features; nothing here featurises. What is written is the
batch the network's in-model sampling starts from: msa, deletion_matrix, msa_mask, cluster_bias_mask,
aatype, residue_index, asym_id, entity_id, sym_id.

The optional third path receives that merged alignment as an A3M - every row ColabFold's sampler can draw
(msa_mask not all zero), deletions written back as lowercase insertions - so LocalFold can fold ColabFold's
rows through its own sampling (`localfold-af2 --sequence=A:B --a3m=<merged.a3m>`), which separates the
merge from everything after it.
"""
import json
import sys

import numpy as np
from colabfold.batch import generate_input_feature, mk_mock_template


def main():
    case = json.load(open(sys.argv[1]))
    chains = case["sequences"]
    model_type = case["model_type"]       # alphafold2_multimer_v3 | alphafold2_ptm
    unique, cardinality = [], []
    for seq in chains:
        if seq in unique:
            cardinality[unique.index(seq)] += 1
        else:
            unique.append(seq)
            cardinality.append(1)
    def read(path, seq, index):
        return open(path).read() if path else f">{101 + index}\n{seq}\n"
    unpaired = [read(case["unpaired"][chains.index(s)], s, i) for i, s in enumerate(unique)]
    paired = None
    if case.get("paired") and len(unique) > 1:
        paired = [read(case["paired"][chains.index(s)], s, i) for i, s in enumerate(unique)]
    templates = [mk_mock_template(s) for s in unique]
    feats, _ = generate_input_feature(unique, cardinality, unpaired, paired, templates,
                                      is_complex=len(chains) > 1, model_type=model_type,
                                      max_seq=int(case.get("max_seq", 508)))
    keep = ("msa", "deletion_matrix", "msa_mask", "cluster_bias_mask", "aatype", "residue_index",
            "asym_id", "entity_id", "sym_id", "num_alignments", "deletion_matrix_int", "between_segment_residues")
    out = {k: np.asarray(feats[k]) for k in keep if k in feats}
    if out["aatype"].ndim == 2:
        # the MONOMER pipeline stops before its TF stage: aatype is a one-hot and the msa is still in
        # HHBLITS order, which correct_msa_restypes maps to AlphaFold's inside the model's input stage
        from alphafold.common import residue_constants as rc
        out["aatype"] = out["aatype"].argmax(-1)
        out["msa"] = np.asarray(rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE)[out["msa"]]
    if "deletion_matrix" not in out and "deletion_matrix_int" in out:
        out["deletion_matrix"] = out.pop("deletion_matrix_int")
    np.savez(sys.argv[2], **out)
    if len(sys.argv) > 3:
        alphabet = "ARNDCQEGHILKMFPSTWYVX-"
        mask = out.get("msa_mask", np.ones_like(out["msa"]))
        lines = []
        for index, (row, dels, live) in enumerate(zip(out["msa"], out["deletion_matrix"], mask)):
            if index > 0 and not np.any(live):
                continue                      # pad_msa's rows, which sample_msa never draws
            text = "".join("x" * int(d) + alphabet[int(c)] for c, d in zip(row, dels))
            lines += [f">row{index}", text]
        open(sys.argv[3], "w").write("\n".join(lines) + "\n")
    print({k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
