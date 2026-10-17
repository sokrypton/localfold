"""biohub's own language-model pair for a sequence - ESM-C's 37 (or 31, or 81) hidden states through ESMFold2's shim -
written as raw float32 [T, T, P], to hold a port's (EF2_SAVE_LMZ) against it.

    ~/.cache/localfold/venv-esm/bin/python tools/esmc/probe-esmfold2-lm-pair.py \\
        --model biohub/ESMFold2-Experimental-Fast-base300M-step1500k --sequence GWS... --out lmz.f32
    python3 -c "..."     # relRMS against metal/ef2's EF2_SAVE_LMZ for the same sequence

The pair is the first thing the folding model reads and has no randomness in it, so it is where a TOWER convention
(the residual scale, the layer mix) is settled without a fold's noise: `esm` 3.4.1's EsmFold2ExperimentalModel, its
ESM-C loaded through config.esmc_id and held in float32 (its finalize_esmc casts to bfloat16, a 1e-2 floor), the shim
called with lm_dropout 0 and no residue masked (lm_mask_pct 0).
"""
import argparse

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()

    from esm.models.esmfold2 import EsmFold2ExperimentalModel
    from esm.models.esmfold2.processor import ESMFold2InputBuilder
    from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput

    features, _ = ESMFold2InputBuilder().prepare_input(
        StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=a.sequence)]), seed=0, device="cpu")
    model = EsmFold2ExperimentalModel.from_pretrained(a.model, load_esmc=True, device="cpu").eval()
    model.esmc = model.esmc.float()           # (finalize_esmc made it bfloat16)
    print("residue scaling:", sorted({round(float(b.scaling_factor), 6) for b in model.esmc.transformer.blocks}))
    with torch.no_grad():
        hidden = model._compute_lm_hidden_states(features["input_ids"], features["asym_id"], features["residue_index"],
                                                 features["mol_type"], features["token_attention_mask"], lm_mask_pct=0.0)
        lm_z = model.language_model(hidden.float(), lm_dropout=0.0)
    out = lm_z[0].float().numpy()
    out.astype(np.float32).tofile(a.out)
    print(f"lm pair {tuple(out.shape)} -> {a.out}")


main()
