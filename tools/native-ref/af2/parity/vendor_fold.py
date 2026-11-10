"""ColabFold's own fold of a case, through `colabfold.batch.run` - what `colabfold_batch` runs.

    ~/venv_colabfold/bin/python cuda/af2/parity/vendor_fold.py <case.json> <out_dir> \
        [--model-type=alphafold2_multimer_v3] [--recycles=N] [--tolerance=T] [--max-msa=508:2048] \
        [--models=1] [--seed=0] [--seeds=1] [--data-dir=~]

The case's per-chain unpaired and paired A3Ms are handed over as ColabFold's own combined A3M (msa_to_str),
so nothing is searched and nothing here featurises. `--data-dir` holds params/params_model_*.npz
(DeepMind's float32 checkpoints). Unset options keep ColabFold's defaults for the model type.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from colabfold.batch import run
from colabfold.input import msa_to_str


def main():
    from colabfold.utils import setup_logging
    os.makedirs(sys.argv[2], exist_ok=True)
    setup_logging(Path(sys.argv[2]) / "log.txt")
    p = argparse.ArgumentParser()
    p.add_argument("case")
    p.add_argument("out")
    p.add_argument("--model-type", default="alphafold2_multimer_v3")
    p.add_argument("--recycles", type=int, default=None)
    p.add_argument("--tolerance", type=float, default=None)
    p.add_argument("--max-msa", default=None)
    p.add_argument("--models", default="1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--data-dir", default=os.path.expanduser("~"))
    a = p.parse_args()
    case = json.load(open(a.case))
    chains = case["sequences"]
    unique, card = [], []
    for s in chains:
        if s in unique:
            card[unique.index(s)] += 1
        else:
            unique.append(s); card.append(1)
    read = lambda path, s, i: open(path).read() if path else f">{101 + i}\n{s}\n"
    unpaired = [read(case["unpaired"][chains.index(s)], s, i) for i, s in enumerate(unique)]
    paired = ([read(case["paired"][chains.index(s)], s, i) for i, s in enumerate(unique)]
              if case.get("paired") and len(unique) > 1 else None)
    a3m = msa_to_str(unpaired, paired, unique, card)
    order = [int(x) for x in a.models.split(",")]
    run(queries=[("fold", chains if len(chains) > 1 else chains[0], [a3m], None)],
        result_dir=a.out, num_models=len(order), model_order=order, is_complex=len(chains) > 1,
        num_recycles=a.recycles, recycle_early_stop_tolerance=a.tolerance, model_type=a.model_type,
        msa_mode="single_sequence" if not any(case["unpaired"]) else "mmseqs2_uniref_env",
        max_msa=a.max_msa, random_seed=a.seed, num_seeds=a.seeds, data_dir=a.data_dir, keep_existing_results=False,
        user_agent="localfold-parity/1.0")


if __name__ == "__main__":
    main()
