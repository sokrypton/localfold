# RFD4-Proteina's protein-binder benchmark, run through the page's Protein Hunter

The 24 PPI targets of RFD4-Proteina's Fig. 2 (the union of the Proteina-Complexa and RFdiffusion3 sets) and
their pass criteria: AF3 refold, single sequence, target templated, 5 recycles, one sample; binder-chain
pTM >= 0.80, minimum interface PAE <= 1.5 A, target-aligned binder backbone RMSD <= 2.5 A
(RFD4-Proteina's `configs/metrics/pass_criteria/all_pass_criteria.yaml` and `af3_oracle.py`).
Results and method: docs/DESIGN-TODO.md, "RFD4-Proteina's binder benchmark".

1. `git clone https://github.com/RosettaCommons/RFD4-Proteina` (BSD-3) and
   `python3 prep.py <clone>/data/benchmarks/ppi/ppi_all_atom` writes `targets.json`.
2. Save `https://rfd4-proteina.org/benchmarks/data.json` here as `rfd4-paper-data.json` (the paper's per-target rows).
3. `run2.sh <jobs-prefix> config-C.json out.jsonl 0 1 2 ...` - one headless Chrome per worker, each driving
   design.html through `bench.js` (a Protein Hunter attempt, its internal check, then the evaluation), over the
   `target:attempt` lists in `<jobs-prefix>.jobs<w>`.
4. `score.py out.jsonl` - per-target pass rates against the paper's, clustered with `cluster.py` (a small TM-align
   standing in for Foldseek, which could not be installed here).
5. `rescore.py out.jsonl native.jsonl` re-evaluates every design with the native CUDA AF3 (cuda/af3, af3-any-model's
   int8 weights) - an evaluator independent of the page's int5 WebGPU fold.

`results-C-2026-10-29.jsonl` is the overnight run: 214 attempts, each design's sequence and both evaluators' terms.
