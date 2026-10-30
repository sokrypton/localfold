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

6. `run-native.sh <jobs-prefix> config-N*.json out.jsonl 0 1 ...` - the same attempt with every fold on the native CUDA
   port: `native-bench.mjs` runs the page's design loop, MPNN and contact check under Node and hands each fold to a
   resident server (`cuda/af3/fold --serve [--model=boltz2|chai1]`, started first), ~5x the browser's folds an hour.
   Its evaluation is the native AF3. `folder` picks the design engine (af3, boltz2, chai1), `checkModel` the internal
   check's, `hotspots` hands Boltz-2 and Chai-1 the benchmark's hotspots as a pocket, `msaDir` gives each target
   chain its MSA instead of the template (one `<sha1[:12]>.a3m` a chain, from the ColabFold server).

`results-C-2026-10-29.jsonl` is the overnight run: 214 attempts, each design's sequence and both evaluators' terms.

7. The small-molecule- and DNA-binder benchmarks: `~/.venv-rdkit/bin/python prep-tasks.py <clone>` writes
   `tasks.json` (77 DISCO small-molecule targets, ligands as SMILES or CCD codes; 3 DNA duplexes as sequence), then
   `run-tasks.sh config-T*.json out.jsonl <workers> <target:attempt> ...` (`task-bench.mjs`, the same loop with
   LigandMPNN / NA-MPNN and each task's pass criteria) against one resident `cuda/af3/fold --serve`, and
   `task-score.py out.jsonl`. Results: docs/DESIGN-TODO.md, "RFD4-Proteina's small-molecule- and DNA-binder
   benchmarks".

## The best settings so far, per task

All on the native port, evaluation always the native AF3 at the benchmark's settings; mean over targets.
The internal check is the evaluator's own model in every row - stated beside every number.

| task | config | what it is | pass rate | RFD4-Proteina high |
|---|---|---|---|---|
| protein binders (24) | `config-NC.json` | AF3 designs with the target templated, 6 cycles x 100 steps x 3 recycles, 3 restarts, early stop at pTM >= 0.85 and iPAE <= 1.2, internal AF3 check of the top 2 | 37.5% | 19.3% |
| small-molecule binders (77) | `config-TP2.json` | the same loop with LigandMPNN, no template, the file's 100-155 lengths, **10 restarts, check the top 3 at two seeds each, then `polish` the nearest misses** | 62.3% (without polish 51.9%) | 34.3% |
| small-molecule binders, `disco` criterion | `config-TS9.json` | TS7 with the internal check on the `disco` criterion (`criterion`); `disco-score.py` rescores | 81.2% (CI 76.6-85.1) | 64.7% (Search 76.4%) |
| DNA binders (3) | `config-TD9.json` | LigandMPNN, no C/D/E (`omit`), 10 restarts, an undocked fold refolded up to 5 seeds (`contactTries`), `stick`, `minFootprint` 8 on checks | 6.7% (4 of 60; CI 1.7-13.3); + polish (TD15) 8.3% | 3.5% |
| enzyme active sites (37, MCSA) | `config-TA6.json` | the motif's tips and the ligands inpainted into every design fold (`cuda/af3 --motif`, released below 1 A), each design's sequence folded again FREE as a `probe` with the motif matched unindexed (`motifMatch`), checks on the probes, then a `climb` on the motif | 21.6% over all 37 (CI 17.6-25.7), a pass on 11 | 14.7% |

What did not help, measured: the warm start (W0/W1, a null), the target's MSA in place of its template (NM1), Boltz-2
or Chai-1 as design engine (NH, NG, NX), bigger small-molecule binders (TS1), and on DNA a beam of 3 with a give-up
(TD1, TD16), a fixed DNA frame (TD17), and for AME guiding to the last step (TA1), a guided-then-free schedule (TA3,
TA4). Untested: 6 restarts on the protein binders, which lifted small molecules from 26.6 to 36.4.
