# Where this left off

Written at the end of the ESMFold2 WebGPU session. `CLAUDE.md` has the durable
findings - this file has only what is UNFINISHED. The problem it was written for
is at the top, now settled, with the one question it left behind.

## SETTLED: ligand contacts, and the confidence built on them

Reproduced, fixed and measured - see CLAUDE.md's ESMFold2 confidence section for
the tables. In short: the separation rule ran on the TOKEN index, a ligand is one
token per heavy atom, and 62% of a ubiquitin+ATP fold's contacts were ATP's own
internal pairs. `partnerKeys` makes the rule "the same chain, and within
`separation` RESIDUES", which is bit-identical on one unmodified protein chain.

Two things that came out of it and are worth carrying:

* **The proposed per-chain fix would have been credited with something it does
  not do.** The protein's own certainties shift by -0.051 when ATP is added, and
  they shift by exactly that under both rules - it is the trunk conditioning on
  a real molecule, not the metric.
* **OPEN: the distogram predicts 0 protein-ligand contacts where the structure
  makes 64.** Either this head does not speak about ligand pairs or the borrowed
  `CONTACT_EDGES` are wrong for them. Nothing depends on it today beyond the
  ligand's own (now honest, and very low) certainty.

## Also open, smaller

* **The ESMFold2 bundles are local-only.** `model-esmfold2-int5` (122 MiB) and
  `model-esmc-600m-int3` (224 MiB) have `directory` but no `remote`, so
  `tools/build_site.py --model` publishes 366 MiB. Every other family is
  hosted. Needs a Hugging Face upload and a pinned commit SHA - see CLAUDE.md's
  hosting section, including why the SHA is pinned.
* **Multi-chain ESMFold2 is untested against an oracle.** It runs - a homodimer
  and a protein+DNA+ligand fold both produce correct geometry - but there is no
  dump to check the chain-aware ESM-C attention or the cross-chain relative
  position encoding against. The featuriser is checked exactly; the model on a
  complex is not.
* **The per-residue confidence ordering was measured and NOT shipped as one.**
  The colour is shipped; a per-residue NUMBER is not, and CLAUDE.md records why
  (worst-fold Spearman goes negative above 40% corruption). The global mean is
  the reliable reading at 0.90 Pearson.
* **`tools/gpu/probe-esmfold2-confidence.js` crashed Chrome at 20 targets x 2
  rates.** 10 x 2 and 16 x 5 both work. Not diagnosed; likely the retained
  per-arm arrays across 11,400 arms.

## What NOT to redo

Every one of these was measured and is written up in CLAUDE.md with its numbers.
They are listed here only so nobody spends a second afternoon on them.

* Restricting the confidence score to the contact bins - ColabDesign's `con`
  loss - scores BELOW a buriedness baseline. It is a design objective, not a
  confidence.
* `exp(-CCE)` on the exact bin is the worst measure of its family; on a radius
  it is the best. The bins are 0.39 A on a borrowed grid.
* Top-N over a residue's best partners loses to using every pair, monotonically.
* A flow sampler arm for ESMFold2 saves nothing - `flow-16` runs 12 steps where
  the shipped `diffusion-15` runs 11.
* Shrinking the trunk's submission window to smooth the progress bar costs 14%.
  The non-awaited `onSubmittedWorkDone` is free.
* A distogram-derived pLDDT NUMBER. Four were removed in commit 588b528 and this
  session's work agrees with that verdict: the ordering is usable, the
  calibration is not.

## The state of the tree

29 commits this session, all pushed to the working branch. `npm test` is 765
passing. The GPU checkers all pass - `check-esmfold2-{trunk-gpu,diffusion-gpu,
atom-stack}`, `check-esmc-{block,tower}` - and `tools/gpu/fold-esmfold2.js`
folds end to end.

🔴 **FOUR FILES IN THE WORKING TREE ARE NOT MINE AND MUST BE LEFT ALONE**:
`src/runtime/quantised-upload.js` (modified), `test/quantised-upload-plan.test.js`,
`tools/gpu/check-delta-upload.js`, `tools/make-delta-fixture.py`. They are a
parallel exploration of delta-encoded model weights.

🔴 **AND USE `python3 tools/serve.py`, NOT `python3 -m http.server`.** The
second sends no cache headers and a plain reload then serves stale ES modules,
which looks exactly like a broken feature and has cost three sessions.
