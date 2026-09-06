# Where this left off

Written at the end of the ESMFold2 WebGPU session. `CLAUDE.md` has the durable
findings - this file has only what is UNFINISHED, and the one open problem is at
the top because it is the reason this file exists.

## OPEN: ligand contacts, and the confidence estimate built on them

🔴 **RAISED BUT NOT INVESTIGATED, AND IT AFFECTS EVERY MODEL.** Reported at the
end of the session: the contact calculation is wrong for ligands, and the
estimated confidence that reads it is wrong in consequence. Nothing was changed
- the report arrived with the context nearly full and the right move was to
write it down rather than start.

What is known, and what is only suspected, kept apart:

**Known, because it is how the code is built.** A ligand is ONE TOKEN PER HEAVY
ATOM, so a ligand of `k` atoms contributes `k` tokens to every token-by-token
matrix. Two consequences follow arithmetically and neither has been measured:

* Its atoms are all within a few angstroms of each other, so a ligand
  contributes roughly `k^2` short-range "contacts" that say nothing about
  anything. On a 60-residue protein with a 40-atom ligand that is 1600 pairs
  against the protein's 3600.
* Sequence separation does not mean what it means for a polymer. A `|i - j| > 6`
  filter, which every contact and confidence path here uses, is designed to
  exclude a chain's own neighbours - and applied across a ligand it excludes an
  arbitrary third of its atoms and keeps the rest.

**The proposed quick fix, from the same report: compute confidence PER CHAIN.**
That is plausible and would sidestep both, since a ligand is its own asym id -
but it is a guess about a bug nobody has yet reproduced, so REPRODUCE IT FIRST.

**Where to look**, in the order that will settle it fastest:

1. `contactAgreement` in `tools/gpu/fold-esmfold2.js` - it already picks a
   representative atom per token and already excludes `|i - j| <= 6`. Fold a
   protein with a ligand (`--ligands=GOL`) and print the contact counts split by
   whether each partner is polymer or ligand. If the ligand's self-contacts
   dominate, that is the bug, visible in one run.
2. `CERTAINTY` in `src/esmfold2/distogram-webgpu.js` - `separation: 3`,
   `cutoff: 12`. The same two objections apply, and its constants were swept on
   PROTEIN-ONLY targets (see CLAUDE.md), so nothing about them is known to hold
   for a ligand token.
3. `src/heads/` and `web/prediction-results.js` for the AF2/AF3 paths, which is
   where "affects all models" would show.

🔴 **AND THE SWEEP CANNOT BE REUSED TO SETTLE IT.** All 46 targets and all 80
corrupted folds were single protein chains. Whatever the right treatment of a
ligand is, no measurement in this repository currently bears on it.

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
