# Where this left off

`CLAUDE.md` has the durable findings; this file has only what is UNFINISHED.
Everything it was previously written for is settled - the ligand contact rule,
the distogram's ligand behaviour, the hosting - and those entries are gone
rather than kept as history.

## The state of the tree

`main` is deployed and verified: `python3 tools/deploy.py --verify` matches
HEAD, and localfold.org folds every model from Hugging Face. All eight bundles
are hosted, so the Pages build carries **0.0 MiB** of parameters. `npm test` is
795 passing; `tools/mobile-layout.py` and `tools/model-terms.py` exit 0.

The one commit that is NOT deployed at the time of writing is `aa3eaec`, which
is documentation and tooling only.

## Open, in the order they are worth doing

### Chain-level pTM: fixed on CUDA, the page and Metal; what is left (reported 2026-11-07)
Reported by Yeqing Lin ("localfold binder pTM bug and fix", against `500165d7`): a short
binder's `chain_ptm` read ~0.10 high against AF3. Three gaps, all now closed to AF3's own
functions (`confidence_head.py`'s interface `tmscore_adjusted_pae`, `confidences.py`'s
`chain_pairwise_predicted_tm_scores` / `get_iptm_xchain`, `model.py`'s `pae_single_mask`):
the chain-level scores reduce the INTERFACE term (d0 from the pair's own chains), `chain_iptm`
is the mean of a chain's pair ipTMs, and only tokens with frames anchor.

* **CUDA** `64685f25` - held by `tools/oracle/check_chain_ptm.py --dump=` on five jobs and two
  OpenDDE folds, exact. The sharded (multi-GPU) head is wired the same way and NOT RUN.
* **Page** `733d77d8` - `--scores=` off `tools/gpu/fold.js --tm-check`, exact on barnase and
  calmodulin (OpenBind-0) and OpenDDE; `test/chain-ptm-af3.test.js`. boltz2's page path untested.
* **Metal** `573a18a0` - **NOT BUILT OR RUN** (no Mac here). The interface term is taken on the
  host off the PAE logits (`metal/af3/frames.h`). Wants `metal/af3/gate.py` and
  `check_chain_ptm.py` on Apple silicon; Metal has no `AF3_PAE_OUT` dump yet, so the check is
  the `--scores`-style comparison or a dump to add.
* **Standardised (2026-11-07, the user's call): every model's AF3-named chain fields are AF3's.**
  `cuda/cpp/featurise/ptm.h` is the native ports' one definition (cuda/af3, cuda/ef2, metal/af3, metal/ef2);
  `shared/heads/tm-score.js` the page's (AF3 lineage, WebGL2 AF3, AF2, ESMFold2). ESMFold2 keeps ESM's own pTM / ipTM
  (ESM's reference: d0 and anchors over every token, no chain_ptm, a directional pair_chains_iptm). cuda/ef2
  barnase-barstar chain_ptm 0.9748/0.9795 -> 0.9585/0.9578. Still without chain fields: cuda/af2, metal/af2, the
  WebGL2 AF2 path. **Metal is edited blind** - docs/A100-TO-METAL.md is the Mac agent's brief.

### The two held-back pages
`single.html` and `proteinhunter.html` are gitignored and on disk, out of the
repository and off the site until they are fixed and checked - they were built
against a model row that has moved under them. Nothing that touched them was
deleted: `tools/build_site.py` copies them when they exist, and
`tools/mobile-layout.py` measures them when they exist and prints "not present,
skipped" when they do not. **Putting them back is one line in .gitignore and the
gates return with them.**

### The certainty's blind spot on short sequences
🔴 On a 35-mer, removing the language model takes the contact count from 27 to
**zero** while the certainty reads 0.8922 against 0.8978 - unmoved. Everything
is placed past the 12 A filter, so the mean falls onto the handful of
near-neighbours that survive, which are trivially peaked. **A certainty over very
few pairs is not a certainty about the fold**, and nothing on the page says how
many pairs it rested on. On a 76-mer the same ablation moves it 0.95 -> 0.42, so
it is a short-sequence failure and not a general one. Surfacing the pair count,
or refusing to report below some count, is the obvious move and is unmeasured.

### A ligand's colour is the least reliable thing on the page
Measured and recorded rather than fixed: the head predicts ligand-protein
distances at r = 0.683 against 0.999 for protein pairs, and a ligand's certainty
rests entirely on those cross-chain partners. `chain_pair_max_contact` says so in
the archive; the colour does not.

### ESMFold2-Fast and ESMFold2 (the 6B models)
Parked deliberately, twice over. They are `ESMFold2Model` and not the
`ESMFold2ExperimentalModel` this port implements - a parcae diagonal state-space
recurrence with a learned decay, a 4-layer LM encoder and a 2-layer coda - so it
is four new pieces, not one. And ESM-C 6B is ~2.9 GB at int3 against the 252 MiB
that ships. `esmfold2-fast-6b/` is downloaded (720 MiB, the fold half only) and
gitignored if that is picked up again. **They do have a real confidence head**,
which is the only reason to want them.

### Smaller
* Two cost models still exist side by side, `webgpu/runtime/cost-model.js` and
  `webgpu/esmfold2/cost.js`. They share shape but no code. Least dangerous kind of
  duplication - neither can corrupt a fold.
* The language band's constants are a two-point fit and carry this machine's
  drift: a second run of the same three folds read 1972, 919 and 63 ms against
  the fitted 2145, 1355 and 0.

## What NOT to redo

* **Do not re-shard `af3-int5`.** Its floor is a tensor: two 216 MiB float32
  tensors become 40.5 MiB each at int5, a tensor is contiguous within one file,
  and the shipped layout already isolates them. A 16-shard build produces the
  same two 40.5 MiB shards. Measured.
* **Do not chase `CONTACT_EDGES`.** Fitting the grid from observed distances
  disagrees with itself across pair kinds - widths 0.405 / 0.417 / 0.449 - and
  the residual bias on protein pairs is half a bin.
* **Do not read a ligand result from one ligand.** ATP 0.131, glycerol 0.211,
  haem 0.665 for the same protein.
* **The four `quantised-upload` files are on a branch of the user's**, not part
  of this work.

## The habit this session kept rewarding

Three separate bugs were one shape: **two readers of one thing that disagree.**
`modelFamily()` read the model row while everything else read `chosenFamily()`,
so the page folded 600M weights and labelled them 300M. `remote_families()`
matched only unquoted keys, so hosted bundles counted as local. `?model=` was a
family selector to one reader and a manifest URL to another, so `?model=monomer`
404'd a model that folds fine from the dropdown. Each was silent and each gave a
plausible wrong answer rather than an error.

The second habit: **a regression next to a change is not evidence it came from
it.** The `?model=` 404 appeared minutes after 129.8 MiB were deleted from the
one bundle that broke, and was unrelated. Folding the same model by another
route took thirty seconds and pointed elsewhere.
