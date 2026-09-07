# Working on LocalFold

`AGENTS.md` has the engineering invariants. This file is the operational half:
how to actually run things here, and the traps that have cost time more than
once. The findings themselves - some five hundred of them, every one
measured - live in
`docs/`, indexed at the bottom of this file. **Read the doc for a stage before
changing that stage**; this file's own recurring lesson is that a whole-fold
gate cannot see an error inside one of them.

## Running anything that needs a GPU

```
node tools/gpu-chrome.mjs tools/gpu/<module>.js [--flags]
```

It serves the repo over HTTP, drives headless Chrome, and calls the module's
`export async function main(device, args)`. Whatever `main` returns is printed
as JSON. Anything under `tools/gpu/` is written to that shape.

🔴 **`npm run test:gpu` DOES NOT WORK ON THIS MACHINE AND NEVER HAS.** The Dawn
node binding fails to load - *"built for macOS 26.0 which is newer than running
OS"* - so every `test/*.gpu.test.js` is unrunnable locally. That is the whole
reason `tools/gpu-chrome.mjs` exists. `npm test` (the CPU suite) does run, and
must pass.

🔴 **AND ONE `.gpu.test.js` NAMES A FIXTURE THAT IS NOT IN THE REPOSITORY.**
`test/evoformer-attention.gpu.test.js` wants
`test/fixtures/evoformer/model1-query-59-block0`, which does not exist; only
`model1-query-59-stack` does. So checking an attention change against official
values means the whole-stack checker, not that file.

## The tools, by what they answer

| Question | Tool |
|---|---|
| Does the AF3 head still match AF3? | `tools/gpu/probe-head-vs-af3-steps.js --dump=/af3-rings20.json` |
| Is a fold still the same fold? | `tools/gpu/probe-sidechains.js --steps=8` |
| Is a MODIFIED residue the right shape? | `tools/gpu/probe-modified.js --code=SEP --at=3` |
| What does AF2 predict, distogram and pLDDT, per recycle? | `tools/gpu/probe-af2-dgram-plddt.js --sample=10` |
| Is the sampler converged at this step count? | `tools/gpu/probe-flow-sigma-by-size.js --panel=churn` |
| Do recycles help a complex? | `tools/gpu/probe-recycles-on-complexes.js` |
| Does MSA depth help a complex? | `tools/gpu/probe-msa-depth-on-complexes.js` (**goes to the network**) |
| Does the sampler setting matter on a real binder? | `tools/gpu/probe-designed-binder-sampler.js` (**network**) |
| Does AF2's stack match AlphaFold? | `tools/gpu/check-evoformer-stack.js` |
| Does AF2 still fold the SAME structure? | `tools/gpu/fold-af2.js` |
| Does AF2's distogram head agree with AF2's structure? | `tools/gpu/probe-af2-contacts.js` |
| Which register tile does AF2's dense projection want? | `tools/gpu/bench-evoformer-linear.js` |
| What does AF2's column attention cost alone? | `tools/gpu/bench-msa-attention.js` |
| What does a sampler step cost besides the denoiser? | `tools/gpu/probe-sampler-overhead.js` |
| Where does a denoiser call's time go? | `tools/gpu/bench-head.js --profile` |
| Where does a trunk pass's time go? | `tools/gpu/bench-trunk.js --profile --msa=1024` |
| Where does an AF2 block's time go? | `tools/gpu/profile-af2-block.js --sequences=512` |
| Just the transformer, in 3 seconds? | `tools/gpu/bench-diffusion-transformer.js` |
| Which attention kernel does this device get? | `tools/gpu/probe-kernel.js` |
| Does this device have matrix units, and in what shapes? | `tools/gpu/probe-subgroup-matrix.js` |
| What do `subgroupMatrixLoad`/`Store` actually mean here? | `tools/gpu/check-subgroup-matrix.js` |
| Are the matrix units worth it on a dense projection? | `tools/gpu/bench-evoformer-linear.js --arms=8x8@f16/f16,matrix4` |
| ...and against AF3's own fused projections? | `bench-{grid,triangle}-project.js --tokens=200 --matrix=1` |
| What does packing the attention key cost, and the value? | `tools/gpu/check-attention-packing.js --dense=f32` |
| What does a dispatch cost before it computes? | `tools/gpu/probe-dispatch.js` |
| What does the page cost per frame? | `tools/gpu/bench-frame.js` |
| Which tile does a pairformer kernel want? | `tools/gpu/bench-{triangle-project,grid-project,transition,single-project,opm}.js` |
| Does the template embedder match AF3 with a REAL template? | `tools/oracle/check_af3_template_geometry.js` |
| Does AF2-multimer's template term match its reference? | `tools/gpu/check-multimer-template.js` |
| ...and AF2-MONOMER's? | `tools/gpu/check-monomer-template.js` |
| Does an AF2 kernel still compute AF2? | `tools/gpu/check-evoformer-{transition,opm,attention}.js`, `check-triangle-residual.js` |
| What is this device's actual ceiling? | `tools/gpu/probe-alu.js` |
| Where does the HOST memory go? | `tools/gpu/probe-memory.js` |
| How long does a fold take, by shape? | `tools/gpu/bench-runtime.js` (fits `src/runtime/cost-model.js`) |
| Is an AF3 fold's f16 path still worth it? | `tools/gpu/fold.js --staged= --weights=` (both arms, one shell) |
| Does the progress bar move at the fold's speed? | `tools/gpu/probe-progress-bar.js` |
| Does a failed fold keep its trunk for the retry? | `tools/gpu/probe-trunk-reuse-after-failure.js` |
| What does a fold hold on the DEVICE? | `tools/gpu/fold.js --budget=0` (prints per stage) |
| Does it still fold on a small device? | `tools/gpu/bench-trunk.js --budget=200` |
| Does the page fit a phone? | `python3 tools/mobile-layout.py` |
| Do the heatmap panel's tabs still work after a vendor bump? | `python3 tools/heatmap-panel.py` |
| Does a REAL fold put contacts on its frames? | `python3 tools/fold-in-page.py --model af3` |
| ...and does a template reach it? | `tools/fold-in-page.py --model af3 --template 1QYS_A` |
| ...and a MODIFIED residue? | `tools/fold-in-page.py --model af3 --modify SEP@3` |
| **Does the archive describe the job it wrote?** | `tools/fold-in-page.py --job-round-trip` (folds, WIPES the rows, drops the zip back) |
| Which of AlphaFold 3's own example jobs load here? | `node --test test/af3-example-jobs.test.js` (8 of 14; the other 6 name their field) |
| **Does the port fold at all?** | `node tools/fold-esmfold2.js` (6.5 min, writes a PDB) |
| **Does LocalFold fold a sequence the way ESMFold2 does?** | `node tools/check-esmfold2-fold.js` |
| Does the diffusion module agree, module by module? | `node tools/check-esmfold2-diffusion.js` |
| ...and on the GPU, a whole denoise step? | `tools/gpu/check-esmfold2-diffusion-gpu.js` |
| Does the sliding-window atom attention compute its reference? | `tools/gpu/check-esmfold2-atom-stack.js` |
| Does the featuriser build what ESMFold2 was handed? | `node tools/check-esmfold2-featurise.js` |
| **Does ESMFold2 fold on the GPU, sequence in, structure out?** | `tools/gpu/fold-esmfold2.js` |
| Which of a sampler step's two coordinate sets is the picture? | `tools/gpu/probe-esmfold2-trajectory.js` |
| What does an ESMFold2 fold cost, by band? | `src/esmfold2/cost.js` (fitted at 40, 150, 300) |
| Can anything stand in for the confidence head this checkpoint lacks? | `tools/gpu/probe-esmfold2-confidence.js` |
| Does the EDM sampler's schedule and step agree? | `node tools/check-esmfold2-sampler.js` |
| Does ESMFold2's trunk still compute ESMFold2's trunk? | `tools/gpu/check-esmfold2-trunk-gpu.js` |
| Does z_init's every term agree? | `node tools/check-esmfold2-featuriser.js` |
| What dtype is the atom attention actually holding? | `tools/esmc/probe-esmfold2-atom-attention.py` |
| ...and does the CPU reference? | `node tools/check-esmfold2-trunk.js` (77 s a loop) |
| Which convention does one ESMFold2 module want? | `node tools/check-esmfold2-modules.js` |
| What does ESMFold2's trunk cost, by length? | `tools/gpu/bench-esmfold2-trunk.js` |
| How small can ESM-C get before ESMFold2 notices? | `tools/esmc/probe-esmc-compression.py` |
| ...and what does that cost the STRUCTURE? | `.venv-esm/bin/python tools/esmc/probe-esmfold2-structure.py` |
| Where do I get ESM-C and ESMFold2? | `tools/esmc/fetch.py` (3.0 GB, ungated, MIT) |
| Turn ESM-C into a bundle the browser reads | `tools/export_esmc_model.py`, then `tools/quantize_af3.py --bits 3 --group 128` |
| What should a WebGPU ESM-C agree with? | `oracle-dumps/esmc-59.json`, from `tools/esmc/dump-esmc-oracle.py` |
| Does the ESM-C CPU reference compute ESM-C? | `node tools/check-esmc-reference.js` |
| ...and does the WebGPU block? | `tools/gpu/check-esmc-block.js` |
| ...and the whole 36-block tower, and the shim? | `tools/gpu/check-esmc-tower.js` |
| What does an ESM-C block cost? | `tools/gpu/bench-esmc-tower.js` |
| ...and is the tower right at more than one length? | `check-esmc-tower.js --dump=/oracle-dumps/esmc-{59,128,180}.json` |

| **Does OpenDDE fold?** | `tools/gpu/fold-opendde.js --target=6mrr` (RMSD 1.68 A, TM 0.865) |
| Does its structural-token expansion conserve the atoms? | `tools/gpu/check-opendde-expander.js` |
| **Are a model's BOND LENGTHS right, not just its fold?** | `tools/gpu/probe-nucleic.js --sequence= --model=` (RMSD cannot see this; OpenDDE is 15% short) |
| ...and a ligand's? | `tools/gpu/probe-ligand-flow.js --ligand=GOL --mode=diffusion` |
| Does OpenDDE's trunk predict a real fold's contacts? | `tools/gpu/trunk-opendde.js` (**and `--model=/model-af3-int5/manifest.json` is the control**) |
| Does a pairformer block match its reference at THIS bundle's widths? | `tools/gpu/check-af3-block-any.js --model=` |
| ...and which pair-track kernel is the one that does not? | `tools/gpu/probe-opendde-kernels.js --model=` |

`tools/gpu/check-af3-*.js` are the per-module AF3 oracle checkers.

🔴 **AND EVERY ONE OF THEM EXCEPT THE TWO ABOVE IS PINNED TO AlphaFold 3's
CONSTANTS.** `check-af3-triangle.js` has `const CHANNELS = 128`,
`check-af3-grid-attention.js` has 128 with 4 heads of 32, and
`check-af3-block.js` hand-builds its weight dict with `heads: 4` and
`pairChannels: 128` typed into it. So the differential suite is blind to a
second bundle's widths, which is exactly where a second bundle breaks - see
docs/OPENDDE.md, where a dispatch sized for 128 against kernels compiled for
384 left two thirds of every pair row unprocessed and every per-kernel checker
passing.

## The traps that repeat

🔴 **SO USE `python3 tools/serve.py` AND NOT `python3 -m http.server`.** It
sends `Cache-Control: no-store` on everything a developer edits and a year's
`max-age` on the weight shards, which are the one thing that must still cache -
a bundle is 346 MiB and re-downloading it per reload is the opposite problem.
The rest of this entry is what happens without it, and it cost three separate
sessions before the server existed.

🔴 **A PLAIN RELOAD SERVES CACHED ES MODULES, AND THAT LOOKS EXACTLY LIKE A
BROKEN FEATURE.** `python3 -m http.server` sends no cache headers, so Chrome
caches `web/app.js`, `src/af3/fold.js` and every other module heuristically -
and `location.reload()` does not refetch them. A change lands, the page is
reloaded, nothing happens, and the code looks wrong. Ask the page what it
actually loaded rather than what is on disk:

```js
(await import('/src/af3/fold.js')).foldBatch.toString().includes('recycle-done')
```

against `fetch('/src/af3/fold.js?v=' + Date.now())`. If they disagree, it is the
cache. ⌘⇧R clears it. `tools/fold-in-page.py` never sees this because it
launches a fresh Chrome profile, which is why it can pass while the browser in
front of you does not.

🔴 **AND `node tools/gpu-chrome.mjs` SOMETIMES DOES NOT EXIT.** The results file
is complete and correct and the node process sits there with a headless Chrome
still running, which in a `for` loop stalls every arm behind it. `pkill -9 -f
"gpu-chrome-"` matches the temporary profile directory and nothing else - not
the browser you are using. A batch of checkers should carry one between arms.

## Measuring, without fooling yourself

🔴 **PROFILE, DO NOT BISECT BY DELETION.** Disabling a pass and re-measuring
attributes scheduling and overlap to whatever was removed and has the bench's
noise for resolution. It has produced wrong answers here twice - once reporting
a *removed* pass as costing negative time, once naming the wrong kernel by 4x.
Two profilers exist and both work:

- `tools/gpu/profile.js` wraps `createCommandEncoder` and times every labelled
  compute pass. AF3 labels all of its passes, so this covers the AF3 side.
- AF2 has its own, older and better: `execution.beginTimestampProfile()` with
  `stack.js`'s `profileBlock` input, driven by `tools/gpu/profile-af2-block.js`.
  It is per *dispatch*, not per pass. `profile.js` cannot see into AF2, which
  batches a block's dispatches into one pass called `localfold.compute`.

Timestamps are quantised by Chrome to about 100 microseconds, so a single short
pass is unmeasurable; totals over many passes are fine.

🔴 **THIS MACHINE DRIFTS BY UP TO 3.2x BETWEEN RUNS.** Interleave A and B in one
process, or take a median of many calls - `bench-head.js` medians nine. A single
run of each is not a comparison.

🔴 **AND ONE PROCESS IS NOT ENOUGH IF THE PROCESS IS LONG.** "Run both arms in
one process" defeats the drift for two things measured back to back, and not for
a sweep that takes two minutes: the shapes run in sequence and the drift
accumulates across them. Two runs of `bench-runtime.js` on the identical shapes
disagreed by **-38% on AF3's trunk at 256 tokens and +25% on AF2's stack at 128
rows**, in opposite directions, which is not a property of either model - and a
fit over one of those columns moves the cubic term by 3x. Interleave the shapes,
not just the arms, and take medians.

## Deploying

```
python3 tools/deploy.py          # push main, dispatch the workflow, verify
python3 tools/deploy.py --verify # what is live right now
```

It ends by polling `https://localfold.org/build.json` until the commit it pushed
is the one being served, so "live" is a fact rather than an impression. Pages
builds from the pushed commit, so an uncommitted file cannot reach the site -
and will not be deployed either.

## Oracle dumps

```
python3 tools/oracle/dump_af3_trunk.py --blocks 48 --recycles 0 --diffusion 20 \
  --float32 --sequence <SEQ> \
  --capture 'diffusion_head/__call__$|evoformer/__call__$' \
  --capture-args 'diffusion_head/__call__$' --out <path>.json
```

`--capture-args` is what records the head's *inputs*, without which its answer
cannot be reproduced.

🔴 **EVERY DUMP LIVES IN `oracle-dumps/`, AND THE CHECKERS FETCH IT FROM
THERE.** They used to be written into the repository root, one `.gitignore`
line per file, and 300 MB of generated tensors sat beside `index.html` where a
reader cannot tell the project from somebody's afternoon. The directory is
ignored whole; the dump scripts default their `--out` into it and the checkers
fetch `/oracle-dumps/<name>.json`.

🔴 **AND THE FIXTURES ARE IN `tools/fixtures/`** - `1qys-crystal.pdb`,
`6mrr-crystal.pdb`, `test.a3m` and the reference AF3 server archive. They are
inputs to the tooling, not repository content, and they were nine PDB files and
two zips deep in the root before.

## Two habits worth keeping

- **Verify against the oracle, not against our own reference.** The side-chain
  bug survived for months because the only checker reaching the diffusion head
  builds its weight dict by hand instead of through the loader, so it passed
  while the shipped pipeline was wrong.
- **When a kernel's shape comes from a device limit, resolve it once and pass it
  down.** Resolving it in two places gave shaders tiling by four under a
  dispatch dividing by eight - half the tokens silently unprocessed, reported by
  the bench as a 30% speedup.

## Where the findings are

Each is a log of measurements, newest material appended: what was tried, what
it cost, and which of them are dead ends nobody should retry. They are long
because the numbers are the point - a claim here without one is a guess.

| doc | what is in it |
|---|---|
| `docs/AF3.md` | the AF3 port's state, costs and dead ends, and the **openbind0 dialect** - the three branches a second set of weights needs, and the caches that mistake one model for the other |
| `docs/EF2FAST.md` | the ESMFold2 600M port end to end: the trunk, the diffusion module, ligands and nucleic chains, the certainty estimate that replaces an absent confidence head, the pAE that was measured and withheld, and the compression study that preceded all of it |
| `docs/AF2.md` | the multimer and monomer template terms and the three dialects they are, the end-to-end fold gate, the four differential gates, and the alignment prep |
| `docs/PERF.md` | this device's ceilings, where the memory goes in each model, what f16 is worth **where**, the pair-scratch and aliasing work, and upstream's optimisations tried here |
| `docs/WEB.md` | the page: mobile layout, the template source menu, the download dial, the archive round trip, and the viewer |
| `docs/OPENDDE.md` | the OpenDDE port, and the place to START on it: two token spaces, its own confidence head, the gates it must hold, and what is open |
| `docs/HOSTING.md` | the weights are on Hugging Face, not Pages: how a bundle names its remote, and why a bundle wants more shards than connections |
| `docs/DEVELOPING.md` | the older orientation notes |
| `docs/HANDOFF.md` | the pLDDT-from-distogram attempt whose code was deleted, kept so the next attempt does not repeat it |
