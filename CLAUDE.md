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
| ...and did a KNOB change the structure, which pLDDT will not tell you? | `python3 tools/diff-fold-coords.py --b="--attn-splits=4"` - **`meanPlddt` matched to sixteen digits across an arm that moves 33 atoms** |
| Is a MODIFIED residue the right shape? | `tools/gpu/probe-modified.js --code=SEP --at=3` |
| What does AF2 predict, distogram and pLDDT, per recycle? | `tools/gpu/probe-af2-dgram-plddt.js --sample=10` |
| Is the sampler converged at this step count? | `tools/gpu/probe-flow-sigma-by-size.js --panel=churn` |
| Do recycles help a complex? | `tools/gpu/probe-recycles-on-complexes.js` |
| Does MSA depth help a complex? | `tools/gpu/probe-msa-depth-on-complexes.js` (**goes to the network**) |
| Does the sampler setting matter on a real binder? | `tools/gpu/probe-designed-binder-sampler.js` (**network**) |
| Does AF2's stack match AlphaFold? | `tools/gpu/check-evoformer-stack.js` |
| Do the outer product mean's TWO paths agree? | `tools/gpu/check-opm-paths.js --length=400 --sequences=512 --cz=128` - they contract the sequence axis in a different order, so the bar is a reordering (~1e-6) and not equality |
| ...and does PAIR BLOCKING change nothing at all? | the same tool - its blocked arm forces thousands of blocks and its bar is `=== 0`, because blocking reorders no sum |
| ...and is the f16 contraction still flat in MSA DEPTH? | the same tool, and **run it at `--sequences=512`**: depth is the only axis that arm's risk lives on, and at `--sequences=8` it proves nothing |
| ...and which path is a fold actually taking? | `useOuterFirstContraction` - it was capped at 64 MiB, i.e. **128 residues**, and the fallback is 93.7% of a block at 825 |
| Does AF2 still fold the SAME structure? | `tools/gpu/fold-af2.js` - and it FAILS now if the chain is not a chain, which is the gate a collapsed 825-residue fold walked through for a whole campaign |
| ...and does forcing an AF2 knob still fold the same one? | `tools/gpu/fold-af2.js --tune=key=value`, the same flag `fold.js` carries. A knob no gate enters is a knob nobody has checked |
| Does AF2's distogram head agree with AF2's structure? | `tools/gpu/probe-af2-contacts.js` |
| Which register tile does AF2's dense projection want? | `tools/gpu/bench-evoformer-linear.js` |
| What does AF2's column attention cost alone? | `tools/gpu/bench-msa-attention.js` |
| What does a sampler step cost besides the denoiser? | `tools/gpu/probe-sampler-overhead.js` |
| Where does a denoiser call's time go? | `tools/gpu/bench-head.js --profile` |
| ...and is a pass filling the device, or just slow? | the same, and read `groupsPerPass` |
| ...and where does the 90% that is NOT a compute pass go? | `tools/gpu/fold.js --buffers` (profile.js sees 10% of a fold) |
| Does the sample dimension leave the one-sample path alone? | `tools/gpu/check-difftx-samples.js` |
| ...and does the batched path compute the same thing S times? | `tools/gpu/check-difftx-batched.js` |
| ...and does every K-split, tile and hoist compute the UNSPLIT answer? | `tools/gpu/check-difftx-splits.js` (**150 arms**; tiles and the conditioning hoist are held to relRms EXACTLY 0, K-splits to 1e-3) |
| ...and does that checker's reference say "off" for the thing under test? | it must - the prior turns `batchedGates` ON, so a reference naming only the split counts compared the hoisted path **against itself**. Falsify every arm you add. |
| ...and is a split worth its pass, below the drift? | `tools/gpu/bench-difftx-splits.js` (paired, interleaved, withholds a timing if the arms disagree) |
| Comparing two tensors in a new checker? | `tools/gpu/relative-rms.js` - **an unguarded relRMS over a non-array returns exactly 0**, which is a perfect score from comparing nothing |
| Where does a trunk pass's time go? | `tools/gpu/bench-trunk.js --profile --msa=1024` |
| Where does an AF2 block's time go? | `tools/gpu/profile-af2-block.js --sequences=512` |
| ...and ALL of it, not the top twenty? | the same, `--top=200` - a block's transitions alone are fifty labels, one a chunk, and 15 ms of pair bias hid under them |
| ...and which value of an AF2 knob does this block want? | `profile-af2-block.js --sweep=opmProjectOutputPairs=1,2,4 --watch=opm.project-output` - arms interleaved, two rounds, minimum per arm, weights loaded once |
| Just the transformer, in 3 seconds? | `tools/gpu/bench-diffusion-transformer.js` |
| Which attention kernel does this device get? | `tools/gpu/probe-kernel.js` |
| Does AF3's `grid.attend` on the MATRIX units still compute `grid.attend`? | `tools/gpu/check-grid-attend-matrix.js` - it takes NO bundle, so OpenDDE's head width of 8 and the template embedder's 16 are checked on a box that has only AF3's weights, and it sweeps the token count across both of the kernel's tails |
| ...and which geometry does it want? | `bench-grid-attend-passes.js --arms=scalar,4x32,2x16`, arms interleaved - but the answer that counts is `bench-trunk.js --profile --tune=gridAttendMatrix=true`, in the trunk |
| Does a pairformer block still match its reference, under a forced knob? | `tools/gpu/check-af3-block-any.js --tune=key=value` - and it FAILS now, on a bound that follows the arm |
| Do all eight attention kernels agree, the matrix one included? | `tools/gpu/check-attention-variants.js` - and it takes `--tune=attentionMatrixTile=4x32`, because the matrix arm has a GEOMETRY and one no checker has run is one nobody has checked. Its bar is f16 (~1e-3), not the 5e-5 the f32 variants hold |
| ...and does the f32 attention path still compute f32? | `check-evoformer-attention.js` - where the device picks the matrix kernel, its f32 arm runs a SECOND time with `attentionMatrix` off, because the bound follows the KERNEL and not the requested storage |
| Does this device have matrix units, and in what shapes? | `tools/gpu/probe-subgroup-matrix.js` |
| ...and what do its type parameters MEAN at a non-square shape? | `tools/gpu/check-subgroup-matrix-shapes.js` |
| What do those matrix units ISSUE at? | `tools/gpu/probe-matrix-ceiling.js` |
| Does the staged matrix projection compute a projection, in every precision? | `tools/gpu/check-staged-matrix.js` |
| Is a projection short of threads, short of arithmetic intensity, or neither? | `tools/gpu/probe-split-k.js` |
| What do `subgroupMatrixLoad`/`Store` actually mean here? | `tools/gpu/check-subgroup-matrix.js` |
| Are the matrix units worth it on a dense projection? | `tools/gpu/bench-evoformer-linear.js --arms=8x8@f16/f16,matrix4` |
| ...and against AF3's own fused projections? | `bench-{grid,triangle}-project.js --tokens=200 --matrix=1` |
| What does packing the attention key cost, and the value? | `tools/gpu/check-attention-packing.js --dense=f32` |
| What does a dispatch cost before it computes? | `tools/gpu/probe-dispatch.js` |
| What does the page cost per frame? | `tools/gpu/bench-frame.js` |
| Which tile does a pairformer kernel want? | `tools/gpu/bench-{triangle-project,grid-project,transition,single-project,opm}.js` |
| Why is one layer norm at half another's bandwidth? | probably its CHUNK, not its reduction - a chunk is its own dispatch and 2048 workgroups is 59% of an A100. `pairTransitionChunkBytes` raises it, and on this box that is a 6% GPU win and a 1% WALL loss for 144 MiB |
| Which block do the staged matrix GEMMs want? | `stagedMatrixBlock`, swept with `bench-esmfold2-trunk.js --profile --tune=stagedMatrixBlock=64x128x16x1x8` - **1.16x**, and the block that won the standalone GEMM bench is 16% off in the trunk |
| Does the matrix triangle projection compute the vector one's answer? | `tools/gpu/check-triangle-project-matrix.js` - the interleaved weight pack is inside the comparison, because an interleave off by a role returns a plausible tensor |
| Which pair transition does a stack take, and why? | `pairTransitionSplit` plus `TRANSITION_SPLIT_MIN_CHANNELS` - the split is 1.51x on an ESMFold2 trunk's GPU time and **1.8% of an AF3 one** for the same 72 MiB, so the default declines it below 192 channels |
| Is a transition still worth FUSING at this channel width? | `bench-transition.js --channels=N --arms=4,8:128,16:128,split` - the fused kernel holds the WIDENED row in workgroup memory, so its row tile halves as the channels double. 128: 1.13x for the split, 256: 2.77x, 384: 3.71x |
| ...and does the split compute the fused kernel's answer? | `tools/gpu/check-transition-split.js` (differential, f16 bar, ragged row counts) |
| What does NATIVE AF3 cost, at settings LocalFold can match? | `tools/oracle/bench_af3_native.py --steps=1 --recycles=0 --num-msa=1` |
| ...and why is sweeping `--num_diffusion_samples` the wrong way to ask? | the same file's header - samples are a vmap axis, not extra trajectories |
| Does the template embedder match AF3 with a REAL template? | `tools/oracle/check_af3_template_geometry.js` |
| Does AF2-multimer's template term match its reference? | `tools/gpu/check-multimer-template.js` |
| ...and AF2-MONOMER's? | `tools/gpu/check-monomer-template.js` |
| Does an AF2 kernel still compute AF2? | `tools/gpu/check-evoformer-{transition,opm,attention}.js`, `check-triangle-residual.js` |
| What is this device's actual ceiling? | `tools/gpu/probe-alu.js` (raise `--iterations` on anything faster than an M2) |
| ...and is its VECTOR ceiling real, or dead lanes? | `tools/gpu/probe-alu-lanes.js` |
| Which per-device kernel knobs does THIS device want? | `tools/gpu/probe-tuning.js` |
| ...and does forcing one still fold the SAME structure? | `tools/gpu/fold.js --tune=key=value` (a knob no gate enters is a knob nobody has checked) |
| ...and the PER-KERNEL tiles and splits, which `--tune` cannot reach? | `fold.js --attn-tile= --out-tile= --attn-splits= --norm-splits=` (they live inside `diffusionSplitK` and `--tune` splits its argument on commas) |
| Is a conditioning projection still inside the block loop? | it should not be - see `packZeroGateWeights`, and `--tune=diffusionBatchedGates=false` is the arm without it |
| What is the f16 path worth, on any tool? | add `--f16=off` / `--f16=on` to it (one switch, all models) |
| Is bfloat16 usable, and would it beat the f16 storage? | `tools/gpu/probe-bf16.js` |
| What does the host-device bus cost, each way? | `tools/gpu/probe-bus.js` (**free on an M2, not on a discrete GPU**) |
| What is `grid.attend` alone, without the copies? | `tools/gpu/bench-grid-attend-passes.js` |
| Where does the HOST memory go? | `tools/gpu/probe-memory.js` |
| How long does a fold take, by shape? | `tools/gpu/bench-runtime.js` (fits `src/runtime/cost-model.js`) |
| Is an AF3 fold's f16 path still worth it? | `tools/gpu/fold.js --staged= --weights=` (both arms, one shell) |
| Does the progress bar move at the fold's speed? | `tools/gpu/probe-progress-bar.js` |
| Does a failed fold keep its trunk for the retry? | `tools/gpu/probe-trunk-reuse-after-failure.js` |
| What does a fold hold on the DEVICE? | `tools/gpu/fold.js --budget=0` (prints per stage) |
| Does it still fold on a small device? | `tools/gpu/bench-trunk.js --budget=200` |
| Does the page fit a phone? | `python3 tools/mobile-layout.py` |
| Do the heatmap panel's tabs still work after a vendor bump? | `python3 tools/heatmap-panel.py` |
| Does a REAL fold put contacts on its frames? | `python3 tools/fold-in-page.py --model af3` - **and it runs on Linux now**, headful under `DISPLAY=:99`; it was pinned to a macOS Chrome path and `--headless=new`, which is why the contact overlay could break here unnoticed |
| ...and does a template reach it? | `tools/fold-in-page.py --model af3 --template 1QYS_A` |
| ...and a MODIFIED residue? | `tools/fold-in-page.py --model af3 --modify SEP@3` |
| **Does the archive describe the job it wrote?** | `tools/fold-in-page.py --job-round-trip` (folds, WIPES the rows, drops the zip back) |
| Which of AlphaFold 3's own example jobs load here? | `node --test test/af3-example-jobs.test.js` (8 of 14; the other 6 name their field) |
| **Does the port fold at all?** | `node tools/fold-esmfold2.js` (6.5 min, writes a PDB) |
| Is a fold from ANY of the four models actually a chain? | they all assert on it now - `tools/gpu/chain-geometry.js` holds the one band, `--allow-broken-geometry` is the escape hatch, and `test/chain-geometry.test.js` gates the rule where the weights are not |
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
| Where do I get any model's WEIGHTS? | **already built and published** - `src/reference/manifests/index.js` has a `remote:` per family, all under `huggingface.co/sokrypton/localfold/resolve/<sha>/<family>/`. Download those; the export pipelines below are for making a NEW bundle, not for getting an existing one |
| Where do I get ESM-C and ESMFold2, to build one? | `tools/esmc/fetch.py` (3.0 GB, ungated, MIT) |
| Turn ESM-C into a bundle the browser reads | `tools/export_esmc_model.py`, then `tools/quantize_af3.py --bits 3 --group 128` - **that group is ESM-C's alone.** AF3's and OpenDDE's bundles are `--group 32`, which is the default, and OpenDDE at 128 folds 6MRR into a 3283 A explosion at pLDDT 46.69 |
| What should a WebGPU ESM-C agree with? | `oracle-dumps/esmc-59.json`, from `tools/esmc/dump-esmc-oracle.py` |
| Does the ESM-C CPU reference compute ESM-C? | `node tools/check-esmc-reference.js` |
| ...and does the WebGPU block? | `tools/gpu/check-esmc-block.js` |
| ...and the whole 36-block tower, and the shim? | `tools/gpu/check-esmc-tower.js` |
| What does an ESM-C block cost? | `tools/gpu/bench-esmc-tower.js` |
| ...and is the tower right at more than one length? | `check-esmc-tower.js --dump=/oracle-dumps/esmc-{59,128,180}.json` |

| **Does OpenDDE fold?** | `tools/gpu/fold-opendde.js --target=6mrr` (RMSD 1.68 A, TM 0.865) - and its bundle wants **`export_af3_model.py --include diffuser`**, because the default is trunk plus distogram head and this tool needs `structural_token_expander` |
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

🔴 **AND `meanPlddt` IS NOT A BIT-EXACTNESS GATE, however many digits it
prints.** A night of kernel work reported it identical to SIXTEEN DIGITS -
84.20887255253277 - on every arm, including one that moves thirty-three atoms.
It is a PREDICTED confidence averaged over every atom in the structure, and it
is flat well past the digits a small coordinate change reaches. Measured at 68
tokens, 200 steps, with `tools/diff-fold-coords.py`:

| arm | max \|dx\| | atoms identical |
|---|---:|---:|
| a token tile, which reorders nothing | 0.000000 A | 574/574 |
| hoisting the conditioning projections | 0.000000 A | 574/574 |
| `attnSplits` 1 -> 4, which regroups a sum | 0.001000 A | **541/574** |

So "pLDDT unchanged" means "not obviously broken", never "bit-exact". For that,
compare coordinates, or take relRMS on the raw tensor with
`tools/gpu/bench-difftx-splits.js`.

🔴 **AND A SHADER THAT INDEXES FROM `.x` ALONE, UNDER A GRID THAT FOLDS, STOPS
DEAD AND SAYS NOTHING.** `execution.linearGrid` and `rowGrid` fold into y past
32768 workgroups; a kernel whose `fn main` reads `id.x` and never `id.y` then
recomputes the FIRST `32768 * 64` elements once per y row and never writes its
own. AF2's global column attention did this, so past 2,097,152 elements the
extra alignment's keys and values were never written and the kernel attended
over recycled scratch. It cost every fold above ~500 residues its entire
structure - an 825-residue chain collapsed into a ball two angstroms across,
consecutive CA 0.06 A apart - **while pLDDT ROSE to 69.31 and pTM to 0.9672**.
`execution.dispatch` throws above 65535 workgroups, so the only silent
combination is a folding caller with a shader that ignores y. The audit is one
grep and it is in docs/AF2.md.

🔴 **AND AN ALLOW-LIST OF OPTIONS IS A LIST THAT GOES STALE.** Twice now, at the
same seam in two files. `predictA3m` hand-copied five named options through to
`predict`; when `pairHost` was added for the distogram overlay the list dropped
it, so `web/app.js` asked for the host pair representation, got `undefined`, and
**the contact overlay silently vanished from the shipped page** - while the one
probe that scores that head had been broken by the same commit in the same way.
`src/multimer/model.js` had already been through this (it dropped the whole
multimer regime and ran multimer weights on the monomer graph) and forwards the
whole object. Forward the object.

🔴 **AND THE FOLD GATE HAD THE EVIDENCE AND NEVER LOOKED AT IT.** `fold-af2.js`
had printed `caca` since it was written and nothing asserted on it, so the
collapse passed as "the same fold" for a whole campaign of kernel work. It
asserts now, and so do the other three: `fold.js`, `fold-opendde.js` and
`fold-esmfold2.js` all computed the same distance under comments saying it was
the check that mattered, and all three reported it and moved on. One rule, in
`tools/gpu/chain-geometry.js`, gated by `test/chain-geometry.test.js` because
three of the four tools need weights that are not on every box. **A number a
tool prints is not a gate until something fails on it.**

🔴 **AND A GATE THAT CANNOT FAIL IS NOT A GATE, SO MAKE IT FAIL ONCE.** The
wiring here has a silent failure mode of its own: the verdict skips a fold with
no alpha carbons, so a field name typo hands it `undefined`, `Number.isFinite`
says no, and every fold "passes" as a ligand. The way to know is to narrow the
band, watch the gate throw on a healthy fold, and put the band back.

🔴 **AND A DIFFERENTIAL GATE IS ONLY DIFFERENTIAL IN WHAT YOU ACTUALLY VARIED.**
Comparing the conditioning hoist on against off ALSO flips `normKSplits`,
because the shader factory forces it to 1 when the projection is batched. The
naive comparison reads 538/574 and looks like the hoist is inexact; pinning
`--norm-splits=1` on both sides gives 574/574. Two knobs moved, one conclusion
drawn, and it was the wrong one until the second knob was held.

🔴 **A RUNTIME LOOP BOUND IN A HOT WGSL LOOP COSTS 4.3x, AND IT LOOKS LIKE
NOTHING.** `opm.project-output`'s inner loop is one workgroup read, one global
read and one multiply-add. Written with a constant trip count it is 26.2 ms at
825 residues; written as `for (var t = 0u; t < available; t += 1u)` where
`available = min(CHUNK, CELLS - cell0)` - the same arithmetic, in the same
order, over the same cells - it is **113 ms**. The compiler cannot unroll a
count it cannot see, and in a three-instruction body the loop overhead is half
the kernel. Chunk loops here are GENERATED, one block per chunk with its exact
count typed in, for exactly this reason.

The failure mode is that the two are indistinguishable in review, and the
tempting conclusion is "chunking is slow" - which was measured and is the
opposite of true: at a constant bound, chunking HELPS.

🔴 **AND A DYNAMIC VECTOR INDEX IN A HOT WGSL LOOP COSTS 4x, FOR THE SAME
REASON.** `unpack2x16float(word)[at & 1u]` subscripts a vec2 by a value the
compiler cannot fold, so WGSL puts the pair in addressable memory instead of a
register: `opm.project-output` went 26.2 ms to **103.2**, four times worse than
the f32 it was meant to beat. Hoisting the choice out of the loop -
`let odd = (z & 1u) == 1u;` once, then `select(word.x, word.y, odd)` - takes it
back to 27.8. Index a vector by a constant or select on a hoisted condition;
never subscript one by a loop variable.

🔴 **AND THE MATRIX UNITS PAY ONLY WHERE K DIVIDES BY FOUR - FOR A PLAIN GEMM.**
docs/A100.md's rule was "the operand is materialised, the problem is past a
billion multiply-accumulates, and K is deep". A fourth condition for a GEMM: the
win is almost all in `vectorStaging`, and a vec4 operand read needs the INNER
extent divisible by four. The outer product mean's K is the alignment depth
(512) and it went 2.25x; the triangle multiplication's K is the protein's
LENGTH, and at 825 it went 0.95x. Check `inner % 4` before writing the kernel.

🔴 **AND `inner % 4` IS A CONSTRAINT ON ONE OPERAND, NOT A VERDICT ON A
KERNEL.** AF3's triangle contraction transposes exactly one of its two operands
- the right one going out, the left one coming in - and the transposed one
cannot be vec4-read because its four consecutive elements are `columns` apart.
Turning vector staging off for BOTH, which is what "the inner extent is the
protein's length" seems to imply, makes the kernel a wash: 64.24 ms against the
vector kernel's 63.94. Vectorising the operand that CAN be takes it to 47.82.
`vectorStaging` is per operand for exactly this.

🔴 **BUT DO NOT CARRY THAT RULE TO A KERNEL THAT IS NOT A PLAIN GEMM.** It was
used to rule out a subgroup-matrix flash attention on the grounds that
`head_dim` is 32 - and that inference was wrong. A GEMM at K = 32 amortises its
panel over two k steps; a flash attention stages the query tile ONCE and reuses
it across the whole key sequence. The reuse is in the loop, not in K.
alphafold2-webgpu measures its matrix flash kernel at 1.66x-1.84x the register
one; `src/evoformer/attention-matrix.js` now measures 1.69x here. **Measure a
kernel in the stack it runs in, never as a standalone dispatch** - which is this
file's own rule, two sections down, and it was not followed.

🔴 **AND A LATENCY-BOUND KERNEL'S TILE IS CHOSEN BY WORKGROUP BYTES A LANE, NOT
BY WHAT IT AMORTISES.** The matrix flash attention's first geometry was 1.26x
slower than the kernel it replaced, and the obvious repair - a bigger key tile,
to spread the barriers and the per-tile output rescale over more keys - made it
worse in exact proportion to the shared memory it took:

| tile | bytes a lane | block ms |
|---|---:|---:|
| 4x32 | 242 | 324.6 |
| 2x32 | 276 | 341.0 |
| 2x64 | 438 | 441.1 |
| 2x128 | 762 | 568.8 |

Occupancy is capped by shared memory, so **every fixed cost such a kernel is
built to amortise is worth less than the bytes it takes to amortise it**. What
paid was moving three arrays into registers - the running output, the row
statistics (two lanes share a row and the pair is adjacent, so
`subgroupShuffleXor(v, 1u)` replaces an array AND two barriers), and the logits
the two softmax passes were handing each other through memory.

🔴 **BUT THAT RULE IS A PROPERTY OF AN OCCUPANCY-STARVED KERNEL, NOT OF THE
UNITS.** The same flash attention on AF3's `grid.attend` does not rank by bytes
a lane at all - at 512 tokens the winner is the tile that takes the MOST of them
(4x32, 169 bytes a lane, 148.6 ms) and the two cheapest are the two slowest
(6x16 at 130 bytes, 166.1; 4x16 at 136, 162.4). A pairformer pass launches
`n * heads` workgroups a block, 16,384 of them at 512 tokens over eight, so the
device is full at any of these tiles and what is left to win is the fixed cost a
bigger tile amortises - the opposite trade. **Count the workgroups before
deciding which of the two rules applies.** docs/A100.md has both.

🔴 **AND THE TILE'S OPTIMUM MOVES WHEN THE REST OF THE KERNEL DOES.** Three
hoists out of the per-key work - a bias index costing two integer multiplies a
key for a base constant across the whole loop, a bounds check true on every tile
but the last, and a per-key mask being read once per query ROW - were worth
10-14%, and they moved the best geometry from 6x16 to 4x32, with each 5-6% worse
at the other's optimum. **Re-sweep the tile after every change to the body**,
and never adopt one from another port.

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
| `docs/A100.md` | the same kernels on an **A100**: how to get a real WebGPU adapter on Linux/NVIDIA at all, what this repository's M2-measured conclusions do here, and the four that invert |
| `docs/WEB.md` | the page: mobile layout, the template source menu, the download dial, the archive round trip, and the viewer |
| `docs/OPENDDE.md` | the OpenDDE port, and the place to START on it: two token spaces, its own confidence head, the gates it must hold, and what is open |
| `docs/HOSTING.md` | the weights are on Hugging Face, not Pages: how a bundle names its remote, and why a bundle wants more shards than connections |
| `docs/DEVELOPING.md` | the older orientation notes |
| `docs/HANDOFF.md` | the pLDDT-from-distogram attempt whose code was deleted, kept so the next attempt does not repeat it |
