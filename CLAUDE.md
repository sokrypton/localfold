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

🔴 **`npm run test:gpu` NEEDS A DAWN BUILT FOR THIS GLIBC, AND THE SHIPPED ONE
IS NOT.** `webgpu@0.6.0`'s `linux-x64` binary wants `GLIBC_2.38`; this box has
2.35, so `import("webgpu")` throws and every `test/*.gpu.test.js` was
unrunnable. That is the whole reason `tools/gpu-chrome.mjs` exists - and it is
NOT the reason the note here used to give, which was a macOS message from the
other machine.

**`webgpu@0.4.0`'s Linux binary wants only `GLIBC_2.34` and loads here**, on the
real adapter: `nvidia / ampere / nvidia-a100-sxm4-40gb`, with
`chromium-experimental-subgroup-matrix` and four matrix configurations.

```
npm i webgpu@0.4.0 --no-save     # not the pin: macOS wants 0.6.0
XDG_RUNTIME_DIR=/tmp/xdg npm run test:gpu
```

🔴 **AND ITS ADAPTER HAS NO `shader-f16`, WHERE CHROME'S DOES.** That is where
this file's old claim that neither machine has the feature came from - it was
measured through Dawn. Chrome on this same card reports `shader-f16` on both the
adapter and the device, and the fold uses it: the f16 paths run here and always
have. Two suites skip their f16 arms under Dawn and must be run through
`tools/gpu-chrome.mjs` instead.

`npm test` (the CPU suite) does run, and must pass.

🔴 **AND IT NEEDS `--js-float16array` ON A NODE THAT LACKS `Float16Array`.**
Twelve tests did not FAIL on node 22, they never RAN - `ReferenceError:
Float16Array is not defined` out of `float16.test.js`, `int5-tensor.test.js` and
their neighbours, reported as twelve known failures for long enough to become
folklore. V8 has the type behind a flag, and with it the suite is **971 pass, 0
fail**. `npm test` asks for the flag only when this node lacks the type and its
V8 lists the flag, so a node that already ships it and a node too old to know it
both run unchanged. Do not polyfill it: `float16.test.js` exists to check this
repository's conversion against the platform's own.

🔴 **AND MOST OF THAT SUITE NAMES FIXTURES THAT ARE NOT IN THE REPOSITORY.**
Run for the first time on this box: 42 tests, 3 pass, 25 fail. **24 of the 25
are `ENOENT`** - `test/fixtures/evoformer/model1-query-59-block0` and
`model1-a3m-59-stack` are absent whole, and ten `*_haiku_*.f32.bin` weights are
missing from the `model1-query-59-stack` that IS present. So checking an
attention change against official values still means the whole-stack checker,
not `test/evoformer-attention.gpu.test.js`. Two more failures are the missing
`shader-f16` above.

🔴 **THE TWENTY-FIFTH WAS A REAL KERNEL BUG, NINE MONTHS OLD AND NEVER RUN.**
See the triangle rows below: `TriangleMultiplicationOutgoing` missed its
OpenFold reference by 8.26e-2 against a 1e-5 bound while three AlphaFold
fixtures passed, because the fixture is **cZ 7 and every shipped width is even**.

## The tools, by what they answer

| Question | Tool |
|---|---|
| Does the GPU diffusion conditioning match the reference, at THIS bundle's widths? | `tools/gpu/check-af3-diffusion-conditioning.js --model=` - it passes on af3, openbind0 and opendde now, at 1e-5 with a separation control of 1232-2362x. 🔴 It passed for as long as OpenDDE's bundle existed WITHOUT COMPUTING ANYTHING, because `NaN > 1e-5` is FALSE; every comparison written `if (x > bound) throw` here has the same hole. 🔴 AND IT SWEEPS TWO DIALECTS OVER ONE BUNDLE, so it must re-cut the weights BOTH ways - it could only splice the openfold3 columns in, never strip them, and an openbind0 bundle's own 833 columns then tripped the LayerNorm assertion inside the kernel it was there to measure |
| Does the AF3 head still match AF3? | `tools/gpu/probe-head-vs-af3-steps.js --dump=/af3-rings20.json` |
| Is a fold still the same fold? | `tools/gpu/probe-sidechains.js --steps=8` |
| ...and did a KNOB change the structure, which pLDDT will not tell you? | `python3 tools/diff-fold-coords.py --b="--attn-splits=4"` - **`meanPlddt` matched to sixteen digits across an arm that moves 33 atoms** |
| Is a MODIFIED residue the right shape? | `tools/gpu/probe-modified.js --code=SEP --at=3` |
| **Is a fold the SAME fold twice in one process?** | `tools/gpu/bench-af2-warm.js` - it checksums every pass and throws unless they match, because three predictions of one graph over one input in one process share their pipelines, their buffers and their clocks, so a difference is a RACE and not a precision question. It caught one in the matrix flash attention; `--allow-nondeterminism` is the escape |
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
| **Does the MULTIMER still fold, and what does a repeat cost?** | `tools/gpu/fold-af2.js --family=multimer --chains=30,29 --repeat=3` - the repeat is the only number that prices weight residency, because a first fold is mostly pipeline compilation (1485 ms against 214 for a repeat), and every repeat is held to the first fold's atom checksum. The bundle is `af2-multimer/` in the registry and needs a manifest.json generated from src/reference/manifests/multimer.js |
| ...and does forcing an AF2 knob still fold the same one? | `tools/gpu/fold-af2.js --tune=key=value`, the same flag `fold.js` carries. A knob no gate enters is a knob nobody has checked |
| **Does the device's nearest-centre search agree with the host loop?** | `tools/gpu/check-nearest-centres.js` - bar ZERO differing assignments over seven cases in one submit, because the assignment picks which cluster an extra row joins and so decides the prediction. 🔴 One arm DUPLICATES centres so whole groups tie, since the host keeps the FIRST at an equal score and a join that broke the other way agrees on every random alignment and disagrees on every real one. Two degenerate arms assert centre 0 outright, so the distinctness control cannot pass them by accident. Flipping the tie rule fails six of seven, 512/512 on the duplicate arm |
| ...and does the FOLD agree, not just the kernel? | `tools/gpu/fold-af2.js --host-features` is the control arm. Same checksum both ways - 26706680 at 825 residues, -1725774 at 59 - which is the only thing that settles it, because the assignment reaches the answer through the cluster profile |
| Where does preparing an alignment actually go? | `featureMilliseconds` on `fold-af2.js`, from `featureStats` in src/input/a3m-features.js. At 825 residues with 512 clusters, 1024 extras and two recycles: nearest-centre 645 ms of 1072, then the 49-channel block at 204, the cluster profile at 81, and nothing else above 55. The search is 43 ms on the device |
| Does AF2's distogram head agree with AF2's structure? | `tools/gpu/probe-af2-contacts.js` |
| Which register tile does AF2's dense projection want? | `tools/gpu/bench-evoformer-linear.js` |
| What does AF2's column attention cost alone? | `tools/gpu/bench-msa-attention.js` |
| What does a sampler step cost besides the denoiser? | `tools/gpu/probe-sampler-overhead.js` |
| Where does a denoiser call's time go? | `tools/gpu/bench-head.js --profile`, and `--calls=4` for the COLD one: at 68 tokens call 0 is 1218 ms against 10 steady, and 904 of it is the transformer's resident weights - not its compile, which `probe-warm.js` measures at 71 ms |
| Where does an AF3 FOLD's time go, by stage? | `tools/gpu/fold.js --folds=2` and read `stageMilliseconds` - a caller's clock attributes the gap between two stages to the EARLIER one, so a fold whose named stages stopped at `trunk-done` hid 2.0 s of a 3.2 s first fold |
| ...and is a pass filling the device, or just slow? | the same, and read `groupsPerPass` |
| ...and where does the 90% that is NOT a compute pass go? | `tools/gpu/fold.js --buffers` (profile.js sees 10% of a fold) |
| Does the sample dimension leave the one-sample path alone? | `tools/gpu/check-difftx-samples.js` |
| ...and does the batched path compute the same thing S times? | `tools/gpu/check-difftx-batched.js` |
| ...and does every K-split, tile and hoist compute the UNSPLIT answer? | `tools/gpu/check-difftx-splits.js` (**150 arms**; tiles and the conditioning hoist are held to relRms EXACTLY 0, K-splits to 1e-3) |
| ...and does that checker's reference say "off" for the thing under test? | it must - the prior turns `batchedGates` ON, so a reference naming only the split counts compared the hoisted path **against itself**. Falsify every arm you add. |
| ...and is a split worth its pass, below the drift? | `tools/gpu/bench-difftx-splits.js` (paired, interleaved, withholds a timing if the arms disagree) |
| Comparing two tensors in a new checker? | `tools/gpu/relative-rms.js` - **an unguarded relRMS over a non-array returns exactly 0**, which is a perfect score from comparing nothing |
| Where does a trunk pass's time go? | `tools/gpu/bench-trunk.js --profile --msa=1024` |
| Where does an OpenDDE FOLD's time go, and is it even GPU? | `tools/gpu/fold-opendde.js --profile --buffers --repeat=2` - the trunk phase is **96.3% GPU-idle**, and the second fold is the row a user sees |
| Where does an AF2 block's time go? | `tools/gpu/profile-af2-block.js --sequences=512` |
| ...and ALL of it, not the top twenty? | the same, `--top=200` - a block's transitions alone are fifty labels, one a chunk, and 15 ms of pair bias hid under them |
| 🔴 ...and were these knobs swept at the LENGTH that matters? | Mostly not - the priors were fitted at 400 residues and below. Re-swept at 825 with 512 sequences, one in four moved: `opmPairBlockBytes` 64 -> 256 MiB is **2.2% of a block and 180 ms of a fold for 193 MiB**, bit-identical because blocking reorders no sum. Flat: `attentionMatrixTile` (4x32 still right), `opmProjectOutputPairs`, `stagedMatrixBlock`, `transitionThreadTarget` over an eightfold range. See docs/AF2.md |
| ...and which value of an AF2 knob does this block want? | `profile-af2-block.js --sweep=opmProjectOutputPairs=1,2,4 --watch=opm.project-output` - arms interleaved, two rounds, minimum per arm, weights loaded once |
| Just the transformer, in 3 seconds? | `tools/gpu/bench-diffusion-transformer.js` |
| Which attention kernel does this device get? | `tools/gpu/probe-kernel.js` |
| Does AF3's `grid.attend` on the MATRIX units still compute `grid.attend`? | `tools/gpu/check-grid-attend-matrix.js` - it takes NO bundle, so OpenDDE's head width of 8 and the template embedder's 16 are checked on a box that has only AF3's weights, and it sweeps the token count across both of the kernel's tails |
| ...and which geometry does it want? | `bench-grid-attend-passes.js --arms=scalar,4x32,2x16`, arms interleaved - but the answer that counts is `bench-trunk.js --profile --tune=gridAttendMatrix=true`, in the trunk |
| Does a pairformer block still match its reference, under a forced knob? | `tools/gpu/check-af3-block-any.js --tune=key=value` - and `--resident`, because residency is a different WEIGHT PATH (the pair transition is decoded on the device) and not a cache - and it FAILS now, on a bound that follows the arm |
| Do all eight attention kernels agree, the matrix one included? | `tools/gpu/check-attention-variants.js` - and it takes `--tune=attentionMatrixTile=4x32`, because the matrix arm has a GEOMETRY and one no checker has run is one nobody has checked. Its bar is f16 (~1e-3), not the 5e-5 the f32 variants hold |
| ...and does the f32 attention path still compute f32? | `check-evoformer-attention.js` - where the device picks the matrix kernel, its f32 arm runs a SECOND time with `attentionMatrix` off, because the bound follows the KERNEL and not the requested storage |
| Does this device have matrix units, and in what shapes? | `tools/gpu/probe-subgroup-matrix.js` |
| ...and what do its type parameters MEAN at a non-square shape? | `tools/gpu/check-subgroup-matrix-shapes.js` |
| What do those matrix units ISSUE at? | `tools/gpu/probe-matrix-ceiling.js` |
| Does the staged matrix projection compute a projection, in every precision? | `tools/gpu/check-staged-matrix.js` |
| Does a REAL bundle's int5/int3/int8 decode the same on the device as on the host? | `tools/gpu/check-bundle-device-decode.js --bundle=` - and `check-quantised-upload.js` cannot answer it, because it lays out its own shard. Four reshape arms too: transpose, two-way interleave, column concatenation, and an f32 strided lane |
| A weight buffer is slower than it was and nothing errored? | 🔴 **the device packer refused and fell back.** It throws `DeviceWeightRefusal` naming the tensor now, because a silent fallback is a bug that looks like a slow machine - `allowHostWeightPacking` is the opt-in for a bundle that genuinely cannot be decoded. Ten descriptors in AF2 copied their tensors into a literal instead of deriving them, which READS the getter and decodes the block |
| Is a projection short of threads, short of arithmetic intensity, or neither? | `tools/gpu/probe-split-k.js` |
| What do `subgroupMatrixLoad`/`Store` actually mean here? | `tools/gpu/check-subgroup-matrix.js` |
| Are the matrix units worth it on a dense projection? | `tools/gpu/bench-evoformer-linear.js --arms=8x8@f16/f16,matrix4` |
| ...and against AF3's own fused projections? | `bench-{grid,triangle}-project.js --tokens=200 --matrix=1` |
| What does packing the attention key cost, and the value? | `tools/gpu/check-attention-packing.js --dense=f32` |
| What does a dispatch cost before it computes? | `tools/gpu/probe-dispatch.js` |
| **Did a speculative WARM compile the right shaders?** | count them: `tools/gpu/fold.js --no-warm` and `fold-opendde.js --no-warm` against the same run without the flag, through `probe-compiles.js`. A warm compiling against a shapes-only stand-in cannot give a wrong ANSWER - the stack still asks the cache for its own keys - so the only failure is silent WASTE. OpenDDE went 269 -> 322 pipelines warming its refiner with the trunk's root, and 285 with the wrong pair precision, before it went back to 269 |
| **Is the same WGSL being compiled twice under two keys?** | `distinctSources` and `duplicateModules` in `probe-compiles.js`. It was, for every model: OpenDDE 269 pipelines from 191 distinct texts, AF3 223 from 156, AF2 96 from 73, ESMFold2 95 from 82. `ComputePipelineCache` indexes by `entryPoint + source` as well as by key now - OpenDDE's fold 2540/2552/2603 ms to **2404/2403/2382**, monomer's page 3418 to 3141. 🔴 The source is the key and NOT a hash of it, because a collision would hand a caller somebody else's kernel |
| **Is a first fold waiting on SHADER COMPILATION?** | `tools/gpu/probe-compiles.js --tool=fold-af2` - it wraps another tool's `main`, so any gate can be measured without ceasing to be one. Read **`busyMs`**, the union of the intervals, and never the sum: twenty pipelines compiled at once and twenty compiled in turn have the same sum. AF2's first fold WAS its compile queue (1133 ms of span in a 1163 ms fold, 1.47 ever in flight); AF3's and OpenDDE's were already packed at ~14 |
| ...and what does a first AF2 fold consist of otherwise? | `tools/gpu/probe-af2-warmup.js` - pipelines, shader modules, buffers, writeBuffer bytes and the on-device weight decode, against the wall and the repeat |
| What does the on-device weight decode cost the HOST? | `blockUploadStats` in src/runtime/quantised-upload.js - staging assembly, submits and buffers, none of which is inside a compute pass and so none of which `profile.js` can see |
| ...and what is still DECODED on the host under those packers? | `tensorDecodeStats` in src/reference/http-tensor-store.js, printed as `hostDecode`. After this session: AF2 5 ms, AF3 32, ESMFold2 0, OpenDDE 84 |
| **A device weight path fell back and said nothing - what did that cost?** | `residentPackStats` in src/runtime/resident.js, printed by `probe-compiles.js` as `hostPack` with a per-label table. Every AF3-side device decode falls back to a host `pack()` SILENTLY by design, and one that costs 400 ms of a fold looks exactly like a slow machine: it found `difftx.zerogate.resident` at 496 ms in ONE call, and `w.pair-transition` at 56 calls because `residentPackedOnDevice` wrote halves only while AF3's pair track is f32 |
| What does the page cost per frame? | `tools/gpu/bench-frame.js` |
| Which tile does a pairformer kernel want? | `tools/gpu/bench-{triangle-project,grid-project,transition,single-project,opm}.js` |
| Why is one layer norm at half another's bandwidth? | probably its CHUNK, not its reduction - a chunk is its own dispatch and 2048 workgroups is 59% of an A100. `pairTransitionChunkBytes` raises it, and on this box that is a 6% GPU win and a 1% WALL loss for 144 MiB |
| What do the staged matrix GEMM's three knobs cost? | `stagedMatrixPrefetch` is bit-exact and ON (1.20x on an ESMFold2 trunk, 1.13x OpenDDE, 1.07x AF2); `stagedMatrixDirectWeights` needs an f16 weight BUFFER and is bit-exact given one; `stagedMatrixResult: "f16"` halves the accumulator registers and is 1.19x, and applied to `tri.contract` - whose K is the protein's length - it is 2178 NaN coordinates |
| Which block do the staged matrix GEMMs want? | `stagedMatrixBlock`, swept with `bench-esmfold2-trunk.js --profile --tune=stagedMatrixBlock=64x128x16x1x8` - **1.16x**, and the block that won the standalone GEMM bench is 16% off in the trunk |
| Does AF2's matrix q/k/v/gate projection compute the vector one's answer? | `tools/gpu/check-attention-project-matrix.js` - no bundle, three widths, and BOTH target storages, because a packed one doubles the column group so a lane owns both halves of its word. The four outputs are four different epilogues - the query scaled, the key and value plain, the gate a logistic of the only bias any of them has - so all four are compared and named |
| Does the matrix GRID projection compute the vector one's answer? | `tools/gpu/check-grid-project-matrix.js` - no bundle, both DIRECTIONS (the transposed one reads `(row % n) * n + row / n` and writes at `row`, and a wrong mapping returns a tensor of the right shape and magnitude), and OpenDDE's 8 heads as well as AF3's 4 |
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
| **Does the triangle multiplication match a reference that is NOT AlphaFold's?** | `tools/gpu/check-triangle.js` - OpenFold's recorded output at cZ 7, and this repository's own CPU path. It is the Chrome-lane twin of `test/triangle-multiplication-outgoing.gpu.test.js` and it is the ONLY independent reference this kernel has. 🔴 It was failing at 8.26e-2 against 1e-5 and is not in any gate list, so nothing ran it - on this branch and at `c881283` alike |
| ...and at which WIDTHS? | `tools/gpu/check-triangle-shapes.js` - the sweep that named it. Every ODD `cZ` was wrong and every even one right: the staged LayerNorm stores in PAIRS and took `count / 2u` words a row, so at cZ 7 it wrote three where the projection read four, dropping the last channel AND aligning every row one channel early. `CZ_STRIDE`/`CH_STRIDE` are the rounded-up row stride now, the tail guard is emitted only where the count is odd, and the two normalised buffers are sized to it |
| **Would the derived tuning be WORSE on a narrower GPU?** | `--occupancy=<n> --no-prior` on any GPU tool answers as a device of that width. Swept on AF3: default 6032 ms, width 16 **5030**, 64 4291, 256 3723, 2048 3446, the real measurement 3365, the prior 3406 - monotone, and no width slower than today. 🔴 It validates the CHOICE and not the OUTCOME: that arm is an A100 running a narrow device's configuration, which is why the atom tile's target is clamped never to fall below the shipped 256 |
| **How many workgroups does this device run at once?** | `tools/gpu/probe-occupancy.js` - the unknown behind `diffusionSplitK`, `atomRowTile` and `transitionThreadTarget`, which every geometry prior is a hand-estimate of. 4542 workgroups here by the plateau edge and 4432 by the slope past it, ~290,000 lanes. 🔴 Its own two traps: at a short chain every count below 1024 measures the submit round trip and the edge is invisible, and ONE workgroup is not the floor - it is 1.4 ms where 2 through 4542 are all 2.3 |
| What is this device's actual ceiling? | `tools/gpu/probe-alu.js` (raise `--iterations` on anything faster than an M2) |
| ...and is its VECTOR ceiling real, or dead lanes? | `tools/gpu/probe-alu-lanes.js` |
| Which per-device kernel knobs does THIS device want? | `tools/gpu/probe-tuning.js` |
| What does a GPU with NO prior get? | `--no-prior` on any GPU tool - and it means an unrecognised device WITH whatever units it has, since the capability layer below sits under the priors. `--default-tuning` is the older question: neither prior nor capability, which is what DEFAULT_TUNING alone is worth. AF2's warm repeat on this card: **416 ms with the prior, 406 with capability alone, 643 with neither** |
| ...and WHICH knobs is a prior actually worth? | `--no-prior=<knob>` restores one at a time from the prior. AF2's 218 ms splits `matrixLinear` 95, `attentionMatrix` 51, `attentionProjectMatrix` 39, `attentionGroup` 18, `opmMatrixContract` 12, `stagedMatrixPrefetch` 7, and `linearTallTile`/`triangleProjectMatrix`/`opmProjectOutputPairs` **zero**. Every significant one is "does this device have matrix units", which the API answers - see `matrixCapabilityTuning` |
| ...and does an unmeasured GPU get AF3's geometry now? | **97% of it.** `sample-start` on an unrecognised device: **4510** with every layer off, then 2784 / 2437 / 2252 / 2129 / **1952** as the K split, atom row tile, batched gate, token tile and weight retention are each derived - against the prior's **1874**, and a whole fold of 3398 ms against 3392. Five mechanisms, no new table: the device's measured WIDTH for the K split, the atom tile and the token tile; its memory BUDGET for the batched gate and for keeping weights between folds. 🔴 `--no-prior` KEEPS the derivations, because measuring is what an unrecognised device does; `--default-tuning` suppresses them, and without that switch the DEFAULT_TUNING arm silently drifted 4525 -> 3765 and was measuring something with no name |
| ...and how? | `derivedSplitRule` in src/af3/diffusion-transformer-webgpu.js asks the device's measured width whether the unsplit dispatch already fills it, so `crossover` - a token count in the prior - falls out of arithmetic instead. `sample-start` on an unrecognised device: 4500 before, **2784** derived, against the prior's 1877 and 2732 for the prior's own rule forced by hand. `fold.js --split-k=<json>` is how the rule is set by hand; `--no-prior` now measures, because an unrecognised device would |
| ...and is AF3's the same shape? | **No, and that is the finding.** `fold.js --folds=2 --no-prior=<knob>`: `sample-start` is 1867 ms with the prior and 4517 without, split `diffusionSplitK` 1792, `diffusionTokenTile` 1103, `diffusionBatchedGates` 700, `diffusionNormSplit` 526, `atomRowTile` 316 - tiles and split counts, none of which a capability states. The capability layer is worth 0 on AF3 (6014 ms against 6015) and that is expected |
| ...and does forcing one still fold the SAME structure? | `tools/gpu/fold.js --tune=key=value` (a knob no gate enters is a knob nobody has checked) |
| ...and the PER-KERNEL tiles and splits, which `--tune` cannot reach? | `fold.js --attn-tile= --out-tile= --attn-splits= --norm-splits=` (they live inside `diffusionSplitK` and `--tune` splits its argument on commas) |
| Is a conditioning projection still inside the block loop? | it should not be - see `packZeroGateWeights`, and `--tune=diffusionBatchedGates=false` is the arm without it |
| What is the f16 path worth, on any tool? | add `--f16=off` / `--f16=on` to it (one switch, all models) |
| Is bfloat16 usable, and would it beat the f16 storage? | `tools/gpu/probe-bf16.js` |
| What does the host-device bus cost, each way? | `tools/gpu/probe-bus.js` (**free on an M2, not on a discrete GPU**) |
| How fast can this browser read a weight shard at all? | `tools/gpu/probe-shard-read.js --bundle=` - **371 MB/s**, and `arrayBuffer()` and the store's streamed read measure the SAME, so the chunk loop the progress dial needs costs nothing. Python reads the same shards from the same server at 2129 MB/s: the cap is six HTTP/1.1 connections |
| ...and how much of a first fold is it? | `weightSeconds` on `fold-af2.js`, `fold-esmfold2.js` and `fold-opendde.js` - **the fold's own clock starts after the weights are loaded and a user's does not.** OpenDDE 1.73 s, AF2 0.88, ESMFold2 0.35 |
| 🔴 ...and are those a USER's seconds? | **No, they are the dev server's, and they are 5-12x optimistic.** Every local timing above reads shards over `tools/serve.py` on loopback. Measured to the real remote from this machine: **one OpenDDE shard at 56 MB/s, all twelve in parallel at 78 MB/s aggregate** - Hugging Face serves HTTP/2, so the six-connection cap does not apply and the constraint is bandwidth, which twelve connections improve by 1.4x and not by twelve. docs/HOSTING.md's own remote measurement is 27.9-30.1 MiB/s. So OpenDDE's 472 MiB is 1.7 s here and about **6 s** over the wire, and a bundle's SIZE matters where its shard COUNT no longer does. One machine's network, one sample - but the direction is not in doubt |
| What is `grid.attend` alone, without the copies? | `tools/gpu/bench-grid-attend-passes.js` |
| Where does the HOST memory go? | `tools/gpu/probe-memory.js` |
| How long does a fold take, by shape? | `tools/gpu/bench-runtime.js` (fits `src/runtime/cost-model.js`) |
| Is an AF3 fold's f16 path still worth it? | `tools/gpu/fold.js --staged= --weights=` (both arms, one shell) |
| Does the progress bar move at the fold's speed? | `tools/gpu/probe-progress-bar.js` |
| Does a failed fold keep its trunk for the retry? | `tools/gpu/probe-trunk-reuse-after-failure.js` |
| What does a fold hold on the DEVICE? | `tools/gpu/fold.js --budget=0` (prints per stage) |
| ...and does AF2 still fold under a CEILING? | `tools/gpu/fold-af2.js --budget=200`, with `--no-resident` as its control - 108.5 MiB against 386.9, the same checksum. 🔴 With residency AND a budget the allocator evicts a buffer an in-flight submit still names; `uploadResident` declines residency on a device with a budget for that reason, and the eviction path is still wrong |
| Does it still fold on a small device? | `tools/gpu/bench-trunk.js --budget=200 --model=` - **its default bundle is the float32 one, which is not on this box**, so without `--model=` it 404s rather than measuring |
| ...and a whole AF3 FOLD under one? | it does not, on this branch or on main: `fold.js --budget=200` dies at `difftx.block needs 15.8 MiB`. 400 MiB of diffusion weights will not stream through a 200 MiB device beside the trunk's scratch. Not a regression, and not a gate |
| 🔴 A device weight pack that ignores `residentWeights` KILLS THE FOLD | `residentPackedOnDevice` raises over a budget, and the pairformer's `run` catches exactly ONE of those before restarting without residency. A second raise out of the restart is uncaught - measured as `w.single-transition needs 3.4 MiB` at `--budget=200`. Gate every one of them on `this.residentWeights` |
| Does the page fit a phone? | `python3 tools/mobile-layout.py` |
| Do the heatmap panel's tabs still work after a vendor bump? | `python3 tools/heatmap-panel.py` |
| **What does a RETURNING visitor pay, rather than a first one?** | `tools/fold-in-page.py --keep-profile`. 🔴 `cdp.launch` wipes the Chrome profile every run, so every page number in this repository is a FIRST visit with no HTTP cache and no SHADER cache. Kept: OpenDDE 7123 -> 5896 ms and AF3 4901 -> 4215, and OpenDDE's status line - the fold alone, after the weights - drops a whole second, so much of it is Chrome serving compiled pipelines. OpenDDE's 1.3-1.8 s of compilation is a first-visit cost, not a standing tax. Never make it the default: the wipe is what stops a stale module looking like a broken feature |
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
| **Does ESMFold2 fold on the GPU, sequence in, structure out?** | `tools/gpu/fold-esmfold2.js` - and 🔴 its default bundle is `model-esmfold2-trunk-f32`, which is **not what the page loads**: the registry points `ef2-fast-600m` at `model-esmfold2-int5`, so a timing taken with the default is not a user's path. `--bundle=/model-esmfold2-int5` |
| ...and what does the SECOND fold cost, which is what a page pays? | `fold-esmfold2.js --repeat=3` - folds from the top reusing nothing, and ASSERTS the alpha carbons match the first fold. 8.26 s then 1.86 and 1.85, because the ESM-C tower keeps its weights now. `--budget=2000` is the arm where it declines to and streams |
| Which of a sampler step's two coordinate sets is the picture? | `tools/gpu/probe-esmfold2-trajectory.js` |
| What does an ESMFold2 fold cost, by band? | `src/esmfold2/cost.js` (fitted at 40, 150, 300) |
| Can anything stand in for the confidence head this checkpoint lacks? | `tools/gpu/probe-esmfold2-confidence.js` |
| Does the EDM sampler's schedule and step agree? | `node tools/check-esmfold2-sampler.js` |
| Does ESMFold2's trunk still compute ESMFold2's trunk? | `tools/gpu/check-esmfold2-trunk-gpu.js` - 🔴 it needs `oracle-dumps/esmfold2-trunk-*.json`, which needs torch, which does not fit on this box |
| ...and does its DEVICE-decoded pair track compute the host packer's answer? | `tools/gpu/check-esmfold2-trunk-pack.js` - both arms in one process over one pair, bar ZERO differing elements (the decode is bit-identical and the layout is the same), and it asserts the device arm actually decoded, because two host arms would agree perfectly |
| Does z_init's every term agree? | `node tools/check-esmfold2-featuriser.js` |
| What dtype is the atom attention actually holding? | `tools/esmc/probe-esmfold2-atom-attention.py` |
| ...and does the CPU reference? | `node tools/check-esmfold2-trunk.js` (77 s a loop) |
| Which convention does one ESMFold2 module want? | `node tools/check-esmfold2-modules.js` |
| What does ESMFold2's trunk cost, by length? | `tools/gpu/bench-esmfold2-trunk.js` |
| How small can ESM-C get before ESMFold2 notices? | `tools/esmc/probe-esmc-compression.py` |
| ...and what does that cost the STRUCTURE? | `.venv-esm/bin/python tools/esmc/probe-esmfold2-structure.py` |
| Does the GPU dequantiser decode what the host decodes, at every codec? | `tools/gpu/check-quantised-upload.js` - it drives the SHIPPED `planBlockUpload`/`runBlockUpload` rather than a copy of the shader, sweeps six (bits, group) pairs, and its bar is ZERO differing elements. It found `readTensorRange`'s int5 path reading a 20-byte group whatever the record said |
| Where do I get any model's WEIGHTS? | **already built and published** - `src/reference/manifests/index.js` has a `remote:` per family, all under `huggingface.co/sokrypton/localfold/resolve/<sha>/<family>/`. Download those; the export pipelines below are for making a NEW bundle, not for getting an existing one |
| Where do I get ESM-C and ESMFold2, to build one? | `tools/esmc/fetch.py` (3.0 GB, ungated, MIT) |
| Turn ESM-C into a bundle the browser reads | `tools/export_esmc_model.py`, then `tools/quantize_af3.py --bits 3 --group 128` - **that group is ESM-C's alone.** AF3's and OpenDDE's bundles are `--group 32`, which is the default, and OpenDDE at 128 folds 6MRR into a 3283 A explosion at pLDDT 46.69 |
| What should a WebGPU ESM-C agree with? | `oracle-dumps/esmc-59.json`, from `tools/esmc/dump-esmc-oracle.py` |
| Does the ESM-C CPU reference compute ESM-C? | `node tools/check-esmc-reference.js` |
| ...and does the WebGPU block? | `tools/gpu/check-esmc-block.js` |
| ...and the whole 36-block tower, and the shim? | `tools/gpu/check-esmc-tower.js` |
| What does an ESM-C block cost? | `tools/gpu/bench-esmc-tower.js` |
| ...and is the tower right at more than one length? | `check-esmc-tower.js --dump=/oracle-dumps/esmc-{59,128,180}.json` |

| **Why does OpenDDE compile its pairformer twice, and can it stop?** | `tools/gpu/probe-token-specialisation.js` - it warms the stack at both token counts, diffs every WGSL text and sorts by bytes. 28 kernels of 44 differ only in a NUMBER (154 KiB) and 2 differ structurally (the triangle contractions, 934 of 1009 lines). 🔴 It is still not worth removing: the largest constants are `(row % 68u) * 68u + row / 68u` index arithmetic, which a uniform turns into integer division in the inner loop - the 4.3x class - and `override` does not help because the backend compiles once per value anyway. **The specialisation IS the compile cost.** See docs/OPENDDE.md |
| **Does OpenDDE fold?** | `tools/gpu/fold-opendde.js --target=6mrr` (RMSD 1.68 A, TM 0.865) - and its bundle wants **`export_af3_model.py --include diffuser`**, because the default is trunk plus distogram head and this tool needs `structural_token_expander`. 🔴 **AND ITS DEFAULT IS 200 STEPS WHERE THE PAGE RUNS 16** - `OPENDDE_COUNTS` prefers 16 and docs/OPENDDE.md shows more steps are WORSE - so a timing taken with the default is 4 s of sampler a user never waits for: `--steps=16` is the page's path (a warm fold 4.09 s -> 1.26) |
| Does its structural-token expansion conserve the atoms? | `tools/gpu/check-opendde-expander.js` |
| Does its CONFIDENCE head still compute its PAE and PDE? | `tools/gpu/check-opendde-confidence.js` - the head's only gate, and a fold's geometry check cannot see a confidence number |
| **Are a model's BOND LENGTHS right, not just its fold?** | `tools/gpu/probe-nucleic.js --sequence= --model=` (RMSD cannot see this; OpenDDE is 15% short) |
| ...and a ligand's? | `tools/gpu/probe-ligand-flow.js --ligand=GOL --mode=diffusion` |
| Does OpenDDE's trunk predict a real fold's contacts? | `tools/gpu/trunk-opendde.js` (**and `--model=/model-af3-int5/manifest.json` is the control**) |
| Does a pairformer block match its reference at THIS bundle's widths? | `tools/gpu/check-af3-block-any.js --model=` |
| ...and which pair-track kernel is the one that does not? | `tools/gpu/probe-opendde-kernels.js --model=` |

| Does the MSA stack still compute the MSA stack, on BOTH arithmetics? | `tools/gpu/check-af3-msa-block.js --model=` - two arms, because three device knobs move that pair track onto the matrix units and one bound would stop checking the vector one: 5.40e-6 with the knobs off and 1.75e-3 with the shipped profile, and 28.5 ms against 16.0 |
| ...and does the template embedder? | `tools/gpu/check-af3-template.js --model=` |

`tools/gpu/check-af3-*.js` are the per-module AF3 oracle checkers.

🔴 **AND THEY USED TO BE PINNED TO A BUNDLE THAT IS NOT ON EVERY BOX.**
`check-af3-msa-block.js` and `check-af3-template.js` opened
`/model-af3-full-f32/manifest.json` as a CONSTANT, so on a machine with the
published int5 bundle and not the float32 one they 404 rather than skip - and
two whole stacks' kernel choices went ungated for exactly that reason. Both take
`--model=` now, and both take a bound that follows the bundle, because an int5
bundle's residue against a float32 reference is not a float32 bundle's.

🔴 **AND EVERY ONE OF THEM EXCEPT THE TWO ABOVE IS PINNED TO AlphaFold 3's
CONSTANTS.** `check-af3-triangle.js` has `const CHANNELS = 128`,
`check-af3-grid-attention.js` has 128 with 4 heads of 32, and
`check-af3-block.js` hand-builds its weight dict with `heads: 4` and
`pairChannels: 128` typed into it. So the differential suite is blind to a
second bundle's widths, which is exactly where a second bundle breaks - see
docs/OPENDDE.md, where a dispatch sized for 128 against kernels compiled for
384 left two thirds of every pair row unprocessed and every per-kernel checker
passing.

## 🔴 THIS BRANCH CHANGES FAR MORE ON AN M2 THAN ON THE A100 - VALIDATE THERE

The capability layer and the five derivations both sit UNDER the priors, so on
the A100 - whose prior sets thirty knobs - almost nothing they decide is ever
used, and every number in docs/A100.md for them was taken with `--no-prior`.

**`metal-3`'s prior sets exactly ONE knob**, `opmMatrixContract`, because
"NOTHING FOR apple ON PURPOSE - its measurements ARE the defaults above". So an
Apple part is, for every other knob, precisely the unrecognised device these
layers were built for, and **ten of them newly take effect there**:

| newly set on an M2 | by |
|---|---|
| `matrixLinear`, `attentionMatrix`, `attentionProjectMatrix`, `stagedMatrixPrefetch` | the capability layer, from `chromium-experimental-subgroup-matrix` + `shader-f16` + an f16 configuration |
| `diffusionSplitK`, `atomRowTile`, `diffusionTokenTile` | the measured occupancy width |
| `diffusionBatchedGates`, `keepTrunkWeights`, `keepSamplerWeights` | the memory budget |

Three things to check there, in this order:

🔴 **MEMORY FIRST, BECAUSE IT IS THE ONE THAT CAN HURT.**
`keepResidentAffordable` returns true for a device with NO budget, and a tool
run sets none - so an M2 will now hold ~561 MiB of trunk weights and ~325 MiB
of sampler weights between folds where it used to release them. That is
affordable against 40 GB and is a different proposition on a laptop, where this
file's own warning is that Metal "takes buffers well past the point where macOS
starts paging". Run `fold.js --budget=0` and read the peak before and after;
if it pages, the fix is a budget, not a revert - `budgetForDevice()` exists and
`requestAlphaFoldDevice` takes it.

🔴 **THEN ACCURACY, because four f16 matrix paths turn on at once.** On the
A100 `attentionProjectMatrix` alone moved mean pLDDT 57.28 -> 57.29 and the
first alpha carbon 0.13 A. Run the differential gates rather than a fold:
`check-attention-variants.js`, `check-evoformer-{transition,opm,attention}.js`,
`check-af3-block-any.js --resident`, `check-triangle.js` - whose f16 arms an M2
can actually run - and `check-difftx-splits.js`.

🔴 **THEN THE OCCUPANCY PROBE ITSELF, whose failure mode is a plausible small
number.** It read **4 workgroups** on a card the tool measures at 4542 before it
grew a warm-up and two rounds. Run `probe-occupancy.js` and check the plateau
edge and the slope estimate agree - `agrees: true` in its output. If they do
not, the three width-driven derivations are being fed a wrong number and
`--default-tuning` is the arm that switches all of this off.

A whole fold is the wrong instrument for any of the three: it mixes them, and
the M2 should be FASTER overall, which hides an accuracy change and a memory
one equally well.

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

🔴 **AND A BUNDLE THE CLI LIKES CAN BE ONE THE PAGE CANNOT LOAD.** The command
tools read the `manifest.json` sitting next to the shards; the PAGE reads the
manifest baked into `src/reference/manifests/<family>.js`, which is pinned to a
commit. A bundle downloaded from a different export satisfies the first and not
the second: `model-opendde-int5/` here was a 32-shard export against a registry
that pins 12, so every `fold-opendde.js` gate passed and
`fold-in-page.py --model opendde` died at "weights-08.int5.bin has an invalid
byte length" at 122/472 MiB, having never reached a fold. Check a bundle against
the MODULE, not against the JSON beside it.

🔴 **AND EVERY ONE OF THOSE KILLS LEAVES ITS PROFILE DIRECTORY BEHIND.** The
harness cleans up `/tmp/gpu-chrome-<pid>-<stamp>` when it exits normally and
cannot when it is killed, so a session that runs hundreds of arms leaves
hundreds of directories at 15-500 MB each. Measured on this box: **2217 of them,
34 GB**, on a machine whose disk is the binding constraint and had 2.2 GB free
by the time anything noticed. Sweep them when a batch is over:

```
find /tmp -maxdepth 1 -name 'gpu-chrome-*' -type d -mmin +5 -print0 | xargs -0 -r rm -rf
```

`-mmin +5` is what keeps it from deleting a profile a running browser is still
using.

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
