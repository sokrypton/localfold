# Working on LocalFold

`AGENTS.md` has the engineering invariants. This file is the operational half:
how to actually run things here, and the traps that have cost time more than
once. The findings themselves - some five hundred of them, every one
measured - live in
`docs/`, indexed at the bottom of this file. **Read the doc for a stage before
changing that stage**; this file's own recurring lesson is that a whole-fold
gate cannot see an error inside one of them.

## Running anything that needs a GPU

🔴 **AND `LOCALFOLD_STOCK_FLAGS=1` IS A GATE, NOT A CURIOSITY: TWO OF THE FOUR
MODELS DID NOT FOLD WITHOUT THE FLAGS.** OpenDDE and ESMFold2 both died on
`extension 'f16' is not allowed in the current environment` - a precision chosen
from the CHECKPOINT and from the BUNDLE rather than from the device - so they
were broken for every visitor on an NVIDIA GPU, and no gate could see it because
every harness here passes the flag that hides it. Fixed; all four fold both
ways. `ComputePipelineCache` now throws with the shader KEY when a source
enables f16 on a device without it, because Dawn's own error names nothing. See
docs/A100.md.

🔴 **AND A KNOB'S WORTH IS A PROPERTY OF THE CONFIGURATION.** Swept under
`LOCALFOLD_STOCK_FLAGS=1`, `linearTallTile` is **1.31x** (379.58 ms against
497.03 on a block) where docs/A100.md's `--no-prior` split prices it at ZERO,
and `attentionGroup` moves 5.4% where it moved 0.13% - both because the matrix
kernels replace the vector ones the knobs belong to. Every prior number in these
docs was taken with the flags on. See docs/AF2.md.

🔴 **AND EVERY NUMBER THIS HARNESS PRODUCES IS BEHIND TWO DEVELOPER FLAGS WORTH
1.95x.** `gpu-chrome.mjs` and `tools/cdp.py` both pass
`--enable-dawn-features=vulkan_enable_f16_on_nvidia` and `--enable-unsafe-webgpu`;
on this A100 with Chrome 152 the first is the only reason `shader-f16` exists
and the second the only reason `chromium-experimental-subgroup-matrix` does. A
stock Chrome has **neither**, and the same 825-residue fold is **10767 ms with
them and 21000 without**. `LOCALFOLD_STOCK_FLAGS=1` drops both and is how to ask
what an NVIDIA visitor actually gets. See docs/A100.md.


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

🔴 **AND ITS ADAPTER HAS NO `shader-f16` UNTIL YOU ASK - WHICH IS A MISSING
TOGGLE AND NOT A PROPERTY OF DAWN.** Every one of the 25 `create(...)` calls in
`test/*.gpu.test.js` passes `[]`, and measured here on webgpu@0.4.0:

```
create([])                                                  f16 no   matrix no   18 features
create(["enable-dawn-features=vulkan_enable_f16_on_nvidia"]) f16 YES  matrix no   19
create(["enable-dawn-features=vulkan_enable_f16_on_nvidia,allow_unsafe_apis"])
                                                            f16 YES  matrix YES  21
```

Note no leading `--`, and that a second toggle goes after a COMMA rather than in
a second array element, which throws "Flags expected argument format is
`<key>=<value>`". So the Dawn lane can run the f16 AND the subgroup-matrix arms
that the paragraph below says it cannot; those 25 call sites want one shared
helper. **AND THIS IS WHY A NODE LIBRARY IS THE ONE PLACE THIS PORT CAN
GUARANTEE ITS OWN FAST PATH**: in a browser the two capabilities need flags the
visitor must pass, and in Node the process sets them itself.

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
`model1-a3m-59-stack` are absent whole. So checking an
attention change against official values still means the whole-stack checker,
not `test/evoformer-attention.gpu.test.js`. Two more failures are the missing
`shader-f16` above.

🔴 **AND "TEN `*_haiku_*.f32.bin` WEIGHTS ARE MISSING FROM THE
`model1-query-59-stack` THAT IS PRESENT" WAS THIS FILE'S OWN CLAIM AND IT IS OFF
BY A FACTOR OF FIFTY.** Counted against its own manifest: it declares **530
tensors and 26 are on disk** - the twenty input features and the six geometry
tables, nothing else. **504 are missing, 328 of them `*haiku*`**, so the
directory holds no weight of any kind and is an input set rather than a partial
fixture. That is why all five of the tools that name it fail rather than
degrade: `npm run bench:a3m-model`, `bench:a3m-stack`, `bench:attention` and
`export:web-model` all die on `ENOENT`, the last one on
`stack_haiku_0024.f32.bin` out of its own default argument. Only
`npm run bench` (`benchmark-triangle.js`) builds its own input and needs none of
this.

🔴 **THE TWENTY-FIFTH WAS A REAL KERNEL BUG, NINE MONTHS OLD AND NEVER RUN.**
See the triangle rows below: `TriangleMultiplicationOutgoing` missed its
OpenFold reference by 8.26e-2 against a 1e-5 bound while three AlphaFold
fixtures passed, because the fixture is **cZ 7 and every shipped width is even**.

## The tools, by what they answer

🔴 **AND THIS TABLE IS THE CURATED VIEW, NOT THE LIST.** It answers "which tool
answers this question" and says which arms are traps, which is what you want
when you know what you are asking. It is not exhaustive and was never meant to
be: an audit found **44 tools named by nothing at all** - not here, not in
`docs/`, not by another tool - including working differential checkers such as
`check-af3-opm.js`, `check-af3-msa-attention.js`, `check-af3-atom-decoder.js`
and `check-chiral-gradient.js`, which is docs/PARITY.md's complaint from the
other end: a checker nobody can find is a checker nobody runs.
`docs/TOOLS.md` is the exhaustive one - **every tool, with the first line of its
own header, derived rather than written** - and `npm run test:tools` fails when
it is stale or when a tool has no header at all. Come here first; go there when
you do not know whether a tool already exists.


| Question | Tool |
|---|---|
| **Does a whole DENOISE STEP match af3-any-model's own?** | `tools/gpu/check-af3-denoise.js --model= --name=` - L3, the level that subsumes the diffusion side. 🔴 It holds the GPU to the ORACLE and the CPU to the GPU, because boltz2's token transformer amplifies its input by ~2.2e4 and this f64 reference cannot agree with an f32 oracle to better than 1e-2 however right the port is. `--stages=on` compares every seam against `dump_af3_denoise_stages.py`; without a stage dump the per-stage arms are GPU-against-CPU and say only that the port agrees with ITSELF. 🔴 **AND ITS DUMP HAS NEVER SEEN A STRUCTURE.** `dump_af3_denoise.py` builds `pos_dense = rng.normal(...) * NOISE` and `s`, `z`, `s_inputs` as `rng.normal(...) * 0.5`, so the flagship AF3 denoise oracle is a WIRING test: this port passed it at 1.30e-5 at sigma 16, 9.85e-6 at 0.2 and 1.14e-5 at 0.02 while folding side chains at 6.7x the reference's bond error, because both sides read the same wrong tensor and neither needs real geometry to agree about it. `tools/oracle/dump_af3_real_denoise.py` (A10) captures a REAL fold's trunk conditioning and re-noises a real structure; `--bonds` scores the clean input, the noisy input, native's D and ours as CHEMISTRY. 🔴 **AND RUN IT ON THE FLOAT32 BUNDLE**: after the weight-name fix AlphaFold 3 reads **9.92e-4 on `model-af3-full-f32` and 3.71e-1 on the int5 bundle that ships**, from the same corrected dump - the first is the port and the second is the quantisation, and reading the int5 arm as a defect cost an hour. int5 costs 3.4-4.6% on each of the four tensors and 0.051 -> 0.059 A of side-chain bond rms, which is the whole of it. 🔴 AND A DUMP IS ONLY AS GOOD AS THE REFERENCE THAT WROTE IT: the AF3 dump here was regenerated 2026-09-16 because af3-any-model's `041ab187` had renamed a weight, and agreeing with the old one at 1.55e-5 was agreeing with the defect |
| **Do IntelliFold-2 and RoseTTAFold3 work?** | **BOTH FOLD, at af3-any-model's own numbers.** 6MRR **from the sequence**: if2 **1.584 A / TM 0.9242** against the reference's 1.517, rf3 **1.68 A / TM 0.911** inside the reference's own five-sample spread (0.967-1.772, mean 1.546). A 5CAJ self-template takes if2 17.794 -> **0.256 A**, the second best of the seven, and its trunk on its own reference batch is at or below boltz2 on all five seams; rf3's trunk is 4.90e-2 at `trunk_out_pair` against boltz2's 4.56e-2. 🔴 rf3 TOOK SIX BRANCHES AND TWO OF THEM MOVED THE FOLD BY ALMOST NOTHING WHILE BEING CORRECT - see the arm table in docs/AF3.md, because judging either on the fold alone would have backed it out. Still unimplemented for rf3: the chirality query term (a no-op without chiral centres) and two confidence-head branches (which cannot move geometry) |
| 🔴 **Getting a model's convention set from the reference** | IMPORT `model_config.py` and introspect every uppercase string tuple - `PYTHONPATH=src ~/af3_venv/bin/python` from a checkout of **`sokrypton/alphafold3`**, 🔴 whose `main` IS af3-any-model now (the multi-model work was merged; there is no separate branch to chase, and `~/alphafold3` in this file's older rows does not exist on this box - clone it fresh), because the system python3 is 3.10 and `base_config.py` wants `typing.dataclass_transform`. A REGEX OVER THE SOURCE UNDER-REPORTS: `TRIANGLE_MUL_DIVIDE_BY_LENGTH` is a conditional expression over an env var and several tuples are built from other tuples. **And the tuples are only half the port**: eleven of rf3's divergences are gated on `global_config.model == 'rosettafold3'` with no tuple at all, which only `grep -rn` finds |
| 🔴 **A convention that changes NOTHING AT ALL** | **suspect the bytes, not the convention.** `packOuterProductMeanWeights` reserved offsets over `ORDER + OPTIONAL` and WROTE over `ORDER` alone, so rf3's outer-product bias read a region of zeros: present in the source, present in the offsets, absent from the buffer, and every one of eight oracle seams matched the previous run to the last digit. One list for both loops. Its twin: `=== undefined` is the wrong presence test for an optional weight, because the loader writes `null` and that null reaches the SOURCES map - which killed boltz2 in a path added for a model boltz2 does not share |
| **Does a model take a TEMPLATE, and does it help?** | `tools/gpu/fold-opendde.js --target=5caj --chain=A --template=<pdb>:<chain>` - it folds ANY AF3-lineage bundle and reports RMSD, and the slot is built by `buildTemplate`, the page's own function. 🔴 **6MRR CANNOT ANSWER THIS**: a 68-residue designed protein folds to 0.5 A from its sequence alone, so a correct template embedder and a corrupt one land in the same place - boltz2's read 0.490 there while it was broken. 5CAJ (255 residues, natural, `tools/fixtures/5caj-crystal.pdb`) goes 17-30 A without and 0.16-0.47 A with, for all five |
| **Do the fused embedder's 108/109 columns come out right?** | `tools/gpu/check-fused-template-features.js --name=protenix2` against af3-any-model's own, and `--name=boltz2` for the forward with a `--pins` arm; `--forward` runs protenix2 through boltz2's path as the control. `check-protenix2-empty-template.js` drives the CPU module with the TRUNK's inputs - four empty slots at 68 tokens - which is what separated a wrong featuriser from a wrong GPU path |
| **Where inside a trunk stage is the disagreement?** | `fold-opendde.js --msa-blocks=N --stop-after-opm` bisects the MSA stack against the oracle's `MSA_BLOCKS=`; `fold.js --templates=N` sets the padded template slot count, which under `templateMeanOverAllSlots` is the DIVISOR and not just work; `CAPTURE=LIST` on `dump_af3_trunk_taps.py` names every module the reference traces, and `CAPTURE=<regex>` records them. 🔴 The MSA stack and the pairformer are `hk.experimental.layer_stack`, so `hk.intercept_methods` sees NOTHING inside them - 29 modules in the whole trunk and not one from either - and a tap in there must go through `io_callback` or it raises `TracerArrayConversionError` |
| **Is a term of AF3's z_init wrong, or all of it?** | `tools/gpu/check-af3-embedder-terms.js` - the four summands separately, and it found the relative encoding while the three single projections read 4.6e-8 |
| **Does OPENDDE's trunk and confidence match it?** | `tools/gpu/fold-opendde.js --trunk-oracle= --dump=` and `tools/gpu/check-opendde-confidence-oracle.js` - both new, and OpenDDE was the last family with neither. Its confidence needed its own dumper (`dump_af3_opendde_confidence.py`): `dump_af3_confidence.py` indexes a tensor OpenDDE does not have and dies with a KeyError. The confidence head WAS wrong (PAE 4.68e-3) and is fixed in 8dbb8cd - **8.46e-7 on the A100, 9.11e-7 on the M2**; its atom encoder is exact (0 to 1.79e-7, masked). 🔴 **THAT ROW SAID "STILL WRONG FROM THE TEMPLATE STAGE ON (2.07e-2, inherited to 1.08e-1)... the one open defect" AND IT WAS STALE.** Re-measured on the f32 bundle against its own reference batch: `target_feat` **3.14e-8**, `z_init_generic` **4.70e-8**, `trunk_in_single` **2.78e-7**, `z_after_template` **8.60e-8**, `z_after_msa` **2.65e-5**, `trunk_out_pair` **7.71e-4**. The whole trunk is in band - 7.71e-4 is exactly what the sweep in docs/AF3.md records for OpenDDE's pair width. Two commits closed it and neither came back to this row: **c014468** ("a padded template slot carries restype ZERO, and a one-hot of zero is a one") is the template stage, which is why it showed on a query with NO template, and **f471ee4** (the outer product mean computing 256 channels where OpenDDE's pair is 384) is the rest. **20cf8fd re-measured the sweep and said "nothing left above 1.4e-3"** - so this file has contradicted itself since then, and the sweep was the half that was right. 🔴 A DOC THAT DISAGREES WITH ITSELF COSTS MORE THAN A MISSING ONE: this row was the top item on a list of things left to fix |
| 🔴 **Does `--f16=off` reach the kernel you are asking about?** | **Not the MATRIX ones, ever.** OpenDDE's confidence head read 4.68e-3 here and "identical with `--f16=off`", which was reported as "not arithmetic precision". It was precision: the flag cannot reach the matrix pair kernels, so on a device with matrix units it moves nothing and an unchanged residual proves nothing. The M2, which has no matrix units at these widths, saw the same flag remove all of it. **A control arm that cannot vary the thing under test is not a control** - check the arm reached the code path before reading its result |
| 🔴 **Two residuals that cannot both be true?** | **Believe the checker is wrong, not the port.** Twice in one file: `check-opendde-encoder-oracle.js` reported the per-atom conditioning at 1.19 while `target_feat` was exact in 439 of 447 columns (it passed `atomReference(store)`, the DIFFUSION head's table, where `buildTargetFeat` reads `targetFeatureWeights().reference`), then 0.466 and 0.137 on two seams while the one DOWNSTREAM of both read 1.79e-7 (an unmasked comparison over padded slots the reference fills and this port zeroes, which `mask_mean` discards on both sides). Each contradiction was the fastest route to the fault |
| 🔴 **Does `fold.js --target=6mrr` FEATURISE ANYTHING?** | **Not without `--sequence=`.** The batch is `sequenceArg !== "" ? af3BatchFromA3m(...) : batchFromDump(dump)`, so `--target=` alone folds `oracle-dumps/af3-6mrr.json` - AlphaFold 3's own featurised batch - through whatever model `--model=` names. It says so on the line after the census ("featurised by AF3, read from the dump") and the census itself gives it away: **51 atom subsets is the reference's DENSE grid where this port compacts to 18**. Four folds were read as a regression test of a FEATURISER change and none of them had touched it. Pass `--sequence=` when the featuriser is what changed, and read the subset count |
| 🔴 **Does this port's ATOM KEY WINDOW match the reference's?** | `node tools/check-atom-windows.js` - CPU-only, seven models, and it compares INTEGERS against `queries_to_keys` in each `oracle-dumps/af3-batch-<model>-6mrr.json`. **No other gate here asks**: `check-af3-denoise.js` takes the reference's windows out of the dump precisely so it can compare score models rather than featurisers, so the window itself was never compared to anything. Three whole conventions lived in that gap - SLIDE, CLAMP-and-mask, and IntelliFold-2's slide against a query-block-PADDED edge - and **four of seven models had the wrong one**. All seven are exact now, and the tool prints the other arm beside each so a flag that changes nothing is visible. 🔴 A FOLD'S RMSD CANNOT SETTLE THIS: the change moved opendde's 6MRR by 0.026 A, inside a seed band this repository has measured at 1 A |
| 🔴 **Is an oracle residual the MODEL's, or this port's CONFORMERS?** | `--dump=<a batch dump>` on `fold.js` or `fold-opendde.js`. LocalFold ships ONE idealised reference conformer set shared by every family and the reference featurises CCD geometry, so a SEQUENCE-featurised oracle run has a floor: openbind0's `target_feat` is **2.89e-2 from a sequence and 4.93e-8 from the reference's batch**, same weights, same code. Every oracle number in docs/AF3.md is a `--dump=` number. `batchFromDump` carries `residueOfToken` now, without which OpenDDE's re-tokenisation died in `kindOfToken` |
| **WHICH parts of WHICH model are not exact?** | the sweep in docs/AF3.md: all five trunks on their own reference batch, re-measured 2026-09-14. `target_feat` 1e-8 across the board, `z_init` 1e-8, the template seam 1e-7 or below on four, `z_after_msa` 2.6e-5 to 1.2e-4, and the finished pair **2.98e-4 / 4.20e-4 / 3.49e-4 / 7.71e-4 / 1.38e-3** ordered by pair width. Nothing above 1.4e-3, and at matched f32 the five line up with no outlier. AlphaFold 3's `z_init` 2.20e-4 is its DUMP's bfloat16 and not this port. 🔴 **The first version of this row listed three open defects and all three closed the same day** - re-read the numbers before quoting them |
| **Does the TRUNK match it, stage by stage, on a real batch?** | `tools/gpu/fold.js --trunk-oracle=` with `tools/oracle/dump_af3_trunk_taps.py`. It found boltz2's `target_feat` at relRMS 1.00e+0 while `check-af3-trunk` read 2.74e-5 - the difference between an oracle and a self-comparison. `BLOCKS=` truncates BOTH sides, which is how a convention is separated from an accumulation |
| **Does the CONFIDENCE head?** | `tools/gpu/check-af3-confidence-oracle.js --model= --name=` with `tools/oracle/dump_af3_confidence.py`. Every module inside the head is traced, so a pair that is a sum of nine terms is attributable. It found protenix2 at 3.21e-1 and boltz2 at 8.26e-2 while `check-af3-confidence` passed at 9.56e-5 |
| 🔴 **Is the page's DEFAULT sampler the one that works?** | **Yes - and the claim that it was not is RETRACTED.** This row used to say AlphaFold 3 with Flow on 1QYS "returns a fold that is not a chain on four seeds out of four: CA median 4.267 / 4.367 / 4.861 / 4.886 A, worst up to 26.02, pLDDT 68.79". **It does not reproduce, and not at the commit that recorded it either.** Re-run at HEAD and in a worktree at `9cc43d7` itself, four seeds, through BOTH tools: CA-CA **3.801 / 3.787 / 3.794 / 3.790**, worst 3.50-4.06, pLDDT **79.09 / 79.65 / 79.74 / 79.49**, RMSD 0.936-0.999. `fold.js` and `fold-opendde.js` agree to the digit. The diffusion CONTROL moved too - recorded 1.072, measured 0.918 - which is the tell: a difference in both arms is a difference in the CONFIGURATION, not in the sampler. 🔴 **THE SCRATCHPAD LINE ABOVE THOSE NUMBERS READS `(clean = diagnostics reverted)`**, so it was taken on a tree that was not clean, and a revert that did not revert produced four consistent-looking broken folds. **A measurement taken on a modified tree is not a measurement, and "I reverted it" is not the same as `git status`.** 🔴 **AND IT COST TWO DECISIONS**: the page's default sampler changed on it, and the whole ODE option was built on "the only step that folds 1QYS for AF3" - a target that never needed rescuing. **Diffusion is still the right default**, for the reason that survives: it wins or ties nearly everywhere measured (1QYS 0.918 against flow's 0.936-0.999, 6MRR 0.650 against 0.687, 1TIM 0.958 against 0.975), and it is the sampler af3-any-model verified. Flow is a close second everywhere and costs the same. 🔴 **AND ODE IS REMOVED FROM THE PAGE ON THE SAME EVIDENCE**: measured against diffusion rather than against a Flow that was never broken it wins **1 of 12** model/target pairs (intellifold2 on 6MRR, 0.769 against 1.549) and loses the other eleven - including that model's own 1TIM row, 1.665 against 1.058, so the win is 68 residues and not the checkpoint - and past ~400 residues it breaks. `--mode=ode` stays on the fold tools; the `<option>` and the `AF3_COUNTS` row are gone, and `test/sampler-options.test.js` gates the two lists against each other in both directions |
| 🔴 **Does this checkpoint HAVE the sampler the page offers?** | **rosettafold3 does not, and the page defaulted to it.** `index.html` has `<option value="flow" selected>`, and rf3 in flow mode is **not a chain**: N-CA **6.94 A** against 1.46, CA-C 3.76 against 1.52, consecutive CA collapsing to **0.23 A** at worst - against `--mode=diffusion`'s 1.693 A and a clean backbone. 🔴 **AND pLDDT READS 81.47, four tenths from the good fold's 81.53**, so the number the page shows says nothing is wrong. Nothing could catch it: every standing gate goes through `fold.js`, whose default is `diffusion`, and af3 (CA-CA 3.58) and if2 (3.88) both fold in flow. `noFlowSampler` in the dialect declares it, `foldBatch` THROWS rather than switching silently, and `samplerModeFor` in web/af3-model.js keeps the page from asking - hiding the row is not enough, because hiding a control does not change its value. 🔴 The first guard read `weights.dialect` where `foldBatch`'s is `weights.trunk.dialect`, so it was a no-op and the broken fold reached the geometry gate unchanged: **a guard that reads the wrong field is a guard that is not there** |
| 🔴 **Does ESMFold2 fold a modified residue, and is the fault the MODEL or the PORT?** | **It folds one now, it places it badly, and the fault is the PORT - measured against the reference.** `foldWithEsmfold2` never took `modifications` and its entities literal never named it, so a SEP@3 folded a plain SERINE (`SER [N,CA,C,O,CB,OG]`, certainty 0.59, nothing on the status line) - CLAUDE.md's allow-list trap, fourth time at this seam, fixed in f332b26. 🔴 **THEN `~/venv_ef2/bin/python tools/esmc/probe-esmfold2-modified.py` SETTLED WHAT WAS LEFT**: the same 58-mer and SEP@3 through `esm` 3.4.1's own `ESMFold2InputBuilder` and `EsmFold2ExperimentalModel`, on the weights this port reads - **native SEP 0.997 against this port's 2.349**, same 171-bond control (1.000 against 0.999). 🔴 **AND THE FEATURISATION IS RULED OUT FIELD BY FIELD**: conformer exact (OG-P 1.609), one `refSpaceUid`, bonds in `bondedGroups` AND consumed as `featuriser/tokenBonds`, and the reference itself reports **67 tokens with the modification as TEN tokens of one atom, all `mol_type` 0 (PROTEIN)** - identical to ours, its tokeniser's docstring saying "Modified residues are atom-tokenized (1 token per atom)". The fault is downstream of all of it. 🔴 **AND `npm run test:modified` COVERS SEVEN AF3-LINEAGE BUNDLES AND NOT THIS ONE** (`probe-modified.js` opens an AF3 store), which is why a user found it; `test/af3-ligand-bonds.test.js` holds the structural half. 🔴 **AND "TORCH DOES NOT FIT ON THIS BOX" IS STALE** - 280 GB free, `~/venv_ef2` has torch 2.7.1+cu126 with CUDA, both checkpoints on disk. 🔴 **AND THE FIRST NATIVE NUMBER WAS 1.101 AND WRONG**: atom names sliced from the CCD keep `OXT`, a leaving atom a mid-chain residue drops, shifting every name after it. **A near-miss number is the dangerous kind** - it would have closed the question the wrong way. See docs/WEB.md |
| 🔴 **Does every model fold a MODIFIED RESIDUE, and does it hold together?** | `npm run test:modified` (`tools/check-modified-path.mjs`) - SEP at position 3 through every bundle, at a CONVERGED setting. 🔴 **EIGHT MODELS NOW, AND THE EIGHTH IS WHY A USER FOUND A BUG THIS GATE EXISTS TO CATCH**: it drove `probe-modified.js` alone, which opens an AF3 store and calls `foldBatch`, so ESMFold2 was unreachable - while the page's guard OFFERED modified residues for it and `foldWithEsmfold2` never took them. An entry may carry its own `tool` and `args` now, as check-template-path.mjs's may; the esmfold2 row drives `fold-esmfold2.js --modify=SEP@3` at **diffusion-200**, because the vendor breaks below 15 steps (at 11 its own control is 2.155) and a low count would fail the gate for the sampler rather than the residue. Watched failing at **2.264** with the restype fix reverted. All eight in band: af3 0.980, protenix2 0.996, if2 0.978, openbind0 0.862, opendde 1.001, rf3 0.984, boltz2 0.938, **esmfold2 0.985**. 🔴 **NOTHING WAS RUNNING THIS ARM AND ONE MODEL IS BROKEN IN IT**: `probe-modified.js` has printed `modifiedBondRatio` since it was written and nothing asserted on it, `test:ligand` folds a GLYCEROL (a separate chain, not an atomised residue inside a polymer) and `test:batch` compares a SEP target's FIELDS without folding it. **boltz2 inflates a phosphoserine 1.8x** - N-CA 2.007 against 1.469, CA-C 3.485 against 1.506 - where every other model is within 6% and its own controls are 0.995. 🔴 **FIXED 2026-09-16: boltz2 KEEPS A MODIFIED RESIDUE IN ONE TOKEN** (`modifiedAsOneToken`, that family alone) where this port atomised it for everyone - **1.813 -> 0.938** against genuine Boltz-2's 0.986-1.011, and `test:batch` 14/14 exact. Three things had to be right and the batch gate caught each: the flag must be named in `batch.js`'s forwarding list (83 tokens against 74 until it was), the pseudo-beta is **CB** and not slot zero, and an atom keeps its **component slot** so a removed leaving atom leaves a HOLE - a mid-chain SEP is `N,CA,CB,OG,C,O,_,P,O1P,O2P,O3P` with a gap at 6 where its OXT was, and compacting shifts the phosphate down one. The reference's SECOND fault does not apply here (it also dropped O3P, a sidechain atom, under a guard a later convention falsified), which is why this port read 1.813 where it read 2.705. Originally found because 🔴 **GENUINE Boltz-2 2.2.1 PLACES IT CORRECTLY** - `pip install boltz`, single sequence, three samples, **0.986 / 0.996 / 1.011** at bond rms 0.043-0.076 A. So it is a defect in **af3-any-model's boltz2 PORT** which this port inherits, not something Boltz-2 cannot do; boltz2 is an EXPECTED FAILURE here that asserts it stays broken. af3-any-model's boltz2 is **2.705** on the identical target through the identical harness where its alphafold3 is **0.988** (`tools/oracle/probe_af3_ptm_bonds.py`, run on the A10). Excluded on this side: the bond matrix and bond-order plane (9 pairs, 9 entries, byte-identical across models), `ref_pos` (10 slots, one space, 6.924 A), and the `_1` weight change (4 of 4 byte-identical for boltz2). A FIXED row means it came good - delete the entry, do not widen it. 🔴 **AND IT MUST BE FOLDED CONVERGED OR IT MEASURES NOTHING**: at the probe's default eight steps in DIFFUSION every model reads 11-21 and so does its CONTROL, which is why the gate asserts the control too |
| 🔴 **A probe whose ideal looks wrong is pairing the wrong atoms** | the first version of `probe_af3_ptm_bonds.py` searched EVERY token for an atom named `CB` and paired it with any `OG` whose `ref_pos` distance fell in a 0.9-2.0 A window - and `ref_pos` is each residue's OWN local frame, so a cross-residue distance is not a distance and lands in that window by accident. It reported `CB-OG` at **16.321 A** on a reference fold whose N-CA, CA-CB, CA-C and C-O were all within 3%, and would have been written up as "the reference does it too, mean ratio 2.894". **The tell was the IDEAL reading 1.571 where the dictionary says 1.428**: when the ideal is wrong the pair is wrong. Select an atomised residue's atoms by `residue_index`, which is what identifies them |
| **Does every model fold a LIGAND, and do its bonds hold?** | `npm run test:ligand` (`tools/check-ligand-path.mjs`) - a 68-residue protein plus GLYCEROL through all six AF3-lineage models, asserting the five bond lengths. 🔴 **NOTHING WAS RUNNING THIS ARM AND IT BROKE THREE TIMES.** Every other fold gate here folds a plain protein, so the whole atomised-token half of the featuriser went unexercised: **boltz2 tore a glycerol apart at bond rms 3.602 A (C1-O1 at 6.97 against a 1.43 ideal) while its pLDDT read 92.38**, rosettafold3 died outright on a ligand with no stereocentre ("invalid allocation size 0 for atom.chiral.centers"), and `atomizedElementNames` renamed the atoms in the OUTPUT PDB as well as the model's input, giving one residue six atoms called C, O, C, O, C, O. 🔴 **boltz2's WAS TWO HALVES AND NEITHER SHOWED ALONE**: the featuriser never built `bondOrderMatrix` (five consumers, no producer - boltz2's z-init reads a second BOND-ORDER plane and is the only family with `tokenBondsTypeEmbed`), and `fold.js`'s embedder-input literal did not name it either, so fixing the featuriser alone left the fold BYTE-IDENTICAL. The comment two lines above that literal already warns about exactly this - "a key the batch carries and this literal does not name is a key thrown away here" - about `bondMatrix`, which was fixed; the orders were added to the consumers and never to the list. Now 0.062 A, and all six pass: **af3 0.061**, protenix2 0.044, if2 0.055, openbind0 0.058, boltz2 0.062, rf3 0.069. 🔴 THE af3 FIGURE READ 0.050 HERE AND WAS STALE - re-measured 2026-09-17 at 0.061, and checked against `main`'s own `fold.js` to be sure it was not the SMILES work landing beside it. The other five are exact. This file's own complaint about a figure in prose, one more time. 🔴 **IT ASSERTS BOND LENGTHS BECAUSE NOTHING ELSE CAN SEE THEM** - the fold's RMSD is dominated by 68 residues of protein, pLDDT said 92 on a ligand 6 A out, and `chain-geometry.js` measures the protein BACKBONE and steps over a ligand by design - and it measures by atom ORDER, not name, because keying on "C1" reports rf3's element-renamed atoms as a missing ligand. A bundle this box lacks is a SKIP, not a failure: openbind0 is f32 here |
| 🔴 **Is it wired to the CONTROL a reader touches, not just the API?** | `tools/fold-in-page.py --smiles-ui` drives the dropdown, the box and the BLUR instead of calling `entityList.set()` - and three bugs survived every `--smiles` run because the API path cannot see a UI one. 🔴 **THE BLUR HANDLER TURNED BENZENE INTO HEXANE**: a row that is not `ligand` fell to `cleanSequence`, which keeps only amino-acid letters, so `c1ccccc1` became `CCCCCC` on a click and biotin lost every ring-closure digit. 🔴 **`setChains` DELETED THE ROW**, keeping only `type === "ligand"` - and it runs when an alignment's query replaces the chain list, so folding with an A3M dropped the ligand silently. 🔴 **AND `.entity-value-ligand` IS `text-transform: uppercase`**, so reusing the ligand's class would have DISPLAYED `c1ccccc1` as `C1CCCCC1`. Switching the row's type now clears the box across a CATEGORY (a CCD `C` switched to SMILES is methane, not cytidine monophosphate) and keeps it WITHIN the polymers, where `ACGT` being a valid protein and a valid DNA chain is this page's own documented rule. 🔴 The probe reported a bug that was its own TWICE - holding an element across a re-render, and typing into the HIGHLIGHT LAYER, since a polymer row has two elements classed `entity-value` and a SMILES row has one |
| 🔴 **Does a job with TWO different SMILES ligands work?** | **It did not, and it was two bugs with one cause: every SMILES ligand was named `LIG`.** The visible half is an output PDB with a benzene and a glycerol under one residue name, which `check-ligand-path.mjs`-style filtering silently mixes. 🔴 **THE INVISIBLE HALF IS THE SERIOUS ONE**: `featuriseProtein` keys a ligand's ENTITY on its code - "identical codes are one entity, and each occurrence is a copy of it" - so the two came out sharing an `entity_id`, telling the model that six carbons and a glycerol are two copies of one thing, and telling chain-permutation scoring they are interchangeable. The RULE was right and its KEY was wrong: a CCD code identifies its contents so the two agree for every dictionary fold, but a SMILES ligand has no code and is GIVEN one. The featuriser keys on what the component IS now (element, charge, bonds), which cannot change a dictionary fold - `test:batch` and `test:ligand` confirm. 🔴 **AND BOTH HALVES ARE FIXED ON PURPOSE**: either alone leaves the other reachable from a different caller, which is not hypothetical, because `tools/gpu/fold.js` took ONE `--smiles-code` for every ligand and wrote ten atoms into one residue with `C1` twice. `ligandName` lives in src/chem/component.js and both callers use it; three characters, because fold.js writes the name with `.padEnd(3)` and `LIG2` truncates back to `LIG` |
| 🔴 **Does a SAVED JOB describe the fold that happened?** | **`node --test test/job-json.test.js`**, and the answer was NO for a SMILES ligand. `jobRequestJson`'s last branch was a catch-all `else` that wrote every non-polymer row as `{ligand: {ligand: value.toUpperCase()}}`, so a benzene folded as `c1ccccc1` went into the archive as **`C1CCCCC1` - CYCLOHEXANE** - labelled as a dictionary code. 🔴 **AND THE LOUD HALF IS THE SAFE HALF**: a long SMILES throws on read-back, but `C` is a valid SMILES for methane AND a valid CCD code for cytidine monophosphate, so that job round-tripped SILENTLY into a nucleotide. The server dialect has no SMILES field at all (`ligand`, `ion`, `count`), so such a job is written in the OPEN dialect - where copies are an `id` LIST whose length is the count, seeds are integers not strings, and 🔴 the `dialect` key must be written EXPLICITLY even though this port's own header says the open dialect has none: upstream's rule is both `dialect` and `version` or NEITHER, and neither means the SERVER dialect, so `version: 1` alone is a malformed server file. Verified through `fold-in-page.py --job-round-trip`, which WIPES the rows first |
| 🔴 **Does the MODEL see the same thing from a SMILES as from a CCD code?** | **`npm run test:smiles-batch`** - and this is the sharpest gate of the SMILES path, because a fold gate compares STRUCTURES and a structure is noisy, sampled and model-dependent. The BATCH is what the model reads, and 43 fields over five ligands and four models come out **0 differing**. 🔴 **IT IS ELEMENTWISE, WHICH TOOK ONE TRICK**: `OCC(O)CO` is the obvious glycerol and puts oxygen first where the CCD's GOL is C1 O1 C2 O2 C3 O3, so the corpus writes it `C(O)C(O)CO` - the same molecule in the DICTIONARY'S atom order - and the permutation disappears. A single flipped formal charge is caught as `refCharge: 1 of 1776 differ`. 🔴 **THREE FIELDS ARE REPORTED RATHER THAN ASSERTED AND EACH SAYS WHY**: `refPos` is a conformer and two are both correct (docs/AF3.md's own 0.65 A floor); `refAtomNameChars` differs only where a dictionary name is not element-plus-counter, and a ligand with no CCD entry has no name to match; and the bond LIST order in `ligandSpans`, whose excuse is not "it does not matter" but that `bondMatrix` and `bondOrderMatrix` are derived FROM it, are what the model reads, and are asserted identical - the gate refuses the excuse if either is missing |
| **Does a SMILES ligand fold?** | `npm run test:smiles-path` (`tools/check-smiles-path.mjs`) - the same molecule from its CCD code and from its SMILES, folded through AlphaFold 3, scored on BONDS. GOL, BTN and EDO: the two routes agree to **0.035 A** and biotin (three stereocentres, two fused rings) is 0.150 rms against the code's 0.142. 🔴 **AND TWO SOLO LIGANDS FOR SCALE, BECAUSE A DICTIONARY TWIN MUST BE HAND-WRITTEN IN THE DICTIONARY'S ATOM ORDER AND THAT DOES NOT SCALE** - the page admits 150 heavy atoms and the largest paired case is 16. A **62-atom paclitaxel core** folds at bond rms 0.052 A and an erythromycin fragment at 0.050, both better than the glycerol the CCD path folds at 0.057. 🔴 **THE PAGE REFUSED `smiles` UNTIL NOW AND THE REFUSAL WAS HONEST** - "names a ligand by structure, and this page folds ligands by CCD code" - because a ligand reached the featuriser as `parseCcdComponent`'s output and nothing else could produce one. `src/chem/` produces one, so `test/af3-example-jobs.test.js` goes from eight of fourteen to NINE. 🔴 **AND ITS REACH IS MEASURED BY BREAKING IT**: a conformer replaced with noise fails all three loudly (bond rms 0.22 against 0.03), and one shrunk 15% changes the fold by NOTHING - so `ref_pos` is a FEATURE, not a template, and this is a plumbing gate with a geometry floor. `check-smiles-conformer.mjs` is what resolves a conformer, at 0.03 A. See docs/SMILES.md |
| **Is a SMILES read correctly at all?** | `npm run test:smiles` - four gates against **real RDKit** at `/home/ubuntu/.venv-rdkit`, which is an ORACLE and not a dependency (nothing under `src/` imports anything derived from it). Over **74 molecules**: the graph 74/74, the stereo 43/43 tetrahedral and 2/2 double bonds, the conformer at **0.022 A mean bond and 2.61 deg mean angle** from RDKit's MMFF with every aromatic ring inside 0.003 A of flat, and 49/49 chiral centres built with the right hand. 🔴 **THE CORPUS WAS DOUBLED AFTER THE FIRST 51 ALL PASSED, AND THAT FOUND THREE MORE BUGS** - a corpus that agrees everywhere proves only that the cases somebody thought of are right. The new half was chosen by reading the parser for branches nothing exercised. 🔴 **AND THEN HAND-PICKING RAN OUT, SO THE FOURTH GATE NEEDS NO CASES**: RDKit re-writes each molecule as arbitrarily many different SMILES - different starting atom, branch nesting and ring-closure digits - and every one must parse to the same graph with the same hand at every centre. **1148 re-writings and 1622 centre checks**, aimed squarely at the ring-closure bookkeeping that once inverted nine of twenty-nine stereocentres. It found nothing, which is the evidence the parser is done. 🔴 **AND ITS FIRST VERSION COULD NOT FAIL** - it compared each conformer against the sign `chiralCentres` had asked for, so restoring the original bug still reported 1622/1622; taking the volume over the neighbours in CANONICAL order gives 28 failures. The third chirality check here to need rescuing from comparing something with itself. 🔴 **IT READ 0.055 AND 11.87 AND FOUR FIXES CLOSED IT**, each of which had produced a chemically plausible molecule: an embedding axis that COLLAPSED (power iteration finds the largest eigenvalue by MAGNITUDE and a random distance draw is not positive semi-definite, so `sqrt(max(v,0))` zeroed a whole coordinate - ATP and biotin were embedding PLANAR); a sulfoxide flattened by the sp2 rule, which is `idealAngle`'s own mistake in a second place; steepest descent replaced by SHAKE-style constraint PROJECTION, because the gradient alone satisfied ATP's bounds in 0 of 24 starts and a CF3 group in 3; an ADAPTIVE attempt count, since four attempts left ATP's purine ring at 176 degrees where it should be 118; a VSEPR lean, because one angle per ATOM cannot say that DMSO's sulfur is 96 between its methyls and 107 to its oxygen; and rings solved as CYCLIC polygons, because thiophene's 1.71 A S-C and 1.37 A C-C make its interior angles nothing like a pentagon's 108. 🔴 **AND ONE OF THEM CANNOT SEE AN INVERTED CONVENTION**: `check-smiles-conformer.mjs` asks whether the molecule built has the hand this port ASKED for and both come from `chiralCentres`, so flipping `ANTICLOCKWISE` still reports 29/29 with every molecule the wrong enantiomer; `check-stereo-vs-rdkit.mjs` reads RDKit's own geometry and goes to 0/29. Two gates, one convention. 🔴 **AND AN AROMATIC BOND'S ORDER IS NOT ASSERTED** - benzene has two Kekulé structures and they are the same substance; asserted bond by bond this failed 6 of 51 on the arbitrary choice alone, so what is asserted is that the structure chosen is LEGAL (`valenceProblems`) |
| **Where does building a conformer on the DEVICE pay?** | `tools/gpu/bench-smiles-conformer.js` - **from 16 conformers a dispatch, and 16.3x at 256** (host 670 ms, device 41). 🔴 **AND IT LOSES 5x ON ONE LIGAND, WHICH IS NOT OVERHEAD.** WGSL forbids a barrier inside control flow depending on a workgroup reduction - the uniformity analysis will not call a value read back out of workgroup memory uniform even when it provably is - so the kernel CANNOT stop early: fixed step count, a `finished` flag, `select` instead of branches. It runs the full budget where the host breaks out, which is strictly more arithmetic. The device's cost is FLAT (35 ms from 1 to 256 conformers) and the host's is linear, so the batch axis is the whole story. 🔴 **AND THE MEASUREMENT MOVED THE TARGET**: the O(N^3) triangle smoothing that looks like the expensive step is **6%** of the host time; the refinement is 66%. `tools/gpu/check-smiles-conformer-gpu.js` is the differential and it took TWO reformulations: comparing each side's BEST attempt compared two selections as much as two solvers, and handing both the SAME start then showed the real obstacle - the objective is not convex, so the two arithmetics reach DIFFERENT local minima from one start (ATP's were **74 degrees apart with errors 0.297 against 0.303**). So coordinates are compared only where the bounds admit one answer, and there the agreement is EXACT (glycerol and benzene, bond 0.0000, angle 0.00); elsewhere what is asserted is QUALITY and chirality, and the device is usually the better solver because it never stops early |
| 🔴 **Why not @rdkit/rdkit, which is official and exists?** | **Because MinimalLib is 2D ONLY, which decides it before size does.** Run here on biotin: every z is `0.0000`, the molblock header reads `RDKit          2D`, and the only coordinate entry points are `set_new_coords`/`generate_aligned_coords` - CoordGen, a DEPICTION library. No `EmbedMolecule`, no ETKDG, no distance geometry, so it cannot produce `ref_pos`, which is the one thing the models need and the hard half of the work. It would replace the PARSER - already 51/51 against real RDKit - at **2.39 MB gzipped against this port's 50 KB** (18 KB with the comments stripped), forty-eight times the bytes - re-measured 2026-09-17, and the 34 KB this row used to claim predated the device kernel. A custom WASM build goes the wrong way (adding DistGeom makes it bigger); OpenChemLib at ~400 KB gzipped is the one real alternative and would still need every gate above pointed at it |
| 🔴 **Does the page OFFER a model the site cannot serve?** | **It did, and the deploy being behind is the only reason nobody saw it.** `bundleBaseUrl` is `bundle.remote ?? bundle.directory`, and a family with `remote: null` falls back to `./model-intellifold2-int5/` - a LOCAL export the build deliberately never copies, because Pages caps a site at a gigabyte and that bundle alone is 612 MiB. `index.html` offers IntelliFold-2 and RoseTTAFold3; **neither has a remote**, and the Pages workflow's three download steps are for monomer, multimer and af3, all of which DO, so every step is a no-op and nothing unpacks a local bundle. The live commit predates both options, so it is correct in the tree and correct on the site and broken the moment they meet. `build_site.py` now DROPS an unservable `<option>` from `dist/index.html` and says so every build - never from the checkout, because a developer with the local export is who those options are for. Publishing the bundle and re-pinning its `remote` brings the option back with no edit anywhere: the registry decides |
| 🔴 **Does the SITE still build - the only gate that sees a CONSTRUCTED path?** | `npm run test:site` (`python3 tools/build_site.py`). It assembles `dist/` and runs its own unresolved-import check over the tree it is about to publish. 🔴 **IT IS IN THE GATE LIST BECAUSE A PATTERN OVER TEXT CANNOT SEE A PATH ASSEMBLED AT RUNTIME.** The `src/` reorganisation broke five python tools that build paths from segments - `(ROOT / "src" / "reference" / "manifests" / "index.js")` - where no literal path exists for any regex to match, and `test/imports-resolve.test.js` found **none** of them. It also broke `tools/write_manifest_module.py`, whose `"module"` entries are WRITE TARGETS: it would have written thirteen manifest modules into a directory nothing loads, and the first symptom would have been a published bundle the page could not find. **Run it after any move under `src/`.** 🔴 And it walked `.ipynb_checkpoints` into the deploy check until today - the fourth walker to learn that an editor's snapshot is not source; `test/helpers/source-files.js` is now the one place the JS side states that rule |
| 🔴 **Does the Colab bridge carry a fold BOTH WAYS, and is the feed LIVE?** | `npm run test:colab` (`tools/check-colab-bridge.py`) - **no GPU and no weights**: it starts `tools/colab_backend.py`, which opens `index.html?role=runtime` headlessly, and drives every route from a reader's side over plain HTTP. 🔴 **THE FEED USED TO BE PULLED AND THAT IS WHY IT WAS NOT LIVE.** The backend collected the page's status writes, bar fractions and sampler frames by evaluating a splice over CDP every 250 ms, so a reader saw the fold only as often as a busy page answered the debugger - reported from a real runtime as the bar **sitting at "embedder · 1%" for a whole fold**, everything arriving at the end. `web/colab-bridge.js` **pushes** now, in the task that made the event, and the reader's commands come back through the broker's second mailbox (`/in` and `/out` beside `/up` and `/down`) - so the backend no longer clicks `#predict`, scrapes the status line for the word "failed" or reads the structure out of the download button: CDP is left with starting the browser and naming the card. What the gate proves is the TRANSPORT - the announcement, a `ping` answered as a `pong` (203-414 ms here), both clocks on every event (`at` from the page, `got` from the broker, so a slow feed and a slow fold are two numbers), the watermark's idempotence, one-fold-at-a-time as a 429, and a token refusal on all five routes. Three mutations caught: the push removed (four arms red), the arrival stamp dropped, the busy refusal removed. 🔴 **IT CANNOT COVER A FOLD** - the command is driven with an empty entity list and what comes back is the page's own refusal, so the command path and the `result` event are real and the model is not exercised. See docs/WEB.md |
| 🔴 **Do two models COLLIDE in one pipeline cache - which is what the PAGE does?** | `npm run test:cache` (`tools/gpu/check-pipeline-key-collisions.js`) - two models' trunks in ONE process against ONE `ComputePipelineCache`, in BOTH orders, because a collision is only reported by the SECOND compile. 🔴 **EVERY OTHER MODEL GATE HERE SPAWNS A FRESH BROWSER PER MODEL**, so nothing tested this: `check-template-path.mjs`, `check-ligand-path.mjs`, `test:stock` and `test:portable` all loop by launching `gpu-chrome.mjs` again - one device and one cache each - while a visitor who folds with AlphaFold 3 and then switches to RoseTTAFold3 reuses the cache. Same shape as `LOCALFOLD_STOCK_FLAGS`: the configuration every gate checked was not the one that ships. It found one on its first run, reported from a real session - `af3-msa:59:128:64:128:0.00001:fast:false:msa:keyMask - line 8 of 35: "const DIMENSION: u32 = 8u;" against "...= 32u;"` - because every kernel in msa-attention-webgpu.js is built from one `common` preamble and so embeds HEADS and DIMENSION, **including `keyMask`, thirteen lines that read neither**, while the stack keyed on the channel widths alone. af3 and openbind0 are msaChannels 64 with 8 heads of dimension **8**; rf3 and boltz2 are the same 64 and 8 with dimension **32**. Identical key, different text. 🔴 **AND A PAIR PROVES NOTHING ABOUT A THIRD**: af3 and rf3 alone were clean on the CONFIDENCE head, and adding intellifold2 found a second one - `af3-confidence:59:24:0.00001:fast:refalse:hntrue:shfalse:psfalse:cdfalse:embedProject` at "line 4 of 39: `const C_Z: u32 = 128u;` against `512u`" - because that key named `tokens` and `dense` and five flags and none of the three channel WIDTHS its shaders embed. So the gate defaults to the whole lineage and skips a bundle the box lacks. 🔴 **AND ITS DEFAULT STAGE IS `fold`, NOT `trunk`**: the confidence stack is not compiled by a trunk pass, so `--stage=trunk` is clean on exactly the pair that collides. A gate that cannot reach the bug it was built for is not a gate; the trunk arm is for bisecting. 🔴 **AND OpenDDE NEEDED ITS OWN WEIGHT SET TO BE SWEPT AT ALL**: it runs a structural-token expander, a refiner and its OWN confidence head, so `foldBatch` with AF3's weights dies before compiling any of them - which is why the first version covered opendde at `--stage=trunk` only and left its diffusion and confidence unswept. The dialect picks the set (`structuralTokens`), and the fold arm now **refuses a model that produced no confidence number**, because `foldBatch` skips a head whose weights are absent and returns a structure all the same - which is exactly how a stack drops out of this gate without anyone noticing. Seven models both ways, and each row says which confidence path ran. 🔴 **AND IT COVERS AF2 AND ESMFold2 TOO, BY DRIVING THEIR OWN TOOLS.** Their setups are nothing like the AF3 lineage's, so `--tools=` imports a tool's `main` and calls it in THIS process against THIS device the way `probe-compiles.js` wraps one - no second copy of an `AlphaFoldFixture` or an ESM-C tower to keep in step. The default runs AF2 monomer, AF2 multimer and ESMFold2 BESIDE the model sweep, so one command covers the page's whole menu: **the four families share the generic kernels** (`block:${kernel.cacheKey}`, `attention:pair-bias:${heads}`, the transition and triangle keys), so a CROSS-family collision is a real case. AF2's two graphs are the sharpest within-family one - they build `block:transition:normalize:${weightPrecision}` and `block:global-attention:query:...` from TWO DIFFERENT FILES. All clean, 46 s. 🔴 Specs are separated by SEMICOLONS because a tool's own arguments contain commas: `--chains=30,29` split into two specs and the second failed with "--chains sums to 30, not 59".  `msaAttentionKeyPart` is the one place that states the MSA half now, and `test/msa-attention-pipeline-key.test.js` is the CPU half. 🔴 THE CACHE IS WHAT CAUGHT IT: it indexes by SOURCE as well as by key, so it refused and named the line rather than handing rf3 AlphaFold 3's kernel |
| **Can a visitor fold with AlphaFold 2's OTHER FOUR MODELS?** | **Now yes, and the page offers the number beside the model row.** `tools/export_monomer_model.py` builds any of the five from `params_model_N_ptm.npz` with numpy alone - the monomer had NO such path (`model/` came out of a JAX capture script over fixture files that are not in the repository), where the multimer had one all along. It is checkable because the tensor NAMES come from the shipped manifest rather than a counter: rebuilding model_1 folds to **-1287025 / 62.646 / 0.3163**, the shipped bundle's fold to the digit. 🔴 **AND THREE OF THE FIVE HAVE NO TEMPLATE EMBEDDER** - model_3, model_4 and model_5 are the template-free ones and those 67 tensors are not in the checkpoint - so this port died in a gather naming a tensor rather than the fact. `templateWeights` returns null, both model paths skip the stage and the page refuses a template by NAME. 🔴 **ALL of a section absent is a model where SOME of it absent is a bug**, and an empty parameter table is the worst of both. Measured, all ten built here (97 MiB, three seconds each): monomer 62.646 / 62.435 / 58.468 / 61.866 / 63.941, multimer 47.582 / 50.373 / 51.676 / 51.271 / 50.399. `fold-af2.js --bundle=<directory>` folds any of them |
| **...and what does offering five cost a visitor?** | **43 MiB each rather than 97, through `tools/pack_delta_model.py`.** The five are one training run continued five ways, so a later model stores as a DIFFERENCE at three bits - an ordinary bundle in ESM-C's int3 group 128 codec plus a `delta` header, read by `src/bundles/delta-tensor-store.js`, applied on the device by `planBlockUpload(..., { accumulate: true })` and gated by `npm run test:delta` at **0 differing of 41,094,464 f16 results**. 🔴 **AND THE WEIGHT relRMS SAYS THREE BITS CANNOT WORK, WHICH IS WRONG.** Read against quantize_model.py's pLDDT table a 3-bit delta on model_3 lands at 0.0545 - worse than int5 symmetric's -7.8 pLDDT - and the five checkpoints really are two families (0.087 within {1,2}, 0.243 across to {3,4,5}). Folded, it is **1.95 A against its own bundle's 1.94** on 5CAJ. The difference was the STRUCTURE MODULE, which quantize_model.py keeps at float32 for the reason a delta must too: delta'd with everything else it is -12.7 pLDDT at three bits and -4.9 at four, and excluded - 2.2% of the weights, 8 MiB of a 43 MiB delta - it is **-0.06**. A whole-model norm cannot see which tensors it is averaging. 🔴 **AND THE 59-RESIDUE GATE SEQUENCE CANNOT JUDGE IT**: model_4's delta reads 3 pLDDT low there and 0.02 A on a real target, and the same sequence puts f16 and int8 encodings of IDENTICAL weights 1.6 pLDDT apart. 🔴 **AND TWO BITS (24 MiB) WAS MEASURED, CHOSEN, SHIPPED AND TAKEN BACK WITHIN THE HOUR - BY A SINGLE-SEQUENCE FOLD.** With a 7907-row alignment it costs a tenth of an angstrom and looked like a fair trade; the same bundles on the 59-residue gate sequence with NO alignment read **61.959 / 51.248 / 37.923 / 57.630** against 62.435 / 58.468 / 61.866 / 63.941 - model_4 loses **twenty-four points of pLDDT** and the geometry gate still passes it. An alignment PINS the answer so a coarse delta hardly moves it, and without one the weights are all there is: a compression measured only on deep-MSA targets is measured on the easy half of what the page does. Three bits is exact on all four both ways; `--bits 2` stays for anyone who always folds with an alignment. 🔴 AND EVEN ON THE EASY HALF two bits is a SHIFT and not scatter - **+0.094 +/- 0.020 A with 4 of 4 seeds the same way** against a seed band of 0.163, with a per-model pLDDT bias (-0.22, -0.47, -0.13, -1.61) that reorders the five. 🔴 A delta host-packs (no shard holds its codes), which is **1092 ms against 637** until the accumulate path is wired into the resident packer. See docs/AF2.md |
| **Does every model's TEMPLATE actually move its fold?** | `npm run test:template` (`tools/check-template-path.mjs`) - 5CAJ chain A through every AF3-lineage bundle **and AlphaFold 2's monomer** TWICE, with its own crystal as a self-template and without. 🔴 **THE ARM NOTHING WAS RUNNING, AND ONE MODEL'S TEMPLATE WAS INERT.** Both oracle checkers for this stage - `check-af3-template.js` and `check-fused-template-features.js` - drive the module with `templates: 1`, and a FOLD pads the slot count to FOUR, so a dialect that averages over the wrong denominator is exact in every module check and a quarter strength in every fold. rosettafold3 was: with a perfect self-template it moved **17.949 A to 17.771** while AlphaFold 3 takes the identical input to 0.281 and IntelliFold-2 to 0.254. Its reference averages the FEATURES over PRESENT templates and runs ONE forward (`a_tij = einsum('t,tijc->ijc', present, feats) / clip(present.sum(), 1)`) where every other family runs a forward per SLOT and averages the outputs. `templateFeatureMeanOnePass` in dialect.js; **rf3 now reads 0.137, the best of the seven**. 🔴 **AND THE MODULE CHECKER'S 0.061 FOR rf3 IS NOT THE DEFECT, IT IS THE int5 BUNDLE** - the control says so: boltz2 int5 **0.167** against boltz2 f32 **8.25e-7**. A residual taken on a quantised bundle is not comparable with one taken on a float32 one. 🔴 **AND IT RUNS BOTH ARMS**, because "0.14 A with a template" is evidence only if the same model is 17 A without one; a checkpoint that had memorised the target would pass a one-armed gate. 6MRR CANNOT be the target here - it folds to 0.5 A from its sequence alone. 🔴 **AND AF2'S MONOMER WAS THE EIGHTH ENTRY AND ITS TEMPLATE REACHED NOTHING** - `monomer.js` and `query-only.js` built their template call from a literal naming neither `template` nor `useTemplateUnitVector`, so the term accepted a slot the driver never passed and the MULTIMER forwarded: **21.195 A to 2.371** once it did, TM 0.1239 to 0.9162, and the gate returns exactly the control's 21.195 with the forward deleted. CLAUDE.md's allow-list trap for the third time at this seam. An entry may carry its own `tool`, `args` and bars, since a 255-residue monomer at 16 rows does not land where an AF3 fold does. 🔴 **AND THE LAYOUT IS NOT INTERCHANGEABLE**: AF3 featurises a DENSE-24 slot in per-residue conformer order and AF2 reads **atom37** (0 N, 1 CA, 2 C, 3 CB, 4 O), so `templateSlotAtom37` converts by atom NAME and `test/template-atom37-layout.test.js` pins it. 🔴 **AND STRIPPING SIDE CHAINS IS A NO-OP HERE**, which is what AF2BIND's "nosc" weights rest on: the fold is byte-identical, and that is not the flag failing - it zeroes 810 mask entries, keeps CB on all 246 residues that have one (ColabDesign's `rm_target_sc` masks `[..., 5:]`, so **CB survives**), and the packed geometry is **0 of 408726** apart. The first reading was 339300 differing and was a NaN artefact from a length-`tokens` chain mask where `templateGeometry` wants `tokens*tokens`. 🔴 **THE MONOMER ORACLE WAS NOT RE-RUN**: `check-monomer-template.js` needs an f32 bundle this box lacks and two dumps whose dumper needs a package the A10 lacks. The two-armed fold gate and the layout test stand in, both watched failing; see docs/AF2.md |
| **Does a COMPLEX fold, and is its interface right?** | `tools/gpu/fold-complex.js --target=1brs --chains=A,D --template=/tools/fixtures/1brs-crystal.pdb` - the AF3 lineage had no complex scorer, so every number in these docs was a single chain of 68 to 92 residues. It fits ONE superposition over every chain together (two chains can each be perfect and still be nowhere near each other) and reports each chain twice: `inComplex` in that shared frame and `alone` re-fitted by itself, which separates a PLACEMENT failure from a FOLD one. 🔴 **AND `interface.fnat` IS THERE BECAUSE THE FIRST TARGET HAD NO INTERFACE.** 5CAJ A:B is two independent copies in the asymmetric unit - closest inter-chain CA **11.08 A**, **zero** contacts under 8 A, centroids 44.2 A apart - so its 9 to 24 A across three seeds was the scorer asking an unanswerable question, and it read exactly like a placement defect. `nativeContacts: 0` now says the TARGET is wrong. 1BRS A:D (barnase and barstar, 36 contacts under 8 A) is the default: af3-int5 with a self-template is **0.475 A, TM 0.9928, fnat 0.944**, against 16.666 A and fnat 0.000 with no template and no MSA. The panel: **rf3 0.137**, protenix2 0.452, if2 0.455 (fnat 1.000), af3 0.475, boltz2 0.517 - and rf3 was **12.189 A** here before the template defect below was found, which this tool is what found. 🔴 **AND THE LARGE TARGET IS 1TIM A:B** - triosephosphate isomerase, a real biological homodimer of **494 residues with 101 native contacts**, nearly three times 1BRS's interface. rf3 **0.151 with fnat 1.000 (101 of 101)**, af3 0.957, boltz2 1.027, protenix2 1.047, if2 1.065, against 10.569 A and fnat 0.139 with no template. 🔴 `--no-span-chains` IS THE ARM THAT ISOLATES AN INTERFACE: the same MERGED slot with the cross-chain block masked, which separates "the template can speak across the boundary" from "a merged slot carries full intra-chain weight where two per-chain slots carry half". Worth **0.227 -> 0.137 for rf3** (whose 66 columns are a CA-CA distance distribution, which a cross-chain distance is an instance of) and **0.478 -> 0.475 for af3** on 1BRS - and then **1.293 -> 0.957 for af3** and **0.403 -> 0.151 for rf3** on 1TIM. 🔴 So it is about the SIZE of the interface, not about the model: 36 contacts buys AlphaFold 3 nothing and 101 buys it 1.35x. "Worth nothing" was one target's answer read as a rule, and the second target reversed it. 🔴 It scores over the best relabelling of INTERCHANGEABLE chains (`chainAssignments`), because a homodimer's chains are the same molecule and a perfect prediction with the labels swapped superposes as a total failure - `asLabelled` is the unpermuted score beside it |
| **Does our whole BATCH match the reference's, field by field?** | `npm run test:batch` - `check-atom-windows.js` then `tools/check-batch-fields.js`, which compares **43 of the dump's 60 fields** (108 for opendde) across all seven AF3-lineage models and **TWO targets**, for **14 exact model/target pairs**. The window gate before it compared TWO fields, and between them those two found the atom key window wrong in four of seven models and a chirality term twice recorded as a no-op. 🔴 **AND ONE TARGET REACHES FOUR FEWER CONVENTIONS THAN THE TABLE HAS.** 6MRR is a plain protein: no ligand, no modified residue, so `atomizedElementNames`, `atomizedUnknownRestype`, `atomizedUnknownMsa` and `atomizedBackboneBonds` are inert in every measurement this port had ever taken - and all four were simply **not implemented**, two of them in models that ship. `dump_af3_batch.py --ligand GOL --ptm SEP@3` builds the target that can see them (83 tokens; the phosphoserine contributes ten and the glycerol six). rf3 renames an atomised atom to its ELEMENT; boltz2 and rf3 give it the UNKNOWN restype, but only rf3 carries that into the ALIGNMENT (the references' own MSA query rows: af3 15, boltz2 15 with aatype 20, rf3 20) - one tuple in the reference, two behaviours here; and rf3 alone bonds the atomised residue back into the chain, 18 bonded pairs where the other six have 14. 🔴 **PER-ATOM FIELDS ARE COMPARED ONLY WHERE `ref_mask` IS LIVE ON BOTH SIDES**, the third time this port has learned that: `ref_element` differing in one slot of 1632 in the four families that DROP the terminal atom read as a defect in four shipped models, and it is the OXT - the reference masks it and keeps its element, name and position, this port never creates the atom, both compute a conditioning row for it, and **no gather on either side reads that slot live**. 🔴 **AND `ref_pos` IS A REPORTED FLOOR, NEVER A FAILURE**, or the gate is red forever: it is the shared ideal conformer set, and the frame-free arm says what KIND of difference it is - intra-token pairwise distances agree to **rms 0.65 A, worst 3.77 A on a LYSINE**, the same molecule in a different rotamer and frame. 🔴 **AND IT GOES THROUGH `af3BatchFromA3m`**, because batch.js forwards the dialect to the featuriser field by field and a gate that skips that stays green while the PAGE featurises with another model's conventions - verified by deleting a flag from that forwarding and watching this go red. Which is why `featuriserDialect()` now exists: three call sites listed those fields by hand. **Nine of ten conventions turn it red under `--falsify`**; `dedupeSelfMsa` cannot be seen here because the alignment is the caller's, and the header says so. 🔴 **AND IT COVERS OpenDDE'S SECOND TOKEN SPACE NOW - 109 FIELDS - WHICH WAS COMPARED AGAINST NOTHING AND HAD TWO DEFECTS.** `structbook` is the mapping that DEFINES that space, so a wrong entry makes every stage after it wrong on a shipped model. **The polymer/ligand chain link**: the neighbour test read `chainOfResidue[residueOfToken[token]] ?? 0`, and `residueOfToken` is **-1 for a ligand atom**, so the fallback made every ligand token chain 0 - what a chain-0 polymer residue gives - and the glycerol's first atom was linked to the protein's last residue. Same shape as the `== null` trap: an ABSENT thing collapsed onto a VALID value. 🔴 **AND THE FIRST REPAIR MADE IT WORSE, 3 wrong links to 10**, because a ligand's atoms ARE linked to each other and only the BOUNDARY is refused - which is what a per-token `asymId` says and `chainOfResidue` cannot. **And `struct/token_index` was zero-based** where featurise.js and the reference are one-based: inert, since it reaches the model as a DIFFERENCE, and that is why nothing caught it. 🔴 **TWO THINGS THE COMPARISON HAD TO GET RIGHT**: the reference PADS to a bucket (130 real subtokens against 160), and `ref_space_uid` is a **PARTITION, not an array** - it answers "same space?", so the labels are arbitrary and this port numbers densely where the reference skips one. Elementwise that is 3116 of 3120 differing and reads as a serious defect; as a partition both give 68 spaces over the same atoms, exact |
| **Is this bundle the weights the reference LOADS?** | `python3 tools/check-bundle-vs-params.py --bundle= --digest=` with `tools/oracle/dump_af3_params_digest.py`. 🔴 THE ONE QUESTION EVERY OTHER CHECKER ASSUMES: they all feed ONE bundle to both sides, so a tensor that is the wrong SIGN or the wrong WIDTH is invisible to all of them. It found four negated tensors in boltz2's export and ten stale ones in OpenDDE's |
| Does the GPU diffusion conditioning match the reference, at THIS bundle's widths? | `tools/gpu/check-af3-diffusion-conditioning.js --model=` - it passes on af3, openbind0 and opendde now, at 1e-5 with a separation control of 1232-2362x. 🔴 It passed for as long as OpenDDE's bundle existed WITHOUT COMPUTING ANYTHING, because `NaN > 1e-5` is FALSE; every comparison written `if (x > bound) throw` here has the same hole. 🔴 AND IT SWEEPS TWO DIALECTS OVER ONE BUNDLE, so it must re-cut the weights BOTH ways - it could only splice the openfold3 columns in, never strip them, and an openbind0 bundle's own 833 columns then tripped the LayerNorm assertion inside the kernel it was there to measure |
| Does the AF3 head still match AF3? | `tools/gpu/probe-head-vs-af3-steps.js --dump=/af3-rings20.json` |
| Is a fold still the same fold? | `tools/gpu/probe-sidechains.js --steps=8` |
| ...and did a KNOB change the structure, which pLDDT will not tell you? | `python3 tools/diff-fold-coords.py --b="--attn-splits=4"` - **`meanPlddt` matched to sixteen digits across an arm that moves 33 atoms** |
| Is a MODIFIED residue the right shape? | `tools/gpu/probe-modified.js --code=SEP --at=3` |
| **Is a fold the SAME fold twice in one process?** | `tools/gpu/bench-af2-warm.js` - it checksums every pass and throws unless they match, because three predictions of one graph over one input in one process share their pipelines, their buffers and their clocks, so a difference is a MEMORY bug and not a precision question - a race, or a write out of bounds. 🔴 It caught the second in the matrix flash attention's staging and that was recorded as the first for a whole campaign: the staging loop's trip count is NOT uniform at a head of eight, which an AF2 fold compiles, so unrolling it to a fixed two iterations writes past the array. `--allow-nondeterminism` is the escape |
| **What does this backend do with an over-dispatched invocation?** | `tools/gpu/probe-grid-overdispatch.js` - it drives the SHIPPED `ADD_IN_PLACE_SHADER` over a grid `linearGrid` rounded up, and reads the last in-range element, which should be exactly 1. On this A100 at the 160-residue shape (over-dispatch **917,504**) it is **1, 1, 1**: Dawn over Vulkan DISCARDS an out-of-range write where Metal CLAMPS it onto the tail - which is the whole reason the M2 raced above 128 residues and this box did not. 🔴 A BACKEND THAT DISCARDS IS LUCK, NOT A GUARANTEE: WebGPU permits either, so a missing bounds check is invisible here and catastrophic there. The over-dispatch is 0 up to 128 residues because 128*128*128 is exactly `GRID_WIDTH * 64` |
| 🔴 **Is a fold the same fold twice - and is a race THIS box's?** | `bench-af2-warm.js` again, and the answer differs by machine. An M2 reports it throwing at 160, 200 and 400 residues; **the A100 does not reproduce it at any length** - 59 through 825, then `--passes=8` four rounds at those three, twelve runs all agreeing with identical checksums across runs (160 -9869845, 200 -704434, 400 -15124842). Read the PATTERN in the error, which prints every checksum: `A, B, B` is a first-touch difference (pass 1 reads zero-initialised buffers, later passes read recycled ones) and NOT a race; `A, B, C` is nondeterminism. `--passes=8` separates them - an uninitialised read stays two values, a race keeps making new ones |
| ...and is it the BUFFER POOL? | `bench-af2-warm.js --no-pool`, one run. A pooled buffer keeps the previous fold's bytes, so a kernel reading a region it did not write differs between fresh and reused - which looks like a race and is not. On the A100 at 160 the checksum is **-9869845 either way**. 🔴 IT RETIRES RATHER THAN DESTROYS AND SO LEAKS BY DESIGN: destroying on release killed every length with "[Buffer] destroyed", because a released buffer is still named by an in-flight command buffer - which is WHY the pool exists. Fine to 160 on 40 GB, `Error.cpp:119` at 200. Short lengths only |
| What does AF2 predict, distogram and pLDDT, per recycle? | `tools/gpu/probe-af2-dgram-plddt.js --sample=10` |
| Is the sampler converged at this step count? | `tools/gpu/probe-flow-sigma-by-size.js --panel=churn` |
| Do recycles help a complex? | `tools/gpu/probe-recycles-on-complexes.js` |
| Does MSA depth help a complex? | `tools/gpu/probe-msa-depth-on-complexes.js` (**goes to the network**) |
| Does the sampler setting matter on a real binder? | `tools/gpu/probe-designed-binder-sampler.js` (**network**) |
| **Where does a fold stop BINDING, and which dispatch decides?** | `tools/gpu/probe-binding-ceiling.js --tool=fold-af2 --lengths=59,118` - it runs the wrapped tool at TWO lengths, because a binding that grows as `L` and one that grows as `L^2` are indistinguishable in one run and give out at completely different lengths. Each label gets a growth exponent and an extrapolated ceiling. On AF2, 86 labels: `opm.contract` at **724** (handled - the tiled path takes over), the pair transitions at **1,448** (handled - `transitionChunkRows`), then **2,047 with TWENTY-EIGHT labels on it** (the MULTIMER's is 42) and 2,896 with 22. 🔴 AND 2,047 IS THIS CARD'S: the ceiling is `sqrt(maxStorageBufferBindingSize / (cZ * 4))`, which is 2 GiB here and **4 GiB on an M2**, where it is 2,896 and coincides with the allocation wall - so on that part there is nothing to window at all. 🔴 RUN IT AT LENGTHS WHERE BOTH SAMPLES ARE ABOVE EVERY CHUNK THRESHOLD, or a handled label reads as a ceiling: at 59/118 the pair transitions read 1,448 and the multimer's 1,023, and at 200/400 they are absent from the ranking entirely because chunking has started. `sampleFraction` and `caveat` say when the extrapolation is a guess - at 59/118 it is 2.7% of the limit. 🔴 READ `lowestCeilingGroup.labels` BEFORE THE NAME. This returned one label once and it was read as a to-do list; the triangle is simply first in a sorted list, and windowing its twelve moves the fold's ceiling by ZERO because the thirteenth member of the tie refuses at the same residue. Every dispatch binding an `L^2 * cZ` f32 tensor is in that group. 🔴 AND READ THE EXPONENT COLUMN: a fractional one means the label already steps against a budget and the ceiling is not a prediction. The idea is @milot-mirdita's; the tool is not. 🔴 **AND IT SEES AF3 NOW, WHICH IT DID NOT.** It patched `WebGpuExecution.prototype.dispatch` - AF2's seam - so `--tool=fold` recorded **zero labels** and returned `lowestCeiling: null`, which reads as "nothing is near the limit" and means the instrument was pointed elsewhere; the AF3 stacks call `dispatchWorkgroups` from six of their own modules. `watchBindings` patches `createBindGroup` and `beginComputePass` instead - the API, not a convention - and records the BINDING's size rather than the buffer's. It also **refuses** when neither seam saw anything. Measured: **af3 2047 residues shared by 29 labels, intellifold2 1023 shared by 29** - exactly `sqrt(2 GiB / (cZ * 4))`, the same group both times (trunk-pair, pair-logits, embed.assemble-pair, the five tri.*, the five grid.*, ...), and `addPair` is not even among the largest. **So windowing AF3's add moves the ceiling by ZERO**, as it did for AF2 - measured now rather than argued by analogy, and IntelliFold-2's 1023 is a length people fold. Memory exhausts later (fitted ~3660 and ~1812), so the binding limit really is what binds |
| **Does one dispatch bind the same buffer range twice?** | `tools/gpu/probe-dispatch-aliasing.js --tool=fold-af2` - the allocator pools whole buffers by `byteLength:usage` and a stack RELEASES a block's allocations so the next block can reuse them, which is deliberate and is where the memory saving comes from. Reuse across dispatches is fine; two tensors of ONE dispatch on one buffer is not, when one is the output - workgroups then read what other workgroups write, which is a race whose symptom is a fold that differs run to run and whose cause is nowhere near the shader that shows it. An AF2 fold: **0 of 2,296**. Written to rule that out with a measurement rather than an argument |
| **How long a chain can the residual add still BIND?** | `tools/gpu/probe-residual-binding-ceiling.js` - `addInPlace` bound both tensors whole, and the pair is `L*L*channels*4` against `maxStorageBufferBindingSize`, 2 GiB here and unraisable because it is Vulkan's `maxStorageBufferRange`. The ceiling was **2,047 residues** on a card with 40 GB free: 2,047 binds, 2,048 was REFUSED outright. It windows now - the residual add alone reaches 2,896 in two windows - and below the ceiling the single-dispatch path is untouched. 🔴 BUT THE FOLD STILL STOPS AT 2,047, AND NO ONE WINDOWING MOVES IT: `probe-binding-ceiling.js` puts **28 labels** on that exact residue - the ten triangle multiplication passes and two gates, both triangle attentions' normalize and output in both stacks, both MSA row attentions' pair-normalize and pair-bias, two pair-transition normalize, `template.output` and the template residual. They all bind `L^2 * cZ` f32, so they give out together. The residual is no longer the FIRST thing to fail; it is not a longer complex. 🔴 THE NEXT WALL IS `maxBufferSize` at 4 GiB, so one f32 pair tops out at 2,896 however it is bound - which is also where a packed f16 pair's BINDINGS land, so 2,896 is the destination by either route and the cheap route is the element, not twenty-eight windows. And the price of arriving is minutes: `profile-af2-block.js --length=2040` is 1,260 ms a block, 60.5 s a recycle for the stack, on an O(L^3) contraction. Found by **@milot-mirdita** upstream, whose packed-f16 pair put their ceiling at 2,896 where our f32 put ours at 2,047. See docs/AF2.md |
| 🔴 **Does the fold survive an ODD MSA depth?** | It did not. `fold-af2.js --sequence=<400 residues> --rows=37` died in WebGPU validation and `--rows=38` folded; 39 died, 40 folded, 61 died. The matrix outer product mean binds `left` at `residuesBefore * cOuter * sequences` elements and a bound range must start on **256 bytes** - with cOuter 32 that is a multiple of 256 only when the depth is EVEN. Three things must hold: the MATRIX contraction (refused at depth 1, which is why a single-sequence fold is safe and why no gate showed it), an ODD depth, and MORE THAN ONE pair block (59 residues has one; 400 has three). Both stacks have a depth: the main one is `min(cap, depth)` and every preset cap is even, so it needs a shallow odd alignment - but the EXTRA stack runs at `min(maxExtra, depth - maxMsa)`, which is odd whenever the alignment's own depth is, so at the default 128:256 preset any alignment of 129-383 sequences with an odd count reached it. 🔴 AND WEBGPU NAMED THE WRONG TENSOR: the allocator pools by size and a pooled buffer keeps its CREATION label, so the same bug blamed `opm.left` once and `extra.msa-row-attention.normalized` the next time. `execution.js` checks the offset itself now and names the pass, the binding and the byte. Fixed by cutting the pair block on an ALIGNED number of residues, so every depth that worked is byte-identical. See docs/AF2.md |
| ...and is a depth that is not a multiple of FOUR still slower? | **Yes, and that half is open.** `vectorStaging` on the outer product mean is gated on `sequences % 4 === 0`, so 510 costs **+23% on `opm.contract` and +2.0% on the whole block** against 512 - which does MORE rows and is still faster. Padding the depth would take it and is provably safe for the OPM (the denominator is the mask product, so a masked row adds nothing to either side), but the padded rows also cross row attention, column attention and the transitions, and that has not been checked |
| Does AF2's stack match AlphaFold? | `tools/gpu/check-evoformer-stack.js` |
| Do the outer product mean's TWO paths agree? | `tools/gpu/check-opm-paths.js --length=400 --sequences=512 --cz=128` - they contract the sequence axis in a different order, so the bar is a reordering (~1e-6) and not equality |
| ...and does PAIR BLOCKING change nothing at all? | the same tool - its blocked arm forces thousands of blocks and its bar is `=== 0`, because blocking reorders no sum |
| ...and is the f16 contraction still flat in MSA DEPTH? | the same tool, and **run it at `--sequences=512`**: depth is the only axis that arm's risk lives on, and at `--sequences=8` it proves nothing |
| ...and which path is a fold actually taking? | `useOuterFirstContraction` - it was capped at 64 MiB, i.e. **128 residues**, and the fallback is 93.7% of a block at 825 |
| Does AF2 still fold the SAME structure? | `tools/gpu/fold-af2.js` - and it FAILS now if the chain is not a chain, which is the gate a collapsed 825-residue fold walked through for a whole campaign |
| **Does the MULTIMER still fold, and what does a repeat cost?** | `tools/gpu/fold-af2.js --family=multimer --chains=30,29 --repeat=3` - the repeat is the only number that prices weight residency, because a first fold is mostly pipeline compilation (1485 ms against 214 for a repeat), and every repeat is held to the first fold's atom checksum. The bundle is `af2-multimer/` in the registry and needs a manifest.json generated from src/bundles/manifests/multimer.js |
| ...and does forcing an AF2 knob still fold the same one? | `tools/gpu/fold-af2.js --tune=key=value`, the same flag `fold.js` carries. A knob no gate enters is a knob nobody has checked |
| **Does the device's nearest-centre search agree with the host loop?** | `tools/gpu/check-nearest-centres.js` - bar ZERO differing assignments over seven cases in one submit, because the assignment picks which cluster an extra row joins and so decides the prediction. 🔴 One arm DUPLICATES centres so whole groups tie, since the host keeps the FIRST at an equal score and a join that broke the other way agrees on every random alignment and disagrees on every real one. Two degenerate arms assert centre 0 outright, so the distinctness control cannot pass them by accident. Flipping the tie rule fails six of seven, 512/512 on the duplicate arm |
| ...and does the FOLD agree, not just the kernel? | `tools/gpu/fold-af2.js --host-features` is the control arm. Same checksum both ways - 26706680 at 825 residues, -1725774 at 59 - which is the only thing that settles it, because the assignment reaches the answer through the cluster profile |
| 🔴 **Is the alignment deduplicated?** | **No, and the query is in it twice.** `extractMmseqs2A3m` joins `uniref.a3m` and the environmental A3M, and `queryBlock` returns each block WHOLE - so both start with their own `>101`. A search finding only the query returns **depth 2, distinct 1**. On `tools/fixtures/test.a3m`, aligned-column comparison: 8076 rows, 7663 distinct, **413 duplicates (5.1%)**, about 20 of the default 384-row budget. `deduplicateUnpairedAgainstPaired` in chains.js does exactly this for the multimer and says why - "a duplicate does not merely add nothing, it evicts a sequence that would have added something" - and AlphaFold's `make_msa_features` keeps a `seen_sequences` set. **FIXED**, and measured: on a deep alignment it is worth **nothing** - paired by seed, delta pLDDT -0.005 ± 0.288 at 128:256 over 8 seeds and -0.036 ± 0.129 at 508:1024 over 6, the delta's sd sixty times its mean. 🔴 BUT ON A QUERY-ONLY SEARCH IT CHANGES THE ANSWER: query-once and query-twice now both give pLDDT 59.974, pTM 0.3965, checksum -459839, where plain gave the duplicate 60.079 and **pTM 0.4148**, a 4.6% shift on a number the page shows. `--no-dedupe` is the control arm. See docs/AF2.md |
| ...and does an alignment carrying only the query become a single-sequence fold? | **Yes, from all four inputs** - searched, pasted, uploaded A3M, uploaded archive - through `singleSequenceIfOnlyQuery`, which returns `text: null`, the state "Single Sequence" mode already produces, keeping the search's template hits. 🔴 AND FOR PASTE AND UPLOAD IT CHECKS WHOSE QUERY IT IS: an A3M's own first record WINS over the sequence box, so `foldsAsSingleSequence` routes only when the alignment's query IS what is being folded - otherwise it would silently fold a different protein. A searched A3M cannot differ. 🔴 THERE IS NO SPEEDUP IN IT: 855/840/843 ms across the arms, because a homolog-free search is already a two-row alignment, `maxExtra` collapses to `max(1, 0) = 1` either way, and the page already runs single mode through the same `predictA3m` with a one-row A3M. It is correctness and honesty, not time |
| 🔴 ...and the synthetic alignment had TWELVE distinct rows | `fold-af2.js` strided its gaps by `row % 11 + 3`, so `--rows=128 --extra-rows=128` was 256 rows carrying 12 sequences. Harmless until deduplication, which would have collapsed every AF2 gate to a depth-12 fold while still reporting 256. The stride is the row's bit pattern now and the tool ASSERTS distinctness. This is what moved the three AF2 baselines, not the dedupe - `--no-dedupe` gives the same -1287025 |
| Where does preparing an alignment actually go? | `featureMilliseconds` on `fold-af2.js`, from `featureStats` in src/input/a3m-features.js. At 825 residues with 512 clusters, 1024 extras and two recycles: nearest-centre 645 ms of 1072, then the 49-channel block at 204, the cluster profile at 81, and nothing else above 55. The search is 43 ms on the device |
| Does AF2's distogram head agree with AF2's structure? | `tools/gpu/probe-af2-contacts.js` |
| Which register tile does AF2's dense projection want? | `tools/gpu/bench-evoformer-linear.js` |
| What does AF2's column attention cost alone? | `tools/gpu/bench-msa-attention.js` |
| What does a sampler step cost besides the denoiser? | `tools/gpu/probe-sampler-overhead.js` |
| Where does a denoiser call's time go? | `tools/gpu/bench-head.js --profile`, and `--calls=4` for the COLD one: at 68 tokens call 0 is 1218 ms against 10 steady, and 904 of it is the transformer's resident weights - not its compile, which `probe-warm.js` measures at 71 ms |
| 🔴 ...and is that profile a FOLD, or a PREFIX of one? | read **`dropped`** in `gpuSummary` first. `profile.js` caps at 2048 passes - `createQuerySet` takes no more than 4096 timestamps - and it turns `batchComputePasses` OFF to get one row per label, which multiplies the count about sevenfold. An OpenDDE fold under `--profile` reports EXACTLY 2048, so every `gpuTotalMs` and `idleShare` ever read off one describes part of a fold. `--profile-batched` on `fold.js` and `fold-opendde.js` keeps the batching, so a whole fold fits and the GPU-busy total is true - at the cost of per-kernel attribution, which is the other question |
| ...and is a stage's time the stage's? | **No.** The head's `stage()` timers measure when the HOST returns, so the stage that awaits a readback absorbs everything queued before it: at 300 tokens `atom-decoder` reads 39 ms of a 43 ms call while its own kernels are nowhere in the profile, and the transformer reads 2 ms while its kernels are 25. The call is GPU-bound at 6% idle; the pass table is the truth |
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
| ...and at 512 TOKENS, which nothing had profiled? | `bench-trunk.js --model= --tokens=512 --msa=128 --passes=2 --profile`. The pairformer is GPU-bound - encode 22.8 ms against a wait of 1644.7 - so nothing on its host side is worth moving. `grid.attend` is 30.6% of the GPU time with 16,384 workgroups a pass, well fed. 🔴 **`pair-transition.down` launches 256 workgroups on a card that fits 4542**, 6% of the trunk's GPU time in a kernel using a twentieth of the device - the clearest starved dispatch in the port, and NOT a knob: `pairTransitionChunkBytes` at 64/128/256 MiB leaves it at 256 groups and 77.5 ms. See docs/AF3.md |
| Where does an OpenDDE FOLD's time go, and is it even GPU? | `tools/gpu/fold-opendde.js --profile --buffers --repeat=2` - the trunk phase is **96.3% GPU-idle**, and the second fold is the row a user sees |
| Where does an AF2 block's time go? | `tools/gpu/profile-af2-block.js --sequences=512` |
| ...and ALL of it, not the top twenty? | the same, `--top=200` - a block's transitions alone are fifty labels, one a chunk, and 15 ms of pair bias hid under them |
| 🔴 ...and were these knobs swept at the LENGTH that matters? | Mostly not - the priors were fitted at 400 residues and below. Re-swept at 825 with 512 sequences, one in four moved: `opmPairBlockBytes` 64 -> 256 MiB is **2.2% of a block and 180 ms of a fold for 193 MiB**, bit-identical because blocking reorders no sum. **`transitionChunkBytes` 32 -> 256 MiB is 5.4% of a block and 4.0% of a fold for 168 MiB**, also ampere-only and also bit-identical - its 32 MiB knee had been measured on a 59-residue fold as a MEMORY trade, and at 825 the transitions are 263 of a block's 339 dispatches. Flat: `attentionMatrixTile` (4x32 still right, and 8x32/6x32/4x64 are all WORSE), `opmProjectOutputPairs`, `stagedMatrixBlock`, `transitionThreadTarget` over an eightfold range. 🔴 `audit-knobs.py` calls `transitionChunkBytes` dead on its default workload, because at 59 residues nothing chunks at any candidate value - a knob whose threshold the workload never crosses looks exactly like a dead one. See docs/AF2.md |
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
| **Does this knob do ANYTHING?** | `python3 tools/audit-knobs.py --tool=fold-af2 --tuning=<probe-tuning.js output>`. For every knob in `DEFAULT_TUNING` it runs the same workload with a value DIFFERENT from the one this device resolves, and compares two digests from probe-compiles.js: every shader compiled by label and content, and every dispatch by label and grid. 🔴 A knob that moves neither is SUSPICIOUS, not dead - `keepTrunkWeights` decides releases, `batchComputePasses` decides pass grouping, `deviceFeaturisationMinBytes` decides a host route, and none of the three can move a shader. It says where to look; only reading says which it is. And the workload matters: a diffusion knob cannot move anything in an AF2 fold |
| ...and what did it find? | Two collisions the check caught and nobody had ever hit, both keys that did not name their shader. **`--tune=triangleProjectMatrix=false` killed AF3** - the triangle's weights are packed INTERLEAVED for the matrix projection and separately without it, `const W_LINEARABWEIGHT` against `const W_LINEARAPWEIGHT`, and the key named neither. **`--tune=singleProjectWorkgroupTarget=` at 220 or 55 killed it too** - `projectSplits` is derived from it, baked into the source and multiplied into the dispatch, and was not in the key. Both arms are differentials CLAUDE.md already lists, and neither had been run |
| ...and the 9 knobs it could not reach, now 0 | `--tune-json={"matrixLinear":false}` on any GPU tool takes a whole tuning patch and parses it, so an object-valued knob is reachable at last; the audit carries second values for the tile-valued ones. AF2 is now **11 moved, 38 unmoved, 0 unreachable**. 🔴 Closing the gap found the SAME bug a third time: `trianglePairProjectTile: false` reached a tile resolver as a boolean and killed AF2 with "projectTile undefinedxundefined". `shapedKnob` in device-profile.js reads `false` as unset for the thirteen knobs that take a shape or a count, and must NEVER be used on a boolean knob, where `false` means off |
| A weight buffer is slower than it was and nothing errored? | 🔴 **the device packer refused and fell back.** It throws `DeviceWeightRefusal` naming the tensor now, because a silent fallback is a bug that looks like a slow machine - `allowHostWeightPacking` is the opt-in for a bundle that genuinely cannot be decoded. Ten descriptors in AF2 copied their tensors into a literal instead of deriving them, which READS the getter and decodes the block |
| Is a projection short of threads, short of arithmetic intensity, or neither? | `tools/gpu/probe-split-k.js` |
| What do `subgroupMatrixLoad`/`Store` actually mean here? | `tools/gpu/check-subgroup-matrix.js` |
| Are the matrix units worth it on a dense projection? | `tools/gpu/bench-evoformer-linear.js --arms=8x8@f16/f16,matrix4` |
| ...and against AF3's own fused projections? | `bench-{grid,triangle}-project.js --tokens=200 --matrix=1` |
| What does packing the attention key cost, and the value? | `tools/gpu/check-attention-packing.js --dense=f32` |
| What does a dispatch cost before it computes? | `tools/gpu/probe-dispatch.js` |
| **Did a speculative WARM compile the right shaders?** | count them: `tools/gpu/fold.js --no-warm` and `fold-opendde.js --no-warm` against the same run without the flag, through `probe-compiles.js`. A warm compiling against a shapes-only stand-in cannot give a wrong ANSWER - the stack still asks the cache for its own keys - so the only failure is silent WASTE. OpenDDE went 269 -> 322 pipelines warming its refiner with the trunk's root, and 285 with the wrong pair precision, before it went back to 269 |
| **Is the same WGSL being compiled twice under two keys?** | `distinctSources` and `duplicateModules` in `probe-compiles.js`. It was, for every model: OpenDDE 269 pipelines from 191 distinct texts, AF3 223 from 156, AF2 96 from 73, ESMFold2 95 from 82. `ComputePipelineCache` indexes by `entryPoint + source` as well as by key now - OpenDDE's fold 2540/2552/2603 ms to **2404/2403/2382**, monomer's page 3418 to 3141. 🔴 The source is the key and NOT a hash of it, because a collision would hand a caller somebody else's kernel |
| **Where does a fold's HOST time go, and which loop issues the submits?** | `tools/gpu/probe-submits.js --tool=<tool> --repeats=3` - it wraps another tool the way probe-compiles.js does and changes NO tuning, so unlike `--profile` it describes the fold that ships. Read `hostMs`, not `gapMs`: every mapAsync, onSubmittedWorkDone and popErrorScope is recorded as an interval, merged into a union (not summed - a fold holds hundreds of unawaited completion promises) and subtracted, so what is left is what the CPU actually did. 🔴 AND `startupMs` IS SEPARATE, because everything before the first submit - the shards, the page - was being charged to whichever encoder submitted first, which is 1562 ms landing on `af3-atom-encoder` whose own stage measures 78. 🔴 AND ONE RUN IS NOT A MEASUREMENT: the same 300-token fold put 760 ms on `af3-confidence-embed` once and 47 the next time, hence `--repeats` |
| ...and what did it say? | **There is no host-side lever in any of the four.** A median OpenDDE first fold is 638 ms of host work in the 1715 ms that are not start-up, 338 of it the weight decode; AF2 is 290, ESMFold2 262. Encoding and submitting together are under 30 ms everywhere. See docs/PERF.md |
| **How fast can host bytes reach a STORAGE buffer, and by which route?** | `tools/gpu/probe-upload-path.js` - a pooled MAP_WRITE buffer copied on the device is **6.85 GB/s** against `writeBuffer`'s 2.15 in 1 MiB pieces and 1.22 in one call, and a big `writeBuffer` is SLOWER than the same bytes in pieces. 🔴 IT IS STILL NOT USABLE IN THE WEIGHT LOOP: a pool that recycles on device completion cannot feed a loop that submits hundreds of packs without draining, and forcing it to (128 buffers a class, 768 MiB) made the fold 417 ms SLOWER for 191 ms of `writeBuffer` saved |
| **What does a FIRST VISIT cost on a user's link, not the dev server's?** | `python3 tools/fold-in-page.py --throttle=8` shapes the whole page through CDP; Hugging Face measures 8 MB/s from this machine and the dev server is 371. AF3: **39.3 s, of which 35.5 is the trunk weight phase and 2 is the fold.** A first visit is bytes and almost nothing else - and the biggest-shard-first prefetch order, whose own comment says it cannot be measured from here, measures as **nothing**. See docs/HOSTING.md |
| **Is a first fold waiting on SHADER COMPILATION?** | `tools/gpu/probe-compiles.js --tool=fold-af2` - it wraps another tool's `main`, so any gate can be measured without ceasing to be one. Read **`busyMs`**, the union of the intervals, and never the sum: twenty pipelines compiled at once and twenty compiled in turn have the same sum. AF2's first fold WAS its compile queue (1133 ms of span in a 1163 ms fold, 1.47 ever in flight); AF3's and OpenDDE's were already packed at ~14 |
| ...and what does a first AF2 fold consist of otherwise? | `tools/gpu/probe-af2-warmup.js` - pipelines, shader modules, buffers, writeBuffer bytes and the on-device weight decode, against the wall and the repeat |
| What does the on-device weight decode cost the HOST? | `blockUploadStats` in src/weights/quantised-upload.js - staging assembly, submits and buffers, none of which is inside a compute pass and so none of which `profile.js` can see |
| ...and what is still DECODED on the host under those packers? | `tensorDecodeStats` in src/bundles/http-tensor-store.js, printed as `hostDecode`. After this session: AF2 5 ms, AF3 32, ESMFold2 0, OpenDDE 84 |
| 🔴 **Does `probe-compiles.js` actually pass your flags to the tool?** | **Only if you pass them BARE.** It forwards `args` minus `--tool=`, so `--args=--model=/model-rosettafold3-int5/manifest.json` reaches `fold.js` as ONE literal token that `option(args, "model")` never matches - and the fold silently runs fold.js's DEFAULT bundle, `/model-af3-full-f32/manifest.json`. It cost a whole comparison here: three models profiled, three answers written down, and the tell was **874.1 MiB identical to the tenth of a MiB across models with 128-, 512- and 128-channel pair tracks**. Write `probe-compiles.js --tool=fold --model=... --sequence=...`, and confirm a per-model number DIFFERS before believing the table |
| **A device weight path fell back and said nothing - what did that cost?** | `residentPackStats` in src/runtime/resident.js, printed by `probe-compiles.js` as `hostPack` with a per-label table. Every AF3-side device decode falls back to a host `pack()` SILENTLY by design, and one that costs 400 ms of a fold looks exactly like a slow machine: it found `difftx.zerogate.resident` at 496 ms in ONE call, and `w.pair-transition` at 56 calls because `residentPackedOnDevice` wrote halves only while AF3's pair track is f32 . 🔴 **AND THE ANSWER DEPENDS ENTIRELY ON THE BUNDLE, WHICH IS EASY TO READ AS A DEFECT.** On the FLOAT32 bundle a fold host-packs **1428 ms / 874 MiB** with `w.tri.out` and `w.tri.in` in the table - and that is CORRECT rather than broken: an f32 bundle carries no codes, so `residentTriangleOnDevice` has nothing to decode on the device and returns undefined by design. On the **int5 bundles that ship**, nothing falls back at all - af3 **12 ms / 26.2 MiB**, rosettafold3 **10 ms / 27.0 MiB**, intellifold2 **53 ms / 63.1 MiB**, with neither the triangle nor the grid in the table. Read the bundle before reading the fallback |
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
| ...and AF2-MONOMER's? | `tools/gpu/check-monomer-template.js` - 🔴 **and it does not run on this box**: it wants `/model.f32-backup/manifest.json` (only the int8 `model/` is here, and docs/AF2.md records that comparing against int8 reports quantisation as a fault) and `oracle-dumps/toy-template-monomer-jax*.json`, whose dumper needs the `alphafold` package beside `params_model_1_ptm.npz`. So the monomer template's WIRING is gated by `npm run test:template` and its LAYOUT by `test/template-atom37-layout.test.js`, and the term itself rests on its author's 2.7e-4 / 4.5e-4 |
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
| **Is a kernel starved because of a knob nobody set?** | `single.project` was 15.0 ms of a 104.3 ms trunk at 68 tokens - 136 workgroups on a card that holds 4542 - because `singleProjectWorkgroupTarget` is 110, which `singleProjectSplits`' own comment calls "an M2's number" and left as a parameter for a device to set. No device set it. The ampere prior now sets 2048 with 128 lanes: **whole-trunk GPU -10.8% at 68 tokens, -6.0% at 150, -1.7% at 300, -0.5% at 512**, the kernel itself 4.05x to 2.22x. 🔴 AND NEITHER KNOB PAYS ALONE - 128 lanes over the unsplit output is WORSE (15.48 -> 16.24), which is what two earlier sweeps found and recorded as NOT TAKEN. Sweeping one axis of a pair says the pair does not pay. 🔴 AND IT IS A TRUNK NUMBER: the trunk is 9% of a warm 68-token fold, so the fold moves **1%** (1.166/1.159/1.219 s against 1.184/1.169) and the first fold not at all. See docs/AF3.md. 🔴 **AND THE PRIOR SET THE TARGET AND NOT THE CEILING, WHICH MADE THE TARGET INERT.** `singleProjectMaxSplits` stayed at its module default of **3**, so the candidate list `[1,2,3,6].filter(<= 3)` could never reach the 6 the kernel's own comment says an A100 wants - and at a target of 2048 the "reaches the target" loop never fires at a realistic token count either, so every fold fell through to "take the most workgroups available" and got **3, at every length**. Now **1200 with a ceiling of 6**. Re-measured at 31 rounds of 64: splits 6 is **0.0578 / 0.0594 / 0.0734** at n = 59 / 128 / 200 against 3's 0.0906 / 0.0922 / 0.1078, and 3 wins at 400 and 512 - so 1200-with-6 picks the best arm at all five where **2048-with-6 would pick 6 at 400 and be 46% worse**. 🔴 **AND THE BENCH'S DEFAULT 11 ROUNDS OF 16 CANNOT SEE IT**: every arm reads 0.15-0.22 there and 3 looks like the winner everywhere, a false negative that nearly kept the bug. 🔴 AND THE 15.0/104.3 ABOVE IS STALE - profiled now, `single.project` is **3.79 ms of 93.1**, so this last step is 2.3% of the trunk's GPU and the wall clock does not move (190/191 ms before, 194/184 after). Taken because a prior should express its own measurement, not because a fold gets faster |
| ...and does an unmeasured GPU get AF3's geometry now? | **97% of it.** `sample-start` on an unrecognised device: **4510** with every layer off, then 2784 / 2437 / 2252 / 2129 / **1952** as the K split, atom row tile, batched gate, token tile and weight retention are each derived - against the prior's **1874**, and a whole fold of 3398 ms against 3392. Five mechanisms, no new table: the device's measured WIDTH for the K split, the atom tile and the token tile; its memory BUDGET for the batched gate and for keeping weights between folds. 🔴 `--no-prior` KEEPS the derivations, because measuring is what an unrecognised device does; `--default-tuning` suppresses them, and without that switch the DEFAULT_TUNING arm silently drifted 4525 -> 3765 and was measuring something with no name |
| ...and how? | `derivedSplitRule` in src/af3/diffusion/diffusion-transformer-webgpu.js asks the device's measured width whether the unsplit dispatch already fills it, so `crossover` - a token count in the prior - falls out of arithmetic instead. `sample-start` on an unrecognised device: 4500 before, **2784** derived, against the prior's 1877 and 2732 for the prior's own rule forced by hand. `fold.js --split-k=<json>` is how the rule is set by hand; `--no-prior` now measures, because an unrecognised device would |
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
| ...and does a template reach it? | `tools/fold-in-page.py --model af3 --template 1QYS_A`, and `--template upload:tools/fixtures/5caj-crystal.pdb@A` is the arm that needs no network. 🔴 **AND `--model monomer` TAKES ONE NOW**, which the page refused under any non-AF3 model - right for ESMFold2 and the multimer, wrong for the monomer, whose term was oracle-checked and wired to nothing. 5CAJ chain A, one recycle, single sequence: **pLDDT 31.9 / pTM 0.215 without, 76.0 / 0.795 with** and `template 261/261` on the status line. `buildTemplate` gained a `layout` argument and nothing else - atom37 for AF2, dense-24 for AF3 - and AF3's arm is verified unmoved through it at pLDDT 95.7 on the same file. The multimer, `?graph=unified` and a SECOND template are refused rather than dropped; see docs/WEB.md |
| ...and a MODIFIED residue? | `tools/fold-in-page.py --model af3 --modify SEP@3` |
| **Does the archive describe the job it wrote?** | `tools/fold-in-page.py --job-round-trip` (folds, WIPES the rows, drops the zip back) |
| 🔴 **What is a fold CALLED?** | **`foldStem` in web/app.js - one resolver for all three fold paths: a pasted FASTA `>header`, else the model prefix, through `uniqueStem`.** There were THREE places that built a stem, and the archive's request carried its own name beside them, so the object in the picker, the `.pdb` button and the saved request could all disagree. `buildFoldArchive` has no `jobName` parameter now: the request's name IS the stem. 🔴 **A JOB'S OWN `name` IS NOT CARRIED, DELIBERATELY** - every AlphaFold 3 example has one and this page names the FOLD instead, so `calmodulin_4calcium.json` saves as `af3_1`. A visible name box was built for it (and deleted the shape-comparison machinery that an invisible name needed, and had to break a trunk continuation on a rename) and then removed as not earning its place. `check-job-archive.py` REPORTS the difference and does not fail on it, the same treatment `ref_pos` gets - a deliberate gap held as a failure is a red gate for ever. `test/model-family.test.js` counts the `foldStem` call sites, which is what stops a fourth path naming its object some other way |
| 🔴 **Where does a declared BOND live on the page?** | **In a `contact` row, beside the chains it names** - `Contact (bond)  A12:SG - B1:C25`. It was a job-level field the page kept in a variable: invisible, and silently invalidated by any edit to those chains, which is what the job NAME was built and removed for - and it matters more here, because a bond that has stopped applying changes the STRUCTURE, not a label. Folded from the row on AF3's own KRAS/sotorasib example, **SG-C25 is 1.62 A against 6.25 with it deleted**, and the archive writes it back as `bondedAtomPairs` in the OPEN dialect (the server's has no field, the same reason a SMILES ligand forces that branch). The atoms are optional: `token_bonds` is token x token, so an atom only decides anything for a ligand or an atomised residue. 🔴 **A CONTACT IS NOT A CHAIN AND `expandEntities` WOULD HAVE MADE IT A LIGAND** - that loop is `if polymer ... else if smiles ... else LIGAND`, so an unknown row type becomes a ligand silently, the same shape as `setChains` deleting a SMILES row. `CHAIN_TYPES`/`isChainEntity` is where that is asked now. 🔴 **AND IT COST TWO OF THAT MISTAKE GOING IN**: the status read "2 chains + 1 ligand" for a protein, a ligand and their bond, and the trunk key said `bonds is not defined` in a function handed chains and never rows. 🔴 **AND ITS BOX FILTERED WHAT WAS TYPED IN, THE THIRD ROW TYPE INTO THE SAME `else`**: the blur handler ends in `cleanSequence`, which keeps only amino-acid letters, so `A12:SG - B1:C25` came back **`A:SGB:C`** on clicking away - the branch that turned benzene into hexane, with a 🔴 comment about exactly that directly above it. Reported by a USER, because the entity list's API cannot see a blur: `--smiles-ui` exists for that reason and a contact had no equivalent. `tools/fold-in-page.py --contact-ui` now drives the dropdown, types and fires the blur, and reads `text-transform` too (borrowing `.entity-value-ligand` would uppercase it on screen). Watched failing at `stored A:SGB:C`. Crosslinks are NOT done - af3x's named linker expands to a ligand plus two bonds and fits this row, but wants a table of linker codes and attachment atoms |
| 🔴 **Where does a reader LOAD a job, and what does a dropped file mean?** | **`Load job JSON…` in the MSA dropdown, or a drop anywhere on the page** - `tools/fold-in-page.py --drop-job`, nine arms. A line of prose under the entity rows was removed for a field and a menu entry. 🔴 **THE MENU ENTRY IS AN ACTION IN A LIST OF STATES AND MUST NEVER STAY SELECTED**: a job is not an MSA mode, so left selected `msaMode()` answers with something nothing maps and the fold runs on it - choosing it opens the picker, `msaMode()` answers with the mode it replaced, and `applyJob` puts that mode back unless the file asked for no alignment, which wins. 🔴 **THE NAME FIELD DELETED MACHINERY RATHER THAN ADDING IT**: the job's name was state nobody could see, so keeping it honest took recording the SHAPE of the rows the job created and re-checking at fold time - `set` and `setChains` both `render()` without notifying, so there was nothing to listen to. A visible box answers it outright and `loadedJob`/`jobShape` are gone; it names the REQUEST only, every file still stemmed `af2_1`. 🔴 **AND PLAIN TEXT MEANS DIFFERENT THINGS AT THE TWO DOORS** - the MSA box is "my alignment", a drop is "what I want to fold", so a dropped FASTA fills the chain rows. Nothing regressed, because the page-wide drop did not exist before this session. 🔴 **BUT AN a3m MUST BE TOLD APART OR THE PAGE STOPS RESPONDING**: `entitiesFromText` makes a row per record, so a 7907-row search becomes 7907 rows - `looksLikeAlignment` asks for an `.a3m` name, lowercase insertion columns, or more than eight records, and the status always says which way it went. Falsified both ways (guard removed: an alignment becomes `protein:10x41`; drop routed as `alignment`: the FASTA stops filling rows). 🔴 **AND ADDING A PASSING ARM TURNED ANOTHER RED FOR THE WRONG REASON** - the structure arm compared against a snapshot from the top of the probe that the new FASTA arm legitimately changed: **an arm's baseline is the state immediately before it**. See docs/WEB.md |
| 🔴 **Does the ARCHIVE describe the job that was handed in?** | **`tools/check-job-archive.py --job=<input>.json --archive=<fold>.zip`, and nine of nine of AlphaFold 3's examples match** - saved with `fold-in-page.py --job-archive=<path>`. 🔴 **IT IS A SEPARATE PROCESS AND NEVER IMPORTS THE PAGE'S READER**: `--job-round-trip` reads the archive back through `web/job-json.js`, the module that WROTE it, and a writer and reader sharing a mistake agree perfectly - which is exactly the SMILES bug docs/SMILES.md records, a benzene written as a CCD code and read back as one. It normalises both sides in Python and compares MEANING, because the comparison is cross-dialect (the examples are open dialect, the archive writes server, except SMILES). Beside the request it holds the save to the fold: pae/contact_probs square at the token count and in range, `token_chain_ids`/`token_res_ids` that long, **`atom_plddts` one per ATOM/HETATM in the PDB** (the one cross-file check), and every ligand and modified-residue code actually in the structure. 🔴 **AND THE TOKEN COUNTS ARE AF3's OWN RULE, COUNTED OUT OF THE SAVED PDB**: one token a polymer residue, one an atom for a ligand or atomised residue - calmodulin 149+4 CA=153, streptavidin 126+16 biotin=142, tetracycline 436+64=500, erk2 360-2+(11 TPO+16 PTR)=385, tetr+DNA 436+40=476. 🔴 **ONE FIELD WAS WRONG: THE JOB'S `name`.** `jobFromJson` read it, `applyJob` threw it away and the archive wrote the fold's STEM, so `calmodulin_4calcium.json` came back as `af3_1` - the same chemistry under an identity its author would not recognise, invisible to any comparison of entity lists. Fixed into the request's `name` only, never the file stem. 🔴 **AND MAKING IT GO STALE FAILED TWICE, OPPOSITE WAYS**: clearing it from `onChange` is a no-op because `set` and `setChains` both `render()` without notifying (and `setChains` is the alignment-query-wins path), so it is a SHAPE COMPARISON now; and the probe that edited a row and re-saved read `goesStale: false` as a bug when `archiveFor` rightly reads the name off the PREDICTION - only a SECOND FOLD can go stale, and it does (`af3_2`). See docs/WEB.md |
| 🔴 **Do AlphaFold 3's own example jobs FOLD, not just load?** | **`tools/fold-in-page.py --job=<path>` - all nine that load, fold.** It drops the file on the page, lets web/job-json.js fill the rows and presses Fold; nothing in the tool sets a sequence, a ligand or a modification, or the run would test the script's reading of the format instead of the page's. 🔴 **`test/af3-example-jobs.test.js` NEVER FOLDED ONE** - it asserts what each file BECOMES - so "nine of fourteen load" was never a claim that nine work, and nothing had asked. AF3 int5, single sequence, 2 passes, seed 42 from each file - 🔴 **at 25 steps and not the 4 asked for**, because `fold-in-page.py` set `#af3-count` before the handlers that rebuild it (fixed since; the `controls:` line is what shows which took): ubiquitin 76 in 2 s, barnase-barstar 199 in 7, u1a+RNA 122 in 6, calmodulin+**CA x4** in 8, streptavidin+**SMILES biotin** in 7, tetr homodimer 436 in 16, +**TAC x2** in 21, erk2 360 +**TPO185/PTR187** in 14, tetr+**DNA** 476 tokens in 18. 🔴 **READ THE STATUS LINE, NOT THE pLDDT** (31-78 here): these are the cheapest settings the page has, so the numbers say nothing about quality - what is evidence is that the line names four calciums off an `id` LIST, both phosphorylated residues, a biotin BUILT from SMILES, and 476 tokens in 4 chains, and that none of the nine appended "NOT A CHAIN". 🔴 **AND THE CORPUS IS COMPLETE**: all thirteen are byte-identical to `sokrypton/alphafold3` **main**'s own `examples/` (re-checked 2026-09-20 against a fresh clone; `~/af3fork` and `~/af3src/alphafold3-af3-any-model` on this box are NON-GIT SNAPSHOTS and are behind). The five refusals are three `bondedAtomPairs` and two modified BASES, and the second is a stated mechanism rather than conservatism - the modified-residue path resolves the parent through the AMINO-ACID table, so a base would fold as a modified amino acid. Type-aware resolution is what takes nine to eleven. See docs/WEB.md |
| Which of AlphaFold 3's own example jobs load here? | `node --test test/af3-example-jobs.test.js` - **10 of 14, and the other 4 name their field**: two modified bases, a five-component glycan in one ligand entry, and an alignment carried inline. 🔴 **`bondedAtomPairs` IS READ NOW AND IT IS NOT COSMETIC** - sotorasib is a COVALENT inhibitor and the KRAS job declares the bond to cysteine 12. Folded through the page, same seed, 50 steps: **SG-C25 is 1.62 A with the bond and 6.25 A with the field deleted** (a C-S bond is ~1.81), so the control is what says the ligand is ATTACHED rather than nearby. An endpoint is addressed by `asymId` - one namespace over polymers AND ligands, which is what the file's chain letters are - and an atom NAME resolves to the TOKEN carrying it, because `token_bonds` is token x token; resolving by name is also what makes an edited row throw rather than bond something else. 🔴 **A POLYMER-TO-POLYMER BOND IS READ AND NOT SENT**: AF3 extracts token bonds only where one side is a LIGAND, so a disulfide is absent from its `token_bonds` too - kept out, and the load SAYS so rather than dropping it in silence. 🔴 THIS ROW SAID 8 OF 14 AND 6 REFUSALS AND WAS STALE - the streptavidin/biotin job moved to the loading side when SMILES landed, which the test asserts as a COUNT so that it is a decision rather than a drift, and only this line drifted. The two gaps are the argument for what to build next |
| 🔴 **Is a hidden py2Dmol element dead, or load-bearing?** | **Read the bundle before deleting it - most of that panel was load-bearing.** index.html carried 138 lines of py2Dmol's website at `display: none` (a fetch row, four examples, an options disclosure, seven checkboxes); 48 remain and 41 of those are the comment saying why seven elements are still there. **Three are read unguarded at startup** (`fetch-btn`, `upload-button`, `file-upload`) and a throw there aborts the rest of `initializeApp` - the symptom is the MSA panel never wiring itself up, not an error. 🔴 **AND `#loadAsFramesCheckbox` IS ON THE FOLD PATH**: `processFiles`, reached by `window.py2dmolLoadFiles`, reads `p.checked` with no guard, so deleting it gives **"Error processing af#_#.pdb: Cannot read properties of null (reading 'checked')"** while the status line still reads "Done in 1.2 s · pLDDT 64.2" - a right answer nobody can see. 🔴 **AND ABSENT IS NOT UNCHECKED**: `!!m && m.checked` makes a deleted box FALSE and `!d || d.checked` makes it TRUE, which is why `alignFramesCheckbox` and `loadMSACheckbox` stay and `loadPAECheckbox` could go. 🔴 **AND `#fetch-id` IS A FLAG**: `isIndexHTML = getElementById("fetch-id") !== null && ...` gates `initializeMSAIndex()`. Deleting it changed nothing measurable on a single-chain fold, so it is kept on a READING and the falsification is missing rather than passed - a two-chain fold is where it would show. See docs/WEB.md |
| 🔴 **Is the site shipping a file no page loads?** | **It was - 620 KB of it.** `py2Dmol.embed.min.js` is for `single.html` and `proteinhunter.html`, both held back at `b0dc258`, and `web/` is copied wholesale, so it published on every deploy to be fetched by nobody. 🔴 **DELETING THE MIRROR IS THE WRONG FIX**: that commit's promise is that putting those pages back is one line, which is why it left every tool that touches them working. `build_site.py` now DERIVES it - a top-level file in `dist/web/vendor/` that no shipped `.html`/`.js`/`.css` names is dropped from `dist/` and the build says `left out py2Dmol.embed.min.js (620 KiB)` - so the day `single.html` returns its bundle returns with it. Verified both ways: 242 files / 23.6 MiB with a stub single.html, 240 / 23.0 without. Same shape as the registry dropping an unservable `<option>` |
| 🔴 **Can a reader DROP an AlphaFold 3 job on the page?** | **`tools/fold-in-page.py --drop-job`** - a real DragEvent on document.body, not the file input, which is the point: `--job-round-trip` sets `input.files` and fires a `change`, so it exercises job-json.js and nothing about who RECEIVES the file. 🔴 **py2Dmol OWNED THE PAGE'S DROP AND READ A JOB AS A BROKEN STRUCTURE FILE.** Measured with our listener disabled, one of DeepMind's own examples gives **"Error processing loose files: No structural files (*.cif, *.pdb, *.ent) found."** and leaves every row as it was - so the page had a reader for that exact file and pointed its one error at a structure problem the file was never supposed to have. The door that did work was the alignment upload box, hidden until the MSA dropdown reads "Upload file". 🔴 **THE FIX IS ONE READER, NOT A BETTER SPLIT.** Claiming `.json` and handing it back when it had no `sequences` was tried and is wrong: every refusal then has to be guessed at twice, because which reader answered depended on how far the other got. This page now takes **all four drag events** in capture on `window` and routes through `readHandedFile`, the same router the upload box uses. It costs the viewer's own `.pdb`/`.cif` drop and py2Dmol's `paeFromJSON` pairing, both of which worked here; the refusal names what the page does take. 🔴 **TAKE ALL FOUR OR THE OVERLAY STICKS**: py2Dmol counts dragenter against dragleave in an unreachable closure and only its `drop` listener hides `#global-drop-overlay`, so claiming just the drop leaves it up for ever - this page drives the overlay itself, off `relatedTarget === null` rather than a tally, because a tally that must balance is what stuck. 🔴 **AND A STRUCTURE MUST BE REFUSED BY NAME, BECAUSE `parseA3m` TAKES IT**: an alignment is "any text that is not JSON" by that branch, so without the guard a dropped PDB returns "A3M sequence data appears before the first FASTA header". 🔴 **AND TWO ARMS FIRST PASSED ON THEIR OWN BUG** - the overlay arm asked "is it showing at the next dragenter", which is `flex` either way because it was still showing from the claimed drag (and the first probe never fired `dragenter` BEFORE the drop, so the counter was never dirty); the hand-back arm inferred from status text, and a wrongly-claimed file says ``no `sequences` in that job``, which it did not recognise. It counts calls into `window.handleFileUpload` now, and the answer is 0. Four falsifications in docs/WEB.md, each watched failing |
| 🔴 **Would a device at WebGPU'S GUARANTEED MINIMUM refuse a kernel?** | **`npm run test:spec-floor`** (`check-portable-limits.mjs --spec-floor`) - and it is a STANDING GATE now that all six pass it - verified to fail with any one of the four fixes below removed, which is what makes it one. It used to be four of six. AlphaFold 3, boltz2 and protenix2 USED TO ask for `workgroup_size(512)` and OpenDDE for 384, where the standard promises **256** invocations and a 256 X extent - so four of six could not create their pipelines on a conforming minimum device. 🔴 **`PORTABLE_CEILINGS` COULD NOT SEE ANY OF IT**: it caps at 1024, which is what an A100 and an M2 report, so it is the weakest machine anyone here has MEASURED and not the weakest a conforming browser may be, and `npm run test:portable` passed all six throughout. 🔴 **AND ALL SIX FOLD THERE NOW, AT NO COST ANYWHERE ELSE.** Four blockers,
every one a performance choice that never asked what the device would run:
`transitionWidth` picked 512 lanes over 256 on a speed knob;
`splitTransitionConfig` and `projectMatrixConfig` chose matrix geometries
without pricing the workgroup STORAGE their shaders stage, where
`resolveGridAttendMatrix` beside them already resolved to false rather than
throwing; and TWO kernels sized their workgroup by the attention's WIDTH - the
grid projection staged one gated element per lane, and the q/k/v/gate projection
used `LANES` as both the workgroup size and a LAYOUT stride, which is why it
looked unclampable. Separating those two meanings was the whole fix: `LANES`
keeps the layout, `WG` is the workgroup, every lane-strided loop strides by WG,
and at 256 or below they are equal and it is the kernel it was.

Nothing moves anywhere it has been measured - both projections are BIT-IDENTICAL
(OpenDDE's block 0.013922382574382865 before and after, AF3's
0.03350964165137652, boltz2's 0.009853641554639692), `test:portable` and
`test:stock` pass all six, 6MRR is 0.72 / 0.507 / 0.715 / 1.545, OpenDDE's trunk
oracle is 7.65e-4 and its 5CAJ self-template 0.279 A, `npm test` 1053/0. At the
floor the answers agree too: AF3 83.1328 against 83.1295, OpenDDE 92.0413
against 92.0396, boltz2 96.4579 against 96.4592. AF2's checksum MOVES there
(-1294937 against -1287025) because 16 KiB of workgroup storage picks a
different tile. 🔴 **THE FIRST TWO OF THOSE PAIRS USED TO READ 83.131/83.128 AND
92.1193/92.1202 AND WERE STALE BY A DAY WHEN THEY WERE WRITTEN.** Bisected: AF3
moved at `7c13e05` (the template precision pin below) and OpenDDE at `8dba05f`
(three models folding a single sequence one MSA row short - its RMSD improved
1.527 -> 1.518), and **both commit messages state the new figure**. boltz2 never
moved. Nothing was wrong except that the numbers lived in prose here and in two
other documents and nothing re-ran them - see docs/A100.md for the bisect and
`tools/gate-baseline.json`, which is where all three arms' signatures live now |
| **Would a WEAKER DEVICE refuse a kernel?** | `npm run test:portable` (`tools/check-portable-limits.mjs`) - it caps this device at `PORTABLE_CEILINGS` and folds all six models. 🔴 **A RAISED LIMIT IS A PREDICTION ABOUT THE NEXT MACHINE**, and one of them is wrong on Apple silicon: `maxComputeWorkgroupStorageSize` is 49152 here and **32768 on Metal**, so a kernel taking a 40 KiB tile compiles here and fails to create its pipeline there, with an error naming a shader rather than a limit. It cannot simulate a LARGER limit, so the M2's 4 GiB binding size is out of reach |
| 🔴 **Has a gate's own figure moved, and did anyone notice?** | **`tools/gate-baseline.json`, and the answer used to be no.** All three whole-model gates record their per-model SIGNATURE there and re-check it on every run; `--write-baseline` re-records. It exists because docs/A100.md's stock table was stale within three days on two of four rows and CLAUDE.md's floor line carried a third value for one of them - four OpenDDE figures across the docs for one measurement - while both causing commits had stated the new number in their own messages. **A figure in prose is re-run when somebody decides to; a figure in a file is re-run by the gate.** 🔴 IT IS KEYED ON THE ADAPTER and a baseline from another box is REPORTED, not enforced, because a signature does not travel between machines. That is also why `gpu-chrome.mjs` prints the adapter now - it always collected `adapter.info` and always threw it away, so every figure here has been machine-anonymous |
| **Does every model fold on the browser a VISITOR has?** | `npm run test:stock` (`tools/check-stock-flags.mjs`) - the four folds with `LOCALFOLD_STOCK_FLAGS=1`, which drops the two developer flags every other gate here passes. 🔴 **TWO OF THE FOUR DID NOT FOLD THAT WAY** and nothing could see it, because the configuration every gate checks is not the one the site ships. It asserts a SIGNATURE - a checksum, an atom checksum or a pLDDT - rather than an absence of errors, since an uncaptured device error leaves the harness reporting success. Needs `DISPLAY=:99 XDG_RUNTIME_DIR=/tmp/xdg` like every GPU lane, and says so when they are missing instead of blaming the models |
| **Does the port fold at all?** | `node tools/fold-esmfold2.js` (6.5 min, writes a PDB) |
| Is a fold from ANY of the four models actually a chain? | they all assert on it now - **`src/af3/chain-geometry.js`** holds the one band. 🔴 **AND IT USED TO LIVE UNDER `tools/`, SO THE PAGE RAN NO GEOMETRY CHECK AT ALL - FOR ANY MODEL.** Every CLI fold gated on it and the one path a visitor takes did not, which is `LOCALFOLD_STOCK_FLAGS`'s shape again: the configuration every gate checks was not the one that ships. Measured - **intellifold2 in Flow returns a fold this rule REFUSES on 1 seed in 6** (CA median 4.255 A against 3.80) with pLDDT 83.30, and the page drew it. `web/app.js` appends "NOT A CHAIN" to the status line now; it WARNS rather than refusing, because a visitor who chose a fast sampler may see what it made, but not as though the confidence number were the whole story. `tools/gpu/chain-geometry.js` keeps only the throw and the `--allow-broken-geometry` advice, which is not advice for a browser, `--allow-broken-geometry` is the escape hatch, and `test/chain-geometry.test.js` gates the rule where the weights are not |
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
| Where do I get any model's WEIGHTS? | **already built and published** - `src/bundles/manifests/index.js` has a `remote:` per family, all under `huggingface.co/sokrypton/localfold/resolve/<sha>/<family>/`. Download those; the export pipelines below are for making a NEW bundle, not for getting an existing one |
| Where do I get ESM-C and ESMFold2, to build one? | `tools/esmc/fetch.py` (3.0 GB, ungated, MIT) |
| Turn ESM-C into a bundle the browser reads | `tools/export_esmc_model.py`, then `tools/quantize_af3.py --bits 3 --group 128` - **that group is ESM-C's alone.** AF3's and OpenDDE's bundles are `--group 32`, which is the default, and OpenDDE at 128 folds 6MRR into a 3283 A explosion at pLDDT 46.69 |
| What should a WebGPU ESM-C agree with? | `oracle-dumps/esmc-59.json`, from `tools/esmc/dump-esmc-oracle.py` |
| Does the ESM-C CPU reference compute ESM-C? | `node tools/check-esmc-reference.js` |
| ...and does the WebGPU block? | `tools/gpu/check-esmc-block.js` |
| ...and the whole 36-block tower, and the shim? | `tools/gpu/check-esmc-tower.js` |
| What does an ESM-C block cost? | `tools/gpu/bench-esmc-tower.js` |
| ...and is the tower right at more than one length? | `check-esmc-tower.js --dump=/oracle-dumps/esmc-{59,128,180}.json` |

| **Why does OpenDDE compile its pairformer twice, and can it stop?** | `tools/gpu/probe-token-specialisation.js` - it warms the stack at both token counts, diffs every WGSL text and sorts by bytes. 28 kernels of 44 differ only in a NUMBER (154 KiB) and 2 differ structurally (the triangle contractions, 934 of 1009 lines). 🔴 It is still not worth removing: the largest constants are `(row % 68u) * 68u + row / 68u` index arithmetic, which a uniform turns into integer division in the inner loop - the 4.3x class - and `override` does not help because the backend compiles once per value anyway. **The specialisation IS the compile cost.** See docs/OPENDDE.md |
| **Do the page and the CLI fold the SAME fold?** | they build the batch through one function now - `af3BatchFromA3m` in src/af3/featurise/batch.js - and the four families agree on 6MRR with a 128-row search: af3 90.9/90.9, openbind0 88.1/88.1, boltz2 96.1/96.1, protenix2 92.5/92.5, opendde 94.6/94.609 - each arm on the SAME bundle as the other, which openbind0 needed two tries to arrange. 🔴 Before it they did not, and the divergence hid a page-only bug for a day. See docs/AF3.md |
| **Does OpenDDE fold WITH AN ALIGNMENT?** | `tools/gpu/fold-opendde.js --a3m=<path>` - new, and until it existed **every OpenDDE number in these docs was a single-sequence fold**, so the page's alignment path through that family had nothing to check against. `tools/gpu/fold.js` still cannot fold OpenDDE (it wants the structural-token expander and AF3's confidence head), which is why both tools exist |
| **Does OpenDDE fold?** | `tools/gpu/fold-opendde.js --target=6mrr` (RMSD **1.501 A, TM 0.934**, pLDDT 92.12 - the old figure here was 1.68 / 0.865 and is stale. 🔴 **The outer product mean computed 256 channels where OpenDDE's pair is 384**, so a third of its MSA pair output was missing until 2026-09-14 - worth THREE ORDERS against the oracle (trunk pair 1.08e-1 -> 7.65e-4) and **0.01 A of structure**: with the bug put back the same command gives 1.492 / 0.9342. A defect can be enormous against a reference and invisible in the fold, and neither number is the other's evidence) - and its bundle wants **`export_af3_model.py --include diffuser`**, because the default is trunk plus distogram head and this tool needs `structural_token_expander`. 🔴 **AND ITS DEFAULT IS 200 STEPS WHERE THE PAGE RUNS 16** - `OPENDDE_COUNTS` prefers 16 and docs/OPENDDE.md shows more steps are WORSE - so a timing taken with the default is 4 s of sampler a user never waits for: `--steps=16` is the page's path (a warm fold 4.09 s -> 1.26) |
| Does its structural-token expansion conserve the atoms? | `tools/gpu/check-opendde-expander.js` |
| Does its CONFIDENCE head still compute its PAE and PDE? | `tools/gpu/check-opendde-confidence.js` - the head's only gate, and a fold's geometry check cannot see a confidence number |
| 🔴 **Which SAMPLER and STEP COUNT keep a molecule's geometry - on ligands and nucleic acids, not just protein?** | `tools/gpu/bench-sampler-geometry.js` - five models x seven systems (protein, +GOL, +ATP, +SEP, RNA, DNA, protein+DNA) x four arms x two seeds, scored on BONDS. 🔴 **THE STEP COUNT MATTERS FAR MORE THAN THE SAMPLER**, and the page's 25 is short: `diffusion200` wins **95 of 105** cells, and at 25 steps rf3's ATP is **0.3201 against 0.0469**, if2's ATP 0.3196 against 0.0476, af3's SEP 0.1260 against 0.0591 - while a plain protein backbone barely moves (af3 0.0416 against 0.0383). A backbone is converged at 25 and a LIGAND is not, and every step count here was chosen on a protein. 🔴 **AND FLOW DOES NOT WIN A BOND CLASS**: diffusion25 against flow16 is 50/17/17, flow is the best arm in 1 cell of 105, and it ties rather than wins on ligands (6-6) and nucleic (4-4) - the 'flow rescues ligands' reading came from af3 and if2 alone and boltz2 goes the other way. 🔴 **AND IT FOUND A DEFECT NO GATE COVERS: boltz2 TEARS APART A PHOSPHOSERINE** - SEP3 N-CA **4.578 A against 1.469** at 200 steps, 0.52-1.22 rms in EVERY arm and worse with more steps, where af3 and if2 are 0.05-0.19. `test:ligand` folds a glycerol, which is not a modified residue, and `test:batch` compares a SEP target's FIELDS rather than folding it. Not investigated. See docs/AF3.md |
| **Can a LIGAND or NUCLEOTIDE be scored at all?** | `componentBonds` in `tools/gpu/bond-geometry.js`, and before it the answer was no: the conformer set is twenty amino acids and an X, so a glycerol's bonds were typed by hand into two files and a base could not be scored by anything. It reads `_chem_comp_bond` and the ideal coordinates from the CCD - **stated bonds with orders, not a distance cutoff**, and the same dictionary the featuriser reads, so a fold is still not scored against itself. Calibrated against the hand-typed glycerol table (1.429 against 1.43, 1.530 against 1.52). A nucleotide is reported in its own `nucleic` class, because calling a guanine a ligand makes an RNA row unreadable |
| **Do a fold's atoms OVERLAP each other - the question bonds cannot answer?** | `tools/gpu/clash-geometry.js` (`clashScore(pdb, conformers)`), MolProbity's 0.4 A threshold in MolProbity's clashscore units, so the number means something outside this repository. 🔴 **NOTHING HERE COULD SEE A NON-BONDED CLASH**: `bond-geometry.js` scores only BONDED pairs by construction, `chain-geometry.js` stops at the backbone, and `web/fold-archive.js` omits AF3's `has_clash` under its own rule that a field we do not compute is left out - so a side chain driven THROUGH its neighbour was invisible to all three. 🔴 **AND ITS FIRST CALIBRATION WAS WRONG, WHICH ONLY THE CRYSTALS CAUGHT**: MolProbity's threshold needs MolProbity's RADII (oxygen 1.40, not Bondi's 1.52) and an exclusion of FOUR bonds, or the trans-peptide O(i)...C(i+1) at 2.78 A - one per residue, in every protein - is counted, and a 1.5 A crystal reads 3.63 where the truth is 0. Both are arms (`radii`, `hops`). Calibrated: 6MRR 0, 5K9P 0, 5CAJ 0.97, 1QYS 5.91, 1BRS 7.33, and a 1976 entry 67.4. 🔴 **THE ANSWER IT WAS BUILT FOR - 'is the page's default sampler squashing side chains' - IS NO.** Five models x five targets at the page's diffusion 25 WITH the alignment the page fetches: af3 3.57, boltz2 4.08, if2 5.27, protenix2 6.12, rf3 9.87, against deposited crystals 0-7.33 and **AlphaFold 3's own SERVER at 2.84**. Step count moves nothing there (af3 4.75/5.22/3.32 at 25/50/200) and nothing under a self-template either. What moves it is whether the fold is DETERMINED: the same targets from a single sequence are 10.9 to 239.7. 🔴 **AND A LOW SCORE IS NOT A GOOD STRUCTURE** - af3's single-sequence 5CAJ at 200 steps reads **0.95, below the crystal**, at 21.5 A RMSD, because a wrong fold that EXPANDS has nothing to overlap (Rg 22.0 -> 24.4 A). Read it beside the RMSD or not at all. 🔴 `bySeparation` IS WHAT NAMES THE FAULT: of 170 clashes over 25 good folds, **2 are local** - nobody is packing a side chain into its own neighbour - and they are enriched **1.67x in the first decile of the chain and 1.27x in the last**, flat on a wrong fold. "The ends are overlapping" is a terminus with nothing to pack against, not a rotamer |
| **Are a PROTEIN's bonds right - mainchain, side chain, peptide and ligand, separately?** | `tools/gpu/bond-geometry.js` (`bondGeometry(pdb, conformers)`), the ideals derived from `tools/oracle/reference-conformers.json` rather than typed in. 🔴 **IT IS THE ONLY INSTRUMENT HERE THAT CAN SEE A BONDED SIDE CHAIN** (`clash-geometry.js` above is the non-bonded half), and it found AlphaFold 3's at 0.339 A rms with **100% of them SHORT** where genuine AF3 is 0.051 and the crystal 0.049. 🔴 **FIXED, AND THE FIGURE THAT SAYS SO LIVED IN A GATE THAT NOBODY RE-RAN**: this port's AlphaFold 3 is **0.041 / 0.059 / 0.044** on 6MRR today against AF3 Server's 0.035 / 0.044 / 0.009 - 1.3x, not 7.8x - and `tools/check-oracle-bonds.js` carried the BEFORE row as a constant and exited 1 on it, printing a defect that had been closed. Re-measured 2026-09-17; the command that produces the row is in the file - while CA-RMSD moved 0.660 -> 0.583 and pLDDT ROSE. `chain-geometry.js` stops at the backbone by design, the ligand gate folds a glycerol which has no side chain, and the denoise oracle was fed noise: four gates, none of which could resolve a 28% contraction. `denseBondGeometry` in the same file scores AF3's dense `[tokens, 24, 3]` grid against the batch's own `ref_pos`, so an oracle dump is scored with no PDB in between. `tools/gpu/bench-sampler-bonds.js` sweeps sigma/steps/seeds over a benchmark set; `tools/check-oracle-bonds.js` scores the AF3 Server archive, which is the only reference here that is neither this port nor af3-any-model. See docs/AF3.md |
| **Is the denoiser's own answer a molecule, at every noise level?** | `tools/gpu/fold.js --bond-trajectory`. 🔴 **READ THE WHOLE COLUMN, NEVER ITS LAST ROW**: at the schedule's final sigma the EDM skip term is 0.99996, so D IS the input and scoring it scores the walk that produced it. On 6MRR the denoiser's side chains never beat 0.293 at ANY sigma while its mainchain reached 0.053, which is what exonerated the sampler and left the head |
| **Are a model's BOND LENGTHS right, not just its fold?** | `tools/gpu/probe-nucleic.js --sequence= --model=` (RMSD cannot see this). 🔴 **"OpenDDE IS 15% SHORT" WAS THIS PROBE, NOT OpenDDE.** It called `featuriseProtein` with nothing but `chainKinds` and **no dialect**, so every nucleic number ever recorded here was AlphaFold 3's featurisation fed to another model's weights - the same fault `fold-opendde.js` had. On a nucleic chain it matters more than anywhere: `dropTerminalAtoms` removes the 5' OP3, and the flat atom axis is built from the LIVE atoms, so getting it wrong shifts every index in the chain. With the bundle's own conventions: **af3 0.980, opendde 0.997, intellifold2 0.989, rosettafold3 0.997** - opendde is 0.3% short and the best of the four. 🔴 **AND ITS DEFAULT SAMPLER IS `flow` WHERE `fold.js`'s IS `diffusion`**, which is how rf3 read 3.271 here and looked broken on DNA. Its PROTEIN scores 2.976 in the same probe and 1.004 at `--mode=diffusion`: it was the sampler, not the modality. See `noFlowSampler` |
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

## 🔴 IF YOU ARE THE M2 (OR A PHONE): WHAT TO CHECK ON THIS BRANCH

Written from the A100 side. The first two rounds already happened and found
real bugs there - the matrix attention crashing on 8x8 units, the derivations
overriding Apple's measured defaults, and then the 160-residue race below.

🔴 **THE RACE IS FOUND, FIXED AND GATED; SEE BELOW.** It was a missing bounds
check in `ADD_IN_PLACE_SHADER` under a folded grid, not a missing barrier. This
section is the record of the hunt and the instruments it produced, not an open
thread.

### 🔴 OPENDDE'S ORACLES, RUN ON THE M2: THE CONFIDENCE HEAD WAS PRECISION AFTER ALL

All four references were generated HERE from `~/af3_ported/opendde.bin.zst` and
the af3-any-model checkout, with the float32 bundle exported locally (481
tensors, 655.8 M parameters - the same counts as yours).

**The trunk reproduces your finding to three figures**, on independent hardware
and an independently generated reference: target_feat 5.63e-8, z_after_template
**2.07e-2**, z_after_msa 5.91e-2, trunk_out_pair 1.08e-1. The template stage is
wrong. Not fixed here.

🔴 **BUT THE CONFIDENCE HEAD WAS NOT "NOT PRECISION".** `openddeConfidence` built
its four-block stack with `pairWeightPrecision` alone, where
`Af3ConfidenceHeadGpu` pins f32 staging, weights and accumulation AND
`pairMatrixKernels: false` - measured there as all four outputs failing without.
OpenDDE's head ran at the trunk's defaults, on the page, through `fold.js`.

| confidence vs af3-any-model | pLDDT | PAE | PDE |
|---|---:|---:|---:|
| M2, shipped | 4.57e-4 | 1.04e-2 | 1.43e-2 |
| M2, `--f16=off` | 1.18e-7 | 9.11e-7 | 1.20e-6 |
| M2, pinned, f16 ON | **1.18e-7** | **9.11e-7** | **1.20e-6** |
| A100, shipped, either flag | 1.42e-4 | 4.68e-3 | 7.50e-3 |

Your "identical with --f16=off" was true and the inference from it was not:
`--f16=off` cannot reach the matrix pair kernels, so on a device with matrix
units it changes nothing, while the M2 - which has none at these widths - lost
the whole residual to the same flag. Fixed by giving the stack the four pins;
`test/confidence-precision-pins.test.js` holds both heads to them. The fold's
pLDDT on 6MRR moves 92.0506 -> 92.0500. **Worth re-running the oracle on the
A100**, where the matrix kernels were the part the flag could not see.

🔴 **AND TWO OF THE THREE GATES CANNOT RUN AS WRITTEN ON A SECOND MACHINE:**
- `fold-opendde.js --dump=` completes the trunk comparison and then dies in
  `structuralBatch`: `batchFromDump` carries no `entityId`, the first per-token
  field the structural re-tokenisation takes. Possibly not the last.
- `check-opendde-encoder-oracle.js` featurises from `--sequence` and has no
  `--dump=`, so it runs our conformer set - 576 atoms - against the reference's
  dense 68 x 24 = 1632, and every stage past the embeddings is a LENGTH
  mismatch at exactly 1632/576 = 2.833x. Its 5.21e-1 on the summed embeddings is
  the conformer floor your commit describes, not the encoder.

### 🔴 ROUND FIVE: TWO NEW MODELS, AND THE THREE QUESTIONS ALREADY ANSWERED HERE

**boltz2 and protenix2 are ported and exact**, so the panel is six models rather
than four. The whole AF3-lineage side is now held to af3-any-model's own
numbers rather than to this port's CPU - see docs/AF3.md, which is where the
nine defects that found are written up.

🔴 **THREE OF THE FOUR THINGS AN M2 WOULD HAVE BEEN ASKED ARE ALREADY MEASURED
HERE**, and none of them needs an Apple part after all:

| | how it was asked from this machine | answer |
|---|---|---|
| do the new models fold on a STOCK browser? | `npm run test:stock`, which now covers all six | all six fold |
| would a smaller LIMIT refuse a kernel? | `npm run test:portable` - a new gate that CAPS this device at the portable ceiling | all six fold |
| does f16 cost the new models anything? | `--f16=off` against the default, whole fold | **nothing**: boltz2 0.542 A both ways, protenix2 1.722/1.723 |

`test:portable` is the one worth knowing about. This port ASKS its adapter for
the most it will give on five limits, and one of them differs by a factor this
card hides: `maxComputeWorkgroupStorageSize` is **49152 here and 32768 on
Metal**. A kernel taking a 40 KiB tile compiles here, measures well here, and
fails to create its pipeline there with an error naming a shader rather than a
limit. `LOCALFOLD_PORTABLE_LIMITS=1` caps the request at `PORTABLE_CEILINGS`
(src/runtime/device.js), so that refusal happens HERE. It is the limits twin of
`LOCALFOLD_STOCK_FLAGS`, which asks the same question about features.

🔴 **IT CANNOT SIMULATE A LARGER LIMIT, AND ONE OF YOURS IS LARGER.** The M2's
`maxStorageBufferBindingSize` is 4 GiB against this card's 2, so your binding
ceiling is LOOSER and nothing here can ask about it. Round three has that half.

**What is actually left for an Apple part, in order:**

1. **Metal CLAMPS an out-of-range write where Vulkan discards, and this session
   wrote new shaders.** Two whole kernels (`reembedProject`, `reembedPair` in
   confidence-webgpu.js), a `build-queries-pre` pass, and up-gate branches in
   three more. `test/folded-grid-guard.test.js` passes and is structural, so the
   guards are THERE - but the guard test cannot see a clamp, and
   `probe-grid-overdispatch.js` is the only thing that can. Run it; it should
   still say CLAMPS.

2. **MEMORY, which is the one where a laptop and a 40 GB card genuinely
   differ - and the heaviest model is IntelliFold-2, not boltz2.** The whole
   panel measured in one sitting on the A100 is in docs/A100.md; the peaks are
   **AF2 387 MiB, AF2-multimer 443, AlphaFold 3 954, RoseTTAFold3 957,
   OpenBind-0 954, protenix2 1181, boltz2 1360, ESMFold2 1512, OpenDDE 1810,
   IntelliFold-2 2229** at 68 tokens, and 1291 / 1593 / 1716 / 2976 for af3,
   protenix2, boltz2 and if2 at 255. 🔴 **THE PEAK IS RESIDENT DECODED WEIGHTS
   AND ALMOST NOTHING A USER CHOOSES**: on af3 the int5 bundle and the float32
   bundle both peak at **954.1 MiB**, 25/50/100/200 diffusion steps are all
   992.6, `--trajectory=off` changes nothing, and an 8076-row alignment makes it
   LOWER. So int5 is a download saving and a cold-fold saving and is worth ZERO
   in device memory - size a device off the decoded width, never off the bundle.
   🔴 **AND THE OLD 1535 FOR boltz2 IS NOT THIS BOX'S**: re-measured at the
   commit that recorded it and again at HEAD, identical at both, it is 1399.1
   with `difftx.zerogate.resident` at 162 MiB where that row names "204 of MSA
   scratch" - a row no arm here produces. All of them fold unchanged under
   `--budget=800`, but this file's own warning is that Metal takes buffers well
   past the point where macOS starts paging, and `keepResidentAffordable`
   returns true for a device with no budget. `fold.js --budget=0` prints the
   peak.

3. **The transition width, which is a DEVICE-WIDTH rule and yours is narrower.**
   The diffusion conditioning's single transition dispatches `tokens` workgroups
   and nothing else, so at 68 tokens it ran 68 groups of 128 threads - 44% of
   boltz2's whole denoiser call on this card. It now takes
   `transitionThreadTarget` and widens to 512 where the intermediate divides by
   it: boltz2's denoiser GPU 21.4 -> 14.2 ms, its 200-step diffusion 4.7 -> 3.2
   s, AF3 and protenix2 unmoved. **On a part with a hundredth of the width that
   trade may invert**, and `transitionThreadTarget` is ampere-prior only, so an
   M2 should see NO change from it - which is the thing to confirm rather than
   assume.

4. **The two new models are not in the registry**, so the PAGE cannot load them
   whatever a fold tool says. docs/HOSTING.md has that, with the three published
   bundles this session made stale.

🔴 **AND ONE TRAP THAT COST 7x AND WOULD HAVE COST IT THERE TOO.** A bound
weight field is a THUNK that decodes when read, so `block.ffwAToB != null` -
a presence test choosing a shader variant - unpacked a 768x1536 int5 tensor once
per block per sampler step. boltz2's fold was **38.5 s** and is 3.7. The GPU was
92% idle and the arithmetic was never the problem. Ask `block[SOURCES]`, never
the value; it is CLAUDE.md's existing note about `blockWeightOffsets` reading
`.length`, one convention later.

### 🔴 ROUND FOUR, ANSWERED FROM THE M2: NO VISITOR HAS THE MATRIX UNITS

**The flags question first, because it is the one that changes something.**
Measured with `LOCALFOLD_STOCK_FLAGS=1`, which on macOS drops exactly one flag -
`--enable-unsafe-webgpu`, since `PLATFORM_FLAGS` is empty off Linux:

| on an M2, stock Chrome | |
|---|---|
| `shader-f16` | **yes** - the README's claim, now measured rather than inferred |
| `chromium-experimental-subgroup-matrix` | **NO** |
| `subgroups` | yes |
| kernel chosen | `attention:flash-registers-32-chunk16`, flags or not |

🔴 **SO THE MATRIX UNITS NEED `--enable-unsafe-webgpu` ON BOTH PLATFORMS, AND
NOBODY VISITING THE PAGE HAS THEM.** That is the consequence round four asked
for and it is the larger one: every matrix-unit knob this repository has measured
- `attentionMatrix`, `matrixLinear`, `opmMatrixContract`, `triangleProjectMatrix`,
`gridAttendMatrix`, the staged matrix GEMM family - is a DEVELOPER-FLAG path. The
A100's 218 ms prior split, of which "every significant one is does this device
have matrix units", is a number no visitor can reach. It does not make any of it
wrong; it makes it a measurement of a configuration the site does not ship, and
the honest place for those numbers is beside a note saying so.

It costs an Apple part nothing either way: 8x8x8 units are refused by a kernel
declaring `<f16, 16, 16>`, so the register kernel is chosen with the flag and
without it.

**And the rest of round four is inert here, confirmed.** All four new knobs are
`null` in `DEFAULT_TUNING` and set in the ampere prior only, and the call sites
take the constant through `?? undefined` and `?? CONSTANT`. Measured on this tip
against this box's own previous values - never against the A100's:

| | before | round four |
|---|---|---|
| `fold-af2.js` | -1282976 | **-1282976** |
| peak device bytes | 405,716,280 | **405,716,280** |
| AF3 | 85.83089054918456 | **85.83089054918456** |
| OpenDDE | 92.05056924853176 | **92.05056924853176** |

`npm test` 1014/0.

**Not run here: the `opmBlockI` crossover.** It wants a trunk profile at 400
tokens, and the question - whether this part's memory system turns over where
that card's does - deserves its own round rather than a number taken beside five
other folds on a machine that drifts 3.2x.

### 🔴 ROUND FOUR: WHAT WANTS AN M2 NOW

Thirteen commits since round three, three of them touching `src/`. All three are
performance settings found the same way - by asking which constants and knobs
were fitted at a length nobody folds - and all three are **ampere-prior only**,
so an Apple part should see NO change from any of them. That is the first thing
to check and it is one command: `fold.js --model=` and `fold-af2.js` checksums
against your own previous run, not against this box's.

**1. `opmBlockITokens`, and this is the real ask.** Past 256 tokens AF3's outer
product mean takes a block of ONE instead of two, worth **8.1% of a trunk pass**
at 400 tokens here - `opm.contract` 245.7 -> 147.8 ms, the second-largest kernel
in the trunk. The full curve is in docs/AF3.md and it is not monotone: one wins
at 59, two wins at 150 and 256, one wins from 300 up, at both 512 and 1024 MSA
rows. 🔴 **I shipped that as a DERIVATION for every device first, and that was
wrong** - 256 tokens is where THIS card's memory system turns over, not
something a device reports - so it is a prior now and your part keeps the block
of two. What would be worth knowing is whether the crossover exists there at
all: `bench-trunk.js --profile --model= --tokens=400 --msa=512
--tune-json={"opmBlockI":1}` against the default.

**2. `transitionChunkBytes` and `pairLogitsCacheBytes`, both 256 MiB, both
ampere.** The first is 5.4% of an AF2 block and 4.0% of a fold for 168 MiB; the
second is 8.7% of an AF3 denoiser call for 166 MiB. Both bit-identical here.
They are memory trades, which is exactly the axis a laptop cares about and this
card does not, so the interesting question is whether they are worth taking on
Metal at all - and `fold.js --budget=0` is what says.

**3. The flags, which may be the most important thing in this round and are not
a code change.** Every number this repository has ever published from the A100
was taken behind `--enable-dawn-features=vulkan_enable_f16_on_nvidia` and
`--enable-unsafe-webgpu`, and a stock Chrome on NVIDIA has NEITHER: the same
fold is 10767 ms with them and **21000 without**. `LOCALFOLD_STOCK_FLAGS=1`
drops both. Two things only an Apple part can answer:
   - does a stock Chrome there report `shader-f16`? The README now tells Linux
     users to pass a flag and says Apple needs none, on the strength of your
     round-three reply rather than a measurement.
   - does `chromium-experimental-subgroup-matrix` appear without
     `--enable-unsafe-webgpu`? If it needs the flag there too, then the matrix
     kernel is unreachable for every visitor on every platform, which changes
     what `attentionMatrix` is worth rather than how fast it is.

**4. And the Dawn lane can run its f16 arms after all**, which this file said it
could not. `create([])` in all 25 `test/*.gpu.test.js` call sites is why:
`create(["enable-dawn-features=vulkan_enable_f16_on_nvidia,allow_unsafe_apis"])`
gives f16 AND the matrix units on the node binding here. Whether Metal's Dawn
wants a different toggle is open.

### 🔴 ROUND THREE: WHAT WANTED AN M2, AND WAS ANSWERED

Newest first. Round two's ask - the windowing - is the section after this one
and still stands; nothing below replaces it.

**The whole `src/` surface of `main..a100` is four files**, and three of them are
this round: `attention-matrix.js`, `attention.js` and one knob in
`device-profile.js`. The fourth is round two's `execution.js`. `main` is an
ancestor, so the merge is a fast-forward.

**1. `attentionMatrixPrefetch`, and the honest position is that it has never run
off this card.** It reads the flash kernel's key tile into registers before the
barrier and moves the barrier between the read and the write. On the A100 both
arms are bit-identical (`bench-af2-warm` -121844157 either way) and it is worth
**0.05%**, so it ships OFF and the ampere prior does not set it.

On an M2 `supportsAttentionMatrix` should refuse this kernel outright - it
declares `<f16, 16, 16>` and Metal supports 8x8 only - so the knob should be
**inert**, and that is the thing to confirm rather than assume:

```
node tools/gpu-chrome.mjs tools/gpu/probe-kernel.js
node tools/gpu-chrome.mjs tools/gpu/fold-af2.js --tune=attentionMatrixPrefetch=true
```

The fold's checksum must not move **against that same box's other arm**. 🔴 NOT
against -1287025, which is what this asked for and was wrong: the M2 folds that
gate at **-1282976** because the two boxes resolve different attention kernels,
and on the first run the comparison would have read as a failure. A checksum
travels between machines no better than a timing does. 🔴 **IF `probe-kernel.js` DOES pick the matrix variant** - a
newer Dawn, a newer part - then this arm is live on the backend that CLAMPS
rather than discards, and it wants `bench-af2-warm.js --passes=8` on both arms
before anyone trusts it.

**2. Why that kernel is the interesting one on your machine specifically.** Its
staging loop is `for (var i = local; i < KEYS * HD4; i += LANES)`, and an AF2
fold compiles it at heads of 8, 16 and 32 - so the count is 64, 128 and 256
against 128 lanes and at a head of EIGHT only half the workgroup enters. That
non-uniformity is what made a bisection's "pure reordering" write out of bounds
and read as a race in the shipped kernel for a whole campaign. It never was
one; docs/A100.md has the correction and the reproduction both ways.
`test/uniform-barrier.test.js` gates the barrier half of it and runs in
`npm test`, so it runs there without a GPU. The unroll half is not gated and
cannot easily be - keep the guard.

**3. Nothing else this round touches a kernel.** The binding-ceiling work is a
probe and documentation: `tools/gpu/probe-binding-ceiling.js` now reports how
many labels share the lowest ceiling (28 on the monomer, **42 on the multimer**)
rather than naming one, because windowing any subset of a tie moves the fold by
zero residues. If you run it, run it at two lengths that both sit ABOVE every
chunk threshold - `--lengths=200,400`, not 59,118 - or a handled label reads as
a ceiling; `caveat` in the output says when that is happening.

**4. Worth a number from your side if it is cheap:** `maxStorageBufferBindingSize`
and `maxBufferSize` on the M2. Every ceiling in docs/AF2.md is 2 GiB and 4 GiB,
which is this card, and the windowing threshold is derived from the first - a
smaller one would start windowing at a length the A100 never reaches.

🔴 **ANSWERED, AND IT RUNS THE OTHER WAY: 4 GiB, TWICE THIS CARD'S.** See the
M2's reply below. The consequence it does not draw is the useful one: on that
part the binding wall and the ALLOCATION wall are the same 4 GiB, so both land
on 2,896 together and **there is nothing to window on Apple silicon at all** -
the 28-label tie sits exactly on the wall no windowing can pass. `addInPlace`'s
windowed path is therefore unreachable there below the length where the pair
cannot be allocated either, which answers round two's first ask by making it
moot rather than by testing it. 2,047 is this A100's number; the port's is
`sqrt(maxStorageBufferBindingSize / (cZ * 4))`.

### 🔴 ROUND THREE, ANSWERED FROM THE M2

🔴 **AND FIRST: `dispatchDigest` IS NOT REPRODUCIBLE ON THIS MACHINE, WHICH
BREAKS `audit-knobs.py` HERE.** Five runs of `probe-compiles.js --tool=fold-af2`
on one branch with one input give `shaderDigest` **a5cc4748 every time** and
`dispatchDigest` **three different values** - 9a4f25f8 three times, then 5512fefe,
then 8893e753 - with `dispatchShapes` constant at 102, so it is a COUNT that
moves and not the set of shapes. Dumping the map and diffing the runs names it
outright: the only key that differs is **`occupancy.chain|4x1x1`, at 3, 2 and 1**.
The occupancy measurement's dispatch count is timing-dependent, so it lands in
the digest of anything that wraps a fold.

That matters because `audit-knobs.py` decides whether a knob "moved" anything by
comparing these two digests, and this file calls it **the main ask** for an M2.
Here it would report a random half of the knobs as having moved a dispatch when
nothing moved at all. The shader digest is sound; only the dispatch one is
affected. Until it is fixed, read the FAILED column and the shader digest, and
treat a dispatch difference as a question rather than an answer.

It cost a wrong conclusion on the way to finding it: two branches were compared
on this digest, read as "identical, therefore nothing changed", and the same
comparison run again disagreed with itself. **A digest that is not reproducible
is not a comparison** - the fix for that reading is below, and it is the fold
checksums, which are stable.


All four, on `670453c`. `npm test` 1014/0 here, `uniform-barrier.test.js`
included.

**1. `attentionMatrixPrefetch` is inert, confirmed rather than assumed.**
`probe-kernel.js` picks **`attention:flash-registers-32-chunk16`** - not a
matrix variant - and `probe-subgroup-matrix.js` says why: this device offers
**8x8x8 and nothing else**, `{f32, 8, 8, 8}` and `{f16, 8, 8, 8}`, against a
kernel that declares `<f16, 16, 16>`. So `supportsAttentionMatrix` refuses it
structurally and the knob cannot reach a kernel this device runs.
`fold-af2.js` with and without `--tune=attentionMatrixPrefetch=true` is
**-1282976 both ways**, pLDDT 62.644, geometry ok. Your prediction holds; the
`--passes=8` contingency is not needed.

🔴 **AND -1287025 IS NOT THIS MACHINE'S NUMBER.** The M2 folds
`fold-af2.js` at **-1282976**. Neither is wrong: the two boxes resolve different
attention kernels, so a checksum travels no better between them than a timing
does. Compare an arm against the SAME box's other arm, never against the other
box's baseline.

**2. The binding ceiling is 4 GiB here, not 2 - so the answer runs OPPOSITE to
the worry.** `probe-limits.js` on the M2:

| | M2 | A100 |
|---|---:|---:|
| `maxStorageBufferBindingSize` | **4 GiB** | 2 GiB |
| `maxBufferSize` | 4 GiB | 4 GiB |
| `maxLengthByBindingLimit` | **2896** | 2047 |

So the windowing threshold derived from the binding size starts LATER on an M2,
not earlier, and 2,047 is this A100's ceiling rather than the port's. Every
ceiling in docs/AF2.md wants reading as one card's. The ratio is exactly the
square root of two, which is what a pair tensor quadratic in length does with
twice the binding.

**3. The ceiling probe was not run here**, since its subject is a limit this box
does not share; the numbers above are the input it would want anyway.

### 🔴 ROUND TWO: WHAT WAS NEW AND WANTED AN M2 BEFORE IT MERGED

`addInPlace` - the same shader - now **windows** its bindings. It bound both
tensors whole, and the pair is `L*L*channels*4` against
`maxStorageBufferBindingSize`: 2 GiB on the A100, unraisable because it is
Vulkan's `maxStorageBufferRange`, so the ceiling was **2,047 residues** whatever
memory the card had. Measured there: 2,047 bound, 2,048 was REFUSED outright,
2,896 folds in two windows now. Below the ceiling the single-dispatch path is
untouched and every A100 checksum is unchanged. Found by @milot-mirdita upstream
- their packed-f16 pair put their ceiling at 2,896 where our f32 put ours at
2,047.

Three things worth an Apple part specifically:

1. **That a normal fold still takes the single-dispatch path there.** The
   threshold is `maxStorageBufferBindingSize / 4` rounded down to 64 elements,
   and Metal may report a different limit - a lower one would start windowing at
   a length the A100 never does, which is a behaviour change nobody has seen.
   Any AF2 checksum moving at an ordinary length means that happened.
2. **The guard and the windowing together, on the backend that CLAMPS.** A
   windowed dispatch's last workgroup runs past its window, and on Metal an
   unguarded one would clamp onto the window's last element rather than the
   buffer's - the same corruption as the race, at a new boundary. The guard
   reads `arrayLength(&base)`, which is the WINDOW's length, so it should hold;
   the A100 cannot check that, because it discards instead of clamping.
3. **`tools/gpu/probe-grid-overdispatch.js`**, whose `backend` arm strips the
   guard and is the only thing that still sees what a GPU does with an
   out-of-range write. On the A100 it says `discards`. On an M2 it should say
   **CLAMPS** - and if it does not, this port's whole account of the race is
   wrong.

🔴 **WHY AN APPLE PART IS THE INTERESTING ONE.** The capability layer and the
five derivations sit UNDER the priors, and `metal-3`'s prior sets three knobs.
On the A100, whose prior sets thirty, almost nothing they decide is ever used and
every number recorded for them was taken with `--no-prior`. An Apple part is,
for most knobs, exactly the unrecognised device these layers were built for -
and `DEFAULTS_ARE_MEASUREMENTS` now yields to that, so what runs there is a
different code path from anything measured here.

### 🔴 THE RACE, AS IT WAS HUNTED - AND THE A100 SIDE

**Answered below; this is the record of the hunt, kept because the dead ends
are most of its value.** The two sections that follow are what each machine
ran; the third is what it turned out to be.

You asked for one command - `bench-af2-warm.js --length=160 --rows=128` here -
because it halves the search space. It does. **This box does not reproduce it**:
ok at 59, 80, 100, 128, 160, 200, 400 and 825, then the three that fail on your
box pressed with `--passes=8` (28 pairs a run to disagree on rather than 3) for
four rounds each - twelve runs, all agreeing, and the checksums identical ACROSS
runs too: 160 is **-9869845** every time, 200 **-704434**, 400 **-15124842**.
Note the standing gate has always defaulted to **825** and passes, so the
longest case has been deterministic here throughout.

That does not clear the kernels - a race can be latent and need a scheduler to
expose it - but the mechanism involves Metal or Dawn on your side.

Three things to run, cheapest first. **The first one costs nothing: you already
have the output.**

1. **READ THE PATTERN IN THE ERROR YOU ALREADY HAVE.** `bench-af2-warm` prints
   every checksum and `new Set(checksums).size`, and the shape separates your two
   candidates outright:
   - `2 different structures: A, B, B` - pass 1 differs, the rest agree. That is
     a FIRST-TOUCH difference and not a race: pass 1 reads WebGPU's
     zero-initialised buffers and passes 2+ read recycled ones. Deterministic,
     and it points straight at a kernel reading a region it did not write.
   - `3 different structures: A, B, C` - genuine nondeterminism, a scheduling
     question.
2. **`--passes=8`**, which sharpens the same distinction: an uninitialised read
   stays at two values however many passes you add, a race keeps producing new
   ones.
3. **`bench-af2-warm.js --no-pool`**, which is your buffer-pooling hypothesis in
   one run - I built it for this. A pooled buffer keeps the previous fold's
   bytes, so a kernel reading what it did not write differs between fresh and
   reused. On the A100 at 160 the checksum is **-9869845 either way**, so
   recycling changes nothing here. 🔴 IT RETIRES RATHER THAN DESTROYS AND LEAKS
   BY DESIGN - destroying on release killed every length with `[Buffer]
   destroyed`, because a released buffer is still named by an in-flight command
   buffer, which is WHY the pool exists. Fine at 59-160 on 40 GB, `Error.cpp:119`
   at 200. **Use short lengths on an M2.**

If `--no-pool` stops the race, it is ours after all - a kernel reading
uninitialised memory, invisible here only because Vulkan's recycled bytes happen
to land the same way - and the next step is finding which kernel, not which
scheduler. If it still races with a fresh buffer every time, the pool is
exonerated on both boxes.

🔴 **AND WHAT I DID NOT RULE OUT.** Only that the A100 does not reproduce it.
Your seven ruled-out hypotheses stand; I added nothing to them. In particular I
did NOT audit the kernels for reads past a written region, which is what
`A, B, B` would point at, and I did not test at 160+ on a device with a budget.

### 🔴 THE M2 SIDE OF THE RACE: ALL THREE STEPS RUN, AND IT IS GENUINE

Answering the three above, on the M2, at 160 residues and 128 rows.

**1. The pattern is `A, B, C` at every length, never `A, B, B`** - 160 gives
-10391520, -10133100, -10645643; 200 gives 4654492, 4621956, 4696004; 400 gives
8551836, 8645466, 8441713. So it is not a first-touch difference.

**2. `--passes=8` returns EIGHT distinct structures** - -7646158, -9311372,
-7692662, -8716331, -8139621, -8399149, -6607298, -8748844. An uninitialised
read stays at two values however many passes are added; this keeps making new
ones. It is nondeterminism.

**3. `--no-pool` STILL RACES** - -7210508, -7612777, -7725825 with a fresh
buffer every time. The pool is exonerated on both boxes, and the M2's own
buffer-pooling hypothesis is dead.

🔴 **AND IT IS UPSTREAM OF THE STRUCTURE MODULE.** `meanPlddt` itself varies
run to run - 32.358, 32.359, 32.357 at 200 residues - and pLDDT comes off the
trunk heads rather than from geometry. So the race is in the EVOFORMER TRUNK,
not in the folding of coordinates.

The signature is now: genuine nondeterminism, in the trunk, at 160 residues and
up, on Metal and not on Vulkan, with buffer reuse ruled out on both machines.
That fits a LATENT INTRA-KERNEL race - a missing `workgroupBarrier` that one
scheduler hides and another exposes - and not anything the tuning selects: six
knobs that switch the OPM, triangle, attention and linear kernels
(`opmMatrixOutput`, `opmMatrixContract`, `triangleProjectMatrix`,
`pairTransitionSplit`, `attentionGroup`, `linearTallTile`) all still race, as do
`--f16=off`, `batchComputePasses=false`, `--no-resident` and `--extra=0`.

### 🔴 FOUND, AND IT WAS NOT A BARRIER. IT WAS A MISSING BOUNDS CHECK

`ADD_IN_PLACE_SHADER` in src/runtime/execution.js, the only compute entry point
in the repository that indexed a FOLDED grid with no guard and a
read-modify-write:

```wgsl
let index = id.x + id.y * GRID_WIDTH * 64u;
base[index] += update[index];        // no bounds check
```

`linearGrid` rounds TWICE - elements up to a whole workgroup, then workgroups up
to a whole row of GRID_WIDTH - so once the y fold engages the dispatch is a
MULTIPLE of 32,768 workgroups and almost never the count that was wanted. The
pair tensor is `L * L * 128` elements, i.e. `2 * L * L` workgroups, which crosses
32,768 at **exactly L = 128**:

| L | elements | groups wanted | dispatched | excess invocations |
|---:|---:|---:|---:|---:|
| 59 | 445,568 | 6,962 | 6,962 | **0** |
| 100 | 1,280,000 | 20,000 | 20,000 | **0** |
| 128 | 2,097,152 | 32,768 | 32,768 | **0** |
| 129 | 2,130,048 | 33,282 | 65,536 | 2,064,256 |
| 160 | 3,276,800 | 51,200 | 65,536 | 917,504 |
| 200 | 5,120,000 | 80,000 | 98,304 | 1,171,456 |
| 400 | 20,480,000 | 320,000 | 327,680 | 491,520 |
| 825 | 87,120,000 | 1,361,250 | 1,376,256 | 960,384 |

**Every length that raced has an excess and every length that passed has none.**

🔴 **BOTH SIDES ARE NOW MEASURED IN ISOLATION, NOT ARGUED.**
`probe-grid-overdispatch.js` drives the add-in-place kernel over a deliberately
over-dispatched grid and reads the last in-range element, which should be exactly
1 after one `+=` each. At the 160-residue shape, 917,504 invocations past the end:

| | tail across three runs |
|---|---|
| A100, Dawn over Vulkan | 1, 1, 1 - **discards** |
| M2, Metal, guard stripped | **876, 326, 948** - clamps, and unstable |
| M2, the shipped kernel | 1, 1, 1 - the guard holds |

The M2 row is the bug on its own, with no fold around it: a different value every
run because hundreds of thousands of non-atomic read-modify-writes land on one
address. It is far below 917,504 because most of them read the same stale value,
which is what a lost update looks like.

🔴 **AND THE PROBE HAD TO BE FIXED TO SAY THAT.** Written against the unguarded
kernel, it imported the shipped text - so once the bounds check landed it
reported "discards out-of-range writes" ON THE M2, the machine whose clamping
caused the race. It was measuring the GUARD and calling it the backend. It runs
two arms now: the shipped kernel, whose tail must be 1 everywhere, and the same
text with the guard line stripped, which is the only arm that can still see the
GPU. The strip is asserted, because a `replace` that matches nothing returns the
string unchanged and would quietly make both arms the same arm.

🔴 **AND THE TWO MACHINES DISAGREEING IS THE SPECIFICATION, NOT A SCHEDULER.**
WGSL leaves an out-of-bounds write either discarded or clamped into the buffer,
and backends pick differently: Vulkan discards through `robustBufferAccess`,
Metal - which has no hardware robustness - takes an explicit index clamp. Clamped,
those 917,504 invocations all execute `base[last] += update[last]` on ONE address,
non-atomically: a random multiple of `update[last]` between 1x and 917,504x, new
on every pass. Discarded, nothing happens. So the A100 agreeing with itself twelve
times proved nothing about the kernels, and no amount of pressing it would have.
**A backend difference is EVIDENCE OF THIS BUG CLASS, not evidence against ours.**

It lands upstream of the trunk: src/af2/model/monomer.js applies the template
residual through `addInPlace` once per recycle, unconditionally, on
`pairWithoutTemplates`. 🔴 AND BELOW 32 MSA ROWS IT IS 48 TIMES A RECYCLE
INSTEAD OF ONCE - the outer product mean writes straight into the pair tensor and
skips its own `addInPlace` only while `sequences >= cOuter`, so a SHALLOW
alignment puts the same unguarded read-modify-write in every block. Counted on
the device: 1 call at 128 rows, 49 at 16. See docs/AF2.md. One corrupted cell - `pair[L-1][L-1][127]` - reaches
everything within two blocks, because the triangle multiplications mix every
`(i, j)` through every `k`. That is why `meanPlddt` ITSELF varied. Two more call
sites on the same tensor: src/af2/multimer/model.js (templates only) and
src/af2/model/query-only.js.

Fixed with `if (index >= arrayLength(&base)) { return; }` - exact, because
`dispatch` binds the tensor's own range and not the whole buffer.
src/runtime/elementwise.js had the same hole and is harmless only because every
excess invocation stores the SAME value, which a `+=` does not.

🔴 **AND THE 2,097,152 WAS NOT A COINCIDENCE AFTER ALL.** This file said it
was: "every folding-grid dispatch was audited and no compute entry point reads
`.x` without a y term". True, and the wrong question. The audit asked whether the
shader reads `id.y` - this one does - and never asked whether it GUARDS the
folded index. The `var<workgroup>` scan that replaced it was nine false
positives: all nine AF2-path kernels stage, barrier, read and barrier correctly,
including the WAR barrier at the top of each staging loop.

`test/folded-grid-guard.test.js` is the gate, and it is structural rather than
numeric: every folded index must be compared against a bound BEFORE anything
subscripts a buffer with it. It was verified to fail on the unfixed line. It also
asserts it found at least 40 such shaders, because a rule that stops matching
passes by finding nothing.

🔴 **AND IT IS MEASURED ON THE M2, INCLUDING THE ARM THAT PUTS IT BACK.**
`bench-af2-warm.js --rows=128 --passes=8`, every length that was ever reported:

| L | excess invocations | distinct structures | checksum |
|---:|---:|---:|---:|
| 128 | 0 | 1 of 8 | -2863903 |
| 129 | 2,064,256 | 1 of 8 | -3940524 |
| 160 | 917,504 | 1 of 8 | -9429913 |
| 200 | 1,171,456 | 1 of 8 | -737980 |
| 400 | 491,520 | 1 of 8 | -15147920 |
| 825 (the standing gate, `--passes=3`) | 960,384 | 1 of 3 | -36459566 |

160 returned EIGHT distinct structures before this. Two controls, both run:

- **The guard deleted again, on the GPU.** 160 races immediately - -8031822,
  -8047564, -7540637, -8782252, -9914993, -7897802, -6635619, -7995441 - so it is
  the guard that fixed it and not something that moved beside it.
- **128 with the guard and without it is the SAME fold**, -2863903 and pLDDT
  39.72 both ways. The guard is inert where nothing over-dispatches, which is what
  the excess column predicts and the only reason to believe the table.

These are M2 checksums and are NOT comparable with the A100's: the two machines
resolve different kernels. What is comparable is the count in the third column.

🔴 **AND EVERY ROW OF THAT TABLE WAS RUN IN THE 53-CALL SHAPE, WHICH IS THE
WORST ONE.** It was taken before `cd366b6` fixed this gate's own alignment
generator, so `--rows` was inert and the alignment carried twelve distinct
sequences however deep it claimed to be - below `cOuter`, so the outer product
mean's residual fired in every block. The checksums above are therefore stale
against the current generator, and the determinism they report is STRONGER than
it looked: 53 corrupted adds a pass rather than one. Re-measured at 160 residues
with distinct rows, both shapes, eight passes:

| shape | `addInPlace` dispatches | distinct structures | checksum |
|---|---:|---:|---:|
| `--rows=128` (outer-first, one call) | 1 | 1 of 8 | -8240180 |
| `--rows=16` (shallow, 48 + 1) | 49 | 1 of 8 | -5662210 |

The A100's count reproduces here exactly - 1 and 49, each over-dispatching by
917,504 - which is `tools/gpu/probe-add-in-place.js` doing the job that reading
the source twice did not.

### What to run, in this order

**1. `python3 tools/audit-knobs.py`, and this is the main ask.** It sets each
knob to a value the device does not resolve today and compares two digests -
every shader by label and content, every dispatch by label and grid. On the
A100 it found five bugs and every one was in an arm nobody had run. Roughly half
its "unmoved" rows here are knobs an A100 fold reaches and an M2 configuration
may not, so this covers different ground rather than repeating it.

```
node tools/gpu-chrome.mjs tools/gpu/probe-tuning.js 2>&1 | grep -v '^\[gpu-chrome\]' \
  | python3 -c 'import json,sys;t=sys.stdin.read();print(json.dumps(json.loads(t[t.index("{"):])["currentTuning"]))' \
  > /tmp/tuning.json
python3 tools/audit-knobs.py --tool=fold-af2 --tuning=/tmp/tuning.json
python3 tools/audit-knobs.py --tool=fold --args=--model=/model-af3-int5/manifest.json --tuning=/tmp/tuning.json
```

Read the FAILED column first: on this box two of those were pipeline key
collisions in arms that had never been run. 🔴 **AND ONE ENTRY FAILS BY DESIGN NOW**:
`attentionQueriesPerLane` is declared and NOT WIRED - nothing reads it - so
`deviceProfile` refuses any value but 1 rather than ignoring it, and the audit
will report it alongside genuine collisions. That is the intended signal, not a
bug: a knob that silently does nothing is worse than one that says so. It keeps
its declaration because it has numbers (M2 0.21x, M4 Pro 0.45x, GB10 1.17-1.42x)
and wiring it needs the pipeline key and the dispatch, not just the destructure. "Moved nothing" is suspicious and
not dead - a diffusion knob cannot move anything in an AF2 fold, and
`keepTrunkWeights`, `batchComputePasses` and `deviceFeaturisationMinBytes` are
all alive and invisible to a shader digest by construction.

**2. Memory, because it is the one that can hurt.** `keepResidentAffordable`
returns true for a device with NO budget, and a tool run sets none - so an M2
holds ~561 MiB of trunk and ~325 MiB of sampler weights between folds. That is
nothing against 40 GB and a different proposition on a laptop, where this file's
own warning is that Metal "takes buffers well past the point where macOS starts
paging". `fold.js --budget=0` prints the peak. If it pages, the fix is a budget,
not a revert.

**3. `tools/gpu-chrome.mjs`, which has taken changes from both sides.** Your
`LOCALFOLD_GPU_ANDROID` lane plus `--occupancy`, `--default-tuning` and
`--tune-json` from here. It is the harness every gate runs through; one gate
through it is the smoke test. `--tune-json` in particular is new and untested
there, and step 1 depends on it.

**4. `probe-occupancy.js`, whose failure mode is a plausible small number.** It
read **4 workgroups** on a card the tool measures at 4542 before it grew a
warm-up and two rounds. Check `agrees: true`. If it disagrees, the three
width-driven derivations are being fed a wrong number and `--default-tuning`
switches all of them off.

**5. Re-measure your own residual.** The ~90 ms attributed to three extra
speculative pipelines may be gone: the pipeline cache now shares kernels built
from identical WGSL, which is 96 -> 73 pipelines on AF2 here and 269 -> 191 on
OpenDDE.

A whole fold is the wrong instrument for any of these. It mixes them, and an M2
should be FASTER overall, which hides an accuracy change and a memory one
equally well.

### What landed here since your last look

| | |
|---|---|
| pipeline cache shares kernels built from identical WGSL | 269 -> 191 pipelines on OpenDDE, ~160 ms |
| **the matrix flash attention's "race" is closed and was never one** | the bisection unrolled a loop whose trip count is not uniform at a head of eight - see docs/A100.md. `attentionMatrixPrefetch` is the reorder it was blocking, written so the barrier stays uniform, and it is **0.05% here** where the old table said 5.2%. Ships OFF; both arms bit-identical. On an M2 this kernel is refused outright for wanting 16x16 units, so the arm should never run - worth confirming it still COMPILES nothing |
| `test/uniform-barrier.test.js` | the structural gate for that class: no barrier inside a lane-strided loop, 189 of them, and it fails when one is put back |
| `opmPairBlockBytes` 64 -> 256 MiB, **ampere prior only** | 2% of a block at 825 residues, +193 MiB |
| two pipeline keys that did not name their shader | `triangleProjectMatrix=false` and `singleProjectWorkgroupTarget=` both used to crash AF3 |
| `shapedKnob` over thirteen sites | `false` now means "unset" for a knob that takes a shape, never for a boolean one |
| `--tune-json`, `--occupancy`, `--split-k`, `--keep-profile`, `elapsedMs` | the arms and instruments the above needed |

🔴 **THE THREE AF2 BASELINES MOVED, AND HERE IS EXACTLY WHY.** They are now
AF2 **-1287025**, the 30,29 multimer **315591**, `bench-af2-warm`
**-121844157**. Nothing about a kernel changed: the SYNTHETIC alignment both
tools build used to vary its gap stride by `row % 11`, so rows 1, 12, 23 and so
on were identical and `--rows=128 --extra-rows=128` was 256 rows carrying
**12** sequences. The rows are distinct now and both tools assert it.
`fold-af2.js` was fixed first (-1846490 -> -1287025) and `bench-af2-warm.js`
after (-67537339 -> -36457799 -> **-121844157**), which is why that one moved
twice. Confirmed
to be that and not the deduplication landing beside it: `--no-dedupe` gives
-1287025 too. AF3, OpenDDE and ESMFold2 are untouched.

### Known open, so you do not chase them

- 24 of the Dawn suite's failures are `ENOENT` for fixtures not in the
  repository; two more are Dawn's adapter lacking `shader-f16`.
- `probe-sidechains.js` 404s on the float32 bundle, which is not on the A100 box.
- Measured and DECLINED with numbers, all in docs: upstream's LayerNorm
  rearrangement (1.3% here), the featurisation tail (needs `atan` on the GPU,
  where WGSL permits 4096 ULP), the two structural triangle contractions, and
  retiling `pair-transition.down` (feeding it more workgroups is achievable and
  still does not help - `wide` loses more than `down` gains).
- **Bundle size is the only lever left worth seconds** and needs the float32
  exports plus a re-publish; see docs/HOSTING.md.

### And the merge itself

Still a fast-forward: `main` is an ancestor of `a100`, no conflicts. Two rounds
are queued on it now - the windowing and this one.

🔴 **AND THE SHIPPING SURFACE IS FOUR FILES.** `git diff main..a100 -- src web`:

```
 src/kernels/attention-matrix.js  | the staging restructure, behind the knob
 src/kernels/attention.js         | the knob in the pipeline key
 src/runtime/device-profile.js      | the knob, declared and defaulting to null
 src/runtime/execution.js           | addInPlace windows its bindings
```

Everything else in the merge is docs, probes and tests. The knob is `null`
in `DEFAULT_TUNING` and set by no prior, so the shipped path on every device is
the arm that measured bit-identical here.

🔴 **AND ITS SHIPPING SURFACE IS THREE LINES.** `git diff main..a100 -- src web`
is three files, and ignoring comments the whole of it is two bounds checks and
one `export`:

```
+  if (index >= arrayLength(&base)) { return; }      execution.js
+  if (index >= arrayLength(&output)) { return; }    elementwise.js
-const ADD_IN_PLACE_SHADER = `
+export const ADD_IN_PLACE_SHADER = `
```

plus `setBufferPooling` in allocator.js, which is a diagnostic that DEFAULTS TO
OFF - `poolingDisabled` starts false and only `bench-af2-warm.js --no-pool` sets
it, so the shipped path is untouched. Everything else in the merge is docs, two
probes, a test and the race gate's alignment generator.

Verified on the M2 that the guard is inert where nothing over-dispatches:
`fold-af2.js` at 59 residues is **-1282976 on both branches**, byte for byte.

## The traps that repeat

🔴 **AND `gh` TALKS TO THE OTHER PORT BY DEFAULT, WHICH LOOKS LIKE A QUIET
DEPLOY HISTORY.** This checkout has two remotes - `origin` is
sokrypton/localfold and `upstream` is martin-steinegger/alphafold2-webgpu, which
21 commits here compare against - and with more than one remote `gh` picks for
itself. It picked UPSTREAM: `gh run list` returned that port's runs, with that
port's commit titles, and the newest was a day old, which read as "nothing
deployed today" while four deploys had in fact succeeded. Nothing errors; the
answer is just about a different repository.

```
gh repo set-default sokrypton/localfold     # once per checkout
```

It writes `remote.origin.gh-resolved` into `.git/config`, which is NOT committed,
so every machine does it once - and until it is done, pass `-R
sokrypton/localfold` explicitly. Check with `gh repo view --json nameWithOwner`
before believing anything `gh` says about this repository.

🔴 **AND PUSHING TO `main` IS THE DEPLOY; `tools/deploy.py` IS THE VERIFY.**
The "Deploy WebGPU demo" workflow runs on PUSH, so every commit that reaches
`main` publishes itself within a couple of minutes. `deploy.py` dispatches the
workflow AGAIN and then polls `build.json` until the commit it pushed is the one
being served - which is why a run it starts shows up twice, once for the push
and once for the dispatch. Its value is the POLL: it is how "live" becomes a
fact rather than an impression. But a push that nobody followed with the script
is deployed all the same, and asking `deploy.py --verify` is the way to find out.

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

🔴 **AND THE STANDING RACE GATE NO LONGER FITS THE DEFAULT TIMEOUT ON AN M2.**
`gpu-chrome.mjs` gives a module 600 s and reports "timed out after 600000 ms",
which reads exactly like a hang. `bench-af2-warm.js` at its default
825/512/1024 now takes **770 s** there - 291 cold and 238/241 warm - because
`cd366b6` made its alignment generator produce DISTINCT rows: the shape used to
collapse to twelve sequences under deduplication and is now a genuine 1536-row
one, about 1.45x the work. Raise the budget rather than doubting the fold:

```
LOCALFOLD_GPU_TIMEOUT_MS=2400000 node tools/gpu-chrome.mjs tools/gpu/bench-af2-warm.js
```

It passes there - deterministic, checksum **-123086553** on an M2 against the
A100's -121844157, which is the usual cross-machine difference and not a
disagreement. The A100 is fast enough that the default still fits.

🔴 **AND `node tools/gpu-chrome.mjs` SOMETIMES DOES NOT EXIT.** The results file
is complete and correct and the node process sits there with a headless Chrome
still running, which in a `for` loop stalls every arm behind it. `pkill -9 -f
"gpu-chrome-"` matches the temporary profile directory and nothing else - not
the browser you are using. A batch of checkers should carry one between arms.

🔴 **AND THE PAGE AND THE CLI BUILT THE BATCH BY HAND, EACH DROPPING SOMETHING
THE OTHER PASSED.** `web/af3-model.js` and `tools/gpu/fold.js` both called
`af3MsaFromA3m` and then `featuriseProtein`, and the CLI passed no
`profileMsa` (profiling the 127 CROPPED rows where the page profiles all 8076 -
a different feature, and it feeds `target_feat` too) and no seeded row choice
(taking the alignment's prefix where the page takes a seeded subset). So every
`--a3m` gate here was measuring a fold the site does not run, and the one bug
only the page had could not be reproduced by the tool written to reproduce the
page. **Both call `af3BatchFromA3m` in src/af3/featurise/batch.js now.** `--prefix-rows`
is the control arm for a baseline recorded before this.

🔴 **AND THE PAGE RAN 48 OF BOLTZ2'S 64 TRUNK BLOCKS FOR A DAY, WHICH READS AS
"THE MSA IS NOT REACHING THE MODEL".** `trunkWeights(store, 48, 4)` with both
depths typed in is right for four of the five AF3-lineage families; boltz2's
trunk pairformer is **64**. A stack is one stacked tensor, so asking for 48 of
64 loads the first 48 slices, runs them, and returns a trunk that never
finished - no error, no wrong shape, a plausible structure. Page pLDDT **72.1
with no MSA, 72.4 with 128 rows, 96.1 after the fix**, which is the CLI's number
on the identical batch. `trunkWeights` RAISES on a disagreeing count now
(`{ allowPrefix: true }` for a bench that means it) and
`test/trunk-depth-from-bundle.test.js` gates that no fold path passes a literal.
See docs/AF3.md. **A constant that is right for the model you developed against
is a silent wrong answer for the next one** - the dialect system takes a second
checkpoint's CONVENTIONS off its weights, and its DEPTHS were still coming off
AlphaFold 3's.

🔴 **AND A BUNDLE THE CLI LIKES CAN BE ONE THE PAGE CANNOT LOAD.** The command
tools read the `manifest.json` sitting next to the shards; the PAGE reads the
manifest baked into `src/bundles/manifests/<family>.js`, which is pinned to a
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

🔴 **AND `-mmin +5` IS NOT ENOUGH WHILE A BATCH IS RUNNING.** A gate that takes
longer than five minutes - `test:portable` and `test:spec-floor` both fold six
models and take twenty - has a profile OLDER than five minutes and still open,
so this sweep kills it. It looks like the gate hanging: no error, no output past
whichever model was mid-fold, and nothing left running. Sweep BETWEEN batches,
never during one.

🔴 **AND A CHECKSUM DOES NOT TRAVEL BETWEEN MACHINES, SO NEVER HAND ONE ACROSS
AS A BAR.** The A100 folded `fold-af2.js` at **-1287025** and the M2 at
**-1282976**, over the same code and the same input, because the two resolve
different attention kernels - `attention:flash-matrix-32-...-4x32` here and
`attention:flash-registers-32-chunk16` there, the M2 offering `8x8x8` units and
nothing else against a kernel that declares `<f16, 16, 16>`. Both folds are
correct.

🔴 **AND BOTH OF THOSE ARE THE int8 BASE'S. AlphaFold 2's TWO BUNDLES ARE int5
FROM 2026-09-18** - asymmetric group 32 through tools/quantize_af3.py, 73 and 74
MiB against 98 each - so this A100 now folds the monomer at **-1309830** (pLDDT
62.924) and the 30,29 multimer at **-393805**. Every AF2 checksum written down
before that date is the old base's and is not a bar for anything today. The
switch is free where it was measured: the single-sequence 59-mer 62.924 against
62.646, 5CAJ with 7907 rows 1.859 A / 95.790 against 1.864 / 95.848, and
barnase-barstar with a paired alignment ipTM **0.9208 against 0.9254**. See
docs/AF2.md.

This file already said so, buried in the race section - "these are M2 checksums
and are NOT comparable with the A100's" - and a checklist written for the M2
still told them "the checksum they must not move is -1287025", which on their
first run would have read as a failure. **An arm is compared against the same
box's other arm.** What travels is the COUNT of distinct structures, the
relative residual, and whether two arms on one machine agree - never the value.

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
`src/af2/multimer/model.js` had already been through this (it dropped the whole
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
one; `src/kernels/attention-matrix.js` now measures 1.69x here. **Measure a
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
| `docs/TOOLS.md` | **every tool in the repository, with the first line of its own header** - generated by `tools/index-tools.py`, gated by `npm run test:tools`. The table above is the curated view; this is the census |
| `docs/DEVELOPING.md` | the older orientation notes |
| `docs/RUNNING.md` | the USER-facing half: running the site from a checkout, the Linux/NVIDIA flags, and using the kernels as a library. README.md points at it; it was the one document this index did not list |
| `docs/PARITY.md` | 🔴 **NINETEEN OF TWENTY-ONE AF3 CHECKERS DO NOT RUN ON THIS BOX** - they 404 on a bundle or a dump rather than compare anything, so a suite run reads as twenty-one things that did not object. Also the reference's level matrix (L0-L6), the two ideas worth copying from it, and the eighteen models it runs against this port's four |
| `docs/ESMFOLD2_PTM.md` | 🔴 **TWO BUGS, NOT ONE, AND THE VENDOR HAS NEITHER.** Same job on one checkpoint: the `esm` package places a phosphoserine at **1.002** and a glycerol at **0.986**; af3-any-model's `esmfold2_lm600m` reads **1.446 and 1.346**; this port **2.349 and 0.958**. All against the same 171-bond control at ~1.00. 🔴 **BOTH FIXED NOW, AND NOT BY THE SAME FIX** - sokrypton's side took three (`7df8d97`: a ref-pos table that rewrote the six atoms a SER has and left the phosphate on its CCD frame; an unsymmetrised bond matrix that half-bonded the glycerol; the restype) and reads **0.991 / 0.991**; this port had only the restype - conformer already exact, already symmetrised, glycerol always correct - and reads **0.994**. 🔴 **THE CAUSE HERE: AN ATOMISED RESIDUE IS UNKNOWN, NOT ITS PARENT.** AF3 gives its atom tokens the PARENT restype, so a phosphoserine's ten all said SER; the vendor's atomised branch is `res_type=PROTEIN_UNK_RES_TYPE, input_id=DNA_RNA_LIGAND_INPUT_ID` - **22 and 24**. 🔴 AND 24 IS NOT WHAT THE RESTYPE TABLE GIVES: `AATYPE_TO_ESM_ID` maps unknown to `<unk>` (3) deliberately, because an `X` in a SEQUENCE is a residue nobody identified, where an atomised residue is a row of ATOMS and the tower sees a ligand's token. Two unknowns, two ids, so it is not one lookup. Gated in `test/esmfold2-lm-mask.test.js`, watched failing. 🔴 **AND NOT THE STEP COUNT, SWEPT RATHER THAN ARGUED**: the checkpoint's default is `inference_num_steps: 15`, and below that the vendor breaks EVERYTHING (at 11 its CONTROL is 2.155) while from 15 up it is flat - 1.003 / 1.002 / 1.000 at 15 / 64 / 138. Matched at **138** both sides are converged and the controls agree to 0.002 (1.001 against 0.999) while the SEP differs **2.35x**; on this port more steps make it WORSE (1.399 → 1.883 → 2.279) where the vendor's is flat. 🔴 AND OUR 11 STEPS IS NOT THE VENDOR'S 11 - at 11 its control is 2.155 and ours 0.994, so `actualSteps` is not a like-for-like count. Not the quantisation (int8 1.548 / fp32 1.542), not one sample (five at 1.47-1.83). Carries what is excluded here - conformer exact, one `refSpaceUid`, bonds present and consumed, `molType` PROTEIN and CORRECT against the vendor's own 67-token layout - and the two traps it cost: a near-miss 1.101 from atom names that kept a leaving atom, and a sorted-distance match that is not a shape match |
| `docs/BOLTZ2_PTM.md` | 🔴 **A BRIEF FOR THE af3-any-model SIDE, NOT A FINDING OF OURS** - its `boltz2` inflates a modified residue 2.7x where GENUINE Boltz-2 2.2.1 is 0.986-1.011 and its own AlphaFold 3 is 0.988. The reproduction for all three references, what is already excluded, where to look (`tokenBondsTypeEmbed`, the second bond-order plane only boltz2 reads), the discriminator that halves the search (does a plain LIGAND break too, or only an atomised residue inside a chain?), and the atom-pairing trap that nearly produced the opposite answer. A copy lives at `~/BOLTZ2_PTM.md` on the A10 |
| `docs/SMILES.md` | 🔴 **FOLDING A LIGAND NOBODY HAS A CODE FOR** - the SMILES parser, the distance-geometry conformer and the device kernel, held to real RDKit; the nine defects that each produced a plausible molecule; why `@rdkit/rdkit` was measured and declined (2D only, and 70x the size); and the batch crossover at which the GPU starts paying |
| `docs/HANDOFF.md` | the pLDDT-from-distogram attempt whose code was deleted, kept so the next attempt does not repeat it |
| `docs/ARCHITECTURE.md` | 🔴 **WHAT A NEW MODEL COSTS, AND WHY `src/` WANTS REORGANISING** - the user's standing ask, with the evidence from one port of two models: a convention is up to ELEVEN edit sites, weight ORDER is written three times, the pack/offset loop five, and three of this port's six defects were the same shape of bug in two of those places. Read it before proposing a shape, and read its last section - the five things a tidy-up would naturally break, each of which was paid for |
