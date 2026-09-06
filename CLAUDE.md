# Working on LocalFold

`AGENTS.md` has the engineering invariants and `docs/AF3.md` the AF3 port's state,
costs and dead ends. This file is the operational half: how to actually run
things here, and the traps that have cost time more than once.

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

`tools/gpu/check-af3-*.js` are the per-module AF3 oracle checkers.

🔴 **AF2-MULTIMER'S TEMPLATE TERM RUNS ON EVERY RECYCLE AND NOTHING CHECKED
IT.** `tools/oracle/template_reference.py` computed a numpy reference and wrote
`toy-template.json`; no JavaScript ever read it. Compared at last, the two
disagree - and the comparison localises where:

| | relRMS |
|---|---|
| the input term, all nine features, masked AND with a real template | **2.15e-7** |
| after the first pair block | 1.2e-1 |
| after the second | 1.1e-2 |

`tools/oracle/dump_multimer_template.py` settled it by capturing the module from
AF2 itself, and the GPU is right:

| against AF2, captured | masked | real template |
|---|---|---|
| `src/multimer/template.js` | **6.5e-5** | **3.0e-4** |
| `tools/oracle/template_reference.py` | 1.0e-2 | 2.5e-1 |

🔴 **SO THE numpy REFERENCE'S PAIR BLOCKS ARE WRONG, AND ITS BANNER SAYS SO.**
Its `construct_input` is right - it agrees with the GPU to 2.15e-7, geometry
included - and everything after that is not. It stays because that input term is
a second, independently written reading of the nine features; the checker
asserts exactly that much of it.

🔴 **AND A TEMPLATE IS REACHABLE FROM THE PAGE NOW.** It lives behind a protein
row's `⋮`, beside the modified residues, because the two are the same kind of
thing: set on ONE chain, changing what is folded, and invisible on the row - so
the badge counts both. It takes one source - `1abc`, `1abc_A` or a UniProt accession, one field because it
is one question - fetched by `web/template-source.js` from the RCSB or AlphaFold
DB and turned into a slot over the complex's TOKENS. The row shows what it
covered, because a template covering 17 of 120 residues folds perfectly well and
says nothing about it. Measured on a 53-residue target with 1QYS_A: ipTM 0.324
without, 0.358 with.

🔴 **AND TEMPLATES CAN COME FROM THE MMseqs2 SEARCH, WHICH ALREADY FOUND THEM.**
`pdb70.m8` is in the MSA job's own tar beside `uniref.a3m` - no second search,
no extra request - and nothing had ever read it. The structures come from
ColabFold's own server, `{api}/template/{1qys_A,7fao_C}`, as a gzipped tar of
mmCIF; the RCSB is not involved. **Ask with the chain suffix**: `/template/1qys`
answers 200 with a tar holding only the hhsearch index and no structure, which
is a success with nothing in it.

ColabFold then runs `hhsearch` over that index to get the alignment. A browser
has no such binary and does not need one: the m8's last column is a CIGAR - but
its target coordinates index pdb70's SEQUENCE while a template offers its
RESOLVED residues, so it is deliberately not used. The query is aligned to the
resolved sequence instead and the coverage line says what came of it.

Measured: a 91-residue query at `--msa-mode search --template auto` found
`1qys_A`, covered 91/91 and folded at pLDDT 84.9.

🔴 **AND A WATER IS NOT A RESIDUE.** `chainResidues` read every HETATM, so 1qys
chain A came out ninety-NINE residues instead of ninety-two - eight waters, each
an X with one atom, each a pseudo-beta position in the distogram. Found by
running the PDB reader and the mmCIF reader over the same entry and comparing.
MSE is the one heteroatom kept: selenomethionine is how a great many structures
were phased, and dropping it puts a hole in the middle of a chain.

🔴 **AND foldAf3 PLACES THE SLOTS, BECAUSE ONLY THE FEATURISER KNOWS THE TOKEN
LAYOUT.** A slot is indexed by TOKEN; a modified residue is one token PER ATOM
and a ligand is a chain of its own, so a chain's first token is not the sum of
the preceding chains' residue counts. Callers hand over TEXT and a chain index,
and `batch.chainOfResidue` / `batch.residueOfToken` do the placing. The earlier
offset version had a matching bug:

🔴 **A SLOT BUILT BEFORE THE BINDER'S LENGTH IS KNOWN IS BUILT AT THE WRONG
OFFSET.** Protein Hunter draws its designed chain inside the loop, so
`chains[0].length` is 0 when the templates are fetched - which made the complex
53 tokens instead of 69 and put the target's template across the binder. It
folded, and it moved ipTM from 0.324 to **0.533**, which looks like a template
working well. The binder's length is passed in now.

🔴 **AF2-MONOMER'S EMBEDDER IS A THIRD DIALECT, NOT A THIRD COPY.** Same six
geometry features, but: ONE `Linear` over an 88-channel CONCATENATION rather
than nine summed projections; the whole concatenation masked by the BACKBONE
mask rather than each feature by its own; its distogram NOT pseudo-beta-masked
at all; `use_template_unit_vector` **False** in every shipped monomer config,
so three of the six are deliberately zeroed; and the query pair enters
afterwards through a pointwise attention the other two do not have. Against
AF2's own module: 2.7e-4 masked, 4.5e-4 with a real template.

🔴 **AND `template_mask = 0` IS NOT "A TEMPLATE WITH NO ATOMS".** AF2-monomer
ends with `embedding *= (sum(template_mask) > 0)`, so with no template the term
is EXACTLY ZERO, while a present-but-empty one gives `embedding2d`'s bias
through two pair blocks and a projection - which is not small. LocalFold has
always computed the second, which is what ColabFold does; measuring it against
the first reports relRMS 14.5 for a path that is right.
`dump_monomer_template.py --masked-template` is the arm that means anything.

🔴 **AND COLABDESIGN2 CANNOT CAPTURE MONOMER TEMPLATES AT ALL.** It puts the
monomer on the multimer graph and raises - the two embedders differ and the
weights do not convert - so `dump_monomer_template.py` transforms AF2's
`TemplateEmbedding` with haiku and runs the module alone. Two version traps on
the way: that checkout's config sets `fuse_projection_weights: True` everywhere
while `model_1_ptm`'s weights use the older `layer_norm_input` /
`left_projection` names, and comparing against the shipped `model/` bundle
reports int8 quantisation as a fault - use `model.f32-backup`.

🔴 **AND ITS CROSS-CHAIN MASK REFUSES TO GUESS, LIKE AF3'S.** The first
version defaulted `asymId` to all zeros - every token in chain 0 - which is
right for a monomer and silently lets a template speak across a complex's
chains. AF3 had the identical bug, measured at relRMS 1.09. A template with no
chain ids now raises, and `src/multimer/model.js` hands the ids over from the
feature set. Inter-chain templates are opt-in per slot there too, and moving
the term by relRMS 7.3e-2 is what `tools/gpu/check-multimer-template.js`
asserts, since AF2 has no oracle for something it does not do.

🔴 **AND FEED THE MODULE THE MASKS IT WAS GIVEN.** `__call__<2` is
`padding_mask_2d` and `<3` is `multichain_mask_2d`, and both are all ones in
these dumps because ColabDesign2's featurisation gives one asym_id.
Substituting a two-chain mask of our own scored 7.3e-2 against a module that is
right - a check reporting a fault in its own setup.

🔴 **AND COMPARE AGAINST `model-multimer-f32`, NOT THE SHIPPED BUNDLE.**
`model-multimer` is int8 at block 64 (`dtype: "int8"` in its manifest) and the
references read float32 parameters, so the same correct code scores 6e-3 on the
input term against one and 2e-7 against the other. An hour went into that
before the manifest was read.

🔴 **AND THE TEMPLATE EMBEDDER IS NO LONGER UNCHECKABLE.**
`tools/oracle/dump_af3_trunk.py --template <pdb>[:CHAIN]` folds a query with a
real structure as its template and captures the module's inputs and its
per-slot outputs. That answers the objection at the top of
`src/af3/template-reference.js` - "with no template the six geometry features
are identically zero, so nothing here can tell a correct implementation of them
from a wrong one" - which was true and is the reason only the empty-slot path
exists. See docs/AF3.md's template entry for the numbers.

It writes the template's mmCIF from the PDB's OWN ATOM NAMES rather than
through atom37. ColabDesign2's `_mmcif_for` does the same job but reaches AF2's
`residue_constants`, which imports `dm-tree` - not installed here, and not
worth installing to copy 37 strings that every PDB line already spells out.

🔴 **THE PHONE LAYOUT IS MEASURED, NOT LOOKED AT, AND `--window-size` CLAMPS AT
500px.** 390 and 320 both report an innerWidth of 500; `--headless=old` clamps
identically. `tools/cdp.py` is sixty lines of WebSocket (no new dependency) and
gives `Emulation.setDeviceMetricsOverride`, which is a true viewport at any
width, plus `Page.captureScreenshot`, which `--screenshot` cannot do on a page
with a running rAF loop. `tools/mobile-layout.py` runs 320, 360 and 390 with
1200 as the control, loading a structure and an alignment first - half the rows
it measures are `display: none` on a bare page. It checks `single.html` too.

🔴 **AND "NO HORIZONTAL OVERFLOW" IS NOT THE TEST.** Under mobile emulation a
page that cannot fit does not overflow: the LAYOUT VIEWPORT GROWS, so
`scrollWidth == innerWidth` while the phone renders everything zoomed out. The
assertion is `innerWidth == the width asked for`. Nor can a size check see an
OVERLAP, or a `1fr` grid track squeezed to nothing - the entity row's sequence
box measured 0px at 320 with "PIA" set one letter per line, and every fit check
passed. Nor can it see a page that is ready to be measured: `processFiles` is
defined while `initializeApp` is still running and before `web/app.js` (a
module) has run at all, and called in that window it resolves having loaded
NOTHING. Wait for `#predict` to be enabled, which is the last thing to happen.

🔴 **AN INLINE WIDTH IS ONE NO STYLESHEET CAN OVERRIDE.** Not a media query, not
a container query, not any specificity - only `!important`, and a page whose
responsive rules all need that has no cascade left. There were four in
`index.html`, five in `single.html`, two on the MSA filter sliders, and one that
py2Dmol's own JS writes on the viewer box (turned off with
`data-autosize="css"` on `#canvasContainer`). `max-width` is a DIFFERENT
PROPERTY and beats an inline `width` with no `!important` at all, which is how
the MSA panel is contained and how `single.html` lost three of its own.

🔴 **AND py2Dmol's MSA PANEL IS STILL 948px.** `src/panels/msa.js` has
`const MIN_CANVAS_WIDTH = 948`, clamps every canvas width up to it and writes
the result as `container.style.width`. Our narrow block gives that box
`max-width: 100%; overflow-x: auto`, so it scrolls sideways inside its own card
instead of taking the whole document with it - measured, it was forcing a 972px
layout viewport on a 320px phone. Fixing it properly is an upstream change.

🔴 **AND A DIFFERENTIAL CHECKER THAT BUILDS ITS OWN KERNEL TESTS WHATEVER IT
BUILT.** Four of them did this session, each found the same way: a shipped path
learned to pick between an f32 and an f16 kernel, the checker went on
constructing the f32 one from a module constant, and the arm labelled "f32" was
either testing a kernel nothing runs or - once - silently testing the f16 one
and failing. Ask the selection function for the kernel, take the precision as
an axis, and hold each arm to the bound its own arithmetic implies. Raising one
bound to cover both stops the f32 path being checked at all.

🔴 **AND AF2 NOW HAS AN END-TO-END GATE, WHICH THE DIFFERENTIAL ONES ARE NOT.**
A per-kernel checker says one kernel still computes its own operation. It
cannot say the assembled model still folds, and after three kernel rewrites
that was the whole of AF2's coverage here. `tools/gpu/fold-af2.js` folds a
59-mer through the driver the page uses and prints a checksum over every
coordinate, plus mean pLDDT, pTM and the backbone CA-CA geometry. Run it, stash
the change, run it again: at 128 rows and at 512 rows with a recycle, the tree
before this session's kernel work and the tree after agree to every digit.

It synthesises its alignment from the query, so the 512-row kernels run without
fetching anything, and it opens `./model/` by directory - `loadModel` resolves
the monomer family to Hugging Face, and a regression tool should not pull
227 MB. That makes it a fingerprint, not an oracle: it does not know what
AlphaFold would say.

🔴 **AF2's KERNELS NOW HAVE FOUR DIFFERENTIAL GATES, BECAUSE IT HAD NONE.**
`npm run test:gpu` cannot load Dawn here and `test/fixtures/evoformer/` is
gitignored, so every `test/*.gpu.test.js` covering AF2 is unrunnable - which
left its transition, its outer product mean, its attention projection and the
residual form of its triangle output projection with nothing checking them at
all. Each new checker writes its own CPU reference in its own file, because a
reference that shares code with the thing it checks tests nothing, and each
uses ragged shapes and ragged masks so the bounds checks and the masking are
actually exercised. They are differential, not oracle: they say the kernel
computes the operation, not that AlphaFold agrees.

🔴 **CHUNKING THE PAIR SCRATCH LOOKS OBVIOUS AND THE TRIANGLE WILL NOT HAVE
IT.** The five pair-sized scratch buffers are 5987 MiB of a 9662 MiB fold at
1530 tokens - 62% - and the budget's only cheaper route gives up WEIGHT
residency, which is ~567 MiB and does not grow with the protein. So: run the
track a few hundred rows at a time. The grid attention takes it happily - q, k,
v and the gate are indexed `((row * N + i) * HEADS + head)`, row outermost, so
a row chunk is a contiguous byte range and binding that SLICE makes the
existing indexing address it with **no kernel change at all**. `run` in
src/af3/pairformer-block-webgpu.js accepts a slice for this, and
`encodePairTrack` has a `rowChunk` that defaults to the whole track.

🔴 **AND IT STOPS AT THE TRIANGLE, FOR TWO REASONS.** Its intermediates are
CHANNEL-major - `a[h * PAIRS + i * L + k]` - so a row chunk is CH separate
ranges rather than one, and no binding offset expresses that. Worse, the
INCOMING direction reads `b[a_k * L + i]`, so chunking the output rows needs a
strided COLUMN slice of b, which is not a range at any stride. Chunking the
triangle therefore needs a stride constant in the kernels (and those kernels
are shared with AF2's evoformer and multimer) or a transposed copy of b, which
is the buffer the chunking was meant to avoid.

🔴 **AND THE PEAK IS THE WORST CASE, SO HALF THE JOB IS WORTH NOTHING.** The
triangle and the grid share the five buffers and the allocation is sized for
whichever needs more, so chunking only the grid leaves the peak exactly where
it was. It is all or nothing, and the "all" is a restructure of the pair track
rather than the afternoon it looks like.

🔴 **AND THE TOTALS CANNOT SAY WHICH TENSOR TO ATTACK.** `memorySnapshot`
returns `byLabel` beside the totals - the allocator was always given a label per
buffer and threw it away - and `fold.js`, `bench-trunk.js` and `fold-af2.js`
print it. The two models fail differently and the breakdown is what says so:
AF3 keeps its WEIGHTS resident (three tensors were 1216 MiB of a 1406 MiB fold)
and AF2 keeps none, so AF2's peak is all ACTIVATIONS (`msa-transition.hidden`
alone was 118 MiB of 681). Both are now smaller - a 59-token AF3 fold holds
**798 MiB against 1406**, and an AF2 fold at 512 MSA rows peaks at **573 MiB
against 681**.

🔴 **AND THE SAME DIFFERENCE DECIDES WHETHER f16 WEIGHTS BUY TIME.** The
question has no answer except one about the traffic, and it was asked three
times here with three answers:

| where | how the weights are read | f16 storage is worth |
|---|---|---|
| AF3's trunk | resident, one scalar at a time | **-2%** (377 vs 378 on the pair track; 163-166 vs 166-168 on the single track) |
| AF2's transition | uploaded every pass, re-read 944 times | **+8%** of the kernel, plus half the upload |
| AF3's diffusion transformer | streamed once per token tile | **+14%** (48 -> 41 ms at 59 tokens, 103 -> 89 at 150) |

Halving the bytes never halves the read INSTRUCTIONS, and the `f32()` at each
read is not free - so where the bytes are not the bottleneck it is a small
LOSS taken for the memory. See docs/AF3.md's memory section and
`TRANSITION_CHUNK_TARGET_BYTES`.

🔴 **AND THE ATOM BLOCKS ARE THE FOURTH ANSWER: NO, ON ACCURACY, NOT ON
TIME.** They are the one stack with no precision axis, and they are the shape
the table above says should pay: `output` streams a block's whole 655 KB
through EVERY workgroup - 600 of them at 200 tokens, 393 MB of weight traffic
in one pass - and it runs at **148 GFLOP/s against this device's 1220 scalar
ceiling**, which is a kernel waiting on memory. Narrowing the weights was
tried and it does not survive the envelopes:

| | f32 | big matrices at f16 precision | all weights f16 |
|---|---|---|---|
| encoder `tokenAct` | **9.48e-6** | 6.19e-4 | 1.12e-2 |
| encoder `skipConnection` | **2.05e-5** | 9.26e-4 | 2.11e-2 |
| decoder position update | **2.47e-7** | 1.29e-4 | 1.41e-4 |
| denoiser (bound 4e-4) | **1.28e-4** | 1.18e-3 | 1.47e-3 |

The middle column is the diagnosis: rounding ONLY the tensors over 1024
elements - so every LayerNorm scale and per-channel bias stays float32 - still
misses the head's bound by 3x. There is nothing to split off, so a two-buffer
version would not help either.

🔴 **AND THE REASON GENERALISES.** The diffusion transformer takes f16 weights
happily (1.88e-2 inside a 4e-2 bound) because what it produces is an
ACTIVATION, and the LayerNorm after it renormalises most of the error away.
The atom decoder produces a POSITION UPDATE in angstroms, which nothing
renormalises: a relative error there is a coordinate error. Ask what a stack's
output IS before pricing its weights.

🔴 **MEMORY HAS TWO HALVES AND THE BENCHES ONLY EVER SHOWED ONE.** The GPU
allocator's snapshot cannot see a `Float32Array`, and until
`src/runtime/device-memory.js` existed nothing counted the buffers created
outside the allocator - which are most of them by size. Host heap comes from
`tools/gpu/probe-memory.js` (it forces a collection first, or the reading
carries 300 MiB of garbage); device memory from `memorySnapshot(device)`,
which `fold.js`, `bench-trunk.js` and `fold-af2.js` print. A 31-residue fold
held **305 MiB of heap and 1390 MiB on the device** before the f16 weight work
of 2026-09-04 and holds about 800 MiB on the device after it; 1190 MiB of the
1390 was weights kept
resident on purpose, which `--budget` makes the code give up when it must.

🔴 **KNOW THE CEILING BEFORE CHASING IT.** `tools/gpu/probe-alu.js` runs
multiply-adds out of registers with no memory in the way. On this M2 it reports
**about 1220-1260 GFLOP/s scalar, 2420-2470 vec2, 4870-4980 vec4**, and ~400
billion workgroup reads a second - so a vec4 multiply-add is 4x a scalar one,
and every one of those is about 640 G instructions a second. **In f16 it reports
2045-2121, 4090-4295 and 8279-8590.**

🔴 **THE RATIO IS THE STABLE PART, NOT THE ABSOLUTES.** Two runs of this probe
an hour apart differ by 3-4% on every arm, so a kernel quoted against one of
them is quoted to about that. What does not move is that an f16 multiply-add
issues at **1.7x** an f32 one for the same instruction. `shader-f16` is now
requested by `requestAlphaFoldDevice`. Read a kernel's number against THAT, not
against a specification sheet: the trunk's kernels sat at 900 GFLOP/s to
1.1 TFLOP/s, which is 70-85% of the scalar ceiling and a quarter of the vector
one. It is an instruction-count machine.

🔴 **AND HALF PRECISION MOVED THAT CEILING, so the sentence above is about f32
only.** After the f16 work of 2026-09-04 AF2's dense kernels run at 1140-1550
GFLOP/s rather than 900-1100 - past the scalar ceiling, because their
arithmetic is no longer scalar-equivalent - and `opm.contract` at 684 is the
one left behind. See tools/gpu/profile-af2-block.js.

🔴 **`grid.attend` IS THE BIGGEST KERNEL IN AN AF3 TRUNK AND IT IS ALREADY
NEAR THIS DEVICE.** It is the only pass that grows as tokens CUBED, so its
share grows with the protein: 18.3% of the trunk's GPU time at 200 tokens and
**34.6% at 700**, where it is 203.8 ms a pass. `tools/gpu/bench-grid-attend.js`
alternates arms in one process. Three plausible wins were tried and all three
are dead:

| what | at 400 tokens | verdict |
|---|---|---|
| skip the softmax rescale when the maximum does not move | 131.1 vs 125.5 ms | **0.957x, a loss** |
| stage the keys and values in f16 rather than f32 | 127.1 vs 135.0 | 1.06x, so not BYTES |
| the staged key chunk, 16 / 32 / 64 | 125.7 / 125.3 / 124.4 | nothing, so not BARRIERS |

🔴 **AND THE ARITHMETIC SAYS WHY.** Per lane-key the kernel does about 160
scalar operations (eight vec4 dot products, sixteen vec4 accumulator updates)
and **64 scalar workgroup reads** (eight vec4 each of the key and the value).
At 400 tokens that is 2.56e8 lane-keys, which against this device's measured
ceilings - 610 G scalar FMA/s, ~400 G workgroup reads/s - is about 67 ms of
arithmetic and 41 ms of workgroup traffic against 125 measured. It is balanced,
and both halves are near their limit. That is why halving the tile's BYTES buys
5% and halving the barriers buys nothing: neither reduces the number of
operations.

🔴 **AND THE REGISTER BLOCK WAS ATTEMPTED, AND IS THE WORST OF THE FOUR.**
Giving a lane Q queries so one key read serves all of them divides the read
term by Q and leaves the arithmetic alone, which is the right idea. It is
bit-identical - checked at n=128 and n=192 against the reference, where Q of 1,
2 and 4 agree to every digit - and it is catastrophically slower:

| tokens | Q=1 | Q=2 | Q=4 |
|---|---|---|---|
| 256 | 45.1 ms | 78.8 (**0.57x**) | 166.5 (**0.27x**) |
| 400 | 131.6 | 330.8 (**0.40x**) | 745.2 (**0.18x**) |

Q x 8 vec4 of accumulators and Q x 8 of the query is 128 registers at Q=2, and
it spills - the same 4x-the-wrong-way that `grid.project`'s row tile records at
16. The code was reverted rather than kept behind a flag: parameterising the
hottest kernel in the trunk over an arm nobody should use costs every later
reader, and these numbers are worth more than the switch. **What is left is f16
ARITHMETIC in the accumulator update**, whose ceiling is 1.7x - and see the
next entry before trusting a bound on it.

🔴 **AND check-af3-grid-attention.js WAS PASSING BY LUCK.** It builds its input
as `deterministic(n * n * CHANNELS, 991 + n)` - a different random pair for
every n - and it had only ever been run at its default of **24 tokens**. Run
anywhere else, the f16 staged arm fails the 2e-3 bound that n=24 happens to
give:

| n | 24 | 32 | 33 | 36 | 48 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| f16 | 5.5e-4 | 1.1e-3 | 6.0e-4 | 1.2e-2 | 1.4e-2 | 7.1e-3 | 7.4e-3 |
| f32 | 9.6e-7 | | | | 1.0e-6 | 1.2e-6 | 1.3e-6 |

Not a trend in n - n=28 is worse than n=33 - but a spread over DRAWS, a factor
of 26 wide. The f32 arm is flat at about 1e-6 throughout. So the bound measured
one lucky input. It takes the worst of four draws now and the f16 bound is
3e-2, which is what this input costs.

🔴 **AND THE INPUT IS HARSHER THAN A FOLD, WHICH IS WHY THE SHIPPING PATH IS
FINE.** f16 holds eleven mantissa bits, so a staged key is good to ~5e-4 - but
the error lands in a LOGIT, and `exp` turns an absolute logit error into a
relative weight error. Uniform noise makes large, poorly conditioned logits; a
real pair representation does not, and the whole trunk still agrees with AF3 to
**3.94e-4 end to end** with this same path on. Do not read 1e-2 here as a fold's
error.

🔴 **AND ONE DRAW AT n=192 PUTS THE f32 ARM AT 3.99e-4, WHICH IS 400x ITS NORM
AND IS NOT EXPLAINED.** Only the UNTRANSPOSED module, and only draw 2 of four;
the transposed module sees the same pair and measures 1.13e-6. It is the
shipping kernel - the checker is what changed - so it is a real property of
`pair_attention1` on some inputs at that size, found the day the checker
stopped running at one shape. **Open.** `--n=192 --seeds=3 --precision=f32` is
the reproduction.

🔴 **THE FOUR KERNEL BENCHES EXIST BECAUSE bench-trunk.js COSTS FORTY SECONDS
AND AVERAGES 48 BLOCKS.** Each synthesises its weights, runs one shader at
several shapes interleaved in one process, and costs about a second an arm - and
each checks every arm's output against the first, because a tile the dispatch
does not match leaves rows unprocessed and reads as a speedup. Tune with those;
confirm with `bench-trunk.js`.

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

🔴 **AND THE TEMPLATE'S SOURCE IS A MENU NOW, NOT A GUESS.** One box took
`1abc`, `1abc_A` or an accession and decided which server to ask by counting
characters - four is the RCSB, anything else AlphaFold DB. It reads well and it
is right most of the time, and both of its failures are silent: a
four-character accession goes to the wrong server, and a typo'd PDB id becomes
an AlphaFold DB lookup whose 404 names a database nobody chose. The dropdown -
`TEMPLATE_KINDS` in `web/entities.js` - carries **PDB entry**, **AlphaFold
DB**, **From the MSA search** (which was a checkbox that silently overrode the
box beside it) and **Upload a structure** (which had no way to be named at
all). `fetchStructure(text, {kind})` takes the kind; `parseSource`'s
count-the-characters rule survives only as the fallback for a caller that has
none.

🔴 **AND AlphaFold DB PUTS ITS VERSION IN THE FILENAME, WHICH MOVES.** The URL
was built as `AF-<id>-F1-model_v4.pdb`, and AlphaFold DB's v6 release retired
v4 outright - `curl -I` says 404 for v4 AND v5 on every accession tried - so
every AlphaFold DB template on the page was a 404 naming a URL the user had not
chosen. It asks `https://alphafold.ebi.ac.uk/api/prediction/<id>` for `pdbUrl`
now: one request, CORS open, and nothing to bump at v7. Measured: P61626 covers
9 of the 58-residue default, 1QYS_A covers 8, and the same file uploaded from
disk covers 8 - the upload and the download agreeing is the cross-check that
the two routes reach the same slot builder.

🔴 **AND THE pLDDT FLOOR IS GONE.** It defaulted to 70 for AlphaFold DB, on the
sound reasoning that a predicted structure has every residue and no way to say
it did not see one - so a disordered tail arrives as geometry. But nothing on
screen said the default had done anything, and a number from 0 to 100 is a
modelling choice the popup cannot explain in the space it has.
`buildTemplate`'s `minConfidence` option and `filterByConfidence` remain for a
caller that wants them; no page sets one.

🔴 **`fold-in-page.py` FOLDS ONE CHAIN, AND A COMPLEX FAILS SILENTLY.** It
types the whole `--sequence` into a SINGLE entity's field, so a colon-joined
`A:B` is rejected by the page - "One sequence per entity - use Add entity for
another chain" - the Fold click does nothing, and the tool waits out its whole
timeout. Two 25-minute runs went that way before `cdp.wait_for` learned to print
the page's status line as it changes (`progress=STATUS_LINE`), which answered it
in ten seconds. Watch those `...` lines: a wait that prints nothing never
started. Driving a real complex needs the tool to build one entity per chain
through `window.__entityList`, which it does not do yet.

🔴 **THE WEIGHTS AND THE MSA SEARCH RUN TOGETHER NOW, AND USED NOT TO.** The
model was loaded inside the fold, which runs after the alignment - so a cold
page with the MSA set to search spent the whole MMseqs2 round trip with the
network otherwise idle, then spent the whole download with the search already
answered. They need nothing from each other: one is a static file from a CDN,
the other a query against a server that queues. `startModelPreload` begins the
load before the templates and the alignment; both loaders memoise, so the fold
awaiting the same call later gets that promise rather than a second download.
Measured with `tools/fold-in-page.py --timeline`, which reads resource timing:

| | search | model | overlap |
|---|---|---|---|
| before | 795-1426 ms | 1472-1902 ms | **-46 ms** (strictly sequential) |
| after | 1296-2227 ms | 1359-1844 ms | **+485 ms** |

🔴 **AND A `--timeline` THAT FILTERS BY NAME ALONE MEASURES PAGE LOAD.** The
first version reported a 1.2-second overlap - and the UNCHANGED tree reproduced
it exactly, because `/mmseqs/` matches `src/input/mmseqs2-api.js` and the
weights directory is probed before the button is pressed. Both spans started at
66 ms, which is not a span of anything a click caused. It stamps
`window.__foldClickedAt` at the click and ignores everything earlier.

🔴 **AND THE DOWNLOAD MUST NOT WRITE TO THE STATUS LINE ANY MORE.** Two writers
several times a second, and the message that loses is the one about a server
that may queue for a minute. It reports on the right instead - a filling dial
plus `AlphaFold 3 · 92 / 265 MiB` - and appears only once a load reports itself
partway, so a model already in the shard cache does not flash it.

🔴 **AND `tabular-nums` DOES NOT STOP A COUNTER RESIZING.** It holds every
DIGIT to one width, which is not the problem; the problem is a number that
GROWS a digit, so `1 / 265` became `10 / 265` became `100 / 265` and the label
stepped wider twice per download - moving the dial right and squeezing the
status line, twice, every time. The loaded figure is padded to the width of the
total with **U+2007 FIGURE SPACE**, which is defined as a digit's width. A
plain space does not work: it collapses, and measured it steps exactly as the
unpadded string does. Measured at 1, 10, 100 and 265 MiB, with the unpadded
string as the control that says the measurement can see a difference at all:

| pad | widths | constant |
|---|---|---|
| none (control) | 132.78, 139.78, 146.77, 146.77 | no |
| a plain space | 132.78, 139.78, 146.77, 146.77 | no |
| U+2007 | 146.77 x4 | **yes** |

🔴 **AND A CSS TRANSITION ON A VALUE THAT CHANGES 8776 TIMES IS PURE LAG.**
The dial's fill carried `transition: stroke-dashoffset .2s linear`, to sweep
rather than jump between shard callbacks. `HttpTensorStore` reports once per
network CHUNK - 8776 callbacks over one load of `model-af3-int5` - and a
transition retargeted that often never arrives: each callback restarts it from
wherever the arc has got to, so the arc trails the true value by about
`duration x rate`, and the rate is one whole arc over the length of the load.
The dial read 99.6% while two thirds drawn, and was hidden there, so a load
that COMPLETED looked like one that stopped. Measured with
`tools/fold-in-page.py --bar`, which reads the drawn dashoffset beside the
label, on a 450 ms local load:

| label | arc, with the transition | without |
|---|---|---|
| 96 / 265 MiB | 0.05 | **0.39** |
| 192 / 265 MiB | 0.28 | **0.74** |
| 264 / 265 MiB | 0.64 | **0.93** |

🔴 **AND IT IS WORSE THE FASTER THE LOAD**, which is why nothing caught it: a
slow link lags 0.2 s in 30, which is 0.7% and invisible, while a returning
visitor with the shards cached sees almost none of the arc at all. The DATA was
right the whole time - `tools/gpu/probe-load-dial.js` reports the store's
fraction ending at exactly 1, and the manifest's 264.6 MiB matches the shards on
disk and the shards Hugging Face serves - so every check that read the numbers
passed. **Sample what is DRAWN, not what was computed.**

🔴 **AND THE DIAL IS NEVER LAID OUT UNLESS SOMETHING FORCES IT.** It is
`hidden` on a bare page, so `tools/mobile-layout.py` had never seen it, and its
label cannot wrap or shrink: at 320px the label took 171px of a 254px row and
left the status line **75px**, which fits none of "MSA search · queued
(PENDING) · 41s". Nothing overflowed, so every fit check passed. The label is
`display: none` below 560px; the dial and its title carry the bytes there.
Measured by forcing the dial visible at each width, which is the only way it is
ever laid out.

🔴 **"DOWNLOAD ALL" WRITES THE AF3 SERVER'S ARCHIVE, AND THE UPLOAD BOX READS
IT BACK.** `web/zip.js` is a writer and a reader in one file; `web/fold-archive.js`
assembles the members. Checked against `tools/fixtures/fold_2026_09_01_10_17.zip` in the repo
root: `full_data_0.json` and `job_request.json` match key for key, and
`summary_confidences_0.json` carries nine of its ten. The tenth, `has_clash`, is
omitted because it is a claim about geometry nothing here computes. The
structure is `.pdb` where the server writes `.cif`, which is the one deliberate
difference.

🔴 **AND "DOWNLOAD ALL" IS THE ONE PATH NOTHING RAN, SO EVERY FIELD A FOLD
FORGOT TO STORE FAILED THERE AND NOWHERE ELSE.** It builds the archive out of
`lastPrediction`, and EF2-fast stores no `confidence` object at all - on purpose,
since an object of zeros would be read as the model's opinion. The archive
recovered its TOKEN COUNT from `confidence.predictedAlignedError.length`, so the
button reported *"Cannot read properties of undefined (reading 'length')"*,
which names neither the model nor the field. `tools/fold-in-page.py --download`
presses it, reads the zip BACK through `web/zip.js` and prints its members and
`full_data`'s keys - because "a zip was written" is not "the fold's numbers are
in it".

🔴 **AND THE CONTACT MAP WAS IN THREE PLACES, OF WHICH THE ARCHIVE KNEW TWO.**
AF3 hands it back with its confidence, AF2 fills it in from a `setTimeout` off
the saved pass (its distogram head costs 131 ms at 128 residues and is
deliberately off the critical path), and EF2-fast kept it beside the prediction
in a field of its own - so the model whose contact map is its ONLY score wrote
an archive without one while the panel on screen showed it. `contactSource` is
now the single field on every path, and it holds the OBJECT rather than a copy,
which is what lets AF2's arrive late.

🔴 **AND A README THAT DESCRIBES A CONTROL THE MODEL DOES NOT HAVE IS WRONG, NOT
MERELY VERBOSE.** EF2-fast's said `recycles: 1` and `max msa: 128` - the shared
dials' values, reported as though they had been used. It reads neither: it folds
from the sequence alone, and its trunk loops a number of times the CHECKPOINT
fixes (four) rather than the dial. `msaOrigin` undefined now means "this model
does not take one", which drops the alignment line AND the paragraph describing
a `msas/` directory the archive does not contain.

🔴 **AND `atom_plddts` MUST NOT CARRY SOMETHING THAT IS NOT A pLDDT.** The
B-factor column is whatever the model put there, and for EF2-fast that is the
distogram certainty under a REMARK naming it - the page is careful that the word
pLDDT appears nowhere for that model, and the key would have undone it in the
one file a reader is most likely to parse. It is `atom_certainty` there, and the
PAE is omitted rather than written from an absent matrix.

🔴 **AND `chain_pair_max_contact` IS THE ONE SCORE A MODEL WITH NO CONFIDENCE
HEAD CAN STILL GIVE.** AlphaFold 3 splits intra- from cross-chain too, but only
through ipTM - `iptm_ichain` and `iptm_xchain` in its own code - and through the
contact-weighted PDE summaries in `confidences.py`, all of which need the
confidence head. The strongest predicted contact needs only the TRUNK: the
diagonal says how sure the model is that a chain touches itself at range, and an
off-diagonal entry says whether it believes in the interface at all. It is not a
server field; it is an addition, and it is the reason EF2-fast's summary file is
worth writing. Measured on a two-chain fold, AF3 reports `chain_pair_max_contact`
0.73 across the interface while its `iptm` is 0.23 - the distogram believing in
a contact more than the confidence head believes in the interface.

🔴 **AND SEQUENCE NEIGHBOURS ARE EXCLUDED OR THE DIAGONAL IS ALWAYS 1.00.** A
token is in contact with itself and with the residue beside it whatever the
fold. The rule is the one used everywhere else here - the same chain and within
six RESIDUES, which drops a ligand's whole self-block. A chain shorter than the
separation reports **null**, not zero: "no pair to measure" is not "the model is
sure this does not fold".

🔴 **AND THE ALIGNMENT ROUND TRIP WAS BROKEN BEFORE IT, IN A WAY THE PAGE
ADMITTED IN A COMMENT.** "A pasted or uploaded A3M is one text and cannot be
split into blocks; it becomes the unpaired one." AF3 reads the paired block
first and takes its profile over the UNPAIRED one alone, so downloading an
alignment and uploading it again folded something else, silently. The archive
carries one a3m per chain per block; `msasFromArchive` feeds them back through
`mergeSearchedChains`, the same call the search path makes.
`tools/archive-roundtrip.py` folds, downloads, re-uploads and folds again -
and asserts on **"trunk reused"**, because the trunk cache key hashes the
alignment blocks, so reuse is the page saying the restored blocks are
bit-identical to the searched ones. A matching structure alone would be weaker.

🔴 **AND THE SCORE KEYS ARE ASYM IDS, NOT CHAIN INDICES.** AF3 numbers chains
from ONE (`featurise.js` writes `identity.asymId + 1`); AF2 uses contiguous
blocks from zero. Reading the keys as indices gave a real two-chain fold
`chain_pair_iptm: [[null, null], [null, null]]` and `chain_ptm: [null, 0.69]` -
"1|2" matching nothing and "1" matching the second chain by accident. **Every
unit test passed**, because they were all written with 0-based keys. The
archive sorts the ids it finds and takes the nth as the nth chain.

🔴 **AND `fold-in-page.py` CAN DRIVE A COMPLEX NOW** - one entity per chain
through `window.__entityList`, instead of typing a colon-joined sequence into
one field and waiting out the timeout. A 108-residue two-chain fold with
`--msa-mode search` takes about 6 s. **A 199-residue one (barnase/barstar with
10,839 hits) sat at "Trunk · 1%" for twenty minutes and did not finish** - not
diagnosed, but it is the shape to avoid in a quick loop.

🔴 **AND AlphaFold 2 SAVES ITS BEST PASS, NOT ITS LAST.** Recycling is not
monotonic and AlphaFold's own pipeline ranks its outputs; the criterion is
ColabFold's `rank_by: auto` - the multimer score for a complex, mean pLDDT for a
monomer - and the search starts from the last pass so a tie keeps the more
converged one. The scores card and the status line report the saved pass, and
the line says `saved pass N of M` when it is not the last, because the play bar
is still sitting on the last one.

🔴 **py2Dmol IS A MIRROR NOW, WITH A COMMIT ON IT.** The four vendored files -
`py2Dmol.app.css`, `py2Dmol.align.js`, `py2Dmol.embed.min.js`,
`py2Dmol.full.min.js` - had been copied by hand and nothing recorded from
where, so "is this current?" could only be answered by diffing 800 KB of
minified JavaScript against a build. `python3 tools/sync-py2dmol.py` runs
upstream's own `tools/bundle.py build`, copies the four, and stamps the commit
and each file's hash into `web/vendor/SOURCE.md`; `--check` says whether the
mirror has drifted. Never edit the mirror.

🔴 **AND BOTH BUNDLES ARE NEEDED, WHICH THE SIZES HIDE.** `full` is the website
plus the embed API and `embed` is the embed API alone, so `full` looks like a
superset and is nearly one - but `index.html` loads `full` while `single.html`
and `proteinhunter.html` load `embed`. Syncing only the larger leaves two of
the three pages on a stale viewer.

🔴 **ACTIVATIONS CAN BE STORED TWO HALVES TO A WORD, AND `pack2x16float` IS
CORE WGSL.** Unlike the `f16` TYPE, it needs no device feature, so a tensor
halves on hardware that cannot compute in half precision at all.
`src/runtime/storage.js` is the whole mechanism and `execution.allocate`'s
fourth argument is how a caller asks. A 59-residue fold at 512 MSA rows went
**603.0 -> 396.4 MiB** across four tensors, for 0.043 pLDDT, and got 4.5%
faster where the reader re-reads (the flash kernel's key and value); where it
does not, time is unchanged.

🔴 **AND A WORD IS OWNED BY ONE INVOCATION OR IT IS A RACE.** WGSL cannot write
sixteen bits, so a lane holding one half would read the word, insert and write
it back while the lane holding the other half does the same. Every kernel
converted had to be rearranged so the pair of elements sharing a word is
produced by one lane: the layer norm walks channel PAIRS, and both tiled GEMMs
give a lane a run of adjacent columns where they gave it lanesX-strided ones.
`storedPair` takes a PAIR index and not an element index so a kernel that has
not been rearranged has nothing to pass it.

🔴 **AND IT IS FREE WHERE THE CONSUMER ALREADY NARROWS.** The transition's
hidden activation is read by a kernel whose first act is `f16(source[...])`, so
storing it narrowed loses nothing already lost - the fold came back BIT
IDENTICAL, coordinates and all, 16 MiB lighter. Look for that shape first.

🔴 **AND BOTH FAILURES WERE SILENT, BECAUSE EVERY SHAPE STILL AGREES.** A
packed tensor and an f32 one of the same element count differ only in bytes,
which nothing validates. Reading `normalized` as f32 in the pair-bias shader -
it is `normalized` itself for the triangle attentions, and a separate tensor
only for an MSA row attention - folded 59 residues at **pLDDT 27 with 5.3 A
between consecutive alpha carbons**. Failing to thread `outputStorage` through
`selectAttentionProjectKernel` had the projection write f32 where the flash
kernel read packed, and the fold came back **NaN**. The unit test written to
catch the second passed, because it compared cache KEYS and they already
differed on the source storage: assert on the generated WGSL.

🔴 **AN AF3 FOLD'S PEAK IS IN THE CONFIDENCE HEAD, NOT THE TRUNK OR THE
DIFFUSION.** It runs four more pairformer blocks after the sampler, so it
allocates the whole pair scratch again while the diffusion transformer's 378 MiB
of resident weights are still held and unreadable by anything.
`releaseResidentWeights(device, prefix)` gives a stage's residency back when the
stage is over - `"w."` after the trunk, `"difftx."` after the sampler - and took
a 272-token fold from **1214 to 671 MiB, 45%, for no time at all** and about half
a second on a REPEAT fold, which is the re-packing. Mean pLDDT identical to every
digit. Do this before reaching for kernels.

🔴 **AF3's PAIR SCRATCH IS NOT PACKED ANY MORE, AND THE PARAGRAPH THAT USED TO
BE HERE PRICED IT WRONGLY.** It recorded 1086.5 -> 896.1 MiB at 408 tokens for
no measurable time and called it "a trade taken for LENGTH". Two things were
missing from that. The COST was never measured on the checkers that could see
it - see the table below, and a factor of 1200 on the pair representation. And
the SAVING was mostly memory nothing was using: a seventh scratch buffer no
code ever read, a readback held across the whole block loop, and a sixth buffer
the grid attention did not need. With those three gone, unpacking costs 21.5
MiB of a 610.8 MiB peak at 300 tokens. What survives from that paragraph is
why AF2's packing DOES pay: its flash kernel re-reads its key and value once
per query tile, so halving the bytes pays for the unpacking twice over, and
nothing in AF3's pair track reads these more than once.

🔴 **AND WHO OWNS A WORD IS A DIFFERENT ANSWER IN EVERY KERNEL.** The layer
norms own whole rows and only had to walk words. `grid.project` gave a lane ONE
output channel, so it had to take a PAIR - twice the accumulators - and its row
tile had to fall from 8 to 4: bench-grid-project.js's `p` arms put packed at
8.21 ms against 8.16 at rows 4, **11.01 against 7.54 at 8, and 42.64 against
10.14 at 16**, which is the same register spill AF2's projection sweep records.
The triangle's a and b pair by CHANNEL instead, because they are channel-major
and `h * PAIRS + row` is odd at odd h when n is odd - n = 59 and n = 68 are the
two sizes checked here, one of each, so half the suite would have passed.
`scratch[3]` is still f32: it is the contraction's output, where `h` is group.z
and one workgroup owns one channel.

🔴 **AND A SUBSTITUTION ACROSS GENERATED SHADERS FAILS SILENTLY IN BOTH
DIRECTIONS.** Two of them in one file in one afternoon: one matched NOTHING,
because the indentation differed, and left a bias loop on the old column
mapping; one matched TWICE, because `tile_weight[k * TILE_COLUMNS + local.x +
column * 8u]` is in projectAB and in projectOutput, and broke the kernel that
was not being changed. `a` was right, the contraction was right, and the fold
came out at relRMS 1.42.

🔴 **AND A DIFFERENTIAL THAT TESTS TWO KERNELS CANNOT FIND A BUG IN THE FIFTH.**
tools/gpu/check-triangle-packed.js was wrong twice before it was right: first it
unpacked with the generic `i >> 1` layout while the kernel pairs by channel -
a permutation, reported as relRMS 1.39 against a correct kernel - and then,
corrected, it declared both kernels sound while the fold stayed broken, because
it ran a configuration nothing runs. Run the WHOLE update, and sweep the axes
the caller varies (`direction`, `accumulatePrecision`), or it is a check of
something else.

🔴 **A STORAGE FORMAT MEASURED ON ONE STACK IS NOT A FACT ABOUT THE OTHER
THREE.** `PAIR_SCRATCH_STORAGE` was a module constant that every caller of
`compilePairTrack` inherited, and it was measured on the pairformer's own
differential checker, which passes either way. FOUR stacks run that pair track,
and every other checker that reaches one was over its bound the whole time:

| | packed | unpacked | bound |
|---|---|---|---|
| `check-af3-confidence` stack pair | 3.71e-3 | **3.12e-6** | |
| ...its PAE head | 2.88e-3 | **5.75e-6** | 7.1x envelope |
| ...its PDE head | 3.29e-3 | **7.47e-6** | |
| ...its pLDDT head | 6.88e-4 | **1.16e-4** | |
| `check-af3-msa-block` | 1.82e-3 | **7.16e-6** | 1e-5 |
| `check-af3-template` | 3.79e-5 | **2.52e-7** | 2e-5 |
| `check-af3-trunk` pair | 1.04e-4 | **1.99e-5** | 4e-5 |

A factor of 1200 on the pair representation that feeds pLDDT and PAE. The
CONFIDENCE head is where it shows, because its four blocks amplify and its
heads have the tightest envelopes in the repository; the trunk's own checker at
n=24 barely moves, which is exactly why one checker is not enough.
`UNPACKED_PAIR_SCRATCH` is what `compilePairTrack` defaults to now, and all
four stacks take it. `PAIR_SCRATCH_STORAGE` stays exported and unused, with
that table beside it.

🔴 **AND A HALFWAY LAYOUT IS WORSE THAN EITHER, WHICH IS WHY IT WAS TRIED.**
Bisected on the trunk's pair term, changing only the MSA stack: `a` and `b`
cost 3x - they are MULTIPLIED against each other in the contraction, so their
rounding squares - `normalized` costs 1.6x, and `hidden` and grid attention's
output cost nothing measurable. Keeping only those two passes the TRUNK's bound
at 3.11e-5 and still misses the MSA block's by 50x. Half the memory is not
worth a checker that has to be told to expect less.

🔴 **AND THE END-TO-END NUMBER COULD NOT SEE ANY OF IT.** `fold.js --dump`
reports `pair vs AF3` at 4.03e-4 with the bad packing and 3.94e-4 without it,
with mean pLDDT 85.6 either way. Forty-eight pairformer blocks are contractive
enough to swallow a 1200x error in the term that feeds them, so the whole-fold
gate is the WRONG instrument for a change inside one stage - and it is the one
that gets run. Run the per-stage checkers when a stage changes.

🔴 **AND THE PAIR TRACK NEEDS FIVE SCRATCH TENSORS, NOT SEVEN.** `scratch[6]`
was never read by anything - `encodePairTrack` indexes 0 to 5, and so does
every caller - and `scratch[5]` did not need to exist either: `grid.project` is
the last pass that reads `scratch[0]` and it is encoded BEFORE the pass that
wrote `scratch[5]`, so the grid attention writes its output back into
`normalized`. 43.9 MiB each at 300 tokens.

🔴 **AND A READBACK BUFFER BELONGS AFTER THE SCRATCH, NOT BEFORE THE LOOP.**
Both pair-track stacks reserved their MAP_READ buffers up front and wrote them
once, at the end - a pair-sized buffer standing beside the scratch for a whole
48-block loop, at exactly the moment the trunk is fullest. Releasing the
scratch first is what makes the peak move, because this allocator does not
pool: release DESTROYS.

Those three together, on a 300-token trunk pass at 32 MSA rows:

| | peak | af3-block.scratch |
|---|---|---|
| packed, six buffers, readback in the peak | 589.3 MiB | 153.8 x6 |
| unpacked, six, readback in the peak | 699.2 | 263.7 x6 |
| unpacked, six, readback after | 654.8 | 263.7 x6 |
| **unpacked, five, readback after** | **610.8** | **219.7 x5** |

🔴 **AND A SAMPLER STEP CHANGES TWO INPUTS AND USED TO REBUILD EVERYTHING.**
The diffusion head is called up to two hundred times down one schedule, and
only the noisy coordinates and the noise level move. The per-atom conditioning,
the reference conformer, the ten gathers, the trunk's pair and single, the
encoder's query and key conditioning and masks, and the pair logits derived
from them are the FOLD - all of it was rebuilt on the host and written across
the bus once per step, and three tensors derived from it were recomputed on the
GPU for the identical answer. `bench-head.js --profile` medians nine calls in
one process, which is what to measure this with:

| | 59 tokens | 200 tokens |
|---|---|---|
| before | 86 ms | 253 ms |
| after | **71** | **206** |

The mechanism is `persistent` beside `persistentUpload` in the atom encoder and
decoder - the first keeps a tensor the blocks WRITE, the second keeps one they
READ - plus `reusePair` in the conditioning module and `#pairNorm` in the
transformer. The build closure is not called on a cache hit, so the host-side
gathering inside it does not run either.

🔴 **AND THE ENCODER HANDS THE DECODER DEVICE BUFFERS, NOT ARRAYS.** Its five
static tensors were read back across the bus and uploaded again to make a
second copy the peak then carried beside the first: 17 MiB at 59 residues.

🔴 **WHAT IS LEFT IN A DENOISER STEP IS THE FOUR HOST-DEVICE ROUND TRIPS.** At
59 tokens the stages sum to 71 ms and the labelled compute passes to about 52;
the rest is one submit and one `mapAsync` per stage, because the head chains
conditioning -> encoder -> transformer -> decoder through Float32Arrays.
Caching the transformer's bind groups and scratch tensors bought nothing
measurable against that - the stage sat at 45-46 ms either way - so the next
thing there is chaining the stages ON THE DEVICE, not another cache.

🔴 **AN ATTENTION'S OUTPUT CAN LIVE IN ITS NORMALISED INPUT, AND THAT IS TRUE
IN BOTH MODELS.** The shape is the same everywhere: normalise into a tensor,
project it into q/k/v/gate, attend into a fresh one, project out. The
projection is the LAST pass that reads the normalised tensor and the attention
is the NEXT pass to write, so they can be one buffer. Worth, per attention, one
pair- or MSA-sized tensor:

| | peak before | after |
|---|---|---|
| AF3 trunk, 300 tokens | 654.8 MiB | **610.8** |
| AF2, 512 MSA rows | 396.4 | **365.2** |
| AF2, 128 rows | 156.1 | **147.1** |

`tools/gpu/fold-af2.js`'s checksum is unchanged at both depths and a 68-token
AF3 fold is bit-identical, which is what says the aliasing is real and not a
race.

🔴 **AND ONLY WHERE THE TWO AGREE ABOUT THE ELEMENT.** AF2's normalised tensor
is always packed and its projected ones are packed only where the
register-resident flash kernel accepts them; where it does not, one is half the
bytes of the other, and sharing would hand a shader a buffer of the wrong
length - which is not something WebGPU can catch. The fallback allocates a
second tensor.

🔴 **AND A READBACK BUFFER IS THE OTHER HALF OF THE SAME HABIT.** Anything
written once at the END of a stack should be allocated there, not beside the
scratch at the top - see the trunk note above. Where the copy is encoded into
the same command buffer as the work (the template embedder, the input
embedder) it cannot be moved without splitting the submit, and those stages are
not the peak.

🔴 **PREPARING AN AF2 ALIGNMENT WAS 525 ms OF MAIN-THREAD JAVASCRIPT AND
NOBODY HAD MEASURED IT.** Three loops, none of them subtle, all of them once
per residue: `parseA3m` ran a regex and a `toUpperCase` per character and built
each row by concatenation; `makeA3mFeatures` looked each residue up in a `Map`
through a one-character string; and the nearest-centre assignment - extras x
centres x residues, 1024 x 508 x 59 - ran once per RECYCLE.

| | before | after |
|---|---|---|
| `parseA3m`, 30,000 rows | 307 ms | **85** |
| `makeA3mFeatures`, 200 residues x 10,000 rows, one pass | 403 | **91** |
| ...`tools/fixtures/test.a3m`, four passes | 525 | **75** |

`tools/gpu/fold-af2.js`'s checksum is unchanged at -2105827, which is what
says the clustering still clusters the same way.

🔴 **AND `(x - 0x01010101) & ~x & 0x80808080` IS THE WRONG ZERO-BYTE TRICK IF
YOU ARE COUNTING.** It is the one everyone reaches for and it is exact only for
"is there a zero byte ANYWHERE": a borrow out of a zero byte marks its
neighbour too. Used to count agreeing residues it changed 1024 assignments'
checksum from 195329 to 199057 - a wrong answer that still looks like a
histogram. `~(((x & 0x7f7f7f7f) + 0x7f7f7f7f) | x) & 0x80808080` has no borrow
between bytes.

🔴 **AND THE OTHER TWO PREP PATHS ARE NOT WORTH TOUCHING, MEASURED.** AF3's
`featuriseProtein` is **1 ms** at 200 tokens, and `perAtomConditioning` - which
fold.js's own comment calls out as 119 ms - is **4 ms at 59 tokens and 17 at
240**. That comment is stale; the one-hot it describes was fixed. Writing the
archive is 28 ms for a 2 MB alignment.

🔴 **QUANTISED WEIGHTS CAN BE DECODED ON THE GPU, AND IT IS 3.7x.** The path to
a resident f16 buffer used to be: decode int5 into float32 on the main thread,
narrow the lot into a Float16Array, upload. `src/runtime/quantised-upload.js`
uploads the CODES instead - an eighth of the bytes - and decodes them into the
destination with one dispatch per tensor. 437 ms of host packing becomes 119 of
compute for the diffusion transformer's 24 blocks, and a real page fold went
**3.31 s to 2.30**. `src/af3/device-weights.js` is the shared entry point;
docs/AF3.md has the per-packer table.

🔴 **AND BIT-IDENTITY WAS THE FIRST THING MEASURED, NOT THE LAST.** JavaScript
computes `code * scale + zero` in f64 and WGSL has no f64. The product is exact
in both - a 5-bit code times an f16 scale needs at most 16 mantissa bits - but
the SUM can need more than f32's 24. `tools/gpu/check-int5-gpu.js` answered it
on 131,072 synthetic elements spanning 10^-4 to 10^4 before any of the
plumbing existed; `tools/gpu/check-block-upload.js` answers it on whole real
blocks against the shipping packer. Both read **0 differ**.

🔴 **AND ON A LAZILY BOUND WEIGHT OBJECT, `.length` IS THE DECODE.** Reading
`block[name].length` materialises that tensor. `blockWeightOffsets` existed
precisely to avoid building a buffer and was doing it anyway; the device
planner would have undone its own point; and in the checker it silently made
the host arm WARM and flattered the GPU by 234 ms of work it had itself caused.
`stacked` records the range it will read, and that is the length.

🔴 **AND `Float16Array.set` FROM A Float32Array IS NOT A MEMMOVE.** 8M elements
measure 9.4 ms through `set`, 6.1 through a plain loop and 4.4 unrolled eight
ways - bit-identical. `writeInto` in src/runtime/float16.js is that loop, and
it leaves same-element copies to `set`, which really is a memmove. On real
shapes it is 26% of the narrowing rather than 52%: a block is forty tensors
averaging 200k elements, so per-call overhead is a much larger share than the
microbenchmark suggests.

🔴 **AND `node tools/gpu-chrome.mjs` SOMETIMES DOES NOT EXIT.** The results file
is complete and correct and the node process sits there with a headless Chrome
still running, which in a `for` loop stalls every arm behind it. `pkill -9 -f
"gpu-chrome-"` matches the temporary profile directory and nothing else - not
the browser you are using. A batch of checkers should carry one between arms.

🔴 **AND `memorySnapshot`'s `byLabel` IS CUMULATIVE, WHICH IS THE WRONG
QUESTION.** It sums every allocation a label ever made, so a scratch tensor
taken and returned once a block reads as forty-eight times its size - that is
what CHURNS. `peakByLabel` is what was on the device when it was fullest and
its rows sum to `peakBytes`; that is what says which tensor to attack, and it
is what said ten tensors of 29.5 MiB were 295 MiB of a 552 MiB fold.
`tools/gpu/fold-af2.js` prints both.

## ESMFold2's trunk: AF3's pair track, minus two of its five

🔴 **IT IS AF3's PAIRFORMER BLOCK WITH THE GRID ATTENTIONS AND THE SINGLE TRACK
REMOVED, AND THAT IS MEASURED.** `tools/check-esmfold2-trunk.js` composes
`src/af3/pairformer-reference.js`'s three surviving pieces 24 times and scores
them against the values the native model recorded going into and coming out of
its trunk at each of its four recycles: **relRMS 1.4e-6** against a 2e-4 bound.
So the port needed no new arithmetic, only the weights in AF3's shapes -
`tools/esmc/esmfold2_trunk_weights.py`, exported by
`tools/export_esmfold2_trunk.py`.

🔴 **AND THE TWO TRIANGLES NEED OPPOSITE CONVERSIONS, WHICH NOTHING IN THE
SHAPES SAYS.** ESMFold2 runs both directions through ONE engine and tells them
apart by which half of `proj_bundle` is the left operand; AF3 keeps the halves
fixed and changes the einsum. The modules are the same class with the same
shapes, so a converter that treats them alike gets the incoming one exactly
backwards. Swept rather than read, by `tools/check-esmfold2-modules.js`:

| | halves in order | halves swapped |
|---|---|---|
| `tri_mul_out` -> outgoing | **2.83e-7** | 3.69e-1 |
| `tri_mul_in` -> incoming | 3.24e-1 | **2.86e-7** |

Every wrong-DIRECTION arm scores 0.32 or worse, which is what says the sweep
discriminates rather than blessing whatever it was handed.

🔴 **AND `pair_transition` RETURNS ITS RESIDUAL WHILE THE TRIANGLES RETURN
THEIR DELTA, IN THE SAME BLOCK.** `x + ffn(norm(x))` against
`proj_emit(...)` - so a correct transition scored against the recorded output
reads **9.01e-1**, which looks exactly like wrong arithmetic and sent an
afternoon into re-reading two implementations that already agreed.
`rms(output - input)` 1.48 against `rms(output)` 12.3 is what settled it. Ask
what a recorded tensor IS before scoring against it.

🔴 **AND THE GRID ATTENTION IS SKIPPED, NOT ZEROED.** An attention whose output
projection is zero adds zero, so a zeroed AF3 block already IS ESMFold2's block
and needs no graph code at all. It is also pure waste: `grid.attend` is the
largest kernel in an AF3 trunk and this trunk runs 24 blocks four times over.
`compilePairTrack`/`encodePairTrack` take a `gridAttention` flag (default true,
so no existing caller moves) and `src/esmfold2/trunk-webgpu.js` sets it false.
Worth **1.42x**, and `tools/gpu/check-esmfold2-trunk-gpu.js` runs both arms over
the same weights at the same shapes and asserts **0 differing elements** - not a
tolerance, because dropping passes from a track whose five updates each read the
pair as the last one left it is precisely the change that returns a plausible
tensor. The zeroed weights are synthesised by the checker; shipping them would
be 37.7 MiB of zeros in the bundle.

🔴 **AND THE TWO f16 KNOBS ARE PRICED VERY DIFFERENTLY, SO THIS TRUNK ANSWERS
DIFFERENTLY FROM AF3's.** The transition's staged tiles and the triangle
projection's eight vec4 of accumulators are separate kernels, and AF3 narrows
both. Separated - error against the native model at 40 residues, time swept by
`bench-esmfold2-trunk.js`:

| staged : accumulate | relRMS | 40 tokens | 150 | 300 |
|---|---|---|---|---|
| f32 : f32 | **1.10e-6** | 1.000x | 1.000 | 1.000 |
| **f16 : f32** (shipped) | **5.46e-4** | 1.163 | 1.240 | **1.306** |
| f32 : f16 | 2.62e-3 | 1.102 | 1.088 | 1.151 |
| f16 : f16 | 2.68e-3 | 1.348 | 1.466 | 1.480 |

The ACCUMULATOR carries 96% of the error and returns the smaller half of the
speedup, at every length. AF3 keeps it because it was priced on that kernel
alone - 1.55x on `bench-triangle-project.js` at 118 tokens - rather than against
the other knob. **Price a precision knob against the other knobs, not against
f32.**

🔴 **AND IT FOLDS END TO END NOW: SEQUENCE IN, ESMFold2's PAIR REPRESENTATION
OUT.** `tools/check-esmfold2-fold.js` runs the whole assembly - ESM-C's 37
hidden states, the shim's pair, all five terms of `z_init`, and four loops of 24
blocks - against the native model's own per-loop values:

| | into the trunk | out of it |
|---|---|---|
| loop 0 | 4.12e-5 | 7.93e-5 |
| loop 1 | 4.17e-5 | 7.98e-5 |
| loop 2 | 4.17e-5 | 7.81e-5 |
| loop 3 | 4.17e-5 | 7.73e-5 |

...and on to the distogram head, which is the trunk's one output today:
**4.82e-5**, against an unsymmetrised control at 5.29e-1. The floor throughout
is the atom attention's own bfloat16, which is why the bound is 2e-3 here and
2e-5 against a `--float32-attention` dump.

🔴 **AND THE DISTOGRAM HEAD SYMMETRISES, WHICH IS PART OF THE HEAD.**
`distogram_head(z + z.transpose(-2, -3))` - a distance is symmetric and the
trunk's pair is not. Feeding it `z` alone conforms in shape and returns a
plausible distogram, which is why the checker runs that as a control rather than
trusting the reading. **The bin EDGES are not in this checkpoint's config**: 128
bins and no stated range, while the (absent) confidence head's config carries
2.0 to 52.0 for its own 128. So the port returns LOGITS and leaves distances to
a caller with real edges, rather than presenting a guess as a fact.

🔴 **AND IT EXISTS BECAUSE EVERY PER-MODULE CHECK CAN PASS WHILE THE ASSEMBLY IS
WRONG.** The featuriser's checker says each term of `z_init` matches and the
trunk's says the 24 blocks do. Neither says the five terms are SUMMED, that the
language model's pair reaches them, that the loop runs `num_loops + 1` times, or
that `z` starts at zero - and every one of those is a plausible tensor when
wrong. `--esmc <dir>` is what makes the language model's term exist at all: the
checkpoint does not carry ESM-C (`esmc_id` names a separate 2.3 GB artefact), so
the dump injects `lm_hidden_states` from `tools/esmc/esmc_forward.py` - the same
tower the WebGPU port is checked against. **And a dump without it is not "the LM
contributing zero"**: the shim's biases make `lm_shim(0)` non-zero, so
substituting zeros is a different model. The checker refuses such a dump rather
than quietly scoring four terms of five.

🔴 **THE INPUTS EMBEDDER IS NOT AF3's ATOM ENCODER, AND REUSING IT WOULD HAVE
BEEN THE OBVIOUS WRONG MOVE.** AF3 runs 32-query/128-key windowed atom attention
biased by a pair representation. ESMFold2 runs plain **sliding-window
self-attention** over atoms, half-window 64, whose only positional signal is a
**3D rotary embedding built from the reference conformer**: `ref_pos` x 3 axes x
2 pairs at base 20, plus `ref_space_uid` x 10 pairs at base 10000, filling
`head_dim / 2 = 16` exactly. The window is over **rank among valid atoms**, not
raw index, the diagonal is always allowed, the blocks are adaLN-Zero with
**affine-free RMSNorm**, and q/k take a second affine-free RMSNorm before the
rotation. Nothing in the shapes says any of this: both are "an atom transformer
at 128 channels". `src/esmfold2/atom-encoder-reference.js`.

🔴 **THE DIFFUSION MODULE IS `structure_head`, AND ITS CONDITIONING IS PORTED.**
345 tensors, and the shape of it: conditioning, then the SAME SWA atom encoder
the inputs embedder uses (with a `coords_linear` of 6 = `r_l | pred_r1`
appended), a 12-block 16-head token transformer at 768 channels, an atom
decoder, and 15 EDM steps at `sigma_data` 16. So the expensive primitive was
already done. The conditioning measures **9.37e-8** on the pair and **7.98e-8**
on the single against the module's own first call.

🔴 **AND ITS TRANSITIONS DO NOT FUSE THEIR GATE, WHERE EVERY OTHER TRANSITION
HERE DOES.** AF3's and ESMFold2's own pair transition pack both halves into one
`w12` and split it, so the only question is which half is the gate.
`TransitionLayer` has `a_proj` and `b_proj` as two Linears and computes
`out_proj(silu(a) * b)`. Reading it as a fused pair indexes one matrix at half
its stride. Swapping the two scores **3.41** and **3.54**, which is the control
that says the checker can see it.

🔴 **AND THERE IS NO `s_trunk` ANYWHERE IN THIS MODEL.** The conditioning takes
one and is handed `None`: the trunk has no single track, so `s_inputs` - the 451
channels the inputs embedder produced - is the only single representation in the
graph. An AF3-shaped port reaches for a trunk single, does not find one, and
synthesising a zero is a different model. The pair inputs are **concatenated**
(`z_input_norm` is 512 wide, which is what says so), not added, and `z` is
cached across the sampler's fifteen steps while `s` is not - only `s` depends on
the noise level.

🔴 **AND THE TOKEN TRANSFORMER IS PORTED TOO: 1.48e-6 ACROSS TWELVE BLOCKS.**
Attention biased by the pair, then a conditioned transition, both wrapped in
adaLN-Zero. Four things in it are shaped so that getting them wrong returns a
plausible tensor:

* **The two LayerNorms in adaLN are not the same kind.** The activation's is
  affine-FREE and the conditioning's has a learned SCALE and no bias. There is
  one weight vector between them and it belongs to `s`.
* **The softmax is over the key axis of an `(i, j, head)` tensor** - `dim=-2`,
  not the last. Over the heads instead it still sums to one.
* **There are two gates, from different things.** `g_proj` gates the per-head
  context from the adaLN-MODULATED activation; `out_gate` gates the whole output
  from the CONDITIONING SINGLE, and only the second has a bias (initialised to
  -2, so a fresh block starts nearly closed).
* **`attn_blocks` and `transition_blocks` are two ModuleLists that the forward
  ZIPS**, not one alternating list. A flat export interleaving them loads the
  right count of the wrong things.

🔴 **AND THE TOKEN TRANSITION FUSES ITS GATE WHILE THE CONDITIONING'S DOES NOT,
IN THE SAME MODULE.** `lin_swish` is one Linear of `2 * hidden` split in half,
gate first; `DiffusionConditioning`'s `TransitionLayer` has `a_proj` and
`b_proj` as two Linears. Both are "a SwiGLU transition in the diffusion module"
and they are packed the two different ways. **Read the shapes, never the
family.**

🔴 **AND IT FOLDS.** `tools/fold-esmfold2.js` runs every stage LocalFold owns -
the featuriser, the atom encoder, 24 trunk blocks four times over, the
conditioning, the token transformer, the atom decoder and eleven sampler steps -
and writes a structure. 40 residues, 6.5 minutes on the CPU:

| | |
|---|---|
| CA-CA spacing | **3.809 A** (min 3.787, max 3.853) |
| RMSD to the native fold | **0.598 A** after superposition |

The spacing is the geometry gate: 3.8 A is a peptide bond, and a port that had
the arithmetic subtly wrong would produce a plausible-looking cloud with the
wrong scale. The RMSD is the fold gate, and **0.6 A is agreement, not identity**
- the sampler draws a rotation, a translation and a noise vector per step from
torch's RNG and this used its own, so the two are independent SAMPLES. That they
land within 0.6 A of each other is the model being confident, and it is the
strongest end-to-end statement available for a stochastic sampler.

🔴 **AND THE PAIR CONDITIONING IS CACHED ACROSS THE STEPS WHILE THE SINGLE IS
NOT.** Only `s` carries the noise level; `z` does not depend on `t_hat` at all,
which is why upstream keeps it in `inference_cache["z"]` and rebuilds `s`.
Caching both freezes `t_hat` at step zero - eleven steps that all think they are
the first - and still converges to a structure.

🔴 **THE DENOISER IS CHECKED AT EVERY NOISE LEVEL THE SAMPLER VISITS, NOT ONE.**
Eleven steps spanning five orders of magnitude, teacher-forced on the model's
own `x_noisy` so each is an independent `f(x) == y` rather than a trajectory
whose first error contaminates the rest:

| t_hat | 411 | 278 | 142 | 68.7 | 30.8 | 12.6 | 2.89 | 0.918 | 0.241 | 0.048 | 0.0064 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| relRMS | 1.5e-4 | 1.1e-4 | 1.6e-4 | 6.1e-5 | 4.5e-5 | 6.3e-5 | 1.6e-4 | 1.9e-4 | 1.6e-4 | 1.4e-4 | 3.4e-5 |

Flat, at the atom attention's bfloat16 floor throughout. **One level would have
said very little**: the EDM preconditioning weights the network's output by
`sigma*t/sqrt(sigma^2+t^2)`, which at the last step is about 0.0064 - so a badly
wrong network still scores well there, and the first step is where it is
load-bearing.

🔴 **AND A WHOLE DENOISE STEP NOW RUNS AT 1.51e-4.** Conditioning, the atom
encoder with the noisy coordinates, the token transformer, the atom decoder and
the EDM preconditioning - checked against the module's own first call out of the
sampler's fifteen. The floor is the atom attention's bfloat16 again, six SWA
blocks of it this time.

🔴 **THE NOISY COORDINATES ENTER THE ACTIVATION AND NEVER THE CONDITIONING.**
`q` starts at `c_base + coords_linear([r_l | pred_r1])` while `c` stays
`c_base`, so the atom stack is conditioned on the reference conformer alone and
only its running activation knows where the atoms currently are. Adding the
projection to both is the natural-looking symmetry and a different model.
`pred_r1` is ZEROS when there is no previous prediction, not absent: the
projection is six channels wide either way, and feeding it three reads the
second half of the matrix at the wrong offset. Measured, the same step: with no
coordinate term at all **1.23e-1**, with the coordinates unscaled **1.02e+2**.

🔴 **AND THE LAST LINE IS EDM PRECONDITIONING, NOT A RESIDUAL.**
`out = sigma^2/(sigma^2+t^2) * x_noisy + sigma*t/sqrt(sigma^2+t^2) * r_update` -
at a large noise level the first term is nearly zero and the answer is almost
all network, at a small one almost all input. Writing it as `x_noisy + r_update`
runs and converges to something.

🔴 **AND `ref_element` IS INDICES TO THE FEATURISER AND A ONE-HOT TO THE MODEL,
WHICH COST AN HOUR.** The module is handed `ref_element` at (atoms, 128) and
`ref_atom_name_chars` at (atoms, 4, 64), already one-hot; the featuriser
produces them as indices at (atoms,) and (atoms, 4). Both reach a JavaScript
port as a flat typed array, and passing the one-hot makes `atomFeatures` read
its first `atoms` entries - all 0 or 1 - as element indices. Every shape
conforms, nothing throws, and the encoder came back at **relRMS 0.89 with corr
0.72**, which reads exactly like a wrong convention somewhere in three SWA
blocks. `atomFeatures` checks the LENGTH now and says which it wants.

🔴 **THE SAMPLER IS STOCHASTIC, SO "THE SAME STRUCTURE" IS NOT A GATE.** Every
step centres the coordinates, rotates them by a RANDOM rotation, translates them
by a random vector and adds Gaussian noise - four draws from torch's global RNG
per step - so a JavaScript port cannot reproduce its coordinates and a checker
that tried would be measuring the RNG. What is deterministic is checked exactly
instead:

| | |
|---|---|
| the noise schedule | 2.78e-7 over 16 entries |
| steps after truncation | 11, against the model's 11 |
| `t_hat` per step | 1.49e-6 worst relative |
| **the last step, against the fold** | **1.19e-7** |
| ...without the alignment (control) | 1.38e-3 |

🔴 **AND `max_inference_sigma` IS A DEFAULT ARGUMENT, NOT A CONFIG FIELD, AND IT
TURNS FIFTEEN STEPS INTO ELEVEN.** `sample(..., max_inference_sigma=256.0)`
drops every schedule entry above the cap and prepends the cap itself, so four of
the sixteen go. Reading `inference_num_steps` off the config gives a sampler
that takes four extra steps at noise levels the model never sees. **The step
count is an output of the schedule, not an input to it.**

🔴 **AND STEP i TAKES `gammas[i + 1]`.** Upstream zips `schedule[:-1]` with
`schedule[1:]` and `gammas[1:]`, so the churn applied to `sigma_tm` is decided
by the NEXT noise level. Off by one it still runs; the first step's `t_hat` is
the tell, 256 x 1.605 = 410.88 against 256.

🔴 **AND THE LAST STEP'S OUTPUT IS THE FOLD**, which is what makes any of this
checkable. No augmentation follows it, so `align(x_noisy, x_denoised)` plus the
update, on the recorded pair, must equal `sample_atom_coords` exactly - and it
does, at 1.19e-7. The alignment moves the NOISY copy onto the denoised answer
and not the other way round; swapped, it walks the structure away.

🔴 **AND THE 3x3 SVD IS py2Dmol's `svd3`, NOT A SECOND ONE.** The first version
written here tested for a zero singular value AFTER the square root, which
halves the exponent - so a numerically-zero eigenvalue of 8e-15 against 196
becomes 9e-8, clears any absolute floor, gets divided by, and leaves that column
of U non-orthonormal. py2Dmol's `src/io/math.js` records being bitten by exactly
that and guards it three ways: the floor is on the EIGENVALUE and is RELATIVE to
the largest, each recovered column is verified to be a unit vector, and any
column the division could not give is completed orthonormally against the ones
it could. A flat or linear point cloud is not a corner case for a sampler whose
input starts as noise. `test/esmfold2-kabsch.test.js` pins it - generic, flat,
linear, four scales, and a mirrored pair that must NOT be fitted exactly.

🔴 **AND THE FOURIER TABLE IS A BUFFER, WHICH IS STILL TRAINED IN.**
`register_buffer("w", randn(c))` is drawn once at construction and saved with
the checkpoint, so a port that redraws it gets a different model that runs. Both
`w` and `b` are exported.

🔴 **AND ITS ROTARY TABLE IS bfloat16 IN A float32 MODEL, WHICH IS WORTH 2.4e-3
AND LOOKS EXACTLY LIKE A CONVENTION BUG.** The whole module read **2.8e-4**
against the native one. Every primitive was then verified twice - against the
torch source and against ../alphafold3's independently written JAX reference -
and nine convention arms were swept (`half_window` 32/64/128, rank against raw
index, the diagonal, the rotation, `qk_norm`, three epsilons). **Not one of them
moved the residual.** What did was capturing the module's own intermediates,
which report:

    cos  bfloat16 [1, 320, 16]
    sin  bfloat16 [1, 320, 16]

`build_3d_rope` computes in float32 and the table reaches the attention at eight
mantissa bits. Rounding this reference's own table to bfloat16 reproduces the
native one on **all 5120 entries, bit for bit** - so the table was right the
whole time and only its PRECISION was not.

🔴 **AND THE ATTENTION ITSELF DOWNCASTS TOO, WHICH IS WHY THE CHECK HAS TWO
ARMS.** `SWA3DRoPEAttention.forward` runs `if q.dtype not in (float16,
bfloat16): q, k, v = q.bfloat16(), ...` unconditionally, so a float32 port
cannot agree with the shipping model below about 2e-4 however right it is.
`dump-esmfold2-trunk.py --float32-attention` neutralises the cast and is the arm
that says the arithmetic is right; the dump records which arm it is so a checker
cannot be handed the wrong bound. Measured, same code both ways:

| | control (f32 attention) | shipping (bf16) |
|---|---|---|
| the atom transformer's pooled output | **7.0e-8** | 2.1e-4 |
| one block's attention | 4.1e-7 | 2.4e-3 |

**A residual that survives every convention sweep is a PRECISION fact, and the
way to find it is to ask the module what dtype it is holding.**

🔴 **`model.py` IS A DIFFERENT MODEL FROM `experimental.py`, AND ITS RECURRENCE
IS NOT THIS ONE.** Both live in `esm/models/esmfold2/`, both define a
`folding_trunk` and a `z_init`, and the class this repository loads is
`EsmFold2ExperimentalModel` from the second. Reading the first gives:

```python
delta = F.softplus(self.parcae_log_delta)                    # model.py
a = torch.exp(-delta * torch.exp(self.parcae_log_a))
z = a * z + F.linear(self.parcae_input_norm(z_inject), b_mat)
```

a diagonal state-space recurrence with a learned decay, an input matrix, a
readout and a second `FoldingTrunk` as a coda. What the experimental model
actually runs is:

```python
z = torch.zeros_like(z_init)                                 # experimental.py
for loop_num in range(n_loops + 1):
    z = z_init + self.pair_loop_proj(z)
    z = self.folding_trunk(z, pair_attention_mask=pair_mask)
```

`pair_loop_proj` is `Sequential(LayerNorm, Linear)` with the Linear
**zero-initialised**, and `z` starts at zero rather than at noise. Neither file
names the other; both are "ESMFold2's trunk loop" to a reader, and porting the
wrong one gives a model that folds. Check the CLASS the checkpoint instantiates,
not the file with the likelier name.

🔴 **AND THE TRUNK IS THE EXPENSIVE HALF OF THIS MODEL, NOT THE TOWER.** ESM-C
600M folds 300 residues in 0.91 s. The trunk at 300 tokens is **8.7 s a loop and
it runs four loops** - 35 s - holding 534 MiB, because `d_pair` is 256 where
AF3's is 128 and 24 blocks x 4 loops is 96 block evaluations where an AF3 trunk
runs 48. Whatever "is this worth shipping" turns on, it is this number and not
the language model's.

🔴 **AND A TILE TUNED AT ONE CHANNEL COUNT IS NOT TUNED AT ANOTHER, WHICH COST
1.45x OF THE WHOLE TRUNK.** `transitionRowTile` was a function of the ROW COUNT
alone, and the row count is not what fills the workgroup: the staged block is
`tile * channels` plus `tile * chunk`, so the CHANNELS decide what a tile costs.
At AF3's 128 it picked 8, correctly; at ESMFold2's 256 it picked 8 again and
that is twice what fits well. It takes the channels now, holding both halves of
the staged block near **1024 floats** - a rule that reproduces every tuned value
already in the tree (the pair track's 8:128, the MSA stack's, the template
stack's, both diffusion transitions') and moves only the new shape. Verified by
hashing the generated WGSL at all seven shapes before and after: **six
byte-identical, one changed**, which is a stronger statement than a fold
fingerprint and free. `test/transition-tile.test.js` pins the table, because
every tile is bit-identical (relRMS 0) so nothing else would notice a
re-tuning.

The trunk at 300 tokens, profiled with `bench-esmfold2-trunk.js --profile`:

| pass | before | after | GFLOP/s after |
|---|---|---|---|
| `pair-transition` | 331.0 ms/block, 62.5% | **169.2, 45.0%** | 837 |
| `tri.project` | 84.8, 16.0% | 87.3, 23.2% | 1113 |
| `tri.project-out` | 60.7, 11.5% | 62.3, 16.6% | 777 |
| `tri.contract` | 33.5, 6.3% | 34.7, 9.2% | 825 |
| the two normalises | 19.8, 3.7% | 22.2, 5.9% | |
| **the trunk** | **12.9 s** | **9.2 s** | |

🔴 **AND THE TRIANGLE'S TILE IS *NOT* MIS-TUNED THE SAME WAY, WHICH IS WHY THIS
WAS WORTH CHECKING RATHER THAN ASSUMING.** The obvious next move after the
transition was to suspect every other kernel tuned at 128 channels.
`bench-triangle-project.js --tokens=300 --channels=256` says the shipped
**32x16 is already the best arm at 256** - 967 GFLOP/s on `project`, and best
combined with `project-out` (83.2 ms against 32x32's 87.8). A shape-dependent
tuning bug in one kernel is not evidence of one in its neighbour.

🔴 **AND HALVING THE STAGED BLOCK MADE f16 STAGING WORTH LESS, WHICH IS THE
TRADE MOVING RATHER THAN BREAKING.** Before the tile fix `f16:f32` was 1.306x
at 300 tokens; after it is 1.059x. The kernel was waiting on its staged tile,
the tile is now half the size, and narrowing what is no longer the bottleneck
buys what narrowing never bottleneck-bound bytes buys. The default stays f16 -
it is still free of any measurable accuracy cost at 5.46e-4 - but **a precision
trade priced before a tiling change is not a trade priced after it**.

## ESMFold2 on the GPU: what transferred, and the one kernel that did not

🔴 **IT FOLDS END TO END IN THE PAGE NOW, AND THE ONLY INPUT IS THE SEQUENCE.**
`tools/gpu/fold-esmfold2.js` runs ESM-C's 36 int3 blocks, the shim's pair term,
the featuriser, the inputs embedder, 24 trunk blocks four times over, the
conditioning, the token transformer, the atom decoder and the sampler.
Ubiquitin's first 40 residues, against the CPU fold that preceded it:

| | CPU, 6.5 min | GPU, 3.3 s |
|---|---|---|
| CA-CA spacing | 3.809 A | 3.806 A (3.781-3.838) |
| RMSD to ESMFold2 | 0.598 A | 1.048 A |

The GPU arm additionally runs the language model at THREE BITS where the CPU one
took its pair term from the dump, so 1.05 A includes the quantisation.

| length | time | peak | where it goes |
|---|---|---|---|
| 40 | 3.3 s | 521 MiB | the language model |
| 76 (ubiquitin) | 4.6 s | 578 MiB | |
| 150 | 9.7 s | 677 MiB | trunk 6.5 s, LM 1.9 s, sampler 0.8 s |
| 300 | 32.3 s | 992 MiB | **trunk 85%** |

🔴 **SO THE TRUNK IS THE MODEL AT ANY LENGTH WORTH FOLDING, AND THE LANGUAGE
MODEL IS NOT.** ESM-C 600M is 1.9 s at 150 residues and does not grow with n^2;
the trunk is 96 block evaluations at 256 channels where an AF3 trunk runs 48 at
128. Whatever "is this worth shipping" turns on, it is that number.

🔴 **THE ONE KERNEL WITH NO AF3 ANALOGUE IS THE SLIDING-WINDOW ATOM ATTENTION,
AND THE REASON IS THE POSITIONAL SIGNAL RATHER THAN THE SHAPE.** AF3's atom
encoder is 32-query / 128-key windowed attention biased by a pair
representation; this is plain sliding-window self-attention whose only
positional signal is a rotary embedding built from the REFERENCE CONFORMER.
Both are "an atom transformer at 128 channels".
`src/esmfold2/atom-transformer-webgpu.js` is four new shaders - `modulate`,
`prepare`, `attend`, `gated` - and everything that is a plain projection comes
from `src/esmc/block-webgpu.js`, whose tiled GEMM and fused SwiGLU are exactly
the shapes `ffnUp` and `lin_swish` are packed in.

🔴 **AND THE WINDOW IS RESOLVED ON THE HOST, BECAUSE IT IS OVER RANK.** Two
atoms 64 apart in the array are adjacent in rank if everything between them is
padding, so the allowed set is not `|i - j| <= 64`. Rank is monotonic, so the
allowed set IS a contiguous range and `atomWindows` computes the bounds once.
The validity test stays in the shader because a padded atom can sit INSIDE a
live range, and the diagonal is allowed unconditionally so a masked atom still
has something for its softmax to normalise.

🔴 **A DENOISE STEP IS RECORDED, NOT REBUILT, AND THAT IS THE WHOLE DESIGN.**
Every buffer a step reads is allocated in `prepare()`, so the pipelines, the
bind groups and the dispatch sizes are constants of the fold: a step is two
`writeBuffer`s, one submit of ~300 recorded passes and one readback. Cold and
warm are both 35 ms at 40 tokens, which is what says nothing is being rebuilt.
AF3's head learned this the expensive way; see the note above about its four
host-device round trips.

🔴 **THREE THINGS ARE CONSTANT ACROSS THE WHOLE SAMPLER AND ONLY ONE OF THEM
LOOKS IT.** The pair conditioning `z` does not depend on the noise level, which
upstream says out loud by caching it. Less obviously nor does the pair BIAS
every token block reads - twelve `(n, n, heads)` tensors derived from `z` alone
- and nor does `s_proj(s_input_norm(s_inputs))`, because the noise enters as a
broadcast vector ADDED after that projection. All three are built once. At 200
steps and 300 tokens that is 12 GFLOP a step against zero.

🔴 **AND THE PAIR CONDITIONING IS BUILT IN ROW CHUNKS.** Its transitions widen
256 channels to 512 and hold four tensors of that shape at once, which at 300
tokens is 640 MiB of scratch for arithmetic that is purely row-wise. Eight
thousand rows at a time costs nothing measurable and bounds it at 132 MiB. The
language-model shim and the distogram head are chunked the same way and for the
same reason.

🔴 **THE LANGUAGE MODEL WAS 26% HOST ARITHMETIC, AND PROFILING SAID SO WHERE
GUESSING WOULD NOT HAVE.** At 150 residues the fold spent 3.5 s in ESM-C and
`bench-esmc-tower.js` says the tower's COMPUTE at that length is 0.66 s. The
rest was the host, in two measurable places: **1121 ms** decoding int3 across 36
blocks and **723 ms** narrowing float32 to float16. Both are fixed and neither
needed a kernel:

* **The loop decoded between submits.** `await blockWeights(layer)` ran after
  the previous block's `onSubmittedWorkDone`, so host and GPU work strictly
  alternated. Asking for block N+1 BEFORE awaiting block N's submit puts the
  decode inside the window the device is busy in. One block ahead, not all of
  them - the point of streaming is that 2190 MiB of float32 never exists at
  once.
* **`readTensorAsFloat16` writes half precision as the codes are unpacked**, so
  there is one pass instead of two and no 2.3 GB of intermediate. It is not the
  same operation - float64 to float32 to float16 rounds twice - so it was
  measured: **0 of 14,894,208 elements differ**, and the tower's checker reports
  2.1473020359613994e-06 before and after, to every digit.

3476 ms to **2116**, and 1903 once the trunk stopped competing for the machine.

🔴 **AND AN ALREADY-NARROW ARRAY MUST NOT BE NARROWED AGAIN.**
`float32ToFloat16Array` returns a Uint16Array of BITS, so a second pass would
read the encoding as numbers. The tower's `narrow` passes it through and refuses
it outright if it was compiled for float32 weights; `scaled` refuses it too,
because dividing a Uint16Array elementwise divides the bits. No checkpoint here
scales its residual, which is exactly why that would have gone unnoticed.

🔴 **AND `BYTES` IN dtype.js AND `DTYPE_BYTES` IN build_site.py EACH KNEW ABOUT
ONE PACKED WIDTH, AND IT WAS int5.** Neither entry was read for anything but a
presence check, and the ESM-C bundle ships int3 - so a correct manifest failed
with "unsupported tensor dtype int3" from a reader that decodes int1 through
int7, and the site build told a reader to regenerate a file that would have come
out identical. Both derive the width from the name now.

🔴 **THE TRUNK'S TWO f16 KNOBS ARE BOTH ON, AND THE SECOND WAS PRICED AGAINST
THE SAMPLER RATHER THAN AGAINST A TENSOR NORM.** The accumulator carries 96% of
the error, which for a long time was the reason to leave it alone. What settled
it was folding the same 150-residue sequence twice at the same seed, changing
only this, and superposing:

| what changed | how far the structure moved |
|---|---|
| the SAMPLER'S SEED, nothing else | **6.32 A** |
| f32:f32 -> f16:f32 | 0.008-0.009 |
| f32:f32 -> f16:f16 | **0.034-0.041** |

0.04 A against a 6.3 A spread is a factor of 160: below the resolution of the
thing being predicted. Contact precision and recall are identical across all
three arms to three decimals.

🔴 **AND THE OLD DEFAULT WAS A LOSS AT 300 TOKENS.** Interleaved in one process,
which is the only instrument this machine's drift does not defeat:

| staged : accumulate | relRMS | 150 tokens | 300 |
|---|---|---|---|
| f32 : f32 | 1.10e-6 | 1.000x | 1.000 |
| f16 : f32 (the old default) | 5.46e-4 | 1.074 | **0.951** |
| **f16 : f16** | **2.68e-3** | **1.279** | **1.195** |

`f16:f32` measured 1.306x at 300 tokens BEFORE this session's transition-tiling
fix and 0.951 after: the staged tile is half the size it was, so narrowing what
is no longer the bottleneck costs the `f32()` at each read and buys nothing. **A
precision trade priced before a tiling change is not a trade priced after it** -
the second time that is true in this file.

🔴 **AND THE CHECKER STILL SEPARATES ALL THREE ARMS, WHICH IS THE OTHER HALF OF
TAKING IT.** Each is held to the bound its own arithmetic implies - 2e-4, 1e-3,
4e-3 - rather than one loose bound covering them, and the bound is chosen from
what the stack REPORTS having run rather than from what was asked for: the
`default` arm passes no options on purpose, so naming the shipped precisions in
the checker would make it agree with itself the moment they moved.

🔴 **AND AF3 IS UNMOVED BY EVERYTHING SHARED.** `terminalAtoms` defaults to the
AF3 rule, the dtype guard is a presence check, and the tower's prefetch is
ESM-C's alone. `tools/gpu/fold.js` on a 40-mer: pLDDT 77.36664729240613 and pTM
0.55038830675185 with the original `featurise.js` and `dtype.js` and with these,
to every digit.

🔴 **THE bf16 ARM IS MORE ACCURATE THAN THE f32 ONE, WHICH IS NOT WHAT A
PRECISION AXIS USUALLY MEANS.** `SWA3DRoPEAttention.forward` downcasts q, k and
v whatever the model's dtype, so an f32 attention is a more accurate computation
of something the checkpoint does not do. Against the module's own recorded call
(`tools/gpu/check-esmfold2-diffusion-gpu.js`):

| attention | relRMS |
|---|---|
| bf16, which is what the checkpoint does | **7.08e-5** |
| f32 (the control) | 1.52e-4 |

The control is that f32 is WORSE. If the two arms ever agree, the downcast has
stopped reaching the kernel - which a bound alone cannot see. And the f32 number
reproduces the CPU reference's 1.51e-4 to three digits, which is what says the
two paths are the same arithmetic.

🔴 **THE PAIR NEVER LEAVES THE DEVICE BETWEEN THE TRUNK'S LOOPS.** The trunk
driver takes a borrowed buffer (`state.buffer`) and the recycle projection
between the loops is itself a GPU pass. Four loops of upload-and-read-back would
be 736 MB of traffic at 300 tokens for a tensor nothing on the host touches.

🔴 **AND `z` STARTS AT ZERO WHILE `pair_loop_proj(0)` IS NOT ZERO.** Its Linear
is zero-INITIALISED upstream and then trained, and the LayerNorm in front of it
has an offset - so the first loop's input is `z_init` plus a real vector.
Skipping the projection on the first loop is the natural shortcut and a
different model.

## ESMFold2 does ligands, DNA and RNA - and this port read the config and said otherwise

🔴 **THE CONFIG IS ABOUT MSAs AND CONFIDENCE, NOT ABOUT CHEMISTRY.**
`disable_msa_features: true` and `confidence_head.enabled: false` are true and
say nothing about what the model can hold. ESMFold2's own constants:

    MOL_TYPE_PROTEIN 0  DNA 1  RNA 2  NONPOLYMER 3
    PROTEIN 2..21 (UNK 22)   RNA 23..27   DNA 28..32

and `prepare_input.py` has `tokenize_ligand_ccd` and `tokenize_ligand_smiles` at
one token per heavy atom. The 33-class `aatype` reverse-engineered from a
protein dump had its whole tail read as padding when it is RNA and DNA.
**Templates really are absent**: `grep -rn template` over the whole upstream
package returns nothing, and `z_init` has five terms with none of them one.

🔴 **SO THE FEATURISER IS AN ADAPTER OVER AF3's, NOT A SECOND FEATURISER.** AF3
already tokenises complexes, nucleic chains, ligands from the CCD and modified
residues at one token per atom, and ESMFold2 uses AF3's all-atom representation
term for term. What is genuinely different is three things:

1. **The atom layout is RAGGED**, not 24 dense slots a token. Reading AF3's
   slots in increasing order reproduces conformer order exactly - checked, no
   conformer in either table has non-monotonic slots.
2. **The alphabets differ and both are one-hots**, so neither complains. AF3's
   twenty amino acids map to ESMFold2's by **`+ 2`** - both are alphabetical by
   THREE-letter code - and nothing after them does: AF3's gap at 21 has no slot,
   its RNA runs 22-25 against 23-26, its DNA 26-29 against 28-31.
3. **There is no terminal atom.** No OXT, no OP3; `terminalAtoms: false` is the
   new option on `featuriseProtein`, and it is what takes ubiquitin's first 40
   residues from 312 atoms to the 311 the model was handed.

`tools/check-esmfold2-featurise.js` holds all fifteen discrete features to
IDENTICAL against the dump. `ref_pos` cannot match and says so - both models
draw a torsion per residue instance - so the check there is the N-CA bond
instead (1.4737 A against 1.4656).

🔴 **AND THE LANGUAGE MODEL NEEDED THREE THINGS A MONOMER NEVER SHOWED.**

* **Only PROTEIN tokens are sent.** `protein_mask = (mol_type == 0) & token_mask`,
  and a nucleotide's or ligand atom's hidden state stays ZERO - which is not
  absent, because the shim's LayerNorm has an offset and its downprojection a
  bias, so `shim(0)` is a fixed non-zero vector. `shimSingleForZeroState` is
  that one row of arithmetic.
* **An atom-tokenised residue collapses to ONE LM row** - several structure
  tokens share one `(asym_id, residue_index)` - and the answer scatters back.
* **A complex is ONE packed run** `[BOS] A [EOS BOS] B [EOS]` with a per-chain
  `sequence_id` mask.

🔴 **AND "RUN ESM-C ONCE PER CHAIN" IS THE OBVIOUS WRONG MOVE.** The attention
IS per chain - `seq_id[i] == seq_id[j]` excludes every cross-chain key - but the
ROTARY POSITIONS ARE ABSOLUTE over the packed array with no per-chain reset, so
chain two's first residue sits at `len(chain one) + 3`. Running the tower
separately per chain gives it position 1: the same shapes, a plausible tensor, a
different phase on every head. `createAttentionShader`'s `chainAware` arm is the
mask, and it is part of the pipeline key so a page that folds a monomer and then
a complex at the same length cannot reuse the wrong shader.

Measured, one fold each:

| input | tokens | geometry |
|---|---|---|
| 40-mer (the regression) | 40 | CA-CA 3.806, RMSD 1.050 A |
| two protein chains | 65 | CA-CA 3.786 (3.739-3.827) |
| protein + 12-mer DNA + glycerol | 58 | CA-CA 3.794, **O3'-P 1.588 A** |

The phosphodiester bond is the nucleic gate the way 3.8 A is the peptide one: a
covalent distance no torsion can change.

🔴 **AND A CA-CA METRIC THAT WALKS THE ARRAY IS WRONG ON A COMPLEX.** The first
two-chain run reported a 22.9 A maximum, which is the distance across the chain
break - the metric walking off the end of chain one, not a broken fold.

🔴 **AND `float32Tensors` IS NOT OPTIONAL FOR THIS BUNDLE.** `quantize_af3.py`
keeps a tensor whose name ends in `/scale`, `/offset` or `/bias`, which is how
AF3 spells its norms; this export spells them `leftNormInputScale`, `gateBias`,
`singleScale` - so **350 of its 377 vectors would have been group-quantised**,
and a 128-wide LayerNorm scale at group 32 is four scales carrying the tensor
whose job is to set the scale of everything after it. The Fourier table goes in
the list too, and it is not a norm: `w` and `b` are read inside
`cos(2 * pi * (t * w + b))`, so an error in them is an error in a PHASE.

| bundle | size | RMSD to ESMFold2 |
|---|---|---|
| float32 | 651 MiB | 1.050 A |
| **int5 g32** | **122.5 MiB** | 1.238 |
| int4 g32 | 102.2 | 1.750 |

With ESM-C at int3 (224 MiB) the pair is **347 MiB**, against AF3's shipped 265.

🔴 **AND `BYTES` IN dtype.js LISTED `int5` AND NOTHING ELSE PACKED.** The entry
is never read - the packed branch returns before it - so it was doing nothing
but satisfying a presence check, and the ESM-C bundle's int3 failed it with
"unsupported tensor dtype int3" from a reader that decodes int1 through int7.
Found only by loading the bundle through `HttpTensorStore`, which is the page's
path and not any checker's.

🔴 **AND THE PAGE OFFERS NO FLOW ARM, BECAUSE FLOW BUYS NOTHING FOR THIS
MODEL.** For AF3 the flow/diffusion switch is a twelve-fold saving: that model's
diffusion default is 200 steps and flow-16 is sixteen. ESMFold2's own sampler is
**eleven** steps - `inference_num_steps: 15` truncated by
`max_inference_sigma = 256` - so there is nothing to escape from, and `gamma0 = 0`
stops noise being re-injected without making a step cheaper. Counted:

| preset | asked | steps actually run |
|---|---|---|
| diffusion-15 (the checkpoint's own) | 15 | **11** |
| flow-16 | 16 | **12** |
| diffusion-32 / flow-32 | 32 | 23 each |

The flow arm at its usual setting runs MORE steps than the shipped sampler, for
a sampler the model was not trained with. `SAMPLER_PRESETS` keeps them - the
measurements are worth having - and the page shows only the step dial. **A
choice whose every option is equivalent-or-worse is the same fault as a control
that is ignored.**

🔴 **AND THE MODE IS FORCED IN CODE, NOT ONLY HIDDEN.** That is the lesson the
MSA row taught an hour earlier: hiding a control does not change its value, and
the shared `#af3-mode` select still reads "flow" behind a hidden row.
`samplerPreset` reads `ESMFOLD2_SAMPLER_MODE` and never the select.

🔴 **THE SAMPLER PRESETS ARE PRICED, AND SIX STEPS IS NOT "FASTER".** Measured
on the 40-mer against ESMFold2's own fold, one seed each:

| preset | steps run | CA-CA | RMSD |
|---|---|---|---|
| diffusion-8 | 6 | **58.0 A** | **48.1 A** |
| flow-8 | 6 | 4.27 | 1.72 |
| diffusion-15 (shipped) | 11 | 3.806 | 1.05 |
| flow-16 | 12 | 3.804 | 1.01 |
| diffusion-32 | 23 | 3.802 | 1.14 |
| diffusion-200 | 138 | 3.808 | 1.48 |

A peptide bond is 3.8 A, so `diffusion-8` is not a structure. The churn is what
breaks - `gamma0` re-noises to `sigma * 1.605` and `step_scale` 1.638 overshoots
- and at six steps the levels are too far apart for either to be corrected. The
flow arm re-noises not at all and is merely poor. **More steps than the schedule
buy nothing**: 138 is no better than 11 and eight times the time.

🔴 **AND THE SEED MEANS SOMETHING DIFFERENT IN EACH MODEL, WHICH IS WHY THE
ARCHIVE'S SETTINGS ARE PER MODEL.** AF2's is load-bearing before the model
runs: `a3m-features.js` shuffles the extra pool and BERT-masks 15% of the
centre positions, four ways - 0.7 mask, 0.1 profile, 0.1 same, 0.1 uniform - so
two seeds are two different INPUTS. AF3's drives the diffusion sampler.
EF2-fast's drives its sampler alone.

🔴 **AND EF2-fast DOES NO INPUT MASKING, WHICH THE CONFIG'S OWN DOCSTRING WOULD
TALK YOU INTO.** `EsmFold2Config` carries `lm_mask_pct` - "Fraction of sequence
residues randomly replaced with the LM mask token before running the PLM
backbone, matching the training-time input corruption" - and its docstring says
**"Single-sequence checkpoints set this to 0.1"**. This checkpoint IS a
single-sequence one (`disable_msa_features: true`) and does NOT set it:

| knob | base600M-step1500k |
|---|---|
| `lm_mask_pct` | **absent, so the 0.0 default**, and `hf_adapter` guards it with `if lm_mask_pct:` |
| `lm_dropout` | **0.0** |
| `force_lm_dropout_during_inference` | **False** |

So the only randomness in an EF2-fast fold is the sampler's rotation,
translation and noise. A port that took the docstring's word for it would mask a
tenth of every sequence before ESM-C, get a plausible structure, and be a
different model - and nothing in the shapes would say so. **Read the
checkpoint's config, not the config class's documentation.**

🔴 **AND IT IS IMPLEMENTED ANYWAY, AT ZERO, BECAUSE A KNOB THAT DOES NOT EXIST
CANNOT BE PRICED.** `maskLanguageModelInput` is upstream's `_mask_input_ids` -
a uniform draw per position against the fraction, the specials exempt, the rest
replaced by `<mask>` (id 32, checked against ESM-C's own tokenizer.json). The
default is `shape.lmMaskPct ?? 0`, so this checkpoint is untouched: the
certainty vector of a 76-residue fold is IDENTICAL to every digit against the
tree before it existed. `tools/gpu/fold-esmfold2.js --lm-mask=` is the arm.

| `--lm-mask` | masked | CA-CA | mean certainty |
|---|---|---|---|
| 0 (the checkpoint) | 0 / 78 | 3.797 | 0.9496 |
| 0.1 | 7 / 78 | 3.797 | 0.9542 |
| 0.3 | 17 / 78 | 3.797 | 0.9528 |

Ubiquitin is an easy target and the language model recovers, so read that as
"the mechanism works and this target does not care", not as a licence.

🔴 **AND THE MASK DRAWS FROM ITS OWN STREAM.** Sharing the sampler's would make
the STRUCTURE move at fraction zero, because a conditional draw shifts every
later value - so `uniforms(seed ^ 0x5bf03635)` is a second stream, and at zero
nothing is drawn at all. `uniforms` is now exported beside `gaussians`, which
builds on it rather than repeating xorshift.

🔴 **AND THE SPECIALS ARE EXEMPT, WHICH IS NOT A DETAIL.** The ids are one
PACKED run - `[BOS] A [EOS BOS] B [EOS]` - so a mask landing on a separator
merges two chains for the tower. Upstream excludes bos, eos and pad by value;
so does this, and the test asserts it at fraction 1 where every residue is
masked and no separator is.

🔴 **AND `(seed >>> 0) || 1` MADE SEED 0 AND SEED 1 THE SAME FOLD.** Zero maps to
one, so the two commonest seeds drew the identical stream - and it looked like a
working seed axis, because seeds 2 and 3 differed.

🔴 **THE DISTOGRAM AND THE STRUCTURE ARE TWO INDEPENDENT READINGS OF ONE TRUNK,
WHICH IS A GATE NEITHER GIVES ALONE.** The distogram head is one projection off
the pair; the coordinates came through the conditioning, twelve token blocks,
two atom stacks and a stochastic sampler. On ubiquitin, over pairs at least six
apart: **predicted 121, actual 148, both 121 - precision 1.00, recall 0.82**. A
fold can be geometrically perfect and be the wrong fold, which CA-CA cannot see.

🔴 **AND THE DISTOGRAM'S BIN EDGES ARE BORROWED, WHICH THE PRECISION ALSO
TESTS.** `distogram_bins: 128` is stated for this head and no RANGE is; the
DISABLED confidence head carries `min_dist: 2.0, max_dist: 52.0` for its own
128. `CONTACT_EDGES` is that, borrowed, and says so - a badly wrong range would
destroy the precision first.

🔴 **AND THE HEAD SYMMETRISES.** `distogram_head(z + z.transpose(-2, -3))` - a
distance is symmetric and the trunk's pair is not, so `z` alone conforms and
returns a plausible distogram.

🔴 **A DIFFUSION TRAJECTORY HAS TWO COORDINATE SETS AT EVERY STEP AND ONLY ONE
OF THEM IS A PICTURE.** `coordinates` is what the sampler carries forward - the
state at the NEXT noise level - and `denoised` is the model's predicted
structure at that call, EDM preconditioning included. The first version of the
page drew the state. `tools/gpu/probe-esmfold2-trajectory.js`, on a 40-mer at
diffusion-15:

| step | state Rg | denoised Rg | denoised moved: raw / fitted |
|---|---|---|---|
| 0 | **35.7 A** | 9.7 | - |
| 1 | **40.4** | 9.7 | 13.5 / **0.47** |
| 4 | 12.9 | 9.9 | 18.3 / 0.64 |
| 8 | 9.9 | 9.9 | 10.2 / 0.17 |
| 10 | 9.9 | 9.9 | 16.3 / **0.02** |

The state is four times the size at the top of the schedule AND NOT MONOTONIC -
it grows before it shrinks - so no fixed camera holds it, and the early frames
are Gaussian noise rather than a structure. The denoised prediction is
protein-sized in every frame. AF3's path records the same finding at its own
sigma: a radius of gyration of 1896 A at step 4 against 11.1 at the end.

🔴 **AND THE FRAMES MUST BE RIGIDLY FITTED, WHICH THE LAST COLUMN IS.**
`centreRandomAugmentation` draws a fresh rotation and translation of the whole
system at the top of every step - it is how the sampler is equivariant, and the
model was trained with it in the loop - so consecutive frames differ by 10-26 A
of rigid motion. Superposed, the real movement is 0.02-0.80 A. Unfitted playback
is a protein tumbling, with the convergence it exists to show invisible
underneath. `fittedPdb` and `alphaCarbons` are AF3's, exported rather than
copied.

🔴 **AND THE DISTOGRAM CAN ORDER RESIDUES BY CONFIDENCE, BUT NOT THE WAY IT
LOOKS LIKE IT SHOULD.** The natural proposal - cross-entropy of the distogram
against the distances the sampler actually produced, `exp(-CCE)` (which is
exactly the predicted probability of the observed bin), meaned over the best N
partners beyond a sequence separation - works, and is beaten by a control that
ignores the structure entirely. `tools/gpu/probe-esmfold2-confidence.js`,
against per-residue **lDDT-Ca** because that is the quantity pLDDT predicts:

| | 1QYS (92 res) | 6MRR (68 res) |
|---|---|---|
| mean lDDT-Ca | 0.918 | 0.930 |
| **exp(-CCE), best top-N** | 0.558 / 0.551 | 0.758 / 0.468 |
| **peakedness alone (control)** | **0.658 / 0.610** | **0.797 / 0.468** |
| neighbour count (baseline) | 0.284 / 0.315 | 0.036 / -0.154 |

Pearson / Spearman. The proposal beats the buriedness baseline by about 2x, so
the signal is real - and the distogram's own PEAKEDNESS, the same aggregate over
the same pairs with the observed bin replaced by the distribution's maximum,
beats it on both targets and both measures.

🔴 **SO EVERY PROPOSAL WAS PUT ON ONE GRID, OVER 46 TARGETS, AND THE FIRST TWO
ROUNDS OF CONCLUSIONS WERE BOTH PARTLY WRONG.** 19 per-pair measures x 25
sequence separations x 24 pair cutoffs = 11,400 arms, aggregated identically -
a plain mean over every kept pair - so no measure is helped or hurt by its
aggregation. The measures: `mode r` (mass within r A of the distribution's
MODE), `obs r` (mass within r A of the distance the SAMPLER PRODUCED, so
`obs 0` is exactly `exp(-CCE)`), `negent` (`exp(-H)`), and ColabDesign's two
contact losses as probabilities.

Median Spearman against per-residue lDDT-Ca, holding the other two axes at the
winner:

| measure | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|---|---|
| `mode r` | 0.415 | **0.426** | 0.417 | 0.411 | 0.408 | 0.387 | 0.373 | 0.314 |
| `obs r` | **0.282** | 0.398 | **0.420** | 0.418 | 0.417 | 0.386 | 0.368 | 0.313 |

| | |
|---|---|
| `negent` | 0.409 |
| `conCat` | 0.302 |
| **`conBin`** | **0.175** |
| neighbour-count baseline | 0.262 |

🔴 **`exp(-CCE)` WAS HANDICAPPED BY THE EXACT BIN, NOT BY THE IDEA.** `obs 0` is
the worst of the non-contact family at 0.282; `obs 1` is 0.420, which is the
best measure in the table. The bins are 0.39 A on a BORROWED grid, so demanding
the exact one is demanding a precision nobody chose. Widen it to a radius and
the agreement family becomes competitive.

🔴 **AND "THE STRUCTURE TERM IS HARMFUL" WAS TRUE ONLY AT RADIUS ZERO.** At
radius 1 `mode` and `obs` are 0.417 and 0.420 - a tie. The structure neither
helps nor hurts; what it costs is availability, since `mode` needs no
coordinates and exists as soon as the trunk has run. That is the reason to
prefer it, and it is a different reason from the one recorded before.

🔴 **THE CONTACT RESTRICTION REALLY DOES FAIL, AND NOW FAIRLY.** `conBin` at
0.175 is BELOW the neighbour-count baseline's 0.262, on the same aggregation as
everything else. `con` is a design objective - one minimises `-log p(contact)`
to PUSH residues together - so a high value is a target, not a claim about
reliability, and a distance confidently predicted to be LARGE is evidence of
confidence too.

🔴 **AND BOTH REMAINING AXES HAVE REAL OPTIMA.** Sequence separation is flat
and best at 3-9 (0.418 at 3, 0.397 at 9, 0.291 at 24). The pair cutoff peaks
sharply: **14 A 0.493, 20 A 0.481, 26 A 0.444, 32 A and beyond 0.39**, equal to
no cutoff at all. So "far pairs are less informative" is TRUE at 14-20 A as a
filter on the PAIR and FALSE at 8 A as a restriction on the BINS - which is
exactly the difference between the winner and `conBin`.

**The arm: `mode 1, separation 3, cutoff 16 A`** - median Pearson 0.664, median
Spearman 0.538 against a baseline of 0.262.

🔴 **AND THE HARD END WAS MANUFACTURED RATHER THAN WAITED FOR.** 43 of 46 real
targets fold above lDDT-Ca 0.9, so the label the sweep was fitted against barely
varied. Replacing a fraction of each sequence's residues at random walks a
target down the scale on demand: 16 targets at 0, 15, 30, 50 and 80% corruption
is 80 folds spanning 0.943 to 0.343.

| rate | mean lDDT-Ca | per-residue median Spearman |
|---|---|---|
| 0% | 0.943 | 0.534 |
| 15% | 0.843 | **0.621** |
| 30% | 0.543 | 0.399 |
| 50% | 0.362 | 0.303 |
| 80% | 0.343 | **0.119** |

🔴 **AND THE LABEL DEGRADES WITH THE CORRUPTION, WHICH HAS TO BE SAID.** A
mutant's true structure is not the crystal, so at 80% a low lDDT may mean "this
sequence really does fold differently" rather than "the model is wrong". That is
why the table is per rate rather than pooled, and why the row that carries the
most weight is 15%: still a protein, and the label finally has spread.

🔴 **SO THE PER-RESIDUE ORDERING SHOULD NOT SHIP.** Median 0.44 over the 80,
best at mild corruption and collapsing to 0.119 on badly corrupted folds - and
the WORST individual fold is NEGATIVE for every arm, -0.042 for the most robust
one and -0.357 for the best-median one. An ordering that is right on average and
inverted on some particular fold is the worst possible per-residue colour,
because the fold somebody is staring at is the one they doubt.

🔴 **AND THE GLOBAL ORDERING SHOULD, BECAUSE IT IS A DIFFERENT QUESTION AND IT
WORKS.** "Which residue is least reliable" needs an ordering INSIDE a fold;
"is this fold worth anything" needs one ACROSS folds. One point per fold - the
mean of the estimate against the mean lDDT - over all 80:

| | Pearson | Spearman |
|---|---|---|
| best arm (`mode 2, sep 6, cut 12`) | **0.902** | **0.862** |
| the per-residue winner, for comparison | 0.864 | 0.838 |

0.90 across a set spanning the whole quality range is a usable answer to the
only confidence question this checkpoint can support. It is still an ORDERING
and not a pLDDT - there is nothing to calibrate a number against - but "this
fold is probably not worth looking at" is a claim it earns.

🔴 **AND THE GLOBAL ARM WANTS A WIDER RADIUS THAN THE PER-RESIDUE ONE**, 2 A
against 0 to 1. Which is a hint about what each is measuring: a per-residue
ordering wants the sharpest possible discrimination between neighbours, and a
per-fold one wants a stable average.

🔴 **AND THE TRUNK IS CACHED, WHICH AF3's PATH HAS DONE SINCE IT HAD ONE AND
THIS HAD NEVER DONE.** Changing only the sampler's step count re-ran ESM-C and
all 96 trunk blocks - 85% of a 300-token fold - to arrive at the same pair.
Reported as "so if a user wants to increase the number of diffusion steps they
have to rerun the trunk?", which is exactly what it was. Measured on a 76-mer:

| | |
|---|---|
| first fold | 4.60 s |
| again, trunk reused | **0.74 s** |
| coordinates | **identical, every atom** |

🔴 **AND THE PAIR IS READ BACK ONLY WHEN A CALLER ASKS FOR IT.** It otherwise
never leaves the device between z_init and the sampler, which is why four trunk
loops cost no traffic; `wantReusable` buys one readback at the end, a quarter of
what looping through the host would have cost.

🔴 **AND THE KEY IS WHAT THE TRUNK DEPENDS ON, WHICH IS NOT WHAT THE FOLD
DEPENDS ON.** The checkpoint, the chains and their kinds, the ligands, the pass
count, and WHICH language model - "none" and ESM-C 600M share a family and
produce different pairs, so the family alone is not enough. The MASK is in it, with the seed behind
it: `lm_mask_pct` replaces a fraction of the residues with the mask token before
the tower runs, drawn from the seed, so with masking ON two seeds are two
different trunk inputs. Measured at `--lm-mask=0.1`, seeds 1 and 2 mask eight
positions each and fold to different structures, with certainty 0.9530 and
0.9485. This checkpoint sets the fraction to 0 so the seed never reaches the
trunk and changing it REUSES - where AF3 re-runs - but the config class
documents single-sequence checkpoints as setting 0.1, so a future bundle turns
this on by existing and the key has to be right before that rather than after.

🔴 **AND THE FIRST VERSION OF THAT KEY HAD NEITHER THE MASK NOR THE SEED IN
IT**, while the note beside it claimed the seed was there when it mattered. The
comment described the intent and the code did not implement it - which on a
checkpoint that masked would have handed the second seed the first one's pair,
silently, with every shape agreeing. Verified by the misses as much as the hits - 300M,
back to 600M and a changed recycle count each re-ran, while a repeat, a step
change and a seed change each reused.

🔴 **AND THE TRUNK NAMES ITS PASS, `Trunk 2/4`, AS AF3's LINE DOES - AND NOT ITS
BLOCK.** That is AF3's rule and its reason, in its own comment: the pairformer
is the one stage that already reports 48 times a pass, so the bar under the line
is visibly moving, and a third field would be the "Trunk · pass 1 of 4 ·
pairformer block 23 of 48" the line was cut down from. A number that changes 96
times sits next to two that barely move and the eye tracks the one part that
does not matter. **The bar is where block-by-block belongs; the line says which
pass.** Measured: the line still changes 7 times over 4 shapes, and the trunk's
96 progress events are 24 per pass.

🔴 **THE STATUS LINE IS THREE PHASES AND A PERCENTAGE, AS AF3's IS.** It used
to name every stage, which gave "recycle 0", "trunk 0", "recycle 1", "trunk
1" - and a recycle is two milliseconds against a trunk loop's several seconds,
so the line flickered between two stages whose costs differ by a thousand.
Reported as exactly that. **A recycle is not a phase; it is the seam between two
trunk passes.** Measured on a 76-mer, the line now changes 7 times over 4 shapes
where it changed 17 over 11.

🔴 **AND THE BAR RUNS OFF A COST MODEL FITTED THIS SESSION**, `src/esmfold2/cost.js`:

| band | at 40 | at 150 | at 300 | the shape |
|---|---|---|---|---|
| language model | 2.4 s | 1.9 | 2.3 | **flat** - it is weight streaming, not n |
| trunk, 4 loops | 0.6 | 6.5 | 27.3 | 0.075 ms * n^2 * loops |
| conditioning | 0.32 | 0.38 | 1.03 | 307 ms + 0.008 * n^2 |
| a sampler step | 34 ms | 72 | 156 | 32 ms + 1.38e-3 * n^2 |

Predicted totals 3.4 / 10.2 / 32.3 s against measured 3.7 / 9.6 / 32.3.

🔴 **AND THE LANGUAGE MODEL IS FLAT IN THE SEQUENCE LENGTH, WHICH LOOKS WRONG
AND IS NOT.** ESM-C's 36 blocks are streamed from a 224 MiB bundle and decoded
on the host, and at these lengths that dominates its own arithmetic - so the
band is about the WEIGHTS and not about the protein. It stops being flat
somewhere above 300 residues.

🔴 **AND A BAND WITH NOTHING TO SAY IS THE ONE THAT JUMPS.** The tower was 53%
of a short fold's predicted time and completed in a single step, so the bar went
from zero to a half. `EsmcTowerGpu` reports per block now; measured, the largest
single jump falls from **0.53 to 0.09** over 45 samples, monotonic throughout.

🔴 **AND THE BAR READS `onBlockDone`, NOT `onBlock`, WHICH IS AF3's IDIOM AND
THE DIFFERENCE BETWEEN SMOOTH AND NOT.** `onBlock` fires when a block is
ENCODED, and sixteen of those happen in the time the device takes over one - so
a bar driven by it sprints to the end of the submission window and then sits
still. Reported as the bar not being smooth.

The fix is a NON-AWAITED `onSubmittedWorkDone()` taken per block: it resolves
once everything submitted so far has finished, so one taken at block i settles
exactly when block i is done, and not awaiting it leaves the encode loop and its
pipelining untouched. They resolve in submission order, so the count cannot go
backwards. AF3's pairformer has done this since it had a status line and took it
from AF2's evoformer stack.

🔴 **AND AWAITING PER BLOCK INSTEAD COSTS 14%, MEASURED.** The obvious fix is to
shrink the submission window until the bar is smooth. At 150 residues:

| submission window | 16 | 8 | 4 | 2 | 1 |
|---|---|---|---|---|---|
| trunk | **6406 ms** | 6452 | 6740 | 6848 | **7294** |

The non-awaited promise gets the same smoothness for nothing: 6464 ms with it,
against 6406 without any reporting at all.

🔴 **AND `onBlock` IS STILL AWAITED, BECAUSE THAT IS WHERE THE PAGE PAINTS.** A
GPU promise resolves as a MICROTASK, which returns control to the microtask
queue and never to the browser - so a page that only moved a bar there would
write it and never paint it. `yieldToBrowser` posts a MessageChannel message,
which is a task and is not clamped to a second in a background tab the way
`setTimeout` is. See src/runtime/yield.js.

Measured in the page, 150 residues: **140 bar samples, largest jump 0.047**,
monotonic from 0.01 to 1.00 - against 45 samples and a 0.53 leap before.

🔴 **AND EVERY BAND REPORTS PER UNIT OF ITS OWN WORK, WHICH IS A COUNT AND NOT
AN IMPRESSION.** `tools/gpu/fold-esmfold2.js` returns `progressEvents`, because
a bar sampled from the page cannot answer "does the trunk report block by
block": 96 updates inside two seconds are far more frequent than any poll, so
the trace shows a handful of values whatever the code does. Counted on a 76-mer:

| phase | events | what one is |
|---|---|---|
| Language model | 36 | an ESM-C block |
| Preparing | 2 | the two ends of the embedder |
| **Trunk** | **96** | a pairformer block - 24 blocks x 4 loops |
| Folding | 11 | a sampler step |

🔴 **AND THE STEP COUNT HAS TO BE KNOWN BEFORE THE TRUNK RUNS**, or the plan
cannot be laid out - so the sampler's settings and schedule are resolved at the
top of the fold. They depend on nothing the fold computes.

🔴 **AND `setColorScheme` / `colorBy` DO NOT EXIST ON py2Dmol's RENDERER.** Both
call sites in `loadIntoViewer` were guarded by `typeof === "function"`, which
turned a wrong API into a silent no-op - so the viewer stayed on
`colorMode: "auto"`, which resolves to **rainbow** for a single chain with no
confidence data.

🔴 **THE AF3 PATH SURVIVED THAT BY ACCIDENT, THROUGH `forcePlddtColours`**,
which sets `renderer.colorSelect.value` and dispatches a change - and py2Dmol's
own handler is what validates the mode, sets `colorMode`, marks both dirty flags
and renders. So the working route was always the SELECT, and the dead API beside
it looked like the one doing the job. There is one `setColourMode` now and it
starts there, which also keeps the visible dropdown in step with what is drawn.

The other route is `py2Dmol.setColor(mode)`. Either way what matters is not the
assignment but the three things after it - `colorsNeedUpdate`,
`plddtColorsNeedUpdate`, `render()` - because setting `colorMode` alone leaves
the cached colours in place, which is a second silent no-op. Valid modes are
`auto, chain, rainbow, plddt, deepmind, entropy, object, hydrophobicity` plus
anything in `window.py2dmol_customColors`.

🔴 **AND THE MODE HAS TO BE SET ON THE FIRST LIVE FRAME, NOT AT THE END.**
Setting it after the fold left every frame drawn WHILE the sampler ran in
rainbow, which is the whole point of a per-frame certainty. Reported separately
from the first colour bug, after it was fixed.

🔴 **AND py2Dmol ORIENTS WHEN IT INGESTS A FILE, WHICH THIS PATH NEVER DOES.**
The ESMFold2 fold draws FRAMES, so nothing ever found a best view: the whole
trajectory ran at whatever camera the blank object had, and then the final
`loadIntoViewer` orientated at the very end. That is both halves of "best view
is not being applied to first frame" and "last frame is different angle" -
one cause, reported as two symptoms. `orientBestView` on the first frame, and
the camera saved across the reload off the RENDERER rather than off `viewer`,
which is undefined until that reload and therefore held no camera to save.
Measured: `0.94, 0.35, 0.05` during the fold and `0.936, 0.348, 0.052` after.

🔴 **AND A PROBE THAT READS THE DATA IS NOT A PROBE THAT READS THE PICTURE.**
`tools/fold-in-page.py`'s `bfactor:` line reported 46.4-98.5 per frame while the
structure was drawn in rainbow. It now prints `colour:` as well - the renderer's
`colorMode` and `resolvedAutoColor` - because what is DRAWN is the thing that
was wrong. Same rule as "sample what is drawn, not what was computed", one panel
over.

🔴 **AND THE CAMERA HAS TO BE SAVED ACROSS THE FINAL RELOAD.** `loadIntoViewer`
ingests a FILE and py2Dmol orients the camera when it parses one, so the
trajectory a reader has been watching - and possibly rotating - snaps to a new
angle the moment the last frame lands. The AF3 path has saved and restored
`viewerState` since it had a trajectory; the ESMFold2 one had not. Reported as
"the frames change angle when last frame is added".

🔴 **AND THE STEP DIAL MUST SHOW WHAT RUNS, NOT WHAT THE CONFIG CALLS IT.** The
dial offered 15 beside a status line reading 11 steps, which is the page
contradicting itself - and the config's number is the one with no operational
meaning, since `max_inference_sigma` drops every schedule entry above 256. The
option's VALUE stays the preset's own number, because that is what names a
preset; only its text changes, and it is computed from the schedule rather than
tabulated: 15 -> 11, 32 -> 23, 64 -> 45, 200 -> 138.

🔴 **AND THE TRAJECTORY IS COLOURED FRAME BY FRAME, WHICH IS WHAT THE `obs` ARM
BUYS.** `mode` needs no coordinates and is therefore FIXED for a fold: every
frame would wear the same colour, and the interesting thing about a trajectory
is watching it become confident. `obs` centres on the distance the sampler
produced, so it changes every step. The two scored a tie - median 0.537 against
0.535, worst fold 0.361 against 0.363 - so this costs nothing in accuracy and
buys a per-frame reading. Measured on ubiquitin, the B-factor range per frame:
46.4-98.5, then 40.1-98.0, then 48.4, 48.2, 49.1... - it moves, and it settles.

🔴 **THE FRAME IS SCORED ON THE DENOISED PREDICTION, WHICH IS WHAT IS DRAWN.**
The sampler's own state is Gaussian noise at the top of the schedule; colouring
the state while drawing the prediction would put one frame's colour on another
frame's structure.

🔴 **AND THE FILTER STAYS ON THE MODE WHILE THE QUANTITY MOVES.** The pairs a
residue is judged on come from what the model PREDICTS, which does not change
between frames; only the mass being averaged is recomputed. A score whose
denominator moves is not comparable down a trajectory.

🔴 **AND `mode` IS STILL THERE, AS THE FALLBACK AND AS THE PRE-SAMPLER READING.**
It comes off the trunk, so a caller could gate whether to sample at all on it -
and the finished structure keeps the LAST FRAME's score rather than the trunk's,
or the play bar would step to a different colour on its final frame and read as
the fold changing its mind at the end.

🔴 **IT COSTS THE DISTOGRAM STAYING RESIDENT: `bins` TIMES THE PAIR
REPRESENTATION, 46 MiB at 300 tokens**, released with the last frame.

🔴 **AND A RETAINED BUFFER HAS TO LEAVE THE RELEASE LIST WHEN IT IS RETAINED,
NOT WHEN IT IS FREED.** The first version handed the scorer a closure that
spliced the buffers out of `held` - and that closure runs when the CALLER is
finished, long after the function's own `finally` has freed them. The failure was
"[Buffer esmfold2.disto.certainty-readback] is destroyed" on the first frame,
which names the buffer and not the lifetime.

🔴 **SO THE COLOUR SHIPS, AND THE PRE-SAMPLER ARM DOES NOT USE THE STRUCTURE.**
`CERTAINTY` in src/esmfold2/distogram-webgpu.js: the mass within **2 A of the
distogram's mode**, meaned over every pair at sequence separation above **3**
whose predicted distance is under **12 A**. Ranked on realistic corruption rates
(0 and 15%) by WORST fold, the two families are a tie -

| family | median | worst |
|---|---|---|
| `mode 1.5, sep 3, cut 12` | 0.537 | 0.361 |
| `obs 3, sep 15, cut 14` | 0.535 | **0.363** |
| `negent, sep 5, cut 12` | 0.572 | 0.288 |
| `conBin, sep 3, cut 8` | 0.456 | 0.239 |

- and `mode` is taken because it needs NO COORDINATES. It comes off the trunk's
distogram, so it exists before the sampler runs: the live frames are coloured
from the first one, and a caller could gate whether to sample at all on it.
`obs` cannot do either, for a tie.

🔴 **AND EVERY INVERSION WAS AT 40% CORRUPTION OR WORSE.** The objection to
colouring - "the worst fold is negative" - was measured over folds nobody would
make. On 0 and 15% the worst of forty folds is **+0.31**, and the negatives live
where the fold is garbage AND the label is meaningless, since a heavily mutated
sequence's true structure is not the crystal.

🔴 **AND IT GOES IN THE B-FACTOR UNDER THE pLDDT PALETTE, WITH THREE THINGS
STOPPING IT BEING READ AS ONE.** The B-factor is the only column a viewer can
colour from and the palette is the one every reader of this page knows, so the
guards are elsewhere: the status line says `certainty 0.94 (not pLDDT)`, the
downloaded PDB carries a `REMARK` naming the quantity and the missing head, and
the word pLDDT appears nowhere. With no certainty at all it falls back to chain
colours rather than painting a zero B-factor as no confidence.

🔴 **AND THE PARTNER RULE IS AF3's OWN lDDT, WHICH IS NOT SYMMETRIC.**
OpenFold3's `all_atom_plddt_loss` builds its pair mask as

    (dx_gt < 15) * protein_atom_mask[..., None, :]
  + (dx_gt < 30) * nucleotide_atom_mask[..., None, :]

- the radius is chosen by the kind of the atom in the SECOND index, the one
doing the scoring, and a ligand atom appears in NEITHER term. Its `rep_index`
says the same thing from the other side: CA for a standard protein residue, C1'
for a standard nucleotide, and a padding sentinel for a ligand or an atomized
residue. **Every atom is SCORED; only polymer representatives do the SCORING.**

That is what "treat a ligand like a protein" means, and taking it removed the
fallback by construction rather than by adding a tier: a ligand token has
partners - the polymer around it - under everyone else's cutoff.
`PARTNER_ANGSTROMS` is **12 A for a protein partner and 24 A for a nucleic
one**: the protein number is the one this repository's own 11,400-arm sweep
peaked at (12-14 A), and what is taken from AF3 is the SHAPE - a nucleotide
reaches twice as far, because a base pair's partners are further off than a
side chain's. AF3's own 15/30 is for a different quantity, a distance-difference
test against a true structure rather than a distogram's peakedness.

🔴 **AND A SEQUENCE SEPARATION NEEDS A SEQUENCE, WHICH A LIGAND HAS NOT GOT.**
Every atom of a component carries residue number 1, so a rule phrased in
residues excluded every pair INSIDE it - and a heme folded ALONE then had no
surviving pair at all: 43 tokens, every one reporting no data, the whole
molecule the worst colour on the scale, beside a contact map that was confident
about it. Reported exactly that way. The rule is about a POLYMER's own backbone
and now applies only where both ends are polymer; for a ligand the covalent
bond is the whole exclusion. The lone heme reads **0.9878** (0.9365 to 0.9974).

🔴 **AND THE EXCLUSION IS "TRIVIALLY CLOSE", OF WHICH SEQUENCE SEPARATION IS
ONLY ONE FORM.** A covalent bond is the other: a glycan or a covalent inhibitor
sits at a fixed bond length from the residue it is attached to, which is exactly
as uninformative as an i+1 neighbour and was being counted as a confident
prediction. `features.tokenBonds` is passed to the certainty now and is
REQUIRED, because "no bonds" and "bonds not passed" are the same buffer of zeros
and the second is a silent wrong answer on the one input this exists for.

🔴 **AND NOTHING REFUSES TO SCORE ON CHEMISTRY, WHICH IS WHERE THIS PARTS
COMPANY WITH AF3's lDDT.** That loss admits only protein and nucleotide atoms as
the partner index and gives a ligand no representative at all - right for a
training loss over structures that always have a polymer, and wrong the moment
somebody folds a ligand alone. What stays chemistry-shaped is only how far a
partner may REACH.

🔴 **AND IT MOVES THE POLYMER'S NUMBER, WHICH HAS TO BE SAID.** With a ligand
eligible to score, ubiquitin beside ATP reads **0.7493** where excluding the
ligand gave 0.9215 - a ligand is one token per heavy atom, so ATP is 31 of 66
tokens and nearly half of every residue's partner set. The same arithmetic that
made it 62% of the contact metric. Which number is right is not measured here;
what is measured is that the rule is now the same one for every token.

🔴 **AND A LONE LIGAND'S 0.99 IS MOSTLY THE CONFORMER COMING BACK.** Its
internal distances were HANDED to the model in `ref_pos`, so reproducing them is
a copy rather than a prediction - the score is honest about the distogram's
confidence and says little about whether the placement is right. Read it as
"the model is sure", not as "the model is correct".

🔴 **AND THE UNFILTERED FALLBACK IS GONE, WHICH IS WHAT MADE A LIGAND'S COLOUR
THE ODD ONE OUT.** A ligand's atoms share one residue number, so the separation
rule dropped its whole self-block and it had NO partner - landing on a `loose`
branch that averaged every partner at any distance, the 12 A cutoff included.
ATP read 0.3562 and every digit of that was the fallback: the only number on
the page computed a different way from the rest. It reads **0.3107** now, under
the same rule as everything else. **Nothing falls back any more**, and a token
with no eligible partner reports -1, which the caller reads as no data rather
than as no confidence.

🔴 **AND THE PER-RESIDUE SCORE DOES NOT SPLIT WITHIN FROM ACROSS, BECAUSE pLDDT
DOES NOT.** A local score is about a token's neighbourhood, and a residue at an
interface really does have neighbours in the other chain - AF3's lDDT admits
them. The pTM/ipTM question is a PER-CHAIN one and is answered per chain
instead:

🔴 **AND THE PER-CHAIN SUMMARY DOES SPLIT THEM, WHICH IS AF3's OTHER SHAPE.** The rule excludes only same-chain sequence neighbours, so a
residue on a complex used to be judged partly on pairs across the interface -
and a chain can be folded well and docked badly, which is why AF3 keeps pTM and
ipTM apart. Measured on a two-chain fold, recomputed on the host over the same
distogram three ways:

| chain | mixed | within | across |
|---|---|---|---|
| A (35 tokens) | 0.451 | 0.477 | 0.343 |
| B (68 tokens) | **0.630** | **0.712** | 0.370 |

A chain can be folded well and docked badly, and one number over both says
neither - so `chain_certainty` and `chain_interface_certainty` go in the
archive beside `chain_pair_max_contact`. `interface_certainty` is -1 where a
token has no eligible partner in another chain, which on a monomer is every
token and next to a LIGAND is every protein token too, since a ligand cannot
score. **A monomer's certainty vector is unchanged to every digit** through all
of this.

🔴 **AND IT SHOWS UP WITH A LIGAND, WHICH IS WHERE IT WAS FIRST REPORTED.**
Ubiquitin with ATP: the protein reads **0.9215** against 0.8989 before, because
a ligand is no longer an eligible partner and so can no longer mark the protein
down for the model's uncertainty about where it goes.

🔴 **AND "(not pLDDT)" IS GONE FROM THE STATUS LINE.** It denied something the
line never claimed - it says `certainty`, not pLDDT - and a parenthesis
refusing a reading nobody offered reads as a disclaimer rather than a result.
The caveat still lives where somebody looking for it will find it: the model
row's tooltip, the PDB's REMARK and the archive's README.

🔴 **AND "NO PARTNER INSIDE THE CUTOFF" IS NO DATA, NOT ZERO CONFIDENCE.** A
terminal residue the model places away from everything has an empty filtered
mean; writing 0 there paints it as the least reliable residue in the structure,
which is a claim and the wrong one. It falls back to the unfiltered mean, and
only a chain shorter than the separation gets nothing. Measured on ubiquitin:
before the fallback the range was 0.0 to 98.6, after it 49.7 to 98.6.

🔴 **AND A SEQUENCE SEPARATION ON THE TOKEN INDEX IS NOT A SEQUENCE SEPARATION
WHEN A LIGAND IS ONE TOKEN PER HEAVY ATOM.** Every contact and certainty path
here excluded a partner when the TOKEN INDICES were close, which is a rule about
a chain's own neighbours - and a ligand has none. Reproduced by folding
ubiquitin with and without ATP (76 residues, 31 atoms), which is the measurement
that was missing: **all 46 sweep targets and all 80 corrupted folds were single
protein chains**, so nothing in this file had ever borne on a ligand token.

| ubiquitin + ATP | token-index rule | residue rule |
|---|---|---|
| contacts predicted | 314, **195 of them the ligand's own** | 112 |
| ...precision / recall | 0.987 / 0.726 | 1.000 / 0.541 |
| the LIGAND's mean certainty | 0.8173 | **0.2193** |
| the PROTEIN's mean certainty | 0.8989 | 0.8989 |

62% of that fold's contacts were ATP's internal pairs, so the precision a
checker prints was mostly a statement about a conformer the model was HANDED.
`partnerKeys` in src/esmfold2/distogram-webgpu.js replaces the arithmetic with
two numbers per token - the asym id and the residue number - and the rule
becomes "the same chain, and within `separation` RESIDUES". A ligand's atoms
share one residue number, so a gap of zero drops the whole self-block; two
chains are never neighbours at all, which fixes a smaller bug in the same line.

🔴 **AND IT IS THE OLD RULE EXACTLY ON ONE UNMODIFIED PROTEIN CHAIN, WHICH IS
WHAT EVERY CONSTANT WAS TUNED ON.** There the residue number and the token index
differ by a constant, so their differences agree - and a ubiquitin fold comes
back with the certainty vector IDENTICAL to every digit. That is the gate;
`test/esmfold2-certainty-partners.test.js` pins it, and asserts on the generated
WGSL, because a partner rule that never reaches the kernel agrees with itself.

🔴 **AND THE PROTEIN'S OWN NUMBERS DID NOT MOVE, WHICH REFUTES THE OBVIOUS
DIAGNOSIS.** Adding ATP shifts the protein's certainties by -0.051 on average
with one residue moving 0.475 - and it shifts them by **exactly that, to four
decimals, under BOTH rules**. So the shift is the TRUNK conditioning on a
molecule that is really there, not the aggregation eating ligand pairs: for a
protein token only a handful of partners change category. The proposed
per-chain fix would have been credited with this and deserved none of it.

🔴 **AND THE DISTOGRAM SAYS NOTHING ABOUT WHERE THE LIGAND GOES.** It predicts
**0** protein-ligand contacts while the structure makes **64**, on the same
fold, and the honest 0.22 certainty above is that fact reaching the colour. It
is not settled whether the head cannot speak about ligand pairs or the borrowed
`CONTACT_EDGES` are wrong for them - **open**, and it is why the ligand-free
number is the one to trust.

🔴 **AND 8 ANGSTROMS IS A PSEUDO-BETA CONVENTION, SO IT IS THE WRONG NUMBER
FOR EVERY PAIR THAT IS NOT TWO RESIDUES.** A distogram predicts a distance
between one representative atom per TOKEN, and for a residue that atom stands in
for a side chain's reach while for a ligand it IS the atom. Calibrated rather
than argued: `tools/calibrate-contact-cutoff.py` downloads real depositions,
takes REAL atomic contact as the ground truth - any heavy atom pair under 5 A -
and sweeps which representative-distance threshold reproduces it. 14 entries,
41,000 real contacts, best F1:

| pair | cutoff | F1 | at 8 A |
|---|---|---|---|
| protein-protein | **8 A** | 0.767 | the convention, confirmed |
| ligand-protein | **7 A** | 0.707 | 0.629 |
| ligand-nucleic | **7 A** | 0.764 | |
| nucleic-protein | **10 A** | 0.607 | 0.444 |
| nucleic-nucleic | **9 A** | 0.777 | |
| ligand-ligand | **5 A** | **1.000** | 0.696 |

`CONTACT_ANGSTROMS_BY_KIND` is that table, and the contact shader reads a
PAIRS-SIZED array of bin counts rather than a constant - the pass is chunked
over a slice of the logits, so its cell index is chunk-relative and cannot
recover i and j to look a kind up.

🔴 **AND THE LIGAND ROW IS EXACT, WHICH IS THE POINT AND NOT A FLUKE.** Both
representatives ARE the heavy atoms, so the representative distance is not an
approximation of the ground truth - it IS the ground truth, and 8 A was doing
nothing there but being the wrong definition. That holds whether the two atoms
are in one molecule or two; **what differs between intra and inter is what the
number MEANS**, not where the line sits. Inside a molecule the geometry came
from the CCD conformer the model was HANDED, so a prediction there is a copy.

🔴 **AND ONE NUMBER CANNOT SERVE TWENTY SIDE CHAINS, WHICH IS THE OTHER HALF OF
THE SAME POINT.** The representative stands at a different DEPTH in each
residue: a ligand touching a tryptophan ring is far from that CB, an alanine's
heavy atoms barely reach past it. `--by-residue` measures the reach - the median
pseudo-beta-to-ligand distance among pairs that really are in contact - and it
runs **4.26 A at cysteine to 7.28 A at arginine**, monotonic in side-chain
length, with the best threshold tracking it 5 A to 8 A. Pooled over every
ligand-protein pair:

| rule | precision | recall | F1 |
|---|---|---|---|
| one threshold, 6 A | 0.819 | 0.586 | 0.683 |
| one threshold, 7 A | 0.631 | 0.803 | 0.707 |
| one threshold, 8 A | 0.458 | 0.932 | 0.614 |
| **per residue** | **0.740** | **0.824** | **0.780** |

It DOMINATES rather than trading, which is what says the residues really do want
different numbers. `LIGAND_PROTEIN_ANGSTROMS` is that table.

🔴 **AND IT IS HELD OUT, BECAUSE "TWENTY FREE PARAMETERS BEAT ONE" IS WHAT FREE
PARAMETERS MANUFACTURE.** `--holdout` fits the thresholds on half the entries
and scores them on the other half: **0.771 against the best single arm's
0.694**, barely below the fitted 0.780. Fitting and scoring on the same rows
would have been no result at all.

🔴 **AND ANY THRESHOLD AT OR UNDER 5 A IS PERFECTLY PRECISE FOR FREE**, because
the representative IS one of the residue's own heavy atoms - so a representative
within 5 A of the ligand atom satisfies the ground truth by construction.
Glycine and alanine reading precision 1.000 is that, not a measurement of
anything. The whole question is how much RECALL a side chain lets you buy before
precision goes.

🔴 **AND `contactAngstromsFor` IS ONE FUNCTION FOR BOTH HALVES OF A METRIC.** A
checker computing "actual" at 8 A against a map computing "predicted" at 7
reports a precision about nothing, and the two halves live in different files.

🔴 **AND A NUCLEOTIDE'S REPRESENTATIVE WAS ITS PHOSPHORUS, WHICH COST A FACTOR
OF SIX.** `representativeAtoms` was CB, else CA, else the token's first atom - and
a nucleotide has neither, so it took whatever came first. Across a duplex the
phosphates are eighteen angstroms apart while the bases stack, so **no threshold
recovers the contact**: nucleic-nucleic reads F1 0.125 at its best arm, which is
the widest one offered. AF3's own `RESTYPE_PSEUDOBETA_INDEX` says CB (CA for
glycine), then **C4 for a purine and C2 for a pyrimidine** - not C1', which is
the obvious guess and a different atom. With that table the same row reads
**0.777**, and ligand-nucleic 0.617 -> 0.764.

🔴 **AND IT IS A GEOMETRIC FACT, WHICH IS WHY IT COULD BE FIXED WITHOUT AN
ORACLE.** The calibration never runs a model: it asks whether a threshold on a
representative distance reproduces real atomic contact in a deposited
structure. There is no dump saying which convention ESMFold2 was trained with,
and two independent arguments - AF3's table, and a 6x geometric improvement -
point the same way. The shipped certainty does not move either way (it reads
the distogram's MODE, not coordinates); what changes is the `obs` arm and every
contact metric.

🔴 **AND `named` MATCHES ALL FOUR NAME CHARACTERS, WHICH IS LOAD-BEARING HERE.**
"C4" must not match C4', which every nucleotide also has and which is back out
on the sugar - a silent 4 A error in the representative.

🔴 **AND AF2 AND AF3'S CONFIDENCE IS NOT AFFECTED, BUT AF3's CONTACT MAP IS.**
The report was "this affects all models". Half of it does. No AGGREGATION is at
risk: `distogramContactProbabilities` in src/heads/distogram.js is per PAIR with
no separation rule, `web/prediction-results.js` exports the matrix as it stands,
and both models take their confidence from a confidence head - so ESMFold2 is
the only one here that DERIVES a confidence from a distogram. But the THRESHOLD
is a different question, and AF3 tokenises ligands one heavy atom at a time
exactly as ESMFold2 does, so its contact map wanted the same table. AF2 does
not: monomer and multimer are protein-only, every pair is two residues, and 8 A
is simply right there.

🔴 **SO THE TABLE LIVES IN `src/heads/contact-threshold.js`, WHICH IS NEITHER
MODEL'S.** `contactAngstromsForClasses` takes two CLASSES - nucleic, ligand, or
one of the twenty amino acids - and each model maps its own alphabet onto them.
The two heads then disagree about one thing only, which is where a bin's edge
is: AF3 counts a bin whose TOP edge is under the threshold and ESMFold2's
borrowed grid counts one whose CENTRE is. Both are prefixes of an ordered
binning, so `contactBinsByPair` takes `binsUnder` from the caller and returns a
COUNT rather than a mask.

🔴 **AND BOTH ALPHABETS ARE THREE-LETTER ALPHABETICAL, WHICH IS WHY ONE TABLE
SERVES THEM AND IS ASSERTED RATHER THAN TRUSTED.** AF3's
`ARNDCQEGHILKMFPSTWYV` is one-letter alphabetical and happens to be
three-letter alphabetical too, so AF3's restype IS the table's index and
ESMFold2's is that plus two. `test/contact-threshold.test.js` walks all twenty
rather than spot-checking, because a silently permuted alphabet conforms in
shape and folds something.

🔴 **AND A LIGAND ATOM AND AN UNKNOWN RESIDUE SHARE AN `aatype`, SO AF3's CLASS
CANNOT COME FROM THE ALPHABET.** `featurise.js` writes `UNK_AATYPE` for every
ligand atom and the same value for an X in a protein chain; only `ligandSpans`
separates them, and reading the alphabet alone gives a ligand a 7 A protein
threshold, which conforms and is wrong. `af3ContactClasses` is that one join.

🔴 **AND THE HEADS REFUSE TO DEFAULT IT.** A caller with no classes would
silently get 8 A on every pair back - the convention this exists to correct -
and the failure would be a plausible contact map rather than an error. Both the
GPU head and its CPU reference throw; the two AF3 trunk checkers pass
`CLASS_PROTEIN` for every token EXPLICITLY, with a comment saying their dumps
are protein-only, which is a stated assumption rather than a silent one.

🔴 **AND AF3's PROTEIN-ONLY FOLD IS UNMOVED.** `tools/gpu/fold.js` on a 40-mer:
mean pLDDT **77.36664729240613** and pTM **0.5503883067518472**, which are this
file's own recorded figures to every digit.

🔴 **AND `tools/gpu/fold.js` TAKES `--ligands` AND `--kinds` NOW, BECAUSE THE
LIGAND BRANCHES HAD UNIT TESTS AND NO END-TO-END RUN.** AF3's featuriser has
handled ligands and nucleic chains for a long time and the only route to one was
the page, so the case the contact threshold is entirely ABOUT could not be
folded from a shell. Ubiquitin's first 40 residues with ATP: 71 tokens, 343
atoms, mean pLDDT 89.5, pTM 0.719, ipTM 0.678.

🔴 **AND IT PRINTS A CONTACT-CLASS CENSUS, BECAUSE "IT FOLDED" DOES NOT SAY THE
LIGAND WAS SEEN AS ONE.** A ligand atom and an unknown residue share an
`aatype`, so a class taken from the alphabet alone would call all 31 of ATP's
tokens protein - and the fold would still come out, with a protein's threshold
on every pair and nothing to see. The line reads `contact classes: 40 polymer,
31 ligand, 0 nucleic`, and it is printed only when there is something to say.

🔴 **AND THE PLM ROW PICKS A CHECKPOINT, NOT A TOWER.** Biohub publish
`base600M-step1500k` and `base300M-step1500k` as separate models whose shims are
trained for **36 layers x 1152** and **30 x 960** - so a tower is not swappable
over one set of folding weights, and choosing ESM-C 300M loads a different fold
bundle with it. The folding model is the same SIZE in both (171 M), so the whole
difference is the tower: **252.1 MiB against 346.1**, a 27% saving. The model row
shows one "EF2-fast" and `PLM_FAMILIES` in web/app.js resolves which. A third
pair is published (ESM-C 6B, 6352 M) and would be a registry entry rather than a
branch - `companion` in `MODEL_BUNDLES` is the link, and the loader reads it.

| | ubiquitin, against the 600M fold | 6MRR vs crystal | bundle | time |
|---|---|---|---|---|
| ESM-C 600M | - | **1.43 A**, TM 0.922 | 346.1 MiB | 4.8 s |
| ESM-C 300M | **0.74 A** | 1.66 A, TM 0.909 | **252.1 MiB** | 3.8 s |
| none | 10.96 A | 1.52 A, TM 0.926 | 122.5 MiB | 2.2 s |

0.74 A is inside the sampler's own seed spread, which is what docs/ESMFOLD2.md's
ablation predicted from its median of 2.55 A against 2.52.

🔴 **AND `loadEsmfold2Weights` IS A MAP NOW, WHICH ITS OWN COMMENT ASKED FOR.**
It memoised ONE promise and said "if a second appears, this becomes a Map on the
same day" - and the two pairs have the SAME trunk shapes with different shims,
so a page folding one then the other would have got the first one's weights with
nothing to signal it. That is `loadAf3Weights`'s recorded bug, one model over.

🔴 **AND `family === "ef2-fast-600m"` WAS IN FOUR PLACES, WHICH IS THE
`family === "af3"` MISTAKE AGAIN.** Each meant "is this the single-sequence
pipeline" and each sent the 300M checkpoint down AlphaFold 2's branch. They ask
`SINGLE_SEQUENCE_FAMILIES.includes(family)` now.

🔴 **AND A DESIGNED PROTEIN DOES NOT NEED THE LANGUAGE MODEL AT ALL.** Reported
from the page and confirmed against the crystal: 6MRR folds to **1.52 A, TM
0.926 with no PLM** against 1.43 A and 0.922 with ESM-C 600M - the tower is
worth nothing on it, where ubiquitin's fold moves 10.96 A without one. 6MRR is a
DESIGNED protein, idealised and canonical, so the structure module folds it from
the sequence embedding alone. **The PLM's value is target-dependent, and a
single ablation on one target says nothing about the next.**

🔴 **AND THAT REFINES WHAT THE CERTAINTY IS DOING, IN ITS FAVOUR.** It does not
detect the ABLATION - it tracks the FOLD. On ubiquitin the fold collapsed and it
fell 0.95 to 0.42; on 6MRR the fold survived and it stayed at 0.95. An estimate
that dropped whenever an input was removed would have been measuring the setting
rather than the answer.

🔴 **AND SCORING AGAINST A DEPOSITED STRUCTURE NEEDS THE ALTLOCS DEDUPED.**
6MRR's chain A has 71 CA records for 68 residues, so a naive walk pairs the model
against a shifted crystal and reports **4.74 A and TM 0.393** for a fold that is
really **1.43 A and TM 0.922**. Take one CA per (chain, residue number) and skip
any `altLoc` outside " " and "A".

🔴 **AND THE LANGUAGE MODEL CAN BE TURNED OFF, WHICH IS THIS MODEL'S "SINGLE
SEQUENCE".** AF2 and AF3 fold without their alignment; ESM-C is where this model's
evolutionary information comes from, so switching it off is the same ablation.
It is not a matter of zeroing anything - `tokenToRow` of -1 already means "no
row", the path every ligand and nucleotide token takes, and
`shimSingleForZeroState` supplies the fixed NON-ZERO vector the shim's offset and
bias produce. Substituting zeros would be a different model. `--no-plm` on the
tool, a `PLM` row on the page in the slot the MSA row leaves empty, and
`--plm none` on `fold-in-page.py`. Measured on a 76-mer:

| | CA-CA | certainty | contacts predicted | time |
|---|---|---|---|---|
| ESM-C 600M | 3.797 | **0.9496** | 114, precision 1.000 | 4.8 s |
| none | 3.755 | **0.4165** | **0** | 2.7 s |

...and the two structures are **10.96 A apart**. The geometry stays valid either
way, which is the point: without the language model it makes a CHAIN it cannot
FOLD, and a reader looking only at CA-CA would not know.

🔴 **AND THE RECYCLE DIAL DRIVES THIS TRUNK NOW, WHICH IT DID NOT.** Its loop
count came from the checkpoint - `{ ...M, loops: (M.loops ?? 3) + 1 }` - while
the Recycles control sat on screen beside it doing nothing, which is the
"quietly ignored control" `syncModelControls` exists to prevent and the reason
the MSA row is hidden for this family rather than left showing. The mapping is
exact and the default does not move: upstream runs `range(num_loops + 1)`, this
checkpoint's `num_loops` is 3, and 3 is the dial's own default, so a default
fold is the same four passes it always was. Measured, `--recycles=` on the tool:
1 gives `Trunk 1/2, 2/2` and 3 gives `Trunk 1/4 ... 4/4`.

🔴 **AND THE PLM ROW HAD TO BE GIVEN A LINE OF ITS OWN.** It sits where the MSA
row sits - it IS that row for this model - but this family also hides the
sampler-MODE group, so the line above has room, and `af3CountGroup` GROWS to
fill a line it is alone on: 535px of 948, which left PLM at x=587 against the
MSA row's 42. `flex-basis: 100%` is the break, the same idiom `.fold-option-wide`
uses for the alignment box. Measured: PLM at 42,278 against AF3's MSA at 42,278.

🔴 **AND THE CERTAINTY CATCHES IT, WHICH IS EVIDENCE FOR THE CERTAINTY.** 0.95 to
0.42 on an ablation nobody tuned it against - the sweep that chose its constants
corrupted SEQUENCES, and this removes a whole input. An estimate that tracked
fold quality only on the perturbation it was fitted to would not have moved.

🔴 **BUT IT DOES NOT CATCH IT ON A SHORT ONE, AND THAT IS A REAL BLIND SPOT.** On
a 35-mer the same ablation takes the contact count from 27 to **zero** while the
certainty reads 0.8978 against 0.8922 - unmoved. The filter keeps pairs the model
places under 12 A, and with no language model it places almost everything
further, so the mean is taken over the handful of near-neighbours that survive,
which are trivially peaked. **A certainty over very few pairs is not a certainty
about the fold**, and nothing on the page says how many pairs it rested on. The
contact count and `chain_pair_max_contact` do tell the truth there. **Open.**

🔴 **AND A FOLD WITH NO PROTEIN DOWNLOADED 224 MiB OF LANGUAGE MODEL IT NEVER
CALLED.** ESM-C is handed protein tokens only - `protein_mask = (mol_type == 0)
& token_mask` - so a ligand, DNA or RNA input has no row to give it, and
`foldEsmfold2` already skips the call at `lm.ids.length === 0`. A lone heme
spends **2 ms** in the language model band and has no "Language model" phase in
its trace at all. What it could not skip was the DOWNLOAD: `towerStore.prefetch()`
starts all 54 shards the moment the model is chosen. `startModelPreload` reads
the entities and passes `languageModel: false` when none of them is a protein;
an empty list still fetches, because a page nobody has typed into yet is most
likely about to hold a protein.

🔴 **AND THE STORE MUST LEAVE THE PROGRESS SUM, NOT MERELY GO IDLE.** The
reporter adds `totalBytes` across both bundles, so a tower that never downloads
still promised its 223.6 MiB and the dial would have stopped at a third. It is
registered only when it is going to be fetched.

🔴 **AND THE SHIM IS STILL READ FROM THAT BUNDLE.** Its LayerNorm has an offset
and its downprojection a bias, so `shim(0)` is a fixed NON-ZERO vector and a
fold with no protein still needs those tensors - see `shimSingleForZeroState`.
They are a few hundred KB fetched on demand; the 36 blocks are the 223.6 MiB,
and those are what go unfetched.

🔴 **AND THE TRUNK STILL RUNS, WHICH IS NOT WASTE.** It is the pair track, not a
protein stage: the diffusion conditioning takes `z` from it and the distogram
head is a projection off it, so a fold without it would condition the structure
head on `z_init` and be a different model. On the lone heme it is 0.63 s of a
1.53 s fold, and the protein-specific stage is the one that already skips
itself.

🔴 **AND THE DIAL HAS TO STOP REPORTING WHEN THE LOAD DOES.** The tower STREAMS
- its 36 blocks are read during the fold - so its store went on firing progress
after the weights promise resolved, and `startModelPreload`'s clear was
immediately undone by the next shard. The dial stuck at "346 / 346 MiB" for the
rest of the session. What it means is the DOWNLOAD; the streaming is the fold's
own business and the status line narrates it.

🔴 **THE PAGE'S CAPABILITY GUARDS ARE `supportsAllAtom`, NOT `isAf3Family`, AND
THAT IS THE SAME MISTAKE ONE MODEL LATER.** ESMFold2 runs a different GRAPH and
the same all-atom representation, so a guard asking "is this AlphaFold 3"
refuses a ligand under a model that has ligand tokens - with a message naming a
capability it has, which is exactly what the AF3/OpenBind split recorded above.
Templates are the one thing that is still an AF3 question and stay on
`isAf3Family`.

🔴 **AND A MODEL WITH NO CONFIDENCE HEAD MUST NOT BE COLOURED BY pLDDT.** Every
other fold here is coloured by the confidence head's per-atom answer; this
checkpoint has ZERO `confidence_head.*` tensors, and a structure drawn under the
pLDDT scheme with a zero B-factor is uniformly the colour of NO confidence -
which reads as a terrible fold rather than an absent measurement. It is coloured
by chain, `lastPrediction` carries no `confidence` object at all, and the status
line says so.

🔴 **AND THE MSA ROW IS HIDDEN FOR IT RATHER THAN IGNORED.** A search left on
screen would run, take a minute of somebody else's server, and be discarded -
which is the "quietly ignored control" `syncModelControls` exists to prevent,
one step worse.

## The model is called EF2-fast 600M, and the name is load-bearing

🔴 **`esmfold2` ALONE READS AS ESM'S RELEASED ESMFold2-Fast, WHICH IS A
DIFFERENT AND BETTER MODEL.** That one folds from ESM-C 6B; this is the 600M
experimental checkpoint, which is the one that fits a browser. The page was
making the confusion twice - the dropdown said "ESMFold2" while the status line
said "ESMFold2 600M". The family, the manifest key and the download stem all say
`ef2-fast-600m` now, and `esmfold2` survives as a `?model=` alias so a saved
link still works - the same shape as openbind0's rename, and for the same
reason.

🔴 **THE SOURCE DIRECTORY AND THE BUNDLE KEEP THEIR OLD NAMES**, as
`src/af3/` does for openbind0: a path is not the model's name, and renaming
`model-esmfold2-int5` would move 366 MiB for nothing.

🔴 **AND A RENAMED KEY BREAKS EVERY LOOKUP THAT WAS SPELLED OUT.**
`MODEL_LABELS.esmfold2` and `MODEL_STEMS.esmfold2` silently became `undefined` -
a status line reading "undefined · loading" and a stem of `undefined_1` - and
`loadManifest("esmfold2")` threw "unknown model family". Two REGEXES also
matched only bare identifiers, so `"ef2-fast-600m"` needed quoting and both
stopped seeing it: `test/model-family.test.js` reported a table with one family
missing, and `build_site.py`'s registry cross-check reported the family as
absent from the file that defines it.

🔴 **AND THE STEP DIAL IS NAMED FOR THE SAMPLER IT DISCRETISES.** Both arms read
"Steps", which says nothing - the numbers differ by an order of magnitude
precisely BECAUSE a flow step walks the whole schedule and a diffusion step
discretises it. They are "Flow" and "Diffusion" now, which also avoids the
"Cycles beside Recycles" reading that made them both "Steps" in the first place.

🔴 **AND AF3's DIFFUSION LADDER LANDS ON 200, WHICH IS WHAT IT WAS TRAINED
WITH.** The powers-of-two ladder (20, 40, 80, 160, 320) never offered the
model's own setting, so it was the one number the page could not select. It is
25, 50, 100, 200 - and 25 keeps the floor that was measured, since below twenty
the sampler does not land and ten gives 5.91 A on 6MRR with a CA-CA of 8.40 A.

## A language model instead of an alignment: ESMFold2

Not shipped, and not started as code. `docs/ESMFOLD2.md` is the investigation:
whether ESM-C 600M can be compressed enough to fold from a single sequence in a
browser, which is the case an MSA search cannot serve at all. The three things
worth knowing without opening it:

🔴 **THE CHECKPOINT IS bfloat16 STORED AS float32**, so the first 2x off a 2.30
GB download is not compression, it is padding - not one of 95.6M weights sampled
has any of its low sixteen mantissa bits set, and float16 is lossless on it to
3.2e-9. Quote a scheme against the 16 bits that are really there.

🔴 **ESM-C QUANTISES LIKE AF3 DOES**, so `tools/quantize_af3.py`'s own int5
group-32 asymmetric packer transfers: relRMS 3.8e-2 in weight space here against
4.3e-2 on AF3's six biggest tensors. The tower passes that through at about unit
gain - 4.4e-2 in the pair representation ESMFold2 receives - and it puts the
600M tower plus its folding model at **533 MiB**, against `model-af3-int5`'s
264.6.

🔴 **AND SPENDING BITS WHERE THE LAYER MIX IS HEAVY DOES NOT WORK.** ESMFold2
takes 58.8% of its softmax from the last three of 37 states, and giving those
blocks more precision loses to a uniform allocation at every budget tried. The
mix says where a state is READ, not where precision matters; every block feeds
every later one.

🔴 **AND MEASURE A SCHEME AGAINST THE SAMPLER'S OWN SPREAD, NOT AGAINST ZERO.**
Two seeds of the SAME float32 weights move a structure by 0.99 A on average and
7.06 A at worst over 32 folds, so int5's 0.43 A mean and 3.32 A worst are not a
cost. Sixteen held-out targets, released after the checkpoint's cutoff, and the
damage is a TAIL - the median target moves 0.26 A even at three bits and the
median crystal RMSD is flat at 2.52-2.57 A the whole way down; what changes is
how many targets flip basin (0 at int8, 2 at int5, 3 at int4, 4 at int3).

🔴 **AND CALIBRATED QUANTISATION IS WORTH ABOUT ONE BIT, ONLY AT THE BOTTOM.**
GPTQ and llama.cpp's importance matrix both fit LocalFold's existing decoder -
same asymmetric codes, same float16 scale and zero per group of 32 - and the
whole 573M tower calibrates in 70 s on an A100 against 288 UniRef50 sequences.
imatrix int3 at 4.00 bits matches plain int4 at 5.00; GPTQ int4 beats plain
int4 by 14%; at int5 there is nothing left to win. It does NOT close the
int4-to-int5 gap, so int5 is still the smallest free scheme. **GPTQ makes the
WEIGHTS worse (9.78e-2 against 7.91e-2) and the OUTPUT better (7.87e-2 against
9.42e-2)** - a weight-space study reports it as the worse method - and at THREE
bits it is the worst arm structurally while still winning on the pair metric,
which is the sharpest argument in this repository for folding rather than
trusting a tensor norm. `tools/esmc/gptq.py`.

🔴 **AND THE COMPRESSIBLE MODEL IS THE WEAK ONE.** ESM-C 300M folds as well as
600M here (median 2.55 A against 2.52, a third of the seed spread) and puts the
bundle at 361 MiB - but both are paper ABLATION checkpoints with no confidence
head at all, and the released ESMFold2-Fast folds from ESM-C 6B, which is 4672
MiB at int5. Compression is not the obstacle; the accuracy of the checkpoint
that fits is.

## A second set of weights: the dialect

🔴 **THE MODEL IS OpenBind-0, AND THE NUMBER IS PART OF THE NAME.** Upstream's
announcement (openbind.uk, 2026-08-21) calls it that and points at
`aqlaboratory/openfold-3` releases/tag/v0.5.0, which is the release ported
here. Their registry's bare `openbind` is a name a LATER release would answer
to as well - which is exactly how `openfold3` came to mean two models with
different forward conventions and sent this port reading notes about the wrong
one. So the family, the dialect, the bundle and the manifest's `model.name` all
say `openbind0`, and `dialectFor("openbind1")` RAISES rather than resolving.
`openbind` stays as an alias in two places only - `DIALECT_ALIASES`, because
upstream publishes the blob under that name, and `MODEL_ALIASES`, for `?model=`.

🔴 **OPENBIND IS NOT OPENFOLD3, AND READING THE OF3 PORTING NOTES WHOLESALE GETS
TWO THINGS BACKWARDS.** `../alphafold3/OF3_AF3_PORTING_NOTES.md` describes
OpenFold3 **preview-2** (`of3-p2-155k.pt`). OpenBind is OpenFold3 **v0.5.0**, a
separate model in that checkout's `model_config.MODELS`, and it moved TOWARD
AlphaFold 3 in exactly the two places that would have cost the most here:

- **the column attention's pair bias.** Preview-2 computes `Linear(z[k, q])`;
  AF3's Algorithm 15 says `Linear(z[q, k])`. Upstream's list is
  `TRANSPOSED_COLUMN_PAIR_BIAS` and **openbind is deliberately not in it**, so
  `swapTransposedBias` stays **false** - the same value stock AF3 uses. The flag
  already in `fold.js` reads "stock AF3 is false, the openfold3 lineage true",
  which is about the OTHER release; taking it as "the non-AF3 dialect" would
  transpose the bias in the pairformer, the MSA stack, the template embedder and
  the confidence head against weights that do not want it.
- **the diffusion transformer's pair LayerNorm**, which preview-2 runs per block
  and v0.5.0 runs once for the whole stack, as AF3 does. Their release note:
  "Moved the pair layer norm in the diffusion transformer out of attention pair
  bias. The pair layer norm is run once to match the AlphaFold3 SI."

🔴 **SO THE RUNTIME PORT IS THREE BRANCHES, NOT THIRTY.** Everything else in
those notes is absorbed by the weight converter, because it is a row permutation
of a weight matrix: the residue-alphabet permutation, the i/j crossing between
AF3's two pair-embedding sites, the SwiGLU gate/value concatenation, and the
element index shift - `one_hot(e - 1) @ W` is exactly
`one_hot(e) @ W[max(0, arange - 1)]`, which `converters/common.py` proves to
max|d| = 0. `src/af3/dialect.js` is the table:

| flag | what changes | where |
|---|---|---|
| `symmetriseBonds` | the token bond matrix sets `[j][i]` too | `featurise.js` |
| `maskPaddedKeys` | `offsets_valid &= keys_mask` in the atom cross-attention | `atom-encoder-{reference,webgpu}.js` |
| `padSingleCondUnknownDna` | the diffusion single conditioning is 833 channels, not 831 | `diffusion-{reference,conditioning-webgpu}.js` |

🔴 **AND THE BUNDLE NAMES ITS OWN GRAPH, so a caller cannot pair them wrongly.**
`af3Dialect(store)` reads `manifest.model.name` - which `export_af3_model.py`
has always written - and `trunkWeights`, `confidenceWeights`, `diffusionWeights`
and `targetFeatureWeights` each stamp it onto what they return. An unnamed
bundle RAISES rather than defaulting to stock: a ported checkpoint folded
through AF3's branches returns a structure, which is the failure this exists to
prevent.

🔴 **A ZERO COLUMN IS FREE BEFORE A LINEAR AND IS NOT FREE BEFORE A LAYERNORM.**
That is the whole of the third flag. OpenFold3's restype and profile blocks
carry 32 classes to AF3's 31, and everywhere else the extra class is dropped
from the converted weights because a column that is always zero contributes
nothing to a matrix multiply. The diffusion single conditioning LayerNorms its
concatenation, so a zero input maps to `-mean/std` AND the width becomes 833:
upstream measures dropping the two columns at 2.2e-3 against 3.4e-7.

🔴 **AND THE GPU FIX IS A SENTINEL, NOT A NINTH BINDING.** `maskPaddedKeys` is a
boolean AND in the CPU reference. On the GPU the keys' reference space is
already in the gathers buffer, real space uids are counters and never negative,
so writing **-1** into a padded key makes the equality test fail on its own -
which is `(q == k) && keys_mask` exactly, with no shader change and no extra
storage buffer. The QUERIES stay at zero: upstream gates on `keys_mask` alone,
and masking both would be a third model. The comment beside it that says a
sentinel "is the tidier choice and a different model; it cost 3.1e-2" is about
the STOCK dialect and is still true there.

🔴 **BOTH ARE CHECKED BY SWEEPING THE DIALECT AS AN AXIS, WITH A DISCRIMINATING
CONTROL.** Two arms agreeing with their reference says the GPU matches the CPU;
it does not say the flag reached either. `check-af3-atom-encoder.js` and
`check-af3-diffusion-conditioning.js` both fail if the openbind arm does not
DIFFER from the stock one. Measured:

| | openbind vs alphafold3 | each arm vs its reference |
|---|---|---|
| atom encoder `pairCond` | **7.77e-2** | 2.48e-7 |
| ...`tokenAct` | 2.96e-6 | 9.48e-6 |
| ...`skipConnection` | 3.72e-6 | 2.05e-5 |
| single conditioning | **1.54e-3** (2370x) | 6.48e-7 |

🔴 **AND THE ENCODER'S OUTPUT BARELY MOVES, WHICH IS NOT A BUG.** The atom pair
conditioning moves by 7.8e-2 because that is where a padded key's offset term
lived; almost none of it reaches `tokenAct`, because the attention masks those
same keys anyway. What leaks is the mask bias being a large FINITE negative
rather than -infinity, so a padded key keeps a softmax weight of about 1e-6. So
the control is "some output moved past its envelope", not "every one did" -
demanding all three fails on a correct implementation.

🔴 **AND THE STOCK PATH IS BIT-IDENTICAL, WHICH IS THE OTHER HALF OF THE GATE.**
`tools/gpu/fold.js` before and after the dialect work: PDB SHA
`aef231158a174daf` both ways, mean pLDDT 86.13324126798517 and pTM
0.7378317753181738 to every digit. Per-stage: target_feat 7.98e-8, denoiser
1.28e-4, atom encoder 9.48e-6 / 2.05e-5 - every one of them the figure already
recorded in this file.

🔴 **AND OpenBind FOLDS, WITH ITS OWN NUMBERS.** `openbind.bin.zst` is
published already converted, `read_blob` in `tools/export_af3_model.py` now
reads that format directly - no torch, no 2.3 GB checkpoint, no second checkout
- and the two existing tools do the rest:

```
python3 tools/export_af3_model.py --model openbind --include diffuser \
  --out model-openbind-full-f32           # 406 tensors, 368.4 M, 1405 MiB
python3 tools/quantize_af3.py --source model-openbind-full-f32 \
  --out model-openbind-int5               # 264.6 MiB, 5.31x
node tools/gpu-chrome.mjs tools/gpu/fold.js --model=/model-openbind-int5/manifest.json
```

Scored against `tools/fixtures/6mrr-crystal.pdb`, both bundles at int5:

| | RMSD | TM | pLDDT | CA-CA |
|---|---|---|---|---|
| alphafold3 | **0.67 A** | **0.952** | 85.8 | 3.82 |
| openbind | 1.99 | 0.833 | 79.4 | 3.78 |
| openbind, `swapTransposedBias` forced TRUE | 2.03 | 0.820 | 76.7 | 3.78 |

The AF3 row reproduces this file's own quantisation table (0.66 / 0.953), which
is what says the harness is sound. **The third row is the experiment that
matters**: it turns on the convention OpenFold3 preview-2 was trained with, and
it is WORSE on every measure - so `swapTransposedBias: false` is confirmed
against the weights themselves rather than against a reading of somebody else's
table. One target, so read it as a direction and not a margin.

🔴 **AND THE SHAPES CONFIRM THE WHOLE ANALYSIS INDEPENDENTLY.** `openbind.shapes.json`
against the AF3 bundle: **406 arrays each, identical names, exactly two shape
differences** - `single_cond_initial_norm/scale` 833 against 831 and
`single_cond_initial_projection/weights` [833, 384] against [831, 384]. Nothing
else. Preview-2's per-block diffusion pair LayerNorm would have added
twenty-four tensors and does not appear, which is the tree agreeing with the
release note.

🔴 **`--ablate` AND `--enable` PRICE A DIALECT BRANCH, because a branch that is
silent when wrong cannot be trusted on a reading.** `tools/gpu/fold.js
--ablate=maskPaddedKeys` turns one off and `--enable=swapTransposedBias` turns
one on. `padSingleCondUnknownDna` cannot be ablated: it changes the LayerNorm's
WIDTH, so the bundle's 833-long scale stops matching and the conditioning
throws - the structural gate doing its job, and the reason that branch needs no
measurement.

🔴 **A PADDED KEY ONLY EXISTS BELOW 128 ATOMS, AND EVEN THERE IT REACHES
NOTHING.** `featurise.js` clamps the 128-wide key window against the REAL atom
count, so at 128 atoms or more every key lands on a real atom and the key mask
is identically one. `tools/gpu/probe-ablate.js` sweeps the boundary:

| residues | atoms | padded keys | `pairCond` | `tokenAct` | control |
|---|---|---|---|---|---|
| 4 | 32 | 288 of 384 | **4.13e-1** | **0.00e+0** | 1.04e-6 |
| 8 | 67 | 366 of 768 | 2.72e-1 | 0.00e+0 | |
| 12 | 106 | 198 of 1152 | 1.37e-1 | 0.00e+0 | 1.03e-6 |
| 16 | 143 | **0** of 1536 | 0.00e+0 | 0.00e+0 | |
| 68 | 574 | **0** of 6528 | 0.00e+0 | 0.00e+0 | 1.00e-6 |

So the branch moves the atom PAIR conditioning by up to 41% and the encoder's
OUTPUT by exactly nothing. The control column is a 1e-6 nudge to the
conditioning, and it is there because "relRMS 0.00e+0" is also what a broken
comparison says.

🔴 **WHICH IS WHY AN ABLATION ON 6MRR CAME BACK BIT-IDENTICAL.** Same PDB, same
pLDDT to every digit, with `maskPaddedKeys` off - because 6MRR has 574 atoms and
therefore no padded keys at all. The dumps differ on this and it decides what a
checker can see: `af3-6mrr.json` has 6528 of 6528 keys live, while
`af3-oracle-atom-f32.json` has 873 of 1152 - a 97-atom molecule, under the
window - which is why `check-af3-atom-encoder.js` measures a 7.77e-2 separation
and a whole fold measures none.

🔴 **AND THE PADDING IS GONE, BECAUSE NOTHING HERE NEEDED IT.** AF3 pads because
JAX wants static shapes; these kernels are generated per shape and size their
workgroup storage from `keys`, so the window is `min(128, atomCount)` now and
**no batch this featuriser produces has a padded key at any size**. The
`ref_space_uid = 0` collision that OpenFold3 trained around cannot occur here.

Measured before and after, `tools/gpu/fold.js --sequence=`:

| | 68 residues, 574 atoms | 12 residues, 106 atoms |
|---|---|---|
| diffusion 50 | **bit-identical** (`1f3a312051898379`) | 84.3979 -> 84.4696 pLDDT |
| flow 16 | **bit-identical** (`8cd298eb6bd61f82`) | 84.7656 -> 84.8080 |

Exactly the scope the table above predicts: at or above 128 atoms the window was
already inside the molecule and nothing moves, and below it the padded keys stop
contributing. Geometry is unchanged either way (N-CA 1.458, CA-CA 3.807 on the
12-mer).

🔴 **AND `maskPaddedKeys` IS NOT DEAD, WHICH IS WHY IT STAYS.** The differential
checkers do not use this featuriser - they feed AF3's OWN gathers out of an
oracle dump, and `af3-oracle-atom-f32.json` is a 97-atom molecule with 279
padded keys of 1152. So `check-af3-atom-encoder.js` still separates the two
dialects by 7.77e-2 on `pairCond`, and the flag still decides what a bundle
converted for OpenBind computes on somebody else's batch. What changed is that
the SHIPPING path can no longer reach the case.

🔴 **A SECOND MODEL IS MISTAKEN FOR THE FIRST IN A CACHE, NOT IN A LOADER.**
Two bundles now build AF3's graph, and every memo keyed on something that does
not distinguish them is a silent wrong answer. Two were found, one of them the
hard way:

* `loadAf3Weights` memoised ONE promise, so the second family's fold got the
  first family's weights. Caught while writing it - `weightsPromises` is a Map
  keyed by family.
* **`trunkKey` did not include the family.** The cached trunk is a pair and a
  single representation, and those have the same shapes whichever parameters
  produced them - so folding with OpenBind and then AlphaFold 3 on the same
  sequence matched every other field and handed AF3's diffusion head OpenBind's
  trunk. Reproduced in the page at 32 residues: **pLDDT 41.5 with the status
  line reading "trunk reused", against 83.3 once `family` was in the key.**
  Nothing errors, nothing warns; the chain comes apart, which is how it was
  reported ("atoms are no longer attached").

Neither is findable by folding one model. The reproduction is
openbind -> af3 -> openbind in one page session, which
`test/model-family.test.js` pins by asserting the key names the family.

🔴 **AND "IS THIS AF3" IS NOT `family === "af3"` ANY MORE.** That comparison sat
in five places and every one meant "is this the AF3 pipeline", not "is this
DeepMind's checkpoint" - so a second AF3-graph family took the AlphaFold 2
branch at each. Three of them were CAPABILITY guards, and they refused ligands,
modified residues and nucleic chains under OpenBind with a message naming a
capability the model has: *"Ligands need AlphaFold 3; the model is set to
openbind"*. `AF3_FAMILIES` in `src/reference/manifests/index.js` is the list;
`isAf3Family` is the test.

🔴 **AND THE DIALOG SAYS "NOT AVAILABLE FOR COMMERCIAL USE" AND NOT "ACADEMIC
USE ONLY".** The second is the phrase that comes to hand and it is wrong twice:
DeepMind's terms cover non-profits, research institutes, journalism and
government bodies as well as universities, and they exclude a researcher
employed by a commercial organisation. The short form has to stay TRUE while
being short; the linked terms carry the detail. "Not open source" was the
earlier version of the same mistake - the CODE is openly licensed and it is the
PARAMETERS that are restricted.

🔴 **THE LICENCE DIALOG ASKS THE PERSON FOLDING, NOT THE DEPLOYER.**
`build_site.py` already refuses to publish DeepMind's parameters without
`LOCALFOLD_ACCEPT_MODEL_TERMS`; the page's `#model-terms` dialog gates the first
AF3 fold and remembers the answer in `localStorage`. It offers OpenBind as the
alternative rather than only an "I agree", because a dialog with one button
teaches people to click it. `tools/model-terms.py` is the check - it drives the
real page and asserts the dialog opens, remembers, switches the model row, and
**still opens for `?model=af3`**, since a URL must not be able to accept
somebody else's terms.

🔴 **AND A `<dialog>` IS NEVER LAID OUT UNTIL IT IS OPENED**, which is the load
dial's blind spot again: `tools/mobile-layout.py` cannot see it. `model-terms.py`
forces it open under a 320px device override and asserts it is neither clipped
nor sideways-scrolling and that its two buttons stack (measured 298px wide,
buttons 260px, left 11px).

🔴 **AND A `?model=` THAT IS IGNORED LOOKS EXACTLY LIKE ONE THAT WORKED.** The
complaint about an unknown name was written twice before it was visible: once
before the vendored viewer's own "Ready." line overwrote it, and once before the
parameter had even been read, because it was called beside the Fold button's
enabling - which runs EARLIER in the file. It waits for that specific string
now. `of3` is deliberately not an alias for `openbind`.

## Upstream's optimisation work, tried here

`martin-steinegger/alphafold2-webgpu` is the `upstream` remote. The trees have
diverged too far to merge - 187 commits there, 518 here, and theirs is
TypeScript - so what transfers is findings, not code. Three were tried on this
M2. One is a large win, one does not reproduce, and one is the opposite of what
their hardware says.

🔴 **THE SUBGROUP MATRIX UNITS EXIST ON THIS DEVICE, AND THEY BEAT THE f16
KERNEL WHILE COMPUTING THE f32 ONE.** `chromium-experimental-subgroup-matrix` is
advertised by this adapter and the WGSL compiles;
`tools/gpu/probe-subgroup-matrix.js` reports what it offers, which is
**8x8x8 at f32/f32 and 8x8x8 at f16/f16** - note there is no f16-input,
f32-accumulate configuration here, so the f16 units accumulate in f16 and are a
different accuracy question. `tools/gpu/gemm-matrix.js` is the candidate kernel
and `bench-evoformer-linear.js` has `matrix<blocks>` arms. Against the shipped
dense projection, medians of nine interleaved in one process:

| shape | f32 8x8 | shipped f16 | **matrix f32** | vs f32 | vs shipped |
|---|---:|---:|---:|---:|---:|
| MSA transition, first half | 17.14 ms | 14.21 | **12.25** | 1.40x | 1.16x |
| ...second half | 16.69 | 14.11 | **10.80** | 1.55x | 1.31x |
| structure/confidence single | 0.188 | 0.150 | **0.088** | 2.14x | 1.70x |
| a long chain's pair transition | 3.63 | 3.13 | **2.73** | 1.33x | 1.15x |

The matrix arm's relRMS against the f32 kernel is **0**, at every shape and
every row count tried - it accumulates in f32, so there is no accuracy gate to
pass. That is the whole point: this repository buys 1.15x-1.31x today by
rounding to half precision, and the matrix units buy the same or more by not.
`requestAlphaFoldDevice` now asks for the feature (optionally, so a browser
without it never sees it requested); nothing in `src/` uses it yet.

🔴 **AND THE SECOND ROW OF THAT TABLE IS NOT A SHIPPABLE 1.31x, BECAUSE THAT
KERNEL'S SOURCE IS PACKED.** Every arm in `bench-evoformer-linear.js` reads an
f32 source, so the arms compare fairly with each other and only the FIRST half's
shape is the configuration that ships: `block.js` stores the transition's hidden
activation as `f16` whenever `hiddenChannels % 4 == 0`, which the MSA
transition's 1024 and the pair transition's are, and the second matmul reads it
through `storedElement` - an `unpack2x16float` expression. `subgroupMatrixLoad`
cannot consume an expression. So taking the matrix path there means storing
`hidden` unpacked, which src/runtime/storage.js records as 16 MiB at 512 MSA
rows for a BIT-IDENTICAL fold - a free win being given back. That is a real
trade to weigh, not a number to quote.

🔴 **AND AN OUT-OF-BOUNDS `subgroupMatrixLoad` RETURNS AN ENTIRELY ZERO MATRIX
HERE, WHICH IS NOT WHAT UPSTREAM'S KERNEL ASSUMES.** Their bounded kernel runs
the matrix path everywhere and bounds-checks only in the store, on the stated
reasoning that "loads past the end of a tensor are clamped by WGSL's robustness
rules, so a partial region computes garbage exactly in the rows and columns that
do not exist". On this device it does not. `check-subgroup-matrix.js` loads an
8x8 tile from a buffer holding five rows and **every row comes back zero** -
relRMS 1.0 across the whole tile, the five present ones included. A scalar read
of that buffer is clamped; a matrix read of it is refused wholesale. The first
version of `gemm-matrix.js` was exact whenever M was a multiple of 32 and read
0.153 at 59 rows, which is what that looks like.

So the load has to stay in range. The last region on each axis **slides back**
to end on the final row and column instead of hanging over the edge; the overlap
recomputes rows with the same inputs and writes the same values, and the kernel
then needs at least one whole region per axis. That is a documented restriction
rather than a silent wrong answer - which is what the 64x128 arm still gives
below 64 rows.

🔴 **AND THE SEMANTICS WERE PINNED BEFORE ANYTHING WAS TIMED.** The type
parameters are `<T, columns, rows>` and at the only shape this device offers -
8x8x8 - getting that backwards is invisible in the declaration and visible only
in the answer. `check-subgroup-matrix.js` multiplies one asymmetric 8x8 pair
whose product is known on the host and scores the seven interpretations a
transpose could produce: plain row-major `A@B` at **relRMS 0**, everything else
above 1.1. A bench run before that check would have been timing a transpose.

🔴 **AND THE 64x128 GEOMETRY IS A REAL LOSS HERE, WHICH TOOK TWO GOES TO SAY.**
Upstream reports the shipped-grid geometry at 1.28x-1.66x, worth about a fifth
of the win. The first arm written here read 122-172 GFLOP/s against 1082-1466
for the 32x32 one, and that was an implementation fault, not a device fact:
holding 8x16 accumulator tiles is 128 of them, 8192 floats a subgroup, and it
spills - the same 4x-the-wrong-way `grid.project`'s row tile records at 16.
`subBlocks`/`subColumnBlocks` walk the region a sub-region at a time instead, so
the register budget is flat and the geometry is the caller's; `matrix8x16x4x4`
is that arm, and it is exact (relRMS 0 at 64 and at 128 rows).

It is still a loss, by a factor of four, and now the number means something:

| shape | 32x32 | 64x64, walked 4x4 | 64x128, walked 4x4 | 64x128, walked 8x8 |
|---|---:|---:|---:|---:|
| transition, first half | **1284** | 590 | 306 | 210 |
| ...second half | **1456** | 717 | 361 | 250 |
| a long chain's pair transition | **1082** | 441 | 230 | 167 |

One workgroup is one subgroup - the store's uniformity requirement - so a 64x128
region is an EIGHTH of the workgroups doing eight times the sequential work, and
this device would rather have the occupancy. Upstream's M4 Pro would not, which
is the same shape of disagreement as the queries-per-invocation one below. **So
the grid-compatibility problem is not a fifth of the win here, it is all of it**:
a caller taking this path needs a matrix-specific dispatch grid, not the one
`gemmGrid` derives from the shipped tile.

🔴 **AND AF3's PROJECTIONS ARE ALREADY AT THE MATRIX CEILING, SO THE WIN IS
AF2's ALONE.** The trunk's two hottest dense passes were measured against a
matrix GEMM of identical M, K and N - 40000 x 128 x 512 at 200 tokens - **timed
in the same process**, because a comparison drawn across two runs of anything
here is inside this machine's drift, and the cross-process version of exactly
this comparison read 4.14 against 4.70 ms and would have said the opposite:

| | shipped | matrix f32 | |
|---|---:|---:|---|
| `grid.project` (row tile 8) | **1287 GFLOP/s** | 1131 | the fused kernel wins by 1.14x |
| `tri.project` (32x16) | 1073 | **1125** | 1.05x, inside the noise |

`--matrix=1` on `bench-grid-project.js` and `bench-triangle-project.js` is that
arm. Both AF3 kernels are FUSED - one read of the normalised pair
representation, four projections out of it, two of them through a sigmoid gate -
and that fusion is worth about what the matrix units are. AF2's transition is a
generic unfused `createLinearShader`, which is exactly why the matrix path beats
it by 1.16x-1.31x and does not beat these. **Ask what a kernel already fuses
before pricing its arithmetic.**

🔴 **PACKING THE ATTENTION VALUE COSTS NOTHING HERE, BECAUSE THIS KERNEL HAD
ALREADY ROUNDED IT.** Upstream found that packing the flash kernel's keys AND
values moved an evoformer block's MSA output 4.06e-4 from AlphaFold's own
intermediates against a 5e-5 allowance, with the value alone reproducing 4.05e-4
- a key's error is normalised away by the softmax, a value's is averaged under
weights summing to one and lands undamped. They now pack keys only. This tree
packs all four projected tensors, so it looked like the same bug.

It is not, and the reason is `chunk16`: the default flash kernel stages the key
and value chunks as `vec4<f16>` in workgroup memory **whatever the tensors are
stored as**, so the value is half precision by the time it is used either way.
`tools/gpu/check-attention-packing.js` against a CPU reference, with the dense
kernels forced to f32 so the storage is the only rounding left:

| | flash f32 | flash chunk16 (the default) |
|---|---:|---:|
| nothing packed | 1.9e-7 | 8.97e-5 |
| query/key/gate packed | 1.30e-4 | 1.53e-4 |
| **value packed** | 8.09e-5 | **8.97e-5** - unchanged |
| both | 1.53e-4 | 1.53e-4 |

So unpacking the value buys nothing under the shipped kernel and costs a
tensor's bytes; `ATTENTION_VALUE_STORAGE` stays `f16`. The mechanism to separate
them exists now and is threaded through `selectAttentionFlashKernel`,
`selectAttentionProjectKernel` and `block.js`, because the answer is a property
of the precision and would change on a device without `shader-f16` - where the
value costs 8.09e-5 and is the SMALLER of the two terms, not the larger.

🔴 **AND `inputStorage` IS THREE TENSORS, NOT THE KEY.** It is the query, the key
and the gate, and the query and the gate are read once per invocation rather than
once per key - so the 1.30e-4 row above is not "what the key costs". Two of those
three are narrowed for no bandwidth at all.

🔴 **AND `AttentionGpu` HAD NEVER RUN THE STORAGE THE MODEL RUNS.** It took no
storage option at all, so `check-evoformer-attention.js` - AF2's only attention
differential - was checking an all-f32 configuration that nothing ships, which
is the same fault as a checker building its own kernel. Storage is an axis
there now, and the packing checker asserts its four arms compiled four DIFFERENT
shaders, because a storage option that never arrives reports perfect agreement.

🔴 **AND A PACKED WORD'S TWO COLUMNS ARE A PROPERTY OF THE LAYOUT, SO EVERY
PACKED BINDING DECIDES IT.** The projection's column mapping was gated on
`packOut` alone - the flag for query, key and gate. Packing only the VALUE kept
the lanesX-strided mapping and then wrote `pack2x16float(hd_0, hd_1)`, two
columns EIGHT apart, into the word belonging to `hd_0` and `hd_0 + 1`. Every
shape agreed, nothing was out of bounds, and the attention scored **relRMS
0.528** against its reference. It follows `packOut || packValue` now.

🔴 **AND TWO QUERIES AN INVOCATION IS 4.8x SLOWER HERE, WHERE UPSTREAM'S OTHER
DEVICE WANTS IT.** Their `attentionFlashKernelForShape` gives an invocation two
queries once a shape reaches 128, from a GB10 measurement of 1.17x-1.42x for 128
to 1024 queries; on their M4 Pro it is 2.2x slower and they replaced the
threshold with a probe. This kernel has the same knob and selection has never
used it, and `bench-msa-attention.js` says why - at 512 queries, 59 batch, 8
heads:

| | q1 | q2 | q4 |
|---|---:|---:|---:|
| `auto/c` | 17.03 ms | 82.48 (**0.21x**) | 196.28 (**0.12x**) |

Bit-comparable at 2.84e-7, and catastrophic, which is the same register-spill
shape AF3's `grid.attend` records at Q=2 and Q=4. So nothing changes here; what
is worth taking from upstream is that the ratio is a DEVICE property and the
answer differs by a factor of six between two of them.

🔴 **AND WHERE THE MATRIX UNITS WOULD PAY, BY MODEL.** The share that is a dense
projection at all, from `profile-af2-block.js --sequences=512` and
`bench-trunk.js --profile --tokens=200`:

| | AF2 (monomer, multimer) | AF3 (af3, openbind0) |
|---|---|---|
| plain GEMM passes | **65%** of an 83.1 ms block | **49%** of a 3372 ms trunk |
| fused GEMM | - | `pair-transition`, a further 18% |
| out of reach | the two flash attentions, 20% | `grid.attend`, 19% |

AF2's dense work is `createLinearShader` and the attention's own projection, so
the table at the top applies to it directly. AF3's two hottest were measured and
are at the ceiling already - see above - which leaves `pair-transition` as the
only one that might still move: it fuses a LayerNorm, two matmuls and a gate,
and is 59% arithmetic rather than bandwidth. It is also the furthest from a
drop-in, and the two that WERE measured both say fusion is worth as much as the
units are.

🔴 **AND THE CALLERS ARE THE OBSTACLE, NOT THE KERNEL.** `subgroupMatrixLoad`
cannot consume a WGSL expression: it needs a typed binding, a base offset and a
stride. Every generated kernel here takes its operands as expressions, which is
what lets a caller read a packed activation through `unpack2x16float` or window
a tensor past a binding limit - so the matrix path cannot be made invisible the
way half precision was. A caller has to declare that its operand IS a plain
array with a known stride. That is the reason this stops at a measurement.

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

## Hosting the weights somewhere other than Pages

GitHub Pages publishes at most a gigabyte, and the weights are most of it: AF2
monomer 227 MB, AF3 150 MB, before a third model exists. A page meaning to offer
five keeps its parameters elsewhere.

Everything a bundle needs is one field. In `src/reference/manifests/index.js`:

```js
af3: {
  directory: "./model-af3-int5/",                       // the fallback
  remote: "https://huggingface.co/USER/REPO/resolve/<sha>/",
  ...
}
```

and that is the whole change. Shard URLs are resolved against the bundle's base,
so the store never learns the difference; `build_site.py` and the Pages workflow
both ask `build_site.py --is-remote <family>` and stop publishing a copy.

🔴 **THE ESM-C BUNDLE IS 54 SHARDS WHERE AF3's IS 8, AND MEASURED, THE 8 IS THE
ONE THAT COSTS.** `export_esmc_model.py`'s `SHARD_LIMIT` is 48 MiB applied to the
FLOAT32 export - 2190 MiB, about 46 pieces - and `quantize_af3.py` preserves that
layout rather than re-sharding, so int3 shrinks each eightfold to a 4.1 MiB
median while the count stays. That looked like an oversight and is not.

Fetched from Hugging Face at eight connections, longest first, two interleaved
passes:

| bundle | shards | MiB/s | with a connection idle |
|---|---|---|---|
| `af3-int5` | 8 | 27.9 / 30.1 | **5.30 / 4.52 s** |
| `esmc-600m-int3` | 54 | 24.0 / 28.1 | **1.70 / 0.62 s** |

🔴 **EIGHT SHARDS ON EIGHT CONNECTIONS IS NO PACKING AT ALL.** Every connection
takes one shard, the first to finish has nothing else to do, and the load ends
when the single SLOWEST shard does - which is half the download running
under-parallel. Fifty-four costs 1.83 s of request overhead against eight's 0.27
(the fixed cost of a shard request is a measured **271 ms**, the 307 to
`cdn.hf.co` included) and recovers more than that in packing. Throughput is the
same within this machine's noise either way.

**So the "eight" recorded elsewhere in this file is eight parallel CONNECTIONS
and a longest-first order, not a shard count** - and a bundle wants comfortably
more shards than connections, not the same number.

🔴 **AND AF3 IS THE EXCEPTION, BECAUSE ITS FLOOR IS A TENSOR AND NOT A LAYOUT.**
Two of its float32 tensors are **216 MiB** each - the stacked `transition1`
weights of the diffusion transformer and of the trunk pairformer's single
transition - which are 40.5 MiB apiece at int5. A tensor is contiguous within
one file, so 40.5 MiB on one connection is the makespan floor in ANY sharding,
and the shipped layout already isolates them: tensors per shard reads
`[1, 1, 65, 66, 66, 68, 69, 70]`. Re-sharding to sixteen produces the same two
40.5 MiB shards and gains nothing, which is why it was not done. **Its 4.5-5.3 s
idle tail is those two tensors.**

🔴 **AND af2-monomer's `.js` SHARDS ARE GONE, 129.8 MiB OF THEM.**
`tools/export-js-weights.py` writes a base64-in-JavaScript copy of every shard
for `file://` pages, which is why that bundle had eighteen files where the
others had nine - and `build_site.py` excludes `weights-*.js` and `manifest.js`
from the site, so nothing ever fetched them over HTTP. They are still generated
locally for the offline page; they are simply not hosted.

🔴 **PIN A COMMIT SHA, NOT `main`.** A shard fetched from a moving branch can
change under a manifest that did not, which is the failure the shard-cache token
exists to prevent - and three separate hours have already gone into "<file> has
an invalid byte length", a message that names neither half.

🔴 **AND A TRAILING SLASH, OR THE LAST SEGMENT IS LOST.** `new URL(file, base)`
against ".../resolve/abc123" puts the shard beside `abc123` rather than inside
it. `bundleBaseUrl` adds one; `test/model-bundles.test.js` holds it to that.

Verified against Hugging Face from the browser: CORS passes, the 302 to
`cdn.hf.co` is followed, `?v=` cache tokens survive, ranges answer 206, and the
responses come back `type: "cors"` so the shard cache can store them.

🔴 **AND EVERY BUNDLE IS HOSTED NOW, SO THE PAGES BUILD CARRIES NO WEIGHTS AT
ALL.** `sokrypton/localfold` holds all eight, one directory each. The four
EF2-fast bundles were the last to go up and were 598.6 MiB of a 1 GB allowance
until they did; the build now publishes **0.0 MiB** of parameters.

```
hf upload sokrypton/localfold model-esmfold2-int5 ef2-fast-600m-int5 --repo-type=model
```

🔴 **AND `remote_families()` MATCHED ONLY UNQUOTED KEYS, WHICH IS THE OPPOSITE
OF SAFE.** Three of the four EF2 families are quoted in `index.js` - a key is,
when it is not a bare identifier - so their `remote:` lines were read as
belonging to no family and the build counted hosted bundles as LOCAL. It would
have published 375 MiB a second time, on top of the copies the browser fetches
from Hugging Face: the allowance spent twice for nothing. The same pattern was
wrong in `registry_mismatches` and in two JavaScript tests, all fixed the same
way.

DeepMind's AF3 parameters carry a Prohibited Use Policy - `build_site.py`
already refuses to publish them without `LOCALFOLD_ACCEPT_MODEL_TERMS`. On
Hugging Face the equivalent is a **gated repository**, which is a better fit
than a CI variable because it asks each downloader rather than the deployer.

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
