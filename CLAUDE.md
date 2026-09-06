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
| **Does LocalFold fold a sequence the way ESMFold2 does?** | `node tools/check-esmfold2-fold.js` |
| Does the diffusion module agree, module by module? | `node tools/check-esmfold2-diffusion.js` |
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

🔴 **PIN A COMMIT SHA, NOT `main`.** A shard fetched from a moving branch can
change under a manifest that did not, which is the failure the shard-cache token
exists to prevent - and three separate hours have already gone into "<file> has
an invalid byte length", a message that names neither half.

🔴 **AND A TRAILING SLASH, OR THE LAST SEGMENT IS LOST.** `new URL(file, base)`
against ".../resolve/abc123" puts the shard beside `abc123` rather than inside
it. `bundleBaseUrl` adds one; `test/model-bundles.test.js` holds it to that.

Verified against Hugging Face from the browser: CORS passes, the 302 to
`cdn.hf.co` is followed, `?v=` cache tokens survive, ranges answer 206, and the
responses come back `type: "cors"` so the shard cache can store them. What is
NOT verified is a real upload - there were no HF credentials on this machine, so
the repository and the push are still to do.

To upload:

```
pip install huggingface_hub && hf auth login
hf upload USER/REPO model-af3-int5 . --repo-type=model
```

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
