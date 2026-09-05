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
