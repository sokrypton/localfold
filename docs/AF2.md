# AlphaFold 2 here: the template terms, the kernels, the alignment prep

Monomer and multimer. `src/evoformer/`, `src/multimer/`.

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

## The monomer embedder, which is a third dialect

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

## The gates: one end-to-end, four differential

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

🔴 **AND AlphaFold 2 SAVES ITS BEST PASS, NOT ITS LAST.** Recycling is not
monotonic and AlphaFold's own pipeline ranks its outputs; the criterion is
ColabFold's `rank_by: auto` - the multimer score for a complex, mean pLDDT for a
monomer - and the search starts from the last pass so a tie keeps the more
converged one. The scores card and the status line report the saved pass, and
the line says `saved pass N of M` when it is not the last, because the play bar
is still sitting on the last one.

## 825 residues, and the two dispatches that had never been folded

The monomer path had a **length ceiling nobody had hit**, because nothing here
folds a long monomer. At 825 residues with AlphaFold's own preset - 512 clusters
against 1024 extra - `fold-af2.js` refused twice, one dispatch apart:

```
triangle.outgoing.project         needs 85079 workgroups in y, over the 65535 limit
triangle.outgoing.normalize-input needs 85079 workgroups in x, over the 65535 limit
```

A pair track has n^2 rows and `ceil(pairs / tile)` passes 65535 at **725
residues** with a row tile of 8. Both kernels have folded ALL ALONG -
`row0 = (group.y + group.z * PROJECT_GRID_WIDTH) * TILE_ROWS` in one,
`base_row = (group.x + group.y * LINEAR_GRID_WIDTH) * NORMALIZE_ROWS` in the
other, the second with a comment addressed to "a caller that has not [folded]".
Only the callers passed flat counts.

🔴 **THE PROJECTION WAS FOLDED IN THE MULTIMER TWIN AND NOT THE MONOMER; THE
NORMALISE WAS FOLDED IN NEITHER.** AGENTS.md warns that a change threaded
through one of those two files and not the other is a bug only a multimer fold
can see. This is that warning backwards - fixed in multimer, never threaded to
monomer, so only a long MONOMER could see it - and the normalise would have
stopped a long complex too. Both fixed in both files. The 59-residue gate
returns checksum **-2537784** before and after each fix.

🔴 **AND THE TOOL WAS ASKING FOR HALF THE WORK IT REPORTED.** Clusters are taken
first and the extra rows come from the remainder (`extraPool =
remainder.slice(maxMsa - 1)`), so a `max(512, 1024)` = 1024-row synthetic
alignment gave 512 clusters and **512** extra while the report said 1024. It
synthesises `rows + extraRows` now. Worth 139.7 s -> 149.7 s at 825 residues.

## Against native AlphaFold 2, and against another WebGPU port

Native is `alphafold3.af2` from sokrypton/alphafold3's `af3-any-model` branch -
AF2 vendored inside the AF3 package - bfloat16, Triton flash attention, 93.2M
parameters, 48 evoformer blocks and 4 extra-MSA blocks, templates instantiated.
The other port is **martin-steinegger/alphafold2-webgpu**, run from its own
published q8 bundle through *this* repository's `gpu-chrome.mjs`, because its
Dawn node binding wants GLIBC 2.38 and this box has 2.35.

All three on the same A100, **one trunk pass**, 512 clusters / 1024 extra:

| 825 residues | one pass | vs JAX |
|---|---:|---:|
| native AF2 (JAX) | **5.27 s** | 1.0x |
| alphafold2-webgpu (warm) | **14.21** | 2.7x |
| **LocalFold** | **149.74** | **28x** |

and at *their* published benchmark shape, 59 residues with 128/256:

| 59 residues | one pass |
|---|---:|
| native AF2 (JAX) | **46.2 ms** |
| alphafold2-webgpu here (warm) | 1260 |
| their published figure, on an L40S | 700 |

🔴 **SO LOCALFOLD'S AF2 IS TEN TIMES THE OTHER WEBGPU PORT, WHICH IS THE FINDING
THAT MATTERS HERE.** They carry machinery this path has none of: a
`planMonomerDevice` that sizes limits from the shape, a scratch budget fitted to
host memory (worth **19.06 -> 14.21 s**, 1.34x, exactly the "third of its speed"
their README predicts), and packed activation storage with an `EXACT_STORAGE`
escape for the differential tests. LocalFold's monomer has no equivalent knob.
Every optimisation in docs/A100.md went into AF3.

🔴 **AND THE FIRST PLACE TO LOOK IS ACTIVATION PRECISION, WHICH IS MEASURED AND
NOT GUESSED AT.** Their monomer defaults three storages to f16 -
`triangleWholeStorage`, `msaStorage`, `pairStorage` - and `WebGpuExecution.allocate`
here takes `storage = "f32"` with **no caller in `src/evoformer/` or `src/model/`
passing anything else**. Every evoformer tensor in this path is f32. At 825
residues that is:

| tensor | f32 | f16 |
|---|---:|---:|
| pair, `L^2 x 128` | 348 MB | 174 |
| MSA, `512 x L x 256` | 433 | 216 |
| triangle whole projection | 348 | 174 |

about **565 MB of activation traffic per touch** that they halve and this does
not, across 48 blocks. The machinery to do it exists - `storageBytes` and the
`storage` argument are already threaded through `allocate` - and nothing calls
it. Not attempted here; written down because it is the cheapest thing to try
first and because "their kernels are better" is not a finding, it is a shrug.

🔴 **AND THE GAP TO JAX IS MOSTLY LAUNCH OVERHEAD, WHICH THE TWO SHAPES SHOW.**
At 59 residues native is **27x** the other port; at 825 it is **2.7x**. JAX
compiles the whole graph; a WebGPU port issues thousands of small dispatches,
and that fixed cost stops mattering as the tensors grow.

## Pair-blocking the outer product mean: 825 residues, 150 s -> 29.6 s

The 64 MiB constant above was one of two things keeping this path off its own
fast kernel. The other was that the fast kernel's intermediate was
`L^2 * cOuter^2 * 4` - **every pair at once** - so its size was a function of
the protein. At 825 residues that is 2.79 GB, which `maxStorageBufferBindingSize`
refuses outright, and the fold fell back to `opm.accumulate` again.

🔴 **SO THE INTERMEDIATE IS NOW A BLOCK OF PAIRS AND THE PROTEIN'S LENGTH IS NOT
IN THE ALLOCATION.** The contraction indexes the intermediate by the LOCAL pair
and its operands by the global one; the projection reads local, masks and writes
global. That one substitution is the whole change, and what it buys is that
**every device takes the fast path at every length** - the device limit only
decides how many blocks it takes. Nothing falls back any more except a shallow
MSA, which is the one condition that was ever algebraic (`sequences >= cOuter`).

| 825 residues, 512/1024 | one block | a fold, 0 recycles |
|---|---:|---:|
| before | 2264 ms | 149.66 s |
| pair-blocked | **396** | **29.57** |

**5.7x on the block, 5.06x on the fold.** Against the two references on the same
card, at the same shape:

| 825 residues, one pass | | vs JAX |
|---|---:|---:|
| native AF2 (JAX) | 5.27 s | 1.0x |
| alphafold2-webgpu (warm) | 14.21 | 2.7x |
| **LocalFold, before** | 149.74 | 28x |
| **LocalFold, pair-blocked** | **29.57** | **5.6x** |

so the gap to the other WebGPU port is **10.5x -> 2.08x**.

🔴 **AND IT IS BIT-EXACT, WHICH IS WHY THE GATE'S BAR IS `=== 0`.** Blocking
reorders no sum - a pair's contraction is untouched, only where it lands in the
intermediate - so `check-opm-paths.js` gets a third arm at a cap small enough to
force 5000 blocks and compares it to the one-block run at exactly zero. A 1e-6
bar there, the right bar for the outer-first-against-tiled arm beside it, would
pass an off-by-a-block index. `fold-af2.js` agrees to the digit too: checksum
-1805925 at the default and at `--tune=opmPairBlockBytes=1048576`, which is 14
blocks instead of one.

🔴 **THE BLOCK SIZE IS A TILE AND THE CURVE IS FLAT, WHICH IS THE USEFUL PART.**
Swept in one process at 825 residues and 512 rows, interleaved, as block time:

| pair block | 16 MiB | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| OPM blocks | 167 | 84 | 42 | 21 | 11 | 6 |
| a block | 411.5 ms | 400.3 | **397.1** | 395.4 | 395.2 | 394.6 |

**4.3% across a 32x range**, and 64 MiB -> 512 MiB buys 0.6% for eight times the
memory. So 64 MiB is the default and `opmPairBlockBytes` is the knob; it is
clamped down by what the adapter will bind, so a small device gets more blocks
rather than a failed fold. On an M2, whose limit is 128 MiB, every real protein
was falling off the path and now none of them do.

## The output projection is bound by a weight read, and P is how many pairs share it

With the fallback gone, `opm.project-output` was the second kernel in the block:
43 ms of 396 at 825 residues and 512 rows. That is **178 GFLOP in 43 ms, 4.2
TFLOP/s against this card's 18.1** - and the reason is arithmetic intensity
rather than the kernel. A workgroup sweeps the whole `CELLS x c_z` output matrix
once, 512 KiB at AlphaFold's widths, for one multiply-add per weight read. Every
pair in the workgroup shares that sweep, so doubling P halves the traffic.

Swept interleaved in one process at 825 residues and 512 rows, as this kernel's
own time:

| pairs a group | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| before the fused reduction | 70.9 ms | 43.2 | 30.2 | - |
| after it | 71.1 | 40.9 | **26.2** | 28.6 |

🔴 **AND THE DENOMINATOR WAS GOING TO EAT THE GAIN, WHICH IS WHY 8 TURNS BACK
UP.** Each pair's count of covering sequences ran its own 64-lane barrier tree -
six barriers - so a group of eight ran 48 barriers to produce eight floats. The
counts reduce as a VECTOR now, one tree whatever P is. What is left at 8 is
residency: 8 pairs is 32 KiB of staged cells, one workgroup a core even on a
48 KiB device.

The constant was **2** because the sweep that fixed it was taken on an M2, whose
32 KiB of workgroup storage leaves two workgroups a core at P = 4. So it is
`opmProjectOutputPairs`, a knob, and Ampere takes 4 while the M2 keeps its own
answer. **Bit-exact in P** - a pair's accumulation order does not depend on how
many neighbours share its workgroup - and `fold-af2.js` returns checksum
-1805925 at 1, 2, 4 and 8, which is also what it returned before the change, so
the fused reduction is exact too.

Worth **29.57 -> 28.66 s** on the 825-residue fold.

🔴 **AND AF2 NOW HAS A `--tune=` FLAG AT ALL.** `fold-af2.js` carries the same
one `fold.js` has, and `profile-af2-block.js --sweep=knob=v1,v2,...` runs the
arms interleaved in one process with the weights loaded once - which is the only
honest way to sweep on a machine that drifts 3.2x. Every table in this section
came out of it. CLAUDE.md's rule is that a knob no gate enters is a knob nobody
has checked; the fold checksum is that gate.

### The contraction is NOT bandwidth-bound, and a tiled pair walk proves it

`opm.contract` is what is left: 67 ms of a 380 ms block, **713 GFLOP at 10.6
TFLOP/s against this card's 18.1**. The arithmetic said memory. A workgroup is
one pair and stages that pair's whole left and right operand - 512 sequences x
32 channels x 2, **128 KiB a workgroup**, 89 GB across the dispatch at 825
residues. Row-major is the worst order for it: for a fixed `i` the left row
stays in L2 across all 825 of its pairs, but `j` sweeps the whole right tensor,
54 MB against a 40 MB L2, so every row of `i` re-reads it.

So the pairs were walked in TILE x TILE squares instead. The intermediate is
indexed by the walk index and the output by the decoded `(i, j)`, so **any**
bijection serves as long as both kernels compute the same one - and `TILE = 1`
is exactly row-major, which makes it a knob with a free default. Padded square,
corner tiles guarded, bit-exact: `fold-af2.js` returns -1805925 at TILE = 1, 8,
16 and 32, and 59 is not a multiple of any of them so the padding is exercised.

🔴 **AND IT IS WORTH EXACTLY NOTHING.** Swept interleaved at 825 residues and
512 rows, as `opm.contract`:

| TILE | 1 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| contract | **66.97 ms** | 67.60 | 67.40 | 67.47 | 67.33 |
| a block | 380.13 | 381.34 | 381.15 | 381.17 | 380.95 |

Flat across a 64x range, and if anything the tiles lose. **The diagnosis was
wrong**: those reads were already cache-served, exactly as
src/af3/outer-product-mean-webgpu.js records for its own version of the same
question ("staging the rows was tried and lost... those reads were cache-served
anyway"). The kernel is issue-bound at 59% of the f32 ceiling, not starved of
bandwidth. The code was reverted; what is left to try on it is a
**two-dimensional (i, j) block** in the kernel itself, which is AF3's answer -
BLOCK_I + BLOCK_J staged rows buying BLOCK_I * BLOCK_J pairs, 48 KiB a pair
against this one's 128 - and that changes the instruction count rather than the
traffic.

🔴 **AND DO NOT PORT AF3's OPM KERNEL WHOLESALE.** It fuses the contraction and
the output projection with no intermediate at all, which looks like the obvious
next step and is not: its inner loop re-reads `left` and `right` per CELL as
well as per sequence, `2 x PRODUCTS x SEQUENCES` floats to consume
`2 x C_OUTER x SEQUENCES` distinct ones. That is free at AF3's 59-150 tokens,
where both operands are a few MB and stay in cache. At AF2's 825 tokens and 512
rows they are 54 MB each and it would read **1.08 TB** where the staged kernel
here reads 89 GB.

## The pair bias read the whole pair track once per head

`createAttentionPairBiasShader` gave a thread an `(i, j, head)` and had it sweep
all c_z channels of `pair[i][j]` for that one head - so the eight heads of an
MSA row attention each streamed the WHOLE pair track. At 825 residues that is
**2.79 GB a dispatch to consume 348 MB**, and it measured **8.72 ms of a 379 ms
block for 1.4 GFLOP**: 0.16 TFLOP/s, the least efficient kernel in the block by
two orders of magnitude, and invisible in a profile until you divide its time by
its arithmetic.

A thread owns a PAIR now and carries one accumulator per head, so the row is
read once and the weight read repeats instead - `heads` consecutive floats,
identical across the workgroup, which is a broadcast. The head count is a
generation parameter and joins the pipeline cache key at all three call sites,
because the shader no longer reads `p.heads` in its loop.

| at 825 residues, 512 rows | before | after |
|---|---:|---:|
| msa-row-attention.pair-bias | 8.72 ms | **0.85** |
| triangle-attention-starting.pair-bias | 3.35 | **0.75** |
| triangle-attention-ending.pair-bias | 3.31 | **0.75** |

**6.5x**, a block 378.9 -> 366.5, checksum unchanged.

## AF2's triangle was not asking the device for its projection tile

`trianglePairProjectTile` has been in the Ampere prior since the pairformer's
own sweep chose 32x32 over src/triangle/shaders.js's 32x16. AF2's evoformer and
multimer blocks call the same `createTriangleShaders` and passed the default, so
the pair track ran one tile in AF3 and another in AF2 **on the same device**.

    triangle.outgoing/incoming.project   7.21 -> 6.46 ms each
    triangle.outgoing/incoming.output    6.74 -> 4.88

a block 366.5 -> 360.8, checksum unchanged - it is occupancy and nothing else.
The tile joins the pipeline cache key, because a kernel compiled for one
geometry under a dispatch dividing by another is a fault CLAUDE.md records as
having happened here before.

## An f16 contraction that is safe at depth, and is still only 1.13x

docs/AF2.md calls alphafold2-webgpu's f16 warning "one negative result worth
more than the positives": accumulating a whole contraction in half precision is
the fastest arrangement and it overflows on a deep MSA, taking a 508-row
prediction from **96.80 pLDDT to 69.94** and its pTM to NaN. The reason is that
the sum runs over every sequence.

So `opm.contract` gets a precision arm that narrows the STAGED TILE and a
per-chunk accumulator and promotes into an f32 running total every `OPM_CHUNK`
sequences - 32 in f16 against 8 in f32, so the promotion is 6% of the loop
instead of 25%. The half-precision sum is then bounded by the chunk however deep
the alignment is, and `check-opm-paths.js` has an arm that says so:

| sequences | 32 | 128 | 512 |
|---|---:|---:|---:|
| relRMS against the f32 contraction | 8.89e-4 | 8.76e-4 | **8.29e-4** |

**Flat, not growing** - the opposite of the shape of the failure it guards
against - and below the 1.7e-3 to 2.4e-3 docs/A100.md prices this block's other
f16 kernels at, so it would not be the largest single contribution. The checker
also asserts every output is finite, because NaN is how the unsafe form
announces itself.

🔴 **AND IT IS STILL OFF BY DEFAULT, BECAUSE THE SPEED IS NOT THERE.** Swept
interleaved at 825 residues and 512 rows: `opm.contract` **67.30 -> 59.38 ms**,
a block 360.8 -> 355.4. **1.13x on the kernel, 1.5% on the block** - about
380 ms of a 27.85 s fold - against checksum -1805925 -> -1801474 and pLDDT
57.280 -> 57.233. That is the same trade the full-f16 flash kernel was refused
on, and it gets the same answer.

Why 1.13x and not the 2.0x this card gives f16: **the staging still reads `left`
and `right` as f32 from global**, 89 GB a dispatch, so halving the arithmetic
only moves the half of the kernel that was arithmetic. The follow-on is to have
`opm.project` WRITE them packed - two halves a word, `pack2x16float`, no device
feature - which needs its column ownership changed from "strided by lanesX" to
adjacent quads, exactly as `createLinearShader` already does under packing. That
is the one activation-packing this path can take: docs/AF3.md measured packing
the PAIR scratch and rejected it (a factor of 1200 on the representation that
feeds pLDDT and PAE, because `a` and `b` are multiplied against each other so
their rounding squares), and left/right have the same shape of risk already
priced at 8.3e-4 by the table above.

### Cell-chunking the output projection, and the runtime loop bound underneath it

`opm.project-output` is bound by a weight read that every pair in its workgroup
shares, so the traffic per pair is `512 KiB / P` and the whole game is raising
P. What caps P is that a workgroup stages every cell for every pair -
`CELLS * P * 4` bytes, 16 KiB at P = 4 - which is why the sweep turned back up
at 8. src/af3/outer-product-mean-webgpu.js chunks the same matrix for the same
reason (`OPM_CELL_CHUNK`, swept to 256 there), so AF2's was chunked to match:
stage `CELL_CHUNK` cells at a time, keep the accumulators across the chunk loop,
take P to 16.

🔴 **AND IT WAS 3x SLOWER, FOR A REASON THAT HAS NOTHING TO DO WITH CHUNKING.**
Written the obvious way - a loop over `cell0` whose inner bound is
`available = min(CHUNK, CELLS - cell0)` - the inner loop's trip count is a
RUNTIME value, and this kernel's inner loop is one workgroup read, one global
read and one multiply-add. A compiler that cannot see the count cannot unroll,
and in a three-instruction body the loop overhead is half the kernel:

| at 825 residues, P = 4 | opm.project-output |
|---|---:|
| the shipped kernel, constant bound | **26.2 ms** |
| chunked, runtime bound, chunk = 1024 (i.e. no chunking at all) | 113.0 |
| chunked, runtime bound, chunk = 512 | 97.5 |
| chunked, runtime bound, chunk = 256 | 79.3 |

The 1024 row is the *same arithmetic over the same cells in the same order* as
the shipped kernel. **A runtime bound cost 4.3x**, and chunking underneath it
was HELPING - which is the opposite of what the first sweep looked like it said.
It is in CLAUDE.md now because it is a WGSL trap and not an OPM one.

🔴 **AND WITH THE BOUND MADE CONSTANT - CHUNKS GENERATED, ONE BLOCK EACH, THE
COUNT TYPED IN - THE WHOLE THING IS WORTH ABOUT 1%.** Swept with the chunk
clamped to `maxComputeWorkgroupStorageSize` (16 pairs against every cell is
69,696 bytes and WebGPU refuses the pipeline outright, so the chunk halves until
it fits rather than the pair count falling back):

| P | 4 | 8 | 16 |
|---|---:|---:|---:|
| opm.project-output | **282.6 ms** | 23.8 | 26.5 |
| a block | 619.0 | **359.7** | 362.2 |

The best arm is 23.8 ms against the shipped 26.2, and its block is 359.7 against
360.8 - inside the run-to-run noise. And the P = 4 arm is **incoherent**: the
same configuration measured 79.3 ms one paragraph above with a *worse* loop
bound, so a constant bound cannot be 3.5x slower than a runtime one. The most
likely reading is that a fully-unrolled 256-iteration body blows the instruction
cache at some register counts and not others, which is a fragile thing to ship
a default on.

**Reverted.** 0.7% of a block for a generator that has to reason about workgroup
storage, unroll factors and a device limit is not a trade worth making, and the
one durable finding - the runtime loop bound - is worth more than the kernel was.

## The extra-MSA stack, where 22% of a long fold was hiding

`fold-af2.js` reports a stage breakdown now, taken from the progress stream -
each stage's unit size is its signature, so the deltas name it without the model
reporting stage names it does not have. At 825 residues, 512/1024, 0 recycles:

| stage | | | share |
|---|---:|---:|---:|
| extra-MSA stack | 4 blocks | 6.20 s | **22.4%** |
| main evoformer | 48 blocks | 17.28 | 62.3% |
| structure module | | 1.08 | 3.9% |
| confidence heads | | 1.19 | 4.3% |

🔴 **1549 ms AN EXTRA BLOCK AGAINST THE MAIN STACK'S 360, FOR A STACK WITH A
QUARTER THE MSA CHANNELS.** Its pair half is the same code, so on any accounting
it should be CHEAPER, and nothing here could see into it: `ExtraMsaStackGpu` had
no `profileBlock`. It has one now - `profile-af2-block.js --stack=extra`.

### Two kernels, one fault, and it is the pair bias's fault again

| at 825 residues and 1024 rows | before | after |
|---|---:|---:|
| `extra.msa-column-global-attention.output` | **289.93 ms** (42.5% of the block) | under 4.9 |
| `extra.msa-column-global-attention.query` | 34.66 | under 4.9 |
| an extra-MSA block | 681.61 | **361.54** |

**`.output` recomputed the GATE once per output channel.** A thread owned one
`(row, c_out)` and computed
`gate = bias + sum_c normalized[row][c] * W[c][head][d]` inside its own loop -
but the gate does not depend on `c_out`, so all `channels` threads of a row
computed the same `heads * head_dim` gates, each of them a `channels`-long dot
product. **450 GFLOP issued to compute 13.8**, and it was the largest single
kernel in the whole fold.

**`.query` did the same thing one pass earlier**, with the column's masked mean
over every sequence: it depends on `(column, c)` alone and was recomputed for
each of `heads * head_dim`. 3.46 G multiply-accumulates for 57 M of work, **60x**.

Both give a WORKGROUP the row or the column now, reduce once into workgroup
memory, and take one output per lane. Bit-exact - the only reordering is a
denominator that is a barrier tree over a mask of ones and zeros, integer-valued
in f32, and the `1e-10` beside it is below the ULP of any count it joins.

🔴 **THREE KERNELS IN THIS PORT HAVE NOW HAD THE SAME BUG**, and it is worth
naming as a shape rather than three accidents: *an invocation owns a fast index
that its inner reduction does not depend on, so the reduction is recomputed once
per value of that index.* The pair bias re-read the whole pair track per head;
these two re-reduced a row and a column per head. Each was two to three orders
of magnitude off its ceiling and each was invisible to a profile sorted by name.
**The test is arithmetic, not the profile: divide a kernel's time by its USEFUL
flops, and anything under a few percent of the device is this.**

🔴 **AND THE MULTIMER'S COPIES ARE GONE RATHER THAN FIXED TWICE.** Both kernels
were verbatim in `src/multimer/block.js` with the same fault. This pair had
already been copied once, so the multimer imports the generators now.

### And 2.5 s of a one-pass fold is one-time cost, paid in whichever stack runs first

The stage breakdown at 825 with **one recycle** - two passes - separates it:

| stage | pass 1 | pass 2 |
|---|---:|---:|
| extra-MSA stack, 4 blocks | **5.40 s** | **2.94** |
| main evoformer, 48 blocks | 17.29 | 16.72 |
| structure module | 1.08 | 0.92 |
| confidence heads | 1.20 | 1.16 |

🔴 **THE EXTRA STACK IS 1.84x ON ITS FIRST PASS AND THE MAIN ONE IS 1.03x**, so
this is not "the first pass is warmer": it is a fixed cost that lands wherever
the first stack is, and the extra stack is 4 blocks so it carries the whole of
it against 48. **2.46 s**, which at 0 recycles is 9% of the fold and at
AlphaFold's default 3 recycles is 2%.

The steady-state extra block is 735 ms against the profiler's 361 ms of GPU, so
about half of even the warm number is host - `encodeExtraMsaBlock` packs and
uploads its weights per block per pass, which src/runtime/execution.js's own note
already prices at "221 ms of packing paid once instead of four" for the main
stack.

🔴 **AND IT IS NOT SHADER COMPILATION, WHICH IS THE FIRST GUESS AND IS WRONG.**
The same measurement at 400 residues puts the extra stack at 1.87 s on pass 1
and 1.17 on pass 2 - **0.70 s** of fixed cost against 825's **2.46**. Compilation
would be flat in length; this is 3.5x for a 2.06x length, so it goes as **L^2**,
which is the size of the pair-track buffers. Large WebGPU allocations on this
driver are the candidate - twenty pair-sized buffers at a hundred milliseconds
each is the right order - and the pool only warms once. Not attacked here;
written down with the number and the scaling so the next attempt does not spend
its first hour on the pipeline cache.

## The confidence heads took the same softmax twice

The stage breakdown put 1.19 s of a 26.4 s fold in the confidence heads, which
compute two per-residue numbers and two per-pair ones. The arithmetic is
`L^2 * bins` and it is all on the main thread:

| at 825 residues, 64 bins | |
|---|---:|
| `softmaxExpected` for `predictedAlignedError` | 726 ms |
| `tmTermFromLogits` for pTM | 729 ms |

**Both are the same softmax.** `predictedAlignedError` is the expectation of the
bin CENTRES; the pTM term is the expectation of `1 / (1 + centre^2 / d0^2)` -
the same probabilities over the same logits, weighted differently - and each
pass called `Math.exp` on all 43.6 million logits.

`softmaxExpectations` takes several weight vectors and one pass, and every score
is bit-identical: same maximum, same denominator, same order per output, term
matching the old one to **max abs diff 0**.

🔴 **AND THE ACCUMULATORS HAVE TO BE LOCALS, WHICH IS THE DIFFERENCE BETWEEN 19%
AND 100%.** Written generally - `outputs[set][row] +=` inside the bin loop - two
expectations cost **1224 ms against 994 for one**, when the second should be
nearly free: the exp() calls are shared and only a multiply-add is not.
Typed-array element accumulation ate it. With scalar accumulators and the one-
and two-set cases written out: **790 ms for two against 794 for one.**

    the confidence heads   1.19 s -> 0.51
    the fold              26.42   -> 25.79

The remaining 790 ms could go to the GPU - the logits are already there, and it
would drop a 174 MB readback with them - and is NOT taken here, because the term
is a `Float64Array` on the CPU and an f32 kernel would move pTM's last digits.
3% of a fold is not worth moving a published number.

### And the output matrix in f16 does not pay, which says what binds it

`opm.project-output` reads the whole `CELLS x c_z` output matrix once per
workgroup - 87 GB a dispatch at 825 residues with four pairs a group, which at
26.2 ms is 3.3 TB/s and looks exactly like this card's L2. So the matrix was
packed two halves to a word, its own `Uint32Array` binding (a Float32Array is
not required to carry arbitrary bit patterns, and packed halves make plenty of
NaNs).

| at 825 residues | opm.project-output |
|---|---:|
| f32 weights | **26.2 ms** |
| f16, `unpack2x16float(w)[at & 1u]` | 103.2 |
| f16, selector hoisted out of the loop | 27.8 |

🔴 **STILL SLOWER, SO THE BYTES ARE NOT WHAT BINDS IT.** 3.3 TB/s is being
served out of cache at a high hit rate rather than streamed, and the unpack
instructions cost more than the halved traffic saves. Reverted. This is the
third bandwidth diagnosis in this file to be wrong about an OPM kernel - after
the tiled pair walk and the cell chunk - and the pattern is worth naming: **a
kernel reading the same small matrix from every workgroup is an L2 HIT rate, not
an L2 bandwidth**, and its time is instructions.

The 4x in the middle row is a WGSL trap and is in CLAUDE.md: a dynamic vector
index puts the vector in addressable memory instead of a register.

## The two GEMMs in the outer product mean, onto the matrix units

Both of the OPM's big kernels are plain matmuls, and neither was one.

**The contraction.** `outer[i][j][cl][cr] = sum_s left[s][i][cl] * right[s][j][cr]`
is `C[(i,cl)][(j,cr)] = sum_s A[s][(i,cl)] * B[s][(j,cr)]` - at 825 residues and
512 sequences, **26,400 x 26,400 x 512 and 713 GFLOP**, run at 10.6 TFLOP/s.
docs/A100.md's rule for when the units pay is "the operand is materialised, the
problem is past a billion multiply-accumulates, and K is deep"; this has the
deepest K in the model.

**The output projection.** `output[pair][z] = (bias + sum_cell outer[pair][cell]
* W[cell][z]) / count[pair]` - rows the block's pairs, inner `c_outer^2`, columns
`c_z`. Unlike the contraction it needs no transpose at all.

| at 825 residues, 512 rows | before | after |
|---|---:|---:|
| `opm.contract` | 67.30 ms | **29.85** (2.25x, 23.9 TFLOP/s) |
| `opm.project-output` | 26.15 | **12.43** (2.1x) |
| `opm.scale` (new) | - | 0.58 |
| a block | 360.99 | **310.58** |
| a fold | 23.67 s | **20.92** |

🔴 **AND NEITHER NEEDED A LAYOUT MIGRATION, WHICH IS THE ONLY REASON THEY WERE
CHEAP.** The shared `createStagedMatrixShader` gained four options a dense
projection never wants: `outputIndex`, so the contraction's store writes by PAIR
and `opm.project-output` and the pair blocking are untouched; `bias: false`,
because a contraction has none; `sourceTransposed`, which neither uses because
`opm.project` writes `left` channel-major instead - one index in a kernel that
already runs, against another sweep of 54 MB; and `rowScaleOffset`, below.

🔴 **VECTOR STAGING IS MOST OF THE CONTRACTION'S WIN AND ALMOST ENDED IT.**
Without it the same kernel is 64.5 ms - a **1.04x**, which reads as "the matrix
units do not help here". docs/A100.md prices the vec4 operand read at 1.24 ->
0.70 ms on the transition; here it is 2.16x.

🔴 **AND A SCALE APPLIED AT THE RESULT INSTEAD OF THE OPERAND COST A WHOLE FOLD.**
Written the obvious way - `(bias + sum) * s`, the arithmetic the vector kernel
does - the staged copy of `outer` is a sum over the WHOLE ALIGNMENT before its
divide, and at 1024 sequences it leaves f16's range: every coordinate came back
NaN. At **512** sequences it did not, and at **200** residues it did not, and
that shape of threshold is always this. Scaled at the STAGING the operand is
O(1) whatever the depth, and the bias is scaled with it so the result is
identical. The two are `rowScaleOffset` and `scaleIndex`, and the shared kernel
refuses both at once.

The denominator becomes its own pass, which is the move
src/af3/outer-product-mean-webgpu.js already made for the same reason: the work
is `pairs x sequences` either way, and a GEMM has nowhere to put a cooperative
reduction. `opmMatrixContract` and `opmMatrixOutput` are separate knobs, which
is what bisected the NaN in three runs.

### And the triangle contraction cannot follow, because K is the protein's length

`z[c][i][j] = sum_k a[c][i][k] * b[c][j][k]` is a batch of `c_hidden`
independent `L x L x L` GEMMs, and the layout is almost free: `a`, `b` and `z`
are all channel-major with the same stride, `a` is already the `[i][k]` source
the kernel reads, and only `b` has to swap its `(j, k)` - one index in the
projection that writes it. Built, gated, and **slower**:

| at 825 residues | vector | matrix |
|---|---:|---:|
| triangle.outgoing.contract | 13.25 ms | 13.89 |
| triangle.incoming.contract | 12.67 | 13.97 |
| triangle.outgoing.project | 6.46 | 7.47 (the transposed store) |
| a block | 310.66 | 314.95 |

🔴 **BECAUSE 825 IS NOT A MULTIPLE OF FOUR.** The matrix path's win here is
almost entirely `vectorStaging` - the outer product mean measured 1.04x without
it and 2.25x with - and a vec4 operand read needs the INNER extent to divide by
four so a row's staging does not straddle the next. The OPM's inner extent is
the alignment depth, 512; the triangle's is the protein's length, which is
whatever the protein is. **So the rule docs/A100.md arrived at gains a fourth
condition: materialised, past a billion multiply-accumulates, K deep, and K
divisible by four.** Reverted.

The cache earned its keep on the way: `project-ab` differs by the transposed
store and the pipeline key did not say so, and
`ComputePipelineCache` refused the collision instead of handing back the wrong
shader.

## Two more phases that were pure overhead

🔴 **781 MB OF CONTINUATION STATE THAT MOST FOLDS NEVER READ.** The trunk's MSA
and pair representation were copied to the host so `web/app.js` can continue a
fold at more recycles - one cache, one caller - and every other caller paid for
it. `resumable: true` now asks for it. **1.26 s -> 0.**

🔴 **AND 348 MB OF JAVASCRIPT ZEROS UPLOADED AS THE FIRST PASS'S RECYCLE STATE.**
`new Float32Array(length * length * 128)` is 87 million elements at 825 residues,
built on the host and pushed across the bus, when WebGPU zero-initialises a new
buffer anyway. **The embedder phase 1.57 s -> 0.90.**

Both were invisible until `fold-af2.js` grew a `phases` report - the progress
stream can only say "a block finished", and a quarter of a long fold is not in a
block.

## Why the last 38% of the block CAN follow, and how the first answer was wrong

After the two OPM GEMMs the block is 310 ms and the four flash attentions are
116.9 of it - **38%, and the only thing left big enough to close the gap to
alphafold2-webgpu**. They run at 9.8 TFLOP/s, 54% of this card's f32 shader
ceiling, with a softmax in the loop. The obvious next move is a tensor-core
flash attention.

🔴 **AND THE UNITS ARE WORSE THAN THE VECTOR KERNEL AT THIS HEAD DIMENSION.** A
flash attention's inner extent is `head_dim`, which AF2's every attention sets
to **32**. `bench-evoformer-linear.js --shape=headdim` puts the best generic
GEMM in the tree against the staged matrix kernel at exactly that K, at the M
and N a query tile against a key chunk has:

| K = 32, M = 52,736, N = 824 | | |
|---|---:|---:|
| `8x8@f16/f16`, the vector kernel | 0.300 ms | **9270 GFLOP/s** |
| staged matrix, 64x64x32, 1x4 | 0.425 | 6544 |
| staged matrix, 128x128x16, 1x8 | 0.438 | 6357 |
| staged matrix, 128x128x32, 2x4 | 0.563 | 4944 |
| staged matrix, 128x128x32, 1x8 | 0.575 | 4837 |

**1.4x to 1.9x slower**, and the shipped flash kernels already reach 9.8 TFLOP/s
- which is what the vector kernel does at this shape. They are at their ceiling.

That is the third point on docs/A100.md's curve: **28.2 TFLOP/s at K = 256,
14.9 at K = 128, 9.3 at K = 32.**

🔴 **AND THE CONCLUSION DRAWN FROM IT WAS WRONG, WHICH IS THE POINT OF THIS
SECTION NOW.** It read: "a head dimension is the opposite of a deep inner
extent, so the remaining 38% is a shape the hardware's fast path does not
serve". alphafold2-webgpu ships a subgroup-matrix flash attention, its runtime
calibration probes it against the register kernel on every device that has the
units, and its own source records the matrix kernel winning by **1.66x to
1.84x** across five workgroup-storage arrangements.

**The bench above answers a different question.** A standalone GEMM at K = 32
amortises its staged panel over two k steps and nothing else. A flash attention
stages the QUERY tile once and reuses it across every key in the sequence, and
stages each key tile once for sixteen queries instead of once per query - so the
reuse that pays for the staging is in the loop structure, not in K. What the
units buy is that the query-key reduction happens in hardware, where the
register kernel does it per lane and every subgroup variant pays cross-lane
traffic for it.

Their file warns about exactly this mistake, having made it once: *"The earlier
note here said the opposite, on a microbenchmark whose key and value tensors
were small enough to sit in L2. The model's are not... Measure this against the
stack, not against a standalone dispatch."* CLAUDE.md's own rule is the same one
- profile in situ, do not infer from an isolated kernel - and it was not
followed here. The `headdim` bench arm is kept because the number is real; what
is deleted is the inference.

### And the kernel: 1.69x, and none of it came from the units

`src/evoformer/attention-matrix.js` is that kernel, written from the published
algorithm rather than from their source, which carries no licence. It reached
**69.3 ms against the register kernels' 116.9** across an 825-residue block's
four attentions - inside the 1.66x-1.84x they report - and every step of getting
there was about something other than the matrix multiplies.

The first version was **1.26x SLOWER**, at 147.0 ms. The obvious repair, a
bigger key tile to amortise the barriers and the per-tile rescale of the running
output, made it far worse:

| tile | workgroup bytes a lane | block ms |
|---|---:|---:|
| 4x32 | 242 | 324.6 |
| 2x32 | 276 | 341.0 |
| 4x64 | ~370 | 411.0 |
| 2x64 | 438 | 441.1 |
| 2x128 | 762 | 568.8 |

🔴 **THAT RANKS EXACTLY BY WORKGROUP BYTES A LANE AND NOT AT ALL BY ARITHMETIC.**
The kernel is latency-bound and its occupancy is capped by shared memory, so the
tile that amortises the most fixed cost is the one that fits fewest workgroups
on an SM. **Every fixed cost this kernel was built to amortise is worth less
than the bytes it takes to amortise it.** Three arrays left workgroup memory:

- **the running output** - ROWS x HEAD floats, 64 bytes a lane, and the one
  array that is never a matrix operand. A lane holds `HEAD / 8` vec4s of it.
- **the row statistics** - two lanes share a query row and the pair is
  ADJACENT, so `subgroupShuffleXor(m, 1u)` combines their halves and the running
  max, sum and rescale need no array and no barrier. That needed the accumulator
  split the other way round: the two lanes of one row take half the CHANNELS
  each, rather than each lane taking a slice of some row, so a lane only ever
  asks for the row it owns. The epilogue then writes straight to global.
- **the corrected logits** - the two softmax passes were handing them to each
  other through the score array, and they never leave the lane that made them.

Then three hoists out of the per-key work, none of them touching a multiply:

- **the pair bias index came out of the key loop.** It was
  `(head * queries + q) * queries + k` - two integer multiplies, per key, per
  lane, for a base that is the same for every key that lane will ever read.
- **the bounds check came out of every tile but the last.** The body is
  generated twice; the checked copy runs at most once.
- **the mask is staged.** It is per KEY and was being read per query ROW: 64
  rows reading the same global float and computing the same `1e9 * (m - 1)`.

| tile | block ms | four attentions |
|---|---:|---:|
| **4x32** | **263.5** | **69.3** |
| 3x32 | 269.3 | 75.0 |
| 5x32 | 269.5 | 75.0 |
| 2x32 | 270.4 | 75.9 |
| 6x16 | 278.9 | 85.8 |
| 4x48 | 290.3 | 95.7 |
| 4x64 | 297.3 | 103.0 |

🔴 **AND THE OPTIMUM MOVED, WHICH IS THE PART TO REMEMBER.** Before the hoists
the best tile was 6x16 and a key tile of 32 was 5% worse; after them 4x32 is
best and 6x16 is 6% worse. The scalar work per key and the workgroup bytes per
lane trade against each other, so **neither the tile nor its ranking survives a
change to the other**, and a geometry adopted from another port - or chosen
before the kernel was finished - would have been the wrong one. It is a knob
(`attentionMatrixTile`, written "subgroupsXkeys" so `--tune` can carry it) and
it is in the pipeline cache key, because it sets the dispatch tile too.

Measured and NOT pursued: three of the five barriers are between a subgroup and
its OWN rows, so a subgroup-scoped barrier would do - and there is none on this
Dawn ("unresolved call target 'subgroupBarrier'"). Omitting them outright, which
is a data race and shipped nowhere, is worth **1% to 4%** of a block. Nobody
needs to price that twice.

Also learned the hard way, twice: `subgroupMatrixStore` needs a subgroup-uniform
offset, and an index computed as `local_invocation_id.x / 32` is not one however
uniform it is in fact - `@builtin(subgroup_id)` is. And `from` is a reserved
word in WGSL.

## Where AF2 stands now

🔴 **BOTH PORTS RE-MEASURED AT `94d3902`, INTERLEAVED IN ONE BROWSER - see "Both
ports, re-measured" at the end of this file. 15.74 -> 11.02 s here and
14.21 -> 13.15 there. The table below is the older measurement.**

One trunk pass, 825 residues, 512 clusters / 1024 extra, 0 recycles, same A100:

| | one pass | vs JAX | vs the other port |
|---|---:|---:|---:|
| native AF2 (JAX, bf16 tensor cores) | 5.27 s | 1.0x | - |
| alphafold2-webgpu (warm) | 14.21 | 2.7x | 1.0x |
| LocalFold, before this work | 149.66 | 28x | 10.5x |
| LocalFold, before the matrix attention | 18.06 | 3.4x | 1.27x |
| **LocalFold now** | **15.74** | **3.0x** | **1.11x** |

🔴 **AND THAT LAST COLUMN COMPARED A COLD NUMBER WITH A WARM ONE FOR THIS FILE'S
WHOLE HISTORY.** `fold-af2.js` reports ONE pass with every fixed cost in it;
their harness reports the minimum of two warm passes after a cold one, and
their own output has always said what that is worth - 16.39 cold against 14.21
warm. The two ports at the same temperature are in the ladder below, and at 825
residues they are **level**, not 1.11x apart.

and by length, all 0 recycles, with the last change on and off:

| | matrix off | matrix on | peak device memory |
|---|---:|---:|---:|
| 59 residues, 128/128 | 1.129 s | 1.123 | 108.5 MiB |
| 400 residues, 512/1024 | 5.544 | **5.129** | 2225.6 |
| 825 residues, 512/1024 | 18.06 | **15.74** | 6331.6 |

The 825 figure was **149.66 s** and **11,340 MiB** at the start of this work:
**9.5x faster in 1.8x the memory headroom**, and peak memory is unmoved at
6331.6 MiB by the last of it.

## The two ports at the same temperature, by length

`tools/gpu/bench-af2-warm.js` runs three passes and reports cold and warm the
way their `bench825.js` does, times `predict` alone with the features built
outside the clock as theirs does, and builds the character-for-character
identical synthetic alignment. Both on this A100, one trunk pass, no recycles:

🔴 **AND `--passes` MEANS THE OPPOSITE THING IN THE TWO HARNESSES, WHICH IS A
SILENT 3x AGAINST THEM.** Here it is the number of TIMED REPETITIONS. In
`bench/bench825.js` the repetition count is hardcoded at three and `--passes` is
read as `recycles: passes - 1` - trunk passes INSIDE one prediction. So
`--passes=3` on both sides times three repeats here and three RECYCLES there,
and their 59-residue figure goes 1.24 s to 3.65 while this one does not move.
It cost an hour of looking for a machine fault that was not there: the clocks
were locked, nothing else held the GPU, and the number was still 3x its
recorded value. Run theirs with no `--passes` at all.

The full grid, both ports at three repetitions and zero recycles:

| | ours warm | theirs warm | | ours cold | theirs cold | |
|---|---:|---:|---:|---:|---:|---:|
| **128 clustered / 256 extra** | | | | | | |
| 59 residues | **0.631** | 1.24 | **1.97x** | **1.065** | 2.42 | **2.27x** |
| 128 | **0.661** | 1.22 | **1.85x** | **1.158** | 3.54 | **3.06x** |
| 256 | **1.087** | 1.31 | **1.21x** | **1.591** | 5.30 | **3.33x** |
| 400 | 2.389 | 2.38 | 1.00x | **2.813** | 3.70 | **1.32x** |
| 600 | 5.016 | 5.07 | 1.01x | **5.475** | 9.17 | **1.67x** |
| 825 | 9.556 | **9.20** | 0.96x | 10.083 | 10.14 | 1.01x |
| **512 clustered / 1024 extra** | | | | | | |
| 59 residues | **0.662** | 1.25 | **1.89x** | **1.088** | 2.17 | **1.99x** |
| 128 | **0.992** | 1.26 | **1.27x** | **1.398** | 2.21 | **1.58x** |
| 256 | 2.278 | 2.26 | 0.99x | **2.688** | 3.61 | **1.34x** |
| 400 | 4.435 | 4.40 | 0.99x | **4.829** | 6.49 | **1.34x** |
| 600 | 8.399 | 8.38 | 1.00x | **8.831** | 9.62 | 1.09x |
| 825 | 14.688 | 14.71 | 1.00x | **15.155** | 15.96 | 1.05x |

🔴 **THE ADVANTAGE IS AT SHORT LENGTHS AND IN COLD START, AND IT IS NOT SMALL.**
Warm, this port is **1.85x-1.97x** to 128 residues, 1.21x-1.27x at 256, and
level from 400 up - at 825 with a full alignment the two are 14.688 against
14.71, inside a single run's noise. The one shape where they lead is **825 with
a SHALLOW alignment**, 9.20 against 9.556, and that is the one place their
whole-tensor f16 packing has the most to work with and the fewest MSA rows to
lose accuracy on. Cold, this port is ahead at every length and depth measured,
from 1.01x to 3.33x.

🔴 **AND THE TWO PORTS PRICE THE ALIGNMENT DIFFERENTLY.** What the deeper
alignment costs, as the ratio of 512/1024 to 128/256 at the same length:

| | 59 | 128 | 256 | 400 | 600 | 825 |
|---|---:|---:|---:|---:|---:|---:|
| ours | 1.05 | **1.50** | **2.10** | 1.86 | 1.67 | 1.54 |
| theirs | 1.01 | **1.03** | 1.73 | 1.85 | 1.65 | 1.60 |

From 256 up the two agree to within 8%, which says the extra-MSA stack costs
both of them the same. Below that they do not: at 128 residues four times the
rows costs this port 1.50x and theirs 1.03x. Their fold at that size is still
dominated by something that does not scale with depth, and ours is not - which
is the same fixed cost the cold column shows, seen from the other side.

**And the length scaling is nowhere near the exponent the kernels have.** Ours
at 512/1024, normalised to 59 residues: 1.5x at 128, 3.4x at 256, 6.7x at 400,
12.7x at 600, 22.2x at 825 - against 195x for L^2 and 2734x for L^3. A trunk
pass is quadratic in the pair track and cubic in the triangle, and at these
lengths it is still mostly neither: the fixed cost per pass dominates to about
256 residues and the growth only approaches L^2 past 600.

🔴 **AND THE REASON IS A FIXED COST THEY PAY AND THIS DOES NOT.** Cold minus
warm, across all twelve shapes:

| | smallest | largest |
|---|---:|---:|
| LocalFold | 0.394 s | 0.527 |
| alphafold2-webgpu | 0.920 | 4.100 |

Theirs is one to four seconds and GROWS with the shape; ours is 0.394 to 0.527
and is flat in both length and depth - a 22x range of work either side of it. That is what a `planMonomerDevice` that sizes limits from the
shape, a `fitScratchBudgetScale` search, and a wider set of specialised
pipelines cost on the first pass - the same machinery docs/AF2.md credits for
their warm 825 number, priced on the other side. **For a user folding one
sequence, cold is the number they experience**, and short single folds are the
common case in a browser.

## 🔴 THE 825-RESIDUE FOLD WAS A COLLAPSED BALL, AND pLDDT SAID 69.31

The row above asked which port was right where their pLDDT is 25.69 and this
one's is 69.31. It is this one that is wrong, and not marginally. Consecutive
CA-CA must be **3.80 A**:

| this port, 512/1024 | pLDDT | pTM | CA-CA median | CA-CA worst | first to last CA |
|---|---:|---:|---:|---:|---|
| 200 residues | 33.03 | 0.2312 | 3.869 | 3.43 | far apart |
| 400 | 37.02 | 0.4232 | 3.658 | 11.60 | far apart |
| 600 | 61.35 | 0.9022 | **1.662** | 9.22 | ~3 A |
| 700 | 65.61 | 0.9374 | **1.443** | 0.06 | ~3 A |
| 825 | 69.31 | 0.9672 | **1.559** | **0.06** | **2.2 A** |

At 825 residues the whole chain is inside a ball a couple of angstroms across,
with consecutive alpha carbons 0.06 A apart, and **pLDDT RISES to 69.31 and pTM
to 0.9672 as it collapses**. This is docs/AF3.md's "17 A of spaghetti at pLDDT
55" and docs/OPENDDE.md's "0.27 A cloud" for the third time, and it is why
CLAUDE.md says pLDDT is not a correctness gate.

Their structure on the identical input is a real chain:

| L=600, 8 clustered / 1024 extra | pLDDT | CA-CA median | CA-CA worst | first to last CA |
|---|---:|---:|---:|---|
| LocalFold | 60.96 | **1.741** | 7.92 | ~3 A |
| alphafold2-webgpu | 31.36 | **3.939** | 4.55 | ~70 A |

and so is theirs at 825: pLDDT 25.68, CA-CA median 3.534, worst 2.45, ends 60 A
apart. **Their 25.69 is a genuine low-confidence prediction of a 14x tandem
repeat. This port's 69.31 is a confidently mislabelled blob.**

### It is the EXTRA-MSA stack, and it is the product of length and depth

The main stack is innocent at every depth. At 600 residues, varying only the
clustered rows with the extra alignment held at 8:

| clustered rows | 8 | 16 | 32 | 64 | 128 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| CA-CA median | 3.940 | 3.936 | 3.926 | 3.918 | 3.908 | 3.826 |

all healthy. Eight clustered rows with **1024 extra** collapses on its own
(1.741). The two axes that matter are the EXTRA depth and the length, and they
trade against each other rather than crossing a cliff:

| L=600, 8 clustered | extra 64 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|
| CA-CA median | 3.940 | 3.935 | 3.948 | 3.926 | **1.741** |
| CA-CA worst | 4.36 | 4.25 | 4.28 | **14.20** | 7.92 |

| extra 1024, 8 clustered | L=200 | 300 | 400 | 500 | 600 |
|---|---:|---:|---:|---:|---:|
| CA-CA median | 3.856 | 3.889 | 3.859 | **3.411** | **1.741** |
| CA-CA worst | 3.26 | 5.95 | 7.73 | **10.66** | 7.92 |

Degradation starts at about 500 x 1024 and 600 x 512 and is total by 600 x
1024. A smooth ramp in the PRODUCT, not a step, which points at numerics rather
than at a dispatch that stops covering its rows - and **`--tune=halfPrecision=false`
collapses identically**, so it is not the f16 storage path.

The high pTM is the clue worth following. pTM comes from the PAE head reading
the PAIR representation, and 0.9672 means that head believes every pair is
close - so the pair track is saturating toward "everything in contact" and the
structure module is doing as it is told. The extra stack writes into the pair
track through its outer product mean, whose contraction depth IS the extra
alignment.

### It is not new, and no optimisation in this file caused it

Tested at the earliest commit where the tool actually synthesises `rows +
extraRows` (`3458551`) and at three later points including the global column
attention rewrite: **all collapse, to the same three digits**. `4bb6fe9` and
its parent are bit-identical here.

| commit | | CA-CA median |
|---|---|---:|
| `3458551` fold-af2: synthesise rows + extraRows | | 1.737 |
| `4348aef` the triangle normalize dispatch | | 1.735 |
| `8e85950` the OPM fast path above 128 residues | | 1.749 |
| `6129cbb` pair-block the outer product mean | | 1.749 |
| `4bb6fe9` the global column attention | | 1.749 |
| `4bb6fe9^` | | 1.749 |
| HEAD | | 1.741 |

It predates the whole campaign. What `3458551` did was make the extra alignment
as deep as it says - before it, a `max(512, 1024)` request got 512 extra - which
moved long folds from the spaghetti this file recorded earlier (708 residues,
CA-CA worst 19.46; 825, worst 70.45, medians still ~3.7) into outright collapse.
Same bug, twice the depth.

### What this does and does not invalidate

- **The timing work stands.** Both arms of every A/B run the same shapes, the
  same dispatch counts and the same kernels; the block profiler times
  dispatches. The 1.69x on the flash attentions and the 18.06 -> 15.74 s are
  measurements of throughput and they are unaffected by the values being wrong.
- **The claim that this port folds an 825-residue protein does not.** It
  produces a structure-shaped output that is not a structure.
- **The precision ladder above is weaker than it was presented.** 69.360 f32 /
  69.376 register / 69.306 matrix still shows the matrix kernel does not change
  the answer, which is what it was for - but all three are the same collapsed
  ball, so it is evidence about the KERNEL and not about the fold.
- The six per-kernel differential gates all pass and are all at short shapes.
  **Nothing in the suite reaches this**, which is the finding underneath the
  finding: `fold-af2.js` reports `caca` and no gate asserts on it.

### The bug: one missing `id.y` in the global column attention

`GLOBAL_ATTENTION_KV_SHADER` projects the extra alignment's keys and values. Its
dispatch is

```js
let grid = execution.linearGrid(shape.length * shape.sequences * headDim);
execution.dispatch(encoder, kvPipeline, [...], grid[0], grid[1], 1, `${label}.kv`);
```

and `linearGrid` **folds into y past 32768 workgroups**. The shader read

```wgsl
let index = id.x;                    // and nothing else
```

so every workgroup with a nonzero y recomputed the FIRST `32768 * 64` elements
instead of its own. Past **2,097,152 elements the keys and values were never
written at all**, and the attention read whatever the recycled scratch held.
`head_dim` is 8 in the extra stack, so the cap is `length * sequences =
262,144` - reached by a 1024-row extra alignment at 256 residues.

That is why the main stack is innocent at any depth: it runs the ordinary column
attention, and this kernel is the one only the extra stack uses. It is also why
`--tune=halfPrecision=false` changed nothing - an unwritten buffer has no
precision - and why the damage grows smoothly with `length * sequences`: the
fraction of the alignment left unprojected is `1 - 2097152 / (L * S * 8)`.

The fix is the `y` term. One line, and it costs nothing - the dispatch was
already issuing those workgroups, they were just writing the wrong place.

| 825 residues, 512/1024 | before | after | alphafold2-webgpu |
|---|---:|---:|---:|
| pLDDT | 69.31 | **25.62** | 25.68 |
| pTM | 0.9672 | **0.1830** | - |
| CA-CA median | 1.559 | **3.567** | 3.534 |
| CA-CA worst | 0.06 | **2.38** | 2.45 |
| one pass | 15.72 s | 15.71 | 14.69 warm |

**Two independent ports now agree to 0.06 pLDDT on the same input**, which is
the strongest statement either of them can make about being right.

🔴 **AND THE MULTIMER HAD ITS OWN COPY.** `src/multimer/block.js` builds the same
kernel from the same source with the same `linearGrid` dispatch, and carried the
same missing term. Fixed identically - but **untested on this box**, which has
only the monomer bundle: `--family=multimer` cannot load its weights here.

### The gates that were missing, and now are not

- **`fold-af2.js` asserts on the geometry.** It has printed `caca` since it was
  written and nothing ever read it, so a collapsed ball passed as "the same
  fold" for a whole campaign. It now fails unless the consecutive-CA median is
  3.4 to 4.2 A and the worst outlier within 2.8 A of 3.80 - wide enough for a
  bad PREDICTION and far too narrow for a broken one. Measured healthy: median
  3.485 to 3.972, worst 1.69 to 4.55. Measured broken: median 1.44 to 3.41,
  worst 0.06, or 7.73 to 70.45. `--allow-broken-geometry` reports anyway.
  The gate was falsified by putting the bug back: it fails with *"the fold is
  not a chain: consecutive CA median 1.559 A, worst 0.06 A"*.
- **Every shader was audited for the same defect.** A `fn main` that indexes
  from `.x` and never reads `.y`, in a file whose dispatches fold, is the whole
  signature. Two files matched and both were this bug. The audit was itself
  falsified against the pre-fix source first - the first version of it matched
  nothing anywhere, because `fn main\(([^)]*)\)` stops at the `)` inside
  `@builtin(global_invocation_id)`, and a checker that finds nothing is a
  checker that has not run.
- The remaining `.x`-only shaders are all in files with **no folding caller**,
  and `execution.dispatch` throws above 65535 workgroups, so they cannot
  truncate silently. The one exception, the global attention's query kernel,
  dispatches `x = length` and is bounded by that throw.

### 🔴 THE SAME FOLD, A SECOND BUG, AND THE AUDIT ABOVE ASKED THE WRONG QUESTION

Nine months later a nondeterministic AF2 trunk was hunted on an M2 - a different
structure every pass at 160, 200 and 400 residues, deterministic at 128 and
below, and not reproducible on an A100 at any length. It was pursued as a latent
missing `workgroupBarrier` for a week. It was a second folded-grid bug in
`ADD_IN_PLACE_SHADER`, and **the audit above would never have found it, because
that audit asked whether a shader READS `id.y`. This one does.**

```wgsl
let index = id.x + id.y * GRID_WIDTH * 64u;   // correct
base[index] += update[index];                  // and no bounds check
```

`linearGrid` rounds twice - elements up to a whole workgroup, then workgroups up
to a whole row of GRID_WIDTH - so past the fold the dispatch is a MULTIPLE of
32,768 workgroups and almost never the count wanted. The pair tensor is
`L * L * 128` elements, `2 * L * L` workgroups, which crosses 32,768 at **exactly
L = 128**. At 160 residues that is 65,536 workgroups dispatched for 51,200 needed:
**917,504 invocations past the end**, and at 129 residues 2,064,256.

🔴 **AND THE BACKENDS DISAGREEING IS THE SPECIFICATION.** WGSL leaves an
out-of-bounds write either discarded or clamped into the buffer. Vulkan discards
through `robustBufferAccess`; Metal has no hardware robustness and takes an
explicit index clamp. Clamped, every excess invocation executes
`base[last] += update[last]` on ONE address, non-atomically - a random multiple
of `update[last]`, new every pass. **So "the other machine does not reproduce it"
is evidence FOR this bug class, not against ours**, and twelve clean A100 runs
said nothing about the kernels. That inference cost most of the hunt.

It enters upstream of the trunk: `monomer.js` applies the template residual
through `addInPlace` once per recycle, unconditionally, so `pair[L-1][L-1][127]`
is corrupt before the first block and the triangle multiplications spread it
through every `(i, j)` within two. Hence `meanPlddt` itself varying - 32.358,
32.359, 32.357 at 200 residues - rather than only the coordinates, which is what
located it in the trunk rather than the structure module.

The whole fix is `if (index >= arrayLength(&base)) { return; }`. `arrayLength` is
exact because `dispatch` binds the tensor's own range, not the whole buffer.
`src/runtime/elementwise.js` had the identical hole and is harmless only because
its excess invocations all store the SAME value, which a `+=` does not.

**The rule that replaces the old audit** is in `test/folded-grid-guard.test.js`,
and it is structural rather than numeric: every index built from
`id.x + id.y * GRID_WIDTH * 64u` must be compared against a bound BEFORE anything
subscripts a buffer with it. Three shaders bind the bound to a `let` first
(`let elements = ...; if (index >= elements)`), so a rule reading only the next
line calls three correct shaders broken - it is the ORDER that matters, not the
line. The test asserts it found at least forty such shaders, because a pattern
that silently stops matching passes by finding nothing, which is the failure the
first version of the `id.y` audit had too.

### And the gate that would have caught it had been broken by an optimisation

`tools/gpu/probe-af2-contacts.js` scores the distogram head against the
structure, and it threw on every run: *"Cannot read properties of undefined
(reading 'length')"*. Chasing that found a second, worse thing.

The pair representation stays on the device now - `L^2 * 128` floats is 348 MB
at 825 residues, and taking it off the host was worth **19.68 -> 18.04 s**. The
change added an opt-in, `pairHost: true`, for the one caller that needs the host
copy. But `AlphaFoldMonomerGpu.predictA3m` forwarded a **hand-copied allow-list**
of five options to `predict`, and `pairHost` was not on it:

```js
{ tolerance: options.tolerance, signal: options.signal, chainLengths: options.chainLengths,
  resume: options.resume, resumable: options.resumable }        // and nothing else
```

`web/app.js` asks for `pairHost: true` **through `predictA3m`**. The list dropped
it, `recycle.pair` came back undefined, and `attachContactMap` begins

```js
if (weights?.distogram === undefined || recycle.pair === undefined) return;
```

so **the distogram contact overlay silently disappeared from the shipped page**,
with no error anywhere. The probe that exists to catch precisely this had been
disabled by the same commit, in the same way, at the same time.

🔴 **AND THE FIX WAS ALREADY WRITTEN, IN THE FILE NEXT DOOR.**
`src/multimer/model.js` forwards the whole options object and says why:

> *"This used to hand-copy five named options, which silently DROPPED the entire
> multimer regime ... so every fold through this entry point ran multimer
> WEIGHTS on the MONOMER graph and reported a plausible number for it. An
> allow-list of options is a list that goes stale every time one is added."*

The monomer had the identical seam and had not learned it. It forwards the whole
object now - and so does the third instance of it,
`AlphaFoldQueryOnlyGpu.predictSequence`, which was copying `tolerance`, `signal`
and `chainLengths` by hand. That one was latent rather than live: the page
dropped that driver and `predict` reads only those three. Three copies of one
mistake, two of them shipped. Verified end to end with `tools/fold-in-page.py --model monomer`:
both frames carry `maps: ["contact"]`, the panel shows its contact tab, and
`probe-af2-contacts.js` is back to AUC 0.999 and 1.000 with the map moving 0.152
between passes.

🔴 **AND THAT PAGE GATE HAD NEVER RUN ON THIS BOX.** `tools/cdp.py` hardcoded
`/Applications/Google Chrome.app/...` and `--headless=new` - so
`fold-in-page.py`, the tool that exists **because a contact map failed to appear
three times in a row**, could not launch on Linux at all, and on this driver
headless Chrome answers `requestAdapter` with nothing (it wants
`VK_EXT_headless_surface`, which NVIDIA does not implement). It finds Chrome on
the PATH now and takes the same Vulkan flags `tools/gpu-chrome.mjs` arrived at.
**macOS keeps exactly the flags it had.**

🔴 **AND THE 59-RESIDUE ROW IS A WASH, WHICH TOOK THREE RUNS AN ARM TO SAY.**

🔴 **AND THE 59-RESIDUE ROW IS A WASH, WHICH TOOK THREE RUNS AN ARM TO SAY.**
One run each read 1.047 s against 1.153 - a 10% REGRESSION, and a plausible
story for it was already written: a bigger generated shader costing compile
time on a fold too short to repay it. Both halves of that were wrong. The
pipeline compiles in **7.5 ms warm against the register kernel's 5.2**, and
three runs an arm give 1135/1111/1140 against 1128/1122/1119. The block
profiler had said the same thing all along - at 59 residues and 128 rows the
four attentions are 0.341 ms against 0.239 and the block 3.21 against 3.09.
**A single run of each is not a comparison**, which is this repository's own
rule, and a 100 ms difference on a 1.1 s fold is inside this box's noise.

🔴 **THE LAST 2.3 s IS THE FLASH ATTENTIONS ON THE MATRIX UNITS**, and it is
measured as a fold with the arms alternating - 18.03, 15.74, 18.08, 15.75, so
0.3% between repeats, which is what locked clocks buy:

| 825 residues, 512/1024 | elapsed | mainStack | pLDDT | pTM |
|---|---:|---:|---:|---:|
| `attentionMatrix` off | 18.03 s | 14.84 | 69.376 | 0.9672 |
| `attentionMatrix` on | **15.74** | **12.55** | 69.306 | 0.9672 |

🔴 **AND IT CHANGES THE STRUCTURE, SO HERE IS THE LADDER RATHER THAN AN
ASSURANCE.** All three measured in one tree, 825 residues:

| | pLDDT | pTM |
|---|---:|---:|
| f32 everywhere (`halfPrecision` off) | 69.360 | 0.9676 |
| f16 storage, register flash attention | 69.376 | 0.9672 |
| f16 storage, **matrix** flash attention | 69.306 | 0.9672 |

0.07 pLDDT from the kernel it replaces and 0.05 from full f32, inside the
spread the f16 storage path already produces, and the same pTM to four digits.
AF2's own inference runs in bfloat16, which has three fewer mantissa bits than
the halves these units multiply.

🔴 **AND A STALE BASELINE ALMOST TURNED THAT INTO A 1.4-POINT REGRESSION.** The
notes carried 70.761 for this fold from an earlier tree; measured here with the
matrix kernel OFF it is 69.376. A before-and-after against the remembered
number would have reported a drop that belongs to three intervening commits.
Re-measure the arm you are comparing against, in the tree you are comparing in.

🔴 **AND THE BLOCK IS FLAT AGAIN, WHICH IS THE USEFUL THING TO KNOW NEXT.** At
825 and 512 rows, 263.8 ms of 403 dispatches:

| | ms | share |
|---|---:|---:|
| the outer product mean, all of it | 45.2 | 17.1% |
| ...of which `opm.contract` | 30.0 | 11.4% |
| the four flash attentions | 69.3 | 26.3% |
| the two triangle contractions | 26.0 | 9.9% |
| the two triangle projections/outputs | 22.8 | 8.6% |
| the four attention q/k/v/gate projections | 37.3 | 14.1% |
| the four attention output projections | 21.9 | 8.3% |
| the four transitions | 31.0 | 11.7% |
| everything else | ~10 | 4% |

The two MSA attention projections issue **16.8 TFLOP/s each, 47% of the 36.0
this card's f16 vector path can reach** - and `src/runtime/matrix-linear.js`
measures its staged matrix form at 28.2, on a K of 256 that satisfies every
condition docs/A100.md sets for the units. That, not a better tile, is the next
thing to try: 37.3 ms of q/k/v/gate and 21.9 of output projection, against a
1.68x that would take about 24 ms out of a 264 ms block. The remaining structural gap to JAX is that a shader
reaches 18.1 TFLOP/s of f32 against 312 of bf16 tensor; the flash attentions
are now the one part of this block that does not pay that.

## What reading alphafold2-webgpu was worth

Their port is 10x this one at 825 residues, so the first question was what they
do differently. The answer was not a kernel trick; it was that **this path had
switched its own fast kernel off**.

### The finding: a 64 MiB constant, crossed at 128 residues

`useOuterFirstContraction` gated the outer-first contraction on a hardcoded
64 MiB. The intermediate is `L^2 * cOuter^2 * 4`, so **every protein longer than
128 residues** fell back to `opm.accumulate` - one invocation per (i, j, z),
whose inner read of `right` does not depend on z, so 128 invocations fetch the
same value. At 825 residues that is **93.7% of a block** and **1.9% of this
card's measured f32 ALU ceiling**.

| 400 residues, 512 rows | one block | a fold |
|---|---:|---:|
| 64 MiB cap | 591.2 ms | 41.34 s, 41.35 |
| the device's own limit | **86.9** | **8.90 s, 8.88** |

**4.65x**, from deleting a number. The cap is now the smaller of what the
adapter will BIND and what the memory budget has left - the two things that are
real. Above L = 724 the tiled path still runs, because 2.79 GB genuinely exceeds
`maxStorageBufferBindingSize`.

### What they do that this still does not

| | theirs | here |
|---|---|---|
| the contraction | one calibrated tiled GEMM serves every dense projection, OPM included | a bespoke kernel per operation |
| activations | **packed f16, two per 32-bit word** via `pack2x16float` - no device feature - for MSA, pair and the triangle whole projection | every evoformer tensor is f32 |
| kernel choice | compiled, **checked against a reference**, then timed at device creation; f16 must beat f32 by a margin | static per-architecture priors |
| falling back | windows the FAST kernel when it does not fit | switched to a slower, different one |

🔴 **AND ONE NEGATIVE RESULT OF THEIRS IS WORTH MORE THAN THE POSITIVES.**
Accumulating a whole contraction in f16 is the fastest arrangement and it is
unsafe: it overflows on a deep MSA, taking a 508-row prediction from **96.80
pLDDT to 69.94 and its pTM to NaN**. They record it in source rather than leave
it to a probe, because it only appears at a depth a cheap probe does not reach.
Anything done here with f16 arithmetic - and f16 is exactly 2.0x f32 on this
card - has to respect that.

🔴 **AND THEIR SUBGROUP-MATRIX RESULT DISAGREES WITH THIS FILE'S, WHICH IS A
HARDWARE FACT AND NOT A CONTRADICTION.** On an L40S their matrix kernel runs the
projection probe at 0.048 ms against 0.143 hand-tiled, and a whole recycle falls
0.86 -> 0.70 s; docs/A100.md measures matrix units LOSING at the diffusion
transformer's shapes here. Ada gives 91.6 TFLOP/s of FP32 shader against this
card's 19.5, so the same kernel sits in a different place on the two.

### What was NOT taken, and why

The 32 MiB `TRANSITION_CHUNK_TARGET_BYTES` looks like the same kind of
self-imposed cap and is not. It chunks the SAME kernel rather than choosing a
different one, and its own sweep prices it at **no measurable time** for
681 -> 573 MiB of peak. Removing it broke three tests that exist to say so. A
limit that changes the algorithm is a bug; a limit that changes the loop
structure at no cost is a measured trade.

## The nearest-centre search, on the device

🔴 **PREPARING AN ALIGNMENT IS 59% ONE LOOP, AND IT IS SERIAL WITH THE FOLD.**
The section below took the whole of featurisation from 525 ms to 75 on a
59-residue query and stopped there, because the fold's own clock calls all of
it "features" and nothing said which loop. `featureStats` says. At 825
residues, 512 clusters, 1024 extras and two recycles:

| phase | host | on the device |
|---|---:|---:|
| **nearest-centre search** | **645 ms** | **43** |
| the 49-channel MSA block | 204 | 196 |
| the cluster profile | 81 | 79 |
| extra rows | 54 | 50 |
| encode | 44 | 46 |
| BERT masking | 25 | 26 |
| the alignment profile | 10 | 10 |
| **featurisation** | **1072** | **450** |
| the fold | 23859 | **23254** |

**15x on the loop and 2.5% of the fold**, and at 59 residues 57 ms to 31 with
the repeat going 441/445 to 418/415. The checksum is **identical on both arms**
at both sizes - 26706680 and -1725774 - which is the only thing that settles
it, because the assignment reaches the answer through the cluster profile
rather than directly.

It is a reduction over residues and an argmax over centres, so it is a kernel:
one workgroup an extra row, the centres split across 64 lanes, a tree join.
`src/input/nearest-centres-webgpu.js`.

🔴 **THE TIE RULE IS THE WHOLE RISK AND IT IS IN THE PACK, NOT IN A
COMPARISON.** The host keeps the FIRST centre at an equal score. The join packs
`(score << 16) | (0xffff - centre)` into one u32 and takes the MAX, so the tie
breaks towards the lower index by construction - there is no comparison for
somebody to get backwards later. An empty lane's candidate is 0, which loses to
every real one because a real candidate carries at least `0xffff - centre` in
its low half.

`tools/gpu/check-nearest-centres.js` holds it to zero differing assignments
over seven cases, and one of them DUPLICATES centres so whole groups tie -
without it the check passes on random alignments and fails on real ones, where
near-identical sequences are the normal case. Flipping the pack to break ties
the other way fails six of the seven, 512 of 512 on the duplicate arm.

🔴 **AND THE LOOP WAS SPLIT SO THE RECYCLES BATCH.** Nothing in a recycle's plan
- its shuffling and its BERT masking - depends on an assignment, and no recycle
depends on another. `planA3mFeatures` produces every plan, one submit runs every
search, `finishRecycle` completes every recycle. Four searches, one round trip.
The host path runs the same two functions with the same loop in between, so the
arms cannot drift: they are literally the same code either side of the argmax.

Not taken from upstream, though they got there first and by the same
measurement: their host loop was 3.2 s of a 3.4 s featurisation where ours was
645 ms of 1072, because the alignment-prep work below had already made this
loop word-parallel. Their kernel is 28x and 45 ms at the same shape; ours is
15x and 43 ms. The remaining 400 ms is the 49-channel block and the cluster
profile, which is where they went next.

## Preparing the alignment

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

## The outer product mean's working set was an M2's, and it costs 2% at 825

🔴 **THE PRIORS WERE SWEPT AT 400 RESIDUES AND BELOW, AND ONE OF THEM MOVES.**
A block at 825 residues and 512 sequences is 192.81 ms and the four flash
attentions are 35.8% of it - already on the matrix units, and `attentionMatrix`'s
tile is confirmed right there too (4x32 gives 199.66 ms against 2x32's 206.14,
4x16's 205.98 and 6x16's 214.81). The outer product mean is the next family at
16.3%, and its WORKING SET was not swept at this length.

`opmPairBlockBytes` decides how many pairs the fast path holds at once. It is
not a limit - the path runs at every length whatever it says - so it trades
dispatch count against occupancy, which is a length question. Swept in situ,
block milliseconds and `opm.contract`:

| MiB | block | contract | | MiB | block | contract |
|---:|---:|---:|---|---:|---:|---:|
| 32 | 211.20 | 24.809 | | 256 | **195.43** | **17.972** |
| 64 (shipped) | 199.81 | 19.436 | | 512 | 194.01 | 17.527 |
| 128 | 198.25 | 18.571 | | 1024 | 193.53 | 17.196 |

**256 is the knee.** 64 -> 256 is 2.2% of a whole block and 1.08x on the
contraction; 256 -> 1024 buys 1.9 ms more for four times the memory.

In a fold at 825 residues, 512 clusters and 1024 extras, two pairs alternating:

| | main stack | whole fold | peak |
|---|---:|---:|---:|
| 64 MiB | 9.53 / 9.53 s | 12220 / 12236 ms | 6610.1 MiB |
| **256 MiB** | **9.36 / 9.36** | **12048 / 12039** | 6803.4 |

**180 ms for 193 MiB**, which is the working set exactly as predicted and 0.5%
of this card. At 59 residues it is inert - 966/947 ms against 971/952, repeat
415/421 against 416/417 - because the whole pair tensor is smaller than either
value there.

🔴 **AND IT REORDERS NO SUM, so there is no accuracy question to ask.** The fold
checksum is -67537339 and pLDDT 25.628 on both arms; check-opm-paths.js holds
the blocked arm to relRMS EXACTLY 0 because a pair's contraction is untouched
and only where it lands in the intermediate moves.

🔴 **AND FOUR KNOBS SWEPT AT THE SAME SHAPE DO NOT MOVE, WHICH IS THREE
QUARTERS OF THE SWEEPS.** Recorded so nobody runs them again:

| knob | at 825, 512 sequences | why |
|---|---|---|
| `attentionMatrixTile` | 4x32 **199.66** ms, 2x32 206.14, 4x16 205.98, 6x16 214.81 | the shipped tile is right at this length too |
| `opmProjectOutputPairs` | 9.153 / 9.170 / 9.155 / 9.157 ms at 1, 2, 4, 8 | the MATRIX output projection does not read it |
| `stagedMatrixBlock` | block 195.50 / 195.46 / 195.53 / 195.46 over four geometries | AF2's triangle contraction has its own geometry and never asks |
| `transitionThreadTarget` | block 195.44 / 195.28 / 195.47 / 195.64 at 50k, 100k, 200k, 400k | flat across an eightfold range |

The last is worth a second look because it contradicts an expectation rather
than confirming one: `probe-occupancy.js` measures this card at ~290,000 lanes
and the target is 100,000, so raising it looked overdue. It is flat, which says
the transition is not thread-starved at 825 in the first place - the shape has
enough rows to fill the card at any of these targets, and the knob only bites
where it does not.

**One sweep in four paid.** That is the honest rate, and it is still worth
doing: the one that paid was 2% of every block on a long protein, and long
proteins are where the seconds are.

## 🔴 AN ODD MSA DEPTH KILLED EVERY FOLD LONG ENOUGH TO BLOCK

Found by asking a different question - "where else would padding help?" - and
sweeping the divisibility gates. `tools/gpu/fold-af2.js` on a 400-residue chain:

| rows | |
|---|---|
| 37 | **dies** in WebGPU validation |
| 38 | folds, checksum −12062221 |
| 39 | **dies** |
| 40 | folds, checksum −14998047 |
| 61 | **dies** |
| 126, 128, 510, 512 | fold |

**Odd depths die.** The matrix outer product mean binds `left` as a view
starting at the block's first residue,

```js
const first = (offset / input.length) * input.cOuter * input.sequences;
```

and a bound range must start on a 256-byte boundary. With `cOuter` 32 that
offset is `residuesBefore * 128 * sequences` bytes, which is a multiple of 256
only when `sequences` is even - or when the block happens to start on an even
residue. At 400 residues with the ampere prior's 256 MiB block the second block
starts at residue **163**, and 163 × 32 × 37 × 4 is byte **771968**, 128 past a
boundary.

### Which folds it actually reached, traced

Three things have to hold at once, and the third is why it was never seen on a
gate: **the matrix contraction, an odd depth, and more than one pair block.**

**The matrix path.** At depth 1 the contraction is refused - K is too small for
a tile - and the vector kernel binds `left` whole with no view. So a
single-sequence fold of a 400-residue chain is safe, and gives −24505515 under
either rule. That is why "fold with no MSA" never showed it.

**An odd depth, and BOTH stacks have one.** The page's dial is the only thing
that sets the cap, and every preset is even - 512 (which becomes 508), 256,
128, 64, 32, 16, with **128:256 the default, not 512:1024**. Then:

| stack | depth it runs at | odd when |
|---|---|---|
| main | `min(cap, depth)`, and `centers.length` is exactly that | the alignment is SHALLOWER than the cap and odd |
| extra | `min(maxExtra, depth - maxMsa)` | the alignment's own depth is odd, since `maxMsa` is even |

So the extra stack is the wide one. At the default preset an alignment of
129-383 sequences runs the extra stack at `depth - 128` - odd whenever the
alignment is - and that is an ordinary depth for an ordinary protein, not an
edge case. Confirmed directly: `--rows=128 --extra-rows=73` died under the old
rule and `--extra-rows=72` folded to −9616646, which is the checksum the fixed
tree still gives.

**And more than one pair block**, which is a length question: at 400 residues
with the ampere prior's 256 MiB budget there are three, starting at residues 0,
163 and 326. A 59-residue fold has one block at offset 0 and cannot show it,
which is what every AF2 gate in this repository runs.

🔴 **THE 512 -> 508 IS NOT A ROUNDING STEP, WHICH IS EASY TO MISREAD.** AF2's
config spends four of its 512 rows on templates, so the CAP is 508 - and
`msaSequences` is `centers.length`, with no template rows added back. The
evenness of every cap is a coincidence of the presets, not a guarantee anything
enforces.

🔴 **AND WEBGPU NAMED THE WRONG TENSOR, WHICH IS WHY IT SURVIVED.** The error
reads `Offset (771968) of [Buffer "opm.left"]` in one place and
`[Buffer "extra.msa-row-attention.normalized"]` in another, for the same bug.
The allocator pools buffers by `${byteLength}:${usage}` and a pooled buffer
keeps the label it was CREATED with, so the name in a validation error is
whatever tensor first happened to want that size. `execution.js` checks the
offset itself now and throws naming the pass, the binding and the byte.

**The fix is the block boundary, not the depth.** `outerFirstPairsPerBlock`
already had to cut on a whole number of `i` rows; it now cuts on a whole number
of ALIGNED ones - two residues when the depth is odd, one when it is even. So
every alignment that worked before is byte-identical (38 → −12062221 and
40 → −14998047, unchanged), and the odd ones fold. `test/opm-pair-block-
alignment.test.js` sweeps length, depth and budget, and carries an arm that
asserts the OLD rule still produces byte 771968 - a gate that cannot fail is
not a gate.

### And the slower half of the same question, which is still open

Even when it does not crash, a depth that is not a multiple of FOUR loses the
vec4 read on two kernels: `vectorStaging` in outer-product-mean.js is gated on
`input.sequences % 4 === 0`. Interleaved, 400 residues, reproducible to
0.005 ms:

| depth | `opm.contract` | `opm.project-output` | whole block |
|---|---:|---:|---:|
| 512 | **3.872 / 3.875** | 1.608 / 1.609 | **59.48 / 59.49** |
| 510 | 4.755 / 4.759 | 1.890 / 1.892 | 60.64 / 60.65 |
| 128 | **1.432 / 1.434** | 1.627 / 1.623 | **30.39 / 30.39** |
| 126 | 1.647 / 1.647 | 1.906 / 1.902 | 30.85 / 30.83 |

**+23% on the contraction and +2.0% on the block**, and note that 512 does MORE
work than 510 and is still faster - the padded arm wins outright.

Padding the alignment depth up to a multiple of four with masked rows would
take it, and it is provably safe for the OPM: the denominator is
`sum(mask[i] * mask[j])` over sequences, so a zero-mask row contributes nothing
to the numerator or the denominator. It is NOT taken here because the padded
rows also flow through row attention, column attention and the MSA transitions,
and whether those are as clean has not been checked - a wrong answer is worse
than 2%. The alternative is a padded sequence STRIDE on `left` and `right`
alone, which is the `CZ_STRIDE` fix's shape and touches the most
performance-critical kernel in AF2.

## 🔴 The monomer alignment is never deduplicated, and the query is in it twice

Asked directly: when the search finds only the query itself, do we dedupe? No.

`extractMmseqs2A3m` joins two database blocks - `uniref.a3m` and
`bfd.mgnify30.metaeuk30.smag30.a3m` - and `queryBlock` returns each block WHOLE,
starting with its own `>101` query row. Run against a result that contains
nothing but the query:

```
environmental=true   depth=2  distinct=1  rows=["101","101"]
environmental=false  depth=1  distinct=1  rows=["101"]
```

The environmental database is the default, so **every monomer fold carries the
query twice**, and every sequence found in both databases twice with it.

On `tools/fixtures/test.a3m`, parsed by this repository's own `parseA3m` so the
comparison is on the aligned columns rather than the raw text: **8076 rows, 7663
distinct, 413 duplicates - 5.1%**. At the page's default 128:256 that is about
**20 of 384 rows** spent on sequences the model has already seen.

🔴 **AND THE CODEBASE ALREADY ARGUES THIS, ONE FILE OVER.**
`deduplicateUnpairedAgainstPaired` in src/input/chains.js does exactly this for
the multimer's paired and unpaired blocks, with the reasoning written out - "it
is not a tidiness pass, it is the MSA budget... a duplicate does not merely add
nothing, it evicts a sequence that would have added something" - and with the
note that the comparison has to be on the aligned columns because AlphaFold
hashes the featurised row. `concatenateA3mBlocks` in the same file skips the
second block's query row for the same reason (`for (let row = 1; ...)`). The
monomer's two-database join does neither.

AlphaFold's own `make_msa_features` keeps a `seen_sequences` set and skips a
sequence it has already taken, across all MSAs - so this is a divergence from
the reference, not only a waste. (From knowledge of that pipeline; there is no
checked-in copy here to point at.)

### Fixed, and what it does to pLDDT: nothing measurable, and one thing that matters

`planA3mFeatures` drops a row whose ALIGNED sequence it has already taken,
keeping the first - so the query stays row 0. `--no-dedupe` on fold-af2.js is
the control arm.

**On a deep alignment it is worth nothing.** `tools/fixtures/test.a3m`, paired
by seed because the seed drives which rows the budget draws:

| budget | seeds | delta pLDDT | delta pTM | seeds favouring dedupe |
|---|---:|---|---|---|
| 128:256 (the page default) | 8 | **-0.005 ± 0.288** | +0.0008 ± 0.0039 | 5/8 |
| 508:1024 | 6 | **-0.036 ± 0.129** | -0.0008 ± 0.0019 | 2/6 |

The paired delta's standard deviation is sixty times its mean at the default
budget. Against the captured AlphaFold reference of 96.625 the deduplicated arm
is nominally closer - mean |error| 0.118 against 0.155 - and that is inside the
noise too. 413 of 8076 rows are dropped, and ~20 of the 384 the budget draws;
the alignment is 8076 deep against a 384-row budget, so a few more distinct
rows change nothing. **The MSA budget is not the binding constraint at this
depth.**

🔴 **BUT ON A QUERY-ONLY SEARCH IT CHANGES THE ANSWER, AND THAT IS THE CASE
THAT MATTERS.** Folding the query alone against the query duplicated - which is
literally what `extractMmseqs2A3m` returns when nothing is found:

| A3M | arm | pLDDT | pTM | checksum |
|---|---|---:|---:|---:|
| query once | either | 59.974 | 0.3965 | -459839 |
| query TWICE | **deduplicated** | **59.974** | **0.3965** | **-459839** |
| query twice | plain | 60.079 | **0.4148** | -426577 |

Deduplicated, a duplicate row is a no-op and the fold is bit-identical to the
single-sequence one. Plain, the second copy of the query moves pTM by 4.6% - a
number the page shows a user. Every monomer fold with the environmental
database carried that second copy.

### And an alignment carrying only the query is a single-sequence fold now

All four ways an alignment reaches a fold - **searched, pasted, uploaded as an
A3M, uploaded as an archive** - go through `singleSequenceIfOnlyQuery`, which
counts DISTINCT rows and at one returns the query-only shape, `text: null`.
That is the state "Single Sequence" mode already produces, so every consumer
below already handles it; the search path keeps its template hits, because a
protein with no homologs may still have a structure to lean on. The status line
says which input it was, and the fold's own summary then says "single sequence"
rather than "2 MSA rows".

🔴 **AND FOR A PASTED OR UPLOADED ONE IT ALSO CHECKS WHOSE QUERY IT IS.** An
A3M's own first record WINS over the sequence box, deliberately, so that a
reader can paste an alignment and fold what it describes - and `text: null`
throws that away. `foldsAsSingleSequence` therefore requires the alignment's
query to BE the sequence about to be folded; where they differ the alignment is
kept and folds the protein it names, exactly as before. A searched alignment
cannot differ, since `generateMmseqs2Msa` already refuses an A3M whose query is
not what it asked about - so the guard costs the search path nothing and
protects the two paths a reader controls.

The decision is `foldsAsSingleSequence` in src/input/a3m.js rather than inline
in web/app.js, because node cannot import that file - it wants a DOM - and
`test/a3m-distinct-sequences.test.js` covers the query-mismatch case, the
complex's concatenated query, insertions, and gaps.

🔴 **THERE IS NO SPEEDUP IN IT, WHICH IS WORTH SAYING.** `elapsedMilliseconds`
over the three arms above: **855, 840, 843** - the same. A homolog-free search
already yields a two-row alignment, `maxExtra` collapses to `max(1, 0) = 1`
either way, and web/app.js already funnels single-sequence mode through the
same `predictA3m` with a one-row A3M (`alignment === null ? ">query\n..."`).
The change is correctness and honesty, not time.

### 🔴 And it exposed that the synthetic alignment had twelve distinct rows

`fold-af2.js` built its rows with a gap stride of `row % 11 + 3`, so rows 1, 12,
23 and so on were IDENTICAL: `--rows=128 --extra-rows=128` was 256 rows carrying
**12** sequences. Harmless while nothing read what a row SAID - the work is the
same, so every timing in this file stands - and fatal beside deduplication,
which would have collapsed every AF2 gate to a depth-12 fold while still
reporting 256. The stride is the row's bit pattern now and the tool ASSERTS
distinctness rather than arguing for it.

That is what moved the three AF2 baselines - AF2 **-1287025**, the 30,29
multimer **315591**, and `bench-af2-warm` -36457799 at the time - **-121844157**
now that its own generator is fixed too - and not the deduplication:
`--no-dedupe` gives -1287025 as well.

## The 160-residue race is not reproducible on the A100

An M2 reports `bench-af2-warm.js` - the race gate, three passes of one graph
over one input in one process - throwing at 160, 200 and 400 residues and
passing at 59, 80, 100 and 128, on `main` and on this branch alike. The same
sweep on this A100, `--rows=128`:

| length | 59 | 80 | 100 | 128 | 160 | 200 | 400 | 825 |
|---|---|---|---|---|---|---|---|---|
| | ok | ok | ok | ok | **ok** | **ok** | **ok** | ok |

🔴 **AND UNCHANGED AFTER THE GUARD LANDED, BYTE FOR BYTE.** The same sweep
against the fix returns the same eight checksums - -354741, -4156549, -2580429,
-2891643, -9869845, -704434, -15124842, -36457799 - which is what a guard never
reached on this backend has to do. Both SHAPES are deterministic too, now that
`--rows` selects one: at 160 and 400 residues, six passes each, the one-call
fold and the forty-nine-call fold each agree with themselves.

Then the three failing lengths pressed harder - `--passes=8`, so 28 pairs a run
to disagree on rather than 3, four rounds each: **twelve runs, all agreeing**,
and the checksums identical ACROSS runs too (160 is -9869845 every time, 200
-704434, 400 -15124842). The standing gate has always run 825 and passes.

So the code is deterministic on Dawn over Vulkan on an A100 at every length the
M2 fails. That does not prove the kernels are correct - a race can be latent and
need a scheduler to expose it - but it does say the mechanism involves the other
backend, which is the half of the search space the question was asked to remove.

### 🔴 `--no-pool`, which tests the leading hypothesis in one run

The M2's remaining lead was buffer recycling: the allocator pools by
`${byteLength}:${usage}`, a pooled buffer keeps the previous fold's bytes, and
any kernel READING a region it did not WRITE gives one answer on a fresh buffer
and another on a reused one. That looks exactly like a race and is not one - it
is deterministic given the pool's state and appears only from the second fold
onwards, which is precisely what this gate compares.

`bench-af2-warm.js --no-pool` answers it. On the A100 at 160 residues the
checksum is **-9869845 either way** - pooled and unpooled - so recycling changes
nothing here.

🔴 **AND IT RETIRES RATHER THAN DESTROYS, WHICH IS WHY THE POOL EXISTS.** The
first version destroyed on release and every length died with `extra-MSA block
0: [Buffer] destroyed`: a released buffer is still named by an encoded or
in-flight command buffer, and the pool keeps it alive by accident of holding it.
Retiring keeps it alive on purpose and frees nothing until teardown, so it
**leaks by design** - fine at 59-160 on a 40 GB card, and `Error.cpp:119` at 200.
Use it at short lengths.

### And the cheapest diagnostic is already in the error message

`bench-af2-warm` prints every checksum and `new Set(checksums).size`, and the
PATTERN separates the two mechanisms:

- `2 different structures: A, B, B` - the first pass differs and the rest agree.
  That is a first-touch difference, not a race: pass 1 reads WebGPU's
  zero-initialised buffers and passes 2+ read recycled ones. Deterministic, and
  it points straight at a kernel reading what it did not write.
- `3 different structures: A, B, C` - genuinely nondeterministic, and a
  scheduling question.

`--passes=8` sharpens it: an uninitialised read stays two values however many
passes are added, a race keeps producing new ones.

## Why the A100 was clean, measured rather than assumed

The M2 found it: `ADD_IN_PLACE_SHADER` indexes a folded grid with no bounds
check, `linearGrid` rounds a dispatch up to whole workgroups and then to whole
rows of y, and on Metal the out-of-range invocations CLAMP onto the last
element - hundreds of thousands of non-atomic `+=` on one address. The
arithmetic, over the pair tensor at 128 channels:

| n | elements | grid | over-dispatch |
|---|---:|---|---:|
| 59 | 445,568 | 6962x1 | **0** |
| 100 | 1,280,000 | 20000x1 | **0** |
| 128 | 2,097,152 | 32768x1 | **0** |
| 160 | 3,276,800 | 32768x2 | **917,504** |
| 200 | 5,120,000 | 32768x3 | 1,171,456 |
| 400 | 20,480,000 | 32768x10 | 491,520 |
| 825 | 87,120,000 | 32768x42 | 960,384 |

Zero up to 128 because 128*128*128 is exactly `GRID_WIDTH * 64`, one full row -
which is where the M2's threshold is, and 917,504 is the figure it reported.

🔴 **AND THE REASON THIS BOX NEVER SAW IT IS THE BACKEND, WHICH IS NOW
MEASURED.** `tools/gpu/probe-grid-overdispatch.js` drives the SHIPPED shader
over a deliberately over-dispatched grid and reads the last in-range element,
which should be exactly 1 after one `+= 1` each. On this A100 at the 160-residue
shape - over-dispatch 917,504 - it is **1, 1, 1** across three runs: Dawn over
Vulkan DISCARDS an out-of-range write where Metal folds it onto the tail. The
128 control reports that nothing over-dispatched, so the probe cannot pass by
accident.

So the missing guard was invisible here rather than absent, and "it does not
reproduce on the A100" was luck of the backend, not evidence about the code.
WebGPU permits either behaviour; neither is a guarantee to lean on.

🔴 **SEVEN CALL SITES EXIST; HOW MANY RUN DEPENDS ON THE MSA DEPTH, AND AT
THE BENCHMARK'S DEPTH IT IS ONE.** `addInPlace` serves the TEMPLATE residual in
monomer.js, multimer/model.js and query-only.js - once a fold - and the OUTER
PRODUCT MEAN residual in evoformer/block.js and multimer/block.js. The OPM ones
are CONDITIONAL: `encodeOuterProductMean` writes straight into the pair tensor
and returns it whenever the outer-first path is taken, and the caller is
`if (update !== pair)`. `useOuterFirstContraction` is
`sequences >= cOuter && bytesPerPair <= limit`, so the branch turns on the
alignment's depth against **32**.

Counted on the device rather than read off the source, `fold-af2.js --length=160
--extra-rows=64`:

| clustered rows | `addInPlace` calls in a fold |
|---:|---|
| 128 | 1 - `monomer.template-residual-0`, and nothing else |
| 16 | **49** - the template residual, plus `outer-product-mean.residual` **48 times**, once per main-stack block |

So at any normal depth the exposure is the template residual alone, and
`--extra=0` still raced because THAT is unconditional - not because of
`evoformer/block.js:1255`, which does not fire at 128 rows. **Below 32 rows in a
stack the same unguarded read-modify-write lands on the pair tensor 48 times a
recycle instead of once**, which is the deeper exposure and is exactly the
shallow-alignment case a novel protein hits. All seven share the one shader, so
one guard covers every one of them - including the two an M2 with only the
monomer bundle cannot fold.

🔴 **AND THE RACE GATE ITSELF WAS ALWAYS IN THE 49-CALL SHAPE, WHATEVER `--rows`
SAID.** The count above is `fold-af2.js`, whose generator had already been
fixed. `bench-af2-warm.js` still carried the degenerate one - gap stride
`row % 11 + 3`, so rows 1, 12 and 23 identical - and its alignment was **12
distinct sequences at every `--rows`**, which the featuriser's deduplication
then collapsed it to. Twelve is below 32, so the outer-first path was never
taken and the OPM residual always fired. Counted on the device with
`tools/gpu/probe-add-in-place.js` at 160 residues:

| `--rows` | dispatches | |
|---:|---:|---|
| 128 | **53** | 48 main-stack + 4 extra-stack OPM residuals, 1 template |
| 16 | **53** | identical - the alignment is the same twelve rows either way |

Every one over the pair tensor at 3,276,800 elements with **917,504**
invocations past the end. So the race was found in the worst shape available,
which is lucky, and the gate could not have tested the one-call shape at all.

With the generator fixed here too, `--rows` selects it: **1** dispatch at 128
rows and **49** at 16, reproducing the table above in the gate. Both shapes are
deterministic on the A100 after the guard - 160 and 400 residues, six passes
each - and the default checksum moves to **-121844157**.

## Reading alphafold2-webgpu again: the residual add could not fold a big complex

Forty commits landed upstream between 2026-09-09 and 09-11, nearly all from
**@milot-mirdita**.

🔴 **THE REPOSITORY IS PUBLIC, AND UNLICENSED, AND THOSE ARE DIFFERENT THINGS.**
An earlier note here said it was marked `"private": true` in a way that read as
"the repository is private". It is not: the API reports `private: false`,
`visibility: "public"`. The `"private": true` is in `package.json` and means
only "do not publish this to npm". What IS true is `license: null` and no
LICENCE file at the root - so it is readable and forkable on GitHub and carries
no copyright grant to copy source into another project. Everything below was
read for the idea and re-derived and measured here.

🔴 **AND THEIR CLAIMS WERE CHECKED, NOT QUOTED.** Two of the three that could be
tested here behaved differently than reading them suggested; see the table.

### Taken: windowing the residual add

`Window the residual add, so a complex past 2,896 residues runs at all` says
`addInPlace` bound both tensors whole, and an ordinary tetramer therefore died
planning 13.1 GB on a 97 GB card - because the limit is not memory, it is
`maxStorageBufferBindingSize`, which is Vulkan's `maxStorageBufferRange` and no
request raises.

**We had the same gap, at a LOWER ceiling.** Their pair is packed f16 at two
bytes a channel, so theirs is `sqrt(2 GiB / 256)` = 2,896 residues; ours is f32,
so ours is `sqrt(2 GiB / 512)` = **2,047**. Demonstrated rather than computed, by
`tools/gpu/probe-residual-binding-ceiling.js` on a card with 40 GB free:

| residues | pair | before | after |
|---:|---:|---|---|
| 2,047 | 2.00 GiB | ok | ok, one window |
| 2,048 | 2.00 GiB | **REFUSED** - "Binding size (2147483648) is larger than the maximum storage buffer binding size (2147483644)" | ok, two windows |
| 2,896 | 4.00 GiB | refused | **ok**, two windows |

The add is elementwise, so it windows: each dispatch binds a range that fits,
and the guard reads `arrayLength(&base)`, which is the WINDOW's length - so the
last workgroup of a window stops at the window instead of running into the next.
Below the ceiling the single-dispatch path is untouched and every checksum is
unchanged.

🔴 **AND THE NEXT WALL IS A DIFFERENT ONE.** At 3,300 residues the buffer fails
before the binding does: `maxBufferSize` is 4 GiB here, so a single f32 pair
tops out at **2,896** residues however it is bound. Windowing cannot help that;
packing the pair to f16 or sharding it across buffers would, which is what
upstream's packing work does.

🔴 **AND THE FOLD'S OWN CEILING IS STILL 2,047, BECAUSE addInPlace WAS NOT THE
ONLY SITE.** An earlier version of this section said a complex past 2,047
residues folds now. It does not: `tools/gpu/probe-binding-ceiling.js` runs the
fold at two lengths, takes each labelled dispatch's growth exponent, and
extrapolates where its largest binding meets the limit. Over 89 labels in an AF2
fold:

| label | exponent | crosses at |
|---|---:|---:|
| `opm.contract`, `opm.project-output` | 2 | **724** - known and HANDLED, the tiled path takes over here |
| `pair-transition.first` / `.second`, and the extra stack's | 2 | 1,448 - handled, `transitionChunkRows` windows them |
| `triangle.outgoing.*`, `triangle.incoming.*` | 2 | **2,047 - NOT handled** |

Windowing the residual removed one wall of several and the fold still stops in
the same place. What the fix is worth is that the residual is no longer the
FIRST thing to fail, and what it is not worth is a longer complex.

🔴 **AND WINDOWING THE TRIANGLE WOULD BE WORTH NOTHING, BECAUSE 2,047 IS A TIE
OF TWENTY-EIGHT.** The table above names the triangle because the probe returned
the first row of a sorted list, and reading it that way is how a session came to
be spent asking whether `src/triangle/webgpu.js` could bind ranges. It cannot,
and it does not matter twice over: that module is the STANDALONE kernel that
`check-triangle.js` drives, and the fold's triangle is `src/evoformer/block.js`
and its multimer twin, which dispatch through `execution.view()`-capable
tensors and could be windowed. The reason not to is the count. Every dispatch
binding an `L^2 * cZ` f32 tensor crosses 2 GiB at the same residue, and there
are twenty-eight of them:

| ceiling | labels | what they are |
|---:|---:|---|
| 724 | 2 | `opm.*` - handled, the tiled path takes over |
| 1,448 | 4 | the four pair transitions - handled, `transitionChunkRows` |
| **2,047** | **28** | ten triangle multiplication passes and two gates; both triangle attentions' `normalize` and `output`, in both stacks; both MSA row attentions' `pair-normalize` and `pair-bias`; two pair-transition `normalize`; `template.output`; `monomer.template-residual-0` |
| 2,896 | 22 | the half-width bindings - the template stack, whose cZ is 64, and both triangle attentions' `pair-bias`/`project`/`flash`. Also where a 4 GiB `maxBufferSize` stops an f32 pair being ALLOCATED at all |

So windowing the twelve triangle labels moves the fold's ceiling by **zero
residues**: the thirteenth member of the tie refuses at the same length. The
whole group has to move together, and the destination is 2,896 whatever route is
taken, because that is the allocation wall.

🔴 **AND 2,047 IS THIS CARD'S NUMBER, NOT THE PORT'S.** Measured on both boxes
with `probe-limits.js`:

| | M2 | A100 |
|---|---:|---:|
| `maxStorageBufferBindingSize` | **4 GiB** | 2 GiB |
| `maxBufferSize` | 4 GiB | 4 GiB |
| where an f32 pair stops BINDING | **2,896** | 2,047 |
| where an f32 pair stops ALLOCATING | 2,896 | 2,896 |

The ceiling is `sqrt(maxStorageBufferBindingSize / (cZ * 4))`, and the A100's
binding limit is Vulkan's `maxStorageBufferRange` at exactly half the M2's -
hence exactly `sqrt(2)` between the two lengths. **So on Apple silicon the two
walls coincide and there is nothing to window at all:** the 28-label tie sits on
the allocation wall, which no windowing passes. The twenty-eight windowings
would buy a length only on the card where the binding limit is the smaller of
the two, and even there only up to where the other one stops it.

🔴 **AND THE CHEAP ROUTE TO 2,896 IS THE ELEMENT, NOT THE WINDOW.** A packed f16
pair is two bytes a channel, so its bindings cross 2 GiB at 2,896 - exactly the
allocation wall - and it needs no windowing anywhere. That is why upstream's
number is 2,896 and ours is 2,047: the difference is the element size, not the
windowing. Ours is f32 across the pair track (`createAttentionPairBiasShader` is
the one AF2 shader that takes an f16 pair today), so that route is a port and
not a patch. Twenty-eight windowings reach the same place.

🔴 **AND THE PRICE OF ARRIVING IS MINUTES.** `profile-af2-block.js --length=2040
--sequences=8` runs on this card today: one main-stack block is **1,260 ms**, so
the 48-block stack is **60.5 s per recycle** before the extra stack, the
featurisation or the structure module. The contraction is O(L^3), so 2,896 is
about 3x that a block. A 2,896-residue monomer is a ten-minute fold in a browser
tab, which is the other half of why this is recorded rather than built.

`probe-binding-ceiling.js` returns `lowestCeilingGroup` and `ceilingGroups` for
exactly this reason now. A one-label fix is worth something only where
`lowestCeilingGroup.labels` is 1.

🔴 **AND THE MULTIMER'S TIE IS FORTY-TWO, NOT TWENTY-EIGHT.** The table above is
the monomer. Run against `--family=multimer`, the same 2,047 carries 42 labels -
the same twelve triangle passes plus the TEMPLATE stack's own triangle
multiplication, its two pair transitions, `template.input` and `template.output`
- against 82 labels in the fold. Same conclusion, more of it.

🔴 **AND BOTH SAMPLED LENGTHS HAVE TO SIT ABOVE EVERY CHUNK THRESHOLD.** This is
the probe's own trap and it fired twice before it was noticed. A transition
chunks against `TRANSITION_CHUNK_TARGET_BYTES` (32 MiB), so below that it grows
as a clean `L^2` and the extrapolation sails through a threshold it cannot see:

| sampled at | what the pair transitions read |
|---|---|
| 59 and 118 (monomer) | a ceiling of **1,448** |
| 60 and 120 (multimer) | a ceiling of **1,023** |
| 200 and 400 | **absent from the ranking** - chunking has started, the exponent is ~0 |

All three describe the same handled code. `sampleFraction` is the largest
sample as a fraction of the limit and `caveat` says it out loud: at 59/118 the
lowest group extrapolates from **2.7%** of the limit, which is a guess across
more than an order of magnitude. At 200/400 the transitions and the outer
product mean both drop out and the 2,047 tie stands on its own - 26 labels
there against 28, the difference being two template labels those lengths do not
reach.

🔴 **AND TWO LENGTHS ARE NEEDED TO SEE ANY OF THIS.** A binding that grows as `L`
and one that grows as `L^2` are indistinguishable in a single run and give out at
completely different lengths. The exponent column is what separates "already
windowed" - the 724 and 1,448 rows, both of which the code handles - from a real
ceiling.

🔴 **AND THEY FOUND OUR RACE FROM THE OTHER SIDE.** The same commit says "the f32
shader gets the bounds check its packed siblings always had. It relied on the
WebGPU bounds clamp, which the native path turns off." That is the identical
missing guard the M2 found here, reached by a third route: for them a native
wgpu path with robustness off, for us Metal clamping onto the last element, and
for this A100 a backend that discards and hid it entirely. Three exposures, one
bug, and the WGSL specification permits all three.

### Read and not taken, with their numbers

| upstream | what it is | why not here, yet |
|---|---|---|
| `Make a bind group once for what it binds` | 5,157 bind groups a recycle at 59 residues, **70%** rebuilding an identical one | 🔴 **THE COUNT TRANSFERS AND THE RATE DOES NOT.** `probe-submits.js` counts 5,436 for AF2, close to their 5,157 - but keyed on what each one actually binds (buffer object, offset, size) only **20%** repeat here, 1,078 of 5,436, and 23% for AF3. That is 6 ms of 32. Their blocks are handed the same pooled scratch and differ only in uniforms; ours bind a different weight buffer per block. **Not worth taking here** |
| `Fold the query normalization into the global gate's weight` | the normalisation is affine per channel and the gate contracts over channels, so scale x weight is one tensor and offset x weight sums into the bias | **Algebra verified here** (1.01e-6) and then **NOT TAKEN, because this port already hoisted it another way** - see below |
| `Give the triangle contraction twice the output rows a workgroup` | halves the weight staging | Priced upstream at 1.7% of a 3,300-residue recycle and **0.6%** at 825 - below what this box can resolve without many paired runs |
| `Report the ceiling a device's binding count sets` | `maxStorageBuffersPerShaderStage`, 8 on this card | Already respected here: several shaders sit at exactly 8 and the atom encoder packs ten gathers into one buffer citing "the eight-buffer guarantee". Not a gap |

🔴 **AND THE BEST LINE IN THE FORTY IS A NEGATIVE RESULT, AGAIN.** The same
commit adds that "the mean stays inside the sum rather than being factored out
against a column sum, which keeps the arithmetic as stable as the form it
replaces". Factoring is the obvious next step - `sum_c (x_c - mean) * W_cj`
becomes `dot - mean * colsum`, one pass instead of a subtraction an iteration -
and it is a trap. Simulated in f32 against an f64 reference, 256 channels:

| row mean | mean kept inside | factored against a column sum |
|---:|---:|---:|
| ~0 | 2.12e-5 | 3.78e-5 |
| ~10 | 3.99e-7 | 5.37e-5 |
| ~1,000 | 2.71e-7 | 7.68e-4 |
| ~100,000 | 1.37e-6 | **2.54e+0** |

At a large mean the two terms nearly cancel and the factored form loses the
answer entirely. Random test data does not show this - it needs a row whose mean
is large against its spread - which is exactly why the caution is worth having
written down rather than rediscovered. Same shape as their f16-accumulator
warning: the negative result travels further than the optimisation.

### And the gate fold does not transfer, because we fixed this kernel already

Their gate "normalized every value as it read it: subtract the mean, scale by
the inverse deviation, then a per-channel scale and offset, which is four loads
and four operations an iteration for one multiply-add of actual work", and they
call that loop the largest cost in their extra-MSA block after the outer product
and the flash kernel. Folding the affine part into the weight removes it.

**Here the gate reads a tensor that is already normalised.**
`ATTENTION_NORMALIZE_SHADER` materialises `(x - mean) * invStd * scale + offset`
in its own pass and `extra.msa-column-global-attention.output` does
`staged[c] = normalized[...]` - one load, one multiply-add. The per-iteration
cost their fold removes is not there to remove.

And the pass cannot be dropped by folding, because `normalized` is shared.
Checked in `encodeGlobalAttention` rather than assumed - an earlier draft of this
paragraph said "the query, kv and flash kernels", and the FLASH kernel does not
read it, it reads query, keys and values. The three that bind it are
`kvPipeline`, `queryPipeline` and the gate itself. Folding into the gate's
weight alone would save the gate's arithmetic - already one load - and leave the
tensor exactly where it is for the other two.

The numbers, `profile-af2-block.js --stack=extra --length=825 --sequences=1024`,
a 189 ms block over 200 kernels:

| kernel | ms | share |
|---|---:|---:|
| `extra.msa-column-global-attention.flash` | 5.77 | 3.08% |
| **`...global-attention.output`** - the gate their fold targets | **3.08** | **1.64%** |
| `...global-attention.normalize` | 1.11 | 0.59% |
| `...global-attention.kv` / `.query` | 0.71 / 0.61 | 0.38% / 0.33% |

🔴 **AND THE REASON IT IS 1.64% IS THAT THIS FILE ALREADY FIXED THE BIG
VERSION.** The note further up records the same kernel recomputing the gate once
per output channel - 33x the arithmetic, **289.93 ms of a 681.61 ms extra-MSA
block, 42.5%, the largest kernel in the whole fold**. A workgroup owns a row
now. Their fold and that fix attack the same kernel from different sides, and
having done the larger one leaves 3 ms where they had hundreds. Two ports, the
same hot spot, two different roads out of it.

## Both ports, re-measured: 11.02 s against 13.15

The three-way table above had not been re-run in a long campaign, and
`martin-steinegger/alphafold2-webgpu` had landed a run of performance commits
the day this was taken - its HEAD is `a7e02ed`, dated 2026-09-11. So both sides
were re-measured, **in one page, in one Chrome, interleaved pass by pass**,
which is the only arrangement this box's 3.2x drift allows.

825 residues, 512 clusters / 1024 extra, one trunk pass, features included,
weights already loaded. Warm is the minimum of two repeats after a cold one:

| | cold | warm | recorded before |
|---|---:|---:|---:|
| alphafold2-webgpu at `a7e02ed` | 14.60 s | **13.15** | 14.21 |
| **LocalFold at `94d3902`** | 11.72 | **11.02** | 15.74 |

**1.19x**, where this file last recorded 10.5x the other way. Both fold the same
structure: pLDDT 26.448 theirs and 26.317 ours, which is the low-confidence
read of a 14x tandem repeat both ports agree on. Against the JAX baseline **as
recorded earlier on this card** (5.27 s, not re-run), ours is 2.09x native.

LocalFold on its own harness at the same shape reads **11.23 s** warm against
11.02 here, so the harness is not the difference - which had to be shown rather
than assumed, and see below for why.

By shape, `fold-af2.js --recycles=0 --repeat=3`:

| | cold | warm | recorded before |
|---|---:|---:|---:|
| 59 residues, 128/256 | 0.764 s | **0.196** | - |
| 400 residues, 512/1024 | 4.055 | **3.500** | 5.129 |
| 825 residues, 512/1024 | 11.892 | **11.230** | 15.74 |

The cold-warm delta stays flat in the shape - 0.568, 0.555 and 0.662 s over a
14x range of length - where theirs was 0.920 to 4.100 and grew with it.

### How to run theirs, and the three traps that make the number wrong

Their `tools/benchmark-a3m-model.ts` cannot run from a clone: it opens
`test/fixtures/evoformer/model1-a3m-59-stack/manifest.json`, which their own
`.gitignore` excludes as a "local development asset", and it imports the
`webgpu` node binding, which is this box's GLIBC 2.38 wall. What works is their
BROWSER harness - playwright against their vite dev server, with a spec that
imports their monomer through vite's `/@fs/` route and folds against their
published q8 bundle at `martin-steinegger.github.io`. Their `predict-a3m.ts`
already reads `AFWEBGPU_MANIFEST` for exactly that bundle.

🔴 **THREE THINGS EACH COST A FACTOR AND EACH LOOKS LIKE A RESULT.** Every one
was caught by checking the machine rather than the output:

| | | |
|---|---|---|
| **their playwright config is `headless: true`** | headless Chrome cannot bring up Vulkan here - it wants `VK_EXT_headless_surface`, which the NVIDIA driver lacks - so it silently takes the software adapter | Chrome's GPU process at **1179% CPU with the card at 0%**. `--headed` under `DISPLAY=:99`. This is the same discovery `gpu-chrome.mjs` is built around |
| **no `--enable-dawn-features=vulkan_enable_f16_on_nvidia`** | Dawn refuses `shader-f16` on every NVIDIA GPU without it, so **BOTH ports lose their f16 path** and the comparison measures neither | theirs 23.20 -> **13.15**; ours 25.03 -> 23.61 with the third trap still in place |
| **a hand-rolled `requiredLimits` for our arm** | our kernels pick their shape from `device.limits`, so a device asked for differently is a differently-configured port | ours 23.61 -> **11.02**. `requestAlphaFoldDevice(adapter)` is what `gpu-chrome.mjs` and the page both call; use it |

Their side needs one thing of its own: their native harness fits a scratch
budget to host memory, which a browser cannot ask for and which their README
prices at a third of the speed on a long chain. Passing this card's 36 GiB to
their own `fitScratchBudgetScale` picks scale 16 and takes them from 33.95 s to
22.11 without f16 - so the arm above is their FAST path, not a browser
handicap.

🔴 **AND THE ORDER OF THOSE THREE IS THE LESSON.** Each one produced a plausible
number: 33.95, 23.20, 23.61. Any of them could have been written down as "the
other port is 3x slower" or "ours is 2x slower", and all three were wrong for
reasons nothing in the output said. The card's own utilisation - `nvidia-smi`
next to the run - is what caught the first, and the first is what made the other
two visible.

## What is left between here and JAX, priced

11.02 s against the recorded 5.27 is **2.09x**, and the question is where it
lives. Two measurements answer it.

**It is the main stack, and nothing else is worth attacking.** A fold at
825/512/1024 splits: main stack 9.22 s (77.6%), extra stack 0.84, confidence
0.41, template 0.33, embedder 0.32, structure 0.27, features 0.26, warm 0.18,
trunk readback 0.04. Everything that is not the 48 main blocks is **2.66 s
together**, so a fold that did all of it instantly is still 9.22 s and 1.75x
JAX. There is no host-side or stage-level lever left; there is one loop.

**Inside the block, the gap is arithmetic rate, and the block contains its own
control.** At 825 residues and 512 rows, by GFLOP over measured milliseconds:

| kernel | ms | GFLOP | TFLOP/s |
|---|---:|---:|---:|
| `opm.contract` | 17.98 | 713.7 | **39.7** |
| `msa-row-attention.project` | 8.45 | 221.5 | 26.2 |
| `msa-column-attention.project` | 8.47 | 221.5 | 26.1 |
| `triangle.outgoing.contract` | 6.73 | 143.7 | 21.3 |
| `triangle.incoming.contract` | 7.22 | 143.7 | 19.9 |
| `msa-column-attention.flash` | 11.20 | 221.5 | 19.8 |
| `triangle-attention-starting.flash` | 17.81 | 287.5 | **16.1** |
| `triangle-attention-ending.flash` | 17.97 | 287.5 | **16.0** |
| `msa-row-attention.flash` | 22.26 | 356.8 | **16.0** |
| these nine | 118.10 | 2597.4 | 22.0 |

They are 60% of a 195.6 ms block. **The four flash attentions are 69.2 ms of it
- 35% of the block - at 16 to 20 TFLOP/s, against 39.7 for the outer product
mean's contraction in the same block on the same units.** That 39.7 is also the
best this card has ever given a real kernel here (docs/A100.md's staged GEMM
reaches 39.3), so it is a rate this port is known to be able to hit, not an
aspiration.

Priced: if all nine ran at 39.7 they would take 65.4 ms instead of 118.1, which
is 27% off the block - main stack 9.22 -> 6.7 s and a fold of about **8.5 s,
1.61x JAX**. So the largest single identified gap is the flash attention's
efficiency, and closing it completely does not reach JAX.

🔴 **AND THE REMAINDER IS PRECISION - BUT NOT AS AN ACTIVATION-PACKING JOB, AND
THE PARAGRAPH THAT USED TO STAND HERE WAS WRONG TWICE.** It said all 79
activation allocations in `src/evoformer/`, `src/model/` and `src/multimer/` are
f32 because "a grep for a storage argument returns zero". The grep matched a
LITERAL `"f16"`, and the storage is passed as a variable: parsing the calls
instead, **16 of 102 pass a storage argument**, and they are the ones that
matter - `attention.query`/`key`/`value`/`gate` are already f16 through
`projectedStorage`, and so are the attention's normalised input and the
transition's hidden. The packing this port could cheaply take, it took.

And the rest would not buy speed, which is the second error. **None of the
block's large kernels is anywhere near bandwidth-bound.** Bytes moved over
measured milliseconds, against this card's ~1555 GB/s:

| kernel | ms | GB | GB/s | of HBM |
|---|---:|---:|---:|---:|
| `triangle.outgoing.contract` | 6.73 | 1.05 | 155 | 10.0% |
| `msa-row-attention.project` | 8.45 | 1.30 | 154 | 9.9% |
| `msa-column-attention.flash` | 11.20 | 1.08 | 97 | 6.2% |
| `opm.contract` | 17.98 | 1.19 | 66 | 4.3% |
| `msa-row-attention.flash` | 22.26 | 1.10 | 50 | 3.2% |
| `triangle-attention-starting.flash` | 17.81 | 0.88 | 50 | 3.2% |

The heaviest reaches a tenth of the bus and the flash attentions a thirtieth, so
**halving an operand's bytes cannot move a kernel that is not waiting on
them.** The flash attentions sit at 16 TFLOP/s for the reason docs/A100.md
already gives - occupancy capped by workgroup memory - and that is a tile and
register problem, not a storage one.

🔴 **AND THE TRIANGLE'S SCRATCH IS THE ONE PLACE PACKING IS ALREADY MEASURED,
AS A LOSS.** `createTriangleShaders` takes `{normalized, hidden, ab}` and AF2's
block passes none of it, which reads like an omission. It is not:
docs/PERF.md's bisection of the SAME shaders in AF3 has `a` and `b` costing
**3x** the error - they are multiplied against each other, so their rounding
squares - `normalized` 1.6x, and `hidden` nothing measurable. That is why
`PAIR_SCRATCH_STORAGE` is exported and unused and all four AF3 stacks take
`UNPACKED_PAIR_SCRATCH`. AF2 would be repeating a measured experiment.

What activation packing WOULD buy here is memory and reach, not time: the fold's
peak is 6803 MiB at 825 residues, and a packed pair is what moves the
28-label binding ceiling from 2,047 residues to 2,896. That is a capacity
argument and belongs with the ceiling section, not this one.

So the honest remainder against JAX is arithmetic: 39.7 TFLOP/s is 13% of the
310.9 this card's units issue at, JAX runs bf16 tensor cores end to end, and the
largest single kernel family here is occupancy-bound at 16.

Caveat on the ratio itself: **the 5.27 s is the recorded JAX number and was not
re-run** - the AF2-vendored branch it came from is not on this box. It is a
bf16, Triton-flash-attention, whole-graph-XLA run of the same model.
