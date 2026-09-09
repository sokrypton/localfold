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

