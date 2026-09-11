# Kernels, memory and precision, measured on this M2

Findings that are about the machine rather than about a model: what the device
can do, where the memory goes, what half precision is worth where, and which
optimisations were tried and did not survive. See CLAUDE.md's "Measuring,
without fooling yourself" before trusting any number here.

🔴 **AND A DIFFERENTIAL CHECKER THAT BUILDS ITS OWN KERNEL TESTS WHATEVER IT
BUILT.** Four of them did this session, each found the same way: a shipped path
learned to pick between an f32 and an f16 kernel, the checker went on
constructing the f32 one from a module constant, and the arm labelled "f32" was
either testing a kernel nothing runs or - once - silently testing the f16 one
and failing. Ask the selection function for the kernel, take the precision as
an axis, and hold each arm to the bound its own arithmetic implies. Raising one
bound to cover both stops the f32 path being checked at all.

## Where the memory goes, and what f16 weights are worth where

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

## This device's ceilings, and the biggest kernel in a trunk

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

## Packed activations, the pair scratch, and buffers that can be one buffer

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

## Uploading weights, and what the host was doing while the device waited

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

🔴 **AND `memorySnapshot`'s `byLabel` IS CUMULATIVE, WHICH IS THE WRONG
QUESTION.** It sums every allocation a label ever made, so a scratch tensor
taken and returned once a block reads as forty-eight times its size - that is
what CHURNS. `peakByLabel` is what was on the device when it was fullest and
its rows sum to `peakBytes`; that is what says which tensor to attack, and it
is what said ten tensors of 29.5 MiB were 295 MiB of a 552 MiB fold.
`tools/gpu/fold-af2.js` prints both.

## Where the other two models' memory is, measured rather than assumed

🔴 **AF3 IS DONE, AND IT IS ONE TENSOR.** A 76-token fold peaks at **486.4
MiB** and `difftx.block.resident` is **378.2 of it - 78%**, across 24 blocks.
That is already the f16 form (it was 756 MiB in f32) and it has to stay
resident: the diffusion transformer is called once per sampler step, 50 to 200
times a fold, so streaming it per block would re-upload 378 MiB per step and
decoding it from int5 per step is 119 ms x 50 against a 5 s fold. Everything
else is under 15 MiB. There is no second thing to take.

🔴 **AND AF2 IS FLAT, WHICH IS A DIFFERENT KIND OF DONE.** 59 residues at 512
MSA rows with a recycle peaks at **365.2 MiB** - this file's own recorded
figure - and the largest row is `embed.msa` at **16%**, with a long tail of
attention tensors at 4% each. 128 rows reads 147.1 MiB, also the recorded
figure. A flat profile has no single thing to attack, which is what the
aliasing and packing work already recorded here left behind.

🔴 **THE ONE CANDIDATE LEFT IS `embed.msa` UNDER RECYCLING, AND IT IS 8%.**
Two live allocations carry that label at 512 rows with a recycle and one at
128 rows without - the previous pass's MSA is the embedder's INPUT while the
new one is its output. The previous is read exactly once, into
`embed.previous-msa-normalized`, by the first dispatch of the encoder; every
later dispatch writes the new one. So they could be one buffer, ordered within
the encoder, for 29.5 MiB of 365. It is not taken because it is an ownership
change through TWO recycle loops (`src/evoformer/input-embedder.js` and
`src/multimer/input-embedder.js`) for 8%, and `fold-af2.js`'s checksum
(-2047044 at 512 rows and one recycle) is what would have to gate it.

🔴 **SO THE EF2 RESULT DOES NOT GENERALISE, AND THE REASON IS INSTRUCTIVE.**
EF2-fast gave up 45% because nobody had ever read its peak by label - its fold
tool printed a total and nothing else, where AF3's and AF2's have printed
`peakByLabel` for a long time. The win was not that EF2 was written worse; it
was that it had never been looked at with the instrument the other two had.
**Check whether a thing has been measured before concluding it is optimal.**

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


## What would go the other way, and the seven things that would not

The section above is what was taken FROM `martin-steinegger/alphafold2-webgpu`.
This is the reverse question, asked after the A100 work took AF2 from 150 s to
15.7 at 825 residues: which of those gains is something they do not have? Read
from their source at the divergence point, checked one by one.

🔴 **SEVEN OF THEM ARE THINGS THEY ALREADY DO, AND THAT IS THE FINDING.** The
AF2 campaign's headline numbers were mostly this port recovering from its own
regressions, not passing theirs:

| what this port gained | their source |
|---|---|
| the PAE softmax taken twice, 1.19 -> 0.51 s | their production path reduces PAE and pTM on the GPU in one pass; the double softmax is only their non-reduced fallback |
| the pair bias read once per head, 15.4 -> 2.4 ms | `createAttentionPairBiasShader` accumulates every head from ONE read of the pair row |
| a folded grid a shader ignored, which collapsed every fold past ~500 residues | their bias shader is `id.x + id.y * GRID_WIDTH * 64u` - correct |
| the running output in registers, not workgroup memory | theirs is in registers (`out_${j}`) |
| the two-lane row reduction by `subgroupShuffleXor(v, 1u)` | they have it, at lines 404 and 411 |
| the outer product mean through a general tiled GEMM | that is their architecture: one calibrated GEMM serves every dense projection |
| releasing scratch when it dies rather than at the end | `releaseScratch` in `evoformer/execution-scratch.ts` |

🔴 **AND THEIR KEY TILE WAS SWEPT, WITH OUR LESSON ALREADY IN IT.** `KEY_TILE`
is 32 and the comment above it records one, two, three and four units measured
**in the stack** - 51.0 -> 47.5 ms at 800 residues, 69.6 and 70.4 past the
occupancy cliff - and ends "Measure this against the stack, not against a
standalone dispatch", which is docs/A100.md's own rule reached independently. It
even records that an earlier note said the opposite on a microbenchmark whose
tensors fit in L2.

### What is left, and it is measurements of THEIR port rather than kernels

🔴 **THEIR COLD START IS 0.92 TO 4.10 SECONDS AND GROWS WITH THE SHAPE; THIS
PORT'S IS 0.394 TO 0.527 AND IS FLAT.** Over the twelve shapes in docs/AF2.md's
grid - 59 to 825 residues, two alignment depths, a 22x range of work - cold
minus warm:

| | smallest | largest |
|---|---:|---:|
| LocalFold | 0.394 s | 0.527 |
| alphafold2-webgpu | 0.920 | 4.100 |

At 256 residues and a shallow alignment that is **3.33x on the number a user
folding one sequence actually experiences**. It is worth telling them because
their own harness cannot see it: `bench/bench825.js` reports the MINIMUM OF THE
LAST TWO of three passes, so the cold pass is measured and then discarded. The
likely cost is what buys them their warm figure - `planMonomerDevice` sizing
limits from the shape, the `fitScratchBudgetScale` search, and a wider set of
specialised pipelines to compile.

🔴 **AND BELOW 256 RESIDUES THEIR FOLD DOES NOT NOTICE THE ALIGNMENT.** Going
from 128 clustered / 256 extra to 512 / 1024 at the same length:

| | 59 | 128 | 256 | 400 | 600 | 825 |
|---|---:|---:|---:|---:|---:|---:|
| ours | 1.05 | **1.50** | 2.10 | 1.86 | 1.67 | 1.54 |
| theirs | 1.01 | **1.03** | 1.73 | 1.85 | 1.65 | 1.60 |

From 256 up the two agree within 8%, so the extra-MSA stack costs them what it
costs us. At 128 residues four times the rows costs this port 1.50x and theirs
1.03x - their short fold is still dominated by something that does not scale
with depth. Same cause as the row above, from another angle.

🔴 **AND ONE AXIS OF THEIR MATRIX FLASH ATTENTION IS UNSWEPT.** `SUBGROUPS = 2`
is a module constant; the sweep in the comment varies the KEY tile at that fixed
value. This port swept both and 4x32 beat 2x32 by 341.0 -> 324.6 ms on an
825-residue block, with the five geometries ranking exactly by workgroup bytes a
lane. Their kernel holds more in workgroup memory than this one does - the three
row statistics are still arrays there, about 6 bytes a lane - so their optimum
need not be ours, and on an L40S it need not be either. It is one cheap
experiment with a method already written down.

**Nothing here is code.** The trees diverged on 2026-08-30, theirs is
TypeScript, and the two findings worth passing on are measurements of their own
port that their harness structurally hides.

## What an unrecognised GPU costs, which is 1.5x and was never measured

The section above says their runtime probe is the thing this port's device
profile is missing, and that a two-entry lookup table is a bet that every user
runs hardware somebody here owns. That was an argument. This is the number.

🔴 **NOTHING IN THIS REPOSITORY COULD MEASURE IT, BECAUSE BOTH MACHINES THAT RUN
IT HAVE PRIORS.** `PRIORS` has `ampere` and `metal-3`; every other GPU in the
world takes `DEFAULT_TUNING`, which is one M2's answers. `ignoreDevicePrior`
makes a machine that HAS a prior answer as one that does not, and `--no-prior`
on any GPU tool reaches it. On this A100, prior against no prior:

| | with the prior | unrecognised | |
|---|---:|---:|---:|
| AF2, 400 residues, 512/1024, warm | 4.417 s | 6.760 | **1.53x** |
| ESMFold2 trunk, 300 tokens | 661 ms | 1014 | **1.53x** |
| AlphaFold 3, a whole 68-token fold | 3.5 s | 4.4 | 1.26x |
| AF3 trunk alone, 400 tokens, 8 blocks | 1450 ms | 1504 | 1.04x |

🔴 **AND THE TRUNK-ONLY ROW IS THE INTERESTING ONE.** AF3's trunk barely moves
because eighteen of the ampere prior's twenty-five knobs are diffusion-side or
AF2-side; the trunk's own kernels are close to their defaults. So the 1.5x is
not spread evenly over the model - it is concentrated in the parts that were
tuned, which is exactly where a probe would have to look and exactly where a
wrong default hurts.

**The peak moves too, and not always the wrong way.** The unrecognised ESMFold2
trunk peaks at 445.8 MiB against the prior's 517.8, because
`pairTransitionSplit` is one of the knobs it does not get: it is slower and
smaller. A probe that optimises time alone would take that memory without
asking, which is the trade `TRANSITION_SPLIT_MIN_CHANNELS` exists to make
deliberately.

### Which knobs carry it: three on AF2, two on ESMFold2, and all of them one question

`--no-prior=a,b` restores exactly the named knobs AT THE PRIOR'S OWN VALUES and
drops the rest, which is what lets a sweep price one knob without spelling an
object on a command line - `diffusionTokenTile` is `{below, atOrAbove,
crossover}` and `--tune` splits its argument on commas.

**AF2, 400 residues, 512/1024, warm.** Baseline 6.752 s, all 25 knobs 4.430:

| knob restored alone | warm | saves | % of the gap |
|---|---:|---:|---:|
| `opmMatrixContract` | 5.891 | 0.861 | **37%** |
| `matrixLinear` | 6.014 | 0.738 | **32%** |
| `attentionMatrix` + tile | 6.085 | 0.667 | **29%** |
| `opmProjectOutputPairs` | 6.568 | 0.184 | 8% |
| `attentionGroup` | 6.578 | 0.174 | 7% |
| `attentionVectorScore` | 6.697 | 0.055 | 2% |
| `trianglePairProjectTile` | 6.710 | 0.042 | 2% |
| `linearTallTile` | 6.759 | -0.007 | -0% |
| `keepTrunkWeights` | 6.764 | -0.012 | -1% |
| `transitionThreadTarget` | 6.774 | -0.022 | -1% |

**ESMFold2 trunk, 300 tokens.** Baseline 1019.3 ms, all 25 knobs 655.5:

| restored | ms | saves | % of the gap | peak MiB |
|---|---:|---:|---:|---:|
| the three below plus `stagedMatrixBlock` | 654.4 | 364.9 | **100%** | 517.8 |
| `pairTransitionSplit` + `triangleProjectMatrix` | 701.4 | 317.9 | 87% | 517.8 |
| `pairTransitionSplit` (+chunk bytes) | 769.0 | 250.3 | 69% | 517.8 |
| `triangleProjectMatrix` | 945.3 | 74.0 | 20% | 445.8 |
| `trianglePairProjectTile` | 1017.6 | 1.7 | 0% | 445.8 |
| `transitionThreadTarget` | 1017.9 | 1.4 | 0% | 445.8 |
| `gridAttendMatrix` + tile | 1019.1 | 0.2 | 0% | 445.8 |
| `matrixLinear` | 1020.9 | -1.6 | -0% | 445.8 |
| `stagedMatrixBlock` | 1021.8 | -2.5 | -1% | 445.8 |

🔴 **THE ZEROS ARE TRUSTWORTHY BECAUSE ONE OF THEM HAS TO BE ZERO.**
`gridAttendMatrix` saves 0.2 ms on ESMFold2 and provably cannot do anything -
an ESMFold2 block is a pairformer block with both grid attentions removed. A
sweep that reported a number there would be measuring its own noise.

🔴 **AND `stagedMatrixBlock` IS -2.5 ms ALONE AND 47 ms IN COMBINATION**, which
is the finding that decides what a probe can be. Restored by itself it is the
block geometry for kernels that are switched off, so it correctly measures
nothing; added to the three that are on it takes 701.4 to 654.4. AF2 says the
same thing from the other side - its ten individual savings sum to 2.680 s
against a joint gap of 2.322, **115%**, because three kernels contend for the
same units and the same occupancy.

**So a probe cannot be "measure each knob, keep the winners".** It would set
`stagedMatrixBlock` to null on a correct measurement, and it would over-credit
AF2's three matrix knobs by 15%. It has to be ordered - settle the matrix-unit
question first, then tune the block for whatever turned on - or measure a few
whole configurations rather than knobs.

**The search is small, though.** On both models every knob that carries the gap
is the same question: `opmMatrixContract`, `matrixLinear`, `attentionMatrix`,
`pairTransitionSplit`, `triangleProjectMatrix` are all "should this kernel use
the subgroup matrix units", and `stagedMatrixBlock` is "at what geometry". Two
decisions, not twenty-five - and the same two that upstream's single attention
probe is answering.

### And a standalone GEMM cannot answer either of them

The obvious cheap probe follows from that paragraph: if every knob that carries
the gap asks "do the matrix units pay for this kernel", time one matrix GEMM
against one vector GEMM at each kernel's shape and read off the answers.
`bench-evoformer-linear.js` already does exactly that, so it costs nothing to
check the prediction against the verdicts the sweeps above measured in situ.
It gets two of five WRONG, and in opposite directions:

| kernel | shape | standalone | in situ | |
|---|---|---:|---:|---|
| `attentionMatrix` | `headdim` | **1.00x** | **1.69x** | **miss** |
| `matrixLinear` | `qkvg` | **1.49x** | **0.93x** | **miss** |
| `opmMatrixContract` | `opmcontract` | 1.58x | 2.26x | right, understates |
| `triangleProjectMatrix` | `pair` | 1.20x | 1.53x | right, understates |
| `pairTransitionSplit` | `ef2wide`/`ef2down` | 1.51x/1.65x | 2.89x | right, understates |

🔴 **AND OPPOSITE DIRECTIONS IS THE FAILURE A MARGIN CANNOT REPAIR.** A
threshold set high enough to reject `qkvg`'s false 1.49x also rejects the
attention's true 1.69x, whose standalone reading is 1.00x. There is no cutoff
that gets both.

Each miss has a cause, and both are the same cause seen twice - **the standalone
bench models a kernel that is not the one that ships**:

- **The attention is not a GEMM.** A GEMM at K = 32 amortises its panel over two
  k steps; the flash kernel stages the query tile ONCE and reuses it across the
  whole key sequence. The reuse is in the loop, not in K. This is the third time
  in this repository that a standalone GEMM at `head_dim` has been read as a
  verdict on the attention, and the third time it was wrong.
- **The projection is four GEMMs, already fused.** `qkvg` times ONE matrix. The
  shipped kernel reads the source once and writes q, k, v and the gate from it,
  so the thing a matrix arm would replace is ~4x cheaper than the bench's model
  of it - which is why the standalone arm looks like a win and loses in place.

The three it gets right it understates by 1.3-1.9x, for the ordinary reason: in
situ these kernels also move occupancy, buffer traffic and what the kernels
around them contend for, and none of that is in a single dispatch.

So the probe cannot be a synthetic GEMM at a representative shape. It is the
actual kernel or nothing - which is what upstream's attention probe does, and it
is why theirs costs about a second.

## The atom stack was sized by the padded grid, and it was mostly padding

🔴 **`subsets` COUNTED (token, slot) CELLS WHERE THE AXIS IT INDEXES IS THE
COMPACTED LIST OF REAL ATOMS.** The atom encoder and decoder run once per
diffusion step and every buffer they hold is `subsets * 32 * 128` wide, so a
subset that holds no atom still allocates a full attention and dispatches over
it. What that cost, by shape:

| | real atoms | subsets before | needed |
|---|---|---|---|
| AlphaFold 3, 68 residues | 574 | 51 | **18** |
| OpenDDE, the same 68 residues | 574 | 98 | **18** |
| AlphaFold 3, 250 residues | 2101 | 188 | **66** |
| OpenDDE, 250 residues | 2101 | 358 | **66** |

OpenDDE is worse by construction: the structural layout is the same atoms in
twice the tokens, so it doubled a count that was already 2.8x too big.

🔴 **AND THE WIN IS IN CHURN AND TIME, NOT IN THE PEAK, EXCEPT WHERE THE TRUNK
IS SMALL.** Both folds are bit-identical after the change - AlphaFold 3 mean
pLDDT 85.93504804019729, OpenDDE RMSD 1.676 and TM 0.8839 - and:

| | peak | wall |
|---|---|---|
| AlphaFold 3, 68 residues | 499.1 -> **458.8 MB** (-8.1%) | 4.69 -> 4.48 s |
| OpenDDE, 6MRR, 64 steps | 647.6 -> **553.2 MiB** (-14.6%) | 22.1 -> 19.6 s |
| OpenDDE, 200 residues, 32 steps | 1507.4 -> **1507.4 MiB** (nothing) | 149.1 -> **130.6 s** |

🔴 **THE 200-RESIDUE PEAK DOES NOT MOVE, AND THAT IS THE USEFUL PART OF THE
RESULT.** Its `peakByLabel` is `af3-block.scratch` at 1132 MB plus
`af3-block.pair` at 226 - the TRUNK, which has finished before the diffusion
starts. The atom stack is nowhere near the high-water mark at that length; it
only reaches it at 68 residues, where the trunk is small. So the 250-residue
ceiling is the pair scratch and nothing about the atom stack was ever going to
move it.

What does move is everything the diffusion allocates and re-allocates fifty or
two hundred times. Cumulative bytes at 200 residues: `atom.k` and `atom.v`
613.8 MB each -> 103.8, `dec.k`/`dec.v` 604.0 -> 100.7, `atom.pair` 114.8 ->
25.2, `atom.logits` 86.1 -> 18.9. The wall time falls 11-12% on two different
shapes, which is one run each on a machine that drifts by up to 3.2x - so read
the direction and the mechanism, not the digits.

## The A100 branch, measured back on this M2

The A100 campaign (docs/A100.md) is a large branch tuned on a card with ten
times the cores, and the question it leaves open is what it does to the machine
this file is about. Every number here is this M2, `a100` against `main` at
`d19e2a7`, folds run one after another rather than side by side.

🔴 **THE ANSWER IS "NOTHING", AND THAT IS THE PROFILE WORKING AS DESIGNED.**
`DEFAULT_TUNING` is every knob at its pre-branch value and the priors are keyed
on architecture, so a part with no entry computes what it always did. Checked
rather than assumed, and checked on COORDINATES because this repository has
already recorded `meanPlddt` matching to sixteen digits across an arm that moved
thirty-three atoms:

| model | a100 against main, on this M2 |
|---|---|
| AF3, `fold.js` | **bit identical** - 574 atoms, 0 moved, sampled and denoised |
| ESMFold2, `fold-esmfold2.js` | **bit identical** - `alphaCarbons` equal, certainty to 16 digits |
| AF3 peak device memory | 476.0 -> 476.1 MiB |
| CPU suite | 954 pass |

🔴 **AND AF2 GOT 3.25x FASTER AND 40% SMALLER WITHOUT A KNOB, BECAUSE THE THING
THAT MOVED WAS A DELETED CONSTANT.** The outer product mean's 64 MiB cap stopped
the fast contraction at 128 residues, so every protein longer than that fell to
`opm.accumulate`. At 200 residues and 128 rows:

| | main | a100 |
|---|---|---|
| fold | 26.56 s | **8.18 s** |
| peak device memory | 779.1 MiB | **465.4 MiB** |
| `opm.intermediate` | 103.13 MiB x2 | 64 MiB x1 |

The memory falls because the fallback was the expensive path in both senses: it
allocated a 103 MiB tile buffer twice, where the blocked contraction holds one
64 MiB working set at any length. `check-opm-paths.js --length=400
--sequences=512 --cz=128` agrees at depth - 14.7x over the tiled arm, the
blocked arm at relRMS exactly 0, and the f16 arm finite at the 512 rows upstream
records overflowing.

🔴 **THE ONE PATH THAT MOVED AND HAS NO GATE ANYWHERE IS THE MULTIMER.**
`fold-af2.js --family=multimer --chains=30,29` shifts systematically, and a seed
sweep is what says "systematically" rather than "noisily":

| seed | main | a100 | delta |
|---|---|---|---|
| 0 | 51.214 | 54.265 | +3.05 |
| 1 | 50.410 | 52.668 | +2.26 |
| 2 | 50.995 | 54.067 | +3.07 |

Main's own seed-to-seed spread is 0.80, the gap is 2.3-3.1 and one-directional,
and pTM and ipTM move with it. The monomer at the same length moves 0.068, so
this is not shared-kernel drift. Every per-kernel differential gate passes here
(transition 0.0017 against a 0.004 bound, attention 0.0012, OPM 2.1e-7, triangle
1.1e-7) and the multimer template term matches its JAX reference at 6.5e-5, so
what is left is summation reordering - the staged OPM output projection, the
rewritten global attention, the hoisted pair bias - accumulating over 48 blocks
on a low-confidence synthetic complex. It is unresolved, and it is unresolved
EVERYWHERE: docs/A100.md records that box has no multimer weights, so
`--family=multimer` cannot run there at all. Nothing has ever gated this path
end to end. `check-evoformer-stack.js` cannot settle it either - the fixture in
this checkout has the features and not `stackInputMsa`, on both branches.

## Two A100 knobs, asked here, and one of them is a win

🔴 **`keepTrunkWeights` DOES NOT CARRY, AGAINST docs/A100.md's OWN PREDICTION.**
That file reasons "most of the win is not the bus, so it should carry to an M2"
and adds `--keep-weights=on` so an M2 can settle it in one command. Settled,
with `fold.js --folds=3`:

| arm | warm folds | peak |
|---|---|---|
| `off`, the shipped default | 4.018, 4.011 s | 476.1 MiB |
| `on` | 4.153, 4.024 s | **855.9 MiB** |

Nothing, for +380 MiB - 1.8x the peak. The A100's 1.95x is the PCIe transfer and
unified memory has no transfer to avoid. `null` is right here and is now
measured rather than inherited.

🔴 **`attentionMatrix` DOES NOT COMPILE HERE, AND FAILS LOUDLY.**
`--tune=attentionMatrix=true` throws `GPUPipelineError: the MSL backend only
supports 8x8 subgroup matrices`. The `ampere` prior wants a 4x32 tile and Metal
will not have it. This is the good failure - a pipeline that refuses, not a
kernel that computes something else - and it is the concrete case behind
`matrixLinear`'s note that an M2 reporting matrix configs is not an M2 that
wants an unmeasured kernel.

🔴 **`opmMatrixContract` IS A WIN HERE AND IS NOW A `metal-3` PRIOR.** The one
kernel whose shape suits 8x8x8 units: a plain GEMM with the model's deepest K.
Swept with `profile-af2-block.js --sweep`, which interleaves its arms:

| length x rows | block off | block on | speedup | `opm.project-output` |
|---|---|---|---|---|
| 59 x 128 | 22.87 ms | 21.55 | 1.06x | 2.017 -> 0.927 |
| 128 x 64 | 49.50 | 44.45 | 1.11x | 9.023 -> 4.171 |
| 200 x 128 | 152.35 | 136.52 | 1.12x | 22.272 -> 10.120 |
| 400 x 256 | 766.50 | 665.71 | 1.15x | 94.242 -> 40.115 |

Monotone across a 34x range of block cost, never inverting. Most of it is the
output projection, which follows the contraction unless `opmMatrixOutput` turns
it off; the contraction itself is 1.32x at 200x128. End to end a 200-residue
fold goes 8135 -> 7446 ms with the prior live, CA-CA gate holding, mean pLDDT
moving 0.035 and peak memory 465.4 -> 464.8 MiB.

🔴 **AND IT IS ONE M2.** `metal-3` spans parts with very different core counts,
and this repository's own `attentionQueriesPerLane` spread - M2 0.21x, M4 Pro
0.45x, GB10 1.17-1.42x - is the standing warning that the badge does not predict
the number. Re-sweep before trusting it on another Apple part.

## A phone, for the first time: a Pixel 9 on Valhall

`PRIORS` has two entries. Every other GPU in the world takes `DEFAULT_TUNING`,
which CLAUDE.md prices at 1.5x on AF2 and ESMFold2 - and the device class most
users actually hold had never run a kernel from this repository. It can now:
`LOCALFOLD_GPU_ANDROID=1` on any GPU tool runs it in Chrome on a USB-attached
phone, because this harness was already one localhost origin and `adb reverse`
is the whole of the port.

Pixel 9, Tensor G4, `arm` / `valhall`, Android 16, Chrome 152:

| | M2 | Pixel 9 |
|---|---|---|
| `shader-f16` | yes | **yes** |
| `timestamp-query` | yes | yes |
| subgroup min/max | 32 / 32 | **16 / 16**, so `supportsSubgroups` is false |
| subgroup matrix | 8x8x8 | **none** |
| AF2 59-residue fold | 1.25 s | **8.43 s**, peak 386.9 MiB, `caca ok` |

pLDDT 57.267 against the M2's 57.213, and a different checksum, which is what
two GPUs with different f16 reduction orders must give. Nothing fell back,
nothing refused, no OOM. Every matrix-unit knob - all seven of `ampere`'s and
`metal-3`'s `opmMatrixContract` - is irrelevant on a part with no matrix units,
so whatever wins here will be different wins.

### 🔴 `probe-tuning`'s recommendations DO NOT SURVIVE THE STACK

It disagrees with the defaults on three knobs and measures `linearTallTile` at
**2.008x**, bit-exact, relRMS 0 - a tile, not a reordering. In real work:

| arm | 59x128 fold, `mainStack` x3 | 200x256 block, wall x2 |
|---|---|---|
| default | 6.30 / 6.29 / 6.27 s | 1474.9 / 1543.1 ms |
| all three recommended | 6.36 / 6.36 / 6.34 s | - |
| `linearTallTile` alone | - | 1472.9 / 1500.9 ms |
| `attentionGroup=4` + `attentionVectorScore` | - | 1476.7 / 1548.9 ms |

The tall tile is worth between 0.1% and 2.7%, inside the drift; the attention
pair is consistently worse; all three together are a 1% LOSS. The knobs are
live - the checksum moves -1877818 to -1891033 and pLDDT 57.267 to 57.241,
which is the two attention knobs reordering sums. So **no `valhall` prior is
written**, because a 2x microbenchmark the stack values at 0% is exactly what
this file's own rule says not to believe. Round 2 is slower than round 1 on
every arm: a phone throttles DOWNWARD monotonically, which is a nastier shape
than this M2's random 3.2x drift, and arms must be alternated within a round.

### 🔴 AND GPU TIMESTAMPS ON VALHALL ARE INFLATED 21x

| | timestamp `blockMs` | the tool's own `wallPerBlockMs` | ratio |
|---|---:|---:|---:|
| M2 | 22.01 | 28.5 | 0.77 |
| Pixel 9 | 3542.03 | 168 | **21.08** |

A ratio above 1 claims more GPU time than wall time elapsed, which is
impossible; 0.77 is what a healthy profiler looks like, the summed passes being
a little less than wall. So `--sweep`, per-kernel attribution and everything
downstream of `beginTimestampProfile` are unusable on this GPU, and only
whole-fold wall clock can be trusted - which is why the table above is wall.
`profile-af2-block.js` has printed `blockMs` and `wallPerBlockMs` side by side
since it was written and nothing compared them; on the two machines that had
priors the ratio was always ~0.8, so it never mattered. **A number a tool prints
is not a gate until something fails on it**, again.

### What the A100 branch was worth here

The Pixel ran both sides of the merge, with the harness patch held in place
across the checkout - the first arm of this was run wrong, with `git checkout`
taking the harness with it, so it silently measured the Mac and reproduced the
M2's numbers exactly. Check which machine an arm ran on.

| | pre-merge `c881283` | merged |
|---|---:|---:|
| AF2 wall | 8685 ms | 8223 / **7814 ms** |
| AF2 weight load | 575 ms | 146 / **215 ms** |
| AF2 `mainStack` | 6.43 s | 6.41 / 6.28 s |
| AF2 checksum | -1877818 | -1877818 |
| ESMFold2 whole fold | 12.35 s | **9.41 s** (1.31x) |
| ESMFold2 language model | 4316 ms | **2014 ms** (2.14x) |
| ESMFold2 host peak | 272.8 MiB | **35.8 MiB** |
| ESMFold2 PDB | `4b9a0176…` | `4b9a0176…` |

Both folds bit-identical across the merge. AF2's win is 5-10% and all of it is
the weight load; its compute stack does not move. ESMFold2 is 1.31x where the
M2 got 1.97x, and the phone's share is concentrated in the ESM-C tower's weight
handling rather than the trunk, whose `trunk 0` went the wrong way, 1062 ->
1235 ms. None of it needed matrix units, which is why it carried to a part that
has none.

🔴 **AND THE HOST-WORK FINDING IS UNTESTED HERE.** `shader-source-cache.js` was
kept on the A100 explicitly because 18 MiB of generated-and-discarded WGSL a
fold "is not free on a phone", having been measured at 86 ms and entirely
hidden behind that card's GPU. On this Pixel an AF2 fold is 7.8-8.7 s of wall
against 6.3 s of `mainStack`, so 1.5-2 s a fold sits outside the stack and that
is where such a cost would live. Nobody has measured it there yet.

## Two cache keys, one shader: 29% of a first fold's pipelines were duplicates

🔴 **269 PIPELINES, 191 DISTINCT SHADERS.** `probe-compiles.js` reports
`distinctSources` now - the number of unique WGSL texts against the number of
modules made - and OpenDDE's first fold compiled 78 kernels it had already
compiled under a different cache key. Every model does it:

| | pipelines before | after | shared |
|---|---:|---:|---:|
| OpenDDE | 269 | **191** | 78 (29%) |
| AlphaFold 3 | 223 | **156** | 67 (30%) |
| AF2 monomer | 96 | **73** | 23 (24%) |
| ESMFold2 | 95 | **82** | 13 (14%) |

The keys differ for good reasons - they carry a token count, a direction, a
geometry, a precision - and two different shapes can still generate
character-for-character the same kernel. Nothing had ever compared the SOURCES,
only the keys.

`ComputePipelineCache` keeps a second index by `entryPoint + source` and hands
back the pipeline it already built. Alternating arms, three pairs, OpenDDE at
16 steps:

| | shared | plain |
|---|---:|---:|
| whole fold | 2404 / 2403 / 2382 ms | 2540 / 2552 / 2603 |

**~160 ms, and every pair separates.** pLDDT 92.0505 and peak 1823.9 MiB in all
six. On the page, cold: monomer 3418 -> **3141 ms**, AF3 4896 -> 4823, OpenDDE's
status line crosses from "in 4 s" to "in 3 s".

🔴 **THE SPAN MOVES LESS THAN THE WORK, WHICH IS WHAT A SATURATED POOL LOOKS
LIKE.** `sumMs` falls 45.0 s to 33.2 and `busyMs` only 1765 ms to 1522: removing
29% of the compiles shortens the union of their intervals by 14%, because the
pool was never idle. That is also why this shows up in the FOLD's time rather
than only in the compile's - the work removed was competing with everything else
the host was doing.

🔴 **THE SOURCE IS THE KEY, NOT A HASH OF IT.** A hash collision here would hand
a caller somebody else's kernel; the texts are already retained by the source
memo, so keying on them costs nothing new. And it is safe only because the
layout is `auto` and therefore derived from the source: two pipelines built from
identical WGSL with the same entry point have the same bind group layouts by
construction. The label differs and is cosmetic - profile.js times labelled
compute PASSES, not pipelines.

Where the duplicates come from is worth knowing before trying to remove them at
the source: OpenDDE runs its pairformer at TWO token counts, 68 residues and 130
structural tokens, and `af3-block` is 111 of the 269 with 86% of the compile
work. Many of those kernels do not actually depend on the token count, so they
generate the same text under two keys. Deduplicating them here is a cure for the
symptom that costs nothing; making the key honest would be a cure for the cause,
and would need each kernel checked for whether the count is a loop bound - which
docs/AF2.md prices at 4.3x when it becomes a runtime one.

## The weight upload is `writeBuffer`-bound, and both ways round it lose

A first fold moves the whole bundle through `device.queue.writeBuffer`, and
nothing had ever measured that call. `blockUploadStats.stagingMs` timed the
staging assembly and the upload TOGETHER, so it could not: a change that traded
one for the other moved neither number. Split into `copyMs` (the memcpy into
the shared scratch) and `writeCallMs` (the driver call), on a two-fold run:

| | bytes staged | `copyMs` | `writeCallMs` |
|---|---:|---:|---:|
| OpenDDE | 536 MiB | 57 | **251** |
| ESMFold2 | 350 MiB | 31 | **179** |
| AF2 | 92 MiB | 8 | **48** |

So the host memcpy the code is written around is a fifth of the cost and
`writeBuffer` is the rest - about 3.0 GB/s, and 250 ms of an OpenDDE first fold,
which is more than every shader compile in it.

🔴 **AND `writeBuffer` IS THE SLOWEST ROUTE THIS DEVICE OFFERS.**
`tools/gpu/probe-upload-path.js` moves 64 MiB into a storage buffer and waits
for it to land, arms interleaved, minimum of five:

| route | GB/s |
|---|---:|
| pooled `MAP_WRITE` buffer, 1 MiB at a time, copied on the device | **6.85** |
| `writeBuffer`, 1 MiB at a time | 2.15 |
| `writeBuffer`, all 64 MiB in one call | 1.22 |
| a fresh `mappedAtCreation` buffer each time | 1.07 |

Two things there are worth keeping even though what follows failed. A big
`writeBuffer` is **slower than the same bytes in 1 MiB pieces** - the opposite
of the usual advice - and the mapped route is genuinely 3-5x.

### 🔴 BUT A POOL THAT RECYCLES ON DEVICE COMPLETION CANNOT FEED A LOOP THAT RUNS AHEAD OF THE DEVICE

Wired into `runBlockUpload` behind a rule that can never add a wait - use a
mapped buffer only if one is already mapped, else take the old path - it fired
**19 times out of 487**. The upload loop submits hundreds of packs without
draining, which is deliberate and is what makes it fast; a buffer returns to the
pool only after the submit that read it completes, so the pool is empty for the
whole of the phase it exists to serve. 62 of 536 MiB went through it and the
wall did not move.

Raising the pool to 128 per size class and 768 MiB made it work - **391 of 487
uploads, 431 MiB, `writeCallMs` 251 -> 60** - and the fold got **417 ms
SLOWER**, 5428 against 5011. Allocating four hundred mapped buffers costs more
than the copy it saves. The 191 ms is real and unreachable: it is paid back at
the allocator either way.

### And writing the big chunks straight from the shard is an exact wash

The other way round: skip `sharedStaging` for any chunk over 64 KiB and
`writeBuffer` it directly out of the shard's own ArrayBuffer, rounding the
length up to four (the pad the packer already leaves, and `chunk.at` is always
four-aligned). The bytes are concentrated in big chunks - OpenDDE 88% of them in
35% of the chunks, ESMFold2 92% in 35% - so this removes most of the memcpy.

| | `copyMs` | `writeCallMs` | wall |
|---|---:|---:|---:|
| OpenDDE staged | 60 | 257 | 4980 |
| OpenDDE direct | **7** | **311** | 4958 |
| ESMFold2 staged | 30 | 179 | 3033 |
| ESMFold2 direct | **2** | 179 | 3020 |

The copy went away exactly as intended and `writeBuffer` grew by the same
amount: 2608 extra driver calls at about **21 microseconds each**. A `writeBuffer`
call and 170 KiB of memcpy cost the same thing on this box, which is the number
to remember. Reverted; only the `copyMs`/`writeCallMs` split was kept, because
without it none of the above is visible.

## 🔴 A truncated profile reads exactly like a fast fold

`tools/gpu/profile.js` caps at 2048 passes - `createQuerySet` will not take
more than 4096 timestamps - and it turns `batchComputePasses` OFF to get one row
per label, which multiplies the pass count by about seven. An OpenDDE fold under
`--profile` reports **exactly 2048 passes**, which is the cap: everything after
it was dropped, and the `gpuTotalMs`, `idleMs` and `idleShare` printed beside it
describe a prefix of a fold rather than a fold.

`summary()` reports `dropped` now, and it is the first thing to read. And
`--profile-batched` on `fold.js` and `fold-opendde.js` keeps the batching, so a
whole fold fits and the GPU-busy total is the truth about the wall - at the cost
of per-kernel attribution, which is the other question and needs the other flag.

## What a fold's submits are, and which loop issues them

`tools/gpu/probe-submits.js` wraps another tool the way `probe-compiles.js`
does, and groups every `queue.submit` by the label of the encoder behind it. It
changes no tuning, so unlike `--profile` it describes the fold that ships.

| | submits | dispatches | bind groups | encode | submit | 
|---|---:|---:|---:|---:|---:|
| AF2 | 602 | 9254 | 5436 | 3.7 ms | 7.6 ms |
| OpenDDE | 883 | 16238 | 6643 | 7.7 | 10.8 |
| AF3 | 1447 | 36738 | 9463 | 14.4 | 15.8 |
| ESMFold2 | 726 | 11915 | 4815 | 17.5 | 8.2 |

Encoding and submitting together are **under 30 ms** in every model, and bind
groups another 25-48, so the host's own command building is not where a fold
goes - which is worth knowing before optimising it. The top submitter in all
four is `int5-upload`, at 422 of AF2's 602 and 443 of OpenDDE's 883, one per
weight pack; see above for why leaving it alone is right.

## Where a fold's HOST time goes, and why it is not the lever

`probe-submits.js` grew three things that make the question answerable:
`--repeats=<n>` for a median, the run's start-up charged to nothing, and the
gap split into what the CPU DID against what it WAITED for.

🔴 **THE FIRST SUBMIT'S GAP IS NOT A GAP.** Everything before it - the shards
over the network, the page's own start-up - was being charged to whichever
encoder happened to submit first, which on an OpenDDE fold is 1562 ms landing on
`af3-atom-encoder`, whose own stage measures 78. It is `startupMs` now.

🔴 **AND A GAP IS NOT HOST WORK UNTIL THE WAITING IS TAKEN OUT.** Every
`mapAsync`, `onSubmittedWorkDone` and `popErrorScope` is recorded as an interval,
merged into a union - not summed, since a fold holds hundreds of unawaited
completion promises at once - and subtracted. The column that is left is the
only one a host-side change can move.

Median of five first folds of OpenDDE at 68 tokens, 16 steps: 2681 ms wall, 967
of it start-up, and **638 ms of host work in the 1715 that remain**:

| encoder | hostMs | gapMs | submits |
|---|---:|---:|---:|
| `int5-upload` | 338 | 805 | 443 |
| `opendde-confidence-pair-init` | 95 | 107 | 1 |
| `af3-embedder` | 74 | 120 | 1 |
| `af3-diffusion-conditioning` | 59 | 344 | 17 |
| `af3-atom-decoder` | 37 | 57 | 16 |

`int5-upload`'s 338 is the weight decode, and the section above shows that is at
its floor. Everything else together is 300 ms of a 1715 ms fold, spread over
five modules. **There is no host-side lever here**, and that is the finding: a
fold is GPU and waiting, not CPU.

🔴 **AND ONE RUN OF THIS TABLE IS NOT A MEASUREMENT.** The same 300-token AF3
fold put **760 ms on `af3-confidence-embed` in one run and 47 in the next** -
the module's own internal marks total 47 - and the whole fold went 5723 to 4528
between them. That is this file's 3.2x drift showing up in a host table, and it
is worth most of a morning if the first number is believed. Use `--repeats`.

### Ruled out with numbers

- **`popErrorScope` is not a stall.** 428 of them in an OpenDDE fold summing to
  3469 ms looks alarming, and 370 come from `src/runtime/validation.js` - which
  is the DeferredValidation class, whose whole point is that the pops are held
  and settled together at a boundary that already synchronises. They resolve
  concurrently; the sum is not a wall. The other **58 calls come from 11 sites
  that DO await one, and they are 175 ms between them** - the largest the atom
  decoder's 32 calls at 62 ms.
- **Encoding and submitting are not the cost.** Under 30 ms in every model, with
  bind groups another 18-48. The command building is free relative to the fold.
- **Uploads by label are not the cost.** The largest is `expand.projection` at
  29 ms for 55 MiB; everything else is under 8.
