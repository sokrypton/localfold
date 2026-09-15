# What a new model costs, and how src/ was reorganised

Raised by the user on 2026-09-14, mid-port: "reference in different location,
scattering identical implementations, I think it's making it hard to add new
models and avoid reinventing and also avoid having to reoptimize for each
model."

🔴 **THE REORGANISATION IS DONE; THIS HEADER USED TO SAY IT WAS WANTED.** The
file began as evidence for a decision and is now the record of one, so the
order below is chronological rather than tidy: the measurement first, then the
duplications closed one at a time, then the layout. **The current state is at
the bottom** - `src/` is twelve directories, four lineages over six shared
layers, with zero import cycles. If you want only that, read the last three
sections.

**The first half of this file is EVIDENCE, not a plan.** It was written the same
night, from one port of two models (IntelliFold-2 and RoseTTAFold3), while every
place a convention had to be threaded was still open in an editor.

Nothing here is a complaint about the code being wrong. Every duplication below
was a reasonable local choice and several are load-bearing on purpose. The claim
is narrower and it is measurable: **adding one convention to one model touched
five files on average, and three of the six defects this port produced were the
same shape of bug in two of those five places.**

## The measurement: six conventions, and where each of them had to go

RoseTTAFold3 and IntelliFold-2 needed six conventions between them. Counting
every edit site that had to agree or the model is silently wrong:

| convention | CPU reference | GPU kernel | weight loader | host pack | device-resident pack | pipeline key | dialect table | test |
|---|---|---|---|---|---|---|---|---|
| `maskAtomActPerBlock` | ✓ | ✓ x2 (encoder, decoder) | ✓ | | | ✓ x2 | ✓ | |
| `triangleMulDivideByLength` | ✓ | ✓ | | | | ✓ | ✓ | |
| `msaPairedQueryRow` | ✓ | ✓ | | | | ✓ | ✓ | |
| the constant atom bias | ✓ (already there) | ✓ (already there) | ✓ x2 names | | | | | |
| OPM projection biases | ✓ | ✓ | ✓ | ✓ | | ✓ | | |
| grid attention biases | ✓ | ✓ x3 (vector, matrix q/k/v/gate, matrix out) | ✓ | ✓ | ✓ | ✓ x3 | | |
| `diffusionNoResidual` | ✓ | ✓ x3 (token, atom encoder, atom decoder) | ✓ | | | ✓ x3 | ✓ | |
| `kq_norm` | | ✓ x2 (token, atom - two kernels, one algorithm) | ✓ x2 roots | ✓ x2 orders | ✓ | ✓ x3 | | |
| `paddedAtomKeys` | ✓ (featuriser) | | | | | | ✓ | ✓ |

**The right-hand columns are the cost.** A convention that is one branch in the
reference is up to eleven edit sites here, and there is no single place that
fails when one of them is missed.

## The three bugs this shape produced, all in one night

Each is the same failure: **two lists that have to agree, in two places, with
nothing checking that they do.**

1. **`packOuterProductMeanWeights` reserved offsets over `ORDER + OPTIONAL` and
   WROTE over `ORDER`.** The bias was in the generated WGSL, in the offsets, and
   not in the buffer. Every oracle seam matched the previous run to the last
   digit, which reads exactly like "this convention does not matter here". Cost:
   about an hour, and it nearly became a wrong published conclusion.

2. **`residentGridOnDevice` is a hand-written mirror of
   `packGridAttentionWeights`' layout** — a second copy of `ORDER`, `QKVG` and
   `TRANSPOSED` in another file. A term added to one and not the other does not
   drop the term; it points the shader's `W_GATE_BIAS` INSIDE the output
   projection, on the resident path only. It was caught by reading, not by a
   gate.

3. **`=== undefined` where the loader writes `null`.** An optional weight's
   absence is `null`, that null reaches the SOURCES map, and a strict check
   falls through and reads `.count` off it — killing boltz2 in a code path added
   for a model boltz2 does not share.

And a fourth, older, found by the same port: **`const CHANNELS = 64` in BOTH
`template-reference.js` and `template-webgpu.js`**, plus `QUERY_CHANNELS = 128`
and `heads: 4, dimension: 16` typed into `check-af3-template.js`. That checker
read NaN on OpenDDE and 1.65e-1 on IntelliFold-2, **and neither was the port.**

## Where the same thing is written twice today

Named so a refactor can be checked against a list rather than a feeling.

**Weight ORDER, three copies.** `BLOCK_ORDER`/`blockOrderFor` in
`atom-encoder-webgpu.js`, `BLOCK_ORDER`/`txBlockOrder` in
`diffusion-transformer-webgpu.js`, and the hand-written mirror in
`pair-track-device-weights.js`. All three encode "these tensors, in this order,
some of them optional". Each had its own `has*` predicate
(`blockHasUpGate`, `txHasUpGate`, `blockHasKqNorm`, `txHasKqNorm`, and
`hasBondTypes` in the embedder) that must ask the SOURCES map rather than the
value — a rule that was a comment in five places rather than one function.
✅ **THAT HALF IS CLOSED**: `carriesTensor(weights, name)` in
`src/weights/weight-sources.js`, beside the symbol it asks. Its gate counts
DECODES through a getter and asserts zero, so the 7x bug - a presence test that
unpacked a 768x1536 int5 tensor per block per sampler step - cannot come back
silently. The ORDER lists themselves are still three copies.

**The pack/offset loop, five copies.** ✅ **CLOSED.**
`packOuterProductMeanWeights`, `packGridAttentionWeights`, `packTriangleWeights`,
`packTransitionWeights`, `packAtomBlockWeights`, `packBlockWeights`. Every one
was: walk a list, reserve an offset, sum a length, then walk it again and copy.
Two of them had the optional-tensor bug; the others did not have optional
tensors YET, which is not a property to rely on.

All six now call `packNamedWeights` in `src/weights/weight-pack.js`, and the fix
is structural rather than careful: **there is one `packing` array, it is a local
variable, and both loops read it** - a caller cannot pass two lists because the
signature does not have two. 🔴 **THE HELPER WAS NOT INVENTED**: the triangle
had already solved this locally and correctly with `packOrder`, which is exactly
why nothing there ever had the bug, so it was promoted rather than rewritten.
`test/weight-pack.test.js` packs DISTINCT NON-ZERO values and reads each back at
its reported offset - a reserved-but-unwritten region reads as zeros - and it
was watched failing with the original line put back.

**The staged LayerNorm, at least four copies.** `stagedLayerNorm` in
`triangle/shaders.js` is shared; the outer product mean, the atom output kernel,
the grid attention and both kq-norm kernels each write their own two-pass
mean/variance/reduce. They differ in layout and in what they fuse, which is why
they were written separately — but the REDUCTION is identical every time, and a
`use_fast_variance` convention has to be got right in each.

**A block's residual wiring, three copies.** ⬜ **On inspection this is not a
duplication at all, and listing it as one was this file overcounting.**
`diffusionNoResidual` is implemented once in the token transformer - a saved
buffer and a copy kernel, because its passes are SEPARATE - and once in the atom
stacks, as three read sites inside ONE FUSED kernel with no buffer at all. The
atom decoder then shares the encoder's factory, so the third "copy" is free and
was never a copy.

Two implementations because the two stacks fuse their passes differently, which
is a property of the kernels and not of how the convention was written down.
There is no shared thing to extract: a buffer-and-copy and a fused three-site
read have no common body. What a reorganisation can fix here is that the FLAG
reaches them by two different routes - which is the dialect-routing question
below, and is the user's call rather than a tidy-up.

**The CPU reference beside every GPU kernel.** This one is DELIBERATE and should
survive any refactor: `check-af3-*.js` is differential, and a shared
implementation would agree with itself. What should not survive is that the
CONVENTION is threaded into both by hand. The reference and the kernel should
read one description of the model.

**How a dialect flag reaches code, three routes.** `input.dialect` (the atom
encoder's `preTrunkQuery`), a field copied onto every block by the weight loader
(`chainedAtomLayerNorm`, `maskAtomActPerBlock`, `diffusionNoResidual` — because
the DECODER sees no dialect object), and a `shape` field derived at compile time
(`kqNorm`, `noResidual`, `opmBiasAfterNorm`). Which route a flag takes is a
property of who needs it, and nothing states the rule.

## What "reoptimize for each model" means concretely

The user's third complaint, and it is the one with numbers behind it.

Every tile, split and workgroup target in this port was fitted on AlphaFold 3's
widths - `cZ` 128, 4 heads of 32 - and the priors are per DEVICE, not per model.
IntelliFold-2's trunk pair is 512 channels and its template stack 256. What
follows from that tonight:

* **it is the heaviest model here, at 2229 MiB with no budget**, against AF3's
  983 - four times the pair width is four times every resident trunk weight;
* **the matrix pair kernels cost it 5.72e-4 at the template seam where AF3
  reads 2.77e-5**, purely because an f16 accumulation over 256 channels is not
  one over 64. The shipped trunk already pins them off for that stage, which is
  the right answer arrived at for a different model's reason;
* nothing in `probe-tuning.js` or the ampere prior knows a model's widths, so
  the sweeps that fitted `opmBlockITokens`, `transitionChunkBytes` and
  `singleProjectWorkgroupTarget` were all taken at AF3's shape and are applied
  at if2's.

None of that is wrong today - every gate passes and the fold is good - but the
knobs are fitted to a shape rather than derived from one, so a seventh model
inherits a sixth model's guesses.

## 🔴 AND ONE COUNTEREXAMPLE, WHICH IS WHAT THE REFACTOR SHOULD AIM AT

`pairTransitionSplit` turns itself on above `TRANSITION_SPLIT_MIN_CHANNELS`, a
THRESHOLD on the width rather than a per-model constant. It was fitted on
AlphaFold 3 (1.8%) and ESMFold2 (1.51x) and nobody has ever swept it at
IntelliFold-2's 512 channels - where it is **15.2x on the kernel and 3.6x on the
whole trunk**, and it selected itself correctly with no one touching it.

That is the shape to generalise. A rule keyed on something the BUNDLE states
transfers to the next checkpoint; a constant fitted at one model's shape does
not. Every item in the section above is a constant fitted at AlphaFold 3's
shape, and this one is not.

## What the refactor would have to keep

Written down because these are the things a tidy-up would naturally break, and
each of them was paid for:

* **the CPU reference must stay an INDEPENDENT implementation.** It is the only
  differential this port has below the oracle.
* **a pipeline key must name every variant in its source text**, including ones
  that only shift an offset. Three collisions here were keys that did not.
* **a convention must have no default.** Every consumer throws on a missing
  dialect flag rather than assuming AlphaFold 3, and that is what turns a
  silently different model into an error.
* **generated WGSL with constants baked in, not uniforms.** A runtime loop bound
  in a hot WGSL loop costs 4.3x here; a uniform in an index costs a division.
* **the shapes come off the BUNDLE.** Every width typed into a file was a bug
  waiting for the next checkpoint, and four of them fired this month.


## The first slice, done — and what it deliberately did not touch

Landed 2026-09-14, after both models were complete, as the opening of the
reorganisation. **Chosen because it needs none of the three design questions
answered**: it changes no module layout, keeps the CPU reference an independent
implementation, and does not touch how a dialect flag reaches code. 28 net lines
smaller across eight files, and all three standing gates byte-identical.

| duplication | status |
|---|---|
| the pack/offset loop, six copies | ✅ one helper, and the bug is now unrepresentable |
| the `has*` SOURCES predicate, five copies | ✅ one function, gated by a decode counter |
| weight ORDER, three copies | ✅ **the two DEVICE mirrors are gated** (grid and triangle, test/grid-layout-agrees.test.js, watched failing from both sides), and the grid's four retyped lists are **collapsed** - `GRID_ORDER`, `OPTIONAL_GRID`, `QKVG`, `TRANSPOSED` are imported from the host packer now. The triangle's still spells its order out: it splices four interleaved projections in where the vector path has eight separate ones, so it is not one list with a filter |
| the staged LayerNorm, four copies | ❌ **MEASURED AND DECLINED - and it is THIRTEEN blocks, not four.** See below |
| a block's residual wiring, three copies | ⬜ **NOT A DUPLICATION - one convention, two necessary implementations.** See below; nothing to collapse |
| how a dialect flag reaches code, three routes | ✅ **decided and collapsed** - outside the weight loader every copied flag is read off the BLOCK, which is the route the loader guards with a throw. `test/dialect-routes.test.js` asserts the rule and was watched failing both ways |

**Why those two first.** They are the only two duplications on the list that
have actually produced defects: the outer product mean's reserved-but-unwritten
bias (an hour, and nearly a published conclusion) and the presence test that
decoded a tensor to ask whether it existed (boltz2's fold, 38.5 s against 3.7).
Both are now structural — a caller cannot express the bug — rather than a
comment asking the next author to remember.

**Why not the rest yet.** The three open items each need a shape decided:

* **weight ORDER** ✅ **done for the grid, deliberately not for the triangle.**
  The differential came first, which is what made the collapse safe: the grid's
  four retyped lists are imported from the host packer now, so a term added
  there reaches the resident path without a second edit. The triangle's layout
  is not one list with a filter - the matrix path splices four interleaved
  projections in where the vector path has eight separate ones - so collapsing
  it is a different change with its own risk. It is gated either way.
* **the staged LayerNorm** ❌ **measured, and declined with numbers.** Counted
  rather than eyeballed: **13 reduction blocks across 5 files, 246 lines** - not
  the four this file first claimed - and after normalising whitespace and the
  channel-count identifier, **no two of them are alike**. They differ on three
  real axes, not on spelling:

  | | lines | tiling | variance | accumulates |
  |---|---:|---|---|---|
  | grid attention | 20 | multi-row tile | fast, `E[x^2] - mu^2` | raw values |
  | transition | 37 | multi-row tile | two-pass | **centred** values |
  | diffusion transformer | 22 | **one row per group** | two-pass | raw values |

  A shared generator would need all three as parameters plus the count, the
  grid-width constant name and the load expression - at which point it is a
  switchboard emitting three different algorithms rather than one shared
  implementation, and each call site is barely shorter than the block it
  replaced. **The duplication here is apparent rather than real**: they look
  alike because a two-pass reduction always looks alike, and they are not the
  same computation.

  🔴 Note this is NOT the runtime-indirection objection this file gave before.
  These emit generated WGSL, so factoring the GENERATOR costs nothing at run
  time as long as the text is unchanged - that objection was wrong and the real
  one is legibility. The reason to decline is that the shared thing would be
  harder to read than the three things it replaced.
* **how a dialect flag reaches code** is the design question, not a tidy-up.
  Three routes exist because three different things need the flag at three
  different times (a call site, a per-block copy for a decoder that sees no
  dialect, a compile-time shape). Picking one route means either giving the
  decoder a dialect or giving the shape resolver a block, and that is the
  user's call rather than mine.

  🔴 **AND THE DIFFERENTIAL FOR IT LANDED FIRST, WHICH IS WHAT MADE THE GRID'S
  COLLAPSE SAFE AND WILL MAKE THIS ONE SAFE.** `test/dialect-routes.test.js`
  reads the copied-flag list OUT OF the loader (a list typed twice is the thing
  it exists to stop) and pins which source every read site uses. Measured
  rather than assumed, and it corrected this file: the loader copies **four**
  flags, not three - `keyMaskedAtomAttention` is the fourth - and

  | flag | read sites | sources |
  |---|---:|---|
  | `maskAtomActPerBlock` | 4 | block |
  | `chainedAtomLayerNorm` | 4 | block |
  | `keyMaskedAtomAttention` | 4 | block |
  | **`diffusionNoResidual`** | **5** | **block AND dialect** |

  Three of the four take exactly ONE route and are consistent everywhere. **One
  takes two**: the atom encoder reads `input.dialect.diffusionNoResidual` for
  its compile-time shape while the decoder and the CPU reference read the copy
  on the block. They agree only because the loader writes the dialect's value
  onto every block - and the decoder reads block **ZERO** where the reference
  reads the block it was handed, so a copy that failed for a later block would
  leave two of the five right and say nothing.

  ✅ **AND THE DECISION IS TAKEN: COLLAPSE TO THE BLOCK.** The user's call, on
  the table above. One line - the atom encoder reads
  `weights.blocks[0]?.diffusionNoResidual` where it read `input.dialect?.` - and
  **every copied flag now takes exactly one route outside the loader.**

  🔴 **AND THE BLOCK IS THE BETTER-GUARDED ROUTE, WHICH DECIDED IT.**
  `atomBlockWith` THROWS on a block carrying no `diffusionNoResidual`, where
  `input.dialect?.X === true` reads a missing dialect as FALSE - silently
  choosing AlphaFold 3's behaviour, which is exactly the default the convention
  rule at the top of this file exists to forbid. Collapsing the other way would
  have kept the unguarded read and grown three signatures for a value that
  never varies within a stack, which is the objection this file already had.

  The test asserts the RULE rather than the list now: outside the loader, a
  copied flag is read off the block and never off a dialect - so a fifth read
  site against a new source fails whatever it is called. Verified to fail by
  putting the dialect read back.

  **Behaviour-preserving, measured rather than argued.** rosettafold3 is the
  only dialect that sets the flag: 6MRR `meanPlddt` **81.53041400210395** and
  1.649 A before and after, af3 **83.52427630142053** and 0.706, rf3's 5CAJ
  self-template 0.137 A at 79.16165604369479 - sixteen digits each way.

None of this is urgent. Every gate is green and both new models fold at the
reference's numbers; the argument for continuing is the measured one at the top
of this file, not a feeling that the code is untidy.


## The reorganisation itself: src/af3 grouped by STAGE

The user's actual ask, and the thing every slice above deliberately did not
touch: **"reference in different location, scattering identical
implementations."** `src/af3` was **54 files and 31,574 lines in one flat
directory - 43% of the whole codebase** - and a stage's CPU reference and its
GPU kernel were paired by FILENAME SUFFIX, so `template-reference.js` and
`template-webgpu.js` sat twenty entries apart in `ls`.

    src/af3/
      featurise/    10 files   3,224 lines   the batch, the CCD, conformers, MSA, template features
      trunk/        19 files  10,188 lines   embedder, template, MSA stack, pairformer, and their kernels
      diffusion/    11 files   9,747 lines   head, transformer, conditioning, atom encoder/decoder, samplers
      confidence/    5 files   2,673 lines   pLDDT/PAE/PDE and OpenDDE's own head
      weights/       4 files   2,446 lines   the loaders and the two device packers
      structure/     2 files     493 lines   OpenDDE's structural-token expander
      (top level)    3 files   2,803 lines   dialect.js, fold.js, chain-geometry.js

Each stage's **reference and kernel now sit in one directory**, which is the
complaint answered: `trunk/template-reference.js` beside
`trunk/template-webgpu.js`. The three files at the top are the ones that are not
a stage - the cross-cutting convention table, the entry point, and the shared
geometry rule.

🔴 **THE SAFETY NET CAME FIRST, AND IT HAD TO.** There is no bundler and no type
checker in the gate path here: the page and `tools/gpu-chrome.mjs` both serve
`src/` over HTTP as ES modules, so an import that resolves nowhere is a 404 at
RUNTIME in whichever lane happens to name it - and docs/PARITY.md records
nineteen of twenty-one AF3 checkers not running on this box at all, so a missed
rename could sit for weeks. `test/imports-resolve.test.js` walks src, tools,
test and web and asserts every relative and root-absolute specifier names a file
that exists. 51 files moved, **imports rewritten in 184 files**, and the gate was
green on the first run after.

🔴 **BUT THE GATE WAS BLIND TO THREE TESTS, AND THEY ALL BROKE AT ONCE.**
`af3-ligand-bonds`, `confidence-precision-pins` and `dialect-routes` open
`src/af3/<name>.js` with `readFileSync` to assert something about the TEXT of a
module - which is not an import, so the one gate written to make a file move
safe could not see them. `test/helpers/af3-source.js` is the fix and it is the
durable one: a structural test names the MODULE, never its directory, because
the directory is a fact about how the tree is organised today and none of those
tests is about that. It raises on a duplicate basename rather than silently
handing back whichever the walk reached last.

**Behaviour-preserving, measured.** The page folds unchanged (af3, 58 residues,
pLDDT 66.5); rf3 6MRR `meanPlddt` **81.53041400210395** and af3
**83.52427630142053**, sixteen digits; `test:portable`, `test:spec-floor` and
`test:stock` identical on all 24 rows; `test:ligand` at exactly its recorded
bond values (af3 0.050, protenix2 0.044, if2 0.055, openbind0 0.058, boltz2
0.062, rf3 0.069); `test:batch` 14/14; `npm test` 1115/0.

Also deleted: `src/af3/.ipynb_checkpoints`, two stale untracked copies of
`batch.js` and `fold.js` totalling 1,436 lines, sitting inside `src/` where a
reader cannot tell them from the real thing.

🔴 **WHAT THIS DOES NOT DO.** It moves files; it does not change what a new
convention costs. `dialect.js` is still 51 flags x 7 checkpoints stated by hand
- that is the price of the no-defaults rule at the top of this file, and it is
gated (`test/af3-dialect.test.js` asserts every dialect states every flag, and
that stock AlphaFold 3 is all-false). And the per-model tuning complaint is
still open and still measures as PROPORTIONATE rather than wrong: IntelliFold-2's
trunk at 256 tokens is 4884 ms and 3173 MiB against AlphaFold 3's 937 and 689,
for a four-times-wider pair. No knob is currently wrong at a non-AF3 shape.

## The token transformer's shape comes off the bundle now

`src/af3/weights/diffusion-weights.js` handed the token transformer
`channels: 768, heads: 16, dimension: 48, transitionFactor: 2,
blocksPerSuperBlock: 4` as typed-in numbers. The comment two lines above them
already records this literal being wrong twice for exactly that reason -
`pairChannels: 128` read protenix2's 256-wide pair through a 128-wide stride,
and `condChannels: 384` allocated boltz2's conditioning at half its size. Those
two were fixed; these five were the rest of the same literal.

**One tensor states all of them.** `transformerq_projection/weights` is
`[superBlocks, blocksPerSuperBlock, channels, heads, dimension]`:

```
af3 boltz2 protenix2 intellifold2 rosettafold3 openbind0 opendde
        all seven:  [6, 4, 768, 16, 48]
```

and `transformerffw_transition1/weights` ends in `channels * factor * 2`,
because the transition is a SwiGLU and its gate and value live in one tensor.
`txShape` reads both, checks that `heads * dimension === channels` and that the
hidden axis is a whole SwiGLU factor, and raises naming the tensor if either
fails.

🔴 **ALL SEVEN AGREE TODAY, SO THIS CHANGES NO FOLD** - 24 gate rows identical
across `test:portable`, `test:spec-floor` and `test:stock`, and af3's 6MRR is
`83.52427848789334` to the digit. That is the honest value proposition: it is
not a fix, it is the eighth checkpoint getting the right widths or a named
failure here instead of a validation error four stages downstream. This is the
`TRANSITION_SPLIT_MIN_CHANNELS` shape applied to a shape rather than to a
threshold - a rule keyed on what the bundle states transfers to the next
checkpoint; a constant fitted at one model's does not.

🔴 **AND IT RAISED ON A TEST STUB IMMEDIATELY**, which is the behaviour working:
`test/af3-diffusion-weights.test.js` answers `[24]` for every tensor it is not
asked about, so the derivation refused with "the token transformer's
q_projection is rank 1". The stub states the two shapes now.

## Item 5, measured and NOT acted on: the AF3 binding ceiling

The question was whether AF3's `addPair` should window its bindings the way
AF2's `addInPlace` does. It is **not resolved, and here is exactly how far it
got**, because a half-answered question recorded as an answer is how this file's
other entries went stale.

🔴 **THE PROBE CANNOT SEE AF3 AT ALL.**
`tools/gpu/probe-binding-ceiling.js` patches
`WebGpuExecution.prototype.dispatch` - AF2's seam. The AF3 stacks call
`dispatchWorkgroups` from six of their own modules, so `--tool=fold` recorded
**zero labels** and returned `lowestCeiling: null` with an empty list. Read
quickly that is "nothing is near the limit"; it means the instrument was
pointed at the wrong place. **It refuses now** rather than returning that, and
still reports AF2 correctly (89 labels, `opm.contract` at 724). Extending it
means watching the BIND GROUP, since AF3 has no single dispatch seam.

🔴 **BUT THE QUESTION IS LIVE, WHICH THE FIRST GUESS GOT BACKWARDS.** I assumed
memory would exhaust before the binding limit, making the whole thing moot.
Measured peak against length, two points each, fitted as `A + B*L²`:

| | peak at 256 | at 384 | memory exhausts ~ | binding limit |
|---|---:|---:|---:|---:|
| af3, pair 128 | 621 MiB | 869 MiB | **L ~3660** | **2047** |
| intellifold2, pair 512 | 2894 MiB | 3863 MiB | **L ~1812** | **1023** |

The binding limit comes **first** for both, so windowing could matter - and
IntelliFold-2's is at about a thousand residues, which is a size people fold.

### ...and then it was measured, and the answer is DO NOT WINDOW

`watchBindings` in the probe patches `createBindGroup` and `beginComputePass` -
the WebGPU objects, not a repository helper, so it catches every caller in
either family - and records the largest BINDING per labelled pass. The quantity
is `resource.size` (or the buffer's remainder past `resource.offset`), which is
what `maxStorageBufferBindingSize` limits, rather than the buffer's whole
length. It needs `batchComputePasses` off for profile.js's reason: a batched
pass is one label over many dispatches.

| `--tool=fold --lengths=200,400` | labels | lowest ceiling | sharing it |
|---|---:|---:|---:|
| af3, pair 128 | 98 | **2047 residues** | **29** |
| intellifold2, pair 512 | 101 | **1023 residues** | **29** |

Both numbers are exactly the arithmetic - `sqrt(2 GiB / (cZ * 4))` - and the
group is the same 29 either way: `trunk-pair`, `pair-logits`,
`embed.assemble-pair`, `template.embed.0`, the five `tri.*`, the five `grid.*`,
and fifteen more. **`addPair` is not even among the largest of them.**

🔴 **SO WINDOWING THE ADD MOVES THE CEILING BY EXACTLY ZERO, WHICH IS WHAT AF2
FOUND WITH ITS 28-LABEL TIE** - and now AF3's is measured rather than reasoned
by analogy. Every dispatch binding an `L² * cZ` f32 tensor gives out together;
windowing any subset of a tie moves nothing. The next group is 2896 for af3 and
1448 for if2, both a single label.

**Read the caveat on the af3 row**: its lowest group extrapolates from 3.8% of
the limit, because at 400 residues a 128-wide pair is nowhere near 2 GiB.
IntelliFold-2 has no caveat - its pair is four times wider so the sample is
close - and **1023 residues is the number that matters anyway**, being a size
people fold.

The memory fit above stands as the other half: memory exhausts around 3660
(af3) and 1812 (if2), so the binding limit really is what binds. Reaching past
it needs the pair ELEMENT to shrink - a packed f16 pair puts the binding at
2896 by either route, which is upstream's answer and is in docs/AF2.md - not
twenty-nine windows.


## src/esmfold2: two renames, not a directory tree

The last flat directory with the reference/kernel split - and the fix is NOT
the one af3 got. Seventeen files do not want five subdirectories of two or
three; that is structure for its own sake. What they wanted was for the pairs
to be findable, and two of them were not:

| was | is | why |
|---|---|---|
| `atom-encoder-reference.js` | `atom-transformer-reference.js` | its own header says "a sliding-window atom transformer", which is `atom-transformer-webgpu.js` |
| `featuriser-reference.js` | `pair-features-reference.js` | its own header says "how the pair representation starts", which is `pair-features-webgpu.js`'s "`z_init` and the recycle projection" |

Both were the CPU reference for a kernel they did not sort next to, so a reader
looking for one had no way to find the other. Every pair is adjacent now -
atom-transformer, diffusion, distogram, pair-features - and the three files with
no reference (`trunk-webgpu`, `language-pair-webgpu`, `featurise`) and the one
reference with no file of its own name (`sampler-reference`, whose kernel lives
inside `diffusion-webgpu.js`) are the honest remainder.

🔴 **AND THE FIRST ATTEMPT BROKE 25 TESTS, FROM ONE LINE.** The rewriter
replaced the old BASENAME anywhere it appeared, and `atom-encoder-reference.js`
is also the name of a **different file** under `src/af3/diffusion/` - so every
AF3 import of its own reference was rewritten to a module that does not exist.
Reverted with `git reset --hard` and redone rewriting only PATH-QUALIFIED
references: 9 files touched instead of 30. A rename is a path operation, and a
basename is not a path. ESMFold2's atom checksum is **-128226** and af3's 6MRR
`83.52427848789334` / 0.706, both unchanged.

## "Keep reference separate, fix dependencies" - the cycles, and what caused them

The user's ask after the af3 move, and the first thing it needed was a
measurement rather than a rearrangement. A directory-level import graph over
`src/` found **three cycles**, and each one turned out to be a single leaf
module filed under the wrong heading:

| cycle | the backwards edge | what it really was |
|---|---|---|
| `reference <-> runtime` | `runtime/quantised-upload.js` -> `reference/dtype.js` | `dtype.js` is "reading a stored tensor" - a RUNTIME concern living in the weight-bundle layer |
| `design <-> af3` | `af3/featurise/template-input.js` -> `design/superpose-pdb.js` | a superposition utility imported by af3 and web and **not by design**, the directory holding it |
| `af3 <-> evoformer` | both ways | the one that is real - see below |

Both of the first two are **leaves** - no relative imports of their own - so
moving them could not create a new edge, and both cycles closed:
`src/weights/dtype.js` and `src/heads/superpose-pdb.js`.

🔴 **AND "reference" MEANS ONE THING NOW.** `src/reference/` held no CPU
reference at all - it is `bundle.js`, `http-tensor-store.js`, `manifest.js`,
`manifests/`, the weight-bundle and tensor-store layer - while the whole
repository uses `*-reference.js` for the independent CPU implementations that
AGENTS.md requires every kernel to have. One word for two unrelated things, in a
tree whose other complaint was that the references were hard to find. It is
**`src/bundles/`**. 105 files rewritten.

### The cycle that was NOT a misfiling - CLOSED by extraction, not by a move

`af3 <-> evoformer` is genuine reuse in both directions: AlphaFold 2's template
module uses AlphaFold 3's template featuriser
(`evoformer/template.js` -> `af3/featurise/template-{features,input}.js`) and
AlphaFold 3's grid attention uses AlphaFold 2's subgroup-matrix geometry
(`af3/trunk/grid-attention-matrix.js` -> `evoformer/attention-matrix.js`).

Closing it needed an **extraction, not a move**, and only ONE of the two
directions had to go: `template-input.js` genuinely depends on af3's conformers
and alphabet, so AF2 borrowing AF3's template featuriser stays and is honest.
What left is the other way round - the seven symbols AF3 took out of AF2's
attention kernel, now in **`src/kernels/attention-geometry.js`**:
`ATTENTION_MATRIX_SUBGROUP_SIZE`, `ATTENTION_MATRIX_UNIT`,
`attentionMatrixGeometry`, `paddedHead`, `strides`,
`attentionMatrixStorageBytes` and `supportsAttentionMatrix`.

None of them emits a kernel. They are a geometry and four capability questions -
what tile the units want, how wide a padded head is, how the strides dodge the
banks, what that costs in bytes, and whether this device can run any of it - and
none of that is AlphaFold 2's. The kernel stays in
`evoformer/attention-matrix.js`, which imports them back and re-exports them, so
its own callers never noticed.

🔴 **AND `allowsAttentionSubgroupSize` HAD TO COME WITH IT.**
`supportsAttentionMatrix` calls it and it lived in `evoformer/attention.js`, so
leaving it there would have made `runtime` import `evoformer` - a worse layering
than the cycle being removed. It is eight lines of `device.adapterInfo`, a
runtime question wherever it is written. `attention.js` imports and re-exports
it so `probe-kernel.js` and `attention-subgroup-size.test.js` are unchanged.

**`src/` now has ZERO import cycles.**

🔴 **AND THE GATE IS WHAT TOLD ME I WAS FINISHED.** `ALLOWED` still held
`af3<->evoformer`, the cycle stopped happening, and the second assertion failed
with *"af3<->evoformer is allowed and no longer happens - drop it"*. That
assertion exists because an allow-list's real failure mode is outliving its
reason; it caught its own entry within the hour. `ALLOWED` is empty now.

Verified: AF2 **-1287025** and `probe-kernel.js` still resolving
`attention:flash-matrix-32-f32f32f32-4x32`, which is the sharp test - a
`supportsAttentionMatrix` that answered differently after the move would pick a
different kernel and change the checksum. af3 6MRR `83.52427848789334` / 0.706,
and all 24 rows of `test:portable`, `test:spec-floor` and `test:stock`
identical.

🔴 **`test/no-import-cycles.test.js` IS THE GATE**, and it has two halves. A new
unexplained cycle fails it. And an ALLOWANCE THAT NO LONGER HAPPENS also fails
it - "af3<->evoformer is allowed and no longer happens, drop it" - because the
failure mode of an allow-list is that it outlives its reason and the next entry
is added beside a stale one. Both halves were watched failing.

🔴 **AND NOT EVERYTHING FLAT IS SCATTERED, WHICH IS WHERE THIS STARTED.** The
obvious reading of fourteen top-level directories is that AlphaFold 2 has been
smeared across `evoformer`, `model`, `multimer`, `structure` and `heads`. The
imports say otherwise: `evoformer` is imported by af3, heads, model, multimer
and structure; `triangle` by four families; `heads` by af3 and esmfold2;
`runtime` by everything; and `multimer` by nothing at all, because it is a leaf
entry point. These are **layers several models share**, not silos that got
scattered, and a `models/` vs `shared/` split would cut across real edges and
make the tree lie. The flat top level stays.

## Two tails: an editor's snapshots, and one more file in the wrong directory

**`src/af3/feature-convergence.js` was AlphaFold 3's, filed under AlphaFold
2's graph.** 180 lines, a leaf, imported by `af3/fold.js` and by nothing else in
`src/`. It is `src/af3/feature-convergence.js` now. Its neighbour
`recycle-convergence.js` stays where it is and that is not an oversight: it is
imported by `model/monomer.js`, `model/query-only.js`, `multimer/model.js`,
`af3/fold.js` and `src/index.js`, so it is genuinely shared and `src/model/` is
as good a home as any.

🔴 **AND `src/testing/` STAYS A ONE-FILE DIRECTORY, DELIBERATELY.** I listed it
as an oddity. It holds a 45-line seeded generator used by `tools/`, `test/` and
`web/dev.js` - and by nothing in `src/` at all. Moving it into `test/` would
make the dev panel import from the test tree; moving it into `runtime/` would
muddy the layer that is otherwise all GPU plumbing. A one-file directory whose
name tells the truth is not a problem worth churning ninety files for.

### 🔴 AND A jupyter-lab IS WRITING INTO src/, WHICH ONE GATE WAS READING

`src/af3/.ipynb_checkpoints/fold-checkpoint.js` and
`src/kernels/triangle/.ipynb_checkpoints/webgl2-checkpoint.js` are written by the
jupyter-lab running against this checkout, every time those files are saved.
They are gitignored and never reach the repository.

**An earlier commit here says it "deletes" the first of them, and that was
wrong**: it came back within the hour, because deleting an editor's snapshot
does not stop the editor. What matters is that nothing READS them, and
`test/no-import-cycles.test.js` was: its `walk` did not skip hidden directories,
so two stale snapshots were contributing to the dependency graph this file now
quotes. Checked rather than assumed - they add **no edge** today, because their
imports are a subset of the live files' - but a snapshot taken mid-edit could
name anything, and the graph would report it as a dependency. The walk skips
them now, like `imports-resolve`, `af3-source` and `modules-parse` already did.

`test/modules-parse.test.js` learned this first and its comment is the best
statement of it: the failure named the checkpoint, but "every module under src/
loads" going red reads as a broken source tree, and the source tree was fine.
## The layout, after mirroring af3-any-model where it fits

The user's steer: stop improving the method, consolidate and organise for
maintenance, and consider copying the reference implementation's shape.
af3-any-model's is:

```
src/alphafold3/
  af2/        39 py   the AF2 lineage, kept apart
  model/      57      network/ pipeline/ components/ scoring/ + model_config, params, weights
  data/       20      featurisation
  structure/  10      constants/ 11      common/ 5      jax/ 6
```

Three of those ideas transfer, and one does not. **`af2/` as a separate
lineage** transfers. **A shared compute backend** (`jax/`) transfers - that is
`runtime/`. **Weights and params beside each other** transfers - that is
`bundles/`. What does NOT transfer is `model/network/` as one flat directory of
stages: af3-any-model duplicates the AF2 and AF3 lineages rather than sharing
kernels, where this port shares them, so our AF3 stages are already grouped
(`af3/trunk`, `af3/diffusion`, `af3/confidence`) and flattening them would undo
the first slice.

### 🔴 `src/evoformer/` WAS TWO THINGS, WHICH IS WHY IT LOOKED SHARED

The obvious reading of the old tree is that AlphaFold 2 was smeared across
`evoformer`, `model`, `multimer` and `structure`. The measured reason it looked
that way is narrower: **`evoformer/` held AF2's trunk AND the shared kernel
library**, under a name that says only the first. `heads/confidence.js` and
`structure/ipa.js` both imported `attention.js` and `transition.js` from it, so
a directory named for AF2's trunk was a dependency of AlphaFold 3's confidence
head.

The boundary was already one-way and clean - **zero kernel-to-trunk edges**:

| kernels (moved to `src/kernels/`) | trunk (stays in `src/af2/evoformer/`) |
|---|---|
| `attention.js` -> `attention-matrix.js` | `block.js` -> attention, attention-project-matrix, opm, transition |
| `attention-project-matrix.js` | `stack.js` -> block |
| `attention-webgl2.js` | `input-embedder.js` -> attention |
| `outer-product-mean.js`, `transition.js` | `template.js` -> attention, block |

After the split `evoformer` is imported by `model`, `multimer` and `index.js`
and **nothing else** - a closed lineage - so the four directories group as
`src/af2/{evoformer,model,multimer,structure}`.

🔴 **AND ONE EDGE WAS MINE.** `af3 -> structure` did not exist before this
session; it appeared when `superpose-pdb.js` went into `structure/` to break the
`design <-> af3` cycle. It is a structure COMPARISON utility, which is what
`heads/tm-score.js` is, and af3 already imports `heads` - so it is
`heads/superpose-pdb.js` now and `structure` is AF2's again. **A move that
closes one cycle can quietly re-aim a directory**, and the only reason this was
noticed is that the dependency graph got re-measured after every step rather
than once at the start.

### Where it landed

```
lineages   af2/ 20   af3/ 56   esmfold2/ 17   esmc/ 3
shared     kernels/ 6   triangle/ 8   heads/ 5   input/ 6   runtime/ 23   bundles/ 21
other      design/ 14   testing/ 1
```

Twelve top-level directories from fourteen, and every name now says what the
thing is: a lineage, a shared layer, or neither. Zero import cycles.

**Behaviour-preserving across 115 rewritten files**: AF2 **-1287025**, the 30/29
multimer **315591**, af3 6MRR **83.52427848789334** / 0.706, ESMFold2
**-128226**, and all 30 rows of test:batch, test:portable, test:spec-floor,
test:stock and test:ligand unchanged.

## All the shared kernels in one place, and the split that is blocked

`src/runtime/` was three things at 23 flat files - device and execution
plumbing, the weight-decode path, and two shared KERNELS that had drifted in.
The kernels are out:

- `matrix-linear.js` (822 lines, the staged subgroup-matrix GEMM, imported by
  every family) and `attention-geometry.js` (192, extracted earlier to break the
  af3/evoformer cycle) are now in `src/kernels/`, beside the six that came out
  of `evoformer/`.
- `src/triangle/` - a self-contained shared kernel package with its own CPU
  reference and WebGL2 backend - is `src/kernels/triangle/`, because a shared
  kernel sitting BESIDE the shared-kernel directory is the same confusion this
  whole pass is about.

🔴 **AND I EXPECTED THE MATRIX MOVE TO BE BLOCKED, AND IT WAS NOT.**
`device-profile.js` names `matrix-linear.js` twice and both are COMMENTS, not
imports - so moving it created no `runtime -> kernels` edge and no cycle. Two
minutes of grep against an afternoon of not doing it; the same check said
`device-memory.js` has no relative imports at all.

### 🔴 THE WEIGHT-DECODE SPLIT IS BLOCKED, AND BY A REAL COUPLING

The other seven - `dtype`, `float16`, `weight-pack`, `weight-sources`,
`device-pack`, `quantised-upload`, `resident` - are one concern ("turn a stored
tensor into a device buffer") and would read well as `src/weights/`. They cannot
move:

```
runtime/execution.js  ->  device-pack.js, resident.js      (runtime -> weights)
resident.js           ->  device-memory.js                 (weights -> runtime)
```

`execution` needs the packers to fill a buffer and the packers need the
allocator's accounting to say what was filled. That is a directory cycle, and
this file's own gate would refuse it. Breaking it means deciding whether the
memory accounting belongs with the allocator or with the packers, which is a
design question and not a tidy-up - the same verdict the `af3 <-> evoformer`
cycle got before someone actually extracted its shared surface.

### Where `src/` ended up

```
lineages   af2/ 20      af3/ 56     esmfold2/ 17   esmc/ 3
shared     kernels/ 16  heads/ 5    input/ 6       runtime/ 21   bundles/ 21
other      design/ 14   testing/ 1
```

Eleven top-level directories from fourteen, no name that lies about its
contents, and zero import cycles. **Behaviour-preserving:** AF2 **-1287025**
still resolving `attention:flash-matrix-32-f32f32f32-4x32`, af3 6MRR
**83.52427848789334** / 0.706, ESMFold2 **-128226**, 24 gate rows identical.

🔴 **AND THE COMMENT-PATH GATE EARNED ITS KEEP AGAIN.** The rewriter fixed
import specifiers and directory-shaped prose and missed two FULL FILE paths -
`src/kernels/matrix-linear.js` and `src/kernels/attention-geometry.js` - named
in 22 comments across 13 files. `test/imports-resolve.test.js` failed on all of
them by name. That gate was written this session after the af3 move left 129 of
exactly this behind.

## 🔴 THE REORGANISATION BROKE THE DEPLOY, AND NO JS GATE COULD SEE IT

Found by running `python3 tools/build_site.py` rather than by any test. Three
separate failures, all of them in the publishing path, all of them invisible to
`test/imports-resolve.test.js` because that gate read `.js` and `.mjs` only.

**1. Literal paths in python, yaml and html - 43 across 13 files.** Mostly
prose, but not all: **`tools/write_manifest_module.py` carries a `"module"`
WRITE TARGET per family**, thirteen of them. After `src/reference/` became
`src/bundles/` that tool would have written thirteen manifest modules into a
directory nothing loads - silently recreating the old tree while the live one
went stale. It is the tool you run straight after `hf upload`, so the first
symptom would have been a published bundle the page could not find.

**2. Paths BUILT FROM SEGMENTS, which no regex over strings can see.**

```python
index = (ROOT / "src" / "reference" / "manifests" / "index.js").read_text(...)
```

There is no `src/bundles/manifests/index.js` literal anywhere in that line.
Five files did this - `build_site.py` (three times), `check_remote_bundle.py`,
`export-js-weights.py`, `export_multimer_model.py`,
`dump_reference_conformers.py` - and the new non-JS rule found none of them.
**The gate for a constructed path is running the thing**, which is how this was
found.

**3. `build_site.py` walked `.ipynb_checkpoints` into the deploy check** - the
third time this session an editor's snapshot has been read as source, after
`test/modules-parse.test.js` (which learned it first) and
`test/no-import-cycles.test.js`. The reorganisation made a pre-existing bug
fatal: those snapshots' imports used to resolve and now do not, so the build
failed on two files nobody wrote, with advice pointing at `.gitignore`. Its walk
skips hidden directories now, like every other walker here.

`test/imports-resolve.test.js` grows a third rule for the literal half, with its
own walker over `.py`, `.yml`, `.html` and `.sh` - **not** by widening the
shared one, which the first attempt did and which immediately made the
comment-path rule read python it has no exclusions for. py2Dmol's own tree
(`src/align/`, `src/app/`, `src/core/`, `src/io/`) is excluded by prefix,
because those are a different checkout's paths named here on purpose.

Verified: `build_site.py` writes **dist/ 230 files, 23.1 MiB**, and
`write_manifest_module.py af3` writes `src/bundles/manifests/af3.js`.

**The standing lesson.** Every rule added this session has been one file type
behind the last mistake: imports, then comments, then python and yaml. The
constructed-path case says where that ends - a pattern over text cannot see a
path assembled at runtime, and the only gate that covers it is executing the
tool. `build_site.py` should be run after any move under `src/`.

## The two fundamental fixes, rather than a fourth pattern

Every gate added this session was one file type behind the last mistake:
imports, then comments, then python and yaml. That is a losing shape, and the
last two changes are the ones that stop it rather than extend it.

**1. `npm run test:site` is a standing gate now.** A pattern over text cannot
see `(ROOT / "src" / "reference" / "manifests" / "index.js")` - there is no
literal path in that line - and five python tools built their paths that way.
The only thing that covers a constructed path is **executing the tool**, so
`tools/build_site.py` joins the gate list. It assembles `dist/` and runs its own
unresolved-import check over the tree about to be published. Run it after any
move under `src/`.

**2. `test/helpers/source-files.js` is the one statement of "what is a source
file".** Four separate walkers had each learned, the hard way, that a
jupyter-lab writes `.ipynb_checkpoints/*-checkpoint.js` next to the file being
edited: `modules-parse` imported one and failed the CPU suite,
`no-import-cycles` fed two into the dependency graph, `af3-source` would have
returned one by basename, and `build_site.py` walked them into the deploy check.
The JS side shares one walker now. `build_site.py` keeps a Python copy with a
comment pointing at it, because a shared rule cannot cross that boundary - worth
saying rather than pretending.

## src/, finished

```
lineages   af2/ 20      af3/ 55     esmfold2/ 17   esmc/ 3
shared     kernels/ 15  heads/ 5    input/ 6       runtime/ 16
           weights/ 5   bundles/ 21
other      design/ 14   testing/ 1
```

`src/weights/` is the last split and the decision behind it is worth recording,
because the first answer was "blocked". The seven weight-decode files in
`runtime/` looked like one group, and moving all seven would have made a cycle:
`execution -> device-pack -> resident -> device-memory`. But **five of them
import nothing at all** - `dtype`, `float16`, `weight-sources`, and
`weight-pack -> float16`, `quantised-upload -> dtype`. Only `resident` and
`device-pack` reach into runtime, and those two are about device RESIDENCY and
accounting rather than about decoding a stored tensor, which is where they
belong anyway. So the five moved and `weights/` is a true bottom layer: **it
imports nothing**, and af3, bundles, esmc, esmfold2, kernels and runtime all
depend on it.

Twelve directories, every name honest about its contents, zero import cycles,
and the layering runs one way: `weights` and `runtime` at the bottom, `kernels`
and the shared services above them, the four lineages on top.

## Markdown was the last blind spot, and the worst one

`test/imports-resolve.test.js` now covers `.md` as well as `.js`, `.mjs`,
`.py`, `.yml`, `.html` and `.sh`. The reorganisation had left **41 stale paths
across nine documents** - and docs are how anything here is FOUND, since
CLAUDE.md's tool table is the index. **A doc naming a module that moved is worse
than code doing it: code fails, a doc just misleads.**

One of the 41 is the example worth keeping in mind.
`src/kernels/attention-webgl2.js` and `src/kernels/triangle/webgl2.js` are 701
lines that **nothing imports**, and a scan for dead code finds them
immediately. They are not dead: docs/DEVELOPING.md says they "exist for
comparison, not as a fallback anything selects", with the measurement that
writing the fragment-shader version is what exposed how much the WGSL kernel was
paying in barriers - the 11.4x. The only thing standing between those 701 lines
and a confident deletion was a sentence in a document, and that sentence was
pointing at a path that no longer existed. **Both files now carry the sentence
in their own headers**, which is where it survives a reader who never opens
docs/DEVELOPING.md - the same repair the four held-out `web/` modules got in the
census below.

### The pattern, stated once

Every gate added this session was one file type behind the last mistake:

| what moved | what caught it | what it could not see |
|---|---|---|
| 51 files under `src/af3` | nothing - found by reading | 129 paths in code COMMENTS |
| comments | a new rule | three `readFileSync` tests |
| those tests | `af3-source.js` helper | python, yaml, html |
| python and yaml | a third rule | paths BUILT FROM SEGMENTS |
| constructed paths | **running `build_site.py`** | markdown |
| markdown | a fourth rule | - |

Two of those are structural rather than another pattern, and they are the ones
that matter: `npm run test:site` **executes** the build, which is the only thing
that can see a path assembled at runtime; and `test/helpers/source-files.js` is
the single statement of "what is a source file", which four walkers had each
learned separately from an editor's snapshots.

## What the reorganisation cost at runtime: nothing, and it is measured

The refactor is thirteen commits touching `src/`, from `66f3fbd` to `36d4784`.
"Behaviour-preserving" was the claim and the gates were the evidence; this is
the question the gates cannot answer, because a gate says pass and not *how
much*. The whole panel at `b9e3eb0` - the commit before the first of the
thirteen - and at HEAD, same script, same sitting, warm being the second of two
folds in one process:

| model | peak before | peak after | warm before | warm after | fold signature |
|---|---:|---:|---:|---:|---|
| AF2 monomer | 386.9 MiB | 386.9 | 0.20 s | 0.22 | **bit-identical** (62.646) |
| AF2 multimer | 442.8 | 442.8 | 0.22 | 0.21 | **bit-identical** (47.582) |
| ESMFold2 | 1511.7 | 1511.7 | 0.87 | 0.87 | **bit-identical** (-128226) |
| AlphaFold 3 | 954.1 | 954.1 | 0.98 | 0.92 | 83.16921321191023 -> ...495311351 |
| RoseTTAFold3 | 956.7 | 956.7 | 0.94 | 1.03 | 81.53339842684814 -> ...809397752 |
| OpenBind-0 | 954.1 | 954.1 | 0.90 | 0.94 | 81.6521641796172 -> ...66033904 |
| Protenix2 | 1180.8 | 1180.8 | 1.09 | 1.10 | 84.66682041314421 -> ...071885165 |
| boltz2 | 1360.2 | 1360.2 | 1.12 | 1.11 | 96.47195216207189 -> ...301422066 |
| OpenDDE | 1810.0 | 1810.0 | 1.31 | 1.31 | 92.03958276090722 -> ...192353464 |
| IntelliFold-2 | 2229.1 | 2229.1 | 1.73 | 1.79 | 83.22122296904186 -> ...278929173 |

🔴 **TEN OF TEN PEAKS IDENTICAL TO THE TENTH OF A MEBIBYTE**, and the same three
labels at the top of each. Warm times move by -6% to +10% with **mixed signs** on
a box this file's neighbour records as drifting up to 3.2x between runs, so the
honest reading is no measurable change rather than a small one. Cold times the
same: 1.85 / 2.26 / 2.15 / 3.08 / 1.93 / 3.60 / 0.73 / 0.73 / 2.50 / 1.51 before
against 1.88 / 2.31 / 2.15 / 2.95 / 1.93 / 3.52 / 0.73 / 0.73 / 2.47 / 1.52.

🔴 **AND THE SEVEN THAT ARE NOT BIT-IDENTICAL ARE ONE COMMIT, WHICH IS NOT A
MOVE.** The differences are in the seventh and eighth significant figure -
1.7e-6 relative at worst - and they are all `9b2cd37`, "two tuning knobs that
could not do what they said". Bisected inside the window, on AlphaFold 3:

| commit | meanPlddt | |
|---|---|---|
| `b9e3eb0` | 83.16921321191023 | before the refactor |
| `b8e2330` | **83.16921321191023** | after all three pure moves - `66f3fbd`, `fc6711c`, `b8e2330` |
| `9b2cd37` | **83.16921495311351** | the knob |
| HEAD | 83.16921495311351 | |

So **the file moves are bit-identical and the arithmetic change is the knob**,
which raises `singleProjectMaxSplits` from a module default of 3 to 6 and so
regroups a sum in `single.project`. A different grouping of a float sum is a
different last digit by construction; CLAUDE.md prices that knob's whole effect
at 2.3% of the trunk's GPU time and nothing on the wall, and this is the
corresponding nothing on the answer. **The three models with no AF3 trunk - AF2,
its multimer and ESMFold2 - are bit-identical through all thirteen commits**,
which is the control: had a move broken something, they are where it would show
without a knob to blame.

### And what it bought

| | before | after |
|---|---|---|
| top-level directories under `src/` | 14 | **12** |
| files directly in `src/af3` | **54** | 4, in six stage directories - `trunk` 19, `diffusion` 11, `featurise` 10, `confidence` 5, `weights` 4, `structure` 2 |
| AF2's lineage | scattered over `evoformer` 10, `model` 4, `multimer` 6, `structure` 7 | one `af2/` of 20, with the shared kernels lifted into `kernels/` 15 |
| `reference/`, which held no CPU reference | 22 files | renamed `bundles/` 21, with `weights/` 5 split out |
| directory-level import cycles | 3 | **0** |
| files / lines | 180 / 71,971 | 181 / 72,214 |

The 243 added lines are headers and comments, including the ones explaining why
four `web/` modules and two WebGL2 kernels have no importer. **No file was
deleted and no line of kernel arithmetic moved** - which is what the table above
is the measurement of, rather than the assertion.

## The census: is every file accounted for?

Asked after the reorganisation, and answered by counting rather than by
impression. **817 tracked files, 0 untracked-but-unignored**; excluding
`web/vendor/`, `web/public/`, `tools/fixtures/` and `oracle-dumps/` - data and
other projects' code - **780**.

| area | files | how each one is reached |
|---:|---|---|
| `tools/` | 334 | a command, now indexed exhaustively in `docs/TOOLS.md` |
| `test/` | 235 | the `test/*.test.js` glob in `npm test`, `*.gpu.test.js` in `test:gpu`, `test/browser/` in `test:browser` |
| `src/` | 181 | imported, or an entry point |
| `web/` | 38 | loaded by a page, or imported by something that is |
| `docs/` | 14 | indexed at the bottom of CLAUDE.md, and `test/docs-indexed.test.js` now checks that both ways |
| root | 13 | the two pages, the configs, the three markdown files |

`tools/`'s 334 reconcile as **304 indexed + 25 under `tools/fixtures/` + 5 that
are neither code nor fixtures** (`esmc/README.md`, `oracle/js_header.txt`,
`oracle/js_tail.txt`, and the two `oracle/reference-conformers*.json`).
`test/`'s 235 are **107 under `test/fixtures/`**, 123 `*.test.js` (25 of them
`.gpu.`, run by `test:gpu` rather than `npm test`), one playwright spec, and
**four shared helpers** - `harness.js` with 76 importers,
`alphafold-references.js`, `helpers/af3-source.js`, `helpers/source-files.js`.

After it, **the only tracked file that nothing else names is `package-lock.json`**,
which is npm's. Every test file is reached by a glob rather than by name, which
is the same thing said differently.

Four things came out of it that were not visible from any single directory.

🔴 **44 TOOLS WERE NAMED BY NOTHING AT ALL** - not CLAUDE.md's table, not
`docs/`, not another tool. Among them `check-af3-opm.js`,
`check-af3-msa-attention.js`, `check-af3-atom-decoder.js` and
`check-chiral-gradient.js`: working differential checkers, each holding a GPU
kernel to its CPU reference, that nobody could find. That is docs/PARITY.md's
complaint from the other end - it counts the checkers that do not RUN on this
box; this counts the ones nobody knows are there. **`docs/TOOLS.md` is the
census, generated by `tools/index-tools.py` and gated by `npm run test:tools`,
and every entry is the first line of the file's own header rather than a
paraphrase** - so an entry cannot drift from the thing it describes, and adding
a tool with no header at all fails the gate. Five tools had no header: the four
`npm run bench:*` benchmarks and `export-web-model.js`, the oldest things here -
and writing one for each turned up that **four of the five cannot run on a fresh
checkout**, which is the whole argument for the gate: the tools nobody had
described were also the tools nobody had run.

`model1-a3m-59-stack` is absent whole. And `model1-query-59-stack`, which IS
present, **declares 530 tensors of which 26 are on disk** - twenty input
features and six geometry tables, no weight of any kind, 328 of the 504 missing
files `*haiku*`. CLAUDE.md said "ten `*_haiku_*.f32.bin` weights are missing"
from it; that is off by a factor of fifty and is corrected. `export:web-model`
dies `ENOENT ... stack_haiku_0024.f32.bin` **out of its own default argument** -
run, not inferred. Only `npm run bench` (`benchmark-triangle.js`) builds its own
input and works.

🔴 **AND `tools/fold-in-page.py` HAD TWO DEAD PATHS FROM THE REORGANISATION,
NEITHER OF WHICH FAILED.** `MANIFESTS = "/src/reference/manifests/index.js"` is
matched against the request path to rewrite the `remote:` line, so when the
module moved it simply stopped matching and `--local-weights` silently fetched
from Hugging Face instead - slower, and dependent on a network the flag exists
to avoid. The other, `await import('/src/reference/http-tensor-store.js')`, sits
inside a `try` that prints `'unavailable: ' + error.message`, so a moved file was
reported as a missing measurement - and not only `hostDecode`: the whole
`af3LoadMilliseconds` block went with it, so `weightPhases` printed one string
where it prints seven numbers.

🔴 **AND THE FIRST HALF IS MEASURED RATHER THAN ARGUED**, because
`--remote-weights` is exactly the state the stale path silently produced. The
first version of this paragraph was one run of each, which is the trap CLAUDE.md
names two sections on - this box drifts up to 3.2x and a single run of each is
not a comparison. Interleaved instead, three rounds, arms alternating:

| arm | trunk weight phase (ms) | wall (ms) |
|---|---|---|
| local | 1005, 973, 985 | 5130, 5140, 5121 |
| remote | 2845, 2793, 3425 | 7010, 6906, 7539 |

**2.89x on the weight phase and 1.9 s of wall**, medians, and the two bands do
not overlap on either column - which is what makes three rounds enough. The
tool's own `--remote-weights` help puts the transfer at ~150 MB. Every run of a
check whose whole point is not to go to the network. **`test/imports-resolve.test.js` could not see
either**, and the reason is the whole lesson: its pattern began
`(?<![/\w-])`, a lookbehind written to stop the rule matching the tail of a
longer path - which also rejects a LEADING SLASH, and a leading slash is exactly
how the browser names a module. The one dialect the gate could not read was the
one the page speaks. It reads both now, and a **directory** without a file on the
end as well, which is how six documents and two tools named
`src/reference/manifests` for a directory that no longer existed.

🔴 **AND SIX `web/` MODULES ARE NOT REACHABLE FROM ANY PAGE IN THE
REPOSITORY, ON PURPOSE.** Of 25 modules, 19 are reachable from `index.html` or
`dev.html`; **five only from `single.html` and `proteinhunter.html`** -
`main.js`, `hunter.js`, `viewer.js`, `mutate.js`, `hydrophobicity.js` - and
`plddt.js` from **neither**, having had no importer at any tracked revision.
`b0dc258` held those two pages out until they can be fixed against a model row
that moved under them.

Two of the five are still exercised: `mutate.js` and `hydrophobicity.js` by
`test/mutate.test.js`, `test/hydrophobicity.test.js` and
`test/browser/webgpu.spec.js`. **The four with no page and no test are
`main.js`, `hunter.js`, `viewer.js` and `plddt.js`**, and each now carries the
explanation in its own header. A dead-code sweep would delete all four, because
the evidence that nothing imports a file is not evidence that nothing needs it -
the same trap as the WebGL2 kernels above, one directory over.

🔴 **AND "NOTHING IMPORTS IT" AND "NO PAGE REACHES IT" ARE DIFFERENT QUESTIONS,
which is worth stating because the first answer here was the wrong one of the
two.** A census keyed on importers reports **three** (`main.js`, `hunter.js`,
`plddt.js`) and misses `viewer.js` entirely - it IS imported, by two files that
nothing reaches. Only a closure from the entry points finds it. Run the closure.

Three files were genuinely dead and are gone: `tools/serve-nocache.py`
(superseded by `tools/serve.py`, and actively worse - it sends `no-store` on the
weight shards too, which is the 346 MiB re-download `serve.py` exists to avoid),
`tools/gpu/_tune.js` (eight lines, superseded by `probe-tuning.js`), and
`tools/gpu/gemm-matrix-staged.js` (a re-export shim left behind when the staged
GEMM moved into `src/kernels/matrix-linear.js`; its three importers now name the
real module).

### Which recorded numbers actually reproduced

🔴 **"EVERY NUMBER IDENTICAL TO THE RECORDED SWEEP" WAS THE FIRST VERSION OF
THIS AND IT WAS NOT TRUE.** Checked row by row against a recorded run rather
than asserted:

| gate | against | result |
|---|---|---|
| `test:ligand` | CLAUDE.md's six bond-rms figures | **6 of 6 exact** |
| `test:template` | docs/AF3.md:4977 - a recorded run of this gate | **7 of 7 exact**, boltz2's 0.837 included |
| `test:portable` / `test:spec-floor` | CLAUDE.md's AF2 checksums | **-1287025 and -1294937, exact** |
| `test:stock` | docs/A100.md's table | AF2 and ESMFold2 exact; **AF3 and OpenDDE do not** |
| `test:batch` | its own verdict line | passes; no per-field figure is recorded to compare |

The two that moved are bisected in docs/A100.md: **AF3 at `28b3965`** (its
diffusion atom encoder read the `_1` pair tensors, denoise 4.19e-1 -> 1.55e-5),
then `7c13e05` and `9b2cd37`; **OpenDDE at `8dba05f`** (three models folding a
single sequence one MSA row short, RMSD 1.527 -> **1.518**). Both are
correctness fixes, both improved agreement with the reference, and **both
commit messages already state the new figure** - so the figures were stale
rather than the gate, and nobody had to notice anything subtle. What failed is
that they lived in prose in three documents. They live in
`tools/gate-baseline.json` now, adapter-keyed and re-checked every run. **This commit cannot be the cause and
that is checkable rather than asserted** - `git show --name-only` puts its whole
`src/` surface at `attention-webgl2.js` and `triangle/webgl2.js`, which the
closure above proves nothing imports, and every added line in `src/` and `web/`
is a comment.

🔴 AND THE LESSON IS ABOUT THE PHRASE, NOT THE NUMBERS. "Identical to the
recorded sweep" was written after reading the values and finding them familiar.
Four of the five rows were exact; the claim was made about all five. **A
comparison is against a recorded value or it is not a comparison** - and where
no recorded value exists, as for `test:batch`, the honest report is that the
gate passed, not that it matched.

**What the census does NOT recommend is a `dev/` directory.** The undocumented
tools are undocumented, not experimental: moving a working differential checker
somewhere marked provisional would make it less likely to be run, which is the
opposite of what `docs/PARITY.md` says this port needs. An index fixes being
unfindable; relocation does not.
