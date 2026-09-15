# What a new model costs, and why src/ wants reorganising

Raised by the user on 2026-09-14, mid-port: "reference in different location,
scattering identical implementations, I think it's making it hard to add new
models and avoid reinventing and also avoid having to reoptimize for each
model."

**This file is EVIDENCE, not a plan.** It was written the same night, from one
port of two models (IntelliFold-2 and RoseTTAFold3), while every place a
convention had to be threaded was still open in an editor. The shape of the
refactor is a decision owed to the user; what follows is the input to it.

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
`src/runtime/weight-sources.js`, beside the symbol it asks. Its gate counts
DECODES through a getter and asserts zero, so the 7x bug - a presence test that
unpacked a 768x1536 int5 tensor per block per sampler step - cannot come back
silently. The ORDER lists themselves are still three copies.

**The pack/offset loop, five copies.** ✅ **CLOSED.**
`packOuterProductMeanWeights`, `packGridAttentionWeights`, `packTriangleWeights`,
`packTransitionWeights`, `packAtomBlockWeights`, `packBlockWeights`. Every one
was: walk a list, reserve an offset, sum a length, then walk it again and copy.
Two of them had the optional-tensor bug; the others did not have optional
tensors YET, which is not a property to rely on.

All six now call `packNamedWeights` in `src/runtime/weight-pack.js`, and the fix
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
| how a dialect flag reaches code, three routes | open — this is the one that wants a decision |

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

  Nothing was wrong today and nothing was changed. What changed is that the
  routing is now a thing a test has an opinion about: adding a fifth read site
  against a new source turns it red, which is the point - **the routing may be
  deliberate, but a change to it must be.** Verified to fail by pointing the
  encoder's read at the block.

None of this is urgent. Every gate is green and both new models fold at the
reference's numbers; the argument for continuing is the measured one at the top
of this file, not a feeling that the code is untidy.
