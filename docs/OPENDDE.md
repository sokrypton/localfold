# OpenDDE: two token spaces, and a fold

OpenDDE (Aureka Research, Apache-2.0) is an independent PyTorch
reimplementation in the AlphaFold 3 family - its own pairformer and its own
primitives, not DeepMind's code at different widths. This is the port's state:
what runs, what it is worth, and what is deliberately absent.

**It folds.** The trunk runs on residues; between the trunk and the diffusion
each standard residue is re-tokenised into a backbone and a sidechain token,
and the diffusion runs on those. `src/af3/fold-opendde.js` is that driver and
`tools/gpu/fold-opendde.js` runs it.

    node tools/gpu-chrome.mjs tools/gpu/fold-opendde.js --target=6mrr

| target | residues -> tokens | RMSD | TM |
|---|---|---|---|
| 6MRR | 68 -> 130 | **1.399-1.639 A** | **0.904-0.917** |
| 1QYS | 92 -> 179 | **0.902-0.956** | **0.939-0.945** |

...at the sampler's own default of 16 steps, over three seeds and two. The
earlier figures here (6MRR 1.678 / 0.865, 1QYS 2.573 / 0.726) were taken at 200
steps and before the pair transition's factor was derived from the weights;
both of those cost real accuracy and both are fixed.

🔴 **THE GEOMETRY IS THE GATE BEFORE THE FOLD IS.** A peptide bond is 3.8 A: a
port with the arithmetic subtly wrong produces a plausible cloud at the wrong
scale, and an RMSD alone does not say which. TM above 0.5 is the same fold.

**`foldingModel` is still `false`**, because the PAGE has no route to this
driver - it folds through `foldBatch`, which would hand the diffusion residue
tokens where it wants structural ones. Every shape conforms and a structure
comes out, so that branch has to land before the flag does.

## What it does

`tools/gpu/trunk-opendde.js` runs the trunk and the distogram head, sequence in
and contact map out, and scores that map against a deposited crystal.

    node tools/gpu-chrome.mjs tools/gpu/trunk-opendde.js
    node tools/gpu-chrome.mjs tools/gpu/trunk-opendde.js \
      --model=/model-af3-int5/manifest.json     # the control

🔴 **THE CONTACT MAP IS THE ONLY END-TO-END STATEMENT THIS PORT CAN MAKE, AND
IT IS NOT AN ORACLE.** There is no dump of OpenDDE's own intermediates to
compare against and no coordinates to superpose. What there is instead is a
claim about a real structure: a trunk assembled with the wrong width, the wrong
head count, the wrong bin grid or a branch inverted does not predict a fold's
contacts by accident. On 6MRR, at MSA depth 1, against real atomic contact -
any heavy-atom pair under 5 A, which `tools/calibrate-contact-cutoff.py`
established here as the ground truth:

| | precision | recall | p@N | chance |
|---|---|---|---|---|
| AlphaFold 3 (the control) | 0.753 | 0.824 | **0.800** | 0.043 |
| **OpenDDE** | 0.605 | **0.953** | **0.718** | 0.043 |

p@N is the strongest 85 pairs, 85 being how many the crystal has, so it does
not move when a threshold does. OpenDDE predicts MORE contacts than AlphaFold 3
for the same target - 134 against 93, for 85 real ones - so it is the less
conservative of the two here: higher recall, lower precision.

🔴 **AND AF3 THROUGH THE SAME HARNESS IS WHY THOSE NUMBERS MEAN ANYTHING.**
Without the control, "OpenDDE scores 0.72" has nothing to be good or bad
against, and a harness fault that flattered both would be invisible. Chance is
0.043 rather than 0 because a fifth of the eligible pairs of a 68-residue chain
really are in contact.

🔴 **AND THE SEQUENCE COMES OUT OF THE CRYSTAL, NOT OUT OF THE TOOL.** The
first version carried a 6MRR sequence typed from memory. It was a different
protein from the deposition beside it, and it folded, and it scored, and it
reported numbers. Both now come from one file. The altLoc rule matters too:
6MRR's chain A has 71 CA records for 68 residues.

## The widths, which are the whole reason the loader changed

Every width in `src/af3/weights.js` used to be a number typed next to the
tensor it describes. A declared width is right for exactly one checkpoint and
silent for every other: this bundle loads through `pairChannels: 128` without
complaint and dispatches every kernel over a third of its own tensor.

Read off the blob, and independently stated in upstream's
`model_registry.OPENDDE_SETTINGS`:

| | AlphaFold 3 | OpenDDE |
|---|---|---|
| pair track | 128 | **384** |
| MSA track | 64 | **128** |
| triangle attention, trunk / MSA / confidence | 4 x 32 | **12 x 32** |
| ...template stack | 4 x 16 | **2 x 32** |
| single track | 384 | 384 |
| distogram bins | 64 | **96** |
| target_feat, relative encoding | 447, 139 | 447, 139 |

🔴 **AND THE MSA VALUE DIMENSION IS THE ONE THAT PROVES THE METHOD.**
AlphaFold 3 is 64 channels and 8 heads with a value dim of 8, so `channels /
heads` is also 8 and that checkpoint cannot tell the two rules apart. OpenDDE
is 128 channels and 8 heads with a value dim of **still 8** - upstream calls it
a decoupled per-head width - so dividing gives 16 and reads twice its own
tensor. Only `v_projection`'s own shape says so.

## The dialect

`src/af3/dialect.js`. OpenDDE and OpenBind-0 are both OpenFold3 by lineage and
**disagree in opposite directions on the two flags they share**, which is the
whole reason that table exists:

| flag | AF3 | OpenBind-0 | OpenDDE |
|---|---|---|---|
| `swapTransposedBias` | false | false | **true** |
| `padSingleCondUnknownDna` | false | **true** | false |
| `symmetriseBonds` | false | true | true |
| `maskPaddedKeys` | false | true | true |
| `pairInitFromSingle` | false | false | **true** |
| `msaUpdateBeforeOuterProduct` | false | false | **true** |
| `distogramBias` | false | false | **true** |
| `keyMaskedAtomAttention` | false | false | **true** |
| `perBlockPairLayerNorm` | false | false | **true** |
| `perBlockAtomPairLayerNorm` | false | false | **true** |
| `chainedAtomLayerNorm` | false | false | **true** |
| `splitPairConditioning` | false | false | **true** |
| `structuralTokens` | false | false | **true** |

🔴 **`swapTransposedBias` IS ON HERE AND OFF FOR OpenBind-0.** Upstream's
`TRANSPOSED_COLUMN_PAIR_BIAS` lists opendde and deliberately omits openbind.

🔴 **AND `padSingleCondUnknownDna` IS OFF HERE AND ON FOR OpenBind-0, EVEN
THOUGH OpenDDE's NATIVE CONDITIONING IS 833 WIDE EXACTLY AS OpenFold3's IS.**
Its converter collapses that to 831 by remapping the 32-class vocabulary onto
AF3's 31 rather than padding it (`converters/opendde.py`,
`_remap_s_inputs_vec`), so what reaches this graph is 831 and stock AF3's
arithmetic is correct. Reading "833 in the checkpoint" as "pad the
conditioning" would LayerNorm over two columns the converter already folded
away. **Lineage is provenance; a convention is a separate question.**

### The three branches that are implemented and checked

* **`pairInitFromSingle`** - the pair track is built from the single embedding
  `s_init`, not from target_feat, so `single_activations` is computed BEFORE
  the pair init instead of after the MSA stack. The shapes state it:
  `left_single` is [447, 128] under AF3 and **[384, 384]** here. And it is
  s_init WITHOUT the recycled term, which is identical on the first pass and a
  different model on every pass after it.
* **`msaUpdateBeforeOuterProduct`** - an MSA block updates the rows FIRST and
  feeds the updated MSA to the outer product; AlphaFold 3 takes the outer
  product off the pre-update MSA. Same modules, same weights, different
  function, compounding over the blocks.
* **`perBlockAtomPairLayerNorm`** - the atom encoder that builds 384 of
  target_feat's 447 columns normalises its pair conditioning inside every block
  rather than once for the stack. The checkpoint states it twice: the scale is
  [3, 16] rather than [16], and haiku names the enclosing stack
  `__layer_stack_no_per_layer` rather than `__layer_stack_with_per_layer`, so
  reading either convention without the other finds no tensors at all.

* **`chainedAtomLayerNorm`** - the atom cross-attention's two adaptive
  LayerNorms are CHAINED: `a = layernorm_a(a, s)` then
  `kv = layernorm_kv(a, s)` reading the ALREADY-NORMALISED a, so the gather
  onto the key layout happens between them and the second norm sees the first
  one's learned scale. AlphaFold 3 normalises the raw activation twice, once
  per side. The GPU's key projection fuses its own normalisation into itself,
  so chaining means handing it a pre-normalised activation - `normaliseQueries`
  writes `adaLN_q(act)` and the caller binds that in place of `act`, while the
  QUERY projection still reads the raw one.
* **`keyMaskedAtomAttention`** - the attention's mask bias is a SUM rather than
  AlphaFold 3's PRODUCT, so a real query cannot attend to a padded key at all
  where AF3 penalises a pair only when both ends are padded.

`test/opendde-branches.test.js` runs BOTH arms of the first two over one set of
weights and asserts they differ, because a flag that never reaches the
arithmetic agrees with itself. `test/opendde-shaders.test.js` does the same for
the GPU arms by asserting on the generated WGSL - comparing cache keys is not
comparing kernels, which this repository records a unit test passing for while
the fold came back NaN.

### What the last two branches are worth, measured

| branch | moves the trunk | moves the contact map |
|---|---|---|
| `chainedAtomLayerNorm` | pairRms 21.6072 against 21.6139, single 12.8431 against 12.8054 | **no** - 0.605 / 0.953 / 0.718 either way |
| `keyMaskedAtomAttention` | **nothing at all**, to every digit | no |

🔴 **AND THE SECOND ONE IS INERT BY CONSTRUCTION, NOT BY LUCK.** The key window
is `min(128, atomCount)`, so no key this featuriser produces is ever padded,
`mask_k` is identically one and the two forms agree exactly. It is implemented
because the differential checkers do NOT use this featuriser - they feed
AlphaFold 3's own gathers out of an oracle dump, where padded keys do occur -
and the same reasoning is already recorded here for `maskPaddedKeys`.

🔴 **AND A NULL RESULT IS ONLY WORTH HAVING IF THE FLAG REACHED THE KERNEL**,
which is why both rows above are backed by a WGSL assertion rather than by the
measurement alone. The first two ablations run here were silently dropped: the
atom blocks carry their OWN copy of these flags, stamped at load time, so
changing the caller's dialect left the bundle's own answer in place and both
arms reported the same number - which is also what "the branch does not matter"
looks like.

### The distogram bias, which is applied twice

Stock AF3's `half_logits` is bias-free and OpenDDE's is not. This head
symmetrises by adding its own transpose, so a bias added once per half is added
**twice per logit** - and that is what OpenDDE trained with, because it
symmetrises after its own linear exactly as this graph does. Two of the four
families upstream lists need theirs HALVED instead, because their natives
symmetrise the pair first. The loader cross-checks the tensor's presence
against the dialect and refuses a bundle where they disagree.

🔴 **AND THE BIN EDGES ARE BORROWED, WHICH IS WORTH SAYING PLAINLY.** Upstream
changes only the COUNT - 96 against 64 - and leaves the two breaks at
AlphaFold 3's 2.3125 and 21.6875, so OpenDDE's grid is 95 edges over the same
span: a finer grid of the same reach, spacing 0.206 A against 0.313. That is
upstream's reading and the only stated source; OpenDDE's published
`config.json` is metadata and carries no distogram range. The contact result
above is the evidence that it is not badly wrong - a wrong range would destroy
the precision first. **Open**, in the sense that a better source would settle
it.

## What does not transfer

🔴 **THE DIFFUSION RUNS ON AN EXPANDED TOKEN SET, AND THAT IS NOT A BRANCH.**
Between the trunk and the diffusion OpenDDE expands each residue into about two
"structural tokens" - a backbone token and a sidechain one, glycine staying
single - and runs the diffusion AND a confidence head of its own design on that
expanded set. `structural_token_expander` is 17 tensors and
`structural_token_refiner` 51; the confidence head is 51 more under
`confidence_head/pairformer_stack` with none of AlphaFold 3's names. That is a
second token space threaded through the atom layouts, not a flag.

The blob says how far apart the two models are:

| | |
|---|---|
| arrays in OpenDDE's blob | **481** |
| ...in AlphaFold 3's | 406 |
| sharing a name | 240 |
| ...and also a shape | **123** |

By group, of the 240 shared: the trunk pairformer, the MSA stack and the
template embedder share every name and differ only in width; the diffusion
shares 51 of 53; the confidence head shares nothing usable.

So the port is a second token space threaded through the atom layouts, which
is what `src/af3/structural-tokens.js` and `src/af3/fold-opendde.js` are. The
bundle is the whole model: 481 tensors, 655.8 M parameters, upstream's own
published `parameter_count` exactly.

🔴 **AND THE CONFIDENCE HEAD IS PORTED, AS ITS OWN PARAMETRISATION.** It shares
not one tensor name with AlphaFold 3's: it initialises its pair from `s_inputs`
as a row and a column, adds a distance embedding of the structure the SAMPLER
PRODUCED - so unlike AF3's it cannot run before the sampler - runs four
pairformer blocks of its own shape, and reads pLDDT and experimentally-resolved
as a per-ATOM einsum against a [24, c_s, bins] tensor selected by the atom's
dense SLOT. Broadcasting that instead of selecting gives every atom of a token
the same pLDDT, which looks right on a backbone and is wrong everywhere else.

Measured on 6MRR against the deposition, with AlphaFold 3 through the same tool
as the control:

| | RMSD | TM | mean pLDDT | pLDDT vs error |
|---|---|---|---|---|
| AlphaFold 3 | 0.642 A | 0.956 | 84.21 | **-0.405** |
| **OpenDDE** | 1.655 | 0.885 | **92.05** | **-0.186** |

The last column is Spearman of per-residue pLDDT against per-residue deviation
after superposition, so NEGATIVE is the head working - confident where the
error is small. Both are negative and AlphaFold 3's is the stronger, on a fold
that is also the better one. **OpenDDE is the more optimistic of the two**: it
reports 92 on a 1.65 A fold where AlphaFold 3 reports 84 on a 0.64 A one. On a
target this good there is little error to rank, which docs/EF2FAST.md records
as the reason its own certainty sweep had to manufacture a hard end - so read
-0.19 as "the head is wired correctly", not as a calibration.

🔴 **AND WHAT IT STILL DOES NOT REPORT IS pTM.** That needs a TM term;
AlphaFold 3's head emits one beside the PAE and OpenDDE's emits pLDDT, PAE, PDE
and experimentally-resolved and nothing else. Deriving one from the PAE would
be a different quantity wearing pTM's name, so pTM and ipTM are ABSENT rather
than approximated.

🔴 **AND NOTHING CAN RUN THE ABSENT HALVES BY ACCIDENT.** `diffusionWeights`
and `confidenceWeights` on this bundle both refuse by naming the first tensor
they cannot find, rather than loading a partial graph - which is the structural
gate doing its job, and the reason four of the dialect's flags
(`perBlockPairLayerNorm` on the token transformer, `splitPairConditioning`, and
the two heads above) are declared and unreachable rather than declared and
silently ignored.

## The bundle

    # 2.47 GB, pinned - a blob from a moving branch can change under a bundle
    # that did not, and the failure names neither half
    mkdir -p ~/af3_ported && cd ~/af3_ported
    curl -sSLO https://huggingface.co/sokrypton/af3-any-model/resolve/\
      13db85d4867fd0d7f7d91f24f9e20c36eca78004/opendde/opendde.bin.zst

    python3 tools/export_af3_model.py --model opendde --out model-opendde-trunk-f32
    python3 tools/quantize_af3.py --source model-opendde-trunk-f32 \
      --out model-opendde-trunk-int5
    python3 tools/write_manifest_module.py opendde

| | |
|---|---|
| float32 | 214 tensors, 387.0 M parameters, **1476 MiB** |
| int5 group 32 | **278.1 MiB**, 5.31x |

Larger than AlphaFold 3's whole 265 MiB bundle for a trunk alone, because the
pair track is three times as wide.

🔴 **IT IS NOT PUBLISHED.** The registry entry has a `directory` and no
`remote`, so `build_site.py --is-remote opendde` exits 1 and the Pages workflow
would try to publish 278 MiB it has no allowance for. Hosting it is one line -
see docs/HOSTING.md - and is a decision about somebody's Hugging Face account
rather than a code change.

🔴 **AND QUANTISATION IS NOT WHAT LIMITS IT.** The contact numbers above are
int5; the float32 bundle scores the same. That was measured first, because
"the port is wrong" and "int5 is too coarse for this checkpoint" look identical
from a bad contact map.

## The three branches the diffusion needed

| | AlphaFold 3 | OpenDDE |
|---|---|---|
| `pair_cond_initial_norm` | [267] = trunk pair 128 + RAW relative 139 | **[256]** = two compressions of 128 |
| token transformer pair norm | shared, [128] beside the stack | **per block, [6, 4, 128] inside it** |
| atom stacks' pair norm | shared, [16] beside the stack | **per block, [3, 16] inside it** |

🔴 **THE PAIR CONDITIONING IS TWO COMPRESSIONS, NOT ONE CONCATENATION.**
AlphaFold 3 concatenates the trunk pair with the RAW relative encoding and
LayerNorms the lot; OpenDDE compresses each to the pair width SEPARATELY -
`z_trunk_projection` [384, 128] and `relpe_projection` [139, 128] - and
concatenates those. The joint norm over the widened concatenation couples the
two terms, so this is a different function and not a re-association. It does
not depend on the noise level, which is why the head already cached it across
the sampler's steps and why `pairConditioning` on its input is enough.

🔴 **AND THE STACK'S NAME FOLLOWS THE CONVENTION.** haiku calls a layer stack
`__layer_stack_no_per_layer` when it carries no per-layer inputs, so a per-block
pair norm moves the PATH as well as the RANK - and reading one convention
without the other finds no tensors at all rather than the wrong ones.

🔴 **AND THE TOKEN TRANSFORMER'S PER-BLOCK NORM IS WHAT MADE IT COLLAPSE.**
Loaded and not applied, its [6, 4, 128] scale was read as [128] and its
[6, 4, 128, 16] projection at AlphaFold 3's [6, 128, 4, 16] layout - the same
element count, a different meaning. The fold came out finite, shape-correct and
with a radius of gyration of **0.27 A**: atoms placed sensibly WITHIN a token,
and tokens not placed relative to one another at all. That is exactly what a
garbage pair bias does, and it is why the symptom localised the cause.

🔴 **AND IT NEEDED NO KERNEL, BECAUSE A PER-BLOCK SCALE IS A PER-BLOCK
PROJECTION.** This LayerNorm has no offset, so `LN(z) * scale_b @ W_b` is
`LN(z) @ (diag(scale_b) W_b)` exactly - the mean and variance it removes do not
depend on the scale. The scales fold into the projections in the loader,
transposing from OpenDDE's [block, channels, head] to AlphaFold 3's
[channels, block, head], and the shared-scale kernel then computes OpenDDE.

🔴 **AND THE WHOLE DIFFUSION IS SHAPE-IDENTICAL TO AlphaFold 3's APART FROM
THOSE TWO CONDITIONING TENSORS**, which is what said the remaining fault had to
be a forward branch rather than a width, and stopped the search being a sweep.

## What went wrong on the way, and what it cost

🔴 **THE DISPATCH WAS SIZED FOR 128 CHANNELS WHILE THE KERNELS WERE COMPILED
FOR 384.** `compilePairTrack` took the width; `encodePairTrack` defaulted it to
`PAIR_CHANNELS`. Two thirds of every pair row went unprocessed, with nothing
out of range and no validation error, and the contact map scored BELOW chance -
precision 0.034 against 0.044, with **zero** true contacts among the 85
strongest. Every kernel in the track measured 6e-7 against its own CPU
reference at OpenDDE's widths the whole time.

This is the second time a shape resolved in two places has done this here;
CLAUDE.md's closing habit records the first. Both functions require the width
now - a default is what let the two drift.

🔴 **AND THE BLOCK-LEVEL DIFFERENTIAL IS WHAT FOUND IT, NOT THE FOLD.** The
per-kernel checkers all passed. `tools/gpu/check-af3-block-any.js` runs a whole
pairformer block against `pairformer-reference.js` THROUGH THE LOADER, so the
widths are the bundle's: it read **1.293** for OpenDDE against 3.7e-3 for
AlphaFold 3, which is the difference between "assembled wrongly" and "rounding".
`tools/gpu/probe-opendde-kernels.js` then attributed it, by running each of the
five pair updates alone at the bundle's widths.

🔴 **AND EVERY EXISTING PER-KERNEL CHECKER IS PINNED TO AlphaFold 3's
CONSTANTS.** `check-af3-triangle.js` has `const CHANNELS = 128`,
`check-af3-grid-attention.js` has 128 with 4 heads of 32, `check-af3-block.js`
hand-builds its weight dict with `heads: 4, pairChannels: 128` typed into it.
The whole differential suite is blind to a second bundle's widths, which is
exactly where a second bundle breaks. The two new tools go through the loader
instead.

🔴 **AND THE GRID ATTENTION'S BIAS KERNEL GENERATED NOTHING AT TWO HEADS.** It
vectorises four heads to a `vec4` as `Array.from({ length: heads / 4 })`, which
is EMPTY at OpenDDE's two-head template stack - so `main` touched none of its
bindings, and WebGPU infers a bind group layout from the bindings a shader
USES. The layout came back with one entry against a bind group of three and the
error named neither the kernel nor the head count. Every head count in this
repository before OpenDDE was 4, 8, 12 or 16. There is a scalar arm now, and a
guard that refuses to emit no body.

🔴 **AND A 384-CHANNEL PAIR TRACK NEEDS A WORKGROUP LIMIT ALPHAFOLD 3 NEVER
CAME NEAR.** The grid projection is one thread per pair channel, and the spec
defaults are 256 in X and 256 invocations per workgroup while this adapter
reports 1024 for both. `maxComputeWorkgroupSizeX` and
`maxComputeInvocationsPerWorkgroup` are requested now. Both are needed: the X
extent alone is not enough, because the total is capped separately.

🔴 **AND `store.shape` THROWS ON A MISSING TENSOR** rather than returning
undefined, which is right for every caller but the one optional tensor in the
tree - the distogram bias.

## AlphaFold 3 is unmoved

The same 40-mer folds to mean pLDDT **72.19283791929007** and pTM
**0.5096721043810248** on this tree and on the tree before any of this work, to
every digit. `npm test` is 833 passing.

## Where a fold's time and memory go

Measured on 6MRR, 68 residues becoming 130 structural tokens, at 50 sampler
steps. `tools/gpu/fold-opendde.js` prints both.

| | |
|---|---|
| peak | **1198.3 MiB** |
| whole fold | 20.4 s |
| trunk (48 blocks) | 3.6 s |
| structural refine | 0.8 s |
| structural expand | 0.065 s |
| **the sampler** | **~15.6 s, 77%** |

🔴 **THE PEAK IS 90% RESIDENT TRUNK WEIGHTS, AND THE SAMPLER IS 77% OF THE
TIME - SO THEY ARE TWO DIFFERENT PROBLEMS.** `peakByLabel` reads
`w.pair-transition` 324 MiB, `w.grid` 272, `w.tri.out` 163, `w.tri.in` 163,
`w.single-transition` 162: 1084 of 1198, against 34 MiB of scratch. That is
three times AlphaFold 3's because these weights go as the SQUARE of the channel
count and the pair track is 384 rather than 128.

🔴 **AND THE SAMPLER IS EXPENSIVE FOR A REASON NO KERNEL WILL FIX: IT RUNS ON
130 TOKENS WHERE AlphaFold 3 RUNS ON 68.** The denoiser's token transformer is
quadratic in them, so OpenDDE pays about four times AF3's sampler cost for the
same protein. The structural expansion is what buys the accuracy; it is also
what costs the time.

### The sampler: diffusion, sixteen steps

🔴 **THE FLOW ARM IS A LOSS AT THE SAME PRICE, WHICH IS THE OPPOSITE OF WHAT IT
IS FOR AlphaFold 3.** On 6MRR at sixteen steps, two seeds each:

| | TM | RMSD | time |
|---|---|---|---|
| **diffusion-16** | **0.9044, 0.9169** | 1.498, 1.399 | 16.1 s |
| flow-16 | 0.8307, 0.8601 | 1.828, 1.641 | 16.1 s |

AlphaFold 3 prefers flow because its diffusion default is 200 steps and flow-16
reaches it in a twelfth of the calls. OpenDDE's sampler is already best at
SIXTEEN, so a flow arm has nothing to escape from - and escaping the re-noising
is what loses the structure. The mode row is hidden for this family and the
value FORCED, because hiding a control does not change it: the shared
`#af3-mode` select still reads "flow" behind a hidden row, which
docs/EF2FAST.md records catching an hour after hiding its own.

🔴 **AND THE PAGE HAD BEEN RUNNING THE WORSE ONE.** Every measurement in this
section was taken in diffusion mode through the shell tool, while the page
defaulted to flow - so the deployed site folded OpenDDE at TM 0.83-0.86 where
it can do 0.90-0.92, for the same twenty seconds. Measure the arm the PAGE
runs, not the one the tool defaults to.

### More sampler steps are WORSE, on two targets and nine folds

TM against the deposition, one column per seed:

| | 16 steps | 100 steps |
|---|---|---|
| 6MRR | **0.9044, 0.9169, 0.9039** | 0.8884, 0.8828 |
| 1QYS | **0.9449, 0.9394** | 0.9285, 0.9319 |

and on 6MRR the whole ladder, one seed: 16 -> 0.9146, 25 -> 0.8925,
50 -> 0.8887, 100 -> 0.8785.

**The ranges do not overlap on either target**, and 16 is 32-40% faster
(15.9 s against 26.3 on 6MRR, 29.6 against 43.2 on 1QYS). `OPENDDE_COUNTS`
carries 16 as the preferred value; the AlphaFold 3 settings stay on the ladder
so they remain selectable and comparable.

🔴 **AND THE SEED SPREAD IS WHAT MAKES THAT A RESULT.** Three seeds at 16 steps
and two at 100: every 16-step fold has a better TM than every 100-step fold and
the ranges do not overlap, on a quantity whose seed-to-seed spread within an arm
is 0.013. It is 40% faster as well. docs/EF2FAST.md records the same shape for
that model - "more steps than the schedule buy nothing" - so this is the second
time here. **One target, so read it as a direction and not a margin**; the
default is unchanged pending more.

### The memory regime INVERTS with length, and both are worth knowing

Every optimisation above was measured at 68 residues. At 200 it is a different
machine:

| | 68 residues, 130 tokens | 200 residues, 384 tokens |
|---|---|---|
| peak | 647.6 MiB | **1507.4 MiB** |
| whole fold, 16 steps | 16.0 s | **129.5 s** |
| resident trunk weights | **90% of the peak** | ~9% |
| pair scratch | 34 MiB | **1080 MiB, 72%** |

🔴 **SO THE SMALL-PROTEIN WINS ARE REAL AND THEY ARE NOT THE WHOLE PICTURE.**
The trunk's block weights are a CONSTANT - about 1084 MiB in f32, 542 in f16 -
so at 68 residues they are everything and at 200 they are a ninth. The five
pair-sized scratch tensors are quadratic in STRUCTURAL tokens, and OpenDDE has
about two of those per residue, so they overtake the weights somewhere near 150
residues. Both decisions above still help at 200 (they remove a fixed 542 MiB
that would otherwise sit on top of the scratch); neither is the lever there.

🔴 **AND THAT PUTS OpenDDE's CEILING AT ROUGHLY HALF AlphaFold 3's, BY
CONSTRUCTION.** The structural expansion doubles the token count and the scratch
is quadratic in it, so a 300-residue chain is about 576 tokens and its scratch
alone extrapolates to 2.4 GiB. AlphaFold 3 folds 1530 tokens on this device;
OpenDDE reaches about 250 residues. The expansion is what buys the accuracy - it
is also what costs the memory, and there is no setting that separates them.

🔴 **AND CHUNKING THE SCRATCH IS THE ONLY LEVER LEFT AT THAT END, WHICH THIS
REPOSITORY HAS ALREADY PRICED AND REJECTED.** CLAUDE.md records it: the grid
attention takes a row chunk for free, the TRIANGLE will not - its intermediates
are channel-major and the incoming direction needs a strided column slice - and
the two share the allocation, so chunking one leaves the peak where it was. It
is all-or-nothing and the "all" is a restructure of the pair track.

### The peak, 1198 -> 648 MiB, in two decisions keyed on one number

Both are the same trade - what the trunk's block weights cost against what they
buy - and both flip between AlphaFold 3's 128 channels and OpenDDE's 384,
because these weights go as the SQUARE of that. `WIDE_PAIR_TRACK` is the
threshold, at 256, and it is a line between two measured points rather than an
optimum.

| | peak | time |
|---|---|---|
| as first written | 1198.3 MiB | 16.1 s |
| f16 resident pair weights | 873.5 | 16.1 |
| **...and not resident at all** | **647.6** | **16.0** |

**-46%, and the structure does not move**: RMSD 1.570 and TM 0.9146 at every
step. AlphaFold 3 takes neither and is bit-identical (mean pLDDT
72.19283791929007), AlphaFold 2 is unmoved (checksum -2105827).

### Residency buys a second pass and costs the whole peak

| recycles | resident | non-resident |
|---|---|---|
| 0 | 873.5 MiB, 16.1 s | **647.6 MiB, 16.1 s** |
| 1 | 894.4, 19.5 | **647.6, 19.6** |
| 3 | 894.4, 26.4 | **647.6, 26.8** |

🔴 **247 MiB TO BUY AT MOST 0.4 SECONDS.** Residency exists so a SECOND trunk
pass does not re-upload 48 blocks - and at OpenDDE's widths the re-upload is
1.5% of a three-recycle fold while holding them is 28% of the peak. AlphaFold 3
keeps its residency: its block weights are a ninth of these, and its peak is
the diffusion transformer regardless, so there is nothing to buy.

### f16 resident pair weights: 27% of the peak, taken

| | peak | time | RMSD | pLDDT |
|---|---|---|---|---|
| OpenDDE f32 | 1198.3 MiB | 20.4 s | 1.680 | 92.051 |
| OpenDDE `--pair-weights=f16` | **873.5** | 20.4 | 1.681 | 92.051 |
| AlphaFold 3 f32 | 476.0 | 4.2 | 0.682 | 85.621 |
| AlphaFold 3 f16 | 476.0 | 4.1 | 0.683 | 85.639 |

Free in time and in accuracy, worth 27% of OpenDDE's peak and NOTHING of
AlphaFold 3's - whose peak is the diffusion transformer's 378 MiB, not the
trunk's.

🔴 **AND MAKING IT A GLOBAL DEFAULT PRODUCED NaN, WHICH WAS A PACKING BUG AND
NOT AN ARITHMETIC ONE.** Bisected across the five stacks that run this track -
the trunk pairformer, the MSA stack, the template embedder, the structural
refiner and the confidence head - it was the MSA stack, and the template
embedder was quietly wrong beside it (finite, but RMSD 2.009 against 1.570).

Both call `packPairTrackWeights` WITHOUT a precision while `compilePairTrack`
generates kernels that read one. So an f16 kernel read f32 bytes as pairs of
halves. Same shape as the dispatch sized for 128 channels against kernels built
for 384, and the same cause: one decision made in two places. Both now read a
single local, and at f16 the MSA stack is finite and RMSD 1.570 - identical to
f32.

🔴 **AND IT IS ON WHERE IT PAYS, BY WIDTH RATHER THAN BY NAME.** `foldBatch`
takes f16 when the pair track is at least 256 channels. AlphaFold 3 at 128
gains NOTHING - its peak is the diffusion transformer - and at f16 its fold
shifts (mean pLDDT 72.19283791929007 -> 72.17922675038298), which is well
inside a seed's spread and still a change for no gain. 256 is not a measured
optimum: the two points are 128 (nothing) and 384 (27%), and it should move
when a third exists.

🔴 **AND f16 ON THE MSA AND TEMPLATE STACKS IS FREE AND WORTHLESS, MEASURED.**
Now that their packing agrees with their kernels, both are finite and identical
to f32 (RMSD 1.570, TM 0.9146) - and the peak does not move at all, because
their weights are not AT it: the fullest moment is inside the 48-block
pairformer, by which time the MSA stack's four blocks and the template's two
have been released. A saving that is not at the peak is not a saving, which is
this file's own recurring lesson one stack further along.

🔴 **AND THE GRID'S 272 MiB STILL DOES NOT TAKE IT.** `w.tri.out`,
`w.tri.in` and `w.pair-transition` are passed `pairWeightPrecision` and
`w.grid1`/`w.grid2` are not, which is exactly why the observed saving is 325
MiB rather than 461. Narrowing them needs the grid shaders to read f16 weights,
which they cannot today.

## State, for whoever picks this up

Live at localfold.org and in the model row. The bundle is 481 tensors and
655.8 M parameters - upstream's own published count - at int5 in twelve shards
(473 MiB), hosted at `sokrypton/localfold`, pinned to
`d9e5e9c3cdaf941f5dd3e57ccdffdcd98e81bff8`.

**The local export is NOT in the checkout** and is gitignored in all four
shapes. To get it back:

    hf download sokrypton/localfold --include 'opendde-int5/*' --local-dir /tmp/dde-dl
    ln -sfn /tmp/dde-dl/opendde-int5 model-opendde-int5

...or rebuild from `~/af3_ported/opendde.bin.zst` through
`tools/export_af3_model.py --model opendde --include diffuser` and
`tools/quantize_af3.py --shards 12`.

**The gates**, all of which must hold:

| | |
|---|---|
| AF3 fold, bit-identical | mean pLDDT **72.19283791929007**, pTM **0.5096721043810248** |
| AF2 checksum | **-2105827** at 128 rows, **-2047044** at 512 with a recycle |
| `npm test` | **866** |
| OpenDDE, 6MRR | RMSD **1.399-1.676 A**, TM **0.884-0.917** |
| OpenDDE, 1QYS | RMSD **0.902-1.052**, TM **0.925-0.945** |
| AlphaFold 3 chemistry, the control | protein bond ratio **1.009**, ligand rms **0.033 A** |
| pLDDT ranking, `plddtVsError` | AF3 **-0.3484**, OpenDDE **-0.3063** on 6MRR |

...and the one that was missing, which is why the section below exists: a gate
on RMSD and TM cannot see a bond length.

**The other three models must stay bit-identical**, and are the gate on any
change to `src/af3/featurise.js`, which all of them share:

| | |
|---|---|
| ESMFold2, `tools/gpu/fold-esmfold2.js` | CA-CA **3.8060627434251515**, certainty **0.7924301467835904**, 27/27 contacts, peak 272.76 MiB |
| OpenBind-0, `fold-opendde.js --model=/model-openbind0-int5/manifest.json` | pLDDT **77.3932004390278**, RMSD **1.963**, TM **0.82** |
| AlphaFold 3, `fold.js --sequence=<6MRR>` | mean pLDDT **85.93504804019729**, pTM **0.7368081900126794** |
| a ligand, `probe-ligand-flow.js --ligand=GOL --mode=diffusion --steps=64` | rms **0.03339435515711982** |
| a modified residue, `probe-modified.js --code=SEP --at=3` | **0.838** against control **1.003** |

🔴 **AND `tools/fold-in-page.py --model af3` / `--model opendde` IS THE ONE
THAT CATCHES WHAT THE TOOLS CANNOT.** Three page-level TypeErrors in this port
were found by driving the real page and by nothing else.

`tools/gpu/fold-opendde.js` is the end-to-end tool and takes `--target`,
`--length`, `--steps`, `--mode`, `--seed`, `--resident` / `--no-resident`,
`--pair-weights` and `--model` (AlphaFold 3 runs through it as the control).

## Open, in the order worth doing

🔴 **THE 15% CHEMISTRY DEFICIT IS THE FIRST THING WORTH DOING, AND IT IS A
CORRECTNESS BUG RATHER THAN A LIMITATION.** Measured, controlled and localised
two sections below: the atom geometry inside a structural token is ~15% short
while the peptide bond between two of them is 1.03, so the suspect is the atom
decoder over the structural layout, not the trunk, the tokeniser or the
sampler. `probe-ligand-flow.js --ligand=GOL --mode=diffusion --steps=64` is a
twenty-five-second reproduction; AlphaFold 3 through the same tool is 0.033 A
where OpenDDE is 0.349.

🔴 **AND THE CONFIDENCE HEAD'S LEVEL IS WORTH ONE CHECK, NOT AN ALARM.** Its
within-fold ranking is -0.31 where AlphaFold 3's on the same target is -0.35,
so it tracks error about as well as the reference does; what differs is the
level. The one port-side suspect is `plddt_weight`, which is `[24, c_s, 50]` -
indexed by an atom's DENSE SLOT. Under the structural layout a slot is the
atom's position within its STRUCTURAL token, so a sidechain token's first atom
reads slot 0's matrix where a residue layout would give that matrix to N. That
is either right or badly wrong depending on which space upstream indexes, and
like everything else on the diffusion side it needs a dump to settle.

🔴 **AND THE 250-RESIDUE CEILING IS WHAT A USER MEETS FIRST**, before any
optimisation matters to them. See the regime section above: it is the
structural expansion, it is quadratic, and the only lever left is the pair-track
chunking CLAUDE.md already priced and rejected. The atom stack's padding has
since been removed - docs/PERF.md prices it - and it moved the peak at 68
residues by 14.6% and at 200 by NOTHING, because the high-water mark there is
the trunk's pair scratch. So the ceiling is unchanged and now measured to be
somebody else's problem.

🔴 **THE pLDDT IS WIRED, AND RANKS AS WELL AS AlphaFold 3's DOES HERE.**
Per-residue Spearman against real deviation is -0.17 to -0.31 on two
near-perfect targets, against AlphaFold 3's own -0.35 on one of them through
the same tool. The right sign, roughly the reference's strength, and measured
where there is almost no error to rank. A statement about calibration - the
LEVEL, which is the part that does differ - needs docs/EF2FAST.md's corruption
sweep, which manufactures the hard end rather than waiting for it.

🔴 **AND pTM IS ABSENT AND SHOULD STAY ABSENT** until something emits a TM
term. OpenDDE's head does not; deriving one from the PAE would be a different
quantity wearing pTM's name.

🔴 **THE GRID'S RESIDENT WEIGHTS ARE THE LAST OPTIMISATION, AND THE LENGTH
SWEEP DEVALUED IT.** `w.tri.out`, `w.tri.in` and `w.pair-transition` take
`pairWeightPrecision` and `w.grid1`/`w.grid2` do not, because
`createGridAttentionShaders` has no weight-precision plumbing at all where the
triangle has a clean pattern to copy. It is worth ~135 MiB of 648 at 68
residues and ~23 MiB of 1507 at 200 - so it was never the right target. It also
needs a two-buffer split rather than a flag: the grid's weights are ONE
interleaved buffer, so narrowing it narrows the LayerNorm scales with it, which
the triangle's own notes say to avoid.

## The chemistry is 15% short, and RMSD could never have said so

🔴 **OpenDDE'S ATOMS ARE ABOUT 15% TOO CLOSE TOGETHER, ON EVERY INPUT SHAPE,
AND EVERY GATE IN THIS FILE PASSED THROUGH IT.** The gates were RMSD and TM
against a crystal, which score where the backbone GOES; nothing scored what the
model builds once it is there. Measured with `probe-nucleic.js`'s median rigid-
pair ratio against the same conformer dictionary the featuriser read - 1.0 is
the dictionary's own chemistry - at 64 steps, recycles 0, one seed:

| arm | protein | nucleic | phosphodiester O3'-P |
|---|---|---|---|
| AlphaFold 3 f32, 68-residue chain alone | **1.009** | - | - |
| AlphaFold 3 f32, + a 4-nt DNA chain | **1.009** | **1.007** | 1.53-1.59 A |
| OpenDDE int5, the same chain alone | **0.892** | - | - |
| OpenDDE int5, the same chain, 200 steps | **0.889** | - | - |
| OpenDDE int5, + the same DNA chain | **0.849** | **0.797** | 1.85-2.03 A |

🔴 **AND IT IS NOT THE SAMPLER, NOT THE QUANTISATION AND NOT THE STEP COUNT.**
Each was ruled out by running AlphaFold 3 through the identical arm, since a
difference between two models measured at different settings is a difference
between the settings:

| control | protein | nucleic |
|---|---|---|
| AF3 f32, flow, 32 steps | 1.009 | 1.012 |
| AF3 f32, **diffusion**, 32 steps | 1.006 | 1.009 |
| AF3 **int5**, diffusion, 32 steps | 1.006 | 1.006 |
| AF3 int5, flow, 32 steps | 1.007 | 1.004 |

AlphaFold 3 lands on 1.004-1.012 in all four. Nothing about the EDM sampler or
about int5 compresses a bond, and OpenDDE at 200 steps is 0.889 where it is
0.892 at 64 - so it is converged onto the wrong chemistry rather than short of
it.

🔴 **THE DEFICIT IS INSIDE A TOKEN, NOT BETWEEN TOKENS, WHICH IS WHY TM = 0.9
SURVIVED IT.** Ratios of OpenDDE's own measurements to AlphaFold 3's on one
31-residue sequence, `probe-sidechains.js`:

| what | OpenDDE / AF3 |
|---|---|
| aromatic ring bonds, 33 of them (all role-2 sidechain tokens) | **0.848** |
| N-CA and CA-C, inside a role-1 backbone token | 0.943, 0.973 |
| C-N, the peptide bond BETWEEN two backbone tokens | **1.032** |
| CA-CA spacing along the chain | 0.964 |

So the chain's trace is close to right and the geometry within a structural
token is compressed, worst in the sidechain token. That is the same shape as
the side-chain bug docs/AF3.md records - "everything short at once" - and it
points at the atom decoder over the structural layout rather than at the trunk,
the tokeniser or the sampler. A uniform coordinate scale is RULED OUT: a scale
would move CA-CA and the peptide bond by the same factor as the ring, and the
peptide bond is 1.03.

🔴 **AND OpenBind-0 IS THE CONTROL THAT NAMES THE STAGE.** It is a second
non-AlphaFold bundle through the SAME tool, the same trunk code, the same
diffusion head and the same sampler - and the one thing it does not have is a
structural-token stage. Its backbone chemistry is right:

| `fold-opendde.js --target=6mrr --steps=32` | N-CA | CA-C | CA-CA |
|---|---|---|---|
| ideal | 1.458 | 1.525 | 3.80 |
| OpenBind-0 | **1.458** | 1.493 | 3.771 |
| OpenDDE | **1.363** | 1.493 | 3.664 |

So the shared AlphaFold 3 diffusion path is not what compresses a bond; the
second token space is. That narrows the suspect list to the expander, the
refiner, and the atom encoder and decoder as they run over the structural
layout.

🔴 **A LIGAND SHOWS IT TEN TIMES OVER, AND IS THE CHEAPEST REPRODUCTION.**
Glycerol beside the same 68-residue chain, bond error against the dictionary's
ideal conformer, `probe-ligand-flow.js --ligand=GOL --mode=diffusion
--steps=64`: AlphaFold 3 rms **0.033 A** (max 0.055), OpenDDE rms **0.349 A**
(max 0.641). Five bonds, one component, twenty-five seconds.

🔴 **THE CONFIDENCE HEAD'S LEVEL IS HIGH, AND ITS RANKING IS FINE - AND AN
EARLIER ENTRY HERE CLAIMED OTHERWISE BECAUSE IT NEVER RAN THE CONTROL.** The
level really is high: OpenDDE reports mean pLDDT 94.3 on the ligand job to
AlphaFold 3's 89.8, 92.1 to 85.9 on the protein alone, 91.6 to 88.1 on the
complex - higher on every arm while being the worse structure on every arm.

But "higher mean than another model" is not "does not track error", and the two
were conflated. What a pLDDT claims is that THIS residue is placed well, so the
question is the within-fold ranking, which `fold-opendde.js` has always
reported as `plddtVsError` - a Spearman where negative is correct:

| 6MRR, 64 steps, recycles 0 | plddtVsError | RMSD |
|---|---|---|
| AlphaFold 3 int5, through the same tool | **-0.3484** | 0.774 |
| OpenBind-0 | **-0.4984** | 1.963 |
| OpenDDE | **-0.3063** | 1.676 |
| OpenDDE, 1QYS | -0.1668 | 1.052 |

🔴 **AlphaFold 3's OWN RANKING ON THIS TARGET IS -0.35.** OpenDDE's -0.31 is
the same number. The -0.19 to -0.29 recorded further down as "weak" was weak
against nothing - the control had never been measured, and measured, it is
where AlphaFold 3 sits too. Two near-perfect targets give any head very little
error to rank, which is a property of the targets and not of the port.

So what is left is a LEVEL offset, and nothing here says whether that is the
port or the model. A differently trained head is entitled to a different level.
The one concrete port-side suspect is named below.

🔴 **AND THE REASON IT SURVIVED IS THAT THERE IS NO OpenDDE ORACLE DUMP.**
`oracle-dumps/` holds nine AlphaFold 3 captures - embedder, MSA, stack, trunk,
atom, denoiser, diffusion, confidence - and every AF3 module was checked
against one. OpenDDE has none: the port was built from the checkpoint's tensor
names plus a dialect, and validated only end to end against crystals. So the
whole diffusion side has never been compared to anything module by module,
which is exactly where this sits. `check-opendde-expander.js` covers the
expander and stops there.

A fix wants a capture of upstream's own denoiser - its inputs, since
`--capture-args` is what makes the answer reproducible - at the shape the
structural layout produces, then `check-af3-diffusion-head.js`'s treatment
applied to it. Everything short of that is guessing at a convention, and this
file already records what guessing at a convention costs.

## The untested surfaces, now tested

The tokeniser's own branches are covered without a GPU by
`test/opendde-structural-tokens.test.js` - sixteen tests over a ligand, a DNA
chain, an RNA chain, a two-chain complex, a modified residue and a plain
protein. They are conservation laws, because the regrouping moves atoms between
(token, slot) rather than changing them: the map is a bijection onto the live
atoms, `residueAtomGather` agrees with the `sources` it was built beside, no
atom changes position, element, charge, name or reference space, twins are
mutual and share their residue's space, chain adjacency stops at a chain
boundary, and the round trip back to the residue layout is the identity. Plus
the four things a shape cannot see: role, centre, twin and adjacency - including
the purine N9 / pyrimidine N1 distinction upstream records as costing a tRNA
fold 4.5 A, which no protein test can reach.

🔴 **AND ALL SIXTEEN PASS, WHICH IS WHY THE FOLD NUMBERS ABOVE ARE A MODEL
PROBLEM AND NOT A TOKENISER ONE.** The atoms arrive in the right tokens; what
happens to them afterwards is 15% short.

## Still open, and pre-existing

🔴 **`tools/gpu/check-af3-block.js` FAILS ON STOCK AlphaFold 3, AND DID BEFORE
ANY OF THIS.** At its own defaults - staged f16, accumulate f16, weights f16 -
it reports pair relRMS 1.02e-1 against a 1.31e-3 bound, 23311x. Verified at
commit 4981ffd, before the first OpenDDE commit, so it is pre-existing and not
a regression here. `--accumulate=f32` brings it to 6.13e-3 and `--staged=f32`
does not move it, so the f16 triangle ACCUMULATOR carries essentially all of
it - which is what docs/EF2FAST.md records for that knob ("the accumulator
carries 96% of the error"). The likely reading is that the bound was written
for f32 defaults and the defaults moved under it, exactly as the denoiser's
arms did in commit 21840ee. Not investigated further: the end-to-end fold is
unaffected, and re-deriving a bound is a decision about what the kernel is
allowed to cost.

🔴 **AND THE FOLD HAS ONLY BEEN SCORED ON TWO TARGETS, AT MSA DEPTH 1.** 6MRR
and 1QYS are both DESIGNED proteins - idealised and canonical, which
docs/EF2FAST.md records as the reason they fold well from a single sequence
where ubiquitin does not. Read the TM scores as "the trunk is assembled
correctly", not as a benchmark.
