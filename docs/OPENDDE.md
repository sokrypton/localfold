# OpenDDE: the trunk transfers, the way it makes coordinates does not

OpenDDE (Aureka Research, Apache-2.0) is an independent PyTorch
reimplementation in the AlphaFold 3 family - its own pairformer and its own
primitives, not DeepMind's code at different widths. This is the port's state:
what runs, what it is worth, and what is deliberately absent.

**It is not a folding model here.** `MODEL_BUNDLES.opendde` says
`foldingModel: false`, the page does not offer it, and the reason is in the
weights rather than in the schedule - see "What does not transfer" below.

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

So the exported bundle is the trunk and the distogram head, and stops there.

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

## Open

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

🔴 **THE STRUCTURAL-TOKEN STAGE IS THE WHOLE OF WHAT WOULD MAKE THIS A FOLDING
MODEL**, and it is a second token space rather than a branch: a featuriser that
splits residues into backbone and sidechain tokens, atom layouts over that set,
and a confidence head with none of AlphaFold 3's names. Upstream's
`structural_tokens.py` is 168 lines of JAX and `opendde_confidence.py` 151, but
the featurisation around them (`attach_structural_batch`) is the larger half.

🔴 **AND IT HAS ONLY BEEN SCORED ON ONE TARGET, AT MSA DEPTH 1.** 6MRR is a
DESIGNED protein - idealised and canonical, which docs/EF2FAST.md records as
the reason it folds well from a single sequence where ubiquitin does not. Read
0.718 as "the trunk is assembled correctly", not as a benchmark.
