# EF2-fast (ESMFold2 600M): the port, and what it cost to get right

The model is `ef2-fast-600m`. The second half of this file, under "Before the port", is the
investigation that preceded it - whether ESM-C could be compressed enough to
fold from a single sequence in a browser. Everything before that is the port.

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

## EF2-fast's memory, and a transition optimisation that was not one

🔴 **A FOLD'S PEAK IS NOT IN THE TRUNK, WHICH IS WHERE ALL THE TIME IS.** The
trunk is 85% of a 300-token fold's SECONDS and its own peak is 446 MiB; the
fold's is 992. Nothing said so until `tools/gpu/fold-esmfold2.js` printed
`peakByLabel` - AF3's `fold.js` and `fold-af2.js` have printed it for a long
time and this tool had only the total, so a 92 MiB saving inside the trunk
could be measured and its complete absence from the fold's peak could not be
explained. The rows at the peak were four pair-sized f32 tensors at 87.9 MiB
each - `rel-pos`, `z-init`, `pair`, `diff.pair-cond` - and two of them were
dead.

| | 300 tokens | 76 |
|---|---|---|
| before | 991.5 MiB | 577.7 |
| `z_init` released when the trunk loop ends | 903.6 | |
| the conditioning written into `relPos` | **815.7** | **566.5** |

**17.7%**, and the structure does not move: PDB sha256 `83c0530b02f867ac` at
300 tokens and `181a74e78959fe5e` at 76, certainty and contacts identical to
every digit at both, time unchanged.

🔴 **`z_init` HAS ONE READER AND IT IS INSIDE THE LOOP.** `z = z_init +
pair_loop_proj(z)` reads it once a loop and nothing after the trunk does. It
was held to the end of the fold, through the diffusion, which is where the
fold is fullest. Same shape as AF3's `releaseResidentWeights`: give a stage's
memory back when the stage is over, before reaching for kernels.

🔴 **AND THE CONDITIONING CAN LIVE IN THE ENCODING IT EATS, BECAUSE IT IS ROW
CHUNKED.** `joined-norm` is the last pass to read `relPos` and it runs before
`z-project` writes - within a chunk. So chunk k's output goes where chunk k's
input was and chunk k+1 reads rows chunk k never touched. It is the aliasing
both models already do where an attention writes into its normalised input;
what is new is that this one crosses a STAGE boundary, so the caller hands the
buffer over rather than the callee allocating one.

🔴 **AND THE WHOLE ARC IS 45%, WITH THE PEAK MOVING THREE TIMES.** Every step
is bit-identical or priced against the sampler's own spread, on a 300-token
fold:

| | peak | what moved |
|---|---|---|
| before | **991.5 MiB** | |
| `z_init` released when the trunk loop ends | 903.6 | |
| the twelve pair biases built after `relPos` is dead | 837.5 | |
| the conditioning written INTO `relPos` | 815.7 | |
| the conditioning's widened scratch released before the biases | 799.2 | |
| **the token transformer's weights in f16** | **629.9** | the peak leaves the diffusion |
| `relPos` rebuilt after the trunk; the pair released | **543.5** | the peak is the TRUNK |

76 tokens goes 577.7 -> **317.9** over the same steps. The structure is
unchanged throughout except where f16 weights move it by 0.0017 A, and the
76-mer's certainty is still 0.9496183936533175 with 114 contacts.

🔴 **AND THE FLOOR IS SIX PAIR-SIZED TENSORS, WHICH IS 527 OF THE 543.** Four
triangle scratch, `z_init` and the pair, all live at once by construction.
Going below it needs the pair track chunked, which this file records as
all-or-nothing and blocked at the triangle.

🔴 **THE DENOISER'S TOKEN TRANSFORMER TAKES f16 WEIGHTS AND ITS ATOM STACKS DO
NOT, AND THE SIZES ARE WHY THE QUESTION IS WORTH ASKING AT ALL.** The twelve
token blocks are **459 MiB** of a 799 MiB fold; the two atom stacks are **6.4
MiB together**. So the stack `docs/PERF.md`'s rule forbids narrowing is not
worth arguing about, and the stack it permits is more than half the model. The
rule is what a stack PRODUCES: the token transformer's output is an activation
an adaLN renormalises, an atom stack's is a position update in angstroms that
nothing renormalises - AF3 records the identical split at 1.88e-2 inside a
4e-2 bound against its atom blocks missing 4e-4 by 3x.

| | peak | sampler step | denoiser relRMS |
|---|---|---|---|
| f32 weights | 799.2 MiB | ~398 ms | 7.08e-5 |
| **f16 weights** | **629.9** | **~344** | 1.47e-4 |

🔴 **AND IT IS FASTER, WHICH THE MEMORY ARGUMENT DOES NOT PREDICT.** These
weights stream once per sampler step, so halving their bytes is the one shape
this file's f16-weight table says pays - unlike AF3's trunk, where they are
resident and read one scalar at a time for a 2% LOSS. Ask how a stack READS
its weights before pricing their width.

🔴 **AND IT IS PRICED AGAINST THE SAMPLER'S OWN SPREAD.** The structure moves
**0.0017 A**; changing the SEED alone moves it **12.05 A**. A factor of 7000,
with the certainty, the contact count and the CA-CA spacing unmoved.
`check-esmfold2-diffusion-gpu.js --weights=f32,f16` is the axis, and each arm
is held to the bound its own arithmetic implies rather than one loosened to
cover both - the f32-weight arms reproduce this file's recorded 7.08e-5 and
1.52e-4 exactly, which is what says the plumbing is sound rather than the
bound generous.

🔴 **AND AN IN-PLACE TRANSFORM MUST BE ASKED FOR.** `prepare()` wrote the pair
conditioning into `relPos` unconditionally for one commit, and the denoiser's
checker prepares TWO arms from one uploaded encoding - so the second arm read
the first arm's conditioning and scored **1.77e-1 against a 4.5e-4 bound**.
`reuseRelPos` is opt-in and the fold is its only caller. The checker caught it
because it runs two arms off one buffer, which is exactly the shape that finds
a destructive input.

🔴 **AND THE PEAK MOVES WHEN YOU TAKE A TENSOR OUT OF IT, SO THE SAVINGS DO NOT
ADD UP.** Releasing `relPos` after the conditioning was worth 66 MiB and not
87.9, because the twelve pair biases are allocated just after it and the
fullest moment moved earlier. Releasing the pair was worth **nothing at all**
on its own, because by then the peak had moved into the trunk. Read
`peakByLabel` again after every change; the row that was second is not the row
that is first, and `peakGroups` in fold-esmfold2.js is what says which STAGE
to price.

🔴 **AND THE PAIR TRACK NEEDS FOUR SCRATCH TENSORS WITHOUT THE GRID ATTENTION,
NOT FIVE.** `tri.contract` is the last pass to read `a`, and it runs before
`tri.normalize-hidden` writes - so the normalised hidden goes back into `a`.
Only the grid attention ever wanted a fifth, because `grid.project` writes q,
k, v and a gate and all four are live at once. `pairScratchCount(gridAttention)`
is the rule. Worth 92 MiB of the TRUNK's peak at 300 tokens (538 -> 446) and
nothing at all of the fold's, per the entry above. AF3 is unmoved:
`check-af3-trunk` reports pair relRMS **1.99e-5**, this file's own figure.

🔴 **THE TRANSITION IS NOT WEIGHT-READ BOUND, WHICH TOOK THREE INSTRUMENTS TO
ESTABLISH AND CONTRADICTS bench-transition.js's OWN BREAKDOWN.** `pair-transition`
is 47-50% of this trunk and reads one scalar weight per multiply-add, and that
tool's comment attributes 19% of the kernel to the first matmul's two weight
reads. Halving those reads - a blocked slot mapping, so a lane owns ADJACENT
intermediate slots and reads them together - changes the kernel by **nothing**:

| arm, normalised by the untouched `tri.project` in the same profile | |
|---|---|
| the shipped kernel | **2.454** |
| one slot a lane, blocked mapping (two runs) | 2.512, 2.475 |
| two adjacent slots a lane (two runs) | 2.397, 2.394 |

Reverted. Every weight read that could be removed was already overlapped with
the arithmetic.

🔴 **AND A SECOND, ALIASED BINDING OF THE WEIGHT BUFFER COSTS 2.3x.** The first
version read the adjacent pair through a `vec2<f32>` view of the same buffer -
two read-only bindings on one GPUBuffer, which WebGPU allows and which is
bit-identical. Normalised the same way, the aliased scalar arm is **5.2-5.7**
against the shipped kernel's 2.454, and the aliased vec2 arm 2.7. Two read-only
views beside a `read_write` input cost this kernel more than a wide load saves
it. Reading `weights[k]` and `weights[k + 1]` and leaving the merge to the
compiler keeps the aliasing information the second binding takes away.

🔴 **AND THAT ALIASED PAIR IS WHAT MANUFACTURED A 1.9x.** With the extra binding
in BOTH arms, `bench-transition.js` reported 312 -> 162 ms and the trunk 1.94x -
which is the vec2 load winning back the cost of the binding it was delivered
in, and nothing else. **An A/B where both arms carry the same new defect
measures the defect.** The control that caught it was normalising against a
kernel the change does not touch.

🔴 **AND THE FIRST ARM MEASURED IN `bench-esmfold2-trunk.js --profile` RUNS
COLD.** Every kernel in it, including ones nothing touched, reads 1.5-1.9x slow:
`tri.project` measured 3807 ms in the first arm of a run and 2002, 2551 and 2078
in the other three. A baseline-then-candidate A/B therefore invents a speedup of
about that size. `--profile` sweeps every arm now rather than the first, and the
number to read is a RATIO against an untouched pass in the same report - the
wall figure at 300 tokens moves more between two runs of one arm than the arms
differ by.

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

🔴 **AND IT CANNOT USE AN MSA EITHER, WHICH WAS MEASURED RATHER THAN READ OFF
THE FLAG.** `disable_msa_features: true` does NOT mean "no alignment was
given": `experimental.py` falls back to the QUERY ONE-HOT when there is no MSA
and only then zeroes the profile and the deletion mean, so this checkpoint's
profile channels are identically zero in TRAINING as at inference. `s_inputs` is
`[atomPooled 384 | aatype 33 | profile 33 | deletionMean 1]`, and the weights
that read those 33 columns say what that cost them:

| `zInit1` block | mean column RMS | coefficient of variation |
|---|---|---|
| atomPooled 0:384 | 0.308 | 0.450 |
| aatype 384:417 | **0.688** | **0.289** |
| **profile 417:450** | **0.027** | **0.029** |

Chance CV for 256 columns of noise is 0.044, so the profile block is FLATTER
than noise while its neighbour is ten times rougher, and it is **26x smaller**.
The control that settles it: the 600M and 300M checkpoints have profile columns
identical to **0.0** while their aatype columns differ by up to 0.075. Two
separately trained models cannot agree on a shared weight unless no gradient
ever reached it. Those columns are the initialisation.

🔴 **AND FEEDING A REAL ALIGNMENT CONFIRMS IT, WITH A SCRAMBLE AS THE CONTROL.**
`tools/gpu/probe-esmfold2-msa-profile.js` folds five arms in one process at
three seeds - a 59-mer against `tools/fixtures/test.a3m`, 8076 rows, a genuine
PSSM (columns sum to 1.000, mean peak 0.43, mode agreeing with the query at
36/59). RMSD from the zero-profile fold after superposition:

| arm | seed 1 | 2 | 3 | mean |
|---|---|---|---|---|
| the query one-hot | 0.031 | 0.027 | 0.028 | 0.029 |
| **a real MSA profile** | 0.020 | 0.021 | 0.019 | **0.020** |
| **that profile, residue axis PERMUTED** | 0.030 | 0.028 | 0.032 | **0.030** |
| **the SAMPLER'S SEED, nothing else** | 0.393 | 0.486 | 0.418 | **0.432** |

A real alignment is indistinguishable from a permuted one - both about a
fifteenth of the sampler's own spread - and certainty is 0.967 on every arm at
every seed. **A profile is a random projection here, not an input.**

🔴 **AND THE PLUMBING CONTROL IS WHAT MAKES THAT A RESULT.** Every arm runs at
the same seed, so a profile that never reached the model would move the fold by
EXACTLY zero - which is also what "the alignment does nothing" looks like if
only the means are read. The probe asserts both: `profileReachedTheModel` (all
three arms nonzero) and `alignmentIsDistinguishableFromNoise` (false). Without
the first, this measures a broken fixture.

🔴 **AND THE REAL PROFILE MOVES THE FOLD LESS THAN THE SCRAMBLE, WHICH IS NOT
IT WORKING.** 0.020 against 0.030, consistently. The two have identical column
norms - a permutation - so the difference is which random columns are hit: a
real profile's mass sits mostly on the residue the aatype one-hot already names,
so its contribution lands nearer the subspace the trained term occupies. The gap
is 0.010 A, two percent of the seed spread. Read it as geometry, not as signal.

🔴 **SO MSA SUPPORT IS A DIFFERENT CHECKPOINT, NOT A FLAG - AND THAT IS UNLIKE
TEMPLATES.** `_MODULE_FLAG_SECTIONS = ("msa_encoder", "lm_encoder", "parcae")`,
and `EsmFold2MsaEncoderConfig` is documented "Large MSA models only". Neither
bundle here carries a single `msa_encoder` tensor - 856 tensors, zero matches -
so a Large MSA release would bring both the encoder AND trained profile columns.
Templates have no module to disable at all; MSAs have one this checkpoint does
not include.

🔴 **AND TEMPLATES ARE ABSENT IN A THIRD WAY: THE HOOK EXISTS AND NO WEIGHTS
CONSUME IT.** `grep -rin template` over the whole `esm/models/esmfold2/` package
returns **0**, and `z_init` has five terms with none of them one. But
`prepare_input.py` builds `disto_cond` and `disto_cond_mask` - a binned distance
matrix over chosen token pairs, on the SAME 2-22 A / 64-bin grid, which is what
a template reduces to - and `model.py` raises on it with the reason in its own
comment: *"No released checkpoint carries the disto_conditioning_proj weights
that would consume these."* `experimental.py`, the class this checkpoint
instantiates, does not take the argument at all. So somewhere a checkpoint was
trained with distogram conditioning; none released was. `pocket_feature` is in
the same state, listed in `_IGNORED_FEATURE_KEYS`.

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

🔴 **AND THE DISTOGRAM IS WEAK ON WHERE THE LIGAND GOES, WHICH IS THE MODEL AND
NOT THE PORT.** Every other reading off the same head is excellent, which is
what makes the comparison worth having - one fold, one head, four kinds of pair,
predicted distance against the structure's own:

| pair kind | n | bias | correlation |
|---|---|---|---|
| protein-protein | 2415 | +0.21 A | **0.999** |
| sequence neighbours | 435 | +0.31 | 0.999 |
| ligand's own atoms | 465 | +0.60 | **0.981** |
| **ligand-protein** | 2356 | **+2.37** | **0.683** |

So the head reads a ligand's GEOMETRY nearly as well as a protein's - it is
handed the conformer in `ref_pos` and reproduces it - and reads the protein
almost perfectly. What it is weak at is the ligand's PLACEMENT against the
protein, which is the one thing it has to infer.

🔴 **AND IT HEDGES LONG, WHICH THREE ESTIMATORS SHOW BETWEEN THEM.** For
ligand-protein the mode is +2.37 A with an RMS of 6.64, the mean +4.11 with
5.00, the median +3.40 with 4.44 - the mode nearest, the mean furthest, which
is a long RIGHT tail: a peak near the right answer with mass trailing off toward
"far away". On protein pairs all three agree to a tenth of an angstrom.

🔴 **AND "ZERO PREDICTED CONTACTS" WAS A STATEMENT ABOUT A 0.5 THRESHOLD, UNTIL
IT WAS CHECKED.** The contact map is a PROBABILITY and the panel draws it, so a
count of pairs over 0.5 is the checker's convention rather than the head's
opinion. Read as probabilities on the ATP fold, protein-ligand pairs max at
**0.131** against protein-protein's 0.997 - so the threshold was not hiding
anything after all, and the head really does refuse to believe in that contact.

🔴 **BUT IT IS LIGAND-DEPENDENT, WHICH ONE TARGET WOULD HAVE MISSED.** The same
protein against three components: **ATP 0.131, glycerol 0.211, haem 0.665**. A
cofactor that appears in a great many structures is placed with some confidence
and a cryoprotectant is not, which is what training coverage would look like.
**Do not generalise a ligand result from one ligand.**

🔴 **AND THE BORROWED BIN GRID CANNOT BE IMPROVED FROM THIS DATA, WHICH WAS WORTH
TRYING.** A predicted distance is linear in the bin index, so regressing the
expected bin on the OBSERVED distance - that direction, because the noise is in
the prediction and the other way round is diluted - recovers the grid the head
was trained with. It disagrees with itself: bin widths of 0.405, 0.417 and 0.449
from the three reliable pair kinds, implying ranges from 53 to 58 A against the
borrowed 52. The +0.21 A bias on protein pairs is half a bin at this resolution,
which is as close as the grid can resolve. `CONTACT_EDGES` stays.

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

🔴 **AND OpenFold3's TRAINING LOSS CONFIRMS THE REPRESENTATIVE TABLE AND
REFUTES ANY CHEMISTRY IN THE HEAD.** `openfold3/core/loss/distogram.py`'s
`all_atom_distogram_loss` is the loss everyone in this lineage trains against,
and it is four lines: take one representative atom per token, bin the Euclidean
distance, cross-entropy. `get_token_representative_atoms`'s own docstring is the
table, verbatim - **Cb for standard amino acids (Ca for glycine), C4 for
purines, C2 for pyrimidines, and the first and only atom for anything atomized**,
which is every ligand and every modified residue. So `representativeAtoms` is
right for the reason it was fixed, and now it is right on an upstream statement
rather than on AF3's feature table plus a 6x geometric argument.

🔴 **AND THE PAIR MASK IS RESOLVED-ATOM PRESENCE AND NOTHING ELSE.** No
`dna_weight`, no `ligand_weight` - the diffusion loss beside it has all three
(5.0, 5.0, 10.0) and the distogram has none, so **every token pair is trained
identically whatever the two molecules are**. There is no per-chemistry
threshold, no per-chemistry bin grid and no per-chemistry weight anywhere in the
head. All the chemistry lives in WHICH ATOM the token is represented by.

🔴 **SO `CONTACT_ANGSTROMS_BY_KIND` IS A DEPARTURE FROM UPSTREAM, DELIBERATELY,
AND A READER COMPARING THE TWO WILL SEE THAT.** OpenFold3 reads a contact off
the same head as **a flat 8 A for every pair** - `distogram_bins_8A =
distogram_bin_ends <= 8.0` in `core/metrics/confidence.py`, which is the TOP-EDGE
rule this repository already uses, confirmed. That number weights its gPDE; it
is not a claim that 8 A means the same thing for two ligand atoms as for two
pseudo-betas. Our table is a post-hoc calibration of exactly that, measured
against real atomic contact in deposited structures, and the upstream flat 8 A is
the arm it beats. **Both are right about different questions**: theirs is a
weight inside a ranking metric, ours is a contact map somebody reads.

🔴 **AND AF2's BIN BREAKS WERE 2 AND 22 IN TWO MANIFESTS AND THEY ARE 2.3125 AND
21.6875.** AlphaFold's `config.py` says `first_break: 2.3125, last_break:
21.6875` and its head is `linspace(first, last, num_bins - 1)`, an exact
0.3125 A grid. `add_distogram_head.py` wrote 2.0 and 22.0, and both AF2 bundles
carried them - a first break a third of a bin low and every edge off by up to
0.3125 A.

🔴 **AND THE ROUND NUMBERS ARE REAL, THEY ARE JUST THE OTHER FORM OF THE GRID.**
OpenFold3 states the identical binning as `bin_min 2.0, bin_max 22.0, no_bins 64`
with **nearest-centre** assignment (`binned_one_hot` is `argmin |d - centre|`), so
its centres are `2.15625 + 0.3125 b` and the midpoints between them are
`2.3125 + 0.3125 b` - **AlphaFold's breaks, to 0.0**. Two independent statements
of one grid, and 2 and 22 belong to the CENTRE form. `test/distogram-head.test.js`
asserts the two coincide, because that identity is what says neither reading was
a guess.

🔴 **AND THE SHIPPED CONTACT MAP DOES NOT MOVE, WHICH IS WHY THIS SURVIVED.**
Both grids put **19 bins under 8 A** - 7.9375 and 7.806 are the last edges that
qualify - so the only quantity anything computes from these numbers was correct
under the wrong ones. What was wrong was the LABEL on every bin, which
`probe-af2-dgram-plddt.js` writes into its dump, and any expected distance a
future caller takes. Found by reading OpenFold3's loss, not by a failing check.

🔴 **AND THE GATE THAT SHOULD HAVE CAUGHT IT LOOKED AT `tensors` ALONE.**
`manifest_mismatches` compared the compiled module's tensor table against the
exporter's `manifest.json` and nothing else, so `distogramHead` - which is a
sibling of `tensors`, not a member - could differ between them for as long as it
liked. It compares EVERY key now, `shardDigests` excepted because the writer
computes that and the exporter does not write it. Held to a negative control:
put 2.0 back into `model/manifest.json` alone and the build fails naming
`distogramHead`; under the previous gate the identical corruption is silent.

🔴 **AND IT HAD STOPPED RUNNING AT ALL, BECAUSE CHECKING SAT INSIDE
PUBLISHING.** The registry check and the manifest check lived under
`if include_model:`, which was sound while bundles were published from here.
Every bundle is hosted now and the Pages workflow runs `build_site.py` with no
`--model`, so the one gate that says the compiled manifest still describes the
shipped weights ran on no deploy. **What a bundle's manifest SAYS ships with the
page wherever its shards live; only the COPY is opt-in.** Both checks are out of
that branch, and a remote family is checked before it is skipped for publishing
- which is why the two AF2 families, the only two with an exporter manifest to
compare against, were the two never compared.

🔴 **AND `--model` NOW EXITS 0 WHEN EVERY BUNDLE IS REMOTE.** It skipped all
eight as hosted, shipped none, and reported *"no export directory exists"* with
the directories sitting right there - a message naming a cause that is not the
cause. "Nothing was published" is not "nothing exists".

🔴 **AND ALL EIGHT BUNDLES ARE REPINNED TO ONE SHA, `71ece357`.** The hosted
`manifest.json` copies carried the wrong breaks too. Nothing fetches them - the
page reads the compiled module - but a second copy that can disagree is this
session's own recurring bug, so they were corrected and uploaded. **The upload
alone would not have changed what is served**: the remotes pin a commit, so the
old snapshot goes on answering until the pin moves. Verified before repinning by
comparing the two commits' blob OIDs through the tree API rather than
downloading anything: **2 files changed across all eight bundles, both of them
these manifests, every shard the same blob.** Then folded through it - AF2
monomer at pLDDT 84.5 and EF2-fast at certainty 0.84, both from the new pin.

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

0.74 A is inside the sampler's own seed spread, which is what the compression study below
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

## A PAE from a distogram, for a model that has no confidence head
🔴 **IT WORKS, OUT OF SAMPLE, AND THE CONTROL IS WHAT SAYS SO.** EF2-fast has no
confidence head at all, and PAE is the score people read off a complex - it is
what says whether two parts are placed correctly against each other, which
neither pLDDT nor a contact map answers. AlphaFold 3 produces a distogram AND a
real PAE out of ONE fold, so an estimator can be scored before being carried to
the model that lacks one. 13 targets, `tools/gpu/probe-pae-from-distogram.js` to
collect and `tools/pae-from-distogram.py` to score, **leave-one-target-out
throughout**:

| | median Spearman | worst |
|---|---|---|
| `d_ij` alone (the baseline) | 0.658 | 0.509 |
| geometry only, sigma removed from the fit | 0.658 | 0.509 |
| **the distogram's own moments, fitted** | **0.876** | **0.734** |
| ...with sigma attached to the WRONG PAIRS | **0.450** | 0.346 |

Median RMSE **2.86 A** against a quantity running 0 to 32.

🔴 **PAE AND A DISTOGRAM ARE NOT THE SAME KIND OF OBJECT, AND THE ARITHMETIC
SAYS EXACTLY HOW THEY DIFFER.** Aligning on token i's frame, let
`Delta_ij = dx_j - dx_i` be the relative displacement error. Then

    PAE(i,j)^2  ~  E||Delta_ij||^2          the full 3-D magnitude
    sigma_ij^2  =  E[(u_ij . Delta_ij)^2]   ONE radial projection of it

So a distogram supplies one scalar projection, along a known direction, of a
3-vector - and the direction it cannot see is the tangential one, which is
precisely how a domain ROTATION displaces things. That is the mechanism behind
sigma systematically under-reading inter-domain error, and it is why no
per-pair function of sigma alone can be the answer.

🔴 **AND PAE^2 IS A SQUARED-DISTANCE MATRIX, SO IT IS LOW-RANK BY
CONSTRUCTION.** `E||Delta_ij||^2 = g_ii + g_jj - 2 g_ij` for `g` the Gram matrix
of displacement covariances, so double-centring PAE^2 gives a Gram matrix whose
rank is the number of collective modes. Measured over thirteen targets rather
than assumed:

| | |
|---|---|
| components for 90% of PAE's energy | **6-10** |
| ...of `sigma`'s, at the same sizes | **13-33** |
| PAE's asymmetry, `mean|asym| / mean|sym|` | **0.12** (0.09-0.29) |
| energy of double-centred PAE^2 in THREE eigenvalues | **0.61-0.88** |

**The distogram is a high-rank per-pair signal and PAE is a low-rank collective
one.** The frame term that makes PAE asymmetric is a 12% correction, not the
substance.

🔴 **SO THE FEATURES ARE THE GRAM TERMS, AND EACH HALF IS WORTH MEASURING
SEPARATELY.** `g_ii + g_jj` is a PER-TOKEN mobility - read off the distogram as
how uncertain a token's distances are in general - and `g_ij` is the pair
coupling, which is sigma itself. Leave-one-target-out throughout, so the fit
never sees the target it is scored on:

| | median Spearman | worst |
|---|---|---|
| `d_ij` alone (the baseline) | 0.658 | 0.509 |
| geometry only, distogram removed from the fit | 0.658 | 0.509 |
| **mobility, with no pair term at all** | **0.745** | 0.618 |
| the distogram per pair | 0.876 | **0.734** |
| **both: mobility + coupling** | **0.885** | 0.728 |
| ...with sigma attached to the WRONG PAIRS | **0.450** | 0.346 |

Median RMSE **2.76 A** against a quantity running 0 to 32.

🔴 **THE SHUFFLED ARM IS THE EXPERIMENT, AND IT LANDS BELOW THE GEOMETRY.** Same
sigma values, same marginal distribution, wrong pairs: 0.450 against
geometry-alone's 0.658. A wrong distogram is WORSE than no distogram, so what is
used is the correspondence between a pair and its spread - not the presence of
another column for a least squares to lean on. Without that arm, "0.885 beats
0.658" is what any extra feature does to a fit.

🔴 **AND THE BASELINE HAD TO BE THE GEOMETRY.** PAE grows with distance whatever
the model believes, so `d_ij` alone already reads 0.658 and an estimator quoted
against zero would look four times better than it is. The `no sigma` arm scores
**identically** to `d_ij` - which is the check that the fit adds nothing by
being a fit.

🔴 **AND `d_ij` IS ALREADY THE LOW-RANK COLLECTIVE SIGNAL, WHICH IS WHY
EXPLICIT LOW-RANK MACHINERY BUYS NOTHING.** A squared-distance matrix of 3-D
points has rank at most 5, so handing the fit the predicted structure's own
distances already gives it the collective frame. Reconstructing sigma^2 by
classical MDS - double-centre, truncate, read the distances back - is a real
improvement **on sigma alone**, median 0.600 -> 0.705 and worst **0.082 ->
0.470**, because it recovers directions no single pair measured. Added on top of
a fit that already has `d_ij`, it is redundant: 0.876 -> 0.874. Projecting the
finished prediction onto rank 4 or 8 is slightly WORSE (0.858, 0.862). **The
low-rank structure is real and is already being supplied by the geometry.**

🔴 **AND THE PRINCIPLED PER-PAIR ESTIMATOR LOST TO ADDING TWO RANKS TOGETHER.**
Differentiating the cosine rule gives the tangential displacement from a
neighbour k as `sigma_kj * d_kj / (r_ik * sin theta)`, independent of `d_ij` -
which is the check that the algebra is right, since an angular error scaling
with distance must give a displacement that does not depend on how the angle was
measured. Taking the best-determined neighbour (a MIN, which is why a mean over
the neighbourhood scored worse) gives median **0.749** with no fitted parameters
- beating sigma alone at 0.586 and the distance at 0.646, and LOSING to
`rank(sigma) + rank(d)` at **0.818** on eleven of thirteen targets. The min is
brittle: one accidentally small `sigma_kj` discards every other constraint, and
the errors it treats as independent are not. **A derivation is a hypothesis.**

🔴 **AND THE DISTOGRAM IS BLIND PAST 22 ANGSTROMS, BY CONSTRUCTION.** Its last
bin is open-ended, so it cannot tell 30 A from 60 while PAE runs to 32.
`min(d, 22)` is a feature for that reason, and it is why the worst-scoring
targets are the ones whose PAE is largest.

🔴 **AND THE DYNAMIC RANGE WAS MANUFACTURED, AS THE CERTAINTY SWEEP'S WAS.** A
well-folded monomer has almost no PAE to rank - 6MRR reads mean 3.36, sd 3.03 -
so each sequence is folded again SPLIT INTO TWO CHAINS, which puts real block
structure in: intra-chain 5.81 against inter-chain 10.20 on that same 68-mer.
The thirteen are five monomers, five splits and three protein-plus-peptide
complexes.

🔴 **AND IT DOES NOT SHIP, BECAUSE IT IS INVERTED ACROSS FOLDS.** It was put on
the page - the heatmap panel gained a `pae` tab - and taken off again the moment
it was folded on inputs nothing had tested: single sequences and random ones.
Five folds at 68-76 residues:

| | contacts | certainty | **pAE** |
|---|---|---|---|
| 6MRR | 73 | 0.9199 | **8.678** |
| ubiquitin | 121 | 0.9496 | **8.836** |
| random 1 | 1 | 0.6151 | **6.557** |
| random 2 | 0 | 0.6124 | **7.056** |
| random 3 | 3 | 0.5285 | **6.906** |

**A sequence that folds to nothing scores BETTER than a real protein.** And
sharper, the same target either way - ubiquitin with the language model removed
loses every contact and drops to certainty 0.4165, and its pAE IMPROVES from
8.836 to **7.969**.

🔴 **AND IT IS NOT THE DISTANCE FEATURES, WHICH IS WHAT MAKES IT FATAL RATHER
THAN FIXABLE.** One point per target, refitted leave-one-out, against the mean
true PAE:

| | within a fold | across folds |
|---|---|---|
| shipped (all eight features) | 0.738 | **-0.867** |
| without `d` and `min(d,22)` | 0.645 | -0.667 |
| the distogram terms only | 0.627 | -0.683 |
| `sigma` alone | 0.644 | -0.467 |

Every arm is negative. The mechanism is that a failed fold COLLAPSES - short
distances, and a distogram that is confidently wrong rather than uncertain - so
every term points the wrong way at once.

🔴 **SO IT IS WITHHELD ON THIS FILE'S OWN PRECEDENT.** The per-residue certainty
colour was measured and not shipped because "an ordering that is right on
average and inverted on some particular fold is the worst possible per-residue
colour, because the fold somebody is staring at is the one they doubt". **A PAE
panel that looks better on a failed fold is the same fault and worse**, because
a PAE is what people check precisely when they suspect a fold. `alignedError` is
opt-in on `foldEsmfold2` and nothing on the page passes it; the estimator, the
fit and the numbers stay in `src/esmfold2/aligned-error.js` and
`tools/pae-transfer.py`.

🔴 **AND THE CERTAINTY CAUGHT EVERY ONE OF THESE CASES, WHICH IS THE OTHER HALF
OF THE RESULT.** 0.92-0.95 on the two real proteins, 0.53-0.62 on the randoms,
0.42 on the ablation - the complement holds up exactly as its own sweep said it
would (0.90 across folds). **The page is not missing a global score. It is
missing a per-pair one.**

🔴 **AND THE TEST THAT FOUND IT COST FIVE FOLDS AND WAS NOT IN ANY SWEEP.** The
46-target sweep, the 80 corrupted folds and the 13 PAE targets were all
sequences that FOLD. "Generate a few random predictions with no MSA or PLM" is
the input class none of them covered, and it inverted the headline result.
`tools/gpu/probe-pae-esmfold2.js --summary --no-plm` is that arm.

🔴 **AND A FOLD PRODUCES ONE NOW, NOT ONLY A TOOL.** `foldEsmfold2` returns
`alignedError`, a tokens^2 matrix. Its three inputs - the distogram's mean,
spread and effective width in ANGSTROMS - are computed on the DEVICE beside the
contacts by `createMomentsShader`, riding the same projected chunk so the
distogram is still projected exactly once; taking them on the host would mean
reading back `pairs * bins * 4`, **46 MiB at 300 tokens for a quantity three
numbers wide**. The estimate itself is assembled after the sampler, because half
its features are distances and those do not exist until there is a structure.

🔴 **AND THE MOMENT BUFFERS NEED COPY_SRC, WHICH THE SCRATCH TENSORS DO NOT.**
`storage` alone is what every other tensor in that function takes, and a buffer
that is read back is not one of them: *"[Buffer esmfold2.disto.mean] usage
(BufferUsage::Storage) doesn't include BufferUsage::CopySrc"* on the first fold.

🔴 **AND IT ORDERS PAIRS INSIDE ONE FOLD AND SAYS ALMOST NOTHING ACROSS FOLDS -
THE OPPOSITE WAY ROUND FROM THE CERTAINTY.** One point per fold, the mean
estimate against the mean true PAE over the nine matched targets: **Pearson
0.340, Spearman 0.117**, with a range of 8.68-9.50 A where the truth's is
3.04-12.71. It is nearly a constant between folds. Within a fold it orders pairs
at 0.746.

| | across folds | within one |
|---|---|---|
| the distogram certainty | **0.90** | 0.44 median, worst NEGATIVE |
| **the pAE** | **0.117** | **0.746** |

So the two answer different questions and neither substitutes for the other -
the pAE for "which parts of THIS fold are placed relative to which", the
certainty for "is this fold worth looking at". Which is how a real PAE is read
anyway: nobody compares the mean PAE of two targets, they look at the block
structure of one.

🔴 **SO THE ANGSTROMS ARE A REGRESSION ONTO ANGSTROMS AND NOT A CALIBRATION.**
Per-target bias runs **-3.88 to +5.99 A** with a mean of +0.69, because it
regresses to the global mean: 6MRR reads 8.68 against a true 3.40, and the
hardest target reads 8.84 against 12.71. **Report the MAP, not the number** -
`fold-esmfold2.js` prints a summary only so a shell run can see the estimate
exists, and says so in the field beside it.

🔴 **AND IT IS CARRIED TO EF2-fast NOW, WHICH IS THE POINT.**
`src/esmfold2/aligned-error.js` is the estimator and
`tools/gpu/probe-pae-esmfold2.js` collects the features. It is a **pAE** in the
literal sense - a predicted aligned error - and the mechanism being a read-off
rather than a head does not change what the matrix is. Nine sequences folded
through both models, leave-one-target-out:

| | median Spearman | worst |
|---|---|---|
| AF3's own distogram (the ceiling) | 0.881 | 0.718 |
| **EF2-fast's distogram** | **0.746** | 0.460 |
| geometry alone | 0.556 | 0.404 |

🔴 **AND THE SPREAD IS EXPLAINED BY WHETHER THE TWO MODELS FOLD THE SAME
THING.** The target is AF3's PAE, which is about AF3's OWN structure, while the
estimate is about EF2-fast's - so where they disagree the comparison is invalid
rather than the estimate wrong. That is not a hand-wave: the correlation between
the two models' distance matrices predicts the score at **Pearson 0.772**, and
splitting on it,

| | this estimate | AF3's own head |
|---|---|---|
| same fold (agreement >= 0.9, n=4) | **0.790** | 0.900 |
| different fold (n=5) | 0.548 | - |

so on comparable targets a distogram read-off comes within 0.11 of a dedicated
confidence head. `tools/pae-transfer.py` prints the agreement column beside
every score for exactly this reason.

🔴 **EVERY FEATURE IS IN ANGSTROMS, OR ONE FIT COULD NOT SERVE TWO GRIDS.**
AF3's distogram is 64 bins over 2-22 A and EF2-fast's is 128 over a borrowed
2-52, so a spread read in BIN INDICES differs by a factor of two between them
for the same physical uncertainty. Entropy is worse: it is not the same unit at
all, since a uniform distribution over 128 bins carries log 2 more nats than one
over 64 for free - so it enters as `exp(H) * binWidth`, an effective width.

🔴 **AND THE UNCONSTRAINED FIT WENT DOWN WHEN THE MODEL GOT LESS SURE.** sigma
and the mobility terms are strongly correlated - a mobility IS a mean of sigma -
so least squares gave them large opposite signs, and raising every pair's spread
by 1 A moved the estimate by **-0.513 A**. Pinning that one aggregate direction
to zero removes it and IMPROVES the fit, 0.738 -> **0.746**, which is what
removing a spurious direction looks like. Ridge is the obvious alternative and a
far worse trade: the slope only turns positive at lambda 1e5, where the median
has fallen to 0.661. **One bad direction wants one constraint, not blanket
shrinkage.**

🔴 **AND THE CONSTRAINT HAS TO NAME EVERY FEATURE THAT MOVES.** `effWidth` is
`exp(H) * binWidth` and a Gaussian's `exp(H)` is `sigma * sqrt(2 pi e)`, so a
uniform widening moves it by 4.13 per angstrom rather than not at all. A first
constraint over sigma and the mobilities alone left a residual slope of -0.15 -
small enough to read as rounding and still the wrong sign. `test/esmfold2-aligned-error.test.js`
caught it, and asserts the scale-invariance directly.

🔴 **AND THE ESTIMATE IS A CONTRAST, WHICH IS WHAT THE TEST HAD TO BE TAUGHT.**
Widening every pair a token takes part in raises that token's own mobility
baseline, which the estimate normalises against - so its row can correctly stay
put or fall. The property that holds is about ONE pair against its two tokens'
baselines, and the first version of that test asserted the row and failed
against correct code.

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

## A language model instead of an alignment

The second half of this file is the investigation that preceded the port:
whether ESM-C 600M can be compressed enough to fold from a single sequence in a
browser, which is the case an MSA search cannot serve at all. The three things
worth knowing without reading it:

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


---

# Before the port: can a language model replace the MSA search?

What follows was written while nothing here shipped, and it answers one
question - **can a protein language model replace the MSA search in a
browser** - with the measurements that answer it. The tools are
`tools/esmc/`. It is kept because the compression numbers are still the
basis of every bundle above.


## Why ask

LocalFold's alignment comes from an MMseqs2 server that queues, and a target
with no homologues gets nothing useful back from it at any wait. ESMFold2-Fast
folds from ESM-C's hidden states instead of from an alignment: no search, no
round trip, no database. The whole question is whether the language model fits.

## The models, and which one this is about

Biohub publish ESMFold2 (MSA-capable) and **ESMFold2-Fast** (single sequence,
no MSA encoder at all), each in a released and an *experimental* line, and the
experimental line additionally as a per-LM-size ablation series that exists to
reproduce the paper.

| | LM | folding model | tower |
|---|---|---|---|
| `ESMFold2-Experimental-Fast` | ESM-C 6B | 179.3 M | 6352 M |
| `...-Fast-base600M-step1500k` | ESM-C 600M | 171.2 M | **573.3 M** |
| `...-Fast-base300M-step1500k` | ESM-C 300M | 171.1 M | 333 M |

The folding model is the same size in all three; only the tower changes. All are
ungated and MIT - there is no terms dialog to answer and nothing
`build_site.py` has to refuse, which is a real difference from AF3.

🔴 **THE TOWER IS PAIRED WITH ONE CHECKPOINT OF ITSELF, NOT WITH A WIDTH.** The
folding model's layer mix is a `(n_layers + 1,)` vector trained against exactly
the tower named in its `esmc_id` - `biohub/ESMC-600M-1500000`, which is NOT the
general-release `esmc-600m-2024-12`. `tools/esmc/fetch.py` reads the pairing out
of the folding model's own config rather than writing it down twice.

🔴 **AND THE `-step` CHECKPOINTS HAVE NO CONFIDENCE HEAD.**
`confidence_head.enabled: false`, zero confidence tensors in the file - so no
pLDDT, no PAE, no pTM. The fold succeeds and `output_to_pdb` raises `KeyError:
'plddt'` on the way out. LocalFold colours by pLDDT and its scores card is built
on it, so a shipped 600M path needs a checkpoint that has one; the ablation
series is for measuring, not for shipping.

## The first 2x is not compression

🔴 **THE CHECKPOINT IS bfloat16 STORED AS float32.** Not one of the 95.6M
weights sampled has any of its low sixteen mantissa bits set. The tower was
trained in bfloat16 and widened on the way out, and ESMFold2's own forward
casts it straight back (`torch.autocast(device_type=..., dtype=torch.bfloat16)`
around the ESM-C call; the experimental loader's `finalize_esmc` is
`self.esmc.bfloat16()`). So half of the 2.30 GB download is zeros.

Consequences, both measured rather than argued:

* the bfloat16 arm reads relRMS **exactly 0.00e+00** on the weights and on every
  downstream column, which is what a correct harness must say and is also how
  this was found - a scheme that costs nothing is either free or not running.
* float16 reads **3.2e-9**, which is not rounding. bfloat16 holds 7 mantissa
  bits and float16 holds 10, so every value in range is exact; the residual is
  the tail below float16's subnormal floor, and the tower's smallest weights go
  down to 3.9e-10.

**So quote a scheme against the 16 bits that are really there.** A table
starting at "32.00 bits" flatters every row in it by a factor of two.

## What compression costs, on the tensor ESMFold2 actually reads

`python3 tools/esmc/probe-esmc-compression.py`. Three targets - ubiquitin and
the two crystal structures the AF3 side already scores against - and the error
is reported at four depths, because they do not agree and only the last one is
the question:

* `weights` - what a weight-space study would report.
* `states` - the flat mean over ESM-C's 37 hidden states.
* `mixed` - the same, weighted by ESMFold2's learned softmax over them.
* `pair` - the pair representation the folding trunk receives. **This is the
  column that prices a scheme.**

| scheme | bits/w | MiB | weights | states | mixed | pair |
|---|---:|---:|---:|---:|---:|---:|
| float32 (control) | 32.00 | 2187 | 0 | 0 | 0 | 0 |
| **bfloat16** | 16.00 | 1094 | **0** | **0** | **0** | **0** |
| float16 | 16.00 | 1094 | 3.5e-9 | 6.7e-7 | 1.3e-6 | 1.2e-6 |
| int8 g64 sym | 8.25 | 564 | 6.1e-3 | 5.4e-3 | 9.6e-3 | 7.4e-3 |
| int8 g128 asym | 8.25 | 564 | 6.0e-3 | 5.3e-3 | 9.2e-3 | 7.3e-3 |
| int6 g32 asym | 7.00 | 478 | 1.9e-2 | 1.6e-2 | 2.8e-2 | 2.2e-2 |
| int6 g64 asym | 6.50 | 444 | 2.2e-2 | 1.8e-2 | 3.2e-2 | 2.5e-2 |
| **int5 g32 asym** | 6.00 | 410 | 3.8e-2 | 3.3e-2 | 5.7e-2 | **4.4e-2** |
| int5 g32 asym+search | 6.00 | 410 | 3.7e-2 | 3.5e-2 | 5.8e-2 | 4.4e-2 |
| int5 g64 asym+2 outliers | 5.69 | 389 | 3.5e-2 | 3.1e-2 | 5.4e-2 | 4.4e-2 |
| int5 g64 asym | 5.50 | 376 | 4.4e-2 | 3.7e-2 | 6.4e-2 | 4.9e-2 |
| int4 g64 asym+2 outliers | 5.19 | 355 | 7.2e-2 | 6.1e-2 | 1.1e-1 | 8.0e-2 |
| int4 g32 asym | 5.00 | 342 | 7.9e-2 | 6.9e-2 | 1.2e-1 | 9.4e-2 |
| int4 g32 asym+search | 5.00 | 342 | 7.6e-2 | 6.9e-2 | 1.2e-1 | 8.8e-2 |
| int4 g64 asym | 4.50 | 308 | 9.1e-2 | 7.7e-2 | 1.3e-1 | 1.0e-1 |
| int3 g32 asym | 4.00 | 273 | 1.7e-1 | 1.5e-1 | 2.5e-1 | 1.9e-1 |
| int3 g32 asym+search | 4.00 | 273 | 1.6e-1 | 1.6e-1 | 2.5e-1 | 1.8e-1 |
| int2 g32 asym | 3.00 | 205 | 4.0e-1 | 4.1e-1 | 7.4e-1 | 5.2e-1 |

Three things worth taking from it.

**ESM-C quantises like AF3 does.** int5 group 32 asymmetric costs relRMS 3.8e-2
in weight space here against 4.3e-2 on AF3's six biggest tensors
(`tools/analyse_quantisation.py`) - the same scheme, the same neighbourhood, a
different model. The AF3 frontier's shape transfers: the zero point is the big
win at low precision, searching the range is worth a few percent, and pulling
outliers out beats clipping them.

**The tower does not amplify.** weights 3.8e-2 -> mixed 5.7e-2 -> pair 4.4e-2 at
int5. Thirty-six residual blocks pass the error through at roughly unit gain,
and the two LayerNorms in the shim take some of it back off. A stack that
amplified would show it here and would settle the question by itself.

**Searching the range stops paying above four bits.** 7% off the pair error at
int4 (9.4e-2 -> 8.8e-2), nothing at all at int5, and it costs a minute a tensor
sweep. Same for outliers: 5.69 bits with two outliers buys exactly what 6.00
bits buys without them, so it is 0.31 bits of saving for a packing format
nothing here can read.

## And what that costs the STRUCTURE

    python3 tools/esmc/fetch_targets.py --out targets --count 16
    .venv-esm/bin/python tools/esmc/probe-esmfold2-structure.py \
        --pdb-dir targets --seed-spread 2 --csv rows.csv

Sixteen single protein chains, 67-205 residues, X-ray, better than 2.0 A,
**released after the checkpoint's September 2021 cutoff** and filtered so no two
are near-duplicates - the RCSB returns deposition groups together and a plain
"newest 20" came back as nine consecutive entries of the same protein, which
would read as nine measurements and be one.

The folding model is loaded once and `forward` takes `lm_hidden_states`
directly, so an arm replaces exactly the tensor the previous table prices and
nothing else - not the folding weights, not the diffusion noise, not the seed.

🔴 **AND THE SAMPLER'S OWN SPREAD IS MEASURED FIRST, BECAUSE NOTHING ELSE HERE
MEANS ANYTHING WITHOUT IT.** The same float32 weights at three seeds, one
diffusion sample each: **mean 0.99 A, worst 7.06 A** over 32 folds. A scheme
that moves a structure by less than that has not moved it.

| scheme | bits/w | bundle | moved: median | mean | worst | targets past 1 A | median vs crystal |
|---|---:|---:|---:|---:|---:|---:|---:|
| float32 | 32.00 | 2840 MiB | - | - | - | - | **2.52 A** |
| **no language model** | - | - | 16.04 A | 15.52 | 23.77 | **16/16** | 17.33 A |
| int8 g64 sym | 8.25 | 732 MiB | 0.04 A | 0.10 | 0.84 | **0/16** | 2.52 A |
| **int5 g32 asym** | 6.00 | **533 MiB** | 0.08 A | 0.43 | 3.32 | 2/16 | **2.53 A** |
| int4 g32 asym | 5.00 | 444 MiB | 0.12 A | 1.27 | 13.17 | 3/16 | 2.53 A |
| int3 g32 asym | 4.00 | 355 MiB | 0.26 A | 1.71 | 15.22 | 4/16 | 2.57 A |

🔴 **THE DAMAGE IS A TAIL, NOT A DEGRADATION.** The median target moves 0.26 A
even at three bits and the median crystal RMSD is FLAT from float32 to int3 -
2.52, 2.52, 2.53, 2.53, 2.57. What actually changes is how many targets get
knocked into a different basin entirely: none at int8, two at int5, three at
int4, four at int3, with worst cases of 13 and 15 A. So a mean is the wrong
summary here and a median alone is too kind; the honest column is **how many
targets moved further than the sampler moves them by itself**.

🔴 **int5 IS FREE ON THAT READING, AND THE SAME SCHEME IS FREE ON AF3.** Its
mean displacement (0.43 A) and its worst (3.32 A) are both inside the sampler's
own 0.99 / 7.06, and its median crystal RMSD is 2.53 against float32's 2.52.
`tools/quantize_af3.py` records int5 group-32 asymmetric costing AF3 nothing
either (0.66 A against float32's 0.69, inside the spread between diffusion
seeds). Two models, two graphs, one packer, the same verdict - and LocalFold
already has the GPU decoder for it (`src/runtime/quantised-upload.js`).

🔴 **int4 IS THE EDGE AND int3 IS OVER IT**, which is again where AF3 lands.
int4's worst case (13.17 A) is nearly twice the worst the sampler produces on
its own, so it breaks targets seeding does not. The floor is about **444 MiB**
for the 600M pair if a rare flipped fold is acceptable and **533 MiB** if it is
not.

🔴 **AND A pair RELATIVE ERROR IS A POOR PREDICTOR OF ANGSTROMS.** int5's 4.4e-2
and int4's 9.4e-2 are a factor of 2.1 apart in the pair representation and a
factor of 3 apart in mean displacement - but 4x apart in the worst case, which
is the number that decides. Use the pair column to choose which schemes are
worth folding; price them on the structure.

## Halving the tower again costs nothing measurable

The other compression axis is the tower itself, and Biohub publish the ablation:
`...-Fast-base300M-step1500k` is the same folding model against ESM-C 300M (30
layers x 960, 333 M parameters). Same sixteen targets, same protocol:

| | median vs crystal | mean | seed-to-seed |
|---|---:|---:|---:|
| ESM-C 600M, float32 | **2.52 A** | 4.88 | 0.99 A |
| ESM-C 300M, float32 | **2.55 A** | 5.31 | 1.10 A |
| ESM-C 300M, int8 | 2.55 A | 5.32 | |
| ESM-C 300M, int5 g32 | 2.57 A | 5.27 | |
| ESM-C 300M, int4 g32 | 2.54 A | 5.26 | |

**0.03 A of median between a tower of 573 M parameters and one of 333 M**, which
is a third of the seed-to-seed spread. On the tails they differ - 600M's mean is
0.4 A better - but on this set the small tower is not the limiting factor.

That puts the **300M pair at int5 at 361 MiB**, against LocalFold's own
`model-af3-int5` at 264.6 MiB and AF2 monomer at 227 MB. It is the smallest
configuration measured that still folds, and it is in the same class as what the
page already ships.

| bundle | float32 | float16 | int8 | **int5** | int4 |
|---|---:|---:|---:|---:|---:|
| ESM-C 300M + folding model | 1923 | 961 | 496 | **361** | 300 |
| ESM-C 600M + folding model | 2840 | 1420 | 732 | **533** | 444 |
| ESM-C 6B + folding model | 24915 | 12457 | 6423 | 4672 | 3893 |

MiB.

## The catch, which is not about compression at all

🔴 **THE COMPRESSIBLE MODEL IS THE WEAK ONE AND THE STRONG ONE IS NOT
COMPRESSIBLE ENOUGH.** The `base300M` and `base600M` checkpoints are paper
ablations - Biohub's own README says "please use ESMFold2 for research work" -
and the RELEASED `ESMFold2-Fast` folds from ESM-C **6B**, which is 4672 MiB at
int5 and out of reach at any precision this study reached. So a browser build
would be shipping an ablation checkpoint, and its accuracy is what the tables
above measure: **median 2.52 A on held-out targets, mean 4.88 A, and two of
sixteen outright failures** (7ILM at 11.9 A, 9TLM at 26.0 A). Compression is not
the obstacle. Whether that accuracy is worth having when no alignment exists is
the actual question, and it is a product question rather than a measurement one.

🔴 **AND THOSE CHECKPOINTS HAVE NO CONFIDENCE HEAD**, so no pLDDT, no PAE, no
pTM - see the model table above. LocalFold colours by pLDDT, its scores card is
built on it, and its archive writer emits it. That is not a small gap to paper
over.

## Chasing 200 MiB, and where the floor actually is

A 200 MiB bundle is an arithmetic statement before it is an experiment:

| bundle | parameters | bits/weight it allows |
|---|---:|---:|
| ESM-C 600M + folding model | 744.5 M | **2.25** |
| ESM-C 300M + folding model | 504.1 M | **3.33** |

🔴 **SO 200 MiB IS OUT OF REACH FOR THE 600M TOWER BEFORE ANYTHING IS
MEASURED.** 2.25 bits a weight is int2 at group 128, the far corner of the
format, and int2 at group 32 - a strictly more generous setting - already reads
pair relRMS 5.2e-1. There is no training run behind that.

### The half of the bundle that had never been priced

🔴 **THE FOLDING MODEL IS 171 M PARAMETERS AND EVERY TABLE ABOVE IGNORED IT.**
It is 41% of the 300M bundle at equal bits, so a size target measured on the
tower alone is a size target for three fifths of the download.
`--fold-bits` quantises it by the same rule `tools/quantize_af3.py` uses - 379
tensors touched, 439 kept float32, 99.2% of the parameters reached. Sixteen
held-out targets, tower held at int5:

| folding model | MiB | median vs crystal |
|---|---:|---:|
| float32 | 653 | 2.55 A |
| int8 g32 | 168 | 2.57 A |
| int5 g32 | 127 | 2.53 A |
| int4 g32 | 107 | 2.54 A |
| **int3 g32** | **86** | **2.58 A** |
| int3 g128 | 71 | **4.59 A** |

🔴 **IT GIVES UP BITS ALMOST FOR FREE AND WILL NOT GIVE UP ITS GROUP.** Three
bits at group 32 costs 0.03 A of median; three bits at group 128 costs **two
angstroms**, and it does so underneath everything else, so a tower ladder run
on top of it measures the folding model and reports the tower. That is how the
first version of this table was nearly written. The reason is the one
`docs/AF3.md` already records for AF3's atom decoder: this model's output is a
POSITION, in angstroms, and nothing downstream renormalises a relative error
in it.

### The tower, with a folding model that is not the problem

Folding model pinned at int3 group 32, ESM-C 300M, same sixteen targets:

| tower | bits/w | tower MiB | moved: mean | TM | median vs crystal |
|---|---:|---:|---:|---:|---:|
| int5 g32 | 6.00 | 238 | 0.53 A | 0.976 | 2.58 A |
| **int3 g32** | 4.00 | 159 | 1.15 A | 0.937 | **2.46 A** |
| **int3 g64** | 3.50 | 139 | 1.44 A | 0.921 | **2.43 A** |
| int2 g32 | 3.00 | 119 | **8.81 A** | **0.484** | **12.15 A** |
| int2 g64 | 2.50 | 99 | **10.04 A** | **0.395** | **12.24 A** |

with `no language model` at 11.99 A / TM 0.299 / median 13.30 A.

🔴 **THE THREE-TO-TWO BIT STEP IS A CLIFF, NOT A SLOPE.** int3 group 64 still
folds - median 2.43 A against the float32 tower's 2.55, inside the sampler's
own spread - and int2 group 32, half a bit later, is **within a whisker of
having no language model at all**: 12.15 A median against 13.30 A for feeding
the trunk nothing. Two bits does not degrade this tower, it deletes it.

### So the floor, by scalar group quantisation, is about 225 MiB

| | tower | folding model | total |
|---|---:|---:|---:|
| comfortable | int3 g32, 159 | int3 g32, 86 | **245 MiB** |
| **tightest that folds** | int3 g64, 139 | int3 g32, 86 | **225 MiB** |
| under 200, and broken | int2 g64, 99 | int3 g32, 86 | 186 MiB |

🔴 **NOTHING IN THE FORMAT REACHES 200 MiB WHILE STILL FOLDING.** The only
scalar combinations that fit put two bits on the tower, and two bits is the
cliff. Getting under 200 needs a different representation - QuIP#-style
incoherence processing with vector codebooks is the class that makes 2-bit
language models work - and that is a new WebGPU decoder, not a new packer.

## What the AF3 port learned about the language model, and what it means here

`../alphafold3` now folds all six ESMFold2 releases against native, and getting
there turned up four things this directory has to know.

🔴 **THE SHIM IS PER MODEL, NOT PER FAMILY.** Every release trains its own
`language_model.*` - the module that turns ESM-C's 37 hidden states into a pair
representation. They share the TOWER and nothing else. The AF3 port shipped one
`esmfold2.lm.npz` for the whole family and fed every variant the BASE model's,
which against native's own `lm_z` for that variant reads **corr 0.026** where the
variant's own shim reads **0.999998**. It cost `esmfold2_exp_fast` **8.798 A
against 0.812** on 6MRR.

🔴 **THIS DIRECTORY AVOIDS IT BY CONSTRUCTION, AND THAT IS WORTH KNOWING BEFORE
SOMEBODY OPTIMISES IT.** `Shim.__init__` reads `language_model.*` out of the
folding checkpoint it is handed, so the tower and the shim cannot come from
different models without the caller naming two directories. Verified against
the `esm` package's own module on the 600M experimental-fast checkpoint:

| | |
|---|---|
| `Shim.pair` against native `model.language_model` | **relRMS 3.20e-7** |
| correlation | 1.000000 |
| standard deviation | 3.354 against 3.354 |

The failure mode returns the moment anyone caches a precomputed `lm_pair` or an
`lm.npz` beside the tower - which is exactly the shape of optimisation this
repository likes, and exactly what made the mistake easy upstream.

🔴 **AND BETWEEN 600M AND 300M IT WOULD RAISE, WHICH IS NOT REASSURANCE.** Their
shims differ in SHAPE - 37 mix entries against 31, a (256, 1152) projection
against (256, 960) - so crossing those two is a loud error. Upstream's case was
two releases of the SAME width, where it is silent and folds anyway. Any check
here has to discriminate on the weights, not on the shapes.

🔴 **AND THE EXPERIMENTAL LINE IS A DIFFERENT IMPLEMENTATION, NOT A DIFFERENT
CONFIG.** `ESMFold2ExperimentalModel` is its own class. Three divergences the
AF3 port found by reading it rather than inferring from `config.json`, all of
which a LocalFold port would have to carry:

* **no `lm_encoder`**: the shim's output is added straight to `z_init`, ONCE,
  outside the recycle loop. Reading the released line's early-return here meant
  the language model never reached the trunk at all - and it still folded, one
  variant at 1.694 A, with ESM-C changing the answer by nothing to three
  decimals.
* **`lm_dropout` is 0.0**, not the released line's 0.25, which lives in
  `lm_encoder.per_loop_lm_dropout` and therefore does not exist here.
* the MSA encoder runs AFTER the recycle and is ADDED - moot for "fast", which
  has no MSA encoder at all.

🔴 **AND ALL SIX VARIANTS THE AF3 PORT WIRES IN USE ESM-C 6B.** Its registry
says so in as many words ("all six share ESM-C 6B") and there is no reference to
a 600M tower anywhere in that tree. The checkpoints this document compresses -
`...-base600M-step1500k` and `...-base300M-step1500k` - are the paper's LM-size
ablation series and are not in it. So the model that is validated end to end is
not the model that fits a browser, and the model that fits a browser is not
validated end to end. See the note on the confidence head above.

## The 600M bundle, all the way down

Everything above measures the tower with a float32 folding model, which prices
about two thirds of a download. This is the whole bundle: ESM-C 600M plus the
folding model at int3 group 32, sixteen held-out targets, against a seed
yardstick of **2.43 A mean and 11.00 A worst** over 32 folds of identical
weights.

| bundle | tower scheme | bits/w | median vs crystal | mean | past the seed noise |
|---:|---|---:|---:|---:|---:|
| - | float32 | 32.00 | **2.13 A** | 5.17 | - |
| 496 MiB | int5 g32 | 6.00 | 2.12 A | 5.22 | 2/16 |
| 428 MiB | int4 g32 | 5.00 | 2.19 A | 5.12 | 2/16 |
| 360 MiB | int3 g32 | 4.00 | 2.39 A | 5.21 | 2/16 |
| 325 MiB | int3 g64 | 3.50 | **2.13 A** | 5.19 | 2/16 |
| 308 MiB | int3 g128 | 3.25 | **2.13 A** | 5.19 | 2/16 |
| **308 MiB** | **codebook 4096 x 4d** | 3.25 | 2.32 A | 5.33 | **1/16** |
| **274 MiB** | **codebook 1024 x 4d** | 2.75 | 2.37 A | 5.26 | **1/16** |
| 291 MiB | int2 g32 | 3.00 | 6.16 A | 8.52 | 10/16 |
| 257 MiB | int2 g64 | 2.50 | 11.37 A | - | - |
| 240 MiB | codebook 256 x 4d | 2.25 | 7.13 A | 11.71 | 9/16 |
| - | *no language model* | - | 14.03 A | 17.12 | 16/16 |

🔴 **THE FLOOR IS 274 MiB, AND IT WAS 533 WHEN THIS DOCUMENT STARTED.** Three
separate things moved it, none of them a better packer: quantising the FOLDING
MODEL, which nobody had priced and which gives up bits almost for free; opening
the tower's GROUP from 32 to 128, which is a quarter of a bit at no measurable
cost; and replacing the uniform grid with a codebook.

🔴 **AND THE TWO-BIT CLIFF IS A PROPERTY OF THE GRID, NOT OF THE BUDGET.**
Scalar int2 group 32 spends **3.00** bits a weight and folds at 6.16 A median;
a 1024-entry codebook spends **2.75** and folds at 2.37 A. A uniform grid puts
its levels where there are no weights, and at two bits there are too few levels
to waste any. That is the whole of the difference - same information budget,
one representation folds and the other does not.

🔴 **AND AT MATCHED RATE THE CODEBOOK IS THE MORE ROBUST ONE, WHICH THE MEDIAN
HIDES.** At 3.25 bits, scalar int3 g128 and the 4096-entry codebook read 2.13 A
and 2.32 A of median crystal RMSD - the scalar arm looks better. On the tail it
is the other way round: the codebook moves **1 of 16** targets past the
sampler's own spread against the scalar arm's 2, and **none at all** past 5 A
against the scalar arm's 1, with a worst case of 3.43 A against 9.06 A. The
median is where these schemes agree; the tail is where they differ, and the
tail is what a user notices.

### What the codebook costs to decode, which is less than what it replaces

🔴 **A TABLE LOOKUP IS CHEAPER THAN UNPACKING FIVE BITS.** `fit-codebook.py`
writes ONE shared table for the whole tower - 1024 entries of 4 float16 is
**8 KB** - so decoding a weight is an index into it and a multiply by the
group's scale, against int5's shift-mask-across-a-byte-boundary. LocalFold
already expands quantised weights into a dense float16 buffer in one dispatch
(`src/runtime/quantised-upload.js`); this is that same dispatch with a simpler
body. It is not the same shader, but it is not a harder one.

🔴 **AND THE ROTATION - QuIP#'s OTHER HALF - IS NOT WORTH IT HERE.** Multiplying
by a random orthogonal matrix before quantising is what makes 2-bit language
models work in the literature, and it is the expensive half: the kernel has to
un-rotate at run time. On ESM-C's weights it buys **1%** (relRMS 0.2090 against
0.2106 at 2.75 bits) and 5.6% on the scalar path. Measured before building it,
which is the only reason it was not built.

🔴 **AND AN EMPTY CENTRE IS A WASTED CODE.** Lloyd's algorithm strands centres
that nothing selects, so a 1024-entry table silently becomes a 900-entry table
at the same price. Restarting a dead centre on the vectors currently worst
served is four lines and it is why all three tables report every entry used.

🔴 **AND THE GROUP SCALE IS AN RMS, NOT A MAXIMUM.** A shared table only works
if every group hands it the same distribution, and dividing by the group's
largest weight standardises the OUTLIER rather than the bulk. There is also no
zero point: a scalar scheme needs one because its grid is symmetric and its
weights are not, and a codebook's entries are already wherever k-means put
them. That is half a bit per group not spent.

### And what sparsity is worth, which is nothing here

Zeroing weights reaches rates dense quantisation cannot - 1:8 with int4
survivors is 1.00 bits a weight - but it is dominated everywhere the model
still folds:

| scheme | bits/w | reconstruction relRMS |
|---|---:|---:|
| 2:8 int4 | 1.85 | 0.5442 |
| scalar int2 g32 | 3.00 | 0.3947 |
| 2:4 int4 | 3.15 | 0.3468 |
| **scalar int3 g64** | **3.50** | **0.2093** |
| 2:4 int8 | 5.15 | 0.3383 |
| **scalar int4 g32** | **5.00** | **0.0845** |

🔴 **BECAUSE THE DISCARDED ENERGY IS A FLOOR NO PRECISION RECOVERS.** The top 4
of every 8 weights carry 96.5% of the energy, so 2:4 throws away 3.5% and
cannot beat relRMS **0.187** at any survivor precision - and scalar int3 g64
*achieves* 0.209 at 3.50 bits. The same table shows it directly: 2:4 with int4
survivors reads 0.3468 and with int8 survivors 0.3383, so **two extra bits a
weight buy 2%**. The error is what was zeroed, not how the rest is stored.

🔴 **AND THE MASK IS NOT FREE.** "90% of the weights are zero" says nothing
until the reader is told WHICH. An arbitrary mask costs the binary entropy of
the density - 0.54 bits a weight at one in eight, more than the surviving
values themselves. n:m fixes the count per block so the mask is log2(C(m,n))/m
and needs no search, which is the only version worth quoting.

## Calibrated quantisation, which is what the LLM world does instead

Everything above rounds each weight to the nearest code and looks at nothing
else. The methods that made 3- and 4-bit language models usable all ask a
different question - *what codes keep this LAYER'S OUTPUT the same on real
data* - and they are cheap enough to be worth trying here. `tools/esmc/gptq.py`
implements two, `tools/esmc/calibrate-esmc.py` drives them, and the calibration
set is 288 UniRef50 sequences (three length buckets, 96 each).

| | what it does | cost |
|---|---|---|
| **RTN** | nearest code, per-group affine range | free |
| **imatrix** | keeps RTN's codes, picks each group's RANGE to minimise error weighted by that channel's activation energy. llama.cpp's importance matrix | one forward pass |
| **GPTQ** | quantise column by column, pushing each column's rounding error into the columns not yet done, along the inverse Hessian of `2 XᵀX` | one forward pass + a Cholesky per matmul |
| AWQ | search a per-input-channel scale, fold it into the neighbouring op | comparable |
| SpQR / SqueezeLLM | keep the few extreme weights per group in float16 | measured above: 0.31 bits |
| QuIP# / AQLM | random rotations to kill outliers, then vector codebooks | **needs a different decoder** |
| SmoothQuant, LLM.int8() | move activation outliers into the weights | **irrelevant here** - LocalFold quantises STORAGE and computes in f16/f32, so there are no activation outliers to migrate |

🔴 **THE BLOCKS ARE CALIBRATED IN ORDER AND SO ARE THE FOUR MATMULS INSIDE ONE**,
so every layer corrects for the error its predecessors actually made rather
than for an error nothing will make. Five passes per block instead of one, and
the whole 573M tower takes **70 seconds on an A100**.

🔴 **AND NEITHER METHOD CHANGES THE STORAGE FORMAT, WHICH IS WHY THESE TWO AND
NOT THE OTHERS.** Both emit exactly what `src/runtime/quantised-upload.js`
already decodes: asymmetric codes, one float16 scale and one float16 zero per
group of 32. GPTQ's group axis lines up for free - LocalFold groups 32
CONSECUTIVE elements of a row-major `(out, in)` tensor, which is 32 consecutive
input channels of one output channel, which is what GPTQ calls `group_size`.
`act-order` is deliberately not implemented: it is worth a few tenths of a bit
and it permutes the input axis, so the groups stop being consecutive and the
shader would need the permutation.

### What they buy

Same sixteen held-out targets, same protocol, same seed yardstick (1.00 A mean,
7.16 A worst between two seeds of identical weights):

| arm | bits/w | tower | moved: mean | TM | median | worst | past 1 A |
|---|---:|---:|---:|---:|---:|---:|---:|
| **GPTQ int5** | 6.00 | 410 MiB | **0.39 A** | **0.979** | 0.06 | 3.07 | 2/16 |
| RTN int5 | 6.00 | 410 MiB | 0.44 A | 0.977 | 0.08 | 3.29 | 2/16 |
| **GPTQ int4** | 5.00 | 342 MiB | **1.14 A** | **0.947** | 0.10 | 13.33 | 3/16 |
| imatrix int4 | 5.00 | 342 MiB | 1.16 A | 0.940 | 0.08 | **9.25** | 3/16 |
| RTN int4 | 5.00 | 342 MiB | 1.29 A | 0.933 | 0.12 | 13.26 | 3/16 |
| **imatrix int3** | 4.00 | 273 MiB | **1.29 A** | **0.929** | 0.20 | 11.20 | 3/16 |
| RTN int3 | 4.00 | 273 MiB | 1.72 A | 0.920 | 0.27 | 15.09 | 4/16 |
| GPTQ int3 | 4.00 | 273 MiB | 1.87 A | 0.904 | 0.60 | 14.03 | **7/16** |

🔴 **GPTQ MAKES THE WEIGHTS WORSE AND THE OUTPUT BETTER, WHICH IS THE WHOLE
IDEA.** At int4 it moves the weights 9.78e-2 from float32 where plain rounding
moves them 7.91e-2 - and the pair representation it produces is 7.87e-2 against
rounding's 9.42e-2. A weight-space study would have reported GPTQ as the worse
method. `tools/analyse_quantisation.py` is a weight-space study.

🔴 **CALIBRATION IS WORTH ABOUT ONE BIT, AND ONLY AT THE BOTTOM.** `imatrix`
int3 at **4.00 bits** matches plain int4 at **5.00 bits** on every column - 1.29
A mean, TM 0.929 against 0.933 - which is a whole bit for one forward pass. At
int5 there is nothing left to win: GPTQ's 0.39 A against rounding's 0.44 A is
inside the sampler's own 1.00 A, so both are the same answer.

🔴 **AND IT DOES NOT CLOSE THE int4 -> int5 GAP.** GPTQ int4 is 1.14 A where
plain int5 is 0.44 A. Calibration recovers about 15% of a bit-step, not a whole
one, so **int5 remains the smallest scheme that is free** and 533 MiB (600M) /
361 MiB (300M) stands as the answer.

🔴 **AND GPTQ IS THE WORST ARM AT THREE BITS, WHICH THE PAIR METRIC DOES NOT
SAY.** Its pair error (1.77e-1) beats plain rounding's (1.92e-1) and its folds
are worse on every structural column - mean 1.87 A against 1.72, TM 0.904
against 0.920, and **seven of sixteen targets past the seed noise against four**.
Pushing a large rounding error down the Hessian spreads it over columns that
were fine; at three bits there is more error than there is room to put it. The
two metrics disagree in SIGN here, which is the strongest argument in this file
for folding the structure rather than trusting a tensor norm.

### Would quantisation-aware TRAINING go further

Probably, by about another bit, and it is not obviously worth it.

The literature's ordering is RTN < imatrix/AWQ < GPTQ < QAT, and the step from
GPTQ to QAT is worth roughly what the step from RTN to GPTQ is - which here was
one bit at the bottom and nothing at the top. The right objective would not be
the masked-LM loss ESM-C was trained on: ESMFold2 never reads the logits. It
would be **self-distillation against the float32 tower's own hidden states** -
no labels, just UniRef50 sequences and the teacher already on disk - because
those are literally the tensors the shim mixes. Straight-through estimator on
the codes, the scales and zeros left as they are so the format does not move.

Cost on this A100: teacher forward plus student forward and backward, about 3x
a plain pass, so ~1-2 days for a few thousand steps at 573M parameters. The
prize is the 600M bundle at 355 MiB instead of 533, or the **300M bundle at
about 240 MiB - which is AF2 monomer's 227 MB**.

🔴 **BUT THE BINDING CONSTRAINT IS NOT SIZE.** Both towers already fit in the
same class as what the page ships, the compute is a tenth of an AF3 trunk pass,
and the checkpoints that fit are paper ablations with **no confidence head at
all** and a median 2.52 A on held-out targets with two failures in sixteen.
Spending two days of GPU to move 533 MiB to 355 MiB does not change any of
that. The experiment to run before any training run is whether ESMFold2-Fast is
accurate enough to be worth shipping at ALL, and that is a question about the
model rather than about its bytes.

## Spending bits where the layer mix is heavy does NOT work

ESMFold2 takes **58.8%** of its softmax from the last three of the 37 states and
**0.017%** from states 12-17 between them. That looks like an obvious place to
allocate precision, and it is not, at any budget tried:

| arm | bits/w | pair | the uniform scheme at that budget |
|---|---:|---:|---|
| int4 everywhere, int6 on the last 6 blocks | 5.33 | 7.4e-2 | ~6.4e-2 interpolated |
| int4 everywhere, int8 on the last 3 blocks | 5.33 | 8.5e-2 | ~6.4e-2 |
| int3 everywhere, int6 on the last 6 blocks | 4.50 | 1.4e-1 | 1.0e-1 (int4 g64) |

🔴 **BECAUSE THE MIX SAYS WHERE A STATE IS *READ*, NOT WHERE PRECISION
MATTERS.** Every block feeds every later one, so a cheap early block damages the
expensive late states too - and it damages them through 30-odd more blocks of
residual stream. Uniform allocation beats every graded one measured, by 15-40%,
which is a large enough margin not to be a tuning question.

## The forward pass is not ours alone

`tools/esmc/esmc_forward.py` is written from the block layout
`../alphafold3/converters/esmc.py` documents, deliberately not by importing the
`esm` package, so that a quantisation arm and its reference do not share a
forward. Against `transformers`' own `EsmcForMaskedLM`, on ubiquitin:

| | relRMS |
|---|---|
| hidden states 0-3 | 0.0, 1.7e-7, 2.2e-7, 2.5e-7 |
| hidden state 18 | 5.2e-7 |
| hidden states 34, 35, 36 | 2.1e-6, 2.2e-6, 2.1e-6 |
| masked-LM logits | 1.0e-6 |

Float32 accumulation order, and it pins every convention that could have been
wrong invisibly: RoPE's split-halves rotation, the QK-LayerNorm being over the
FULL `d_model` rather than per head, the fused projection's `[q|k|v]` order, the
SwiGLU's `[gate|up]` order, the ESM3 residual scale, and the final LayerNorm
being applied to the last state only.

🔴 **`transformers` CANNOT LOAD THESE CHECKPOINTS DIRECTLY.** Its ESMC expects
its own converted layout (`esmc.layers.N.self_attn.q_proj.weight`) while Biohub
publish the fused TransformerEngine one
(`esmc.transformer.blocks.N.attn.layernorm_qkv.weight`), so
`from_pretrained` loads **nothing**, reports every parameter MISSING, silently
builds an 80-layer model from a default config and answers with random weights.
The check above builds the state dict by hand. A first attempt that did not
would have been comparing against noise.

🔴 **AND THE FIRST SANITY CHECK WAS A MASKED-LM ONE, BEFORE ANY ORACLE.** Mask
every fifth residue of ubiquitin and see what the head puts back: **86.7%**
against a chance rate of about 5%. That is cheap, needs no reference, and no
wrong RoPE convention survives it.

## The shim, and why the 37 states never have to exist at once

ESMFold2's `LanguageModelEncoder` is: LayerNorm the states, project 1152 -> 256,
mix them by `softmax(base_z_combine)`, downproject, outer product carrying BOTH
a product and a difference, two-layer MLP, LayerNorm.

🔴 **THE MIX IS A CONSTANT AND THE NORM AND PROJECTION ARE SHARED ACROSS
LAYERS**, so `sum_k combine[k] * LN(h_k) @ W` is a running accumulator of
`(tokens, 256)` and the `(37, tokens, 1152)` tensor never needs to be
materialised - 4.5x smaller at any length, and the tower can be streamed one
block at a time. `Shim.accumulate` is that form and it agrees with
`Shim.single` to **1.1e-7**, which is what says it is the same arithmetic.
The downprojection stays outside the sum because it is affine and its bias must
not be added thirty-seven times.

That is the fact that decides device memory. A block is 15.9M parameters, so at
float16 the tower's resident footprint on the GPU is **32 MB one block at a
time**, not 1.15 GB - the same shape of trade `releaseResidentWeights` makes for
AF3's stages.

## Would it run

Arithmetic, not measurement, but the terms are not close:

| | |
|---|---|
| ESM-C 600M, matmuls | **1.15 GFLOP per token** |
| at 300 residues | 359 GFLOP, ~0.3 s at this M2's measured 1.0-1.5 TFLOP/s |
| weight traffic, one pass | 1147 MB at float16, **358 MB read as packed int5** |
| resident, streamed | one block, 32 MB at float16 |

For scale, `bench-trunk.js` measures AF3's own trunk at **3372 ms at 200
tokens**. The language model is a fraction of that, and ESMFold2's trunk is 24
PAIR-ONLY blocks - no triangle attention, no single track - against AF3's 48
pairformer blocks. The compute is not the problem. **The download is the whole
question**, which is why this document is mostly a quantisation table.

| bundle, tower + folding model | float32 | float16 | int8 | int5 | int4 |
|---|---:|---:|---:|---:|---:|
| ESM-C 300M | 1923 | 961 | 496 | **361** | 300 |
| ESM-C 600M | 2840 | 1420 | 732 | **533** | 444 |
| ESM-C 6B | 24915 | 12457 | 6423 | 4672 | 3893 |

MiB. For comparison LocalFold ships `model-af3-int5` at **264.6 MiB** and AF2
monomer at 227 MB. So the 600M pair at int5 is about twice the AF3 bundle and
the 6B is out of reach by an order of magnitude at any precision - which is what
makes the 600M line the interesting one and the reason the question was asked
about it.
