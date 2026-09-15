# AlphaFold 3 in LocalFold

## 🔴 TWO NEW MODELS: IntelliFold-2 AND RoseTTAFold3

Both are in af3-any-model's `ALL_MODELS`, so both are dialect ports rather than
new graphs, and the two are almost opposite in size. **Read the convention set
before the code**: the reference's `model_config.py` tuples are the authority,
and `PYTHONPATH=src ~/venv/bin/python` on the reference node with a script that
imports the module and introspects every uppercase tuple is how to get it - a
regex over the source MISSES the ones built from other tuples and misses
`TRIANGLE_MUL_DIVIDE_BY_LENGTH`, which is written as a conditional expression.

### IntelliFold-2: two conventions, and it folds

The reference names it in exactly two tuples - `KEY_MASKED_ATOM_ATTENTION` and
`MASK_ATOM_ACT_PER_BLOCK` - and it is deliberately NOT in `OPENFOLD3_LINEAGE`:
it forks boltz's FEATURISER, not OpenFold's network. Only the second was new
here, and the reference records that the two are one bug between them: with the
atom key window aligned and the key-side mask wrong, block 1 reads 9.5e-03 and
blocks 2-3 blow up on windows 16 and 17 alone.

`maskAtomActPerBlock` re-zeroes the padded atom slots at the TOP of every atom
block, because if2 pads the flat atom axis INSIDE each attention call
(`pad_at_dim(a_row, ..., value=0.)`). It reuses the mask-act kernel that already
ran once after the stack, so only its POSITION differs, and it is carried on
each atom block - like `chainedAtomLayerNorm` - so the DECODER, which sees no
dialect object, reads it off its weights the same way.

**It folds 6MRR at pLDDT 83.5, pTM 0.684, N-CA 1.44, CA-C 1.51, CA-CA 3.81,
radius of gyration 11.2 A**, and its trunk on its own reference batch sits
INSIDE the family's int5 band on every seam:

| seam, `--dump=` reference batch, int5 | intellifold2 | boltz2 | protenix2 |
|---|---:|---:|---:|
| `target_feat` | **2.41e-2** | 3.68e-2 | 5.51e-2 |
| `z_init` | **1.68e-2** | 1.64e-2 | 2.07e-2 |
| `z_after_template` | **9.51e-3** | 1.64e-2 | 1.30e-2 |
| `z_after_msa` | **9.14e-3** | 4.32e-2 | 4.64e-2 |
| `trunk_out_pair` | **4.54e-2** | 4.56e-2 | 7.15e-2 |

🔴 **THOSE ARE int5 NUMBERS AND THE SWEEP BELOW IS f32'S.** The 1e-8 column in
the older table was taken on float32 bundles; an int5 bundle's residue against
an f32 oracle is 1e-2 for every model, and the only useful reading is one model
against another on the same route. if2 is at or below boltz2 on all five.

🔴 **AND ITS FEATURISER IS WORTH 6x AT `target_feat`.** Featurised from the
sequence rather than read from the reference's batch, if2 reads **1.45e-1**
against 2.41e-2 - the shared-ideal-conformer floor plus three boltz-forked
conventions this port does not implement: `qblock_keys` (the atom key window's
edge is the atom count rounded UP to a whole 32-atom query block, so on 6MRR its
last two blocks take keys 448..575 where AF3's slide gives 446..573),
`dedupe_self_msa` (a chain with no alignments gets a DEPTH-1 MSA where AF3 hands
the query twice) and `drop_atoms` (no terminal OXT, no 5' OP3). All three are
open.

### RoseTTAFold3: ten tuples, eleven more branches, and the trunk is done

Its SHAPES are stock AF3's - c_z 128, four triangle-attention heads - so unlike
protenix2 there is no widening. Two config divergences (a 65-bin distogram and
an MSA module that holds ONE set of weights and runs it four times, which the
converter replicates so this port sees four blocks) and then a long list of
forward branches, most of them gated on the model NAME in the reference rather
than on a convention tuple. `src/af3/dialect.js`'s ROSETTAFOLD3 lists every one,
implemented or not; four are done and they are worth an order of magnitude:

| rf3 trunk, `--dump=` reference batch, int5 | start | +conformer bias | +is_paired | +OPM bias | +grid bias |
|---|---:|---:|---:|---:|---:|
| `target_feat` | 1.40e+0 | 7.00e-2 | 7.00e-2 | 7.00e-2 | **7.00e-2** |
| `z_init` | 6.61e-1 | 3.97e-2 | 3.97e-2 | 3.97e-2 | **3.97e-2** |
| `z_after_template` | 2.53e-1 | 2.63e-2 | 2.63e-2 | 2.63e-2 | **2.01e-2** |
| `z_after_msa` | 2.83e-1 | 1.36e-1 | 1.15e-1 | 6.14e-2 | **6.29e-2** |
| `trunk_out_pair` | 7.28e-1 | 1.68e-1 | 1.49e-1 | 1.29e-1 | **4.90e-2** |
| `single` | 4.00e-1 | 1.24e-1 | 2.82e-2 | 2.89e-2 | **1.47e-2** |

**4.90e-2 is boltz2's 4.56e-2**, on the same route with the same quantisation,
so rf3's TRUNK is at the family's band. Its diffusion side is not: `kq_norm`,
the `no_residual` block wiring, the chirality query term and two confidence-head
branches are declared and unimplemented, and the reference's own converter says
"WIP: converter trunk+heads+diffusion(cond+token) done; atom path + branches
next" - so an rf3 oracle dump can be trusted through the trunk and the token
transformer and NOT through the atom encoder or decoder.

**IT FOLDS.** 6MRR at **RMSD 1.68 A, TM 0.911**, which is inside af3-any-model's
own five-sample spread for rf3 (0.967 / 1.621 / 1.676 / 1.694 / 1.772, mean
1.546). Six branches took it there and no single one of them did; the table is
worth keeping because two of the six moved it and did NOT fix it, which is
exactly the state in which a correct change looks like a wrong one:

| rf3, 6MRR, each arm on top of the one above | Rg over 68 CA | CA-CA | N-CA | CA-C |
|---|---:|---:|---:|---:|
| trunk conventions only | 3.3 A | 1.32 | 0.28 | 0.43 |
| + `diffusionNoResidual`, token transformer | 3.0 | 1.50 | 0.34 | 0.48 |
| + `kq_norm`, token transformer | **10.6** | **3.29** | 0.79 | 0.97 |
| + `kq_norm`, both atom stacks | 10.7 | 3.31 | 0.87 | 1.03 |
| + `diffusionNoResidual` in the ATOM stacks, `paddedAtomKeys`, `dropTerminalAtoms` | **11.1** | **3.81** | **1.42** | **1.52** |
| (a compact 68-mer / ideal) | 11-12 | 3.80 | 1.46 | 1.52 |

The middle two rows are the lesson. `diffusionNoResidual` in the token
transformer moved the fold by almost nothing and was correct; `kq_norm` in the
atom stacks moved it by almost nothing and was correct. Judging either on the
fold alone would have backed it out.

`kq_norm` is a TRAINED LayerNorm on q and k over the FLATTENED
`num_head * key_dim` axis - not per head, so one mean and one variance serve
every head of a row - applied after the projection and before the key_dim
scaling, with two-pass variance and both a scale and an offset. Its tensors are
`transformer{query,key}_layer_norm` [6, 4, 768] in the token transformer and
[3, 128] in each atom stack, all under `__layer_stack_no_per_layer`, which for
the atom stacks is the same root their other weights take because rf3 is in
`PER_BLOCK_ATOM_PAIR_LAYER_NORM`. In the token transformer it is one workgroup
a token row between `qkvg` and `attend`; in the atom stacks q and k have
DIFFERENT row counts, so one dispatch covers `QUERY_ROWS + KEY_ROWS` and the
first `QUERY_ROWS` of them are q. It normalises `k` AFTER `expand-keys`, because
the reference's `x_k` is already in keys layout and the gather zeroes the padded
slots the reference normalises along with the rest.

🔴 **AND THE REFERENCE'S rf3 IS FINE. I SAID IT WAS WIP AND THAT WAS WRONG.**
This section previously read the note in af3-any-model's `model_registry.py` -
"WIP: converter trunk+heads+diffusion(cond+token) done; atom path + branches
next" - as a live caveat and concluded an rf3 oracle dump could not be trusted
past the token transformer. **That note is STALE**, and their own PARITY.md
contradicts it in the same checkout: rf3 is `✓` at L0, L1 pairformer, L3
denoise, L4 confidence, L5 fold and L6 modality, `~` at L2 in exactly the way
protenix2 and openbind0 are, and its own L5 log from today reads

    rosettafold3: 68 residues from 6MRR.pdb
      5 samples, CA-RMSD: 1.772 1.621 1.676 0.967 1.694   best 0.967  mean 1.546

**So every rf3 oracle number here is trustworthy and the collapse is OURS.**
Reading one file's comment as the state of a port, when the repository ships a
parity matrix and a dated log per level, was the error - and it pointed the
search at the wrong repository for an afternoon.

🔴 **AND ITS CONFIDENCE HEAD READ 53.8 pLDDT FOR THAT 1.68 A STRUCTURE.** The
last branch is `confidenceGlobalNorm`: a parameter-free LayerNorm over the WHOLE
tensor - not along the feature axis - applied to each detached trunk input
before the head reads it. **pLDDT 53.8 -> 81.5 and pTM 0.175 -> 0.843**, and
AlphaFold 3's control is unmoved at 83.08.

Three things about it are worth keeping:

  * **the statistics are over REAL TOKENS ONLY, and that is the whole
    difficulty.** A per-feature norm cannot see padding; one that reduces across
    the feature axis can. Upstream measured a 76-residue chain padded into a
    128-token bucket at PAE ~28 A everywhere and pTM 0.04 against 0.89, for the
    same fold.
  * **`target_feat` is normalised over 449 columns where this port has 447.**
    The two missing ones are residue-vocabulary classes our alphabet does not
    carry; they are zero on every input built here, which a per-feature norm
    would not care about, and each still contributes `mean^2` to the variance.
  * **it runs on the HOST, deliberately.** It reduces the whole tensor to two
    scalars, so a GPU version is a full reduction, a readback and a second pass
    - three dispatches to save an O(n) loop that runs once per FOLD. The pair is
    590k floats on a 68-mer, about a millisecond, against a denoiser measured in
    seconds. It is the one thing the GPU path and the CPU reference share, and
    it is allowed because it is a STATISTIC and not a kernel.

🔴 **AND `weights.confidence.dialect` NEVER EXISTED.** The head is called
`run(input, weights, dialect)` and the fold passed `weights.confidence.dialect`,
which `confidenceWeights` has never set - so the argument has been `undefined`
for every model since the head was written, and every branch behind it ran ONLY
in `check-af3-confidence-oracle.js`, which passes a real one. That is
protenix2's and boltz2's `preSymmetrisedPde` as well as rf3's two: measured by
the checker, never reached by a fold. It passes `weights.trunk.dialect` now.

🔴 **AND "IDENTICAL TO FOUR FIGURES" WAS THE PRINTER, NOT THE WIRING.** The
CA-CA dgram appeared to change nothing - pLDDT 53.8 before and after - and that
was read here as a flag not reaching the kernel, on the strength of the
outer-product packing bug earlier the same night. It was live: 53.814360520
against 53.814360514, and pTM 0.17502894 against 0.17499594. The console rounds
to one decimal. **Ask for the digits before concluding a change is inert.**

**What is left, and PARITY.md names most of it.** Its "FIFTEEN PORT BUGS" table
is a list of the conventions an AF3-lineage atom path gets wrong, and two of its
rf3 rows are ours now:

  * **`slid the key window where it CLAMPS AND MASKS`** - rf3's atom attention
    clamps its key window and masks the out-of-range slots
    (`Cs = arange(nq)*32 + 16`, `patchk = arange(128) - 64` -> keys
    32i-48 .. 32i+79, then `clamp(indices, 0, L-1)` and `-1e9 * (maskQ|maskK)`),
    where this port slides the window bodily in bounds. Same convention as
    opendde and protenix; upstream missed it too, and its note says why - rf3
    was already in `KEY_MASKED_ATOM_ATTENTION` and that list is about the MASK,
    not about where the window SITS.
  * **the chirality query term** - the gradient of the chiral-centre dihedral
    error w.r.t. the noisy coordinates, added to the diffusion atom encoder's
    query. The reference HAS it (their own oracle-bug table records a gate that
    had it switched off); we do not. It is the only reflection-asymmetric signal
    in the network and a NO-OP on a batch with no chiral centres, which 6MRR is,
    so nothing measured here can see it yet.

Both of the confidence-head branches landed; see above.

🔴 **AND THE OPM DIVISOR IS ALREADY RIGHT, WHICH IS WORTH RECORDING BECAUSE THE
PARITY LINE READS THE OTHER WAY.** "OPM applies its output bias BEFORE the
divide" describes the BUG they fixed, not the convention: `OPM_BIAS_AFTER_NORM`
contains rosettafold3, the forward is `act / max(norm, 1) + output_b`, and that
is boltz2's branch, which is what `opmBiasAfterNorm: true` already gives us.
Their own note adds that rf3's source divides `right` by `float(N)` before the
einsum, which equals the pairwise count on an unmasked MSA and does not on a
masked one - so the reference is knowingly approximating there and so are we,
identically.

rf3 is also absent from the reference's `_SAMPLER_CONSTANTS`, which its own
comment calls "the honest state for one nobody has checked" - so it runs
AlphaFold 3's EDM schedule on both sides. That is a shared unknown rather than a
divergence, and their fold is a chain under it, so it is NOT the explanation for
ours being short.

The four that landed, in the order they were worth:

1. **`conformer_embedding_bias`, one [128] tensor, and it was 20x at
   `target_feat` on its own.** rf3's atom single rep takes
   `process_atom_level_embedding(f['atom_level_embedding'])`, whose input is all
   ZEROS here - but the MLP has biases and a LayerNorm tail, so it emits a fixed
   NONZERO vector, the same for every atom and two thirds the magnitude of the
   ref-feature embedding. **A zero feature is not a zero contribution.** The
   converter collapses that subtree to one constant, exactly as boltz2's Linear
   bias is one, so the two share `embedAtomFeaturesBias` and the forward needs
   no second branch - `constantAtomBias` in diffusion-weights.js reads either
   name and refuses a bundle carrying both.
2. **The MSA `is_paired` column, ZERO on the query row.** rf3's
   `msa_activations` is [35, 64] like boltz2's, and this port already built the
   35th column - with boltz2's meaning. boltz2 marks the QUERY row paired; rf3's
   `add_residue_is_paired_feature` marks rows PAIRED ACROSS CHAINS, which an
   unpaired alignment never has, so it is 0 everywhere INCLUDING the query.
   `msaPairedQueryRow` decides the VALUE; the column's existence still comes off
   the weight's width. Worth 1.24e-1 -> 2.82e-2 on `single`.
3. **The outer product mean's `left_projection`/`right_projection` biases**, worth
   `z_after_msa` 1.15e-1 -> 6.14e-2. Added BEFORE the mask, because the
   reference writes `mask * Linear(act)` - a masked row still contributes
   nothing, and an unmasked one gains two cross terms, since the outer product
   is bilinear and dropping a bias is therefore NOT a constant offset.
4. **The grid attention's `gating_query`/`output_projection` biases**, worth
   `trunk_out_pair` 1.29e-1 -> **4.90e-2** - the largest of the four. The gate's
   bias initialises to 1.0 against a ZERO-initialised weight, so the gate is
   bias-DOMINATED, and this runs 96 times in an rf3 trunk. It reaches the MATRIX
   projection too: the gate's rides in lane 3 through `laneBias` (the generic
   bias would add it to q, k and v as well) and the output projection's is the
   generic one. Vector and matrix agree at 4.91e-2 and 4.90e-2.

🔴 **AND THE OPM BIAS WAS BIT-IDENTICAL TO HAVING NO BIAS FOR AN HOUR.**
`packOuterProductMeanWeights` reserved the offsets over `ORDER + OPTIONAL` and
WROTE over `ORDER` alone, so the shader read a region of zeros: the term was
present in the source, present in the offsets, and absent from the buffer. Every
seam matched the previous run to the last digit, which reads exactly like "this
convention does not matter here". The grid pack had the same two loops and was
written with one list from the start because of it. **When a change moves
nothing at all, suspect the bytes before the convention.**

🔴 **AND `=== undefined` IS THE WRONG PRESENCE TEST FOR AN OPTIONAL WEIGHT.** The
loader sets an absent one to `null`, and that null reaches the SOURCES map, so
`residentGridOnDevice`'s strict check fell through and read `.count` off it -
killing boltz2, a model with none of these tensors, in a code path added for a
model that has them.

## 🔴 THREE MODELS WERE FOLDING A SINGLE SEQUENCE ONE MSA ROW SHORT

Counted in af3-any-model's own batches on 6MRR, by summing `msa_mask`:

| live MSA rows, no alignment | |
|---|---|
| **two** (the query twice) | alphafold3, openbind0, opendde |
| **one** | boltz2, protenix2, intellifold2, rosettafold3 |

This port gave all seven ONE. AlphaFold 3 concatenates a PAIRED and an UNPAIRED
block, so a chain with no homologs contributes its own sequence to each; boltz's
featuriser - which protenix, IntelliFold-2 and RoseTTAFold3 all fork or match -
emits a depth-1 `dummy_msa` and finds nothing to pair.

**It is not cosmetic.** The outer product mean over two identical rows is
unchanged, but the pair-weighted averaging and the row transition are
DEPTH-sensitive; upstream measured the same convention at 4.3% of esmfold2's MSA
injection. AlphaFold 3's 6MRR fold from the sequence goes **83.084 -> 83.169**.

The flag is `dedupeSelfMsa`, named the reference's way round so stock AlphaFold 3
stays the all-false baseline this table's own test asserts, and inverted once in
`af3BatchFromA3m` where it meets the featuriser's `duplicateQueryRow`.

🔴 **AND FINDING IT EXPOSED THAT `fold-opendde.js` PASSED NO DIALECT AT ALL.**
That tool handed `af3BatchFromA3m` nothing but `max-msa` and `seed`, so **every
OpenDDE number in these docs was featurised with AlphaFold 3's conventions** -
uncentred reference conformers, a sliding atom key window where OpenDDE clamps
and masks, and the query once where it wants it twice. `tools/gpu/fold.js` has
passed them since the batch was extracted and `fold.js` CANNOT fold OpenDDE, so
the two tools were never compared. With its own conventions OpenDDE's 6MRR goes
**1.527 -> 1.518 A** and its portable-gate pLDDT 92.1200 -> **92.0396**.

🔴 **THE SHAPE OF THIS IS THE NIGHT'S RECURRING ONE.** A convention applied
UNIFORMLY where the reference splits, plus a second tool that never got the
split at all. It is the same shape as the atom key window, the terminal OXT and
the confidence head's missing dialect - four in one night - and every one of
them was found by comparing against the reference's own batch rather than by any
fold looking wrong.

## 🔴 THE ATOM KEY WINDOW WAS WRONG IN FOUR OF SEVEN MODELS, AND NOTHING ASKED

`tools/check-atom-windows.js` compares this port's `queries_to_keys` against the
one in each `oracle-dumps/af3-batch-<model>-6mrr.json` - the reference's OWN
gather, integer for integer. It did not exist until 2026-09-14 and no other gate
here asks the question: `check-af3-denoise.js` reads the reference's windows out
of the dump precisely so it can compare score models rather than featurisers, so
the window itself was compared to nothing.

Three conventions were living in that gap, and there are three of them, not two:

| rule | who | the last window starts at |
|---|---|---|
| **SLIDE** | alphafold3 | `atoms - keys`, shifted bodily in bounds |
| **CLAMP and mask** | opendde, protenix2, boltz2, rosettafold3 | `32i + 16 - keys/2`, out-of-range slots masked |
| **SLIDE against a PADDED edge** | intellifold2 | `ceil(atoms / 32) * 32 - keys` |

and a fourth thing that is not a window rule at all:

| | who |
|---|---|
| **no terminal OXT and no 5' OP3** | openbind0, boltz2, intellifold2, rosettafold3 |

Before: 3 of 7 exact. After: **7 of 7 exact**, zero index differences anywhere.

**How each was found, because the order matters.** Setting CLAMP for the three
families the reference lists took opendde and protenix2 to exact and left four
models differing - and every difference was in subsets 16 and 17 alone, the last
two, which is the signature of an END effect. Printing the first differing slot
named the rest in one line each: boltz2 and rf3 read `ours idx 573 mask 1,
theirs idx 573 mask 0` (an atom we have and they do not), openbind0 `ours 446,
theirs 445` (a SLIDING window whose edge is the atom COUNT, off by that same
atom), and intellifold2 `ours 446, theirs 448` (off by two the other way, which
is `ceil(573/32) * 32 - 128`). One atom explained three of the four.

That atom is the C-terminal OXT. boltz's canonical table does not list one
(`const.ref_atoms["GLU"]` ends at OE2) and its CCD mol flags OXT
`leaving_atom: True`; IntelliFold-2 forks that table whole; rf3 calls atomworks'
`remove_protein_terminal_oxygen`; OpenFold3 and OpenBind-0 drop both it and OP3.
**protenix and opendde deliberately KEEP it**, so this is not an OpenFold-lineage
question and could not have been derived from one.

🔴 **AND THE NUCLEIC HALF IS THE WORSE HALF.** OP3 is the FIRST atom of residue
1, so carrying it shifts the ENTIRE flat atom axis of a nucleic chain by one,
where OXT only displaces a protein chain's tail. On 6MRR this was one atom in
574; on a nucleic chain it is every index.

🔴 **AND A FOLD CANNOT SETTLE ANY OF IT.** The clamp moved opendde's 6MRR by
0.026 A - inside a seed band this repository has measured at 1 A - so RMSD said
nothing and the gather said everything. From the sequence, after:

| `fold.js --sequence=<6MRR>` | atoms | pLDDT | CA-CA |
|---|---:|---:|---:|
| af3 (keeps OXT) | 574 | 83.1 | 3.73 |
| boltz2 | **573** | 96.5 | 3.80 |
| protenix2 (keeps OXT) | 574 | 84.7 | 3.74 |
| intellifold2 | **573** | 83.2 | 3.81 |

🔴 **AND THE FIRST FOUR RUNS OF THAT TABLE MEASURED NOTHING.** `fold.js
--target=6mrr` WITHOUT `--sequence=` folds `oracle-dumps/af3-6mrr.json` -
AlphaFold 3's own featurised batch - through whichever model `--model=` names.
Every one reported 574 atoms and 51 atom subsets after the change, which reads
exactly like "the flag is not wired". 51 is the reference's DENSE subset grid
where this port compacts to 18, and that number is the tell.

### 🔴 IntelliFold-2 DID NOT FOLD AT WebGPU's GUARANTEED MINIMUM, AND ONLY ADDING IT TO THE GATE FOUND THAT

`test:portable` and `test:spec-floor` covered six models. Adding the two new
ones took ten minutes and immediately failed:

    GPUPipelineError: The total use of workgroup storage (16960 bytes) is
    larger than the maximum allowed (16384 bytes).

**16,960 is `8 * 512 * 4 + 64 * 4 * 2 + 8 * 4 * 2` exactly** - an eight-row
staged LayerNorm at IntelliFold-2's 512 channels. At AlphaFold 3's 128 the same
tile is 4,672 bytes and fits anything, which is why a constant nobody priced
survived five models.

🔴 **AND IT IS IN TWO FILES, WHICH MADE THE FIRST HALF OF THE FIX LOOK LIKE NO
FIX AT ALL.** `grid-attention-webgpu.js` and `src/kernels/triangle/shaders.js` each
carry their own copy of that LayerNorm, and both come to **exactly** 16,960
bytes at 512 channels - so fixing one left the error byte-for-byte identical and
read as "the flag is not reaching the kernel". docs/ARCHITECTURE.md lists this
LayerNorm among the things written four times, and this is what that costs.

`tileThatFits` now takes the largest of [8, 4, 2, 1] whose storage fits
`maxComputeWorkgroupStorageSize`, and it lives in the LOWER module because af3
depends on triangle and not the reverse. It is the FIFTH instance of this exact
bug - `transitionWidth`, `splitTransitionConfig`, `projectMatrixConfig` and two
projection workgroups were the first four - and every one was a performance
choice that never asked what the device would run.

| | channels | tile | bytes |
|---|---:|---:|---:|
| AF3, at the floor | 128 | 8 | 4,672 |
| **if2, at the floor** | 512 | **4** | **8,736** |
| if2, on this A100 | 512 | 8 | 16,960 |

Nothing moves where the limit is not binding: if2 unrestricted is
83.22122296904186 before and after, to every digit, and AF3 83.08440884314348.
At the floor if2 is 83.2191685848003 - the tile changes, so the answer moves in
the fifth digit, which is what AF2's checksum already does there.

**All eight now pass all three gates.**

### 🔴 IntelliFold-2 IS THE HEAVIEST MODEL HERE, AND IT FOLDS ON 800 MiB

Its no-budget peak is **2229 MiB**, against boltz2's 1535 and AlphaFold 3's 983
- the trunk pair is 512 channels where AF3's is 128, so every resident trunk
weight is four times the size:

### 🔴 WHY IT IS LARGE: 4x THE PAIR WIDTH IS 16x THE PAIR WEIGHTS, AND NOTHING ELSE CHANGED

"The trunk pair is 512 channels where AF3's is 128" is the cause and it
understates the effect by a factor of four, because **a pair weight is
`C x kC`** - both of its dimensions carry the pair width - so the width enters
squared. Counted out of the two bundles' own manifests:

| | AlphaFold 3 | IntelliFold-2 | |
|---|---:|---:|---|
| parameters | 368.4 M | **851.0 M** | 2.31x |
| bundle | 265 MiB | **611 MiB** | 2.31x - int5 costs 6.02 bits/param in both |
| tensors | 406 | 406 | the same tensors, **191 of them the same size** |

and by module:

| | AF3 | IF2 | |
|---|---:|---:|---|
| diffusion head | 204.3 M | 207.7 M | **1.02x - untouched** |
| trunk pairformer | 147.4 | 549.4 | 3.73x |
| confidence head | 12.9 | 46.8 | 3.63x |
| MSA stack | 3.0 | 41.8 | **13.9x** |
| template stack | 0.3 | 4.0 | 13.3x |

🔴 **AND "int5" IS 6.02 BITS A PARAMETER, NOT 5 - A FIFTH MORE THAN THE NAME.**
The scheme is `asymmetric-per-group`, **5 bits, group 32, a float16 scale AND a
float16 zero per group**, so a quantised parameter costs `5 + 32/32` = **6.0
bits** and the two extra bytes per group of thirty-two are 20% of the bundle.
Both bundles also keep **151 of their 406 tensors in float32** - the layer norms
and biases, 0.7 M of IntelliFold-2's 851.0 M parameters, 0.08% - which rounds
the whole-checkpoint figure to 6.02.

That model predicts the artefacts exactly rather than approximately:

| | parameters | predicted | on disk | float32 would be |
|---|---:|---:|---:|---:|
| AlphaFold 3 | 368.4 M | **265 MiB** | 265 | 1405 MiB (5.31x) |
| IntelliFold-2 | 851.0 M | **611 MiB** | 611 | 3246 MiB (5.31x) |

🔴 **AND IT BUYS NOTHING ON THE DEVICE.** Measured on the A100, the int5 bundle
and the float32 bundle both peak at **954.1 MiB** on AlphaFold 3 - identical to
the tenth of a MiB - because what is resident is the DECODED tensor and a
decoded tensor has one width. int5 is 5.31x on the download and 1.86 s against
3.47 on the cold fold, and zero in memory. See docs/A100.md.

🔴 **THE 3.73x AND THE 13.9x ARE THE SAME NUMBER SEEN THROUGH DIFFERENT
MIXTURES.** Split the pairformer by which track a weight belongs to:

| trunk pairformer | AF3 | IF2 | |
|---|---:|---:|---|
| single track | 120.5 M | 120.9 M | **unchanged** |
| pair track | **26.9** | **428.6** | **15.9x** |

There it is: 4x the channels, **15.9x the weights**, and the single track beside
it identical. AlphaFold 3's pair track is only 18% of its own pairformer and 7%
of the whole checkpoint, which is why quadrupling it multiplies the MODEL by
2.31 rather than by 16 - and why the MSA stack, which is nearly all pair-width
weights, shows the raw 13.9x almost undiluted.

The individual tensors say the same thing. The extra 482.6 M parameters are
half accounted for by eight tensor classes, every one of them `C x kC`:

| | AF3 | IF2 | share of the delta |
|---|---:|---:|---:|
| `pair_transition/transition1/weights` | 6.3 M | **100.7 M** | 20% |
| `pair_transition/transition2/weights` | 3.1 | 50.3 | 10% |
| the four `triangle_multiplication_*/{gate,projection}` | 1.6 each | 25.2 each | 20% |
| `pair_attention1/{gating_query,k_projection}` | 0.8 each | 12.6 each | 4% |

`[48, 128, 1024] -> [48, 512, 4096]` is the first of those, and 16x exactly.
**The stack is not deeper** - 48 trunk blocks, 4 MSA blocks and 4 confidence
blocks in both - and **the diffusion head is not wider**, because it works on the
single and atom tracks at 384 channels in both. IntelliFold-2 is one wide track
in an otherwise ordinary AlphaFold 3.

Which is also the whole device-memory story: the two rows at the top of its 2229
MiB peak are `w.grid` 567 MiB and `w.pair-transition` 384, and both are the same
`C x kC` weights resident in decoded form.

🔴 **AND THE 2229 IS THE ONE FIGURE IN THIS DOCUMENT'S MEMORY TABLES THAT
REPRODUCES EXACTLY** - re-measured on the A100 on 2026-09-15 at **2229.1 MiB**,
where boltz2's 1535 comes back 1399.1 and AlphaFold 3's 983 comes back 992.6.
And the panel in docs/A100.md adds the half a single length cannot show: this
2.34x memory multiple is **flat in length** (2.31x at 255 tokens, because a
weight does not grow with the protein) while the TIME multiple is not - 1.87x at
68 tokens and **2.66x at 255**.


| held, no budget | |
|---|---:|
| `w.grid` (116 buffers) | 567.2 MiB |
| `w.pair-transition` (56) | 384.1 |
| `difftx.block.resident` (24) | 378.2 |
| `w.tri.out` / `w.tri.in` (58 each) | 195.6 each |
| `w.single-transition` (52) | 189.1 |

**A budget takes all of it and the fold does not move**: pLDDT 83.5, CA-CA 3.81
A, the same structure at `--budget=800` as at no budget, peaking at **139 MiB**.
What it costs is the second fold, which is the row a user sees:

| `fold.js --model=/model-intellifold2-int5/manifest.json` | trunk, first | trunk, warm | peak |
|---|---:|---:|---:|
| no budget | 2.4 s | **0.7 s** | 2229 MiB |
| `--budget=1200` | 11.4 | 7.2 | 804 |
| `--budget=800` | 11.4 | - | 139 |

So the residency trade is doing exactly what it is for, and the number to quote
for a laptop is 139 MiB and 11 s rather than 2229 MiB and 0.7 s. Nothing here
needed changing; this is the measurement, not a fix.

### 🔴 THE TRANSITION SPLIT IS 3.6x ON IntelliFold-2's TRUNK, AND NOBODY TUNED IT

`pairTransitionSplit` was fitted on AlphaFold 3 (worth 1.8%) and on ESMFold2
(1.51x), and it turns itself on above `TRANSITION_SPLIT_MIN_CHANNELS`, which is
192. IntelliFold-2's pair is 512, and at that width:

| if2 trunk, 150 tokens, 128 MSA rows | GPU total | the transition |
|---|---:|---:|
| split (the default at 512 channels) | **1072.5 ms** | 112.29 + 84.95 = 197.2 |
| `--tune=pairTransitionSplit=false` | 3854.2 | **2998.67** |

**15.2x on the kernel and 3.6x on the trunk.** The fused kernel holds the
WIDENED row in workgroup memory, so its row tile halves as the channels double
and at 512 it is holding almost nothing per workgroup - 11,042 groups a pass
doing very little each.

Nothing needed changing, and that is the finding: a THRESHOLD picked the right
answer for a model nobody swept. It is the counterexample to "reoptimize for
each model" in docs/ARCHITECTURE.md - a rule derived from a width transfers
where a fitted constant does not.

The rest of if2's trunk at that shape: `tri.project` 137.1 ms (3755 groups a
pass), `grid.project` 127.5 (5528), `grid.attend` 115.0 (3600), then
`pair-transition.wide` 112.3 (1877) and `pair-transition.down` **85.0 at 235
groups a pass**, which is the same starved dispatch CLAUDE.md records for AF3
and the same one docs record as retiled, measured and DECLINED - `wide` loses
more than `down` gains. At 235 groups it is 8% of the trunk and it is the price
of the 15.2x above, not a missed opportunity.

### Templates: IntelliFold-2 is the best of the seven, rf3 refuses

`fold-opendde.js --target=5caj --chain=A --template=tools/fixtures/5caj-crystal.pdb:A`,
which is the gate 6MRR cannot answer:

| 5CAJ, 255 residues, no MSA | without a template | with |
|---|---:|---:|
| af3 | 16.523 | 0.258 |
| openbind0 | 29.606 | 0.224 |
| boltz2 | 17.461 | 0.470 |
| protenix2 | 19.192 | 0.164 |
| opendde | 20.496 | 0.317 |
| **intellifold2** | **17.794** | **0.256** |
| **rosettafold3** | 17.858 | **17.041** |

🔴 **AND rf3's IS NOT THE SAME KIND OF THING, WHICH THAT ROW IS THE EVIDENCE
FOR.** Its 66 columns are a CA-CA distance HISTOGRAM, a coverage flag and a
noise level - distance-distribution conditioning rather than a geometry
embedding - and the reference's own docstring calls it "a flexible hint (define
the target without pinning exact coordinates)". Measured here, that is exactly
what it behaves like:

| rf3 with a self-template | without | with |
|---|---:|---:|
| 5CAJ, 255 residues | 17.858 A, TM 0.176 | 17.041, **TM 0.251** |
| 6MRR, 68 residues | 1.680 A, TM 0.911 | 1.684, TM 0.908 |

🔴 **AND THE PARAGRAPH THAT STOOD HERE WAS WRONG.** It read "this is the model's
behaviour rather than the port's", on the strength of rf3's own docstring
calling its template "a flexible hint", and flagged the reference comparison as
the thing to do before believing it. That comparison has now been run, and it
says the opposite:

| af3-any-model's OWN rosettafold3, 5K9P chain A, five samples | best | mean |
|---|---:|---:|
| no template | 1.603 A | 1.706 |
| **self-template** | **0.125 A** | **0.131** |

**Thirteen times, and it pins the structure exactly as the other five do.** So
rf3's template is not a weak hint, and the gap on our side is a defect
downstream of the features - which are themselves exact:
`distogram_condition` and `has_distogram_condition` both read relRMS
**0.000e+0** against af3-any-model's own tensors, element for element, and the
noise column matches the reference's own constant (the dump's `feat:noise_scale`
is a different, per-token quantity the module does not read).

**The lesson is the one this file keeps recording.** A vendor's own prose
described the module accurately and still supported the wrong conclusion about
OUR output, because "a flexible hint" is a statement about the ARCHITECTURE and
17.041 A was a statement about a fold. Nothing but running the reference on the
same target could separate them, and it took twenty minutes.

🔴 **AND THEN THE RETRACTION WAS ITSELF TOO STRONG.** "The gap is a defect
downstream of the features" was written before our rf3 had been run on the same
target, and on 5K9P it is not a defect at all:

| 5K9P chain A, self-template, one seed through `fold-opendde.js` | without | with |
|---|---:|---:|
| **our rosettafold3** | 10.164 A, TM 0.246 | **1.431 A, TM 0.901** |
| our alphafold3, same harness, same target | - | 1.520 A, TM 0.925 |
| af3-any-model's rosettafold3, best of five | 1.603 | 0.125 |

**Our rf3's template works - seven times on RMSD - and lands slightly AHEAD of
our own AlphaFold 3 on the same target through the same harness.** What is left
is a HARNESS gap that is not rf3's: we reach 1.4-1.5 A on 5K9P where the
reference reaches 0.125, for AF3 as much as for rf3, and the reference takes the
best of five samples where this tool folds one.

So the size dependence was real after all, and it is the one thing all three
measurements agree on:

| rf3 self-template | residues | without | with | ratio |
|---|---:|---:|---:|---:|
| 5K9P | 76 | 10.164 | 1.431 | **7.1x** |
| 6MRR | 68 | 1.680 | 1.684 | 1.0x (already folded) |
| 5CAJ | 255 | 17.858 | 17.041 | 1.05x |

rf3's bins stop at 20 A. At 76 residues they see the whole fold; at 255 almost
every pair lands in the last bin and the conditioning carries nothing. **What is
NOT measured is whether af3-any-model's rf3 has the same size dependence** - its
own template loader refuses 5CAJ, which has two polymer chains - so that is the
remaining open question, and it is a question about the MODEL rather than about
this port.

🔴 **THE SEQUENCE OF WRONG READINGS IS THE POINT.** First "a flexible hint, so
this is the model" (from the vendor's prose, and wrong). Then "a defect
downstream of the features" (from the reference's 13x, and also wrong). The
truth needed THREE measurements - their model on their target, our model on
their target, and our AF3 on their target as a control - and no two of them
would have found it. **A vendor's prose is not a measurement, and one
measurement against a different target is not a comparison.**

🔴 **AND ITS BINS STOP AT 20 A** where every other distogram in this port runs
to 50.75: `concat(arange(1, 4, 0.1), arange(4, 20.5, 0.5))`, 63 boundaries, 30 of
them at 0.1 A resolution inside 4 A. It is a CLOSE-range histogram, so on a
255-residue chain most pairs land in the last bin and carry nothing.

if2 needed nothing for this: it runs AlphaFold 3's nine-projection embedder, so
the whole template path transfers. rf3's `a_proj` is [66, 64] where protenix2's
is [108, 64] and boltz2's [109, 64] - its 66 columns are a 64-bin CA-CA distance
DISTRIBUTION plus has_condition and noise_level, not the distogram, restype
one-hots, unit vector and frame mask the other two concatenate. `a_proj`'s first
dimension is the ONLY thing that separates the three, because all of them ride
the same scopes. It is built now; see the template section.

### The template stack's width was a constant, and IntelliFold-2's is 256

`const CHANNELS = 64` sat in BOTH template-reference.js and template-webgpu.js.
It is 64 in five checkpoints and **256 in if2**, whose template grid attention is
8 heads of 32 against AF3's 4 of 16. `templateWeights` reads it now from the norm
after the stack - `output_layer_norm/scale` for the nine-projection embedder and
`v_norm/scale` for the fused one - which is the one tensor both forms carry that
states it, and both consumers require it rather than defaulting.

The failure was loud rather than silent, which is the only reason it was cheap:
`splitInterleaved` refused the triangle weights outright with "fused weight has
131072 elements; expected 8192".

🔴 **AND `check-af3-template.js` HAD THREE OF AlphaFold 3's CONSTANTS TYPED
INTO IT**, which is CLAUDE.md's standing note about hand-built weight dicts, one
file later. `QUERY_CHANNELS = 128` (OpenDDE's is 384) and `heads: 4,
dimension: 16` (OpenDDE's template stack is 2 x 32, if2's 8 x 32), plus a pinned
`{ swapTransposedBias: false }` where every other `--model=` checker derives the
dialect from `manifest.model.name`. **OpenDDE read NaN on this checker and
IntelliFold-2 1.65e-1, and neither was the port.** With all three off the bundle:

| `check-af3-template.js --model=` | 0 slots | 1 | 4 | 1 spanning | 4 spanning |
|---|---:|---:|---:|---:|---:|
| af3 | 2.77e-5 | 2.38e-5 | 2.75e-5 | 2.48e-5 | 2.98e-5 |
| opendde (was **NaN**) | 2.21e-5 | 5.92e-5 | 8.08e-5 | 9.40e-5 | 1.21e-4 |
| intellifold2 (was **1.65e-1**) | 5.72e-4 | | | | |
| intellifold2 `--matrix=off` | **3.80e-7** | 3.45e-7 | 3.16e-7 | 3.65e-7 | 3.37e-7 |

That last row is the finding: if2's template embedder is EXACT, and its 5.72e-4
is entirely the matrix pair kernels accumulating over a 256-channel stack rather
than a 64-channel one. The shipped trunk already pins `pairMatrixKernels: false`
on this stage, so nothing a user runs sees it; the checker's matrix bound now
scales by `sqrt(channels / 64)`, which is what a sum of independent roundings
does, and it is not a licence to raise it further.

### `triangleMulDivideByLength`, and why it is applied at the centre norm

rf3 computes `out = einsum("bikd,bjkd->bijd", left, right / float(L))` and then
the centre LayerNorm. It is observable ONLY because a LayerNorm's epsilon does
not commute with a scale - which is also why it cannot be folded into a weight,
and the reference says so. On the GPU the scale is applied at the CENTRE NORM's
INPUT rather than at the contraction's output: the two are the same number, that
pass already reads every element once, and the contraction is the most expensive
kernel in the track and does not need a multiply in its inner loop. Divided
rather than multiplied by a reciprocal, so it is the CPU reference's `total / n`
to the last bit.


Where the AF3 port stands, what it costs, and the things that have already been
got wrong once. Written to be read before touching any of it.

`AGENTS.md` holds the invariants; this holds the state.


## 🔴 THE MATRIX PAIR TRACK COSTS THREE ORDERS IN THE BLOCK, AND TWO CHECKERS PASS IT

`check-af3-confidence.js` fails all four heads - pLDDT 1902x, PAE 3463x, PDE
3718x, resolved 522x their envelope. It is not precision: `--f16=off` returns
byte-identical numbers, and the head already pins `stagedPrecision`,
`weightPrecision` and `accumulatePrecision` to f32.

**It is not any one kernel either.** `tools/gpu/probe-confidence-kernels.js`
runs the six updates of a block one at a time against the reference, on the
bundle's REAL weights, and every one is clean on both stacks - at an all-ones
mask and at the checker's 80% one, at input scale 1 and at the 177 the pair
actually reaches:

    triangle.outgoing 4.6e-7   grid.1 6.4e-7   pair-transition   4.8e-7
    triangle.incoming 4.5e-7   grid.2 6.6e-7   single-transition 7.3e-7

**The COMPOSED block is where it appears, and the trunk has it worse than the
confidence head.** One block, real weights, f32 accumulators, against the
reference's own `pairformerBlock`:

| stack | GPU vs reference | 1e-7 rounding envelope | ratio |
|---|---:|---:|---:|
| confidence | 4.55e-3 | 1.02e-6 | **4469x** |
| trunk | 2.03e-2 | 3.80e-6 | **5334x** |

🔴 **AND IT IS THE DEVICE TUNING, NOT THE PORT'S ARITHMETIC.** The same block
with `--no-prior` or `--default-tuning` reads **1.24e-5, 3.3x its envelope**.
The ampere prior is what turns on the matrix kernels; the capability layer
leaves `gridAttendMatrix` null. Removing them from the SHIPPED configuration,
one at a time and then together:

| trunk block, one block, real weights | BLOCK.pair | xEnvelope |
|---|---:|---:|
| shipped | 2.03e-2 | **5334x** |
| minus `triangleProjectMatrix` | 3.74e-3 | 985x |
| minus that and `gridProjectMatrix` | 2.82e-3 | 743x |
| ...and `gridAttendMatrix` | 3.21e-4 | 84x |
| the whole prior off | 1.24e-5 | 3.3x |

🔴 **NO ONE KNOB OWNS IT - THEY COMPOUND, AND THAT IS WHY BISECTING FROM THE
EMPTY SIDE LIED.** Restoring `gridAttendMatrix` alone onto an empty prior reads
737x, which named it the culprit; removing it from the full prior changes
5334x to 5334x. Both measurements are right and the first conclusion was wrong.
`triangleProjectMatrix` is the largest single term from the full side (5.4x),
and the last 84x to 3.3x survives every remaining knob tried individually -
`matrixLinear`, `stagedMatrixBlock`, `attentionMatrix`, `attentionMatrixTile`,
`attentionVectorScore`, `trianglePairProjectTile`, `transitionChunkBytes`,
`attentionGroup`, `linearTallTile`, `pairTransitionSplitMinChannels` - and
`--f16=off` does not move it either. **Subtract from the shipped configuration,
never add to an empty one**, and expect a residue that only the whole prior
explains.

🔴 **AND `--tune=` IS NOT A HARNESS FLAG, WHICH COST THE FIRST BISECTION.**
`gpu-chrome.mjs` handles `--tune-json=`, `--f16=`, `--occupancy`,
`--default-tuning` and `--no-prior`; `--tune=key=value` is parsed by the eleven
TOOLS that implement it (`fold.js`, `fold-af2.js`, `profile-af2-block.js` and
friends), not by the runner. A tool that does not implement it - such as this
probe when it was written - takes the flag, ignores it silently, and every arm
reads identical. That is indistinguishable from a knob that does nothing, and
it produced five such rows here before `--tune-json=` was used instead. **An
unrecognised flag is silently ignored by every tool in this repository**, so an
arm that changes nothing wants the flag checked before the knob is believed
inert.

🔴 **AND THERE ARE FOUR MATRIX KERNELS IN THE PAIR TRACK, NOT THREE.** Fifteen
single-knob arms all read 84.4x before the answer came from printing what
`compilePairTrack` actually RESOLVES under each configuration, which named it in
one run. The four, every one carrying `matrixElement: "f16"`:

    triangleProjectMatrix   gridProjectMatrix   gridAttendMatrix   pairTransitionSplit

`pairTransitionSplit` is the last 84x to 3.3x, and it is not a knob anyone would
look at for accuracy - its name and its documentation are both about SPEED (1.13x
on AF3's 128 channels, 3.71x on OpenDDE's 384). **Print the resolved
configuration before naming knobs.**

🔴 **FIXED FOR THE CONFIDENCE HEAD, AND DELIBERATELY NOT FOR THE TRUNK.**
`Af3PairformerStackGpu` takes `pairMatrixKernels: false`, and
`Af3ConfidenceHeadGpu` sets it beside the three precision axes it already
pinned - it had pinned three of four. On the SHIPPED tuning the head now reads
stack pair **2.97e-6** where it read 3.93e-3, and all four heads pass: pLDDT
201x, PAE **7.0x**, PDE **7.3x**, resolved 62.9x. The trunk keeps its matrix
kernels and its speed.

🔴 **AND THE TRUNK KEEPS THEM, BECAUSE THE APPROXIMATION IS THE POINT.** The
four are worth **14% of a trunk pass** (911/946 ms against 1046/1044) and 27% of
the pairformer (452/458 against 579/582), and running some of this port in f16
to go faster is a deliberate trade. What was wrong was not the kernels but the
BOUND, which priced only the three precision axes and so reported an accepted
trade as a defect on every run. `check-af3-trunk` takes `--matrix=off` now and
its pair bound follows the KERNEL, which is what check-evoformer-attention.js
already does for the same reason:

| arm | pair | contact | logits |
|---|---:|---:|---:|
| shipped: f16 axes + matrix | 1.12e-4 | 5.09e-3 | 7.21e-5 |
| `--matrix=off` | 1.85e-5 | 3.76e-4 | 1.18e-5 |
| `--matrix=off` + f32 axes | **6.66e-7** | 1.10e-4 | 4.90e-7 |

🔴 **AND THE LAST ROW IS THE ONE THAT MATTERS, BECAUSE IT COULD NOT BE REACHED
BEFORE.** Asking for `--staged=f32 --weights=f32 --accumulate=f32` used to
return **1.12e-4, the same number as the f16 default** - the request reached
none of the four kernels, so the f32 path was not being checked at all and had
not been for as long as the prior has set them. It reaches 6.66e-7, 5.1x the
rounding envelope, which is the evidence that the port's arithmetic is right and
the 1.12e-4 is approximation rather than error. Without an arm that genuinely
gets f32 there is no way to tell those two apart.

🔴 **AND THE TRUNK HAS THREE PAIR TRACKS, WHICH IS WHY THE FIRST PIN DID
NOTHING.** `pairMatrixKernels` wired into `pairformer-block-webgpu.js` alone
moved an f32 request from 1.12e-4 to 1.11e-4: the MSA stack and the template
embedder compile their own `compilePairTrack` and kept theirs. All three take
the option now, and both of the other two had to learn to read it from the
CONSTRUCTOR, because `Af3TrunkGpu` pins there while passing its run-time options
to only two of the three `run`s.

🔴 **BUT THE TRUNK'S OWN 5334x IS NOW AN OPEN QUESTION AND NOT A CLOSED ONE.**
Nothing here says 2.03e-2 a block is acceptable over 48 of them; it says the
confidence head could not afford it. `check-af3-trunk` holds 4e-5 and cannot
currently run - it wants `dialect.msaUpdateBeforeOuterProduct` named - so the
trunk's matrix path has no oracle check at all. A fold moves little
(meanPlddt 85.8300957 shipped, 85.8303909 with the head pinned, 85.8337307 with
the whole prior off), which is evidence about the MEAN and not about the pair
representation those kernels actually compute.

🔴 **AND THE KERNEL'S OWN CHECKER REPORTS IT AND PASSES.**
`check-grid-attend-matrix.js` prints `matrixVsReference: 1.34e-3` beside
`scalarVsReference: 1.51e-6` and returns `"ok": true`, because its bar is the
f16 one (~1e-3) the matrix units warrant. That is defensible for the kernel and
is not defensible for the CONFIDENCE head, which pins three precision axes to
f32 precisely because pLDDT and PAE amplify - and `gridAttendMatrix` is a
FOURTH axis it does not pin, so f16 matrix units run inside a head that
believes it is in f32.

🔴 **AND `check-af3-block.js` READS 3.93e-2 AT 68x ITS OWN ENVELOPE AND
PASSES.** Its bound is `envelope * 300` when the accumulators are f16, and its
envelope is 5.74e-4 because it builds its weight dict BY HAND: on random
weights a pairformer block is chaotic and one kernel's worth of rounding grows
560x over four blocks. On the bundle's real weights the same perturbation grows
~3x. So the checker with synthetic weights has an envelope three orders too
wide to see this, which is the whole reason a real-weights probe was needed.

**What it costs a fold is small, and that is the last piece rather than the
reassurance.** AF3 at int5, shipped against `--no-prior`: meanPlddt
**85.8300957** and **85.8337307**. The trunk's 48 blocks do not amplify it the
way the confidence head's tight envelopes do - but pLDDT and PAE are per residue
and per pair on the page, and those differ at 1e-3 while their mean does not.


## 🔴 THE PAGE RAN 48 OF BOLTZ2'S 64 TRUNK BLOCKS, AND IT LOOKED LIKE A DEAD MSA

Reported as "boltz2 is not using the MSA - results similar to single sequence
input". It was not the MSA. `web/af3-model.js` loaded the trunk as

```js
const trunk = await trunkWeights(store, 48, 4);
```

with both depths typed in. That is right for four of the five AF3-lineage
families and wrong for boltz2, whose trunk pairformer is **64 blocks**. A stack
is ONE stacked tensor, so asking for 48 of 64 reads the first 48 slices, runs
them, and returns a trunk that never finished. Nothing raises, nothing is the
wrong shape, and the structure that comes out is a plausible one.

On 6MRR's 59 residues with a 128-row search, through the page:

| | pLDDT |
|---|---:|
| boltz2, page, no MSA | 72.1 |
| boltz2, page, 128-row MSA | **72.4** |
| boltz2, CLI, same batch | **96.1** |
| boltz2, page, after the fix | **96.1** |

An alignment that moves the answer by 0.3 reads exactly like an alignment that
is not reaching the model. What it actually meant was that sixteen blocks of
trunk were missing, and the MSA's contribution - which enters at the FRONT of
the trunk and is refined all the way through it - is what a truncated stack
loses first.

🔴 **AND THE TOOL WRITTEN TO REPRODUCE THE PAGE COULD NOT REPRODUCE IT**, for
two reasons that had nothing to do with the bug and everything to do with why it
survived. `tools/gpu/fold.js` and `web/af3-model.js` each built the batch by
hand, and each dropped something the other passed:

- the CLI never passed `profileMsa`/`profileDeletionMatrix`, so it profiled the
  **127 cropped rows** where the page profiles all **8076**. That is a different
  feature, not a different sample of one, and it feeds `target_feat` as well as
  the profile - `profile#` and `tf#` both differ in the batch.
- the CLI never seeded the row subsample, so it took the alignment's **prefix**
  where the page takes a **seeded random subset** of the whole file.

So every `--a3m` gate in this repository was measuring a fold the site does not
run. Both call `af3BatchFromA3m` in src/af3/featurise/batch.js now, and with the two
batches made identical field by field the CLI reproduced 72.4 - which is how the
weights became the only thing left.

**The fix is three parts, and only the first one is the bug:**

1. `web/af3-model.js` passes no counts; `trunkWeights(store)` reads them.
2. `trunkWeights` RAISES on a count that disagrees with the bundle rather than
   silently handing back a prefix. A bench that means to walk a short stack
   passes `{ allowPrefix: true }` and says so at the call site.
3. `test/trunk-depth-from-bundle.test.js` gates both halves: boltz2 is still 64
   in the pinned manifest, the families do not all agree on one depth (a rule
   that stops discriminating passes by finding nothing), and no fold path
   anywhere under `src/`, `web/` or `tools/gpu/` passes a literal count. Verified
   to fail with the old line put back.

**Parity after it, page against CLI on one sequence and one alignment:**

| family | page | CLI |
|---|---:|---:|
| alphafold3 | 90.9 | 90.9 |
| openbind0 | 88.1 | 88.1 |
| boltz2 | 96.1 | 96.1 |
| protenix2 | 92.5 | 92.5 |
| opendde | 94.6 | 94.609 |

🔴 **AND openbind0 TOOK TWO TRIES TO ASK PROPERLY, BOTH TIMES BY COMPARING TWO
BUNDLES.** `fold-in-page.py` STRIPS every `remote:` line so the page reads
shards off disk, and this box has `model-openbind0-f32/` and no
`model-openbind0-int5/` - so the page died at "failed to load tensor
.../single_transition/transition2/weights: 404" and it read as a broken family
on the live site. It is not: the published `weights-02.int5.bin` at the pinned
sha hashes to exactly the digest committed in the manifest module. Then
`--remote-weights` gave the page 88.1 against the CLI's 89.6 - which is not a
parity gap either, because the CLI was on the f32 bundle. Pointed at the SAME
remote int5 manifest the CLI gives **88.1**. Two numbers that differ because
they name different weights say nothing about the code between them; both
comparisons had to be made same-bundle before either meant anything.

🔴 **AND OPENDDE HAD NO CLI FOLD THAT TOOK AN MSA AT ALL** until this - every
OpenDDE number in these docs is a single-sequence fold, so the page's alignment
path through that family had nothing to be checked against. `fold-opendde.js
--a3m=` closes it, through the same builder. `tools/gpu/fold.js` still cannot
fold OpenDDE (it wants the structural-token expander and AF3's confidence head),
which is why the two tools both exist.

**The lesson is the one this file keeps relearning in a new place: a constant
that is right for the model you developed against is a silent wrong answer for
the next one.** The dialect system exists so a second checkpoint's CONVENTIONS
come off its weights; its DEPTHS were still coming off AlphaFold 3's.

## What works

A protein chain typed into `index.html` folds with AlphaFold 3 entirely in the
browser: featurisation, trunk, diffusion and confidence, no server. Pick **AF3**
in the Model dropdown.

- **From a sequence, not a dump.** `src/af3/featurise/featurise.js` builds AF3's whole
  batch in JavaScript. Checked array-by-array against AF3's own batch for 6MRR
  and for a three-chain complex: `node tools/oracle/check_af3_featurise.js`.
- **Complexes**, chains separated by `:`. Chain identity comes from
  `src/input/chains.js` - the same `chainIdentity()` AlphaFold-multimer uses.
- **Two samplers.** *Diffusion* (**the default**) is AF3's own stochastic
  sampler, 25 steps on the page. *Flow* draws once at the top of the schedule
  and walks it down deterministically, 16 cycles. Both are seeded. Diffusion is
  the default because it wins or ties nearly everywhere measured - 1QYS 0.918
  against flow's 0.936-0.999, 6MRR 0.650 against 0.687, 1TIM A:B 0.958 against
  0.975 - and it is the sampler af3-any-model verified. Flow is a close second
  everywhere and costs the same (25 steps against 16 cycles). 🔴 THE REASON
  THIS ROW USED TO GIVE - "flow returns a fold that is NOT A CHAIN on 1QYS
  across four seeds" - IS RETRACTED; it does not reproduce, not even at the
  commit that recorded it. See the retraction at the end of this file.
- **DNA and RNA chains**, as their own entity types. A standard nucleotide is
  ONE TOKEN PER RESIDUE, so this needed no tokeniser change - only teaching the
  featuriser that a chain has a KIND, because `ACGT` is a valid protein as well
  as a valid DNA chain and nothing about the letters says which. Checked
  array-by-array against AF3 for protein+DNA, protein+RNA and a three-chain
  complex; folded geometry checked by `tools/gpu/probe-nucleic.js` (bond ratio
  1.013 DNA, 1.009 RNA, against 1.017 for the protein control). Their reference
  conformers are `src/af3/featurise/reference-conformers-nucleic.js`, generated from the
  oracle rather than typed. No MSA: AF3 searches an RNA database this page has
  no server for, and DNA gets none in AF3 either.
- **Modified residues** on protein chains; modified BASES are refused, since the
  modified-residue path resolves parents through the amino-acid table.
- **Recycles** for AF3 as well as AF2.
- **MSAs**, through the page's own alignment controls - search, paste or upload,
  shared with both AlphaFold 2 models. `src/af3/featurise/msa-features.js` is the whole of
  the adapter. On the 59-residue demo sequence a 512-row alignment moves pLDDT
  55.8 -> 65.7 and costs about 2 s (the MSA stack goes from nothing to 239 ms at
  512 rows).
- The trajectory animates as it computes, and the finished structure gets
  py2Dmol's PAE panel and prediction-quality card.

### Accuracy, against crystal structures

| | RMSD | TM | notes |
|---|---|---|---|
| 6MRR, flow 8 | 0.69-0.75 | 0.949 | four seeds, sigma0 160 |
| 6MRR, flow 8, sigma0 2560 | 0.65-0.77 | 0.950 | four seeds, AF3's schedule |
| 6MRR, diffusion 200 | 0.66 | 0.953 | |
| 1QYS (Top7), flow 8 | 0.89-0.92 | 0.944 | four seeds, sigma0 160 |
| 1QYS, flow 8, sigma0 2560 | 0.85-0.88 | 0.948 | four seeds, AF3's schedule |
| 1QYS, diffusion 200 | 0.93-1.12 | 0.92-0.94 | across four seeds |

🔴 **1QYS DOES NOT REPRODUCE ITS ROW AND HAS NOT FOR SOME TIME.** Measured
2026-09-02 at `--mode=flow --steps=8 --recycles=1`, seeds 1, 2 and 3: **1.24,
1.27 and 1.25 A**, TM 0.899-0.903, against the 0.89-0.92 above. Three recycles
gives 1.23 A and pLDDT 81.2 rather than 69.8, so it is not a recycle count. It
is not a regression from the kernel work either - the same input on the tree as
of `ea063a0`, before any of it, gives 1.24 A and pLDDT 69.8 to the digit. 6MRR
still measures inside its row (0.64 A, TM 0.960). The likeliest explanation is
that this table predates the side-chain fix, which changed what the denoiser
produces; it has not been re-run.

🔴 THE FLOW STARTS AT 160 A NOW, WHICH COSTS 1QYS 0.04 A. Most of AF3's
schedule sits above the level where the denoiser begins trusting the
coordinates it is handed, so a walk from 2560 spends its first calls on a
regime a flow does not need - and a ligand pays for it, HEM's bond error at
eight steps being 0.218 A from 2560 against 0.129 A from 160. On the proteins
6MRR is unchanged and 1QYS loses 0.04 A with non-overlapping seed ranges. That
trade was made deliberately; `schedule: {sigmaMax: 160}` restores AF3's own.
AF3's DIFFUSION sampler is untouched. See tools/gpu/probe-sigma0.js.

Flow matches or beats the 200-step sampler with ~25x fewer denoiser calls, on
both proteins measured. It is two proteins, both small designed alpha/beta
folds, both single-sequence - an observation, not a result.

🔴 AND BOTH ARE 68 AND 92 RESIDUES, WHICH IS WHY THEY KEPT PASSING. Every AF3
number in this file was a small single-sequence protein until 3RPF; the fold
that dropped a 512-row alignment on the floor scored the same on these two,
because they never had an alignment to drop. With an MSA, 3RPF's 146-residue
chain reaches 1.10 A and its complex 0.32 A - see the section on the AlphaFold
Server below. A regression suite of two proteins under 100 residues, both
folded single-sequence, is not one.

 ### Speed, 68 tokens

🔴 **THE FIRST FOLD OF A SESSION IS NOT THE FOLD, AND THIS FILE ONLY EVER
QUOTED THE FIRST.** Pipelines and the resident f16 weights are cached for the
life of the DEVICE, so the page pays for them once and every fold after is
cheaper. `tools/gpu/fold.js --folds=3` runs three in one process:

| | cold | warm | device |
|---|---|---|---|
| every f16 path off | 3.23 s | 2.18, 2.17 | 1463 MiB |
| as shipped | 2.48 s | 1.97, 1.90 | 855 MiB |

So a flow-8 fold is **1.9 s** once a session is going, and 2.5 the first time.
The f16 work is worth 23% cold and 11% warm - bigger cold because the f32 arm's
larger weight UPLOAD costs more than the f16 arm's one-off conversion.

It was 3.2-3.4 s and 1406 MiB before the f16 work of 2026-09-04, ~150 s when the first end-to-end fold ran, and 7 s before
the kernel work of 2026-09-02. A diffusion-200 fold was 25.9 s at the 3.0 s
era.

Where it goes: two trunk passes, then eight denoiser calls of which the FIRST is
~550 ms and the rest ~85. It is the largest single item left, and it is not
compilation, which measures 6 ms.

🔴 **AND IT IS NOT THE f16 CONVERSION EITHER, WHICH IS WHAT THIS FILE USED TO
SAY.** `tools/gpu/probe-pack.js` splits it over all 24 blocks of the int5
manifest, which pack to 378.2 MiB:

| cold, f16 (decode + convert + concatenate) | 494 ms |
| warm, f16 (convert + concatenate) | 244 |
| warm, f32 (concatenate alone) | 66 |

So the int5 DECODE is **250 ms**, the f16 conversion 178, and the concatenation
66. The decode is the store binding a block lazily - the first read of each of
its ~40 tensors decodes that tensor out of the shard - and it is paid whatever
precision the buffer ends up in. Dropping to f32 weights would save 178 ms of
494 and cost 378 MiB more on the device.

🔴 **AND IT IS A COMPUTE PASS NOW, NOT HOST WORK AT ALL.** Hiding it behind the
trunk was considered and would have cost 378 MiB; decoding it on the GPU costs
nothing and is 3.7x faster. `src/weights/quantised-upload.js` uploads the int5
CODES - about an eighth of the bytes - and decodes them straight into the
resident buffer. Over the 24 blocks, both arms cold: **437 ms on the host
against 119 on the device**, and 0 of 198 million elements differ.

The same helper (`src/af3/weights/device-weights.js`) took the trunk's two f16 labels,
which were the next largest:

| packer | MiB | host, cold |
|---|---|---|
| the transformer's 24 blocks | 378.2 | 440 ms |
| the trunk's single transition | 162.1 | 200 |
| its single attention | 67.6 | 77 |
| its pair transition (f32) | 36.0 | 16 |
| its outgoing triangle (f32) | 18.2 | 15 |
| its first grid attention (f32) | 15.1 | 19 |

What a real page fold does, through the dev panel's own timeline:

| | before | + the transformer | + the trunk |
|---|---|---|---|
| Trunk 1/2 | 673 ms | 647 | **355** |
| Folding | 920 | 311 | **257** |
| the whole fold | 3.31 s | 2.62 | **2.30** |

🔴 **THE f32 PACKERS ARE NOT WORTH CONVERTING, AND NEITHER WAS THE LAST f16
ONE, ON TIME ALONE.** Moving the single ATTENTION changed the fold by 2.31 s
against 2.30 - nothing - because once the single TRANSITION was off the main
thread the rest of the trunk's packing fits inside the pairformer's own GPU
time. It is kept for the 77 ms of main-thread work, which matters on a slower
CPU, not for the clock here. The f32 ones total about 50 ms, already hidden,
and would need an f32 variant of the kernel.

🔴 **AND THE RELEASE OF THE STAGING BUFFERS MUST NOT BE AWAITED.** The first
version awaited `onSubmittedWorkDone` per block, which put 48 host-device
synchronisations inside the pairformer's block loop - a loop written to run
ahead of the device on purpose. Trunk 1/2 went 647 ms to **778**. It rides an
unawaited promise now, which is the idiom that loop already uses for its
progress callback. Inside a trunk pass the order is now pair-transition
86 ms, grid.project 58, tri.project 48, grid.attend 47, tri.project-out 39, at
118 tokens over two passes.

🔴 **THE 12 ms THIS FILE ATTRIBUTED TO THE SAMPLER'S PER-STEP WORK WAS NOT
THERE.** It said a denoiser call cost 123 ms inside the sampler against 111 on
tools/gpu/bench-head.js, blamed the gap on the random augmentation, the noise
injection, the Euler step and the two trajectory copies, and called it 2.4 s of
a 200-step fold that nobody had looked at. Someone has now:
`tools/gpu/probe-sampler-overhead.js` times each phase inside the loop.

    59 tokens    step 112.8 ms   head.run 112.2   everything else 0.6
    150 tokens   step 250.5 ms   head.run 249.0   everything else 1.5

So the loop costs **0.6%**, not 10%, and the host arithmetic is 0.4 ms of it -
nearly all the noise injection's 4,248 gaussian draws. The augmentation, the
Euler step and both copies measure zero. They were always going to: at 59
tokens the whole of that work is about fifteen thousand float operations, which
is microseconds, and the count was checkable without running anything.

The 12 ms was two numbers from two processes - the drift this file warns about
three times over. `head.run` itself is within 0.2 ms of the GPU time it
reports, so there is no round trip to win back either. **Do not build a GPU
sampler step; there is nothing under it.**

## Modified residues, and why they need sixteen steps

Supported as of 2026-09-02, verified against AF3 array by array
(`check_af3_featurise.js` with a `--modification` dump) and structurally
(`tools/gpu/probe-modified.js`).

A modified residue is **one token per heavy atom**, inside the chain: SEP at
position 3 of a twelve-residue chain is 21 tokens, the ten belonging to it each
carrying one atom, all holding the PARENT residue's aatype (serine), all
sharing that residue's index, and all keeping the chain's asym, entity and sym.
Its own bonds go through the ligand-bond machinery; its peptide bonds to its
neighbours stay implicit in `residue_index`, as a standard chain's do. The
dictionary describes a FREE amino acid, so `polymerResidue` drops the OXT it
loses on forming a peptide bond and puts it back at a C-terminus.

🔴 **MSE IS NOT ONE OF THESE.** AF3 folds selenomethionine into methionine's
alphabet slot and leaves it one token with its own chemistry - SD becomes SE.
Every other modification tried is atom-tokenised. The page refuses MSE with
that reason rather than offering it and doing something else.

🔴 **AN ATOM-TOKENISED RESIDUE IS PLACED LESS PRECISELY THAN A STANDARD ONE,
AND THAT IS AF3'S BEHAVIOUR.** Its atoms are each their own token rather than
coming from a shared residue conformer, so the sampler places them
individually. Folding `ACSEFGHIKLWY` with SEP at 3, as the median
predicted-to-ideal bond ratio:

| | control | modified |
|---|---|---|
| AF3 itself, 32 diffusion steps | 1.003 | **0.956** |
| this port, 32 diffusion steps | 1.005 | **0.953** |

We match AF3 to three thousandths on both, so the gap between 0.95 and 1.00 is
the architecture's price and not a porting bug. What IS ours to get right is the
step count: at eight flow steps the residue comes out visibly compressed while
its neighbours are fine.

    flow-8    0.835   (control 1.003)
    flow-16   0.974   (control 1.007)
    flow-32   0.996   (control 1.010)

So sixteen is the lowest the dial offers, in both modes. It looked exactly like
the side-chain bug below - everything short, worse with distance from the
backbone, P-O3P at 0.483 - and the two are told apart by the fact that this one
improves with more steps and that one did not.

## Running it

    python3 -m http.server 8080          # then open /index.html

    # The GPU lane. Dawn (`npm run test:gpu`) cannot load on this macOS; Chrome
    # can. Every checker and bench is a module for this harness.
    node tools/gpu-chrome.mjs tools/gpu/<module>.js [--flags]

    node tools/gpu-chrome.mjs tools/gpu/fold.js --sequence=GWSTELEK... \
      --mode=flow --steps=8 --recycles=1 --model=/model-af3-int5/manifest.json
    python3 tools/score_fold.py <the log> --reference tools/fixtures/6mrr-crystal.pdb

    # The featuriser, including the MSA path, against AF3's own batch.
    python3 tools/oracle/dump_af3_trunk.py --blocks 0 --a3m rows.a3m --out d.json
    node tools/oracle/check_af3_featurise.js d.json rows.a3m

    node tools/gpu-chrome.mjs tools/gpu/bench-blocks.js   # AF2 vs AF3 per block
    node tools/gpu-chrome.mjs tools/gpu/bench-ab.js --skip=single

Checkers, all differential against the CPU reference:
`check-af3-{triangle,transition,grid-attention,single-attention,opm,msa-attention,embedder,template,block,msa-block,trunk,confidence,diffusion-*,atom-*,sampler-gpu,target-feat-gpu}.js`

## Traps

Each of these cost real time. They are in the code as `🔴` comments too.

**Benchmarks drift up to 3.2x between processes.** Two numbers from two
invocations of the same bench cannot be compared. `tools/gpu/bench-ab.js`
alternates A and B inside one process and reports medians; within a process the
spread is about +-10 ms. A whole round of per-pass profiling was thrown away
after this was ignored - the tell was "everything skipped" measuring *slower*
than "full".

**The parts do not sum to the whole.** Skipping one pass measures what it costs
on top of everything else pipelining, not its share. Individually the pair
track's passes account for 79 ms; together they cost 348.

**pLDDT is not the check.** It comes off the trunk and can look healthy over
coordinates that are not a molecule. A batch with one broken gather folded a
17 A spaghetti at pLDDT 55 with 15 A between consecutive CA. Backbone CA-CA is
what a wrong sampler cannot fake, which is why the fold prints it.

**A gather's `count` is not decoration.** `convert()` sizes its output from it,
so a gather without one silently yields a zero-length tensor and the model runs
anyway.

**`sigma` is a claim about the input, not a dial.** It reaches the network
through the Fourier noise embedding. Feeding a black hole at sigma 4 - "this
structure is nearly right" - diverges; at sigma 2560 - "this is noise, ignore
it" - the same input gives 1.39 A.

**The diffusion head has its own five reference embeddings**, distinct from the
conditioning module's. Same shapes, different weights. Reusing one for both
type-checks.

**Four conditioning weights exist twice in the checkpoint**, `..._1` and
unsuffixed, identical shapes. Dropping the suffix loads clean and gives the
wrong `target_feat`.

**AF3's MSA gap is 21, not 31.** The alphabet is 21 protein codes, then the
gap, then the nucleotides - the gap is in the MIDDLE of the 32-wide one-hot. A
gap at 31 type-checks, folds, and tells the model every gap is an unknown
nucleotide. Related and just as quiet: the deletion counts stay RAW, because
AF3's embedder does the `atan(n/3)` squashing itself, and AF2's featuriser does
it on the way in.

**AF3's unpaired chain merge is NOT block-diagonal.** `merge_msa_features` pads
each chain's alignment to the deepest and concatenates along the TOKEN axis, so
merged row r is chain A's row r beside chain B's row r, for every chain, with no
notion of entity - there is no `block_diag` anywhere in AF3. AlphaFold-Multimer
is the one that distinguishes: `_merge_homomers_dense_msa` merges copies of one
sequence densely and block-diagonalises only distinct entities, so it agrees
with AF3 on a homo-oligomer and differs on a heteromer.
`mergeUnpairedChainA3ms`, which block-diagonalises copies too, is neither: it
belongs to the AF2-MONOMER hack, where the +200 residue offset stands in for
chain awareness. AF3 has `mergeRowAlignedChainA3ms`. Two consequences, both
silent: the block-diagonal
form halves the information in every row and doubles the depth to carry it; and
for a HOMO-oligomer the row-aligned merge already IS the paired construction, so
supplying a paired block as well duplicates every row. That combination made a
homodimer fold worse with an MSA than without one, and a monomer shows neither.

**AF3's `msa` is two blocks and its `profile` is over one of them.** The array
the model reads is the paired block followed by the unpaired one; the profile
and deletion_mean are computed upstream, per chain, over the unpaired block
ALONE. So a 32-row A3M gives a 33-row `msa` - the query appears twice, because
an absent paired block becomes the query alone - and a profile over 32. Deriving
the profile from the array instead double-counts the query in every column, and
looks completely reasonable. `unpairedFrom` is threaded through featurise.js for
exactly this.

**AF3 resamples the reference conformer per residue instance** - fixed bond
lengths and angles, random torsions - so a baked table cannot reproduce a dump.
Measured cost of baking one: 0.01 A of structure. `check_af3_featurise.js`
therefore holds the chemistry (bonded pairs, from the bond graph) and lets the
torsions go.

**A ligand's bonds were read correctly and then dropped TWICE, on two paths
that could not see each other.** `ccd-component.js` parses the CCD bond table
and `featurise.js` turns it into the contact matrix AF3's `_embed_bonds` wants -
one direction per bond, `[0,0]` cleared, symmetrised only for the OF3 dialect.
After that:

- **The model never saw it.** `fold.js` assembles the trunk's input as an object
  literal and did not name `bondMatrix`, so the embedder got `undefined` - which
  is indistinguishable from a fold with no ligand. And `embedder-webgpu.js`, the
  one a browser fold runs, had neither the `bondEmbedding` weight nor the term,
  while `embedder-reference.js` had both. `diffuser/evoformer/bond_embedding/
  weights` was in the shipped bundle, downloaded on every fold, multiplied by
  nothing. `tools/gpu/check-af3-embedder.js` passed throughout because its
  fixture carried no bond matrix either: **a feature absent from both sides of a
  differential test is not tested by it.**
- **The viewer never saw it either.** `toPdb` wrote no CONECT records, so
  py2Dmol derived the ligand's bonds from the DISTANCE between atoms - and
  re-derived them from every trajectory frame, whose coordinates are noise until
  the last few diffusion steps. Measured on a six-atom cofactor whose truth is
  five bonds: **4 sticks at convergence, then 4/4/3, 3/4/3 and 2/3/1 as the
  noise grows** - a different molecule every frame. With CONECT it is 5 at every
  noise level, because the bonds stop being a function of the geometry.
- **And `ELEMENT_SYMBOL` had four entries** - C, N, O, S - with everything else
  falling through to carbon. Across a corpus of 51 distinct hetero components,
  **28 carry an element it dropped**: every phosphate-bearing ligand, every
  heme, every metal ion. That is a wrong colour and a wrong radius, and it
  breaks the distance fallback a second way, because that rule is per ELEMENT
  PAIR: a disulfide at 2.05 A read as C-C (ceiling 1.8) vanishes.

`test/af3-ligand-bonds.test.js` covers all of it on the CPU lane; seven
mutations, each caught.

🔴 **AND THE PAE WAS CROPPED TO THE POLYMER, ON A PREMISE THAT WAS WRONG.**
`paeSize` took the top-left `residues x residues` block of AF3's token matrix,
because a ligand is one token per heavy atom and "the matrix is wider than the
residues the viewer draws". The second half of that is not true: **py2Dmol
carries one POSITION per ligand heavy atom too**, and reads them in file order,
which is the order `toPdb` writes, which is token order. Measured across the two
repos on a 20-residue chain plus an 8-atom ligand: AF3 says **28 tokens**,
py2Dmol says **28 positions**, ligand starting at index **20 on both sides**, and
every cell of the matrix lands where its coordinates say - including the
protein-to-ligand corner, which is the whole reason to look at a mixed fold's
PAE. Reported as the PAE missing the ligand part.

Driven on py2Dmol's own page with that 28x28 matrix: the panel sizes itself from
what it is handed (`n = 28`), `pae_n` equal to that width makes its
cell-to-residue crossings the identity, 112k pixels of plot are drawn, and a
drag over the ligand block selects positions **20-27** - the ligand's own. A
ligand-only fold now falls out of the same rule instead of needing the special
case it used to have.

🔴 **AND THE FIRST TEST FOR IT DID NOT CATCH THE BUG.** It called `paeMatrix`
with the full stride and asserted the rows survived - proving the function keeps
what it is given, when the fault was in what the CALLER asked for. Restoring the
crop left it green. `paeSize` is a closure and cannot be called from a test, so
it is READ, the same way the trunk input's `bondMatrix` key is.

**py2Dmol read CONECT partners one column late**, which `trim()` hid up to 9,999
atoms - a right-justified four-digit serial survives a one-column slip, a
five-digit one does not. Serial 10000's partner came back as **1**: not a
dropped bond but a stick drawn to a real atom somewhere else. Fixed in py2Dmol's
`src/io/parse.js`, with the case in its `tests/interaction.js`.

**This build's py2Dmol renderer has no `setColor` or `setColorScheme`** - those
belong to the embed build. Drive the app's own colour `<select>` instead.
Writing `renderer.colors` directly is overwritten on the next recompute.

### The two traps nucleic acids set

🔴 **AN A3M COLUMN IS NOT A TOKEN INDEX.** `msa.set(row, ...)` copies an
alignment row in flat, which is right for exactly as long as one token means
one residue. A modified residue is ten tokens, so from the first one onward
every column of the alignment sat over the wrong residue - silently, with every
array still the right shape. It survived because the two features were tested
apart: the modified-residue work was checked without an alignment and every
alignment dump was a plain protein. ONE DUMP WITH BOTH FOUND IT. The batch now
carries a residue->column map and the msa, the deletion matrix and the profile
all read through it.

🔴 **THE PSEUDO-BETA IS C4 FOR A PURINE AND C2 FOR A PYRIMIDINE**, read out of
AF3's own gather rather than reasoned about. C1' is the plausible wrong answer -
the sugar carbon the base hangs off - and it disagreed. It has to be decided by
BASE rather than by name, because a pyrimidine has a C4 as well, so matching the
name alone takes the wrong atom in three components of five. And a nucleotide's
extra terminal atom is at the OTHER END from a protein's: OP3 at the 5-prime,
where a protein takes OXT at its last residue.

## Performance, and what has already been tried

The pairformer went 3468 ms -> 621 ms over 48 blocks at 59-68 tokens, and then
to **261 ms** with the f16 work of 2026-09-04.

🔴 **AND IT IS NO LONGER SLOWER THAN AF2'S BLOCK, WHICH THIS FILE SAID TWICE.**
"AF3's block is 1.09x AF2's evoformer block - for a block with no MSA row
attention, no column attention and no outer product mean in it" was a standing
complaint, and it is now the other way round: `tools/gpu/bench-blocks.js` at 59
tokens reports **5.44 ms a pairformer block against AF2's 8.93, or 0.61x.**

The reason is not that AF3 got better at the same thing. Both models took the
same class of change, but at 5 MSA rows an AF2 block is almost all PAIR track -
its triangle and its transition, none of which was touched - while the
pairformer is nothing but pair track and every one of its kernels was. At 512
MSA rows, where AF2's block is nine tenths MSA track, AF2 is the one that
moved: 109.25 -> 87.1 ms.

🔴 **AND THE TRUNK IS COMPUTE BOUND, WHICH THIS FILE USED TO SAY WAS THE NEXT
LEAD.** It said ~5 ms a block in the encoder, submit and validation path was
unexplained. It is not there. The labelled compute passes do sum to well under
the wall clock - it was 352 ms against 1201 when this was written - and three
separate measurements say that gap is the instrument and not the machine. The
absolutes below have all moved since (the trunk is much faster now); the SHAPE
is the argument and it has not:

- `Af3PairformerStackGpu` now returns its own `split`. A 16-block pass is
  encoded in **1.3-2 ms** and spends **82, 308 and 1345** inside
  `onSubmittedWorkDone` at 59, 118 and 236 tokens.
- Doubling the tokens quadruples the time, three times over: those same three
  shapes give **87, 318 and 1379 ms**. A pass paying a fixed cost per block does
  not scale like that.
- `tools/gpu/probe-dispatch.js` prices a dispatch before it computes anything:
  **3.5 us** in a shared compute pass, 4.9 in its own, 5.1 when it changes
  pipeline and builds a bind group, 29 when it gets its own encoder and
  submission, and **294 us** for a full round trip (submit, drain, map back).
  At ~500 dispatches a pass that is 2.5 ms.

Merging every dispatch of a block into ONE compute pass was tried on the
strength of the first number and measured 1278 ms against 1269 - nothing. The
gap is `tools/gpu/profile.js`: it adds 30% to the wall clock it is measured
against, and its timestamps are quantised to ~100 us across 1,521 short passes.

What worked, in order of size:

1. **Weight layout.** q, k and the gate were stored `(out, channels)`, so
   consecutive threads read 128 floats apart and nothing coalesced. Transposed
   at pack time into `v`'s `(channels, out)`. This was nearly all of the win.
2. **Row-blocked projection.** One pair row per workgroup meant one multiply per
   weight loaded - arithmetic intensity 1. Eight rows per workgroup, measured
   against 4 and 16.
3. **Submission window 16.** Each `onSubmittedWorkDone` is a full pipeline
   drain: 1 gives 881 ms, 8 gives 622, 16 gives 609, 48 gives 607.
4. **Flash attention** for the grid attention - online softmax, one thread per
   query, vec4 accumulators. Correct and the better kernel, but worth only 2.7%.
5. **`target_feat`'s atom encoder onto the GPU**: 5267 ms -> 160 ms, 33x. It
   reuses `Af3AtomEncoderGpu` by zeroing the three inputs this encoder does not
   have; they enter through bias-free linears of layer-normed values, so zeros
   contribute exactly zero. Checked by `check-af3-target-feat-gpu.js`.

7. **f16 accumulators in the two triangle projections**, 2026-09-04.
   `tri.project` and `tri.project-out` were 23% of the trunk's GPU time between
   them, and are 19% after this and hold their accumulators in WGSL ARRAYS - eight vec4 and eight vec2 -
   which is the shape a driver spills first. In f16: **1.688 -> 1.087 ms and
   1.250 -> 0.875** at 118 tokens, 1.55x and 1.43x, at the tile they already
   had. As the pairformer's wall time: **118 tokens 180 -> 162 ms, 236
   772 -> 702, 384 2200 -> 2034.** `accumulatePrecision`.

6. **f16 for the staged workgroup blocks**, 2026-09-04. Grid attention's key and
   value tile and the pair transition's two blocks are read once per output by
   every lane, and narrowing them halves the workgroup memory that bounds the
   occupancy. As the pairformer's wall time: **59 tokens 55 -> 53 ms, 118
   387 -> 361, 236 848 -> 777, 384 2461 -> 2226** - it grows with the problem.
   The arithmetic is untouched; only the staged copy narrows. `stagedPrecision`.

What did **not** work, measured, so it is not retried:

- **Uploading all 48 blocks' weights once** instead of per block: 30% *slower*
  (640 -> 830 ms), reproducibly. Recycling a few buffers beats holding 384, and
  the up-front burst serialises ahead of all compute.
- **Caching bind groups**: exactly zero, 636-639 either way, though ~1,680 are
  created per stack.
- **f16 weights for the triangle, for SPEED.** `src/kernels/triangle/` has had a
  precision option since before this port and `bench-triangle.js` reports 1.40x
  for it at L=128, which reads exactly like an unclaimed win. Wired through to
  the pairformer it measured **377 ms against 378**. The bench's 1.40x is its
  per-call weight UPLOAD shrinking; in the trunk the weights are resident and
  never uploaded, and halving their bytes does not halve the read INSTRUCTIONS -
  these kernels read weights one scalar at a time and this machine is
  instruction-bound. It is still worth doing for MEMORY; see below.
- **The same f16 staging in the MSA STACK, which is the same kernels one stage
  earlier.** Its pair track and transition give 580 -> 516 ms at 236 tokens -
  11%, a bigger relative win than the pairformer's - and the trunk's contact
  probabilities go **1.86e-4 to 5.21e-3**, 28x, with the pair at 6.64e-5
  against 4e-5. Staging only its pair track and leaving the MSA transition
  alone still costs 3.72e-3 for 44 ms.

  🔴 **THE DIFFERENCE IS POSITION, NOT ARITHMETIC.** The MSA stack writes the
  pair representation that all 48 pairformer blocks then read, so its rounding
  is amplified by everything downstream; the pairformer's own is not. That is
  the rule for the next one of these: the same trade is worth taking near the
  output and not near the input.
- **Preparing the diffusion head's weights during the trunk.** A pairformer
  pass encodes in ~5 ms and waits ~340 for the device, and none of the
  transformer's 24 blocks depends on the trunk - so packing them into that gap
  should be free, and the first denoiser call would stop being several times a
  steady one. Measured, the work MOVED and the fold did not: **trunk 1.1 -> 1.4
  s, diffusion 1.4 -> 1.0, total 2.5 either way.** The host time is idle but
  the BUS is not - filling 378 MiB of resident buffers competes with the
  trunk's own traffic for exactly what it saves. It is the same finding as the
  up-front weight burst above, in a new place.
- **Anything that adds registers to the flash attention kernel.** See
  src/kernels/attention.js: a vec4 q.k accumulator is worth exactly zero
  (the compiler already does it), grouping the keys to amortise the softmax
  rescale is 2.3x SLOWER, and two queries a lane is 4.7x slower. The query and
  the accumulators are already 64 registers a lane and that is the ceiling.
- **Binding AF2's attention kernel directly** rather than rewriting on its
  principles: it takes a uniform for its shape, folds `1/sqrt(d)` into the query
  projection and applies the gate itself. The adapter was wrong at relRMS
  2.96e-1 and cost more to find than the rewrite took.

~~Where the remaining time goes, at 59 tokens: 348 ms of dispatch work and
284 ms of per-block overhead that is not uploads (40 ms) and not bind groups
(0 ms). About 5 ms a block in the encoder, submit and validation path is
unexplained. That is the next lead and it is a small one.~~

🔴 **STRUCK OUT 2026-09-04: THERE IS NO SUCH OVERHEAD.** The paragraph above
survived because the numbers it quotes are real - the labelled compute passes
genuinely do sum to less than the wall clock - but the gap is the profiler and
not the machine. See the compute-bound note in the section above, which prices
a dispatch, splits the encode from the wait, and shows the pass scaling as a
clean square. It was a lead for a long time and it was never there.

### The f16 budget, spent and accounted for

Half precision is used in four places now, and it is worth seeing what they cost
together rather than one at a time. The trunk against the all-f32 tree, 48
blocks and real weights:

| | pair | single | contact | logits |
|---|---|---|---|---|
| all f32 | 6.18e-7 | 6.21e-7 | 9.76e-5 | 4.46e-7 |
| + staged tiles | 1.04e-5 | 2.91e-6 | 1.86e-4 | 6.83e-6 |
| + resident weights | 1.04e-5 | 8.06e-5 | 1.86e-4 | - |
| + triangle accumulators | 1.99e-5 | 8.06e-5 | 1.33e-3 | 1.26e-5 |

So the contacts - the most sensitive thing the trunk emits - are 13.6x their
f32 value. What that bought, measured by forcing every f16 path off with
`--staged=f32 --weights=f32 --accumulate=f32`:

| | all f32 | shipped | |
|---|---|---|---|
| pairformer, 236 tokens | 854 ms | 727 | 15% |
| whole trunk pass, 236 tokens | 1677 ms | 1558 | 7% |
| a fold, cold / warm | 3.23 / 2.18 s | 2.48 / 1.93 | 23% / 11% |
| a fold's device memory | 1406 MiB | 798 | 43% |

🔴 **THE FOUR ROWS ARE NOT THE SAME NUMBER SEEN FOUR WAYS, and quoting the
biggest one is the mistake.** A trunk pass is 7% because the MSA stack and the
template are most of what is left in it and neither was touched; a fold is 23%
cold because the diffusion head's weights are most of its memory traffic and
the f32 arm uploads 608 MiB more of them. The pairformer's 15% is the one that
matches the kernel work.

🔴 **WHAT DECIDES IS WHETHER A NUMBER A USER SEES MOVES, AND NONE DOES.** On
6MRR and 1QYS at flow-8: CA RMSD 0.032 A and 0.005 against the f32 tree, where
AF3's own accuracy on these is 0.7-0.9 A and the sampler's seed-to-seed spread
is ~0.1. Mean pLDDT within 0.01 and worst per-residue 0.04 on 6MRR and 0.22 on
1QYS, on a number the page shows to one decimal. pTM within 2e-4, shown to two. A contact probability
moves by ~1e-3 in [0, 1].

🔴 **AND RMSD IS NOT MONOTONE IN THE ERROR, so do not read one structure as a
verdict.** Adding the triangle accumulators took 6MRR from 0.0086 to 0.0322 A
and 1QYS from 0.0119 to 0.0053 - the sampler is chaotic and both are noise
around a small perturbation. The deterministic trunk numbers above are the
signal; the folds say the scale.

## Memory, which is a separate question from speed

🔴 **THE TOTALS COULD NOT SAY WHICH TENSOR TO ATTACK, AND NOW THEY CAN.** The
allocator was already given a label for every buffer and threw it away.
`memorySnapshot(device)` returns `byLabel` as well as the totals, and
`bench-trunk.js`, `fold.js` and `fold-af2.js` all print it. The answer for AF3
was three rows out of a 1406 MiB fold:

    difftx.block.resident   756.4 MiB   24 diffusion transformer blocks
    w.single-transition     324.1       48 pairformer blocks, 384 channels x4
    w.single                135.2       48 single-track attentions

1216 of 1406 MiB, all of it weights. In f16 they are 608, and **a fold now
holds 798 MiB against 1406; the trunk alone 337 against 567.**

🔴 **IT BUYS NO TIME AND IS NOT SUPPOSED TO - IT COSTS ABOUT 2%.** Halving the
bytes does not halve the read instructions, and the `f32()` at each read is not
free. Two separate measurements, both on the pairformer:

- the PAIR track's weights (triangle and pair transition): 377 ms against 378,
  which is nothing;
- the SINGLE track's, which is where the memory is: 163, 163, 166 ms in f32
  against 166, 167, 168 in f16 across three interleaved pairs.

What it buys is a device small enough to hold the model at all - a 4 GiB
phone's whole budget is 1.3 GiB, which a fold was exceeding on its own.

🔴 **AND THE DIFFUSION TRANSFORMER IS THE EXCEPTION, so do not generalise the
paragraph above to it.** That stack streams its whole weight set once per token
tile instead of keeping it in cache, so there the bytes ARE the cost: 48 -> 41
ms at 59 tokens and 103 -> 89 at 150. See its shader factory.

Two folds say what THIS CHANGE costs, against the same seeds on the all-f32
tree at flow-8 with one recycle: **6MRR 0.0077 A CA RMSD, 1QYS 0.0093 A**, worst
per-residue pLDDT 0.02 and 0.24, bond geometry identical to five decimals. The
triangle accumulators landed after it and moved those to 0.032 and 0.005; the
budget table above is the combined figure and this one is the weights alone.

🔴 **TWO PLACES KEEP f32 ON PURPOSE, AND BOTH ARE ABOUT THE RATIO.**

- **The confidence head's four pairformer blocks.** pLDDT and PAE are what the
  page shows and they are a softmax over 50 and 64 bins - the most amplifying
  thing either model emits. In f16, pLDDT's relRMS goes 1.16e-4 to **2.32e-2**
  and PAE's 5.75e-6 to 2.51e-3. Four blocks of 52 is ~14 MiB, so f32 here costs
  almost nothing and keeps both numbers checked at the tolerance their own
  arithmetic reaches.
- **The pair track's own weights**, offered as `pairWeightPrecision` and off by
  default. They save 38 MiB - 6% of the 608 - and cost the worst amplification
  measured anywhere here: `check-af3-block`'s pair goes from 17x its rounding
  envelope to 51x. A caller that would otherwise not fold can still ask, which
  is what `--budget` already exists for.

🔴 **AND `createTriangleShaders` CONFLATED TWO FORMATS.** Its `precision` named
the weights AND the activations: at "f16" the normalize shader declared
`source` - the pair representation itself - as an f16 array, which is right for
the standalone runner (it converts `z` on the way in) and wrong for anything
sharing that buffer with a track. Wired into AF3 it read every f32 pair value
as two halves of one float and the trunk produced NaN.
`shape.weightPrecision` now narrows the weight buffer alone.

🔴 **EVERY CHECKER THE CHANGE REACHES GREW A PRECISION AXIS RATHER THAN A RAISED
BOUND**, and both arms run: `check-af3-trunk` (two axes, four combinations),
`check-af3-block`, `check-af3-diffusion-head`,
`check-af3-diffusion-transformer`, `check-af3-grid-attention`,
`check-af3-confidence`. Raising one bound would have stopped the f32 path being
checked at all - which is the whole reason the f32 arms still measure what they
did before. The bounds are derived from the arithmetic and the table of
measurements is in each file.

## Open

- **ipTM**, which is what a complex is actually judged by, is still not
  implemented for AF3 - the confidence head emits PAE and PDE, and pTM/ipTM are
  absent rather than approximated.
- **The A3M parser is narrower than AF3's alphabet.** `src/input/a3m.js` rejects
  B, Z, J, O and U, which AF3 maps to D, E, X, X and C. The codes are in
  `AF3_MSA_CODES` and unreachable through that parser - for AlphaFold 2 too, so
  widening it is a change to all three models rather than to AF3's path.
- **No ipTM**, which is what a complex is actually judged by. The confidence
  head emits PAE and PDE; pTM and ipTM are not implemented.
- **Per-atom conditioning is still on the CPU**, but it is no longer a few
  hundred milliseconds and most of it never needed a kernel. Two of its five
  embeddings multiplied by a materialised ONE-HOT - the element, 128 columns
  indexed by atomic number, and the atom name, four 64-way one-hots flattened
  to 256 - which is 384 of its 389 input columns, so 99% of its arithmetic was
  multiplying by zero. As gathers: **72.5 -> 3.7 ms at 59 tokens, 219.8 -> 9.2
  at 150, 343 -> 14.6 at 300**, bitwise identical, and
  test/af3-atom-conditioning.test.js holds it to the matmul form's exact
  floats. Both the summation ORDER and the float64 accumulation are load-
  bearing there; getting either wrong moved 1e-7 through a reference the GPU
  checkers compare against at 1e-6, which is a tolerance nobody chose. What is
  left is ~4 ms of genuine dense work and is no longer worth a kernel.
- ~~**Templates raise** rather than compute.~~ Closed on the CPU 2026-09-04:
  `src/af3/featurise/template-features.js` computes all six geometry features and
  `template-reference.js` loops over real slots. Against AF3 on a 16-residue
  query with Top7 in slot 0 of four:

  | | relRMS |
  |---|---|
  | slot 0, real | 5.03e-7 |
  | slots 1-3, empty | 2.35e-7 |
  | the module's 128-channel output | 1.74e-7 |

  The GPU path computes them too, checked against the CPU reference at 32
  tokens with 0, 1 and 4 of 4 slots occupied: relRMS 2.5e-7, 2.2e-7, 1.6e-7.

  🔴 **AND A REUSED BUFFER LOOKED EXACTLY LIKE A WRONG KERNEL.** The first
  version wrote each slot's aatype and geometry into one buffer inside the slot
  loop. `queue.writeBuffer` is ordered against SUBMITS, not against the
  recording of a command encoder - so all four writes landed before the single
  submit and every slot ran against the LAST slot's data. With one occupied
  slot of four the module computed the all-empty answer, which differs by only
  that slot's quarter share: relRMS 2.1e-2, small enough to read as precision
  and wrong enough to lose the template entirely. One buffer per slot.

  🔴 **AND IDENTICAL SLOTS RUN ONCE BETWEEN THEM.** Four empty slots produce
  the same embedding by construction, so a naive per-slot loop is 4x the work
  for an identical answer on every de novo fold - the trunk's template stage
  went 83 ms -> 150 ms before this was put back. Each pass carries how many
  slots it stands for and the accumulate shader multiplies by it, so a fold
  with no templates runs one pass, which is what it always did.

  🔴 **AND THE CROSS-CHAIN MASK WAS DEFAULTING PERMISSIVELY.** AF3 masks the
  geometry features ACROSS chains, because its `Template` is one protein chain
  and a complex's chains are templated by separate searches - so a cross-chain
  distance is computed from two structures that were never in one frame. The
  embedders defaulted to "every pair is intra-chain", which is right for a
  one-chain query and is what every check had. Measured on a two-chain query
  with a template on EACH chain:

  | mask | slot 0 |
  |---|---|
  | per chain, as AF3 | 5.5e-7 |
  | all-ones | **1.09** |

  So it is most of the module's answer, not a correction. The mask is derived
  per slot from `asymId` now and there is no permissive default: a template
  with neither `asymId` nor an explicit mask raises.

  ### Inter-chain templates, which AF3 does not do

  The masking is about PROVENANCE, not modelling. When two chains come from
  ONE file - a real co-crystal, or a complex this page predicted - they ARE in
  one frame, and the cross-chain distances are exactly the interface geometry a
  binder method wants. So `spanChains` is a flag on the SLOT, not a setting on
  the model: one slot built from one structure may span while another built
  from a separate search may not, in the same fold. Spanning opens only the
  pairs the slot covers at BOTH ends, because a pair with one end outside the
  template is still two frames apart.

  Measured at 32 tokens over two chains, against the same slots masked per
  chain: spanning moves the module's output by relRMS 1.5e-2 with one occupied
  slot and 4.9e-2 with four. There is no oracle for this - AF3 does not do it -
  so it is checked by construction: the GPU and CPU paths agree to 2e-7 on
  every arm, and an arm that failed to move would fail the check.

  The page is unchanged: nothing yet builds `slots` from a user's structure.

  🔴 **ONE THING WAS WRONG AND ONLY THE ORACLE COULD HAVE SAID SO.** The unit
  vector is `R_i^-1 (t_j - t_i)` - the FRAME is the row index and the POINT is
  the column - because AF3 writes
  `rigid[:, None].inverse().apply_to_point(points)` and the broadcast puts
  frames on axis 0. Written the other way it is the exact transpose: still unit
  length, still smooth, still masked correctly. Measured both ways against
  AF3's own `make_backbone_rigid`:

  | | unit vector | distogram | both masks |
  |---|---|---|---|
  | frame on the row | **1.29e-7** | 0 | 0 |
  | frame on the column | 1.51e+0 | 0 | 0 |

  Everything else was bit-exact either way, so nothing but a real template
  could have found it - which is precisely the argument the module's own
  header used for not writing these features at all.

  The rest of the entry, kept because it is what the checkers read: It was "the geometry features are unverifiable without a
  reference", which was true: with no template all six are identically zero, so
  nothing here could tell a correct implementation from a wrong one.
  `tools/oracle/dump_af3_trunk.py --template <pdb>` now produces one. Measured
  on a 16-residue query with Top7 as its template, four slots:

  | slot | template | module output |
  |---|---|---|
  | 0 | real, 374 atoms | mean -0.0781 std 2.0651 |
  | 1, 2, 3 | empty | mean +0.2583 std 0.7561, all three IDENTICAL |

  which is the module's documented behaviour seen from outside: an empty slot
  is a learned transform of the QUERY and three of them produce the same
  answer. The dump carries `template_aatype [4, 16]`,
  `template_atom_mask [4, 16, 24]` and `template_atom_positions [4, 16, 24, 3]`
  as inputs, and the per-slot 64-channel output beside the module's 128-channel
  contribution - so the six features can be checked in isolation from the two
  pairformer blocks that follow them.

  The two constant tables they need are small and, for protein, trivial:
  `RESTYPE_PSEUDOBETA_INDEX` is dense slot 4 (CB) for every amino acid except
  glycine, which takes slot 1 (CA); the backbone frame is
  `RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX[:, 0]` = (2, 1, 0) = (C, CA, N) for all
  twenty. Nucleotides differ and AF3's own `Template` is documented as one
  protein chain.
- ~~**AF3's block is still 1.09x AF2's** for strictly less work.~~ Closed
  2026-09-04: it is 0.61x. See the performance section.
- **No RNA alignment.** AF3's pipeline searches an RNA database; single-sequence
  is what an RNA chain gets here, and the status line says so. The seam is
  ready for one - featuriseProtein reads the MSA by ALIGNMENT COLUMN now, and
  a nucleic chain's columns are simply absent from a protein A3M.
- **No modified bases**, and no nucleic ligand bonds: a DNA chain and a ligand
  in the same job are two separate molecules to the featuriser.

## Pairing, as the server actually returns it

`generateMmseqs2PairedMsa` posts every distinct sequence of a complex to
`ticket/pair` as `>101, >102, ...` with `mode=pairgreedy`, and `pair.a3m` comes
back holding one NUL-separated block per query. Confirmed live on 3RPF's two
chains (146 and 74 residues): 9,904 rows for each, equal depth, each block
carrying its own query and its own width, and no all-gap padding rows.

The pairing is real and not merely aligned. Row 2 of both chains is
`UniRef100_UPI00129C3066`, one 214-residue protein matching chain A over 83-209
and chain B over 1-75 - a single partner supplying both halves, which is the
signal the paired block exists to carry.

It took 88 s for that pair, against about the same for the unpaired searches
that run alongside it. A homomer skips this entirely.

End to end through the page, AlphaFold-Multimer on those two chains at three
recycles: **pLDDT 96.5, pTM 0.907, ipTM 0.897** in 684 s. ipTM is the number
that matters and the one pairing is for - a confident INTERFACE needs
cross-chain coevolution, which is precisely what the paired block carries and
what a heteromer folded without until now.

AF3 takes the same alignment and the budget split is AF3's own: of 9,904 paired
and 7,283 unpaired rows, it reads 255 paired + 256 unpaired + the query = 512,
with `unpairedFrom` at 256. That is `max_paired_sequences = msa_size // 2` with
the remainder to the unpaired block, which is what featurise.js needs to compute
the profile over the right half.

🔴 THE FOLD PASSED ONE MSA ROW TO THE TRUNK, and that was the whole of it.
`src/af3/fold.js` called the trunk with `sequences: 1` and sliced every MSA
array down to the query, so the MSA stack never saw an alignment. Fixed; the
numbers below are after.

It is worth knowing how it hid, because the next bug of this kind will hide the
same way. An alignment still reached the model: `profile` and `deletion_mean`
are computed over all of it and ride into `target_feat`, so supplying one DID
improve the fold (44.5 -> 62.6 pLDDT on chain A) and the status line honestly
reported the depth that had been FEATURISED. Nothing reported the depth the
trunk was handed. `foldBatch` now emits it and `tools/gpu/fold.js` prints a
marker when the two disagree.

It also explains evidence that fitted none of the theories being tested: more
recycles made the structure worse while raising pLDDT, the sampler and the
precision barely mattered, and every component measured exact against AF3 while
the assembled fold was poor. The model was right the whole way down.

### Against the AlphaFold Server, on 3RPF

Its own run of the same two chains is the reference (`useStructureTemplate:
false`, ptm 0.91 / iptm 0.91 across five seeds), and its per-chain MSAs are in
the zip, which is what makes this comparison clean - no MMseqs2 in the loop.

| | before | after |
|---|---|---|
| chain A, 146 res | 9.96 A, TM 0.409, pLDDT 58.1 | **1.10 A, TM 0.962, pLDDT 87.9** |
| both chains, 220 res | 17.21 A, TM 0.196 | **0.32 A, TM 0.997, pLDDT 93.9** |

217 of 220 CA within one angstrom. 6MRR is unchanged at 0.76 A, because a
single-sequence fold always had one row.

Through the page, with ColabFold's own MSAs rather than the server's: the same
complex reaches **pLDDT 94.4**, against 62.6 before.

🔴 AND IT NOW COSTS WHAT AN MSA COSTS: 227 s against 84 s, because the MSA stack
has 512 rows to work on instead of one. The old number was cheap because it was
not doing the work. `--max-msa` trades this back if a fold has to be quick.

### Templates are not the difference

The server scores the same with them off, so do not implement templates to
chase a complex that folds badly. That was the prime suspect for an hour and it
was wrong.

### Fixed: a fold crawled in a background tab

The per-block yield was `setTimeout(resolve, 0)`, and Chrome clamps setTimeout
to >=1 s in a hidden tab - so a 48-block pass that takes under a second took
the better part of a minute and the trunk appeared to hang. Measured at the
time: pass 1 reached block 11 in five minutes hidden, then jumped to block 28
the moment the tab was touched.

`src/runtime/yield.js` is the fix (commit 882e3f2) and the whole of it: a
MessageChannel message is a task, so it still lets the page paint and still
lets Stop respond, but it is not a timer and is not clamped.

## The weights

🔴 **`model-af3-int5` is DeepMind's AlphaFold 3, not OpenFold3.** Every manifest
in the lineage says so - `model.name` is `alphafold3`, `source` is
`af3.bin.zst`. They carry a Prohibited Use Policy. `tools/build_site.py` reads
the manifest and refuses to publish them without
`LOCALFOLD_ACCEPT_MODEL_TERMS=alphafold3`, and the deploy workflow demands the
same repository variable before it untars the release - that check is on the
only path that actually publishes.

An Apache-2.0 bundle needs an OpenFold3 export. The blob is at
`~/af3_converted_cd2/of3_ported_weights.bin.zst` and
`tools/export_af3_model.py --model openfold3` exists, but the OF3 dialect turns
on four branches the stock graph does not have - a column-attention pair-bias
swap, a symmetrised bond matrix, an element index shift, and Fourier weights
read from the checkpoint - and none of those are implemented or verified here.

int5 costs nothing measurable: 1405 MiB -> 265 MiB, and 6MRR folds to 0.66 A
against float32's 0.69, which is the spread between diffusion seeds.

## What a denoiser call costs, and what made it cost less

A sampler calls the diffusion head up to 200 times and everything else once, so
the head is the whole optimisation target. On a 59-residue chain, steady state:

| stage           | as found | now |
|-----------------|---------|-----|
| conditioning    |    48   |  11 |
| atom encoder    |   100   |  18 |
| single-proj     |    -    |   2 |
| transformer     |   549   |  72 |
| atom decoder    |    48   |  24 |
| **one call**    | **760** | **134** |

🔴 **AND AT 150 TOKENS IT IS A DIFFERENT KERNEL LIST, WHICH IS WHERE THE 2026-09-03
WORK CAME FROM.** The table above is a 59-residue chain, where the atom blocks
are small. At 150 tokens a call was 267 ms - transformer 143, atom decoder 59,
atom encoder 46 - and **three kernels were reading the conditioning from global
memory once per token per channel, on every lane**: `ffw-out` and
`attention-output` in the transformer, and the atom blocks' `output`. The value
is indexed by (token, d) and never by the channel a lane owns, so all 256 lanes
of a workgroup wanted the same tile-by-C_COND floats. `conditionedProject` did
the same with two tensors.

| kernel | before | after | |
|---|---:|---:|---|
| ffw-out | 33.4 | **14.1** | the token tile lifted from two, then the conditioning staged |
| attention-output | 19.1 | **9.0** | the zero-gate loop was 60% of it |
| output (x3) | 11.6 | **10.0** | five global reads of the conditioning became one stage |
| project / project-keys | 3.3, 2.1, 2.1 | **1.8** | act AND queries_cond, four loops each |

A denoiser call **267 -> 221 ms**, its transformer 143 -> 111. A 200-step fold
at that length is about nine seconds less.

🔴 **AND WHERE THE STAGE GOES IS DECIDED BY RESIDENCY, NOT BY STYLE.** Three of
these reuse an array that is already dead - `wt` in ffw-out, `gated` in
attention-output, the raw tensors in conditionedProject - because a second
array would have taken those kernels from 6 to 12 KiB, or 8 to 16, and halved
how many workgroups a core can hold. The atom `output` kernel is the exception
and gets a fifth array: it already holds 24 of this device's 32 KiB, so it is
one workgroup a core either way and the array is free. It also CANNOT reuse
`cond_norm`, because its second adaptive-zero projection reads the RAW
conditioning after the normalised form has been written - which would have been
silent.

So 200 steps is about 27 seconds on a 59-residue chain, where it was 152, and a
whole diffusion-200 fold measures 26.3 s end to end against a flow-8 fold's 2.6.

🔴 **AND AT 150 TOKENS THE HEAD IS A DIFFERENT SHAPE, WHICH IS WHERE THE LAST
WIN CAME FROM.** A call there is 261 ms - transformer 133, atom decoder 61, atom
encoder 45, conditioning 15 - and `ffw-out` alone was 33.4 of it. Its token tile
was pinned at two by `Math.min(2, tile, fits(intermediate))`, a sizing term that
assumed a workgroup staged `outTile * INTERMEDIATE` floats. Chunking had removed
that long before: it stages `outTile * outChunk`, 6 KiB at four. The cap was
never lifted, so the measurement that set it - outTile 4 at 85 ms against 2's 74
- stood against a kernel that no longer existed.

The tile is what amortises the weight read, one per step of the intermediate
multiplied into every token of the tile. Re-measured with the chunking in place,
as medians of repeated runs of bench-diffusion-transformer.js: **at 150 tokens
2 -> 138 ms, 4 -> 128, 8 -> 132; at 59 tokens they tie.** `ffw-out` 33.4 -> 25.0,
the transformer 143 -> 133, a call 267 -> 261. Arithmetically neutral to every
digit - each token's sum runs over the same intermediate in the same order
whatever the tile - and check-af3-diffusion-transformer.js reports the identical
relRMS either side.

The lesson is not the tile. It is that a cap and the measurement justifying it
outlive the kernel they were about, and nothing fails when they do.

THE REST OF THE FOLD, for scale, all on the same 59-mer:

| trunk pass, 32 MSA rows   | 756 -> 570 ms  |
| trunk pass, 1024 MSA rows | 1093 -> 804 ms |
| AF3 checkpoint load       | 5470 -> 1364 ms |
| AF2 monomer / multimer load | 1012 / 874 -> 417 / 400 ms |

🔴 **AND BOTH HOT PATHS ARE NOW FLAT, WHICH IS WHERE THE CHEAP WORK ENDS.** The
head's largest kernel is ffw-out at 21 ms of 134; the trunk's top six were
pair-transition 84, tri.project 63, tri.project-out 43, grid.project 41,
grid.attend 40, grid.project-out 39, with no outlier. Everything left is a
kernel rewrite - tiling both operands in shared memory - rather than a shape
fix, and the failures listed below are what that has to beat.

## The pairformer's kernels, rewritten for the shape rather than the arithmetic

A trunk pass went **540 -> 408 ms** at 59 tokens, **3.38 -> 2.25 s** at 150, and
**670 -> 544 ms** at 59 tokens with a 1024-row alignment; its pairformer 435 ->
311 and 2879 -> 1900, and its MSA stack at that depth 334 -> 196. A denoiser
call went 134 -> 111. 6MRR folds to 0.64 A, TM 0.960. AF2's evoformer shares five of these
kernels and its triangle projection went 0.581 -> 0.422 ms a block and its
output projection 0.405 -> 0.327, with the contraction dropping off the
profiler's list entirely.

🔴 **AND ITS BLOCK TOTAL IS NOT MEASURABLE TO THAT PRECISION HERE.**
`profile-af2-block.js` reports a block at 11.0 to 12.7 ms for the SAME build
across processes, so a 4% change in it says nothing; the per-dispatch numbers
above are stable to about 1% and are what a claim about AF2 should rest on. An
earlier "12.6 -> 10.4" in this file was two numbers from two processes and has
been withdrawn.
Nothing computes anything different; every checker is unmoved and the denoiser's
worst error against AF3's own moved 1.19e-5 -> 5.86e-6.

🔴 **THOSE ARE UNPROFILED NUMBERS AND THE TABLE BELOW IS NOT.** `--profile`
writes a timestamp pair per compute pass, and at these shapes that is about a
fifth of the trunk: the same build measures 439 ms without it and 528 with. Use
the per-pass numbers to rank kernels against each other, never to quote a total,
and take before/after totals from two runs that are both unprofiled.

| kernel | before | after | what changed |
|---|---|---|---|
| pair-transition   | 83.3 | 73.0 | chunked, then its rows made vec4 lanes |
| tri.project       | 61.3 | 45.4 | 2x2 -> 4x2, one vec4 a cell, rows staged as one |
| tri.project-out   | 42.8 | 32.8 | the same |
| grid.project      | 41.1 | 38.4 | q/k/v/gate interleaved, read as one vec4 |
| grid.attend       | 39.0 | 21.7 | a chunk of keys staged in workgroup memory |
| add               | 11.2 |  2.4 | four of the five folded into their producer |
| grid.project-out  | 39.1 | 15.6 | a tile of rows, where it was one |
| single-transition | 28.1 | 29.6 | untouched |
| tri.contract      | 23.3 |  8.6 | 1 output a thread -> 4x4, both tiles vectors |
| opm.contract      | 15.7 |  9.0 | a block of (i, j) pairs, and a bigger chunk |
| pair-logits       | 11.8 |  4.9 | heads as vec4, normalised once |
| grid.bias         |  5.6 |  3.0 | the same |
| single.project    | 20.4 | 11.1 | the width split over workgroups |
| single.project    | 20.4 | 11.1 | the width split over workgroups, outputs blocked |
| tri.normalize     | 13.7 |  8.3 | the LayerNorm staged, to coalesce |
| grid.normalize    | 11.8 |  7.4 | the same |

Every one of those is a ratio of reads to multiply-adds or a count of
workgroups, and every one is measured by a bench that runs in about a second an
arm - `bench-triangle-project.js`, `bench-grid-project.js`,
`bench-transition.js`, `bench-single-project.js` - against `bench-trunk.js`'s
forty seconds and 48-block average. Each checks its arms against the first,
because a tile the dispatch does not match leaves rows unprocessed and reads as
a speedup.

🔴 **AND THE NEXT PROTEIN IS NOT THIS ONE. `grid.attend` IS CUBIC IN N.**
Everything else in the pairformer is quadratic. Before it was staged, the
attention was 39 ms of 400 at 59 tokens and **564 of 2429** at 150 - the largest
kernel in the trunk by half again. Staged it is 21.7 and 318, second at both,
but the exponent has not changed and it will lead again on a longer chain.
Anything further should be measured at 150, not at 59.

🔴 **AND ON A DEVICE WITH MATRIX UNITS THIS KERNEL IS NOW A DIFFERENT ONE.**
`src/af3/trunk/grid-attention-matrix.js` is the same online-softmax flash attention
AF2 runs, and it is 1.40x to 1.53x over the staged scalar kernel below across
200 to 640 tokens on an A100 - the whole measurement, and the AF2 tile rule that
does NOT transfer to it, are in docs/A100.md. It is off unless
`gridAttendMatrix` is set, and everything in this section is still what runs
everywhere else.

**Staging is what fixed it, and the reason generalises.** The dispatch gives a
workgroup one (pair row, head) and sixty-four queries, and the key loop runs over
the same axis for all of them - so each of the `2 * dimension/4` vectors a key
needs was fetched by sixty-four lanes issuing sixty-four IDENTICAL global loads.
A chunk of keys in workgroup memory makes that one load and sixty-four workgroup
reads: 1.9x at every length measured (0.425 -> 0.237 ms at 59 tokens, 5.75 ->
3.05 at 150, 36.9 -> 19.5 at 300). Chunks of 16 and 32 tie and 64 loses, so the
bound is 8 KiB. Two shape notes: the kernel reads `workgroup_id` rather than
`global_invocation_id`, because WGSL's uniformity analysis has to SEE that row
and head are workgroup-uniform or it rejects a barrier under the branch on them;
and a lane past the last query no longer returns, because it has to reach every
barrier the staging loop makes.

### What was tried on these and lost

- **Accumulating the query-key score into vec4s** reduced once, instead of a
  chain of `dot()`s - which looked like the problem, since `dot()` is four
  multiplies and four DEPENDENT adds and eight of them are a chain about
  thirty-two deep. Interleaved in one process at 150 tokens: the dot form
  5.10 ms, one accumulator 5.80, two 6.00, four 5.05.
- **More than one query per attention invocation.** The attention reads
  `dimension/4` vectors of k and as many of v per key and does the same number
  of vector operations with them - one load per multiply-add - and those loads
  do not depend on the query, so two queries an invocation should halve them.
  At 150 tokens it was **1.85x slower**, and four queries 3.9x: a query costs
  `dimension/4` vectors of q plus as many accumulators, so two is already 128
  floats of register and it spills.
- **Widening the transition's workgroup to 256 lanes** where the single track
  has only 59 rows to hand out: 0.728 ms against 128 lanes' 0.591. The
  LayerNorm's reduction grows a level and 384 channels split unevenly.
- **Raising the transition's row tile to 8 without chunking the intermediate**:
  1.77x slower. It fits in the 32 KiB this device grants and leaves one
  workgroup resident per core.
- **Blocking the transition's first matmul over i**, on its own: nothing.
- **Barriers.** Priced by removing them from the projection's k loop: exactly
  zero. The step stays at 8.
- **Batching a pairformer block's dispatches into ONE compute pass**, the way
  AF2's stack does. A trunk pass opens 1,332 of them, and the profiler's timed
  GPU work summed to 335 ms of a 451 ms wall - which looked like 116 ms of pass
  boundaries. It is not: batching measured 312 ms against 311. The gap was the
  profiler's own timestamp writes (451 profiled against 403 not) plus the
  labels the report does not list. Pass boundaries cost nothing here, and
  splitting them per dispatch is what lets profile.js see in, so they stay.
- **Splitting the SINGLE track's transition into two dispatches.** Its rows are
  its tokens, so the fused kernel gets 59 workgroups on a 59-residue chain and
  cannot tile out of it - 27 ms of a 307 ms pairformer for a fiftieth of its
  arithmetic. Splitting the two matmuls apart, with the widened intermediate
  travelling through a 363 KB buffer, gives twelve times the workgroups and the
  same weight traffic. It measured **65 ms** - a widening pass of 53.5 and a
  contraction of 11.9 - against the fused 27. The widening repeats the row's
  LayerNorm once per slice of the intermediate, twelve tree reductions a row
  where there was one, and that is more than the occupancy was worth. A third
  dispatch to normalise once would remove it; the fused form is 27 ms and this
  would have to beat it from 53.5, so it was not pursued.
- **Reading the transition's widening weights as vec4.** Its two weight reads a
  channel were half its instructions, and consecutive lanes read consecutive
  slots - so four consecutive slots to a lane makes those two reads two vec4
  reads and the multiply-adds eight: eleven instructions to buy thirty-two where
  it was five to buy eight. It needs a chunk of four workgroup widths, and at
  the tile that then fits it measured 1.63 ms against the current shape's 1.39.
  A third measurement saying this kernel is not waiting on its weight reads.

### What the MSA stack cost, once anyone measured it at depth

At 59 tokens and 1024 rows the stack was 334 ms against a 310 ms pairformer -
the untouched half of the trunk. Two kernels were most of it and both had the
same shape of fault:

| kernel, 1024 rows | before | after |
|---|---|---|
| opm.contract | 113 | 60 |
| msa.project | 62 | out of the top twelve |

`msa.project` gave a ROW TO A THREAD, walking WIDTH outputs by C_M channels and
re-deriving the normalised activation inside both loops - so a row's 64 values
were recomputed 64 times each, and two thirds of the kernel was that. A
workgroup a row: 64 lanes share the reduction, stage the normalised row once,
and take an output each.

`opm.contract` needed a two-dimensional block, and the reason generalises. A
cell's product is `left[i][c] * right[j][e]`, so an i-by-j block of pairs reads
BLOCK_I values of left and BLOCK_J of right to make BLOCK_I * BLOCK_J products;
a run of consecutive SLOTS shares an i only by accident, and nothing the
compiler can see says so.

### The ceiling these are measured against

`tools/gpu/probe-alu.js` asks the device directly, with no memory in the way:

| | GFLOP/s |
|---|---|
| scalar f32 multiply-add | 1287 |
| vec2 | 2526 |
| vec4 | 5034 |

and 396 billion workgroup-memory reads a second. Every one of those is about
**640 billion instructions a second**, which is the number that actually
governs: a vec4 multiply-add and a scalar one and a workgroup read all cost one
instruction, so this is an instruction-count machine and vectorising pays
exactly when it reduces the count.

That reframes everything above. The trunk's kernels ran at 900 GFLOP/s to
1.1 TFLOP/s, which is not 25% of a 3.6 TFLOP/s paper peak - it is **70-85% of
the scalar ceiling**, and about a third of the instruction rate once their loads
are counted. The remaining factor is in the loads, not the arithmetic.

🔴 **AND HALF PRECISION MOVED THE CEILING ITSELF, which is the one way past
that argument.** An f16 multiply-add issues at 1.7x an f32 one for the same
instruction, so a kernel at 85% of the f32 scalar ceiling is not finished - it
is finished IN f32. The kernels that took f16 accumulators now run past that
line; see the f16 budget section above and tools/gpu/profile-af2-block.js.

🔴 **WHICH IS WHY VECTORISING THE TRANSITION BOUGHT NOTHING BY ITSELF.** Its
tile's rows became vec4 lanes - four multiply-adds into one, and a quarter of
the workgroup reads - and the pair shape measured 1.488 ms against the scalar
1.475. The kernel waits on its two weight reads per channel, not on its
arithmetic. What the vectorisation DID buy is room for the tile: as scalar code
tile 8 lost to tile 4 (1.556 against 1.525), and vectorised it wins (1.394
against 1.494), because the rows now cost a quarter of the workgroup memory they
did.

🔴 **AND RANKING THE TRUNK'S KERNELS BY GFLOP/s MISLEADS, WHICH COST A DAY'S
LEAD TO FIND OUT.** `tri.project-out` measures 672 GFLOP/s against
`tri.project`'s 977 on the same shape, and it had the fault to match: it
contracts TWO matrices at each (k, output channel) cell and staged their
weights as two SCALAR arrays, where projectAB packs its four into one vec4.
Scalar, its inner loop was two source reads, four weight reads and sixteen
multiply-adds a step of k - 0.73 useful operations an instruction against 2.9.
Packing the pair into a vec2, with one vec2 accumulator a cell, should have
been worth about 1.5x.

It was worth **3%** (210.5 -> 205.2 ms), because the kernel is not instruction
bound. Counted properly, per dispatch at 150 tokens:

| | source | weights | store | total | achieved |
|---|---:|---:|---:|---:|---:|
| tri.project | 92 MB | 184 | 23 | 300 MB in 2.675 ms | **112 GB/s** |
| tri.project-out | 184 MB | 92 | 12 | 288 MB in 1.950 ms | **148 GB/s** |

Both are AT or ABOVE the 114 GB/s `probe-alu.js` measures for streamed global
reads. They differ in GFLOP/s because they differ in multiply-adds per BYTE -
project-out reads two source tensors and one set of weights, project reads one
and four - and not because one is better written than the other. The change is
kept because the code is simpler and the 3% is real, not because the reasoning
that motivated it was right.

The lever on a memory-bound kernel is traffic, and traffic is what the tile
divides - which is why every tile here has been swept and why every sweep ends
at the same place: a bigger tile cuts traffic and loses more to occupancy. 32x16
beats 32x32, 32x8, 16x16 and 16x32 at 150 tokens.

🔴 **AND THE KERNELS ARE NOW AT THE PRACTICAL CEILING, WHICH IS NOT THE PAPER
ONE.** At 150 tokens with a 512-row alignment - a real fold - a trunk pass is
2.60 s, and every one of its top kernels runs at about **270 billion
instructions a second** against the 640 billion probe-alu.js measures with no
memory in the way. Their instruction counts are within about 1.3x of what the
arithmetic needs. The remaining factor is latency the machine is not hiding, and
it does not yield to another tile: the list above is eight attempts at cutting
instructions further, and every one measured worse or level.

🔴 **AND PRICING A READ BY SUBSTITUTING A CONSTANT OVERSTATES IT.** Replacing
the projection's weight-tile reads with a constant took it from 0.525 to 0.375
ms, suggesting 29% to win; packing those four reads into one vec4 - which is as
far as that goes - was worth 5%. A constant lets the compiler hoist the
multiply-add too, so the arm measures the read AND the arithmetic that depended
on it. Useful for ranking, useless as a target.

🔴 **AND ONE ALGEBRAIC IDENTITY IS NOT ONE HERE.** `grid.attend` subtracts 1e9
from each masked logit inside an `if`. Computing that penalty once per key and
ADDING it - with `select(-1.0e9, 0.0, masked > 0.0)`, or with a plain `var` set
in an `if` - is the same expression and measures **relRMS 2.24e-1** against the
CPU reference where the `if` measures 9.63e-7. Deterministic, and identical to
the last digit whichever of the two rewrites is used, so it is a real difference
and not noise. It was not run down. Do not rewrite it.

The matrix kernel routes around it rather than resolving it: it hoists the
mask's READ out of the per-query-row loop into a staged per-key array, which is
what the redundancy was, and leaves the conditional itself written exactly as it
is here. Whatever this is, it is still unexplained.

What paid, in order of size:

1. **Two kernels had no grid at all.** `aggregate` and `single-initial` both
   dispatched ceil(TOKENS/64) workgroups with one thread to a token - ONE
   workgroup for any protein under 64 residues - and each lane then walked a
   whole matmul, 2.4M multiply-adds in `aggregate`'s case. It was 43 ms of the
   atom encoder's 82: more than its three cross-attention blocks put together,
   in the pass that only pools their output.
2. **The transformer was weight-bandwidth bound by 25x.** One workgroup per
   token meant every workgroup read the block's entire weight set: 5.9M floats
   for 2.4M MACs, a quarter of a MAC per byte where the device needs about
   twelve. A call read 33 GB of weights, which at ~350 GB/s is the 107 ms it
   took. Tiling over tokens - with the output range split so tiling does not
   cost occupancy - took the stack to 74.
3. **The block loop awaited `popErrorScope()` AND `onSubmittedWorkDone()` per
   block**, two host-device round trips a block, 48 a call. `DeferredValidation`
   exists for exactly this and the pairformer had used it for a year.
4. **Per-token workgroups were 64 lanes wide**, so the token count was the
   occupancy. 256 is the ceiling and the optimum.
5. **Weights packed and uploaded per call.** Now packed once per weight object
   and resident on the device - src/runtime/resident.js.
6. **The key rows are a gather of the query rows**, four slots to an atom, and
   the atom transformer projected each slot separately. Projecting per atom and
   expanding is a quarter of the work and numerically identical.
7. **The pair conditioning does not depend on sigma** and was rebuilt 200 times.

🔴 **AND SIX THINGS THAT LOOKED OBVIOUS AND WERE WORTH NOTHING**, which is most
of what this section is for. Batching the 24 per-block submits into one encoder
(199 vs 205 ms). Skipping ~14 MB of per-call readback that is immediately
re-uploaded (4 ms - unified memory makes a copy back nearly free). Widening the
ATOM kernels the way the transformer's were widened (they already launch 1440
workgroups). Replacing the atom kernels' redundant serial LayerNorms - all 64
lanes walking all 128 channels, four times - with workgroup reductions, which is
strictly less work and landed inside the noise. Raising
maxComputeWorkgroupStorageSize to lift the token tile from four to eight, which
lifts a ceiling that was never binding. And tiling further in general: 4 and 2
beat every larger pair measured.

🔴 **TWO MEASUREMENTS LIED, BOTH BECAUSE THEY WERE TOO CHEAP.** A bisect of the
atom encoder on a bench that averaged two calls, with a ten millisecond spread,
reported a REMOVED pass as costing negative time and named the attention blocks;
`aggregate`, four times bigger than anything guessed, only appeared once
bench-head.js reported a median over nine calls with its range. And a 30%
"speedup" from tile 8 was the shader factory defaulting the tile to 4 while the
dispatch divided the token count by 8 - half the tokens were never projected. It
was caught by two numbers disagreeing that should have been identical, not by a
checker; the factory now throws rather than defaulting.

🔴 **THE TRANSFORMER IS AT A LOCAL OPTIMUM AND FOUR THINGS SAY SO.** Its tile
and lane counts are both measured maxima; giving each workgroup ONE of q/k/v/gate
instead of four - on the theory that 4 x TILE accumulators were spilling - was
worse at 59 tokens and at 200 (the token tile is then staged four times);
reading the token tile from global instead of workgroup memory, to let the tile
grow past what that memory caps, was worse again; and `splits` makes no
difference at all, so the lane imbalance it creates at SLICE 384 against 256
lanes is not costing anything. f16 is settled too, in README.md: 13% SLOWER at
1.89e-4 error, because Apple GPUs run f32 and f16 ALU at the same rate and these
kernels are ALU-bound.

🔴 **AND THERE IS A BETTER TOOL THAN THE ONE USED HERE.** Every bisect above
disabled a pass and re-measured, which is noisy and cost two wrong conclusions.
`timestamp-query` is in the features src/runtime/device.js already requests, and
README.md records per-kernel timestamp profiling being used on the triangle
stack. Use that first next time.

The remaining floor is the transformer's arithmetic intensity, which tiling by
four improves and does not fix. Measure with
tools/gpu/bench-diffusion-transformer.js, which takes about three seconds
because it synthesises its weights, and gate any change on
tools/gpu/probe-head-vs-af3-steps.js.

## The denoiser, and the law that governs it

A call is **125 -> 122 ms** at 59 tokens, its transformer 73 -> 67, from chunking
and vectorising `ffw-out` (20.0 -> 16.7 ms of the 24 blocks). That is small, and
the reason it is small is the useful part.

🔴 **AND THAT MODEL WAS WRONG, WHICH TOOK A PROFILE AT 240 TOKENS TO SEE.**
Halving the traffic by doubling the tile bought nothing once the structure
allowed it (323 ms against 324), so the stack is not bandwidth-bound; the
numbers below were a coincidence of scale. What it IS bound by is the
instruction count of four kernels that each read one or two weights per
multiply-add, and restructuring those took a 240-token stack **324 -> 234 ms**
and a 59-token denoiser call 121 -> 117:

| kernel, 240 tokens, 8 blocks | before | after | what changed |
|---|---|---|---|
| pair-logits | 27.2 | 7.4 | heads as four vec4, channels outside |
| adaln | 8.6 | 3.1 | a tile of tokens |
| ffw-adaln | 7.9 | 2.8 | the same |
| attention-output | 12.5 | ~10 | a tile, and one output an invocation |

`pair-logits` is the one that mattered: it is quadratic in tokens where the
token projections are linear, so it leads on any real protein. It looped heads
OUTSIDE channels, re-reading the normalised pair row for each of the sixteen and
reading one weight per multiply-add - 48 instructions to buy 16. The heads are
contiguous in the projection, so they are the vector: channels outside, four
vec4 accumulators, nine instructions.

🔴 **AND ONE OUTPUT AN INVOCATION IS WHAT MADE THE REST OF IT WORK.** These
kernels' accumulators are (matrices x tile groups x outputs a lane) vectors and
every one is live across the whole channel loop, so a second output a lane
doubles the registers - enough to spill at eight tokens, where tile 8 measured
542 ms against tile 4's 332. Each kernel now splits its own output range to
exactly `lanes` wide rather than sharing one `splits`; their ranges differ
(heads*dimension, the doubled intermediate, the channels) so one number cannot
make all three exact.

The superseded reasoning, kept because the arithmetic is still worth seeing:

🔴 **THE TOKEN TRANSFORMER READS ALL 566 MB OF ITS WEIGHTS ONCE PER TILE OF FOUR
TOKENS.** Twenty-four blocks of 5.9M floats
is 566 MB; the tile is 4, so a 59-token call makes fifteen passes over it, 8.5
GB. At the 114 GB/s `tools/gpu/probe-alu.js` measures for STREAMED global reads
- and 23.6 MB a block is far past any cache - that is 74 ms. The transformer
measured 73. The model holds at every length:

| tokens | tiles | measured | per tile |
|---|---|---|---|
| 59 | 15 | 67 ms | 4.5 |
| 120 | 30 | 139 | 4.6 |
| 240 | 60 | 326 | 5.4 |
| 480 | 120 | 912 | 7.6 |

The drift upward is the attention, which is quadratic; the linear term is 4.5 ms
a tile, which is 566 MB at 126 GB/s.

**The tile is capped by workgroup memory, and lifting the cap changed nothing.**
`xt` holds TILE x 768 activations - 12 KB at four tokens, 24 at eight. Chunking
the channels unties that, and it is implemented; with it, tile 8 at 240 tokens
measures 253 against tile 4's 235, and tile 4 wins at 59 tokens too (65 against
82). So the traffic the tile divides was not what the stack was waiting on.

🔴 **AND f16 WEIGHTS ARE NOT THE LEVER EITHER, WHICH IS THE MEASUREMENT THAT
SETTLES THE MODEL.** README records f16 COMPUTE being rejected (13% slower;
Apple runs f32 and f16 ALU at the same rate), and that says nothing about f16
STORAGE with f32 accumulation - a bandwidth change rather than an arithmetic
one, and the obvious move if 630 MB of resident weights were the problem. Built
(the device feature, half-width packing, and one rewrite of the finished WGSL
turning every `weights[...]` into `f32(weights[...])`) it measured **85 ms
against 65 at 59 tokens** and 232 against 234 at 240. Slower, or level. The
bytes were never the constraint; the conversions are instructions and
instructions are.

It also costs accuracy that is not free: relRMS against the f32 reference goes
1.88e-2, against a 3.02e-6 rounding envelope. That is still inside AF3's own
bfloat16 noise - eleven mantissa bits against eight - but there is no reason to
spend it for nothing.

### What else was tried on the head, and lost

- **Tiling `attention-output` over tokens, on its own.** 69 ms against 65 for
  the stack, and 70 with a split of the output range added. It only became a
  small win once every kernel took one output an invocation.
- **Lifting the token tile past four**, which is what the traffic model above
  said to do. 253 ms against 235 at 240 tokens with the channels chunked, and
  worse without.
- **Vectorising `qkvg` and `ffw-wide` over the token tile**, which takes qkvg
  from 24 instructions a channel to nine: 68 ms against 67. The same lesson the
  trunk's transition taught - these kernels wait on their weight reads, not on
  their arithmetic. It is kept because the code is simpler, not because it is
  faster.

## Fixed: the side chains were compressed, and the loader was reading four wrong tensors

Reported from the page: side chains badly placed and rings wrong, at any number
of steps. `tools/gpu/probe-sidechains.js` measured it against the reference
conformers' own rigid tables, and against AF3's own 200-step sample of the same
59-mer:

|                    | bond ratio | 1-3 ratio | PHE ring bonds |
|--------------------|-----------|-----------|----------------|
| AF3 itself         | 1.017     | 1.015     | 1.407 1.404 1.404 1.405 1.409 1.408 |
| this port, before  | 0.927     | 0.908     | 1.122 1.099 1.287 1.198 1.164 1.303 |
| this port, after   | 1.015     | 1.017     | 1.407 1.403 1.407 1.402 1.407 1.401 |

**The bug.** `diffusionWeights` loaded the diffusion atom encoder's four pair
tensors under their UNSUFFIXED names. The checkpoint has each of them twice, at
identical shapes: the unsuffixed set belongs to the pair conditioning computed
over a token's own 24 dense atom slots (AF3 captures it as `[tokens, 24, 24,
16]`), and the `_1` set to the queries-keys layout the atom transformer actually
works in (`[subsets, 32, 128, 16]`). Loading the wrong four threw nothing,
changed no shape, and folded a plausible protein - with every side chain about
8% short. `targetFeatureWeights`, ten lines above in the same file, carries a
comment warning about exactly this trap.

**Why nothing caught it.** The only checker that reaches the whole head,
`tools/oracle/check_af3_denoiser.js`, builds its weight dict BY HAND rather than
through the loader - so it scored 6.8e-6 against AF3 the whole time the shipped
pipeline was wrong. Everything downstream compared the GPU against our own CPU
reference, which was fed the same hand-built weights. A checker that does not go
through the loader does not check the loader.

**How it was found**, in the order the possibilities died:

1. `tools/gpu/probe-af3-trunk-sample.js` substituted AF3's OWN trunk into
   `foldBatch`'s `reuse` path. The side chains stayed at 0.921, which
   exonerated the trunk and its 3.7e-2 pair disagreement.
2. `tools/gpu/probe-head-cpu-vs-gpu.js` ran one denoising step both ways on
   AF3's own 59-token batch: 5e-7. Not the shaders either - the CPU reference
   and the GPU shared the error.
3. A 20-step oracle dump WITH the head's arguments captured
   (`--capture-args 'diffusion_head/__call__$'`) gave AF3's own answer at every
   rung of the schedule. `tools/gpu/probe-head-vs-af3-steps.js` asked ours the
   identical twenty questions and got 2e-2 to 6e-2 at EVERY level - which said
   the divergence was the molecule, not the noise level, and killed the
   hypothesis that the EDM preconditioning was wrong at low sigma.
4. `tools/gpu/probe-head-stages-vs-af3.js` then ran the same comparison on the
   TWELVE-mer, where the head is supposed to be exact, and got 0.102. That is
   the moment it stopped being about the molecule: the same dump, the same
   reference code, two weight dicts.

After the fix the head reproduces AF3 to about 1e-6 at all twenty noise levels
from 4608 A down to 0.03, and the bond ratios agree to three decimals.

🔴 **AND EVERY SAMPLER MEASUREMENT PREDATED THE FIX.** The sigma0 sweep, the
ligand-flow knee at sigma_data, the flow-versus-diffusion step counts and the
160 A default were all measured against a denoiser that was 3-6% wrong at every
noise level. Two have been re-run and their docstrings now carry the new
tables; what changed is worth reading, because it is a lesson about what a
sampler sweep can and cannot tell you:

- **The rankings did not move.** sigma0 still trades a ligand's bond lengths
  against a protein's backbone in the same direction, with the knee still at
  sigma_data. A sweep that ranks settings is nearly blind to whether the model
  underneath is the right one.
- **The magnitudes did.** On 1QYS the 160 A default cost 0.043 A before and
  0.025 A after, and the seed ranges that used to be disjoint now overlap. On
  HEM at sixteen steps AF3's own top of schedule got WORSE, 0.065 to 0.168,
  while 160 A improved to 0.047 - so the gap the default exists to close is
  wider against the correct weights, not narrower.
- **pLDDT moved most of all**, 6MRR to 85.8 and 1QYS to 79.8 from the high
  fifties, which is the clearest single sign that the weights were wrong: the
  confidence head was reading a structure the trunk did not predict.

Still to re-run: the flow-versus-diffusion step counts, and the ligand-only
HEM-forms-by-8-steps observation the default was chosen from.

## Templates, and where the slots are placed

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

🔴 **AND THE TEMPLATE EMBEDDER IS NO LONGER UNCHECKABLE.**
`tools/oracle/dump_af3_trunk.py --template <pdb>[:CHAIN]` folds a query with a
real structure as its template and captures the module's inputs and its
per-slot outputs. That answers the objection at the top of
`src/af3/trunk/template-reference.js` - "with no template the six geometry features
are identically zero, so nothing here can tell a correct implementation of them
from a wrong one" - which was true and is the reason only the empty-slot path
exists. See docs/AF3.md's template entry for the numbers.

It writes the template's mmCIF from the PDB's OWN ATOM NAMES rather than
through atom37. ColabDesign2's `_mmcif_for` does the same job but reaches AF2's
`residue_constants`, which imports `dm-tree` - not installed here, and not
worth installing to copy 37 strings that every PDB line already spells out.

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
openbind"*. `AF3_FAMILIES` in `src/bundles/manifests/index.js` is the list;
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

🔴 **AND `?model=` HAD TWO READERS, WHICH IS WORSE THAN BEING IGNORED.**
`applyModelFromUrl` takes it as the model row's value; `web/model.js` took it as
a manifest URL whenever the family was monomer, which is how a page is pointed
at weights somewhere else. `?model=monomer` is the one spelling that reaches
both - it selected AlphaFold 2 and then fetched `<origin>/monomer`, so the live
page said **"failed to load model manifest: 404"** for a model that folds
perfectly well from the dropdown, at pLDDT 96.5. The override now requires a
PATH - a slash, or a `.json` ending - because a path is what it was for; a bare
family name belongs to the other reader.

🔴 **AND IT WAS FOUND BY MISTAKING IT FOR SOMETHING ELSE.** The 404 appeared
minutes after 129.8 MiB of `af2-monomer/*.js` were deleted from Hugging Face, on
the one bundle those files belonged to, which is as convincing a coincidence as
this repository has produced. The deletion was innocent - `ScriptTensorStore`
reads `manifest.js` only under `file://`, and from the bundle's own directory,
never from a remote - and folding monomer from the dropdown proved it. **A
regression that appears next to a change is not evidence it came from it.**

🔴 **AND A `?model=` THAT IS IGNORED LOOKS EXACTLY LIKE ONE THAT WORKED.** The
complaint about an unknown name was written twice before it was visible: once
before the vendored viewer's own "Ready." line overwrote it, and once before the
parameter had even been read, because it was called beside the Fold button's
enabling - which runs EARLIER in the file. It waits for that specific string
now. `of3` is deliberately not an alias for `openbind`.

## The trunk at 512 tokens, which nothing had profiled

The AF3 priors were fitted at 200 tokens. AF2's were fitted at 400 and one of
them moved 2% when re-swept at 825 (docs/AF2.md), so the same question is worth
asking here. `bench-trunk.js --model=/model-af3-int5/manifest.json --tokens=512
--msa=128 --passes=2 --profile`, steady pass:

| stage | ms |
|---|---:|
| pairformer | 1896 |
| msa-stack | 628 |
| embedder | 514 |
| template | 409 |
| distogram | 213 |
| **whole** | **3724** |

🔴 **AND THE PAIRFORMER IS GPU-BOUND HERE, WHICH IS WORTH KNOWING BEFORE
OPTIMISING ANYTHING ELSE.** `pairformerSplit` reports encode **22.8 ms**, wait
**1644.7**, release 0 - the host finishes encoding in a fortieth of the time the
GPU takes, so nothing on the host side of this stage is worth moving.

The GPU passes, 1374 ms over 2048 dispatches:

| pass | ms | share | groups a pass |
|---|---:|---:|---:|
| `grid.attend` | 397.1 | 30.6% | 16384 |
| `tri.contract` | 167.3 | 12.9% | 5837 |
| `tri.project` | 107.9 | 8.3% | 11878 |
| `pair-transition.wide` | 94.3 | 7.3% | **2048** |
| `grid.project` | 94.0 | 7.2% | 15974 |
| `pair-transition.down` | 77.5 | 6.0% | **256** |
| `tri.project-out` | 66.6 | 5.1% | 6656 |

🔴 **`pair-transition.down` LAUNCHES 256 WORKGROUPS ON A CARD THAT FITS 4542.**
That is 6% of the trunk's GPU time in a kernel using about a twentieth of the
device, and it is the clearest starved dispatch anywhere in this port -
`probe-occupancy.js` is what makes it legible as one rather than as a number.
`pair-transition.wide` at 2048 is short of the same mark by half.

It is NOT a knob. `pairTransitionChunkBytes` at 64, 128 and 256 MiB leaves the
group count at exactly 256 and the pass at 77.5 / 77.48 / 77.87 ms - the
transition is not chunked at this shape, so the chunk target never binds and the
count comes from the split kernel's own tiling. Fixing it means changing that
tiling, which is kernel work and wants its own differential.

The ceiling is worth stating before anyone starts: eliminating the pass entirely
is 2% of the trunk, because `grid.attend` is 30% of the GPU time and is already
on the matrix units with 16,384 workgroups a pass - well fed, and the reason the
trunk looks the way it does.

### 🔴 And `pairTransitionChunkBytes` never reached this track at all

Chasing that starved dispatch found the reason a knob could not move it.
`pairformer-block-webgpu.js` and `msa-stack-webgpu.js` both put
`pairTransitionChunkBytes` into the options they hand `encodePairTrack`, and
`pair-track-gpu.js` - the only caller of `transitionSplitChunkRows` - never read
it, so the rule fell back to its own 64 MiB default. **Every AF3 and OpenDDE arm
ever measured with that knob was measured at 64 MiB**, which is why the first
sweep of 64, 128 and 256 moved the group count not at all and the pass by 0.4 ms
of 77.5. The same shape as `matrixLinear: false` falling through into the matrix
path: a knob with no off position, and a knob with no effect, are both worse than
no knob.

Wired through, it does what it says and still does not pay:

| chunk | down | groups | wide | groups | whole trunk | peak |
|---|---:|---:|---:|---:|---:|---:|
| 64 MiB | 77.12 ms | 256 | 94.08 | 2048 | 3717 ms | 1284 MiB |
| 256 | 66.72 | 1024 | 121.95 | 8192 | 3671 | 1500 |
| 512 | **63.04** | 2048 | **120.52** | 16384 | 3741 | 1788 |

The starved pass gets its workgroups - 256 to 2048, and 1.22x - and the `wide`
pass loses 27 ms, more than `down` gains. `wide` was never starved at 2048
groups, so a bigger chunk buys it nothing and costs it locality: the widened
buffer it streams grows with the chunk. The trunk is 3717 / 3671 / 3741, a wash,
for up to 504 MiB of extra peak.

So the default stays at 64 MiB and the fix is the plumbing, not the value. It
matches what docs/PERF.md already records for the ESMFold2 trunk - "a 6% GPU win
and a 1% WALL loss for 144 MiB" - reached there by a different route.

## The single projection was running an M2's constant, and it is 4x

`single.project` is one workgroup per token and per split. At 68 tokens that is
136 workgroups on a card that holds 4542, and it was **15.0 ms of a 104.3 ms
trunk** - more than any kernel in the model except `grid.attend`, for a track
that is O(n) where the pair track is O(n²).

Two knobs fix it and **neither works alone**, which is why two earlier attempts
found nothing. `bench-trunk.js --profile`, reading `gpuTotalMs` (every pass, not
the listed rows), three rounds an arm, arms interleaved:

| n=68 | trunk GPU | `single.project` | groups |
|---|---:|---:|---:|
| shipped (target 110, 64 lanes) | 104.3 / 104.3 / 104.3 | 15.00 | 136 |
| target 2048 alone | 96.9 / 96.8 / 96.8 | 7.48 | 204 |
| **target 2048 + 128 lanes** | **93.0 / 92.8 / 93.1** | **3.69** | 204 |

| n=300 | trunk GPU | `single.project` | groups |
|---|---:|---:|---:|
| shipped | 607.8 / 609.3 / 608.4 | 15.67 | 300 |
| target 2048 alone | 601.1 / 601.3 / 601.3 | 7.86 | 900 |
| **target 2048 + 128 lanes** | **599.0 / 597.7 / 596.9** | **4.99** | 900 |

Across four sizes, whole-trunk GPU: **68 tokens −10.8%, 150 −6.0%, 300 −1.7%,
512 −0.5%**, and the kernel itself 4.05x, 3.97x, 3.13x, 2.22x. The win is
largest at the SHORT chains, which is what a page mostly folds, because the
starvation is: fewer tokens, fewer workgroups, same fixed cost.

🔴 **WHY NEITHER KNOB PAYS ALONE, AND BOTH TOGETHER DO.** 128 lanes over the
unsplit 384-wide output leaves each lane three outputs deep and buys nothing -
measured at 15.48 -> **16.24**, worse, which reproduces docs' earlier
"`project` at 128 lanes is WORSE (15.60 -> 16.09)" exactly. Split three ways the
output is 128 wide, one lane an output, and the two compose. The earlier split
sweep used `maxSplits` 6, whose `perSplit` is 64 and which therefore cannot use
a wider lane at all. **Sweeping one axis of a pair says the pair does not pay.**

🔴 **AND A LANE WIDTH IS A REQUEST, NOT A VALUE.** `perSplit` is `width /
splits` and the split count is derived from the token count, so 128 divides it
at one length and not at another: at AF3's 384 the rule picks 3 splits for every
n below 1024 - `perSplit` 128 - and **2 from 1024 to 2047, where `perSplit` is
192**. A prior naming 128 outright would have folded every chain up to a
thousand tokens and thrown on the next one. `createSingleAttentionShaders`
halves the request until it divides, landing on the 64 every caller had before
the knob existed; and the pipeline key names the RESOLVED width, not the
request, because that is what the shader contains.

🔴 **IT IS A REORDERING.** Three workgroups normalise the row where one did and
128 lanes reduce it in a different tree, so `check-af3-block-any`'s single
residual moves in the eighth figure (2.0899e-4 either way) and its pair not at
all. On a fold: **max |dx| 0.001 A, 531 of 574 atoms identical** - the same
class as `attnSplits` 1 -> 4, which the ampere prior already ships at 541/574.
pLDDT 84.26493204096884 against 84.26493426067073.

**ampere only.** `singleProjectSplits`' own table shows 6 splits LOSING on an M2
at every n it was measured at, which is why these were left as parameters in the
first place - see DEFAULTS_ARE_MEASUREMENTS.

🔴 **AND EVERY NUMBER ABOVE IS A TRUNK NUMBER, NOT A FOLD NUMBER.** The trunk's
GPU time is about 104 ms of a 1.18 s warm AF3 fold at 68 tokens - 9% - so 11 ms
off it is **1%**, and that is what a whole fold shows:

| | first | warm |
|---|---:|---:|
| AF3 68 tokens, new | 2.159 / 2.153 / 2.177 s | **1.166 / 1.159 / 1.219** |
| AF3 68 tokens, old | 2.142 / 2.156 s | 1.184 / 1.169 |
| OpenDDE 6mrr, new | | **2423 / 2420 ms** |
| OpenDDE 6mrr, old | | 2432 / 2493 ms |

Consistently in the right direction and consistently about 1%, with the first
fold unmoved because it is compilation and weights. Take the change - it is
free, and it is a kernel running at four times its old rate - but do not quote
the trunk figure as a fold figure. This is the distinction docs/PERF.md keeps
making about `--profile`: a share of a stage is not a share of a wall.

## The pair-logits cache was fitted at 200 tokens, and covers a quarter at 400

`PAIR_LOGITS_CACHE_BYTES` is 64 MiB and its own note says the cache is
`64 x tokens^2` bytes a block - so at the 200 tokens it was measured at it
covers **all twenty-four blocks**, and at 400 it covers **six**. That is the
same shape as AF2's `TRANSITION_CHUNK_TARGET_BYTES` (docs/AF2.md): a byte cap
fitted where it happened to cover the whole workload, found by parsing every
`export const` in `src/` for a comment that cites only a short length.

`pairLogitsCacheBytes` is the knob now, null taking the constant.
`bench-head.js --tokens=400 --calls=9 --model=/model-af3-int5/manifest.json`,
median of the eight steady calls, two rounds:

| | round 1 | round 2 |
|---|---:|---:|
| 64 MiB (the constant) | 51.5 ms | 51.5 |
| 128 MiB | 50.0 | 50.5 |
| **256 MiB** | **47.0** | **47.5** |
| 512 MiB | 47.0 | 47.0 |

**8.7% of a denoiser call**, and 256 is the knee because 24 blocks of 10.24 MiB
is 246 MiB - 512 buys nothing because it caches the same twenty-four.

🔴 **AND AT THE PAGE'S DEFAULT IT IS WORTH ABOUT 1.5%, WHICH IS THE HONEST
NUMBER.** `AF3_COUNTS.flow` prefers **16** steps, so the sampler is about 0.8 s
of a 4.5 s fold and a whole fold reads **4.5 s on both arms** - the saving is
under the resolution of that clock. It is the diffusion mode, whose dial reaches
**200** steps because that is what AF3 was trained with, where 8.7% of a call
becomes about 0.9 s. Peak goes **1457.7 -> 1623.8 MiB** at 400 tokens.

Set in the **ampere prior only**, beside `opmPairBlockBytes` and
`transitionChunkBytes`, which is now three 256 MiB caps on a 40 GB card.

**Bit-exact, and checked with the right instrument.** A cache of a recomputed
value cannot change an answer unless it goes stale, and
`tools/diff-fold-coords.py --b='--tune-json={"pairLogitsCacheBytes":67108864}'`
says so: **574/574 atoms identical, max |dx| 0.000000 A**. 🔴 The first attempt
to check it compared two PDB captures that were both EMPTY - `fold.js --folds=1`
prints JSON and the grep matched nothing - and reported "STRUCTURE IDENTICAL",
which is this repository's "a gate that cannot fail is not a gate" in one line.

## The outer product mean's block was fitted at 150 tokens and inverts at 300

`OPM_BLOCK_I` is 2 and its note reasons from "the workgroup count still wins at
these sizes", measured at 59 and 150 tokens. At 400 tokens `opm.contract` is the
**second-largest kernel in the trunk** - 245.9 ms against `grid.attend`'s 259.4 -
and the block that wins there is the one that note measured as worst.
`bench-trunk.js --profile --model=/model-af3-int5/manifest.json --msa=512`:

| tokens | 59 | 150 | 256 | 300 | 350 | 400 |
|---|---:|---:|---:|---:|---:|---:|
| blockI 1 | **3.67** | 19.21 | 53.51 | **75.67** | **108.61** | **147.76** |
| blockI 2 | 5.53 | **14.41** | **38.81** | 111.52 | 178.05 | 245.69 |

Two wins only in a band, and **the cliff between 256 and 300 is sharp and
reproducible** - 256 reads 38.81/39.63 for two against 53.51/53.98 for one, and
300 reads 111.52/112.33 against 75.67/75.73. It is not a depth effect: at 1024
rows the ordering is identical (59: 6.92 against 11.06; 150: 39.01 against
33.15; 256: 121.60 against 84.26; 400: 325.20 against 499.72). `blockI 2` goes
superlinear past 256 where `blockI 1` stays smooth in the pair count; the
mechanism is not established and the L2 working set does not explain it, since
256 tokens at 1024 rows already exceeds L2 and two still wins there.

`opmCellChunk` swept beside it stays at 256: 128 is 247.8, 256 is 147.9, 512 is
150.7, 1024 is 173.2. Only the block moves.

**A PRIOR, one-sided and conservative** - `opmBlockITokens`, above which the
stack takes 1. 🔴 It shipped for one commit as a DERIVATION applying to every
device, which was wrong: the five derivations this repository has all read
something the device reports - its measured width, its memory budget - and 256
tokens is not that, it is where this A100's memory system turns over. Another
part keeps the measured block of two until someone measures it there. The band where two wins is left exactly as measured, and so is
the 59-token case, where one wins by 1.9 ms. A trunk pass at 400 tokens is
**1244.4 -> 1143.6 ms, 8.1%**, and a fold at 200 tokens is untouched
(`opm.contract` 39.08 ms, the block-of-two path). Bit-exact where it fires:
`diff-fold-coords.py --sequence=<400> --b='--tune-json={"opmBlockI":2}'` gives
**3046/3046 atoms identical, max |dx| 0.000000 A**.

## A recycle criterion for a trunk that returns no structure

AF2 stops recycling on ColabFold's `compute_tol` - the RMS change of every
C-alpha pair distance - and `src/af2/model/recycle-convergence.js` implements it.
**AF3 has no early stop at all**, and the reason is structural rather than an
oversight: `src/af3/fold.js`'s recycle loop is a bare `for (let pass = firstPass;
pass <= recycles; pass += 1)` because AF2 recycles a STRUCTURE and AF3 recycles
the single and pair representations alone, running the sampler once at the end.
There are no coordinates to compare until every recycle is already paid for.
OpenDDE runs the same fold; ESMFold2 has its own recycle path and does not.

So the only signal is the representation. `src/af3/feature-convergence.js` is
the metric - relative RMS change, `||b - a|| / ||b||`, dimensionless so it can
be compared across models and token counts - and it costs nothing: the loop
already holds `previousPair` and `previousSingle` as HOST arrays, because the
pairformer reads them back each pass. No kernel, no readback, no device memory.

`recycleDeltas` reports it per pass and `--recycle-tolerance` acts on it, **0 and
off by default**. On the 68-token default input:

| pass | pair | single |
|---:|---:|---:|
| 0 | 1 (the zero seed) | 1 |
| 1 | 0.0725 | 0.0554 |
| 2 | 0.0159 | 0.0070 |
| 3 | 0.0045 | 0.0023 |

Monotone, and decaying about 4x a pass. `--recycles=3 --recycle-tolerance=0.02`
runs three passes instead of four, 2.243 s -> 2.067 s.

🔴 **AND THE OBVIOUS WAY TO CALIBRATE IT DOES NOT WORK, WHICH IS THE FINDING.**
The natural experiment is to fold at 0, 1, 2 and 3 recycles and see where the
STRUCTURE stops moving. Done that way the 68-token input reads 0.551, 0.120,
0.076 and 0 A against the deepest fold, which looks like convergence by recycle
1. It is not a measurement of the trunk. **The sampler is stochastic, so the
control is to vary the seed and hold the recycles fixed** - and superposed, at
three recycles:

| | sampler noise, same trunk | recycle 2 vs 3 |
|---|---|---|
| 68 tokens, pLDDT 87.5 | **0.133 - 0.294 A** | **0.023 A** |
| 250 tokens, pLDDT 42 | **4.580 - 12.702 A** | 3.265 A |

**At both confidence levels the recycle-to-recycle difference is at or below the
sampler's own noise**, by an order of magnitude at high confidence. A structure
RMSD cannot resolve one recycle count from another on a diffusion model, so it
cannot calibrate a criterion either, and the reading that suggested it could was
noise.

🔴 **AND THE FIRST VERSION OF THAT CONTROL WAS WRONG TOO, IN THE OTHER
DIRECTION.** Unsuperposed, the seed-to-seed numbers are 7.4 to 15.1 A at
pLDDT 87.5, which reads as chaos. Two folds of the same molecule are in
arbitrary rigid-body poses: different seeds start from different noise and there
is no canonical frame. Same-SEED comparisons share one, which is why the recycle
sweep gave small numbers without alignment and why mixing the two designs is a
trap. Kabsch first, always, unless the seed is held.

pLDDT is better behaved but not clean either - its own seed spread is 0.520 at
68 tokens and 1.146 at 250, against recycle spreads of 1.707 and 7.747. Usable
at 3-7x the noise, and not a fine instrument.

**What IS deterministic is the trunk.** The pair delta reads 0.0479 on all three
seeds of the 250-token input, to four figures, because only the sampler is
stochastic. That is the whole argument for measuring convergence on the
representation rather than on what comes out of it.

🔴 **AND THE FIRST PASS'S SEED IS NOT THE RIGHT SHAPE, WHICH THE GATE CAUGHT.**
The loop seeds `previousPair` with `tokens^2 * 128` zeros - AF3's pair width -
and OpenDDE's pair is 384 wide, so the two differ by exactly 3x and a strict
comparison raised `feature convergence over 591872 and 1775616 elements` on a
model that had folded a moment earlier. The strictness is worth keeping for the
passes that DO compare, so pass 0 is reported as 1 rather than measured: it has
no previous by construction, which the zero seed was only ever standing in for.

🔴 **SO THE CRITERION READS THE DISTOGRAM, WHICH IS THE ONE TRUNK QUANTITY IN
ANGSTROMS.** Every pass computes one already, to draw the contact map while the
trunk is still recycling, so the expectation over its 64 bins is a predicted
distance matrix for free - and the RMS change of that matrix is exactly what
ColabFold's `compute_tol` takes over a structure. AF2's criterion and this one
are the same measurement in the same unit, one from coordinates and one from
the trunk. `expectedDistances` and `distanceChange` in
src/af3/feature-convergence.js; the tolerance is angstroms and ColabFold's
default for the analogous number is 0.5.

| pass | pair | single | **distogram** |
|---:|---:|---:|---:|
| 68 tokens, 1 | 0.0725 | 0.0554 | **0.1244 A** |
| 2 | 0.0159 | 0.0070 | **0.0393** |
| 3 | 0.0045 | 0.0023 | **0.0184** |
| 250 tokens, 1 | 0.0970 | 0.0373 | **0.4195** |
| 2 | 0.0705 | 0.0115 | **0.2602** |
| 3 | 0.0479 | 0.0084 | **0.2289** |

🔴 **AND IT IS BIT-IDENTICAL ACROSS SAMPLER SEEDS** - 0.4195, 0.2602 and 0.2289
on every seed of the 250-token input, to four figures, while the structures
those same folds produced differ by 4.6 to 12.7 A. That is the whole argument
for the instrument: the trunk is deterministic and only the sampler is not, so
this is the one number a stochastic structure cannot contradict.

**And it resolves the 250-token puzzle.** That input looked unconverged because
its structures moved 3.3 A between recycle counts. Its TRUNK was converging the
whole time - 0.42 to 0.26 to 0.23 A, under ColabFold's tolerance from the first
pass - and every angstrom of that structural variation was the sampler on a
pLDDT-42 input. Low confidence does not mean the trunk is still moving; it means
the sampler cannot commit, which is a different thing and the structure cannot
tell them apart.

At 0.5 A the default input runs **two passes instead of four**, 2.301 s ->
1.944 s, for a pLDDT change of 0.126 - smaller than that fold's own seed spread
of 0.520.

🔴 **AND SIX REAL SEQUENCES KILLED THE ONE-CROSSING RULE.** The two inputs above
both settle monotonically, which is exactly what made a first-crossing threshold
look sound. Folded at three recycles, the per-pass change in angstroms:

| | tokens | pLDDT | pass 1 | pass 2 | pass 3 |
|---|---:|---:|---:|---:|---:|
| 1qys (Top7, de novo) | 91 | 81.64 | 1.329 | 0.715 | 1.240 |
| 6mrr | 71 | 70.20 | 0.386 | 0.122 | 0.140 |
| **GB1** | 56 | 74.13 | **0.488** | **1.092** | 0.394 |
| lysozyme C | 129 | 36.23 | 0.329 | 0.132 | 0.095 |
| ubiquitin | 76 | 90.43 | 1.684 | 2.418 | 0.336 |
| villin HP36 | 36 | 90.55 | 2.842 | 0.611 | 0.134 |

**The trunk does not settle monotonically.** GB1 dips under 0.5 A at pass 1 and
then moves **1.092 A** at pass 2; ubiquitin rises from 1.684 to 2.418 before
falling; 1qys never settles at all. A rule that stops at the first crossing
throws GB1's second pass away.

**Two consecutive passes under the tolerance is safe on all six** and still
stops 6mrr and lysozyme a pass early. It is not fitted to these numbers - one
step under a threshold is the textbook thing not to trust - but GB1 is why it is
there, and `test/feature-convergence.test.js` pins that sequence by name so the
rule cannot be simplified back. At 0.5 A the default input then runs three
passes instead of four, 2.255 -> 2.131 s, for a pLDDT change of 0.084 against
its own seed spread of 0.520.

🔴 **IT STILL SHIPS AT ZERO.** Six sequences are a corpus in the sense that they
falsified a rule, and not in the sense that they calibrate one: four of the six
never reach two consecutive passes under 0.5 A within three recycles, so what
the tolerance buys on a real workload is two folds' worth of evidence. The
saving where it does fire is one pass of four. What is established is the
instrument, its unit and the shape of the rule; the threshold is not, and a
default that silently drops a recycle should be worth more than that before it
is one. `recycleDeltas` reports all three numbers on every fold, so the corpus
can keep growing from runs people were doing anyway.

## PROTENIX-V2: OPENDDE'S DIALECT WITH FOUR FLIPS, AND THREE THINGS THE TENSORS SETTLED

Protenix-v2 (ByteDance, **Apache 2.0**, best-A **0.703** in the reference's
table - the strongest model this port can legally serve). Added as a `BLOBS`
entry and a dialect, with **no exporter change at all**: the trunk exported on
the first attempt at 207 tensors, and the full bundle is 404 tensors /
464.7 M parameters / 1773 MiB. That is `export_af3_model.py`'s own claim - "a
second model becomes a different `--blob`, not a second exporter" - holding.

**Its widths are the tensors' and none is written down.** 48 trunk blocks of 8
triangle heads at c_z 256, a 2-block template stack of 2 heads at 64, four MSA
blocks at c_m 128 with value_dim 8, a 64-bin distogram with a biased half-logit
projection. Every one matches the reference's `PROTENIX2_SETTINGS`, which is
what says `src/af3/weights/weights.js`'s derivation works rather than a table here
having to keep step.

### What the assertions caught, one run each

🔴 **`splitPairConditioning` WAS WRONG, AND THERE ARE THREE SHAPES NOT TWO.**
Read off the reference's `DIFFUSION_PROJECTED_RELPOS` membership it looked like
a `true`; `diffusion-weights.js` threw at once - *"this bundle does not carry
z_trunk_projection and its dialect says otherwise"*. The tensors:

| | relpos | trunk pair | `pair_cond_initial_projection` |
|---|---|---|---|
| AF3 | raw 139 | passed through | [267, 128] |
| OpenDDE | projected | **also** projected | [256, 128] |
| **protenix2** | projected | passed through | **[512, 256]** |

512 is 256 + 256, `relpe_projection` is [139, 256] and there is no
`z_trunk_projection`. So `splitPairConditioning` is OpenDDE's BOTH-projected
case and protenix2 wanted a new `projectedRelpos`. The reference's list is about
the featurisation; LocalFold's flag was about the tensor layout. They are not
the same question and the guard is the only reason that took one run.

🔴 **`padSingleCondUnknownDna` WAS WRONG TOO, COPIED FROM OpenDDE.** *"single
conditioning is 831 channels but its LayerNorm scale is 833"*. protenix2 carries
the two unknown-DNA columns where OpenDDE does not - both being
OPENFOLD3_LINEAGE, which is exactly why this is a flag and not a lineage
property.

🔴 **AND `pairInitFromSingle` WAS SETTLED BY A SHAPE BEFORE ANYTHING RAN.**
OpenDDE's `left_single` is [384, 384] and builds the pair from `s_init`; this
bundle's is **[447, 256]**, so it builds it AlphaFold 3's way. The reference has
no convention list for this - the shape is the statement - and
`check-af3-embedder` passing is the confirmation.

### Where it stands

| | |
|---|---|
| `check-af3-block-any` | **PASS** - the pairformer computes its reference at protenix2's widths |
| `check-af3-embedder` | **PASS** |
| `check-af3-diffusion-conditioning` | **PASS** - initialPair 3.20e-7, widths 256/256/384 derived |
| `check-af3-msa-block` | `fused weight has 131072 elements; expected 32768` - exactly 4x, which is c_m 128 against 64 times 8 heads against 4. A fused MSA weight is still sized from AlphaFold 3's constants |
| `check-af3-trunk`, `check-af3-template` | `missing tensor .../single_template_embedding/query_embedding` - protenix2's template embedder has a different module tree (43 tensors), not just different widths |

🔴 **AND TWO DECLARED FLAGS ARE NOT IMPLEMENTED, WHICH IS WHY A FOLD WOULD BE
WRONG IN TWO PLACES THAT NO FOLD CAN SEE.** `preSymmetrisedPde` symmetrises the
PDE logits before the head rather than after - the reference found it with
`confidence_parity.py` reading pde corr **0.87** while pae, plddt and resolved
were all at parity, and records that **no fold caught it**, because a symmetric
plausibly-scaled error metric stays symmetric and plausible.
`templateMeanOverAllSlots` divides the template term by every slot rather than
the occupied ones. Both are in `PROTENIX2` and nothing reads them yet.

### protenix2's template embedder is a different MODULE, and here is all of it

`check-af3-trunk` and `check-af3-template` stop at
`missing tensor .../single_template_embedding/query_embedding`, and the reason
is not widths. protenix2 runs **boltz2's fused template module**, and the pieces
correspond to AF3's one for one:

| AF3 / OpenDDE | protenix2 | |
|---|---|---|
| `query_embedding_norm` + `template_pair_embedding_8` | `z_norm` + `z_proj` | renamed |
| `output_layer_norm` + `output_linear` | `v_norm` + `u_proj` | renamed |
| `template_pair_embedding_0..7`, nine projections summed | **`a_proj` [108, 64]**, one projection of the concatenation | **fused** |
| `single_template_embedding/template_embedding_iteration` | `__layer_stack_no_per_layer/tmpl_pairformer` | renamed, one level up |

A sum of projections of the parts IS one projection of their concatenation, so
this is packing and naming, not a different model - which is what the
reference's own note means by "its Protenix-specific bits are the CONVERTER's
naming/feature conventions, not forward-graph shape".

**The forward:**

    v = z_proj(z_norm(z)) + a_proj(a_tij)
    v = v + pairformer(v)   x2
    v = v_norm(v)
    aggregate over templates
    u = u_proj(relu(u))

**The 108-wide feature, in order** - `[disto(39), pb_ch(1), rt_j(32), rt_i(32),
uvec(3), bb_ch(1)]`:

- `disto` 39 bins, one-hot of CB-CB squared distance against
  `linspace(3.25, 50.75, 39)**2` with the last upper edge at 1e8, masked by
  `pb2d * asym_mask_2d`.
- the frame is **Boltz's, not AF3's**: `e1 = norm(C - CA)`,
  `e2 = norm((N - CA) - e1 ((N - CA).e1))`, `e3 = e1 x e2`, rot columns
  `[e1, e2, e3]`; `uvec = R_i^T (ca_j - ca_i)`, normalised, masked by
  `fr2d * asym_mask_2d`. N/CA/C come from rigid-group 0, whose atom order is
  **[C, CA, N]** and not [N, CA, C].
- restypes are **32**-class, not AF3's 31, through
  `_AF3_TO_OF3 = range(21) + (31,) + (21,22,23,24) + (26,27,28,29) + (25,)`.

🔴 **AND `rt_j` COMES BEFORE `rt_i`, WHICH IS NOT A TYPO.** protenix appends
`expand_at_dim(aatype, -3)` then `expand_at_dim(aatype, -2)`, and the first
inserts the new axis first, leaving the tensor varying along **j**. `a_proj` is
converted with no column permutation, so the order has to be native's exactly.
The reference records having these the other way round as worth **corr 0.9985
against 0.999998**, unnoticed until `template_parity.py` existed.

🔴 **AND THE DISTOGRAM IS MASKED BY THE MULTICHAIN MASK IN THE FORWARD, NOT
ONLY BY pb2d IN THE FEATURISER.** Missing that half is invisible on a monomer
and actively harmful on a complex: a distogram one-hot is nonzero for EVERY
pair, so unmasked cross-chain entries are not zeros but confident FABRICATED
inter-chain distances. Measured there: a template made a 146+74 heterodimer
WORSE, interface 31.76 -> 46.76 A, while the same template rescues four other
ports to 1-2 A.

🔴 **WHAT IS DONE HERE, AND WHY THE FORWARD IS NOT.** `templateWeights` takes
the dialect and loads either shape, guarded against the bundle
(`fusedTemplateEmbedder` against `a_proj`'s presence), so protenix2's template
weights load. The forward is deliberately NOT written yet: LocalFold's
`check-af3-template` compares a GPU path against this repository's own CPU
reference, so writing both halves from this specification would produce two
pieces of new code agreeing with each other and a checker that cannot fail.
That is the trap CLAUDE.md names - "verify against the oracle, not against our
own reference" - and the ladder is the reference's `template_parity.py`, which
is what found the `rt_j`/`rt_i` order above. Dump it first.

### ...and the forward, written against the oracle, at 1.52e-7

`tools/oracle/dump_af3_template.py` records af3-any-model's own
`template_parity.ours` - that gate's entry point, real protenix2 weights, 34
scopes mapped and 0 unmapped - as
`oracle-dumps/af3-oracle-template-protenix2.json`: 76 tokens, the 108 feature
columns and the module's output separately. `fusedTemplateEmbedding` in
src/af3/trunk/template-reference.js is held to it by
`tools/gpu/check-af3-template-fused.js` at **relRMS 1.52e-7**, ours rms 12.4434
against native's 12.4434.

🔴 **AND THE ORACLE EARNED ITS KEEP ON THE FIRST RUN.** The specification above
says protenix concatenates the "j-varying block FIRST", and the reference
records the other order as worth corr 0.9985 against 0.999998. Read literally
that gives `restype_j` then `restype_i`, and that scores **5.77e-2** - which IS
corr 0.9985. The right order against `our_features`' output is `restype_i` then
`restype_j`, for 1.52e-7.

Both statements are true. The reference is describing NATIVE's tensor naming,
where a name says which index the tensor varies along; the feature dict has
already resolved it. **A specification read off someone else's source cannot
settle which convention its words are in** - and the failure landed on the exact
number that source had written down for this mistake, which is what a correct
oracle looks like when you are wrong.

Had the forward been written from the specification and checked against a CPU
reference written the same way, both halves would have carried the same swap and
agreed at 1e-7.

**Still not written: the featuriser.** The 108 columns go in from the dump. The
frame convention, the bin edges, the 32-class remap and the multichain masking
are all specified above and none of them is gated yet.

## BOLTZ-2: THREE CHECKERS ON THE FIRST EXPORT, AND TWO WIDTHS AF3 HID

Boltz-2 (MIT, best-A **0.430** in the reference's table - the strongest model in
it). A `BLOBS` entry and a dialect, no exporter change: **442 tensors, 507.5 M
parameters, 1936 MiB**. `check-af3-block-any`, `check-af3-embedder` and
`check-af3-msa-block` all passed on the first run at its widths.

**It is a bigger port than protenix2 - seven new conventions against three** -
but it shares protenix2's hardest piece: both run the FUSED template embedder,
already written and held to an oracle at 1.52e-7. What boltz2 adds there is an
OUTER residual around the stack (`templateStackOuterResidual`), which protenix2
does not have, and the reference's note on that is worth keeping: *"protenix
inherited the shared forward and got the wrong convention; rf3 escaped by not
inheriting it. Either a per-vendor convention is named or the next subclass gets
whichever behaviour its parent happened to have."*

### Two widths that were constants because AlphaFold 3 makes them coincide

🔴 **`targetFeatWidth: 447` WAS TYPED IN, UNDER A COMMENT NAMING THAT EXACT
FAULT.** boltz2's target_feat is **384**, and the checkers read "targetFeat has
10728 elements; expected 9216". It was always derivable - the single
conditioning's LayerNorm states its INPUT width, less the trunk single, less the
two unknown-DNA columns where the dialect pads:

    af3        831 - 0 - 384 = 447
    protenix2  833 - 2 - 384 = 447
    boltz2     768 - 0 - 384 = 384

🔴 **AND THE CONDITIONING'S OUTPUT WIDTH IS NOT THE SINGLE IT READS.** AF3's
`single_cond_initial_projection` is [831, 384] - 384 out, and the trunk single
it concatenates is also 384 - so `seqChannels + targetFeatWidth` happened to be
the input width and one variable served both. boltz2's is **[768, 768]**: 768
out, 384 in. Read as one number that gives 1152 against a LayerNorm of 768.
`trunkSingleChannels` is its own field now.

Both are the same shape of fault and neither is visible on AF3, OpenDDE or
protenix2, because all three have output == trunk single == 384. **A constant
that three models agree on is still a constant.**

### Where it stops

`check-af3-diffusion-conditioning` reads *"the pair conditioning's initial
projection is NaN against its own reference"* - a real signal and correctly
caught, where the same checker once passed an all-NaN OpenDDE arm because
`NaN > 1e-5` is false. boltz2's pair path is 128 wide with `relpe_projection`
[139, 128] and `pair_cond_initial_projection` [256, 128], so 256 = 128 + 128 and
`projectedRelpos` derives the trunk pair at 128 correctly; the NaN is downstream
of the widths.

Seven conventions are declared and unimplemented - `opmRowCountNorm`,
`opmBiasAfterNorm`, `noHeadNorm`, `reembedConfidencePair`,
`templateVisibilityByCoverage`, `rawRefCharge`, `templateStackOuterResidual`.
🔴 **AND `opmRowCountNorm` NEEDS MSA DEPTH > 1 TO BITE**: at depth 1 its bias
term is `(1 - 1/1) * b = 0` and the two normalisers agree, which is why the
reference's boltz2 single-sequence fold was exact while its MSA module was not.
A single-sequence gate cannot see that one.

### 🔴 A CHECKER PASSED BECAUSE BOTH SIDES WERE WRONG THE SAME WAY

`check-af3-diffusion-conditioning` read **3.20e-7 on protenix2** and that number
meant nothing. There are THREE pair-conditioning shapes and the code knew two:

| | relpos | trunk pair | concatenation |
|---|---|---|---|
| AF3 | raw 139 | passed through | `trunkPair + 139` |
| OpenDDE | projected | **also** projected | `2 * c_z`, on `z_trunk_projection` |
| protenix2, boltz2 | **projected** | passed through | `trunkPair + c_z` |

The third has `relpe_projection` and no `z_trunk_projection`, so `split` is false
and it fell into AF3's raw-139 arm:

    protenix2   256 + 139 = 395   against a norm of 512
    boltz2      128 + 139 = 267   against a norm of 256

🔴 **AND ONLY boltz2 WAS LOUD ABOUT IT.** Its 267 is LONGER than its scale, so
the LayerNorm read past the end and all 73728 elements came out NaN - on BOTH
sides, which is what made it obvious. protenix2's 395 is SHORTER than its 512,
so it read a prefix, stayed finite, and the GPU made the identical mistake. Two
wrong computations agreeing to 3.20e-7.

With the reference corrected, the same arm reads **1.01 on protenix2** - order
one, the GPU computing a different function - while AF3 stays at 2.10e-7 and
OpenDDE at 3.78e-7. The defect was there from the moment protenix2 was added and
the suite reported it as a pass.

**The only reason it surfaced is that a second model rounded the other way.** A
differential that compares two implementations of the same misunderstanding is
worth nothing, and nothing in its output says so - this one printed a number
four orders inside its bound. Adding a second model to a dialect is worth more
as a test of the FIRST one than the numbers suggest.

The reference and the checker carry the third branch now. The GPU does not: its
`pairWidth` is `pairChannels + RELATIVE_WIDTH` with a `split` flag and no third
mode, so protenix2 reads 1.01 and boltz2 NaN until it is written.

## THE FUSED TEMPLATE ON THE GPU, AND WHY protenix2 STILL DOES NOT FOLD

`Af3TemplateEmbedderGpu` takes the fused embedder now - a second `embed` shader
(`v = z_proj(z_norm(z)) + a_proj(a)`), a second weight ORDER, the mode in the
cache key, and the 108 columns as a per-pair buffer. Everything after the input
stage is the same code, because the fused module differs only in how the stack's
input is built.

`emptyFusedFeatures` builds a de novo fold's columns and **refuses a template**,
because the featuriser is not written and building it from docs/AF3.md's
specification would be unverifiable. The empty columns are measured, not
assumed: zero everywhere except restype_i and restype_j one-hot at column 31.

`check-af3-template-fused.js` now holds both halves: the CPU forward against
af3-any-model (1.52e-7 templated, 1.58e-7 empty) and **the GPU shader against
that CPU forward on the empty slots a fold actually builds (3.09e-5)**. The
shader had no check at all before, which is the only reason it was worth
suspecting when the fold came out wrong.

### Three hardcoded widths, and the one that is still open

🔴 **`check-af3-trunk` SEEDED `previousPair` AT `tokens^2 * 128`** - AlphaFold
3's c_z, typed in - so protenix2 at 256 read a recycling buffer half the length
its trunk expects and the WHOLE TRUNK came out NaN, envelope included. **An
envelope that is NaN is the tell**: it is a CPU reference disagreeing with
itself, which cannot be a port difference and has to be a malformed input.
Derived from the bundle, protenix2's trunk reads **pair 6.31e-5 at 1011x its
envelope**, beside AlphaFold 3's 855.9x.

`src/af3/fold.js` had the same constant in two places, under a comment that
already recorded OpenDDE's being 384 and "the two differ by exactly 3x". Fixed,
though it changed nothing here.

🔴 **AND protenix2 STILL FOLDS TO A BROKEN CHAIN** - 0.96 A backbone bonds
against an ideal 1.46, consecutive CA 6.5 A against 3.8 - with every stage
running and nothing erroring. The trunk is right (6.31e-5), the conditioning is
right (1.89e-7), the template is right on both paths. What is NOT checked is the
diffusion half: **75 tensors exist only in protenix2 and 74 only in AlphaFold
3**, because its atom transformer sits under `__layer_stack_no_per_layer` with
differently-concatenated leaf names.

🔴 **AND `check-af3-diffusion-transformer` "PASSED" FOR protenix2, WHICH MEANT
NOTHING.** It takes no `--model=`, so it opened AlphaFold 3's bundle and
reported on AlphaFold 3. The same is true of `check-af3-atom-encoder`,
`-atom-decoder`, `-diffusion-head` and `-sampler-gpu`: fourteen of the twenty
AF3 checkers are pinned, so a second model's diffusion path has no coverage at
all and a suite run says so only if you read which bundle each one opened.

**The control that proved it is protenix2's and not the harness's:** AlphaFold 3
folded from the SAME dumper's batch, the same 12-residue sequence and the same
tool reads N-CA **1.46** against an ideal 1.46, CA-C 1.53 against 1.52, CA-CA
3.81 against 3.80, pLDDT 89.7. The dumper, the sequence and `fold.js` are all
sound.

## protenix2 FOLDS A CHAIN NOW, AND THE LAST 0.73x IS THE ATOM DECODER'S

Two more AlphaFold 3 constants, both in `diffusionWeights`:

🔴 **`transformer: { pairChannels: 128 }` WAS TYPED IN.** The token transformer
reads the diffusion conditioning's PAIR, and PROTENIX2_SETTINGS widens
`heads.diffusion.conditioning.pair_channel` with the trunk - so protenix2's
`pair_logits_projection` is [6, 4, **256**, 16] where AF3's is [6, 128, 4, 16].
The stack was reading a 256-wide pair through a 128-wide stride.

🔴 **AND THE TWO LAYOUTS NEST AND ORDER DIFFERENTLY**, so one expression cannot
read both: AF3 is singly nested with the width at axis 1, protenix2 doubly
nested with it at axis 2. `txStackFor` cannot be reused either - it appends a
trailing `/transformer` because most leaves in that stack are named
`transformer<leaf>` CONCATENATED, and this one is not.

`encoder.trunkPairChannels: 128` was the same, and `diffusion_embed_trunk_pair_
cond` states it: [128, 16] under AF3 and [256, 16] here.

**What that bought, on 6MRR at 68 residues:**

| | N-CA | CA-C | CA-CA | radius of gyration |
|---|---:|---:|---:|---:|
| before | 0.96 | 1.02 | 6.52 | 4.0 A |
| after | 1.08 | 1.09 | **3.55** | **11.2 A** |
| AF3, same dumper and target | 1.46 | 1.52 | 3.79 | 11.1 A |
| ideal | 1.46 | 1.52 | 3.80 | 11-12 for a 68-mer |

So the topology is right - a compact 68-mer with pTM **0.908** where it was a
4 A ball - and the geometry gate passes. **It is not finished:** every bond is
still about **0.73x** ideal.

🔴 **AND THE SHAPE OF THAT ERROR NAMES THE STAGE.** Intra-residue bonds are
0.73x while CA-CA - between token centres - is 0.93x. A uniform scale would move
both equally. Per-atom offsets from the token centre being compressed while the
centres stay put is the ATOM DECODER's output, not the trunk's and not the token
transformer's.

**Everything upstream is verified:** trunk 6.31e-5, conditioning 1.89e-7,
template 1.52e-7 (CPU) and 3.09e-5 (GPU), embedder, msa-block and pairformer
block all at their own widths. And the reference folds protenix2 on this target
to **best 1.009 A, mean 1.391 A**, so ~1 A is what a correct port should reach.

🔴 **THE NEXT STEP IS AN L3 ORACLE, NOT MORE WIDTHS.** Four of the five hardcoded
constants found today were caught by a shape mismatch that threw; this one does
not throw, because every tensor is the right shape and only the ANSWER is wrong.
`denoise_parity.py` in the reference runs one whole denoise step - conditioning,
atom encoder, token transformer, atom decoder and the EDM scaling at once - and
that is what localises a wrong answer with correct shapes.

---

## boltz2 AND protenix2 ARE FINISHED, AND THE GATES THAT FINISHED THEM

The section above ends with "the next step is an L3 oracle, not more widths."
That was right, and it was not enough: an L3 oracle localises the DENOISER. Two
more instruments were needed, and between them they found nine defects nothing
here could see.

| | RMSD to the 6MRR crystal | TM | pLDDT | pTM | N-CA | CA-C | CA-CA |
|---|---:|---:|---:|---:|---:|---:|---:|
| **boltz2** | **0.537 A** | **0.973** | 96.2 | 0.798 | 1.46 | 1.52 | 3.80 |
| alphafold3 | 0.643 | 0.956 | 84.3 | 0.741 | 1.44 | 1.53 | 3.82 |
| OpenDDE | 1.525 | - | 92.1 | - | - | - | - |
| **protenix2** | **1.564** | **0.929** | 86.5 | 0.858 | 1.46 | 1.52 | 3.75 |
| ideal | | | | | 1.46 | 1.52 | 3.80 |

boltz2 is the best of the four here, which is what the reference's own table
says it should be.

### 🔴 THE ONE LESSON: A CHECKER THAT FEEDS ONE BUNDLE TO BOTH SIDES CANNOT SEE A CONVENTION

Every AF3 gate in this repository compares **the GPU against this port's own CPU
reference**. That catches a kernel, and it is structurally incapable of catching
a convention, a depth or a weight - because both sides read the same bundle and
are wrong together. Measured: at the moment boltz2's whole trunk was 1.34e+0
from af3-any-model's, `check-af3-trunk` read 2.74e-5 and `check-af3-confidence`
read 9.56e-5. Both passed. Both had passed all week.

Three oracles now close that, and each one found defects on its first run:

| | what it compares | what it found |
|---|---|---|
| `tools/oracle/dump_af3_denoise_stages.py` + `dump_af3_scopes.py` | af3-any-model's four denoiser seams, and every one of its 124 hk.Module outputs | ten affine LayerNorms, the transition up-gate, a negated weight in the bundle |
| `tools/oracle/dump_af3_trunk_taps.py` + `fold.js --trunk-oracle=` | its Evoformer's z at each stage, on the REAL batch | target_feat as a sum, two z-init terms, the MSA feature's 35th column, the MSA double-add, a 64-block pairformer |
| `tools/oracle/dump_af3_confidence.py` + `check-af3-confidence-oracle.js` | its ConfidenceHead on a real atom layout, module by module | protenix2's two missing terms, boltz2's 8-block stack |
| `tools/check-bundle-vs-params.py` | the BUNDLE against the params the reference loads | four `embed_pair_offsets` tensors negated by a converter fix that landed after the export |

### 🔴 AND THREE OF THE NINE WERE A DEPTH OR A WIDTH TYPED INTO A LOOP

| | boltz2 | everyone else | what it cost |
|---|---:|---:|---|
| trunk pairformer | **64** | 48 | ran three quarters of the trunk |
| confidence pairformer | **8** | 4 | ran half the head on inputs exact term for term |
| MSA feature width | **35** | 34 | dropped `is_paired`, which on a single-sequence batch is the whole alignment |

None of them throws. A stacked tensor with 64 blocks read 48 times is a valid
read; a 34-wide prefix of a [35, 64] matrix has correct strides. `trunkDepths`
and the confidence loader read both depths off the stack's leading axis now, and
the MSA width off `msa_activations`.

🔴 **AND THE DEPTH BISECT COULD NOT SEE THE FIRST ONE**, which is the part worth
remembering. Truncating BOTH sides to 1, 4, 16, 32, 36, 40 and 44 blocks agreed
to 1.55e-4 at every one of them - because every arm truncated to a depth under
48, where the two configurations are the same model. The cliff was between 44
and 48 and it was not chaos: at 48 the reference used blocks 48..63 and this
port had never loaded them.

### The conventions, in the order they were found

**In the diffusion head.** Ten of boltz2's LayerNorms are AFFINE where AF3's are
scale-only; the offsets are read from the BUNDLE (the converter has already
decided by emitting the tensor) rather than from a model-name table, and a zero
offset IS the scale-only LayerNorm, so no shader variant is needed. Its
conditioned transition has a THIRD projection - `SwiGLU(a) * a_to_b(a)` in four
stacks - and that one IS a shader variant, because a zero multiplier is not the
identity but a dead block.

**In the trunk.** `target_feat` is a SUM of seven terms, not AF3's
concatenation: the atom encoder's token activation plus six bias-free
projections, four of which are constant on an ordinary monomer and all six
trained NON-ZERO. z-init carries two more constant terms
(`token_bonds_type_embed` row 0 and `contact_encoding_unspecified`). The MSA
module ADDS ITS INPUT PAIR TWICE - its MSAModule returns the updated z and its
caller adds z to that. And its OPM divides before adding the bias, clamping
rather than nudging, which is worth `(1 - 1/n) * b` and therefore nothing at
depth 1.

**In the confidence head.** It rebuilds z and s from nine terms under
`~_boltz2_reembed`, normalises before no logit head, and splits both pair heads
into intra- and inter-chain halves - which on a monomer never fire, so a
single-chain gate cannot see whether they exist.

**And protenix2's head was wrong all along**: it needed `distance_feat_project`
(a second, UNBINNED distance term) and `input_single_norm` (the trunk single
LayerNormed and clamped to +/-512 before ANY use). Its fold went pLDDT 59.1 ->
86.5 on unchanged coordinates. `preSymmetrisedPde` had been declared in the
dialect and read by nothing; it is implemented now, on both models.

### 🔴 boltz2 WILL NOT RUN ITS TOKEN TRANSFORMER AT f16

| `--f16` | one denoise step against af3-any-model |
|---|---:|
| off | **3.50e-3** |
| on | 2.21e-1 |

Its 24-block token transformer amplifies its input by about **2.2e4**, measured
and LINEAR - half the input gap gives half the output gap (1.18e-2 against
2.29e-2). That is a property of the up-gate: no other model here does it. The
same amplification is why this repository's f64 CPU reference reads 2.20e-2
where the f32 GPU reads 3.50e-3 - the GPU accumulates the way the oracle does.
`check-af3-denoise.js` holds the GPU to the oracle and the CPU to the GPU for
exactly that reason, and prices the model's own arithmetic envelope beside both.

The fold is unchanged either way at 200 steps, so this is recorded rather than
acted on; a per-dialect precision floor is the fix if a longer chain shows it.

### The MSA is capped at num_msa now, and it was not

AF3's Evoformer subsamples to `config.num_msa` (1024 in every checkpoint of this
lineage) before the MSA stack sees a row. This port ran the whole array, which
for a featurised batch padded to 16384 rows was sixteen times the rows the model
takes: **3.2 GiB of `af3-msa.msa-scratch` and 1.5 s of a 3.3 s trunk**, now 280
MiB and 515 ms.

🔴 **AND THE FIRST num_msa ROWS ARE NOT AF3's num_msa ROWS.** It gumbel-shuffles
first, so which rows survive is a draw from a PRNG this port cannot reproduce -
and on a deep alignment the query itself survives only with probability
`num_msa/depth`. Taking the prefix keeps the query and keeps the alignment's own
order, which is `subsample_msa_keep_query`'s rule rather than `shuffle_msa`'s.
A coverage limit, named; `DETERMINISTIC_MSA=1` on the oracle side makes the two
comparable.

### What is still open

- The bundles are LOCAL. `model-boltz2-f32` was corrected in place by
  `tools/negate-bundle-tensors.py` after `check-bundle-vs-params.py` named the
  four tensors; a published bundle must be re-exported from a converter at or
  after the fix (`~/ported/boltz2/boltz2.bin.zst`, 2026-09-09 or later).
- boltz2's `templateStackOuterResidual` and `templateVisibilityByCoverage` are
  implemented and have never been exercised: 6MRR folds with no template.
- The inter-chain confidence heads and `opmRowCountNorm` both need a COMPLEX and
  an MSA of depth > 1 respectively. Neither can be seen on this target.

### OpenDDE AND OpenBind-0, MEASURED AGAINST THE ORACLE FOR THE FIRST TIME

| | one denoise step | trunk pair | confidence pLDDT | RMSD to 6MRR | TM |
|---|---:|---:|---:|---:|---:|
| alphafold3 | 1.55e-5 | 2.96e-4 | 3.46e-5 | 0.643 A | 0.956 |
| **boltz2** | 3.50e-3 | 3.49e-4 | 1.44e-6 | **0.537** | **0.973** |
| openbind0 | 8.22e-5 | 4.27e-4 | 4.26e-4 | 1.732 | 0.904 |
| opendde | 9.79e-7 | 7.65e-4 | ~1e-6 | 1.545 | 0.921 |
| protenix2 | 1.92e-6 | - | 1.47e-7 | 1.564 | 0.929 |

**openbind0 needed nothing.** It was already right, at every level, and this is
the first time anything measured it: its trunk's `target_feat` reads 4.93e-8,
its pair 4.27e-4, its confidence head 4.26e-4, and its bundle agrees with the
reference's params 406 of 406.

**OpenDDE was not**, and both causes were in the BUNDLE:

  * it joined `PADDED_SINGLE_COND` upstream on 2026-09-10 and this export
    predates it - its diffusion single conditioning normalises over 833
    channels, not 831, and the two re-inserted zero columns are not free
    because a LayerNorm maps a zero to -mean/std. A uniform 0.12%, exactly
    `1 - sqrt(831/833)`.
  * eight base-name encoder tensors were ZERO in the export where the reference
    has values.

Re-exported, both bundles agree 481 of 481 and the denoise step is 9.79e-7.

🔴 **AND THE WIDTH ASSERTION THAT SHOULD HAVE CAUGHT THE FIRST ONE COULD NOT
FIRE.** `targetFeatWidth` was derived as `scale - pad - trunkSingle`, i.e. FROM
the tensor it was checked against, so whatever the scale said the derived width
absorbed it. The stale bundle folded silently at target_feat 445 instead of 447.
It comes off `single_activations` now, which states 447 outright, and the stale
bundle raises.

### 🔴 OpenDDE's TRUNK AND CONFIDENCE, MEASURED AT LAST - AND THE TEMPLATE STAGE IS WRONG

Both gates now exist. `fold-opendde.js --trunk-oracle=` (the comparator is
shared with fold.js, in tools/gpu/trunk-oracle.js) and
`tools/gpu/check-opendde-confidence-oracle.js`, whose dump needed a new script:
`dump_af3_confidence.py` indexes
`confidence_head/~_embed_features/left_target_feat_project` and OpenDDE has no
such tensor, so it died with a KeyError. `tools/oracle/dump_af3_opendde_confidence.py`
drives the reference's own `OpenDDEConfidenceHead` with 0 unmapped scopes.

🔴 **AND THE FIRST THING BOTH GATES NEEDED WAS THE REFERENCE'S OWN BATCH,
BECAUSE THIS PORT'S CONFORMERS ARE NOT AF3'S AND THAT IS DELIBERATE.** LocalFold
ships ONE idealised reference conformer set shared by every family; the
reference featurises CCD geometry per input. Measured, same weights and same
code, on 6MRR:

| openbind0 `target_feat` | from a SEQUENCE | from the reference's batch |
|---|---:|---:|
| | **2.89e-2** | **4.93e-8** |

So every number in the table above is a `--dump=` number - the MODEL, with the
featuriser taken out - and a sequence-featurised oracle run has a floor of about
3e-2 that is the conformer choice and not a defect. That is worth knowing before
reading any oracle residual: `fold-opendde.js --dump=` exists for the same
reason, and needed `batchFromDump` to carry `residueOfToken`, which the
featuriser records and that path did not (OpenDDE re-tokenises and reads it, so
it died in `kindOfToken` on a batch that is complete for AF3).

**OpenDDE's trunk, on the reference's batch:**

| seam | relRMS |
|---|---:|
| `target_feat` | **3.14e-8** |
| `tap.z_init_generic` | **4.70e-8** |
| `tap.trunk_in_single` | **2.78e-7** |
| `tap.z_after_template` | **2.07e-2** |
| `tap.z_after_msa` | 5.91e-2 |
| `tap.trunk_out_pair` / `pair` | 1.08e-1 |
| `single` | 1.88e-2 |

**Exact into the trunk and wrong from the template stage on.** Everything after
inherits it, so there is one defect here and not four - and 6MRR carries NO
template, which makes it the empty-template path. openbind0's same seam reads
4.07e-6, so it is OpenDDE's, not the module's.

🔴 **FIXED, AND IT WAS THE EMPTY TEMPLATE SLOT'S RESTYPE.** OpenDDE takes
protenix's featuriser, which "fills its one empty template with the GAP restype
and zero-pads the rest" - so on a query with NO template its `template_aatype`
is **21 across slot 0 and 0 across slots 1..3**, which the reference's own batch
dump shows exactly (protenix2 the same; AlphaFold 3, openbind0 and boltz2 write
0 in every slot). This port wrote 0 everywhere, and an empty slot is not a
no-op: the aatype one-hot picks a row out of `template_pair_embedding_2`/`_3`,
so row 0 (ALA) went in where row 21 belongs.

| | before | after |
|---|---:|---:|
| the template module alone | 2.83e-1 | **1.32e-7** |
| the trunk seam `z_after_template` | 2.07e-2 | **3.80e-6** |
| `trunk_out_pair` | 1.08e-1 | 7.10e-2 |

AlphaFold 3 (1.17e-4 / 2.96e-4) and openbind0 (4.07e-6 / 4.27e-4) are unchanged
to every digit, and boltz2 and protenix2 still fold 6MRR at pLDDT 96.1 and 92.5,
because the flag is the dialect's.

🔴 **AND THE GPU PATH NEEDED THE SAME FIX FOR A DIFFERENT REASON: IT COLLAPSES
THE EMPTY SLOTS.** `Af3TemplateEmbedderGpu` runs all the empty slots as ONE pass
with a repeat count, on the argument that they "produce the same embedding by
construction - same all-ALA aatype". Under this convention they do not: slot 0
carries the gap and the rest carry 0, so they are two distinct embeddings. The
CPU reference alone read 1.32e-7 while the fold still read 2.07e-2, which is
what named the second half. The pass splits now, and only where a dialect
supplies a gap - elsewhere it is the one pass it always was.

### 🔴 FIXED: THE OUTER PRODUCT MEAN COMPUTED 256 CHANNELS AND OpenDDE'S PAIR IS 384

`@compute @workgroup_size(256)` and `let f = local` - **one lane per channel,
with no loop.** `has_f = f < C_Z` guards the SHORT case and nothing covered the
long one, so on a pair track wider than 256 the top channels were never computed
and never written. Every other stack in every model is 128 or 256 wide:

| MSA stack | pair channels | |
|---|---:|---|
| alphafold3 | 128 | fits |
| openbind0 | 128 | fits |
| boltz2 | 128 | fits |
| protenix2 | 256 | fits exactly |
| **opendde** | **384** | **a third of it missing** |

| | before | after |
|---|---:|---:|
| `check-af3-msa-block.js --model=opendde` | **4.94e-1** | **3.29e-7** |
| trunk `tap.z_after_msa` | 6.01e-2 | **2.68e-5** |
| trunk `tap.trunk_out_pair` / `pair` | 1.08e-1 | **7.65e-4** |

**OpenDDE's trunk is now exact end to end** - 3.14e-8, 4.70e-8, 2.78e-7,
3.80e-6, 2.68e-5, 7.65e-4 - and that last number is in line with AlphaFold 3's
own 2.96e-4, openbind0's 4.27e-4 and boltz2's 3.49e-4. The other four are
unmoved to every digit.

🔴 **AND THE GATE EXISTED AND HAD NEVER BEEN POINTED AT THIS BUNDLE.**
`check-af3-msa-block.js` takes `--model=` and runs in nine milliseconds; four
models read 1e-6 on it and OpenDDE read 4.94e-1. This whole hunt - an
io_callback tap in the reference, a `--stop-after-opm` stop point, `--msa-blocks`,
a row-slice comparator - ended at a differential checker that was already
written. **Run every gate against every bundle before building an instrument.**

🔴 **AND THE FOLDS BARELY MOVED - RETRACTING WHAT THIS PARAGRAPH FIRST SAID.**
It claimed the fix "cost OpenDDE 3.6 pLDDT", comparing 94.88 against 91.26. Those
are two different proteins: 94.88 is a 59-residue chain and 91.26 is 6MRR. Taken
properly, by putting the bug back and running both arms on one input:

| | with the bug | fixed |
|---|---:|---:|
| 6MRR, `--target=6mrr` | 1.492 A / TM 0.9342 / pLDDT 92.086 | 1.501 / 0.9343 / 92.120 |
| a 59-residue chain, 128-row MSA | pLDDT 94.884 | 94.879 |

**A third of the MSA stack's pair output was missing and the structure moved by
0.01 A.** The defect is three orders against the oracle and invisible in the
fold, and neither number is evidence about the other. This fix buys a trunk that
agrees with the reference, not a better prediction.

🔴 **THE INSTRUMENTS THIS NEEDED, AND THE TWO THAT LIED FIRST.** The MSA stack
is a `hk.experimental.layer_stack`, so `hk.intercept_methods` sees NOTHING
inside it - 29 modules traced in the whole trunk and not one of them in the
stack. Two things were needed:

  * an **io_callback** tap in the reference's `EvoformerIteration`
    (`AF3_MSA_BLOCK_TAPS=1`), because a direct record inside the stack raises
    `TracerArrayConversionError`; it exposes `msa_block_post_opm` and
    `msa_block_msa_act`.
  * `--stop-after-opm` here, which ends the stack after the OPM so the pair the
    loop already reads back IS that seam.

🔴 **AND `--stop-after-opm` WAS AN EARLY `return` THAT SKIPPED `queue.submit`,
SO IT THREW THE WHOLE BLOCK AWAY.** The stopped pair came back identical to
`z_after_template` and read as "our outer product mean contributes EXACTLY
zero" - which matched the reference's own z_after_template-vs-post_opm distance
to three digits (4.873e-2 against 4.87e-2), and that coincidence is what made it
convincing. The control that broke it: the same arm on the SEQUENCE path, where
the MSA plainly works (pLDDT 87.3 with none against 94.9 with 128 rows) and the
stopped pair was also unchanged. **A number that agrees with a prediction to
three digits is still worth one control.**

### 🔴 TEMPLATES ON A TARGET THAT NEEDS THEM: 19 A TO 0.2 A

6MRR was the wrong test. A 68-residue designed protein folds to 0.5-1.7 A from
its sequence alone, so a perfect template has almost nothing to add and the
whole experiment fits inside the noise. **5CAJ, 255 residues, a natural
protein**, is the test: single-sequence fails outright and the template has to
carry the fold. `fold-opendde.js --target=5caj --chain=A --steps=16
--template=/tools/fixtures/5caj-crystal.pdb:A`, no MSA:

**All five, matched settings** - 16 flow steps, `--recycles=3`, no MSA:

| model | no template | + self-template |
|---|---:|---:|
| alphafold3 | 16.523 A / TM 0.258 | **0.258 / 0.998** |
| openbind0 | 29.606 / TM 0.079 | **0.224 / 0.999** |
| boltz2 | 17.461 / TM 0.239 | **0.470 / 0.994** |
| protenix2 | 19.192 / TM 0.198 | **0.164 / 0.999** |
| opendde | 20.496 / TM 0.156 | **0.317 / 0.997** |

**Every model now takes a template and every one of them lands on the crystal**,
from folds that are not folds at all (TM 0.08-0.26). Two of the five could not
take one at the start of this session and a third was silently corrupt.

🔴 **AND openbind0's "TEMPLATES BARELY HELP" WAS THE TARGET, NOT A DEFECT.** On
6MRR it reads 1.743 -> 1.579 and looked like the weak one of the three that ran;
on 5CAJ it is 29.606 -> 0.224, the largest rescue of the five. A 68-residue
designed protein cannot distinguish a working template embedder from a broken
one, and it was about to be written up as a suspect.

**That is what a working template looks like** - a fold that is not a fold at
all (TM 0.18-0.26) becoming the crystal. And it is the gate 6MRR could not be:
on 6MRR every one of these lands between 0.47 and 0.53 whether the featuriser
is right or wrong, which is exactly how boltz2's shipped without one.

### 🔴 boltz2's TEMPLATE SCRATCH WAS HALF THE WIDTH ITS ATTENTION WRITES

Found by the target above and by nothing else. boltz2's self-template fold of
5CAJ read **3.859 A** where AF3 and protenix2 reached 0.2, and on 6MRR it read
0.490 against its own 0.507 baseline - indistinguishable from correct.

**The grid projection writes `heads * dimension` per pair per role, and the
template stack sized its scratch by the CHANNEL count.** In every other stack in
every model those are the same number:

| stack | heads x dim | channels | |
|---|---|---|---|
| AF3 trunk | 4 x 32 = 128 | 128 | equal |
| boltz2 trunk | 4 x 32 = 128 | 128 | equal |
| protenix2 trunk | 8 x 32 = 256 | 256 | equal |
| OpenDDE trunk | 12 x 32 = 384 | 384 | equal |
| AF3 template | 4 x 16 = 64 | 64 | equal |
| protenix2 template | 2 x 32 = 64 | 64 | equal |
| **boltz2 template** | **4 x 32 = 128** | **64** | **WIDER** |

boltz2's template attention is the only place anywhere that the two differ, so
the buffer held half of what the projection wrote and the kernel's own bounds
check dropped the tail. Nothing raises; the fold is plausible.

The bisect, each step a seam that had to be built:

| | boltz2 | protenix2 (control) |
|---|---:|---:|
| the 109/108 columns, CPU forward vs the oracle | 8.25e-7 | 1.54e-7 |
| the EMBED alone, GPU vs CPU | 2.2e-7 | 1.6e-7 |
| the whole module, GPU vs CPU | **0.748** | 3.9e-5 |
| ...after sizing scratch by the attention width | **5.11e-4** | 3.9e-5 |

The embed being exact either side is what said it was the two pairformer blocks
and not the projection or the features. Ruled out on the way, each measured:
the outer residual (forced off on BOTH sides the GPU still disagreed by 0.756),
the head count (`gridHeads` is read off `blocks[0].pairAttention1.heads`), and
the pipeline key (it already carries `fused<width>` and `:or`).

🔴 **AND THE 5.11e-4 THAT REMAINED IS PRECISION, MEASURED THROUGH AN ARM THAT
COULD ACTUALLY REACH IT.** With the four pins AlphaFold 3's confidence head
carries - f32 staging, f32 weights, f32 accumulation, matrix pair kernels off -
boltz2's fused template reads **9.11e-7** against its CPU reference instead of
5.11e-4.

**Both arms used to rule precision out first were inert**, which is the third
time today: `--f16=off` cannot reach the matrix kernels, and `--tune=` reaches
nothing at all in a checker that constructs `Af3TemplateEmbedderGpu` itself -
three `--tune` arms read the identical 5.11e-4 and said only that they had not
run. The pins go through the CONSTRUCTOR, which is what `--pins` on
`check-fused-template-features.js` now sets.

🔴 **SHIPPED, ONCE THE COST WAS MEASURED INSTEAD OF ASSUMED.** "Pinning four
settings across every model's template stage has a cost nobody has measured" was
the reason not to - so it was measured, interleaved at 256 tokens over four
runs: the stage is **8.0 ms either way** against a 468 ms trunk, no arm above
8.1. It is 1.7% of a trunk and its output is added to z, so it is paid once and
inherited by all 48 pairformer blocks.

And it was worth more than boltz2. Every model's template seam against
af3-any-model:

| `z_after_template` | before the pins | after |
|---|---:|---:|
| openbind0 | 4.07e-6 | **1.24e-7** |
| protenix2 | 2.76e-6 | **9.68e-8** |
| opendde | 3.80e-6 | **8.60e-8** |
| alphafold3 | 1.17e-4 | 1.17e-4 (its own dump's bf16 - see above) |
| boltz2 | 5.05e-8 | 5.05e-8 (inert: its empty term is zero) |

Thirty-fold on three of the five, for nothing measurable. **A cost nobody has
measured is not a reason; it is a measurement nobody has taken.**

🔴 **AND THE SAME LINE IS IN THE TRUNK, WHERE IT IS LATENT.**
`pairformer-block-webgpu.js` sizes its scratch by `pairChannels` too, and every
trunk shipped here has `heads * dimension == channels`, so it has never been
wrong. Written the same way now, so the next checkpoint does not pay for it
twice.



### 🔴 THE FUSED TEMPLATE FEATURISER, WHICH DID NOT EXIST

Both fused models used to throw on a supplied template: "the fused template
embedder has no featuriser yet". Two featurisers, because **the two models do
not share a feature set**:

  * **protenix2's 108 are AlphaFold 3's own six features concatenated** - 39
    distogram + 1 pseudo-beta mask + 32 restype_i + 32 restype_j + 3 unit
    vector + 1 frame mask - and `templateGeometry` already computed four of
    them for the nine-projection path. Checked against af3-any-model's
    `our_features` by `tools/gpu/check-fused-template-features.js`: distogram
    **0**, both masks **0**, unit vector 3.64e-7, both restypes **0**.
  * **boltz2's 109 are a different construction**: 38 distogram bins on
    `linspace(3.25, 50.75, 37)`, a unit vector that is the element-wise SIGN of
    `R_j^T (ca_i - t_j)` (Boltz normalises along a size-1 axis, so the division
    is by `abs()` per component - a quirk the trained weights depend on), a
    restype vocabulary shifted by two over 33 classes, and its own frame from
    the side-chain table's group 0.

🔴 **AND `restype_i` VARIES ALONG j IN protenix2 AND ALONG i IN boltz2.** The
name says which index the tensor varies along in each vendor's own naming, not
which one selects its value. Built the other way protenix2's two restype
columns score **1.36** and the fold still looks plausible. The aatype needs no
remap for protenix2 - the table recovered from the dump is the identity, which
is worth stating because docs/AF3.md's "32-class remap" reads as though it does.

🔴 **AND THE DUMP DID NOT RECORD ITS OWN INPUT.** `dump_af3_template.py` wrote
the 108 columns and the module's output and not the STRUCTURE they came from, so
the only thing checkable was the forward - which is the half that was already
right. It records `template_aatype`, `template_atom_positions` and
`template_atom_mask` now. **A featuriser needs its input recorded beside its
output to be checkable at all.**

### 🔴 DO TEMPLATES ACTUALLY WORK? TWO OF THE FIVE THREW

The sharpest functional test there is, and nothing had ever run it: fold the
target from its OWN crystal as a template, with NO alignment. If templates work
at all, that fold must land on the crystal. 6MRR, 16 flow steps, int5 (openbind0
f32, the only one on this box), `fold-opendde.js --target=6mrr --template=`:

| model | no MSA, no template | + self-template | |
|---|---:|---:|---|
| alphafold3 | 0.72 A | **0.50** | works |
| opendde | 1.542 | **0.531** | works - the largest gain of the five |
| openbind0 | 1.743 | 1.579 | **barely moves**, on the same nine-projection embedder |
| boltz2 | 0.507 | **THROWS** | not implemented |
| protenix2 | 0.714 | **THROWS** | not implemented |

🔴 **boltz2 AND protenix2 CANNOT TAKE A TEMPLATE AT ALL**: "the fused template
embedder has no featuriser yet: a supplied template needs the 108 columns
built". The fused embedder's FORWARD is exact - `check-af3-template-fused.js`
reads 1.52e-7 with a real template, because the dump hands it the 108 columns -
and the featuriser that would build them from a structure was never written. So
the gate that exists passes and the feature does not exist, which is the
cleanest example in this repository of a checker measuring the half that works.

🔴 **AND NO CLI GATE HAD EVER SUPPLIED A TEMPLATE.** Every template number in
these docs before this one is an EMPTY slot; `fold-in-page.py --template` drives
the page and reports pLDDT, which is not a correctness gate. `fold-opendde.js
--template=<pdb>:<chain>` builds the slot through `buildTemplate`, the same
function the page uses, and reports RMSD against the crystal.

**openbind0 is the open question of the three that run.** Its embedder is
AlphaFold 3's, its empty-template seam is exact (4.07e-6), and a 100%-coverage
self-template moves it from 1.743 to 1.579 where AF3 goes 0.72 to 0.50 and
OpenDDE 1.542 to 0.531. A template that good should dominate the prediction.

### 🔴 THE WHOLE TRUNK, ALL FIVE FAMILIES, ON EACH ONE'S OWN REFERENCE BATCH

The sweep that answers "what else is not exact". `fold.js --dump= --trunk-oracle=`
(and `fold-opendde.js` for OpenDDE), f32 bundles, 6MRR, one pass, so the
featuriser and the conformers are out of the comparison and what is left is the
graph:

| family | `target_feat` | `z_init` | `z_after_template` | `z_after_msa` | `pair` |
|---|---:|---:|---:|---:|---:|
| alphafold3 | 5.76e-8 | 2.20e-4 * | 1.17e-4 * | 1.16e-4 | 2.98e-4 |
| openbind0 | 4.93e-8 | 3.65e-8 | 1.24e-7 | 6.03e-5 | 4.20e-4 |
| boltz2 | 9.71e-8 | 5.05e-8 | 5.05e-8 | 4.55e-5 | 3.49e-4 |
| protenix2 | 2.37e-8 | 1.89e-8 | 9.68e-8 | 5.40e-5 | 1.38e-3 |
| opendde | 3.14e-8 | 4.70e-8 | 8.60e-8 | 2.65e-5 | 7.71e-4 |

\* AlphaFold 3's two are its DUMP's bfloat16, not this port's - see below; every
other cell is the port's.

🔴 **RE-MEASURED 2026-09-14, AND THE FIRST VERSION OF THIS TABLE IS WHY.** As
first published it read protenix2's template seam at 3.89e-3 and OpenDDE's MSA
at 6.01e-2, and both were fixed the same day - a padded template slot carrying
restype ZERO and the outer product mean computing 256 of 384 channels - along
with a template precision pin worth thirty-fold on three of the five. **A table
of measurements is stale the moment the thing it measures is fixed**, and this
one sat wrong for several commits inside the same document that described the
fixes.

**Every atom encoder is exact** - `target_feat` is 1e-8 for all five, which is
the one row of this table with no exception in it.

**And nothing is left above 1.4e-3.** All three of the things this section
originally listed are closed - OpenDDE's outer product mean, protenix2's template
stage, and AlphaFold 3's `z_init`, which turned out to be the oracle's own
precision rather than this port's. What follows is the record of each; the
ordering below is the state on the morning of 2026-09-14, not now:

1. **OpenDDE's outer product mean, 6.01e-2.** Bisected to that one module above;
   the MSA embedding and the MSA update either side of it are exact.
2. ~~**protenix2's template stage, 3.89e-3**~~ - **FIXED: a padded template
   slot carries restype ZERO, and a one-hot of zero is a ONE.** protenix2's
   featuriser fills its one empty template with the GAP restype and zero-pads
   the rest - and `template_aatype = 0` is zero-padding of the AATYPE, not of
   the feature: `one_hot(0, 32)` sets restype column 0. So the four slots the
   trunk runs are `[gap, restype-0, restype-0, restype-0]`, not four gaps and
   not `[gap, 0, 0, 0]`. Measured against af3-any-model's own
   `evoformer/template_embedding` on 6MRR, which has no template:

   | four empty slots built as | template term | `z_after_template` |
   |---|---:|---:|
   | four gaps (shipped) | 7.39e-3 | 3.89e-3 |
   | gap then all-ZERO (tried first) | worse | 4.55e-3 |
   | **gap then restype-0** | **5.26e-6** | **2.76e-6** |

   protenix2's whole trunk follows: `z_after_msa` 3.42e-3 -> **5.38e-5** and the
   pair 3.98e-3 -> **1.39e-3**. The other four are unmoved to every digit, and
   `--template` on 5CAJ still reads 0.165 A.

   🔴 **AND IT WAS BUILT WRONG TWICE BEFORE THE BATCH WAS READ CAREFULLY
   ENOUGH.** Gating the gap to the first slot was the obvious symmetry with
   OpenDDE's fix, it made the seam WORSE, and that was recorded as evidence the
   fused featuriser sets the gap everywhere. It was evidence of nothing except
   that all-zero is not what a padded slot holds.

3. ~~**AlphaFold 3's own `z_init_generic`, 2.20e-4**~~ - **NOT A DEFECT IN THIS
   PORT; it is the dump.** See below: the reference's own two implementations,
   run fresh, agree with this port at relRMS 0 and differ from the captured
   scope by the same 1.255e-3. Originally read as - four orders worse than
   every other family's, on the model this port was written against first. The
   template stage HALVES the relative error (1.17e-4) because it roughly doubles
   the pair's magnitude while adding an exact term, which says the error is
   entirely in the z init and nothing downstream adds to it. **Localised to one
   term and NOT yet explained; see below.**

#### 🔴 AF3's z_init IS THE RELATIVE ENCODING, AND EVERY TEST SAYS OUR SIDE IS RIGHT

`tools/gpu/check-af3-embedder-terms.js`. The pair init is four summands and
three of them are exact:

| term | relRMS |
|---|---:|
| `single_activations` | 4.61e-8 |
| `left_single` | 4.96e-8 |
| `right_single` | 4.56e-8 |
| **`position_activations`** | **1.26e-3** |
| `bond_embedding` | rms 0.0000 on both sides - a protein has no covalent links |

So it is the relative encoding alone. What has been ruled out, each with a
measurement rather than a reading:

  * **Not the formulas.** `featurization.relative_encoding_segments` clips
    `offset + 32` to [0, 64] with 65 for a different chain, the token block the
    same with 65 off-residue, the chain block keyed on ENTITY (AF3 is not in
    `CHAIN_BUCKET_ON_SAME_CHAIN`, which is ESMFold2's), and `same_entity` as one
    scalar column - 66 + 66 + 1 + 6 = 139, in that order. This port's
    `relativeEncoding` is the same arithmetic in the same order.
  * **Not the indices.** Subtracting the entity and chain rows from the NATIVE
    tensor and brute-forcing which (pos, token) rows explain the remainder
    recovers exactly the indices we pick, on every pair tried.
  * **Not our projection.** The reference's own `_RelativeEncodingProjection` is
    `w_pos[i] + w_token[j] + entity * w_entity + w_chain[k]`, and our value
    equals that four-row gather at relRMS **0**.
  * **Not the weight.** `check-bundle-vs-params.py` agrees 404 of 404.
  * **Not a bias.** The bundle carries no `position_activations/bias`, and the
    module takes none.
  * **Not bfloat16**, which the magnitude suggested: the dumper sets
    `bfloat16 = "none"`, and rounding the weights, the output, or the running
    sum all score WORSE than plain f32 (1.26e-3 f32; 1.26e-3, 2.09e-3, 2.12e-3).

🔴 **ANSWERED: IT IS THE DUMP, NOT THE PORT.** The reference has TWO
implementations - `create_relative_encoding` builds the (L, L, 139) one-hot and
contracts it, `relative_encoding_segments` returns indices for a gather - and
run fresh from the checkpoint on this batch they agree with each OTHER at
relRMS **0** and land on rms **2.542316**, which is THIS PORT's value. The
captured scope is rms 2.542363, and the reference's own code differs from its
own captured scope by **1.255e-3** - the same number this port scores. So:

| | rms | vs the captured scope |
|---|---:|---:|
| af3-any-model, one-hot path, fresh | 2.542316 | 1.255e-3 |
| af3-any-model, gather path, fresh | 2.542316 | 1.255e-3 |
| this port | 2.542316 | 1.255e-3 |

**The port agrees with the reference's code exactly and the DUMP is the
outlier.** So AF3's `z_init_generic` 2.20e-4, and the 1.17e-4 / 1.16e-4 /
2.96e-4 that follow from it in the sweep above, are a property of that dump and
NOT of this port; AF3's true trunk residual is unmeasured and smaller.

🔴 **AND IT IS NOT THE FILE.** `RELENC=1` on `dump_af3_trunk_taps.py` does the
comparison IN ONE PROCESS - the captured scope against a fresh gather of the
same parameters, with the JSON round trip, the file and the `BLOCKS=` slicing
all removed. Still **1.255e-3**, `calls 1`. What is left is arithmetic inside
af3-any-model's own module:

  * the checkpoint's `position_activations` is **bfloat16** (139 x 128), and a
    fresh gather upcasts it and sums in f64;
  * the captured values and the fresh ones both sit on a 2^-10 grid and differ
    by a few ULPs - `1.720703` against `1.717773`, max |d| 0.03125;
  * casting the fresh result to float16 scores **1.268e-3** against the observed
    1.255e-3, within one percent, and bfloat16 scores 2.09e-3.

So the captured scope is that gather at reduced precision, and this port
computes it at f32 and matches the exact one. **The remaining question is about
the reference's internals and not about anything this port ships**, which is
where it is being left.

**The lesson is the one this file keeps paying for: an oracle is a measurement
too.** Three of the four "not exact" cells found this session turned out to be
the instrument - two of my own checkers and now the reference's own dump - and
each was found the same way, by a residual that could not be true alongside
another one.

`z_after_template` is 5.05e-8 for boltz2 because its whole empty-template term
is ZERO under `templateVisibilityByCoverage` - the seam equals `z_init`, so that
cell is not evidence its template embedder is right, only that it is inert with
no template supplied.

### 🔴 AND THE CONFORMERS ARE NOT CENTRED, WHICH IS FIVE OF THE SIX FAMILIES

Found while chasing the above. `CENTRE_REF_CONFORMERS` in af3-any-model is
`('boltz2', 'openfold3', 'openbind0', protenix*, 'opendde')` - everything except
stock AlphaFold 3, which is the reference implementation, and intellifold2,
which passes `centering=False`. Each of their featurisers subtracts the group
mean per `ref_space_uid`; this port did not. Measured: glycine's four atoms mean
to exactly (0, 0, 0) in the reference's batch and to (1.31, -0.02, 0.58) here.

It moves the RAW `ref_pos` channel only - the atom encoder also reads a
translation-invariant pairwise difference - which is why half the module could
not see it. What it was worth, on openbind0's sequence-featurised `target_feat`:
**2.89e-2 -> 2.47e-2**. So it is real and it is NOT most of the conformer floor;
the shared idealised geometry is. Folds are unchanged within the seed spread
(boltz2 6MRR 0.507 A / TM 0.976 against 0.537 / 0.973; opendde 1.542 / 0.920
against 1.525 / 0.933).

**OpenDDE's confidence head**, on the reference's own seeded inputs - FOUND
WRONG, FIXED BY THE M2 IN 8dbb8cd, AND CONFIRMED HERE:

| readout | before | after (A100) | after (M2) |
|---|---:|---:|---:|
| `predicted_lddt` | 1.42e-4 | **1.14e-7** | 1.18e-7 |
| `predicted_experimentally_resolved` | 1.94e-4 | **9.53e-8** | - |
| `full_pae` | 4.68e-3 | **8.46e-7** | 9.11e-7 |
| `full_pde` | 7.50e-3 | **9.73e-7** | 1.20e-6 |

The head ran at the TRUNK's precision: `Af3ConfidenceHeadGpu` pins f32 staging,
f32 weights, f32 accumulation and the matrix pair kernels OFF, because pLDDT and
PAE are softmaxes over 50 and 64 bins - the most amplifying thing the model
emits - and this stack took only the bundle's weight precision. Four orders.

🔴 **AND I CALLED IT "NOT PRECISION" ON THE STRENGTH OF A CONTROL THAT COULD NOT
VARY PRECISION.** The arm was `--f16=off`, which **cannot reach the matrix
kernels**: on a device with matrix units the flag moves nothing, so an unchanged
residual reads as proof that precision is not the cause and is proof of nothing
at all. The M2 - which has no matrix units at these widths - saw the same flag
remove the whole error. **A control arm that cannot vary the thing under test is
not a control**, and "identical with the flag off" must be read as "the flag
reached this code path" first. The checker's bound is 1e-5 now, verified to fail
with the pins removed; leaving it at the defect's own 1e-2 would have let the
defect back in silently.

**And OpenDDE's target-feat atom encoder is exact**, which is what isolates the
trunk defect to the template stage alone. `check-opendde-encoder-oracle.js
--dump=`, masked to the atoms both sides call real:

| seam | relRMS |
|---|---:|
| per-atom conditioning (5 embeds summed) | 3.90e-8 |
| `pair_mlp_3` | **0** |
| `atom_transformer_encoder` | 1.18e-7 |
| `project_atom_features_for_aggr` | 1.79e-7 |

🔴 **THE MASK IS NOT A TOLERANCE, IT IS THE COMPARISON.** Unmasked, those arms
read 0.466 and 0.137 while `project_atom_features_for_aggr` - DOWNSTREAM of both
- read 1.79e-7, which cannot be true. The reference's per-atom embeddings are
nonzero on all 1632 dense slots (`embed_ref_element` and `embed_ref_atom_name`
embed index-0 one-hots, which are real vectors) and this port zeroes them; both
are masked out at `mask_mean`, so the fold is identical and the unmasked residual
is dominated by rows neither model reads. That is the second impossible pair of
residuals in this one checker, and both times the checker was at fault.

🔴 **AND THE ENCODER BISECT CAUGHT ITS OWN AUTHOR FIRST.**
`check-opendde-encoder-oracle.js` reported the per-atom conditioning at 1.19 -
near-orthogonal to the reference - while `target_feat` was exact in 439 of its
447 columns. Both cannot be true, and that contradiction is what found the bug:
the checker passed `atomReference(store)`, the DIFFUSION head's reference table,
where `buildTargetFeat` reads `targetFeatureWeights(store).reference`. With the
right weights the conditioning came back to 0.26 and then, on the reference's
batch, to exact. **A residual that is impossible given another residual is
evidence about the checker, not about the port.**

### What the two new models COST, and the 5.4x that was hiding in a presence test

68 tokens, int5 bundles, 200 diffusion steps, warm fold (the second of two):

| | trunk | diffusion | total | peak device | trunk at 256 tokens |
|---|---:|---:|---:|---:|---:|
| alphafold3 | 0.3 s | 2.7 s | **3.1 s** | 983 MiB | 917 ms |
| **boltz2** | 0.4 | 4.7 | **5.2** | **1535** | 1044 |
| **protenix2** | 0.6 | 2.8 | **3.5** | 1297 | 1962 |

🔴 **AND THIS TABLE DOES NOT SAY WHICH MACHINE, WHICH IS ITS DEFECT - THE boltz2
ROW IS NOT THE A100'S.** Re-measured there on 2026-09-15 at `4df2b99`, the
commit this table was written in, AND at HEAD 127 commits later, identical at
both: alphafold3 **992.6 MiB / 3.138 s**, boltz2 **1399.1 / 3.69**, protenix2
**1220.3 / 3.357**. So nothing drifted - AlphaFold 3's row is essentially that
box (983 against 992.6, 3.1 against 3.138) and boltz2's is 9.7% under on memory
and 41% under on time. Its breakdown below names "204 of MSA scratch" as the
third row; on the A100 the third row is `difftx.zerogate.resident` at 162 MiB in
**every** arm that can be run - from a sequence, from the dump, at 50 steps, at
200, and with a real 8076-row alignment. A memory figure is a property of the
machine exactly as a checksum is. **The whole panel, measured in one sitting on
the A100 with the controls for what does and does not move a peak, is in
docs/A100.md.**

boltz2 is 1.7x AlphaFold 3 for 1.6x the device memory, which is what its shape
costs: 64 pairformer blocks against 48, an 8-block confidence stack against 4,
and a token transformer carrying a third projection per block. Its peak is 513
MiB of resident diffusion-transformer blocks, 270 of trunk single transitions
and 204 of MSA scratch. protenix2's trunk is 2.1x AF3's at 256 tokens - its pair
track is 256 channels wide against 128 - while its diffusion is the same.

🔴 **AND boltz2's SAMPLER WAS 193 ms A STEP BEFORE THIS, AGAINST AlphaFold 3's
13.** Not a leak - flat from 25 steps to 200 - and not arithmetic: the GPU was
**92% IDLE** through a denoiser call, 21 ms of work in a 267 ms span, with the
same ten submits and the same 309 passes AF3 has. The head's own stage timers
put 9.1 s of a 10.2 s 50-step fold in the TOKEN TRANSFORMER stage, whose GPU
kernels are under 2 ms.

It was the presence test for the up-gate. `txHasUpGate(block)` read
`block.ffwAToB != null` - and a bound block's fields are THUNKS that decode when
read, so asking whether the tensor exists unpacked a 768x1536 int5 tensor, once
per block per sampler step. Asking the SOURCES map instead is the same answer
for free:

    boltz2, 50 steps    diffusion 10.2 s -> 1.9 s
    boltz2, 200 steps   diffusion 38.6 s -> 4.7 s

Three tests in this session had the same shape - `blockHasUpGate` in the atom
stacks and `hasBondTypes` in the embedder - and all three now ask the thunk.
It is CLAUDE.md's own note about `blockWeightOffsets` reading `.length`, one
convention later: **a bound weight field is not a value, and `!= null` on one is
a decode.**

🔴 **AND `bench-trunk.js` COULD NOT MEASURE EITHER MODEL** until this: it passed
the imported `DIALECT` constant rather than the bundle's, so pointing it at
boltz2 or protenix2 died in `emptyFusedFeatures` before producing a number. Same
fault docs/PARITY.md records across the checkers.

## The batch, all of it, against the reference's - and the four conventions no plain protein reaches

`npm run test:batch`. Previously `check-atom-windows.js` compared two fields of
the reference's sixty - `queries_to_keys` and rf3's chiral centres - and those
two alone found the atom key window wrong in four of seven models, the terminal
OXT/OP3 carried by four families that drop it, and a chirality term twice
recorded here as a no-op. The other fifty-eight were being reasoned about.

`tools/check-batch-fields.js` compares 43 of them (108 for opendde, which
carries its own second token space) for every dumped model. **14 model/target
pairs exact**: every gather, `aatype`, `profile`, `ref_element`, `ref_charge`,
`ref_atom_name_chars`, `ref_space_uid`, the masks and the bond sets.

### 🔴 The four conventions 6MRR cannot see, and all four were missing

6MRR is a plain 68-residue protein. It has no ligand and no modified residue, so
four entries in the dialect table were inert in **every measurement this port
has ever taken** - and four of them had never been implemented. Two belong to
models that ship.

`tools/oracle/dump_af3_batch.py --ligand GOL --ptm SEP@3` builds the target that
can: 83 tokens, the phosphoserine contributing ten and the glycerol six, through
`_fold_setup(chains=...)` so the conventions still come from the reference.

| convention | who | what |
|---|---|---|
| `atomizedElementNames` | rf3 | an atomised atom is renamed to its ELEMENT. "CA", "CB", "OG", "O1P" become "C", "C", "O", "O"; glycerol's "C1" becomes "C" |
| `atomizedUnknownRestype` | boltz2, rf3 | an atomised token's aatype is UNKNOWN (20), not the parent residue's (15, serine) |
| `atomizedUnknownMsa` | **rf3 alone** | ...and that carries into the ALIGNMENT, and so into the profile |
| `atomizedBackboneBonds` | **rf3 alone** | the atomised residue is bonded back into the chain |

**The third exists because the gate refused the second.** Implemented as one
flag, boltz2's profile came out 20 where the reference has 15. The references'
own MSA query rows at the phosphoserine's tokens settle it:

    alphafold3    15 15 15 ...    aatype 15
    boltz2        15 15 15 ...    aatype 20
    rosettafold3  20 20 20 ...    aatype 20

boltz2 moves the restype and leaves the alignment holding the parent; rf3 moves
both. One tuple in the reference, two behaviours here.

**The fourth is rf3's alone, and counting is what showed it.** The reference
lists **14** bonded token pairs for six families and **18** for rf3. The four
extra are `1-2 2-1 6-12 12-6` - the peptide bonds either side of the
phosphoserine, each way round, where the nine internal SEP bonds appear once
each. featurise.js's own comment said backbone connectivity is "left implicit in
residue_index, exactly as for an unmodified chain", which is right for AF3 and
six others and wrong for the seventh: without them the modified residue is a
ligand floating beside the chain as far as the pair track is concerned.

**All four are provably inert on a plain protein**, which is why nothing that
shipped moves: featurising 6MRR with them on and off, for all seven models,
gives **0 differing elements across 20 array fields**. That is a stronger
statement than a fold comparison on a box that drifts 3.2x.

### 🔴 Three things the gate had to be corrected on before it could say any of that

**1. Per-atom fields are compared only where `ref_mask` is live on BOTH sides.**
The first run reported `ref_element` differing in one slot of 1632 in exactly
the four families with `dropTerminalAtoms`, and `ref_atom_name_chars` in three
of 6528 - one atom's name. It reads as a defect in four shipped models. It is
the C-terminal OXT: the reference drops it by MASKING it while leaving its
element (8), its name and its CCD position, where this port never creates the
atom and leaves the slot zeroed. Both then compute a per-atom conditioning row
for it - `rows = tokens * dense`, and the element embedding indexes its weight
table by atomic number with no mask, so we add row 0 where they add row 8 - and
**no gather on either side references that slot with a live mask**, checked for
all four. A padded slot the reference fills and this port zeroes: the same shape
as docs/OPENDDE.md's `mask_mean` note and the encoder checker corrected before
it. **Third occurrence.**

**2. `ref_pos` is reported, never failed**, or the gate is red forever and
nobody reads it. It is the shared ideal conformer set, which is a deliberate
decision, and 485 live slots of 1722 differ by up to 9.66 A. That number is
partly meaningless - an idealised conformer has an ARBITRARY rigid frame - so
the frame-free arm asks the other half. Intra-token pairwise distances, which
are invariant to rotation and translation:

    rms 0.6483-0.6517 A, worst 3.7679 A, on token 18 of 6MRR - a LYSINE

The longest, most flexible side chain in the sequence. So the two sets are the
same molecules in a different **rotamer** and a different frame, not wrong
geometry. Neither number says that alone, and the elementwise one stays in the
output because it is what `embed_ref_pos` actually consumes.

**3. The bond comparison was wrong twice.** First it asked "does this port build
this field" and answered no for all seven, because the reference carries bonds
as a pair list and this port as a dense matrix - representation, not content,
and six of the seven had every bond. Then it compared UNDIRECTED edges, which
throws away `symmetriseBonds`, the convention the field exists for. Measured:
AlphaFold 3's **directed** set is this port's exactly, 14 against 14 with
nothing either way, and each symmetrising dialect has exactly 14 more, one per
internal bond reversed. The expectation is therefore the reference's own pairs
plus their reverses where the dialect symmetrises - exact, and `symmetriseBonds`
became observable, which it had not been.

### 🔴 And it goes through `af3BatchFromA3m`, which is the half that catches a caller

src/af3/featurise/batch.js forwards the dialect to `featuriseProtein` **field by field**.
A gate that calls the featuriser directly would stay green while the page and
every fold tool silently dropped a convention. Verified by deleting
`atomizedBackboneBonds` from that forwarding: the gate goes red with "4 bonds
the reference has and this port does not".

Which is why `featuriserDialect(dialect)` exists now. `tools/gpu/fold.js`,
`tools/gpu/fold-opendde.js` and `web/af3-model.js` each listed those fields by
hand - the allow-list shape CLAUDE.md records twice as having shipped a bug,
once taking the contact overlay off the page and once running multimer weights
on the monomer graph. Adding four conventions would have made it four places to
forget. All three forward the object; `test/af3-dialect.test.js` asserts it
carries every featuriser convention and none of the graph ones, because a
featuriser has no business with `noResidual`.

### What it can still not see

- **`dedupeSelfMsa`**, the tenth convention, is the one `--falsify` leaves
  green. It decides whether the query appears twice in the alignment, and the
  alignment is the one input this port takes from its caller rather than from
  the reference, so `msa` is not compared. Its evidence is elsewhere (AF3 6MRR
  83.084 -> 83.169).
- **opendde's second token space** - 65 `struct/` and `structbook/` fields that
  this port DOES build, in src/af3/featurise/structural-tokens.js, and that nothing
  compares against the reference. Named in the output rather than silently
  absent.

### 🔴 `restype_alignment`: declared by rosettafold3, and measured INERT in the reference

The last of rf3's named featuriser conventions, and the reason this port matches
the reference without implementing it is not that it does not matter - it is
that **the reference's own implementation cannot fire on a batch it produces.**

What it is meant to do (`chiral_features.apply_restype_alignment_on_atomized`,
rf3 only): find every token with no alignment of its own and replace its
`profile` with a one-hot of its restype and its `msa` column with that restype.
Their docstring is emphatic about why it matters - `profile` is 31 of the 449
columns of rf3's `s_inputs`, which feeds `to_s_init`, both z-init projections
and the MSA embedder - and about the predicate:

> BOTH channels use the SAME test: a column is rewritten only when EVERY
> alignment row in it is a gap, which cannot happen for a polymer because row 0
> is the query sequence itself.

The earlier, weaker test was `profile.argmax != aatype`, which fired on real
polymer columns and cost "~4 A of backbone accuracy" on 1STP with a 2144-row
alignment, making rf3 **worse** with an MSA than without one. So the current
predicate is a deliberate fix.

**And the fix is unreachable.** The batch's MSA is padded to 16384 rows, and the
padding value is **0**, not the gap index 21. `(msa == 21).all(axis=0)` is
therefore false for every column of every batch. Measured on three of the
reference's own dumps:

| dump | rows | columns that are GAP in every row |
|---|---:|---:|
| rosettafold3 6MRR | 16384 | **0** |
| rosettafold3 6MRR + GOL + SEP@3 | 16384 | **0** |
| boltz2 6MRR + GOL + SEP@3 | 16384 | **0** |

And the consequence is visible in the dump directly rather than inferred: at a
LIGAND token, which is exactly what the convention is for, rf3's own batch has
`profile` one-hot at column **21** (gap) with `aatype` **20** (unknown). Had the
rewrite fired, the profile would be one-hot at 20. `gap_idx = 21` confirmed
against `residue_names.POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP` on the
reference itself.

**So this port does not implement it, and that is why the batch is exact.**
Implementing the INTENDED behaviour would diverge from the reference this port
is held to. What would change that: the reference padding its MSA with the gap
index, or testing the predicate against `msa_mask` rather than every padded row.
If either happens, rf3 needs this and `tools/check-batch-fields.js` will say so
the moment a re-dumped batch disagrees.

🔴 **AND THIS GATE COULD NOT HAVE FOUND IT ANYWAY**, which is the second entry
in its "cannot see" list beside `dedupeSelfMsa` and for the same reason: the
convention only bites when there IS an alignment, and the alignment is the one
input this port takes from its caller rather than from the reference, so `msa`
is not compared. It was found by reading the reference after the gate had run
out of things to say - not by the gate.

## The ligand path: nothing was running it, and it was broken in three ways

`npm run test:ligand`. A 68-residue protein plus GLYCEROL through all six
AF3-lineage models, asserting the ligand's own five bond lengths.

Every other fold gate in this repository folds a plain protein. So the whole
atomised-token half of the featuriser - four dialect conventions, the bond
matrix, the bond orders and the atom names - was exercised by nothing, and three
separate defects were sitting in it. Two of the three are in models that ship.

| | | |
|---|---|---|
| **boltz2 tore the ligand apart** | bond rms **3.602 A**, C1-O1 at **6.97** against a 1.43 ideal | pLDDT read **92.38** |
| **rosettafold3 died outright** | `RangeError: invalid allocation size 0 for atom.chiral.centers` | glycerol has no stereocentre |
| **rf3's PDB had six atoms named C, O, C, O, C, O in one residue** | `atomizedElementNames` reached the output as well as the model | not unique, which the format does not allow |

### boltz2's was two halves, and neither showed alone

boltz2's z-init reads TWO planes - the contact flag and the bond ORDER - and it
is the only family with `tokenBondsTypeEmbed`, so it is the only one that
notices a missing one.

**Half one**: the featuriser never built `bondOrderMatrix`. Five consumers read
it - `embedder-webgpu`, `embedder-reference`, `confidence-webgpu`,
`confidence-reference`, and `fold.js` forwarding it to the confidence head - and
nothing produced it. The order is in the component's own bond table and
`parseCcdComponent` has always returned it: a channel parsed, forwarded, and
never filled.

**Half two**: `fold.js`'s embedder-input literal did not name it. So fixing the
featuriser alone changed the fold by **exactly nothing** - byte-identical bond
lengths, 6.97 and 2.14 and 6.41 again - because the plane was still arriving as
zeros from the caller.

🔴 **And the comment two lines above that literal is about this exact trap:**

    // 🔴 NAMED, BECAUSE THIS OBJECT IS BUILT FIELD BY FIELD. A key the batch
    // carries and this literal does not name is a key thrown away here, and
    // the embedder cannot tell that from a fold with no ligand: both arrive
    // as `undefined` and both fall back to zeros. That is how the whole bond
    // feature came to be computed, shipped and never applied.

It is about `bondMatrix`, which was fixed. `bondOrderMatrix` was added to the
consumers beside it and never to this list. Fourth instance of the shape in one
session, after the three `featuriserDialect` call sites, and the first where the
warning was already written at the site.

    boltz2   bond rms 3.602 -> 0.062 A   C1-O1 6.97 -> 1.43   pLDDT 92.38 -> 92.63

All six now: af3 0.050, protenix2 0.044, if2 0.055, openbind0 0.058, boltz2
0.062, rf3 0.069.

### Why the gate has the shape it has

- **It asserts bond LENGTHS, because nothing else can see them.** The fold's
  RMSD is dominated by 68 residues of protein; `meanPlddt` said 92 on a ligand
  6 A out; and `tools/gpu/chain-geometry.js` measures the protein BACKBONE and
  steps over a ligand by design. A ligand that comes apart is invisible to all
  three.
- **It measures by atom ORDER, not by name.** rf3 renames an atomised atom to
  its element symbol, so a checker keying on "C1" finds nothing and reports the
  ligand MISSING - which is what the first version did, and the convention
  working correctly read as a dropped ligand.
- **It asserts the names are unique within the residue**, which is a property of
  the FILE rather than of the fold, and is the third defect above.
- **A bundle this box does not have is a SKIP.** openbind0 is f32 here and the
  rest are int5; hard-coding the suffix reported a 404 as a failure.

## rosettafold3 has no flow sampler, and the page defaulted to one

Found by running `probe-nucleic.js` on the two new models, which nothing had
done. The finding is not about nucleic acids.

**6MRR, `--mode=flow`:**

    N-CA 6.94 A (ideal 1.46)   CA-C 3.76 A (ideal 1.52)
    consecutive CA 3.07 A median, worst 0.23 A   <- collapsed
    pLDDT 81.47

against `--mode=diffusion`'s **1.693 A** and a clean backbone. AlphaFold 3
(CA-CA 3.58) and intellifold2 (3.88) both fold in flow, so it is the checkpoint
and not the sampler.

🔴 **AND pLDDT IS 81.47 AGAINST THE GOOD FOLD'S 81.53.** Four tenths. Whatever
number the page puts on screen says nothing is wrong. This is the same lesson as
the collapsed 825-residue AF2 fold whose pLDDT *rose* to 69.31, and the only
thing that catches it is `chain-geometry.js`.

🔴 **AND `index.html` HAS `<option value="flow" selected>`.** The page's sampler
select defaulted to Flow at the time and `web/app.js` forced diffusion for
OpenDDE alone, so
a visitor choosing rosettafold3 would have got that fold. **No standing gate
could see it**: they all go through `fold.js`, whose default is `diffusion`.

Three changes, and the middle one is the point:

* `noFlowSampler` in the dialect, inverse polarity like `dedupeSelfMsa` so stock
  AlphaFold 3 stays all-false.
* **`foldBatch` throws** rather than switching quietly, with the measurement in
  the message. A fold that silently ran a different sampler than the caller
  asked for is the kind of thing that gets measured for a week.
* the page hides the row *and* forces the mode through `samplerModeFor` -
  hiding a control does not change its value, which is the trap already
  recorded at that line for OpenDDE and ESMFold2.

🔴 **AND THE FIRST GUARD DID NOTHING.** It read `weights.dialect`, the name
`buildTargetFeat` uses on a different object, where `foldBatch`'s is
`weights.trunk.dialect`. So it was `undefined`, the check was a no-op, and the
broken fold reached the geometry gate exactly as before - which is how it was
noticed. **A guard that reads the wrong field is a guard that is not there**,
and that is the third time in one session a check quietly did not run.

### And the nucleic numbers it started from were the probe's fault

`probe-nucleic.js` called `featuriseProtein` with nothing but `chainKinds` and
**no dialect**, so every nucleic number this repository has recorded was
AlphaFold 3's featurisation fed to another model's weights - the same fault
`fold-opendde.js` had, where fixing it moved every published OpenDDE figure. It
matters more on a nucleic chain than anywhere: `dropTerminalAtoms` removes the
5' OP3 and the flat atom axis is built from the live atoms, so getting it wrong
shifts every index in the chain rather than one slot.

With each bundle's own conventions, DNA `ACGTACGT`, intra-residue bond ratio
against the baked conformer:

| model | ratio | worst pair |
|---|---:|---|
| af3 | 0.980 | G7 P-OP2 0.751 |
| opendde | **0.997** | T8 P-OP2 0.828 |
| intellifold2 | 0.989 | T4 C4'-C1' 1.191 |
| rosettafold3 (diffusion) | 0.997 | - |

**So "OpenDDE is 15% short", which CLAUDE.md carried as an open defect, was the
probe.** It is 0.3% short and the best of the four.

🔴 **AND I THEN READ rf3's 3.271 AS "BROKEN ON DNA", WHICH WAS ALSO WRONG.** Its
PROTEIN scores 2.976 in the same probe and **1.004** at `--mode=diffusion`: the
probe's default mode is `flow` where `fold.js`'s is `diffusion`, and that was
the whole difference. The reference's own rosettafold3 folds the 1LMB DNA duplex
at **2.18 / 2.13 A** C1' RMSD, so the model is fine on DNA. Two wrong readings
of one number before the control arm was right.

### And the schedule is NOT why - measured, so the next person need not check

The obvious explanation for rosettafold3's flow walk failing is a different
noise schedule, which would also mean this port's DIFFUSION fold was running on
a wrong constant. It is not that. Read off the reference's own configs:

    alphafold3     gamma_0 0.8  gamma_min 1.0  noise_scale 1.003  rho 7.0
                   sigma_max 160.0  sigma_min 0.0004  step_scale 1.5  steps 200
    rosettafold3   identical
    intellifold2   identical

So the eval schedule is shared, and rf3's diffusion fold is not paying for a
wrong sigma - which is consistent with it landing at 1.693 A inside the
reference's own 0.967-1.772 spread.

🔴 **WHY IT FAILS: TWO HYPOTHESES TESTED, BOTH DEAD, AND A MECHANISM LEFT.**
The flow walk is THIS PORT's shortcut - one draw at the top of the schedule then
a deterministic descent, about 8 calls instead of 200 - and not something the
reference has, so there is no oracle for it.

**Hypothesis 1, `diffusionNoResidual`: not supported.** rf3's transformer is
`act + attn + transition(act)` rather than the usual residual chain, it is the
ONLY family carrying that flag, and it is the only family without a flow walk -
a perfect correlation over seven models. Forced OFF as a diagnostic, rf3's flow
fold is still not a chain and is WORSE: N-CA 14.97 A against the flagged run's
6.94, CA median 5.136. Caveat, stated because it matters: the diagnostic also
makes the network compute something its weights were not trained for, so a
recovery could in principle be masked. Suggestive against, not decisive.

**Hypothesis 2, undersampling: refuted decisively.** If a deterministic descent
merely needed a finer discretisation, more steps would help. They make it
monotonically WORSE, on the unmodified model:

| steps | CA median | worst | pLDDT |
|---:|---:|---:|---:|
| 16 | 3.071 | 0.23 | 81.47 |
| 60 | 2.656 | 0.75 | 81.46 |
| 200 | **2.122** | 0.67 | 81.38 |

against 3.80 expected. **The walk CONTRACTS**: every extra step pulls the chain
tighter toward a point, so its fixed point for this checkpoint is a collapsed
structure rather than a fold it has not reached yet. That is a mechanism, and it
points at the SCALE of the denoiser's output rather than at the schedule (which
is identical to AF3's - measured below) or the step count.

🔴 **AND pLDDT SITS AT 81.4 THROUGH ALL OF IT**, moving 0.09 while the chain
collapses from 3.07 to 2.12. Three separate step counts, three wrecks, one
confident number.

🔴 **RE-MEASURED 2026-09-15, AND THE RADIUS OF GYRATION SAYS IT BETTER THAN THE
CA MEDIAN DOES.** The fixed point is not merely "tight", it is a POINT:

| steps | CA-CA | worst | N-CA | **Rg** | pLDDT |
|---:|---:|---:|---:|---:|---:|
| 16 | 4.54 | 79.74 | 6.30 | **19.0** | 81.54 |
| 60 | 3.19 | 10.31 | 6.76 | **7.7** | 81.47 |
| 200 | 2.10 | 0.67 | 10.87 | **2.2** | 81.37 |
| *diffusion, 16* | *3.82* | *3.72* | *1.42* | ***11.2*** | *81.53* |

68 residues inside a radius of **2.2 A** at 200 steps. The CA median at 200
reproduces the row above it to three digits - 2.10 against 2.122, pLDDT 81.37
against 81.38 - so the FIXED POINT is stable across everything that has landed
since. What moved is the early trajectory: at 16 steps it now EXPANDS first
(Rg 19.0, worst CA-CA 79.7) where it used to already be collapsing (worst 0.23).
It passes through a disordered state on its way to the point.

🔴 **AND IT IS NOT A BAD FIRST DRAW, WHICH HAD TO BE CHECKED** because this
session retracted a structurally identical "flow returns no chain" claim that
turned out to be one measurement on a dirty tree, and because `db4b74f` found
flow's risk elsewhere IS the first draw. Four seeds, 16 steps, 6MRR:

| seed | flow N-CA / CA-C / CA-CA / Rg | diffusion |
|---|---|---|
| 1 | 5.94 / 3.19 / 4.59 / 16.0 | 1.42 / 1.53 / 3.80 / 11.1 |
| 7 | 6.22 / 3.02 / 3.70 / 13.7 | 1.42 / 1.51 / 3.81 / 11.2 |
| 21 | 5.86 / 3.01 / 4.31 / 19.4 | 1.42 / 1.51 / 3.80 / 11.1 |
| 20260831 | 6.30 / 2.89 / 4.54 / 19.0 | 1.42 / 1.53 / 3.82 / 11.2 |

Four of four broken under flow, four of four clean under diffusion, and pLDDT
**81.51 to 81.56 in BOTH COLUMNS** - a range of 0.05 across folds that are a
chain and folds that are not. Deterministic, so there is no seed to find.

🔴 **AND THE SIGNATURE IS INTRA-RESIDUE, WHICH IS A LEAD NOBODY HAS FOLLOWED.**
N-CA is 4.2x its ideal and CA-C 2.0x, while CA-CA is only 1.2x - the alpha
carbons are roughly where they should be and the backbone N and C around them
are not. A uniform output-scale error would move all three by one factor. This
one does not, so "the SCALE of the denoiser's output" above is too coarse: it
points at the per-atom offsets the ATOM DECODER adds to a token position rather
than at the token walk. Untested.

What is settled is the shipping question: the failure is loud (`foldBatch`
throws), the page cannot ask for it, and the diffusion path is measured good.

### intellifold2 in flow: worse, but not broken - so the control STAYS

Measured after rosettafold3's, because the same question applies to the other
new model. 6MRR through `fold-opendde.js`, three seeds:

| seed | flow | diffusion |
|---|---|---|
| default | 1.822 A / TM 0.8589 | 1.579 A / TM 0.9241 |
| 7 | *(no result - see below)* | 1.605 A / TM 0.9240 |
| 21 | 1.962 A / TM 0.8383 | 1.553 A / TM 0.9219 |

Flow's TM is 0.838-0.859 against diffusion's 0.922-0.924, **no overlap** - the
same shape as the OpenDDE measurement that got its mode row hidden (0.831/0.860
against 0.904/0.917).

🔴 **AND THE CONTROL STAYS ANYWAY, WHICH IS THE OPPOSITE CALL FROM rf3's.**
Three cases, three answers:

* **rosettafold3**: flow is a BROKEN STRUCTURE - N-CA 6.94 A, collapsed
  backbone, refused by the geometry gate. The page must not offer it, and
  `foldBatch` throws.
* **OpenDDE**: flow is worse AND costs the same 16.1 s at sixteen steps, so
  there is nothing to trade. Row hidden.
* **intellifold2**: flow is worse and much FASTER - about 8 denoiser calls
  against 200. That is a trade a visitor is entitled to make, and the sampler
  select exists to let them. Hiding it would take away a real choice on the
  strength of a quality number alone.

So this is recorded rather than acted on. A user picking Flow for intellifold2
gets a worse fold, knowingly; a user picking it for rosettafold3 got a wreck
with a confident pLDDT, which is why only that one is refused.

🔴 **THE MISSING ARM WAS RUN, AND IT REVERSES THE PARAGRAPH ABOVE.**
`--mode=flow --seed=7` returned nothing in the sweep because **the fold is not a
chain**:

    consecutive CA median 4.255 A, worst 4.70 A, against 3.80 expected
    pLDDT says 83.30

Six seeds of flow on 6MRR: 1.822, **refused**, 1.962, 2.142, 1.988, 2.203 -
against diffusion's 1.579, 1.605, 1.553. So flow is not "worse but not broken"
for intellifold2; it is broken **one seed in six**, and the conclusion written
above with that arm missing was wrong. Waiting for it would have cost one fold.

**The control still stays, but for a reason that survives the arm.** Flow is
about 8 denoiser calls against 200, the failure is occasional rather than
certain, and - see below - the page now SAYS when it happened. rosettafold3
stays refused outright, because its flow fold is broken every time.

## The page ran no geometry check at all, for any model

The larger finding, and it is not about intellifold2. `chainGeometryVerdict`
lived in `tools/gpu/`, so **every command-line fold in this repository gated on
it and the one path a visitor takes did not.** Same shape as
`LOCALFOLD_STOCK_FLAGS`: the configuration every gate checks was not the one
that ships. A visitor folding if2 in Flow would have been handed that seed-7
structure with "pLDDT 83.3" beside it and nothing else.

* the rule is `src/af3/chain-geometry.js` now, one implementation;
* `tools/gpu/chain-geometry.js` keeps the terminal's half - the throw and the
  "--allow-broken-geometry" advice, which is not advice for a browser - and
  re-exports the rest. A test compares the function IDENTITIES, so a second copy
  cannot appear;
* `web/app.js` appends **"🔴 NOT A CHAIN - the backbone is broken, and pLDDT
  does not measure that"** to the status line.

🔴 **IT WARNS RATHER THAN REFUSING, DELIBERATELY.** The structure is still
drawn: a visitor who chose a fast sampler is entitled to see what it made, and
hiding it is worse than labelling it. What is not acceptable is showing it as
though the confidence number were the whole story - this repository's oldest
lesson, from the 825-residue AF2 fold that collapsed into a ball while pLDDT
ROSE to 69.31.

### Is the flow sampler safe for the models that already SHIP? Yes - measured

rosettafold3's flow fold is broken every time and intellifold2's one seed in
six, and the page defaulted to Flow at the time - so the obvious next question
was whether any
model already published has the same problem. It does not. Twelve folds of 6MRR
in `--mode=flow`, three seeds each, verdict by `chainGeometryVerdict`:

| model | seeds 1 / 7 / 21 | chains |
|---|---|---|
| af3 | pLDDT 77.79 / 78.04 / 78.55 | **3/3** |
| boltz2 (published) | 96.48 / 96.15 / 96.42 | **3/3** |
| protenix2 (published) | 84.67 / 84.68 / 84.68 | **3/3** |
| openbind0 | 82.22 / 82.29 / 79.50 | **3/3** |
| intellifold2 | - | 5/6, one refused |
| rosettafold3 | - | broken every time |

So the Flow default is sound for everything on Hugging Face today, and the
exposure is confined to the two models this session ported - which are not
published yet, and one of which now refuses flow outright while the other warns.

🔴 **THIS IS A NEGATIVE RESULT AND IT IS WORTH THE FORTY MINUTES.** "The
published models are probably fine" was the assumption; the reason to spend
twelve folds on it is that the page's default had just been shown to produce a
wrecked structure for one model with a confident pLDDT beside it, and the same
default has been live for boltz2 and protenix2 for weeks. An assumption about
what a visitor is getting is not a measurement of it.

### Both new models fold THROUGH THE PAGE, and the sampler fix is visible there

`tools/fold-in-page.py --model intellifold2` and `--model rosettafold3`, which
drives the real page in a real browser rather than a fold tool. This is the
check CLAUDE.md's own trap demands - "a bundle the CLI likes can be one the PAGE
cannot load", because the page reads the manifest baked into
`src/bundles/manifests/<family>.js` and not the JSON beside the shards, and
opendde once died at 122/472 MiB with every CLI gate passing.

    IntelliFold-2 · 58 residues · in 5 s · single sequence · 2 passes · pLDDT 54.5
    RoseTTAFold3  · 58 residues · in 4 s · single sequence · 2 passes · pLDDT 72.1

Both load, fold, draw, colour by pLDDT and populate the PAE and contact panels.
Neither status line carries the new "NOT A CHAIN" warning, which is the other
half of the check: the warning exists and stayed quiet on a good fold.

🔴 **AND THE FRAME NAMES PROVE THE SAMPLER DECISION, WHICH IS THE POINT:**

    rosettafold3   "diffusion_0" ... 25 frames   <- the page FORCED it
    intellifold2   "flow_0"      ... 16 frames   <- its control was kept

Before `noFlowSampler`, rosettafold3 on that page would have run the flow walk
and drawn the wreck - N-CA 6.94 A, collapsed backbone, pLDDT 81.47 beside it.
The two models get the two different decisions this session argued for, on the
one path a visitor actually takes, and nothing else moved.


## Harder problems: the first COMPLEX this side has ever scored

Every AF3-lineage number above this line is a single chain of 68 to 92
residues. `fold-opendde.js` reads ONE chain out of a crystal and folds its
sequence alone, so "does the sampler matter" had been decided on 6MRR - which
CLAUDE.md's own table calls too easy to discriminate ("a 68-residue designed
protein folds to 0.5 A from its sequence alone, so a correct template embedder
and a corrupt one land in the same place"). `tools/gpu/fold-complex.js` is the
missing instrument:

```
node tools/gpu-chrome.mjs tools/gpu/fold-complex.js --target=1brs --chains=A,D \
  --model=/model-af3-int5/manifest.json --mode=diffusion \
  --template=/tools/fixtures/1brs-crystal.pdb
```

🔴 **ONE SUPERPOSITION OVER EVERY CHAIN, AND THREE NUMBERS BESIDE IT.** Fitting
each chain separately reports two perfect chains that are nowhere near each
other as a perfect answer, which is the only failure mode a complex has that a
monomer does not. So the fit is over all chains together (`complex`), and then:

- each chain gets `inComplex` - itself in that shared frame - beside `alone`,
  re-fitted by itself. Good alone and bad in complex is a **placement** failure;
  bad both ways did not **fold**, and they want different fixes.
- `interface.fnat` is the fraction of the crystal's inter-chain alpha-carbon
  contacts (under 8 A) the model also makes. It needs no superposition at all,
  so it says nothing about the fit and everything about the interface.
- the score is taken over the best relabelling of INTERCHANGEABLE chains
  (`chainAssignments`, gated by test/chain-assignments.test.js), because a
  homodimer's two chains are the same molecule and a perfect prediction with the
  labels the other way round superposes as a total failure. AlphaFold 3 does
  this itself and calls it chain permutation alignment. `asLabelled` is the
  unpermuted score, so the difference is visible rather than hidden in a
  minimum.

### 🔴 THE FIRST TARGET WAS THE WRONG ONE, AND `nativeContacts` IS WHY THAT IS NOW LOUD

5CAJ was the obvious choice - it is already a fixture, it is 522 residues over
two chains, and this file records it as the target that separates a working
template embedder from a broken one. It produced **9.1 / 23.7 / 19.6 A across
three seeds while each chain folded to 0.28-0.50 A alone**, which reads exactly
like a placement defect, and was written up as one.

It is not. **5CAJ's A and B are two independent copies in the asymmetric unit.**
Measured on the deposition: the closest inter-chain alpha carbons are **11.08 A**
apart, there are **ZERO contacts under 8 A** (12 under 12 A, out of 68,121
pairs) and the centroids are **44.2 A** apart. Their relative placement is
crystal packing. No model in this panel predicts crystal packing, none should,
and an RMSD over the pair measures nothing - the 15 A of seed spread was the
scorer asking an unanswerable question three times.

The chain permutation was not the explanation either, and that was checked
rather than assumed: 5CAJ's dimer is near-C2, so the swapped labelling scores
**9.109 against 9.113**. Two candidate stories for a 9 A number and neither was
it.

**`interface.nativeContacts` is in the report so this cannot happen quietly
again.** Zero native contacts means the target is the wrong question, and the
tool's header says so.

### 1BRS A:D - barnase and barstar, an actual heterodimer

`tools/fixtures/1brs-crystal.pdb`, chains A (barnase, 108 residues) and D
(barstar, 87). Closest inter-chain CA **4.83 A**, **36 contacts under 8 A**. 195
residues, so a fold is five seconds.

| af3-int5, seed 20260831 | complex | TM | fnat | A `alone` | D `alone` | pLDDT |
|---|---:|---:|---:|---:|---:|---:|
| no template | 16.666 | 0.1563 | **0.000** | 13.562 | 12.923 | 38.76 |
| merged template slot | **0.475** | 0.9928 | **0.944** | 0.221 | 0.641 | 92.99 |
| one slot per chain | 0.488 | 0.9923 | 0.972 | 0.229 | 0.645 | 93.22 |

And the whole panel with a merged self-template, 25 diffusion steps:

| model | complex | TM | fnat | pLDDT |
|---|---:|---:|---:|---:|
| **rosettafold3** | **0.137 / 0.118** | 0.9993 | 0.917 / 0.944 | 80.90 |
| protenix2 | 0.452 | 0.9936 | 0.972 | 95.49 |
| intellifold2 | 0.455 | 0.9936 | **1.000** | 96.26 |
| alphafold3 | 0.475 | 0.9928 | 0.944 | 92.99 |
| boltz2 | 0.517 | 0.9921 | 0.972 | 96.87 |

rf3 is first here and it was **12.189 A with fnat 0.222** when this panel was
first run - see the template defect below, which is what the complex scorer
found on its first honest target.

**The complex path works end to end**: the ":"-joined sequence, the per-chain
`asymId`, the per-chain template token offsets, the shared-frame scoring. With a
template AlphaFold 3 puts barnase and barstar together to under half an
angstrom and recovers 34 of the 36 native contacts. Without one - no MSA either
- it does not even fold the chains (13 A each, pLDDT 38.8), which is the same
single-sequence wall 5CAJ shows for a monomer.

### The merged template slot: worth nothing to five models and 1.7x to one

The page builds one template SLOT PER CHAIN, which is AF3's convention.
`multichainMaskFor` opens a cross-chain pair only where a slot covers BOTH ends,
so per-chain slots contribute nothing across the boundary however `spanChains`
is set - each covers one end. `mergeTemplateSlots` (src/af3/featurise/template-input.js)
folds them into ONE slot that does; `--per-chain-templates` is the arm without
it, and `--no-span-chains` is the same MERGED slot with the cross-chain block
masked.

🔴 **THE MASK REALLY OPENS, AND THE TOOL COUNTS IT.** 1BRS A:D: 18,792 of the
18,792 cross-chain pairs open on the merged slot, 0 on the per-chain arm and 0
under `--no-span-chains`. A `spanChains` that failed to reach `chainMaskFor`
would have produced the per-chain answer while the report said "merged", which
is the whole reason the count is printed.

🔴 **AND `--no-span-chains` IS THE ARM THAT ISOLATES THE INTERFACE**, because
merged against per-chain confounds two things at once under rf3's dialect: the
merged slot both opens the cross-chain block AND carries full intra-chain
weight, where two per-chain slots each carry HALF - `templateFeatureMeanOnePass`
divides by the number of PRESENT templates, so two half-covering templates are
two present ones. Same single slot, only the cross-chain block differing:

| 1BRS A:D, seed 20260831 | complex | fnat | cross pairs open |
|---|---:|---:|---:|
| rosettafold3, chains open | **0.137** | 0.917 | 18,792 |
| rosettafold3, chains masked | 0.227 | 0.917 | 0 |
| alphafold3, chains open | 0.475 | 0.944 | 18,792 |
| alphafold3, chains masked | 0.478 | 0.972 | 0 |

**The interface is worth 1.7x to rosettafold3 and nothing to AlphaFold 3**, and
the reason is what the feature IS. rf3's 66 columns are a CA-CA distance
distribution with a coverage flag - distance conditioning, which a cross-chain
distance is exactly an instance of. AlphaFold 3's featuriser cannot produce a
cross-chain template feature at all (`evoformer.py:519` builds
`asym_id[:, None] == asym_id[None, :]` with no option beside it), so for that
checkpoint the block is out of distribution and reads as noise it has learnt to
ignore. boltz2, the family whose reference masks templates by COVERAGE rather
than by chain (`TEMPLATE_VISIBILITY_BY_COVERAGE`, whose comment says "a row that
covers two chains makes their cross-chain block visible"), was the one expected
to gain and did not: 21.065 / 22.557 merged against 21.162 per-chain on 5CAJ,
with the chains folding slightly WORSE under the merged slot.

🔴 **AND DO NOT READ rf3's 0.137 AGAINST 0.676 AS THE INTERFACE.** Its per-chain
arm is 0.676 and its merged arm 0.137, a 5x gap - and the table above says only
0.227 to 0.137 of that is the cross-chain block. The rest is the denominator:
under `templateFeatureMeanOnePass` two per-chain templates halve each other's
intra-chain weight. Two mechanisms in one comparison, and the arm that separates
them was one flag.

### The sampler on a real complex, and ODE loses on both hard targets

1BRS A:D, merged self-template, everything else held:

| sampler | complex | fnat | pLDDT |
|---|---|---|---|
| diffusion, 3 seeds | **0.448 / 0.475 / 0.491** | 0.944 / 0.944 / 0.944 | 92.8-93.0 |
| flow, 2 seeds | 0.465 / 0.503 | 0.944 / 0.917 | 91.9-92.3 |
| **ode**, 2 seeds | **0.549 / 0.661** | 0.889 / 0.917 | 91.2-91.6 |

And on 5CAJ at 522 residues, where the complex number is meaningless but the
per-chain one is not:

| sampler | chain A `alone` | chain B `alone` | pLDDT |
|---|---:|---:|---:|
| diffusion, 3 seeds | 0.504 / 0.343 / 0.352 | 0.469 / 0.279 / 0.306 | 86.2-86.9 |
| flow | 0.356 | 0.324 | 85.85 |
| **ode**, 2 seeds | **11.098 / 3.998** | **10.048 / 5.592** | **61.1 / 63.0** |

🔴 **THE 68-RESIDUE SAMPLER PREFERENCE DOES NOT SURVIVE TO 195, LET ALONE 522.**
Two sections up, ODE was said to be the step that made AlphaFold 3's 1QYS a
chain again while Flow returned none - and that claim is RETRACTED below. Here ODE is last on a 195-residue
heterodimer by a margin larger than the seed spread of either other arm, and at
522 residues it **collapses the chains themselves** - 4 to 11 A where diffusion
and flow are both under 0.51, with pLDDT dropping to 61 from 86.

Both findings are real and they are about different lengths. That is the
argument for ODE being a named option a user picks rather than a per-model
default this port picks for them, and the honest guidance is: **ODE for short
chains, Diffusion for anything long.** The page defaults to Diffusion.

🔴 **AND READ THE `alone` COLUMN ON 5CAJ, NOT THE COMPLEX ONE.** The complex
score there is placement over two chains that do not touch, and its seed spread
within one sampler (9.1 to 23.7) is larger than any difference between samplers.
A sampler winner read off that column would be noise.

### 1TIM A:B - a LARGE complex where every number is trustworthy

`tools/fixtures/1tim-crystal.pdb`, chains A and B: triosephosphate isomerase,
**494 residues, an actual biological homodimer** with closest inter-chain CA
3.26 A and **101 contacts under 8 A** - nearly three times 1BRS's interface, on
two and a half times the residues. This is the target that answers "larger
complexes": big, real, and with a metric that means something.

af3-int5, merged self-template, 25 diffusion steps:

| | complex | TM | fnat | pLDDT |
|---|---:|---:|---:|---:|
| no template | 10.569 | 0.4619 | 0.139 | 52.51 |
| **merged template** | **0.957** | 0.9866 | **0.871** | 91.30 |
| merged, cross-chain block MASKED | 1.293 | 0.9781 | 0.713 | 88.14 |

and the panel, all with the merged self-template:

| model | complex | TM | fnat | pLDDT | seconds |
|---|---:|---:|---:|---:|---:|
| **rosettafold3** | **0.151** | 0.9996 | **1.000 (101/101)** | 79.15 | 24 |
| af3 | 0.957 | 0.9866 | 0.871 | 91.30 | 20 |
| boltz2 | 1.027 | 0.9849 | 0.901 | 93.08 | 29 |
| protenix2 | 1.047 | 0.9840 | 0.832 | 90.05 | 42 |
| intellifold2 | 1.065 | 0.9839 | 0.891 | 94.67 | 91 |

🔴 **AND HERE THE CROSS-CHAIN TEMPLATE BLOCK PAYS FOR AlphaFold 3 TOO**, which
1BRS said it did not: 1.293 to **0.957** and fnat 0.713 to **0.871**, sixteen
more native contacts. rosettafold3 goes 0.403 to **0.151** with fnat 0.950 to
1.000. So `spanChains` is worth nothing on a 195-residue heterodimer with 36
contacts and worth 1.35x (af3) and 2.7x (rf3) on a 494-residue dimer with 101 -
the finding is about the SIZE of the interface, and the earlier "worth nothing"
was one target's answer read as the rule.

🔴 **AND THE SAMPLER ANSWER IS UNAMBIGUOUS AT THIS LENGTH.** Same target, same
template, af3-int5, four seeds each, arms interleaved in one batch:

| sampler | complex, 4 seeds | mean | fnat | pLDDT | seconds |
|---|---|---:|---|---:|---:|
| diffusion, 25 steps | 0.957 / 0.934 / 0.931 / 1.010 | **0.958** | 88 / 88 / 89 / 88 of 101 | 91.2-91.7 | 20.5 |
| flow, 16 cycles | 0.980 / 0.992 / 0.968 / 0.959 | **0.975** | 88 / 89 / 90 / 88 | 92.2-92.6 | 19.8 |
| **ode**, 16 cycles (2 seeds) | **7.853 / 5.394** | 6.62 | **26 / 38** | **70.3 / 75.1** | 19.8 |

🔴 **FLOW AND DIFFUSION ARE NOT DISTINGUISHABLE HERE, AND THE SEED SPREAD SAYS
SO RATHER THAN THE MEANS**: flow's whole range (0.959-0.992) sits INSIDE
diffusion's (0.931-1.010), the two differ by 0.017 A on a per-seed spread of
0.079 and 0.033, and their interface counts overlap seed for seed. Flow's pLDDT
is consistently about one point higher, which is not a correctness statement.

**So ODE's collapse is the STEP and not the WALK.** Flow and ODE are the same
schedule, the same 16 cycles and the same code path, differing only in
`x <- D` against `x <- D + (sigma_next/sigma)(x - D)` - and one of them is
0.975 while the other is 6.62.

🔴 **AND FLOW IS NOT MEANINGFULLY CHEAPER AT THIS SIZE, WHICH IS WORTH SAYING
BECAUSE 16 AGAINST 25 STEPS LOOKS LIKE IT SHOULD BE.** 19.8 s against 20.5, a
3% difference: at 494 tokens with 3 recycles the fold is the TRUNK, and the
sampler is a sliver of it. The step-count saving is real at 68 tokens and gone
by 494.

🔴 **AND MORE FLOW CYCLES ARE WORSE, NOT BETTER.** At 25 cycles instead of 16 -
matching diffusion's budget exactly - flow reads **0.999 / 1.020 with pLDDT
89.13 / 89.45**, against 16 cycles' 0.980 / 0.992 at 92.6. So the walk's length
is not a quality dial, which is the same shape docs/OPENDDE.md records for
OpenDDE ("more steps are WORSE") and the reason `AF3_COUNTS` prefers 16. Put beside the 68-residue result where ODE was the only step that
folded 1QYS at all, the shape of the whole finding is now clear:

| residues | ODE against diffusion |
|---|---|
| 68-92 (1QYS/6MRR, af3) | ODE **loses** - 1QYS diffusion 0.918, flow 0.936-0.999, **ODE 1.215**; 6MRR 0.650 / 0.687 / **1.488**. The "Flow is not a chain here" that made ODE look like a rescue is retracted below |
| 195 (1BRS A:D) | ODE mildly worse - 0.549/0.661 against 0.448-0.491 |
| 494 (1TIM A:B) | ODE **breaks** - 6.62 mean against 0.958 (diffusion) and 0.975 (flow), fnat 0.87 to 0.26 |
| 522 (5CAJ A:B) | ODE wrecks the CHAINS - 4 to 11 A alone, pLDDT 61 against 86 |

**ODE degrades with length and it is not subtle.** The page defaults to
Diffusion, ODE is opt-in, and its tooltip now says so.

### 🔴 AND IT FOUND A DEFECT: RoseTTAFold3's TEMPLATE WAS A QUARTER STRENGTH

The complex panel put rf3 at **12.189 A on 1BRS A:D** where the other five are
0.45 to 0.52. Two folds separate a complex failure from a fold failure and it
was neither: **rf3 on ONE chain of barnase, with a full self-template, was
13.215 A** where AlphaFold 3 on the identical input is 0.231.

The established instrument on the established target says it plainly. 5CAJ chain
A, self-template, 25 diffusion steps, every model int5:

| | with a self-template | without | pLDDT |
|---|---:|---:|---:|
| intellifold2 | 0.254 | - | 94.76 |
| alphafold3 | 0.281 | 17.624 | 92.12 |
| **rosettafold3, before** | **17.771** | 17.949 | 63.73 |
| **rosettafold3, after** | **0.137** | 17.949 | 79.16 |

0.137 A is the best of the seven.

**What it was.** rf3's reference has its own template forward, and the part that
matters is the DENOMINATOR:

```python
a_tij = (jnp.einsum('t,tijc->ijc', present, feats)
         / jnp.clip(present.sum(), 1.0, None))     # mean over PRESENT templates
...                                                # then ONE forward pass
```

against every other family, which runs a forward PER SLOT and averages the
outputs over the slot count. A fold pads the template slots to **four**
(`templates: options.templates ?? 4` in fold.js) and `TEMPLATE_SCALE` is
`1/slots`, so one real template reached rf3's trunk as **a quarter of itself**,
mixed with three quarters of an empty-slot pass. A quarter of a
distance-distribution conditioning is a template that does almost nothing -
which is exactly what 17.949 to 17.771 looks like.

`templateFeatureMeanOnePass` in dialect.js is the convention, true for
rosettafold3 and false for the other six.

🔴 **AND NO MODULE CHECKER COULD SEE IT, BECAUSE ALL OF THEM PASS `templates:
1`.** At one slot a slot mean and a present mean are the same number. The
checker that exists reads **0.061** for rf3 - and that is not the defect either:
it is the int5 bundle, which the control names outright, **boltz2 int5 0.167
against boltz2 f32 8.25e-7**. So the module check was green, the number that
looked bad was quantisation, and the real fault only existed at a slot count no
checker used.

🔴 **THE UNTOUCHED ARMS, SHOWN UNTOUCHED** rather than argued from the diff:
rf3's 6MRR with no template is `meanPlddt` **81.53041400210395** and 1.649 A
before and after, identical to sixteen digits - which is the empty path, and the
whole reason rf3's trunk seam stayed exact through the bug. boltz2's 5CAJ
self-template is 0.837 both ways. `npm test` 1103/0.

🔴 **AND THE GATE THAT WOULD HAVE CAUGHT IT NOW EXISTS**: `npm run
test:template` (`tools/check-template-path.mjs`) folds 5CAJ chain A through
every AF3-lineage bundle **twice** - with the self-template and without - and
fails unless the templated fold lands on the crystal AND the control does not.
Both arms, because "0.14 A with a template" is evidence only if the same model
is 17 A without one; a checkpoint that had memorised the target would otherwise
pass. This is the template twin of `npm run test:ligand`, and the same sentence
applies to both: **every other fold gate here folds a plain protein with no
template, so the stage went unexercised.**

The whole panel, which is the first time these seven have been measured side by
side on a template:

```
ok    alphafold3    with a template   0.281 A   without  17.624 A   pLDDT  92.12
ok    openbind0     with a template   0.384 A   without  19.273 A   pLDDT  93.63
ok    boltz2        with a template   0.837 A   without  16.990 A   pLDDT  91.57
ok    protenix2     with a template   0.249 A   without  18.994 A   pLDDT  94.32
ok    opendde       with a template   0.233 A   without  17.159 A   pLDDT  91.30
ok    intellifold2  with a template   0.254 A   without  17.919 A   pLDDT  94.76
ok    rosettafold3  with a template   0.137 A   without  17.949 A   pLDDT  79.16
```

🔴 **AND IT WAS VERIFIED TO FAIL**, because a gate that cannot is not one:
with `templateFeatureMeanOnePass` put back to false the rosettafold3 row reads
`FAIL ... with a template 17.771 A   without 17.949 A   pLDDT 63.73` and the
run exits 1.

### What the complex work leaves open

Each of these is a run rather than an argument, and none of them is blocking:

- **An MSA on a complex.** Every number in this section is single-sequence with
  a self-template, and the interface is exactly the part co-evolution is for.
  `af3BatchFromA3m` already takes `{paired, unpaired, unpairedProfile}`, so the
  path exists; what is missing is a paired alignment to feed it, which means the
  network and a pairing mode. Until then "no template, no MSA" is the floor
  every complex number here is measured against, and it is a harsh one -
  af3 is 10.569 A on 1TIM and 16.666 on 1BRS that way.
- **A self-template is not a template.** The whole panel lands between 0.15 and
  1.07 A because it is being handed the answer. A homolog at 40% identity is the
  question a user actually asks, and nothing here has asked it.
- **Three or more chains.** `chainAssignments` enumerates up to 720 relabellings
  and is tested to six identical chains, but no fold has run past two. A
  homotrimer would exercise both the permutation and the featuriser's chain
  numbering at once.
- **Whether the page should offer the merged slot.** It is worth 1.35x to
  AlphaFold 3 on a 494-residue dimer and nothing on a 195-residue one, and it is
  out of AF3's training distribution either way - `evoformer.py:519` cannot
  produce a cross-chain template feature. The page's template UI is per-chain by
  construction and stays that way until someone decides that trade.

## 🔴 RETRACTED: "AlphaFold 3 with Flow returns no chain on 1QYS"

This file and CLAUDE.md both carried it, the page's default sampler was changed
on it, and the whole ODE option was justified by it. **It is false.**

The recorded finding: af3 + Flow + 1QYS, "not a chain on four seeds out of
four - CA median 4.267 / 4.367 / 4.861 / 4.886 A against 3.80, worst up to
26.02, pLDDT 68.79", against diffusion's 1.072 A.

Re-measured, four seeds, through BOTH fold tools:

| af3, 1QYS, 16 cycles | CA-CA median | worst | RMSD | pLDDT |
|---|---:|---:|---:|---:|
| flow, seed 1 | 3.801 | 4.04 | 0.956 | 79.09 |
| flow, seed 7 | 3.787 | 3.50 | 0.944 | 79.65 |
| flow, seed 21 | 3.794 | 4.06 | 0.936 | 79.74 |
| flow, seed 20260831 | 3.790 | 4.01 | 0.999 | 79.49 |
| diffusion, seed 1 | 3.806 | 4.02 | **0.918** | 79.61 |

`fold.js --sequence=` and `fold-opendde.js --target=1qys` agree to the digit
(both 3.801 and 79.09 at seed 1), so it is not a tool difference.

🔴 **AND IT IS NOT A FIX THAT LANDED SINCE, WHICH IS THE CHECK THAT SETTLES IT.**
A worktree at **`9cc43d7` - the commit that recorded the claim** - run on the
same GPU with the same bundle gives **CA-CA 3.801 / 3.787 / 3.794 and pLDDT
79.09 / 79.65 / 79.74**, identical to HEAD. The code at the commit that reported
a wreck does not produce one.

🔴 **THE TELL WAS IN THE NUMBERS ALL ALONG: THE CONTROL MOVED TOO.** Diffusion
was recorded at 1.072 A and measures 0.918; pLDDT was recorded at 68.79 and
measures 79.5. **A difference that appears in both arms is a difference in the
CONFIGURATION, not in the thing under test** - this file's own rule about
control arms, pointed the other way round.

🔴 **AND THE SCRATCHPAD LINE ABOVE THOSE NUMBERS READS `(clean = diagnostics
reverted)`.** The run was taken on a tree that had been edited and hand-reverted
during a debugging session. **A revert you performed is not the same as `git
status`, and a measurement taken on a modified tree is not a measurement** - it
produced four internally consistent broken folds, which is exactly what makes it
convincing and exactly why four seeds agreeing is not evidence of anything on
its own.

**What survives.** Diffusion is still the right page default, for the reason
that holds up rather than the one that did not: it wins or ties nearly
everywhere - 1QYS 0.918 against flow's 0.936-0.999, 6MRR 0.650 against 0.687,
1TIM A:B 0.958 against 0.975 over four seeds - and it is the sampler
af3-any-model verified. Flow is a close second at the same price everywhere it
has been measured, and rosettafold3's `noFlowSampler` is unaffected: that
collapse was measured on a clean tree, reproduces, and is a different finding.

### And what it does to ODE

ODE's headline was "the only step that folds 1QYS for AlphaFold 3", on a target
that never needed rescuing. Measured against DIFFUSION rather than against a
broken Flow - one seed, 68 and 92 residues, the lengths ODE is supposed to suit:

| model | target | diffusion 25 | flow 16 | **ode 16** |
|---|---|---:|---:|---:|
| af3 | 6MRR | **0.650** | 0.687 | 1.488 |
| boltz2 | 6MRR | **0.476** | 0.541 | 0.509 |
| protenix2 | 6MRR | **0.693** | 0.733 | 1.223 |
| intellifold2 | 6MRR | 1.549 | 1.565 | **0.769** |
| af3 | 1QYS | **0.918** | 0.956 | 1.215 |
| boltz2 | 1QYS | **0.838** | 0.928 | 1.012 |
| protenix2 | 1QYS | 1.017 | **0.837** | 1.082 |
| intellifold2 | 1QYS | 1.585 | **1.136** | 1.851 |

**ODE wins 1 of 8 here and 1 of 9 counting 1TIM** - intellifold2 on 6MRR, where
it is genuinely twice as good (0.769 against 1.549) - and loses everywhere else,
by 2.3x on af3's 6MRR. Add 195 residues (0.549/0.661 against 0.448-0.491), 494
(6.62 mean against 0.958) and 522 (it wrecks the chains), and there is no length
and no model where it is the right pick except that one square.

The earlier per-model table that made ODE look good - "af3, boltz2 and
intellifold2 are better under ODE" - compared ODE against **Flow**, never
against Diffusion, on one target. That was the gap.

### The samplers on 1TIM, per model - and two findings bigger than the ODE question

1TIM A:B, 494 residues, merged self-template, seed 20260831 unless noted.

| model | diffusion 25 | flow 16 | **ode 16** |
|---|---:|---:|---:|
| af3 | **0.958** (4 seeds) | 0.975 (4 seeds) | 6.62 (2 seeds) |
| boltz2 | **1.027 / 1.090** | **17.706 / 1.042 / 19.598** | 20.256 |
| protenix2 | **1.047** | 1.269 (fnat 0.653) | 4.819 |
| intellifold2 | 1.065 | **1.058** | 1.665 |

🔴 **ODE'S ONE WIN DOES NOT SURVIVE THE LENGTH.** intellifold2 on 6MRR is the
single square where ODE beats everything (0.769 against 1.549) - and at 494
residues it is **1.665 against flow's 1.058 and diffusion's 1.065**, the worst of
the three for that model too. So the win is a property of 68 residues, not of
the checkpoint. Counting 1TIM, **ODE wins 1 of 12 measured model/target pairs**
and loses every other one.

🔴 **AND FLOW HAS A FAILURE OF ITS OWN AT LENGTH, WHICH IS boltz2's.** Three
seeds on 1TIM: **17.706, 1.042, 19.598**. Two of three are catastrophic, and
they fail in two DIFFERENT ways - which the precision column is what separates:

| boltz2, flow, 1TIM | complex | recall | precision | predicted / native | chains `alone` | pLDDT |
|---|---:|---:|---:|---:|---|---:|
| seed 20260831 | 17.706 | 0.000 | 0.000 | 51 / 101 | 1.118 / 1.071 | 85.11 |
| seed 7 | **1.042** | 0.881 | 0.840 | 106 / 101 | 1.023 / 0.939 | 93.00 |
| seed 21 | 19.598 | 0.901 | **0.192** | **474** / 101 | 16.785 / 16.576 | 73.26 |
| diffusion, control | 1.027 | 0.901 | 0.843 | 108 / 101 | 0.976 / 1.039 | 93.08 |

Seed 20260831 folds both chains correctly (1.1 A each) and puts them in the
wrong place - **zero of the 101 native contacts**. Seed 21 folds them wrongly
(16.8 A each) into a mass that touches everywhere. It is a length effect and not
a boltz2 property: the same model in flow on **1BRS A:D (195 residues) is 0.499
against diffusion's 0.517**, and one chain of 1TIM alone (247) is 1.592.

🔴 **AND NOTHING THE PAGE SHOWS WOULD TELL A USER.** Both bad seeds **PASS the
chain-geometry gate** - CA-CA 3.784 and 3.788, worst 3.64 and 4.47, which is a
healthy backbone - and seed 20260831's pLDDT is **85.11**, three points off a
good fold's. The gate measures whether the backbone is a chain; two chains that
are each a chain and are 17 A from where they belong is not a question it asks.
`interface.fnat` and `interface.precision` are the only things here that can see
it, and they exist because of this run.

🔴 **AND `fnat` ALONE WAS FOOLED WITHIN AN HOUR OF BEING WRITTEN.** Seed 21 reads
**recall 0.901** - 91 of the crystal's 101 contacts recovered - on a fold whose
chains are 16.8 A out of shape, because a model that puts everything close to
everything recovers every contact by accident. It made **474** inter-chain
contacts where the crystal has 101. Recall is not an interface score without
precision beside it; both are reported now, and a healthy fold is ~104-108
predicted against 101 native at precision 0.84.

### ODE is removed from the page, and the step is kept for the CLI

Decided on the twelve pairs above: **ODE wins 1 of 12** - intellifold2 on 6MRR,
0.769 against 1.549 - and loses the other eleven, including the same model's own
1TIM row (1.665 against 1.058 and 1.065), so the win is a property of 68
residues rather than of a checkpoint. Past ~400 residues it breaks outright.
**A third option that is never the right pick is a way for a visitor to get a
worse fold**, so `index.html` no longer offers it and `AF3_COUNTS` has no row.

What stays: `--mode=ode` on any fold tool, and the step in
`diffusion-sampler-webgpu.js`. The intellifold2 square is real and nobody has
explained it, so removing the code would throw away the question along with the
option. `foldBatch` still validates the three modes and still throws on a
fourth.

🔴 **AND THE TWO LISTS ARE GATED AGAINST EACH OTHER NOW**, because they live in
different files: the `<option>`s in index.html, `AF3_COUNTS` in
web/af3-model.js, and `web/app.js` subscripting the table with the select's
value. **One of app.js's two readings had a `?? table.flow` fallback and the
other did not**, so removing a mode from one file and not the other is a "cannot
read properties of undefined" in the middle of starting a fold - the
stale-allow-list trap CLAUDE.md already records twice. `test/sampler-options.test.js`
runs it both ways (no option without a row, no row without an option), asserts
the marked-up default is `diffusion`, and was verified to fail in each
direction. The unguarded reading is guarded.

### 🔴 FLOW'S RISK IS A BAD FIRST DRAW, NOT LENGTH - and the earlier claim here was wrong

The 1TIM section above says the sampler comparison is about length, and the
page's tooltip said Flow is "weaker past about 400 residues". **That is not what
is happening.** Five seeds each, 1TIM A:B, 494 residues, merged self-template:

| | seeds 20260831 / 7 / 21 / 42 / 99 |
|---|---|
| **boltz2, flow** | **17.706** / 1.042 / **19.598** / 1.013 / 1.058 |
| boltz2, diffusion | 1.027 / 1.090 / - / 1.113 / 1.076 |
| af3, flow | 0.980 / 0.992 / 0.968 / 0.959 / 0.943 |
| intellifold2, flow | 1.058 / - / 1.086 / - / - |

**af3's Flow is clean on all five at that length**, so "Flow past 400 residues"
is false. And boltz2's Flow is clean on a LONGER target - 5CAJ A:B at 522
residues, four seeds, 0.566-0.842 alone with pLDDT 90.5-94.4 - so it is not
boltz2's Flow at length either.

🔴 **IT IS THE SEED, AND THE TEMPLATE IS EXONERATED.** The two failing seeds
fail under **every** template configuration:

| boltz2 flow, 1TIM | merged | cross-chain masked | per-chain slots | no template |
|---|---:|---:|---:|---:|
| seed 20260831 | 17.706 | 17.585 | 17.008 | 19.452 |
| seed 21 | 19.598 | 17.909 | 7.783 | 23.191 |

So the merged cross-chain block - which is the feature this session added, and
the obvious suspect - is not the cause. What is left is the initial draw: **Flow
takes one at the top of the schedule and walks down deterministically, so it
never escapes a bad one, where diffusion re-noises at every step and does.**
That explains the shape exactly - intermittent, locked to the seed, and never
seen in diffusion.

Two failure modes from that one cause: seed 20260831 folds both chains
correctly (1.1 A each) and puts them 17.7 A apart with **zero** of the 101
native contacts; seed 21 folds them wrongly (16.8 A each) into a mass that
touches everywhere.

🔴 **AND NEITHER IS VISIBLE TO ANYTHING THE PAGE SHOWS.** Both pass the
chain-geometry gate - CA-CA 3.784 and 3.788 - and seed 20260831's pLDDT is
**85.11** against a good fold's 93. The tooltip says what is true now: Flow is
close behind on average and cannot recover from a bad first draw, so a minority
of seeds come out far worse. Diffusion stays the default.

**Why boltz2 and not af3 is not established.** Five seeds on one target for one
model is where this stops; the honest claim is the mechanism and the
measurement, not a rule about which checkpoints are susceptible.
