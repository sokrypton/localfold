# cuda/grad: backpropagation for the native ports, for training

Every CUDA port in this tree is forward-only, and fast because of that: fused kernels, int5 weights folded at load,
LayerNorm scales folded into the next GEMM, inputs of a Linear dropped the moment its GEMM retires. Training needs
the opposite of each of those choices, so the training graph is written **beside** each port, not inside it:

- **The inference forward is untouched.** Nothing under `cuda/af2`, `cuda/af3`, `cuda/ef2`, ... includes
  `cuda/grad`. A fold compiles and runs exactly as before; every `cuda/*/gate.py` baseline still holds because no
  byte of what it builds changed.
- **Weight gradients as well as input gradients.** bindcraft2's native AF2 (sokrypton/bindcraft2, `native/`) has
  the forward and backward for design, where the weights are frozen: every backward GEMM there is `dX = dY W^T`
  alone, and nothing a `dW` would need is kept. Training needs `dW = X^T dY` too, so the saves are different.
- **Float32 master weights**, read from the original checkpoints (AF2's `params_*.npz`), never the int5 bundles.
- **Memory by rematerialisation**: a stack keeps each block's input and replays one block's forward, with its saves,
  immediately before that block's backward (AF2's `use_remat`). Inside a block every module's backward runs in an
  arena `Scratch` scope, so only the saves and the running gradients outlive it.

## Status

Every port's main network has a training forward and backward. Two gates each (`run_gates.sh` runs all of them):
the float32 forward held to the PORT's own float32 path on its real weights, and the float64 build's backward held
to central differences of that same forward (`gradcheck.cuh`, below) - so the training graph computes what the
port computes, and its gradient is the gradient of that.

| port | what trains | forward vs the port | backward |
|---|---|---|---|
| **AF2** (`af2.cuh`) | Evoformer block, main and extra-MSA stacks, OPM first/last; the rematerialised 48-block stack | JAX `jax.vjp` in float64 (`oracle_af2_block.py`) | <= 3.3e-5, and 0.00-0.03 of the JAX bar |
| **AF2 structure module** (`af2sm.cuh`) | the eight shared-weight fold iterations (IPA with its point and pair terms, the transition, the quaternion backbone update) and the last iteration's side chain (angles -> the eight group frames -> atom14/atom37); cotangents on the whole trajectory, the last act, both angle forms and both atom layouts; `stopRotGradient` for AlphaFold 2's own training (off: af3-any-model's folding.py, which the port follows, keeps the rotation's gradient) | `structureModule()` on model_1_ptm and the trunk's real single and pair (data-test59's oracle): act 3.5e-7, rigid 1.8e-7, atom14 1.2e-7 - the port's own sensitivity to a 1e-7 nudge is 3.4e-7 | 51 tensors <= 3.5e-7 |
| **AF2 embedder and heads** (`af2io.cuh`) | the input embedder (msa_feat and target projections, left/right singles, the recycled position's distogram, the recycled pair and first-row norms, both relative encodings, the extra MSA) and every head (single, pLDDT, PAE, distogram, masked MSA, experimentally resolved) | **af3-any-model's JAX itself** (cuda/af2/oracle.py's dump): embedded msa 1.5e-9, pair 2.9e-8; the heads on JAX's own representations 0 (PAE, distogram: the same bits) to 2.8e-7 | 37 tensors <= 1.1e-6, both relative encodings |
| **AF2's template embedders** (`af2tmpl.cuh`) | the multimer's (nine pair-input projections and the query's, the pair stack in its order, the ReLU mean, the projection) and the monomer's (embedding2d, the stack with both attentions first, the pointwise attention over templates); the template torsion rows into the MSA | `templateEmbedding` on 1BRS with a template: 1.2e-7; `templateEmbeddingMonomer` on 5CAJ: 2.3e-7; `templateRows` 0 | 129 and 114 tensors <= 4.1e-5, two templates (the second moved 0.7 A) |
| **AF3 lineage**, nine families (`af3.cuh`, `af3diff.cuh`) | pairformer block (sequential, chai-1's parallel, Mol-1's PairMixer), MSA block (AF3, update-first, chai-1 grouped), the 24-block diffusion transformer (AdaLN, chai-1's raw AdaLN, Boltz-2's up-gate, rf3's q/k norm and noResidual) | `pairformerBlockAt`, `msaBlock`, `diffusionTransformer` <float>: 2e-7..3.3e-5 on af3, boltz2, protenix2, intellifold2, rosettafold3, opendde, openbind0, mol1, chai1 | <= 8.5e-6 under every dialect switch |
| **AF3 lineage: the whole denoiser** (`af3denoise.cuh`) | D(x, sigma) from the trunk's single and pair and target_feat: the diffusion conditioning (relpos, Fourier noise, the split/relpe pair forms), the atom encoder (reference embedding, atom pair and its MLP, the atom stack, aggregation), the transformer, the decoder, EDM's skip/out - into every parameter, the trunk's outputs and x | `denoise()`: af3 8.9e-6, protenix2 1.5e-6, intellifold2 1.7e-6, mol1 1.6e-6, rf3 9.9e-8 (its chirality term as the port computes it, a constant here), chai1 4.2e-6 (its structure-pair conditioning, raw AdaLN, decoder conditioning), opendde 4.9e-7 (on its 130 structural tokens), openbind0 2.0e-4, boltz2 4.1e-4 - the last two the transformer's float32 arithmetic (below) | every tensor of 688 <= 1.1e-5 (af3), at h = 1e-8; chai1 433 <= 6.3e-5; opendde 699 <= 1.2e-5 at h = 1e-7 |
| **AF3 lineage: the input embedder and distogram** (`af3embed.cuh`) | target_feat (its atom encoder and the concatenation; boltz2's atom-only sum of six token tables; chai-1's token embedding with its ESM2 rows, both projections and its encoder - LayerNormed conditioning, the 14-feature biased atom pair, the two-layer MLP, the reference-space mask), z_init (the left/right outer sum, the recycled pair, the relative encoding, the bond terms; chai-1's own relative encoding, chain-entity and declared-bond terms, its first pass recycling z_init and s_init through themselves), s_init and the recycled single, the MSA embedding (chai-1's 41 columns), the distogram head (chai-1's MLP form) | `buildTargetFeat()` and `embed<float>()`'s z_init seam, single and msa on pass 0: <= 3.1e-7 on af3, protenix2, intellifold2, rosettafold3, opendde, openbind0, mol1, boltz2 (7.3e-8), chai1 (1.6e-7, its target_feat with ESM2 3B) | 109 tensors <= 2.4e-5 (af3), boltz2 121, chai1 99 <= 8.8e-5 at h = 1e-8 |
| **OpenDDE's structural tokens** (`opendde.cuh`) | the expander (s_inputs and the single gathered from each subtoken's residue with their role embeddings, the split MLP, each structural pair through one of 49 role-pair matrices plus five feature tables, the attention bias), the refiner (four pairformer blocks, the bias on every single-attention head), OpenDDE's own confidence head (the clamped single's norm, s1/s2, the distance bins and raw distance, its blocks, PDE LN(z + z^T), PAE, the slot-major pLDDT) | `expandStructural` + `ddeConfidence` on 6MRR's 130 subtokens: s_inputs and bias exact, the refined pair 2.0e-6, single 4.9e-6, PAE/PDE/pLDDT <= 1.1e-6 | 137 tensors <= 6.6e-6 (one block each), the trunk's outputs included |
| **AF3 lineage: the template embedder** (`af3tmpl.cuh`) | the query's LayerNorm and projection, each slot's features (the nine projections with the distogram and geometry, or the fused feature columns; chai-1's bias), the pair-only stack with grid attention (chai-1's parallel), boltz2's outer residual, the output norm (chai-1's per-template coverage), the weighted sum, its ReLU and projection | the template term against `embed()`'s z_init_generic -> z_after_template on 6MRR with its own crystal in two slots: 1.4e-7..2.5e-7 on af3, boltz2, protenix2, intellifold2, rosettafold3, opendde, openbind0, mol1; chai-1 3.3e-7 against `templateEmbedding` alone (its target_feat reads ESM2 3B) | 80-88 tensors <= 9.6e-6 (af3, boltz2, chai1, rosettafold3, protenix2) |
| **AF3 lineage: the confidence head** (`af3conf.cuh`) | target_feat's projections and the pseudo-beta distogram into the pair, its four pairformer blocks, the PDE (AF3's symmetrised-after, protenix2's LN(z + z^T)), PAE and per-atom pLDDT logits; boltz2's re-embedded pair and single (its norms, the relative encoding, the s_inputs projections and their outer product, the 64-bin distance and bond terms) and split intra/inter-chain heads; rf3's parameter-free global norms (target_feat's over 449 columns, its two absent columns through the statistics) and CA distogram; chai-1's 16-bin unmasked distances, its transposed output twins (W + W_T, both trained) and ATOM37 pLDDT | `confidenceHead()`'s expectations: PAE/PDE <= 4.5e-6 on af3, protenix2, intellifold2, openbind0, mol1, boltz2 (4.8e-7), rf3 (1.6e-7), chai1 (1.5e-6); pLDDT held to 4x the port's own sensitivity (below) | 219 tensors <= 5.5e-6 (af3), boltz2 433 <= 4.1e-6, rf3 237 <= 1.9e-5, chai1 227 <= 5.5e-6 |
| **ESMFold2** (`af3.cuh`, `msap.cuh`) | pair trunk blocks of the trunk, coda, LM and MSA encoders | `trunkBlock`: 1.9e-7..3.8e-7 | the pairformer's |
| **ESMFold2's atom stack** (`ef2atom.cuh`) | c0 and the sliding-window adaLN-Zero blocks (q/k rms-normed per head and rotated by the 3D RoPE table, the window by valid rank, the live gate, the gated SwiGLU) the inputs embedder and the diffusion module's encoder and decoder share | `prepareAtoms` + `swaBlock` in their float32 arm: <= 9.0e-8; **the vendor's own float32-attention dump**: <= 1.3e-7 | 21 tensors <= 5.5e-7 |
| **ESMFold2's diffusion module** (`ef2diff.cuh`) | one denoiser call: the conditioning pair (its relative encoding, the two transitions) and every token block's bias, the noise-conditioned single, the atom encoder and decoder (the SWA stacks), the 12 adaLN token blocks with pair-biased gated attention and gated SwiGLU, the pooling and broadcast, EDM's combine | `makeDenoiser` + `denoise` in their float32 arm: 1.1e-6 | 349 tensors <= 2.6e-5 |
| **ESMFold2's trunk glue** (`ef2trunk.cuh`) | the language-model shim (the softmax mix of all 37 hidden states - their cotangents go back into the tower through `towerBackward`'s `dstate` - the projection, the zero row of a token the tower never saw, the pair MLP over `[a*b \| a-b]`, rebuilt a block of rows at a time in the backward rather than kept), the inputs embedder (c0, the windowed atom stack, toToken, the masked pool, the tail), z_init (the outer sum, the relative one-hot, the bonds, lm_z), both recycles (the experimental tier's `Linear(LN(z)) + z_init`; the parcae recycle's lmEncoder over the dropped-out pair, the per-channel decay, the readout and the coda - with the port's own dropout draw, so a seed drops what it drops), the full ESMFold2's MSA encoder (the embedding and s_inputs' projection, the count-clamped outer product, the unmasked pair-weighted averaging, the transition, the pair block), the distogram | **the vendor's own tensors** on 6MRR (its float32-attention dump): lm_z 1.8e-7, s_inputs 1.0e-7, z_init 5.7e-8, all four passes in <= 1.2e-7 and out <= 9.5e-7, the distogram 5.2e-7; `foldingTrunk` 7.4e-7. The parcae recycle against `foldingTrunk` (EF2_DETERMINISTIC): ESMFold2 4.1e-5, ESMFold2-Fast 1.2e-4 - the port moves 4.1e-5 and 5.5e-5 by itself for a 1e-7 nudge of these random inputs, and the bar is 4x that. The MSA encoder against `msaEncode` on a real 59-residue alignment's first 1,024 rows: 1.9e-7 | 67 tensors <= 1.3e-5 (experimental), 114 <= 1.9e-5 (parcae, dropout on), the inputs embedder's 22 <= 3.0e-6, the MSA encoder's 140 <= 8.3e-6 (a column mask on) |
| **ESMFold2's confidence head** (`ef2conf.cuh`) | Synthyra's head: the four s_inputs projections and their outer product, the normalised trunk pair with z_init's relative and bond tables (shared parameters, both gradients), the distance embedding, four pair-only blocks around a residual, the row-attention pooling, the per-slot pLDDT tables, the PAE | **the vendor's own tensors** on 6MRR: per-atom pLDDT 7.2e-8, PAE logits 4.6e-7; `confidenceHead` 9.4e-8 | 102 tensors <= 4.5e-6 |
| **ESM-C / ESM2** (`plm.cuh`) | the towers, ESM-C's sequence head, a masked-LM cross entropy | `esmcTower`, 36 layers: 5.8e-7 | <= 5.3e-7 |
| **MSA Pairformer** (`msap.cuh`) | its layer: PWA, transition, query-biased OPM (differential sequence weights), pair block | the port's float32 path: <= 4.2e-7 | 2.6e-6 |
| **MSA Pairformer's whole model** (`msapmodel.cuh`) | the MSA embedding, the initial pair (relative position over the break-shifted index, the bond), the layers, both contact heads (Cb, ConFind) at their layers and the LM head on the query row | `msaEmbedK`, `pairInitK`, `contacts()`, `lmHead()`: 0 to 1.5e-7 (the layers: the row above) | 115 tensors <= 2.8e-6 |
| **BioEmu2** (`bioemu2.cuh`) | MSA layer and PairMixer layer (Boltz modules, stacked (out, in) weights) | `pairWeightedAveraging`/`transition`/`outerProductMean`/`trimul`: <= 2.1e-6 | <= 2.1e-6 |
| **BioEmu2's denoiser** (`bioemu2den.cuh` on `tape.cuh`) | the static context (the pairwise conditioner's pair and relative encoding, the atom features, the windowed atom pair's geometry, token term and MLP tail, every layer's window and token biases, the single conditioner's static half, a's trunk half) and one D(x, sigma): the single conditioning, 16 dual-stream layers (windowed and full conditioned attention, the coupling mixers), the readout | `staticContext` + `denoise()` float32: the context <= 1.1e-7, D 5.7e-5 at sigma 5 and 5.3e-6 at 40; d/dx against the port's own vjp tape (jax.grad's) at float32's floor (below) | 709 tensors <= 7.6e-5, the context's weights and the trunk's three inputs included |
| **BioEmu2's trunk** (`bioemu2trunk.cuh`) | the input embedder (the per-atom features, the windowed pair and its tail, three windowed conditioned layers - on the tape - the mean over each token's atoms, the tail columns), s_init and z_init (the method rows, the bonds, the contact conditioning's class vectors and its threshold encoder through its Fourier features, the relative encoding), the recycle, the MSA embedding, the MSA and PairMixer layers, the distogram | `inputEmbedder` + `trunkInit` + two `trunkPass` + `distogram`, float32, on 6MRR: s_inputs 1.0e-7, z_init 6.5e-8, the pair 2.4e-7, the distogram 1.2e-6 | 161 tensors <= 5.0e-5 (one MSA layer, two PairMixer layers, contact classes on some pairs) |
| **RFD4-Proteina** (`rfd4p.cuh`) | token layer (RMS AdaLN, pair-biased attention, SwiGLU, output scales) and pair update | `tokenLayer`/`pairReprUpdate` <float>: 5e-8 / 1.6e-7 | 1.9e-6 |
| **RFD4-Proteina's whole trunk** (`rfd4trunk.cuh`) | one cycle: the conditioning and its two SwiGLU transitions, s_init, z_init (the pair factory's RMS-normed projection adaLN-conditioned on the pair condition), the 14 token layers and 7 pair updates, the CA, local-latent, sequence and insertion heads | `forward()` FP32, one recycle, on a real 54-token input: every stage and head <= 4.4e-7 | 93 tensors <= 1.9e-6 |
| **RFD4-Proteina's autoencoder** (`rfd4ae.cuh`) | the layer both halves share (LayerNorm throughout, no pair bias, the zero conditioning's one-row gammas, betas and gates - whose weights train through that row), the decoder's features (two embedding tables) and three heads, the encoder's per-atom features (three tables) and mean / log-scale head | `decode()`: every layer <= 1.5e-5 (the port's own floor for this near-uniform softmax), heads <= 8.5e-6; `encode()`: <= 4.7e-7 | 69 + 35 tensors <= 4.2e-6 |

Every CUDA model's forward now has a backward here; what is not is said in each header (e.g. rf3's chirality term is a
constant input to the denoiser, a template's geometry and a sampler's coordinates take no gradient).

Also: AdamW (`adam.cuh`); `train-af2-smoke` - real weights perturbed 10% and trained back, loss 918 -> 14.7 in 60
steps.

🔴 **THE DENOISER'S 4e-4 IS THE PORT'S ROUNDING, AND IT WAS SHOWN RATHER THAN ASSUMED.** boltz2 reads 4.1e-4 against
`denoise()` and openbind0 2.0e-4, where every seam up to the encoder's skip is 1e-6 or under. Three measurements put it
in the transformer's float32 and not in a convention: this denoiser in **float64** is as far from the port as in
float32 (4.0e-4, 1.9e-4); its float32 is only 8.5e-5 / 4.4e-5 from its own float64; and `test-af3diff` - the
transformer alone, on random inputs at 68 tokens - reads 7.4e-5 (af3) and 1.45e-4 (openbind0) against the port.
Its bar is 1e-3, and `GRAD_FALSIFY=1` (pre-trunk queries and the padded-key mask flipped) reads **0.74 and 0.88**.
`GRAD_STAGES=1` prints every seam; `GRAD_SENS=1` the port against itself with its inputs moved 1e-7.

🔴 **THE STRUCTURE MODULE ON RANDOM INPUTS IS CHAOTIC, SO ITS GATE READS THE TRUNK'S REAL ONES.** With unit-variance
random single and pair it read 9.3e-4 against the port - and the port moves 1.2e-3 against ITSELF when those inputs are
moved 1e-7: eight iterations of sharp IPA softmaxes amplify a last place by 1e4. On the evoformer's real outputs the
same comparison is 3.5e-7, the port's own sensitivity 3.4e-7.

🔴 **AND THE CONFIDENCE HEAD'S pLDDT IS HELD TO THE PORT'S OWN CONDITIONING, MEASURED EACH RUN.** PAE and PDE agree to
4.5e-6, while intellifold2's pLDDT read 6.6e-4 and openbind0's 3.6e-4 - and the single after the blocks already 3.4e-4,
the pair 3e-6. The single track reads the pair through the single attention's bias: moving the port's PAIR by 1e-7 moves
its own pLDDT logits 2.0e-4 (intellifold2) and 3.9e-3 (openbind0), where moving its single moves them 2.3e-5. So the
gate takes the port against itself with both inputs moved 1e-7 and holds pLDDT to max(1e-4, 4x that).

🔴 **A LAUNCH BOUND BY `blocks(n)` IS A 256-THREAD BOUND.** `blocks(n, t = 256)` - five kernels here launched 128
threads a block over `blocks(n)` blocks, so the second half of their work never ran: ESMFold2's q/k preparation (the atom
stack read 0.12 against the port at its first block) and the AF2 structure module's rigid update and identity, which no
gate saw because both its inputs were under 128 residues. The structure module's gate takes 1BRS (199) now.

`tape.cuh` is a reverse-mode tape over `real` whose weights are Params (cuda/bioemu2/src/vjp.cuh's tape, op for op,
with the weights' gradients): BioEmu2's denoiser is written on it, the static context included, because a graph of
~2,000 ops with a handful of custom ones ported more faithfully op by op than as hand-written backwards.

🔴 **BioEmu2's d/dx IS FLOAT32-LIMITED, AND THE PORT'S TWO FLOAT32 PATHS AGREEING DOES NOT SAY OTHERWISE.** At sigma 5 on
random noise the network's part of d/dx read 2.1e-2 against the port's vjp tape, where the port against itself (x moved
1e-7) reads 3.1e-4 - so it looked like a defect. It is not: both sides cut to the same depth agree to 1.3e-7 with no
layers, the context agrees to 1e-7, the float64 gradcheck holds every tensor, and this denoiser's float32 is 1.0e-2 from
its own float64 - as far as the port's float32 is. The port's vjp and denoise() agree to 8e-6 because they make the
same cuBLAS calls in the same order. A nudge of the INPUT measures conditioning; it does not measure rounding.

## The reference-free gate (`gradcheck.cuh`)

The whole library is written on `real`, float or (`-DGRAD_F64`) double. In double, each tensor's analytic gradient
dotted with a random direction is held to a Richardson-extrapolated central difference of the same forward, the
outputs differenced elementwise (never two large losses subtracted), at h = 1e-7 x the tensor's RMS: a correct
gradient reads ~1e-6, and deleting one term of the LayerNorm backward reads 1.5 (74 tensors fail). Up to three
directions a tensor, because a ReLU input within h of its kink spoils one direction and a wrong gradient all of them.
When all three fail, three more are tried with the direction scaled elementwise by |w| / rms(w): a weight with
near-dead columns (protenix2's `projectAtomFeaturesForAggr` has 130 output channels whose largest entry is 3e-3 of the
tensor's rms) holds ReLU inputs that a step sized to the tensor crosses in EVERY random direction - 14% of that
projection's outputs sit within 1e-6 rms of zero - and stepped in proportion to itself each entry stays on its side.
🔴 A loss over padded positions is not defined: a query masked in every key has logits q.k - 1e9, and even float64
rounds q.k there to 1e-7, so every gate's cotangent is zero on padding - as a loss over real residues is.

## The gate

`oracle_af2_block.py` runs af3-any-model's JAX `EvoformerIteration` in **float64** on random inputs with a padded
tail and a partly masked MSA row, the block's parameters sliced out of the real checkpoint, and takes `jax.vjp` of
`sum(msa_out * ct_msa) + sum(pair_out * ct_pair)` to both inputs and to all 85 parameter tensors. It runs the same
vjp in **float32** too and records, per tensor, how far the reference's own float32 lands from its float64.

`test-af2-block` holds every tensor to `max(1e-4, 3 x JAX float32's error)`: a gradient ill-conditioned in JAX is
ill-conditioned here, and one that is analytically zero (the row attention's `feat_2d_norm/offset` - that norm feeds
only the pair bias, and a softmax gradient sums to zero over the keys) has only rounding on either side. Every
parameter gradient starts as NaN, so one the backward never writes fails.

```
~/.venv-lfjax/bin/python cuda/grad/oracle_af2_block.py --out cuda/grad/data/af2-block --seed 1
~/.venv-lfjax/bin/python cuda/grad/oracle_af2_block.py --extra --out cuda/grad/data/af2-extra-block
~/.venv-lfjax/bin/python cuda/grad/oracle_af2_block.py --opm-last --out cuda/grad/data/af2-block-opmlast
~/.venv-lfjax/bin/python cuda/grad/oracle_af2_block.py --pad 0 --L 48 --S 7 --block 20 --out cuda/grad/data/af2-block-nopad
~/.venv-lfjax/bin/python cuda/grad/oracle_af2_block.py --L 48 --S 7 --block 20 --out cuda/grad/data/af2-block-b20
nvcc -O2 -std=c++17 -arch=sm_80 -o cuda/grad/test-af2-block cuda/grad/test_af2_block.cu -lcublas
for d in cuda/grad/data/*/; do cuda/grad/test-af2-block $d | tail -1; done
```

`--nblocks 3 --block 10 --seed 3 --out cuda/grad/data/af2-stack3` dumps a three-block stack, which the same
binary runs through `af2::Stack` - inputs kept, every block replayed before its backward.

All five single-block dumps pass, the worst tensor at 0.26-0.45 of its bar (2026-11-03, A100). The forward agrees with float64 at
1e-7 to 3e-6, inside JAX float32's own error on every case.

🔴 **A FAILING SEED CAN BE A ReLU TIE, NOT A DEFECT.** Two seeds failed first, both the same way: ONE element of
the pair transition's hidden layer within float32 rounding of zero, so its ReLU gate went the other way from
float64 - one channel of `transition1/bias` off by 0.75 (or 2.3) with every other channel within 1e-6, and every
tensor upstream inheriting it. A block has ~900k ReLU inputs, so a tie is a few-percent event, and JAX's float32
happened to land on float64's side. `GRAD_SHOW=<param>` prints a tensor's worst elements; a single outlier
channel is the signature. The gate's seeds are chosen tie-free.

🔴 **AND A FULLY MASKED SOFTMAX ROW HAS NO DEFINED GRADIENT.** A padded residue's query row is masked in every key,
so its logits are `q.k - 1e9`: float64 keeps `q.k`, float32 keeps nothing. The oracle zeroes the cotangent of every
padded output, which is what a loss over real residues does; with it nonzero the comparison was 5e8 on one tensor
and 1e-3 everywhere, measuring nothing.

## Cost

`bench-af2-block --tf32` at the AF2 training crop (L 256, 128 MSA rows), A100, random weights:

| | ms |
|---|---:|
| forward (the stack's, no saves) | 16.1 |
| replay with saves | 16.1 |
| backward | 33.7 |
| **one block trained** | **65.9** (3.2 s for 48 blocks) |

One block's activations peak at **3.0 GiB** (8.6 before the module scratch scopes); the 48 blocks' saved inputs are
3.1 GiB. Without TF32 (`--tf32` off, the gate's arithmetic) a block is 107 ms.

What is still on the table, in the order it pays: attention is materialised a batch chunk at a time and recomputed
in the backward - a flash forward/backward (bindcraft2's `flash_bwd_body.cuh` returns dQ, dK, dV and the bias
gradient) removes the logits and the chunk loop; bf16 saves halve the 3 GiB; the triangle contraction and the OPM
go through cuBLAS strided batches with channel-major copies around them, which bindcraft2's fused bodies avoid.
