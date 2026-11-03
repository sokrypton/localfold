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
| **AF3 lineage**, nine families (`af3.cuh`, `af3diff.cuh`) | pairformer block (sequential, chai-1's parallel, Mol-1's PairMixer), MSA block (AF3, update-first, chai-1 grouped), the 24-block diffusion transformer (AdaLN, chai-1's raw AdaLN, Boltz-2's up-gate, rf3's q/k norm and noResidual) | `pairformerBlockAt`, `msaBlock`, `diffusionTransformer` <float>: 2e-7..3.3e-5 on af3, boltz2, protenix2, intellifold2, rosettafold3, opendde, openbind0, mol1, chai1 | <= 8.5e-6 under every dialect switch |
| **AF3 lineage: the whole denoiser** (`af3denoise.cuh`) | D(x, sigma) from the trunk's single and pair and target_feat: the diffusion conditioning (relpos, Fourier noise, the split/relpe pair forms), the atom encoder (reference embedding, atom pair and its MLP, the atom stack, aggregation), the transformer, the decoder, EDM's skip/out - into every parameter, the trunk's outputs and x | `denoise()`: af3 8.9e-6, protenix2 1.5e-6, intellifold2 1.7e-6, mol1 1.6e-6, rf3 9.9e-8 (its chirality term as the port computes it, a constant here), openbind0 2.0e-4, boltz2 4.1e-4 - the last two the transformer's float32 arithmetic (below) | every tensor of 688 <= 1.1e-5 (af3), at h = 1e-8 |
| **AF3 lineage: the input embedder and distogram** (`af3embed.cuh`) | target_feat (its atom encoder and the concatenation), z_init (the left/right outer sum, the recycled pair, the relative encoding, the bond terms), s_init and the recycled single, the MSA embedding, the distogram head | `buildTargetFeat()` and `embed<float>()`'s z_init seam, single and msa on pass 0: <= 3.1e-7 on af3, protenix2, intellifold2, rosettafold3, opendde, openbind0, mol1 | 109 tensors <= 2.4e-5 (af3) |
| **AF3 lineage: the confidence head** (`af3conf.cuh`) | target_feat's projections and the pseudo-beta distogram into the pair, its four pairformer blocks, the PDE (AF3's symmetrised-after, protenix2's LN(z + z^T)), PAE and per-atom pLDDT logits | `confidenceHead()`'s expectations: PAE/PDE <= 4.5e-6 on af3, protenix2, intellifold2, openbind0, mol1; pLDDT held to 4x the port's own sensitivity (below) | 219 tensors <= 5.5e-6 (af3), 222 (protenix2) |
| **ESMFold2** (`af3.cuh`, `msap.cuh`) | pair trunk blocks of the trunk, coda, LM and MSA encoders | `trunkBlock`: 1.9e-7..3.8e-7 | the pairformer's |
| **ESMFold2's atom stack** (`ef2atom.cuh`) | c0 and the sliding-window adaLN-Zero blocks (q/k rms-normed per head and rotated by the 3D RoPE table, the window by valid rank, the live gate, the gated SwiGLU) the inputs embedder and the diffusion module's encoder and decoder share | `prepareAtoms` + `swaBlock` in their float32 arm: <= 9.0e-8; **the vendor's own float32-attention dump**: <= 1.3e-7 | 21 tensors <= 5.5e-7 |
| **ESM-C / ESM2** (`plm.cuh`) | the towers, ESM-C's sequence head, a masked-LM cross entropy | `esmcTower`, 36 layers: 5.8e-7 | <= 5.3e-7 |
| **MSA Pairformer** (`msap.cuh`) | its layer: PWA, transition, query-biased OPM (differential sequence weights), pair block | the port's float32 path: <= 4.2e-7 | 2.6e-6 |
| **BioEmu2** (`bioemu2.cuh`) | MSA layer and PairMixer layer (Boltz modules, stacked (out, in) weights) | `pairWeightedAveraging`/`transition`/`outerProductMean`/`trimul`: <= 2.1e-6 | <= 2.1e-6 |
| **RFD4-Proteina** (`rfd4p.cuh`) | token layer (RMS AdaLN, pair-biased attention, SwiGLU, output scales) and pair update | `tokenLayer`/`pairReprUpdate` <float>: 5e-8 / 1.6e-7 | 1.9e-6 |

Not yet: the template embedders (AF2's, AF3's), the dialect branches named in each header (boltz2's re-embedded
confidence head and atom-only target_feat, rf3's global norms, chai-1's embeddings, OpenDDE's structural tokens), and, ESMFold2's atom stack and diffusion module, BioEmu2's denoiser,
RFD4-P's encoder/decoder glue). The layers they are made of are mostly here already.

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
