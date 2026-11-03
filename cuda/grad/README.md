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

| model | forward + backward (inputs and every parameter) | gate |
|---|---|---|
| AF2 Evoformer block - main stack, extra-MSA stack (global column attention), OPM first (multimer) or last (monomer) | **done** | `test-af2-block`, 5 dumps |
| AF2 stack (48 + 4 blocks, rematerialised) | written (`af2::Stack`) | not yet gated end to end |
| AF2 embedder, template stack, structure module, heads, losses | next | |
| AF3 lineage (pairformer, MSA module, templates, diffusion, atom transformer, confidence) | planned | |
| ESMFold2, BioEmu2 (has `vjp.cuh`: activations only), RFD4-P, MSA-P | planned | |
| ESM-C / ESM2 (`cuda/plm`) | planned - esmc-cuda (sokrypton/esmc-cuda) is a complete trainer to port from | |

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

All five pass, the worst tensor at 0.26-0.45 of its bar (2026-11-03, A100). The forward agrees with float64 at
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
