# ESMFold2 in CUDA

A native CUDA/cuBLAS ESMFold2 (`esmfold2-fast-600m`), held stage by stage to the references on
exactly the same input:
- the trunk and sampler to biohub's own forward (the `esm` package's `EsmFold2ExperimentalModel`);
- the confidence head to Synthyra's `ConfidenceHead`, out of their own bundle. biohub ships the
  checkpoint without a head; Synthyra trained one on the same frozen trunk.

It runs:
- ESM-C 600M (36 blocks) and ESMFold2's language-model shim;
- the inputs embedder, a sliding-window atom transformer with 3D RoPE;
- z_init and the 24-block pair trunk, four passes;
- the distogram, the diffusion module and its EDM sampler (11 steps);
- the confidence head: pLDDT, PAE, pTM and ipTM.

The input is the page's own: `export_input.mjs` calls `featuriseForEsmfold2` and
`languageModelInput` (src/esmfold2/featurise.js), and the PDB is the page's writer's records.

## Run

```
native/ef2/fold 6mrr.pdb --sequence=GWSTELEKHREEL...
native/ef2/fold 1brs.pdb --sequence=<A>:<D>                  # chains joined by ':'
```

`fold` builds `ef2` if it is missing and exports the weights once into `native/ef2/weights`
(`export_weights.mjs`, 2.9 GB float32). It then starts `ef2`, which uploads the weights and warms up
on a synthetic input of about the right size while the input is exported. It folds with `--fast`;
flags after `--` go to `ef2` (`--seed=`, `--steps=`, `--inputs-window=`).

The weights come from two float32 bundles: `model-esmc-600m-f32`, and `model-esmfold2-conf-f32`.
The second is biohub's trunk plus Synthyra's head:

```
python3 tools/export_esmfold2_trunk.py --esmfold2 esmfold2-fast-600m --out model-esmfold2-conf-f32 \
    --confidence ~/.cache/huggingface/hub/models--Synthyra--ESMFold2-600/snapshots/*/model.safetensors
```

`ef2` refuses weights without a head, as the page does.

## Exactness

`oracle.py` runs biohub's forward on the CPU in float32, with its own ESM-C (biohub/ESMC-600M-1500000)
loaded in float32. On CUDA the model autocasts its language model, inputs embedder and trunk to bf16,
and its loader always takes ESM-C in bf16, so neither is a float32 reference. The oracle records
every seam, then runs Synthyra's head on the fold's own trunk pair and coordinates.

`--float32-attention` neutralises the one bf16 cast the module makes whatever the model's dtype: q, k
and v of every atom attention. That is the control `ef2 --atom-f32` is held to. 6MRR:

| seam | float32 | `--fast` |
|---|---:|---:|
| ESM-C's 37 hidden states | 9.2e-7 | |
| the language model's pair | 4.3e-7 | |
| `s_inputs` (each atom block 1e-7) | 1.0e-7 | |
| trunk, each of 4 passes | 1e-6 | 8.2e-4 |
| distogram | 5.4e-7 | 5.1e-4 |
| denoiser, the reference's own inputs (step 0 / 10) | 3.7e-7 / 4.7e-8 | 3.8e-4 |
| confidence: per-atom pLDDT / PAE logits | 1.3e-7 / 7.1e-7 | 1.1e-4 |
| pTM | 0.750974 both | 0.751067 |

1BRS A:D, with per-chain attention in ESM-C: hidden states 2.0e-6, pTM 0.967692 against 0.967691.
With the model's own bf16 atom attention left in, `s_inputs` is 1.6e-4, the module's own rounding.

## The window, which the two references disagree about

biohub's code windows every atom stack at ±64, counted in rank among valid atoms (flash-attn
`window_size=(64, 64)` on CUDA, a rank mask on the CPU). Synthyra's fastplms never windows the inputs
embedder, and the page follows it (docs/EF2FAST.md). This port defaults to the vendor's
reading; `--inputs-window=0` is the page's.

Against the crystals neither wins everywhere:

| target | windowed (default) | dense (page) | seeds |
|---|---:|---:|---:|
| 6MRR | 0.92 Å | 1.59 Å | 5 |
| 1QYS | 0.89 Å | 0.94 Å | 3 |
| 5CAJ | 2.10 Å | 1.86 Å | 3 |

## `--fast`

- **f16 trunk.** The trunk and the head's four blocks run cuBLASLt f16 GEMMs and native/af3's fused
  triangle input (`triInK`), instantiated at 256 channels.
- **Contraction.** The triangle contraction runs on zero-padded f16 planes, with a channel-major centre
  norm after it.
- **TF32.** Every other GEMM uses TF32.
- **Rejected:** native/af3's fused transition at 256 channels is slower (504 against 466 ms of trunk:
  255 registers, four warps an SM). A CUDA graph of the trunk's passes measured nothing.

| A100, warm | 6MRR (68 tokens) | 5CAJ (261) |
|---|---:|---:|
| language model | 12 ms | 54 ms |
| trunk, 4 passes | 43 ms | 453 ms |
| sampler, 11 steps | 55 ms | 145 ms |
| confidence | 5 ms | 68 ms |
| sequence to PDB (`fold`, cold process) | 1.15 s | 2.1 s |

## Gate

```
python3 native/ef2/gate.py            # folds against gate-baseline.json, oracles against their bounds
python3 native/ef2/gate.py --write
```

| case | CA RMSD | pLDDT | pTM |
|---|---:|---:|---:|
| 6MRR from its sequence | 0.844 Å | 78.32 | 0.751 |
| 1QYS from its sequence | 0.880 Å | 84.87 | 0.899 |
| 5CAJ from its sequence | 2.085 Å | 91.88 | 0.948 |
| 1BRS A:D from its two sequences | 0.526 Å | 94.52 | 0.968 (ipTM 0.959) |

The gate also checks every `data-*/` that has an `oracle-f32att/`, in both precisions:

```
node --js-float16array native/ef2/export_input.mjs native/ef2/data-x --sequence=...
~/venv_ef2/bin/python native/ef2/oracle.py native/ef2/data-x --out native/ef2/data-x/oracle-f32att --float32-attention
```

I tightened its bounds once to confirm it fails.

## Traps it cost

- **The reference is not float32 on CUDA.** The model autocasts three stages to bf16 there, and
  `from_pretrained` discards `esmc_precision`, so ESM-C is always bf16. Casting after the load keeps
  bf16-rounded weights. Measured against that, ESM-C read 6.8e-3, and it reads 9.2e-7 against a real
  float32 reference.
- **Half of q and k were never normalised.** A kernel launched 128-thread blocks over a grid sized
  for 256, the same class as native AF2's side-chain kernel. Block 0 of the inputs embedder read 0.14.
- **A padding zeroed once is not zeroed for another size.** The unfused triangle zeroed its padded
  planes once per buffer. A warm-up at another size leaves data where the next layout's padding is,
  so it now re-zeroes when the padded size changes.
