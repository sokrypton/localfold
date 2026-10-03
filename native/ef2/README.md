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
native/ef2/fold gol.pdb --sequence=<SEQ> --ligands=GOL --modify=SEP@3
native/ef2/fold dna.pdb --sequence=GCGATCGATCGC:GCGATCGATCGC --kinds=dna,dna
native/ef2/fold lig.pdb --sequence=<SEQ> "--smiles=OCC(O)CO"
native/ef2/fold kras.pdb --job=tools/fixtures/af3-jobs/kras_g12c_sotorasib.json
```

The input options are native/af3's exporter's, resolved the same way:
- `--kinds`: one per chain, protein, dna or rna.
- `--ligands`: CCD codes, fetched from the RCSB.
- `--smiles`: built by src/chem.
- `--modify`: `CODE@position[@chain]`, modified residues and bases.
- `--job`: an AlphaFold 3 job file read by the page's own reader (`web/job-json.js`): its ligands,
  glycans, ions, modified residues and bases, declared bonds and `userCCD`.

All nine of AlphaFold 3's loadable example jobs fold. On the covalent KRAS/sotorasib job, Cys12 SG to
the ligand's C25 is 1.73 Å: bonded, through the declared bond.

`fold` builds `ef2` if it is missing and reads the page's own published bundles as they are - the int5
trunk and the int3 ESM-C, fetched once by `native/fetch_bundles.py` from their Hugging Face remotes, 0.35 GB:
their codes go to the device and are decoded there (native/af3's `Model::loadBundle`; bit-identical to
decoding them on the host, and the load is 127 against 241 ms for the float32 file).

The quantisation costs accuracy, measured on the gate's cases: 6MRR 0.84 -> 1.42 A (pLDDT 78.3 -> 77.2),
1BRS 0.53 -> 0.92 A, ligand bonds 0.047 -> 0.115 A rms, nucleic 0.036 -> 0.063, KRAS pLDDT 88.1 -> 83.7
with the covalent SG-C25 still bonded (1.79 A). The float32 export (`export_weights.mjs`, 2.9 GB, from
the unpublished float32 bundles) is what the oracle checks below use, through `ef2 --weights=`. It then starts `ef2` while the input is exported. `ef2`
uploads the weights and, during the upload, warms up on a synthetic input of up to 96 tokens. It folds
with `--fast`; flags after `--` go to `ef2` (`--seed=`, `--steps=`, `--inputs-window=`).

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

Beyond plain protein, against the float32-attention oracle in float32:

| input | trunk | denoiser | pLDDT per atom | pTM, both |
|---|---:|---:|---:|---:|
| protein + glycerol | 1.0e-6 | 9.0e-7 | 1.3e-7 | 0.775944 |
| protein with phosphoserine at 3 | 1.4e-6 | 2.4e-7 | 1.4e-7 | 0.773445 |
| DNA duplex | 7.6e-7 | 4.0e-7 | 2.1e-7 | 0.182829 |
| RNA hairpin | 6.7e-7 | 4.9e-7 | 1.8e-7 | 0.063898 |

With no protein token, as in a nucleic or ligand-only input, the tower is skipped, and every token
takes the shim's value at a zero state, as the reference does.
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

**The 256-channel pair track, on fused kernels of its own** (`src/fused256.cuh`). native/af3's fused kernels hold
a whole 128-wide tile or weight on the chip; at 256 channels that is 255 registers a thread
or more shared memory than a block gets. Its fused transition measured slower (504 against
466 ms), so these stream instead:

| kernel | does | 261 tokens, 4 passes |
|---|---|---:|
| `triIn256K` | LN, projection and gate, the interleaved a/b split into padded planes, the gating linear | 89 ms (native/af3's `triInK` at 256: 110) |
| `triangleOutK` | the centre norm, the output projection (weight streamed 32 columns at a time), the gate and the residual | 79 ms (unfused: ~100) |
| `transitionUpK` | LN, the [gate, value] widening and SwiGLU: the 2I-wide rows never leave the chip | 93 ms (unfused: ~100) |

What made them pay, each measured:
- **Fragments, not staging buffers.** `triangleOutK` builds its MMA fragments straight from the
  channel-major product tile and normalises them on the way, so there is no f16 row buffer.
- **Shared memory recycled.** The weight stages reuse the memory of rows already loaded into
  registers, so two blocks fit an SM: `triIn256K` 110 → 89 ms.
- **Coalesced epilogue.** `triangleOutK` stages its output chunk and writes 16 bytes a thread:
  145 → 83 ms.
- **But not always.** The same staging in `transitionUpK` cost it an SM's second block and was
  slower (93 → 112 ms), so it was taken out.

The tiles leave a small pair track's device idle, so the fused path starts at 80 tokens:

| warm trunk, 4 passes | 68 tokens | 92 | 195 | 261 | 476 |
|---|---:|---:|---:|---:|---:|
| cuBLASLt | **43.2 ms** | 71.5 | 201 | 368 | 1131 |
| fused | 52.7 | **58.3** | **190** | **320** | **953** |

`--no-fused256` is the cuBLASLt arm. Its accuracy is the same: trunk 7.4e-4 against the
float32-attention oracle, against 8.2e-4 unfused.

**The sampler** (warm, 6MRR / 5CAJ: 55 / 145 ms before, 28 / 48 after; the fold byte-identical
across the graph and warm-up arms, the denoiser 3.3e-4 against the oracle under `--fast`):
- **The atom attention is windowed, not masked.** It built `[heads, A, A]` scores and masked all but
  ±64 ranks; `swaWindowK` takes 16 consecutive valid queries and their whole window in one
  shared-memory tile, four queries a warp. 7.6 → 3.7 ms at 261 tokens, and the A² buffer is gone.
- **Every projection of the single alone is one batched GEMM a step**: the 12 blocks' adaLN gates
  and shifts (from one LN of the single, scaled per block) and their out gates, 72 GEMMs.
- **The denoiser's GEMMs run on f16 tensor cores** (`--no-sampler16`: TF32), at 68 tokens no faster,
  at 261 a third.
- **One step is captured as a CUDA graph and replayed** (`--no-sampler-graph`): the noise level's
  scalars live on the device, so the graph serves every level.
- **The pooling walks each token's atoms**, not every atom for every token (5.8 ms at 261).
- **Every weight starts on 16 bytes on the device** (native/af3's loader lays the file out so, for
  all three ports): cuBLAS was choosing its align1 kernels for most of them, and a batched GEMM,
  which cannot see its pointers, faulted. That alone took the sampler from 54 to 32 ms on 6MRR.

Elsewhere:
- The contraction runs on zero-padded f16 planes.
- Every other GEMM uses TF32.
- A CUDA graph of the trunk's passes measured nothing.

| A100, warm | 6MRR (68 tokens) | 5CAJ (261) |
|---|---:|---:|
| language model | 10 ms | 15 ms |
| trunk, 4 passes | 43 ms | 320 ms |
| sampler, 11 steps | 28 ms | 48 ms |
| confidence | 4 ms | 18 ms |
| sequence to PDB (`fold`, cold process) | 0.64 s | 0.99 s |

## Start-up

`fold`, sequence to PDB in a cold process: 6MRR **1.15 → 0.64 s**, 5CAJ **1.97 → 0.99 s**.
The fold's own compute is 0.09 and 0.40 s of that. The rest:
- **The CUDA context: 0.20 s.** Created while the input is exported (0.16 s).
- **The weights, read with `pread` into pinned buffers rather than mapped: 0.24 s for 2.9 GB.**
  native/af3's loader, so all three ports read this way. A mapping cost page faults to read, and
  275 ms to tear down when the process exited, after the PDB was written. A file is mapped now only
  when the host reads one of its entries.
- **The warm-up runs while the weights are still arriving** (`M.uploadAsync`). Its kernels read a copy
  still in flight, so its answers are garbage, and every weight derived from the copy is dropped
  afterwards (`forgetDerivedWeights`: the f16 mirrors, the f16 bias copies). The fold is
  byte-identical to one with no warm-up.
- **The warm-up is capped at 96 tokens.** A warm fold the input's size outlasted the upload: 5CAJ's
  261 tokens cost 0.2 s. The warm-up loads kernel modules and cuBLAS plans, and 96 tokens is already
  in the fused kernels' range.

- **`fold` returns once the PDB is written.** On success `ef2 --detach-output` prints `ef2: done` and
  closes stdout, and the driver releases the device after the wrapper has returned (0.14 s), as
  native/af3's does.

`EF2_STARTUP=1` prints the context and upload times.

## Memory

Peak device memory at 783 tokens (5CAJ's chain three times), `--fast`: **15.9 → 6.8 GB**.
- The ESM-C tower's four matrices a block run on their f16 mirror (`gemmH`), and their f32 device copy
  is dropped once the mirror is built (native/af3's `compactWeights`): 2.29 GB, the bulk of the 2.9 GB of
  weights. LM pair 2.4e-4 against the oracle, as under TF32, and slightly faster (783 tokens: 58.9 against
  61.2 ms). `--no-tower16` keeps the f32 copies.
- From ~350 tokens (a pair over 128 MB) each phase gives its predecessor's scratch back. Below that the
  scratch is kept, because re-allocating it cost every phase `cudaMalloc`s (6MRR's language model 12 → 9 ms).
- Each phase gives its buffers back when it is done: the language model's pair after z_init, z_init and
  the trunk's scratch after the trunk, the denoiser's after the sampler. The warm-up fold used to keep
  everything it allocated under the real fold.
- The relative position encoding is computed where it is read (z_init, the diffusion's conditioning,
  the confidence head), not kept as a pair tensor.
- The pair-sized intermediates that only feed a projection are made 64 MB of rows at a time: the
  recycle's LN, the diffusion's `[z | rel_pos]` and its per-block normalised pair, the confidence head's
  outer product.
- The distogram is computed only under `--oracle`; nothing else reads it.
- The diffusion's 12 pair biases are kept, in f16 under `--fast`, and the conditioning pair is not.
- The weights' f16 mirror covers the folding bundle only. It used to convert ESM-C as well (1.2 GB),
  which reads only its f32 copy.

`EF2_MEM=1` prints what is in use at each phase.

## Gate

```
python3 native/ef2/gate.py            # folds against gate-baseline.json, oracles against their bounds
python3 native/ef2/gate.py --write
```

The folds run on the published bundles (as `fold` does); the oracle checks on the float32 export.

| case | CA RMSD | pLDDT | pTM |
|---|---:|---:|---:|
| 6MRR from its sequence | 1.422 Å | 77.21 | 0.737 |
| 1QYS from its sequence | 0.865 Å | 83.86 | 0.878 |
| 5CAJ from its sequence | 2.077 Å | 90.82 | 0.944 |
| 1BRS A:D from its two sequences | 0.923 Å | 94.46 | 0.967 |
| 6MRR + glycerol + phosphoserine at 3 | 1.625 Å, ligand bonds 0.115 | 80.15 | 0.790 |
| DNA duplex | nucleic bonds 0.063 | 57.24 | 0.155 |
| RNA hairpin | nucleic bonds 0.063 | 60.07 | 0.063 |
| KRAS + sotorasib (AF3's job) | ligand bonds 0.031, SG-C25 1.79 Å | 83.70 | 0.911 |

Bond rms is against CCD ideals, scored by `native/af3/bonds.mjs`; it may move 0.01 Å. The covalent
distance must stay under 2.2 Å, which is bonded and not merely near.

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
