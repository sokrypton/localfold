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

The input is the page's own, built natively: `cuda/featurise/esmfold2-featurise` is `export_input.mjs` in C++ -
`featuriseForEsmfold2` and `languageModelInput` (shared/esmfold2/featurise.js) over the AF3 featuriser, its
ligands, SMILES and modified residues included - writing the same files byte for byte
(`tools/check-native-featuriser.py`), and the PDB is the page's writer's records. No JavaScript runs on a fold.
Beside it every fold writes AlphaFold 3's `<stem>_confidences.json` (the expected PAE, `token_plddts`,
the token layout) and `<stem>_summary_confidences.json` (pTM, ipTM, mean pLDDT), as cuda/af3 does -
what the CUDA backend (cuda/worker.py) hands the page. The 300M checkpoint folds through the
same binary (`--fold-bundle=model-ef2-fast-300m-int5 --esmc-bundle=model-esmc-300m-int3`; `cuda/esmfold2/fold
x.pdb --model=esmfold2-fast-300m --job=<job.json>`).

## Run

```
cuda/esmfold2/fold 6mrr.pdb --sequence=GWSTELEKHREEL...
cuda/esmfold2/fold 1brs.pdb --sequence=<A>:<D>                  # chains joined by ':'
cuda/esmfold2/fold gol.pdb --sequence=<SEQ> --ligands=GOL --modify=SEP@3
cuda/esmfold2/fold dna.pdb --sequence=GCGATCGATCGC:GCGATCGATCGC --kinds=dna,dna
cuda/esmfold2/fold lig.pdb --sequence=<SEQ> "--smiles=OCC(O)CO"
cuda/esmfold2/fold kras.pdb --job=tools/fixtures/af3-jobs/kras_g12c_sotorasib.json
```

🔴 **ONE BINARY, ONE COMMAND, NOTHING ELSE TO RUN** (cuda/featurise/standalone.h): `cuda/esmfold2/esmfold2 --job=<job.json>
--out=<pdb>` (or `--sequence=`, and any input flag `fold` takes) fetches the model's published weights the first
time (`cuda/featurise/fetch-weights`'s code, into the checkout or `LOCALFOLD_HOME`), featurises the input in the same
process while the device starts - the featuriser's own object, so the input is byte for byte what
`cuda/featurise/esmfold2-featurise` writes - and folds; a searched alignment is kept as `<out>.a3m`, and `--frames=<dir>` writes
each intermediate result as it lands (what the page draws live). No Node, no Python, no script: `bash cuda/build.sh` builds it. `fold` is a wrapper over it; a featurised directory as the first argument,
and `--serve`, are the resident server cuda/worker.py drives for the page, unchanged.

The input options are cuda/af3's exporter's, resolved the same way:
- `--kinds`: one per chain, protein, dna or rna.
- `--ligands`: CCD codes, fetched from the RCSB.
- `--smiles`: built by shared/chem.
- `--modify`: `CODE@position[@chain]`, modified residues and bases.
- `--job`: an AlphaFold 3 job file read by the page's own reader (`web/job-json.js`): its ligands,
  glycans, ions, modified residues and bases, declared bonds and `userCCD`.

All nine of AlphaFold 3's loadable example jobs fold. On the covalent KRAS/sotorasib job, Cys12 SG to
the ligand's C25 is 1.73 Å: bonded, through the declared bond.

`fold` builds `esmfold2` if it is missing and reads the page's own published bundles as they are - the int5
trunk and the int3 ESM-C, fetched once by `cuda/featurise/fetch-weights` from their Hugging Face remotes, 0.35 GB:
their codes go to the device and are decoded there (cuda/af3's `Model::loadBundle`; bit-identical to
decoding them on the host, and the load is 127 against 241 ms for the float32 file).

The quantisation costs accuracy, measured on the gate's cases: 6MRR 0.84 -> 1.42 A (pLDDT 78.3 -> 77.2),
1BRS 0.53 -> 0.92 A, ligand bonds 0.047 -> 0.115 A rms, nucleic 0.036 -> 0.063, KRAS pLDDT 88.1 -> 83.7
with the covalent SG-C25 still bonded (1.79 A). The float32 export (`export_weights.mjs`, 2.9 GB, from
the unpublished float32 bundles) is what the oracle checks below use, through `esmfold2 --weights=`. It then starts `esmfold2` while the input is exported. `esmfold2`
uploads the weights and, during the upload, warms up on a synthetic input of up to 96 tokens. It folds
with `--fast`; flags after `--` go to `esmfold2` (`--seed=`, `--steps=`, `--inputs-window=`).

The weights come from two float32 bundles: `model-esmc-600m-f32`, and `model-esmfold2-conf-f32`.
The second is biohub's trunk plus Synthyra's head:

```
python3 tools/export_esmfold2_trunk.py --esmfold2 esmfold2-fast-600m --out model-esmfold2-conf-f32 \
    --confidence ~/.cache/huggingface/hub/models--Synthyra--ESMFold2-600/snapshots/*/model.safetensors
```

`esmfold2` refuses weights without a head, as the page does.

## The released models: ESMFold2-Fast and ESMFold2

The same binary folds biohub's two released checkpoints, which the page does not ship (their tower is ESM-C 6B):

```
cuda/esmfold2/fold 6mrr.pdb --model=esmfold2-fast --sequence=GWSTELEKHREEL...
cuda/esmfold2/fold 6mrr.pdb --model=esmfold2 --sequence=GWSTELEKHREEL... --a3m=<one A3M per protein chain>
```

What they add over the experimental tier:
- the parcae recycle. z starts as truncated-normal noise. Each of four passes refines a dropped-out copy of the
  language model's pair (p 0.25, at inference too) through a 4-block lm_encoder, injects it into a per-channel
  decay of z, and runs the trunk. Then a readout and a 2-block coda.
- ESM-C 6B, kept resident as af3-any-model's int8 codes (6.4 GB) and expanded a layer at a time.
- AF3's 64-bin distogram, its sampler constants (14 steps clipped at sigma 256), and a PAE LayerNorm in the
  confidence head.
- the alignment's profile and mean deletion in s_inputs (`msaFeatures` in the bundle). With no alignment, the
  profile is one row of the query, as biohub's builder makes it. The experimental tier zeroes both
  (`disable_msa_features`), and its bundles predate the field.
- ESMFold2 only: 48 trunk blocks, and an MSA encoder (4 blocks, 128 channels) over z_init whose output replaces
  the injection. Each block is an outer product mean, `Wout(outer) / max(pair count, 1)` with the bias divided
  too; then, except on the last block, a pair-weighted averaging and a SwiGLU transition; then the pair-only
  block. Neither its inputs nor z_init change between passes, so it runs once a fold (src/msa.cuh).

The alignment follows biohub's `esm` builder (prepare_input.py, paired_msa.py):
- a protein chain's rows come from its A3M, or from its sequence alone;
- any other token gets its residue type in row 0 and a gap below;
- the deletion value is `(pi / 2) atan(d / 3)`, the vendor's transform (af3-any-model uses AF3's `2 / pi`);
- past 1,024 rows it is subsampled, keeping the query and the A3M's order;
- 10% of the non-query columns are masked, drawn from the seed, as the vendor does at inference.

Rows of different chains are stacked unpaired: the vendor's taxonomy pairing is not done here.
`EF2_DETERMINISTIC=1` turns off every random draw (z noise, LM dropout, column mask, subsample).

The bundles come from the old-format checkpoints (biohub/ESMFold2-Fast at c6c7958d63, biohub/ESMFold2 at
8fc3ff4710: the ones carrying `parcae_*` names). Each folding bundle carries its own language-model shim, because
the two checkpoints' shims differ in all twelve tensors. The tower is shared.

```
python3 tools/export_esmfold2_trunk.py --esmfold2 <ESMFold2 dir> --confidence <ESMFold2 dir>/model.safetensors \
    --out model-esmfold2-f32                                     # 894 MiB; model-esmfold2-fast-f32 is 719
python3 tools/export_esmc6b.py --tower <af3-any-model's esmc.unpacked> --esmfold2 <either dir> --out model-esmc-6b-int8
```

Measured against references:
- ESMFold2, two passes against af3-any-model's `esmfold2` (float32, dropout off, `EF2_DETERMINISTIC=1`): the
  trunk's pair agrees to relRMS **1.89e-3**, the same as ESMFold2-Fast's 1.7e-3.
- The MSA encoder at 256 rows of a real alignment, against biohub's own `MSAEncoder` fed native's z_init and
  s_inputs: **3.5e-7**. The rows and deletions are identical to the vendor's `construct_paired_msa`.
- 5CAJ from its sequence: **1.97 Å**, pLDDT 97.4 (ESMFold2-Fast 1.69-1.91, 92.3).
- Warm on the A100, the encoder costs 19 ms at 261 tokens with one row and 44 ms with 1,024 rows, against a
  470 ms trunk.

**On a T4: ESMFold2 folds up to about 1,750 tokens.** That is measured on a simulated T4 (the T4's shared-memory
limit, plus a second process holding all but 14.6 GiB of the A100); 2,000 runs out. Two things a short card does
that a large one does not, both decided by free memory (`roomFor`) and both bit-identical (relRMS 0 through
`LOCALFOLD_BIG=1`):
- **the tower leaves the device after the language model.** ESM-C 6B's 6.4 GB of codes are idle until the next
  fold, which reads them back from the shards (`Model::parkResident` / `unparkResident`; 0.84 s here, from the
  page cache). Without this, 1,500 tokens ran out.
- **the language model's pair waits in pinned host memory**, copied into the injection once a pass. The loop
  otherwise holds four pairs (z_init, z, the injection and this), 3.1 GB each at 1,750 tokens. z itself is
  allocated only after the MSA encoder.

The next wall is the three pairs the loop cannot do without, beside the triangle's scratch. Going past it would
mean float16 pairs.

## Exactness

`oracle.py` runs biohub's forward on the CPU in float32, with its own ESM-C (biohub/ESMC-600M-1500000)
loaded in float32. On CUDA the model autocasts its language model, inputs embedder and trunk to bf16,
and its loader always takes ESM-C in bf16, so neither is a float32 reference. The oracle records
every seam, then runs Synthyra's head on the fold's own trunk pair and coordinates.

`--float32-attention` neutralises the one bf16 cast the module makes whatever the model's dtype: q, k
and v of every atom attention. That is the control `esmfold2 --atom-f32` is held to. 6MRR:

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

**The 256-channel pair track, on fused kernels of its own** (`src/fused256.cuh`). cuda/af3's fused kernels hold
a whole 128-wide tile or weight on the chip; at 256 channels that is 255 registers a thread
or more shared memory than a block gets. Its fused transition measured slower (504 against
466 ms), so these stream instead:

| kernel | does | 261 tokens, 4 passes |
|---|---|---:|
| `triIn256K` | LN, projection and gate, the interleaved a/b split into padded planes, the gating linear | 89 ms (cuda/af3's `triInK` at 256: 110) |
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
- **Every weight starts on 16 bytes on the device** (cuda/af3's loader lays the file out so, for
  all three ports): cuBLAS was choosing its align1 kernels for most of them, and a batched GEMM,
  which cannot see its pointers, faulted. That alone took the sampler from 54 to 32 ms on 6MRR.

Elsewhere:
- The contraction runs on zero-padded f16 planes.
- Every other GEMM uses TF32.
- A CUDA graph of the trunk's passes measured nothing.
- **The pair in bf16 through each run of blocks** (`trunkBlocks`, `ef2Pair16`: Ampere on, the fused 256-channel path,
  not on a card short of room): the language model's encoder blocks, the 24 and the coda work on a bf16 copy of the
  pair converted in before the run and out after it (the recycle's own arithmetic stays f32) - the triangle's
  input and output kernels read and write half the bytes, and the transition writes its gated rows in bf16 so its
  second GEMM runs bf16 throughout and accumulates straight into the pair. 5CAJ, warm: trunk **204.2 -> 190.9 ms
  (-6.5%)**; CA RMSD 2.099-2.118 against 2.095-2.102 A on three seeds, mean pLDDT 90.83 -> 90.69.
  `LOCALFOLD_PAIR_F32=1` keeps the f32 pair. The gate's 5CAJ case 2.102 -> 2.103 A.

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
  cuda/af3's loader, so all three ports read this way. A mapping cost page faults to read, and
  275 ms to tear down when the process exited, after the PDB was written. A file is mapped now only
  when the host reads one of its entries.
- **The warm-up runs while the weights are still arriving** (`M.uploadAsync`). Its kernels read a copy
  still in flight, so its answers are garbage, and every weight derived from the copy is dropped
  afterwards (`forgetDerivedWeights`: the f16 mirrors, the f16 bias copies). The fold is
  byte-identical to one with no warm-up.
- **The warm-up is capped at 96 tokens.** A warm fold the input's size outlasted the upload: 5CAJ's
  261 tokens cost 0.2 s. The warm-up loads kernel modules and cuBLAS plans, and 96 tokens is already
  in the fused kernels' range.

- **`fold` returns once the PDB is written.** On success `esmfold2 --detach-output` prints `ef2: done` and
  closes stdout, and the driver releases the device after the wrapper has returned (0.14 s), as
  cuda/af3's does.

`EF2_STARTUP=1` prints the context and upload times.

## Memory

Peak device memory at 783 tokens (5CAJ's chain three times), `--fast`: **15.9 → 6.8 GB**.
- The ESM-C tower's four matrices a block run on their f16 mirror (`gemmH`), and their f32 device copy
  is dropped once the mirror is built (cuda/af3's `compactWeights`): 2.29 GB, the bulk of the 2.9 GB of
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
python3 cuda/esmfold2/gate.py            # folds against gate-baseline.json, oracles against their bounds
python3 cuda/esmfold2/gate.py --write
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

Bond rms is against CCD ideals, scored by `cuda/af3/bonds.mjs`; it may move 0.01 Å. The covalent
distance must stay under 2.2 Å, which is bonded and not merely near.

The gate also checks every `data-*/` that has an `oracle-f32att/`, in both precisions:

```
cuda/featurise/esmfold2-featurise cuda/esmfold2/data-x --sequence=...
~/venv_ef2/bin/python cuda/esmfold2/oracle.py cuda/esmfold2/data-x --out cuda/esmfold2/data-x/oracle-f32att --float32-attention
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

## How large a fold fits (in progress)

"Short of room" is cuda/af3's free-memory line now (2026-10-07; `shortPair`, common.cuh): 18x the f32 pair
against the room this process had at its first ask, where it was a 64th of the card - 790 residues on 40 GB, ~480 on
a T4. At 1044 residues on the A100 the ordinary paths take the trunk 3424 -> 3197 ms (a cold fold) for a peak of 9.0
GB against 7.8; under a simulated T4 (all but 14.6 GiB held, its shared-memory limit) 780 residues fold on the
ordinary paths in ~6.7 GB. `LOCALFOLD_SHORT_PAIR_TIMES=0` is the old rule. (cuda/af2 keeps the old rule: at 1275
residues its two modes are 9.90 against 9.84 s.)

On a card short of room (`shortPair`): **z_init is streamed** - it is row-local (the language model's pair
term comes from per-token states through a pair MLP, a block of rows at a time), so each recycle makes it
a block at a time and adds it into z, in the stored form's block size and order: byte-identical, and
neither z_init nor the language model's pair (two 256-channel f32 pairs) is kept. The **triangle
multiplication goes in output blocks** (`triangleBlockedEf2`, on cuda/af3/src/triblocked.cuh, shared
with cuda/af2) where the whole form would not fit with room to spare. `LOCALFOLD_BIG=1` forces both;
6MRR 1.422 A either way, and check-cuda-worker passes under it. A simulated T4 folds 1,600 residues.
On the same short card the pair transition normalises a chunk of rows at a time (the whole f16 normalised
pair was 2.95 GB at 2,400), the confidence head's last call normalises the trunk's z in place and parks its
residual in pinned host memory while its blocks run (two 256-channel pairs fewer), and the distogram goes a
block of pair positions at a time (z + z^T and the logits were 5.9 and 2.95 GB whole) - each the same
arithmetic, byte-identical (contacts identical at 600). **A simulated T4 now folds 2,000 residues** (trunk
30 s, CA-CA median 3.793, none out of band). At 2,400 the trunk finishes and the diffusion conditioning's own
256-channel pair (5.9 GB, `dc.pair`) is the next holder - it would want AF3's streamed conditioning, which was
not ported; at 2,800 the triangle's fixed operand (4 GB) no longer fits.
