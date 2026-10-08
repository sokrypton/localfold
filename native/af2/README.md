# AlphaFold 2 in CUDA

A native CUDA/cuBLAS AlphaFold 2, monomer and multimer, held stage by stage to af3-any-model's
JAX AF2 on exactly the same input. It runs that reference's ONE graph: every checkpoint goes
through the multimer network, a monomer's parameters converted at export the way the reference
converts them at load (`alphafold3/af2/convert.py`). What differs between the two is a regime
written beside the weights: position scale 10 against 20, the outer product mean after the MSA
stack against before it, and which template embedder runs.

The input is the page's own: `export_input.mjs` calls `makeA3mFeatures`
(shared/input/a3m-features.js), the function the page and `tools/gpu/fold-af2.js` fold with, once per
recycle. So a difference against the browser is the network's, never the featuriser's.

## Run

```
native/af2/fold 6mrr.pdb --sequence=GWSTELEKHREEL...
native/af2/fold 5caj.pdb --sequence=<SEQ> --a3m=oracle-dumps/5caj-a.a3m
native/af2/fold 5caj.pdb --sequence=<SEQ> --template=tools/fixtures/5caj-crystal.pdb:A
native/af2/fold 1brs.pdb --sequence=<A>:<D> --model=model_1_multimer_v3 \
    --template=tools/fixtures/1brs-crystal.pdb:A+D
native/af2/fold 1brs.pdb --sequence=<A>:<D> --model=model_1_multimer_v3 --search
```

`--search` gets the alignment from the ColabFold MMseqs2 server through the page's own client
(shared/input/mmseqs2-api.js). For a complex, each distinct chain is searched, the paired block is
added for distinct chains, and the merge follows the weights' regime: dense within an entity and
block-diagonal between entities for the multimer, block-diagonal throughout for a monomer.
Barnase-barstar from its two sequences folds to **0.584 A, ipTM 0.92**, against 17.4 A without an
alignment, and the search takes 3.5 s of a 5.7 s command. It sends the sequences to
api.colabfold.com, so it is a flag, never a default.

`--model` is `model_1_ptm` (default) or `model_1_multimer_v3`, the two the page publishes whole
(models 2-5 are published as int3 deltas on them: `af2 --bundle=../../model --delta=../../model-mono-3-delta
--map=maps/model_3_ptm.map` reads one as the page does, bit-exact against shared/bundles/delta-tensor-store.js,
and `make_map.py --delta` builds its map), and `fold` folds with the page's own weights: the int5 bundle (`model/`, `model-multimer/`, fetched
once by `native/fetch_bundles.py`), read as it is - its codes decoded on the device - through
`maps/<model>.map`. `make_map.py` builds a map (~/.venv-lfjax): the monomer's parameters are found in
the decoded bundle by value (within int5's half step, a tie settled by the manifest's names) and the
multimer's are joined exactly, every checkpoint element's id run through the page's exporter and the
reference's loader both; each native tensor is then strided parts of bundle tensors. The template
torsions' chi tables are compiled in (`src/chi_tables.cuh`). Chains are joined by `:`. `--template`
takes `<structure>:<chain>[@<query chain>][+...]` (PDB or mmCIF), each part aligned to its chain by
the page's own `buildTemplate` in the atom37 layout and merged into the one slot AF2 takes, as the
page builds it (`1brs.pdb:A+D` is A onto chain 0, D onto chain 1); `--template-search-chains=` with
`--search` takes the search's best hit per listed chain, as the page's "from the MSA search" does,
and `--job=` reads an AlphaFold 3 job with the page's reader. `af2 --tolerance=<A>` stops early by
the page's rule (ColabFold's compute_tol between passes, from the second on), and every fold writes
AlphaFold 3's `<stem>_confidences.json` / `_summary_confidences.json` beside its PDB (expected PAE,
pTM/ipTM, the token layout, contact probabilities where the weights carry a distogram head). The page's monomer bundle carries the templates' single features too (`templateSingle`, float32,
added by `tools/append_template_single.py` without touching a published shard - the exporter's
section scopes had missed them): 5CAJ with its own crystal is 0.217 A / pLDDT 97.22 against
DeepMind float32's 0.205 / 97.4, and 2.531 / 74.5 without them. The int5
cost elsewhere is small - 6MRR 1.900 A / 84.58 against 1.882 / 85.17. Input flags (`--recycles`,
default 3, `--max-msa` 512, `--max-extra` 1024, `--seed`) go to the exporter. Flags after `--` go
to `af2`, which `fold` runs with `--fast`.

## What it computes

The embedder (relpos, recycled pair and first row, the recycled distogram); the extra-MSA stack (4
blocks, global column attention); 48 Evoformer blocks; the template embedder:
- the multimer's: unit-vector and backbone-frame features, a 2-block pair stack, and the
  embedding of a BLANK template even when none is given, because that embedding is not zero;
- the monomer's: a 2-block pair stack, point attention over template slots, and torsion-angle MSA
  rows.

Also the structure module (IPA, 8 layers, side chains through the rigid groups), and the heads:
pLDDT, PAE with pTM and ipTM, the distogram, and masked MSA. The output is a PDB with pLDDT in the
B-factor column, one chain per `asym_id`.

## Exactness

`oracle.py` runs the reference's `RunModel.apply` on the features `export_input.mjs` wrote, in
strict float32 (`jax_default_matmul_precision=highest`, XLA attention). It taps every seam by
re-running with fewer blocks; a 0-block stack is one identity block with its output projections
zeroed, because `layer_stack` cannot be empty. `af2 <data> --oracle=<data>/oracle` compares as it
goes.

| input | float32 atoms | `--fast` atoms | pLDDT |
|---|---:|---:|---|
| 59 residues, 512-row alignment (monomer) | 1.3e-6 | 1.4e-3 | 96.60 both |
| 1BRS A:D, single sequence (multimer) | 2.5e-6 | 5.5e-3 | 36.55 / 36.56 |
| 1BRS A:D with its crystal (multimer template) | 4.5e-7 | 3.9e-4 | 92.86 both |
| 5CAJ with its crystal (monomer template) | 2.0e-5 | 1.2e-3 | 47.41 both |
| model_1_ptm, four recycled passes (`--passes 4`) | 1.2e-6 | | |

Every intermediate seam is 1e-8 to 5e-7 in float32 (the monomer template's torsion rows are
1.5e-5, computed in double because a degenerate dihedral is ill-conditioned in f32).

## `--fast`

- f16 tensor-core GEMMs through cuBLASLt, with bias and ReLU in the epilogue and residual adds as
  beta.
- native/af3's flash attention for every 16- and 32-wide head; the extra stack's 8-wide heads are
  padded to 16.
- q/k/v/gate as one GEMM straight into the flash layout.
- A strided copy of the kernel, so column attention and the ending node read their tensor where it
  lies and write into the residual's layout, with no transposes.
- The triangle's three projections as one GEMM, a channel-major centre norm, and a batched f16
  contraction.
- The MSA transition's second bias carried by the product, through a ones column.
- Masks skipped when they are all ones.
- CUDA graphs from 8 passes up (below that, capture costs more than it saves).

On the A100, 5CAJ (261 residues, 512 + 1024 rows):

| | steady pass | four passes |
|---|---:|---:|
| native `--fast` | **444 ms** | **2.02 s** |
| af3-any-model JAX AF2 (bf16) | 637 ms | 2.55 s |

**Against ColabFold 1.6.3 with its Pallas kernels** (`--use-fast-kernels`), re-measured 2026-10-05 on
this A100: model_1_ptm, 4 passes with early stop off, no relax, exact lengths, 512:1024 alignment
budget, warm (ColabFold's per-model `took` on a repeated length, its compile excluded; native's fold
time after its warm-up pass, median of five; re-run after the 2026-10-05 overnight kernel work). Peak is GPU memory over idle, sampled at 100 ms
(ColabFold with `XLA_PYTHON_CLIENT_PREALLOCATE=false`; its single-sequence figure is the run's peak,
so the 522-residue one). Harness and inputs: `/tmp/claude-1000/af2bench` (the scripts are short; native's is
`native_bench.sh`).

| | 68 res | 261 res | 522 res | 261 + 7,907-row MSA |
|---|---:|---:|---:|---:|
| native AF2, now | **0.086 s** | **0.39 s** | **1.37 s** | **1.46 s** |
| native AF2, 2026-10-03 | 0.11 s | 0.68 s | 2.67 s | 1.73 s |
| ColabFold 1.6.3, fast kernels | 0.2 s | 1.2 s | 4.6 s | 3.2 s |
| native's speed-up | ~2.3x | 3.1x | 3.4x | 2.2x |
| peak, native | 1.2 GB | 1.6 GB | 2.8 GB | 3.2 GB |
| peak, ColabFold | — | — | 3.2 GB | 3.3 GB |

Longer single chains (5CAJ repeated 4, 6, 8 and 10 times; native median of three, ColabFold its warm
second copy, each length its own process):

| | 1,044 res | 1,566 res | 2,088 res | 2,610 res |
|---|---:|---:|---:|---:|
| native AF2 | **6.4 s** | **16.5 s** | **34.3 s** | **61.0 s** |
| ColabFold 1.6.3, fast kernels | 19.4 s | 47.2 s | 80.7 s | 178.1 s |
| native's speed-up | 3.0x | 2.9x | 2.4x | 2.9x |
| peak, native | 7.2 GB | 10.5 GB | 16.7 GB | 24.6 GB |
| peak, ColabFold | 5.8 GB | 9.8 GB | 14.9 GB | 22.7 GB |

Native spends memory the card has whenever that makes it faster. Grid attention runs over every row in
one pass when the device has the room for its buffers and for the triangle multiplication's whole form
beside them (`roomFor`, free memory at the time); it was capped at a fixed 32nd of the card, which chunked
it from ~1,100 residues on 40 GB with 25 GB free - 17.5 / 35.9 / 63.7 s at 1,566 / 2,088 / 2,610 then,
peaks 7.7 / 11.7 / 16.7 GB. Taken alone, without reserving the triangle's room, it starved the triangle
into its blocked form at 2,610 and the fold went to 104 s. The rest of what native keeps or gives back
by length (the embedder's chunks, stage scratch released past a 64th of the card) was measured at no
time either way here, so it stays lean. Every lean form forced (`LOCALFOLD_BIG=1`, which also turns on
the costly ones such as parking the residual in host memory) is 3.6 GB at 11.4 s at 1,044 against 7.2 GB
at 6.4 s. On a smaller card the chunked forms come in by themselves where the memory runs out.

The earlier table had ColabFold's 261-residue fold at 2.1 s; it measures 1.2 s now under either
allocator and the other three of its numbers are unchanged, so that cell was the outlier. Native
uses the page's int5 weights with TF32/f16; ColabFold DeepMind's float32 with bf16 and its kernels;
both fold 6MRR and 5CAJ to within 0.02 A of each other against the crystal.

`fold`, sequence to PDB in a cold process: 6MRR **0.59 s** (110 ms of it the fold), and 5CAJ with
its 7907-row alignment 2.47 s. On success `af2 --detach-output` prints `af2: done` and closes stdout,
so `fold` returns while the driver releases the device (0.16 s). Weights are read with `pread`
rather than mapped (native/af3's loader).

## A deep alignment's featurisation (2026-10-08)

The features are the page's own (shared/input/a3m-features.js, through export_input.mjs). For 5CAJ's 7907-row
alignment one recycle's featurisation was ~440 ms in Node - parsing and encoding the A3M 185, the nearest-centre
search 158, the rest ~95 - and the four recycles ran in four workers that **each parsed the whole A3M again**. Now
the alignment is planned once (parse, encode, profile, every recycle's masking) and the workers take only a
recycle's search and finishing: byte-identical (5CAJ, 1TIM's 16469 rows and the 59-residue fixture), 1.0 -> 0.74 s
here and **1.58 -> 1.25 s (5CAJ) and 2.0 -> 1.4 s (1TIM) on two CPUs**, a Colab T4's.

**And the nearest-centre search runs on the card where the machine has fewer cores than recycles** - the page's
device search (webgpu/input/nearest-centres-webgpu.js) in CUDA (`--nearest=<out>`, a serve job; the zero-byte count and
the first-centre tie of the host loop, integers both ways). The exporter plans and writes the searches
(`--search-out`), the AF2 server assigns them, and the exporter finishes from the assignments (`--assignments`), its
plan held between the two requests by native/export_server.mjs's one process. Byte-identical; through a warm
exporter on two CPUs 5CAJ 0.99 -> 0.82-0.92 s and 1TIM 1.04 -> 0.92-0.99, and with thirty cores ~20 ms slower (the
round trips), which is why tools/native_worker.py takes it only below as many cores as recycles
(`sched_getaffinity`, so a VM's or a `taskset`'s limit counts). A shallow alignment never takes it. Exercised end to
end by the gate's searched AF2 case on two CPUs (`taskset -c 0,1 python3 tools/check-native-worker.py --no-page`).

## Memory

Peak device memory, `--fast`: 783 residues from a single sequence **9.0 → 5.7 GB**; 5CAJ with
512 + 1024 alignment rows 4.5 → 3.5 GB. No slower either way.
- The transitions run in ~128 MB row chunks. The whole widened tensor was 1.26 GB of the pair track
  at 783 residues.
- The transition's f32 buffers are allocated only on the f32 path. Under `--fast` they were made and
  never touched: 1.9 GB at 783.
- The triangle multiplication's five projections are made in row chunks. a and b go to their
  planes and only the output gate is kept whole (`[pairs, 5C]` was 0.78 GB). The centre norm reuses
  the input norm's buffer.

The chunked GEMMs round differently. A converged fold is unmoved (5CAJ 0.000 Å). An unfolded one,
783 residues at pLDDT 34, lands somewhere else, 31.6 Å away, with pLDDT and pTM the same to four digits.
`LOCALFOLD_MEM=1` prints what is in use after the Evoformer.

## Gate

```
python3 native/af2/gate.py            # folds against gate-baseline.json, oracles against their bounds
python3 native/af2/gate.py --write    # re-record after a deliberate change
```

Six folds, scored against the deposited structure with native/af3/score.py:

| case | CA RMSD | pLDDT |
|---|---:|---:|
| 6MRR from its sequence | 1.882 A | 85.17 |
| 5CAJ from its sequence | 21.736 A | 31.93 |
| 5CAJ with its crystal as a template | 0.205 A | 97.39 |
| 1BRS A:D from its sequences | 17.456 A | 37.00 |
| 1BRS A:D with its crystal as a template | 0.287 A | 94.94 |
| 5CAJ with its 7907-row alignment | 1.827 A | 96.07 |

Both arms of each template case are run, because a template proves itself only by moving the
fold. The gate also checks every local `data-*/` that has an `oracle/`, in both precisions, on the atoms
and on the Evoformer's pair: an unconverged fold's structure module amplifies (1BRS from its
sequences turns 3e-4 at its input into 9e-3 on its atoms), so the pair is the seam that separates
a reordered sum from a defect. Those
directories are gitignored and built by hand:

```
node native/af2/export_input.mjs native/af2/data-x --sequence=... [--template=...] --recycles=0 \
    --weights=native/af2/weights-model_1_ptm
~/.venv-lfjax/bin/python native/af2/oracle.py native/af2/data-x --weights native/af2/weights-model_1_ptm \
    --model model_1_ptm --out native/af2/data-x/oracle
```

The oracles compare against af3-any-model on DeepMind's float32 weights, so their arm runs from
`export_weights.py`'s export (`af2 ... --weights=<dir>`), which is not published. The gate fails when its bounds are tightened under the measured figures (all eight oracle arms
and the fold). The baseline is this machine's.

## Traps it cost

- **The flash bias's pad columns must be zero.** The bias is laid out at a stride rounded up to
  8, and the kernel's 16-byte loads read the pad beside the last real column. A reused buffer's
  stale NaN there turned every templated `--fast` fold to NaN, and only a templated one, because
  only the template stack had used the buffer at another head count first.
- **A width typed into a helper is the next stack's bug.** `pairBiasFast` assumed the pair was
  128 wide, and the template stack's is 64. The resulting out-of-bounds read surfaced as cuBLAS
  error 15 several kernels later.
- **The side-chain kernel launched 128 threads over a 256-residue grid**, so it computed the first
  128 residues only. It was invisible on every input shorter than that.

## Shared with native/af3: the fused triangle multiplication

AlphaFold 2's triangle multiplication at 128 channels is AlphaFold 3's with biases, and it had its own
unfused path (LayerNorm, a [C, 5C] GEMM, a gate kernel, the contraction, a centre norm, the output GEMM, a
gated add). It now runs native/af3's fused kernels (src/fusedtriangle.cuh: `triInK`, `triOutPK`) through
their raw-pointer launchers, with a BIAS option those kernels gained (AF3's own instantiations unchanged,
byte-identical folds) and the weights re-laid once (AF3 interleaves a and b's columns where AF2 stores the
halves one after the other: `interleaveTriK`). a and b in f16 and the product in f32, as AF2's path had
them. 262 tokens with 512 alignment rows: 1.80 -> 1.71 s on an A100; every `test:native` AF2 case at its
previous RMSD. The template stack's 64-channel triangle keeps the unfused path.

...and the pair stacks' transition (128 channels) on native/af3's fused transition, in a ReLU-with-bias form
that kernel gained (`fusedTransitionK<..., RELU>`; AF3's SwiGLU instantiation unchanged): the LayerNorm'd
and the 512-wide rows never written. 1% on an A100 (1.71 -> 1.69 s at 262 tokens and 512 rows), where the
GEMMs bind; the bytes it saves are what a T4's or an L4's transition is made of.


...and the pair itself in bf16 (`AF2_P16`, evoformer.cuh), on Ampere and later, where every update both stacks
run has a bf16 form (`af2Pair16Ok`: either fused triangle form or a T4's streaming one, the fused grid attention,
the fused transition, and not a card short of room): native/af3's pair kernels take it through `PAIR16`, set
around a block's pair updates only; the embedder and the templates add into it; the outer product mean's
folded GEMM goes through an f32 block and an add; the row-direction attention output takes `gridOutK` as the
column one does (cuBLAS has no f16-in, bf16-out GEMM). After the stacks it is converted once into the recycled
pair (f32), which the structure module, the heads, the confidences and the next pass's embedder read - so no
f32 pair is ever kept beside it. A100: 5CAJ with 512 + 1024 rows 1676 -> 1664 ms (-0.7%), 494 residues from a
single sequence 1363 -> 1343 (-1.5%); the gate's AF2 rows move in the third decimal (6MRR 1.903 -> 1.901 A,
5CAJ templated 0.214 -> 0.213, 1BRS's delta 0.285 -> 0.283). 🔴 A COLAB T4 IS 1-3% SLOWER WITH IT (9.76 -> 9.79
s, 9.23 -> 9.50): Turing has no f32 -> bf16 conversion instruction, and its biased float-tile triangle output
went 1237 -> 1520 ms of a 494-residue fold - so it is off below sm_80. Tried and not taken: the outer product
mean's product in f16 to halve its add (14 ms faster at 494 residues, and 6MRR from a single sequence 1.898 ->
2.001 A, pLDDT 84.6 -> 81.1). `LOCALFOLD_PAIR_F32=1` keeps the f32 pair. Nor native/af3's cached cuBLASLt plan for the bf16
contraction (`triContractBf16`, the 128x128 tile from np 200 to 352): 5CAJ 1658 -> 1653 ms, a 494-residue fold
1342 -> 1360 (outside the window the plan's cuBLASLt pick, with no workspace, is slower than cuBLAS's own).

...and the post-fold reductions on the device (`paeTmPairsK`, `contactPairsK`): the expected PAE, each pair's TM term
and the distogram's P(< 8 A) were four host passes over pairs x 64 logits with an exp a bin, behind two 17 MB
downloads - 218 ms between a 261-residue fold's last pass and its result, now 69 (a warm templated 5CAJ job through
the worker 393 -> 237 ms). Same double accumulation in the same bin order; every PAE and contact value, pTM and ipTM
identical at the precision the files carry (5CAJ, 1BRS), the gates unchanged.

Measured and left (2026-10-03, A100, 262 residues, 512 alignment rows): the MSA column attention's
strided flash kernel at 8 warps rather than 4 is slower (235 against 224 ms of strided flash a fold), and
its transposed-copy form (the masked path's) runs the attention at the same speed and adds 110 ms of
transposes - the strided reads are not what it spends. At ~73 TFLOP/s (70 GFLOP a call) it is not the
outlier its time share suggests.

## How large a fold fits

On a card short of room (`shortPair`) AlphaFold 2 takes native/af3's approach - each stage gives back what
it alone used, and what is read once is taken in chunks of rows rather than held whole - and the levers
that cost time engage only when the whole form would not fit with an eighth of the card to spare
(`roomFor`). A fold that fits runs exactly as before (6MRR byte-identical); `LOCALFOLD_BIG=1` forces every
path at any size, and the whole of `tools/check-native-worker.py` passes under it.

- **the recycled pair re-embedded in place**: every term of a row (the outer sum, the previous positions'
  distogram, the LayerNorm'd old row, the relative encoding) reads only that row, so no second pair.
- **the triangle attention in chunks of attention rows**: the bias from the pair in chunks, then per chunk
  its LayerNorm, q/k/v/gate, the flash kernel and the output - never the LayerNorm'd pair, the q/k/v/gate
  or the output whole, nor the ending node's transposed copy of the pair (its rows are gathered from the
  columns, its mask read transposed: the flash kernel's row offset indexes only the mask).
- **`attentionCore` in chunks of rows** where its buffers would not fit (the MSA row attention's q/k/v/gate
  grow with the alignment's depth), and the row attention's pair bias from the pair in chunks.
- **the triangle multiplication in output blocks** (native/af3's `triangleBlocked`, with AF2's biases and
  its projection halves one after the other).
- **the structure module's LayerNorm'd pair never stored**: each pair position's mean and inverse (8 bytes)
  and the IPA reads `(pair - mean) * inv * scale + offset` where it needs it - layerNormK's own
  arithmetic, so the same values.
- **the pair heads in chunks**: the PAE logits a chunk of pairs at a time (to the host for the output, into
  the PAE and TM terms for the per-pass frames) and the distogram's contact probabilities a block of rows
  at a time.
- no CUDA graph on such a card (graphs run only at 8+ passes anyway), so a choice made from the free
  memory cannot differ between a pass and its capture.

Measured single sequence, 4 passes: a simulated T4 (`LOCALFOLD_SMEM_LIMIT=65536 LOCALFOLD_FLASH_REG=1` and
a second process holding all but 14.6 GiB) folds **3,500 residues** in 4.8 minutes of A100 arithmetic,
where ~1,300 was its limit; on the A100 3,144 residues peak at 34.9 GB on the whole forms (there is room
for them). At 2,096 residues the big paths against the ordinary ones: pLDDT 20.18 / 20.17, 0.39 A apart
on a fold of pLDDT 20.
