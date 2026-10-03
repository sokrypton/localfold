# AlphaFold 2 in CUDA

A native CUDA/cuBLAS AlphaFold 2, monomer and multimer, held stage by stage to af3-any-model's
JAX AF2 on exactly the same input. It runs that reference's ONE graph: every checkpoint goes
through the multimer network, a monomer's parameters converted at export the way the reference
converts them at load (`alphafold3/af2/convert.py`). What differs between the two is a regime
written beside the weights: position scale 10 against 20, the outer product mean after the MSA
stack against before it, and which template embedder runs.

The input is the page's own: `export_input.mjs` calls `makeA3mFeatures`
(src/input/a3m-features.js), the function the page and `tools/gpu/fold-af2.js` fold with, once per
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
(src/input/mmseqs2-api.js). For a complex, each distinct chain is searched, the paired block is
added for distinct chains, and the merge follows the weights' regime: dense within an entity and
block-diagonal between entities for the multimer, block-diagonal throughout for a monomer.
Barnase-barstar from its two sequences folds to **0.584 A, ipTM 0.92**, against 17.4 A without an
alignment, and the search takes 3.5 s of a 5.7 s command. It sends the sequences to
api.colabfold.com, so it is a flag, never a default.

`--model` is `model_1_ptm` (default) or `model_1_multimer_v3`, the two the page publishes whole
(models 2-5 are published as int3 deltas on them: `af2 --bundle=../../model --delta=../../model-mono-3-delta
--map=maps/model_3_ptm.map` reads one as the page does, bit-exact against src/bundles/delta-tensor-store.js,
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

`fold`, sequence to PDB in a cold process: 6MRR **0.59 s** (110 ms of it the fold), and 5CAJ with
its 7907-row alignment 2.47 s. On success `af2 --detach-output` prints `af2: done` and closes stdout,
so `fold` returns while the driver releases the device (0.16 s). Weights are read with `pread`
rather than mapped (native/af3's loader).

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
