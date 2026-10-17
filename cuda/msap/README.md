# MSA Pairformer in CUDA

A native CUDA/cuBLAS port of MSA Pairformer (Akiyama et al.; [yoakiyama/MSA_Pairformer](https://github.com/yoakiyama/MSA_Pairformer),
weights [yakiyama/MSA-Pairformer](https://huggingface.co/yakiyama/MSA-Pairformer)). It takes an alignment and returns:
- residue contacts: Cβ, after layer 15, and ConFind, after layer 18;
- the query's 26-token language-model logits;
- optionally, the embeddings.

The model is 22 layers, 111.4 M parameters:
- a 464-wide MSA track and a 256-wide pair track;
- each layer's MSA update is the pair-weighted averaging (AF3's Algorithm 10) followed by a SwiGLU transition;
- the pair is updated by a query-biased outer product (sequences weighted by a differential attention against the query row), then the two triangle multiplications and a transition;
- there is no triangle attention.

Every module but the outer product, the input embeddings and the heads is one ESMFold2's MSA encoder already has. This port runs cuda/ef2's kernels unchanged. `tools/export_msap.py` writes the checkpoint in ESMFold2's names and layouts:
- linears input-major;
- each SwiGLU's halves swapped (MSA Pairformer gates the second half);
- the triangle projections interleaved;
- the incoming triangle's a and b swapped, because its contraction is Σₖ a[k,i] b[k,j] where ESMFold2's is Σₖ a[k,j] b[k,i].

## Use

```
bash cuda/build.sh msap                                  # cuda/msap/localfold-msap
~/venv_msap/bin/python tools/export_msap.py              # cuda/msap/weights (needs torch; the checkpoint is fetched
                                                         #   by MSAPairformer.from_pretrained(weights_dir=~/msap/weights))
cuda/msap/localfold-msap --a3m=x.a3m --out=x/ --fast     # x/cb_contacts.npy, confind_contacts.npy [L][L],
                                                         #   logits.npy [L][26], subsample.a3m
```

Options:
- `--chain-breaks=i,j` gives the residue each chain after the first starts at. A ColabFold `#len1,len2` header gives them too. The token index is shifted by 100 from each break on, as the reference does.
- `--embeddings` adds `query.npy` [L][464] (the final MSA's query row) and `pair.npy` [L][L][256].
- `--contacts-only` stops after layer 18: the same contacts, about 15% faster.
- `--depth=N` sets the subsample's depth (default 512, the reference's).
- Without `--fast` every activation is float32, the reference arm.

## The subsample

The reference (`MSA_Pairformer.dataset.MSA`) does two things:
1. It runs hhfilter (`-id 90 -diff N -cov 70 -qid 30 -qsc -20`).
2. It then greedily adds the row with the largest mean Hamming distance to the rows already chosen, until it has N rows.

Here, step 1 is coverage ≥ 70% and identity to the query ≥ 30% (msa-tools' `filter`, [sokrypton/msa-tools](https://github.com/sokrypton/msa-tools)). Step 2 is the reference's own greedy, on the device with integer sums. It takes 40-72 ms for up to 21,008 rows.

hhfilter's `-id 90 -diff` is left out. The table is the reference model's top-L long-range Cβ contacts (|i-j| ≥ 24, Cβ < 8 Å against the crystal), with top-L/2 in brackets:

| subsample | 5CAJ | 1TIM | 5K9P |
|---|---|---|---|
| the reference's (hhfilter + greedy) | 0.885 (0.946) | 0.891 (0.959) | 0.853 (0.892) |
| this one | 0.858 (0.954) | 0.895 (0.943) | 0.813 (0.919) |
| the alignment's first 512 rows | 0.724 | 0.781 | 0.240 |

The greedy stage does the selecting. A redundancy cut before it is quadratic in the depth.

## Accuracy

`tools/oracle/dump_msap.py` runs the reference in float32 (TF32 off, the plain PyTorch triangle) and writes:
- the subsample's tokens;
- every module of layer 0 and the pair and MSA after chosen layers;
- the heads.

`localfold-msap --oracle=<dump> --check` reads the same tokens and compares every stage. The numbers below are relative RMS:

| 5CAJ, 512 × 261 | float32 | `--fast` |
|---|---|---|
| layer 0, each module | ≤ 9e-7 | ≤ 3e-4 |
| pair after layer 21 | 1.2e-6 | 8.3e-4 |
| Cβ contacts | 4.9e-7 | 7.8e-4 |
| ConFind contacts | 5.7e-7 | 7.7e-4 |
| logits | 2.1e-7 | 1.4e-4 |

For comparison, the reference's own bfloat16 is 5.3e-3 from its float32 on contacts (1B70).

1BRS A:D, which has two chains with a break, matches as closely: pair init exact, contacts 4.9e-7.

## Speed and size (A100 40 GB, warm, the same rows both sides)

The reference is run in bf16 with cuEquivariance, its README's settings. The time is the best of three warm calls. The native peak is the whole process on the card (nvidia-smi), weights and CUDA context included; the reference's peak is PyTorch's allocator. Where the card has room the triangles take faster whole-plane forms, so a native peak is what a run took on an otherwise empty 40 GB card, not what it needs.

| rows × residues | this port, `--fast` | the reference | speed-up |
|---|---|---|---|
| 512 × 247 (1TIM) | 175 ms, 2.5 GB | 437 ms, 2.9 GB | 2.5× |
| 512 × 261 (5CAJ) | 191 ms, 2.6 GB | 480 ms | 2.5× |
| 1,024 × 247 | 304 ms, 3.2 GB | 756 ms, 5.1 GB | 2.5× |
| 2,048 × 247 | 565 ms, 3.8 GB | 1,398 ms, 9.5 GB | 2.5× |
| 4,096 × 247 | 1.09 s, 5.5 GB | 2.71 s, 18.3 GB | 2.5× |
| 512 × 1,023 (1JZ7) | 1.27 s, 7.4 GB | 3.46 s, 13.0 GB | 2.7× |
| 512 × 1,040 (1B70, two chains) | 1.34 s, 7.6 GB | 3.28 s, 13.4 GB | 2.4× |
| 1,024 × 1,023 | 1.86 s, 9.2 GB | 4.97 s, 20.8 GB | 2.7× |
| 2,048 × 1,023 | 2.99 s, 12.6 GB | out of memory | |
| 512 × 2,046 (1JZ7 twice, two chains) | 4.45 s, 17.1 GB | out of memory | |
| 1,024 × 2,046 | 5.73 s, 20.5 GB | out of memory | |
| 512 × 3,069 | 10.1 s | out of memory | |
| 512 × 4,092 | 21.1 s, 27.6 GB | out of memory | |
| 512 × 5,115 | 36.0 s, 38.1 GB | out of memory | |

With all but 15 GiB of the card held by another process (a T4's room, though a T4 itself keeps the pair f32), 512 × 3,069 runs in 16.1 s.

What the `--fast` path does (each step's numbers are in its commit message):
- **The pair is bf16** from its first value to the last layer, as cuda/ef2 keeps ESMFold2's (`--pair-f32` keeps it f32). Contacts on 5CAJ move 7.8e-4 -> 3.2e-3 from the float32 reference, still inside the reference's own bf16 (5.3e-3).
- **The MSA transition's SwiGLU is fused into its up-projection** (`swiglu.cuh`). It is a GEMM shaped as cuBLAS's own kernel for this product, the gate applied in registers; the 3,712-wide intermediate never reaches memory. 133,632 rows: 2.61 ms against cuBLASLt plus the SwiGLU pass's 3.61.
- **The pair-weighted averaging's per-head sum is one GEMM a head** (`pwaHeadK`). It reads the values in place and gates them on the way out, at 149 TFLOPS where cuBLAS's per-(sequence, head) form ran 35. Values and gates come from one GEMM.
- **The query-biased outer product:**
  - k, left and right come from one f16 GEMM;
  - the scores LayerNorm k's halves on the fly;
  - the contraction reads both operands as they lie, a block of rows i at a time;
  - its projection, SwiGLU and residual are one kernel reading the product in place (`opmProjectK`).
- **Weights:** the MSA side's f32 weights are dropped after their f16 copies are made (338 MB).
- **Scratch:** the MSA side's scratch is given back before a short card's pair block, whose blocked triangle wants a whole operand plane (13.4 GB at 5,115).
- **Tried and not taken:** the transition in chunks small enough to stay in L2 (the GEMMs lose more than the cache gains).

The pair block (two triangle multiplications and a transition) is cuda/ef2's, unchanged. It is 21% of a 261-residue pass and 44% at 1,023.

## License

The weights are under the authors' modified MIT license ("coffee edition", pip `msa-pairformer` 1.0.2; the Hugging Face card names a "pizza-ware" license). They are not redistributed here: export them from the authors' own release.
