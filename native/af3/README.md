# AlphaFold 3 in CUDA

A native CUDA/cuBLAS port of this repository's AlphaFold 3, transcribed stage by
stage from the CPU references under `src/af3/` (the specification) and checked
against af3-any-model's own oracle dumps in `oracle-dumps/`.

Status: **the pairformer (all 48 blocks) runs and matches AF3.** The rest of the
model is not ported yet; see "Next" below.

## Build and run

```
cd native/af3
nvcc -O3 -std=c++17 -arch=sm_80 --default-stream per-thread -DUSE_FP16 pairformer.cu -lcublas -o pairformer16

python3 ../../tools/serve.py 8791 &        # the exporters read bundles over HTTP
node --js-float16array --max-old-space-size=16000 export-pairformer.mjs data48   # f32 bundle, 569 MiB
python3 index.py data48

./pairformer16 data48 --bf16 --graph --repeat=3            # AF3's own 6MRR input, vs AF3's output
./pairformer16 data48 --tokens=1024 --bf16 --graph         # timing at any size (random input)
./pairformer16 data48 --tokens=256 --bf16 --stages         # per-stage profile
```

`--bf16` selects the 16-bit path; with `-DUSE_FP16` that path is IEEE half,
otherwise bfloat16. `--no-fused` is pure FP32 (the accuracy reference), `--tf32`
TF32 linears. `export-block.mjs` exports one block plus the CPU reference's
output (`--real` for AF3's input, otherwise random) for checking a single block.

## Accuracy: 48 blocks on 6MRR (68 tokens), against AF3's `trunk_out_pair`

f32 weights, AF3's own pairformer input (`tap.trunk_in_pair`, `tap.trunk_in_single`):

| path | pair | single |
|---|---|---|
| FP32 | 4.2e-7 | 1.9e-7 |
| TF32 | 2.7e-4 | 4.7e-5 |
| **FP16** | **2.95e-4** | 1.9e-4 |
| BF16 | 2.4e-3 | 4.7e-4 |
| WebGPU, shipped (docs/AF3.md, f32) | 2.98e-4 | |

BF16 is no faster than FP16 here and its 7-bit mantissa drifts to 8x the error
over 48 blocks; FP16 lands on WebGPU's error. 🔴 A RANDOM N(0,1) INPUT IS
USELESS FOR THIS: through real weights it reads 1-3e-2 in any 16-bit format
while AF3's real input reads 3e-5 - always check on `--real`/the oracle.

## Speed: 48 blocks, A100-SXM4-40GB

| tokens | WebGPU (dev flags) | CUDA FP16 | |
|---|---|---|---|
| 64 | 51 ms | 15.3 ms | 3.3x |
| 128 | 87 ms | 36.9 ms | 2.4x |
| 256 | 274 ms | 142 ms | 1.9x |
| 512 | 1202 ms | 601 ms | 2.0x |
| 1024 | 6635 ms | 2952 ms | 2.2x |

WebGPU here is `bench-trunk.js --msa=1` steady pairformer with the developer
flags; a stock-Chrome NVIDIA visitor gets about half that. No 2 GiB binding
ceiling: 2500 and 3000 tokens run (18 and 25 GB).

## What made it fast (in order)

1. Scratch buffers cached, not `cudaMalloc`'d per call: 13.7 -> 3.7 ms/block at 64.
2. Grid attention over all rows at once where it fits.
3. Tensor cores. TF32 first; then FP16 end to end - LayerNorm and centre-norm
   write 16-bit directly, intermediates stay 16-bit, residuals stay f32.
4. The grid attention as a FlashAttention-2 kernel (`flashGridAsync`): S, P and O
   in registers via `mma.sync` m16n8k16 (WMMA's fragment layout is unspecified),
   P's accumulators reused as the A operand of P V, cp.async double-buffered
   K/V/bias tiles, ldmatrix (.trans for V), scores in the log2 domain. One
   direction at 1024 tokens: ~13 ms, against PyTorch's cutlass attention 17.9 ms.
5. Fused GEMMs: q/k/v/gate as one, triangle projection+gate as one; residual
   adds folded into GEMMs with beta = 1.
6. CUDA graph replay (small sizes).

## Next

The rest of AF3, each against the oracle: embedder + target_feat, template
embedder, MSA stack, recycling and distogram, diffusion (conditioning, atom
encoder/decoder, 24-block transformer, sampler), confidence head.
