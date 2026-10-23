# CUDA Optimization Notes from BindCraft2 (`ColabDesign2/dev/native_block`)

During the optimization of the BindCraft2 native CUDA port (`ColabDesign2/dev/native_block`, which shares `flash.cuh` and AF2 trunk/structure module shapes with `localfold`), we profiled and verified several optimizations on Ampere (`sm_86`, NVIDIA A10) that directly apply to `localfold` (`cuda/af2`, `cuda/af3`, and `cuda/ef2`). We also ported `localfold`'s `ipaPairTermK` (1-block-per-query across all 12 heads) back into BindCraft2's `ipa.cuh`, which eliminated 12x redundant pair HBM reads and dropped `ipa::gatherK` from `14.05 ms` to `< 4 ms`.

Below are the concrete optimizations from our work that can be applied directly to `opus-55-opt`:

---

## 1. `flashGridHalf`: Unpadded XOR-Swizzled Shared Memory (`fa2Ldk<D>()` / `fa2Kv<D>()`) + Running-Pointer `issue()` (`-12.2%` kernel time, bit-identical)

**Where:** [`cuda/af3/src/flash.cuh:204-335`](../af3/src/flash.cuh) (used by `cuda/af2`, `cuda/af3`, and `cuda/ef2` whenever `MASKED = true` or `D = 48`).

**What:**
In `cuda/af3/src/flash.cuh`, `flashGrid2R` (lines 1265–1385) uses:
- `fa2Ldk<D>()` (`D` when `D == 32` instead of `D + 8`) and `fa2Kv<D>(r, c)` (`r * D + (c ^ (((r >> 1) & 3) << 3))`), eliminating the `+8` shared-memory padding on `K` and `V` stages while avoiding bank conflicts on `cp.async` 16-byte stores and `ldmatrix.x4` / `ldmatrix.x4.trans` loads.
- Running pointers `kvCur[KV_PER]` and `bCur[B_PER]` advanced by `kvStep` / `BK` inside the `!last` fast path of `issue()`.

However, `flashGridHalf` (lines 204–490) still uses the old padded `LDK = D + 8` (`faStage` at line 205) and base+offset `kvBase[k]` / `bBase[k]` indexing (lines 279–315). Because **every masked attention** in `cuda/af2/src/evoformer.cuh` (when `seqMask != nullptr`), `cuda/af3`, and `cuda/ef2` dispatches to `flashGridHalf` rather than `flashGrid2R`, porting `fa2Ldk<D>()` / `fa2Kv<D>()` and running-pointer `kvCur` / `bCur` into `flashGridHalf`:
1. Shrinks `faStage<32, WARPS, BK>()` by `2 * BK * 8 * 2` bytes per stage (`3,072` bytes across double-buffered stages at `BK = 48`, or `4,096` bytes at `BK = 64`).
2. Cuts 2 registers per thread and eliminates per-tile 64-bit address offset calculations on all non-final tiles.
3. **Measured on A10 (`sm_86`)**: `261.26 ms -> 229.46 ms` (`-31.8 ms` / **`-12.2%`**) across 208 launches at `N = 384`, **bit-identical**.

```cpp
// Move fa2Ldk / fa2Kv above faStage and flashGridHalf:
template <int D> __host__ __device__ constexpr int fa2Ldk() { return D == 32 ? D : D + 8; }
template <int D> __device__ __forceinline__ int fa2Kv(int r, int c) {
  if constexpr (D == 32) return r * D + (c ^ (((r >> 1) & 3) << 3));
  else return r * (D + 8) + c;
}
template <int D, int WARPS, int BK = FA_BK> __host__ __device__ constexpr size_t faStage() {
  return (size_t)2 * BK * fa2Ldk<D>() * 2 + (size_t)(16 * WARPS) * (BK + 8) * 2 + BK * 4;
}
```
Then inside `flashGridHalf`:
- Set `constexpr int BQ = 16 * WARPS, LDK = fa2Ldk<D>(), LDB = BK + 8, NT = WARPS * 32;`
- Store `kvOff[k] = fa2Kv<D>(jj, c)` during setup and replace `K + nt * 8 * LDK + (lane & 7) * LDK + ...` and `V + ...` in `ldsm4` / `ldsm4t` with `fa2Kv<D>(...)` (matching `flashGrid2R`).
- Replace `kvBase[k]` / `bBase[k]` with running pointers `kvCur[k]` (stepped by `pos(BK)`) and `bCur[k]` (stepped by `BK`).

---

## 2. Single-Pass Register-Cached + Vectorized Warp LayerNorm (`layerNormTK` & `lnStatsK`)

**Where:**
- [`cuda/af2/src/fast.cuh:223-236`](src/fast.cuh#L223-L236) (`layerNormTK`)
- [`cuda/af2/src/structure.cuh:270-282`](src/structure.cuh#L270-L282) (`lnStatsK`)

**What:**
Currently, `layerNormTK` and `lnStatsK` iterate `for (int c = lane; c < C; c += 32)` with scalar float loads and re-read `xr[c]` from global memory in 2–3 separate loops (mean pass, variance pass, and normalize+write pass).
- In `lnStatsK` (where `C = 128` for the pair representation or `C = 384` for the single representation, i.e. at most `4` or `12` floats per lane), caching the values in a fixed register array during Pass 1 (when `C == 128` or `C == 384`) avoids the second global read of `pair` (`[N, N, 128]` = `75 MB` at `N = 384`) and uses `float2` / `float4` coalesced loads.
- Note on summation order: to remain **100% bit-identical** to `for (int c = lane; c < C; c += 32)`, keep the per-lane accumulation at stride 32 (`v[k] = xr[lane + k * 32]`) before the warp `__shfl_xor_sync` tree!

---

## 3. Register-Buffered Half-Warp Channel-Major `centerNormHK` (`mov.b32` Packed 16-bit Trick, `-43.5%` kernel time)

**Where:** [`cuda/af3/src/elementwise.cuh:165-178`](../af3/src/elementwise.cuh#L165-L178) (`centerNormHK` general fallback used by 64-channel template stack and T4 paths when `C != 128 && C != 256`).

**What:**
In `centerNormHK`, when the shared-memory transpose specialization (`C == 256` or `C == 128`) is not taken (e.g. `C == 64` in the template pair stack), each thread walks `for (int c = 0; c < C; ++c)` over channel-major planes (`xs[c * M]`) **twice** from HBM — once to compute `(mean, rstd)` and a second time to normalize and write `out[r * C + c]`. When `[C, N, N]` exceeds L2 cache, the second loop re-reads the entire tensor from HBM.

Splitting `C` channels across a half-warp (`2` threads per pair position `r`, each holding `C / 2` channels) and loading 2 adjacent positions (`r0, r0 + 1`) as a packed 32-bit word (`uint32_t`, containing `half2` or `__nv_bfloat162`) allows caching all `C / 2` channel pairs in registers (`uint32_t vbuf[C / 2]`) with **zero register spills**.

**Crucial `ptxas` detail:** If you unpack `half2` / `__nv_bfloat162` in Pass 1 and unpack the same C++ variable again in Pass 2, `ptxas` CSE promotes the unpacked `float` values live across the warp reduction, doubling register usage and causing local-memory spills. Reading the register buffer in Pass 2 through an opaque inline PTX move prevents CSE expansion while costing 0 instructions:
```cpp
__device__ __forceinline__ __half2 u32ToHalf2(uint32_t raw) {
  __half2 v;
  asm volatile("mov.b32 %0, %1;" : "=r"(*(uint32_t*)&v) : "r"(raw));
  return v;
}
```
On A10 (`sm_86`), this cut channel-major center LayerNorm stats/normalization time by **`-43.5%`** (`33.43 -> 18.88 ms`) with `0` spill stores/loads.

---

## 4. Destination-Contiguous Indexing in Transposed Orientation Epilogues (`swapAddK`)

**Where:** [`cuda/af2/src/evoformer.cuh:86-93`](src/evoformer.cuh#L86-L93) (`swapAddK`).

**What:**
In `swapAddK`:
```cpp
template <class T, class TO = T>
__global__ void swapAddK(const T* x, TO* out, int L, int C) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)L * L * C;
  if (idx >= total) return;
  int c = idx % C; size_t ij = idx / C; int i = ij / L, j = ij % L;
  out[((size_t)j * L + i) * C + c] = (TO)((float)out[((size_t)j * L + i) * C + c] + (float)x[idx]);
}
```
Here, `idx` indexes `(i, j, c)`, so the read-only load `x[idx]` is contiguous, while **both** the load AND the store of the read-modify-write `out[(j * L + i) * C + c] += ...` (2/3 of the HBM transactions, including all dirty L2 writebacks) are strided by `L * C`!
By interpreting `ij = idx / C` as the **destination** index `(j, i)` (`out[idx] += x[(i * L + j) * C + c]`):
- `out[idx]`'s read and write become **100% sequential** across warps/blocks.
- Only the single read-only load `x[((size_t)(ij % L) * L + (ij / L)) * C + c]` is strided.
- Every output element is **100% bit-identical** (same float addition `out + x` at every `(j, i, c)`).
- **Measured on A10 (`sm_86`)**: `-19.3%` on transposed RMW epilogue (`21.37 -> 17.25 ms`).
