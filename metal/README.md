# LocalFold on Metal

The Apple-native port, begun 2026-10-08. It exists because WebGPU cannot reach
what Apple silicon has: a stock browser exposes no subgroup matrix units (they
need `--enable-unsafe-webgpu`, on Apple as on NVIDIA - see CLAUDE.md's round
four), WebGPU kernels pay index clamps that Metal's lack of hardware
robustness forces on every access, and the Neural Accelerators in an M5's GPU
cores are reachable only through Metal 4's tensor APIs.

**What it is today: two kernels and the gate that holds them**, the trunk's two
largest - `grid.attend` and `pair-transition` are 2.08 s of a 5.58 s AF3 trunk
pass on an M2 at 256 tokens. No fold runs on Metal yet.

```
bash metal/build.sh
metal/check/check-kernels [--tokens=128,256,400] [--rounds=7]
```

`check-kernels` holds each kernel to a CPU reference written from the
specification and exits 1 when one misses its bound. Watched failing: the
attention with its softmax rescale removed (relRMS 3.1e-1) and the transition
off by 1% (1.0e-2).

## What it is worth, measured on an M2 (10-core GPU, macOS 13.2)

| kernel, 256 tokens | WebGPU, stock Chrome | Metal | |
|---|---:|---:|---:|
| `grid.attend` | 12.21 ms | **4.3-4.5 ms** (~1.9-2.0 TFLOP/s) | ~2.8x |
| `pair-transition` | 19.2 ms in a trunk pass (24.7 standalone) | **9.8-9.9 ms** (2.6 TFLOP/s) | ~2.0x |

At 128 and 400 tokens the attention is ~2.2x. The WebGPU figures are
`tools/gpu/bench-grid-attend-passes.js`, `tools/gpu/bench-transition.js` and a
`bench-trunk.js --profile` pass; they ran in another process on a machine that
drifts, so read a ratio to about 15%.

Where the gain comes from, separated:

- **Leaving WebGPU alone is 1.2-1.7x.** The WebGPU attention's own scalar
  algorithm, written natively, ran 0.93 / 7.28 / 31.0 ms against 1.27 / 12.21 /
  35.92.
- **The 8x8 simdgroup matrix units are the rest**, and only with the softmax in
  registers (`thread_elements()`) and every tile loop fully unrolled - see the
  headers of the two kernels for the arms that lost.
- **f16 buys nothing on the M2's matrix units.** Its MPS multiplies a
  4096-cube at 2.42 TFLOP/s in f32 and 2.77 in f16, and accumulating the
  transition in f16 was no faster and 600x less accurate.
- **The transition is AT the hardware's practical ceiling**: 2.6 TFLOP/s against
  MPS's 2.4-2.8 on its best shape. On the pair track's own tall, narrow shapes
  MPS reaches only 0.7-1.1, so a library is not the route either.

**The trunk, estimated: ~2x** - 5.58 s to roughly 2.7-2.9 s a pass at 256
tokens. The two kernels here are measured; the projections (`grid.project`,
`tri.project` and their outputs, ~36% of the pass) and `tri.contract` are
extrapolated at the transition's rate, and folding the LayerNorms into the
projections - as the transition already does - would take it toward 2.1x.

On an M5 the case is a different one and unmeasured: its GPU has matrix
hardware a browser cannot use at all.

## Layout and conventions

- `kernels/*.metal` - one file a kernel family. Tensors are laid out as the
  WebGPU kernels lay them out (each header says how), so the two backends can be
  compared directly.
- `runtime.h` - Objective-C++: device, queue, the library, pipelines, shared
  buffers. Objective-C++ rather than Swift so the C++ featuriser
  (`cuda/featurise`) and the CUDA port's host logic can be linked as they are.
- `check/` - gates. A number a tool prints is not a gate until something fails
  on it.
- Kernels are compiled from source at start-up (58 ms cold, ~1 ms once macOS
  has cached them): the offline `metal` compiler needs a full Xcode. `build.sh`
  embeds them in the binary as `kernels.inc`.

🔴 **A PIPELINE CAN SILENTLY REFUSE THE THREADS IT IS DISPATCHED WITH.**
`maxTotalThreadsPerThreadgroup` falls as a kernel uses more registers, and a
dispatch past it does not run - no error, the output untouched, and a GPU time
that read as a 2x win in the prototype (a 64-row transition tile, relRMS 1.0).
`Runtime::pipeline` takes the thread count and refuses.

🔴 **`MTLCreateSystemDefaultDevice()` RETURNS nil TO A COMMAND-LINE PROCESS**
here, saying so only on stderr. `MTLCopyAllDevices()` does not.

## Next

1. Prototype `grid.project`, so the trunk estimate is measured end to end.
2. The pairformer block: the remaining pair-track kernels against
   `cpu/af3/trunk/`, then a block against an oracle dump, as cuda/af3 was built.
3. Decide how the native featuriser is shared: it lives in `cuda/featurise` and
   is plain C++ that clang builds, but its directory says CUDA.
