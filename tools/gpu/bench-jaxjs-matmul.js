/**
 * jax-js's WebGPU matmul at the shapes this repository's kernels run.
 *
 *     npm i @jax-js/jax && cp -r node_modules/@jax-js/jax/dist .jaxjs-tmp
 *     node tools/gpu-chrome.mjs tools/gpu/bench-jaxjs-matmul.js
 *
 * (.jaxjs-tmp is gitignored; gpu-chrome.mjs serves the repository root, and
 * jax-js has to be reachable from the page for the same reason src/ is.)
 *
 * WHY. "Would a JS tensor framework close the gap to JAX" is answerable, and
 * jax-js (https://jax-js.com) is the serious candidate: a JAX-shaped API with
 * vmap, jit with operation fusion, and a WebGPU backend that reports over
 * 7000 GFLOP/s on an M4 Max. The question is not whether it is good - it is
 * whether a general framework's GEMM beats a hand-fused kernel AT THE SHAPES
 * A FOLD ACTUALLY RUNS, which are small in one dimension and where this
 * repository's own generic GEMM only reaches 17.6% of the device ceiling.
 *
 * 🔴 THREE THINGS HAD TO BE RIGHT BEFORE THE NUMBER MEANT ANYTHING, and each
 * wrong version of this bench reported a figure that would have libelled the
 * library. Reading the result matrix back measures this platform's ~1 GB/s
 * readback and not the kernel - the 30208x1024 output is 124 MB, 124 ms of the
 * 172 the first version reported. Calling without `jit` re-traces and
 * re-synthesises every iteration: 0.49 TFLOP/s on a 2048 cube against 4.29
 * with it. And `blockUntilReady` is the force, not `.data()`.
 *
 * 🔴 AND THE `--dtype=float16` ARM DOES NOT TAKE. It reports numbers identical
 * to f32 to three digits, which means the argument is not reaching the dtype
 * rather than that half precision is worth nothing here. The comparison below
 * is f32 against f32 and should not be read as anything about jax-js in f16.
 *
 * 🔴 IT BRINGS ITS OWN DEVICE. jax-js calls init() and takes a GPUDevice of
 * its own, so this does NOT run on the device gpu-chrome.mjs hands in - the
 * two are separate adapters on the same card. That is fine for a throughput
 * comparison and would not be fine for interop, which is the practical
 * obstacle to using it for one stage of a fold.
 */
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const SHAPES = [
  // The diffusion transformer's qkvg. This repository gets 21% of the ceiling
  // here with a fused kernel and 17.6% with its best generic GEMM.
  { name: "difftx qkvg", m: 240, k: 768, n: 768 },
  // AF2's MSA transition, where the same generic GEMM reaches 81%.
  { name: "AF2 transition", m: 30208, k: 256, n: 1024 },
  // A square one, for a number comparable to anybody's matmul benchmark.
  { name: "square 2048", m: 2048, k: 2048, n: 2048 },
];

export async function main(device, args) {
  const rounds = Number(option(args, "rounds", "9"));
  const jax = await import("/.jaxjs-tmp/index.js");
  const { numpy: np, init, defaultDevice, jit, blockUntilReady } = jax;
  const devices = await init();
    const backend = option(args, "device", "webgpu");
  if (!devices.includes(backend)) return { error: `no ${backend} backend`, devices };
  defaultDevice(backend);
  const gpu = await jax.getWebGPUDevice?.();
  const adapter = { vendor: gpu?.adapterInfo?.vendor, architecture: gpu?.adapterInfo?.architecture };

  const median = (v) => [...v].sort((a, b) => a - b)[v.length >> 1];
  const rows = [];
  for (const { name, m, k, n } of SHAPES) {
    // 🔴 jit, NOT A BARE CALL. Without it every iteration re-traces and
    // re-synthesises the kernel, which is what a first version of this bench
    // measured and reported as 0.49 TFLOP/s on a 2048 cube - a number about
    // this bench and not about jax-js.
    const f = jit((x, y) => np.matmul(x, y));
    // 🔴 AND f16 IS AN ARM, because the device has shader-f16 and jax-js
    // supports it - comparing its f32 against a hand-tuned f32 kernel and
    // stopping there would not be a fair reading.
    const dtype = option(args, "dtype", "float32");
    const a = np.ones([m, k], dtype);
    const b = np.ones([k, n], dtype);
    const once = async () => {
      const start = performance.now();
      const c = f(a.ref, b.ref);
      // 🔴 AND blockUntilReady, NOT .data(). Reading the matrix back measures
      // this platform's ~1 GB/s readback: the 30208x1024 output is 124 MB,
      // which was 124 ms of the 172 the first version reported.
      await blockUntilReady(c);
      const ms = performance.now() - start;
      c.free?.();
      return ms;
    };
    for (let i = 0; i < 3; i += 1) await once();   // compile and warm
    const times = [];
    for (let i = 0; i < rounds; i += 1) times.push(await once());
    const ms = median(times);
    const gflops = 2 * m * k * n / ms / 1e6;
    rows.push({ shape: name, m, k, n, ms: Number(ms.toFixed(4)),
      tflops: Number((gflops / 1000).toFixed(2)),
      percentOfCeiling: Number((100 * gflops / 1000 / 18.1).toFixed(1)) });
    a.free?.(); b.free?.();
  }
  return { devices, backend, adapter, note: "ceiling is this device's measured 18.1 TFLOP/s f32", rows };
}
