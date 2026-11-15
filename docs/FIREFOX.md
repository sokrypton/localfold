# Firefox

Firefox's WebGPU is wgpu and naga, not Dawn and Tint: another validator and another WGSL compiler over the same
shaders. No gate here had run through it until a reader reported "very strange protein structures" (issue #10,
2026-11-13). Measured 2026-11-14 on an Apple M2, macOS 13, Firefox 157.0.1 against Chrome 157; on Linux the reader's
fold was not WebGPU at all - see "Linux" at the end.

## Running anything in Firefox

```
LOCALFOLD_GPU_FIREFOX=1 node tools/gpu-chrome.mjs tools/gpu/<module>.js [--flags]
```

The same harness, a visible window, a fresh profile (`LOCALFOLD_FIREFOX` names the binary). None of the Chrome
flags apply, so it is always the stock configuration. What the adapter offers on the M2: `shader-f16` yes,
`subgroups` **no**, no matrix units, 8 storage buffers a stage, 2 GiB bindings, 32 KiB of workgroup storage. So
Firefox takes the kernels a device without subgroups takes - `LOCALFOLD_GPU_WITHHOLD=subgroups,subgroup-size-control`
asks Chrome to select the same ones (`withholdFeatures` on `requestAlphaFoldDevice`).

## What was wrong: two defects, one silent

Before the fixes, through the fold tools on the local bundles:

| model | Firefox | cause |
|---|---|---|
| the seven AF3-lineage models, OpenDDE | a fold that is not a chain (OpenBind-0: N-CA 1.08 A, CA-CA 6.57), **no error** | the atom attention's softmax |
| ESMFold2 | NaN coordinates, **no error** | its LayerNorm and softmax reductions |
| AlphaFold 2, the multimer | refused at pipeline creation; NaN once that compiled | the f16 bitcast, then the LayerNorm preamble |
| BioEmu2 | refused at pipeline creation | the f16 bitcast |

On the page the AF3 lineage is the default, so a Firefox reader got a structure with a confidence number and
nothing saying it was wrong - `target_feat`, the first tensor of the trunk, was already wrong (rms 0.067 against
0.404), so the distogram was off as well as the atoms.

### 1. One lane storing what the rest of the workgroup reads

```wgsl
if (local == 0u) { reduce[0] = max(m0, m1); }   // one lane finishes the reduction
workgroupBarrier();
let largest = reduce[0];                        // every lane reads it
```

In Firefox on Apple silicon the other lanes get `reduce[0]` as it stood BEFORE lane 0's store - on about half
the lanes of four workgroups in five in isolation, on all 63 in the kernel. The softmax then exponentiates
against too small a maximum and overflows. Chrome computes every shape correctly on the same GPU.

`tools/gpu/probe-barrier-visibility.js` is the reproduction, sixteen shapes of "few lanes write, every lane reads
after a barrier" with a known answer. What it says for Firefox: a single lane's store to a FIXED address in
workgroup memory (a constant index, a scalar) is not seen; a store indexed by the lane, by a value the compiler
cannot fold, or into a storage buffer is.

🔴 **AND THE PROBE IS NOT THE WHOLE RULE.** Rewriting `row_stat[0] = ...` as `row_stat[lane] = ...` passes the
probe (a one-element array included) and still failed in AlphaFold 2's LayerNorm, where the same slot is stored
and read twice in one kernel (the mean, then the inverse deviation): the second read returned the first value.
So the kernels were fixed in the form that depends on no compiler's view of another lane's store, wherever the
shared value is cheap:

- **every lane computes it** - the eight partials folded by each lane (`atom-encoder.js`'s attention, the
  LayerNorm preamble of `kernels/attention.js`, `kernels/transition.js`, `kernels/outer-product-mean.js`,
  `af2/structure/iteration.js`). The same arithmetic in the same order;
- where one lane must do serial work for the rest (57 stores in ESMFold2, ESM-C, OpenDDE's readouts, BioEmu2,
  the multimer's template), the slot is **named by the writing lane** in an array a lane can index (never a
  one-element array, whose every index clamps to 0), written once and read once.

Chrome's results are unchanged by it: AlphaFold 2 -1287409, ESMFold2 -133814, OpenBind-0 59.0216051723824 before
and after. The multimer moved in the last place (-891782 to -891784).

### 2. naga refuses a bitcast that changes the element count across f16

`bitcast<vec4<f16>>(vec2<u32>)`, `bitcast<vec2<f16>>(u32)`, `bitcast<u32>(vec2<f16>)` and their kin are how the
packed-f16 kernels read and write their operands, and naga types them wrongly and then refuses the use ("the
type of `v` is expected to be `vec2<f16>`, but got `f32`"). `tools/gpu/probe-wgsl-support.js` lists what a
browser's compiler takes. `webgpu/runtime/wgsl-compat.js` asks once per device (`recordCompilerSupport`) and,
where they are refused, `ComputePipelineCache` and RFD4-Proteina's engine compile the same text with the
bitcasts spelled as `unpack2x16float` / `pack2x16float` - the same halves, exactly. A browser that takes the
bitcast is given the text it always was. `test/wgsl-compat.test.js` holds the rewrite and fails on a spelling
it does not cover.

A refused pipeline now says why: `compileNamed` appends the compiler's message and the line to "ShaderModule
with '<label>' label is invalid", which was all there was.

## After

Every family through its fold tool, Firefox against Chrome on one machine (pLDDT, or the tool's checksum):
AF3 68.87 / 68.87, OpenBind-0 59.01 / 59.02, Boltz-2 61.09 / 61.06, Protenix-v2 59.44 / 59.50, IntelliFold-2
53.73 / 53.74, RoseTTAFold3 71.74 / 71.74, Mol-1 76.52 / 76.53, OpenDDE 84.699 / 84.696, AlphaFold 2 62.906 /
62.910, multimer 51.272 / 51.251, ESMFold2 -133808 / -133814, BioEmu2 worst CA-CA 3.892 / 3.892. The page folds
OpenBind-0, Boltz-2, AlphaFold 2, the multimer and EF2-fast in stock Firefox.

## Speed: four things, none of them the arithmetic (2026-11-14)

A reader's question - AlphaFold 3 "seems slower on Firefox" - measured: `fold.js` on the 58-residue gate sequence,
int5 bundle, 25 steps, warm fold, M2, Firefox 160 against Chrome 157: **5.2 s against 1.33**. After the first three
changes below **1.6 s** (1.5 with the fourth), the first fold 1.9-2.1 s against Chrome's 2.0, and the same structure on every page load
(it was not, see 3).

| arm (Firefox, warm fold) | s |
|---|---:|
| as it was | 5.2 |
| + the Apple prior | 3.6 |
| + waits pumped | 2.7 |
| + short loops unrolled, the startup race skipped | 1.6 |

**1. Firefox's `adapter.info` is blank**, vendor and architecture both "", so no prior matched and an M2 ran as a
device nobody had measured. `identityOfUnnamedAdapter` (webgpu/runtime/device-profile.js) names it from what Firefox
does report: a Mac whose adapter offers ASTC texture compression is Apple silicon (an Intel Mac has BC alone; a
phone and a Windows-on-Arm part have ASTC and are not Macs, and stay unrecognised).
`tools/gpu/probe-adapter-identity.js` prints every route a page has. Missing `subgroups` is not a cost: Chrome with
them withheld folds in the same 1.33 s.

**2. Firefox reports finished GPU work on a 100 ms tick.** `tools/gpu/probe-readback-latency.js`: a 256-byte copy
awaited by `mapAsync` or `onSubmittedWorkDone` is 99-101 ms every time, Chrome 0.2 ms. A fold waits on a readback
every sampler step, so a denoiser call of 45 ms was 100. The device looks when something is submitted: an empty
`queue.submit([])` while the wait is open brings the answer in 1-8 ms; a `writeBuffer`, a `createBuffer`, another
`onSubmittedWorkDone` and ONE empty submit straight after the ask do not. `webgpu/runtime/completion-pump.js`
wraps the two waits of a Firefox device (known by its compiler) so an empty submit goes out every
half millisecond while one is open. 🔴 The turns are MessageChannel messages, not timers: a timer turn in a window
that is not in front read 5 to 682 ms. `LOCALFOLD_GPU_NO_PUMP=1` is the arm without.
🔴 **And nothing is measured to decide it.** The pump used to turn on where one startup wait took over 30 ms; the delay is not always 100 ms - 70 was read,
and now and then under the bar - and a session that measured short folded unpumped for its whole life: the page's
default example (59 residues, 7664 MSA rows, 4 passes) 11 s on a first fold where a pumped session is 3, `fold.js`
6.3 s against 3.2, with the unpumped arm's stage signature to the millisecond. One in seven page loads did it here.

**3. Every loop iteration costs, whatever is in it.** `tools/gpu/probe-loop-cost.js`, 1536 iterations a lane, GPU
clock, Firefox against Chrome:

| body | Firefox loop | Firefox unrolled x8 | Chrome loop | Chrome unrolled x8 |
|---|---:|---:|---:|---:|
| arithmetic only | 60.9 ms | 16.1 | 23.9 | 14.0 |
| one storage read | 52.7 | 15.7 | 14.6 | 15.6 |
| one workgroup read | 53.0 | 18.7 | 9.5 | 13.4 |
| the denoiser's dense loop (both, one multiply-add) | 61.7 | 28.2 | 18.6 | 22.5 |

So it is the loop and not the reads or their bounds checks (naga guards each loop it emits against never ending,
and the guard is in the iteration; the translator's output was not read - installing it was not permitted here - so
that mechanism is the known one, not one seen). `withUnrolledLoops` (webgpu/runtime/wgsl-compat.js) rewrites an
innermost `for` whose count is constant and whose body is short and has no barrier - eight copies an iteration,
each with its own `let i`, the remainder as straight code - for that compiler alone (`compilerText`; Chrome's dense
loop is SLOWER unrolled). One denoiser call, GPU time: Firefox 81.6 ms -> 40.8, Chrome 34.8; the call 100 -> 49 ms
against 45. 🔴 Only a short body: the triangle contraction's loop over k (two barriers and a 4x4 tile an iteration)
went 2.7 ms to 16 unrolled, which is what `UNROLLED_BODY_LIMIT` and the barrier rule are for.
`--unroll-loops` / `--no-unroll-loops` on any GPU tool are the arms; `tools/gpu/dump-shaders.js` prints the text a
tool compiles.

**4. A workgroup array over 8 KiB costs every workgroup before it runs.** `tools/gpu/probe-workgroup-init.js`, 4096
workgroups that touch 64 floats of an array, by the size the array is DECLARED at:

| declared | Firefox | Chrome |
|---|---:|---:|
| 2048 floats (8192 bytes) | 0.98 ms | 0.40 |
| 2052 floats | 36.2 | - |
| 4096 floats | 108 | 0.47 |
| 8192 floats | 428 | 1.8 |
| two arrays of 2048 | 2.0 | - |
| rows of 2048 nested in one variable, 4096 | 108 | - |

About 3 ns a byte a workgroup past the line (WGSL zeroes workgroup memory at entry; past 8 KiB it is done slowly),
by ARRAY and not by kernel, and nesting does not help. ESMFold2's hidden LayerNorm stages 8 rows of 260 floats, 8320
bytes: **108 ms of a 284 ms trunk pass where Chrome's is 3** (`tools/gpu/time-shader.js`, the kernel alone: 3.17 ms
against 0.18, and 0.27 rewritten). OpenDDE at 384 channels has twenty-two such kernels, AlphaFold 3 two, AlphaFold 2
none. `withSmallWorkgroupArrays` (the same file) cuts each such array into parts of at most 8 KiB behind a getter and
a setter and rewrites every use; an array used in a way it does not rewrite is left whole. The arithmetic is
untouched, so the folds are the same to the digit.

Warm folds after all four, Firefox against Chrome (the 58-residue gate sequence, 25 steps where there are steps):

| model | Firefox before 4 | Firefox | Chrome |
|---|---:|---:|---:|
| AlphaFold 3 | 1.6 s | 1.5 | 1.33 |
| OpenDDE | 11.8 | 4.6 | 3.85 |
| ESMFold2 | 1.29 | 0.89 | 0.66 |
| Boltz-2 | 2.08 | 2.05 | 1.76 |
| AlphaFold 2 | 1.15 | 1.15 | 0.89 |

Chrome's ESMFold2 trunk pass is 152 ms with its startup race's answers and 198 without; Firefox's is now 187, level
with the second (see Open for why the first is not available to it).

**And the startup race is not run in Firefox** (webgpu/af3/trunk/grid-attend-race.js, which an Apple GPU runs once
to choose between kernel forms an M2 and an M5 split on). Naming the device Apple switched it on, and Firefox's
pass timestamps cannot call it: one kernel read 12.8 to 47.7 ms over six passes (Chrome 9.09 to 9.34), the answers
changed from load to load, and with them the fold's kernels, its speed (1.8 to 3.3 s) and its last digits (pLDDT
68.74 to 68.88; `check-af3-block-any.js --repeat=6` showed one process agreeing with itself, so it was the choice
and not a race in a kernel). Timed three passes to a stamp the arms are near ties that still fall either way, for
2-4 s of racing. So in Firefox the race's knobs stay unanswered - the form every kernel shipped in before the race.

An AlphaFold 3 trunk pass at 128 tokens and 64 rows is at parity (1345 ms against 1348); what is left of a fold's
gap is the sampler (a denoiser call 49 ms against 45) and the first fold's compiles.

## Open

- **An M5 in Firefox** takes the M2's forms (no race there), and nothing has measured it (the M5 has no Firefox).
- **What is left is spread evenly.** An AlphaFold 2 block is 23.1 ms against 17.1 and Boltz-2's denoiser call 52.9
  against 43.3, with no kernel standing out (pass timestamps blur there, but the totals do not). Two things tried
  for it and not kept: unrolling the loops whose start or bound is not a constant (88 of AF2's 115; nothing), and
  racing in Firefox (its f16 projection does not beat f32 there - 11.4 against 9.7 ms - so the race's answers are
  the defaults it already has, and Chrome's 198 -> 152 ms on ESMFold2's trunk is not there to be had).

- **Linux** was the WebGL2 path and is the next section; Firefox's WebGPU on Linux (Nightly) is not measured.

## Linux: the report was the WebGL2 path, on the desktop OpenGL compiler (2026-11-15)

The reader of issue #10 was on Ubuntu, in Firefox and in Chromium, and neither fix above reached them. Firefox
160 for Linux has **no `navigator.gpu`**, and a stock Linux Chrome has WebGPU on few drivers, so both folded
through `webgl/` - and on Linux a browser's WebGL2 is the vendor's desktop OpenGL driver, which no gate had run:
this harness puts WebGL on Vulkan (`--use-angle=vulkan`), the Macs go through Metal.

Reproduced on the A100 through NVIDIA's OpenGL driver (`LOCALFOLD_GPU_ANGLE=gl-egl`, below), AlphaFold 3, 48
blocks, 25 steps, the 59-residue gate sequence:

| | CA-CA median (lowest - highest) | pLDDT | |
|---|---|---:|---|
| as it was | 3.344 A (1.927 - 3.696) | 72.4 | not a chain, **no error** |
| fixed | 3.856 A (3.798 - 3.925) | 56.1 | the WGSL fold's |

The trunk's pair was 1.38e-1 from the WGSL fold's and is 1.41e-4, the same as through Vulkan.

**The cause is one line, in every attention kernel.** A key is masked the way every reference masks it,

```glsl
float logit = score + 1e9 * (mask - 1.0);     // adds exactly zero where the mask is one
```

and NVIDIA's GLSL compiler reassociates it into `(score + 1e9 * mask) - 1e9`. A float near 1e9 moves in steps
of 64, so every logit of a LIVE key was rounded to a multiple of 64 and the softmax ran over what was left:
`check-webgl-kernels.js`'s attention arms read 0.28-0.41 against 1e-5. GLSL ES has no `precise`, so the constant
is **selected** instead of summed - `score + (mask == 1.0 ? 0.0 : 1e9 * (mask - 1.0))`, the same value for every
mask, one addition the compiler cannot rearrange - in `webgl/kernels/attention.js`, `attention-flash.js` (three),
the atom attention (three), AF3's MSA weights and AF2's invariant point attention (1e5).
`test/webgl-mask-select.test.js` holds the rule and fails on each of the twelve lines put back.

```
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json LOCALFOLD_GPU_ANGLE=gl-egl \
  node tools/gpu-chrome.mjs tools/gpu/fold-af3-webgl.js --seams       # WebGL2 on desktop GL, WGSL on Vulkan
LOCALFOLD_GPU_FIREFOX=1 node tools/gpu-chrome.mjs tools/gpu/check-webgl-kernels.js --no-webgpu
```

`--no-webgpu` runs a module with a null device, which is the only way to run anything in a browser that has no
WebGPU; the adapter line then names the WebGL renderer. After the fix every `check-webgl-*` gate reads the same
on NVIDIA's OpenGL as on Vulkan (45 pass; the seven that do not are this A100's own, identical on both).

**A second pass for the same class (2026-11-15), nothing further found.**

- *WebGL2 on NVIDIA's OpenGL against WebGPU, family by input* (`fold-af3-webgl.js --seams --recycles=1 --steps=25`):
  AlphaFold 3, OpenBind-0, Boltz-2 and Protenix-v2, each single sequence, the 8076-row alignment, + GOL, two
  chains, + a DNA chain and SEP@3, and IntelliFold-2 single - 25 folds, the trunk pair 7.1e-5 to 1.7e-4 from the
  WGSL fold's in every one, the masks that are really zero (padding, a second chain, atom tokens) included.
  RoseTTAFold3, Mol-1 and the rest of IntelliFold-2 were not run.
- *No other shader sums a large constant into a live term*: the remaining ones are selects (`... ? x : -1e9`), a
  clamp and a bin edge.
- *Firefox's own WebGPU on Linux* is off in release 160 and a reader can turn it on
  (`LOCALFOLD_FIREFOX_PREFS=dom.webgpu.enabled=true,gfx.webgpu.ignore-blocklist=true` with `LOCALFOLD_GPU_FIREFOX=1`;
  wgpu over NVIDIA's Vulkan here). `probe-barrier-visibility.js` passes every shape, and each family folds at
  stock-flag Chrome's numbers on the same card (pLDDT, Firefox / Chrome): AlphaFold 3 86.88 / 86.89, Boltz-2
  96.34 / 96.32, Protenix-v2 84.78 / 84.78, RoseTTAFold3 81.53 / 81.52, OpenDDE 92.046 / 92.045, AlphaFold 2
  62.930 / 62.924, multimer 51.272 / 51.249, ESMFold2 73.347 / 73.346, BioEmu2 worst CA-CA 3.892 / 3.892.

### Speed on that driver: a fresh texture is the cost, so the pool stops making them

A 262-residue AlphaFold 3 fold (48 + 4 blocks, 2 passes, 25 steps, single sequence) through WebGL2 on NVIDIA's OpenGL
was **79 s** where the same draws through ANGLE's Vulkan on the same A100 were 10.9 s and WebGPU is 4.1 s; at 59
residues the two WebGL2 paths were level (3.9 / 3.8 s), so it is not the arithmetic. The first draw into a texture
nothing has written costs **0.86 ms a MiB** on that driver (110 ms for 128 MiB, 19 more to delete it) against 0.08
through Vulkan - `texImage2D` and `texStorage2D` alike, and a clear of our own before the draw moves nothing, so it
is the driver's allocation and not the browser's zeroing. At 262 tokens the pool's 512 MiB could not hold what a
pairformer block frees and asks for again, so every block created ~1.5 GiB of textures.

Two changes, neither a measurement of the device (a start-up timing with a threshold was tried first and dropped):

- **A pooled texture serves any shorter tensor** (`Storage.anyFit`). The pool matched a shape exactly, so every
  shape's spare sat idle at once. A kernel addresses a tensor by element index and draws its own rectangle, so a
  taller texture works as it stands. The one thing that refused this before - a draw into several targets needs
  attachments of one size (0x8CD9) - is settled where the targets are bound: an undrawn target takes a texture of
  the others' height (`Storage.retexture`, ~90 a fold).
- **The bound follows the fold**: 512 MiB or thirty-two pair tensors, whichever is more (`af3PoolLimitBytes`) -
  1.07 GiB at 262 tokens, 512 MiB up to ~180.

262 tokens, 8 trunk blocks, NVIDIA's OpenGL - seconds at GiB of peak, exact-match pool / any-fit:

| bound | exact | any-fit |
|---|---|---|
| 512 MiB | 12.8 at 1.43 | 9.4 at 1.65 |
| 768 MiB | 10.2 at 1.70 | 3.9 at 1.84 |
| 1 GiB | 5.5 at 1.97 | **2.9 at 2.09** |
| 4 GiB | 3.1 at 2.49 | 2.8 at 2.13 |

The whole 262-residue fold: **79 s -> 12.4 s** on NVIDIA's OpenGL and 10.9 -> 10.0 s through Vulkan, the peak
1.74 -> 2.42 GiB on both; 59 residues level (3.8 s) at a slightly lower peak (1006 -> 963 MiB). Same pLDDT and the
same trunk seams against WGSL in every arm. `fold-af3-webgl.js --pool-mib=512 --no-any-fit` is the fold as it was.
Any-fit alone does not make 512 MiB enough at that length: what a block frees is more than that.

🔴 **And an Apple renderer keeps the pool as it was** (`af3PoolFollowsFold`, 2026-11-15). On an M2 through ANGLE's
Metal a fresh texture is cheap, so the same 262-token fold (48 + 4 blocks, 25 steps) is 28.3 s at a 1.64 GiB peak
with the exact-match 512 MiB pool, 30.9 s at 1.96 with any-fit under the same bound (a taller texture under a
shorter tensor is live bytes) and 27.0 s at 2.29 with both rules - the same structure in all three, in Chrome and
in Firefox with `--no-webgpu`. `fold-af3-webgl.js --any-fit --pool-mib=<n>` is the arm with, there.

🔴 **What was not run.** Mesa on Intel or AMD hardware, and Firefox on a hardware driver: under Xvfb both
browsers get Mesa's llvmpipe unless EGL is pointed at the vendor's, which Chrome takes and Firefox does not.
On llvmpipe the kernel gates pass in Firefox and in Chrome before and after (Mesa does not reassociate this),
and a whole fold is refused loudly - a float render target is `FRAMEBUFFER_UNSUPPORTED` once ~480 MiB of weight
textures are live. The reader's GPU is not known; an NVIDIA card is the one case shown to give their symptom.
