# Firefox

Firefox's WebGPU is wgpu and naga, not Dawn and Tint: another validator and another WGSL compiler over the same
shaders. No gate here had run through it until a reader reported "very strange protein structures" (issue #10,
2026-11-13). Measured 2026-11-14 on an Apple M2, macOS 13, Firefox 157.0.1 against Chrome 157; Linux Firefox
(Vulkan) has not been measured from here.

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
measures the delay once a device and, where it is over 30 ms, wraps the two waits so an empty submit goes out every
half millisecond while one is open. 🔴 The turns are MessageChannel messages, not timers: a timer turn in a window
that is not in front read 5 to 682 ms. `LOCALFOLD_GPU_NO_PUMP=1` is the arm without.

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
- **RFD4-Proteina does not run in Firefox**: every kernel of `webgpu/rfd4p/engine.js` says `enable subgroups;` and
  shares its row statistics by `subgroupShuffleXor`, and Firefox offers no `subgroups` (naga: "not yet
  implemented"). The engine refuses by name now instead of with "ShaderModule ... is invalid"; a form without
  subgroups is a port of nineteen shuffle sites, not a rewrite of text. (Its weights' address was also dead in every
  browser - the github.io URL redirects without CORS - and is the site's own domain now.)
- **What is left is spread evenly.** An AlphaFold 2 block is 23.1 ms against 17.1 and Boltz-2's denoiser call 52.9
  against 43.3, with no kernel standing out (pass timestamps blur there, but the totals do not). Two things tried
  for it and not kept: unrolling the loops whose start or bound is not a constant (88 of AF2's 115; nothing), and
  racing in Firefox (its f16 projection does not beat f32 there - 11.4 against 9.7 ms - so the race's answers are
  the defaults it already has, and Chrome's 198 -> 152 ms on ESMFold2's trunk is not there to be had).

- **Linux.** The report was Ubuntu, Firefox and Chromium. Defect 2 is the compiler's and is the same on Linux;
  defect 1 was measured on Metal only, and Chromium on Linux is not explained by either (it may be the WebGL2
  path - the status line says which backend folded). Ask a Linux reader for the status line and `about:support`.
- The 57 lane-named stores rest on measurement (every model's fold, both browsers), not on a rule the probe can
  state. A new kernel should take the every-lane form.
- No standing gate runs Firefox; `LOCALFOLD_GPU_FIREFOX=1` on the fold tools is the check, and `test/wgsl-compat.test.js`
  and `test/firefox-runtime.test.js` hold the rewrite, the identity rule and the pump without a browser.
