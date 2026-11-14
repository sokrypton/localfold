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

## Open

- **Linux.** The report was Ubuntu, Firefox and Chromium. Defect 2 is the compiler's and is the same on Linux;
  defect 1 was measured on Metal only, and Chromium on Linux is not explained by either (it may be the WebGL2
  path - the status line says which backend folded). Ask a Linux reader for the status line and `about:support`.
- The 57 lane-named stores rest on measurement (every model's fold, both browsers), not on a rule the probe can
  state. A new kernel should take the every-lane form.
- No standing gate runs Firefox; `LOCALFOLD_GPU_FIREFOX=1` on the fold tools is the check.
