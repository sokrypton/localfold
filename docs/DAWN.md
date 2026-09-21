# The `dawn` branch: folding in a process, and what it cost to find out

The Colab runtime used to be a headless Chrome opening `index.html?role=runtime`
and driving its own controls. It is a node process over Dawn now, and nine
families fold through it with no browser anywhere. This is the record of what
was measured, so the next session does not pay for it twice.

Seventeen commits on `dawn`, branched from `20b065d`. Not pushed, not merged.

## What is verified, and on what

Everything below is a Colab **T4** through `tools/colab_backend.py --runtime
node`, driven from the reader's side over HTTP.

| | |
|---|---|
| AF3 through the bridge | 4,379 progress · 136 status · **25 frames** · 467 atoms · the full twelve-field prediction · typed arrays tagged `Float32Array` |
| EF2-fast through the bridge | **11 frames** · 466 atoms · `chainCertainty 0.62` · no `confidence` object · twice with identical counts |
| a fold on its own (`tools/gpu/fold.js`) | 7.5 s · pLDDT 70.1 · backbone 1.45 / 1.54 / 3.86 A · Rg 10.9 A |
| install, cold, counterbalanced on two fresh boxes | Dawn **6.7 / 6.9 s** against Chrome **15.4 / 16.7 s** |
| whole setup | **25 s** against ~34 with Chrome (the four X11 libraries were Chrome's alone) |
| `shader-f16` on a T4 | 11.4 s -> **7.5 s**, 1.52x, structure identical to the digit (the A100 records 1.74x) |

## What is NOT verified, stated rather than implied

- ~~**The page's own fold path after the extractions.**~~ **VERIFIED on the
  A100**, one sequence per family through `tools/fold-in-page.py`, which drives
  the page's own controls. Nine of ten fold; see the table below. `web/app.js`
  losing ~1,000 lines did not break the page's fold path.
- **AlphaFold 2 in the runtime.** Refused by name. Its compute/display split is
  designed and not written; the seam is in `web/af2-model.js`.
- **The notebook end to end on a fresh runtime.** Its pieces were exercised; the
  cell as a cell was not.

## The page still folds: ten families on an A100

Asked because the branch says it was not asked. `tools/fold-in-page.py --model
<family>`, one 58-mer each, one recycle, four steps - the page's own controls,
not an API:

| family | |
|---|---|
| af3 | 58 residues · 2 s · pLDDT 67.5 |
| opendde | 58 residues · 6 s · pLDDT 87.2 |
| boltz2 | 58 residues · 4 s · pLDDT 66.2 |
| protenix2 | 58 residues · 4 s · pLDDT 61.4 |
| intellifold2 | 58 residues · 5 s · pLDDT 53.0 |
| rosettafold3 | 58 residues · 4 s · pLDDT 72.1 |
| monomer | 1.2 s · pLDDT 64.2 · pTM 0.338 |
| multimer (26:31) | 3.5 s · pLDDT 63.9 · pTM 0.394 · ipTM 0.207 |
| ef2-fast-600m | 58 res · 11 steps · 1.9 s · certainty 0.59 |
| **openbind0** | **fails** - `failed to load tensor .../single_transition/transition2/weights: 404` |

🔴 **AND openbind0 FAILS ON `main` TOO, WHICH IS THE ONLY REASON IT IS NOT A
REGRESSION.** The same command on `20b065d` gives the same class of failure at a
different tensor (`.../transformerk_projection/weights: 404`), so the bundle
this box can reach is incomplete and the extraction is not implicated. A branch
verification without the control would have read as "dawn broke openbind0".

The page's other gates, on the branch: `--drop-job` eight arms true,
`--contact-ui` `kept: true`, `--job-round-trip` `same`/`seedSame` true, and
AlphaFold 3's KRAS/sotorasib job folded from its rows at 189 residues + MOV.
`npm test` 1273, `test:site`. `mobile-layout.py` reports its one standing
failure - the desktop resize handle, bisected on main to the py2Dmol vendor bump
`72d1bcc` and not this branch's.

🔴 **AND `docs/TOOLS.md` WAS STALE ON THE BRANCH**: `colab_runtime.mjs` is a new
tool and the census is gated, so `npm run test:tools` was red here. Regenerated.

## Three findings worth keeping

🔴 **`webgpu@0.6.0`'s Dawn ABORTS an AF3 fold on a T4, and it is f16.** The trunk
completes and the process dies of SIGABRT as the sampler starts - no uncaptured
error, no lost device, nothing reaching JavaScript. Four arms name it:

    f16 + matrix   diffusion   ABORT      trunk 7.9 s
    f16            diffusion   ABORT      trunk 1.1 s
    f16            FLOW        ABORT      trunk 1.5 s
    no f16         diffusion   COMPLETE   pLDDT 70.1

Flow aborts too, so it is not the sampler; dropping the matrix path does not
help, so it is not the matrix kernels. It is `vulkan_enable_f16_on_nvidia`,
which fits Dawn refusing f16 on NVIDIA by default pending a CTS investigation
(crbug.com/42251215). **0.4.0 is pinned**, in the notebook and in `src/node.js`.
If a future Dawn does this again, dropping f16 is a working fallback - it costs
peak memory 590 MiB against 360 and a 7.2 s sampler against 2.8, and it folds.

🔴 **`timestamp_quantization` was destroying every small kernel's measurement.**
Dawn rounds `timestamp-query` to a **65536 ns** grid - not the 100 us usually
quoted. Over 64 dispatches of a ~10.5 us kernel: quantised, the gcd of the
deltas is 65536 and **63 of the 64 read ZERO**; unquantised, gcd 32 ns and 21
distinct values. Every kernel in `tools/gpu/` under ~65 us was being swept
against noise. Now disabled in `tools/gpu-chrome.mjs` and `createNodeDevice`.

🔴 **A 300 ms command poll starves a weight download on two CPUs.** EF2-fast's
shard fetch reported "0 bytes arrived ... after three attempts", and the same
job with a poll beside it aborted natively (`std::system_error: Invalid
argument`). `serveCommands` backs off to 2 s while folding. The fast poll bought
nothing there anyway: the command loop cannot be heard during a fold because the
thread is held, which is why the fold ceiling exists.

## Dead ends - measured, and not worth re-running

- **`disable_robustness`**: null. Six alternated arms, minimum of three each -
  trunk 1.0 against 1.1, diffusion 2.9 against 2.9, fold 7.3 against 7.3. It
  would trade clamping for corruption.
- **The subgroup-matrix path is not 7x slower.** That reading was 0.6.0's broken
  f16. On 0.4.0, alternated: trunk 1.3 s with it, 1.0 / 0.9 without. And it is
  not evidence against `src/runtime/device-profile.js`'s `turing` prior either - that was
  measured at 300 tokens where `grid.attend` is 15.4% of the pairformer, and
  this is a 58-mer. **Do not touch that table on the strength of a 58-mer.**
- **Four wrong causes for the EF2 failure**, each eliminated by an arm rather
  than an argument: the loader (succeeds standalone, with and without the
  224 MiB tower), the `timestamp_quantization` flag, `navigator.gpu` being unset
  by `createNodeDevice`, and `createNodeDevice` itself (one core dump that did
  not reproduce).
- **`chromium-experimental-subgroup-matrix` is a FEATURE, not a toggle** - it is
  `allow_unsafe_apis` that exposes it.

## The arrangement this branch leaves behind

Three graphs, one convention. Every assembly is `(result, about)` and returns
the object every surface reads; no DOM, no viewer, no downloads in any of them.

| graph | orchestration | assembly |
|---|---|---|
| AF3, 7 families | `foldAf3` - `web/af3-model.js` | `predictionFromAf3` |
| ESMFold2, 2 | `foldEsmfold2Job` - `web/esmfold2-model.js` | `predictionFromEsmfold2` |
| AlphaFold 2 | `foldWithAf2` - still a PAGE function in `web/app.js` | `predictionFromAf2` - `web/af2-model.js` |

`fold(event)` went from 895 lines to 336: it dispatches now rather than
containing a model.

## Traps that cost real time here

- 🔴 **`pkill -f` ON THE BACKEND'S OWN NAME KILLS THE KILLER** - the shell running it has
  that string in its own command line. A stale runtime survived a "restart" and
  obeyed every command beside the new one: two `fold-begin` events for one
  press, and the stale process's error ended the watch while the real fold ran
  on. `pgrep -fc` counts itself the same way and reported two runtimes where
  there was one. Read `ps` once and signal by pid.
- 🔴 **node buffers `console.log` to a FILE, and a crashing process loses it.**
  A late abort then reads as a hang at startup. Write marks with `fs.writeSync`.
- 🔴 **A `colab exec` is serial and its client times out at 120 s.** Long work
  goes detached with its output to a file; queuing more reads behind a slow one
  just stacks them.
- 🔴 **`node --check X && echo "ok"` chained off the wrong command** printed
  "ok" while the check was failing. Use an explicit `if`.
- 🔴 **A comment naming a path from another repository fails
  `comments name a module that does not exist`.** The free-variable tool used
  while cutting these functions lives in the py2Dmol tree, and citing it by
  path here is a dead pointer the gate catches - correctly.
- 🔴 **A free-variable scan's count is an upper bound, not a parameter list.**
  It cannot tell module scope from function-local: 27 of 37 names needed no
  passing when AF2 came out of the handler, and 13 of 30 before that.

## Verifying it yourself

    colab new --session x --gpu T4
    # clone main, run the notebook's own SETUP, upload this branch's files
    python3 tools/colab_backend.py --port 8710 --token T --runtime node
    # then drive /in and /down from the reader's side; see the runs above

`--runtime chrome` still works and is the fallback for a machine with no usable
Dawn - this repository's own Mac is one, where the prebuilt `dawn.node` wants a
newer macOS than it runs. **Nothing on this branch is reachable from the local
test lanes**, which is why every number above names the box it came from.
