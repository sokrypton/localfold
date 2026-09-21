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
- ~~**The notebook end to end on a fresh runtime.**~~ **RUN, on a fresh Colab
  T4, and it found a bug that fired on every WARM re-run.** See below.
- 🔴 **AND THE QUESTION NOBODY HAD ASKED IS THE ANSWER: the node runtime folds
  ONE PROTEIN CHAIN and nothing else.** Measured, not read. See below.

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
`npm test` 1273, `test:site`, and `mobile-layout.py` is green again - see below.

🔴 **THE LAYOUT GATE WAS RED FOR A MECHANISM CHANGE, NOT A BROKEN PAGE.** It
asserted that `#canvasContainer .resize-handle` is NOT hidden at desktop, as a
guard that the narrow-only rule had not leaked. py2Dmol's slot layout (vendor
bump `72d1bcc`, on main) moved the resize onto the slot **body** - `resize:
both`, with its own `::-webkit-resizer` hidden - and hides the old handle at
EVERY width through a stylesheet **its JavaScript injects**:
`.py2dmol-slot-body .resize-handle { display: none !important }`. That is in
neither `.css` file, which is why grepping for it found nothing; the engine's
own `CSS.getMatchedStylesForNode` named it in one call.

Measured at 1200px: the slot body is `resize: both`, so a reader could drag the
corner the whole time. The vendored files are not ours to edit, so the fix was
never to put the handle back - the gate asks for the **capability** now
(resizable on a desktop, not on a phone), which is what a reader needs and
survives upstream moving the corner again. Watched failing by pinning
`resize: none !important` at the vendor's own specificity.

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

## 🔴 WHAT THE NODE RUNTIME ACTUALLY PORTS: ONE PROTEIN CHAIN

Asked as "is every part of the website available through the Colab backend",
and answered by driving the broker's own `/in` and `/down` on a T4 rather than
by reading the source. Every row is a fold that happened: AF3 int5, 58-mer,
4 steps, one recycle, seed 42, against a `baseline` arm of the same.

| what the reader asks for | through `--runtime node` | evidence |
|---|---|---|
| one protein chain | **folds** | 472 atoms, 58 CA, pLDDT 44.97, 4 frames |
| the same, `seed: 7` | honoured | coordinates differ from baseline |
| **two copies of it** | **REFUSED** | `: is not an amino acid code` |
| **two different chains** | **REFUSED** | same |
| **any ligand** (`CA`, `TAC`) | **REFUSED** | same |
| **a SMILES ligand** | **REFUSED** | `:, 1 are not amino acid codes` |
| **a contact row** | **REFUSED** | `:, 1, 2,  , -, B are not ...` |
| an RNA chain | refused, wrong reason | `U is not an amino acid code` |
| **a DNA chain** | 🔴 **FOLDS AS A PEPTIDE** | `ACGT...` -> **ALA CYS GLY THR**, 111 atoms |
| the MSA mode the page sends | **ignored** | byte-identical to baseline |
| a template | **ignored** | byte-identical to baseline |
| the sampler (`af3-mode`) | **ignored** | byte-identical to baseline |
| AlphaFold 2 monomer / multimer | refused BY NAME | its own message, correctly |
| ESMFold2 | folds | 11 frames, no confidence object |

🔴 **THE REFUSALS ARE ALL ONE LINE, AND IT IS THE GUARD ITSELF.**
`runFold` builds `sequence = chains.join(":")` over EVERY entity and hands that
to `af3SequenceProblem`, whose alphabet is the twenty amino acids - so the
colon refuses every multi-entity job before the ligand or the bond is ever
reached. The page's own guard, two files away, is
`chains.filter((_, i) => chainKinds[i] === "protein").join("")` with a comment
saying exactly why both halves are there: a ligand-only fold has no sequence,
and `T` is correct for the row it is in. The runtime has the same call with
neither half.

🔴 **AND THE DNA ROW IS THE DANGEROUS ONE, BECAUSE IT SUCCEEDS.** `chainKinds`
is never passed, so `ACGTACGTACGTACGTACGT` is read as Ala-Cys-Gly-Thr and comes
back as a 20-residue protein with a plausible confidence. This is CLAUDE.md's
own documented trap - `ACGT` is a valid protein AND a valid DNA chain - landing
where nothing was watching for it.

🔴 **THE RUNTIME IS NOT THE ONLY HALF AT FAULT: `foldOnBackend` SENDS FIVE
FIELDS.** It builds `{entities, model, steps, recycles, msa}` and drops
`templates` - which it accepts as a parameter and never forwards - along with
the sampler, the seed, `max-msa`, the PLM row and the AF2 model picker. The
runtime, meanwhile, reads `request.alignment`, which no page ever sends: an
alignment passed by hand moves the same fold **44.965 -> 51.262 pLDDT** and
comes back in `prediction.a3m`, so the plumbing works and nothing fills it.
**Every Colab fold through the node runtime is a single-sequence fold**, whatever
the MSA dropdown says, and the reader is never told.

What is NOT at fault, checked rather than assumed: the result shape. `wireResult`
serialises the whole prediction into `predJson` - `model`, `stem`,
`chainLengths`, `seed`, `recycles`, `tokens`, `contactSource`, the nine
confidence fields - and `revivePrediction` registers it under the reader's own
stem, so provenance, the archive and the download buttons are whole. A first
reading of this called `context` and `settings` missing; they are spread into
the top level by `predictionFromAf3`, and the probe was wrong rather than the
runtime.

🔴 **AND THE CONTROL SAYS WHICH HALF OF THIS IS THIS BRANCH'S. IT IS THE BIG
HALF.** `git diff origin/main..dawn` over the Colab surface:
`web/colab-bridge.js` is **IDENTICAL**, `foldOnBackend` in web/app.js is
**identical in the part that matters** - the same five-field request, the same
`templates` parameter accepted and never forwarded - `tools/colab_runtime.mjs`
is **new**, and `colab_backend.py` on main **has no `--runtime` flag at all**:
chrome was the only path there. So:

| | on `main` | on `dawn`, default `--runtime node` |
|---|---|---|
| multi-chain, copies, ligands, SMILES, contacts | the page's own `expandEntities` | **refused on the colon** |
| a DNA chain | the page's own `chainKinds` | **folded as a peptide** |
| the MSA search | the runtime page runs it off `msa-mode` | **never happens** |
| AlphaFold 2 | the runtime page folds it | **refused by name** |
| templates, sampler, seed, `max-msa`, PLM | **already dropped** | already dropped |

**The last row is the only one that predates this branch**, and it is the mild
one: the reader silently gets the runtime page's defaults. Everything above it
is a capability the headless-Chrome runtime had because it DELEGATED to the
real page, and that the node runtime lost because it re-implements the fold
path in 380 lines. Making node the default shipped that loss.

🔴 **AND THE DOCUMENTED FALLBACK MAY NOT EXIST ON A BOX THIS NOTEBOOK SET UP.**
The section below says "`--runtime chrome` still works and is the fallback",
and the same commit removed the browser from SETUP - "the four X11 libraries
were a zipped Chrome's alone, 6.0 s of apt for a browser that is no longer
installed". Whether Colab's own image carries one is UNMEASURED: the probe for
it lost its VM to the keep-alive fault below before it ran.

🔴 **AND `--runtime chrome` IS THE WORKAROUND FOR ALL OF THE BIG ONES,
MEASURED ON THE SAME CARD.** It sets four controls on a real page -
`model-family`, `recycles`, `af3-count`, `msa-mode` - and calls
`window.__entityList.set(entities)`, so the page's own `expandEntities` runs.
Same T4, same 58-mer, AF3 at 4 steps, the arms node refused:

| | `--runtime node` | `--runtime chrome` |
|---|---|---|
| one protein chain | 472 atoms | 472 atoms, 22 s |
| two copies | **refused** | **944 atoms, chains A and B**, 20 s |
| protein + `CA` x4 | **refused** | **476 atoms, HETATM 4, chains A-E**, 14 s |
| a DNA chain | **ALA CYS GLY THR** | **DA DC DG DT**, 411 atoms, 14 s |
| an MMseqs2 search | never happens | **21,577 rows, 128 used, pLDDT 91.0** (ubiquitin) |
| AlphaFold 2 monomer | refused by name | **601 atoms**, `AlphaFold 2 (monomer)`, pTM 0.355 |
| AlphaFold 2 multimer | refused by name | **1202 atoms, chains A and B**, ipTM 0.098 |

🔴 **AND THE SEARCH ARM TOOK TWO TRIES TO READ, BOTH TIMES THE PROBE'S FAULT.**
First it was sent `msa: "mmseqs2"` and came back `unknown alignment mode` - the
select's values are `none`, `search`, `paste`, `upload` and `mmseqs2` is the
LABEL, which web/colab-bridge.js's own comment warns about one line above where
it sets the control. Then with `search` it folded and the status line read
**"single sequence"** - which looked like the search being dropped and is
`singleSequenceIfOnlyQuery` working correctly, because the 58-mer under test was
SYNTHETIC and has no homologs. Ubiquitin is what settles it. **A capability
probe needs an input the capability can act on.**

It still drops the template, the sampler, the seed, `max-msa` and the PLM row -
`foldOnBackend` never sends them and this runtime never sets them - so the
reader silently gets the runtime page's defaults for those five. That part
predates this branch.

🔴 **AND THE COLD SETUP IS 42.9 s WITH A BROWSER AGAINST ~25 s WITHOUT**,
measured here, which is the whole price of the default going back. A warm
re-run is under a second either way now that the `wait` bug below is fixed.
The adapter through Chrome reads `nvidia / turing`, `shaderF16 true`,
`subgroupMatrix true`, `maxBufferSize` **4 GiB** - where the same card through
Dawn in node reports **1 TiB**, which is worth knowing before any ceiling is
derived from one of them.

🔴 **AND A PROBE THAT SENT `msa: "mmseqs2"` GOT `unknown alignment mode` AND
THAT WAS THE PROBE.** The select's values are `none`, `search`, `paste`,
`upload`; `mmseqs2` is the LABEL. web/colab-bridge.js's own comment warns that
a select silently refuses a value it has no option for, one line above where it
sets this one.

## The notebook, run as a cell on a fresh T4

🔴 **THE SETUP FAILED ON EVERY WARM RE-RUN, WHICH IS THE CASE ITS OWN HEADER
CALLS "NEARLY FREE".** `wait $small_libs` with `small_libs` unset is a bare
`wait`, which reaps EVERY background job including the clone; the later
`wait $repo_clone` then returns **127** on a pid that is no longer a child, and
`set -e` ends the script with the last `say` unprinted and the `2>/dev/null`
swallowing the reason. A cold box sets `small_libs` and passes; a warm box -
`need_icd=0 need_loader=0`, which is every second run - does not. The notebook's
own failure message is "run this cell again", which is the one thing that could
not work. Reproduced in five lines on the VM, fixed with an `if`, and the warm
setup now completes in **0.93 s** with the backend and `colab_runtime.mjs` both
up and the adapter reading `nvidia / turing / tesla-t4`, `shaderF16 true`,
`subgroupMatrix true`.

🔴 **AND THE CELL CANNOT BE FINISHED HEADLESSLY, WHICH IS A PROPERTY OF COLAB
AND NOT A BUG.** It ends at `eval_js('google.colab.kernel.proxyPort(...)')`,
which waits on a reply from the notebook FRONTEND; under `colab exec` there is
no frontend and the call blocks for ever in
`google.colab._message.read_reply_from_input`. Everything before it - setup,
service, runtime, adapter, token - is verified. Worse, that stuck thread
POISONS THE KERNEL: it holds the input queue, so every later `colab exec` dies
with `TimeoutError: Timeout waiting for output` and the session looks wedged.
Run the cell up to the display block when driving headlessly, and
`colab url -s <name>` for the genuine end-to-end.

## Reaching a Colab box from the A100

`uv tool install google-colab-cli --with "jupyter-kernel-client==0.15.0"`, then
`colab --auth=adc new -s <name> --gpu T4`. 🔴 **THE PIN IS NOT OPTIONAL**:
colab-cli 0.6.0 calls `jupyter_kernel_client.KernelClient`, which 1.0.2 renamed
to `JupyterKernelClient`, and unpinned every `colab exec` dies on the rename
while `colab new` still prints `Session READY` - so a dependency fault reads as
a session fault. Auth is the ADC from `gcloud auth application-default login`;
the bundled `colab skill` says the `colaboratory` scope is mandatory.
🔴 **AND ITS ABSENCE DOES NOT SHOW AT ALLOCATION.** A T4 minted without it
allocated, folded and served for twenty minutes, then `colab log` read
`KEEP: stopped reason=consecutive_4xx_errors ... 404 .../keep-alive/` and the
session was terminated under the work. `colab sessions` says only that the VM
is gone; `colab log -s <name>` is what names the cause.
