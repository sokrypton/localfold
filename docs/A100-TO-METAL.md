> 🔴 **SUPERSEDED, 2026-11-11: the Metal port is its own repository now** -
> [sokrypton/colabfold-metal](https://github.com/sokrypton/colabfold-metal) (private), LocalFold's `metal/` with its
> history. Its first job on a Mac is **`docs/HANDOFF-SPLIT.md` there**: build it, run its gates, report back in that
> file's Log. LocalFold's side is branch `split-native`, where `metal/` is that repository as a submodule. What
> follows is the 2026-11-07 round, kept as the record.

# A100 to Metal: what wants checking on Apple silicon (`opus-55-opt`, 2026-11-07)

Written on the A100 for the agent sitting on a Mac, whose job is testing `metal/`, the page's WebGPU path and its
WebGL2 path. **This file is how we talk**: answer in the `## Log` at the bottom - newest last, dated, signed with
your chip (`(M2)`, `(M5)`, ...) or `(A100)` - with what you ran, the numbers, and any commit hash. Do not rewrite an
earlier entry; correct it with a new one. A sibling file on the B200 (`~/B200-HANDOFF.md` there) does the same for
the training work; nothing in it is yours.

Ground rules are CLAUDE.md's and AGENTS.md's. The ones that matter here: commit as
`git -c user.name=sokrypton -c user.email=so3@mit.edu commit`, end every message with
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`, push to `opus-55-opt` (never `main`: a push there is a
deploy), `git pull --rebase` first. No new `LOCALFOLD_*` env knobs. Fail loudly. A checksum does not travel between
machines - compare an arm against your own box's other arm.

## What changed, and why it needs a Mac

### 1. `opus-55-opt-backprop` is merged into `opus-55-opt` and retired (`c94ea737`)

Almost all of it is `cuda/grad/` (OpenFold3 training on CUDA - nothing for you). What reaches you is section 2.

### 2. Chain-level pTM is AlphaFold 3's now, in every port (reported by Yeqing Lin)

A short binder's `chain_ptm` read ~0.10 high against AF3: every port reduced the chain-level scores from the GLOBAL
TM term (d0 from every token). AF3 (`confidence_head.py`, `confidences.py`, `model.py`):
- pTM / ipTM off the global term, anchored on the tokens WITH FRAMES (`pae_single_mask`: an ion, a ligand of colinear
  atoms anchors nothing);
- `chain_pair_iptm` off the INTERFACE term (d0 from the chain's own tokens on the diagonal, the two chains' summed
  off it), same anchors; `chain_ptm` its diagonal;
- `chain_iptm` = `iptm_xchain`, the mean of a chain's row of `chain_pair_iptm` off the diagonal.

**One definition**: `cuda/featurise/ptm.h` (host C++, on Metal's include path) for every native port, and
`shared/heads/tm-score.js` for the page. ESMFold2 keeps ESM's own pTM / ipTM (every token an anchor); its AF3-named
chain fields are AF3's.

| where | state | checked here |
|---|---|---|
| cuda/af3, cuda/ef2 | done | AF3's own functions (`tools/oracle/check_chain_ptm.py`), PASS |
| page WebGPU AF3 lineage, OpenDDE | done | same, PASS |
| page WebGL2 AF3 | done | against the WebGPU fold: chain pTM 0.2413/0.3912 vs 0.2402/0.3933 |
| page AF2 (WebGPU) | chain fields on the interface term | multimer folds; unit tests |
| page ESMFold2 (WebGPU) | chain fields NEW (it had pTM / ipTM only) | barnase-barstar folds, the pair ipTM = ipTM as it must |
| **metal/af3** | done, **NEVER BUILT** | host compile of `frames.h` + `ptm.h` only |
| **metal/ef2** | chain fields NEW, **NEVER BUILT** | nothing |
| cuda/af2 | done (`chain_*` on the interface term, every token an anchor) | the page's `af3TmScores` on the same PAE logits: 1BRS, 6MRR, a 3-chain multimer, worst 4.4e-5 |
| python/localfold/worker.py | passes every native port's chain fields to the page | they were dropped: no Colab archive had any; `test:native` now requires them |
| page AF2 (WebGL2), metal/af2 | no chain fields at all | - |

## Asks, in order

1. **Build and gate the two Metal ports** - they were edited blind:
   ```
   bash metal/build.sh af3 ef2
   python3 metal/af3/gate.py && python3 metal/ef2/gate.py        # against your chip's baselines
   ```
   The files: `metal/af3/{frames.h,confidence.mm,structural.mm,main.mm,output.mm,af3.h}`,
   `metal/ef2/{confidence.mm,output.mm,ef2.h}`. Expect pTM / ipTM unmoved except where a token has no frame (an ion,
   e.g. calmodulin_4calcium), and the summary's `chain_ptm` lower for every chain of a complex.

2. **Hold metal/af3's chain scores to AF3's own functions.** `check_chain_ptm.py` wants the fold's TM terms dumped
   (cuda/af3 writes them under `AF3_PAE_OUT=<file>`: PAE, PDE, pLDDT, then the interface term, the global term and the
   frame mask as float32 - see `cuda/af3/src/af3.cu` near "AF3_PAE_OUT"). metal/af3 has no such dump. Either add the
   same one (it is an existing knob name, not a new LOCALFOLD_ one) and run
   `python3 tools/oracle/check_chain_ptm.py --dump= --out= --log=` with an AF3 install, or compare metal/af3's
   `_summary_confidences.json` against cuda/af3's on the same job, seed and model - I can fold the cuda side here; ask
   in the log with the job and settings. Jobs worth it: `tools/fixtures/af3-jobs/{barnase_barstar,calmodulin_4calcium,tetr_dimer_tetracycline}.json`.

3. **metal/ef2**: fold barnase_barstar and calmodulin_4calcium; on this A100 cuda/ef2 gives (same job files, default
   settings) barnase-barstar `chain_ptm [0.9585, 0.9578]`, `chain_pair_iptm` off-diagonal 0.9564 = ipTM; calmodulin
   `chain_ptm [0.5213, null x4]` (the ions have no frames), `chain_iptm[0] 0.6488`. Close, not equal, is expected.

4. **The page on a Mac** (Chrome and Safari): `npm run check:safari` first (new syntax in `shared/heads/tm-score.js`,
   `webgpu/esmfold2/confidence.js`, `webgpu/heads/confidence.js`, `webgl/af3/model/fold-batch.js`). Then fold a
   two-chain job with an AF3-lineage model, AF2 multimer and ESMFold2 and read the archive's
   `summary_confidences_0.json` - every model now writes `chain_ptm` / `chain_iptm` / `chain_pair_iptm`.
   The AF3 confidence shader gained a binding (`chain_info`, binding 6; boltz2's features moved to 7) and OpenDDE's
   pair readouts one too - **check the WebGPU spec floor of 8 storage buffers a stage still holds on your adapter**
   (`npm run test:spec-floor` is the A100's emulation; a real Metal adapter is the better witness).

5. **WebGL2** (`webgl/`, all new on `opus-55-opt` since your last pass): `tools/gpu/check-af3-webgl-page.js` (the
   page's WebGL2 AF3 fold against its WebGPU one) and the `check-webgl-*.js` checkers. On this A100 the AF3 page
   check passes (Cα 0.048 A apart, 21 s against 3 s).

## Log

- 2026-11-07 (A100): this file. `opus-55-opt` at the commit that adds it. Asks 1-4 are this round's; 5 is
  standing.
