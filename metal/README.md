# LocalFold on Metal

The Apple-native port: **AlphaFold 3's lineage (all nine models, chai-1 with its ESM2 3B and Mol-1 with its split trunk), AlphaFold 2 (monomer and multimer, models 1-5) and
ESMFold2, written for Metal** - no CUDA translation, no patches. Each port reads the same featurised input as
`cuda/<port>` (`cuda/featurise`, linked in) and the same published weights, and folds to the same structures; the
arithmetic is chosen for Apple's GPUs (8 x 8 simdgroup matrices, unified memory, no integer divider).

```
bash metal/build.sh af3 af2 ef2                # -> metal/<port>/localfold-<port>; only the command-line tools needed
metal/af3/localfold-af3 --out=6mrr.pdb --model=boltz2 --sequence=GWSTELEKHREEL...
metal/ef2/fold 6mrr.pdb --sequence=GWSTELEKHREEL...    # cuda/<port>/fold's interface
bash python/build_wheel_macos.sh               # the macosx_13_0_arm64 wheel: `pip install localfold` on a Mac
```

The binaries take the CUDA binaries' flags (`--help`), fetch their weights into `~/.cache/localfold` (or
`--weights-dir=`), and need macOS 13 (Metal 3: a buffer's GPU address) on Apple silicon. Kernels are Metal source
compiled at run time; the GEMM instances a run used are listed in `~/.cache/localfold/metal/native-<port>-<hash>.specs`
(a wheel ships one) and compiled up front, in parallel, the next time.

## Layout

```
metal/
  build.sh       builds metal/<port>/localfold-<port> (and selftest, bench)
  core/          what every port is written against:
    core.h/.mm     the device, memory (shared buffers addressed by GPU address), dispatch, the GEMM and its fused
                   epilogues, the gated flash attention, LayerNorm and elementwise kernels, profiling
    args.h         every kernel's argument struct, one declaration read by C++ and by Metal
    common.metal   the core's kernels (attention, LayerNorm, the bundle decoder, conversions)
    gemm.metal     the GEMM on 8 x 8 simdgroup matrices, compiled per instance
    model.h/.cpp   weights and inputs by name: page bundles and af3-any-model blobs decoded on the device, through
                   cuda/featurise's weight walks (af2_weights.h, af3_weights.h), delta bundles, featurised inputs
    host.h/.cpp    the drivers' shared host pieces: --serve, frame superposition, whole-file writes
  af3/ af2/ ef2/ one per port, flat: the port's kernels (kernels.h, <port>.metal), its stages (*.mm), main.mm,
                 fold, gate.py, gate-baseline.<chip>.json
  bioemu2/       BioEmu2: --job= on af3-any-model's int8 blob, read as the page's bundle through a weight walk
                 (weights.mm); held stage by stage to webgpu/bioemu2 on the int5 bundle (--bundle=, --oracle=,
                 tools/dump-bioemu2-webgpu.mjs; docs/BIOEMU2.md). On the worker (`localfold serve`); steering by cuda/bioemu2's steer.h
                 (edm_gd, edm_fk) - edm_fkc refused: it needs the denoiser's backward
  selftest/      the GEMM, attention and LayerNorm against host references
  bench/         GEMM and attention arms, interleaved, each checked against the first
```

A port's Metal source is `core/args.h + <port>/kernels.h + core/common.metal + <port>/*.metal`, embedded in the
binary by `build.sh`; the GEMM's is `args.h + gemm.metal`, compiled per (types, tile, epilogue) instance on demand.

## The website on this machine: `localfold serve`

`localfold serve` (python/localfold/serve.py, in the wheel) serves the website's own page on 127.0.0.1 and sends
every fold to these ports instead of the browser's WebGPU - python/localfold/server.py and worker.py, the same
broker and worker Colab's runtime runs, driving metal/ here and cuda/ on NVIDIA. From a checkout:
`python3 python/localfold/server.py --native --local --open`. The page says where it folds ("Local server · Apple M2 ·
Metal"); its link is plain, with no token - it is this machine's own server on loopback.

`python3 tools/check-native-worker.py --offline --no-af3` (`npm run test:native`) holds it here: every case of the
CUDA backend's gate - 21 folds over every family but AlphaFold 3 itself (whose weights need DeepMind's terms), the
refusals, and a real page folding three families through the server - passes on the M2, one bar at Metal's own
number (protenix2 6MRR, 1.6 A). A warm fold on the page: AF2 6MRR 2.0 s click to result, ESMFold2 7.2, protenix2 4.9.

## Pocket and contact constraints: written, not yet built here

A job's `constraints` (cuda/af3/README.md, Input) reach Metal as they reach CUDA: the featuriser is the shared
cuda/featurise one, and `constraint_tables.h` adds each site's term from its table with `af3_add_table` after the
kernels that build the pair - Boltz-2's z-init (trunk.mm) and confidence re-embed (confidence.mm), chai-1's z-init and
diffusion structure pair (diffusion.mm). Written on a Linux box with no Mac, so it has not been compiled or run: build,
then fold `tools/fixtures/constraints/contact.json` with `--model=boltz2` and with `--model=chai1` against the CUDA
port's numbers before trusting it.

## Is it right? The gates

The CUDA gates' cases, scored against crystals, with this Mac's baseline in `metal/<port>/gate-baseline.<chip>.json`
(0.05 A and 0.5 pLDDT) and the A100's printed beside each case:

```
python3 metal/af3/gate.py [--write] [--only=<case or model>]   # 19: nine models, ligand + phosphoserine, DNA,
python3 metal/af2/gate.py [--write]                             #   methylated DNA, kitchen sink, glycan, templates
python3 metal/ef2/gate.py [--write]                             # AF2 5, ESMFold2 8
python3 metal/bench/gate.py [--write]                           # lf_gemm's SPEED, this chip's own times (below)
```

A chip without its own answer baseline is held to its generation's (an M5 Pro to the M5's), else the M2's. The answer
gates cannot see a kernel slow down - 5646656b made every `lf_gemm` 15x slower on the M2 and passed them all on the M5
(docs/M2-TO-M5.md) - so `metal/bench/gate.py` times the bench's plain and extended-epilogue arms against
`metal/bench/bench-baseline.<chip>-<cores>c.json` and fails at 1.3x. Speed is not shared between chips: a new chip
records its own. A change to `gemm.metal` wants it run on an M1/M2 (static registers) as well as on the chip that wrote
it.

On an M2 (10-core GPU), 2026-10-09: every case passes. Against the CUDA-translated port this replaced, case by case:

| | native | translated |
|---|---|---|
| AF3 6MRR | 0.771 A, pLDDT 86.16 | 0.768, 86.15 |
| protenix2 / boltz2 / intellifold2 6MRR | 1.561 / 0.578 / 1.539 A | 1.561 / 0.577 / 1.539 |
| rosettafold3 / openbind0 / opendde 6MRR | 1.699 / 1.759 / 1.457 A | 1.706 / 1.756 / 1.454 |
| AF3 5CAJ / 1BRS self-template | 0.205 / 0.457 A | 0.205 / 0.454 |
| AF2 6MRR / 5CAJ + template / 1BRS multimer + template | 1.899 / 0.216 / 0.272 A | 1.900 / 0.216 / 0.272 |
| ESMFold2 6MRR / 1QYS / 5CAJ / 1BRS | 1.353 / 0.865 / 2.105 / 0.912 A | 1.362 / 0.865 / 2.104 / 0.911 |
| chai-1 6MRR / + GOL and SEP@3 (ligand bonds) | 0.976 / 1.744 A (0.075) | 1.020 / 1.733 (0.077) |

🔴 **THAT TABLE IS THE EULER SAMPLER'S.** Since 2026-10-17 every AF3-lineage fold samples as the page and cuda/af3
do (opus-55-opt's d6a37b64, docs/AF3.md "The default sampler"): the exponential update `x' = r^S x + (1 - r^S) D` on
noise levels even in log sigma, with churn - `SamplerOptions` in af3.h. `--update=euler --schedule=karras` is
AlphaFold 3's own and reproduces the table; `--no-churn` drops the injection; `--flow` is refused by name. chai-1
keeps its own schedule and step, now the exponential one at its scale of 2 (opus-55-opt's 8469ca1: its 200 steps' bonds
at 25), and takes the flags; `--update=euler` is its sampler before. The baselines are re-recorded on it (AF3 6MRR 0.850 A at the gate's 200
steps). metal/ef2 takes the same three flags; its default stays biohub's Euler.

The AF3 trunk's pair after one pass agreed with the translated port's at relRMS 8.5e-4 (single 5.6e-4) on the first
build. `LOCALFOLD_SAVE_SEAMS=<dir>` writes AF3's stage outputs as `.f32` files for exactly that kind of comparison.

chai-1's ESM2 3B tower (`metal/af3/esm2.mm`) keeps its int8 matrices resident as their codes, read where they lie in
the mapped blob and expanded to half a GEMM at a time: the whole chai-1 fold peaks at 1.4 GB.

### biohub's released ESMFold2-Fast, on ESM-C 6B (2026-10-17)

It folds here: the parcae recycle (`trunk.mm`, cuda/ef2's), ESM-C 6B kept RESIDENT as the page's own quantised codes
and expanded a matrix at a time (`Model::loadBundle`'s `resident`, `Model::expand`; any tower whose float16 would pass
4 GB, `LOCALFOLD_TOWER_RESIDENT=1` forces it), and the language model's pair in float where the tower's states are
large (`tower.mm`: 6B's reach |a| 160-190, and a * a passed half's range - a NaN pair; 600M's and 300M's reach 29-50 and
keep their half path byte for byte). Built from biohub's own MIT checkpoints, nothing else:

```
hf download biohub/ESMC-6B --local-dir ESMC-6B                                   # 25.4 GB, float32
hf download biohub/ESMFold2-Fast --revision c6c7958d63 --local-dir ESMFold2-Fast  # the parcae_* layout cuda/ef2 reads
python3 tools/export_esmc_model.py --esmc <abs>/ESMC-6B --esmfold2 <abs>/ESMFold2-Fast --out <abs>/model-esmc-6b-fast-f32
python3 tools/quantize_af3.py --source <abs>/model-esmc-6b-fast-f32 --out <abs>/model-esmc-6b-fast-int3 --bits 3 --group 128
python3 tools/export_esmfold2_trunk.py --esmfold2 <abs>/ESMFold2-Fast --confidence <abs>/ESMFold2-Fast/model.safetensors \
    --out <abs>/model-esmfold2-fast-f32 && python3 tools/quantize_af3.py --source <abs>/model-esmfold2-fast-f32 --out <abs>/model-esmfold2-fast-int5
metal/ef2/fold out.pdb --model=ef2-fast --sequence=... --weights-dir=<dir>
```

`--model=ef2-fast` (cuda/featurise/fetch.h) reads `<dir>/model-esmfold2-fast-f32` and `<dir>/model-esmc-6b-int8` by
those names, whatever they hold: link the int5 folding bundle and the tower you want there. The exporter now reads the
published sharded layout (q|k|v and gate|up concatenated by rows, as `esm`'s checkpoint_layout.py does) and derives the
residual scale, sqrt(layers / 36): 1.4907 at 80 layers.

**The tower at 3 bits loses almost nothing.** Six crystals x two seeds, the folding model int5 g32 in every arm, CA RMSD
against the int7 tower (the quantiser's widest, 5.5 GB):

| tower | bundle | mean abs dRMSD | worst | mean dRMSD |
|---|---:|---:|---:|---:|
| int7 g128 | 5.49 GB | - | - | - |
| int5 g32 | 4.44 GB | 0.008 A | 0.043 | -0.004 |
| int4 g32 | 3.70 GB | 0.007 A | 0.016 | -0.004 |
| **int3 g128** (the 600M tower's recipe) | **2.46 GB** | 0.028 A | 0.117 (5CAJ) | +0.024 |

int7: 6MRR 1.620 / 1.681 A, 1QYS 0.931 / 0.936, 5K9P 1.732 / 1.302, 5CAJ 1.710 / 1.779, 1TIM A 1.114 / 1.099, 1BRS
A:D 0.608 / 0.631. int3 is the only arm that moves anything: 5CAJ ~0.1 A worse on both seeds, and 1TIM's pLDDT up 3.7
(90.0 -> 93.9) at the same RMSD - a confidence it has not earned. int4 g32 is as good as int7 at two thirds of its
size. On an M2: the language model 1.5-2.7 s (6.2 B weights expanded a GEMM at a time), the trunk 0.9 s at 68 tokens.

**And ESMFold2 itself, with its MSA encoder** (`msaEncode` in trunk.mm, cuda/ef2's msa.cuh): biohub/ESMFold2 at
8fc3ff4710, its own adapter on the same 6B tower (`model-esmc-6b-full-int3`, exported with `--esmfold2` pointing at it),
folded with `--model=ef2 --a3m=<a3m per chain>` and `<dir>/model-esmfold2-f32` linked to its bundle. Against biohub's
own `MSAEncoder` (`tools/esmc/probe-esmfold2-msa-encoder.py`: esm 3.4.1's module on the port's z_init, s_inputs and
rows, float32 on the CPU), 1,024 rows x 59 tokens: **8.7e-4**, the pair blocks' half floor, with the encoder held float32
in the bundle (`quantize_af3.py --prefix-bits msaEncoder/=32`: 194 MiB against 169); at int5 it reads 5.9e-2 and the
fold moves 0.02 A. Its outer product mean is a float GEMM, because a sum over 1,024 rows passes half's range (inf at 59
tokens through lin's half rounding). 6MRR from its sequence: 1.563 A, pLDDT 71.4; the encoder 0.2 s, the 48-block trunk
1.7 s at 68 tokens.

**On the page, through `localfold serve`.** The server's `/health` lists the native-only models whose exports are on
disk (`models`: `ef2-fast`, `ef2`), and the page shows their options only then. ESMFold2 is one entry on the model row
and the variant row picks the checkpoint: Fast on ESM-C 600M or 300M (the page's own, WebGPU), Fast on ESM-C 6B and
Full on ESM-C 6B (native only: shown when the server has them). Every variant folds from its language model, and an alignment is an opt-in: the 600M and
300M checkpoints read none and have no MSA row; for the 6B two the row starts at "ESM-C only" and offers a search or an
upload (the worker runs the search through AlphaFold 3's featuriser). The released Fast reads an alignment only as two
profile features, and on the 59-residue gate protein its 8,076 rows move pLDDT 83.18 -> 83.12; Full's encoder takes the
same alignment 89.8 -> 91.4. The worker reads them where
`fetch-weights` does - `LOCALFOLD_WEIGHTS_DIR`, else `~/.cache/localfold` on a Mac - by fetch.h's names, whatever they
hold: `model-esmfold2-fast-f32` (Fast's folding bundle), `model-esmfold2-f32` (ESMFold2's, its encoder float32) and one
`model-esmc-6b-int8` (the 6B tower: each folding bundle carries its own adapter, `lm/*`, which `tower.mm` prefers over
the tower's, so one tower serves both). `tools/check-native-worker.py` folds them where present: Fast 6MRR 1.606 A in
3.4 s, ESMFold2 6MRR 1.564 A, and 5CAJ with a searched alignment 1.963 A at pLDDT 97.3 (the CUDA port: 1.97 / 97.4),
and through the page in 8-10 s.

Fixed: a float32 folding bundle died on loading here ("Insufficient Memory" at its first command): 4 of its 16 shards
are a whole number of pages, and `mapFile`'s two bytes past the end needed a page past the file, which a file mapping
cannot back. The span is anonymous memory with the file over its front (core.mm); it folds 6MRR at 1.608 A.

**The 300M tower's residual scale was wrong everywhere, and is derived now.** `tools/export_esmc_model.py` wrote
residualScale 1.0 for every tower; ESM-C scales by sqrt(layers / 36), and biohub's own 300M runs 0.912871. Held against
biohub's language-model pair (`tools/esmc/probe-esmfold2-lm-pair.py`, `esm` 3.4.1 in float32, 6MRR), the float32 300M
tower reads 8.1e-3 with 1.0 and 2.2e-3 with 0.9129 on Metal (half GEMMs), and **7.97e-5** on WebGPU's float32 path.
metal/ef2/main.mm and webgpu/esmc/tower.js derive it from the depth (1 at 600M, 1.4907 at 6B), so the published 300M
bundle folds right unchanged; WebGPU folds the division into the device decode's group scales
(planBlockUpload's `multiplier`), where a scale other than 1 used to send the tower to the host. The CUDA port still
reads the manifest's 1.0 (cuda/ef2/src/ef2.cu:116).

## How fast

On the M2, warm, the fold alone unless said:

| | time | |
|---|---|---|
| AF3 6MRR (68 tokens, 3 recycles, 200 steps) | 6.3 s: trunk 1.3, diffusion ~5 | 6.2 s translated, 12.0 WebGPU |
| AF3 5CAJ self-template (255 tokens) | 31 s with start-up | |
| ESMFold2 6MRR / 5CAJ | 0.3 / 0.8 s | |
| AF2 6MRR / 5CAJ + template | 1.2 / 3.9 s | 14.4 s translated |

On an M5 (8-core GPU, macOS 26.6), warm, the matrix units on (gemm_tensor.metal) against the M5's own old kernels:

| | M2 | M5, old kernels | M5, matrix units |
|---|---|---|---|
| AF3 6MRR (68 tokens, 3 recycles, 200 steps) | 6.1 s | 3.79 s | 1.62 s |
| a diffusion step, 68 tokens | 24 ms | 15.3 ms | 6.4 ms |
| AF3 trunk pass, 261 tokens (5CAJ with its template) | 3.9 s | 2.64 s | 1.17 s |
| attention, 255 keys, 4 heads of 32, 255 rows | | 3.88 ms | 1.37 ms |
| GEMM, 4096 square | 2.93 TFLOP/s | 3.05 | 12.0 |
| weights load, AF3's int5 bundle (warm) | | 178 ms | 96 ms |

(As of 523aaed3. The matrix units' column: the half GEMMs on matmul2d with their epilogues in registers, attention v3
(the online softmax in cooperative tensors) from 48 keys, the triangle gate in registers, fast math in those
epilogues, the bundle decoder's and the MSA kernels' indices in 32 bits, the diffusion's conditioning batched over
noise levels; docs/M5-HANDOFF.md's last sections have each step's numbers and what lost.)

Profile any fold with `LOCALFOLD_PROFILE=1` (every labelled dispatch its own command buffer: an upper bound, good for
proportions) and AF3's denoiser by stage with `AF3_STAGES=1` (a sync between stages: real time).

What won, natively:

- **The diffusion transformer's GEMMs on rows rounded up to 16.** A short n (68 tokens) takes one tile row of 80, and a
  ragged tile runs the GEMM's bounds-checked staging and scalar stores: 0.31 against 0.22 ms on 768 x 3072. The
  transformer 33.7 -> 22.6 ms a step at 68 tokens; the padding rows compute values nobody reads.
- **The atom attention on simdgroup matrices** (`af3_atom_attention_mma`: 32 queries against 128 keys a threadgroup,
  the softmax in registers): the encoder and decoder 3.4 -> 1.8 ms a step each. `AF3_SCALAR_ATOM_ATTENTION=1` is the
  control.
- **Every attention on one flash kernel** (`core/common.metal`, heads 8-64 wide): the grid attention's column
  direction, AF2's MSA column and ending-node attentions, ESMFold2's diffusion - read ACROSS a tensor's leading axis
  through strides, so nothing is transposed. 1.22 TFLOP/s at D 32 on the M2 (0.15 before its loops were unrolled).
- **The diffusion transformer's 96 conditioning projections folded into two GEMMs** (the per-block LayerNorm scales
  folded into the weights, the biases as GEMM biases).

- **LayerNorm's row array sized to the row** (`lf_layernorm4_<V>`, `lf_center_norm_<NK>`): one sized for the widest row
  took a 128-channel row's occupancy - 42.6 against 83.5 GB/s on 68121 x 128 (`localfold-bench ln`), the trunk 6-7% at
  261 tokens. `LOCALFOLD_LN_WIDE=1` is the control.

What was tried and **lost**, so nobody repeats it blind:

- **The flash attention reading K and V straight from the device** (`simdgroup_load` from device memory, no staging and
  no barrier, the last tile staged): both arms fell to about half - 0.72 / 0.67 against 1.33 TFLOP/s at 261 tokens - the
  second path cost the kernel its registers. It is register-bound, as its comment says.
- **The diffusion transformer's gated residual and the next adaLN in one kernel** (cuda/ef2's fusion on `opus-55-opt`,
  48 launches a step fewer): byte-identical and level - 2.49 / 2.47 / 2.47 against 2.49 / 2.43 / 2.46 s at 68 tokens,
  0.7% at 261. Launches from one serial encoder cost almost nothing here; on CUDA the gain was the launches.
- **The gated residual and the next adaLN in one kernel, again (2026-10-11, the M5)**: byte-identical, the two kernels'
  565 -> 501 ms of a profiled 255-token diffusion, and level or 1-2% slower in wall time (a simdgroup a row starves the
  GPU at 68 tokens). The kernel already moves ~125 GB/s, this machine's bandwidth at that size.
- **A 72-row GEMM tile** (simdgroups 1 x 4) so 68 tokens pad to 72, not 80: 0.181 against 0.189 ms on 3072 x 768, level
  on the conditioning, worse on 768 columns (no 16-column tile) - about 2% of a step, not worth a second layout.

- **Split K for the skinny GEMMs** (partial products a K slice, then one reduction applying the epilogue): 0.395
  against 0.334 ms on 3072 x 68 x 768, and the trunk slower. These GEMMs are not short of threadgroups.
- **Matrix stores for the whole 8-row blocks of a ragged tile**: 0.31 -> 0.30 ms, and one instance hit a Metal
  compiler internal error. Padding the rows (above) does it for free.
- From the translated port's measurements, which hold here: double-buffered GEMM tiles (1.13-1.37x the time), eight
  simdgroups a threadgroup (1.04x), 64 x 128 or 128 x 64 tiles (spilled registers), MPS (level at best, more error),
  a "skinny" GEMM reading W straight into the matrix units (uncoalesced), register prefetch (2.43 -> 1.48 TFLOP/s).

## Memory

Every weight held once, in the precision it is used in: large tensors decoded straight to float16 at load (never
through a float32 copy), the rest float32. A weight read only through a derived form - a triangle's interleaved
projection and gate, an attention's q|k|v|g, a SwiGLU interleaving, the token transformer's folded conditioning - gets
its own allocation at load, the derived form is built up front (`metal/af3/weights.mm`) and the source given back
(`Model::retire`). ESMFold2 gives its language model back once it has run (not under `--serve`) and interleaves its
tower's weights in place. Scratch buffers are named and grown as asked; each stage gives its own back when it is done,
and the pair track's stages, which run one after another, share theirs (one buffer is the grid attention's q|k|v|g or
the triangle's planes) and give it back at the end of each pass. The sampler's scratch is given back before the
confidence head builds its own. The pair recycles in place (its recycle term projected back over itself, a row reading
only its own normalised row): a copy of it and an f32 term were two pairs more, 1.1 GB at 1044 tokens. (From cuda/af3's
big-input work on `opus-55-opt`, whose pair stays bf16 past the trunk; its launch fusions were measured here and lost.)

**Where the matrix units run, the trunks' residual streams are half** (`LOCALFOLD_HALF_PAIR=0` the control): AF3's
pairformer stack, AF2's whole evoformer (the MSA, the extra MSA and the pair) and ESMFold2's folding blocks each take a
half copy on entry and hand float back on exit, so every stage outside them is unchanged. Every residual is summed in
float and rounded once on its store (saturating at 65504; the largest values seen are ~13000, openbind0's pair); every
norm reads the stored half values. The updates' last GEMMs are bandwidth-bound on the residual, so it is speed - AF3
-5 to -6% a trunk pass, AF2 -4.5 to -8%, ESMFold2 -2.5 to -3.6% - and, since 2026-10-15, memory too: the float copy is
given back while a stack runs on the half one (below). bf16 was 3-15 A off on the low-confidence
folds where half sat at the f32 noise floor; docs/M5-HANDOFF.md has the numbers.

**The triangle's contraction is stored in half too, scaled by 2^-10** (`halfProductScale`, `LOCALFOLD_HALF_PROD=0` the
control): the float product is a pair update's largest write and the centre LayerNorm's whole read, and the norm is
scale-free (its epsilon scaled to match). Seen per update at rms 4 to 97,000 and peaks of 2.5e6 (IntelliFold-2, 5CAJ),
so the scale keeps the smallest in half's full precision and the peak 27x under 65504 (the store saturates past it).
Per pass the contraction is 16-18% faster and the norm 10-18% (AF3: 80 -> 67 and 64 -> 54 ms; ESMFold2: 226 -> 185 and
176 -> 144), about 2-3% of a trunk.

🔴 **A RELEASED BUFFER IS STILL RESIDENT UNTIL THE WORK ISSUED BEFORE IT IS DONE**, and the host runs far ahead of the
GPU: a 522-token trunk pass held 0.9 GB of released buffers beside its own 1.9, invisible to the allocation count and
exactly the gap to the process footprint. Past 128 MB of them (`LOCALFOLD_BURIED_MB`) the allocator waits for the GPU
instead - no measurable cost (5CAJ self-template 31.7 / 31.9 s against 31.8 / 41.0 without). `AF3_MEM=1` prints both
peaks and the scratch by name.

| peak | before | now |
|---|---|---|
| AF3 6MRR: allocation / process footprint | 2.21 / 2.28 GB | 0.85 / 1.14 GB |
| ESMFold2 5CAJ: allocation / footprint | 2.90 / 2.92 GB | 1.70 / 2.23 GB |
| AF3, 522 tokens, one pass: footprint | 2.96 GB | 2.00 GB |
| AF3, 1044 tokens, one pass: allocation / footprint | 9.0 / 11.3 GB | 5.2 / 5.3 GB |
| AF3, 1044 tokens, two passes: footprint | | 4.7 GB |

**Scratch shared between modules, and the scribble that proves it** (`shareScratch`, core.h; cuda's version and its
lessons are docs/CUDA-TO-METAL.md on `opus-55-opt`). A port names its scratch FAMILIES (name prefixes) and which are
nested - live around another's turn; a buffer of 32 MB or more aliases a slot no buffer of its own family, or of a
nested one, holds. AF3's MSA stack shares the pair track's slots; AF2's block modules (attentions, triangle
multiplications, outer product, transitions) are families of their own, so its evoformer holds its largest module's
scratch rather than the sum. `LOCALFOLD_SCRIBBLE=1` fills every slot with NaN bytes at each change of family, and a fold
must stay byte-identical to `LOCALFOLD_SHARED_SCRATCH=0`'s, at 32 MB and at `LOCALFOLD_SHARED_MIN=65536` (a long fold's
sizes); `LOCALFOLD_SHARED_SKIP=<family>` bisects a failure. It found two, both cuda's: AF2's outer-product mask norm,
written once a stack and read by every block after (in no family now), and the triangle's planes' padding zeroed once
and remembered by pointer, in AF2 and ESMFold2 (`padZero` every call, as AF3's pair track did). Every AF3 gate case and
four AF2 inputs pass it. **Where the stacks run half, the float pair is given back while they run** (AF3's pairformer
and confidence head, AF2's evoformer with its MSAs, ESMFold2's trunk), AF2's recycled pair is normalised in place, the
triangles' centre norm and gated value are written over their planes, and each confidence head gives the pair track's
scratch back before its heads. `LOCALFOLD_PEAK=1` prints each stage's peak and what was held at it.

**The triangle's middle in one buffer, and attention's q|k|v|g a block at a time** (the second round, same day). Core's
`triContract`: the gate GEMM writes a and b interleaved by channel, and the contraction runs a quarter of the channels
at a time, each quarter's product written over the planes the quarter before read - so the product ends up contiguous
at the buffer's start, with room for one quarter ahead of the planes; where the product is half, the centre norm fits
after it and the gated value over it. 2.25 pair-sized planes where there were 4 plus two pair buffers: 0.63 GB at 1044
tokens and 128 channels against 1.7. AF3's grid attention and AF2's every attention build q|k|v|g for at most 256 MB of
batch rows at a time, the blocks balanced (1.1 GB for the whole pair at 1044; below ~500 tokens one block). And AF2 gives
its structure module and heads back at the start of each next pass: kept, the next evoformer's buffers landed in the
slots they had grown. All bit-identical to before on every input compared (AF3 and AF2 at 522 and 1044 tokens, ESMFold2
at 1044), and level in time (interleaved: AF2 522 22.24 / 22.41 s, 1044 127.9 / 128.2 s warm; ESMFold2 1044 67.85 / 67.80
s). `LOCALFOLD_TRI_BLOCKS=1` and `LOCALFOLD_QKVG_MB=0` are the controls.

| peak at 1,044 tokens (5CAJ x4, each templated; ESMFold2 untemplated) | before | now |
|---|---:|---:|
| AF3 | 5.33 GB | 4.20 GB, then 3.54 GB |
| AF2 multimer | 8.88 GB | 4.08 GB, then 2.76 GB |
| ESMFold2 | 8.67 GB | 5.32 GB, then 4.90 GB |

**A fold that will not fit is refused up front** (`foldFits`, core.h): its peak estimated from the bytes held once the
input is loaded plus measured factors times the f32 pair and MSA, against `recommendedMaxWorkingSetSize` (12.71 GB on a
16 GB M5). Past that, unified memory pages and the fold runs many times slower instead of failing. The factors (bytes
at the peak per f32 pair byte) were measured by folding at 1,044 tokens on both paths: AF3 5.1 (5.4 where its stacks run
float, as on an M2), AF2 4.7, ESMFold2 4.2 (5.3; 3.7 since 2026-10-19, each pair update emitting its next norm over its own input) for its trunk's stage with its tower given back; the MSA's 3.2 (AF3) and
2.2 (AF2); OpenDDE's structural stage, asked before the trunk, 6 per byte of its structural f32 pair (measured on
the M2 at 544 residues, 1,040 structural tokens, where the trunk's estimate alone read 3.66 GB of an 11.08 GB peak). The estimates sit 1-3% over the peaks they were fitted to; below ~600 tokens a fold's fixed costs dominate and
they can read 5-10% under, which decides nothing there. On this M5 the longest folds are about AF3 2,100 tokens, AF2
2,250 residues, ESMFold2 1,690 (before this round 1,900, 1,850 and 1,620). `LOCALFOLD_NO_FOLD_FITS=1` folds anyway; a `--serve` process
answers the job with the refusal and stays up.

## Traps, each paid for

🔴 **AN ARRAY OF SIMDGROUP MATRICES MUST BE UNROLLED** (`_Pragma("clang loop unroll(full)")` on every loop over one),
or it spills to memory: the flash attention ran at 0.15 TFLOP/s until it was.

🔴 **APPLE'S GPUs HAVE NO INTEGER DIVIDER.** `a / d` by a runtime `d` is a software routine at any width; a kernel that
divides an element runs at a fraction of its bandwidth. `lf_udiv` (`core/common.metal`) divides through a float
reciprocal with one correction, exact below 2^24.

🔴 **A BLOB'S DATA IS NOT ALIGNED.** af3-any-model's records have headers of any length, so a float32 tensor can start
at any byte; four-byte loads read garbage and protenix2 folded NaN. The decoder assembles words from bytes.

🔴 **32 KB OF THREADGROUP MEMORY, AND NO `double`.** A kernel staging more does not compile on Apple's GPUs (the CUDA
ports assume 48 KB or more); rf3's chirality gradient, double precision on CUDA, is float here with a larger step.

🔴 **THIS MACHINE THROTTLES, AND THE GPU IS SHARED WITH THE WINDOW SERVER.** The same GEMM read 0.22 and 0.42 ms in two
runs minutes apart. Compare arms interleaved in one process (`metal/bench`), never two runs.

🔴 **A FLOAT ATTENTION MASK IS ADDITIVE AND FINITE.** The flash kernel's sentinels are -1e30 past the keys and -1e9 for
a masked one, so a row whose every key is masked is uniform rather than NaN - as the references are.

🔴 **`MTLCreateSystemDefaultDevice()` RETURNS nil TO A COMMAND-LINE PROCESS** sometimes; `MTLCopyAllDevices()` does not.
And a pipeline can silently refuse the threads it is dispatched with (`maxTotalThreadsPerThreadgroup` falls as a kernel
uses more registers): no error, the output untouched.

## Next

1. The skinny GEMMs (a single sample's diffusion): weights read at ~25 GB/s where the M2 has ~100. On the M5 (matrix
   units) 38-96 GB/s: a 68-token step is ~311 MB of transformer weights, and its floor (6.0 ms at 16 tokens, 7.6 at
   68) is those reads - not the dispatches (72 fewer a step: level). Int8-resident weights (3.) or more samples a batch
   are what move it.
2. Fusions the CUDA port has: the gated residual into the next adaptive LayerNorm, the trunk's transition and
   triangle kernels.
3. An int8-weight GEMM for ESM2 3B (its matrices are expanded to half a GEMM at a time).
4. An M5: its GPU's matrix hardware through Metal 4's tensor APIs - done (`core/gemm_tensor.metal`, chosen at run
   time on an Apple10 GPU): the half GEMMs with their epilogues in registers, the f32-X GEMMs that allow half staging,
   attention from 48 keys at heads 16-64 wide (v3, the online softmax in cooperative tensors). Left: heads 8 and 24
   wide (AF2's extra MSA, AF3's single attention) on lf_attention, and attention itself at 6-7 TFLOP/s against the
   GEMMs' 10-12 - 37% of a 1020-token trunk. docs/M5-HANDOFF.md's last sections have the numbers and what lost.
5. The pair LayerNorm emitted by the previous GEMM's epilogue (Gemm::lnOut, the matrix units): done for the 128-wide
   pairs on 64 x 128 tiles and the 256-wide on 32 x 256 (05c066c5).
