# From the CUDA ports to the Metal ones (2026-10-15)

Written on the A100 (branch `opus-55-opt`) for whoever works on `metal/` (now on `opus-55-opt` too: the `metal` branch was merged and removed 2026-10-18). Nothing here was run on a
Mac. Each item is something the CUDA ports learned this round, with how it was measured there and why it might
matter more on a 16 GB Mac, where unified memory is the ceiling. The opposite direction was checked too: the `metal`
branch's trunk ideas were profiled on the A100 and are already in CUDA or are below 1% there (the end of this note).

## 1. Scratch shared between modules, and how to prove it safe

CUDA's trunk kept every module's large named scratch for the whole fold: 14.7 GB at 1,500 tokens where the largest
module needs about 5. The modules of a block run one after another, so a buffer of one module is dead while another
runs. `cuda/af3/src/common.cuh` ("SHARED SCRATCH") aliases each large buffer (32 MB or more) of a *family* (a name
prefix: `tri.`, `grid.`, `tr.`, ...) to a shared slot that no other buffer of its family holds. Result on the A100,
atoms bit-identical: AF3 1,850 tokens 27.9 -> 20.7 GB peak, AF2 2,000 residues 16.4 -> 10.9, ESMFold2 2,000 residues
16.2 -> 15.2. `metal/af3`'s "the pair track's stages share their scratch" (9.0 -> 7.1 GB at 1,044 tokens) is the same
idea within the pair track; the CUDA version reaches the template stack, the MSA stack, the embedding and the
confidence head too.

**The proof is a scribble, and it is what made this safe.** At every change of family, fill every shared slot with
garbage (except slots of a family nested with either side), and require the fold to be bit-identical to the unshared
build. Plain sharing can pass by luck of slot placement; the scribble cannot. Run it also with every buffer over 64 KB
shared: that stands in for a long fold, where a buffer that is small at 262 residues passes 32 MB. The instrument and
the bisection over the family list are in commit `872a136`'s `common.cuh`; they were taken out with their env knobs.

What the scribble found, each a fold that moved or crashed, so each port keeps its own family list and its own
"nested" pairs (a family whose buffers stay live around another's turn):

- AF3: the template stack holds `tmpl.*` across its blocks' triangle, attention and transition modules, and the
  embedding holds `emb.*` across the f16 conversion buffer.
- AF2: the template stacks around everything their blocks run, the heads around the structure module, row and column
  attention and the attention bias around the attention kernel, and **the MSA masks**: written once at the embedding,
  read by every block. A buffer written once and read later must never be shared.
- ESMFold2: the trunk's own `trunk.*` (live across its blocks, a crash when shared), and `shim.tokens`, read by the
  streamed z_init inside the trunk (an illegal address across two GPUs).
- **A pad "written once" breaks as soon as its bytes are shared.** CUDA's AF2 and ESMFold2 triangle zeroed their
  planes' padding once and cached that; under sharing another family's data landed there and a 700-residue multimer
  moved. Metal already zeroes only the padding on every call (`padZero`), which is the fix CUDA adopted.
- **A captured graph pins its buffers.** Freeing a shared slot when one holder is released freed memory a captured
  replay still used (AF2: an illegal address). Metal has no CUDA graphs, but an encoded command buffer that is
  reused holds pointers the same way.
- **Per-type statics.** The scribble's "last family" was first a `static` inside a template function, one copy per
  element type, so switches were missed or fired mid-turn. One global.

## 2. The template stage was the length ceiling, twice

At 9,200 tokens on 40 GB the template stack's f32 activation (64 channels, 256 bytes a token pair) was made while the
bf16 pair (256 bytes) was still resident: 512 bytes a pair, the highest stage of the fold. With one live pass the pair
is read before the stack (the query) and after it (the output) and not in between, so CUDA now **parks the pair before
the activation is made** and reads the query from the parked copy, and **ends the stage in f16**: relu, scale and
convert the activation to f16 before the pair comes home, then project from the f16 copy. The peak is 384 bytes a pair.
The f16 end changes rounding on that path only; a 5CAJ fold with its template was unchanged.

`metal/af3/trunk.mm`'s `templateEmbedding` makes the query and the activation as whole pair-sized f32 tensors beside
the pair. On a Mac that is likely the same top stage. On unified memory "parking" is not a copy to the host; it would be
freeing the pair (or keeping it only as the stage's input) while the stack runs.

## 3. A blocked triangle must not rebuild its operand from the pair it is writing

At 8,600 tokens CUDA's limit was the blocked triangle: the pair plus a second pair-sized operand `b`. I tried building
`b` in bands of the contracted index `k`, rebuilt for each block of output rows, which took 9,200 tokens through. **It
computed the wrong model.** Each block adds its output into the pair before the next block rebuilds `b` from it, so
every block after the first contracted against a partly updated pair. Whole, `b` is made once before any write. Nothing
in a fold showed it: 5CAJ moved 0.19-0.29 A, but block width alone (a different GEMM shape) moves that fold 0.27 A, and
its RMSD to the crystal was unchanged. It was found by reading the code. If `metal/` ever blocks its triangle, the
update must stay out of the pair until the last block (a second pair-sized buffer, or another home for it), or `b` must
be built whole.

## 4. Size the blocks near the limit against a small reserve

Near the card's limit CUDA's blocked triangle sized its row blocks by what was free beyond a 16th of the card. Beside
the pair and `b` almost nothing was, so blocks were ~100 rows, cuBLAS picked small tiles, and the triangle's GEMMs took
238.7 s a pass at 8,300 tokens against 60.5 at 7,800. `b` is made and nothing else is allocated before the blocks'
buffers are given back, so the reserve can be a 64th: the trunk 545.7 -> 396.6 s, the cubic's expectation. Any
memory-sized chunk on Metal (grid attention chunks, transition chunks) has the same failure mode: a reserve sized for
safety elsewhere can starve the GEMM it sizes.

## 5. The fit check, measured past the line

CUDA refuses a fold up front (`foldFits`) at a factor of the f32 pair. The factor was found by folding past the check
(`LOCALFOLD_NO_FOLD_FITS=1`) and reading which allocation failed and by how much: 8,600 tokens ran out by 50 MB in the
triangle's `b`, so the factor is 1.06 where the fused bf16 triangle runs, and 8,300 tokens fold on 40 GB (it was
7,900). A first guess of 0.80 admitted 9,550 tokens, which then ran out by 0.2 GB in the diffusion preparation. A Mac
has no fit check today; folding past the working set ends in paging rather than an error, which is worth knowing
before a user finds it.

## 6. The diffusion preparation's floor at the limit

At the limit the diffusion keeps the f16 LayerNorm'd pair (256 bytes a pair), the atom encoder's pair conditioning
(64), and its per-block biases. CUDA rebuilds the biases of one block at a time (not a super block's four) where even a
super block's set does not fit beside the f16 pair: bit-identical, one buffer instead of four.

## 7. The extra stack's global attention and row attention (2026-10-18)

CUDA's column global attention on the 5,120-row extra MSA had been eleven f32 passes with two transposes (7.9 ms a call
at 261 residues); it is four kernels now (cuda/af2/src/globalattn.cuh), ~0.6 ms. metal/af2 never had the transposes -
it already runs in the MSA's own layout with the keys and values from one GEMM - but it still writes the LN'd rows,
the f32 gate and the gated rows as passes of their own. What CUDA fuses beyond that: the LN, `[k | v | gate]` and the
query's masked column sums in ONE pass over the rows (the sums per block, then one atomic a column-channel), and the
gate times the column average, the output projection, its bias and the residual in another. The extra row attention's
LN + q/k/v/gate is one pass too (`gridInK` at 64 channels), and its output projection adds into the f32 MSA in one
(`projAdd64K`) where a GEMM with an f32 output ran an epilogue pass over the whole MSA. Together ~7% of a 261-residue
fold on the A100.

## 8. Mol-1: a ninth AF3-lineage model - ON metal since 2026-10-27

**Done on the M2**, items 1-5 below: the PairMixer stack in `prepareWeights` (a block's grid attention skipped when absent), the
incoming triangle emitting the transition's norm when the block has none, the split trunk in `runTrunk` (both the half
pair and the `LOCALFOLD_HALF_PAIR=0` control), `transitionF32` for the single transition (the LN'd input half; the
product, `silu(a) b` and the second GEMM float), and the Protenix scores and rebuilt C-terminal oxygens through
scores.cuh as CUDA has them. 6MRR from its sequence: **1.579 A, pLDDT 87.95, pTM 0.8941**, ipTM 0, ranking 0.18; the
float-pair control 1.579 A / 88.12. 🔴 Without `transitionF32` one pass folds (1.609 A) and four are NaN, as item 4
said: lf_gemm's SwiGLU store converts to half unclamped, so the 6.6e4 is an infinity. `metal/af3/gate.py`: all 19
cases ok against the M2's baseline, `mol1-6mrr` recorded.

**Against the trunk oracle below** (exported on the A100 with `export-model.mjs --oracles=trunk`, folded here with
`--recycles=0` and `LOCALFOLD_SAVE_SEAMS`; the narrow pair is tapped as `trunk_in_pair`/`trunk_out_pair`, the full one
after the adapter as `pair`), relRMS, the half pair / the float pair: target_feat 2.5e-5, z_init 8.5e-5, template
9.3e-5, MSA stack 3.3e-4, narrow pair in 8.6e-4 / 4.6e-4, trunk_in_single 2.4e-4, narrow pair out **6.4e-3 / 4.9e-3**,
final pair **4.5e-3 / 3.7e-3**, single 4.6e-4. 🔴 **THE CONTROL SAYS THAT IS METAL'S BAND, NOT MOL-1'S**: Protenix-2 -
its parent, 48 pairformer blocks - through the same procedure against a trunk oracle dumped the same way (its int8 blob,
af3-any-model's mol1-port checkout) reads target_feat 6.8e-5, MSA stack 3.4e-4, trunk_out_pair **4.5e-3 / 3.4e-3**,
single 1.5e-3. Half GEMMs everywhere (CUDA's f32 path reads 1e-6 on the same oracle): the pairformer accumulates ~1e-2
of drift in f16 arithmetic, whichever model. The denoise and confidence oracles were not run: Mol-1 shares Protenix-2's
code for both.

The note as written before it was done:


Mol-1 (Molence Bio, `--family=mol1`; af3-any-model's `mol1-port` branch, blob `mol1/mol1.int8.bin.zst` at HF
1b41e333) folds natively in CUDA (3b98f446, a85cb169) and on the page (b0743630). metal/af3 does not have it: written
on Linux, untestable there, and a blind edit to the emitted-norm chain would break eight models on the Mac to add one.
What metal needs, each with its CUDA twin:

1. **The weight walk is already done** - cuda/featurise/af3_weights.h is shared, and with `--family=mol1` it emits
   `trunk.pairMixerBlocks.k.*` (pairformer blocks with NO `pairAttention1/2`), `trunk.pairformerBlocks.k.*` at the narrow
   width, `trunk.trunkPairChannels` (320) and `trunk.trunkPairIn` / `trunk.trunkPairOut` (448x320, 320x448, no bias).
   Add `"trunk.pairMixerBlocks."` to `prepareWeights`' stack list (weights.mm) so its blocks get `blockWeights`.
2. **A block with no grid attention** (pairtrack.mm `pairUpdatesT`): skip both `gridAttention` calls when
   `M.has(pre + ".pairAttention1.heads")` is false, and point the incoming triangle's emitted norm at
   `pre + ".pairTransition.inputLayerNorm"` instead of `pairAttention1.actNorm` (cuda: pairtrack.cuh `pairUpdates`).
3. **The split trunk** (trunk.mm `runTrunk`, cuda: trunk.cuh `runTrunk`): after the MSA stack, the PairMixer blocks on the
   448 pair; then `narrow = pair W_in` (every pair row, written not added), the 40 pairformer blocks on `narrow` at
   C = 320, then `pair = narrow W_out`. Seams: `trunk_in_pair` / `trunk_out_pair` are the NARROW pair, and
   `trunk_in_single` is taken AFTER the PairMixer (that is where the reference taps them). The half-pair stack works as
   it does for every other model - both widths are multiples of 8.
4. 🔴 **THE SINGLE TRANSITION'S GATED ACTIVATION OVERFLOWS f16** (`singleTransitionF32` in dialect.js). Measured on
   CUDA's f32 path, 6MRR: in pairformer block 35 the widening peaks at 273 but `silu(a) * b` reaches **6.43e4 on the
   first pass and 6.61e4 on the second** - so in f16 one pass folds and every pass after is NaN everywhere. metal's
   `transitionT` stores `g` in half (`gemmSwiglu`), so it WILL hit this. Only `g` needs headroom: either a float `g`
   for this flag, or `g` stored scaled (x 1/4 in the SwiGLU epilogue, `alpha` 4 on transition2). CUDA runs the whole
   transition in f32 (n rows). Mol-1 itself runs its trunk in bf16 autocast, which never meets it.
5. **Its scores** (`protenixScores`, protenix2 too; `rebuildCterminalOxygens`, mol1): a single chain's ipTM 0, the ranking
   0.8 ipTM + 0.2 pTM - 100 has_clash with no disorder term, has_clash over polymer chain PAIRS, and each chain's last
   O/OXT rebuilt on the written coordinates (cuda/af3/src/scores.cuh `hasClashChainPairs`, `writtenCoordinates`).

To check it, with the CUDA port's oracles (made on the A100: af3-any-model at f32 on the int8 blob, 6MRR on its own
batch): `oracle-dumps/af3-batch-mol1-6mrr.json`, `af3-oracle-{trunk,denoise,stages,confidence}-mol1.json` - copy them
over, `node cuda/af3/export-model.mjs <dir> --no-weights --family=mol1 --oracle-model=mol1`, and the metal binary on that
directory. CUDA's numbers there: every trunk seam 5.7e-8 to 1.3e-6, denoise 1.4e-6, confidence 1e-7 to 8e-7; 6MRR from
its sequence 1.45-1.53 A over five seeds, pLDDT ~88.

## What did not transfer from `metal` to CUDA (measured on the A100)

| `metal` idea | CUDA |
|---|---|
| outer product mean's product stored permuted by its epilogue | the separate permute is 4.2 ms of a 544 ms pass (0.8%) at 600 tokens, 1,024 MSA rows; cuBLAS takes no custom store |
| the next LayerNorm emitted by the previous GEMM | CUDA's input kernels (`gridInK`, `triIn256K`, `fusedTransitionK`) normalise the rows they load; no separate pass to remove |
| wide centre LayerNorm statistics in one read | `triangleOutK` stages the product tile once and takes both statistics from registers |
| the triangle's product in half | the fused path's product is bf16 already |

The rest of the branch is Apple-specific (no integer divider, 8x8 matrix stores, Metal 4 tensor units, the M2/M5 tile
and register budgets).
