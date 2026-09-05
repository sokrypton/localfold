# ESMFold2 and ESM-C in LocalFold

Nothing here ships yet. This is the answer to one question - **can a protein
language model replace the MSA search in a browser** - and the measurements that
answer it.

`docs/AF3.md` holds the AF3 port's state; this holds the ESMFold2 investigation's.
The tools are `tools/esmc/`.

## Why ask

LocalFold's alignment comes from an MMseqs2 server that queues, and a target
with no homologues gets nothing useful back from it at any wait. ESMFold2-Fast
folds from ESM-C's hidden states instead of from an alignment: no search, no
round trip, no database. The whole question is whether the language model fits.

## The models, and which one this is about

Biohub publish ESMFold2 (MSA-capable) and **ESMFold2-Fast** (single sequence,
no MSA encoder at all), each in a released and an *experimental* line, and the
experimental line additionally as a per-LM-size ablation series that exists to
reproduce the paper.

| | LM | folding model | tower |
|---|---|---|---|
| `ESMFold2-Experimental-Fast` | ESM-C 6B | 179.3 M | 6352 M |
| `...-Fast-base600M-step1500k` | ESM-C 600M | 171.2 M | **573.3 M** |
| `...-Fast-base300M-step1500k` | ESM-C 300M | 171.1 M | 333 M |

The folding model is the same size in all three; only the tower changes. All are
ungated and MIT - there is no terms dialog to answer and nothing
`build_site.py` has to refuse, which is a real difference from AF3.

🔴 **THE TOWER IS PAIRED WITH ONE CHECKPOINT OF ITSELF, NOT WITH A WIDTH.** The
folding model's layer mix is a `(n_layers + 1,)` vector trained against exactly
the tower named in its `esmc_id` - `biohub/ESMC-600M-1500000`, which is NOT the
general-release `esmc-600m-2024-12`. `tools/esmc/fetch.py` reads the pairing out
of the folding model's own config rather than writing it down twice.

🔴 **AND THE `-step` CHECKPOINTS HAVE NO CONFIDENCE HEAD.**
`confidence_head.enabled: false`, zero confidence tensors in the file - so no
pLDDT, no PAE, no pTM. The fold succeeds and `output_to_pdb` raises `KeyError:
'plddt'` on the way out. LocalFold colours by pLDDT and its scores card is built
on it, so a shipped 600M path needs a checkpoint that has one; the ablation
series is for measuring, not for shipping.

## The first 2x is not compression

🔴 **THE CHECKPOINT IS bfloat16 STORED AS float32.** Not one of the 95.6M
weights sampled has any of its low sixteen mantissa bits set. The tower was
trained in bfloat16 and widened on the way out, and ESMFold2's own forward
casts it straight back (`torch.autocast(device_type=..., dtype=torch.bfloat16)`
around the ESM-C call; the experimental loader's `finalize_esmc` is
`self.esmc.bfloat16()`). So half of the 2.30 GB download is zeros.

Consequences, both measured rather than argued:

* the bfloat16 arm reads relRMS **exactly 0.00e+00** on the weights and on every
  downstream column, which is what a correct harness must say and is also how
  this was found - a scheme that costs nothing is either free or not running.
* float16 reads **3.2e-9**, which is not rounding. bfloat16 holds 7 mantissa
  bits and float16 holds 10, so every value in range is exact; the residual is
  the tail below float16's subnormal floor, and the tower's smallest weights go
  down to 3.9e-10.

**So quote a scheme against the 16 bits that are really there.** A table
starting at "32.00 bits" flatters every row in it by a factor of two.

## What compression costs, on the tensor ESMFold2 actually reads

`python3 tools/esmc/probe-esmc-compression.py`. Three targets - ubiquitin and
the two crystal structures the AF3 side already scores against - and the error
is reported at four depths, because they do not agree and only the last one is
the question:

* `weights` - what a weight-space study would report.
* `states` - the flat mean over ESM-C's 37 hidden states.
* `mixed` - the same, weighted by ESMFold2's learned softmax over them.
* `pair` - the pair representation the folding trunk receives. **This is the
  column that prices a scheme.**

| scheme | bits/w | MiB | weights | states | mixed | pair |
|---|---:|---:|---:|---:|---:|---:|
| float32 (control) | 32.00 | 2187 | 0 | 0 | 0 | 0 |
| **bfloat16** | 16.00 | 1094 | **0** | **0** | **0** | **0** |
| float16 | 16.00 | 1094 | 3.5e-9 | 6.7e-7 | 1.3e-6 | 1.2e-6 |
| int8 g64 sym | 8.25 | 564 | 6.1e-3 | 5.4e-3 | 9.6e-3 | 7.4e-3 |
| int8 g128 asym | 8.25 | 564 | 6.0e-3 | 5.3e-3 | 9.2e-3 | 7.3e-3 |
| int6 g32 asym | 7.00 | 478 | 1.9e-2 | 1.6e-2 | 2.8e-2 | 2.2e-2 |
| int6 g64 asym | 6.50 | 444 | 2.2e-2 | 1.8e-2 | 3.2e-2 | 2.5e-2 |
| **int5 g32 asym** | 6.00 | 410 | 3.8e-2 | 3.3e-2 | 5.7e-2 | **4.4e-2** |
| int5 g32 asym+search | 6.00 | 410 | 3.7e-2 | 3.5e-2 | 5.8e-2 | 4.4e-2 |
| int5 g64 asym+2 outliers | 5.69 | 389 | 3.5e-2 | 3.1e-2 | 5.4e-2 | 4.4e-2 |
| int5 g64 asym | 5.50 | 376 | 4.4e-2 | 3.7e-2 | 6.4e-2 | 4.9e-2 |
| int4 g64 asym+2 outliers | 5.19 | 355 | 7.2e-2 | 6.1e-2 | 1.1e-1 | 8.0e-2 |
| int4 g32 asym | 5.00 | 342 | 7.9e-2 | 6.9e-2 | 1.2e-1 | 9.4e-2 |
| int4 g32 asym+search | 5.00 | 342 | 7.6e-2 | 6.9e-2 | 1.2e-1 | 8.8e-2 |
| int4 g64 asym | 4.50 | 308 | 9.1e-2 | 7.7e-2 | 1.3e-1 | 1.0e-1 |
| int3 g32 asym | 4.00 | 273 | 1.7e-1 | 1.5e-1 | 2.5e-1 | 1.9e-1 |
| int3 g32 asym+search | 4.00 | 273 | 1.6e-1 | 1.6e-1 | 2.5e-1 | 1.8e-1 |
| int2 g32 asym | 3.00 | 205 | 4.0e-1 | 4.1e-1 | 7.4e-1 | 5.2e-1 |

Three things worth taking from it.

**ESM-C quantises like AF3 does.** int5 group 32 asymmetric costs relRMS 3.8e-2
in weight space here against 4.3e-2 on AF3's six biggest tensors
(`tools/analyse_quantisation.py`) - the same scheme, the same neighbourhood, a
different model. The AF3 frontier's shape transfers: the zero point is the big
win at low precision, searching the range is worth a few percent, and pulling
outliers out beats clipping them.

**The tower does not amplify.** weights 3.8e-2 -> mixed 5.7e-2 -> pair 4.4e-2 at
int5. Thirty-six residual blocks pass the error through at roughly unit gain,
and the two LayerNorms in the shim take some of it back off. A stack that
amplified would show it here and would settle the question by itself.

**Searching the range stops paying above four bits.** 7% off the pair error at
int4 (9.4e-2 -> 8.8e-2), nothing at all at int5, and it costs a minute a tensor
sweep. Same for outliers: 5.69 bits with two outliers buys exactly what 6.00
bits buys without them, so it is 0.31 bits of saving for a packing format
nothing here can read.

## And what that costs the STRUCTURE

    python3 tools/esmc/fetch_targets.py --out targets --count 16
    .venv-esm/bin/python tools/esmc/probe-esmfold2-structure.py \
        --pdb-dir targets --seed-spread 2 --csv rows.csv

Sixteen single protein chains, 67-205 residues, X-ray, better than 2.0 A,
**released after the checkpoint's September 2021 cutoff** and filtered so no two
are near-duplicates - the RCSB returns deposition groups together and a plain
"newest 20" came back as nine consecutive entries of the same protein, which
would read as nine measurements and be one.

The folding model is loaded once and `forward` takes `lm_hidden_states`
directly, so an arm replaces exactly the tensor the previous table prices and
nothing else - not the folding weights, not the diffusion noise, not the seed.

🔴 **AND THE SAMPLER'S OWN SPREAD IS MEASURED FIRST, BECAUSE NOTHING ELSE HERE
MEANS ANYTHING WITHOUT IT.** The same float32 weights at three seeds, one
diffusion sample each: **mean 0.99 A, worst 7.06 A** over 32 folds. A scheme
that moves a structure by less than that has not moved it.

| scheme | bits/w | bundle | moved: median | mean | worst | targets past 1 A | median vs crystal |
|---|---:|---:|---:|---:|---:|---:|---:|
| float32 | 32.00 | 2840 MiB | - | - | - | - | **2.52 A** |
| **no language model** | - | - | 16.04 A | 15.52 | 23.77 | **16/16** | 17.33 A |
| int8 g64 sym | 8.25 | 732 MiB | 0.04 A | 0.10 | 0.84 | **0/16** | 2.52 A |
| **int5 g32 asym** | 6.00 | **533 MiB** | 0.08 A | 0.43 | 3.32 | 2/16 | **2.53 A** |
| int4 g32 asym | 5.00 | 444 MiB | 0.12 A | 1.27 | 13.17 | 3/16 | 2.53 A |
| int3 g32 asym | 4.00 | 355 MiB | 0.26 A | 1.71 | 15.22 | 4/16 | 2.57 A |

🔴 **THE DAMAGE IS A TAIL, NOT A DEGRADATION.** The median target moves 0.26 A
even at three bits and the median crystal RMSD is FLAT from float32 to int3 -
2.52, 2.52, 2.53, 2.53, 2.57. What actually changes is how many targets get
knocked into a different basin entirely: none at int8, two at int5, three at
int4, four at int3, with worst cases of 13 and 15 A. So a mean is the wrong
summary here and a median alone is too kind; the honest column is **how many
targets moved further than the sampler moves them by itself**.

🔴 **int5 IS FREE ON THAT READING, AND THE SAME SCHEME IS FREE ON AF3.** Its
mean displacement (0.43 A) and its worst (3.32 A) are both inside the sampler's
own 0.99 / 7.06, and its median crystal RMSD is 2.53 against float32's 2.52.
`tools/quantize_af3.py` records int5 group-32 asymmetric costing AF3 nothing
either (0.66 A against float32's 0.69, inside the spread between diffusion
seeds). Two models, two graphs, one packer, the same verdict - and LocalFold
already has the GPU decoder for it (`src/runtime/quantised-upload.js`).

🔴 **int4 IS THE EDGE AND int3 IS OVER IT**, which is again where AF3 lands.
int4's worst case (13.17 A) is nearly twice the worst the sampler produces on
its own, so it breaks targets seeding does not. The floor is about **444 MiB**
for the 600M pair if a rare flipped fold is acceptable and **533 MiB** if it is
not.

🔴 **AND A pair RELATIVE ERROR IS A POOR PREDICTOR OF ANGSTROMS.** int5's 4.4e-2
and int4's 9.4e-2 are a factor of 2.1 apart in the pair representation and a
factor of 3 apart in mean displacement - but 4x apart in the worst case, which
is the number that decides. Use the pair column to choose which schemes are
worth folding; price them on the structure.

## Halving the tower again costs nothing measurable

The other compression axis is the tower itself, and Biohub publish the ablation:
`...-Fast-base300M-step1500k` is the same folding model against ESM-C 300M (30
layers x 960, 333 M parameters). Same sixteen targets, same protocol:

| | median vs crystal | mean | seed-to-seed |
|---|---:|---:|---:|
| ESM-C 600M, float32 | **2.52 A** | 4.88 | 0.99 A |
| ESM-C 300M, float32 | **2.55 A** | 5.31 | 1.10 A |
| ESM-C 300M, int8 | 2.55 A | 5.32 | |
| ESM-C 300M, int5 g32 | 2.57 A | 5.27 | |
| ESM-C 300M, int4 g32 | 2.54 A | 5.26 | |

**0.03 A of median between a tower of 573 M parameters and one of 333 M**, which
is a third of the seed-to-seed spread. On the tails they differ - 600M's mean is
0.4 A better - but on this set the small tower is not the limiting factor.

That puts the **300M pair at int5 at 361 MiB**, against LocalFold's own
`model-af3-int5` at 264.6 MiB and AF2 monomer at 227 MB. It is the smallest
configuration measured that still folds, and it is in the same class as what the
page already ships.

| bundle | float32 | float16 | int8 | **int5** | int4 |
|---|---:|---:|---:|---:|---:|
| ESM-C 300M + folding model | 1923 | 961 | 496 | **361** | 300 |
| ESM-C 600M + folding model | 2840 | 1420 | 732 | **533** | 444 |
| ESM-C 6B + folding model | 24915 | 12457 | 6423 | 4672 | 3893 |

MiB.

## The catch, which is not about compression at all

🔴 **THE COMPRESSIBLE MODEL IS THE WEAK ONE AND THE STRONG ONE IS NOT
COMPRESSIBLE ENOUGH.** The `base300M` and `base600M` checkpoints are paper
ablations - Biohub's own README says "please use ESMFold2 for research work" -
and the RELEASED `ESMFold2-Fast` folds from ESM-C **6B**, which is 4672 MiB at
int5 and out of reach at any precision this study reached. So a browser build
would be shipping an ablation checkpoint, and its accuracy is what the tables
above measure: **median 2.52 A on held-out targets, mean 4.88 A, and two of
sixteen outright failures** (7ILM at 11.9 A, 9TLM at 26.0 A). Compression is not
the obstacle. Whether that accuracy is worth having when no alignment exists is
the actual question, and it is a product question rather than a measurement one.

🔴 **AND THOSE CHECKPOINTS HAVE NO CONFIDENCE HEAD**, so no pLDDT, no PAE, no
pTM - see the model table above. LocalFold colours by pLDDT, its scores card is
built on it, and its archive writer emits it. That is not a small gap to paper
over.

## Calibrated quantisation, which is what the LLM world does instead

Everything above rounds each weight to the nearest code and looks at nothing
else. The methods that made 3- and 4-bit language models usable all ask a
different question - *what codes keep this LAYER'S OUTPUT the same on real
data* - and they are cheap enough to be worth trying here. `tools/esmc/gptq.py`
implements two, `tools/esmc/calibrate-esmc.py` drives them, and the calibration
set is 288 UniRef50 sequences (three length buckets, 96 each).

| | what it does | cost |
|---|---|---|
| **RTN** | nearest code, per-group affine range | free |
| **imatrix** | keeps RTN's codes, picks each group's RANGE to minimise error weighted by that channel's activation energy. llama.cpp's importance matrix | one forward pass |
| **GPTQ** | quantise column by column, pushing each column's rounding error into the columns not yet done, along the inverse Hessian of `2 XᵀX` | one forward pass + a Cholesky per matmul |
| AWQ | search a per-input-channel scale, fold it into the neighbouring op | comparable |
| SpQR / SqueezeLLM | keep the few extreme weights per group in float16 | measured above: 0.31 bits |
| QuIP# / AQLM | random rotations to kill outliers, then vector codebooks | **needs a different decoder** |
| SmoothQuant, LLM.int8() | move activation outliers into the weights | **irrelevant here** - LocalFold quantises STORAGE and computes in f16/f32, so there are no activation outliers to migrate |

🔴 **THE BLOCKS ARE CALIBRATED IN ORDER AND SO ARE THE FOUR MATMULS INSIDE ONE**,
so every layer corrects for the error its predecessors actually made rather
than for an error nothing will make. Five passes per block instead of one, and
the whole 573M tower takes **70 seconds on an A100**.

🔴 **AND NEITHER METHOD CHANGES THE STORAGE FORMAT, WHICH IS WHY THESE TWO AND
NOT THE OTHERS.** Both emit exactly what `src/runtime/quantised-upload.js`
already decodes: asymmetric codes, one float16 scale and one float16 zero per
group of 32. GPTQ's group axis lines up for free - LocalFold groups 32
CONSECUTIVE elements of a row-major `(out, in)` tensor, which is 32 consecutive
input channels of one output channel, which is what GPTQ calls `group_size`.
`act-order` is deliberately not implemented: it is worth a few tenths of a bit
and it permutes the input axis, so the groups stop being consecutive and the
shader would need the permutation.

### What they buy

Same sixteen held-out targets, same protocol, same seed yardstick (1.00 A mean,
7.16 A worst between two seeds of identical weights):

| arm | bits/w | tower | moved: mean | TM | median | worst | past 1 A |
|---|---:|---:|---:|---:|---:|---:|---:|
| **GPTQ int5** | 6.00 | 410 MiB | **0.39 A** | **0.979** | 0.06 | 3.07 | 2/16 |
| RTN int5 | 6.00 | 410 MiB | 0.44 A | 0.977 | 0.08 | 3.29 | 2/16 |
| **GPTQ int4** | 5.00 | 342 MiB | **1.14 A** | **0.947** | 0.10 | 13.33 | 3/16 |
| imatrix int4 | 5.00 | 342 MiB | 1.16 A | 0.940 | 0.08 | **9.25** | 3/16 |
| RTN int4 | 5.00 | 342 MiB | 1.29 A | 0.933 | 0.12 | 13.26 | 3/16 |
| **imatrix int3** | 4.00 | 273 MiB | **1.29 A** | **0.929** | 0.20 | 11.20 | 3/16 |
| RTN int3 | 4.00 | 273 MiB | 1.72 A | 0.920 | 0.27 | 15.09 | 4/16 |
| GPTQ int3 | 4.00 | 273 MiB | 1.87 A | 0.904 | 0.60 | 14.03 | **7/16** |

🔴 **GPTQ MAKES THE WEIGHTS WORSE AND THE OUTPUT BETTER, WHICH IS THE WHOLE
IDEA.** At int4 it moves the weights 9.78e-2 from float32 where plain rounding
moves them 7.91e-2 - and the pair representation it produces is 7.87e-2 against
rounding's 9.42e-2. A weight-space study would have reported GPTQ as the worse
method. `tools/analyse_quantisation.py` is a weight-space study.

🔴 **CALIBRATION IS WORTH ABOUT ONE BIT, AND ONLY AT THE BOTTOM.** `imatrix`
int3 at **4.00 bits** matches plain int4 at **5.00 bits** on every column - 1.29
A mean, TM 0.929 against 0.933 - which is a whole bit for one forward pass. At
int5 there is nothing left to win: GPTQ's 0.39 A against rounding's 0.44 A is
inside the sampler's own 1.00 A, so both are the same answer.

🔴 **AND IT DOES NOT CLOSE THE int4 -> int5 GAP.** GPTQ int4 is 1.14 A where
plain int5 is 0.44 A. Calibration recovers about 15% of a bit-step, not a whole
one, so **int5 remains the smallest scheme that is free** and 533 MiB (600M) /
361 MiB (300M) stands as the answer.

🔴 **AND GPTQ IS THE WORST ARM AT THREE BITS, WHICH THE PAIR METRIC DOES NOT
SAY.** Its pair error (1.77e-1) beats plain rounding's (1.92e-1) and its folds
are worse on every structural column - mean 1.87 A against 1.72, TM 0.904
against 0.920, and **seven of sixteen targets past the seed noise against four**.
Pushing a large rounding error down the Hessian spreads it over columns that
were fine; at three bits there is more error than there is room to put it. The
two metrics disagree in SIGN here, which is the strongest argument in this file
for folding the structure rather than trusting a tensor norm.

### Would quantisation-aware TRAINING go further

Probably, by about another bit, and it is not obviously worth it.

The literature's ordering is RTN < imatrix/AWQ < GPTQ < QAT, and the step from
GPTQ to QAT is worth roughly what the step from RTN to GPTQ is - which here was
one bit at the bottom and nothing at the top. The right objective would not be
the masked-LM loss ESM-C was trained on: ESMFold2 never reads the logits. It
would be **self-distillation against the float32 tower's own hidden states** -
no labels, just UniRef50 sequences and the teacher already on disk - because
those are literally the tensors the shim mixes. Straight-through estimator on
the codes, the scales and zeros left as they are so the format does not move.

Cost on this A100: teacher forward plus student forward and backward, about 3x
a plain pass, so ~1-2 days for a few thousand steps at 573M parameters. The
prize is the 600M bundle at 355 MiB instead of 533, or the **300M bundle at
about 240 MiB - which is AF2 monomer's 227 MB**.

🔴 **BUT THE BINDING CONSTRAINT IS NOT SIZE.** Both towers already fit in the
same class as what the page ships, the compute is a tenth of an AF3 trunk pass,
and the checkpoints that fit are paper ablations with **no confidence head at
all** and a median 2.52 A on held-out targets with two failures in sixteen.
Spending two days of GPU to move 533 MiB to 355 MiB does not change any of
that. The experiment to run before any training run is whether ESMFold2-Fast is
accurate enough to be worth shipping at ALL, and that is a question about the
model rather than about its bytes.

## Spending bits where the layer mix is heavy does NOT work

ESMFold2 takes **58.8%** of its softmax from the last three of the 37 states and
**0.017%** from states 12-17 between them. That looks like an obvious place to
allocate precision, and it is not, at any budget tried:

| arm | bits/w | pair | the uniform scheme at that budget |
|---|---:|---:|---|
| int4 everywhere, int6 on the last 6 blocks | 5.33 | 7.4e-2 | ~6.4e-2 interpolated |
| int4 everywhere, int8 on the last 3 blocks | 5.33 | 8.5e-2 | ~6.4e-2 |
| int3 everywhere, int6 on the last 6 blocks | 4.50 | 1.4e-1 | 1.0e-1 (int4 g64) |

🔴 **BECAUSE THE MIX SAYS WHERE A STATE IS *READ*, NOT WHERE PRECISION
MATTERS.** Every block feeds every later one, so a cheap early block damages the
expensive late states too - and it damages them through 30-odd more blocks of
residual stream. Uniform allocation beats every graded one measured, by 15-40%,
which is a large enough margin not to be a tuning question.

## The forward pass is not ours alone

`tools/esmc/esmc_forward.py` is written from the block layout
`../alphafold3/converters/esmc.py` documents, deliberately not by importing the
`esm` package, so that a quantisation arm and its reference do not share a
forward. Against `transformers`' own `EsmcForMaskedLM`, on ubiquitin:

| | relRMS |
|---|---|
| hidden states 0-3 | 0.0, 1.7e-7, 2.2e-7, 2.5e-7 |
| hidden state 18 | 5.2e-7 |
| hidden states 34, 35, 36 | 2.1e-6, 2.2e-6, 2.1e-6 |
| masked-LM logits | 1.0e-6 |

Float32 accumulation order, and it pins every convention that could have been
wrong invisibly: RoPE's split-halves rotation, the QK-LayerNorm being over the
FULL `d_model` rather than per head, the fused projection's `[q|k|v]` order, the
SwiGLU's `[gate|up]` order, the ESM3 residual scale, and the final LayerNorm
being applied to the last state only.

🔴 **`transformers` CANNOT LOAD THESE CHECKPOINTS DIRECTLY.** Its ESMC expects
its own converted layout (`esmc.layers.N.self_attn.q_proj.weight`) while Biohub
publish the fused TransformerEngine one
(`esmc.transformer.blocks.N.attn.layernorm_qkv.weight`), so
`from_pretrained` loads **nothing**, reports every parameter MISSING, silently
builds an 80-layer model from a default config and answers with random weights.
The check above builds the state dict by hand. A first attempt that did not
would have been comparing against noise.

🔴 **AND THE FIRST SANITY CHECK WAS A MASKED-LM ONE, BEFORE ANY ORACLE.** Mask
every fifth residue of ubiquitin and see what the head puts back: **86.7%**
against a chance rate of about 5%. That is cheap, needs no reference, and no
wrong RoPE convention survives it.

## The shim, and why the 37 states never have to exist at once

ESMFold2's `LanguageModelEncoder` is: LayerNorm the states, project 1152 -> 256,
mix them by `softmax(base_z_combine)`, downproject, outer product carrying BOTH
a product and a difference, two-layer MLP, LayerNorm.

🔴 **THE MIX IS A CONSTANT AND THE NORM AND PROJECTION ARE SHARED ACROSS
LAYERS**, so `sum_k combine[k] * LN(h_k) @ W` is a running accumulator of
`(tokens, 256)` and the `(37, tokens, 1152)` tensor never needs to be
materialised - 4.5x smaller at any length, and the tower can be streamed one
block at a time. `Shim.accumulate` is that form and it agrees with
`Shim.single` to **1.1e-7**, which is what says it is the same arithmetic.
The downprojection stays outside the sum because it is affine and its bias must
not be added thirty-seven times.

That is the fact that decides device memory. A block is 15.9M parameters, so at
float16 the tower's resident footprint on the GPU is **32 MB one block at a
time**, not 1.15 GB - the same shape of trade `releaseResidentWeights` makes for
AF3's stages.

## Would it run

Arithmetic, not measurement, but the terms are not close:

| | |
|---|---|
| ESM-C 600M, matmuls | **1.15 GFLOP per token** |
| at 300 residues | 359 GFLOP, ~0.3 s at this M2's measured 1.0-1.5 TFLOP/s |
| weight traffic, one pass | 1147 MB at float16, **358 MB read as packed int5** |
| resident, streamed | one block, 32 MB at float16 |

For scale, `bench-trunk.js` measures AF3's own trunk at **3372 ms at 200
tokens**. The language model is a fraction of that, and ESMFold2's trunk is 24
PAIR-ONLY blocks - no triangle attention, no single track - against AF3's 48
pairformer blocks. The compute is not the problem. **The download is the whole
question**, which is why this document is mostly a quantisation table.

| bundle, tower + folding model | float32 | float16 | int8 | int5 | int4 |
|---|---:|---:|---:|---:|---:|
| ESM-C 300M | 1923 | 961 | 496 | **361** | 300 |
| ESM-C 600M | 2840 | 1420 | 732 | **533** | 444 |
| ESM-C 6B | 24915 | 12457 | 6423 | 4672 | 3893 |

MiB. For comparison LocalFold ships `model-af3-int5` at **264.6 MiB** and AF2
monomer at 227 MB. So the 600M pair at int5 is about twice the AF3 bundle and
the 6B is out of reach by an order of magnitude at any precision - which is what
makes the 600M line the interesting one and the reason the question was asked
about it.
