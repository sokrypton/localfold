# AlphaFold 2 here: the template terms, the kernels, the alignment prep

Monomer and multimer. `src/evoformer/`, `src/multimer/`.

🔴 **AF2-MULTIMER'S TEMPLATE TERM RUNS ON EVERY RECYCLE AND NOTHING CHECKED
IT.** `tools/oracle/template_reference.py` computed a numpy reference and wrote
`toy-template.json`; no JavaScript ever read it. Compared at last, the two
disagree - and the comparison localises where:

| | relRMS |
|---|---|
| the input term, all nine features, masked AND with a real template | **2.15e-7** |
| after the first pair block | 1.2e-1 |
| after the second | 1.1e-2 |

`tools/oracle/dump_multimer_template.py` settled it by capturing the module from
AF2 itself, and the GPU is right:

| against AF2, captured | masked | real template |
|---|---|---|
| `src/multimer/template.js` | **6.5e-5** | **3.0e-4** |
| `tools/oracle/template_reference.py` | 1.0e-2 | 2.5e-1 |

🔴 **SO THE numpy REFERENCE'S PAIR BLOCKS ARE WRONG, AND ITS BANNER SAYS SO.**
Its `construct_input` is right - it agrees with the GPU to 2.15e-7, geometry
included - and everything after that is not. It stays because that input term is
a second, independently written reading of the nine features; the checker
asserts exactly that much of it.

## The monomer embedder, which is a third dialect

🔴 **AF2-MONOMER'S EMBEDDER IS A THIRD DIALECT, NOT A THIRD COPY.** Same six
geometry features, but: ONE `Linear` over an 88-channel CONCATENATION rather
than nine summed projections; the whole concatenation masked by the BACKBONE
mask rather than each feature by its own; its distogram NOT pseudo-beta-masked
at all; `use_template_unit_vector` **False** in every shipped monomer config,
so three of the six are deliberately zeroed; and the query pair enters
afterwards through a pointwise attention the other two do not have. Against
AF2's own module: 2.7e-4 masked, 4.5e-4 with a real template.

🔴 **AND `template_mask = 0` IS NOT "A TEMPLATE WITH NO ATOMS".** AF2-monomer
ends with `embedding *= (sum(template_mask) > 0)`, so with no template the term
is EXACTLY ZERO, while a present-but-empty one gives `embedding2d`'s bias
through two pair blocks and a projection - which is not small. LocalFold has
always computed the second, which is what ColabFold does; measuring it against
the first reports relRMS 14.5 for a path that is right.
`dump_monomer_template.py --masked-template` is the arm that means anything.

🔴 **AND COLABDESIGN2 CANNOT CAPTURE MONOMER TEMPLATES AT ALL.** It puts the
monomer on the multimer graph and raises - the two embedders differ and the
weights do not convert - so `dump_monomer_template.py` transforms AF2's
`TemplateEmbedding` with haiku and runs the module alone. Two version traps on
the way: that checkout's config sets `fuse_projection_weights: True` everywhere
while `model_1_ptm`'s weights use the older `layer_norm_input` /
`left_projection` names, and comparing against the shipped `model/` bundle
reports int8 quantisation as a fault - use `model.f32-backup`.

🔴 **AND ITS CROSS-CHAIN MASK REFUSES TO GUESS, LIKE AF3'S.** The first
version defaulted `asymId` to all zeros - every token in chain 0 - which is
right for a monomer and silently lets a template speak across a complex's
chains. AF3 had the identical bug, measured at relRMS 1.09. A template with no
chain ids now raises, and `src/multimer/model.js` hands the ids over from the
feature set. Inter-chain templates are opt-in per slot there too, and moving
the term by relRMS 7.3e-2 is what `tools/gpu/check-multimer-template.js`
asserts, since AF2 has no oracle for something it does not do.

🔴 **AND FEED THE MODULE THE MASKS IT WAS GIVEN.** `__call__<2` is
`padding_mask_2d` and `<3` is `multichain_mask_2d`, and both are all ones in
these dumps because ColabDesign2's featurisation gives one asym_id.
Substituting a two-chain mask of our own scored 7.3e-2 against a module that is
right - a check reporting a fault in its own setup.

🔴 **AND COMPARE AGAINST `model-multimer-f32`, NOT THE SHIPPED BUNDLE.**
`model-multimer` is int8 at block 64 (`dtype: "int8"` in its manifest) and the
references read float32 parameters, so the same correct code scores 6e-3 on the
input term against one and 2e-7 against the other. An hour went into that
before the manifest was read.

## The gates: one end-to-end, four differential

🔴 **AND AF2 NOW HAS AN END-TO-END GATE, WHICH THE DIFFERENTIAL ONES ARE NOT.**
A per-kernel checker says one kernel still computes its own operation. It
cannot say the assembled model still folds, and after three kernel rewrites
that was the whole of AF2's coverage here. `tools/gpu/fold-af2.js` folds a
59-mer through the driver the page uses and prints a checksum over every
coordinate, plus mean pLDDT, pTM and the backbone CA-CA geometry. Run it, stash
the change, run it again: at 128 rows and at 512 rows with a recycle, the tree
before this session's kernel work and the tree after agree to every digit.

It synthesises its alignment from the query, so the 512-row kernels run without
fetching anything, and it opens `./model/` by directory - `loadModel` resolves
the monomer family to Hugging Face, and a regression tool should not pull
227 MB. That makes it a fingerprint, not an oracle: it does not know what
AlphaFold would say.

🔴 **AF2's KERNELS NOW HAVE FOUR DIFFERENTIAL GATES, BECAUSE IT HAD NONE.**
`npm run test:gpu` cannot load Dawn here and `test/fixtures/evoformer/` is
gitignored, so every `test/*.gpu.test.js` covering AF2 is unrunnable - which
left its transition, its outer product mean, its attention projection and the
residual form of its triangle output projection with nothing checking them at
all. Each new checker writes its own CPU reference in its own file, because a
reference that shares code with the thing it checks tests nothing, and each
uses ragged shapes and ragged masks so the bounds checks and the masking are
actually exercised. They are differential, not oracle: they say the kernel
computes the operation, not that AlphaFold agrees.

🔴 **AND AlphaFold 2 SAVES ITS BEST PASS, NOT ITS LAST.** Recycling is not
monotonic and AlphaFold's own pipeline ranks its outputs; the criterion is
ColabFold's `rank_by: auto` - the multimer score for a complex, mean pLDDT for a
monomer - and the search starts from the last pass so a tie keeps the more
converged one. The scores card and the status line report the saved pass, and
the line says `saved pass N of M` when it is not the last, because the play bar
is still sitting on the last one.

## Preparing the alignment

🔴 **PREPARING AN AF2 ALIGNMENT WAS 525 ms OF MAIN-THREAD JAVASCRIPT AND
NOBODY HAD MEASURED IT.** Three loops, none of them subtle, all of them once
per residue: `parseA3m` ran a regex and a `toUpperCase` per character and built
each row by concatenation; `makeA3mFeatures` looked each residue up in a `Map`
through a one-character string; and the nearest-centre assignment - extras x
centres x residues, 1024 x 508 x 59 - ran once per RECYCLE.

| | before | after |
|---|---|---|
| `parseA3m`, 30,000 rows | 307 ms | **85** |
| `makeA3mFeatures`, 200 residues x 10,000 rows, one pass | 403 | **91** |
| ...`tools/fixtures/test.a3m`, four passes | 525 | **75** |

`tools/gpu/fold-af2.js`'s checksum is unchanged at -2105827, which is what
says the clustering still clusters the same way.

🔴 **AND `(x - 0x01010101) & ~x & 0x80808080` IS THE WRONG ZERO-BYTE TRICK IF
YOU ARE COUNTING.** It is the one everyone reaches for and it is exact only for
"is there a zero byte ANYWHERE": a borrow out of a zero byte marks its
neighbour too. Used to count agreeing residues it changed 1024 assignments'
checksum from 195329 to 199057 - a wrong answer that still looks like a
histogram. `~(((x & 0x7f7f7f7f) + 0x7f7f7f7f) | x) & 0x80808080` has no borrow
between bytes.

