# AlphaFold 3 in LocalFold

Where the AF3 port stands, what it costs, and the things that have already been
got wrong once. Written to be read before touching any of it.

`AGENTS.md` holds the invariants; this holds the state.

## What works

A protein chain typed into `index.html` folds with AlphaFold 3 entirely in the
browser: featurisation, trunk, diffusion and confidence, no server. Pick **AF3**
in the Model dropdown.

- **From a sequence, not a dump.** `src/af3/featurise.js` builds AF3's whole
  batch in JavaScript. Checked array-by-array against AF3's own batch for 6MRR
  and for a three-chain complex: `node tools/oracle/check_af3_featurise.js`.
- **Complexes**, chains separated by `:`. Chain identity comes from
  `src/input/chains.js` - the same `chainIdentity()` AlphaFold-multimer uses.
- **Two samplers.** *Flow* (default) draws once at the top of the schedule and
  walks it down deterministically, ~8 calls. *Diffusion* is AF3's own stochastic
  sampler, 20+ steps. Both are seeded.
- **Recycles** for AF3 as well as AF2.
- **MSAs**, through the page's own alignment controls - search, paste or upload,
  shared with both AlphaFold 2 models. `src/af3/msa-features.js` is the whole of
  the adapter. On the 59-residue demo sequence a 512-row alignment moves pLDDT
  55.8 -> 65.7 and costs about 2 s (the MSA stack goes from nothing to 239 ms at
  512 rows).
- The trajectory animates as it computes, and the finished structure gets
  py2Dmol's PAE panel and prediction-quality card.

### Accuracy, against crystal structures

| | RMSD | TM | notes |
|---|---|---|---|
| 6MRR, flow 8 | 0.63-0.70 | 0.95 | across four seeds |
| 6MRR, diffusion 200 | 0.66 | 0.953 | |
| 1QYS (Top7), flow 8 | 0.86 | 0.947 | |
| 1QYS, diffusion 200 | 0.93-1.12 | 0.92-0.94 | across four seeds |

Flow matches or beats the 200-step sampler with ~25x fewer denoiser calls, on
both proteins measured. It is two proteins, both small designed alpha/beta
folds, both single-sequence - an observation, not a result.

### Speed, 68 tokens

A flow-8 fold is about 7 s, of which the trunk is ~1 s. It was ~150 s when the
first end-to-end fold ran.

## Running it

    python3 -m http.server 8080          # then open /index.html

    # The GPU lane. Dawn (`npm run test:gpu`) cannot load on this macOS; Chrome
    # can. Every checker and bench is a module for this harness.
    node tools/gpu-chrome.mjs tools/gpu/<module>.js [--flags]

    node tools/gpu-chrome.mjs tools/gpu/fold.js --sequence=GWSTELEK... \
      --mode=flow --steps=8 --recycles=1 --model=/model-af3-int5/manifest.json
    python3 tools/score_fold.py <the log> --reference 6mrr-crystal.pdb

    # The featuriser, including the MSA path, against AF3's own batch.
    python3 tools/oracle/dump_af3_trunk.py --blocks 0 --a3m rows.a3m --out d.json
    node tools/oracle/check_af3_featurise.js d.json rows.a3m

    node tools/gpu-chrome.mjs tools/gpu/bench-blocks.js   # AF2 vs AF3 per block
    node tools/gpu-chrome.mjs tools/gpu/bench-ab.js --skip=single

Checkers, all differential against the CPU reference:
`check-af3-{triangle,transition,grid-attention,single-attention,opm,msa-attention,embedder,template,block,msa-block,trunk,confidence,diffusion-*,atom-*,sampler-gpu,target-feat-gpu}.js`

## Traps

Each of these cost real time. They are in the code as `🔴` comments too.

**Benchmarks drift up to 3.2x between processes.** Two numbers from two
invocations of the same bench cannot be compared. `tools/gpu/bench-ab.js`
alternates A and B inside one process and reports medians; within a process the
spread is about +-10 ms. A whole round of per-pass profiling was thrown away
after this was ignored - the tell was "everything skipped" measuring *slower*
than "full".

**The parts do not sum to the whole.** Skipping one pass measures what it costs
on top of everything else pipelining, not its share. Individually the pair
track's passes account for 79 ms; together they cost 348.

**pLDDT is not the check.** It comes off the trunk and can look healthy over
coordinates that are not a molecule. A batch with one broken gather folded a
17 A spaghetti at pLDDT 55 with 15 A between consecutive CA. Backbone CA-CA is
what a wrong sampler cannot fake, which is why the fold prints it.

**A gather's `count` is not decoration.** `convert()` sizes its output from it,
so a gather without one silently yields a zero-length tensor and the model runs
anyway.

**`sigma` is a claim about the input, not a dial.** It reaches the network
through the Fourier noise embedding. Feeding a black hole at sigma 4 - "this
structure is nearly right" - diverges; at sigma 2560 - "this is noise, ignore
it" - the same input gives 1.39 A.

**The diffusion head has its own five reference embeddings**, distinct from the
conditioning module's. Same shapes, different weights. Reusing one for both
type-checks.

**Four conditioning weights exist twice in the checkpoint**, `..._1` and
unsuffixed, identical shapes. Dropping the suffix loads clean and gives the
wrong `target_feat`.

**AF3's MSA gap is 21, not 31.** The alphabet is 21 protein codes, then the
gap, then the nucleotides - the gap is in the MIDDLE of the 32-wide one-hot. A
gap at 31 type-checks, folds, and tells the model every gap is an unknown
nucleotide. Related and just as quiet: the deletion counts stay RAW, because
AF3's embedder does the `atan(n/3)` squashing itself, and AF2's featuriser does
it on the way in.

**AF3's unpaired chain merge is NOT block-diagonal.** `merge_msa_features` pads
each chain's alignment to the deepest and concatenates along the TOKEN axis, so
merged row r is chain A's row r beside chain B's row r, for every chain, with no
notion of entity - there is no `block_diag` anywhere in AF3. AlphaFold-Multimer
is the one that distinguishes: `_merge_homomers_dense_msa` merges copies of one
sequence densely and block-diagonalises only distinct entities, so it agrees
with AF3 on a homo-oligomer and differs on a heteromer.
`mergeUnpairedChainA3ms`, which block-diagonalises copies too, is neither: it
belongs to the AF2-MONOMER hack, where the +200 residue offset stands in for
chain awareness. AF3 has `mergeRowAlignedChainA3ms`. Two consequences, both
silent: the block-diagonal
form halves the information in every row and doubles the depth to carry it; and
for a HOMO-oligomer the row-aligned merge already IS the paired construction, so
supplying a paired block as well duplicates every row. That combination made a
homodimer fold worse with an MSA than without one, and a monomer shows neither.

**AF3's `msa` is two blocks and its `profile` is over one of them.** The array
the model reads is the paired block followed by the unpaired one; the profile
and deletion_mean are computed upstream, per chain, over the unpaired block
ALONE. So a 32-row A3M gives a 33-row `msa` - the query appears twice, because
an absent paired block becomes the query alone - and a profile over 32. Deriving
the profile from the array instead double-counts the query in every column, and
looks completely reasonable. `unpairedFrom` is threaded through featurise.js for
exactly this.

**AF3 resamples the reference conformer per residue instance** - fixed bond
lengths and angles, random torsions - so a baked table cannot reproduce a dump.
Measured cost of baking one: 0.01 A of structure. `check_af3_featurise.js`
therefore holds the chemistry (bonded pairs, from the bond graph) and lets the
torsions go.

**This build's py2Dmol renderer has no `setColor` or `setColorScheme`** - those
belong to the embed build. Drive the app's own colour `<select>` instead.
Writing `renderer.colors` directly is overwritten on the next recompute.

## Performance, and what has already been tried

The pairformer went 3468 ms -> 621 ms over 48 blocks at 59-68 tokens, and AF3's
block is now 1.09x AF2's evoformer block - for a block with no MSA row
attention, no column attention and no outer product mean in it.

What worked, in order of size:

1. **Weight layout.** q, k and the gate were stored `(out, channels)`, so
   consecutive threads read 128 floats apart and nothing coalesced. Transposed
   at pack time into `v`'s `(channels, out)`. This was nearly all of the win.
2. **Row-blocked projection.** One pair row per workgroup meant one multiply per
   weight loaded - arithmetic intensity 1. Eight rows per workgroup, measured
   against 4 and 16.
3. **Submission window 16.** Each `onSubmittedWorkDone` is a full pipeline
   drain: 1 gives 881 ms, 8 gives 622, 16 gives 609, 48 gives 607.
4. **Flash attention** for the grid attention - online softmax, one thread per
   query, vec4 accumulators. Correct and the better kernel, but worth only 2.7%.
5. **`target_feat`'s atom encoder onto the GPU**: 5267 ms -> 160 ms, 33x. It
   reuses `Af3AtomEncoderGpu` by zeroing the three inputs this encoder does not
   have; they enter through bias-free linears of layer-normed values, so zeros
   contribute exactly zero. Checked by `check-af3-target-feat-gpu.js`.

What did **not** work, measured, so it is not retried:

- **Uploading all 48 blocks' weights once** instead of per block: 30% *slower*
  (640 -> 830 ms), reproducibly. Recycling a few buffers beats holding 384, and
  the up-front burst serialises ahead of all compute.
- **Caching bind groups**: exactly zero, 636-639 either way, though ~1,680 are
  created per stack.
- **Binding AF2's attention kernel directly** rather than rewriting on its
  principles: it takes a uniform for its shape, folds `1/sqrt(d)` into the query
  projection and applies the gate itself. The adapter was wrong at relRMS
  2.96e-1 and cost more to find than the rewrite took.

Where the remaining time goes, at 59 tokens: 348 ms of dispatch work and 284 ms
of per-block overhead that is **not** uploads (40 ms) and **not** bind groups
(0 ms). About 5 ms a block in the encoder, submit and validation path is
unexplained. That is the next lead and it is a small one.

## Open

- **The paired block is not verified against the live server.** A heteromer now
  gets one, through `ticket/pair` with `mode=pairgreedy`
  (`generateMmseqs2PairedMsa`), and the parser and assembly are covered by tests
  against mocked archives - but no run against api.colabfold.com has confirmed
  that `pair.a3m` really arrives NUL-separated per query with equal depths. That
  is one heteromer search away. A homo-oligomer needs none of this: its unpaired
  merge is already the paired construction.
- **The A3M parser is narrower than AF3's alphabet.** `src/input/a3m.js` rejects
  B, Z, J, O and U, which AF3 maps to D, E, X, X and C. The codes are in
  `AF3_MSA_CODES` and unreachable through that parser - for AlphaFold 2 too, so
  widening it is a change to all three models rather than to AF3's path.
- **No ipTM**, which is what a complex is actually judged by. The confidence
  head emits PAE and PDE; pTM and ipTM are not implemented.
- **Per-atom conditioning is still on the CPU** - a real gap against AGENTS.md,
  now a few hundred ms rather than five seconds. It has no kernel.
- **Templates raise** rather than compute: the geometry features are
  unverifiable without a reference.
- **AF3's block is still 1.09x AF2's** for strictly less work.

## The weights

🔴 **`model-af3-int5` is DeepMind's AlphaFold 3, not OpenFold3.** Every manifest
in the lineage says so - `model.name` is `alphafold3`, `source` is
`af3.bin.zst`. They carry a Prohibited Use Policy. `tools/build_site.py` reads
the manifest and refuses to publish them without
`LOCALFOLD_ACCEPT_MODEL_TERMS=alphafold3`, and the deploy workflow demands the
same repository variable before it untars the release - that check is on the
only path that actually publishes.

An Apache-2.0 bundle needs an OpenFold3 export. The blob is at
`~/af3_converted_cd2/of3_ported_weights.bin.zst` and
`tools/export_af3_model.py --model openfold3` exists, but the OF3 dialect turns
on four branches the stock graph does not have - a column-attention pair-bias
swap, a symmetrised bond matrix, an element index shift, and Fourier weights
read from the checkpoint - and none of those are implemented or verified here.

int5 costs nothing measurable: 1405 MiB -> 265 MiB, and 6MRR folds to 0.66 A
against float32's 0.69, which is the spread between diffusion seeds.
