# Protein design in LocalFold: Protein Hunter, and what it borrows

Written 2026-10-24 for whoever builds the design page. It is a plan, not a record of work done: **nothing in
section 4 onward has been built or measured here.** Every number attributed to an upstream is that upstream's own
measurement on its own stack, reproduced nowhere in this repository yet, and labelled as theirs.

---

## 1. Where design lives, and why

**Protein Hunter goes here, in this repository. RFD4-Proteina stays in its own.** The split is by dependency, not
by subject:

| | Needs a folding model? | Home |
|---|---|---|
| **Protein Hunter** | Yes — the method *is* fold ↔ MPNN in a loop | here |
| **RFD4-Proteina** | No — generative, its own encoder, trunk and decoder | `sokrypton/rfd4-proteina-webgpu` |
| **MPNN** | No, but Hunter needs it | here, mirrored from `sokrypton/mpnn` |

Hunter's dependency on folding is not a library call, it is the whole AF3/Boltz-2/ESMFold2 WebGPU stack — most of
this repository. Three things follow from putting it beside that stack rather than across a repository boundary:

1. **Generator ensembles cost almost nothing here.** Rotating the folding model every cycle is ProteinHunter2's one
   solid measured win, and this page already runs nine of those models.
2. **The validity checks have their chemistry to hand.** Ring closure in Å, Watson-Crick pairs formed, antibody
   framework RMSD — these need `shared/chem` and the heads, not a fold's own confidence.
3. **The site build resolves the whole module graph** (`tools/build_site.py`) and has no bundler, so a page in
   another repository that imports this one's fold path would have to carry a mirror of it.

**🔴 AND THE CONDITION THAT FLIPS THIS HAS BEEN MET, so the table above is the reasoning and not the answer.**
The page is to carry a **Method** picker - `{ProteinHunter, RFD4-Proteina}`, Protein Hunter the default - with the
controls below it changing to suit (Sergey, 2026-10-24). One page offering both methods needs both engines reachable
from it, and that is exactly the product case named here as the thing that would change the recommendation. So:

- **RFD4-Proteina's ENGINE comes here**: its shared pipeline, its WebGPU engine and its float32 CPU reference, which
  are already this repository's idiom and collide with nothing. Its parity checks come with it.
- **Its PAGE does not.** AtomWorks Studio, Mol*, the Rust/wasm annotation studio and the nebula assets stay behind:
  this page is `index.html`'s light theme with py2Dmol, and two viewer stacks in one page is a cost with no return.
- **Its weights are fetched from the port that publishes them** (Sergey, 2026-10-24), not copied here and not
  re-uploaded: the loader's chain is a local export → `sokrypton/rfd4-proteina-webgpu`'s Pages URL → Hugging
  Face, so this repository's build keeps carrying 0.0 MiB of parameters and there is no second 420 MB copy to
  drift from the first. 🔴 That makes the other repository's Pages URL a **dependency**: if it stops
  publishing, RFD4 stops loading here, and our own Hugging Face copy is the fallback to reach for then rather
  than a thing to do twice over now.
- 🔴 **Its 293 KB tensor manifest IS tracked here, and it can drift.** The tests read it without a network,
  but it records byte offsets into shards it does not contain - so if that port re-exports, these offsets describe
  the old shards while the fetched ones are new, every tensor reads from the wrong place, and nothing says so. It
  is the failure `model-boltz2-int5/` has today. `weights/SOURCE.md` pins the commit; a `--check` against
  upstream, as `tools/sync-mpnn.py` and `tools/sync-py2dmol.py` both have, is the missing piece.
- 🔴 **And NVIDIA's terms become this repository's problem.** Today every model here has terms that
  `tools/model-terms.py` can express. An NVIDIA research checkpoint offered from the same page has to be expressible
  the same way, and nobody has read those terms yet. **That is the real cost of merging, and it is the user's call,
  not a detail to be absorbed quietly.**

What does NOT change: the two methods remain different machines. Hunter hallucinates through a folding model and
needs the fold stack; RFD4 generates and needs none of it. They share a page, an input model (below) and a viewer -
not a code path.

**Hygiene, so design can live here without touching what ships.** All four already exist as patterns:

| Want | How |
|---|---|
| Off the live site until it is good | `design.html` in `build_site.py`'s `OPTIONAL` list, not linked from `index.html` |
| No design weights in the Pages build | gitignored, hosted like the nine fold bundles (the build carries 0.0 MiB of parameters) |
| In-flight work off the shipping branch | a branch, as `metal/design` already does |
| Its own gates | a `test:design` script |

---

## 2. What is already here

| path | what | state |
|---|---|---|
| `webgpu/design/hunter-loop.js` | the v1 loop: fold a mostly-`X` sequence, MPNN on the previous backbone, re-fold, keep the best cycle. `fold` and `design` are injected, so it tests without a GPU | works, green tests |
| `cpu/design/sample-sequence.js` | the per-cycle schedules: percent-`X` ramp, first-cycle proline omission, alanine bias and guard | works, green tests |
| `cpu/design/mpnn/` | the MPNN engine, mirrored from `sokrypton/mpnn` @ `a80942f0` | works |
| `cpu/design/mpnn-bridge.js` | the fold → MPNN join; the interface is a PDB string | works |
| `cpu/design/designers.js` | the checkpoint registry and the automatic choice | works |
| `web/public/mpnn/` | **4** checkpoints at noise 0.2 (soluble, protein, ligand, NA), ~16 MB | checked in |
| `design.html`, `web/design.js` | the page: a Method picker, the controls, and a row plus a sequence line per cycle | works, `npm run test:design` |
| `web/hunt.js` | what makes the loop real: a true AF3 fold and MPNN redesign, with progress across the whole hunt | works, end to end in the gate |
| `web/viewer.js` | the py2Dmol wrapper: builds on the first frame, appends the rest, keeps the camera. Written for the withheld pages; **`design.html` is the first page to load it** | works |
| `tools/check-design-page.py` | loads the page, runs its module, requires the auto-designer to respond | works |
| `web/hunter.js` | the v1 page | 🔴 **retire.** Its page was withheld (`.gitignore`) and is not in this checkout |
| `tools/protein-hunter-in-page.py` | drives a real run over CDP and checks the table came back | works, needs a page |
| `test/hunter-loop.test.js`, `test/sample-sequence.test.js`, `test/mpnn-bridge.test.js`, `test/designers.test.js` | the loop, the schedules, the bridge, the registry | green |

**Keep the loop and the schedules, write a new page.** The schedules encode v1's method and are checked against its
numbers, so they are the reference to diff a new loop against, not dead weight. The page is replaced outright.

---

## 3. The four upstreams, and what we take from each

### 3A. `sokrypton/mpnn` — ours, the MPNN engine
Pure JS plus a 26 KB WASM kernel, no runtime dependency. **15 checkpoints** where this repository mirrors 4:
ProteinMPNN and SolubleMPNN at noise 002/010/020/030, LigandMPNN at 005/010/020/030, NA-MPNN, and two membrane
MPNNs. `sync-mpnn.py` mirrors only the import closure; Potts/trMRF are deliberately excluded.

Take: **more checkpoints as the page needs them** — Hunter needs LigandMPNN for ligand targets and NA-MPNN for
nucleic binders, both already mirrored, and the noise levels are a knob v1 never exposed. Its `FUTURE.md` lists
side-chain packing (LigandMPNN), NA-MPNN specificity mode and NA-MPNN pair bias as open; none is needed for a first
page. Its `app/constrainttable.js` is a constraint-table UI worth reading before designing ours.

🔴 The mirror is one commit behind upstream's HEAD and `sync-mpnn.py --check` is run by nothing. Run it before
trusting the mirror, and note its recorded upstream path is a different machine than this one.

### 3B. `maraxen/aminx` — Marielle Russo, MIT, **permission granted** (2026-10-24)
JAX/Equinox ProteinMPNN and LigandMPNN, with a browser half built for precisely our constraints — her
`docs/browser_integration.md` says it is "for porting the exported ProteinMPNN sampler into a plain static site
(for example a local ProteinHunter.org build). No bundler, no server-side Python." Her path is ONNX Runtime Web on
the wasm provider, pinned `onnxruntime-web@1.30.0`, with the whole autoregressive sampler inside one ONNX graph, so
noise and every knob are runtime inputs.

**Attribution is a condition of the licence, not a courtesy.** MIT requires her copyright and permission notice to
travel with any copy or substantial portion. Any file we take keeps a header naming `maraxen/aminx`, the commit, and
`Copyright (c) 2025 Marielle Russo`; `cpu/design/mpnn/SOURCE.md` gains a second section for her; her
`NOTICE-weights.md` travels with any converted weights we take.

Taking her sampler wholesale is a **trade**, and the first page should not make it:

| | `sokrypton/mpnn` (here now) | aminx's browser export |
|---|---|---|
| Checkpoints | 15 incl. **Ligand, NA**, membrane | ProteinMPNN only (`v_48_020`) |
| Length | unbounded | **L ≤ 256**, two buckets |
| Download | 3.3 MB default | 19.4 MB / 33 MB per bucket |
| Runtime deps | none | `onnxruntime-web` |
| Sampling | JS loop | in-graph, likely much faster — **unmeasured here** |
| Knobs | partial | seed, temperature, bias, omit, **fixed and tied positions**, SplitMix64 determinism |

Hunter needs Ligand and NA, which her browser half does not export. So:

- **Take now: her RunSpec semantics.** The knob surface our bridge lacks — fixed positions, tied positions,
  decoding order, bias/omit, and a seeded PRNG. **Specified in docs/DESIGN-RUNSPEC.md**: ten fields with their
  defaults, the eleven input tensors, the SplitMix64 draw, and the tie semantics. Adopt the *semantics and the
  field names* against our engine.
  🔴 **One finding there changes how reproducibility must be implemented:** she pads the structure to its length
  bucket *before* building the inputs, so the PRNG is consumed over the padded length. The same structure at the
  same seed therefore samples differently depending on whether it lands in the 128 or the 256 bucket. A
  reimplementation that draws over the real length is not her sampler, and the difference is silent.
- **Take now: her parity discipline.** A per-checkpoint ledger that states where parity is incomplete rather than
  implying it is finished.
- **Measure later, decide later: the ONNX sampler.** A speed arm against the JS engine for the ProteinMPNN case,
  interleaved, on the same structures. Adopt only if it wins enough to justify the dependency and the download.
- **Later, if ever: her extra families.** Her `NOTICE-weights.md` carries PottsMPNN, LASErMPNN and Proton-PottsMPNN
  conversions, with upstream notices — model families this repository does not have.

### 3C. `yehlincho/ProteinHunter2` — private, no licence file, **permission granted**
The repository carries no LICENSE, so the default is all rights reserved — but the author has given permission to
build on this work (Sergey, 2026-10-24, covering every upstream named here). So code may be ported, not only the
method.

🔴 **Permission is not a licence, and the difference shows up in what has to be written down.** A grant from
an author is enough to use the work; it is not a public grant that travels with a copy. So anything taken from an
unlicensed repository carries, in the file, what it came from and that it is used with permission and when — because
the next reader cannot infer a private grant from a repository that still says nothing. Where an upstream DOES carry
a licence, that licence's own terms are what travel: `maraxen/aminx`'s MIT notice, `sokrypton/py2Dmol`'s Beerware
notice.

The same applies backwards: `webgpu/design/hunter-loop.js` is documented as "transcribed from
`boltz_ph/pipeline.py`" from her v1, and that provenance line should say it is used with permission now that it is.

Her v2 is v1's loop on one folding runtime instead of six vendored backends. The ideas, and what each costs us:

| her idea | what it is | cost here |
|---|---|---|
| **Generator ensembles** | rotate the folding model every cycle so a design satisfies several models | low — we have nine |
| **Validity checks** | *physical* measurements: ring closure Å, WC pairs formed, antibody framework RMSD | low, and the most valuable thing in her repo |
| **Epitope targeting** | contact residues plus a contact-rejection filter | low (the filter), high (trained conditioning) |
| **Antibody CDR mode** | ANARCI bookkeeping on the binder sequence | medium |
| **Cyclic peptides** | a runtime flag, plus refusing to archive an open ring | medium |
| **PTM / modified-nucleotide targets** | things v1 cannot express | medium — this repo already folds both |
| **Gumbel sampling, warm start, hotspots, chain masking** | `ph2/{gumbel,steering,hotspots,chainmask}.py` | unread; medium |
| **Pocket conditioning** | Boltz-2's *trained* `ContactConditioning`, which she notes ships upstream with its input hardcoded off | 🔴 Boltz-2 only, cannot generalise |

**Her headline measurement, and how to treat it.** AF3-judged ipTM on PD-L1, n≈30 per arm, **her stack, not ours**:

| models in the ensemble | her AF3 ipTM | 95% CI |
|---|---|---|
| 2 | 0.653 | [0.577, 0.729] |
| 3 | 0.599 | [0.520, 0.678] |
| 1 | 0.519 | [0.436, 0.601] |
| 5 | 0.515 | [0.433, 0.596] |
| 6 | 0.480 | [0.408, 0.551] |

She states what it establishes — 2 beats 1 (P=0.99, d=0.60), 3 beats 1 (P=0.92), both beat 5 and 6 (P=0.99–1.00) —
and what it does not: 2 vs 3 (P=0.83), or whether 6 is worse than 1 (P=0.76). **Carry those limits with the
numbers.** Her proposed mechanism, that weak members get an equal vote, is her explanation and not a measured cause.

🔴 **The sharpest thing she found, and the reason the validity checks come early:** a model can run, score itself
well, and produce the wrong molecule. In her table ESMFold2 reports fine confidence and gets 0.000 of a DNA duplex's
Watson-Crick pairs. A design loop that trusts a fold's own confidence cannot see this.

### 3D. `sokrypton/rfd4-proteina-webgpu` — ours, MIT code
Not a dependency of this page. It matters here as **proof the conventions port**: it is a working deployed WebGPU
model in this repository's exact idiom: the same shared / webgpu / cpu split under an rfd4p subtree, with the CPU
half a float32 reference the GPU is held to, GPU checks on the same `gpu-chrome.mjs` harness with
`LOCALFOLD_STOCK_FLAGS`, and its own AGENTS and HANDOFF notes. Follow that CPU-reference-plus-oracle pattern for any
new kernel here.
(🔴 Its directories are described rather than written as paths on purpose: `test/imports-resolve.test.js` reads a
bare `shared/...` as a claim about OUR tree, and catching another project's path written as though it were ours is
exactly what that gate is for.)

🔴 Its weights come from the `nvidia/RFD4-Proteina` checkpoint and carry **NVIDIA's own terms, which nobody here has
read and which must not be accepted on the user's behalf.** If Hunter ever offers RFD4 as a backbone generator,
those terms enter this repository's licence story and belong in `tools/model-terms.py`.

---

## 4. The loop's contract

Keep v1's shape — it is why the loop tests without a GPU — and widen it for ensembles.

```
runDesign({ fold, design, spec }) -> async generator of cycles
```

- **`fold(chains) -> { structure, confidence }`** stays the only thing the loop knows about folding. Ensembles are a
  *list* of folders the loop rotates; nothing else in the loop changes.
- **`design(structure, runspec) -> { sequence, logProbs }`** takes aminx's RunSpec field names.
- **The objective** stays v1's: per-chain-pair ipTM for a complex, mean pLDDT / 100 for a monomer.
- **Validity is separate from the objective and can veto.** A cycle that produces the wrong molecule is not a
  candidate however it scores. Open ring, unpaired duplex, broken antibody framework: rejected, with the reason
  recorded and shown.
- **Every cycle is a frame**, so a run plays back. v1 did this and it is worth keeping.

---

## 5. The page

New file, `design.html`. **`web/hunter.js` is not reused.**

**The goal is an interface as easy as `index.html`'s, highly interactive, with enough visual feedback that it is
always obvious what the machine is doing.** That is a harder target here than for folding, and for one reason: a
fold is a single action with one answer, while a hunt is minutes of nested work — runs, then cycles, then a fold and
a design inside each cycle. 🔴 **An interface that only shows the final answer is wrong for this.** Every cycle must
land something visible the moment it exists, and at any instant the page should answer three questions without being
asked: *what is it doing now*, *what has it found so far*, and *why did it keep or reject the last thing*.

### The input model: every object is a chain, and a chain is a contig

🔴 **AND IT IS ALREADY WRITTEN.** `shared/rfd4p/design-spec.js` came in with the engine: pure, no imports, an
atom array in and a plain object out. It reads a structure's condition annotations and returns per-chain roles, length
ranges, an RFdiffusion contig string and the task that amounts to. The span axes below are its
`condition_coordinate` (geometry) and `condition_sequence` (sequence), with `condition_index` saying whether a motif's
POSITION is pinned - which is the whole difference between indexed and unindexed scaffolding - and
`expsegmin`/`expsegmax` carrying a generated span's range.

Its seven example structures are the proof the abstraction holds, and are now a test
(`test/rfd4p-design-spec.test.js`, no Studio needed - the annotations are in the mmCIF and the parser keeps them):

| example | task it reads | contig |
|---|---|---|
| `uncond_short` | unconditional | `100` |
| `ppi_pdl1` | binder | `50-120/0 B1-115` |
| `motif_idx_1prw` | motif | `5-20/A1-20/10-25/A22-41/5-20` |
| `motif_unidx_1prw` | motif | `60-105/[A1-40]` |
| `sm_sam` | small-molecule binder | `120-165 + SAM` |
| `dna_7m5w` | binder | `A1-9/0 B1-10/0 120-150` |
| `5o45_cropped` | redesign all | `115` |

The two motif rows are the same protein and the same motif, differing only in whether its position is pinned - the
brackets. That is one model producing seven modalities with no per-modality branch, which is the reason to build on
it rather than invent a second vocabulary.

🔴 **It is RFD4's vocabulary, though, not a shared one yet.** Protein Hunter has no notion of a length range or
of generating geometry, so generalising these roles across both methods is the next piece of input work.

🔴 **ONE INPUT MODEL FOR BOTH METHODS, AND IT IS THE PART WORTH GETTING RIGHT.** The page is a list of objects you
add to, and each object is a chain — which in the general case is a *contig*: an ordered run of spans, some given and
some to be made. That is RFdiffusion's own notion, and it turns out to describe Hunter's inputs too.

A span has **two independent axes**, and almost every modality in section 6 is a combination of them:

| | geometry | sequence |
|---|---|---|
| a target chain | given | given |
| a binder | **generated** | **free** |
| a motif | given | given |
| the scaffold around a motif | **generated** | **free** |
| inverse folding | given | **free** |
| a ligand or nucleic target | given | given (or N/A) |

Read down those two columns and the modality list stops being a list of features:

- **protein binder** = one given chain + one object that is a single generated span of fixed length
- **unconditional** = one object, one generated span, nothing else
- **inverse folding** = one object, geometry given, sequence free — no generated span at all
- **motif scaffolding** = one object, spans `[generated 5-15][given: motif][generated 10-30]`; *indexed* fixes where
  the motif sits, *unindexed* leaves the generated lengths as ranges and lets the method choose
- **antibody CDR** = one object whose framework spans are given and whose CDR spans are free in sequence
- **enzyme design** = motif scaffolding plus a ligand object

🔴 **AND THE SPAN AXES ARE THE RESIDUE ROLES.** The colours in the legend below are not a separate concept: `target`
is given/given on a chain nobody designs, `motif` is given/given inside a designed one, `free` is sequence-free,
`fixed` is a sequence-given position inside a free span. One model, shown once, in the track and in the 3D view.

**What differs per method is which spans are legal, not what a span is.** Hunter cannot generate geometry — it
hallucinates through a folding model, so a "generated" span means *fold it from X and see*, and a length RANGE is
not something it can express today. RFD4 generates geometry natively and takes ranges. So the Method picker's real
job is to grey out the spans the chosen method cannot honour, **and to say why** rather than silently dropping them.

🔴 **A length RANGE on Hunter is the first thing that will be asked for and the first thing to refuse clearly.** v1
took a single length; a range implies a search over lengths, which is a different loop and is not in section 7's
order of work.

### Layout and style

- **A Method picker at the top** — `{ProteinHunter, RFD4-Proteina}`, Hunter the default — with the controls beneath
  it changing to suit. 🔴 **What is shared stays put when the method changes**: the object list, the viewer, the
  seed, the downloads. A control that moves when the method changes is a control the reader has to find again, which
  is `index.html`'s own stated rule about its model picker. And a control the chosen method cannot honour is
  disabled **with its reason**, not removed — a control that vanishes teaches nothing.
- `index.html`'s light theme: `web/vendor/py2Dmol.app.css` plus `web/localfold.css`. 🔴 **Not** `web/style.css`, the
  dark theme the withheld pages used.
- Every rule anchored to an id or class this project introduces; the existing palette, the 600 + 8 + 340 shell, the
  58 px inputs, option rows content-sized rather than hand-sized.
- One viewer: py2Dmol, as `index.html` has it. Not Mol*.

### What has to be visible

| | shown as |
|---|---|
| Where the run is | run *i* of *n*, cycle *k* of *m*, and which of fold / design is in flight |
| **Which model folded this cycle** | named per cycle — with ensembles the folder rotates, and a result is meaningless without it |
| The sequence changing | a residue track per cycle, the `X` positions visibly filling in |
| **What is being decided, and what is not** | **done for the two roles a Hunter run has** — the designed chain takes `free`, every other chain `target`, in the 3D view and named in the legend. `fixed`, `tied` and `motif` arrive when the RunSpec's positions reach the page |
| Why a cycle won | its objective, against the best so far |
| **Why a cycle was rejected** | the physical number, not a verdict: `ring 3.87 Å`, `0/12 Watson-Crick pairs`, `framework RMSD 17.6 Å` |
| The whole hunt | **done** — every cycle is a frame of one py2Dmol object, labelled `cycle_N`, so the play bar walks the hunt |

That rejection row is the one most likely to be cut for time and the one worth keeping most: a model can score
itself well and produce the wrong molecule, so a page that reports only the objective is a page that lies.

### Input should be picking, not typing

Hotspots, epitope contacts, fixed positions and motif residues are all *sets of residues*. Typing `54,56,58` is the
v1 interface and it is the part least like `index.html`'s ease. **Picking them in the viewer is the target.**
`sokrypton/rfd4-proteina-webgpu` already went this way — its guide documents picking in the viewer, a query
language, selection paths and PyMOL commands — so the idea is proven in this family of pages; scale it down to what
a design page needs rather than adopting the whole editor.

### py2Dmol is ours, and the audit says what is actually missing

🔴 **Beerware, `so3@mit.edu`** — "do whatever you want with this stuff" as long as the notice is retained. So where
the viewer is the limiting factor, extend it rather than work around it. **docs/DESIGN-PY2DMOL.md** audits the three
things this page needs, so only what is missing gets built:

| what the page needs | verdict | what to do |
|---|---|---|
| per-residue colouring by **role** | **EXISTS** | use `setColor(colour, { positions })` — it takes an array or a `Set`, and writes an `advanced` colour on the object. Verified at `src/parts/embed.js:448`. Nothing upstream needed |
| a **labelled** frame per cycle | **PARTIAL** | frames, stepping, the play bar and a `py2dmol-frame-change` event all exist, and `frame.name` is already stored — but it is drawn by the full app's label element, which the embed path does not create. The upstream change is small: let the embed show a frame's name |
| residue **picking** the page can read | **PARTIAL** | picking, selection and a change event exist. The gap: the event carries **no detail**, so a page must read `v.selection` — 0-based indices — and translate them itself via the renderer's chain and residue-number arrays. Either live with the translation or give the event a payload upstream |

So the viewer is much closer than assumed: the role colouring that looked like the big ask is free, and the two
partials are a label and an event payload rather than new machinery.

🔴 **The vendored copy is a mirror and must not be edited in place.** `web/vendor/SOURCE.md` pins `sokrypton/py2Dmol`
by commit; change it upstream, rebuild its bundle, re-run `python3 tools/sync-py2dmol.py`, and check with `--check`.
Editing `web/vendor/py2Dmol.full.min.js` directly is how a change gets silently reverted by the next sync.

### UI already written, and ours to reuse

`sokrypton/mpnn`'s own page has the pieces a design UI needs and this repository has never taken: a sequence view, a
residue trace, and a **constraint table** — which is the fixed/tied-position interface, already built against the
same engine. Read those before designing ours. They are not in the mirror: `sync-mpnn.py` copies only the import
closure, not the app.

---

## 6. The target: modality parity with RFD4-Proteina

### Where it stands, measured

Every row below was run through `design.html` on an M5. 🔴 **"runs" is not
"designs well"** - a 24-mer against a lone HEM is a hard case and its cycles
went over the alanine ceiling, which is the loop's veto working rather than a
pass. What these establish is that the job can be *expressed* and *reaches the
engine*.

🔴 **AND "THE PRESET LOADS" IS NOT "THE PRESET RUNS".** Every RFD4 row below was
marked ✅ on the editor round-tripping its preset - and when they were actually
generated, **seven of the eight conditioned ones failed**. The reason is in
section 3D's note now: a preset without `specMode` is driven by its structure's
own ANNOTATIONS, and its contig string is written in the original entry's
numbering. `motif_idx_1prw` asks for `A52-71` where its example CIF has chain A
at residues 0-42. Running it from the spec gave "Preserved segment 'A52-71'
matched no atoms". Every column below is now a generation or a hunt, not a
round trip.

| | Protein Hunter | RFD4-Proteina |
|---|---|---|
| unconditional | ✅ objective 0.802 | ✅ 100 tokens / 30 steps |
| protein binder | ✅ objective 0.589, two folders rotating | ✅ 183 tokens (`ppi_pdl1`) |
| ligand target | ✅ runs, LigandMPNN chosen automatically | ✅ 165 tokens (`sm_sam`) |
| nucleic target | ✅ runs, NA-MPNN chosen automatically | ✅ 147 tokens (`dna_7m5w`) |
| a designed DNA / RNA chain | ✅ runs (2026-10-28), NA-MPNN designs it; recovers a duplex partner, the objective does not yet see pairing (DESIGN-TODO.md) | — |
| from a structure | ✅ its sequence is read out | ✅ its coordinates are conditioned on |
| motif, indexed | ✅ objective 0.835, the motif survives exactly | ✅ 67 tokens |
| motif, unindexed | ❌ Hunter cannot choose where the motif sits | ✅ 112 tokens |
| inverse folding | ✅ objective 0.906, whole chain resampled | ✅ 67 tokens |
| partial diffusion | ❌ no analogue | ✅ 67 tokens over 10 of 20 steps, as its fraction asks |
| antibody CDR | ✅ objective 0.882, framework held either side | ❌ no preset |
| cyclic | 🔴 **blocked in the fold path, not here**: `cyclicPeriod` is READ by `cpu/af3/diffusion/atom-encoder.js` and produced by nothing, so the model takes the feature and no featuriser computes it | ❌ no preset; `cif.js` knows a `cyclic-pseudo-peptide` chain type |
| PTM target | ✅ objective 0.356, a phosphoserine swapped in by CCD code | ⚠️ through a loaded structure that already carries it |
| epitope / hotspots | ✅ one field, reached by a contact filter: fold, then reject a cycle that is not touching | ✅ the same field, as a bracketed `C_HOT` condition |
| labelled spans (secondary structure) | ❌ nothing to condition on; it folds what the sequence gives | ✅ 54 tokens (`custom_cath_monomer`), which is the 54-residue protein its own description names |

`--cases` re-runs the Hunter column, asserting the automatic designer follows the
job and that held residues survive; **`--presets` generates all ten of RFD4's
presets**, which is the arm that caught the seven failures above; `--generate`
is the quick unconditional one.

🔴 **That primitive now exists, and it was one checkbox.** A given span is HELD
by default; ticking `redesign` keeps its place and length and lets the designer
rewrite its letters. At the scale of a whole chain that is inverse folding; at
the scale of part of one it is an antibody's CDR; and a given span between two
Design spans is a scaffolded motif. All three run, and the gate asserts the held
residues SURVIVE in the designed sequence - the only check that catches a hold
which is not holding, since the run looks identical either way.

🔴 **AND TWO SHIPPED PRESETS WERE CONDITIONING ON ALMOST NOTHING.**
`evalPathSelection` splits a selection on commas *before* it splits a path on
slashes, so a residue field reading `54,56,66,115` is four tokens of which only
the first is a path - and an unbracketed `16-35` reaches `parseInt` and becomes
the single residue 16. Measured on `motif_idx_1prw`: the bare form selects
residue 16 where the bracketed form selects 16-35. `ppi_pdl1`'s four hotspots
were written the same way. Both are bracketed now, with the finding recorded at
the top of `shared/rfd4p/presets.js` and the DSL pinned by
`test/rfd4p-selection.test.js`.
🔴 **One question that leaves open for whoever owns those presets:** bracketed,
`[16-35,52-71]` still selects only 16-35, because the example's chain A is
renumbered and has no residue 52-71 at all. So those numbers may belong to the
original 1PRW rather than to the cropped example. Bracketing is strictly closer
to what the preset says it does; whether the numbers are right is not something
this repository can settle.

What is left is genuinely different in kind, not more of the same. 🔴 **cyclic is
blocked below this page**: `cyclicPeriod` is read by the atom encoder and
produced by nothing, so adding it is fold-path work in the shipping path and not
a design feature. **Unindexed** motif placement needs the loop to search over
where the motif sits, which is a different loop. And **labelled spans** (CATH, secondary structure) need a row that
carries a name and a polymer type.



The standing goal is that Protein Hunter eventually offers **every modality RFD4-Proteina does**. Its Studio ships
nine (its Studio examples manifest), plus a DNA case among its page examples and motif scaffolding in both an
indexed and an unindexed form.

🔴 **The two tools get there by different means, and that decides what each modality costs.** RFD4 is a generative
flow-matching model: a modality is a conditioning input. Hunter hallucinates through a folding model: a modality is
a constraint on the sequence, the geometry, or the objective. So some of RFD4's modalities are nearly free here and
some need machinery that does not exist yet.

| RFD4 modality | Hunter's route | Cost here |
|---|---|---|
| `unconditional` | single chain, all-`X`, fold and redesign | free — v1 does it |
| `protein_binder` | the core case | free — v1 does it |
| `small_molecule_binder` | ligand target, LigandMPNN designs the binder | low — the checkpoint is mirrored |
| DNA / RNA binder or target | nucleic chain, NA-MPNN designs the binder | low — the checkpoint is mirrored |
| `antibody_design`, `nanobody` | CDR-only redesign, framework held fixed | medium — ANARCI-style bookkeeping, plus **fixed positions** |
| `inverse_folding` | the designer alone on a given backbone: one design step, no re-fold | low — a degenerate loop, mostly UI |
| `motif_scaffolding` (indexed) | motif residues pinned in sequence **and** placed in geometry | 🔴 **high** — fixed positions gives the sequence half; the geometry half needs a template or a geometric term in the objective |
| `motif_scaffolding` (unindexed) | as above, but the loop chooses where the motif sits | 🔴 **higher** — placement becomes part of the search |
| `enzyme_design` | active-site geometry around a substrate | 🔴 **high** — motif scaffolding plus a ligand, i.e. the two hardest together |
| `partial_diffusion` | **not a native concept.** The nearest analogue is warm-starting an AF3-lineage sampler from an existing structure instead of from noise | 🔴 **unknown — this mapping is an inference, not an established equivalence.** Yehlin's `scripts/analysis/13_warmstart.py` suggests she has measured something in this direction; unread |

**What this reorders.** Fixed and tied positions stop being a nicety and become the load-bearing primitive: they are
what antibody mode, inverse folding and both motif modalities are built on. That is already step 2 below, which is
the right place for it. Geometric conditioning — a template, a contact term, or an RMSD-to-motif term in the
objective — is the one genuinely missing piece, and it gates the three expensive modalities. Nothing about it is
designed yet.

🔴 **Parity of capability is not parity of quality.** That a modality runs says nothing about whether the molecule
is right — which is the whole point of the validity checks in step 4, and why they come before the hard modalities
rather than after.

---

## 7. Order of work

Each step ends green and is committed before the next begins.

1. ~~**Scaffold and the page shell.**~~ **Done**, and py2Dmol audited
   (docs/DESIGN-PY2DMOL.md: role colouring EXISTS, frames and picking PARTIAL).
   The gate is **`npm run test:design`** (`tools/check-design-page.py`): it loads the page in headless
   Chrome, requires the module to actually run, and requires the automatic designer to MOVE when the target gains a
   ligand - a note reading "Auto → SolubleMPNN" is also what a broken wire prints, since soluble is the fallback.
   🔴 **Both of those gates are already red on this checkout, for reasons that predate this work and are not
   design's to fix:** `npm run test:site` fails because `model-boltz2-int5/` has drifted from its manifest module
   (it is missing `weights-08.int5.bin`), and `tools/mobile-layout.py` reports four failures about `index.html`'s
   own option grouping. Verified by re-running both with this work stashed: identical failures. Until they are
   green, a new page's module graph has to be resolved by hand.
2. **The RunSpec — built and tested, NOT yet wired.** `cpu/design/runspec.js` translates aminx's field names into
   `Model.sample`'s options, with 10 tests. 🔴 **Nothing routes through it yet:** `web/hunt.js` still hands
   `temperature` and `omit` to the loop, which hands them to `designChain`, which builds its own bias. Wiring it
   means teaching the bridge to accept pre-built sample options, and it would have no visible effect until the page
   can express a fixed position - so the honest order is **UI for positions first, then the wiring**, and
   `runspec.js` is the thing both will meet at. **Unlocks** inverse folding, antibody mode and motif scaffolding.
3. ~~**The loop on one folder.**~~ **Done.** `web/hunt.js` supplies the loop's `fold` and `design` - a real AF3
   fold and a real MPNN redesign - and `web/design.js` draws a row and a sequence line per cycle as it lands.
   `npm run test:design` runs the whole thing end to end: a 31-residue target, a 12-residue binder, one cycle,
   about twelve seconds on an M5 off the local bundle - weights, fold, redesign, fold, two rows, best marked.
   🔴 It also checks the loop's own rule through the page: **cycle 0 can never be best**, because its sequence
   is the mostly-X draw the run started from, and on a 12-mer of mostly X it folds deceptively well (pLDDT 99 in a
   stub, and it was the top scorer).
4. **Validity checks — the framework is in, with two checks in it.** The loop takes a `veto` callback, because
   whether a MOLECULE is right is geometry and the loop has no business parsing structures; it keeps the reason on
   the record, and the page prints the reason rather than re-describing it. In it today: the alanine ceiling, and
   **contact rejection** (`cpu/design/contact.js`) - CA to CA at `shared/heads/contact-threshold.js`'s 8 Å
   protein-protein convention. 🔴 That one is what makes an epitope mean anything for Hunter, which cannot condition
   on hotspots the way RFD4 does: fold, then reject a cycle whose binder is somewhere else. Proved to fire by asking
   for residue 9999 - every cycle rejected, `touches 0/1 hotspots (nearest none, needs 8 A)`.
   Still to add, each reachable only with the modality that produces it: ring closure, Watson-Crick pairing,
   antibody framework RMSD.
5. ~~**Ensembles.**~~ **Done.** The folder rotates per cycle - not per run, since a design only has to satisfy
   several models if the model changes UNDER the sequence being refined. `npm run test:design` reads the rotation
   off the table and fails if the column does not alternate: a four-cycle run with two folders gives
   `af3, opendde, af3, opendde`, and every folder is loaded before the first cycle rather than mid-bar. Two
   AF3-lineage models stay resident together on a 16 GB M5.
   🔴 Reproducing her ipTM result is a *measurement*, not a gate, and it needs n≈30 per arm. Nothing here has
   measured whether ensembles help; only that they run.
6. **Then, in her order of value:** epitope targeting by contact residues (filter first), PTM targets, cyclic,
   antibody CDR mode.
7. **Then the expensive modalities**, in this order: ligand and nucleic binders (checkpoints already mirrored),
   inverse folding, antibody and nanobody, then geometric conditioning and the two motif forms, then enzyme design.
   `partial_diffusion` stays an open question until someone reads her warm-start analysis.

**Unsettled, needs a decision when reached:** whether the first page exposes more than one MPNN noise level; whether
the ONNX sampler earns its dependency (step 2 is where to measure it); whether pocket conditioning is worth a
Boltz-2-only path; and how RFD4 reaches the page - its engine is in the tree and the Method picker says plainly that
it is not joined to it, which needs the 420 MB fetch and a generation to be worth claiming.

---

## 8. The worker protocol

Design tasks may be delegated to the `agy` CLI (Antigravity, `~/.local/bin/agy`).

- **Model and effort:** Gemini 4 Argon at medium by default —
  `agy -p "<task>" --model gemini-4-argon-medium --effort medium --mode auto --output-format json`, resumed with
  `--conversation <id>`. Low for mechanical work; high for numerics that must match a reference, or a task that
  already failed at medium.
- 🔴 **The worker never commits, pushes, or rewrites history.** It ends each task with a report: what changed and
  why, measurements with their arms, which gates ran and what they said, and what it did not do. We read the diff,
  run the gates, and write the commit message from that report plus our own check.
- **Porting code from an upstream is allowed** (permission granted by every author, 2026-10-24), and every ported
  file must name where it came from. For a licensed upstream that means carrying the licence's notice; for an
  unlicensed one it means recording that it is used with permission and when.
- The repository invariants in `AGENTS.md` and `CLAUDE.md` bind it: never widen a tolerance or rewrite a baseline to
  pass a gate, GPU benchmarks one at a time and interleaved, no bundler, and `package-lock.json` and
  `localfold-fixes-for-sergey.md` are never staged.

---

## 9. Open items

| item | who | blocks |
|---|---|---|
| ~~Permission from the upstream authors~~ — **granted, all of them** (2026-10-24) | done | nothing |
| Write the permission into each ported file, and onto `hunter-loop.js`'s existing provenance line | worker | nothing; do it as code lands |
| 🔴 NVIDIA's RFD4-Proteina weight terms — **unread, and NOT covered by the authors' permission**: NVIDIA is not one of them. Not to be accepted on the user's behalf | Sergey | only if Hunter offers RFD4 as a generator |
| Is the goal a product site, or a tool beside the folder? | Sergey | section 1's repository choice |
| `sync-mpnn.py --check` against `sokrypton/mpnn` HEAD | worker | step 2 |
| Whether Marielle will export LigandMPNN and NA-MPNN to the browser | Sergey / Marielle | would change 3B's trade entirely |
| ~~Does py2Dmol already do role colouring, labelled frames and residue picking?~~ — **audited, docs/DESIGN-PY2DMOL.md**: colouring EXISTS, frames and picking PARTIAL | done | nothing |
| 🔴 RFD4's GPU parity checks need `model.safetensors` - FLOAT weights from **NVIDIA's own** HF repo, under their terms, a large download. The harness itself works | Sergey | running those checks here |
| Rename `weights/` to `model-rfd4p-int8/` for this repository's convention, now the hosting is settled | worker | nothing |
| How geometric conditioning works - template, contact term, or an RMSD-to-motif term in the objective | unassigned | motif scaffolding, enzyme design |
| Does `partial_diffusion` have a Hunter analogue? Read her `13_warmstart.py` | unassigned | that one modality |
