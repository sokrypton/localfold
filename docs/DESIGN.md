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

**What argues the other way, and the condition that would flip this.** The established pattern in this project is
one repository per tool, same conventions, each at its own Pages URL — `sokrypton/mpnn` and
`sokrypton/rfd4-proteina-webgpu` are both that, and both self-contained. Hunter is not self-contained, which is why
it is the exception. But if the goal is a **product** — one site offering hallucination and generative design behind
one UI — then one repository is right, and the fold path gets mirrored in exactly as MPNN already is
(`tools/sync-mpnn.py` + `cpu/design/mpnn/SOURCE.md`). 🔴 The cost of that is real and worth stating: the fold
kernels are the fastest-moving part of this tree, and a design loop pinned to a stale mirror of them is a bad place
to debug from. Decide the product question first; the rest of this document is unaffected by the answer.

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
| `design.html`, `web/design.js` | the new page's shell: controls real, loop not wired | works, `npm run test:design` |
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

### Style

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
| **What is being decided, and what is not** | per-residue roles coloured in both the 3D view and the track: free, fixed, tied, motif, target |
| Why a cycle won | its objective, against the best so far |
| **Why a cycle was rejected** | the physical number, not a verdict: `ring 3.87 Å`, `0/12 Watson-Crick pairs`, `framework RMSD 17.6 Å` |
| The whole hunt | every cycle a frame, so it plays back — v1 did this and it is worth keeping |

That rejection row is the one most likely to be cut for time and the one worth keeping most: a model can score
itself well and produce the wrong molecule, so a page that reports only the objective is a page that lies.

### Input should be picking, not typing

Hotspots, epitope contacts, fixed positions and motif residues are all *sets of residues*. Typing `54,56,58` is the
v1 interface and it is the part least like `index.html`'s ease. **Picking them in the viewer is the target.**
`sokrypton/rfd4-proteina-webgpu` already went this way — its guide documents picking in the viewer, a query
language, selection paths and PyMOL commands — so the idea is proven in this family of pages; scale it down to what
a design page needs rather than adopting the whole editor.

### py2Dmol is ours, so the viewer can change to suit this

🔴 **Beerware, `so3@mit.edu`** — "do whatever you want with this stuff" as long as the notice is retained. So where
the viewer is the limiting factor, the answer is to extend it rather than work around it. What this page is likely
to need and should be checked for first: per-residue colouring by *role* rather than by value, a frame API that
takes a labelled cycle, and residue picking that emits a selection the page can read.

🔴 **The vendored copy is a mirror and must not be edited in place.** `web/vendor/SOURCE.md` pins
`sokrypton/py2Dmol` by commit; the workflow is to change it upstream, rebuild its bundle, re-run
`python3 tools/sync-py2dmol.py`, and check with `--check`. Editing `web/vendor/py2Dmol.full.min.js` directly is how
a change gets silently reverted by the next sync.

### UI already written, and ours to reuse

`sokrypton/mpnn`'s own page has the pieces a design UI needs and this repository has never taken: a sequence view, a
residue trace, and a **constraint table** — which is the fixed/tied-position interface, already built against the
same engine. Read those before designing ours. They are not in the mirror: `sync-mpnn.py` copies only the import
closure, not the app.

---

## 6. The target: modality parity with RFD4-Proteina

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

1. **Scaffold and the page shell.** `design.html` in the light style, the ensemble and designer pickers, no loop
   wired. Gate: `npm run test:site` resolves it, `tools/mobile-layout.py` passes.
   Alongside it, audit py2Dmol for the three things section 5 needs - role colouring, a labelled frame API, residue
   picking - and record which already exist, since that decides whether the viewer needs an upstream change early.
   The gate that does run is **`npm run test:design`** (`tools/check-design-page.py`): it loads the page in headless
   Chrome, requires the module to actually run, and requires the automatic designer to MOVE when the target gains a
   ligand - a note reading "Auto → SolubleMPNN" is also what a broken wire prints, since soluble is the fallback.
   🔴 **Both of those gates are already red on this checkout, for reasons that predate this work and are not
   design's to fix:** `npm run test:site` fails because `model-boltz2-int5/` has drifted from its manifest module
   (it is missing `weights-08.int5.bin`), and `tools/mobile-layout.py` reports four failures about `index.html`'s
   own option grouping. Verified by re-running both with this work stashed: identical failures. Until they are
   green, a new page's module graph has to be resolved by hand.
2. **The RunSpec.** aminx's field names and semantics on our engine, with her attribution in place; fixed and tied
   positions reaching `cpu/design/mpnn/`. Gate: a seeded RunSpec reproduces its sequence exactly; new bridge tests.
   **Unlocks** inverse folding, and is the primitive antibody mode and motif scaffolding are built on.
3. **The loop on one folder.** v1's method, new code, diffed against `cpu/design/sample-sequence.js`'s schedules.
   Gate: `test/hunter-loop.test.js` adapted and green; the CDP driver completes a real run.
4. **Validity checks.** Ring closure, Watson-Crick pairing, antibody framework RMSD, as vetoes with recorded
   reasons. Gate: each check has a fixture that fails it and one that passes.
5. **Ensembles.** Rotate the folder per cycle. Gate: a run with two folders alternates them and records which folded
   each cycle. 🔴 Reproducing her ipTM result is a *measurement*, not a gate — and it needs n≈30 per arm.
6. **Then, in her order of value:** epitope targeting by contact residues (filter first), PTM targets, cyclic,
   antibody CDR mode.
7. **Then the expensive modalities**, in this order: ligand and nucleic binders (checkpoints already mirrored),
   inverse folding, antibody and nanobody, then geometric conditioning and the two motif forms, then enzyme design.
   `partial_diffusion` stays an open question until someone reads her warm-start analysis.

**Unsettled, needs a decision when reached:** whether the first page exposes more than one MPNN noise level; whether
the ONNX sampler earns its dependency (step 2 is where to measure it); whether pocket conditioning is worth a
Boltz-2-only path.

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
| Does py2Dmol already do role colouring, labelled frames and residue picking? | worker | how much of section 5 needs an upstream change |
| How geometric conditioning works - template, contact term, or an RMSD-to-motif term in the objective | unassigned | motif scaffolding, enzyme design |
| Does `partial_diffusion` have a Hunter analogue? Read her `13_warmstart.py` | unassigned | that one modality |
