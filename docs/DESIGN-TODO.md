# The design page: what was asked for, and what is actually built

🔴 **THIS IS A LIST OF REQUESTS AND THEIR STATE, NOT A RECORD OF MEASUREMENTS.**
`docs/DESIGN.md` is the plan and the evidence; this is the queue. A line here
saying DONE means it was verified the way the item itself describes - and where
that verification was weak, the line says so rather than claiming more.

Status is one of **done** (built and checked), **partial** (built, with a named
gap), **queued** (agreed, not started), **open** (wants a decision first).

---

## 1. The input model — the big one

> *"page opens for the first time, user should be able to add span, if it's a
> structure, they can upload a structure, specify the span (auto filled in),
> they can then add another structure if they want to add another span, default
> to dropdown menu showing the first structure, or it can be a second structure
> they load for example... or a sequence. we need potential for multistructure
> support."*

**queued — nothing started, and it is the largest item here.**

Today the page holds **one** loaded structure (`loaded`, a single binding in
`web/design.js`), and a `From structure` span names a chain of *that* one. Every
piece of the request above is blocked on that being a list rather than a
variable:

| what is asked | what exists | what it needs |
|---|---|---|
| upload a structure from the span row | one structure, loaded from the options strip | a per-row source, and the upload moved to where the span is |
| the span auto-filled from the structure | typed by hand | read the chain's real first/last residue on load |
| a second structure for a second span | impossible | `loaded` becomes a keyed list |
| dropdown defaulting to the first | no dropdown | a source picker per row, defaulting to the first loaded |
| a sequence instead | exists (`Sequence` span kind) | unchanged |

🔴 **AND IT CHANGES WHAT A CONTIG MEANS, WHICH IS THE PART TO GET RIGHT FIRST.**
A contig segment is written `A18-132` — a chain letter and a residue range, with
no room to say *which structure*. With two structures loaded, `A18-132` is
ambiguous, and RFD4 is handed one input file. So multistructure needs a decision
before any UI: either the page **merges** the chosen chains into one input and
relabels them (so the contig stays RFD4's own notation and the relabelling is
ours), or it refuses more than one structure per job and multistructure means
only "pick which one this row came from". The first is more useful and is real
work; the second is an afternoon. **This is the open decision.**

## 2. Fixing geometry during generation

> *"I wonder if there is a way for us to FIX things during diffusion? like
> target or motifs etc, similar to how rfdiffusion would've done it, or it can
> be specified as template."*

Three different answers, because there are three different samplers:

- **RFD4-Proteina — done, and native.** Given segments *are* held coordinates;
  that is what the contig is. Hotspots and `C_SEC` are the other two channels.
  Nothing to build.
- **As a template to the folder — queued, and the cheap win.** Hunter folds
  every cycle, and the fold path has an oracle-checked template slot
  (`npm run test:template`: 5CAJ 17.9 Å → 0.14 Å with a self-template, AF2
  monomer 21.2 → 2.4). Handing the motif in as a template each cycle *is*
  motif-fixing, through a mechanism that already works and is already gated.
- **Re-noising inside AF3's diffusion — open, and much heavier.** RFdiffusion's
  actual mechanism re-asserts the motif at every step rather than once. This is
  a sampler change in `cpu/af3/diffusion/`, it is not gated by anything that
  exists, and it overlaps code another agent works in. Do the template route
  first and see whether it is enough.

## 3. Secondary structure in Protein Hunter

> *"how is secondary structure encouraged during protein hunter diffusion?"*

**open — and the premise needs correcting first: Hunter has no diffusion.** It
is a hallucination loop (fold a mostly-X sequence → MPNN redesigns → re-fold),
so there is no trajectory to intervene in. Secondary structure is conditionable
on RFD4-Proteina only, via labelled spans → `C_SEC` (`web/contig-ui.js`), and
the page refuses the control for Hunter in those words:

> Protein Hunter cannot be asked for a secondary structure; it folds whatever
> the sequence gives.

Two ways in, ranked:

1. **As a term in the objective.** Hunter is a search over `objectiveOf`, and
   secondary structure is measurable from the backbone that already exists every
   cycle. The label becomes something the loop *searches for* rather than
   something imposed — which is honest about what Hunter is.
2. **As a template** — see item 2; the same mechanism covers both.

A third, biasing MPNN toward helix/strand formers at labelled positions, is
indirect and weak. Not recommended.

## 4. Chain colour indicators — **done**

> *"we need chain color indicators... this is for both colors in viewer and also
> for the span blocks, which which block will be which color"*

- **done** — the palette, the blend, and the chain walk, in `web/contig-ui.js`
  (`CHAIN_COLOURS`, `chainColour`, `shade`, `chainOfSpans`), held to `chainsOf`
  by `test/design-chain-colours.test.js` and watched failing. 🔴 The two walks
  must agree or the indicator is *worse* than nothing: a swatch naming one chain
  while the structure paints another reads as an answer.
- **done** — the span rows carry a `::before` bar, so the grid keeps exactly
  three children and the phone breakpoint is unchanged.
- **done** — the viewer takes hue from the chain and expresses role as a
  treatment of it. The legend is built from the spans; it listed five fixed
  roles, two of which nothing painted.
- **gated at runtime**: `tools/check-design-page.py` compares the row bars
  against the legend swatches, because they come from two walks of the same
  spans and a disagreement points at the wrong chain, which reads as an answer.

## 5b. The generation was drawn as one picture — **done**

The flow runs its steps and the page pushed only the finished structure, so the
play bar — which hides itself below two frames — never appeared and RFD4 looked
static beside a hunt that animates. Live frames are opt-in because each is an
all-atom decoder pass, so the page asks for a frame COUNT and the cadence falls
out of it: 16 decodes whether the run is 20 steps or 200. The gate asked for
`>= 1` frame, which a single finished structure satisfies — it asks for a
trajectory now.

## 5c. `weights/` is not a directory in this repository — **done**

The shards were never committed, but the top-level directory was called
`weights/` and advertised something the repository does not have. It is
`model-rfd4p-int8/` now, matching the nine fold bundles, holding only the config
and the tensor manifest. 🔴 Upstream's layout is NOT ours to change, so
`tools/check-rfd4p-manifest.py` maps our paths onto theirs; it reaches the
network by necessity and reports both files identical to the pinned commit.

## 5d. A contig segment was mis-split — **done**

Reported from the page: `A16-35` read as chain `A1`, first 6, because the chain
group was greedy and could absorb digits. RFD4's own parser is letters-only with
an OPTIONAL range. 🔴 A round trip cannot see it — chain "A1" with first 6
re-joins to exactly "A16-35" — which is why it survived a byte-for-byte check
against six presets. Only the fields catch it.

## 5. Side chains in the viewer — **done**

> *"no sidechains"*, then *"still seeing sidechains"*

Fixed in `1cce2d28`, and worth reading because the first fix looked right and
did nothing: `showSidechains()` runs after **every** frame while `onPaint` runs
on the **first** only, so the page hid them once and frame two put them back.
Skipping the show is also not enough — py2Dmol builds them regardless — so the
viewer now takes `sidechains: false` and hides per frame, on the same cadence as
the rebuild it is undoing. 🔴 Two measurements of this were worthless first: a
hand-rolled PDB with its atom-name column off by one (both arms read zero), then
`renderer.sidechains`, which is a persistent table and truthy either way. Pixels
on a real fixture settled it.

## 6. Everything else already landed

| request | where |
|---|---|
| align frames step to step, fade what is not designed | `cpu/design/align.js`, `232bba86` |
| show which parts are kept | held residues painted, `viewer.colourResidues` |
| beam search — N sequences per round | `webgpu/design/hunter-loop.js` |
| one folder, not a rotation | options strip |
| a validation model at the end | `cpu/design/validate.js` |
| hide the designer, make it automatic | `#designerGroup` hidden; the note stays |
| diffusion steps controllable, advanced hidden | `#advancedGroup` disclosure |
| cut-off range fields | `.contig-number` 9ch |
| beauty and organisation pass | one card, `index.html`'s idiom |

## 6b. Pocket conditioning — a port gap, and reachable

🔴 **"Protein Hunter has no pocket conditioning" was the wrong claim, and it was
mine.** It reads as a property of the method; it is a property of this tree.

Genuine Boltz-2 takes a `pocket` constraint — a binder chain plus the residues
it should contact — and Hunter folds with whatever is in `FOLDERS`, so a folder
honouring one *would* condition the fold on the pocket. What is actually
missing, verified here:

- the AF3-lineage featuriser has **no constraint path at all** — nothing in
  `shared/af3/featurise/` builds one;
- `FOLDERS` is `{af3, openbind0, opendde}`, and **boltz2 is not in it** although
  it is published (it has a `remote`) and shares `foldAf3` with all three;
- pocket conditioning *does* exist in this repo, in **BioEmu2**
  (`pocket_feature`, `contact_conditioning`, `contact_threshold`) — but
  `pocket_feature` is allocated all-zeros and populated by nothing, and
  `docs/EF2FAST.md` records it sitting in upstream's `_IGNORED_FEATURE_KEYS`,
  i.e. no released checkpoint consumes it. `contact_conditioning` is real and
  populated. BioEmu2 is not reachable from the design page.

So closing it means a constraint path in the featuriser plus boltz2 in
`FOLDERS`. Not verified: whether af3-any-model's boltz2 port exposes the
constraint at all, which decides whether this is plumbing or a port.

## 7. Known gaps nobody has asked about yet

- ~~RFD4's `#validate` and `#beam` are ignored by its generate path~~ — **done**.
  Both honoured, and the picker offered two models (Boltz-2, Protenix-2) that
  are not in `FOLDERS` and validated nothing at all; the list is gated against
  that map now.
- ~~Validation results computed and not displayed~~ — **done** on both paths:
  the validity column carries the sentence, and the status line repeats it.
- **Modality parity with RFD4-Proteina** is a standing goal, measured in
  `docs/DESIGN.md` §6 and not closed.
- **Residue picking in the viewer** — **done for conditions** (2026-10-26), not
  yet for spans. A loaded structure is drawn in the viewer, residues are picked
  with py2Dmol's own click-to-select (`renderer.residueSelection`; the event's
  missing detail did not matter with one viewer on the page), and
  `web/condition-editor.js` sets them as any of the fifteen conditions AtomWorks
  Studio's editor offers - hotspot, coordinates, sequence, secondary structure,
  RASA, H-bond, salt bridge, metal coordination, pi stack, pi-cation,
  hydrophobic, coordinate group, unindexed motif, chain centroid, CATH -
  narrowed to atom names where the condition is per atom. It writes the job's
  `conditions` spec, merged after a preset's and the spans' (the last arm wins),
  so job.json and `localfold-rfd4p --job=` read what the page ran. The viewer
  draws a PDB written from the parsed array, so a click maps to the selection
  language's own chain ids and `res_id`, not an author numbering.
  `test/condition-editor.test.js` holds an editor-built spec to the batch a
  hand-written one builds; `tools/check-rfd4p-batch.py` holds those conditions'
  batch to upstream's; `tools/check-design-page.py` clicks a residue for real
  (and with `--generate` generates from it).
- **Spans picked from the structure** — **done** (2026-10-26). The fold page's
  own sequence strip (py2Dmol's `window.SEQ` panel) is reused unchanged: the
  design page points its callbacks at its viewer while the loaded structure is
  drawn (and at nothing over a design, whose positions are not the input's), so
  a drag along the sequence sets the same `residueSelection` a click does.
  "Add as spans" appends the selection's contiguous runs as `From structure`
  spans; the condition editor reads the same selection. Select all / Unselect /
  Invert are wired through the renderer's selection API, since py2Dmol's own
  app (which wires them on index.html) does not run here.
  `tools/check-design-page.py` drags along the strip for real.
- **AtomWorks Studio's generation options** — **done** (2026-10-26): Thinking,
  Backbone K, Latents K and Partial % on the page, recorded in job.json.
- **Laid out for a newcomer** — **done** (2026-10-27). Five steps on the left in
  the order a first visit decides them (what to design, the starting structure,
  what gets built, constraints, generate), the viewer, sequence strip and results
  on the right; one column on a phone. Eight goal cards replace the method picker
  as the first question (each loads its example, New protein is a blank 100-mer);
  the method is an Advanced setting and the other method's controls are hidden
  rather than greyed with a sentence; notes appear only when something needs
  doing. The job is drawn as a chain diagram (a block per piece, click to edit),
  pieces are named for what is done to them ("Design new", "Keep from
  structure"), kept pieces are painted on the structure in their chain's colour,
  and the constraints step answers "what can I do with this selection" with quick
  actions and chips. 🔴 It found a wrong note: an example that runs from its
  structure's annotations IGNORES edits to its pieces, where the page said edits
  "switch to the contig"; it now says the edits are not used, when one is made.
  `tools/check-design-page.py` holds the new defaults, the goal cards, the
  diagram (watched failing with the block click unwired) and the quick actions.
