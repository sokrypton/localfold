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
- **The job drawn before it runs, and the picture made the page** — **done**
  (2026-10-27). The viewer was empty until a structure was loaded and then drew
  only that structure. It now draws the JOB from arrival (`web/job-sketch.js`):
  the structure's real atoms, and every piece still to be designed as a C-alpha
  trace of its length (a range at its midpoint), pale and translucent - a
  circular arc of even 3.8 A steps between the two kept pieces a linker joins, a
  protein-sized wire globe on a piece with one neighbour or none, set beside the
  target's hotspots for a binder and round the ligand for a pocket. Every edit in
  step 3 redraws it, keeping the camera and the selection. A click on a sketched
  piece opens its row; a sketch is no residue a constraint can name, so picks and
  Select all skip it. A legend above the viewer says what each colour is and
  that the faded part is a sketch, not a prediction. A design on screen stays
  until "Show the job" is pressed. The stage is as tall as the window and sticks
  beside the steps; Generate moved into its bar, in view on arrival (it was below
  the fold at the end of step 5); on a phone the picture comes right after step
  1. `tools/check-design-page.py` holds the arrival sketch, the motif's three
  sketched pieces, the sketch click (watched failing with the handler off) and
  Show the job after a generation; `test/job-sketch.test.js` holds the geometry.
  🔴 Found on the way: the design page's box was `#viewer-container`, the id
  py2Dmol's website app starts itself on, so every load threw inside py2Dmol and
  wrote a 600 px size into this page's viewer - renamed `#design-viewer`. And
  py2Dmol's full-screen button drew an empty box on Linux (U+26F6 is in few
  fonts): an inline SVG now, on sokrypton/py2Dmol#32's branch and re-vendored.
  **Closed since** (sokrypton/py2Dmol#33, vendored): the sketch is see-through on
  py2Dmol's 2D painter too - which is the painter EVERY browser gets here, not only
  one without WebGL2 as this entry said: the full bundle carries both painters and
  takes the GPU one only when a page asks (`rendering.gpu`), and this page never
  did. The 2D painter now draws a faded selection as two whole frames crossfaded,
  exact to 8-bit rounding (py2Dmol's `tests/opacity_2d.py`). And the strip leaves
  the sketch out (`listsChain`), so it no longer reads as a row of G.

- **The page reorganised around the method, an example, and segments** — **done**
  (2026-10-27). Protein Hunter is the default method, and the method and its
  models ("Folds with", "Check with") are step 1. The examples are one grouped
  dropdown (step 2), named as a reader asks for them with a one-line note in the
  page's words; choosing one fills in the segments, loads and draws its
  structure, and sets its sampler. It keeps the method it was chosen under, and
  one the method cannot run is listed disabled with its reason. The eight goal
  cards and the second "More examples" picker are gone. Each segment is a card
  carrying every setting it has (step 3):
  - a designed segment's length, and its secondary structure (which labels itself)
    and polymer under RFD4-Proteina;
  - a kept segment's chain, residues and **hotspots**, with "Use selection";
  - a sequence's polymer and letters;
  - a ligand's code.

  Modified residues and a label sit under "More". "Add a segment" names what is
  added (a DNA sequence, a ligand, a chain break). Hotspots live on the segment for
  both methods: an example's file hotspots are written onto its target's card
  (`web/example-spans.js`), and Mark as hotspot writes there too. The separate
  hotspot line is gone. Step 4 is the selection, shown while a structure is loaded:
  Keep in the design and Mark as hotspot under either method, and RFD4-Proteina's
  constraints only under it. The sequence strip stays through a run, showing the
  design's sequence and following the play bar.

  **The binder and the DNA binder now run under Protein Hunter**: they load as
  their files describe them (`test/design-preset-methods.test.js` measures it
  through the same module). Partial diffusion and inverse folding of 1PRW are
  withheld from Hunter: their segments are the motif example's, and `--presets`
  showed each running as that same hunt. That leaves Hunter 6 of 10. A
  kept DNA strand folds as DNA (`polymerOf`); it folded as a protein of unknowns
  before.

  🔴 Found on the way:
  - `hotspotsOf` read `60-64` as 60 and 64, so a hunt was aimed at the two ends of
    the site;
  - the SAM example's own file named its ligand `S;M`; the file now says `SAM`
    (see the third pass).

  **Second round, same day** (from a reader's report: the sequence "not changing"
  and its box "disappearing" during a design, hotspots hard to find, and the cards
  hard to read):
  - **The strip was below the fold.** The viewer was as tall as the window, so the
    strip under it was off screen on a laptop through every run. The stage now
    shortens the viewer while the strip is up (`has-sequence`).
  - **Pressing Design blanked the picture.** It wiped the viewer and the strip for
    the whole model load and first fold. The job now stays drawn until the run's
    first frame (`beginRunPicture`).
  - **The strip trailed the picture.** It waited a frame to rebuild, and stayed
    empty through a hunt's first cycle because building the viewer raises no
    frame change. It now follows every frame as it lands.
  - **Cards are a two-column form,** label beside control, one setting a line, with
    no "More". The hotspots line is marked in the hotspot colour and says what a
    hotspot is. A protein-sequence target takes hotspots too, under Protein Hunter.
  - **Seed** is a main Run setting with a die beside it.
  - 🔴 **Protein Hunter was checking hotspots in the wrong numbering.** It was given
    the structure's chain letter and residue numbers; it checks its own fold, whose
    chains are lettered by job order and numbered from 1. The two agree only for
    the PD-L1 example: a target kept as B5-50 was asked for the fold's B14 when the
    reader meant B10. `hotspotsOf` now numbers them as the fold does, per run
    (`test/contig-ui.test.js`).
  - Not a bug: RFD4-Proteina at 20 steps designs a 100-mer as poly-alanine (seed 0),
    and the strip shows it. At the page's 60 it does not.

  `tools/check-design-page.py` holds the new order, labels on every control, the
  binder under both methods, Add a segment, Use selection, and the strip during
  and after a real hunt. With `--presets` it runs every example as a generation
  and every Hunter one as a one-cycle hunt: all 18 finished (2026-10-27).

### Third pass over the options (2026-10-27) - done

- Steps number themselves (a CSS counter), so a hidden Selection step no longer leaves "3, 5".
- Run: sentence-case labels and the page's own field style; Advanced is its own titled box under the
  main settings, holding the method's sampler knobs and Protein Hunter's designer/fold notes.
- Adding a protein/DNA/RNA sequence or a ligand adds the chain break before it.
- More constraints: each control labelled (Constraint, Value, Atoms); Hotspot is no longer offered
  there - it is a segment's, set on its card or by "Mark as hotspot".
- Shorter hotspot help; a chain break's card says "the next segment starts a new chain"; a
  ligand-only chain is described as "ligand SAM", not "chain B 1-1".
- Verified: check-design-page default and --generate ok, npm test 1584 pass, no overflow at 390 px.
- `web/examples/sm_sam.cif.zst` renamed its ligand `S;M` -> `SAM` in every table (atom_site,
  chem_comp, bonds), so the strip and card read SAM and the two display workarounds are gone. 🔴 The
  featurised batch is IDENTICAL from both files - all 80 fields, seeds 0 and 7, a comparison that
  catches one atom moved 0.7 A - because no RFD4 table knows SAM and an annotated example fetches
  no dictionary entry. A RAW structure would: `componentsNeedingCcd` fetches RCSB's entry for any
  CCD-shaped name, which is a reason upstream's `S;M` may have been deliberate. Not checked against
  the upstream oracle: `oracle-dumps/` holds no sm_sam dump on this box.

### Every example, checked (2026-10-27) - done

All eleven entries under both methods, loaded and screenshot, then `--presets` run: RFD4 on all ten,
a 1-cycle hunt on the six Protein Hunter takes - 16 of 16 finished. Fixed on the way:
- 🔴 an example sharing its file with the one before (the three 1PRW ones) said "edits here are not
  used" with no edit: its loading was compared against the previous example's segments;
- the unconditional example's file holds only the design's placeholder, and the page offered a
  Selection step and an empty strip for it - it now says "only a placeholder, nothing to keep";
- the legend's "not used" appeared whenever a structure was loaded, now only when part of it is
  left out of the job;
- a "Keep" card's chain tag reads "→ chain B" (the chain it becomes), since "chain B" beside a field
  saying chain A of the structure (the 5O45 binder) read as a contradiction;
- under Protein Hunter, an example it cannot run now greys out Design with the reason, as the note
  already said.
Open: at the check's speed settings (20 steps, 1 cycle) uncond_short and the 5O45 binder hunts end
"no cycle was eligible" (the alanine ceiling) - known from the earlier sweep, not a page fault.


### Chain colours during a generation (2026-10-27) - done

Reported: chain colours mismatched during generation. 🔴 RFD4-Proteina's frames were never coloured by
the page - py2Dmol's palette goes by the output's own letters - and RFD4 letters chains its own way:
a kept chain keeps the structure's letter, new chains take the first free one
(`createChainIdGenerator(claimedChainIds)`). So the 5O45 binder, design first and target from chain A,
came out target A / design B, and was drawn with each in the other's colour; the cards and diagram
said A / B the other way round. `chainLetters(spans, method, claimed)` (web/contig-ui.js) is now the
one rule - Hunter's job order, or RFD4's - for the cards' "→ chain X", the diagram, the results legend
and the colour of every RFD4 frame; a chain RFD4 writes that the rule did not predict is an error.
Verified: all ten RFD4 examples ran through `--presets` with the check live (no error, results
unchanged), and 5O45 / PD-L1 / DNA screenshot before and after a run in the same colours.
