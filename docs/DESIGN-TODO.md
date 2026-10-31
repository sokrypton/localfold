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
  pieces are named for what they are ("Designed", "Structure", "Sequence";
  2026-10-28 - they were "Design new", "Keep from structure", "Known sequence"),
  kept pieces are painted on the structure in their chain's colour,
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


### Refold, as its own step (2026-10-27) - done

Asked for: a validate button that predicts and shows the (mis)alignment, separate from the run so the
reader decides, with its own model picker. 🔴 The check used to run after EVERY run with a model picked
in step 1 ("Check with"), cost every run a fold, and showed only a line of text - the refolded
structure was computed and never drawn. Now:
- step 6, **Refold**: a finished design (an RFD4 candidate, or a hunt's scored cycle - its best by
  default), a model of its own (AF3, Boltz-2, Protenix-2, OpenBind-0, OpenDDE), a Refold button, the
  verdict, and the refold as a PDB in the design's frame;
- the viewer draws the design in the run's colours with the refold over it, grey where it agrees and
  red where it moved (1 A to 4 A and up), fitted the way the run is - on what was kept, the designed
  chains held out - so a binder that lands elsewhere shows as moved; "Show the run" gives the
  trajectory back;
- the refold reads every chain off the design (protein, DNA/RNA, ligands - `cpu/design/refold.js`),
  pairs chains by order not letter (RFD4 letters the 5O45 binder design B / target A), and refuses a
  residue it cannot name (a modified one) rather than refolding an X;
- an RFD4 beam no longer ranks or marks a best candidate: nothing scores one until it is refolded,
  and the verdict lands in its row.
Found on the way: "Folds with" offered Boltz-2 and Protenix-2 and `hunt` threw on both ("no folder
boltz2") - they were missing from `FOLDERS`; `test/design-validate-options.test.js` now holds both
pickers to it. Verified: `--generate` refolds candidate 2 with AF3 (two chains drawn, verdict in the
row, Show the run back to 31 frames); 5O45 at 60 steps and a 2-cycle hunt refolded and screenshot.
A single-sequence refold misfolded the natural target - 5O45's PD-L1 came back 8.3 A from its
crystal, so the binder's check was dominated by the target. **Done (2026-10-27): the refold
templates every kept protein chain off the design itself** (`keptTemplates`, one slot a chain, never
spanning, so where the binder sits is still the model's answer; more than four kept chains is
refused; the verdict names the templated chains, and a slot that did not arrive throws). 5O45 at 60
steps with AF3: pLDDT 49.1 / ipTM 0.099 -> **88.9 / 0.826**, binder RMSD 1.01 -> 0.72 A, and the
target drawn grey on its crystal. **And the refold is its own viewer object** (2026-10-27, the
user's ask): the design and the refold are two py2Dmol objects shown together, the refold in the
design's letters, where it had been one structure with the refold under spare letters C, D. Its
shading is addressed by chain and residue NUMBER, because py2Dmol does not keep a file's chain
order - counted by position, the 5O45 binder's red landed on the target.

### Protein Hunter designs a DNA or RNA chain (2026-10-28) - done, the score is the open part

A Designed segment "Made of" DNA or RNA now stays nucleic under Hunter (it was forced to protein),
starts as mostly `N` - AF3's base-less nucleotide, as `X` is its unknown amino acid - and is
designed by NA-MPNN alone (`chooseDesigner({designs})`; an explicit pick of another family is
refused). The sampler opens only NA-MPNN's four DNA tokens (`designBias(..., {designs: "nucleic"})`;
shared tokens store an RNA base as the DNA one) and the bridge reads them back in AF3's letters by
residue name (`baseLetter`: DT is U in an RNA chain). The alanine ceiling is a protein rule and does
not apply - the results table shows the strand's GC share instead. Two fixes it needed: NA-MPNN's
parser did not know AF3's `DN`/`N`, so a strand of unknowns came back one letter long (hand-applied
to the mirror, which had already diverged, and sokrypton/mpnn#4, merged upstream as eaaa8ba - see
cpu/design/mpnn/SOURCE.md); and the contact veto parsed protein only, so a nucleic chain was
invisible to it - a binder against DNA alone had nothing to touch and was never checked. It reads
DNA and RNA now, each pair at its calibrated cutoff (10 A nucleic-protein, 9 nucleic-nucleic).

Measured, AF3, one run of 5 cycles, single sequence:

| job | converged strand | against the native | objective |
|---|---|---|---|
| `dna_7m5w`, strand A kept, strand B (10) designed | `GATTCATAAC` (cycle 3) | contains A's reverse complement exactly, 9/9, with a 5' G overhang (native: a 3' C) | ipTM 0.003 -> 0.010; pLDDT 63 (all N) -> 87, native 89 |
| full 7M5W, protein A and strand B kept, strand C (18) designed | `GAGCTTTTTCATTTCATA` | the complement shifted by 2 - it pairs with B out of register | ipTM **fell**, 0.499 at cycle 0 -> 0.358 |

🔴 **THE OBJECTIVE IS THE OPEN QUESTION, AS EXPECTED.** On the bare duplex ipTM is ~0.01 for every
strand (a 19-nt complex is too short for a TM-based score to mean much), so it orders cycles but its
value says nothing; pLDDT tracks the pairing. With the protein, single-sequence AF3 does not fold
capicua's HMG domain (pLDDT 42 for the native complex) and scores the base-less starting strand
best (ipTM 0.353 against the native strand's 0.089) - so the loop's own selection works against it.
A nucleic design wants an objective that sees pairing (pLDDT, or base-pair geometry as a veto) and a
target that folds (an MSA or a template), neither of which is wired.

### Protein Hunter with history: Stay within and Warm start (2026-10-28) - done, measured on one target

Nothing carried a cycle's structure to the next but the backbone MPNN read, so a run drifted. Two
remedies, both off by default (the reference's method):

- **Stay within (A)** (`accept` in webgpu/design/hunter-loop.js): up to Sequences designs a cycle,
  one at a time, keeping the first whose designed chain moved less than this from the previous
  cycle (C-alpha RMSD on itself), else the closest; from cycle 2.
- **Warm start (A)** (`warm` in web/hunt.js): each cycle's AF3 diffusion starts from the previous
  cycle's structure this far down the noise schedule and runs only the steps left - partial
  diffusion (`start` in webgpu/af3/diffusion/diffusion-sampler.js). The previous PDB is laid out for
  the new sequence by `startFromPdb` (shared/af3/structure/warm-start.js): an unchanged residue
  copies every atom, a mutated one its backbone and CB, every other atom on its C-alpha. On 6MRR a
  four-mutation variant folded 1.5 A from the previous fold warm at 10 A against 2.3-2.5 cold.

The results table's **Moved** column shows each cycle's drift either way.

5O45 binder, AF3, 6 cycles, three seeds, the best design refolded by Protenix-2 (chain B templated):

| seed | arm | moved, cycles 3-6 | best objective | refold ipTM, RMSD |
|---|---|---|---|---|
| 0 | reference | 2.3 / 1.0 / 2.1 / 2.7 | 0.134 | 0.285, 1.32 |
| 0 | warm 10 | 2.0 / 1.8 / 1.6 / 1.7 | 0.155 | 0.307, 1.73 |
| 0 | stay 2 | 1.4 / 0.8 / 0.7 / 0.6 | 0.146 | 0.342, 1.14 |
| 0 | both | = warm 10 (every first try was within) | 0.155 | 0.307, 1.73 |
| 1 | reference | 3.5 / 5.1 / 7.2 / 3.7 | 0.192 | 0.359, 2.05 |
| 1 | warm 10 | 2.9 / 2.5 / 1.0 / 1.5 | 0.223 | 0.610, 1.84 |
| 1 | stay 2 | none within, 4.4-7.7 | 0.355 | 0.599, 8.77 |
| 1 | both | 2.5 / 2.3 / 1.5 / 1.6 | 0.210 | 0.308, 7.32 |
| 2 | reference | 6.3 / 8.4 / 13.2 / 7.8 | 0.351 | 0.603, 6.80 |
| 2 | warm 10 | 3.7 / 2.5 / 2.0 / 1.6 | 0.545 | 0.708, 6.41 |
| 2 | stay 2 | none within, 5.3-17 | 0.283 | 0.325, 12.6 |
| 2 | both | 3.3 / 1.1 / 1.8 / 1.9 | 0.527 | 0.683, 9.01 |

🔴 **The warm start is what stabilises a run**: it settles to ~1.5-2 A on every seed where the
reference keeps jumping 3-13 A, and beats the reference's best objective and refold ipTM on all
three. **Stay within alone is a filter, not a force**: where no candidate comes near (seeds 1 and
2) every cycle falls back to "closest" and its best design fails the refold badly. Stabler is not
the same as right - seed 2's refold disagrees in every arm. One target, three seeds; a
monomer and a second binder are the next measurements.

**Warm start is ON by default at 10 A** (2026-10-28, the user's call on these numbers); Stay within
stays off - alone it failed two seeds of three. Blank either for the reference's method. OpenDDE takes it too:
the start is gathered onto its structural tokens (`residueToStructural`, the inverse of the table
that scatters its answer back). 5O45 with OpenDDE, 5 cycles, seed 0: moved 9.6-11.6 A a cycle cold,
4.1-5.6 warm; best objective 0.516 -> 0.549; the AF3 refold disagrees with both (12-14 A).

**Measured again in the target's frame (2026-10-28), and the first table above is about SHAPE.**
The Moved column and Stay within superposed the binder on ITSELF, so a binder that slid across its
target read 1-2 A while the viewer - fitted on the target - showed it wandering (the user's report).
`designedDrift` (cpu/design/align.js) fits on everything but the designed chain and takes the
binder's RMSD unrefitted; a monomer is still measured on itself. Same job, seeds 0-2, moved per
cycle 1-6 (A), best objective, Protenix-2 refold ipTM / RMSD:

| seed | reference | warm 10 | warm 5 | warm 3 |
|---|---|---|---|---|
| 0 | 36/42/34/2.8/29/21 · 0.134 · 0.285/1.32 | 36/3.0/3.7/2.5/2.7/2.6 · 0.155 · 0.307/1.73 | 36/2.1/2.0/1.7/1.1/1.3 · 0.165 · 0.201/11.2 | 36/1.2/1.1/1.2/1.0/1.3 · 0.229 · 0.265/12.9 |
| 1 | 6.2/5.6/5.0/10.7/9.8/5.1 · 0.192 · 0.359/2.05 | 6.2/4.2/4.6/3.2/2.2/3.0 · 0.223 · 0.610/1.84 | 6.2/2.4/2.0/1.6/1.3/1.7 · 0.225 · 0.505/11.8 | 6.2/1.3/1.2/0.9/1.0/1.1 · 0.191 · 0.380/15.0 |
| 2 | 20/27/12/9.9/20/23 · 0.351 · 0.603/6.80 | 20/3.9/4.4/3.3/2.6/2.1 · 0.545 · 0.708/6.41 | 20/2.1/1.8/1.5/1.1/1.2 · 0.479 · 0.695/6.51 | 20/1.4/1.2/1.1/1.0/1.2 · 0.349 · 0.578/12.9 |

🔴 The reference's binder JUMPS SITES (10-42 A a cycle); warm 10 holds it at 2-4.5. The big first
number is cycle 1, which folds cold by design (it leaves cycle 0's mostly-X guess). 🔴 **HOLDING
TIGHTER OVERFITS**: at 5 and 3 A the run barely moves and the independent refold rejects most of its
designs (11-15 A) - the sequence is tuned to a structure AF3 was held at, not one it finds. 10 A is
the only level whose refold agrees on seeds 0 and 1, so it stays the default. Warm 5 + Stay within
3 = warm 5 on every seed (every first try was within).

**A second binder, a monomer, and warming cycle 1 too (2026-10-28).** Same harness (the page, 6
cycles, seeds 0-2, best design refolded by Protenix-2); "from 1" starts cycle 1 from cycle 0's fold
of the mostly-X guess (`warm.from = 1`, injected for the measurement - not a page setting). Moved
per cycle 1-6 (A) · best objective · refold ipTM / RMSD (the monomer has no ipTM):

| target | seed | reference | warm 10 | warm 10 from 1 |
|---|---|---|---|---|
| 5O45 | 0 | 36/42/34/2.8/29/21 · 0.134 · 0.285/1.32 | 36/3.0/3.7/2.5/2.7/2.6 · 0.155 · 0.307/1.73 | 3.9/4.6/4.5/3.8/3.4/2.8 · 0.181 · 0.178/15.7 |
| 5O45 | 1 | 6.2/5.6/5.0/10.7/9.8/5.1 · 0.192 · 0.359/2.05 | 6.2/4.2/4.6/3.2/2.2/3.0 · 0.223 · 0.610/1.84 | 5.6/3.0/3.7/2.8/2.2/3.6 · 0.237 · 0.532/3.65 |
| 5O45 | 2 | 20/27/12/9.9/20/23 · 0.351 · 0.603/6.80 | 20/3.9/4.4/3.3/2.6/2.1 · 0.545 · 0.708/6.41 | 4.6/4.3/3.9/3.5/4.4/3.1 · 0.362 · 0.524/16.6 |
| PD-L1 | 0 | 25/9.2/22/5.5/7.2/23 · 0.272 · 0.528/8.66 | 25/4.9/2.7/4.5/2.8/2.9 · 0.198 · 0.532/11.0 | 6.3/5.7/5.6/3.7/3.6/2.1 · 0.186 · 0.290/13.9 |
| PD-L1 | 1 | 22/25/32/41/50/31 · 0.142 · 0.174/19.1 | 22/5.6/4.5/4.2/4.4/3.7 · 0.156 · 0.190/18.9 | 6.6/4.7/5.0/3.6/3.5/4.0 · 0.192 · 0.275/12.1 |
| PD-L1 | 2 | 18/36/24/42/39/47 · 0.083 · 0.199/17.3 | 18/5.3/5.1/3.3/2.9/4.7 · 0.113 · 0.163/11.5 | 5.6/5.8/4.1/3.5/5.6/4.6 · 0.242 · 0.158/12.1 |
| monomer 100 | 0 | 9.5/4.7/2.9/0.6/0.8/0.8 · 0.870 · 1.44 | 9.5/2.0/1.2/1.8/1.6/1.3 · 0.830 · 1.22 | 3.8/2.6/2.2/2.4/1.8/1.9 · 0.597 · 9.69 |
| monomer 100 | 1 | 10.7/1.1/1.2/0.6/0.5/0.4 · 0.835 · 0.49 | 10.7/1.4/1.0/1.3/1.0/1.0 · 0.799 · 1.03 | 3.3/3.1/3.0/1.7/2.0/0.9 · 0.757 · 0.82 |
| monomer 100 | 2 | 12.4/3.4/8.0/3.4/1.3/0.9 · 0.914 · 1.85 | 12.4/1.7/1.5/0.9/1.0/0.7 · 0.791 · 0.93 | 2.4/1.6/2.2/1.7/0.9/0.5 · 0.809 · 0.49 |

🔴 **WARMING CYCLE 1 IS WORSE, so it stays cold**: it fails the refold on 5O45 seeds 0 and 2 (15.7,
16.6 A) and the monomer's seed 0 (9.7) - cycle 0's fold of an all-X chain is a poor place to be held
near. 🔴 **PD-L1 HOPS SITES TOO (18-50 A a cycle) AND WARM 10 HOLDS IT AT 3-6, BUT NOTHING PASSES
ITS REFOLD** (8.7-19 A in every arm, ipTM <= 0.53): six cycles is too few for that target, so it says
the drift fix is general and nothing about design quality. **A monomer converges by itself** (under
1 A by cycle 4 cold); warm 10 costs it a little objective (0.87/0.84/0.91 -> 0.83/0.80/0.79) and its
refolds all pass (0.9-1.2 A against cold's 0.5-1.9). Warm 10 from cycle 2 stays the default.

**More diffusion steps (2026-10-28) make it worse, not better.** The same three targets and seeds at
100 and 200 steps against 25 (the page's Diffusion steps, Advanced), the reference and warm 10. Refold
ipTM / RMSD, seeds 0-2:

| target | steps | reference | warm 10 |
|---|---|---|---|
| 5O45 | 25 | 0.285/1.32 · 0.359/2.05 · 0.603/6.80 | 0.307/1.73 · 0.610/1.84 · 0.708/6.41 |
| 5O45 | 100 | 0.582/12.6 · 0.172/14.4 · 0.421/18.4 | 0.540/19.1 · 0.282/11.3 · 0.650/17.4 |
| 5O45 | 200 | 0.212/6.82 · 0.388/6.91 · 0.172/2.41 | 0.427/7.10 · 0.141/13.9 · 0.264/1.08 |
| PD-L1 | 100 | 0.680/1.36 · none · none | 0.719/2.83 · none · none |
| PD-L1 | 200 | none · 0.473/11.5 · 0.588/2.56 | none · none · 0.168/13.0 |
| monomer | 100 | 2.17 · 0.85 · 0.93 | 0.91 · 8.14 · 1.59 |
| monomer | 200 | 0.37 · 0.41 · 0.83 | 1.32 · 2.74 · 0.46 |

"none": no cycle was eligible - every design missed all 3 of PD-L1's hotspots, 7 runs of 12 here
against 0 of 9 at 25 steps. 🔴 **THE BINDER MOVES MORE WITH MORE STEPS**: warm 10 held 5O45 at
2-4.5 A a cycle at 25 and lets it move 6-18 at 200 (seeds 1, 2), the reference 10-54. The likely
cause, not yet tested: the sampler's churn (gamma 0.8 above sigma 1,
webgpu/af3/diffusion/diffusion-sampler.js) re-noises EVERY step to 1.8x its level, so a warm walk from 10 A takes more fresh noise
the more steps it is cut into. The monomer converges either way. 25 steps stays the default.

**PD-L1 again with no hotspots required (2026-10-28)**, so every run has a best design to refold
(`window.__designHotspots = {}`; an empty set used to be read as "touch none of the target" and
refused every cycle - fixed in cpu/design/contact.js). Refold ipTM / RMSD, seeds 0-2:

| steps | reference | warm 10 |
|---|---|---|
| 25 | 0.528/8.66 · 0.444/37.4 · 0.182/9.54 | 0.532/11.0 · 0.190/18.9 · 0.250/3.99 |
| 100 | **0.680/1.36** · 0.302/5.82 · **0.400/0.93** | 0.719/2.83 · 0.467/4.24 · 0.182/9.17 |
| 200 | **0.710/0.59** · 0.473/11.5 · 0.588/2.56 | **0.885/0.91** · drifted off (touches nothing) · 0.228/5.97 |

🔴 **ON PD-L1 MORE STEPS HELPS, THE OPPOSITE OF 5O45**: 0 of 6 refolds pass at 25 steps, 2 of 6 at
100 and 2 of 6 at 200 (plus a 2.56), and the best design measured anywhere here is warm 10 at 200
(ipTM 0.885, 0.91 A). Warm start does not help this target's refold - reference 4 passes, warm 1. Three
seeds a cell, and the two binders disagree, so no default moves on this.

**Hotspots as boltz2's pocket constraint (2026-10-28)**, as ProteinHunter v1 (`{"pocket": ...}` in the
Boltz job) and v2 (ph2/hotspots.py; it reports ipTM 0.276 -> 0.610 on PD-L1) both do by default:
`hotspotPocket` in cpu/design/contact.js, handed by web/hunt.js to any folder in `CONSTRAINT_FAMILIES`
(boltz2 here; chai-1 is not a Hunter folder), at boltz's 6 A. The contact check still runs on top, and
the other folders of a rotation get the check alone. PD-L1, its 3 hotspots required, Boltz-2 folding,
5 cycles, seeds 0-2; the off arm strips the constraint from a served copy of hunt.js. Best boltz2
objective, then the Protenix-2 refold's ipTM / RMSD:

| steps | pocket | no pocket |
|---|---|---|
| 25 | 0.771: 0.531/5.68 · 0.584: 0.523/**0.96** · 0.181: 0.546/9.82 | 0.649: 0.458/9.10 · 0.393: 0.741/**0.59** · 0.306: 0.559/11.8 |
| 100 | 0.731: 0.551/6.39 · 0.271: 0.644/10.3 · 0.615: 0.287/14.2 | 0.790: 0.433/9.74 · 0.384: 0.726/2.18 · 0.855: 0.380/**0.63** |

🔴 **NOT THE WIN v2 REPORTS, ON THIS MEASUREMENT**: refolds pass 1 of 6 with the pocket and 2 of 6
(plus a 2.18) without, and the boltz2 objective is higher with it at 25 steps (2 of 3) and lower at 100
(3 of 3). The objective is not comparable across the arms anyway - with the pocket it is boltz2's ipTM
after being TOLD where the binder goes. Three seeds a cell; it stays on, as upstream has it, until a
larger set says otherwise.

**...and the hotspots are a PRIOR there, not a filter (2026-10-28)**: a cycle a pocket-reading folder
folded is no longer rejected for missing a hotspot, only for touching nothing; other folders keep the
filter. Re-run on the same six cells, every best and every refold came out IDENTICAL to the pocket arm
above - so with the pocket no cycle had been missing its hotspots, and every rejection there was the
alanine ceiling. The pocket does put the binder on the epitope; what it does not yet do is refold better.

**Steering toward the hotspots, for any model (2026-10-28)**: cpu/af3/diffusion/steering.js, ProteinHunter2's
ph2/steering.py reimplemented - each denoised prediction takes one descent step on relu(softmin over binder
token centroids of the distance to each hotspot - 8 A)^2, scaled by tHat/sigmaMax - and "Steer to hotspots"
under Advanced (off by default). A steered cycle treats the hotspots as a prior: rejected only for touching
nothing. OpenDDE (structural tokens) refuses it. PD-L1 with AF3, 3 hotspots, 5 cycles, seeds 0-2; cycles
1-5 touching a hotspot (CA within 8 A), then the Protenix-2 refold of the best, ipTM / RMSD:

| steps | filter (today) | steer 1 | steer 3 |
|---|---|---|---|
| 25 | 5/5 0.53/11.0 · 5/5 0.18/15.7 · 3/5 0.16/11.5 | 5/5 0.19/9.5 · 5/5 0.32/19.5 · 2/5 0.13/8.5 | 5/5 0.53/11.4 · 5/5 0.19/16.1 · 1/5 0.14/8.9 |
| 100 | 5/5 **0.72/2.83** · 0/5 none · 0/5 none | 4/5 0.35/7.8 · 5/5 0.70/13.8 · 0/5 0.37/0.72 | 5/5 0.47/10.7 · 5/5 **0.63/3.13** · 0/5 0.38/0.90 |

🔴 **IT RESCUES THE RUNS THE FILTER LOSES, AND NOT YET THE REFOLD.** At 100 steps two of three filter runs
ended with no eligible cycle; steering at 3 put seed 1 on the epitope every cycle (nearest 3.9-4.7 A
against 10.4-15.6) and moved seed 2 from 12.7-17.8 A to 8.9-10.1 - closer, not touching. At 25 steps the
filter already hits and steering changes little. Refolds pass nowhere on the epitope; the two that pass
(0.72, 0.90, seed 2 at 100) are binders that never reached it, eligible only because a steered cycle is
not filtered. Weight 1 is v2's and too weak here: the term is zero inside 8 A between centroids, so it
stops pulling at about the filter's own edge. Three seeds a cell; off by default.

**Steering, six seeds (2026-10-29).** The three-seed table above flattered it. PD-L1, AF3, 100 steps,
5 cycles, seeds 0-5; each cell is cycles 1-5 touching a hotspot, then the Protenix-2 refold ipTM/RMSD:

| arm | s0 | s1 | s2 | s3 | s4 | s5 | cycles on |
|---|---|---|---|---|---|---|---|
| filter | 5 0.72/2.8 | 0 none | 0 none | 5 0.22/14.2 | 3 0.14/8.4 | 5 0.60/5.5 | 18/30 |
| steer 3 | 5 0.47/10.7 | 5 0.63/3.1 | 0 0.38/0.9 | 0 0.21/3.3 | 0 0.44/11.1 | 0 0.23/11.2 | 10/30 |
| steer 10 | 5 0.47/9.5 | 5 0.68/12.2 | 0 0.17/14.3 | 2 0.31/10.5 | 0 0.29/18.2 | 0 0.12/8.2 | 12/30 |
| steer 3, cutoff 6 | 5 0.63/7.1 | 0 0.20/12.9 | 0 0.41/8.3 | 2 0.61/10.7 | 5 0.73/5.5 | 5 0.14/11.9 | 17/30 |

Keeping the hotspot filter beside steering changes no trajectory (the veto picks what may win, not
what the next cycle starts from); it only turns the off-epitope refolds into "none". Steering puts the
binder on the epitope LESS often than the unsteered fold (10-12 of 30 against 18), a stronger pull
does not help, a 6 A cutoff draws even, and no arm refolds under 2 A on the epitope. It stays off by
default; not worth more tuning until a target shows it winning.

**Steering was broken, and fixed (2026-10-29).** The table above measured a bug. The first version
(ProteinHunter2's planned ph2/steering.py: squared violation, one plain gradient step on every
centroid) moved the three or four NEAREST residues of each side alone by twice the violation - a
binder 20 A off had three residues thrown ~20 A out of its chain at weight 1, 70 and 240 A at 3 and
10, with the energy RISING - and the denoiser put them back, so cycle 0 came out within 0.1 A of the
unsteered fold at every weight. It also scaled by tHat/sigmaMax, which is ~6% at the 10 A warm start
every later cycle joins at. The original Protein Hunter never steered at all: its Boltz pocket has no
`force`, so Boltz's contact potential gets no atom pairs. Now (cpu/af3/diffusion/steering.js): a
LINEAR restraint, the binder translated as one body, at most `weight` angstroms a step and never past
the largest violation, on Boltz's 1 / 0.5 / 0 schedule over the steps actually walked. Same six seeds
(cycles 1-5 on a hotspot, refold ipTM/RMSD):

| arm | s0 | s1 | s2 | s3 | s4 | s5 | cycles on |
|---|---|---|---|---|---|---|---|
| filter | 5 0.72/2.8 | 0 none | 0 none | 5 0.22/14.2 | 3 0.14/8.4 | 5 0.59/5.5 | 18/30 |
| steer 1 A | 4 0.20/16.0 | 2 0.26/5.0 | 5 0.24/30.6 | 5 0.31/14.4 | 5 0.15/15.5 | 4 0.47/3.0 | 25/30 |
| steer 3 A | 5 0.25/11.8 | 5 0.48/18.6 | 4 0.70/5.8 | 5 0.71/18.1 | 4 0.24/15.5 | 5 0.21/16.5 | 28/30 |
| steer 10 A | 5 0.56/9.6 | 5 0.16/13.1 | 4 0.21/15.6 | 5 0.22/16.5 | 5 0.41/8.7 | 4 0.38/16.4 | 28/30 |

It does its job now - the binder sits on the epitope in 28 of 30 cycles, every seed, the two the
filter lost included - and the designs do not survive a refold: not one under 2 A, and the best
(3.0 A) is weaker than the filter's 2.8. A steered pose is one AF3 holds only while pulled; the
sequence MPNN writes for it does not carry it. Off by default still.

**Steering, then folding unpulled (2026-10-29): no better.** The last 1 or 2 cycles folded without
the pull, filtered on the hotspots, and only those could win (tried, not kept). Same six seeds,
winner's refold ipTM/RMSD: 3 A releasing 1 - 0.21/16.9, 0.31/18.2, none, 0.73/17.7, 0.18/4.2,
0.22/16.9; 3 A releasing 2 - none, none, none, 0.73/17.7, 0.24/7.7, 0.26/16.6; 10 A releasing 2 -
0.57/6.5, none, 0.40/16.5, 0.18/16.2, 0.41/8.8, 0.13/12.2. Released at 3 A the binder often walks
off (5 of 12 runs leave nothing); where it stays, Protenix-2 does not agree on the pose (ipTM up to
0.73 at 17 A). The limit on this target is the refold agreeing at all - the filter alone passes
nothing either and its best is 2.8 A - not where the binder is aimed.

**Folding with two models at once (2026-10-29): worse, reverted.** Built in `8906518a` and reverted
after: every diffusion step AF3 and Protenix-2 denoise the same noisy structure, the partner fitted
onto the leader and averaged, the step taken toward the mean; each model's confidence head then
scores the one structure. PD-L1, 100 steps, 5 cycles, seeds 0-5, refolded with OpenBind-0 (neither
designer), best design's ipTM/RMSD then cycles 1-5 on a hotspot:

| arm | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| AF3 alone | 0.77/2.2 · 5 | none · 0 | none · 0 | 0.23/18.9 · 5 | 0.09/6.1 · 3 | 0.61/3.1 · 5 |
| AF3+Protenix-2 rotating per cycle | 0.49/2.7 · 5 | 0.11/15.4 · 3 | 0.12/3.9 · 2 | none · 0 | 0.09/7.2 · 1 | none · 0 |
| AF3+Protenix-2 together | 0.44/11.9 · 5 | 0.12/11.2 · 3 | none · 5 | none · 2 | 0.64/12.2 · 2 | 0.34/12.0 · 4 |

Every together design refolds at 11-12 A, two seeds lost every cycle to the 20% alanine ceiling,
and a cycle costs 1.5x.

**...and taking turns instead (2026-10-29): no better, not kept.** The same machinery with the
diffusion's steps going to AF3 and Protenix-2 alternately, one model's answer taken whole a step,
so no mean and no blur. Same six seeds: 0.14/6.4 · 1, 0.20/10.1 · 1, none (every cycle 30-46%
alanine), 0.48/4.0 · 4, 0.36/11.7 · 4, 0.75/13.8 · 5 - better than AF3 alone on seed 3 and worse on
0, 4 and 5. 🔴 AND THE ALANINE IS NOT BLUR: cycles over the 20% ceiling are 3/30 for AF3 alone and
for the rotation, 12/30 averaged and 6/30 alternating - mixing models inside one walk raises it
either way, so "a mean is a smoother backbone" was a guess the alternating arm falsifies.

## RFD4-Proteina's binder benchmark, run through Protein Hunter (2026-10-29, overnight)

**The benchmark** (RFD4-Proteina preprint Fig. 2 and S8; their repo's `configs/metrics`): 24 PPI targets, the
union of Proteina-Complexa's and RFdiffusion3's sets, binder lengths as each target specifies. A design passes when
AF3 refolds it - single sequence, the target templated with its own coordinates, 5 recycles, one sample, a random
seed - at binder-chain pTM >= 0.80, minimum interface PAE <= 1.5 A (`chain_pair_pae_min`, both directions averaged,
min over partners) and target-aligned binder backbone RMSD <= 2.5 A. Their per-target rows are the site's
`benchmarks/data.json`, 200 attempts a target. Harness, configuration and per-attempt results: `tools/ppi-bench/`.

**Ours**: the page's own Protein Hunter (`web/hunt.js`) driven in headless Chrome, AF3 int5 on WebGPU, with:
- the target templated in every design fold (`3084bc37`; the benchmark's evaluator templates it, and folding it
  from its sequence alone designs against a target nobody scores);
- 6 cycles, 100 steps, 3 recycles, warm start from cycle 2 at 10 A; a run stops early at binder pTM >= 0.85 and
  interface PAE <= 1.2 A, and an attempt restarts from a new seed up to 3 runs;
- 🔴 **an internal check**: each run's two most promising cycles (design-fold binder pTM >= 0.8, interface PAE
  <= 2.5 A) are refolded under the benchmark's own settings with a seed of the pipeline's, and the first that
  passes is the design (else the checked one closest to passing). This is AF3 feedback inside the loop: the rules
  allow it (internal folding and selection count as generation cost; BindCraft does the same with AF2), but
  RFD4-Proteina uses none, and the internal check and the evaluator are the same model.

Each attempt submits one design. 214 attempts, 8-10 a target. Evaluated twice: by the page's AF3 (int5, WebGPU) and
by the native CUDA AF3 (cuda/af3, af3-any-model's int8 weights), which share no weights file or kernel:

| mean over the 24 targets | pass rate | 95% CI (bootstrap over attempts) | clustered | targets with a pass |
|---|---|---|---|---|
| **Protein Hunter + internal check, native AF3** | **24.1%** | 19.7-28.7% | 23.2% | 15/24 |
| Protein Hunter + internal check, page AF3 | 22.3% | 18.1-26.8% | 20.9% | 17/24 |
| RFD4-Proteina, high thinking (paper) | 19.3% | | 10.6% | 22/24 |
| BindCraft, single backbone (paper) | 9.9% | | 5.9% | 19/24 |
| Protein-Hunter, original (paper) | 6.9% | | 4.1% | |

Per target (native, ours / RFD4 high): PD1 22/14, IFNAR2 33/0, CD45 0/1.5, Claudin1 40/10, BBF14 0/12.5, CrSAS6
0/14.5, DerF7 0/29, DerF21 60/34.5, BetV1 10/3.5, SpCas9 56/43, CbAgo 0/5, HER2-AAV 0/0.5, BHRF1 89/71, SC2RBD
10/14.5, TrkA 38/33.5, H1 25/21, VEGFA 11/17, IL17A 0/9, TNFa 0/0, IL2Ra 0/14.5, IL7Ra 12.5/27, InsulinR 67/49,
PD-L1 56/35.5, Tie2 50/3.

🔴 **What this does and does not show:**
- The mean pass rate is ahead under both evaluators; under the native one the CI clears RFD4's 19.3%, under the
  page's it does not. With 8-10 attempts a target against their 200, it is a lead, not a settled one.
- **The clustered number is not comparable yet.** Clusters are counted among passing designs, and at 8-10 attempts
  a target almost nothing can share a cluster, where their 200 attempts merge about half. `tools/ppi-bench/cluster.py`
  is a small TM-align on CA traces standing in for Foldseek (TM 0.5, merging on the shorter chain's TM-score,
  the generous side) because Foldseek could not be installed here.
- **Coverage is behind**: 15-17 targets with a pass against their 22, from a twentieth of their attempts; nine
  targets got none (CD45, BBF14, CrSAS6, DerF7, CbAgo, HER2-AAV, IL17A, TNFa, IL2Ra).
- **Throughput is far behind**: ~39 attempts per GPU-hour, evaluation included (five browsers on one A100,
  464 s an attempt each), against their 986 designs per GPU-hour before evaluation.
- **The evaluator is not the paper's AF3 binary**: both of ours are AF3-lineage ports (int5 and int8). Over the
  214 designs they agree on 197 verdicts (42 pass both) and differ on 17 - 7 pass only the page's, 10 only the
  native one.

**What moved it, measured on the way** (same 24 targets, 1-2 attempts each, page evaluator):
- the baseline loop (5 cycles, the target templated, the winner by the benchmark's terms): 4.2%, the original
  Protein-Hunter's level. The binder becomes confident on its own and never docks: design folds sat at interface
  PAE 15-25 A for whole runs, and where a design fold did dock the evaluator agreed.
- restarts, early stop and warm start: 1/10, with many near misses at interface PAE 1.8-2.3 A - the design folds
  (3 recycles, 100 steps) read ~0.5 A lower than the evaluation (5 recycles, 200 steps).
- **the internal check: 18.8-22%** - the step that turned near misses into choices.
- not taken: three sequences a cycle at 50 steps (0/6, twice the time); giving up on a run not docked by cycle 2
  with five restarts (0/10 - docking often comes late).

Open: wire the target template into the design page itself (the hunt takes `templates`; design.html does not yet
pass them); more attempts for a settled comparison and a real clustered rate; the nine targets with no pass.

### Engines, target input and a cross-evaluation (2026-10-29, the day after)

Every fold on the native CUDA port now (`tools/ppi-bench/native-bench.mjs`: the page's `runDesign`, MPNN and contact
check under Node, each fold handed to a resident `cuda/af3/fold --serve`; ~2 s a fold against 10-15 s in the
browser). The evaluation is the native AF3 (int8) with the benchmark's criteria. Mean over the 24 targets, bootstrap
95% CI over attempts:

| run | design engine (target input) | internal check | attempts | pass rate | targets with a pass |
|---|---|---|---|---|---|
| **NC** | AF3 (template) | AF3 | 240 | **37.5%** (33.3-42.1) | 19 |
| NM1 | AF3 (target MSA) | AF3 | 336 | 22.9% (19.3-26.5) | 17 |
| C (browser, overnight) | AF3 int5 (template) | AF3 int5 | 214 | 24.1% (19.5-28.7) | 15 |
| NH | Boltz-2 + hotspot pocket (template), original-style: 7 cycles, 200 steps, best ipTM | none | 240 | 9.2% (6.2-12.5) | 11 |
| NHM | the same with the target MSA - the published Protein Hunter's own input | none | 240 | 6.7% (4.2-9.6) | 8 |
| NG | Boltz-2 + pocket (template) | Boltz-2 | 54 | 4.9% (0-9.0) | 3 |
| NX | Chai-1 + pocket (template) | Chai-1 | 121 | 3.0% (0.7-5.7) | 3 |
| NXH | Chai-1 + pocket (template), original-style: 5 cycles, 200 steps | none | 120 | 2.6% (0-5.3) | 3 |
| *RFD4-Proteina high (paper)* | | | *200 a target* | *19.3%* | *22* |

- **The reproduction holds**: NHM - Boltz-2, the target's MSA, no check - is 6.7%, RFD4-Proteina's 6.9% for the
  original. Templating the target instead lifts the same loop to 9.2%.
- **The template beats the MSA for both engines** (AF3 37.5 against 22.9; Boltz-2 9.2 against 6.7).
- **An internal check helps only with the evaluator's model**: Boltz-2 checking Boltz-2's designs (NG, 4.9%) is below
  not checking at all (NH, 9.2%) - it accepts what Boltz-2 likes and AF3 rejects, and the early stop ends runs on it.
- **AF3 is the engine**, against the preprint's ranking (Boltz-2 first there, AF3 last): with the target templated and
  an AF3 check it is the only engine above RFD4-Proteina; Chai-1 is the weakest here.
- 🔴 **NC IS AN UPPER BOUND**: its design, check and evaluation are the same int8 weights. C - designed with the
  page's int5 copy - is the fairer AF3-evaluated number.
- 🔴 **AND 444 ATTEMPTS WERE LOST TO THE MACHINE FIRST**: three resident servers (AF3, Boltz-2, Chai-1) and the MSA
  folds ran the 40 GB card out of device and pinned host memory, and the failed attempts were scored as failed designs
  for an afternoon. Re-run one server at a time (`phase.sh`); `score.py` skips error rows now.

**Cross-evaluation** - "AF3 designed, AF3 checked, AF3 scored" asked of models that took no part
(`tools/ppi-bench/xeval.py`: the same criteria, target templated, single sequence, 5 recycles, 200 steps; ESMFold2-fast
600M with Synthyra's head takes no template and reports no chain pTM, so it is judged on interface PAE and RMSD alone):

| designs | AF3 | Boltz-2 | Chai-1 | Protenix-2 | ESMFold2 (iPAE+RMSD) |
|---|---|---|---|---|---|
| NC (AF3-designed) | 37.5% | **23.8%** | **26.7%** | **25.1%** | 6.3% |
| C (AF3 int5-designed) | 24.1% | 17.2% | 21.6% | 18.5% | 8.3% |
| NH (Boltz-2-designed) | 9.2% | 19.1% | 16.1% | 11.5% | 3.5% |

- AF3's designs are not AF3 agreeing with itself: three models that did not design them pass 24-27% of NC's,
  and **Boltz-2 passes AF3's designs more often than its own** (23.8 against 19.1).
- Every engine favours its own: NH's Boltz-2 designs pass Boltz-2 at 19.1% and AF3 at 9.2%.
- 🔴 It is not yet a like-for-like win over RFD4-Proteina, whose designs were only ever scored by AF3; re-scoring
  theirs (the paper's archive) the same way is what would make it one.
- ESMFold2 rejects nearly everything because it refolds the natural target from its sequence alone, so its
  target-aligned RMSD tests the target as much as the binder.

**The warm start's own A/B** (W0/W1: C's recipe without and with the 10 A warm start from cycle 2, in the browser -
the native port has no warm start, so NC against C could not answer it; same lengths and evaluation seeds per attempt
through `pair`, attempts 30 and 31 on all 24 targets, page-AF3-evaluated, no errors):

| arm | attempts | pass rate | targets with a pass | median hunt |
|---|---|---|---|---|
| W0, no warm start | 48 | 20.8% (14.6-27.1) | 7 | 493 s |
| W1, warm start 10 A from cycle 2 | 48 | 16.7% (10.4-22.9) | 6 | 447 s |

- **It does not help.** Paired by target and attempt: 5 pass in both arms, 5 only without it, 3 only with it. Seven
  of eight discordant pairs would be needed to call it either way, so it is a null at this size, and the margins lean
  the same way: median W1 - W0 interface PAE +0.65, binder RMSD +0.35 A, binder pTM +0.006.
- It is ~10% faster per attempt, which is not enough to matter.
- So the winning recipe's lift over the original Protein Hunter is the template, AF3 as engine, restarts and the
  internal check - not the warm start. It stays in C as recorded; dropping it would not cost anything measurable.

### RFD4-Proteina's small-molecule- and DNA-binder benchmarks (2026-10-30)

The paper's other two binder tasks, on the native CUDA AF3 port: `tools/ppi-bench/prep-tasks.py` turns their repo's
targets into AF3 jobs (`tasks.json`), `task-bench.mjs` is native-bench.mjs's loop over them, `task-score.py` scores
against the paper's per-target rates (`rfd4-paper-data.json`).

- **Small molecules**: the 77 DISCO Figure 2c targets (`disco_figure_2c.yaml`). Each ligand chain is written as SMILES
  from the file's own atoms, bonds and formal charges (RDKit); a metal complex as its CCD code (HEM, B12, SF4, FES,
  FE2+AKG, the ions), since RDKit writes coordination as dative bonds. Unfixed ligand, as theirs. Pass (S8.2):
  protein-ligand ipTM >= 0.80, min interface PAE <= 1.5, protein backbone RMSD <= 1.5 A, ligand RMSD <= 5 A after
  protein-backbone alignment, design against refold. MPNN is LigandMPNN (it sees the ligand's atoms).
- **DNA**: the three RFD3 targets, sequence context (the paper's "unfixed": the duplex is given as sequence and
  co-folded), binder 120-150. Pass: min interface PAE <= 1.5 and binder backbone RMSD <= 8 A after aligning all DNA
  atoms. MPNN is NA-MPNN.
- The interface terms are the paper's oracle's (`oracle_metric.py`): chain_pair_* averaged both ways, min for PAE and
  mean for ipTM over the binder's partners. The evaluation is the native AF3, 200 steps, 5 recycles, one sample, its
  own seed; the design loop is recipe C's (AF3, 6 cycles, 100 steps, 3 recycles, internal AF3 check), no template.

Small molecules, 2 attempts a target (theirs 200), the same attempts in every arm:

| run | change | pass rate (95% CI) | targets with a pass |
|---|---|---|---|
| T | recipe C, the file's 100-155 lengths, 3 restarts, check top 2 | 26.6% (22.1-31.2) | 32 |
| TS1 | the paper's 150/200/250 lengths | 22.4% at 67 attempts (served seed), stopped | 15 |
| **TS2** | **6 restarts, check top 3** | **36.4% (31.2-41.6)** | **43** |
| TS3 | 10 restarts, check top 3 (served seed; 101 of 154 re-evaluated) | 48.1% | 41 |
| TS3f | 10 restarts, check top 3 - on the fixed server, every fold at its own seed | 48.1% (43.5-52.6) | 51 |
| TS4 | TS3f with a 2-seed internal check: a candidate passes only if it passes at two seeds | 51.9% (47.4-56.5) | 53 |
| TS6 | TS4 with LigandMPNN at temperature 0.2 (0.1) | 53.2% (48.7-57.8) - a null, 82 against 80 paired | 55 |
| **TS4 + polish** | **TS4, and where no restart passed: 8 MPNN draws on each of the 2 nearest misses' backbones, the best 3 checked (`polish`)** | **62.3% (57.8-66.9)** | **60** |
| TS7 | TS4 + polish at MPNN temperature 0.2 | 62.3% (57.8-67.5) - level with TS4 + polish; temperature a null again | 62 |
| *RFD4-Proteina high (paper)* | | *34.3%* | *77* |
| *original Protein Hunter / BoltzGen / DISCO (paper)* | | *0.8% / 4.0% / 2.6%* | |

- 🔴 **CORRECTED 2026-10-30, 04:30: THESE WERE 33.1% AND 56.5% AND BOTH WERE WRONG.** Every fold went to a resident
  `cuda/af3/fold --serve`, which parsed each job's `--seed` and folded at its own start-up seed (fixed in 5a2e9484).
  The internal check and the evaluation run at the same settings, so the evaluation of a checked design WAS its
  check, byte for byte - and a check that passes is selected to pass. Re-evaluated at each attempt's own seed
  (`task-bench.mjs --rescore`): T 51 -> 41 passes, TS2 87 -> 56, TS3 71 -> 49. The table is the re-evaluation.
  The protein-binder NC barely moves under the same correction (37.5 -> 37.9%, `rescore.py`, batch mode, which
  always honoured the seed): with the target templated the fold hardly depends on the seed; a small molecule's
  interface PAE sits on the 1.5 line and does.
- **TS2 is level with RFD4-Proteina** (36.4 against 34.3%), and 10 restarts (TS3, partial) points higher.
- An internal check that passes is 64% likely to pass the evaluation at another seed (TS2: 56 of 87); the losses
  are interface PAE moving across 1.5. **Checking at two seeds** (TS4, `check.seeds`) lifts that to 78% (70 of 90,
  against TS3f's 67 of 103) and the paired pass count 74 -> 80 of 154, for 13% more GPU a attempt (18.1 s against
  16.0) - flat per GPU-hour (104 against 108 pooled).
- **Polish is the largest single gain after restarts:** run on TS4's 74 failures it rescued 16 (80 -> 96 of 154),
  at ~11 extra design folds where it ran. Its restarts reproduce TS4's records exactly (seeded, and the fixed server
  is deterministic at a seed), so TS4 + polish is TS4's passes plus polish's. MPNN temperature 0.2 moved nothing.
- **Ten restarts on the fixed server, honest: 48-52% against RFD4-Proteina's 34.3%,** a pass on 51-53 of 77 targets
  at two attempts a target. Per GPU-hour it is ~5x behind (104-108 against their 559, A100 against H200).
- Bigger binders hurt (TS1: 15 passes against T's 19 on the same 67 attempts).
- 🔴 Caveats, as for PPI: the internal check is the evaluator's model; 2 attempts a target cannot give a clustered rate
  (theirs 200); their AF3 read each ligand from the input file's chemistry and ours from SMILES or a CCD code; and
  their 100% target coverage at 200 attempts is not comparable with 60 of 77 at 2.

DNA: **TD9 is ahead of RFD4-Proteina, on few passes: 7.1% (CI 1.8-14.2), 4 of 56 attempts** (theirs 3.5%,
original Protein Hunter 0.3%) - `config-TD9.json`: LigandMPNN for the binder (not NA-MPNN), no C/D/E, 10 restarts,
`contactTries` 5, `stick`, `minFootprint` 8 screening the checks only. By target 7m5w 3/19, 7rte 1/18, 7n5u 0/19.
- Designs dock on only 4-9% of design cycles; near-misses slide along the duplex with binder and DNA each rigid.
- TD12 (TD9 with a K/R logit bias of 0.3, `residueBias`): 0/32, stopped. At +1 the designs were 45% K/R and none docked.
- TD14 (TD9 with ESMFold2 designing cycles 0-2, AF3 after): 0/42, stopped. Pure EF2 design docks every cycle and
  AF3 then reads interface PAE ~23 - ESMFold2's dock is not one AF3 agrees with.
- Earlier, before TD9:
- T (30 attempts): 21 never produced a candidate - the binder drifted 20-40 A off the 10-bp duplex every cycle.
- TD1 (beam 3, 6 restarts, give up an undocked restart at cycle 2) and TD2 (a fold that leaves the binder undocked is
  refolded with up to 5 seeds, `contactTries`): 0%, and the designs were ACIDIC - D/E ~25% against K/R 16-21% -
  because MPNN was designing a soluble protein that never touched the DNA.
- TD3 (no C/D/E, `omit`, plus `contactTries` 5): still 0/30, but the designs dock: final interface PAE mostly 1.5-2.5
  (one 1.53 at 7.6 A), and a few within the 8 A RMSD. TD4 adds TS2's 6 restarts and top-3 check.

**Passes per GPU-hour against theirs** - the paper's own measure: passes over (generation + AF3 evaluation) GPU-hours,
theirs on H200s, ours on this A100. Ours is estimated, not metered: each attempt's folds counted from its records
(design folds at 100 steps x 3 recycles, checks and the evaluation at 200 x 5) times the resident server's own
per-fold GPU time at 179 tokens (0.34 s and 0.59 s), scaled linearly by the attempt's token count - which UNDER-states
the larger PPI complexes, whose trunk grows faster than linearly. MPNN and featurisation run on the CPU and are not
counted, as their CPU MPNN is not.

| task | run | GPU s an attempt | passes / GPU-hour (pooled, mean over targets) |
|---|---|---|---|
| small molecules | T (3 restarts) | 7.0 | 137, 164 |
| small molecules | TS2 (6 restarts) | 11.1 | 118, 172 |
| small molecules | *RFD4-Proteina high (H200)* | *2.2* | *559, 570* |
| small molecules | *BoltzGen / original Protein Hunter / DISCO* | *3.2 / 58 / 79* | *46 / 0 / 1* |
| protein binders | NC | 11.2 | 120, 153 |
| protein binders | *RFD4-Proteina high (H200)* | *6.5* | *106, 126* |
| protein binders | *BoltzGen / original Protein Hunter / BindCraft* | *6.3 / 81 / 156* | *28 / 3 / 2* |

- On protein binders Protein Hunter is level with RFD4-Proteina per GPU-hour on a SLOWER card, and ahead per attempt.
- On small molecules it is level per attempt (36.4 against 34.3%) and behind per GPU-hour by ~3-4x, of which the
  A100-to-H200 gap is perhaps 2x: RFD4-Proteina's attempt is 2.2 GPU-seconds, ours 11. (Re-evaluated passes.)
- Against every other baseline in the paper, including the original Protein Hunter, it is ahead on both by one to
  two orders of magnitude.

### Toward every reported number (2026-10-30, afternoon)

The goal: at or above RFD4-Proteina high on every task the paper reports. The paper's `rfd4-paper-data.json` has
FIVE criteria, not three - `ppi`, `sm_binder`, `disco` (a second small-molecule criterion), `dna`, and `ame` (enzyme
active sites, MCSA) - all defined in their `configs/metrics/pass_criteria/all_pass_criteria.yaml`.

| task | criterion | RFD4-Proteina high | ours, best | status |
|---|---|---|---|---|
| protein binders | ipTM >= 0.8, min iPAE <= 1.5, binder RMSD <= 2.5 | 19.3% | 37.9% (NC, native, each attempt at its own seed; 24.1% was the page's recipe C) | ahead |
| small molecules | `sm_binder` (above) | 34.3% | 62.3% (TS7) | ahead |
| small molecules | `disco`: backbone RMSD <= 2, ligand centroid <= 2 A | 64.7% | **81.2% (TS9, CI 76.6-85.1)**; 74.7% (TS7) | ahead |
| DNA | min iPAE <= 1.5, binder RMSD <= 8 after DNA alignment | 3.5% | 8.3% (TD15), 6.7% (TD9) | ahead, on 4-5 passes |
| enzyme active sites | `ame` (below) | 14.7% | **21.6% (TA6, all 37 targets, 74 attempts, CI 17.6-25.7), a pass on 11** | ahead |

**Against RFD4-Proteina's SEARCH pipeline too.** One attempt in all three is "do what you like, submit one design":
their one-shot "high" generates once, their "Search" scores several candidates with AF2-multimer (PPI) or OF3 (the
rest) and keeps the best, and ours runs restarts, polish/climb and an internal AF3 check - search, with the
EVALUATOR's model as the scorer (a different seed), which favours us. So Search is the like-for-like column:

| task | ours | RFD4 high | RFD4 + Search |
|---|---|---|---|
| protein binders | 37.9% | 19.3% | 28.1% |
| small molecules | 62.3% | 34.3% | 46.6% |
| small molecules, `disco` | **81.2% (TS9, CI 76.6-85.1)** | 64.7% | 76.4% |
| DNA | 6.7-8.3% | 3.5% | 4.8% |
| AME | 21.6% | 14.7% | 18.7% |

Not run here: the variants with a FIXED target (small molecules 15.9 / 22.9%, disco 45.3 / 52.7%, DNA 2.2 / 4.5%),
AME with the ligand unfixed (15.2 / 19.6%), PPI all-atom (16.9%). Per GPU-hour RFD4-Proteina leads on small
molecules (~5x) and is level on protein binders.
- **TS9 = TS7 with its internal check on the `disco` criterion** (`criterion: "disco"`: backbone RMSD <= 2 A and every
  ligand centroid within 2 A, the attempt judged the same way): each criterion is its own benchmark, and TS7 was
  selecting for the stricter sm_binder one. **81.2% (CI 76.6-85.1) over all 77 targets, a pass on 74** (150 of 154
  attempts; 4 lost to a full disk, re-run); the harness's verdict and `disco-score.py`'s rescoring from the PDBs agree
  to the digit. Against TS7's 74.7%: selecting on the criterion that is judged is worth ~6 points.
- **`disco` rescored, nothing refolded** (`disco-score.py`): from each row's own design and refold PDBs, the binder's
  backbone fitted, the largest ligand-chain centroid distance. The paper groups a ligand by connectivity including
  coordination bonds (FE2 + AKG is one molecule there, two chains here); the max over two centroids is never smaller
  than over their union's, so ours can only be stricter. TP2 (74 attempts) 67.0%, TS7 74.7%, TB1 (67) 80.6%.
- **TB1** (TP2 + beam 3, the beam fixed in 8bca8fa0): level with TS7 on the same 61 attempts, 44 against 44 (10 won
  by each alone), for ~3x the design folds. Stopped at 67 to free the GPU.
- **TD15** (TD9 + polish), finished: 5/60, 8.3% (CI 1.7-15.0) against TD9's 4 on the same 60 (one more, from
  polish). 7n5u still 0 - no arm has passed it.
- **TD16** (TD9 + beam 3): 0 of 14 (TD9 1 of the same 14), still running.

**Motif inpainting in the native sampler (`cuda/af3 --motif=<file>`, 7cc0de8a).** At every diffusion step the listed
atoms' target coordinates are Kabsch-fitted onto the denoised estimate's own copies of those atoms and written over
them; the noisy walk then steps toward the edited estimate. The target keeps its internal geometry; the fold
chooses where it sits. Lines are `<token> <atom> x y z`; a negative token is a whole chain (asymId), for a ligand's
one-atom tokens. `--motif-until=<A>` stops guiding below a noise level. Checked on M0050 (6 motif residues + U5P):
motif RMSD 0.0007 A guided, 6.26 A unguided, same seed. Only DESIGN folds are guided; the internal check and the
evaluation fold free, as the paper's oracle does.

**AME (enzyme active sites), first arm TA1.** The 37 `mcsa_37_fixed_ligand` targets (the paper's `mcsa_fixed`):
1-7 catalytic residues each, given as their COORDINATE-CONDITIONED atoms only (the non-NaN rows of
`_condition_coordinate_1_atom` - side-chain tips; the backbone atoms their README says were recovered from the PDB
entry are not conditioned, and we do not use them), 1-3 ligands by CCD code with their conditioned atoms, a
180-residue scaffold, the motif UNINDEXED (the designer places it). Pass (`ame`): min interface PAE < 1.5, backbone
RMSD <= 2 A design to refold, the refold's motif atoms within 1.5 A of the input's (one superposition over all of
them - every conditioned atom is coordinate group 0), and no polymer-ligand pair under 1.5 A in the design.
- Recipe: TP2's loop (6 cycles, 10 restarts, top-3 check, polish), the motif residues at random positions >= 4
  apart and >= 8 from either end (a new placement each restart), held through every LigandMPNN draw; every design
  fold inpainted with the motif tips AND the ligands' conditioned atoms as one rigid body.
- 🔴 Not carried: M0110's inter-ligand bonds (FAD-PER-DAL fold as three molecules).
- **TA1 (guided every step, every cycle): 0 of 8, stopped.** Confident designs (refold ipTM ~0.9, iPAE 0.9-1.5,
  backbone 0.4-2.6 A) whose free refold puts the motif 2.7-14.6 A off, and many designs clash (polymer-ligand pairs
  under 1.5 A: the rigid motif-plus-ligand body forced into the chain). A design guided to the end holds the motif
  exactly, so nothing selected for one the sequence carries.
- **TA2 (`motifUntil` 1 A - the last ~40 of 100 steps free - and checks gated on the design's own motif <= 1.5 A,
  unclashed): 0 of 2, stopped.** The design folds now vary (motif 0.8-2.6 A; 24 of 70 under 1.5 on M0040), but the free
  check refolds still put it 4.7-15 A off.
- **TA3 (guided cycles 0-2, then 6 cycles folding free; only a free cycle is checked):** smoke test on M0040 - the
  first free cycle loses the motif (1.2 -> 11.3 A) while the interface improves (iPAE 4.8 -> 1.2). AF3 binds the
  ligand from residues of its own choosing, not the held ones. Replaced by TA4 before it finished.
- **TA4 = TA3 + `motifMatch`**: on each free fold, the motif re-found UNINDEXED among the fold's residues of the right
  types near a ligand (depth-first over assignments, pruned by pairwise tip-centroid distances within 3 A, the full
  RMSD at each leaf), and the held positions moved there. Smoke test M0040: free-cycle motif 2.1-2.3 A where the
  held positions read 11-15. **0 of 6, stopped** (free-cycle motif best 1.48 A on M0040, 9.8 on M0050).
- **Where RFD4-Proteina's 14.7% comes from:** metal and small sites - M0129 (Fe/aKG, 4 residues) 63%, M0097 (Zn) 47%,
  M0630 (1 residue) 46%, M0738 43%, M0710 37% - against 0-2% on most 5-7-residue motifs. So the strong targets
  are tested first.
- **TA5 (`probe`): every cycle guided, AND the same sequence folded again FREE at the design settings, the motif
  matched unindexed in it; the probe's interface, motif, positions and structure are what is judged, checked and
  submitted; MPNN designs from the guided backbone.** On RFD4's eight best targets: **M0630 and M0097 PASS** (the
  first AME passes), M0129 not (one probe in 70 under 1.5 A; the evaluation 17 A).
- **TA6 = TA5 + `climb`** (instead of polish): from the 3 free structures nearest the motif, 6 generations of 6
  MPNN draws with the matched residues held, each folded free and matched again, the best kept when closer
  (motif RMSD, then interface PAE); every structure reached is a check candidate. Smoke test M0097: 3.5 -> 1.28 A in
  two generations. With no passing check the submitted design is now the nearest to passing, never a guided one.
  Results so far: 0 of 6 on the alphabetically first (mostly hard) targets. The climb does not rescue a large motif
  - it starts 5-8 A off on M0050/M0058/M0078 and never improves; it reached 1.37 A once on M0040 (check 2.43).
- **TA5 final on RFD4-Proteina's eight best targets: 3 of 8** (M0630, M0097, M0710) against their 39% mean there.
- **TA7 = TA6 + `mpnnMsa` (the user's idea): an alignment of MPNN's own draws** - 32 sequences off the previous
  cycle's backbone at temperature 0.3, the motif held - handed to AF3 as the binder's unpaired MSA in every guided
  design fold, its depth falling linearly to none at the last cycle so the run ends single-sequence, as it is
  judged. Probes, checks and the evaluation stay single-sequence. Run in RFD4-Proteina's per-target order, best
  first. **A null, stopped at 32 of 37 targets.** Paired with TA6 on the same attempt-0 targets (same placements and
  seeds through `pair`): **10 passes against 9**, 5 targets only with the MSA and 4 only without - where TA6's own
  second attempt on those targets passes 6, so one is noise. The probes (the single-sequence folds that judge each
  design) are slightly WORSE with it: motif within 1.5 A on 7.5% of cycles against 8.6%, median iPAE 4.79 against
  3.75. The 31% it showed beside TA6's 21.6% was a strongest-first subset against all 37. It also costs 32 MPNN draws
  a cycle on the CPU. TA6 stays the AME recipe.
- **TA6, 2 attempts on each target (TA6 + TA6b): 22.1% (CI 17.6-26.5) over 34 targets, a pass on 10**, against
  RFD4-Proteina high's 14.9% on the same 34. Ours both times: M0129 (theirs 63%), M0710 (37%), M0110 (13%), M0717
  (8%), M0096 (1.5%); once: M0630, M0636, M0209, M0739, M0664. Missed where theirs is high: M0097 (47%), M0738 (43%),
  M0555 (26.5%), M0157, M0500. The probe and the matching carry it; the climb rescues no large motif.
- 🔴 M0365, M0663 and M0907 errored on the first design fold: their input conditions a ligand atom AF3's CCD
  component does not build (FBP, RIB and CAP's O1, leaving atoms) and `--motif` refuses an atom it cannot find.
  The harness now folds a stub once a target and guides only the ligand atoms AF3 builds (e3d2219a; only the
  PROTEIN motif atoms are scored, so this changes nothing judged); the six attempts re-run: **all 37 targets, 21.6% (CI 17.6-25.7),
  a pass on 11, against RFD4-Proteina high's 14.7%.**

**DNA held in a fixed frame, TD17.** The DNA benchmark gives the duplex as SEQUENCE only (`*_sequence_context`,
coordinates zero; `*_structure_context` is the fixed variant only BoltzGen and RFD3 ran, at 0.8%), so every design
fold made its own duplex and MPNN designed against a different one each cycle. TD17 = TD9 + `dnaFrame`: the duplex
folded once per target from its sequence alone (beside a 5-residue stub; no information the unfixed input lacks),
then inpainted into every design fold token by token. Checks and evaluation fold free. Same 60 attempts as TD9.
- **0 of 10, stopped** (TD9 1 of the same 10). Docking unchanged: 11 of 92 design cycles at interface PAE < 5
  against TD9's 17 of 113. A fixed duplex is not what keeps the binder off it.
- **TD16 (TD9 + beam 3): 0 of 18, stopped** (TD9 2 of the same 18).
- Not tried, if DNA is revisited: `--motif-until` (release the frame for the last steps, so the design agrees better with a free
  refold), a partial pull, and holding only the part of the target the binder does not touch.

### The page against native: why recipe C scored 24.1% and NC 37.9% (2026-10-30, night)

The same Protein Hunter recipe (6 cycles x 100 steps x 3 recycles, 3 restarts, target templated, internal AF3 check
of the top 2) scores 24.1% through the page (`bench.js`, C) and 37.9% natively (`native-bench.mjs`, NC). Narrowed:
- **Not the evaluation.** The page's AF3 scores NC's own designs at 37.5% against native's 37.9%, agreeing on 223
  of 240 attempts (`R-NC.jsonl`'s page columns).
- **Not the internal check.** A passing page check passes the evaluation 44 times in 50.
- **Not the loop.** `hunt()` and `native-bench.mjs` call the same `runDesign`, SolubleMPNN both (the page's "auto"
  picks it for protein-only), the same contact veto (no hotspots in either config), the same check and submission
  rule. Lengths vary per attempt in both.
- **Not the seed.** `hunt()` folds every cycle of a run at ONE diffusion seed where `native-bench.mjs` varies it per
  cycle - but NC's design folds ran on the resident server before the served-seed fix, which ignored every job's seed,
  so NC was one-seed too and still 37.9%. (`sameSeed` is in `native-bench.mjs` as the control, not run.)
- **Not the folding engine on finished designs** (`engine-probe.py`): NC's submitted binders, attempts 0-2 on every
  target, two seeds, at the DESIGN settings - check-worthy (pTM >= 0.8, iPAE <= 2.5) native 74 / page int5 71 / page
  float32 76 of 144, agreeing with native on 137 and 140; median iPAE 1.76 / 2.32 / 1.82.
- **Where it diverges: the design trajectories.** The page vetoes 4-10% of design cycles for touching nothing (native
  ~1%), and its cycle-by-cycle iPAE falls more slowly (cycle 6 median 10-12 against 7.3): 5.6% of page design cycles
  are check-worthy against 10.9%, so fewer attempts reach a passing check (23% against 38%). The warm start is not it
  (W0, no warm start, 20.8%). What is left is the page's int5 weights on weak, early-cycle sequences - a 0.5 A worse
  median iPAE on finished designs that agree, and possibly more on the X-rich ones a run starts from.
- **The test running: PI/PF**, the page's own loop on its int5 bundle against the float32 export (`cfg.bundle`, a
  `bench.js`-only override of AF3's registry entry), no warm start, attempts 40-41 on all 24 targets, paired; NCn is
  native on the same attempts. **44 paired attempts: page int5 10, page float32 10, native 14** - int5 is not it; the
  page's loop still trails native by about a third (means 20.8 / 27.1 / 33.3% over 24 targets). Its design cycles
  are vetoed for touching nothing ~4% of the time against native's <1%, and it folds every cycle of a run at one
  seed where native varies it per cycle - the next arm.
- **Final, 48 paired: page int5 12, page float32 13, native 16 - and the page's designs RE-SCORED NATIVELY
  (`rescore.py`) pass 16 (PI) and 14 (PF).** So today's page design loop is native's: the deficit was the page
  scoring its own designs with one AF3 seed, a label that reproduces at another seed only 64-78% of the time. The
  funnel agrees: check-worthy cycles 8.6 / 10.2 / 10.6%, attempts with a passing check 16 / 18 / 18. The
  per-cycle-seed arm (PS) was stopped as moot.
- **So `bench.py` now scores natively** (`native_eval.py`, shared with `rescore.py`): every submitted page design is
  evaluated by the native AF3 at the attempt's own evaluation seed, and the page's own score of it is kept beside it
  as `pageBinderPtm`/`pageIpae`/`pageRmsd` (`evaluator: "native"` on the row). A page run and a native run are
  judged by the same model from here on.
- **Not the cause: an untemplated design loop.** `hunt()` gained `templates` at 07:19 on 10-29 and C finished at
  11:07, so C looked like it might have designed against an untemplated target - but its design structures sit as
  close to the crystal target as every templated run's (target CA RMSD median 1.05 A, against 1.04-1.12).
- **The 10-29 page runs are low on native scoring too: W0 (no warm start) 8 of 48, W1 (warm start) 7 of 48**, where
  today's same configuration (PI) passes 16 of 48 on other attempts. Either 10-29's code or those attempts (W's
  lengths and seeds) were worse. **W0r** replays W0's exact attempts (same `pair`, so the same lengths and seeds) on
  today's code, scored natively: **10 of 48 against W0's 8**, with 29 of the 48 binders identical. So the code
  barely moved; W's attempts (their lengths and seeds) are simply harder ones than PI's (16 of 48). 8 / 10 / 16 of 48
  is within what attempt sampling gives, so there is no page deficit left to explain.
- **The engines at EVALUATION settings (200 steps, 5 recycles; `engine-probe.py`, 2026-10-31):** PI's 48 designs
  (attempts 40-41) x seeds 101-104, 192 folds an arm. Check-worthy (pTM >= 0.8, iPAE <= 2.5) native 110 / page
  float32 102 / page int5 101; **evaluator-grade (pTM >= 0.8, iPAE <= 1.5) native 77 / page float32 76 / page int5
  63**, median iPAE 1.71 / 1.68 / 2.19. Paired with native: float32 dIPAE median -0.01, worse on 22 of 48 designs -
  **the page's AF3 is native's**; int5 +0.08, worse on 32 of 48. So the only engine difference is the int5
  bundle's quantisation, a small upward shift in iPAE that lands right on the 1.5 threshold (18% fewer passes there).
  It matters for scoring, which is native now, and little for the design loop (71 against 74 check-worthy of 144 at
  design settings). No engine bug.

### The other datasets the paper reports (2026-10-30, night; scored 2026-10-31)

Besides each benchmark's figure-2 set, `rfd4-paper-data.json` reports a FIXED-target variant (the ligand's or the
duplex's coordinates given) and, for AME, an UNFIXED-ligand one. RFD4-Proteina high / + Search:

| variant | ours | RFD4 high | + Search | how |
|---|---|---|---|---|
| AME, ligand unfixed (`mcsa_37`, "U-" targets) | **20.3%** (CI 18.9-21.6; 37 targets, 74 attempts) | 15.2% | 19.6% | TA6's recipe; only the atoms that file conditions are inpainted (the tips, and a few ligand atoms) |
| DNA fixed (`*_structure_context`, "F-") | **TD18F 2 of 62 (3.2%, CI 0-7.9)**: above one-shot, below Search; TD20F (released below sigma 4) running | 2.2% | 4.5% | TD9's recipe; the given duplex inpainted token by token into every design fold |
| small molecules, ligand fixed ("F-") | **50.6%** (CI 46.1-55.2; 77 targets, 154 attempts; TS7F) | 15.9% | 22.9% | the file's conformer inpainted: a SMILES chain by token, in RDKit's `_smilesAtomOutputOrder`, checked element by element against what AF3 builds; a CCD chain by atom name; fewer than 3 atoms (an ion) unguided |
| `disco`, ligand fixed | **64.9%** (CI 62.3-67.5; 112 attempts) | 45.3% | 52.7% | the same rows, `disco-score.py` |
| PPI all-atom | - | 16.9% | - | same targets and criterion as `ppi_backbone`; NC's 37.9% is not re-run |

In every variant only the DESIGN folds are guided; the internal check and the evaluation fold free, and the
criterion is the task's own. `task-score.py` reports each variant against its own dataset and against Search.

- **DNA fixed is the one variant behind: 0 of 60 (0 of the first 34 below)** (7m5w, 7n5u, 7rte), where TD15 passes 5 of 60 on the SAME three
  targets unfixed. The design funnel is no worse - 5.0% of design folds reach iPAE <= 2.5 against TD15's 6.7%, 9
  checks against 23 at 0.26 against 0.38 an attempt, 3 checks at iPAE <= 1.5 each - but the submitted designs fail
  the free evaluation's placement: binder RMSD <= 8 after DNA alignment on 4 of 34 against 10 of 60. A binder
  designed on the crystal's duplex does not keep its place on AF3's own. **TD18F** adds TD15's polish and releases the
  guide below sigma 1.0 (`motifUntil`), so the duplex settles toward AF3's form in each design fold's last steps.
  **2 of 32 on attempts 0-10** (7m5w:1 iPAE 1.37 RMSD 4.3; 7rte:3 iPAE 1.35 RMSD 2.6), where TD9F's same attempts are
  0 of 32: 6.3% against RFD4 + Search's 4.5%, a two-pass difference, so attempts 11-20 are being added.
  **Final: 2 of 62 (3.2%)** - attempts 11-20 added none. On these three targets the guide is the cost: held to the
  last step 0/60, released below sigma 1 2/62, unguided (TD15, unfixed) 5/60. **TD20F** releases it below sigma 4, so
  only the duplex's coarse placement is held.
- **The DNA designer, paired (TD19 = TD15 with NA-MPNN, TD15's attempts): NA-MPNN 0 against LigandMPNN 3 on the
  first 18** - LigandMPNN stays the recipe's designer; running to 60.

- 🔴 **THE DISK FILLED (497 GB, 4.5 MB free)** while the page A/B's browsers each cached the 1.4 GB float32 bundle:
  native folds failed in the `fold` wrapper (`exit: : numeric argument required` - its `.done` file could not be
  read), and three result files got truncated lines. Error and truncated rows were dropped and re-run; `~/.cache`
  was cleared (the user's call; the CCD cache kept, the featuriser reads it every fold) for 33 GB.
- The page A/B (PI/PF/NCn) was restarted lean, attempts 40-41.

### Motif inpainting on the page (2026-10-31)

`cuda/af3 --motif`, ported to the WebGPU sampler: `cpu/af3/diffusion/inpaint.js` (`motifSlots` resolves the native
file's entries - a token and an atom name, a negative token a whole chain - against the batch; `inpaintDenoised`
fits the target onto the denoised answer's own copies of those atoms and writes it in), applied in
`sampleOnGpu` after steering (`motif: {slots, xyz, until}`), threaded through `foldBatch`, `foldAf3`
(`motif: {atoms, until}`) and `hunt()` (`motif: {atoms: length => [...], until}`, a function of the run's drawn
length because a token after the designed chain moves with it). Refused for OpenDDE (structural tokens), as
steering is. Checked: M0050 (six catalytic tips + U5P) through the page's own `foldAf3`, same seed - motif RMSD
**0.0007 A guided against 19.14 unguided** (native: 0.0007 against 6.26). `test/motif-inpaint.test.js` holds the
name resolution and its refusals, and the fit (watched failing with the fit removed). Not yet a control on
design.html: the page has no way to pick a motif's atoms or a ligand's conformer.

### The benchmark recipe on design.html: Search (2026-10-31) - built, and it is the bench's loop

Every winning run is one loop around a hunt with a few switches that the job's own contents decide. So that loop now
lives in the page rather than in the bench harnesses:

- **`cpu/design/protocol.js`** holds the recipes as data, named by the run that measured them:
  - `PROTOCOLS.ppi` = NC;
  - `PROTOCOLS.sm` = TS7;
  - `PROTOCOLS.dna` = TD15.

  `protocolFor(contents)` picks one from what the job carries: a ligand, a DNA/RNA chain, or a protein partner. It
  returns none, with the reason, for a job with nothing to bind, a designed nucleic chain, or a ligand and DNA
  together (no benchmark measured that).

  The same module holds what the recipes are judged by:
  - `interfaceTerms`: the designed chain's pTM, its mean pair ipTM, and its lowest two-way interface PAE;
  - `verdict`: the paper's three criteria on a design and its refold;
  - `footprint`.
- **`web/search.js`** is the loop:
  - restarts at `seed + 101 r`, each a one-run `hunt()`, stopped by `controller.abort()` on a confident cycle;
  - the restart's top cycles refolded FREE at 200 steps / 5 recycles, at `7 seed + cycle + 1000 r`, every one of
    `check.seeds` seeds required;
  - a polish of the nearest checked misses when nothing passed;
  - the answer: the first passing check, else the nearest miss.

  Those seeds are bench.js's, so a PPI search is bench.js's NC attempt. Its folds and designer are injectable, and
  `test/design-search.test.js` runs the whole loop with scripted folds. It was watched failing with the early stop
  removed and with the check-pass exit removed.
- **`web/hunt.js`** gained the DNA recipe's two fold-side switches, both off by default so a plain hunt is unchanged:
  - `foldTries`: an undocked fold is refolded at `seed + 7919 k`;
  - `stick`: an undocked cycle hands the next design the run's last docked fold, marked `stuck`, and the search
    never chooses it.
- **design.html** has **Search: One hunt / Benchmark recipe** beside Cycles:
  - the note under it names the recipe it will run and why, or why none fits;
  - the recipe's cycles, temperature, omitted letters and designer replace the page's own, and the note says so;
  - the answer is marked best in the table and the refold picker, and a polished design is added to the picker,
    since it was never a cycle.

  Driven in a fresh page:
  - from scratch: "nothing to bind";
  - PD-L1: NC;
  - SAM: TS7;
  - 7M5W: TD15.
- **The check:** `bench.js` with `search: true` (config-PIS) sends a PI attempt through `web/search.js`.
  - **01_PD1:40 reproduced PI's row exactly**: same binder, restart 3 cycle 4, 3 restarts.
  - **04_IFNAR2:40 did not, and the cause is upstream of the search.** Cycle 0, the same seed and the same starting
    sequence, already read iPAE 6.204 against PI's 6.244.
  - A fresh-process fold at a seed repeats to the digit (fold.js, pLDDT 33.5505 twice). So the page's folds depend
    on what the page did before: PI ran its attempts in another order in a long-lived browser.
  - **The A/B, each arm in its own fresh browser: the old loop and `web/search.js` give the same binder, the same
    restart count and the same best cycle on 04_IFNAR2:40, 05_CD45:40 and 12_Claudin1:40, with every design fold's
    interface PAE identical to the last digit.** The fresh IFNAR2 run also reproduces
    PI's own row (1 restart, restart 1 cycle 3). So the search is the bench's loop.
  - 🔴 **The PIS divergence is a page-state leak, and it is open:** the same attempt folds differently after the page
    has run another one (PIS ran PD1 first). A long-lived design.html session is therefore not a fresh one. Every
    resident-state bug this repository has had (CLAUDE.md, "A resident server is not a fresh process") had this
    shape, and none was found by a per-process gate.

### Held coordinates and the enzyme recipe on design.html (2026-10-31) - built, checked on the page's engine

What was "not on the page yet" above:

- **Coordinates: Free / Held to the structure** (beside Search). Held inpaints into every design fold:
  - the loaded structure's ligands;
  - its kept DNA/RNA strands;
  - the motif residues inside the designed chain.

  It works for a plain hunt (guided to the last step) and for a search (as its recipe says). The check and the refold
  still fold free.
  - `cpu/design/motif.js heldCoordinates` builds the entries from `chainsOf`'s new `structureParts` and lays them on
    the fold's tokens:
    - a polymer residue at its chain's offset, with the designed chain at the run's length;
    - a ligand as its whole chain (`-(chains + 1 + j)`), by atom name.
  - Every entry is `optional`: `motifSlots` drops and counts a name the fold does not build (a leaving atom, a 5'
    phosphate, an OXT) and still refuses a name it has twice.
  - Each run's job builds its own, so a drawn length moves the tokens with it.
  - Held with nothing to hold, or no structure, is refused with a sentence before any weights load.
- **Checked through the page's own `foldAf3`**, one seed, 30 steps, unguided against guided:
  - the 7M5W fixed duplex (all 388 atoms resolved by name): **DNA RMSD to the given coordinates 2.38 A free, 0.0007 A
    held**;
  - M0040's catalytic tips plus ADP, from their names: **motif 18.9 A free, 0.0007 A held; ADP 1.86 A free, 0.0006 A
    held**.
- **The recipes, chosen from the job when Search is on** (`protocolFor`, with `motif` and `fixed`):
  - **smFixed = TS7F** (the conformer guided to the last step);
  - **dnaFixed = TD18F** (guided, released below sigma 1, `guideUntil`);
  - **ame = TA6:**
    - the motif and ligand guided, released below sigma 1;
    - `probe`: every design refolded UNGUIDED through `hunt`'s new `examine` hook, its motif found unindexed
      (`motifMatch`), and those terms and that structure are what the cycle is judged and checked by;
    - the stop and check want the motif within 1.0 / 1.5 A and no clash;
    - a climb on motif RMSD when no restart passed;
    - the paper's `ame` criterion in `verdict`;
    - the motif-score fallback.
- **Refusals:**
  - a motif with no ligand is refused (only enzyme design was benchmarked);
  - a motif that is not held is refused (there is nothing to score it against).
- **Tests** (`test/design-motif.test.js`):
  - the token layout;
  - the motif terms and the unindexed match;
  - the recipe choice;
  - AME through the loop with scripted folds: the guide and its release reaching the hunt, every cycle judged by its
    unguided refold (it fails with the refold ignored), and the climb on a miss.
- **Not yet:**
  - **No page search has been benchmarked:** the fixed and AME recipes ran on the native port (task-bench.mjs), and
    the page's versions of them are checked piece by piece, not by a pass rate.
  - **Motif placement is the contig's (indexed),** where TA6 drew random positions per restart. The unindexed match
    still finds the motif wherever each unguided refold put it, and the climb holds those residues.
  - **A SMILES ligand cannot be held:** the page holds a ligand by its CCD atom names; the fixed-sm benchmark's
    SMILES-order mapping (task-bench `ligandAtomsInOrder`) is not ported.
