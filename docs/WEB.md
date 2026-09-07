# The page: layout, templates, the archive, and the progress bar

`index.html`, `single.html`, `web/`, and the tools that drive them -
`tools/fold-in-page.py`, `tools/mobile-layout.py`, `tools/model-terms.py`.

## Templates, from the page

🔴 **AND A TEMPLATE IS REACHABLE FROM THE PAGE NOW.** It lives behind a protein
row's `⋮`, beside the modified residues, because the two are the same kind of
thing: set on ONE chain, changing what is folded, and invisible on the row - so
the badge counts both. It takes one source - `1abc`, `1abc_A` or a UniProt accession, one field because it
is one question - fetched by `web/template-source.js` from the RCSB or AlphaFold
DB and turned into a slot over the complex's TOKENS. The row shows what it
covered, because a template covering 17 of 120 residues folds perfectly well and
says nothing about it. Measured on a 53-residue target with 1QYS_A: ipTM 0.324
without, 0.358 with.

🔴 **AND TEMPLATES CAN COME FROM THE MMseqs2 SEARCH, WHICH ALREADY FOUND THEM.**
`pdb70.m8` is in the MSA job's own tar beside `uniref.a3m` - no second search,
no extra request - and nothing had ever read it. The structures come from
ColabFold's own server, `{api}/template/{1qys_A,7fao_C}`, as a gzipped tar of
mmCIF; the RCSB is not involved. **Ask with the chain suffix**: `/template/1qys`
answers 200 with a tar holding only the hhsearch index and no structure, which
is a success with nothing in it.

ColabFold then runs `hhsearch` over that index to get the alignment. A browser
has no such binary and does not need one: the m8's last column is a CIGAR - but
its target coordinates index pdb70's SEQUENCE while a template offers its
RESOLVED residues, so it is deliberately not used. The query is aligned to the
resolved sequence instead and the coverage line says what came of it.

Measured: a 91-residue query at `--msa-mode search --template auto` found
`1qys_A`, covered 91/91 and folded at pLDDT 84.9.

🔴 **AND A WATER IS NOT A RESIDUE.** `chainResidues` read every HETATM, so 1qys
chain A came out ninety-NINE residues instead of ninety-two - eight waters, each
an X with one atom, each a pseudo-beta position in the distogram. Found by
running the PDB reader and the mmCIF reader over the same entry and comparing.
MSE is the one heteroatom kept: selenomethionine is how a great many structures
were phased, and dropping it puts a hole in the middle of a chain.

## The phone layout, measured rather than looked at

🔴 **THE PHONE LAYOUT IS MEASURED, NOT LOOKED AT, AND `--window-size` CLAMPS AT
500px.** 390 and 320 both report an innerWidth of 500; `--headless=old` clamps
identically. `tools/cdp.py` is sixty lines of WebSocket (no new dependency) and
gives `Emulation.setDeviceMetricsOverride`, which is a true viewport at any
width, plus `Page.captureScreenshot`, which `--screenshot` cannot do on a page
with a running rAF loop. `tools/mobile-layout.py` runs 320, 360 and 390 with
1200 as the control, loading a structure and an alignment first - half the rows
it measures are `display: none` on a bare page. It checks `single.html` too.

🔴 **AND "NO HORIZONTAL OVERFLOW" IS NOT THE TEST.** Under mobile emulation a
page that cannot fit does not overflow: the LAYOUT VIEWPORT GROWS, so
`scrollWidth == innerWidth` while the phone renders everything zoomed out. The
assertion is `innerWidth == the width asked for`. Nor can a size check see an
OVERLAP, or a `1fr` grid track squeezed to nothing - the entity row's sequence
box measured 0px at 320 with "PIA" set one letter per line, and every fit check
passed. Nor can it see a page that is ready to be measured: `processFiles` is
defined while `initializeApp` is still running and before `web/app.js` (a
module) has run at all, and called in that window it resolves having loaded
NOTHING. Wait for `#predict` to be enabled, which is the last thing to happen.

🔴 **AN INLINE WIDTH IS ONE NO STYLESHEET CAN OVERRIDE.** Not a media query, not
a container query, not any specificity - only `!important`, and a page whose
responsive rules all need that has no cascade left. There were four in
`index.html`, five in `single.html`, two on the MSA filter sliders, and one that
py2Dmol's own JS writes on the viewer box (turned off with
`data-autosize="css"` on `#canvasContainer`). `max-width` is a DIFFERENT
PROPERTY and beats an inline `width` with no `!important` at all, which is how
the MSA panel is contained and how `single.html` lost three of its own.

🔴 **AND py2Dmol's MSA PANEL IS STILL 948px.** `src/panels/msa.js` has
`const MIN_CANVAS_WIDTH = 948`, clamps every canvas width up to it and writes
the result as `container.style.width`. Our narrow block gives that box
`max-width: 100%; overflow-x: auto`, so it scrolls sideways inside its own card
instead of taking the whole document with it - measured, it was forcing a 972px
layout viewport on a 320px phone. Fixing it properly is an upstream change.

## The template source menu, and the download dial

🔴 **AND THE TEMPLATE'S SOURCE IS A MENU NOW, NOT A GUESS.** One box took
`1abc`, `1abc_A` or an accession and decided which server to ask by counting
characters - four is the RCSB, anything else AlphaFold DB. It reads well and it
is right most of the time, and both of its failures are silent: a
four-character accession goes to the wrong server, and a typo'd PDB id becomes
an AlphaFold DB lookup whose 404 names a database nobody chose. The dropdown -
`TEMPLATE_KINDS` in `web/entities.js` - carries **PDB entry**, **AlphaFold
DB**, **From the MSA search** (which was a checkbox that silently overrode the
box beside it) and **Upload a structure** (which had no way to be named at
all). `fetchStructure(text, {kind})` takes the kind; `parseSource`'s
count-the-characters rule survives only as the fallback for a caller that has
none.

🔴 **AND AlphaFold DB PUTS ITS VERSION IN THE FILENAME, WHICH MOVES.** The URL
was built as `AF-<id>-F1-model_v4.pdb`, and AlphaFold DB's v6 release retired
v4 outright - `curl -I` says 404 for v4 AND v5 on every accession tried - so
every AlphaFold DB template on the page was a 404 naming a URL the user had not
chosen. It asks `https://alphafold.ebi.ac.uk/api/prediction/<id>` for `pdbUrl`
now: one request, CORS open, and nothing to bump at v7. Measured: P61626 covers
9 of the 58-residue default, 1QYS_A covers 8, and the same file uploaded from
disk covers 8 - the upload and the download agreeing is the cross-check that
the two routes reach the same slot builder.

🔴 **AND THE pLDDT FLOOR IS GONE.** It defaulted to 70 for AlphaFold DB, on the
sound reasoning that a predicted structure has every residue and no way to say
it did not see one - so a disordered tail arrives as geometry. But nothing on
screen said the default had done anything, and a number from 0 to 100 is a
modelling choice the popup cannot explain in the space it has.
`buildTemplate`'s `minConfidence` option and `filterByConfidence` remain for a
caller that wants them; no page sets one.

🔴 **`fold-in-page.py` FOLDS ONE CHAIN, AND A COMPLEX FAILS SILENTLY.** It
types the whole `--sequence` into a SINGLE entity's field, so a colon-joined
`A:B` is rejected by the page - "One sequence per entity - use Add entity for
another chain" - the Fold click does nothing, and the tool waits out its whole
timeout. Two 25-minute runs went that way before `cdp.wait_for` learned to print
the page's status line as it changes (`progress=STATUS_LINE`), which answered it
in ten seconds. Watch those `...` lines: a wait that prints nothing never
started. Driving a real complex needs the tool to build one entity per chain
through `window.__entityList`, which it does not do yet.

🔴 **THE WEIGHTS AND THE MSA SEARCH RUN TOGETHER NOW, AND USED NOT TO.** The
model was loaded inside the fold, which runs after the alignment - so a cold
page with the MSA set to search spent the whole MMseqs2 round trip with the
network otherwise idle, then spent the whole download with the search already
answered. They need nothing from each other: one is a static file from a CDN,
the other a query against a server that queues. `startModelPreload` begins the
load before the templates and the alignment; both loaders memoise, so the fold
awaiting the same call later gets that promise rather than a second download.
Measured with `tools/fold-in-page.py --timeline`, which reads resource timing:

| | search | model | overlap |
|---|---|---|---|
| before | 795-1426 ms | 1472-1902 ms | **-46 ms** (strictly sequential) |
| after | 1296-2227 ms | 1359-1844 ms | **+485 ms** |

🔴 **AND A `--timeline` THAT FILTERS BY NAME ALONE MEASURES PAGE LOAD.** The
first version reported a 1.2-second overlap - and the UNCHANGED tree reproduced
it exactly, because `/mmseqs/` matches `src/input/mmseqs2-api.js` and the
weights directory is probed before the button is pressed. Both spans started at
66 ms, which is not a span of anything a click caused. It stamps
`window.__foldClickedAt` at the click and ignores everything earlier.

🔴 **AND THE DOWNLOAD MUST NOT WRITE TO THE STATUS LINE ANY MORE.** Two writers
several times a second, and the message that loses is the one about a server
that may queue for a minute. It reports on the right instead - a filling dial
plus `AlphaFold 3 · 92 / 265 MiB` - and appears only once a load reports itself
partway, so a model already in the shard cache does not flash it.

🔴 **AND `tabular-nums` DOES NOT STOP A COUNTER RESIZING.** It holds every
DIGIT to one width, which is not the problem; the problem is a number that
GROWS a digit, so `1 / 265` became `10 / 265` became `100 / 265` and the label
stepped wider twice per download - moving the dial right and squeezing the
status line, twice, every time. The loaded figure is padded to the width of the
total with **U+2007 FIGURE SPACE**, which is defined as a digit's width. A
plain space does not work: it collapses, and measured it steps exactly as the
unpadded string does. Measured at 1, 10, 100 and 265 MiB, with the unpadded
string as the control that says the measurement can see a difference at all:

| pad | widths | constant |
|---|---|---|
| none (control) | 132.78, 139.78, 146.77, 146.77 | no |
| a plain space | 132.78, 139.78, 146.77, 146.77 | no |
| U+2007 | 146.77 x4 | **yes** |

🔴 **AND A CSS TRANSITION ON A VALUE THAT CHANGES 8776 TIMES IS PURE LAG.**
The dial's fill carried `transition: stroke-dashoffset .2s linear`, to sweep
rather than jump between shard callbacks. `HttpTensorStore` reports once per
network CHUNK - 8776 callbacks over one load of `model-af3-int5` - and a
transition retargeted that often never arrives: each callback restarts it from
wherever the arc has got to, so the arc trails the true value by about
`duration x rate`, and the rate is one whole arc over the length of the load.
The dial read 99.6% while two thirds drawn, and was hidden there, so a load
that COMPLETED looked like one that stopped. Measured with
`tools/fold-in-page.py --bar`, which reads the drawn dashoffset beside the
label, on a 450 ms local load:

| label | arc, with the transition | without |
|---|---|---|
| 96 / 265 MiB | 0.05 | **0.39** |
| 192 / 265 MiB | 0.28 | **0.74** |
| 264 / 265 MiB | 0.64 | **0.93** |

🔴 **AND IT IS WORSE THE FASTER THE LOAD**, which is why nothing caught it: a
slow link lags 0.2 s in 30, which is 0.7% and invisible, while a returning
visitor with the shards cached sees almost none of the arc at all. The DATA was
right the whole time - `tools/gpu/probe-load-dial.js` reports the store's
fraction ending at exactly 1, and the manifest's 264.6 MiB matches the shards on
disk and the shards Hugging Face serves - so every check that read the numbers
passed. **Sample what is DRAWN, not what was computed.**

🔴 **AND THE DIAL IS NEVER LAID OUT UNLESS SOMETHING FORCES IT.** It is
`hidden` on a bare page, so `tools/mobile-layout.py` had never seen it, and its
label cannot wrap or shrink: at 320px the label took 171px of a 254px row and
left the status line **75px**, which fits none of "MSA search · queued
(PENDING) · 41s". Nothing overflowed, so every fit check passed. The label is
`display: none` below 560px; the dial and its title carry the bytes there.
Measured by forcing the dial visible at each width, which is the only way it is
ever laid out.

## The archive, and the round trip back in

🔴 **"DOWNLOAD ALL" WRITES THE AF3 SERVER'S ARCHIVE, AND THE UPLOAD BOX READS
IT BACK.** `web/zip.js` is a writer and a reader in one file; `web/fold-archive.js`
assembles the members. Checked against `tools/fixtures/fold_2026_09_01_10_17.zip` in the repo
root: `full_data_0.json` and `job_request.json` match key for key, and
`summary_confidences_0.json` carries nine of its ten. The tenth, `has_clash`, is
omitted because it is a claim about geometry nothing here computes. The
structure is `.pdb` where the server writes `.cif`, which is the one deliberate
difference.

🔴 **AND "DOWNLOAD ALL" IS THE ONE PATH NOTHING RAN, SO EVERY FIELD A FOLD
FORGOT TO STORE FAILED THERE AND NOWHERE ELSE.** It builds the archive out of
`lastPrediction`, and EF2-fast stores no `confidence` object at all - on purpose,
since an object of zeros would be read as the model's opinion. The archive
recovered its TOKEN COUNT from `confidence.predictedAlignedError.length`, so the
button reported *"Cannot read properties of undefined (reading 'length')"*,
which names neither the model nor the field. `tools/fold-in-page.py --download`
presses it, reads the zip BACK through `web/zip.js` and prints its members and
`full_data`'s keys - because "a zip was written" is not "the fold's numbers are
in it".

🔴 **AND THE CONTACT MAP WAS IN THREE PLACES, OF WHICH THE ARCHIVE KNEW TWO.**
AF3 hands it back with its confidence, AF2 fills it in from a `setTimeout` off
the saved pass (its distogram head costs 131 ms at 128 residues and is
deliberately off the critical path), and EF2-fast kept it beside the prediction
in a field of its own - so the model whose contact map is its ONLY score wrote
an archive without one while the panel on screen showed it. `contactSource` is
now the single field on every path, and it holds the OBJECT rather than a copy,
which is what lets AF2's arrive late.

🔴 **AND A README THAT DESCRIBES A CONTROL THE MODEL DOES NOT HAVE IS WRONG, NOT
MERELY VERBOSE.** EF2-fast's said `recycles: 1` and `max msa: 128` - the shared
dials' values, reported as though they had been used. It reads neither: it folds
from the sequence alone, and its trunk loops a number of times the CHECKPOINT
fixes (four) rather than the dial. `msaOrigin` undefined now means "this model
does not take one", which drops the alignment line AND the paragraph describing
a `msas/` directory the archive does not contain.

🔴 **AND `atom_plddts` MUST NOT CARRY SOMETHING THAT IS NOT A pLDDT.** The
B-factor column is whatever the model put there, and for EF2-fast that is the
distogram certainty under a REMARK naming it - the page is careful that the word
pLDDT appears nowhere for that model, and the key would have undone it in the
one file a reader is most likely to parse. It is `atom_certainty` there, and the
PAE is omitted rather than written from an absent matrix.

🔴 **AND `chain_pair_max_contact` IS THE ONE SCORE A MODEL WITH NO CONFIDENCE
HEAD CAN STILL GIVE.** AlphaFold 3 splits intra- from cross-chain too, but only
through ipTM - `iptm_ichain` and `iptm_xchain` in its own code - and through the
contact-weighted PDE summaries in `confidences.py`, all of which need the
confidence head. The strongest predicted contact needs only the TRUNK: the
diagonal says how sure the model is that a chain touches itself at range, and an
off-diagonal entry says whether it believes in the interface at all. It is not a
server field; it is an addition, and it is the reason EF2-fast's summary file is
worth writing. Measured on a two-chain fold, AF3 reports `chain_pair_max_contact`
0.73 across the interface while its `iptm` is 0.23 - the distogram believing in
a contact more than the confidence head believes in the interface.

🔴 **AND SEQUENCE NEIGHBOURS ARE EXCLUDED OR THE DIAGONAL IS ALWAYS 1.00.** A
token is in contact with itself and with the residue beside it whatever the
fold. The rule is the one used everywhere else here - the same chain and within
six RESIDUES, which drops a ligand's whole self-block. A chain shorter than the
separation reports **null**, not zero: "no pair to measure" is not "the model is
sure this does not fold".

🔴 **AND THE ALIGNMENT ROUND TRIP WAS BROKEN BEFORE IT, IN A WAY THE PAGE
ADMITTED IN A COMMENT.** "A pasted or uploaded A3M is one text and cannot be
split into blocks; it becomes the unpaired one." AF3 reads the paired block
first and takes its profile over the UNPAIRED one alone, so downloading an
alignment and uploading it again folded something else, silently. The archive
carries one a3m per chain per block; `msasFromArchive` feeds them back through
`mergeSearchedChains`, the same call the search path makes.
`tools/archive-roundtrip.py` folds, downloads, re-uploads and folds again -
and asserts on **"trunk reused"**, because the trunk cache key hashes the
alignment blocks, so reuse is the page saying the restored blocks are
bit-identical to the searched ones. A matching structure alone would be weaker.

🔴 **AND THE SCORE KEYS ARE ASYM IDS, NOT CHAIN INDICES.** AF3 numbers chains
from ONE (`featurise.js` writes `identity.asymId + 1`); AF2 uses contiguous
blocks from zero. Reading the keys as indices gave a real two-chain fold
`chain_pair_iptm: [[null, null], [null, null]]` and `chain_ptm: [null, 0.69]` -
"1|2" matching nothing and "1" matching the second chain by accident. **Every
unit test passed**, because they were all written with 0-based keys. The
archive sorts the ids it finds and takes the nth as the nth chain.

🔴 **AND `fold-in-page.py` CAN DRIVE A COMPLEX NOW** - one entity per chain
through `window.__entityList`, instead of typing a colon-joined sequence into
one field and waiting out the timeout. A 108-residue two-chain fold with
`--msa-mode search` takes about 6 s. **A 199-residue one (barnase/barstar with
10,839 hits) sat at "Trunk · 1%" for twenty minutes and did not finish** - not
diagnosed, but it is the shape to avoid in a quick loop.

🔴 **py2Dmol IS A MIRROR NOW, WITH A COMMIT ON IT.** The four vendored files -
`py2Dmol.app.css`, `py2Dmol.align.js`, `py2Dmol.embed.min.js`,
`py2Dmol.full.min.js` - had been copied by hand and nothing recorded from
where, so "is this current?" could only be answered by diffing 800 KB of
minified JavaScript against a build. `python3 tools/sync-py2dmol.py` runs
upstream's own `tools/bundle.py build`, copies the four, and stamps the commit
and each file's hash into `web/vendor/SOURCE.md`; `--check` says whether the
mirror has drifted. Never edit the mirror.

🔴 **AND BOTH BUNDLES ARE NEEDED, WHICH THE SIZES HIDE.** `full` is the website
plus the embed API and `embed` is the embed API alone, so `full` looks like a
superset and is nearly one - but `index.html` loads `full` while `single.html`
and `proteinhunter.html` load `embed`. Syncing only the larger leaves two of
the three pages on a stale viewer.


## Saving the session, which is py2Dmol's and not ours

🔴 **py2Dmol ALREADY SERIALISES A SESSION, AND WRITING A SECOND FORMAT WOULD
HAVE RESTORED LESS.** `buildViewerState` produces exactly what its Save button
writes to a `.py2dmol.json` - every frame, the camera, the colour mode, the
style, the side chains, the PAE, every heatmap, the MSA - and `loadViewerState`
has always been the reader for a dropped one. A first attempt stored a fold
ARCHIVE instead and restored **one frame** where this restores **sixteen**: the
archive holds the answer, not the trajectory.

Both were reachable only through a file - the builder downloaded its result and
the loader was not exported - so the change upstream is to split the two
(`py2Dmol 6ee50b8`).

🔴 **AND THE EXPORT MUST BE THE FUNCTION, NOT AN ARROW THAT CALLS IT BY NAME.**
The bundle is concatenated rather than module-scoped, so a top-level
`function buildViewerState` IS `window.buildViewerState`; assigning
`window.buildViewerState = () => buildViewerState()` replaces the global with an
arrow whose body resolves to the arrow. It recursed until the stack ended, and
the only symptom was `RangeError: Maximum call stack size exceeded` from a
function that reads correctly. The stack - the same frame nine times - is what
named it.

🔴 **WHAT py2Dmol DOES NOT CARRY IS THE JOB, AND THAT HALF IS OURS.** Its frames
know coordinates and maps; nothing in them says which model ran, against which
sequence, with which alignment, or what the confidence head said. That goes
under a `localfold` key, which the loader ignores and a round trip preserves.
Without it the structure comes back with a blank score card, because
`updateScoresCard` hides its box outright when handed undefined.

🔴 **AND THE SESSION IS SAVED WHEN THE READER LEAVES, NOT WHEN THE FOLD ENDS.**
Measured: at the moment a fold completes the viewer object holds ONE frame -
`framesAtSave: 1` against the sixteen it ends with - because the trajectory
lands in it after the prediction is stored, and AF2's contact map arrives later
still in a `setTimeout` off the finished pass. Saving at completion captured a
session that was not yet the one on screen, and every fix for that is a guessed
delay. `visibilitychange` needs no guess: whatever is on screen when the tab is
hidden IS the session, with the camera and colour mode the reader chose. The
save at completion stays as a floor. With that, the record went from 26,641
bytes and 1 frame to **188,767 bytes and 16**, contact map included.

🔴 **AND `loadViewerState` RESOLVES BEFORE IT IS FINISHED.** Its last act is a
`setTimeout(..., 100)` that picks the current object and syncs the heatmap, so
`refreshHeatmap` straight after the await runs while `currentObjectName` is
still unset and returns at its first guard. The restore waits for the condition
rather than sleeping on it.

### What is verified, and what is not

Gated by `tools/fold-in-page.py --session`, which folds, drives
`visibilitychange`, reads the record out of the real IndexedDB, **reloads**, and
restores:

| | |
|---|---|
| saved | 16 frames, PAE and maps on them, camera, `colorMode: plddt`, 188,767 B |
| job | stem, model, 58 residues, pLDDT 71.18, the MSA origin |
| offer | `Last fold: AlphaFold 3 · 58 residues · pLDDT 71.2 · just now` |
| restored | 16 frames, `pae: 58`, `contact: true`, score card 71.2 / 0.39 |

🔴 **AND SAVING AT FOLD COMPLETION WAS STILL WRONG, WHICH THE GATE HID BY
DRIVING THE EVENT.** `visibilitychange` catches a settled session, but only for
a reader who hides the tab; press reload straight after a fold and that signal
never comes, so the only save that ran was the floor at completion - one frame
of sixteen, no contact map. The gate drove a `visibilitychange` before
reloading and was green throughout. It now reloads the way a reader does, and
`--session-hidden` is the other arm. The save waits for the FRAME COUNT TO STOP
GROWING - six still checks - rather than for a guessed interval, and AF2's
contact map keeps its own re-save where it arrives.

| | reload, no hide (before) | after |
|---|---|---|
| frames | 1 | **16** |
| maps on frames | none | **pae + contact** |
| bytes | 26,641 | 188,767 |

🔴 **AND THE HEATMAP PANEL WAS HIDDEN BY A MULTI-MODE GUARD, ON A VIEWER
SHOWING ONE OBJECT.** `heatmapObjectName` returned null for ANY `shownObjects`
Set, and a restored session has a Set of exactly one - so the panel had no
object to describe. Its own rule is "the matrix belongs to one, so it waits
until the viewer is back to one", and a one-element set is back to one; the
guard made the panel depend on HOW the viewer arrived at one object rather than
on whether it is showing one. Fixed in `py2Dmol db883b6`.

🔴 **AND IT FAILED ONLY ON A RESTORE, WHICH IS WHAT MADE IT HARD TO SEE.**
`updateVisibility` runs when the panel is asked to change, so a live page that
showed the panel while the set was still null keeps it on screen once the set
becomes a Set - visible by inertia, with nothing to recompute it. Loading a
session recomputes from nothing, and the panel that had been on screen all
along did not come back. Everything measurable said the data was fine -
`hasData: true`, `mapKeysOf: ['pae','contact']`, frame 0 holding a well-formed
`{data, n: 58}` and a 58-wide `pae` - and `heatmapRenderer.maps` was empty
because `heatmapObjectName` had already answered null. `heatName` is what named
it.

Restored now: `panelShown: true`, `panelTabs: ['pae','contact']`,
`heatName: 'af3_1'`, `rendererMaps: ['pae','contact']`. The old open note:
it blamed `resolveMapFrame`'s backward search, which was wrong - the search was
never reached. `heatmapObjectName` had returned null before it, so `_show` had
no object at all. Reasoning down the call chain named the wrong function; the
probe that printed `heatName` named the right one in a single run.

### The structure travels, and the record is gzipped

🔴 **BOTH DOWNLOAD BUTTONS WERE ON SCREEN AND BROKEN AFTER A RESTORE.**
py2Dmol's session carries coordinates, element symbols and residue numbers -
enough to DRAW a fold and not the text the fold produced - and both buttons
read `prediction.pdb`. "PDB" wrote the word `undefined` into a file and "All"
threw inside the archive builder. Rebuilding the text from the frames would be
a second PDB writer to keep in step with the first, so the record carries the
one the fold wrote, along with everything `buildFoldArchive` reads: the token
layout, the confidences, the PAE and the contact map.

🔴 **AND THE MATRICES GO IN AS PLAIN ARRAYS, BECAUSE THE RECORD IS JSON.** A
Float32Array survives `structuredClone` and does not survive `JSON.stringify` -
it returns as `{"0":1.2,...}`, an object with numeric keys that every reader
here treats as a matrix of undefined. Converted going in and typed coming out,
so there is one shape to restore rather than two to tell apart.

🔴 **AND A RESTORED SESSION HAS NO ALIGNMENT TO INCLUDE, WHATEVER THE BUTTON
ASKS FOR.** "Download all" asks for one because a live fold has one; the README
still has to say the archive does not carry it rather than describe an `msas/`
that is absent. `archiveFor` forces `alignmentOmitted` on a restored
prediction - measured, `readmeOmits: true`.

🔴 **GZIP IS WORTH 3.5x AND IS IN THE BROWSER ALREADY.** `CompressionStream`,
measured on a real session: 358,152 bytes of JSON become **103,372**. The
payload is rounded decimal coordinates repeated over every frame of a
trajectory, and it grows with both the chain length and the sampler's step
count - py2Dmol's own note records a 212 MB session for a 305,004-position
structure, so this is not a small-case optimisation. The record went from
188,767 bytes holding the answer alone to 103,372 holding the structure, the
matrices and the whole trajectory. A stored `Uint8Array` also skips IndexedDB's
structured clone of a deep object graph. Where `CompressionStream` is missing
the object is stored as it is, and the reader tells the two apart by TYPE
rather than by a flag that could disagree with the bytes.

Measured after a reload and restore, with both buttons actually pressed:

| | |
|---|---|
| PDB | 36,896 B, 467 ATOM records, no `undefined` |
| All | 21,589 B zip, five members, `full_data` with `contact_probs` and `pae` |
| README | says the alignment is not in this archive |
| panel | visible, tabs `['pae','contact']` |

### The matrices are read back out of the frames, not stored twice

🔴 **py2Dmol'S SESSION ALREADY HOLDS THE PAE AND THE CONTACT MAP**, so keeping
float copies beside them wrote every pair twice - and n^2 is the term that
grows fastest with chain length, which makes this the copy worth not making.
Both invert exactly enough:

| | how it is stored | recovered to | what the archive writes |
|---|---|---|---|
| PAE | float rows, rounded to 1 dp by the session writer | 0.05 A | 2 dp - loses one digit |
| contact | bytes, `round(p * 255)`, vmin 0 vmax 1 | 0.004 | 2 dp - **no practical loss** |
| pLDDT | `frame.plddts`, rounded to integers | 1 | feeds only `fraction_disordered`, threshold 50 |

The per-ATOM pLDDTs the archive writes come off the stored PDB's B-factor
column rather than from any of this, so they are unrounded.

🔴 **AND THE DECODE USES THE MAP'S OWN BOUNDS, NOT A CONSTANT.** `contactMapFor`
writes vmin 0 / vmax 1 and `paeMapFor` writes vmin 0 / vmax 32 - quantised
against a fixed range rather than its own, so two folds are comparable - and
`mapsOfFrame` normalises every producer to the same `{data, n, vmin, vmax}`.
Reading the bounds from the entry is what keeps this right for a map some other
path encoded differently; a hard-coded 255 would fill the key with plausible
nonsense instead of failing.

🔴 **AND THE FIRST FRAME THAT HAS ONE WINS.** A trajectory carries coordinates
on every frame and a contact map on one - AF3 attaches it to `flow_0` - so a
search that looked only at the frame on screen would find nothing on the
fifteenth.

Measured, on the archive a restored session writes:

| | |
|---|---|
| `pae` | diagonal **0.8**, off-diagonal **19.9**, max 23 |
| `contact_probs` | diagonal **1**, range 0-1 |
| `atom_plddts` | 63.23 to 84.24, against the fold's 71.2 mean |

Present is not correct, which is why these are values and not key names: a
decode against the wrong bounds fills the file with a plausible matrix.

The record, over three shapes of the same session:

| | raw | stored |
|---|---|---|
| the answer alone, uncompressed | 188,767 | 188,767 |
| structure and matrices, gzipped | 358,152 | 103,372 |
| structure, matrices rebuilt, gzipped | 226,641 | **51,968** |

### The three models, and the one that caught a regression

🔴 **A MODEL WITH NO CONFIDENCE HEAD STILL HAS A CONTACT MAP, AND IT IS ITS
ONLY SCORE.** The restore collapsed its whole confidence object to undefined
whenever the summary was absent - which threw away the matrices just recovered
from the frames, so EF2-fast's restored archive lost `contact_probs` AND its
`_summary_confidences_0.json` entirely, the one file `chain_pair_max_contact`
lives in. The summary being absent says nothing about the maps. Caught by
running the gate on all three models rather than on AF3 alone, which is this
repository's recurring lesson in a new place.

🔴 **AND pLDDT IS ATTACHED ONLY WHERE THERE IS ONE.** `fullDataJson` chooses
`atom_plddts` over `atom_certainty` on exactly that field's presence, so
handing it EF2-fast's B-factors - a distogram certainty, under a REMARK saying
so - would label them as the model's pLDDT in the file a reader is most likely
to parse. Restored EF2-fast still writes `atom_certainty`, and its score card
stays hidden rather than drawing dashes.

| after a reload and restore | AF3 | AF2-mono | EF2-fast |
|---|---|---|---|
| frames | 16 | 2 | 11 |
| panel tabs | pae, contact | pae, contact | contact |
| score card | 71.2 / 0.39 | 61.9 / 0.32 | **hidden** |
| PDB | 36,896 B, 467 atoms | 36,862 B, 466 | 36,940 B, 466 |
| archive | 5 members | 5 | 5, `atom_certainty` |
| stored | 51,968 B | 31,255 B | 36,429 B |
| gzip ratio | 4.36 | 3.42 | 6.53 |

The saved-session row is measured at phone widths with its text forced on, for
the same reason the download dial's label is - `hidden` until there is a
session to offer, and a box that is not laid out cannot overflow. At 320px the
row is 254px with the text at 79 and Restore's right edge at 205 against the
row's 287: `overflows: false`. `mobile-layout.py` names `#session` as refusing
to shrink below 575px, which is the same min-content false positive the tool
reports for any `white-space: nowrap` text - what decides is the row against
the box, which is why it is measured separately.

**Not yet exercised through a session:** a multi-chain or ligand fold (the
`chainLengths` and `tokens` round trip), OpenDDE and OpenBind-0, and anything
long enough for the n^2 matrices and the frame count to matter together.

### A complex and a ligand through a session

🔴 **A LIGAND IS THE CASE THE ARCHIVE REFUSES TO GUESS.** It is one token per
heavy atom, so a fold with one has MORE TOKENS THAN RESIDUES, and
`tokenIdentifiers` throws rather than numbering them - "this fold's token
layout must be passed in, not inferred". That makes it the sharpest test of a
session carrying `tokens` back: 58 residues plus GOL is **64 tokens**, the
restored PAE is 64 wide, and the archive is written rather than refused. Had
`tokens` been lost, the token count would still have come out as 64 from the
PAE's own length and the builder would have thrown against 58 residues.
`fold-in-page.py --ligand GOL` drives it.

🔴 **AND THE COMPLEX EXERCISES THE KEYING THAT WAS ONCE WRONG.** Per-chain
scores are keyed by ASYM ID, which AlphaFold 3 numbers from one and AlphaFold 2
from zero - read as indices they produced `chain_pair_iptm` all null while
every unit test passed. Measured through a saved session, on a 58 + 76 fold:

| | |
|---|---|
| chains | `[58, 76]`, 134 residues |
| token layout | 134 tokens, chains `['A','B']`, last residue id **76** - numbering restarts per chain |
| `chain_ptm` | `[0.46, 0.54]` |
| `chain_pair_max_contact` | `[[0.61, 0.2], [0.2, 1]]` |
| `pae` | diagonal 0.8, off-diagonal 19.3, max 28.9 |
| PDB | 84,458 B, 1,069 atoms |

**Still not exercised through a session:** OpenDDE and OpenBind-0, and anything
long enough for the n^2 matrices and the frame count to matter together.

### Templates travel; the alignment does not

🔴 **A TEMPLATE IS INPUT, AND LOSING IT DESCRIBED A DIFFERENT JOB.** The record
did not carry `templates`, so a restored prediction had `templates: undefined` -
and by this file's own rule an ABSENT array means "this model has no such
control", which is what drops the README's templates line entirely. An empty
array means "none were used". Neither means "there were some and they are
gone", which is what had happened. The archive also lost the template
structures themselves.

They travel now. Unlike the alignment they are small - a structure or three
rather than a 3 MB a3m - and unlike the alignment they are a CHOSEN input: a
fold that quietly forgot which template it was given is a different job from
the one that ran, where a re-searched MSA is at least the same question asked
again. `text` and `chain` are what the archive writes and `source` names the
hit; `origin` is dropped, because it carries the live fetch's status and that
request is long finished.

Measured, `--template 1QYS_A` through a save and reload:

| | |
|---|---|
| archive member | `templates/af3_1_template_hit_0_chains_a.pdb` |
| README | `- templates: 1 used` |
| session | 310,669 B raw, **70,458** gzipped |

against 226,641 / 51,968 for the same fold with no template - so a template
costs about 18 KB stored.

### Every model and shape now through a session

| | frames | panel | card | archive |
|---|---|---|---|---|
| AF3 | 16 | pae, contact | 71.2 / 0.39 | 5 members |
| AF2-monomer | 2 | pae, contact | 61.9 / 0.32 | 5 |
| AF2-multimer, 2 chains | 2 | pae, contact | 45.1 / 0.28 | 5 |
| EF2-fast | 11 | contact | **hidden** | 5, `atom_certainty` |
| OpenBind-0 | 16 | pae, contact | 61.3 / 0.34 | 5 |
| OpenDDE | 16 | pae, contact | 83.6 / **no pTM** | 5 |
| complex 58+76 | 16 | pae, contact | 50.3 / 0.33 | 5, `chain_ptm` [0.46, 0.54] |
| ligand GOL | 16 | pae, contact | 75.7 / 0.48 | 5, **64 tokens** for 58 residues |
| template 1QYS_A | 16 | pae, contact | 67.9 / 0.38 | 6, with `templates/` |

And the state machine: folding again after a restore works - the button is
live, `uniqueStem` gives `af3_1_2` rather than colliding with the restored
`af3_1`, the record is overwritten with the new fold, and the offer row is
re-asked so it does not go on advertising the old one.

### State, for whoever picks up the saved session

**Where it lives.** `web/fold-session.js` is the store, `rememberSessionWhenSettled`
/ `offerSession` / `restoreSession` in `web/app.js` are the three verbs, and the
row is `#session` in `index.html`, under the status line and outside
`#viewer-container` on purpose. Upstream: `py2Dmol 6ee50b8` split
`buildViewerState` from the download and exported both; `py2Dmol db883b6` fixed
the Multi guard that hid the heatmap panel.

**The gate** is `python3 tools/fold-in-page.py --model <m> --session`. It folds,
reads the record out of the real IndexedDB, RELOADS, restores, presses both
download buttons and reads the zip back, then folds something else to check the
offer does not go stale. `--session-hidden` drives a `visibilitychange` first,
the other save signal. `--ligand GOL` and `--template 1QYS_A` are the two input
shapes worth re-running after any change here.

🔴 **THE GATE MUST RELOAD THE WAY A READER DOES.** It drove its own
`visibilitychange` before reloading once, and was green for two rounds while a
plain reload restored one frame of sixteen. If a change here needs the gate
adjusted, check first whether the adjustment is the bug.

**Numbers to compare against** - AF3, the page's default 58-mer, no MSA:

| | |
|---|---|
| session | 226,641 B raw, **51,968** gzipped, 16 frames |
| restore | 16 frames, `pae: 58`, contact, card 71.2 / 0.39, panel `['pae','contact']` |
| PDB button | 36,896 B, 467 ATOM records |
| All button | ~19 KB zip, 5 members, README saying the alignment is not included |
| 152 residues | 723,105 B raw, 166,223 gzipped |

**Open, in order:**

1. ~~**Modified residues**~~ - done, and it found a bug that was not in the
   session at all. `fold-in-page.py --modify SEP@3` sets the modification on the
   entity the way the row's `⋮` popup does, and the round trip is clean: 67
   tokens for 58 residues, PAE 67 wide, `token_res_ids` ending at 58, both
   download buttons working. But **`job_request.json` did not name the
   modification**, on a live fold as much as a restored one - the request is the
   file a reader hands back to reproduce a job, and one listing the parent
   sequence alone describes a different fold. Silently, because a modified
   residue changes no residue COUNT: `SEP3` shows in the status line and nowhere
   in the archive. Now written in the server's own dialect,
   `{ptmType: "CCD_SEP", ptmPosition: 3}`, and ABSENT rather than `[]` on an
   unmodified chain, since an empty array claims the chain was checked. The gate
   reads it back off the zip.

   The same argument reached the offer row, which said "58 residues" for a fold
   whose parent would say exactly the same thing - so it names the ligands and
   the modifications now, capped at three, the way the status line does:
   `Last fold: AlphaFold 3 · 58 residues + SEP3 · pLDDT 72.1 · just now`.
2. **A quota-exhausted save.** `saveSession` returns `"quota"` and the page says
   so, and that branch has never run.
3. **The no-`CompressionStream` fallback.** `pack`/`unpack` decide by TYPE, so an
   uncompressed record still reads; untested in a browser that lacks it.
4. **Nothing longer than 152 residues** has been saved. The n^2 matrices and the
   frame count compound; a 500-mer is expected around 1-2 MB gzipped.

**Not deployed.** Nine commits here and two in py2Dmol are unpushed as of this
note.

## The job JSON, read as well as written

The archive has written a `*_job_request.json` since it existed, and its README
told the reader to drop the .zip back on the page "to fold again with exactly
these alignments". That restored the **alignment and nothing else**: the
sequence, the copies, the ligands, the modified residues and the seed all had
to be retyped out of the request file by hand. A format written in one place
and read in none drifts, which is how the templates and then the modifications
came to reach the fold and not the file, twice in a week.

`web/job-json.js` now holds both halves - `jobRequestJson` moved there out of
`fold-archive.js`, which re-exports it - and `jobFromJson` reads a job back.

🔴 **"THE AlphaFold 3 FORMAT" IS TWO FORMATS.** They differ in every field that
matters, and reading one as the other is silent rather than loud:

| | server dialect | open-source dialect |
|---|---|---|
| marker | `dialect: "alphafoldserver"`, `version: 3` | `dialect: "alphafold3"` or absent, `version` 1-4 |
| seeds | `["7"]`, strings | `[7]`, integers |
| a chain | `proteinChain: {sequence, count}` | `protein: {id: "A", sequence}` |
| copies | `count` | the LENGTH of an `id` list |
| a template | `useStructureTemplate: true`, and nothing about which | `templates: [{mmcif, queryIndices, templateIndices}]` |
| an alignment | not expressible | `unpairedMsa` / `pairedMsa`, inline or by path |

Both are read. **Only the server one is written**, because the archive's whole
justification is matching `tools/fixtures/fold_2026_09_01_10_17.zip` file for
file - trading that for a marginal gain is a bad trade.

### What it refuses, and why refusing is the point

Every unsupported field parses perfectly well as far as the sequence, so a
reader that read past it would fold a real structure of the right protein
**without the inhibitor bonded to it**, or with unmethylated DNA, and report it
as the job that was asked for. So each is refused by name:
`bondedAtomPairs`, `userCCD`, a `smiles` ligand, `ccdCodes` with several
components (that is one bonded chain, not several ligands), `unpairedMsaPath`,
an inline `unpairedMsa`, `queryIndices`/`templateIndices` (they set the
template's residue mapping and this page computes its own), modified bases, and
a sequence-entry key the page has never heard of.

🔴 **AN EMPTY `unpairedMsa` IS AN INSTRUCTION, NOT AN ABSENT FIELD.** AlphaFold
3 reads `""` as "fold this chain with no alignment" and an absent field as "go
and search" - opposite jobs, several minutes apart. `""` sets the MSA dial to
none and says so in the status line.

### AlphaFold 3's own examples are the corpus

🔴 **AND THE REFERENCE ARCHIVE ITSELF IS FED BACK NOW.**
`tools/fixtures/fold_2026_09_01_10_17.zip` is the real AlphaFold Server export
this whole format was read off, and until this gate nothing ever handed it to
the page - so the reader was only ever checked against the archive we write,
which shares its source. Dropped on the upload box it comes back as
`archive · 2 chains, 2 with paired rows · 2 chains · seed 819505351`, with rows
`protein:146x1` and `protein:74x1`. That is a file DeepMind wrote, restoring
both the job and four alignment blocks.

`tools/fixtures/af3-jobs/` holds all thirteen `examples/*.json` from
google-deepmind/alphafold3 plus its kitchen-sink `alphafold_input.json`,
vendored under Apache 2.0, and `test/af3-example-jobs.test.js` runs the reader
over every one. **These are the only inputs here we did not write** - the rest
of `test/job-json.test.js` checks the reader against our own reading of the
spec, which is the same mistake the archive made from the writing side. They
carry things our fixtures did not think to: `version: 4`, an `id` LIST standing
for four calcium ions, `description` keys inside a chain body, and
`modificationType`/`basePosition` where a protein says `ptmType`.

**Eight of the fourteen load. Six do not**, and the split is the roadmap:

| | files | why |
|---|---|---|
| loads | 8 | complexes, homodimers by `id` list, ions, CCD ligands, protein PTMs, DNA and RNA chains |
| bonded chemistry | 3 | `bondedAtomPairs` - a covalent inhibitor, a glycan, the kitchen sink |
| modified bases | 2 | `5CM` on DNA, `PSU`/`5MC`/`OMG` on RNA |
| SMILES | 1 | a ligand named by structure rather than code |

All fourteen behaved as predicted on the first run, including ERK2's `TPO@185`
and `PTR@187` passing the page's own parent-residue validator against a real
360-residue sequence - an independent check of `modificationProblem` that
nothing else here provided.

### The gate

    python3 tools/fold-in-page.py --model af3 --modify SEP@3 --ligand GOL \
      --job-round-trip

🔴 **IT WIPES THE ENTITY ROWS BEFORE DROPPING THE ARCHIVE BACK.** The rows are
still on screen from the fold that just ran, so "they match afterwards" is true
of a page that read nothing at all. It sets them to `AAAAAAAA` and the seed to
999 first, then drops the zip and compares. Measured, all green:

| arm | result |
|---|---|
| archive round trip | 26,079 B zip; sequence, `SEP@3`, `GOL`, copies and seed all back |
| open dialect, hand written | `2 chains + 1 ligand · seed 1234 · MSA off`, dial actually moved |
| a refusal | `bondedAtomPairs describes chemistry this page does not build`, rows unchanged |
| one of AF3's own example files | dropped on the real input, not passed to the reader |

Compared on the fields the job file can carry, not on the whole row: a restored
row has no `template.origin` and no coverage status, which are discovered when
a template is FETCHED and were never part of the job.

🔴 **AND THE SERVER DIALECT LOSES WHICH TEMPLATE.** `1QYS_A` goes in and
"search for a template" comes back, because `useStructureTemplate` is a boolean.
That is a real loss, asserted in `test/job-json.test.js` rather than left to be
discovered by somebody whose re-fold used a different structure than the one
they picked. The open-source dialect could carry it; writing that one would
cost the archive's file-for-file claim.

### Checked against AlphaFold 3's parser, not against its documentation

The writer was diffed against `tools/fixtures/fold_2026_09_01_10_17.zip` and
against `src/alphafold3/common/folding_input.py` - **the code that actually
reads these files**. The writer came out clean:

| written | upstream says |
|---|---|
| top-level `name, modelSeeds, sequences, dialect, version` | exactly its allowed set, and `dialect`+`version` must both be present or both absent |
| `modelSeeds: ["42"]` | `int(seed)` over the list, so strings are right |
| `count` on a chain | how it expands copies |
| `useStructureTemplate: false` | read as "use no templates" - a meaningful false, not noise |
| `ptmType: "CCD_SEP"` | `mod['ptmType'].removeprefix('CCD_')` |
| `full_data_0.json` keys | identical set and order to the reference archive's |

🔴 **WITH ONE EXCEPTION, AND IT IS UPSTREAM'S OWN SPLIT RATHER THAN OURS.**
`folding_input.py` sets `ALPHAFOLDSERVER_JSON_VERSION = 1` and RAISES on
anything else - while the real AlphaFold Server stamps `"version": 3` on the
archive it hands you, as `tools/fixtures/fold_2026_09_01_10_17.zip` does. **So
the reference parser refuses the reference archive**, and it refuses ours for
the same reason and the same value. The writer stays at 3: this archive's whole
justification is being the server's file for file, and a job request nobody
else writes is worth less than one the pipeline needs a version bump to read.
Worth knowing before somebody hands `_job_request.json` to `run_alphafold.py`
and reads "unsupported version: 3, expected 1" as our bug.

The reader takes both numberings - server 1 and 3, open-source 1 through 4,
which is upstream's own `JSON_VERSIONS` - and refuses anything else by number,
because a later version may give a field we already read a different meaning.
It also follows upstream's both-or-neither rule for `dialect` and `version`,
and its **absent-means-the-server's** default, which is not the obvious one:
defaulting the other way put such a file under the wrong version table.

🔴 **AND THAT RULE CAUGHT A FLAW IN OUR OWN TESTS.** The `open()` helper in
`test/job-json.test.js` wrote `version` with no `dialect` beside it and leaned
on our default - so every open-dialect case below it had been written against a
file AlphaFold 3 itself would refuse. All fourteen of its example files carry
both fields; the helper does now too.

🔴 **AND THE SAME READING FOUND THREE BUGS IN THE READER, NONE OF WHICH THE
EXAMPLE CORPUS COULD CATCH** - all fourteen of those files are the open-source
dialect, and every one of these is about the server's:

1. **An `ion` entry was refused.** AlphaFold Server spells a magnesium
   `{"ion": {"ion": "MG", "count": 1}}`, and `Ligand.from_alphafoldserver_dict`
   takes `ligand` or `ion` alike. Reading only `ligand` refused every real
   server job with a metal in it - half of what `COMMON_IONS` exists for.
2. **`CCD_ATP` stayed whole.** Upstream does `removeprefix('CCD_')`; kept, it
   is a five-letter code this page would fetch a component for and not find.
3. **`glycans` and `maxTemplateDate` were ignored.** Both are in the server's
   allowed key set and both RAISE upstream. A glycan is chemistry this page does
   not build; a template date changes which template is found, so honouring the
   sequence and dropping the date folds a different job.

Unknown keys are refused now, per entry kind, with the allowed sets copied from
`folding_input.py` - upstream calls `_validate_keys` and raises, so leniency
here would fold a job the reference implementation would not have run.

🔴 **`ligand` IS THE ONE ENTRY KEY BOTH DIALECTS USE** and they mean different
bodies by it - `{"ligand": "GOL", "count": 1}` against
`{"id": "B", "ccdCodes": ["GOL"]}`. Which fields are legal comes from the BODY.
Keyed off the entry name alone, this page's own archive stopped being readable,
which is how the mistake surfaced.

### One stale comment, found by diffing rather than reading

`fold-archive.js`'s header claimed `has_clash` **and** `chain_pair_pae_min`
were both left out as uncomputed. `has_clash` still is. `chain_pair_pae_min`
has been computed and written for some time - it is the minimum over ordered
pairs of a PAE we already have, with the server's own values quoted in the code
beside it. The comment was corrected, and it names the two keys that are ours
rather than the server's: `chain_pair_max_contact` and `mean_plddt`.
