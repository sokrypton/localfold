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
