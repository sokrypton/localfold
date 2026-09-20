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

### 🔴 AND ALPHAFOLD 2's MONOMER TAKES ONE NOW, WHICH IT WAS REFUSED FOR

`chosenFamily`'s guard read "Templates need AF3 or OpenBind-0", and for the
monomer that was right for the wrong reason: its term exists, is
oracle-checked against AF2's own module, and was reachable from nothing,
because `src/af2/model/monomer.js` built its template call from a literal that
named neither `template` nor `useTemplateUnitVector`. See docs/AF2.md. With
the driver forwarding, the page can offer it. Measured, 5CAJ chain A with its
own crystal uploaded, one recycle, single sequence:

| monomer, from the page | pLDDT | pTM |
|---|---:|---:|
| no template | 31.9 | 0.215 |
| self-template, 261/261 | **76.0** | **0.795** |

🔴 **AND `buildTemplate` GAINED A `layout` AND NOTHING ELSE.** Everything the
AF3 path needs it already did - it sniffs PDB against mmCIF, ALIGNS a homolog
to the query, drops low-confidence residues - and a page template is a homolog
or an upload, never the query's own sequence, so `tools/gpu/fold-af2.js`'s
identity map is exactly what does NOT work here. AF2 differs only in ending on
`templateSlotAtom37` rather than the dense-24 builder. **The two are the same
rank and neither throws on the other**, so that one argument is the whole
difference and all of the risk; AF3's arm is verified unmoved through the same
function, pLDDT 95.7 on the identical file.

Three things are refused rather than dropped, which is the rule the rest of
this section is built on:

- **the MULTIMER.** Its driver has forwarded a template since it was written,
  but its embedder is a different dialect and nothing on this page builds a
  slot for it - which is precisely the gap that let the monomer's term look
  supported for a year.
- **`?graph=unified`**, which runs that same multimer graph over a monomer. A
  template there would be ignored, so the fold stops instead.
- **a second template.** `QueryOnlyTemplateGpu` reads `input.template`,
  singular: AF3 runs a forward per slot and averages the outputs, and this one
  does not.

🔴 **AND THE TEMPLATE IS IN `af2Key`.** That key's own comment calls itself
everything a pass reads, and a trunk cached from a fold WITHOUT the template is
not this fold's trunk - so raising the recycle count would have continued from
the wrong state. The SOURCE is hashed rather than the slot: the slot is
megabytes of float and the text plus the chain decides every one of them.

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

🔴 **AND A LIGAND IS A CHAIN THE SCORES COUNT AND `chainLengths` DOES NOT.**
`web/app.js` builds `chainLengths` as `chains.map((chain) => chain.length)` -
the POLYMER rows - while the confidence scores are keyed by asym id, and
AlphaFold 3 gives a ligand one of its own. So a 58-residue protein folded with
GOL reached `summaryConfidencesJson` as ONE chain against TWO scored asym ids,
`asymOrder` found the counts disagreed, fell back to the plain index, and every
per-chain lookup missed. Measured on the page: `chain_ptm: [null]` and
`chain_pair_max_contact: [[0.73]]` for a fold whose own `chain_ids` names A and
B - so the protein's pTM was written as a null and the protein-ligand contact,
the one number somebody folding with a ligand is looking for, was never written
at all. 🔴 **AND THE SCALARS BESIDE THEM WERE RIGHT**, `ptm` 0.43 and
`mean_plddt` 72.61, which is why nothing looked wrong. The chain list comes off
the TOKEN chain ids now, which is what the rest of that file already writes:
`chain_ptm: [0.44, 0.87]`, `chain_pair_max_contact: [[0.73, 0.17], [0.17,
null]]`. The B diagonal stays null by design - a ligand's atoms share a residue
number, so the within-six-residues rule drops its self-block. Gated by
`test/fold-archive.test.js`, verified to fail first.

🔴 **AND THE CHAIN-PAIR DIAGONAL WAS null WHERE THE SERVER WRITES A NUMBER.**
`chain_pair_iptm[i][i]` is the chain against itself, which is that chain's own
pTM - `tools/fixtures/fold_2026_09_01_10_17.zip` has
`[[0.9, 0.91], [0.91, 0.86]]` beside `chain_ptm` `[0.9, 0.86]` - and this wrote
null there, on the reasoning that an unscored interface must be null rather than
zero. Sound reasoning, applied to the one cell that is not an interface. The
same file read the same convention CORRECTLY one function down, where
`chain_pair_pae_min`'s own test cites the server's real diagonal.

🔴 **AND THIS IS THE LIMIT OF A KEY-FOR-KEY COMPARISON.** The check recorded
above - "nine of ten keys", `has_clash` omitted - passes a wrong VALUE inside a
key that is present, which is what this was for however long it stood. The
fixture is in the repository and was never read back for values. Fixed; the
diagonal carries the chain's own pTM, and stays null only where the fold has no
per-chain pTM at all - absent, not invented, which is the rule everywhere else
in that file. Gated by `test/fold-archive.test.js` against the fixture's own
numbers, verified to fail first.

🔴 **AND AN AF3-GRAPH PDB SAID NOTHING ABOUT ITSELF.** `toPdb` opened with
`const lines = []` and pushed no REMARK, so a file saved from the page named
neither the model that produced it nor the quantity in its B-factor column -
measured as ZERO header lines in a downloaded `af3_1.pdb`, against the
AlphaFold 2 path's `REMARK   1 ALPHAFOLD2 WEBGPU PREDICTION` and EF2-fast's two.
Seven families share that writer, so all seven wrote an anonymous file with a
pLDDT-shaped column nothing labelled. 🔴 **THE HEADER IS A CALLER'S, AND IN TWO
PARTS FOR A REASON**: the page supplies the model - `modelName`, never a
literal, since an OpenBind-0 fold claiming to be AlphaFold 3 is worse than an
anonymous one - and `web/af3-model.js` adds the B-factor line only where
`result.scores.plddt` exists, because a family on this graph with no confidence
head writes zeros there and the word would be the mislabelling EF2-fast's own
REMARK exists to prevent. `toPdb` emits nothing unless asked, which is what
keeps three tests and several tools reading the output they already parse.
`tools/fold-in-page.py --download-pdb` presses the button and reads the bytes
BACK - it is the other half of `--download`, and nothing had ever pressed it.

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


## A token is not a position, and a modified residue is where they part

🔴 **THE PAE WAS ARRANGED WRONGLY ON ANY FOLD CARRYING A MODIFIED AMINO ACID,
AND THE NOTE EXPLAINING WHY IT COULD NOT BE WAS ABOUT LIGANDS.** AF3 scores
TOKENS. A ligand's heavy atoms are one token each AND one position each in the
viewer, so those two spaces really do agree - which is what `paeSize`'s comment
says, and it is why the crop that used to be there was removed. A MODIFIED
residue is the other case: every family but boltz2 atomises it into one token
per ATOM, while py2Dmol draws it as ONE residue.

Measured end to end, with both real implementations and nothing synthetic
between them - `featuriseProtein` on `GWSTELEKHR` with a phosphoserine at
position 3, written out by `toPdb`, loaded into the viewer:

| | |
|---|---|
| tokens the featuriser makes | **19** |
| positions py2Dmol's parser makes | **10**, named GLY,TRP,SEP,THR,... all type P |
| `viewerTokens(batch).length` | **10** |

So nine rows of that matrix addressed nothing and every residue after the
modification read somebody else's. `viewerTokens` / `matrixForViewer` in
`web/prediction-results.js` are the map and the gather, and `onBatch` (added to
`foldAf3`) is how the LIVE contact map gets the same treatment - it arrives per
recycle, long before the batch is returned.

🔴 **AND THE RULE IS THE PARSER'S OWN, NOT "IS IT MODIFIED".** py2Dmol keeps a
residue whole when it carries a BACKBONE - N + CA + C, or C4' + O4' + C1' - and
draws it at the CA or the C4'. Anything else is a ligand to it. Measured, three
modifications inside a six-residue chain:

| written as | positions | type |
|---|---|---|
| SEP, full backbone | 1 | P |
| a bare phosphate: P, O1P, O2P, O3P | **4** | L |
| a ribose-carrying nucleotide | 1 | R |

A rule that collapsed every modified span is three columns wrong on the second
of those - the same fault pointing the other way - which is what the first
version of this did.

**AND THE MODIFICATION IS DRAWN NOW.** A cartoon runs the ribbon through a
phosphoserine's alpha carbon exactly as it runs it through the serine it was
made from, so the phosphate - the reason the residue is in the job - was
invisible. Those residues, and no others, get their side chains shown when the
fold lands. Asked WITH THE OBJECT NAMED, because `positions` index what is
DRAWN: with two folds merged, residue 3 of this one is residue 3 of the first.
Measured - two objects merged, `{object, positions: [2]}` lands on owner 8,
which is that object's offset of 6 plus 2.

🔴 **ESMFold2 WAS OFFERED MODIFIED RESIDUES AND NOT GIVEN THEM, AND NOW IT IS -
WHICH IS HOW WE FOUND THAT IT CANNOT PLACE ONE.** `modelFamily` refuses a
modification for AlphaFold 2 by name - "AF2 tokenises one residue per
letter... folding under it would drop the modification and return a confident
structure of the unmodified chain" - and ACCEPTS one for ESMFold2, whose fold
call then dropped it: a `SEP@3` job came back `GLY,TRP,SER,...`, status "13
res", the exact failure that refusal exists to prevent.

It travels now - the component is fetched as a ligand's is, it goes into the
featuriser's `entities`, into the trunk cache key (a modification changes what
is folded, so a trunk cached for the plain chain is not this fold's), and the
batch reaches the page through `onBatch` so this path's contact map is
collapsed like AF3's. The status line says **13 res + SEP3** and counts
RESIDUES rather than tokens, which read "22 res" for a thirteen-residue chain
the moment one was atomised.

🔴 **AND THE RESIDUE COMES OUT BROKEN, WHICH IS WHY NOTHING WAS DRAWN EVEN
AFTER IT ARRIVED.** py2Dmol builds a side-chain table only from atoms that are
within bonding distance, so a scattered one has nothing to draw. Measured on
GWSTELEKHRSVQ + SEP@3, the same job on both models:

| bond (A) | AF3 | EF2-fast | ideal |
|---|---|---|---|
| CA(2)-CA(3) | 3.91 | 4.45 | 3.8 |
| SEP N-CA | 1.45 | 1.37 | 1.46 |
| SEP CA-C | 1.48 | **1.83** | 1.52 |
| SEP CB-OG | 1.48 | **2.94** | 1.43 |
| SEP OG-P | 1.63 | **4.25** | 1.61 |

🔴 **AND `modifiedAsOneToken` IS NOT THE ANSWER HERE, THOUGH IT LOOKS LIKE
ONE.** It is what fixed exactly this shape for boltz2, and on this fold it
brings CB-OG to 1.38, CA-C to 1.55 and CA(2)-CA(3) to 3.61 - the side chain
then draws - while **OG-P stays at 4.07**, so the phosphate is misplaced
either way. And it contradicts this file's own upstream finding: *"AF3 already
tokenises ... modified residues at one token per atom, and ESMFold2 uses AF3's
all-atom representation term for term"*, with OpenFold3's loss docstring
naming "the first and only atom for anything atomized" as a token's
representative. One fold against a documented convention is not evidence.

🔴 **ANSWERED, ON THE A100: THE MODEL PLACES A LIGAND'S ATOMS AND NOT AN
ATOMISED RESIDUE'S.** Both measurements above, run on
`GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK` (58 residues,
certainty 0.57 rather than the 13-mer's 0.45), scored as a mean bond ratio
against the ideals - 1.000 is perfect, CLAUDE.md's band is 0.70-1.30:

| arm | protein control | the SEP | a glycerol |
|---|---:|---:|---:|
| SEP, preset 15 (11 steps) | 0.994 | 1.399 | - |
| SEP, preset 64 (45 steps) | 1.002 | 1.883 | - |
| SEP, preset 200 (138 steps) | 1.001 | **2.279** | - |
| GOL alone, 138 steps | 1.003 | - | **0.958** |
| GOL **and** SEP, 138 steps | 0.999 | **2.349** | **0.958** |

**The ligand arm is the one that decides it.** In ONE fold, at one setting, the
protein backbone is 0.999, a plain CCD glycerol - atomised the same way, one
token per atom - is **0.958**, and the phosphoserine is **2.349**. So this is
not a model that cannot place atoms; it is an atomised residue INSIDE a polymer
chain specifically.

🔴 **AND MORE STEPS MAKES IT WORSE, WHICH RULES OUT THE SAMPLER'S BUDGET
OUTRIGHT.** 1.399 -> 1.883 -> 2.279 as the steps go 11 -> 45 -> 138, while the
control holds at 1.00 and the glycerol at 0.96. A converging sampler moving
steadily AWAY from the chemistry is a wrong target, not an unfinished walk -
which is the opposite of what a step sweep usually shows and the opposite of
what "the model is not confident here" would predict.

🔴 **AND THE FIRST RUN OF THAT SWEEP MEASURED NOTHING**, which is worth more
than the numbers. Four arms at 15/32/64/200 came back four identical folds -
same certainty, same 2.0 s, same "11 steps" - because `fold-in-page.py` set
`#af3-count` FOURTH, before `msa-mode` and `plm-mode` had fired their change
handlers, and `syncAf3Count` REBUILDS that dial from the chosen model's table
and resets it to the preferred value. Every `--steps` this tool has ever been
given went the same way: `--steps=4` on AlphaFold 3 folded at **25**. It is set
last now, and the `controls:` line it already printed is what shows it took.
An arm that changes nothing is usually an arm that did not run.

🔴 **AND THE FEATURISATION IS RULED OUT, WHICH IS MOST OF THE SEARCH SPACE.**
Four checks, on the batch `featuriseForEsmfold2` actually produces for
`GWSTELEKH` + GOL + SEP@3:

| asked | answer |
|---|---|
| is the SEP's reference conformer right? | **exact** - N-CA 1.469, CA-CB 1.529, CB-OG 1.428, OG-P 1.609, P-O1P 1.480 |
| do its ten atoms share one `refSpaceUid`? | yes (uid 2), as the glycerol's six share uid 9 |
| are its bonds in the matrix, and does the model read them? | yes - `modifiedSpans` is in `bondedGroups`, and `tokenBonds` is uploaded and multiplied by `featuriser/tokenBonds` in the trunk |
| is `molType` right for an atomised residue? | **PROTEIN, and that is correct** |

That last one looked like the bug and is not. `molType` is built from
`ligandSpans` alone, so a modification's ten atom tokens come back PROTEIN -
which is exactly the shape of the bug the comment above that loop records being
fixed *for ligands*. But the reference derives `is_protein` from the **chain
type** (`features.py`: `all_tokens.chain_type == PROTEIN_CHAIN`), not from
whether a token was atomised, so a modified residue inside a polypeptide is
protein by construction. And `molType` reaches only the distogram's contact
classes here, never the structure decoder.

🔴 **WHAT THE COORDINATES SAY: THE BACKBONE LANDS AND THE SIDE CHAIN DOES NOT.**
Superposing the predicted SEP onto its ideal conformer on N, CA, C, O alone:

```
N   -> N   0.57 A      C   -> C   0.74 A
CA  -> CA  0.80 A      CB  -> 2.09 A from CA
                       OG, P, O1P, O2P, O3P: 1.4 - 4.1 A from any ideal position
```

A ligand in the same fold is all "side chain" and is placed correctly, so this
is not the atom decoder being unable to place a rigid group; it is this residue's
side chain specifically.

🔴 **AND ONE CONCLUSION HERE WAS WRONG FOR TEN MINUTES, WHICH IS WORTH THE
SPACE.** Comparing the 45 intra-residue distances SORTED, labels ignored, gives
rms 0.592 A against the labelled 1.892 - which reads as "right shape, wrong
names: a permutation bug". It is not. A sorted-distance distribution is a weak
fingerprint: any compact blob of ten atoms matches another compact blob of ten
atoms to about that, and the backbone superposition above refutes it outright.
**Match the labels before believing a shape.**

🔴 **ANSWERED OUTRIGHT: IT IS THIS PORT, NOT THE MODEL.** The native checkpoint
places the phosphoserine perfectly. `tools/esmc/probe-esmfold2-modified.py`
folds the same 58-mer with the same `SEP@3` through `esm` 3.4.1's own
`ESMFold2InputBuilder` and `EsmFold2ExperimentalModel`, on the same weights this
port reads:

| same input, same checkpoint | control | the SEP |
|---|---:|---:|
| **native ESMFold2** | 1.000 | **0.997** |
| this port | 0.999 | **2.349** |

Both controls are the same 171 backbone bonds of the 57 unmodified residues.
Native's worst SEP bond is 2.6% out; this port's `OG-P` is 2.06-4.25 A against a
1.610 ideal.

🔴 **AND THE TOKENISATION IS NOT THE DIFFERENCE**, which is worth knowing before
anyone goes looking there. The reference reports **67 tokens**, the modified
residue as **10 tokens of one atom each**, all `mol_type` 0 - PROTEIN. This port
produces 67 tokens, ten atom tokens, `molType` PROTEIN. Its own tokeniser's
docstring says so: *"Modified residues (from modifications) are atom-tokenized
(1 token per atom)"*. So the layout agrees, the conformer agrees, the bonds
agree, and the answer is downstream of all of it.

🔴 **AND "THE BOX WOULD NOT FIT IT" WAS STALE.** docs/EF2FAST.md says the EF2
oracle needs torch, which "does not fit on this box" - from when the disk was
down to 2.2 GB. There is **280 GB free**, `~/venv_ef2` has torch 2.7.1+cu126
with CUDA, and both checkpoints are already on disk (`esmfold2-fast-600m` 654
MB, `esmc-600m` 2.2 GB). The comparison cost one probe, not a download.

🔴 **AND THE FIRST NATIVE NUMBER WAS 1.101 AND WRONG, BY THE MISTAKE THIS
SECTION HAD JUST FINISHED WARNING ABOUT.** The atom names came from slicing the
CCD's list to the first ten - which keeps **OXT**, a leaving atom a mid-chain
residue drops - so every name after it shifted by one and `OG-P` was measured
against the phosphorus's neighbour. Read through the reference's own
`get_ccd_leaving_atoms`, the order is `N,CA,CB,OG,C,O,P,O1P,O2P,O3P` and the
mean is 0.997. A near-miss number is the dangerous kind: 1.101 would have read
as "native is imperfect here too" and closed the question the wrong way.

**What is still open**: WHERE in this port's atomised path. The ligand control says the
atom encoder and the sampler can place a rigid group; what differs for a
modification is that its atoms carry a residue's `residueIndex` and share a
chain with polymer tokens. `modifiedAsOneToken` (above) is the boltz2-shaped
fix and was measured as a dead end here.

**The original framing**, kept: is this the MODEL or this port's
atomised path? The measurements that would separate them are EF2 on a plain
CCD ligand (known good - `npm run test:ligand` folds a glycerol) against a
modified residue in the same job, and the same fold at more sampler steps; and
the gate for it is `check-modified-path.mjs`'s shape, which needs a GPU. Note
the fold this was measured on reports certainty 0.45 on a 13-mer, so the model
is not confident here about anything.

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

### Every model on the page, through the round trip

The reader and the offer row are model-independent by construction, which is an
argument and not a measurement. All six were run:

| model | archive | what came back |
|---|---|---|
| AF3, `SEP@3` + `GOL` | 26,079 B | sequence, modification, ligand, copies, seed |
| AF3, two chains | 40,408 B | both chains, in order |
| AF2-mono | 20,112 B | sequence and seed |
| OpenBind-0 | 22,462 B | sequence and seed |
| OpenDDE | 20,737 B | sequence and seed |
| EF2-fast | 13,468 B | sequence and seed |

🔴 **AND EF2-fast WENT THROUGH THE SESSION TOO, BECAUSE IT IS THE SHARP CASE.**
It has no confidence head, and the discipline the archive was taught about
absent scores has to survive a save, a reload and a restore - it would be very
easy for a round trip to put zeros where the model has no opinion. Measured, on
a restored fold:

    offer row   Last fold: EF2-fast · 58 residues · just now
    score box   hidden, cells "-"
    panel tabs  ['contact']            - no PAE tab, because there is no PAE
    full_data   atom_certainty         - NOT atom_plddts
    summary     chain_pair_max_contact only
    README      no "not in this archive" line: this model takes no alignment
    frames      11 restored

The offer row says no pLDDT rather than `pLDDT 0.0`, which is the whole rule in
one line.

## The alignment travels now

The session deliberately dropped the MSA, because it is 96.8% of a fold archive
- 3.0 MB of ubiquitin's 3.1 MB. That reasoning was about an a3m's RAW size,
which is not what gets stored: an alignment is thousands of near-identical rows,
about the most compressible thing in the record. The reference archive's four
real blocks are 1,288,080 bytes and gzip to 280,784, a ratio of 4.6.

What it buys is the difference between a fold that can be **reproduced** and one
that can only be looked at - a re-search finds different hits, so without it a
restored fold's archive has to carry the "may find different hits" caveat.
Measured on ubiquitin, 76 residues, MMseqs2:

| | before | after |
|---|---|---|
| session record | ~250 KB raw / ~55 KB gz | **9,244,617 raw / 2,756,924 gz** |
| restored archive | 19 KB, no `msas/` | **1,233,759 B, with `msas/`** |
| README | "may find different hits" | no caveat: it reproduces |

🔴 **AND IT IS THE ONE FIELD ALLOWED TO BE DROPPED.** Everything else in the
record is bounded by the fold; an alignment is bounded by what a public server
returned. So a `"quota"` save retries WITHOUT it and says
`saved without its alignment - there was no room for it`, rather than letting
one deep MSA cost the whole session. That is also the first thing that has ever
exercised the quota branch's neighbourhood.

🔴 **THE OFFER ROW READS A SEPARATE RECORD NOW, AND HAS TO.** It needs a model
name, a residue count, a score and a timestamp - and it was getting them by
ungzipping the entire session on every page load. That was 52 KB before; after
this it is 2.8 MB of gzip over 9.2 MB of JSON, on a phone, to decide whether to
show one line. `current-meta` is written in the SAME transaction as the session
so the two cannot disagree, and it is named by what it DROPS - the alignment,
the structure, the templates - because a summary built by listing what it keeps
goes stale the moment `jobMeta` gains a field.

### The fourth README state, which every single-sequence archive got wrong

Found while making `archiveFor` ask what is actually held rather than where the
prediction came from. The "here is `msas/`" branch was reached by any model with
an alignment CONTROL - so a fold that deliberately used none described a
directory that was not in the file and told the reader to drop the zip back "to
fold again with exactly these alignments". **Every single-sequence archive this
page has ever written said that**, and it is live on the site right now.

Whether the archive CARRIES an alignment is a different question from whether
the model TAKES one, and they are asked separately now - the carrying half
answered by looking at the files that were written, not by a caller's flag:

| the fold | the README |
|---|---|
| searched, alignment carried | ``msas/`` holds one alignment per chain |
| searched, alignment dropped | left out, "may find different hits" |
| ran on the single sequence | **ran on the sequence alone; folding it again reproduces it** |
| model takes no alignment | folds from the sequence alone |

### Forget deletes both records, and did not

A second record is a second thing to delete, and only the write path knew it.
`clearSession` removed the session and left the summary, so **Forget** deleted
the fold while the offer row stayed on screen advertising it - and pressing
Restore then said "there is no saved session to restore". Both are deleted in
one transaction now.

🔴 **AND A GATE ARM RELOADS AFTERWARDS, because the click alone cannot show
this.** The row hides itself on the click, from memory; it is only redrawn from
the store on the next page load, which is where a stale summary reappears. The
arm also reads the store's remaining keys directly - `keysLeft: []` is the
assertion, since "the row is hidden" was true even while the bug was there.

    forget:       {"hiddenAfter": true, "keysLeft": []}
    after forget: {"offered": false, "text": ""}

## Every page timing here is a FIRST visit, and a returning one is much cheaper

🔴 **`cdp.launch` WIPES THE CHROME PROFILE ON EVERY RUN**, which is the right
default for a checker - CLAUDE.md's trap about a cached ES module looking
exactly like a broken feature is about the other direction - and it means
nothing in this repository had ever measured what a RETURNING user pays. A
fresh user-data-dir has no HTTP cache and no SHADER cache, so every number
recorded for the page includes downloading the whole bundle and compiling every
pipeline.

`fold-in-page.py --keep-profile` reuses it. Same machine, same fold, back to
back:

| | first visit | second | third |
|---|---:|---:|---:|
| OpenDDE, whole page | 7123 ms | **5896** | 5862 |
| ...its status line, the fold alone | in 4 s | in 3 s | in 3 s |
| AlphaFold 3, whole page | 4901 | **4215** | 4262 |

**1.26 s off OpenDDE and 0.66 off AF3**, and the split matters: the status line
is the fold's own clock, which starts after the weights are loaded, and it drops
a whole second on OpenDDE. So a large part of what a second visit saves is
Chrome serving the 269 compiled pipelines out of its shader cache, not merely
the weights out of its HTTP cache. The status line rounds to whole seconds, so
that is a direction and not a split.

🔴 **WHICH REFRAMES THE COMPILE WORK.** OpenDDE's 1.3-1.8 s of shader
compilation - `busyMs` in probe-compiles.js, against a 2.5 s fold - is a
FIRST-VISIT cost, not a standing tax. It is worth reducing for a first
impression and it is not what a returning user waits for. The weight download
does not get cheaper in the same proportion, which leaves it the dominant term
in both cases and puts bundle SIZE where docs/HOSTING.md already says it is.

### And where the load itself goes, which is one phase

`af3LoadMilliseconds` times each half of `loadAf3Weights`, read back by
`fold-in-page.py` as `weightPhases`. OpenDDE:

| phase | first visit | returning |
|---|---:|---:|
| open the store | 9 ms | 9 |
| **trunk** | **1780** | **747** |
| diffusion | 25 | 24 |
| the structural expander | 63 | 62 |
| atomReference, targetFeat, refiner, confidence | 8 | 8 |
| **total** | **1884** | **849** |
| ...of which host tensor decode | - | **95** (138 calls) |

The whole load is the trunk phase, and on a returning visit it is 747 ms of
which only 95 is decoding: the other ~650 is moving 472 MiB out of Chrome's
disk cache and onto the GPU, which is about 725 MB/s and close to what that
path can do. `--timeline` reports `model: null` on that visit, so none of it is
network.

🔴 **SO A RETURNING VISITOR'S LOAD IS ALREADY NEAR ITS FLOOR, AND THE FLOOR IS
THE BYTE COUNT.** 472 MiB costs 650 ms even from local disk. Nothing in the
decode path is worth attacking - 95 ms - and the download is already cached.
The only thing that moves this number is a smaller bundle, which is where
docs/HOSTING.md already points and which needs the float32 exports and a
re-publish.

### Creating the GPU device early: measured, already overlapped, not taken

`getDevice` memoises one promise and its first caller used to be the fold, so
the natural next move after the weight preload and the shader warm was to create
the device at page load too. Measured on a cold page, the first `getDevice()` is
**148 ms** and every call after it is 0.

It buys nothing. Alternating arms, three pairs on the monomer page: 3148 / 3399
/ 3151 ms with the device created at load against 3361 / 3424 / 3147 without -
the spread inside each arm is larger than the difference between them. The
reason is that the shader warm above already calls `getDevice()` at click time,
concurrently with the weight download, so for every AF3-family model the 148 ms
was already hidden the moment that landed. Reverted rather than kept as a
plausible-looking three lines that move nothing.

Two things this does NOT say. It is one browser on one machine, and a shader
cache is a heuristic with an eviction policy nobody here controls. And
`--keep-profile` must never become the default for a checker: the whole reason
the wipe exists is that a stale module is indistinguishable from a broken
feature, and this file is the wrong place to learn that again.


## 🔴 THE DOWNLOAD DIAL FLICKERED FOR A DELTA MODEL, AND ONLY FOR THE FIRST ONE

Reported from the page: "progress wheel is flickering if I go directly to model
2 (and haven't tried running model 1 yet)". Both halves of that sentence are the
diagnosis.

A delta family downloads TWO bundles - 43 MiB of difference and the 73 MiB base
it is added to - and `openStore` handed each store the caller's `onProgress`.
They then take turns owning one arc: "4 of 43 MiB", "20 of 73", back to "6 of
43". The parenthesis is the other half: it only happens when the base is NOT
already cached, because a visitor who has folded with model_1 gets that store
back from the `stores` map without a byte moving.

Measured with `tools/download-dial.py`, which samples every change to
`#model-load` and fails on either a loaded count that DECREASES or more than one
TOTAL:

| | totals offered | backwards steps |
|---|---|---:|
| both stores on one callback | 43 and 73 MiB | **273** |
| reported as one stream | 116 MiB | 0 |

It ends at "43 / 43 MiB" in the broken arm, because the smaller download
finishes last and overwrites what the bigger one had reported.

🔴 **AND THE FIRST FIX LEFT HALF OF IT**, which is why the second column of that
table is not the whole check. Summing the two streams stops the numerator
falling, and the arc still snapped back ONCE: the delta's store reports
"0 / 43 MiB" the moment it opens, before the base has a manifest, and the next
update says "0 / 116" - the same bytes, a third of the arc. A manifest is
compiled in rather than fetched, so the base's total is seeded before the first
report. Only when it is actually going to be fetched: web/esmfold2-model.js's
rule for its language model, from the other direction.

web/esmfold2-model.js has had the one-stream rule since its own 347 MiB
download, and this is the second place to need it. They are not shared yet -
that one must also STOP reporting when the load ends, its tower going on
streaming through the fold, which nothing here does.

### "All 5" saves five structures now, not one

Asked after the sweep shipped: *"if all 5 models are selected, are we saving the
best from each model or best over all"*. It was best over ALL - one
`_model_0.pdb`, the single best pass of the five, with the other four visible on
the play bar and absent from every file. Five models folded and one structure
saved is four thrown away, and the one that is kept has nothing to be compared
against.

The archive now writes, beside the file it always wrote:

```
fold_all5_1_model_0.pdb                 the winner, where every reader looks
fold_all5_1_rank_001_model_3.pdb        ...and the same structure named by model
fold_all5_1_rank_001_model_3_scores.json
fold_all5_1_rank_002_model_1.pdb
...                                     one per model, ranked
```

Ranked by the criterion the page itself used to pick the winner - the multimer
score for a complex, mean pLDDT otherwise - so `rank_001` IS `_model_0.pdb`
under a name that says which of the five it was. ColabFold's convention, which
is what anyone running all five expects.

🔴 **AND A SINGLE-MODEL FOLD IS BYTE FOR BYTE THE ARCHIVE IT ALWAYS WAS.**
`perModel` is undefined unless a sweep ran, so nothing that reads these files
has to learn a second layout for the common case. `test/fold-archive.test.js`
gates both directions and was watched failing with the loop removed.

### ...and re-running a sweep refetched four deltas, until it kept the right half

Reported twice: *"when I rerun, it redownloads?"*, then *"models still appear to
be redownloaded each time I hit fold"*. The second report is the one that found
the bug, because the first measurement had answered the wrong question.

**A single model was never the problem.** Two folds in one session with model 1,
counting shard requests through `performance.getEntriesByType("resource")`:
run one fetches 8, **run two fetches 0**. The weight tree and the store are both
cached by family and nothing drops them.

**A sweep was.** "All 5" releases each delta once its passes are in hand,
because five models held at once is **3409 MiB of JS heap** against Chrome's
~4 GB ceiling. The release was `stores.delete(family)` - and that threw away two
different things that happen to live in one object:

| the store holds | size for a delta | wanted |
|---|---|---|
| `#fileBuffers`, the shards as downloaded (int5 codes) | 43 MiB | **keep** |
| `#cache`, what they decode to (float32) | ~8x that | drop |

So the memory came back and the BYTES went with it: the next sweep fetched all
forty shards again and ran the download dial for something the browser already
had. `releaseDecoded()` on both stores drops the decode and keeps the shards,
and `releaseModel` calls it instead of deleting the entry:

| | before | after |
|---|---:|---:|
| shard requests on a second sweep | 40 | **0** |
| JS heap after a sweep | 302 MiB | **300 MiB** |

Identical memory, no refetch, and no dial - `openStore` returns the cached store
without wiring a progress callback at all, so nothing animates. What the second
sweep pays is the decode, which is what it was always going to pay.

🔴 **AND THE FIRST MEASUREMENT SAID "NO SLOWER" AND WAS NOT WRONG, JUST NARROW.**
Two sweeps timed 16.0 s and 15.3 s, and that was read as "the refetch is free,
the dial is cosmetic" - but a reader watching 172 MiB count up again has no way
to know that, and **a number that says "no slower here" is not an answer to
"why is it doing that at all"**.

🔴 **AND THE REASON GIVEN FOR IT WAS WRONG, WHICH TOOK A `curl -I` TO SEE.**
That paragraph used to say the shards came back from the browser's HTTP cache
"because the bundles are pinned to a commit and their URLs are immutable". The
pinned URL is immutable and it is **`cache-control: no-store`**: Hugging Face
answers `resolve/<sha>/<file>` with a 302 to a SIGNED CDN url carrying its own
`Expires` and `Signature`, so the final url differs on every request and the
HTTP cache can never match it.

What actually serves a repeat is this port's OWN cache, which
`http-tensor-store.js` has had all along: `#cacheMatch` / `#cachePut` against
Cache Storage, keyed by the stable manifest-relative url, so the signed
redirect underneath is irrelevant. Measured on the live site, AF2 monomer:

| | wall | shard requests on the network |
|---|---:|---:|
| first visit | 4.0 s | 8 |
| after a page RELOAD | 4.0 s | **0** |

with a bucket named `localfold-model-model_1_ptm-76465608-337` holding them. So
the three layers are: the weight tree and store in memory within a session
(what `releaseModel` now keeps), Cache Storage across reloads, and the network
once.

🔴 **AND `transferSize` IS ZERO FOR ALL OF THEM, WHICH IS NOT EVIDENCE OF
ANYTHING.** Hugging Face sends no `Timing-Allow-Origin`, so a cross-origin
resource-timing entry reports 0 bytes whether it came from the wire or not -
which is where "40 served from cache, 0.0 MiB over the wire" came from in the
first place. The COUNT of entries is usable; the bytes are not.

### The options row pairs by question now: Model+Seed, Recycles+Early Stop, MSA+Max MSA

Asked for directly. The narrow layout already paired by meaning - that is what
the flex bases in `localfold.css` are for, and `tools/mobile-layout.py` asserts
the MEMBERSHIP of each line rather than a line count - but the pairs were the
old ones, and `Early Stop` arrived beside Recycles in the DOM and nowhere in the
rule. Measured at 390px, every family:

| family | rows |
|---|---|
| monomer / multimer | `Model + Model# + Seed` / `Recycles + Early Stop` / `MSA + Max MSA` |
| af3 | `Model + Seed + Recycles` / `Sampler + Steps` / `MSA + Max MSA` |
| ef2-fast-600m | `Model + Seed + Recycles` / `Steps + PLM` |

Two changes and no new machinery: the DOM order became Model, Model#, Seed,
Recycles, Early Stop, ... so that reading order and line order agree, and
`#af2ModelGroup` joined the thirds while `#toleranceGroup` joined the halves.
🔴 **RECYCLES STAYS A THIRD AND THAT IS THE LOAD-BEARING PART**: AlphaFold 3
hides Early Stop, so at a half Recycles would take a line of its own and drag
Sampler up beside it - splitting Sampler/Steps and MSA/Max MSA across lines,
which is the exact fault the bases were written to fix. At a third it joins
Model and Seed instead, and the pairs below it are untouched.

🔴 **AND MOVING TWO BLOCKS OF HTML BY SCRIPT ATE A COMMENT.** The first attempt
cut each group from a blank-line boundary, which took the `-->` of the comment
above it and left the comment open - so everything to the next `-->` became
comment, the seed group vanished from the DOM, and the desktop row "broke across
2 lines". That looked like a CSS regression and was a mangled document; the
gate caught it, and the check that would have caught it sooner is one line:
`<!--` and `-->` counts, and `<div` and `</div>` counts, before writing.


## The job JSON was readable and unreachable, which is the same as unread

Asked by a user: can they upload an AlphaFold 3 style JSON and run it? The
answer was *yes, and almost nobody could have found out*. Three questions came
out of it and only the third needed code.

**Is one written correctly?** Yes, gated from both ends. `jobRequestJson`
writes the server dialect and switches to the open one for the job that dialect
cannot express - a SMILES ligand, which has no field there at all.
`test/job-json.test.js` round-trips it, and
`tools/fold-in-page.py --job-round-trip` folds, **wipes the entity rows**, drops
the archive back and compares: `same: true`, `seedSame: true`.

**Is one read correctly?** Both dialects, against a corpus that is not ours:
`test/af3-example-jobs.test.js` runs DeepMind's thirteen `examples/*.json` plus
the pipeline's kitchen-sink `alphafold_input.json`. **Nine of fourteen load;
five refuse by name** - three `bondedAtomPairs`, two modified bases. The
refusals are half the value: every one parses perfectly well as far as the
sequence, so a lenient reader folds a real structure of the right protein with
the inhibitor unbonded and calls it the job that was asked for. 🔴 CLAUDE.md
said **8 of 14 and six refusals** and had been stale since SMILES landed. The
test asserts the count precisely so that moving a file between the lists is a
decision somebody makes out loud; the only thing that drifted was the prose.

**Could a reader reach it?** No, and this is the part that was broken.

🔴 **THE ONLY DOOR WAS THE ALIGNMENT UPLOAD BOX, WHICH IS HIDDEN.** `#msa-file`
is `hidden` unless the MSA dropdown reads "Upload file" (`syncMode`), so loading
a job meant setting a control about alignments to a value about alignments in
order to hand over a file that is not an alignment. Nothing on the page said so.

🔴 **AND THE GESTURE A READER ACTUALLY TRIES CAME BACK AS A STRUCTURE ERROR.**
py2Dmol's `initDragAndDrop` binds the four drag events on `document.body` and
hands every file to `handleFileUpload`. Measured with our listener disabled,
dropping `tetr_homodimer.json` on the page gives

```
Error processing loose files: No structural files (*.cif, *.pdb, *.ent) found.
```

with every row unchanged. Not silence - worse in one way, because the one error
a reader sees points at a structure problem in a file that has no structures in
it and never should have.

### One reader for the gesture, which is the whole fix

The first attempt kept both: claim `.json`, parse it, hand it back to py2Dmol
when the top level had no `sequences`. That was the wrong trade, and the reason
is not taste - **every refusal then has to be guessed at twice**, because which
reader answered depended on how far the other one got. A `.json` with a typo in
a field name is a job-json refusal; a `.json` that is a py2Dmol session is a
viewer error; and a reader holding `fold_scores.json` out of our own archive got
"No valid objects found in state file", which is true of nothing they did.

So **this page owns dragenter, dragover, dragleave and drop**, in the capture
phase on `window`, and routes every dropped file through `readHandedFile` - the
same router the alignment upload box uses, so the two entry points cannot
drift. It reads what the page folds *with*: a job JSON in either dialect, a fold
archive, an alignment. Anything else is refused by name.

🔴 **ALL FOUR EVENTS, NOT JUST `drop`.** py2Dmol shows its overlay on dragenter
and counts enters against leaves in a closure this module cannot reach. Taking
only the drop left that count stuck and the overlay up for ever - an earlier
version had to undo it with a synthetic empty drop dispatched at the body.
Taking all four means its counter never moves, and this page drives
`#global-drop-overlay` itself, off `relatedTarget === null` rather than a depth
tally: a tally has to be right on every enter and leave or it sticks, which is
precisely how the overlay got stuck in the first place.

🔴 **AND A STRUCTURE HAD TO BE REFUSED BY NAME, BECAUSE `parseA3m` WOULD TAKE
IT.** By the time control reaches the last branch an alignment is "any text that
is not JSON", and a PDB is text. With the guard removed a dropped structure
comes back as **"A3M sequence data appears before the first FASTA header"** -
a confusing alignment error for a file that is not an alignment, which is the
same wrong-message failure one layer down.

**What it costs, said out loud:** dropping a `.pdb` or `.cif` no longer shows it
in the viewer, and py2Dmol's `paeFromJSON` pairing - a structure and its PAE
`.json` dropped together - goes with it. Both worked here before (measured:
`structure, 1 residues, 1/1 PAE matrices paired`). They are the price of one
reader, and the refusal names what the page does take so nobody is left
wondering whether the drop registered.

🔴 **`#file-upload` AND `#upload-button` STAY IN index.html, HIDDEN.** Deleting
them throws inside py2Dmol's `setupEventListeners`, which silently aborts the
rest of `initializeApp` and takes the MSA panel's wiring with it - index.html
warns about exactly this beside them. They sit in a panel at `display: none`, so
nothing reaches them.

### 🔴 Two of the arms first passed on the bug they were written for

`tools/fold-in-page.py --drop-job` - a real `DragEvent` on the body, not the
file input. That distinction is the point: `--job-round-trip` sets `input.files`
and fires a `change`, so it exercises job-json.js and nothing whatever about who
*receives* the file.

| arm | the check that did not work | why it passed anyway | what it asks now |
|---|---|---|---|
| the job loads | - | - | rows wiped first, then `filled` and `claimed` |
| the overlay survives | is it showing at the next `dragenter`? | `flex` either way - with the reset deleted it was **still showing from the drag that was claimed**. The first probe also never fired `dragenter` *before* the drop, so the counter was never dirty at all: half a gesture measures nothing | it must go to `none` when the file is taken *and* come back at the next enter |
| a stray JSON is ours | does the status look like one of our refusals? | a handler that wrongly claimed one produced ``no `sequences` in that job``, which that check did not recognise | it wraps `window.handleFileUpload` and **counts calls** - 0, always |

Measured, shipped code against each falsification:

| | `filled` | `claimed` | `overlayShown` | `structureRefused` | `nothingHanded` |
|---|---|---|---|---|---|
| shipped | true | true | `flex` | true | true |
| drop listener removed (*the old behaviour*) | **false** | **false** | `flex` | **false** | **false** |
| only `drop` claimed, not the other three | true | true | **`none`** | true | true |
| structure guard removed | true | true | `flex` | **false** | true |

Row two is what a reader got before this: `protein:8x1` in, `protein:8x1` out,
py2Dmol's structure-file error, and a dropped PDB loading into the viewer
(`structure, 1 residues`).

**And one line of HTML**, under the entity rows, because the drop target being
the whole page is exactly why nothing on screen implied it existed. A sentence
rather than a dashed drop zone: a box drawn round part of the page would be a
lie about where the file may be let go, and a visible affordance that misstates
itself is worse than an invisible one. The probe asserts it is on screen, since
a working drop nobody knows about is the same bug one layer up.

🔴 **WHAT IT DELIBERATELY DOES NOT DO IS PICK THE MODEL.** A ligand, a nucleic
chain or a modified residue needs AF3, and `applyJob`'s own comment says why the
decision does not belong there: the guard already exists at fold time with a
message naming the model that is set, and a second reader of the same control is
the mistake `chosenFamily` was written to end. A reader who drops
`calmodulin_4calcium.json` under AF2 and presses Fold is told *"Ligands need
AF3, OpenBind-0 or ESMFold2; the model is set to monomer"* - one more click, and
never a quietly dropped calcium.

## Purging py2Dmol's website: 138 lines of it were load-bearing

LocalFold offers prediction; py2Dmol.solab.org offers the viewer. So the fetch
row, the four example buttons, the options disclosure and the seven checkboxes
that `index.html` carried at `display: none` were py2Dmol's website showing
through, and none of it was reachable. **Most of it went. Seven elements did
not, and finding out which is the whole of this entry.**

🔴 **THREE ARE READ WITHOUT A GUARD AT STARTUP.** `setupEventListeners` does
`getElementById("fetch-btn").addEventListener(...)` and the same for
`#upload-button` and `#file-upload` - everything else in that function is
`&&`-guarded, which is why `#drawCheckbox`, `#saveStateButton`,
`#prevObjectButton` and `#nextObjectButton` have been absent all along with
nothing to show for it. A throw there aborts the rest of `initializeApp`, and
the symptom is not an error: it is the MSA panel never wiring itself up.

🔴 **AND `#loadAsFramesCheckbox` IS ON OUR OWN FOLD PATH, WHICH IS THE ONE THAT
WOULD HAVE HURT.** `processFiles` - reached by `window.py2dmolLoadFiles`, which
is how `web/app.js` puts a prediction on screen - reads `p.checked` with no
guard at all. Measured by deleting it:

```
Folding # residues · # passes · monomer
Error processing af#_#.pdb: Cannot read properties of null (reading 'checked')
Done in 1.2 s · pLDDT 64.2 · pTM 0.338
```

The fold computes, the status line reports a perfectly good pLDDT, and the
structure never arrives. A hidden checkbox in a panel nobody can see decides
whether a prediction is displayed.

🔴 **AND ABSENT IS NOT UNCHECKED.** Three of the remaining boxes are read as
`!!m && m.checked` and two as `!d || d.checked`, so deleting one is *false* in
the first case and *true* in the second. `alignFramesCheckbox` and
`loadMSACheckbox` are the first kind and stay, with their `checked` attribute,
because losing them would silently turn frame alignment and the MSA hand-off
off. `loadPAECheckbox` is the second kind and means the same thing absent, so it
went - along with `biounitCheckbox`, `loadLigandsCheckbox` and
`filterAdditivesCheckbox`, whose values `initializeViewerConfig` hard-codes as
defaults anyway, and `alignChainInput`, read as `?.value || ""`.

🔴 **AND ONE IS A FLAG RATHER THAN A CONTROL - KEPT ON A READING, WHICH IS
WEAKER AND IS RECORDED AS SUCH.** The bundle computes, once at load,

```js
const isIndexHTML = getElementById("fetch-id") !== null
                 && getElementById("fetch-uniprot-id") === null;
```

and that is the only thing gating `initializeMSAIndex()`. Deleting `#fetch-id`
was tried and **changed nothing observable**: a single-chain fold with a search
is identical either way, because the chain selector that function fills hides
itself below two chains and LocalFold prints its own "Loaded MSAs" line. So the
falsification is *missing, not passed*; a two-chain fold is where it would show.
One line to keep, against an MSA panel that quietly stops wiring itself up.

138 lines to 48, of which 41 are the comment explaining why the other seven are
still there.

### 620 KB published for pages that are not on the site

`py2Dmol.embed.min.js` is loaded by `single.html` and `proteinhunter.html`, both
**held back at `b0dc258`** - out of the repository and gitignored until the model
row they were built against stops moving, with every tool that touches them
skipping a page it cannot find. `web/` is copied wholesale, so their bundle
shipped on every deploy to be fetched by nobody: the second-largest file on a
site Pages caps at a gigabyte.

🔴 **DELETING THE MIRROR WAS THE FIRST ATTEMPT AND IT WAS WRONG.** That commit's
own promise is that putting those pages back is *one line*, and it kept every
tool that touches them working for exactly that reason. Removing the vendored
bundle would have made it one line plus a `sync-py2dmol.py` run against an
upstream checkout that is not on this machine.

So `build_site.py` derives it instead: after the copy, a top-level file in
`dist/web/vendor/` that no shipped `.html`, `.js` or `.css` names is removed
from **`dist/`**, and the build says so.

```
   left out py2Dmol.embed.min.js (620 KiB): no page in this site loads it
dist/  240 files, 23.0 MiB
```

Derived for the reason the model registry is - the day `single.html` returns its
bundle returns with it, with no edit here and nothing to forget. Verified both
ways: a stub `single.html` loading that bundle gives **242 files, 23.6 MiB** with
the bundle present, and removing it again gives 240 and 23.0. `.md` is excluded
from the search because `SOURCE.md` names every bundle by definition, and only
top-level files are considered because `mpnn/kernels.wasm` is loaded from
JavaScript rather than from a page.

## Do AlphaFold 3's own examples work? Nine of fourteen, and now that is measured

`test/af3-example-jobs.test.js` asserts what each of DeepMind's example jobs
*becomes* - the entity list, the seed, the dialect - and **never folds one**. So
"nine of fourteen load" was never a claim that nine of fourteen work, and
nothing here had asked the second question.

First, the corpus is complete and faithful. Checked against the upstream
checkout (`sokrypton/alphafold3` main, whose `examples/` this is; re-checked against a fresh clone 2026-09-20, since `~/af3fork` on this box is a non-git snapshot and is behind): **all thirteen are byte-identical**
to ours, with `alphafold_input.json` the extra kitchen-sink from
`src/alphafold3/common/test_data/`. Nothing has been added upstream that we
lack, and nothing here was edited.

`tools/fold-in-page.py --job=<path>` folds one the way a reader would: it drops
the file on the page, lets `web/job-json.js` fill the entity rows, and presses
Fold. Nothing in the tool sets a sequence, a ligand or a modification - all of
that comes out of the file, or the run would be testing this script's reading
of the format instead of the page's.

**All nine that load, fold.** AF3 int5, single sequence, 2 passes, 4 diffusion
steps, seed 42 out of each file:

| example | what the file asks for | in | pLDDT |
|---|---|---:|---:|
| `ubiquitin_monomer` | protein 76 | 2 s | 61.6 |
| `barnase_barstar` | protein 110 + protein 89 | 7 s | 39.1 |
| `u1a_rna_hairpin` | protein 101 + **RNA 21** | 6 s | 60.0 |
| `calmodulin_4calcium` | protein 149 + **CA x4** | 8 s | 78.5 |
| `streptavidin_biotin_smiles` | protein 126 + **biotin as SMILES** | 7 s | 45.8 |
| `tetr_homodimer` | protein 218 **x2** | 16 s | 36.4 |
| `tetr_dimer_tetracycline` | protein 218 x2 + **TAC x2** | 21 s | 42.5 |
| `erk2_phosphorylated` | protein 360 + **TPO@185, PTR@187** | 14 s | 31.3 |
| `tetr_dimer_dna` | protein 218 x2 + **DNA 20 x2**, 476 tokens | 18 s | 36.4 |

🔴 **READ THE STATUS LINE, NOT THE pLDDT.** These are single-sequence folds at
four diffusion steps - the cheapest setting the page has - so the confidence
numbers say nothing about quality and are not evidence of anything except that
a number came out. What is evidence is the *content* of each line, because it
is built from what the featuriser actually received:

```
Fetching ligand CA        → 149 residues + CA, CA, CA, CA
Fetching modified residue TPO, PTR → 360 residues + TPO185, PTR187
Building LIG from its structure    → 126 residues + OC(=O)CCCC[C@@H]1SC[C...
                                     476 residues in 4 chains
```

Four calciums from an `id` LIST, both phosphorylated residues by CCD code, a
biotin built from SMILES rather than looked up, and a protein/DNA complex at 476
tokens. And none of the nine appended **"NOT A CHAIN"**, which is the geometry
rule `src/af3/chain-geometry.js` applies to every fold the page draws.

**The five that refuse are two gaps, both real.** Three carry `bondedAtomPairs`
(`kras_g12c_sotorasib`, `rnaseb_glycosylated`, and the kitchen-sink
`alphafold_input`) - covalent chemistry this port does not build. Two are
modified BASES: `methylated_dna` (8 x 5CM across two strands) and `modified_rna`
(PSU, 5MC, OMG). That second refusal is not conservatism - `web/entities.js`
states the mechanism, and it is worth quoting because it is the shape of the fix:
the modified-residue path *resolves the parent through the amino-acid table*, so
a modified base would be featurised as a modified amino acid and fold to
something plausible. Making that resolution type-aware is what takes the corpus
from nine to eleven, and it needs `test:batch` rather than a fold to prove.

## Is what the archive SAVES the job that was handed in? Now checked, and one field was not

"All nine fold" is not "all nine save correctly". The archive is the file a
reader hands back to reproduce a fold, and the only thing that had ever checked
it was `fold-in-page.py --job-round-trip` - which reads the archive back through
`web/job-json.js`, **the module that wrote it**. A writer and a reader that
share a mistake agree perfectly; that is exactly the shape of the SMILES bug in
docs/SMILES.md, a benzene written as a CCD code and read back as one,
round-tripping in silence into cyclohexane.

So `tools/check-job-archive.py` is a separate process that never imports the
page's reader. It normalises the archive's `_job_request.json` and the
**original input file** in Python and compares them as meaning rather than text
- which it has to, because the comparison is cross-dialect: AlphaFold 3's
examples are the open dialect (`protein: {id: ["A","B"]}`, integer seeds,
`ccdCodes`, `modificationType`/`basePosition`) and this page writes the server
one (`proteinChain: {sequence, count}`, string seeds, `ligand`,
`ptmType`/`ptmPosition`), except a SMILES ligand, which the server dialect
cannot express and which goes out open.

**Nine of nine match.** Saved with `fold-in-page.py --job-archive=<path>`:

```
ubiquitin_monomer           tokens  76  chains A          atoms  602  ptm 0.54
barnase_barstar             tokens 199  chains A,B        atoms 1598  ptm 0.25
u1a_rna_hairpin             tokens 122  chains A,B        atoms 1256  ptm 0.46
calmodulin_4calcium         tokens 153  chains A,B,C,D,E  atoms 1178  ptm 0.46
streptavidin_biotin_smiles  tokens 142  chains A,B        atoms  951  ptm 0.37
tetr_homodimer              tokens 436  chains A,B        atoms 3442  ptm 0.19
tetr_dimer_tetracycline     tokens 500  chains A,B,C,D    atoms 3506  ptm 0.19
erk2_phosphorylated         tokens 385  chains A          atoms 2923  ptm 0.19
tetr_dimer_dna              tokens 476  chains A,B,C,D    atoms 4264  ptm 0.19
```

🔴 **AND THE TOKEN COUNTS ARE AlphaFold 3's OWN RULE, CHECKED AGAINST THE SAVED
PDB'S ATOMS RATHER THAN ASSERTED.** One token per polymer residue, one per atom
for a ligand or an atomised residue - counted out of each archive's own
structure file:

| | polymer | hetero atoms in the PDB | tokens |
|---|---:|---|---:|
| calmodulin_4calcium | 149 | `CA` x 4 | 149 + 4 = **153** |
| streptavidin_biotin_smiles | 126 | `LIG` 16 (biotin, built from SMILES) | 126 + 16 = **142** |
| tetr_dimer_tetracycline | 436 | `TAC` 64 (2 x 32) | 436 + 64 = **500** |
| erk2_phosphorylated | 360 | `TPO` 11 + `PTR` 16 | 360 - 2 + 27 = **385** |
| tetr_dimer_dna | 436 | none (DNA is one token a base) | 436 + 40 = **476** |
| u1a_rna_hairpin | 101 | none | 101 + 21 = **122** |

Beside the request, the verifier holds the rest of the save to the fold: `pae`
and `contact_probs` square at the token count and in range (a decode with the
wrong bounds fills the key with plausible nonsense), `token_chain_ids` and
`token_res_ids` as long as those matrices, **`atom_plddts` one per ATOM/HETATM
record in the PDB** - the one cross-file check, and the one that would catch a
ligand counted in one file and not the other - and every ligand code and
modified-residue code actually present in the structure, which no confidence
number can see.

### 🔴 The job's own name: built, then removed

Every AlphaFold 3 example carries a `name`, `jobFromJson` reads it, and the
archive wrote the fold's stem in its place - so `calmodulin_4calcium.json` came
back out of the page as a job called `af3_1`. Nothing could have caught it:
`--job-round-trip` compares entity lists, and a name is not an entity.

It was fixed three ways in turn, and then taken out. The record is here because
the reasoning is worth more than the code was:

1. **A `jobName` beside the stem.** Half a fix - the object in the picker, the
   `.pdb` button and every member of the archive still said `af3_1` while one
   field of one file said otherwise.
2. **A visible name box**, which deleted machinery rather than adding it: the
   name had been state nobody could see, and keeping it honest took recording
   the SHAPE of the rows the job created and re-checking it at fold time,
   because `set` and `setChains` both `render()` without notifying any edit
   hook. The box also had to break a trunk continuation on a rename, or
   renaming did nothing at all.
3. **Removed**, as not earning its place.

🔴 **WHAT SURVIVED IS THE PART WORTH KEEPING: ONE RESOLVER.** There were three
places that built a stem. `foldStem` is now the only one - a pasted FASTA
`>header` names the fold, the model prefix is the fallback - so the object, the
download button and the archive cannot drift apart, and `buildFoldArchive` has
no `jobName` parameter for them to drift through. `test/model-family.test.js`
counts the call sites, which is what stops a fourth path naming its object some
other way.

🔴 **AND THE GAP IS NOW REPORTED RATHER THAN ASSERTED.**
`check-job-archive.py` prints `named 'af3_1', not 'ubiquitin_monomer' (this page
names a fold, not a job)` beside the token counts, and does not fail on it - the
same treatment `ref_pos` gets in `check-batch-fields.js`, and for the same
reason: a deliberate difference held as a failure is a red gate for ever. The
chemistry is what must match.

**What is still dropped, deliberately:** the `description` strings inside a
chain body. They are free text about the job rather than part of it, nothing in
the fold reads them, and the server dialect has no field for one.

## The hint became a field, and the dropdown grew a door

The line under the entity rows - *"Drop an AlphaFold 3 job JSON anywhere on this
page to fill these rows in, then Fold — either dialect..."* - was three clauses
of prose standing permanently above the thing it described. It is gone, and the
two things it was trying to say are now controls.

**`Load job JSON…` in the MSA dropdown.** That dropdown was already the only
door - the file input lives behind "Upload file", which is a control *about
alignments* - so loading a job meant setting an alignment dial to hand over a
file that is not an alignment. Now it says so.

🔴 **IT IS AN ACTION IN A LIST OF STATES, AND IT NEVER STAYS SELECTED.** A job
is not an MSA mode: left selected, `msaMode()` would answer with a value nothing
maps and the fold would run on it - the same trap as a hidden control keeping
its old value, which this page has already been bitten by twice. So choosing it
opens the picker immediately (a label without a door is not a door),
`msaMode()` answers with the mode it replaced while the picker is open, and
`applyJob` puts that mode back the moment the file is read. The one exception
wins over it: a file carrying `unpairedMsa: ""` is asking to fold with **no**
alignment, so the dial goes to Single Sequence and the status says so -
restoring the stashed mode over that would run the search the file asked us to
skip.

**A job-name field above the rows** was added here and later removed; see "The
job's own name: built, then removed" above for what it was for and what
survived it.

🔴 **THE NAME WAS STATE NOBODY COULD SEE.** `applyJob` remembered it and the
archive wrote it, so a saved job could be called `calmodulin_4calcium` with
nothing on screen saying so - and keeping that honest took recording the SHAPE
of the rows the job created and re-checking it at fold time, because `set` and
`setChains` both `render()` without notifying any edit hook and there was
nothing to listen to. All of it existed to guess whether a name the reader
could not see still applied. The field answers it outright: what the box says is
what gets saved, the reader can see it and change it, and a name that no longer
fits is *theirs* rather than a stale one this page invented. `loadedJob` and
`jobShape` are gone.

It names the request only. Every file in the archive is still stemmed `af2_1`,
because the README describes that layout and the prediction, the viewer object
and the session all key on it. One consequence worth knowing: re-loading your
own archive fills the box with that stem, since the request genuinely says the
job is called `af2_1` - honest, visible, and one edit from whatever you want.

🔴 **AND THE FIELD FOUND A BUG THE MOMENT IT WAS WIRED.** The name push landed
above `const said = []` in `applyJob` - a temporal dead zone - so the rows and
the box both filled and the status line read **"Cannot access 'said' before
initialization"**. The probe caught it on the first run; reading the diff had
not.

### A dropped FASTA fills the rows, and a dropped a3m does not

The other half of the ask. Plain text is the one thing that means different
things at the two doors, so `readHandedFile` takes a named parameter rather than
growing a second copy:

- the **MSA box** is a control that says *this is my alignment*, and always has
- a **drop on the page** says *this is what I want to fold*, so a FASTA fills
  the chain rows through `entitiesFromText`

Nothing regressed on that split, and the reason is worth stating: **the
page-wide drop did not exist until this session** - it went to py2Dmol - so no
reader has ever dropped an a3m here and had it taken as one.

🔴 **BUT AN a3m HAD TO BE TOLD APART FROM A CHAIN LIST, OR THE PAGE STOPS
RESPONDING.** `entitiesFromText` makes a row per record, so a 7907-row search
dropped on the page becomes 7907 entity rows: not an error, not a fold, just a
page rendering for a very long time. `looksLikeAlignment` asks three things an
alignment has and a handful of chains does not - an `.a3m` name, lowercase a3m
insertion columns, or more than eight records - and the status line always says
which way it went (`2 chains from two.fasta · MSA ▸ Upload file if it was meant
as an alignment`), so a wrong guess is visible and one click from fixed.

Measured, with both arms falsified:

| | `fastaFillsRows` | `a3mStaysAlignment` | what the a3m became |
|---|---|---|---|
| shipped | true | true | `41 sequences · 10 columns` |
| guard removed | true | **false** | **`protein:10x41`** - the alignment as rows |
| drop routed as `alignment` | **false** | true | `2 sequences · 20 columns` |

🔴 **AND ADDING A PASSING ARM TURNED ANOTHER ONE RED FOR THE WRONG REASON.** The
structure arm compared its result against `after` - the job's rows, captured
near the top - and the FASTA arm above it legitimately replaces them, so
`structureRefused` went false over something that had nothing to do with
structures. **An arm's baseline is the state immediately before it**, not a
snapshot from the top of the probe.

`tools/fold-in-page.py --drop-job` now runs nine arms; `--job` prints
`job name:` with `fileFilledTheBox` and `followsTheBox`, the second by renaming
the box by hand and folding again.


## Folding somewhere else, and a page that is not dead while it happens

`tools/colab_backend.py` runs this page in a headless Chrome on a Colab
runtime and serves it to a reader's browser, which asks it to fold
(`?backend=colab&t=…`, `remoteBackend`/`foldOnBackend` in `web/app.js`). The
first version answered `POST /fold` with the finished structure. That is
correct and it is also **a page that sits blank for the whole fold**: no
progress bar, no status line, no sampler frames - every one of which the page
was drawing perfectly, on the other machine, where nobody could see it.

🔴 **WHAT TRAVELS IS THE PAGE'S OWN CALLS, NOT A SECOND FOLD PATH.**
`remoteTap(kind, payload)` is called from `status()`, `progress()` and both
`drawLiveFrame` closures, and hands each event to `web/colab-bridge.js`. The
reader's page replays them in order: `status` → `status()`, `progress` →
`progress()`, `frame` → a frame appended to the viewer. There is no
remote-only rendering path to keep in step with the real one, which is the
whole reason the runtime runs this page rather than a port of it.

Measured on a 13-mer, streamed against blocking, same fixture:

| | status lines | bar values | frames arriving DURING |
|---|---|---|---|
| streamed | 8 distinct | 5 | **22** |
| blocking (the mutation) | 1 | 1 (indeterminate) | 0 |

🔴 **AND IT WAS STILL NOT LIVE IN COLAB, BECAUSE THE EVENTS WERE PULLED.** The
backend armed `window.__remoteTap` before the click and collected it by
evaluating a splice over CDP every 250 ms, in the same round trip as the watch
loop. Every event therefore travelled only as often as a busy page answered
the debugger - reported from a real runtime as the bar **sitting at "embedder ·
1%" for a whole fold**, with the finished structure appearing at the end.
Measured here against a deliberately busy page: with 300 ms tasks the drain
interval stretched 250 ms → 600 ms and events aged ~300 ms, which is chunky
rather than frozen, so the real fold blocks the thread harder than a synthetic
one - but the shape of the fault is the same and the mechanism is the same.

**THE PAGE PUSHES NOW, IN THE TASK THAT MADE THE EVENT.** `tapOut` posts to
`/up` at once. The page has to be running to produce an event at all, so
asking it again later can add nothing.

🔴 **AND "ONE REQUEST AT A TIME" PUT THE FAULT STRAIGHT BACK, MEASURED.** The
first version of the push held the next batch until the last one RESOLVED -
which needs the main thread to run the response, and a page in the middle of a
fold does not give it up. Twenty events pushed across six seconds of 300 ms
tasks reached the broker at **p50 3,002 ms, worst 5,702 ms**: a pulled feed
wearing a push's clothes, and the gate said so on its first run. Started and
not awaited, the same twenty are **p50 1 ms, worst 2 ms**.

**WHAT THAT COSTS IS ORDERING, AND `seq` IS WHAT PAYS IT.** Several requests in
flight can arrive in any order, so every event carries the page's own count and
`foldOnBackend` sorts each polled batch by it before applying. On loopback
nothing has yet arrived out of order; the sort is for the Colab proxy, which is
not loopback.

🔴 **AND THE COMMANDS COME BACK THE SAME WAY - TWO MAILBOXES, ONE BROKER.**
`tools/colab_backend.py` is a post office: `EVENTS` is what the runtime page
has said, `COMMANDS` is what readers have asked, both append-only and both
read by watermark, so an overlapping poll or a reload re-applies rather than
losing anything. The reader posts `/in {op, payload}` and reads `/down?since=`;
the runtime page reads `/out?since=` and posts `/up`. `fold`, `stop` and
`ping` are the ops.

**WHAT LEFT THE BACKEND WITH IT**: the `#predict` click, the status-line word
list that decided a fold had failed, the download-button readback, the
`__foldState` watch and the three tap drains - about 180 lines of CDP driving
a page by imitation. All of it is in `web/colab-bridge.js` now, where setting
a control is setting a control. **CDP keeps the two jobs only it can do**:
start the browser, and say what card it got.

🔴 **AND STOPPING HAS TO REACH THE OTHER MACHINE.** Aborting locally ends the
polling loop and leaves the runtime folding with its GPU held, so the next
fold is refused. The reader posts `{op: "stop"}` and the runtime page clicks
`predict` - the same toggle a reader would press - so there is no second stop
path to drift.

**ONE GPU, ONE FOLD, AND THE REFUSAL IS THE BROKER'S.** A second Fold click on
a running page is how a fold gets STOPPED, so the page cannot answer "busy" by
refusing a press: the broker raises a flag when it accepts a `fold` and lowers
it on the page's own `result`, which is the event that says the fold ended in
every way a fold can end.

🔴 **AND EVERY EVENT CARRIES TWO CLOCKS.** The page stamps `at` and the broker
stamps `got` on arrival, so "the fold was slow" and "the feed was slow" are
two numbers rather than an argument - which is exactly what the pulled version
could not tell apart, and what cost a session of guessing. The reader keeps
them in `window.__remoteLag`.

🔴 **AND A COMMAND LOOP THAT AWAITS A FOLD CANNOT HEAR `stop`.** The runtime
page obeys its commands in order and a fold is minutes long, so awaiting one
left the reader's Stop sitting in the mailbox until the fold it was meant to
interrupt had finished by itself. A fold is a job: it is started and not
awaited, the loop goes on listening, and the broker refuses a second fold.

🔴 **AND A FOLD THAT FAILED IS NOT WAITED FOR.** The readback loop polls for up
to two minutes because `loadIntoViewer` clears the object's frames before it
re-adds them - a window on the way to a structure. A fold that died has nothing
coming, so `runFold` asks the page: `status(text, true)` marks the line
`.error`, which is the same signal a reader gets, where the word list it
replaces ("stopped", "failed", "refus") was a guess at the page's vocabulary
kept in another file in another language.

🔴 **AND A RELOADED RUNTIME PAGE REPLAYED THE WHOLE SESSION - FOUND BY AN ARM
WRITTEN FOR SOMETHING ELSE.** The page polled `/out` from zero, so coming back
from any reload it obeyed every command the notebook had ever sent. The
heartbeat arm navigates that page away and back, and it came back and re-ran a
fold from an arm ten minutes earlier: 681 MB of weights fetched and a real
58-residue fold on a developer's laptop, reported in the gate's own output as
*"AlphaFold 3 · 58 residues · in 3 s · pLDDT 68.7"*. A page that has just
loaded is not owed the past, so it starts at the queue's current head
(`/out?head=1`).

🔴 **AND THAT FOLD MADE A MUTATION PASS.** The frames arm asked for "three or
more frames", which a 58-residue fold satisfies handsomely - so the mutation
it was written for (frames collected and never drawn) walked straight through
it. The assertion names the structure it pushed now: **three frames of exactly
four positions**, which is the gate's own four alpha carbons and nothing else.
The weights stay blocked for the whole run, too: **nothing in this gate may
ever fold**.

🔴 **AND A RUNTIME THAT HAS GONE MUST NOT BE POLLED FOR EVER.** Colab recycles
a runtime when the notebook is closed or left idle, and the busy flag is raised
by the BROKER and lowered by the PAGE - so a page that died mid-fold took the
flag with it, leaving the reader polling for the rest of the session and every
later fold refused 429. The runtime page asks for its commands three times a
second, and that poll IS the heartbeat: `runtimeSeen` rides on `/down` and
`/health`, the reader gives up past twenty seconds with words that say what to
do ("its notebook may have been closed... run the Colab cell again"), and an
announcement from a freshly loaded page clears the busy flag, because a page
that has just started is not folding. Measured: **212 ms fresh, 3,341 ms with
the page away, 135 ms once it is back** - the last number also being the bridge
restarting by itself on a reload.

🔴 **AND `threading.Lock` IS NOT REENTRANT, WHICH COST THREE INNOCENT ARMS.**
`_seen()` takes `MAIL_LOCK`, and one branch called it from INSIDE a `with
MAIL_LOCK` - so the first `head=1` request never returned and never released,
every later request queued behind it, and the gate failed three arms downstream
in a route that had nothing wrong with it. **A hang reported far from its
cause**: read the value before taking the lock.

`npm run test:colab` (`tools/check-colab-bridge.py`) is the gate, and it needs
**no GPU and no weights**. Over the wire: the announcement, a `ping` answered as
a `pong` (202-414 ms), both clocks, the watermark's idempotence, the 429, and a
token refusal on all five routes.

**AND TWO ARMS THAT NEEDED MORE THAN CURL.** The READER'S OWN PAGE is opened in
a second browser at `?backend=colab`, handed a sequence and clicked - with the
WEIGHTS BLOCKED on the runtime page (`Network.setBlockedURLs`, `*huggingface.co*`)
so the fold fails in seconds rather than pulling hundreds of megabytes. The
command arrives carrying its entity and its model, and the runtime's `Failed to
fetch` is what the reader's own status line ends up reading, three events
applied at a worst feed of 2-6 ms. *Its first run failed for a reason worth
keeping: a fresh profile has accepted no model terms, so the click opened the
terms dialog and nothing was sent - a page with a sequence, an enabled button
and a status line still reading "Ready. Paste a sequence and press Fold."* And
the FEED UNDER LOAD is the regression guard: twenty events from a page blocking
its thread in 300 ms chunks, bounded at a second, measured at 1-2 ms.

🔴 **AND A READER CAN ARRIVE IN THE MIDDLE OF A FOLD, which only the page that
pressed Fold used to survive.** A Colab fold is minutes long and the page in
front of it is an ordinary tab - reloaded, reopened from the notebook's link,
opened in a second window - and each of those left a reader watching nothing
while their own fold ran on, with the result landing in a page nobody was
looking at. `head=1` carries `folding`, so a page that opens during one
attaches to it from the CURRENT head (the rest of this fold, not a replay of
the session). `followRemoteFold` is the one loop both callers share, and the
attached reader deliberately gets no Stop: they did not start it.

**THE ARM HOLDS THE FOLD BY TAKING `/out` AWAY FROM THE RUNTIME PAGE**, rather
than folding something slow - the broker raises `folding` when it ACCEPTS the
command, and a page that cannot collect its commands never finishes it, which
makes the state deterministic instead of a race against a real fold's first
seconds. A page opened then reads *"a fold is already running on the runtime -
following it"*.

🔴 **AND THAT HELD FOLD IS WHAT FINALLY REACHES THE FRAMES AND THE INGESTION,
which are the reader's own code at the end of every remote fold and were
covered by nothing.** With the fold held and a reader attached, three frames
are pushed from the runtime page through its own `tapOut` - four alpha carbons
that MOVE between frames, so "the frames arrived" cannot be satisfied by one
frame drawn three times - and then a `result`. The attached page draws **3 of
3** and ends reading its status with **4 positions drawn**, which is
`loadIntoViewer`, the sequence strip and the download buttons having been
handed a structure. No weights, no card, no fold: the transport carries a
structure that was written down in the gate. Mutated (frames collected but not
drawn): 0 of 3.

AND THE RUNTIME GOING AWAY is the ninth arm: the runtime page is navigated to
`about:blank` - a page with no `role` runs no bridge, which is exactly what a
recycled runtime looks like from here - and the heartbeat must age, then come
back when it is navigated home.

Four mutations caught: the push removed (the pulled version's behaviour - four
arms red), the arrival stamp dropped, the busy refusal removed, and the reader's
command never sent. What it cannot cover is a fold: the model never runs, and
what comes back is the page's own refusal.

### 🔴 "A new object with frames" could not see a SECOND fold

The watcher waited for an object name that was not on the page before the
click. That is true of a first fold and **false of every one after it**:
`openBlankFold` reuses the stem and rewinds it, so after two folds the page
held one object, `af3_1`. Measured: the second blocking fold timed out after
120 s while its own status line already read *"AlphaFold 3 · 13 residues · in
1 s (trunk reused)"* - a fold that had finished in a second, reported as a
failure.

The button is no good either (it stays enabled throughout - it is how you stop
one), and a status line without a percentage is true *before* the click has
been acted on. The page states it instead: `window.__foldState = {running,
since}`, written at the top of `fold()` and in its `finally`, compared against
a click time taken from **the page's own clock**. Two blocking folds back to
back now answer in 11.7 s and 1.05 s. That watch lives in `runFold`
(`web/colab-bridge.js`) now, where both clocks are the same one.

## The Colab setup minute, measured on a T4 - and the three ways of cutting it that do not work

Driven from `colab exec` on a real runtime rather than guessed at. The whole
setup is **71 s**, and it is dpkg unpacking onto a **2-CPU** box:

| stage | time | share |
|---|---|---|
| Chrome `.deb` download, 136 MB | **0.65 s** | 1% |
| Chrome `dpkg` install | 21.4 s | 30% |
| **`libnvidia-gl-<major>`** | **44.3 s** | **62%** |
| `git clone --depth 1` | 5.1 s | 7% |

The download is free - 209 MB/s, it is Google's own network - so **every
second is archive extraction**, and nothing about the network or the repository
is worth optimising.

🔴 **AND THERE IS NOTHING ON THE IMAGE TO REUSE.** Taken before anything was
installed: no browser of any kind (no Chrome, no Chromium, no Playwright
cache), an EMPTY `/usr/share/vulkan/icd.d`, and no `libGLX_nvidia` in the
linker cache. What IS there: `libvulkan.so.1` (the loader), `libcuda.so.1`,
`nvidia-smi`, node v20.19, and a populated apt index. So the nicest hypothesis
- that the driver's libraries are on disk and only the ICD *file* is missing,
which would have turned 44 s into writing one JSON - is false.

### Three measured dead ends

- 🔴 **`--force-unsafe-io` IS A NULL RESULT.** dpkg fsyncs every file it
  writes and this box's disk is slow, so it looks like the answer. Same
  package, same box, reinstall either way: **18.4 s plain against 16.7 s**,
  and Chrome **17.1 against 17.6** - the wrong way round. Inside the noise
  both times.
- 🔴 **"AN EXTRACTED CHROME HAS NO WEBGPU" WAS MY PROBE, NOT THE PACKAGING -
  AND THE CONTROL IS WHAT SAID SO.** `dpkg-deb -x` unpacks Chrome in **9.6 s**
  against 17-21 s through dpkg (it needs four small libraries the image lacks:
  `libatk-1.0`, `libatk-bridge-2.0`, `libatspi`, `libXcomposite`, 7.6 s), and
  the adapter check then reported `navigator.gpu` **undefined**. I wrote that
  up as extraction breaking WebGPU, with an isolation to match. Both were
  wrong: running the SAME check with **both halves properly apt-installed**
  also reported `webgpu: false`, which is the configuration that has been
  folding on Colab all along.
  **`about:blank` IS NOT A SECURE CONTEXT.** `cdp.launch` leaves the browser
  on it, `navigator.gpu` does not exist off a secure origin, and every
  "webgpu: false" in this section was that. Navigate to a served page first
  and the same browser answers **nvidia / turing / shader-f16** with
  `isSecureContext: true`. The two readings, one page apart, same process:

  | page | answer |
  |---|---|
  | `about:blank` | `{webgpu: false, secure: false}` |
  | `http://127.0.0.1:8799/index.html` | `{vendor: nvidia, architecture: turing, f16: true}` |

  🔴 **A PROBE THAT CANNOT SAY YES CANNOT SAY NO.** Three experiments were
  scored against it and one of them was committed as a finding. The rule this
  file keeps relearning, in its sharpest form yet: **an instrument needs a
  positive control before its negatives mean anything** - here, one known-good
  configuration answering `nvidia` would have caught it before the first
  conclusion was drawn.

- 🔴 **AND WITH A SOUND PROBE, EXTRACTING THE DRIVER IS REJECTED ON ITS
  MERITS.** Re-run against a served page, the timing half is real: the driver
  downloaded and unpacked by hand takes **13.6 s and hides entirely underneath
  Chrome's dpkg**, because it never touches the dpkg lock - **36.8 s total
  against 71.4**. The GPU half fails, three ways, each measured:

  | the extracted driver, reached by | adapter |
  |---|---|
  | `VK_ICD_FILENAMES` + `LD_LIBRARY_PATH` | `null` |
  | copied into `/usr/lib` + `/usr/share/vulkan/icd.d`, `ldconfig` (3.7 s) | `null` |
  | ...plus `libnvidia-gpucomp` and `libnvidia-compute` extracted too, `ldd` clean | `null` |

  `apt-get download` fetches ONE package, and `libGLX_nvidia.so.0` needs
  `libnvidia-gpucomp.so` from another - which `ldd` named exactly, and
  supplying it changed nothing. `webgpu: true` throughout, so the browser and
  the probe are fine; it is `requestAdapter()` that returns nothing. Whatever
  else `dpkg` does for this driver, a file copy does not reproduce it.
  **So the 44 s is not avoidable by unpacking it differently**, and the ~35 s
  version of this setup does not fold.

- **Skipping what is already present buys nothing on a FIRST run**, which is
  the run people complain about. The probes are still right for a rerun - a
  runtime keeps its filesystem between cells - but on a cold T4 the ICD
  directory is empty and Chrome is absent, so every branch fires.

🔴 **AND THE ANSWER WAS A ZIP, NOT A LIGHTER BROWSER.** Chrome for Testing
ships the same binaries as a plain archive, so the 21.4 s dpkg becomes
**1.8 s of download and 11 s of unzip** - and `chrome-headless-shell`, which
is the half this backend uses, is **261 MB against 393**. Both report
**nvidia / turing / shader-f16** through a served page: a zip is not a lesser
browser, it is the same browser without a package manager. What a `.deb`
brought as dependencies has to be asked for once - `ldd` named exactly four
(`libatk-1.0`, `libatk-bridge-2.0`, `libatspi`, `libXcomposite`) - and they
join the driver's apt call, which now runs UNDERNEATH the unzip instead of
after a dpkg.

Measured end to end on a clean T4, the cell's own script followed by a real
fold: **setup 36.5 s against 71.4**, service ready at 40.5 s, adapter
`nvidia / turing` with `shaderF16` and `subgroupMatrix` both true, and 13
residues folded to 25 frames and 110 atoms. **The 44 s driver install is
still there and is still the floor** - it is the one thing that must go
through dpkg, and the three ways of extracting it all end at
`requestAdapter() === null`.

🔴 **AND THE DRIVER IS THE FLOOR BECAUSE WEBGPU ON LINUX IS VULKAN.** Chrome
reaches the card through Dawn -> Vulkan, and NVIDIA's Vulkan ICD lives inside
`libGLX_nvidia.so.0` with its `nvidia_icd.json`. A Colab runtime ships the
kernel module, `nvidia-smi` and `libcuda` - the COMPUTE userspace, which is
what CUDA uses - and an empty `/usr/share/vulkan/icd.d`. There is no CUDA path
into WebGPU, so the graphics userspace has to be installed. That is what the
44 s buys.

🔴 **AND INSTALLING LESS OF IT SAVES NOTHING, MEASURED.** Unpacked the package
is **447 MB and 83% of it is not Vulkan**: `libnvoptix` + `nvoptix.bin`
(166 MB) and `libnvidia-rtcore` (105 MB) are ray tracing, `libnvidia-present`
(66 MB) is presentation, `libnvidia-eglcore` (35 MB) is EGL, `libnvidia-vksc-core`
is Vulkan SC. Excluded with `dpkg --path-exclude` - and the excludes provably
applied, those files absent afterwards while `libnvidia-glvkspirv` stayed and
the adapter still reported **nvidia / turing / f16 / subgroup-matrix** - the
install took **44.0 s against 44.3**. Not one second.
**Because dpkg DECOMPRESSES the whole archive either way**; `--path-exclude`
declines to WRITE what comes out of the stream, and the cost here is xz on two
cores. The dependencies say the same thing from the other side:
`libnvidia-compute-580` is 335 MB and `libnvidia-gpucomp-580` 70 MB, both
pulled by `libnvidia-gl`, both bigger than the file savings being chased.

🔴 **AND THE DRIVER IS UNPACKED, NOT INSTALLED: 44.3 s BECOMES 15, AND THE
WHOLE SETUP IS 22.3 s.** The working driver's own memory map is what found
it - it loads from **`/usr/lib64-nvidia/`**, Colab's own directory, already on
the image at **580.82.07, the version the kernel module is**. What Colab does
NOT ship there is a `libGLX_nvidia` carrying the Vulkan ICD entry point, and
the loader says so exactly:

```
loader_scanned_icd_add: Could not get 'vkCreateInstance' via
'vk_icdGetInstanceProcAddr' for ICD /usr/lib64-nvidia/libGLX_nvidia.so.0
```

So one package has to be fetched - and `dpkg-deb -x` it into `/` is **11.5 s
where apt takes 44.3**, because apt also unpacks `libnvidia-compute` (335 MB)
and `libnvidia-gpucomp` (70 MB), **which nothing here ever loads**. The ICD and
layer json come out of the package, so nothing is hand-written.

🔴 **INTO `/`, AND WITH NO `LD_LIBRARY_PATH`, BECAUSE THE WORKING STACK IS
MIXED.** This is the whole reason three earlier attempts failed and it is not
obvious: the ICD is the package's 580.178.04 `libGLX_nvidia`, and its
companions are **Colab's 580.82.07**, resolved by the linker the way a real
install resolves them. Force the extracted tree onto the library path and every
library becomes 178.04 - a single consistent version, against an 82.07 kernel -
and the driver answers `ERROR_INCOMPATIBLE_DRIVER / Found no drivers!`. The
mismatch is not the bug; it is the configuration that works.

Four attempts, and each failed differently, which is why the order matters:

| attempt | result |
|---|---|
| register Colab's own libraries, download nothing (11.2 s) | no Vulkan entry point in that build |
| extract to `/opt`, point the ICD at it by env | `Could not get vkCreateInstance` |
| ...with an ABSOLUTE `library_path` and companions extracted | `ERROR_INCOMPATIBLE_DRIVER` |
| **extract the package into `/`, `ldconfig`, no env** | **nvidia / turing / f16 / subgroup-matrix, device created** |

Measured end to end on a clean T4, the cell's own script then a real fold:
**setup 22.3 s**, service ready 26.3 s, and 13 residues folded to 25 frames.
**71.4 -> 36.5 (the Chrome zip) -> 22.3 (the driver unpack).**

**What is left, and it is small**: the clone shares nothing with apt (one is
git's network, the other dpkg's lock), so it is started first and waited for
last - about 5 s of 71. The honest summary is that ~65 s of this is dpkg
unpacking half a gigabyte on two cores, and no arrangement of the same
packages avoids it.

🔴 **AND `colab exec` IS SERIAL AND ITS CLIENT CAN HANG.** A Jupyter kernel
runs one cell at a time, so a probe whose client hangs blocks every later
call - three experiments in this session appeared to run and never did, and
the wall-clock was blamed on the runtime. Long work goes **detached**
(`subprocess.Popen(..., start_new_session=True)` writing timings to a file)
and is read with cheap polls; the first `exec` against a new session pays
~29 s of kernel connection, every later one is ~3 s.

## `bondedAtomPairs`: a covalent inhibitor, bonded

Three of AlphaFold 3's fourteen example jobs were refused for declaring covalent
bonds, and the KRAS one is why it matters: **sotorasib IS a covalent inhibitor**,
bonded to cysteine 12. Folding it beside its target rather than attached to it
is a different answer, and pLDDT will not tell you which you got.

Measured through the page, same job, same seed, 50 steps:

| | SG(CYS 12) - C25(sotorasib) |
|---|---:|
| `bondedAtomPairs` as the file declares it | **1.62 A** |
| the same job with that field deleted | **6.25 A** |

A C-S bond is about 1.81 A. The control is the point: the ligand is genuinely
attached, not merely nearby.

🔴 **AN ENDPOINT IS ADDRESSED BY `asymId`, WHICH IS ONE NAMESPACE OVER POLYMERS
AND LIGANDS** - the same namespace the file's chain letters are in. Polymer
chains take 0..chains-1 in order and each ligand the next, which is exactly what
`expandEntities` and `featuriseProtein` already do, so one number is enough and
the reader can compute the letter map itself. The letters are carried on the
entities only as far as that map and then deleted, rather than becoming a second
source of truth about what a chain is called.

🔴 **AND AN ATOM NAME RESOLVES TO A TOKEN, NOT TO AN ATOM.** `token_bonds` is
token x token: for a standard residue that is its single token whichever atom
was named, and for a ligand or an atomised residue it is the token carrying that
atom. Reading the name as an index would bond whatever happened to sit there.

🔴 **A POLYMER-TO-POLYMER BOND IS READ AND NOT SENT, AND THE JOB IS TOLD.**
AlphaFold 3 extracts token bonds only where one side is a LIGAND -
`get_polymer_ligand_and_ligand_ligand_bonds` - so a disulfide between two
cysteines is absent from `token_bonds` there too. Keeping it would invent a
feature the reference does not have; dropping it silently is the failure this
file exists to avoid, so it becomes a note on the load.

**The corpus goes nine to ten**, and the two that did not move revealed their
next reason rather than losing one: `rnaseb_glycosylated` asks for a
five-component glycan in a single ligand entry, and `alphafold_input` carries
its alignment inline. `modified_rna` and `methylated_dna` are unchanged - that
is the modified-BASES gap, which is separate.

🔴 **AND THE PAGE HOLDS THESE BONDS BY CHAIN POSITION, WHICH EDITING THE ROWS
CAN INVALIDATE.** What protects it is that resolution is by atom NAME: change
the sequence so residue 12 is no longer a cysteine and the featuriser throws
"no atom SG" rather than bonding whatever now sits there. The residual is an
edit that leaves the same atom at the same position, which is narrow and loud
everywhere else. Said here rather than guarded with machinery, after the job
NAME's own comparison was built and then removed for not earning its place.

### What a remote fold was losing: the end of the trajectory, and everything but coordinates

Two faults, both found by watching one run rather than by a test.

🔴 **THE LAST FRAMES WERE COLLECTED AND THROWN AWAY.** The watch loop drains
the page's tap as it polls and then BREAKS - so every frame the sampler emitted
between the final poll and the end of the fold never travelled. It is the END
of the trajectory that goes, which is the part worth watching. The tap is
drained after the loop now, **and again after the readback**, because
`loadIntoViewer` runs at the end of a local fold and the readback WAITS for it:
the frames it re-adds and the status line it writes are emitted inside that
wait. Measured on a 109-residue fold with a real MSA search: **25 frames of 25**,
beside 254 status and 241 progress events.

🔴 **AND THE PAGE WAS HANDED COORDINATES AND NOTHING ELSE.** The readback only
ever clicked the download button, so `foldOnBackend` called
`loadIntoViewer({pdb, scores: {}})` - no alignment, no confidence, no scores
card. A remote fold looked like a structure with the rest of the page switched
off, which is most of what LocalFold shows about a fold.
`window.__lastPrediction()` exposes the object the LOCAL path already hands to
`loadIntoViewer` - a function rather than the value, because it is reassigned
on every fold and a captured reference hands back the one before - and the
same fold now returns **2.84 MB of a3m**, a confidence object carrying
`plddt`, `meanPlddt`, `ptm`, `iptm`, `chainPairIptm` and `chainPtm`, the scores
JSON and the chains.

**What this costs is the a3m's size**: it rides in the job result as text, and
a deep search is megabytes. Over the Colab proxy that is a local hop and it has
not been a problem; it is the first thing to compress if it becomes one.

### The download buttons, and a select that refused the mode it was given

🔴 **THE BUTTONS HAD NOTHING TO DOWNLOAD, AND FAILED TOWARDS THE WRONG
ANSWER.** `activePrediction()` reads `predictions.get(name)` and falls back to
`lastPrediction`, both written by the LOCAL fold paths - so after a remote fold
Download PDB was a silent no-op, or worse handed back whatever that tab had
folded BEFORE: the wrong structure, saved without a word. The runtime now sends
its prediction WHOLE, as a JSON string, and the client registers it under this
page's stem, which is the name the viewer knows the object by and therefore the
one `activePrediction` looks up.

**Whole, not field by field**, because the archive reads `stem`, `model`,
`settings`, `entities`, `msas`, `msaOrigin`, `confidence` and `chainLengths` -
naming them at the readback is the rebuild this repository has been bitten by
six times, and the failure is a zip with a piece missing. The replacer turns
typed arrays into arrays because a `Float32Array` crossing CDP's JSON becomes
`{"0":...}`, which every reader downstream sees as an object with no length.
Measured: **231,651 characters**, carrying every one of those fields.

🔴 **AND A SELECT SILENTLY REFUSES A VALUE IT HAS NO OPTION FOR.** `msaMode()`
maps the control's `none` to `single` for the code below it, and the remote
path was sending that RESOLVED mode - which the runtime then wrote into its own
`msa-mode` SELECT, whose options are `none`, `paste`, `search` and `upload`.
The assignment left the control EMPTY, and the fold threw *"unknown alignment
mode"* before any work began. **A reader folding with the default Single
Sequence setting hit this every time**; Search happened to work, which is why
the streaming test passed and this did not surface until a fold was driven
through the page rather than through curl. The raw control value travels now
and the runtime resolves it with the same function.

*Found by asking the page what it was showing - `status-message` said it in
words - rather than by reading the bridge code, which looked right.*
## `contact`: a bond as a ROW, not as hidden state

`bondedAtomPairs` landed first as a job-level field the page kept in a variable
- set when a job loaded, invisible thereafter, and silently invalidated by any
edit to the chains it named. That is precisely what the job NAME was built and
then removed for, and the same objection applies harder here, because a bond
that has quietly stopped applying changes the STRUCTURE rather than a label.

A `contact` row sits in the entity list beside the chains it names:

```
Protein          GWCTELEKH...
Ligand (CCD)     MOV
Contact (bond)   A12:SG - B1:C25
```

Visible, editable, and deleted when the reader deletes it. The spec reads the
way somebody would say it: chain letter, residue number, and an atom after a
colon. **The atoms are optional** - `token_bonds` is token x token, so for a
standard residue the atom decides nothing; it matters only for a ligand or an
atomised residue, and `featuriseProtein` refuses by NAME when it does.

Folded from the row, on AlphaFold 3's own KRAS/sotorasib example:

| | SG(CYS 12) - C25 |
|---|---:|
| the contact row | **1.62 A** |
| the same job, bond deleted | **6.25 A** |

and the archive writes it back as `bondedAtomPairs` in the open dialect - which
it must, because the server dialect has no field for one, the same reason a
SMILES ligand forces that branch.

🔴 **A CONTACT IS NOT A CHAIN, AND `expandEntities` WOULD HAVE MADE IT A
LIGAND.** That loop is `if polymer ... else if smiles ... else LIGAND`, so a row
type it has not heard of becomes a ligand - silently, with a chain in the fold
nobody asked for. It is the same shape as `setChains` deleting a SMILES row by
keeping only `type === "ligand"`, which this file already records. `CHAIN_TYPES`
and `isChainEntity` are where that question is asked now, and the gate for it
asserts a contact adds neither a chain nor a ligand.

🔴 **AND IT COST TWO OF EXACTLY THAT MISTAKE ON THE WAY IN.** The status line
read **"2 chains + 1 ligand"** for a protein, a ligand and the bond between
them, because `applyJob` counted every row that was not a ligand as a chain. And
the trunk key referenced `bonds` in a function that is handed chains and ligands
and never the rows - `bonds is not defined`, at fold time, in the browser. Both
are the row type falling into code that predates it.

🔴 **AND THE BOX FILTERED WHAT WAS TYPED INTO IT - THE THIRD ROW TYPE TO WALK
INTO THE SAME `else`.** The blur handler ends `entity.value =
cleanSequence(value.value)`, which keeps only amino-acid letters, so a contact
typed as `A12:SG - B1:C25` came back **`A:SGB:C`** the moment the reader clicked
away: the residue numbers, the colon's right-hand side, the spaces and the
hyphen all gone, and what was left still looked like a sequence. It is the
branch that turned benzene into hexane, with a 🔴 comment directly above it
about exactly this - and the comment did not stop it happening again.

Reported by a user, because nothing here could see it: `--smiles-ui` exists
precisely because the entity list's API cannot, and a contact had no equivalent.
`tools/fold-in-page.py --contact-ui` drives the dropdown, types, and fires the
BLUR, which is where the handler lives:

```
typed  A12:SG - B1:C25
stored A12:SG - B1:C25   kept: true
```

and with the branch removed, `stored A:SGB:C  kept: false`. It also checks the
box's `text-transform` is `none`, since borrowing `.entity-value-ligand` would
uppercase a contact on screen the way it would a SMILES.

**Crosslinks are not done.** af3x's form - a named linker between two residues,
which expands to the linker as a LIGAND plus two bonds - fits this row exactly
(`DSSO A53 - C66`), and the parser leaves room for it, but it needs a table of
linker codes and the atoms each attaches by. That is the next piece.