<!--
docs/DESIGN-PY2DMOL.md
----------------------
AI Context: WHAT py2Dmol ALREADY DOES THAT THE DESIGN PAGE NEEDS

🔴 THIS DESCRIBES `sokrypton/py2Dmol` AT COMMIT `1efde273`, NOT THIS
REPOSITORY. py2Dmol is ours (Beerware, so3@mit.edu), so where it is the limit
the answer is to extend it - but only what is actually missing, which is what
this audit establishes. Its paths are written as `src/...` because they are
HERS; the vendored bundle here is `web/vendor/py2Dmol.full.min.js`.

🔴 THE VENDORED COPY IS A MIRROR AND MUST NOT BE EDITED IN PLACE. Change it
upstream, rebuild its bundle, re-run `python3 tools/sync-py2dmol.py`, check with
`--check`. See `web/vendor/SOURCE.md`.

THE THREE ANSWERS, which docs/DESIGN.md section 5 asked for:
  1. per-residue colouring by an arbitrary category - EXISTS, no change needed
  2. a labelled frame API - PARTIAL: frames, stepping and the play bar exist;
     a per-frame LABEL is shown by the full app and not by the embed path
  3. residue picking the page can read - PARTIAL: picking, selection and a
     change event exist; the event carries NO detail payload, so a page reads
     `v.selection` (0-based indices) and translates them itself

HOW IT WAS MADE, AND HOW FAR IT HAS BEEN CHECKED. Produced by an `agy` worker
(Gemini 4 Argon, medium effort) reading her source over the GitHub API, then
spot-verified here on its load-bearing claim: `renderer.setColor(colour, where)`
is at `src/parts/embed.js:448` and does accept `{chain}`, `{positions}` (an
array or a Set) or `{range}`, writing an `advanced` colour onto the object -
so verdict 1 holds and role colouring needs nothing upstream. The other two
verdicts and every line citation are the worker's reading and are NOT
independently verified. Treat a claim here as a claim with a citation.

🔴 A SNAPSHOT, PINNED. py2Dmol moves; re-read before trusting a detail.
-->

# py2Dmol Capability Audit for Protein-Design Page

**Repository**: `github.com/sokrypton/py2Dmol`  
**HEAD Commit**: `1efde27334c9f57f24f34981ff87fc3e26c4bedd`

---

## 1. VERDICT TABLE

| # | Capability | Verdict | Exact API (if present) | Source Location (`file:line`) |
|---|---|---|---|---|
| **1** | **Per-residue colouring by an arbitrary category** | **EXISTS** | **JS Embed / Renderer**: `v.setColor(colour, where)` / `renderer.setColor(colour, where)`; `v.setSidechainColor(colour, sel)`; custom mode registry `window.py2dmol_customColors[name] = (atomIndex, renderer) => ({r,g,b})`; direct `object.color` / `frame.color` (`{ type: 'advanced', value: { chain, position } }`)<br>**Python**: `view.set_color(color, name=None, chain=None, position=None, frame=None)` | `src/parts/embed.js:448-499`<br>`src/core/mol.js:63-201`<br>`src/core/mol.js:853-952`<br>`src/core/mol.js:11038-11095`<br>`src/core/mol.js:20-22, 346-364`<br>`src/parts/sidechains.js:216-258`<br>`py2Dmol/viewer.py:3870-4143` |
| **2** | **A labelled frame API** | **PARTIAL** | **Multi-frame + play bar + stepping (EXISTS)**: `v.loadFrames(src, opts)`, `v.addFrame(frame, objectName)`, `v.replaceFrame(frame, objectName)`, `v.setFrame(frameIndex)`, `v.play()`, `v.pause()`, `py2Dmol.frameFromText(text, opts)`, `CustomEvent('py2dmol-frame-change', { detail: { frameIndex } })`<br>**Per-frame label (PARTIAL)**: `frame.name` is stored on frame objects and displayed in `#frameNameLabel` in the full web app (`src/app/main.js`), **but** the play bar counter (`#frameCounter` in `src/core/mol.js`) only renders `"${current} / ${total}"`, `src/parts/embed.js` does not create `#frameNameLabel`, and Python `view.add()` overwrites `data_dict["name"] = None`. | `src/parts/embed.js:121-167, 235-256, 574-596`<br>`src/core/mol.js:4211-4555, 5879-6012, 6366-6396`<br>`src/app/main.js:848, 1260-1275`<br>`src/app/session.js:93`<br>`py2Dmol/viewer.py:2133-2134, 3095` |
| **3** | **Residue picking that the page can read** | **PARTIAL** | **Click-selection, hit-testing, programmatic set/readback & change event (EXISTS)**: `renderer.pickResidueAt(clientX, clientY)`, `renderer.pickGroupAt(i)`, `v.select(sel)`, `v.unselect(sel)`, `renderer.selectTo(sel, mode)`, `renderer.setResidueSelection(positions)`, `renderer.clearResidueSelection()`, `v.selection` / `renderer.residueSelection` (`Set<number>`), `document` `CustomEvent('py2dmol-residue-selection-change')`<br>**What is missing / partial**: `py2dmol-residue-selection-change` is dispatched on `document` with **no `detail` payload** (neither viewer reference nor resolved `{object, chain, resSeq, resName}` records), so the page must read `v.selection` (0-based indices) and translate via `renderer.chains[i]`, `renderer.residueNumbers[i]`, `renderer.positionNames[i]`, and `renderer.ownerOf(i)`. Hover (`mousemove`) updates `renderer.setHover(...)` and optional global `window.SEQ.setHoveredResidue({object, chain, resName, resSeq})`, not a DOM event or per-viewer callback. | `src/core/mol.js:2270-2288, 2437-2494`<br>`src/core/mol.js:4582-4931`<br>`src/parts/embed.js:221, 569-634`<br>`src/parts/multi.js:206-225`<br>`src/panels/seq.js:2918-2928` |

---

## 2. CAPABILITY DETAILS

### Capability 1: Per-Residue Colouring by an Arbitrary Category (`EXISTS`)

`py2Dmol` does **not** restrict colouring to continuous built-in ramps (`plddt`, `rainbow`, etc.). It has a 5-level colour hierarchy (`position > chain > frame > object > global`) resolved by `resolveColorHierarchy(context, colorSpec)` (`src/core/mol.js:853-952`) and evaluated per residue in `getAtomColor(atomIndex, effectiveColorMode)` (`src/core/mol.js:11038-11095`). Crucially, an explicit literal colour at a higher specificity level (such as `position`) always overrides a colour mode (such as `'plddt'` or `'chain'`) at a lower level (`src/core/mol.js:879-943`, `11078-11092`).

There are **four real APIs** for painting arbitrary per-residue categories/roles:

#### 1A. JS Embed / Renderer Selector API: `v.setColor(colour, where)`
- **Location**: Defined in `src/parts/embed.js:448-499`; attached to both the `v` handle returned by `py2Dmol.show(el, opts)` (`src/parts/embed.js:583`) and `renderer.setColor` (`src/parts/embed.js:417`).
- **Signature**:
  ```js
  v.setColor(colour, where) -> v // returns the viewer handle for chaining
  ```
- **What `colour` accepts** (`src/parts/embed.js:449-473`):
  1. Any `#rrggbb` hex string (`colour[0] === '#'`).
  2. Any named colour string from `namedColorsMap` (`src/core/mol.js:585-590` — 30 names: `'red'`, `'salmon'`, `'orange'`, `'gold'`, `'yellow'`, `'lime'`, `'green'`, `'teal'`, `'cyan'`, `'sky'`, `'blue'`, `'navy'`, `'purple'`, `'magenta'`, `'pink'`, `'brown'`, `'olive'`, `'white'`, `'silver'`, `'grey'`, `'gray'`, `'charcoal'`, `'black'`, `'pymol_carbon'`, `'pymol_nitrogen'`, `'pymol_oxygen'`, `'pymol_sulfur'`, `'pymol_phosphorus'`, `'pymol_halogen'`, `'pymol_metal'`).
  3. An RGB object `{ r, g, b }` (`0..255`), which `setColor` converts to `#rrggbb` (`src/parts/embed.js:450-455`).
  4. Any valid colour mode name returned by `getAllValidColorModes()` (`src/core/mol.js:358-364`: `'auto'`, `'rainbow'`, `'chain'`, `'plddt'`, `'deepmind'`, `'hydrophobicity'`, `'entropy'`, `'object'`, `'ss'`, plus any key registered in `window.py2dmol_customColors`).
- **What `where` accepts** (resolved via `positionsFor(renderer, where)` at `src/core/mol.js:63-201` and `renderer._writeTargets(where)` at `src/parts/sidechains.js:59-77`):
  - `undefined` / `null`: sets the whole viewer's colour mode or resets `obj.color` (`src/parts/embed.js:474-480`).
  - `{ chain: 'A' }` (chain-only): writes `obj.color.value.chain['A'] = value` (`src/parts/embed.js:481-487`).
  - `{ chain: 'A', residues: [54, 56, 58] }` or `{ chain: 'A', range: [10, 30] }`: matches author PDB residue numbers (`renderer.residueNumbers`, which come from PDB cols 22–26 / mmCIF `auth_seq_id` via `src/io/parse.js`) intersected with `chain` (`src/core/mol.js:116-144`).
  - `{ positions: [0, 1, 5] }`, or a bare `Array<number>` / `Set<number>`: matches 0-based position indices (`src/core/mol.js:76-86, 128-131`).
  - `{ object: 'target', chain: 'B', residues: [101, 102] }`: scopes to a specific named object (`src/core/mol.js:110-115`, `src/parts/sidechains.js:60-71`).
  - `{ type: 'P' | 'L' | 'N' | 'D' | 'R' }`, `{ near: sel, angstroms: 5, sidechainsOnly: false }`, `{ not: sel }`: type, spatial proximity, and complement selectors (`src/core/mol.js:90-108, 132-135, 165-199`).
- **How it stores per-residue categories**: Each call with a residue/position selector merges into `g.object.color = { type: 'advanced', value: { chain: {...}, position: { [localPosIdx]: value } } }` (`src/parts/embed.js:488-493`) and triggers `renderer.render('embed.setColor')`. Calling `v.setColor(roleColor, roleSelector)` once per design role paints every category in a handful of calls.

#### 1B. Direct Per-Position Colour Map on `object.color` or `frame.color`
- **Location**: Consumed by `resolveColorHierarchy(context, colorSpec)` (`src/core/mol.js:853-952`) and `getAtomColor(atomIndex, effectiveColorMode)` (`src/core/mol.js:11038-11095`).
- **Shape**:
  ```js
  object.color = {
    type: 'advanced',
    value: {
      object: 'plddt', // optional base mode for unlabelled residues
      chain: { B: '#999999' }, // optional per-chain colour or mode
      position: { 0: '#e41a1c', 1: '#377eb8', 2: 'gold' } // 0-based object-local position index -> colour or mode
    }
  };
  ```
- Can also be set per frame (`frame.color = { type: 'advanced', value: { position: { ... } } }`), where `frame` specificity sits between `chain`/`position` and `object` (`src/core/mol.js:856-859, 11056-11066`).

#### 1C. Custom Colour Mode Function Registry: `window.py2dmol_customColors`
- **Location**: Initialized at `src/core/mol.js:20-22`, validated in `isValidColorMode(mode)` / `getAllValidColorModes()` (`src/core/mol.js:346-364`), and invoked per residue in `_calculateCustomColor(atomIndex, modeName)` (`src/core/mol.js:10992-11004`).
- **Signature**:
  ```js
  window.py2dmol_customColors[modeName] = function(atomIndex, renderer) {
    // atomIndex: 0-based drawn position index
    // renderer.chains[atomIndex]: chain ID string
    // renderer.residueNumbers[atomIndex]: author PDB residue number
    // renderer.positionNames[atomIndex]: 3-letter residue name
    // renderer.ownerOf(atomIndex): { name, object, local, off }
    return { r: 228, g: 26, b: 28 }; // 0..255 RGB object
  };
  ```
- Once registered, `modeName` is accepted everywhere a built-in mode is accepted (`v.setColor(modeName)`, `v.setSidechainColor(modeName, sel)`, or `#colorSelect` dropdown).

#### 1D. Python API: `view.set_color(...)`
- **Location**: `py2Dmol/viewer.py:3870-4143` (also `view.add(..., color=...)` at `py2Dmol/viewer.py:363-407`).
- **Signature**:
  ```python
  view.set_color(color, name=None, chain=None, position=None, frame=None)
  ```
- **Per-residue data accepted**:
  - `view.set_color({0: "red", 1: "#377eb8", 2: "grey"}, position=True, name="design")` (`py2Dmol/viewer.py:3918, 3994-4000`).
  - `view.set_color("#e41a1c", position=[53, 55, 57], name="design")` or `position=(10, 30)` (`py2Dmol/viewer.py:4001-4033`).
  - **Note on Python vs JS**: In Python `set_color`, `position` takes **0-based position indices** (not author PDB residue numbers) and passing both `chain` and `position` raises `ValueError("Cannot specify both 'chain' and 'position'...")` (`py2Dmol/viewer.py:3963-3972`). In JS `v.setColor(colour, where)`, `{ chain: 'A', residues: [54, 56, 58] }` natively intersects chain and author residue numbers.

#### Short Example Calls (Real API)
```js
// JS Embed API: colour residues by design role using chain + author residue numbers
const v = py2Dmol.show('#viewer', { style: 'cartoon' });
v.load(pdbText, { name: 'complex' });

// 1. Target chain held fixed in grey
v.setColor('grey', { chain: 'B' });

// 2. Binder scaffold defaults to pLDDT, then override specific roles on chain A:
v.setColor('plddt', { chain: 'A' });
v.setColor('#2ca02c', { chain: 'A', range: [1, 25] });          // held fixed
v.setColor('#ff7f0e', { chain: 'A', residues: [32, 35, 39] });  // motif
v.setColor('#d62728', { chain: 'A', residues: [54, 56, 58] });  // tied / hotspot

// Or hand a per-residue array (length = N positions) via a custom colour mode:
const rolePalette = {
  fixed:    { r: 150, g: 150, b: 150 },
  redesign: { r:  55, g: 126, b: 184 },
  motif:    { r: 255, g: 127, b:  14 },
  tied:     { r: 228, g:  26, b:  28 },
  target:   { r:  77, g: 175, b:  74 },
};
// Suppose roleByResidue is an array of role strings per 0-based position:
window.py2dmol_customColors.design_role = (atomIndex, renderer) =>
  rolePalette[roleByResidue[atomIndex]] || { r: 200, g: 200, b: 200 };
v.setColor('design_role');
```

---

### Capability 2: A Labelled Frame API (`PARTIAL`)

#### What Exists: Multi-Frame Storage, Stepping, Live Streaming, and Play Bar UI
1. **`py2Dmol.frameFromText(text, opts)`** (`src/parts/embed.js:121-167`, exposed on `window.py2Dmol.frameFromText` at `src/parts/embed.js:1049`):
   - **Signature**: `py2Dmol.frameFromText(text, opts = {}) -> FrameObject`
   - Parses PDB or mmCIF text via `parseStructureText(String(text), { format: opts.format, caOnly: !opts.sidechains })` (`src/parts/embed.js:122-123`) and returns the first model as a frame object `{ coords, chains, plddts, residue_numbers, position_names, position_types, sidechains, ... }`.
2. **`v.load(src, opts)`** (`src/parts/embed.js:527-573`):
   - **Signature**: `v.load(src, opts = {}) -> v`
   - If `src` is a multi-model PDB (`MODEL` / `ENDMDL`) or multi-model mmCIF, `framesFromText(src, opts)` (`src/parts/embed.js:121-158`) parses every model into an array of frames and loads them into a single object (`opts.name || 'mol'`), showing the play bar automatically when `frames.length > 1`.
3. **`v.loadFrames(src, opts)`** (`src/parts/embed.js:574-588`):
   - **Signature**: `v.loadFrames(src, opts = {}) -> v`
   - **Accepts**: `src` can be an array of PDB/mmCIF strings or an array of pre-parsed frame objects; `opts` accepts `{ name, format, sidechains, align }` (plus style options).
   - **Behaviour**: Calls `renderer.newObject(name)` (`src/core/mol.js:4144-4176`), then loops over every item calling `renderer.addFrame(frame, name)` (`src/core/mol.js:4211-4471`). By default `frame.align !== false` (`src/core/mol.js:4253`), so `addFrame` Kabsch-aligns each new frame onto the previous frame unless `align: false` is set on the frame.
4. **`v.addFrame(frame, objectName)` and `v.replaceFrame(frame, objectName)`** (`src/parts/embed.js:593-596`, delegating to `renderer.addFrame` at `src/core/mol.js:4211-4471` and `renderer.replaceFrame` at `src/core/mol.js:4498-4555`):
   - **Signatures**:
     ```js
     v.addFrame(frame, objectName) -> v
     v.replaceFrame(frame, objectName) -> v
     ```
   - Appends a frame to `objectsData[targetName].frames` (or replaces the last frame in place while holding camera centre/extent constant), updates the play bar slider/counter, and renders.
5. **Play bar UI & Stepping (`v.setFrame`, `v.play`, `v.pause`)**:
   - `buildControls(host)` (`src/parts/embed.js:235-256`) creates `#controlsContainer` with `#playButton`, `#frameSlider`, `#frameCounter`, and `#speedButton` (enabled unless `opts.controls === false`).
   - `v.setFrame(frameIndex, skipRender = false)` (`src/parts/embed.js:592`, `src/core/mol.js:5879-6012`): steps to 0-based `frameIndex`, updates `#frameSlider` and `#frameCounter`, and dispatches on `document`:
     ```js
     document.dispatchEvent(new CustomEvent('py2dmol-frame-change', {
       detail: { frameIndex: frameIndex }
     })); // src/core/mol.js:5997-5999
     ```
   - `v.play()` / `v.pause()` (`src/parts/embed.js:590-591`, calling `renderer.startAnimation()` / `renderer.stopAnimation()` at `src/core/mol.js:7054-7127`).

#### What Is Partial: Per-Frame Labels / Captions (`frame.name`)
- **Where `frame.name` ALREADY works**:
  - `renderer.addFrame(data, objectName)` pushes `data` directly onto `object.frames` (`src/core/mol.js:4301`), so any `.name` string on `data` is stored on `object.frames[i].name`.
  - `_subsetFrames` preserves `name: frame.name` (`src/core/mol.js:5513`).
  - Session export `buildViewerState` preserves `if (frame.name) frameData.name = frame.name` (`src/app/session.js:93`).
  - Python `_create_light_frame` preserves `if frame.get("name") is not None: light_frame["name"] = frame["name"]` (`py2Dmol/viewer.py:2133-2134`).
  - In the **standalone web application** (`index.html` + `src/app/main.js`), `updateFrameNameLabel()` (`src/app/main.js:1260-1275`, wired to `py2dmol-frame-change` at `src/app/main.js:848` and called in `src/app/objects.js:385, 481, 497`) reads `frames[idx].name` and displays it in `#frameNameLabel` inside the sequence viewer header when `frames.length > 1`.
- **Where `frame.name` is MISSING**:
  1. **Play bar UI (`#frameCounter` in `src/core/mol.js:6366-6375`)**: `updateFrameCounter(current, total)` unconditionally writes `this.frameCounter.textContent = `${total > 0 ? current : 0} / ${total}`` (`src/core/mol.js:6372`). It never reads `frame.name`.
  2. **Embed UI (`src/parts/embed.js:235-256`)**: `buildControls(host)` creates `#frameCounter` in the play bar and does not create `#frameNameLabel` or wire `updateFrameNameLabel`.
  3. **Embed helpers (`src/parts/embed.js:121-167, 574-588`)**: `frameFromText(text, opts)` does not copy `opts.name` or `opts.label` onto the returned frame, and `loadFrames(src, opts)` does not accept a `labels` array when `src` is an array of PDB strings (though callers passing pre-built frame objects can attach `.name` themselves).
  4. **Python `view.add()` (`py2Dmol/viewer.py:3095`)**: `add()` treats its `name` parameter as the *object* name and explicitly clears the frame name: `data_dict["name"] = None  # Don't set frame-level name; use object name instead` (`py2Dmol/viewer.py:3095`).

#### Short Example Call (Real API Today)
```js
const v = py2Dmol.show('#viewer', { controls: true, style: 'cartoon' });

// Build frame objects and attach .name (preserved on object.frames[i].name)
const f0 = Object.assign(py2Dmol.frameFromText(cycle0Pdb), {
  name: 'cycle 0 - initial seed',
  align: false,
});
const f1 = Object.assign(py2Dmol.frameFromText(cycle1Pdb), {
  name: 'cycle 1 - folded by boltz2',
  align: true, // Kabsch-aligned onto cycle 0
});

v.loadFrames([f0, f1], { name: 'design_run' });

// Stream another cycle later:
const f7 = Object.assign(py2Dmol.frameFromText(cycle7Pdb), {
  name: 'cycle 7 - folded by boltz2',
});
v.addFrame(f7, 'design_run');
v.setFrame(2); // step to cycle 7 (0-based index 2)

// Read back the label on frame step today:
document.addEventListener('py2dmol-frame-change', (e) => {
  const idx = e.detail.frameIndex;
  const obj = v.renderer.objectsData[v.renderer.currentObjectName];
  const caption = obj?.frames?.[idx]?.name || `Frame ${idx + 1}`;
  console.log('Now showing:', idx, caption);
});
```

---

### Capability 3: Residue Picking That the Page Can Read (`PARTIAL`)

#### What Exists: 3D Canvas Hit-Testing, Click Selection, Programmatic Set/Readback, and Selection Event
1. **3D Canvas Click Selection (`src/core/mol.js:2437-2494`, enabled by default in `embed.js:221`)**:
   - In `py2Dmol.show(el, opts)`, `renderer.selectionEnabled = opts.select !== false;` (`src/parts/embed.js:221`).
   - On `canvas` `mouseup` (`src/core/mol.js:2457-2494`), if the pointer moved `< 4` px and was not panning (`!this.isPanning && moved < 4` at `src/core/mol.js:2466`):
     - Calls `const i = this.pickResidueAt(e.clientX, e.clientY)` (`src/core/mol.js:2473`).
     - If `i < 0` (clicked empty background): calls `this.clearResidueSelection()` (`src/core/mol.js:2474-2476`).
     - If `e.shiftKey`: toggles all positions in `this.pickGroupAt(i)` in `this.residueSelection` and calls `this.setResidueSelection(next)` (`src/core/mol.js:2477-2489`).
     - Otherwise (plain click): replaces selection with `this.setResidueSelection(new Set(this.pickGroupAt(i)))` (`src/core/mol.js:2490-2492`).
   - On `canvas` `dblclick` (`src/core/mol.js:2437-2455`): calls `pickResidueAt(e.clientX, e.clientY)`, finds all positions belonging to the same object and chain (`src/core/mol.js:2442-2453`), and selects the whole chain via `this.setResidueSelection(next)`.
2. **Low-Level Hit-Testing API (`renderer.pickResidueAt` & `renderer.pickGroupAt`)**:
   - **Locations**: `src/core/mol.js:4582-4754` (`pickResidueAt`) and `src/core/mol.js:4775-4838` (`pickGroupAt`).
   - **Signatures**:
     ```js
     renderer.pickResidueAt(clientX, clientY) -> number // 0-based position index, or -1 if background
     renderer.pickGroupAt(i) -> Array<number>           // [i] for a residue, [owner] for a side-chain atom, connected component for a ligand, [idx1, idx2] for a contact
     ```
   - Hit-tests frontmost backbone segments, lone atoms, side-chain sticks (mapped back to their parent residue via `this.sidechainMap.get(i).owner` at `src/core/mol.js:4797-4798`), and nucleic acid base plates, respecting the active clip slab (`src/core/mol.js:4608, 4768-4773`).
3. **Programmatic Selection Set & Readback (`v.select`, `v.unselect`, `v.selection`, `renderer.selectTo`, `renderer.setResidueSelection`, `renderer.clearResidueSelection`)**:
   - **Embed API (`src/parts/embed.js:569-634`)**:
     ```js
     v.select(sel)   -> v          // relative add: adds positionsFor(renderer, sel) to selection; v.select() with no arg selects all
     v.unselect(sel) -> v          // relative remove: removes positionsFor(renderer, sel); v.unselect() with no arg clears selection
     v.selection     -> Set<number> // getter returning new Set(renderer.residueSelection || []) (0-based position indices)
     ```
   - **Renderer API (`src/core/mol.js:4893-4931`)**:
     ```js
     renderer.selectTo(sel, mode = 'replace') -> void // mode: 'replace' | 'add' | 'remove'; resolves sel via positionsFor(this, sel)
     renderer.setResidueSelection(positions)  -> void // accepts Set<number> or Array<number>; stores Set or null on renderer.residueSelection
     renderer.clearResidueSelection()         -> void // sets renderer.residueSelection = null
     ```
   - Note that in `embed.js:591-600`, `setResidueSelection` and `clearResidueSelection` are wrapped so that every selection change (whether from a canvas click or a programmatic call) automatically calls `renderer.render('embed.selection')` to draw the yellow selection halo (`#ffd400`).
4. **Selection Change Event (`py2dmol-residue-selection-change`)**:
   - **Location**: Dispatched in `setResidueSelection` (`src/core/mol.js:4920-4922`) and `clearResidueSelection` (`src/core/mol.js:4928-4930`):
     ```js
     document.dispatchEvent(new CustomEvent('py2dmol-residue-selection-change'));
     ```
5. **Translating 0-Based Position Indices to `(object, chain, resSeq, resName)`**:
   - On `renderer`, loaded arrays are indexed by the 0-based drawn position index `i`:
     - `renderer.chains[i]`: chain ID string (e.g. `'A'`).
     - `renderer.residueNumbers[i]`: author PDB residue number (`number`, e.g. `54`).
     - `renderer.positionNames[i]`: 3-letter residue name (e.g. `'ALA'`).
     - `renderer.positionTypes[i]`: `'P'` (protein), `'D'`/`'R'`/`'N'` (nucleic), `'L'` (ligand).
     - `renderer.ownerOf(i)` (`src/parts/multi.js:206-225`): returns `{ name, object, local, off }` identifying which object owns drawn index `i` and its 0-based `local` index within that object.

#### Why It Is `PARTIAL` (What Is Missing for a Design Page)
1. **Empty Event Payload**: `new CustomEvent('py2dmol-residue-selection-change')` (`src/core/mol.js:4921, 4929`) passes **no `detail` object** — not even which viewer dispatched it, which residue was just clicked, or the resolved `{ object, chain, resSeq, resName, position }` list.
2. **0-Based Indices Only on `v.selection`**: `v.selection` (`src/parts/embed.js:618`) returns a `Set<number>` of 0-based position indices. The page must reach into `v.renderer.chains[i]` and `v.renderer.residueNumbers[i]` to get chain + PDB residue numbers.
3. **Plain Click Replaces vs. Accumulates**: On the 3D canvas, a plain click calls `this.setResidueSelection(new Set(this.pickGroupAt(i)))` (`src/core/mol.js:2491`), which **replaces** the selection unless the user holds `Shift` (`e.shiftKey` at `src/core/mol.js:2477`). (By contrast, clicking residues in the 1D sequence strip in `src/panels/seq.js:2690` toggles them additively without `Shift`.) If a user clicks hotspots one by one in 3D without holding `Shift`, each click replaces the previous hotspot unless the page either accumulates clicks in its own state or configures additive click behaviour.
4. **Hover Is Not an Event**: Hovering the 3D canvas (`mousemove` at `src/core/mol.js:2270-2288`) draws the built-in bottom-right HUD pill (`"A ALA 54"`) via `this.setHover(...)` and calls `window.SEQ.setHoveredResidue({ object, chain, resName, resSeq })` if `window.SEQ` exists (`src/core/mol.js:2275-2284`). It does not dispatch a DOM event or invoke an embed callback.

#### Short Example Call (Real API Today)
```js
const v = py2Dmol.show('#viewer', { select: true, style: 'cartoon' });
v.load(pdbText, { name: 'target' });

// Programmatically set initial hotspots (Chain A, PDB residues 54, 56, 58):
v.select({ chain: 'A', residues: [54, 56, 58] });

// Helper to read back current selection as [{ object, chain, resSeq, resName, position }]
function readSelectedResidues(viewer) {
  const r = viewer.renderer;
  return Array.from(viewer.selection, (i) => ({
    position: i,
    object: r.ownerOf ? r.ownerOf(i)?.name : r.currentObjectName,
    chain: r.chains[i],
    resSeq: r.residueNumbers[i],
    resName: r.positionNames[i],
  }));
}

// Subscribe to 3D canvas clicks / selection changes:
document.addEventListener('py2dmol-residue-selection-change', () => {
  const picked = readSelectedResidues(v);
  console.log('Selected residues:', picked);
  // e.g. [{ position: 53, object: 'target', chain: 'A', resSeq: 54, resName: 'TYR' }, ...]
});
```

---

## 3. SMALLEST CHANGES FOR `PARTIAL` CAPABILITIES

### For Capability 2 (Labelled Frame API — `PARTIAL`)

To make per-frame captions first-class in both the play bar UI and the embed/Python APIs while fitting how `py2Dmol` already works:

1. **Carry `name` in `frameFromText` and `loadFrames` (`src/parts/embed.js`)**:
   - In `frameFromText(text, opts)` (`src/parts/embed.js:164-167`), if `opts && typeof opts.frameName === 'string'` (or `opts.label`), set `f.name = opts.frameName` on the returned frame object.
   - In `loadFrames(src, opts)` (`src/parts/embed.js:574-588`), accept `opts.labels` (an `Array<string>` parallel to `src`): when converting item `k` of `src`, if `Array.isArray(opts.labels) && opts.labels[k] != null` and `!f.name`, assign `f.name = String(opts.labels[k])`.
2. **Display `frame.name` in the play bar (`src/core/mol.js` and `src/parts/embed.js`)**:
   - In `Pseudo3DRenderer.prototype.updateFrameCounter(current, total)` (`src/core/mol.js:6366-6375`), look up the active frame of `this.currentObjectName`:
     `const frame = this.objectsData[this.currentObjectName]?.frames?.[current - 1];`
     If `frame && frame.name`, format `this.frameCounter.textContent` (or an adjacent `#frameNameLabel` span inside `#controlsContainer` created by `buildControls` in `src/parts/embed.js:235-256`) as `` `${current} / ${total} — ${frame.name}` `` (or render `frame.name` in `#frameNameLabel` and widen `#frameCounter` from its fixed `width:46px` at `src/parts/embed.js:97`).
   - Include `name: frame?.name || null` in the `py2dmol-frame-change` `CustomEvent` detail in `setFrame` (`src/core/mol.js:5997-5999`):
     `detail: { frameIndex, name: object.frames[frameIndex]?.name || null }`.
3. **(If Python generation is used) Allow per-frame label in `view.add()` (`py2Dmol/viewer.py`)**:
   - In `viewer.add()` (`py2Dmol/viewer.py:2989, 3095`), add an optional `label: str | None = None` (or `frame_name: str | None = None`) argument and set `data_dict["name"] = label` at line `3095` instead of hardcoding `data_dict["name"] = None`. (`_create_light_frame` at `py2Dmol/viewer.py:2133-2134` already serializes `frame["name"]` into the JSON payload.)

### For Capability 3 (Residue Picking That the Page Can Read — `PARTIAL`)

To give the host page structured residue records and event callbacks without reaching into `renderer` internals:

1. **Add residue metadata helper & enrich `py2dmol-residue-selection-change` (`src/core/mol.js`)**:
   - Add a small method `Pseudo3DRenderer.prototype.describePositions(positions = this.residueSelection)` right beside `setResidueSelection` (`src/core/mol.js:4909-4931`) that maps each 0-based position index `i` to:
     ```js
     {
       position: i,
       object: this.ownerOf ? this.ownerOf(i)?.name : this.currentObjectName,
       chain: this.chains?.[i] ?? null,
       resSeq: this.residueNumbers?.[i] ?? null,
       resName: this.positionNames?.[i] ?? null,
       type: this.positionTypes?.[i] ?? null,
     }
     ```
   - In `setResidueSelection(positions)` (`src/core/mol.js:4921`) and `clearResidueSelection()` (`src/core/mol.js:4929`), pass `detail: { renderer: this, positions: this.residueSelection ? Array.from(this.residueSelection) : [], residues: this.describePositions() }` to `new CustomEvent('py2dmol-residue-selection-change', { detail })`.
2. **Expose `v.selectedResidues` and `onSelect` / `onPick` on the embed handle (`src/parts/embed.js`)**:
   - In `mount(el, opts)` (`src/parts/embed.js:569-634`), add a getter `get selectedResidues() { return renderer.describePositions(); }` alongside `get selection()`, and if `typeof opts.onSelect === 'function'`, invoke `opts.onSelect(renderer.describePositions(), renderer)` inside the `setResidueSelection` / `clearResidueSelection` wrapper (`src/parts/embed.js:591-600`).
   - Optionally, if `opts.additiveClick === true`, change the plain-click branch in `canvas` `mouseup` (`src/core/mol.js:2477`) from `if (e.shiftKey)` to `if (e.shiftKey || this.additiveSelect)` so single clicks toggle residues in and out of the selection without requiring `Shift`.

---

## 4. ANY EXISTING MECHANISM WE SHOULD USE INSTEAD

 auditing the codebase surfaced **four built-in mechanisms** that directly solve protein-design page problems and are much better than building custom page-side equivalents:

1. **Use `v.setColor(colour, where)` with selector objects instead of building per-residue index arrays by hand (`src/parts/embed.js:448-499`, `src/core/mol.js:63-201`)**:
   - You do **not** need to map PDB `(chain, resSeq)` to 0-based array indices in your page code. `positionsFor` (`src/core/mol.js:63-201`) already parses `{ chain: 'A', residues: [54, 56, 58] }`, `{ chain: 'A', range: [10, 30] }`, `{ near: { chain: 'B' }, angstroms: 5 }` (interface/epitope contact shells!), and `{ not: sel }`.
   - Even better: because `setColor` merges into `object.color.value.position` (`src/parts/embed.js:488-493`) and `resolveColorHierarchy` gives `position` overrides priority over `chain`/`object` modes (`src/core/mol.js:879-943`), you can leave the binder coloured by `'plddt'` and layer categorical role colours on top (`v.setColor('#ff7f0e', { chain: 'A', residues: motifRes })`).
2. **Separate Backbone Role Colour from Side-Chain Colour (`v.showSidechains` + `v.setSidechainColor`, `src/parts/sidechains.js:167-258`)**:
   - `py2Dmol` stores side-chain colours (`object.sidechainColor`) independently of backbone colours (`object.color`). You can colour the backbone by pLDDT and colour the side chains of hotspot/motif residues by their design role (or vice versa), and show side-chain sticks specifically on hotspot/epitope positions via `v.showSidechains({ chain: 'A', residues: [54, 56, 58] })`.
3. **Ghost Non-Design / Context Regions with `v.setOpacity(sel, alpha)` (`src/core/mol.js:4960-5002`, `src/parts/embed.js:615`)**:
   - Instead of colouring held-fixed or target context chains a dull grey that still occludes buried design positions, `v.setOpacity({ chain: 'B' }, 0.25)` uses ordered-dither fragment dropping on the GPU (`src/core/mol.js:4943-4956`) to fade the interior fill while keeping crisp outlines, with zero mesh rebuild cost.
4. **Built-in 1D Sequence Strip (`src/panels/seq.js`) and Selection Mark Styles (`src/parts/selectpanel.js:10-43`, `docs/SELECTION_MARK.md`)**:
   - If the page mounts the sequence panel (`window.SEQ` in `src/panels/seq.js`), clicking and dragging residues in the 1D sequence strip is already unified with 3D canvas picking through `renderer.residueSelection` and `py2dmol-residue-selection-change` (`src/panels/seq.js:786-826`). Notably, in the sequence strip, plain clicks **already** toggle residues additively (`src/panels/seq.js:2690`) without needing `Shift`.
   - For the 3D selection indicator, `renderer.selectionIndicator` (`src/core/mol.js:1765-1774`) can be switched between `'halo'` (default yellow outline `#ffd400`), `'tint'` (blends selection colour into the residue fill), and `'band'` (`renderer.selectionBand = true`).

---

## 5. PROVENANCE

- **Repository**: `https://github.com/sokrypton/py2Dmol`
- **HEAD Commit SHA**: `1efde27334c9f57f24f34981ff87fc3e26c4bedd`
- **Files Read**:
  1. `README.md`
  2. `CLAUDE.md`
  3. `docs/PLUGINS.md`
  4. `docs/SELECTION_MARK.md`
  5. `py2Dmol/viewer.py`
  6. `src/parts/embed.js`
  7. `src/parts/plugins.js`
  8. `src/parts/selectpanel.js`
  9. `src/parts/sidechains.js`
  10. `src/parts/multi.js`
  11. `src/parts/ui.js`
  12. `src/app/main.js`
  13. `src/app/objects.js`
  14. `src/app/session.js`
  15. `src/core/mol.js`
  16. `src/panels/seq.js`
