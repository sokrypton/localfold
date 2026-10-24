<!--
docs/DESIGN-RUNSPEC.md
----------------------
AI Context: ANOTHER PROJECT'S SAMPLING INTERFACE, SPECIFIED FOR REIMPLEMENTATION

🔴 THIS DESCRIBES `maraxen/aminx`, NOT THIS REPOSITORY. It is the knob surface
step 2 of docs/DESIGN.md adopts - the RunSpec - written down precisely enough to
reimplement against our own engine in `cpu/design/mpnn/`. Nothing here is a
description of code that exists in this tree.

PROVENANCE. Read from `maraxen/aminx` at commit
`0729c4589355f65ff0719fa143cf311f6b44b598`, MIT, Copyright (c) 2025 Marielle
Russo, who has given permission to integrate her work (2026-10-24). Her files
are named with their paths inside HER repository. Any code we take from her
carries her copyright and permission notice, as the licence requires.

🔴 A SNAPSHOT, PINNED. It describes one commit of a repository that moves. Check
it against upstream before trusting a detail, and re-read rather than assume.

HOW IT WAS MADE, AND HOW FAR IT HAS BEEN CHECKED. Produced by an `agy` worker
(Gemini 4 Argon, medium effort) reading her source, then spot-verified here
against that source on three claims, all of which held:
  - the RunSpec is exactly the ten fields in section 1 (checked: every `spec.`
    reference in `runspec_core.mjs`);
  - padding happens BEFORE the inputs are built, so the PRNG is consumed over
    the padded length (checked: `padStructure` then `buildInputs` in
    `aminx_sampler.mjs`) - section 3's consequence, that the same structure at
    the same seed samples differently depending on which bucket it lands in, is
    the single most important thing in this document;
  - section 8.3's untested list is faithful, including that the WebGPU probe was
    NEGATIVE and its conclusions are design rather than measurement.
Everything else is the worker's reading and is NOT independently verified. Treat
a number here as a claim with a citation, not as something this repository
measured - and note section 8's split is hers, not ours.

🔴 Her `docs/` is written as bare `browser_integration.md` throughout, because
`test/imports-resolve.test.js` reads a bare `docs/...` as a claim about OUR
tree. Another project's path written as though it were ours is what that gate
exists to catch.
-->

# `aminx` RunSpec & Browser Sampling Specification

This specification documents the exact runtime behavior of `maraxen/aminx`'s browser sampling surface (`runspec_core.mjs`, `aminx_sampler.mjs`, `split_loop.mjs`, and `split_driver.mjs`) for reimplementation against an independent JavaScript ProteinMPNN engine. Where `browser_integration.md` and the executable code diverge, both are cited and the executable code governs.

---

## 1. FIELD TABLE

`buildP07TypedInputs(structure, runspec, options = {})` (`aminx/runspec_core.mjs:254-355`) normalizes `const spec = runspec || {}` (`runspec_core.mjs:255`) and reads **10** fields from `runspec`.

> **Model-family `options` parameter (distinct from `runspec`):** `buildP07TypedInputs` also accepts a third parameter `options = {}` with `{ alphabet?: string, omitBias?: number }` (`runspec_core.mjs:251-257, 288`), defaulting to `MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"` (`nTokens = 21`, index `20` = `"X"`) and `OMIT_BIAS = -1e8` (`runspec_core.mjs:21-22`). In `split_driver.mjs:113-130, 271-274`, these come from `MANIFEST.json` (`manifest.alphabet`, `manifest.n_tokens`, `manifest.omit_bias`).

| # | Exact Name | Type / Accepted Shape | Default When Omitted | Accepted Range / Form | Exact Effect |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | `seed` | `number` (coerced via unsigned right-shift `spec.seed >>> 0`) | `0` (`undefined >>> 0 === 0` at `runspec_core.mjs:319`) | Any JS value; coerced by `>>> 0` to a 32-bit unsigned integer in `[0, 4294967295]` (`0x00000000..0xffffffff`), then passed to `BigInt(...)` (`runspec_core.mjs:39`). Non-numeric strings, `NaN`, `null`, `undefined`, and multiples of $2^{32}$ all coerce to `0`. | Seeds the per-call SplitMix64 PRNG (`makePrng(seed)` at `runspec_core.mjs:38-52, 319-320`) that drives (1) the Fisher-Yates shuffle when `decoding_order === "random"` and (2) the `[L, nTokens]` `gumbel_noise` draw. |
| 2 | `temperature` | `number` | `0.1` (`toFloat32(0.1)` = `0.10000000149011612`, `runspec_core.mjs:273`) | Checked via `spec.temperature === undefined ? 0.1 : spec.temperature`, then cast via `toFloat32(...)` (`new Float32Array(1)` round-trip, `runspec_core.mjs:31-36, 273`). **No range validation in JS** (`0`, negative values, `Infinity`, `NaN`, or `null` — which coerces to `0` because `null !== undefined` — are not rejected in JS). | Emitted as scalar `float32` tensor `temperature` (shape `[]`, `runspec_core.mjs:352`) and passed to the sampling / Graph F step (`split_loop.mjs:199`). How Graph F scales logits/Gumbel noise by `temperature` (and whether `temperature <= 0` is guarded) is inside the ONNX graph and **unclear from JS source**. |
| 3 | `bias_AA` | `Record<string, number>` (plain object mapping single-letter AA strings to numeric logit deltas) | `{}` (`spec.bias_AA \|\| {}`, `runspec_core.mjs:275`) | Keys: single-character strings present in `alphabet` (`"ACDEFGHIKLMNPQRSTVWYX"`). Values: numbers (or numeric strings) converted via `toFloat32(value)`. Unknown or multi-char keys throw `Error("letter ... is not in alphabet ...")` (`runspec_core.mjs:162-164`). | Applied **first** when building `bias` (`[L, nTokens]`, initialized to `0`). For each `[letter, value]` in `Object.entries(spec.bias_AA \|\| {})`, sets `bias[pos * nTokens + column] = toFloat32(bias[pos * nTokens + column] + toFloat32(value))` across every residue `pos` in `0 .. L - 1` (`runspec_core.mjs:275-281`). |
| 4 | `bias_AA_per_residue` | `Record<number \| string, Record<string, number>>` (outer key: 0-based position; inner dict: single-letter AA -> numeric delta) | `{}` (`positionEntries(spec.bias_AA_per_residue)` returns `[]` when falsy, `runspec_core.mjs:228-231, 282`) | Outer keys: 0-based integer residue index `0 .. L - 1` (coerced via `Number(key)` in `positionEntries`). Inner value: non-null object whose keys are single letters in `alphabet` and values are numbers. | Applied **second** (accumulating onto `bias_AA`). For each `[pos, letters]` and `[letter, value]`, updates `bias[pos * nTokens + column] = toFloat32(bias[pos * nTokens + column] + toFloat32(value))` (`runspec_core.mjs:282-287`). Out-of-range `pos` (`pos < 0` or `pos >= L`) is **not** checked and silently writes out-of-bounds on `Float32Array` (a no-op). |
| 5 | `omit_AA` | `string` or `Iterable<string>` of single-letter AAs (e.g. `"C"` or `"CX"`, or `["C", "X"]`) | `""` (`spec.omit_AA \|\| ""`, `runspec_core.mjs:289`) | Iterated with `for (const letter of spec.omit_AA \|\| "")`. Every yielded element must be a single character in `alphabet` (else `letterIndex(letter)` throws). | Applied **third**. **Overwrites** (does not add to) prior biases: for each `letter` and for all `pos` in `0 .. L - 1`, sets `bias[pos * nTokens + column] = omitValue` where `omitValue = toFloat32(options.omitBias ?? -1e8)` (`-100000000.0`, `runspec_core.mjs:22, 288-292`). |
| 6 | `omit_AA_per_residue` | `Record<number \| string, string \| Iterable<string>>` (outer key: 0-based position; inner value: string of letters like `"P"` or `"CX"`, or array of single-letter strings) | `{}` (`positionEntries(spec.omit_AA_per_residue)` returns `[]` when falsy, `runspec_core.mjs:228-231, 293`) | Outer keys: 0-based integer position index `0 .. L - 1` (via `Number(key)`). Inner value: any iterable of single-character strings in `alphabet` (strings and arrays of 1-char strings both work via `for (const letter of letters)`). | Applied **fourth**. **Overwrites** `bias[pos * nTokens + letterIndex(letter)] = omitValue` (`-1e8`) at that `pos` (`runspec_core.mjs:293-297`), superseding `bias_AA` and `bias_AA_per_residue` at `(pos, letter)`. Out-of-range `pos` is not validated and silently no-ops. |
| 7 | `chains_to_design` | `Iterable<string>` (e.g. `string[]` of chain identifiers such as `["A"]`, passed to `new Set(spec.chains_to_design)`) | `undefined` (when falsy, no chains are auto-fixed by chain ID; `runspec_core.mjs:301`) | Requires `structure.chain_ids` to be provided with `chainIds.length === L`, otherwise throws `Error("chains_to_design requires structure.chain_ids of length L")` (`runspec_core.mjs:302-304`). Elements are matched with `Set.prototype.has(chainIds[pos])`. | For every `pos` in `0 .. L - 1` where `!design.has(chainIds[pos])` (including padded positions if their repeated `chain_ids` entry is not in `chains_to_design`), sets `fixedMask[pos] = 1` and `fixedTokens[pos] = nativeTokens[pos]` (`runspec_core.mjs:306-311`). Note: if `structure.native_tokens` is omitted, `nativeTokens` defaults to all zeros (`0` = `'A'`, `runspec_core.mjs:265-267`). |
| 8 | `fixed_positions` | `Record<number \| string, string>` | `{}` (`positionEntries(spec.fixed_positions)` returns `[]` when falsy, `runspec_core.mjs:228-231, 313`) | **Keyed by 0-based integer residue index** into the structure arrays (`0 .. L - 1`, parsed via `Number(key)`), **NOT** by chain ID and **NOT** by 1-based PDB `residue_index` (`browser_integration.md:141, 146-147`). **Values must be single letters** in `alphabet` (e.g. `{ 0: "M", "4": "G" }`); lists or multi-char strings throw in `letterIndex(letter)` (`runspec_core.mjs:158-166, 313-316`). | Applied **after** `chains_to_design`. For each `[pos, letter]`, sets `fixedMask[pos] = 1` and `fixedTokens[pos] = letterIndex(letter)` (`runspec_core.mjs:313-316`). Can override `fixedTokens[pos]` set by `chains_to_design`, and pins `sequence[pos]` to `fixedTokens[pos]` before wave decoding starts (`split_loop.mjs:145-147`). Out-of-range `pos` still validates `letterIndex(letter)` (throwing on bad letters) before silently dropping the write on `fixedMask`/`fixedTokens`. |
| 9 | `tied_positions` | `Array<Array<number>>` (list of tie-group arrays of 0-based integer position indices, e.g. `[[3, 40], [4, 41]]`) | `[]` (`spec.tied_positions \|\| []`, `runspec_core.mjs:318`) | Outer iterable of arrays (`group`). Empty groups (`group.length === 0`) are skipped (`runspec_core.mjs:173`). Every `member` in a non-empty group must satisfy `0 <= member && member < L` (where `L` is the structure length passed to `buildP07TypedInputs`, i.e. `bucket` when padded), else throws `Error("tie member ... outside 0..${length - 1}")` (`runspec_core.mjs:178-181`). | Builds `tie_group_map` (`int32[L]`): initialized to identity `ids[i] = i`, then for each `group`, sets `gid = min(group)` and sets `ids[member] = gid` for every `member` in `group` (`runspec_core.mjs:169-185`). Alters `designedFlags` (if *any* member of a tie group is designed, *all* members of that tie group get `designed[i] = 1`, `runspec_core.mjs:99-105`) and forces tied positions to decode together and receive the same sampled token. |
| 10 | `decoding_order` | `"random"` or `ArrayLike<number> \| Iterable<number>` (explicit permutation of `0 .. L - 1`) | `"random"` (`spec.decoding_order === undefined ? "random" : spec.decoding_order`, `runspec_core.mjs:322`) | Either the literal string `"random"`, or an iterable/array-like converted via `Int32Array.from(requested)` whose `.length === L` and whose coerced `int32` elements have `new Set(decodingOrder).size === L` (`runspec_core.mjs:326-335`). (**Caveat:** range `0 .. L - 1` is not checked on explicit orders; see Section 7.) | When `"random"`, consumes PRNG draws via `fixedFirstShuffle(L, prng, designed)` (`runspec_core.mjs:323-325`) so non-designed/fixed/padded positions come first and designed positions come second. When explicit, uses `Int32Array.from(requested)` verbatim without advancing the PRNG before Gumbel generation (`runspec_core.mjs:327-340`). |

### Guide vs. Code Discrepancies on RunSpec Fields
1. **`chains_to_design` without `structure.native_tokens`:** `aminx/browser_integration.md:140, 145-146` states that `chains_to_design` "needs `chain_ids` and `native_tokens`". In code (`aminx/runspec_core.mjs:265-267, 301-312`), only `chain_ids` is validated (`if (chainIds.length !== length) throw ...`); if `structure.native_tokens` is omitted, `nativeTokens` silently defaults to `new Int32Array(length)` (all `0`, which maps to `'A'`).
2. **Prose knob names in validation section:** `aminx/browser_integration.md:288` refers in prose to `tied_pairs` and `explicit_order`, whereas the RunSpec table (`browser_integration.md:143-144`) and code (`runspec_core.mjs:318, 322`) use `tied_positions` and `decoding_order`.

---

## 2. INPUT TENSORS

`buildP07TypedInputs` (`aminx/runspec_core.mjs:342-354`) returns an object with **11** named tensors in the exact positional order pinned by `P07_INPUT_ORDER` (`aminx/aminx_sampler.mjs:24-36`). Each entry has the shape `{ dtype: string, shape: number[], data: Float32Array | Int32Array }` (`runspec_core.mjs:235-237`), or `{ dtype, shape, data: number[] }` when called via `buildP07Inputs` (`runspec_core.mjs:368-375`).

Below, `L` denotes the `length` of the `structure` passed to `buildP07TypedInputs` (`structure.coords_shape ? structure.coords_shape[0] : structure.coords.length`, `runspec_core.mjs:260`). When called through `createSampler` (`aminx_sampler.mjs:176-179`) or `createSplitSampler` (`split_driver.mjs:268-274`), `structure` has **already been padded to `bucket`**, so `L === bucket` (`128` or `256`). `nTokens` is `alphabet.length` (`21` for `MPNN_ALPHABET`, `runspec_core.mjs:256-257`).

| # | Tensor Name (`P07_INPUT_ORDER`) | `dtype` | `shape` | Backing Array | Exact Computation from `structure` and `runspec` |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | `coords` | `"float32"` | `[L, 4, 3]` | `Float32Array(L * 12)` | Flattens `structure.coords` row-major via `asFloat32(structure.coords, structure.coords_shape \|\| inferCoordsShape(structure.coords))` (`runspec_core.mjs:206-215, 259, 343`), checking that total element count equals the product of `coords_shape` (or `inferCoordsShape`, which is `[coords.length, coords[0].length, coords[0][0].length]`). Note: `return` hardcodes the output tensor shape metadata to `[length, 4, 3]` (`runspec_core.mjs:343`). |
| 2 | `mask` | `"float32"` | `[L]` | `Float32Array(L)` | Flattens `structure.mask` via `asFloat32(structure.mask, [length])` (`runspec_core.mjs:261, 344`). `1.0` for valid/real residues, `0.0` for missing or padded residues (`aminx_sampler.mjs:64-65, 91-92`). |
| 3 | `residue_index` | `"int32"` | `[L]` | `Int32Array(L)` | Flattens `structure.residue_index` via `asInt32(structure.residue_index, [length])` (`runspec_core.mjs:262, 345`). |
| 4 | `chain_index` | `"int32"` | `[L]` | `Int32Array(L)` | Flattens `structure.chain_index` via `asInt32(structure.chain_index, [length])` (`runspec_core.mjs:263, 346`). |
| 5 | `gumbel_noise` | `"float32"` | `[L, nTokens]` | `Float32Array(L * nTokens)` | Drawn **after** `decoding_order` is computed from `prng = makePrng(spec.seed >>> 0)`. Filled in row-major order (`i = 0 .. L * nTokens - 1`, i.e. residue `0` tokens `0..nTokens-1`, then residue `1`, etc.): `gumbel[i] = gumbelFromUniform(uniformFromU32(prng.nextU32()))` (`runspec_core.mjs:337-340, 347`). |
| 6 | `decoding_order` | `"int32"` | `[L]` | `Int32Array(L)` | If `spec.decoding_order === undefined` or `"random"`, computed as `fixedFirstShuffle(length, prng, designedFlags(length, mask, fixedMask, tie))` using `prng` before `gumbel_noise` is drawn (`runspec_core.mjs:321-325`). Otherwise, `Int32Array.from(spec.decoding_order)` (`runspec_core.mjs:327-335`). |
| 7 | `bias` | `"float32"` | `[L, nTokens]` | `Float32Array(L * nTokens)` | Row-major `[L, nTokens]` array initialized to `0.0` (`runspec_core.mjs:274`). Constructed in 4 ordered stages (`runspec_core.mjs:275-297`):<br>1. `bias_AA`: for each `[letter, value]`, `delta = toFloat32(value)`, for `pos = 0..L-1`, `bias[pos*nTokens + col] = toFloat32(bias[pos*nTokens + col] + delta)`.<br>2. `bias_AA_per_residue`: for each `[pos, letters]` and `[letter, value]`, `bias[pos*nTokens + col] = toFloat32(bias[pos*nTokens + col] + toFloat32(value))`.<br>3. `omit_AA`: for each `letter`, for `pos = 0..L-1`, `bias[pos*nTokens + col] = toFloat32(options.omitBias ?? -1e8)`.<br>4. `omit_AA_per_residue`: for each `[pos, letters]` and `letter`, `bias[pos*nTokens + col] = toFloat32(options.omitBias ?? -1e8)`.<br>(Passed as `cond_bias` to Graph F in `split_driver.mjs:293`, `split_loop.mjs:194`.) |
| 8 | `fixed_mask` | `"float32"` | `[L]` | `Float32Array(L)` | Initialized to `0.0` (`runspec_core.mjs:299`). First, if `spec.chains_to_design` is truthy, sets `fixedMask[pos] = 1` for every `pos` where `!new Set(spec.chains_to_design).has(chainIds[pos])` (`runspec_core.mjs:301-312`). Second, for each `[pos, letter]` in `spec.fixed_positions`, sets `fixedMask[pos] = 1` (`runspec_core.mjs:313-316`). |
| 9 | `fixed_tokens` | `"int32"` | `[L]` | `Int32Array(L)` | Initialized to `0` (`runspec_core.mjs:300`). First, if `spec.chains_to_design` is truthy, sets `fixedTokens[pos] = nativeTokens[pos]` at non-designed chain positions (`runspec_core.mjs:306-311`). Second, for each `[pos, letter]` in `spec.fixed_positions`, sets `fixedTokens[pos] = letterIndex(letter)` (`runspec_core.mjs:313-316`). At positions where `fixed_mask[pos] === 0`, `fixed_tokens[pos]` remains `0`. |
| 10 | `temperature` | `"float32"` | `[]` (0-D scalar) | `Float32Array(1)` | `Float32Array.of(toFloat32(spec.temperature === undefined ? 0.1 : spec.temperature))` with shape `[]` (`runspec_core.mjs:273, 352`). |
| 11 | `tie_group_map` | `"int32"` | `[L]` | `Int32Array(L)` | Computed by `tieGroupMap(length, spec.tied_positions \|\| [])` (`runspec_core.mjs:169-186, 318, 353`): starts with `ids[i] = i` for `i = 0..L-1`; for each non-empty group in `tied_positions`, finds `gid = Math.min(...group)` and assigns `ids[member] = gid` for every `member` in `group`. |

---

## 3. RANDOMNESS

### 3.1 PRNG Algorithm and Constants (`SplitMix64`)
Implemented in `makePrng(seed)` (`aminx/runspec_core.mjs:24-52`):
- **Constants (`runspec_core.mjs:24-29`):**
  - `MASK64 = (1n << 64n) - 1n` (`0xffffffffffffffffn`)
  - `GOLDEN = 0x9E3779B97F4A7C15n`
  - `MIX_1 = 0xBF58476D1CE4E5B9n`
  - `MIX_2 = 0x94D049BB133111EBn`
  - `U32_SPAN = 0x100000000` ($2^{32} = 4294967296$)
  - `MANTISSA_DENOM = 16777217` ($2^{24} + 1$)
- **Seeding (`runspec_core.mjs:38-39, 319-320`):**
  - `const seed = spec.seed >>> 0;`
  - `let state = BigInt(seed >>> 0);` (initial 64-bit state has upper 32 bits `0` and lower 32 bits equal to `seed >>> 0`).
- **State advancement (`nextU64` and `nextU32`, `runspec_core.mjs:41-50`):**
  ```js
  state = (state + GOLDEN) & MASK64;
  let z = state;
  z = ((z ^ (z >> 30n)) * MIX_1) & MASK64;
  z = ((z ^ (z >> 27n)) * MIX_2) & MASK64;
  const u64 = (z ^ (z >> 31n)) & MASK64;
  const u32 = Number(u64 >> 32n); // high 32 bits of the 64-bit output
  ```
  Every `nextU32()` call advances `state` by one 64-bit SplitMix64 step and returns the **upper 32 bits** (`u64 >> 32n`) as a JS `Number` in `[0, 2^32 - 1]`.

### 3.2 Bounded Integer Draws and Fisher-Yates (`nextBelow` & `shuffle`)
- **`nextBelow(prng, bound)` (`runspec_core.mjs:70-80`):**
  - Rejects `bound <= 0 || bound > 0x100000000` with an `Error`.
  - Computes rejection threshold `limit = U32_SPAN - (U32_SPAN % bound)`.
  - Repeatedly draws `draw = prng.nextU32()` while `draw >= limit`, then returns `draw % bound`.
- **`shuffle(length, prng)` (`runspec_core.mjs:82-92`):**
  - Initializes `order = Int32Array(length)` with `order[i] = i`.
  - Loops `for (let i = length - 1; i > 0; i -= 1)`: draws `j = nextBelow(prng, i + 1)` and swaps `order[i]` with `order[j]`.
  - Draws at least `length - 1` `u32` values (plus any rejection-sampling retries in `nextBelow`).

### 3.3 Uniform and Gumbel Noise Generation
- **`uniformFromU32(u32)` (`runspec_core.mjs:54-57`):**
  - Takes the upper 24 bits of `u32`: `const mant = (u32 >>> 8) & 0xffffff;` (an integer in `0 .. 16777215`).
  - Returns `toFloat32((mant + 1) / 16777217)`, which lies strictly inside the open interval `(0, 1)` (`[1 / 16777217, 16777216 / 16777217]`), avoiding `0` and `1` so `Math.log` never sees `0`.
- **`gumbelFromUniform(u)` (`runspec_core.mjs:59-63`):**
  - Applies float32 rounding at **every intermediate operation**:
    ```js
    const uu = toFloat32(u);
    const inner = toFloat32(-Math.log(uu));
    return toFloat32(-Math.log(inner));
    ```
- **Filling `gumbel_noise` (`runspec_core.mjs:337-340`):**
  - Allocates `gumbel = new Float32Array(length * nTokens)`.
  - For `i = 0` to `length * nTokens - 1` (row-major over `[L, nTokens]`):
    `gumbel[i] = gumbelFromUniform(uniformFromU32(prng.nextU32()))`.
  - Note: `badGumbelFromUniform(u)` (`runspec_core.mjs:65-68`, computing `-log(u)` instead of `-log(-log(u))`) and `biasedShuffle` (`runspec_core.mjs:145-155`) are exported only as negative-control defect injections for instrument sensitivity testing (`browser_integration.md:171`) and are **not** called by `buildP07TypedInputs`.

### 3.4 Where Gumbel Noise Enters Sampling
- In the monolithic model (`aminx_sampler.mjs:29, 179-188`), `gumbel_noise` (`float32[L, 21]`) is fed as the 5th positional input to the ONNX session.
- In the split loop (`split_loop.mjs:192-201`), the full `gumbelNoise` (`float32[L, nTokens]`) tensor is passed to `runFuse` (Graph F) on every active wave alongside `group_id` (`int32[G]`). As noted in `runspec_core.mjs:18-19`, Graph F gathers the noise row by `group_id` (`0 .. L - 1`, the minimum position index of the tie group) to perform Gumbel-max sampling for each active group.

### 3.5 Reproducibility and Which Fields Change the PRNG Draw
A single SplitMix64 stream is created per `buildP07TypedInputs(structure, runspec, options)` invocation (`runspec_core.mjs:13-17, 320`):
1. **What changes the PRNG stream alignment before `gumbel_noise`:**
   - `seed` (sets initial PRNG state).
   - `L` (`length`, i.e. the padded `bucket` length when run via `createSampler`/`createSplitSampler`, or raw `structure` length when calling `buildP07TypedInputs` directly): determines how many steps `shuffle(L, prng)` takes (`L - 1` calls to `nextBelow`) and how many elements `gumbel_noise` (`L * nTokens`) draws.
   - `decoding_order`:
     - When `decoding_order === "random"` (or `undefined`), `fixedFirstShuffle(L, prng, designed)` calls `shuffle(L, prng)` first (`runspec_core.mjs:123-124`), advancing the PRNG by `L - 1` bounded draws before `gumbel_noise` is generated.
     - When `decoding_order` is an **explicit permutation array**, `shuffle(L, prng)` is **not called** (`runspec_core.mjs:326-340`), so `gumbel_noise` consumes the very first `L * nTokens` draws from `makePrng(seed)`! Thus switching between `"random"` and an explicit `decoding_order` (even if the explicit array equals the permutation `"random"` would have produced) **changes `gumbel_noise`**.
   - `options.alphabet`: changes `nTokens = alphabet.length`, changing the number of Gumbel draws per position.
2. **What does NOT change the `gumbel_noise` draw or base permutation `shuffle(L, prng)`:**
   - Because `fixedFirstShuffle` runs one unconditional `shuffle(length, prng)` and then performs a deterministic stable partition using `designed` without drawing any further random numbers (`runspec_core.mjs:114-122, 123-143`), changing `mask`, `fixed_positions`, `chains_to_design`, or `tied_positions` **does not alter the PRNG state entering `gumbel_noise`** (though those fields do alter the resulting `decoding_order` via the stable partition).
   - Changing `temperature`, `bias_AA`, `bias_AA_per_residue`, `omit_AA`, or `omit_AA_per_residue` does not touch `prng` at all (`gumbel_noise` and `decoding_order` are bit-identical).

---

## 4. DECODING ORDER

### 4.1 Computing `designedFlags(length, mask, fixedMask, tie)`
When `decoding_order` is `"random"` (the default), `buildP07TypedInputs` first computes `designed` (`Uint8Array(length)` of `0` or `1`) via `designedFlags(length, mask, fixedMask, tie)` (`aminx/runspec_core.mjs:94-107, 324`):
1. **Initial per-position flag (`runspec_core.mjs:96-98`):**
   For `i = 0 .. length - 1`, `designed[i] = 1` if and only if `mask[i] !== 0 && fixedMask[i] === 0`. Otherwise `designed[i] = 0` (this includes padded rows where `mask[i] === 0`, missing backbone rows where `mask[i] === 0`, positions fixed by `fixed_positions`, and positions on chains excluded by `chains_to_design`).
2. **Expansion through tie groups (`runspec_core.mjs:99-105`):**
   - Allocates `group = new Uint8Array(length)`.
   - For `i = 0 .. length - 1`: if `designed[i]` is `1`, sets `group[tie[i]] = 1`.
   - For `i = 0 .. length - 1`: if `group[tie[i]]` is `1`, sets `designed[i] = 1`.
   - **Semantic consequence:** If **any** member of a tie group is designed (`mask[i] !== 0 && fixedMask[i] === 0`), **every** member of that tie group is marked `designed[i] = 1` (even members that had `fixedMask[i] === 1` or `mask[i] === 0`).

### 4.2 `fixedFirstShuffle(length, prng, designed)`
Computed in `aminx/runspec_core.mjs:123-143`:
1. Draws a uniform random permutation `order = shuffle(length, prng)` of `0 .. length - 1`.
2. Counts `nFixed = count of i in 0..length-1 where designed[order[i]] === 0`.
3. Stable-partitions `order` into `out = new Int32Array(length)`:
   - All `pos = order[i]` with `designed[pos] === 0` are written to `out[0 .. nFixed - 1]` in the relative order they appeared in `order`.
   - All `pos = order[i]` with `designed[pos] !== 0` are written to `out[nFixed .. length - 1]` in the relative order they appeared in `order`.
4. Returns `out` as `decoding_order` (`int32[L]`).

### 4.3 Explicit `decoding_order`
When `spec.decoding_order !== undefined && spec.decoding_order !== "random"` (`runspec_core.mjs:322, 326-335`):
- Converts via `decodingOrder = Int32Array.from(requested)`.
- Validates `decodingOrder.length === length` and `new Set(decodingOrder).size === length`.
- Does **not** partition fixed positions first; the caller's permutation is used as-is.

### 4.4 Tie Semantics and Wave Execution
Tied positions are governed by `tieGroupMap` (`runspec_core.mjs:169-186`) and the wave loop (`split_loop.mjs:141-238`):
1. **Group ID assignment (`runspec_core.mjs:169-186`):**
   - Each position `i` defaults to `ids[i] = i`.
   - For each group in `spec.tied_positions`, `gid = Math.min(...group)`, and every `member` in that group gets `ids[member] = gid`.
2. **Initialization (`split_loop.mjs:144-147`):**
   - `sequence` (`Int32Array(L)`) starts filled with `UNDRAWN_TOKEN = -1` (`split_loop.mjs:21, 144`).
   - For `pos = 0 .. L - 1`, if `fixedMask[pos] > 0.5`, `sequence[pos] = fixedTokens[pos]` (`split_loop.mjs:145-147`).
3. **Wave scheduling (Graph W, `split_loop.mjs:135-169`):**
   - Graph W takes `decoding_order` (`int32[L]`) and `tie_group_map` (`int32[L]`) and returns `group_ids` (`[nWaves, G]`), `group_positions` (`[nWaves, G, ...]`), `group_valid` (`[nWaves, G]`), `position_valid`, `ar_mask`, `group_first_rank` (`[L]`), and `pos_first_rank` (`[L]`).
   - In wave `w` (`0 .. nWaves - 1`) and slot `g` (`0 .. G - 1`):
     - `pos0 = at(group_positions, w, g, 0)`
     - `groupId[g] = at(tieGroupMap, pos0)`
     - `thisRank = w * G + g`
     - `isFirst[g] = isActive && at(group_first_rank, groupId[g]) === thisRank ? 1 : 0` (`split_loop.mjs:155-159`).
     - If `isFirst[g]` is `1`, `maskGroup[g * L + pos] = 1` for **all** `pos` in `0 .. L - 1` where `tieGroupMap[pos] === groupId[g]` (`split_loop.mjs:162-168`).
   - If no slot `g` in wave `w` has `isFirst[g] === 1`, the decoder and fuse graphs are skipped for wave `w` and a zero tensor `[G, nTokens]` is pushed to `logitsStack` (`split_loop.mjs:170-174`).
4. **Logit Combination vs. Sample Combination (`split_loop.mjs:182-217`):**
   - **Logits are combined (averaged across the tie group inside Graph F), and a single token is sampled per tie group:**
     - Graph D (`runDecoder`) runs once per active wave given `sequence_oh` (`float32[L, nTokens]`, where `sequence[pos] === -1` produces an all-zero one-hot row, `split_loop.mjs:176-189`) and outputs raw `logits` (`[1, L, nTokens]`).
     - Graph F (`runFuse`) receives `logits` (`[1, L, nTokens]`), `cond_bias` (`[L, nTokens]`), `mask_group` (`bool[G, L]`), `fixed_mask` (`[L]`), `fixed_tokens` (`[L]`), `group_id` (`int32[G]`), `temperature` (`[]`), and `gumbel_noise` (`[L, nTokens]`), and returns:
       - `final_token` (`int32[G]`): one sampled token per group slot `g`.
       - `avg_stored` (`float32[G, nTokens]`): the combined per-group logits for each slot `g` (`split_loop.mjs:192-202, 216`).
     - **Note on exact inside-Graph-F formula:** Because Graph F is an ONNX graph, the exact internal order of operations inside Graph F (e.g., whether `cond_bias` is added before or after averaging logits across `mask_group`, whether `mask` weights the average, and how mixed fixed/designed tie groups are resolved inside Graph F) is inside `p07_fuse_L*.onnx` and **unclear from the local JS files alone** (see Section 10).
5. **Forcing Tied Positions Equal (`split_loop.mjs:204-214`):**
   - For every `pos` in `0 .. L - 1`, `split_loop.mjs` checks all `g = 0 .. G - 1`: if `maskGroup[g * L + pos]` is `1`, it accumulates `tokenSum += at(final_token, g)` and sets `sequence[pos] = tokenSum`. Since `isFirst[g]` is true at most once per unique `groupId` in a wave, `maskGroup[g * L + pos]` is `1` for at most one `g`, so **every position `pos` belonging to `groupId[g]` is assigned the exact same integer token `at(final_token, g)` simultaneously in that wave**.
6. **Post-hoc Log-Probabilities (`split_loop.mjs:79-86, 219-233`):**
   - After all `nWaves` complete, for each `pos` in `0 .. L - 1`:
     - Reads `rank = at(pos_first_rank, pos)` and checks `scheduled = rank < nWaves * G`.
     - If `scheduled`, extracts `row[k] = at(logitsStack[Math.floor(rank / G)], rank % G, k)` (the `avg_stored` row from the wave `w` and slot `g` of that group's first occurrence). If not scheduled (`rank >= nWaves * G`), `row` is all zeros (`new Array(N_TOKENS).fill(0)`).
     - Applies `logSoftmaxRow(row)` in JS (`v - (Math.log(sumExp) + max)`) and writes the result to `logProbsData[pos * N_TOKENS + k]`. Thus tied positions also receive identical rows in `logProbs`.

---

## 5. STRUCTURE FORMAT

`structure` is passed to `buildP07TypedInputs(structure, runspec, options)` (`aminx/runspec_core.mjs:254-272`) and `padStructure(structure, bucket)` (`aminx/aminx_sampler.mjs:74-127`).

| Field Name | Required? | Accepted Input Shape / Types | Internal Tensor `dtype` & `shape` | Semantics |
| :--- | :--- | :--- | :--- | :--- |
| `coords` | **Yes** | Nested JS arrays `[L][4][3]` of numbers, or any flat/nested `TypedArray` (except `DataView`) / array flattening to `L * 4 * 3` numbers (`runspec_core.mjs:191-215, 239-241, 259`). | `float32`, `[L, 4, 3]` | Backbone atomic coordinates in **N, CA, C, O** order (`browser_integration.md:117`). |
| `coords_shape` | Optional (required if `coords` is a flat `TypedArray`) | `[L, 4, 3]` (`number[]` of length 3) | Used as shape `[L, 4, 3]` (`runspec_core.mjs:259-260`, `aminx_sampler.mjs:57-59`) | Explicitly states `[L, 4, 3]` when `coords` is a flat `Float32Array` (since `coords.length` on a flat array would be `12 * L` instead of `L`). |
| `mask` | **Yes** | Array or `TypedArray` of `L` numbers (`0` or `1`) | `float32`, `[L]` | `1` = real residue with valid backbone atoms; `0` = missing residue or padding (`browser_integration.md:118, 124`). |
| `residue_index` | **Yes** | Array or `TypedArray` of `L` integers | `int32`, `[L]` | Per-residue integer numbering (e.g. PDB residue number), used for relative-position edge features (`browser_integration.md:119`). |
| `chain_index` | **Yes** | Array or `TypedArray` of `L` integers | `int32`, `[L]` | Integer chain identifier per residue (`browser_integration.md:120`). |
| `chain_ids` | Optional (required if `runspec.chains_to_design` is set) | `Array<string>` of length `0` or `L` (`runspec_core.mjs:264, 269-271, 302-304`; `aminx_sampler.mjs:113-117`) | Not a model tensor; consumed in JS to populate `fixed_mask` and `fixed_tokens` | Per-residue chain string (e.g. `"A"`, `"B"`), matched against `new Set(runspec.chains_to_design)`. |
| `native_tokens` | Optional | Array or `TypedArray` of `L` integers in `0 .. nTokens - 1` (`runspec_core.mjs:265-267`; `aminx_sampler.mjs:119-124`) | Not a direct model input; copied into `fixed_tokens` for chains not in `chains_to_design` | Integer token indices in `alphabet` (`"ACDEFGHIKLMNPQRSTVWYX"`). Defaults to `new Int32Array(length)` (all `0` = `'A'`) if omitted (`runspec_core.mjs:265-267`). |

### Atom Order and What the Model Computes Itself
- **Atom order:** Exactly 4 backbone atoms per residue in the order **N, CA, C, O** (`browser_integration.md:117, 123`).
- **What the model computes itself:**
  - **Virtual `CB` atom:** No `CB` coordinates are supplied in `coords`; the encoder computes virtual `CB` from `N, CA, C, O` (`browser_integration.md:117`).
  - **k-NN graph and geometric features:** The encoder (`Graph E` / in-graph encoder) computes pairwise distances, `TopK` k-NN `neighbor_indices`, relative-position features from `residue_index` and `chain_index`, `node_features`, and `edge_features` (`split_driver.mjs:29, 50`, `browser_integration.md:229, 358-360`).
  - **Autoregressive wave schedule and masks (`Graph W`):** In the split architecture, `Graph W` computes `group_ids`, `group_positions`, `group_valid`, `position_valid`, `ar_mask` ($L \times L$), `group_first_rank`, and `pos_first_rank` from `decoding_order` and `tie_group_map` (`split_driver.mjs:30, 51-59`).

---

## 6. PADDING AND BUCKETS

### 6.1 Bucket Selection (`BUCKETS` and `pickBucket`)
Defined in `aminx/aminx_sampler.mjs:41-55`:
- `export const BUCKETS = Object.freeze([128, 256]);`
- `pickBucket(length)` iterates through `BUCKETS` (`[128, 256]`) in ascending order and returns the first `bucket` where `length <= bucket`:
  - `length <= 128` (including `1..128`, and non-positive numbers since there is no lower-bound check) $\rightarrow$ `128`.
  - `129 <= length <= 256` $\rightarrow$ `256`.
  - `length > 256` $\rightarrow$ throws `new Error("length ${length} exceeds the largest browser export bucket 256")`.

### 6.2 Structure Padding (`padStructure(structure, bucket)`)
Defined in `aminx/aminx_sampler.mjs:74-127`:
- Computes `const nReal = structureLength(structure)` (`structure.coords_shape ? structure.coords_shape[0] : structure.coords.length`, `aminx_sampler.mjs:57-59, 75`).
- If `bucket < nReal`, throws `new Error("bucket ${bucket} is smaller than n_real ${nReal}")` (`aminx_sampler.mjs:76-78`).
- If `bucket === nReal`, returns `structure` unchanged (same object reference, no copy; `aminx_sampler.mjs:79-81`).
- If `bucket > nReal`, returns a shallow clone `{ ...structure, ... }` padded along axis 0 from `nReal` to `bucket`:
  1. `coords`: `Float32Array(bucket * 4 * 3)` — copies first `nReal * 12` floats from `structure.coords`, pads rows `nReal .. bucket - 1` with **`0.0`**, and sets `coords_shape: [bucket, 4, 3]` (`aminx_sampler.mjs:88-90, 106-107`).
  2. `mask`: `Float32Array(bucket)` — copies first `nReal` floats from `structure.mask`, pads indices `nReal .. bucket - 1` with **`0.0`** (`aminx_sampler.mjs:91-92`).
  3. `residue_index`: `Int32Array(bucket)` — copies first `nReal` ints, pads indices `nReal .. bucket - 1` by **repeating the last real value** `lastResidue = nReal > 0 ? residueIndex[nReal - 1] : 0` (`aminx_sampler.mjs:94-97`).
  4. `chain_index`: `Int32Array(bucket)` — copies first `nReal` ints, pads indices `nReal .. bucket - 1` by **repeating the last real value** `lastChain = nReal > 0 ? chainIndex[nReal - 1] : 0` (`aminx_sampler.mjs:99-102`).
  5. `chain_ids`: if `Array.isArray(structure.chain_ids) && structure.chain_ids.length === nReal`, pads to length `bucket` by **repeating the last real element** `lastChainId = nReal > 0 ? chainIds[nReal - 1] : undefined` (`aminx_sampler.mjs:113-117`).
  6. `native_tokens`: if `structure.native_tokens` is truthy, creates `Int32Array(bucket)`, copies first `nReal` ints, and pads indices `nReal .. bucket - 1` with **`0`** (`aminx_sampler.mjs:119-124`).

### 6.3 How Padding Interacts with `buildP07TypedInputs`
Because `createSampler` (`aminx_sampler.mjs:176-179`) and `createSplitSampler` (`split_driver.mjs:268-274`) call `padStructure(structure, bucket)` **before** calling `buildP07TypedInputs(padded, runspec)`:
- `buildP07TypedInputs` sees `length = bucket` (`128` or `256`), **not** `nReal`!
- Consequently:
  - `shuffle(bucket, prng)` permutes all `0 .. bucket - 1` indices (including padded positions `nReal .. bucket - 1`), and `fixedFirstShuffle` places padded positions (`mask[i] === 0` $\Rightarrow$ `designed[i] === 0`, unless tied to a designed position) into the non-designed head partition alongside fixed positions.
  - `gumbel_noise` is drawn for `[bucket, nTokens]` (`bucket * 21` draws).
  - `bias_AA` and `omit_AA` populate all `0 .. bucket - 1` rows of `bias`.
  - If an engine wants bit-for-bit PRNG parity with `aminx`'s bucketed sampler for `L < bucket`, it must run `shuffle(bucket, prng)` and `gumbel_noise` at `length = bucket`, not `length = nReal`.

### 6.4 Meaning of `nReal` and Which Outputs Are Padding to Ignore
- **`nReal`:** The unpadded residue count `structure.coords_shape ? structure.coords_shape[0] : structure.coords.length` of the input structure before `padStructure`, returned alongside `bucket` by both `sampler.sample` (`{ outputs, nReal, bucket }`, `aminx_sampler.mjs:176, 189`) and `splitSampler.sample` (`{ tokens, logProbs, nReal, bucket }`, `split_driver.mjs:268, 301`).
- **Outputs to slice/ignore:**
  - **Monolith (`aminx_sampler.mjs` + `browser_integration.md:86-88, 98-101`):**
    - `outputs["cond_out_0"].data` (`Int32Array` of shape `[bucket]`): use only `tokens.slice(0, nReal)`; indices `nReal .. bucket - 1` are padding.
    - `outputs["log_softmax_out_0"].data` (`Float32Array` of shape `[bucket, 21]`, length `bucket * 21`): use only the first `nReal` rows (`logProbs.slice(0, nReal * 21)`); rows `nReal .. bucket - 1` are padding.
  - **Four-graph split (`split_loop.mjs:235-238`, `split_driver.mjs:301`, `browser_integration.md:318-321`):**
    - `tokens`: `{ shape: [bucket], data: Int32Array(bucket) }` — use `tokens.data.slice(0, nReal)`; indices `nReal .. bucket - 1` are padding.
    - `logProbs`: `{ shape: [bucket, nTokens], data: Float32Array(bucket * nTokens) }` — use the first `nReal` rows (`logProbs.data.slice(0, nReal * nTokens)`); rows `nReal .. bucket - 1` are padding.

---

## 7. EDGE CASES AND REFUSALS

### 7.1 Explicit Validations That Throw Errors

| Location | Condition Checked | Exact Error Message Thrown |
| :--- | :--- | :--- |
| `runspec_core.mjs:71-73` (`nextBelow`) | `bound <= 0 \|\| bound > 0x100000000` | `` `nextBelow bound out of range: ${bound}` `` |
| `runspec_core.mjs:162-164` (`makeLetterIndex`) | Letter key in `bias_AA`, `bias_AA_per_residue`, `omit_AA`, `omit_AA_per_residue`, or `fixed_positions` is not a single character in `alphabet` | `` `letter ${letter} is not in alphabet ${alphabet}` `` |
| `runspec_core.mjs:179-181` (`tieGroupMap`) | Any `member` of a non-empty group in `tied_positions` has `member < 0 \|\| member >= length` (or is `NaN`/non-comparable) | `` `tie member ${member} outside 0..${length - 1}` `` |
| `runspec_core.mjs:211-213` (`asFloat32`) | Flattened element count of `coords` or `mask` does not match expected shape product | `` `float tensor has ${data.length} values, shape ${shape} wants ${expect}` `` |
| `runspec_core.mjs:222-224` (`asInt32`) | Flattened element count of `residue_index`, `chain_index`, or `native_tokens` does not match `length` | `` `int tensor has ${data.length} values, shape ${shape} wants ${expect}` `` |
| `runspec_core.mjs:269-271` (`buildP07TypedInputs`) | `chainIds.length !== 0 && chainIds.length !== length` | `` `chain_ids length ${chainIds.length} != L ${length}` `` |
| `runspec_core.mjs:302-304` (`buildP07TypedInputs`) | `spec.chains_to_design` is truthy and `chainIds.length !== length` | `"chains_to_design requires structure.chain_ids of length L"` |
| `runspec_core.mjs:328-330` (`buildP07TypedInputs`) | Explicit `spec.decoding_order` has `decodingOrder.length !== length` | `` `explicit decoding_order length ${decodingOrder.length} != L ${length}` `` |
| `runspec_core.mjs:332-334` (`buildP07TypedInputs`) | Explicit `spec.decoding_order` has duplicate entries (`new Set(decodingOrder).size !== length`) | `"explicit decoding_order is not a permutation"` |
| `aminx_sampler.mjs:53-54` (`pickBucket`) | `length > 256` (`BUCKETS[BUCKETS.length - 1]`) | `` `length ${length} exceeds the largest browser export bucket ${largest}` `` |
| `aminx_sampler.mjs:76-78` (`padStructure`) | `bucket < nReal` | `` `bucket ${bucket} is smaller than n_real ${nReal}` `` |
| `aminx_sampler.mjs:161-163`, `split_driver.mjs:89-91` | `fetch(url)` returns `!res.ok` | `` `fetch ${url} failed: ${res.status} ${res.statusText}` `` |
| `aminx_sampler.mjs:168-172` (`createSampler`) | `session.inputNames.length !== 11` (`P07_INPUT_ORDER.length`) | `` `session has ${session.inputNames.length} inputs, P07_INPUT_ORDER declares ${P07_INPUT_ORDER.length}` `` |
| `split_loop.mjs:51-53` (`makeTensor`) | Flattened element count does not equal product of `shape` | `` `tensor has ${data.length} values, shape ${shape} wants ${expect}` `` |
| `split_loop.mjs:59-61` (`flatIndex`) | `indices.length !== shape.length` | `` `index arity ${indices.length} != shape arity ${shape.length} (shape ${shape})` `` |
| `split_loop.mjs:125-127` (`runSplitDecode`) | `!Number.isInteger(N_TOKENS) \|\| N_TOKENS < 1` | `` `nTokens must be a positive integer, got ${N_TOKENS}` `` |
| `split_loop.mjs:128-130` (`runSplitDecode`) | `condBias.shape[1] !== N_TOKENS` | `` `condBias has ${condBias.shape[1]} tokens per position, nTokens is ${N_TOKENS}` `` |
| `split_driver.mjs:98-103` (`checkInputCount`) | Split graph session input count mismatches expected order length | `` `${name} session has ${session.inputNames.length} inputs, expected ${expected.length} (${expected.join(", ")})` `` |
| `split_driver.mjs:118-128` (`resolveFamily`) | Invalid `manifest.alphabet`, `manifest.n_tokens`, or `manifest.omit_bias` | `"manifest.alphabet must be a non-empty string"`, `` `manifest.n_tokens ${manifest.n_tokens} != alphabet length ${nTokens}` ``, or `` `manifest.omit_bias must be a finite number, got ${omitBias}` `` |
| `split_driver.mjs:147-160` (`resolveGraph`) | Invalid `manifest` per-graph `input_names` or `input_dtypes` | Non-empty array check, non-empty string check, duplicate check, session input count check, and `input_dtypes.length === declared.length` check |
| `split_driver.mjs:175-178` (`buildFeeds`) | Missing tensor for a declared graph input name | `` `no input tensor for graph input "${order[i]}" (session input ${session.inputNames[i]})` `` |
| `split_driver.mjs:218-224` (`createSplitSampler`) | `manifest.buckets` is an array, but `opts.manifestBucket` is omitted or not found | `"createSplitSampler: opts.manifestBucket is required when the manifest has buckets"` or `` `manifest has no bucket ${manifestBucket}` `` |

### 7.2 Hard Limits
- **Maximum structure length:** `256` residues (`BUCKETS = [128, 256]`, `aminx_sampler.mjs:41, 54`, `browser_integration.md:128`).
- **PRNG `nextBelow` bound:** `0x100000000` ($2^{32}$, `runspec_core.mjs:28, 71`).
- **Alphabet size default:** `21` (`MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"`, `runspec_core.mjs:21`, `split_loop.mjs:25`).

### 7.3 Unvalidated Edge Cases (Silent Coercions or Silent No-Ops in JS)
1. **Out-of-bounds residue keys in `fixed_positions`, `bias_AA_per_residue`, and `omit_AA_per_residue`:** `positionEntries` (`runspec_core.mjs:228-231`) parses keys with `Number(key)` without checking `0 <= pos < length` or `Number.isInteger(pos)`. Indexing a JS `TypedArray` with a negative index, `>= length`, or `NaN` silently writes nothing and does **not** throw. Furthermore, when `createSampler` or `createSplitSampler` pads `structure` to `bucket` first, indices in `nReal .. bucket - 1` are accepted as valid indices on the padded tensors!
2. **Explicit `decoding_order` values outside `0 .. L - 1`:** `runspec_core.mjs:327-335` checks only `decodingOrder.length === length` and `new Set(decodingOrder).size === length` after `Int32Array.from(requested)`. An array of `L` distinct negative or `>= L` integers passes JS validation without throwing. Also, an explicit `decoding_order` of length `nReal` (when `nReal < bucket`) throws in `createSampler`/`createSplitSampler` because `buildP07TypedInputs` checks `decodingOrder.length === bucket`, not `nReal`.
3. **Overlapping groups in `tied_positions`:** `tieGroupMap` (`runspec_core.mjs:169-186`) iterates groups sequentially and sets `ids[member] = gid` for each group's minimum member, without a transitive union-find merge. If `[[0, 2], [2, 5]]` is passed, after the first group `ids[0] = 0, ids[2] = 0`, and after the second group `ids[2] = 2, ids[5] = 2`, splitting `[0]` and `[2, 5]` instead of merging `[0, 2, 5]`.
4. **Empty `structure` (`L = 0`):** `inferCoordsShape([])` (`runspec_core.mjs:239-241`) throws `TypeError: Cannot read properties of undefined (reading 'length')` on `coords[0].length` unless `coords_shape: [0, 4, 3]` is explicitly supplied. `pickBucket(0)` returns `128`, and `padStructure` pads an `L = 0` structure with `0` for `residue_index`/`chain_index` and `undefined` for `chain_ids` (`aminx_sampler.mjs:96, 101, 115`).

---

## 8. MEASURED vs ASSERTED

### 8.1 Directly Tested in Local `aminx/aminx_sampler.test.mjs` (5/5 tests)
The only test file included in `./aminx/` is `aminx/aminx_sampler.test.mjs` (`lines 1-89`), which tests **5** behaviors:
1. `pickBucket` returns `128` for `1` and `128`, `256` for `129` and `256`, throws `/exceeds/` for `257`, and `BUCKETS` equals `[128, 256]` (`aminx_sampler.test.mjs:33-40`).
2. `padStructure(s, 6)` on an `L = 3` structure zero-pads `mask` (`[1, 1, 1, 0, 0, 0]`), `coords` (indices `36..71` all `0`, `coords_shape` `[6, 4, 3]`), and `native_tokens` (`[0, 1, 2, 0, 0, 0]`), and repeat-last-pads `residue_index` (`[10, 11, 12, 12, 12, 12]`), `chain_index` (`[0, 0, 1, 1, 1, 1]`), and `chain_ids` (`["A", "A", "B", "B", "B", "B"]`) (`aminx_sampler.test.mjs:42-59`).
3. `padStructure(s, 4)` on an `L = 4` structure is a reference identity no-op (`assert.equal(padStructure(s, 4), s)`, `aminx_sampler.test.mjs:61-64`).
4. `buildInputs(s, { seed: 7, temperature: 0.3 })` returns `Float32Array` or `Int32Array` `.data` for all 11 `P07_INPUT_ORDER` keys whose contents and shapes match `buildP07Inputs(s, runspec)` (`aminx_sampler.test.mjs:66-80`).
5. `pickBucket` + `padStructure` + `buildInputs` compose for an `L = 5` structure with `{ seed: 1 }` to produce `mask.shape[0] === 128` and `coords.data.length === 128 * 4 * 3` (`aminx_sampler.test.mjs:82-89`).

*(Note: `aminx_sampler.test.mjs` does **not** directly unit-test `bias_AA`, `bias_AA_per_residue`, `omit_AA`, `omit_AA_per_residue`, `chains_to_design`, `fixed_positions`, `tied_positions`, explicit `decoding_order`, `split_loop.mjs`, or `split_driver.mjs`. `browser_integration.md:172` mentions a separate `runspec.test.mjs` (11/11), which is not among the 8 files in `./aminx/`.)*

### 8.2 Measured Parity and Performance Numbers Cited in `browser_integration.md`
Reported in `aminx/browser_integration.md:161-295, 332-336` from recorded runs (notably bathos run `fd80f81d-3788-49c1-919f-e86acde3c4eb` at commit `443c5b4f`):
- **RunSpec knobs gate (`fd80f81d`, `browser_integration.md:168`):** JAX vs ORT-CPU vs ORT-Web across all 7 RunSpec knobs, `L128` and `L256`, 288 cells — **288/288** bitwise-identical tokens against ORT-CPU and **288/288** against ORT-Web; max log-prob difference **1.013e-04** (bound `2e-4`).
- **Teacher-forced vs reference ProteinMPNN (`fd80f81d`, `browser_integration.md:169`):** Per-position log-probs vs `LigandMPNN@26ec57ac` (checkpoint `proteinmpnn_v_48_020`, same fed sequence and order) — max difference **4.482e-05 nats** (bound `1e-4`).
- **Sampling correctness (`fd80f81d`, `browser_integration.md:170`):** Gumbel-max draws vs `softmax(logits/T)` total variation **0.00458** (bound `0.03`) over 200,000 draws; decoding-order uniformity $p = 0.753$ over 240,000 draws; **0** fixed-first violations; **0** omitted-class draws.
- **Instrument sensitivity (`fd80f81d`, `browser_integration.md:171`):** **4/4** planted defects detected (bias-frozen wrapper, `badGumbelFromUniform`, `biasedShuffle`, and unpartitioned uniform shuffle).
- **Split-graph parity (`browser_integration.md:259-267, 332-336`):**
  - Split vs monolith and vs JAX on ORT-CPU: **56/56** exact tokens at `L128` (run `fba872ce`) and **32/32** exact tokens at `L256` (run `ed2a2617`).
  - `split_loop.mjs` on ORT-Web wasm in Node: **56/56** exact tokens at `L128` (run `bc8eb3d7`).
  - `split_loop.mjs` in headless Chromium: **8/8** exact tokens at `L128` 1-thread (run `13e06211`), **8/8** exact tokens and bit-identical log-probs at `L128` 4-threads with COOP/COEP (run `da02516e`), and **4/4** exact tokens at `L256` 4-threads (run `8a3bb1d6`), with log-prob difference `5.7220458984375e-06` (`4.768e-06` gap noted at line 284).
  - Split vs reference ProteinMPNN (`LigandMPNN@26ec57ac`), teacher-forced direct comparison (scoped to untied lanes on fully-designed structures at `L=128`): **3.822e-05 nats** (bound `1e-4`, run `ed4f0b77`, `browser_integration.md:266-272`).
- **Measured performance (`browser_integration.md:190-220, 249-251`):**
  - Timer control: planted `250.0 ms` delay measured as `250.2 ms` (ratio `1.001`).
  - Node wasm single-threaded at `L=128` (run `be45e729`): Monolith median **19.96 s** (session create `589 ms`, peak RSS `550 MB`, download `19.43 MB` at L128 / `31.52 MB` at L256); Four-graph split median **17.19 s** (session create `364 ms`, peak RSS `550 MB`, download `6.64 MB` at L128 / `6.71 MB` at L256).
  - Headless Chromium split-graph wasm: `L=128` 1 thread = **15.77 s** (`15.48–16.85 s`, run `69bac70f`); `L=128` 4 threads = **6.80 s** (`6.58–7.10 s`, run `7fdb4001`); `L=256` 4 threads = **27.18 s** (`27.03–28.63 s`, run `4669e76f`).

### 8.3 Pending, Untested, or Unmeasured Claims (Must NOT Be Promoted)
1. **Free-sampling distributional agreement vs. reference ProteinMPNN — PENDING (`browser_integration.md:9-10, 173, 183-184`):** Explicitly marked **"Pending. Calibration running on titanix; validate shards next on Engaging"** and described as *"the one remaining gap in this table"*. Teacher-forced log-probs are measured, but free-sampling distributional parity against `LigandMPNN@26ec57ac` is **not** completed.
2. **Direct split-graph teacher-forced parity on fixed/tied/L256 cases (`browser_integration.md:270-272`):** The direct teacher-forced check against PyTorch reference (`3.822e-05 nats`, `ed4f0b77`) was scoped to **untied lanes on fully-designed structures at `L=128`**; fixed-position, tied-position, and `L=256` split cases were checked via exact-token parity against monolith/JAX rather than direct PyTorch teacher-forcing.
3. **WebGPU execution provider — UNTESTED / NEGATIVE PROBE (`browser_integration.md:159, 175, 236, 338, 340-369`):** Probed in run `ca0ea202` on WSL2 without GPU passthrough; `navigator.gpu` was present but **no adapter could be obtained**, so ORT was never invoked with WebGPU. All statements about WebGPU graph partitioning (`TopK` `int64` on CPU, Graph D on GPU) are explicitly labeled *"from design rather than measurement"* (`browser_integration.md:352`).
4. **Mobile browser memory and speed — NOT TESTED (`browser_integration.md:158`):** Explicitly states *"Mobile hasn't been tested."*
5. **PottsMPNN / custom manifest support in `split_driver.mjs` (`split_driver.mjs:15-19, 113-162`):** `resolveFamily` and `resolveGraph` support custom `MANIFEST.json` overrides for a second model family (PottsMPNN), but no tests or measurements for PottsMPNN or non-default manifests exist in the provided files.

---

## 9. ATTRIBUTION FACTS

- **Exact copyright line (`aminx/LICENSE:3`):**
  ```text
  Copyright (c) 2025 Marielle Russo
  ```
- **Upstream repository HEAD SHA (`aminx/PROVENANCE.txt:1`):**
  ```text
  0729c4589355f65ff0719fa143cf311f6b44b598
  ```
- **Original upstream paths and byte sizes of all input files (`aminx/PROVENANCE.txt:2-8`):**

| Local Path | Original Upstream Path in `maraxen/aminx` | Byte Size in `PROVENANCE.txt` |
| :--- | :--- | ---: |
| `aminx/runspec_core.mjs` | `browser/aminx-sampler/runspec_core.mjs` | `13586` |
| `aminx/aminx_sampler.mjs` | `browser/aminx-sampler/aminx_sampler.mjs` | `7295` |
| `aminx/split_loop.mjs` | `browser/aminx-sampler/split_loop.mjs` | `9040` |
| `aminx/split_driver.mjs` | `browser/aminx-sampler/split_driver.mjs` | `12254` |
| `aminx/aminx_sampler.test.mjs` | `browser/aminx-sampler/aminx_sampler.test.mjs` | `3010` |
| `aminx/browser_integration.md` | `browser_integration.md` | `22317` |
| `aminx/LICENSE` | `LICENSE` | `1071` |
| `aminx/PROVENANCE.txt` | *(local provenance metadata file)* | — |

---

## 10. OPEN QUESTIONS

Because the ONNX graphs (`p07_sample_L*.onnx` and `p07_{encoder,wave,decoder,fuse}_L*.onnx`) and Python export scripts (`src/aminx/inference/decode/autoregressive.py`, `src/aminx/export/wrappers.py`) are not present locally, a reimplementer building against a custom JS ProteinMPNN engine would still have to determine the following behaviors that are **unclear from the local JS source**:

1. **Exact logit-fusion and Gumbel-sampling formula inside Graph F (`runFuse`, `split_loop.mjs:192-202`):**
   - When a tie group spans multiple positions (`mask_group[g, pos] === 1`), how does Graph F combine per-position decoder `logits[0, pos, :]` and `cond_bias[pos, :]` into `avg_stored[g, :]`? Specifically:
     - Is `avg_stored[g, :]` an arithmetic mean $\frac{1}{|S_g|}\sum_{i \in S_g} (\text{logits}_i + \text{cond\_bias}_i)$ or a sum, and is `cond_bias` added before or after averaging across the group?
     - Are padded or masked positions (`mask[pos] === 0`) inside a tie group weighted by `0` or included equally in `mask_group`?
     - Does `avg_stored` store the biased logits $\ell + b$, the temperature-scaled logits $(\ell + b)/T$, or the raw un-biased averaged logits? (Note: `split_loop.mjs:231` runs `logSoftmaxRow(row)` directly on `avg_stored` without dividing by `temperature` or adding `condBias` in JS.)
     - For sampling `final_token[g]`, is the formula $\arg\max_k \left( \frac{\text{avg\_stored}[g, k]}{\text{temperature}} + \text{gumbel\_noise}[\text{group\_id}[g], k] \right)$, or is temperature applied to Gumbel noise instead ($\text{avg\_stored}[g, k] + T \cdot g_k$)?
     - Does Graph F force the unknown token `'X'` (index `20`) to be masked out by default, or can `'X'` be sampled whenever `omit_AA` does not include `"X"`?
2. **Mixed fixed + designed tie groups (`runspec_core.mjs:99-105`, `split_loop.mjs:145-147, 196-214`):**
   - If position `A` is fixed (`fixed_positions: { A: "M" }`) and tied to an unfixed position `B` (`tied_positions: [[A, B]]`), `designedFlags` marks **both** `A` and `B` as `designed = 1` (`runspec_core.mjs:99-105`), and `sequence[A]` is initialized to `"M"` (`split_loop.mjs:146`). When the wave containing `{A, B}` runs, Graph F receives `fixed_mask` and `fixed_tokens`, and then `split_loop.mjs:204-214` overwrites `sequence[pos] = final_token[g]` for **all** members of the group (including `A`). Whether Graph F forces `final_token[g]` to `fixed_tokens[A]` when any member (or `group_id[g]`) has `fixed_mask === 1`, or samples freely and overwrites `A`, is inside Graph F and **unclear from JS source**.
3. **Graph W's wave packing and `ar_mask` construction (`split_loop.mjs:135-142`):**
   - What is the wave slot width `G` (e.g., `G = 1` vs. parallel independent groups per wave), how does Graph W pack `decoding_order` + `tie_group_map` into `[nWaves, G]`, and under what conditions can a position remain unscheduled (`pos_first_rank[pos] >= nWaves * G`, `split_loop.mjs:142, 221-222`)? (For instance, are fixed positions scheduled into waves by Graph W or skipped via `pos_first_rank = nWaves * G` so their `logProbs` row comes from `logSoftmaxRow([0, ..., 0])`?)
   - How does `ar_mask` (`[L, L]`) treat tied peers (can members of the same tie group attend to each other during their joint decoding step, and do fixed positions attend to each other according to `decoding_order` or all-to-all)?
4. **Padding vs. unpadded PRNG alignment for non-bucketed engines (`aminx_sampler.mjs:176-179`):**
   - `aminx` pads `structure` to `bucket` (`128` or `256`) **before** calling `buildP07TypedInputs`, so `shuffle` and `gumbel_noise` always run at `length = bucket`. If a custom JS engine runs unpadded at `length = nReal`, should it still pad or simulate `bucket` (`128`/`256`) during `buildP07TypedInputs` to achieve bit-identical sequences for the same `(structure, RunSpec, seed)`?
5. **User-facing indexing when `nReal < bucket` and explicit `decoding_order` is supplied (`aminx_sampler.mjs:178-179`, `runspec_core.mjs:327-330`):**
   - Because `createSampler` and `createSplitSampler` pass the `bucket`-padded structure into `buildP07TypedInputs`, any explicit `runspec.decoding_order` passed to `sampler.sample(structure, runspec)` must have length `bucket` (`128` or `256`), not `nReal`, or `buildP07TypedInputs` throws `explicit decoding_order length != L`. Whether callers are expected to pad explicit `decoding_order` arrays to `bucket` themselves is not documented in `browser_integration.md`.
