/**
 * The model bundles this page can load, and how to reach each one.
 *
 * 🔴 ONE DESCRIPTION OF A BUNDLE, NOT FOUR. Adding a model used to mean
 * touching the loader, the deploy workflow, the site build and the manifest by
 * hand, and the multimer one arrived with its manifest FETCHED while the
 * monomer's was compiled in - so the two failed differently, and the one that
 * fetched failed on a 404 that named nothing. Everything a bundle needs is
 * here; its Python twin is BUNDLES in tools/write_manifest_module.py, and
 * tools/build_site.py checks the two agree.
 *
 * 🔴 THE MANIFESTS ARE LOADED LAZILY, and that is not a micro-optimisation.
 * Compiled in, each is ~100 KiB of JSON in the JS - so a static import of both
 * would put 200 KiB in front of every visitor, including the many who fold a
 * single chain and never touch multimer. A dynamic import is fetched when a
 * fold asks for that family and never otherwise.
 *
 * The loaders are written as literal `import()` calls rather than built from
 * `directory` because a bundler cannot follow a computed specifier; these
 * resolve statically and survive bundling.
 *
 * 🔴 `remote` IS WHERE THE SHARDS ACTUALLY LIVE, AND `directory` IS THE FALLBACK.
 * GitHub Pages publishes at most a gigabyte and the weights are most of it -
 * AF2 monomer is 227 MB and AF3 150 MB before a third model exists - so a page
 * that means to offer five of them cannot bake them into its own artefact. A
 * bundle with a `remote` fetches its shards from there and ships none; without
 * one it behaves exactly as before, which is what keeps an offline build
 * (tools/bundle.py) and a self-hosted copy working.
 *
 * 🔴 THE REVISION IS PINNED IN THE URL, NOT LEFT AT `main`. A shard fetched
 * from a moving branch can change under a manifest that did not, which is the
 * failure the shard cache token exists to prevent - and three separate hours
 * have gone into "<file> has an invalid byte length" already. A commit SHA
 * makes the URL immutable, which is also what lets the browser cache it
 * forever.
 */
export const MODEL_BUNDLES = {
  monomer: {
    model: "model_1_ptm",
    directory: "./model/",
    remote: "https://huggingface.co/sokrypton/localfold/resolve/71ece3571b80e6cdb06f3b95f785187171c718da/af2-monomer/",
    release: "model1-ptm",
    variable: "LOCALFOLD_INCLUDE_MODEL",
    load: () => import("./monomer.js"),
  },
  multimer: {
    model: "model_1_multimer_v3",
    directory: "./model-multimer/",
    remote: "https://huggingface.co/sokrypton/localfold/resolve/71ece3571b80e6cdb06f3b95f785187171c718da/af2-multimer/",
    release: "model1-multimer-v3",
    variable: "LOCALFOLD_INCLUDE_MULTIMER_MODEL",
    load: () => import("./multimer.js"),
  },
  // The whole AF3 diffuser at int5: trunk, diffusion head, confidence head.
  // 265 MiB, and af3.html folds a sequence with it.
  //
  // 🔴 DEEPMIND'S PARAMETERS, NOT OPENFOLD3'S, whatever the family name
  // suggests. They carry a Prohibited Use Policy, so build_site.py will not
  // publish them without LOCALFOLD_ACCEPT_MODEL_TERMS=alphafold3.
  af3: {
    model: "alphafold3",
    directory: "./model-af3-int5/",
    remote: "https://huggingface.co/sokrypton/localfold/resolve/71ece3571b80e6cdb06f3b95f785187171c718da/af3-int5/",
    release: "af3-int5",
    variable: "LOCALFOLD_INCLUDE_AF3_MODEL",
    load: () => import("./af3.js"),
  },
  // The same graph and the same 265 MiB, under the Apache License 2.0.
  //
  // 🔴 OpenBind-0 IS OpenFold3's v0.5.0 RELEASE, AND NOT ITS PREVIEW-2 WEIGHTS.
  // The two differ in forward conventions - see src/af3/dialect.js - so they
  // are different models, and this bundle's manifest names `openbind` so the
  // loader picks the right one. Reading the OpenFold3 porting notes as if they
  // described this release turns on branches it does not want.
  //
  // 🔴 AND IT CARRIES NO PROHIBITED USE POLICY, which is the point of offering
  // it: build_site.py publishes it with no LOCALFOLD_ACCEPT_MODEL_TERMS, and
  // the page's licence dialog offers it as the way past AF3's terms rather than
  // as a second-best.
  //
  // 🔴 AND THE RELEASE NUMBER IS PART OF THE NAME. Upstream's announcement
  // calls the model OpenBind-0; their registry's bare `openbind` is a name a
  // later release would answer to as well, which is how `openfold3` ended up
  // meaning two models with different forward conventions. See dialect.js.
  openbind0: {
    model: "openbind0",
    directory: "./model-openbind0-int5/",
    remote: "https://huggingface.co/sokrypton/localfold/resolve/71ece3571b80e6cdb06f3b95f785187171c718da/openbind0-int5/",
    release: "openbind0-int5",
    variable: "LOCALFOLD_INCLUDE_OPENBIND0_MODEL",
    load: () => import("./openbind0.js"),
  },
  // 🔴 OpenDDE FOLDS, AND ITS PIPELINE HAS TWO TOKEN SPACES. It is an
  // independent PyTorch reimplementation in the AlphaFold 3 family (Aureka
  // Research, Apache-2.0): the trunk runs on residues, and between the trunk
  // and the diffusion each standard residue is re-tokenised into a backbone
  // and a sidechain token, which the diffusion and its confidence head run on.
  // `src/af3/fold-opendde.js` is that driver; every stage inside it is the
  // AlphaFold 3 code at OpenDDE's widths.
  //
  // 481 tensors, 655.8 M parameters - upstream's own published count - at int5
  // in twelve shards, 473 MiB. Measured on 6MRR against the deposited
  // structure: RMSD 1.68 A, TM 0.865, CA-CA 3.68 A.
  opendde: {
    model: "opendde",
    directory: "./model-opendde-int5/",
    // 🔴 PINNED TO A COMMIT, NOT A BRANCH. A shard fetched from a moving branch
    // can change under a manifest that did not, and the failure names neither
    // half - see docs/HOSTING.md, where three hours went into exactly that.
    remote: "https://huggingface.co/sokrypton/localfold/resolve/068c905dfb0f8cf9b9432eef80d2220ef3ff697f/opendde-int5/",
    release: "opendde-int5",
    variable: "LOCALFOLD_INCLUDE_OPENDDE_MODEL",
    load: () => import("./opendde.js"),
  },
  // Boltz-2, an AlphaFold 3-lineage model with its own conditioning and its own
  // confidence module.
  //
  // 🔴 ITS WIDTHS ARE WHERE AlphaFold 3 HID A DISTINCTION. Its trunk single
  // is 384 and its diffusion single conditioning projects to 768, where
  // AlphaFold 3's are both 384 - so `seqChannels + targetFeatWidth` read as the
  // input width for one model and gave 1152 against a LayerNorm of 768 for
  // this one. `trunkSingleChannels` is derived from
  // `single_activations/weights` now rather than assumed; see
  // src/af3/diffusion-weights.js.
  //
  // 🔴 AND ITS CONFIDENCE HEAD IS A DIFFERENT MODULE IN FRONT OF THE SAME
  // PAIRFORMER - `~_boltz2_reembed` rather than `~_embed_features`, and EIGHT
  // blocks where every other model has four. `confidenceWeights` detects that
  // from the tensors rather than from the family name, which is why this entry
  // needs no branch of its own.
  //
  // 442 tensors at int5 in eight shards, 364 MiB.
  boltz2: {
    model: "boltz2",
    directory: "./model-boltz2-int5/",
    remote: "https://huggingface.co/sokrypton/localfold/resolve/068c905dfb0f8cf9b9432eef80d2220ef3ff697f/boltz2-int5/",
    release: "boltz2-int5",
    variable: "LOCALFOLD_INCLUDE_BOLTZ2_MODEL",
    load: () => import("./boltz2.js"),
  },
  // Protenix-v2, the other AlphaFold 3-lineage model in this round.
  //
  // 🔴 ITS TEMPLATE TERM CONCATENATES WHERE AlphaFold 3 PROJECTS. AF3 runs
  // nine separate feature projections (`template_pair_embedding_0..8`); this
  // one and boltz2 concatenate the features and apply a single `a_proj`. That
  // is read from the bundle rather than typed in - see
  // `src/af3/template-webgpu.js`.
  //
  // 404 tensors at int5 in eight shards, 334 MiB.
  protenix2: {
    model: "protenix2",
    directory: "./model-protenix2-int5/",
    remote: "https://huggingface.co/sokrypton/localfold/resolve/068c905dfb0f8cf9b9432eef80d2220ef3ff697f/protenix2-int5/",
    release: "protenix2-int5",
    variable: "LOCALFOLD_INCLUDE_PROTENIX2_MODEL",
    load: () => import("./protenix2.js"),
  },
  // IntelliFold-2 (intelligenAI, Apache 2.0) - the smallest dialect in the
  // family and the widest bundle.
  //
  // 🔴 ITS SHAPES ARE ITS OWN AND ITS CONVENTIONS ARE ALMOST AlphaFold 3's. The
  // trunk pair is 512 channels against AF3's 128 and the template stack 256
  // against 64, while the reference names it in exactly two convention tuples -
  // so everything that reads a width off the bundle transfers and everything
  // that had one typed in did not. That is how `const CHANNELS = 64` in the
  // template embedder was found. See docs/AF3.md.
  //
  // 406 tensors at int5 in forty shards, 612 MiB.
  intellifold2: {
    model: "intellifold2",
    directory: "./model-intellifold2-int5/",
    remote: null,
    release: "intellifold2-int5",
    variable: "LOCALFOLD_INCLUDE_INTELLIFOLD2_MODEL",
    load: () => import("./intellifold2.js"),
  },
  // RoseTTAFold3 (Baker lab / RosettaCommons foundry, BSD 3-Clause).
  //
  // 🔴 IT TOOK EIGHT BRANCHES AND TWO OF THEM MOVED THE FOLD BY ALMOST NOTHING
  // WHILE BEING CORRECT. 6MRR at 1.68 A / TM 0.911, inside af3-any-model's own
  // five-sample spread, with pLDDT 81.5 and pTM 0.843 - and the confidence head
  // read 53.8 / 0.175 for that same structure until its global norm landed. See
  // the arm table in docs/AF3.md.
  //
  // 446 tensors at int5 in sixteen shards, 266 MiB.
  rosettafold3: {
    model: "rosettafold3",
    directory: "./model-rosettafold3-int5/",
    remote: null,
    release: "rosettafold3-int5",
    variable: "LOCALFOLD_INCLUDE_ROSETTAFOLD3_MODEL",
    load: () => import("./rosettafold3.js"),
  },
  // ESMFold2-Experimental-Fast: no alignment, no template, one sequence.
  //
  // 🔴 IT IS TWO BUNDLES AND THE FIRST ENTRY IN THIS TABLE THAT IS. The folding
  // half is 122 MiB and the LANGUAGE MODEL it reads is a separate 224 MiB with
  // its own exporter - and the shim that joins them is per folding model, which
  // is why the ESM-C manifest carries the names of both. `companion` is that
  // link, and a loader that ignored it would fold with a language model whose
  // shim was trained against a different trunk: every shape agrees.
  // 🔴 THE NAME IS THE CHECKPOINT'S, NOT THE ARCHITECTURE'S. "esmfold2" alone
  // reads as ESM's released ESMFold2-Fast, which folds from ESM-C 6B and is a
  // different and better model; this is the 600M experimental one, which is the
  // one that fits a browser. Same reasoning as openbind0's number above, and
  // `esmfold2` survives as a ?model= alias so a saved link still works.
  "ef2-fast-600m": {
    model: "esmfold2-trunk",
    directory: "./model-esmfold2-int5/",
    remote: "https://huggingface.co/sokrypton/localfold/resolve/71ece3571b80e6cdb06f3b95f785187171c718da/ef2-fast-600m-int5/",
    release: "esmfold2-int5",
    variable: "LOCALFOLD_INCLUDE_ESMFOLD2_MODEL",
    companion: "esmc",
    load: () => import("./esmfold2.js"),
  },
  // 🔴 THE SAME FOLDING MODEL AGAINST A SMALLER TOWER, AND A SEPARATE CHECKPOINT
  // RATHER THAN A SWAP. `base300M-step1500k` is published beside
  // `base600M-step1500k` and its shim is trained for 30 layers x 960 against
  // the other's 36 x 1152, so the two towers are not interchangeable over one
  // set of folding weights - which is why this is a FAMILY with its own
  // companion and not an option on the loader. The folding model is the same
  // SIZE in both (171 M), so only the tower's 94 MiB separates the pairs:
  // 252.1 MiB against 346.1. Measured over sixteen held-out targets in
  // docs/EF2FAST.md, the median is 2.55 A against 2.52 - inside the sampler's
  // own 0.99-1.10 A seed spread - and the tails are 0.4 A worse.
  "ef2-fast-300m": {
    model: "esmfold2-trunk",
    directory: "./model-ef2-fast-300m-int5/",
    remote: "https://huggingface.co/sokrypton/localfold/resolve/71ece3571b80e6cdb06f3b95f785187171c718da/ef2-fast-300m-int5/",
    release: "ef2-fast-300m-int5",
    variable: "LOCALFOLD_INCLUDE_ESMFOLD2_MODEL",
    companion: "esmc-300m",
    load: () => import("./ef2-fast-300m.js"),
  },
  // 🔴 NOT A MODEL A PAGE OFFERS, AND THAT IS WHY IT IS NOT IN MODEL_FAMILIES.
  // ESM-C folds nothing on its own; it exists here so the loader, the shard
  // cache, the download dial and build_site.py can all treat it as a bundle.
  esmc: {
    model: "esmc",
    directory: "./model-esmc-600m-int3/",
    remote: "https://huggingface.co/sokrypton/localfold/resolve/71ece3571b80e6cdb06f3b95f785187171c718da/esmc-600m-int3/",
    release: "esmc-600m-int3",
    variable: "LOCALFOLD_INCLUDE_ESMC_MODEL",
    companion: undefined,
    foldingModel: false,
    load: () => import("./esmc.js"),
  },
  // ...and the 300M tower, whose shim belongs to the 300M folding model.
  "esmc-300m": {
    model: "esmc",
    directory: "./model-esmc-300m-int3/",
    remote: "https://huggingface.co/sokrypton/localfold/resolve/71ece3571b80e6cdb06f3b95f785187171c718da/esmc-300m-int3/",
    release: "esmc-300m-int3",
    variable: "LOCALFOLD_INCLUDE_ESMC_MODEL",
    companion: undefined,
    foldingModel: false,
    load: () => import("./esmc-300m.js"),
  },
};

/**
 * The families a page may be SET to, as opposed to the bundles it can load.
 *
 * 🔴 A COMPANION IS A BUNDLE AND NOT A CHOICE. ESM-C is in MODEL_BUNDLES so the
 * loader and the site build can reach it by name; offering it in the model
 * picker would be offering a language model as a structure predictor.
 */
export const FOLDING_FAMILIES = Object.entries(MODEL_BUNDLES)
  .filter(([, bundle]) => bundle.foldingModel !== false)
  .map(([family]) => family);

/**
 * The families whose featuriser has ligand tokens, nucleic chains and modified
 * residues - which is not the same question as whose GRAPH is AlphaFold 3's.
 *
 * 🔴 THIS USED TO BE `AF3_FAMILIES` DOING BOTH JOBS, AND THE SECOND MODEL THAT
 * NEEDED THEM SPLIT IT. ESMFold2 runs a different graph and the SAME all-atom
 * representation - `ref_pos`, `ref_element`, `ref_charge`,
 * `ref_atom_name_chars`, `ref_space_uid`, `atom_to_token`, `token_bonds` - so
 * a capability guard written as "is this an AF3 family" refuses a ligand under
 * a model that has ligand tokens, with a message naming a capability it has.
 * That exact mistake is recorded in CLAUDE.md for the AF3/OpenBind split; this
 * is the same mistake one model later.
 */
export const ALL_ATOM_FAMILIES = ["af3", "openbind0", "opendde", "boltz2",
                                  "protenix2", "intellifold2", "rosettafold3",
                                  "ef2-fast-600m", "ef2-fast-300m"];

/**
 * The families that predict a structure and nothing about it.
 *
 * 🔴 A MODEL WITHOUT A CONFIDENCE HEAD MUST NOT BE COLOURED BY pLDDT. Its
 * B-factor column is zero everywhere, and the pLDDT ramp paints zero RED - so
 * the whole structure reads as a uniformly terrible fold rather than an absent
 * measurement.
 *
 * 🔴 OpenDDE IS NOT ON THIS LIST ANY MORE. Its head is its own
 * parametrisation - see src/af3/opendde-confidence.js - and it is ported, so a
 * fold reports a pLDDT and a PAE. What it still does NOT report is pTM: that
 * needs a TM term, which AlphaFold 3's head emits and OpenDDE's does not, and
 * deriving one from the PAE would be a different quantity wearing pTM's name.
 */
export const MODELS_WITHOUT_CONFIDENCE = ["ef2-fast-600m", "ef2-fast-300m"];

/** Which models fold from a single sequence and take no alignment at all. */
export const SINGLE_SEQUENCE_FAMILIES = ["ef2-fast-600m", "ef2-fast-300m"];

/**
 * The families that build AlphaFold 3's graph, as opposed to AlphaFold 2's.
 *
 * 🔴 A LIST, NOT `family === "af3"`. That comparison was in five places and
 * every one of them meant "is this the AF3 pipeline" rather than "is this
 * DeepMind's checkpoint" - so a second AF3-graph bundle would have taken the
 * AlphaFold 2 branch at each of them, which is not a failure that announces
 * itself.
 */
export const AF3_FAMILIES = ["af3", "openbind0", "opendde", "boltz2", "protenix2",
                             "intellifold2", "rosettafold3"];

/** @typedef {keyof typeof MODEL_BUNDLES} ModelFamily */

/**
 * Where a family's shards are fetched from: its remote if it has one, and its
 * directory beside the page if not.
 *
 * 🔴 A TRAILING SLASH OR THE LAST SEGMENT IS LOST. Shard URLs are resolved with
 * `new URL(file, base)`, and a base of ".../resolve/abc123" without the slash
 * puts the shard next to `abc123` rather than inside it - a 404 naming a path
 * that looks almost right.
 *
 * @param {ModelFamily} family
 */
export function bundleBaseUrl(family) {
  const bundle = MODEL_BUNDLES[family];
  if (bundle === undefined) throw new RangeError(`unknown model family ${family}`);
  const base = bundle.remote ?? bundle.directory;
  return base.endsWith("/") ? base : `${base}/`;
}

/**
 * The tensor table for one family.
 * @param {ModelFamily} family
 */
export async function loadManifest(family) {
  const bundle = MODEL_BUNDLES[family];
  if (bundle === undefined) {
    throw new RangeError(`unknown model family ${family}:`
      + ` expected ${Object.keys(MODEL_BUNDLES).join(" or ")}`);
  }
  return (await bundle.load()).MANIFEST;
}
