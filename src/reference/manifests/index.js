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
    remote: "https://huggingface.co/sokrypton/localfold/resolve/2afc5a814499006d54f6ddd889944f0bae3e9442/af2-monomer/",
    release: "model1-ptm",
    variable: "LOCALFOLD_INCLUDE_MODEL",
    load: () => import("./monomer.js"),
  },
  multimer: {
    model: "model_1_multimer_v3",
    directory: "./model-multimer/",
    remote: "https://huggingface.co/sokrypton/localfold/resolve/2afc5a814499006d54f6ddd889944f0bae3e9442/af2-multimer/",
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
    remote: "https://huggingface.co/sokrypton/localfold/resolve/2afc5a814499006d54f6ddd889944f0bae3e9442/af3-int5/",
    release: "af3-int5",
    variable: "LOCALFOLD_INCLUDE_AF3_MODEL",
    load: () => import("./af3.js"),
  },
  // The same graph and the same 265 MiB, under the Apache License 2.0.
  //
  // 🔴 OpenBind IS OpenFold3's v0.5.0 RELEASE, AND NOT ITS PREVIEW-2 WEIGHTS.
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
  // 🔴 NO `remote` YET. Its shards have not been uploaded, so this bundle is
  // published beside the page. Give it a pinned `remote` the moment they are -
  // a commit SHA, and a trailing slash; see the notes at the top of this file.
  openbind: {
    model: "openbind",
    directory: "./model-openbind-int5/",
    release: "openbind-int5",
    variable: "LOCALFOLD_INCLUDE_OPENBIND_MODEL",
    load: () => import("./openbind.js"),
  },
};

/**
 * The families that build AlphaFold 3's graph, as opposed to AlphaFold 2's.
 *
 * 🔴 A LIST, NOT `family === "af3"`. That comparison was in five places and
 * every one of them meant "is this the AF3 pipeline" rather than "is this
 * DeepMind's checkpoint" - so a second AF3-graph bundle would have taken the
 * AlphaFold 2 branch at each of them, which is not a failure that announces
 * itself.
 */
export const AF3_FAMILIES = ["af3", "openbind"];

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
