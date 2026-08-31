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
 */
export const MODEL_BUNDLES = {
  monomer: {
    model: "model_1_ptm",
    directory: "./model/",
    release: "model1-ptm",
    variable: "LOCALFOLD_INCLUDE_MODEL",
    load: () => import("./monomer.js"),
  },
  multimer: {
    model: "model_1_multimer_v3",
    directory: "./model-multimer/",
    release: "model1-multimer-v3",
    variable: "LOCALFOLD_INCLUDE_MULTIMER_MODEL",
    load: () => import("./multimer.js"),
  },
};

/** @typedef {keyof typeof MODEL_BUNDLES} ModelFamily */

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
