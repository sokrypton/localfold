/**
 * The monomer tensor table, under the name it has always had here.
 *
 * The manifests now live in ./manifests/, one generated module per model, with
 * a registry beside them - see MODEL_BUNDLES there for why. This re-export
 * keeps `DEFAULT_MANIFEST` working for the published API in src/index.js and
 * for anything outside this repository that imports it.
 */
export { MANIFEST as DEFAULT_MANIFEST } from "./manifests/monomer.js";
