/**
 * ProteinEBM on the page: its bundle, its network kept between runs, and the starting structure.
 *
 * 🔴 IT IS NOT A FOLDING MODEL. ProteinEBM is an energy E(x | sequence) over a CA trace, and the page runs its
 * Langevin dynamics FROM a structure: the last fold of the same sequence (any model's), or an uploaded PDB. The
 * sequence box says what the structure must be of; a structure of anything else is refused by name rather than
 * simulated as though it were this protein.
 *
 * 🔴 AND THE NETWORK IS KEPT, WITH ITS WEIGHTS ON THE DEVICE, until the page runs another model (web/app.js calls
 * releaseProteinEbm beside releaseAllWeights). Each length records its own forward and backward once.
 */
import { HttpTensorStore } from "../shared/bundles/http-tensor-store.js";
import { bundleBaseUrl, loadManifest } from "../shared/bundles/manifests/index.js";
import { caTraceFromPdb, centre } from "../shared/proteinebm/model.js";
import { ProteinEbmGpu } from "../webgpu/proteinebm/network.js";

let weightsPromise;

/** The bundle as a reader webgpu/proteinebm takes, memoised; `onProgress` reports the download. */
export function loadProteinEbmWeights(onProgress) {
  weightsPromise ??= (async () => {
    const manifest = await loadManifest("proteinebm");
    const store = await HttpTensorStore.fromManifest(bundleBaseUrl("proteinebm"), manifest, onProgress);
    store.prefetch();
    const read = (name) => store.tensor(name);
    return { read, manifest, megabytes: store.totalBytes / 1048576 };
  })();
  weightsPromise.catch(() => { weightsPromise = undefined; });
  return weightsPromise;
}

const networks = new WeakMap();

/** One network a device, its weights resident between runs; recorded for `trace`'s length. */
export async function proteinEbmNetwork(device, read, trace) {
  let entry = networks.get(device);
  if (entry === undefined || entry.read !== read) {
    entry?.net.destroy();
    entry = { read, net: await ProteinEbmGpu.load(device, read), key: null };
    networks.set(device, entry);
  }
  const key = `${Array.from(trace.aatype).join(",")}|${Array.from(trace.residueIdx).join(",")}`;
  if (entry.key !== key) {
    await entry.net.prepare({ aatype: trace.aatype, residueIdx: trace.residueIdx });
    entry.key = key;
  }
  return entry.net;
}

export function releaseProteinEbm(device) {
  const entry = networks.get(device);
  if (entry === undefined) return;
  entry.net.destroy();
  networks.delete(device);
}

/**
 * The starting CA trace out of a PDB, checked against the sequence being run. `sequence` is the entity list's one
 * protein chain; the structure's first chain whose sequence it is is taken, centred.
 */
export function startingTrace(pdbText, sequence, where) {
  const chains = [...new Set(pdbText.split(/\r?\n/)
    .filter((line) => (line.startsWith("ATOM  ") || line.startsWith("HETATM")) && line.slice(12, 16).trim() === "CA")
    .map((line) => line[21]))];
  for (const chain of chains) {
    const trace = caTraceFromPdb(pdbText, chain);
    if (trace.sequence === sequence) return { ...trace, ca: centre(Float32Array.from(trace.ca)) };
  }
  const found = chains.map((chain) => caTraceFromPdb(pdbText, chain).sequence.length).join(", ");
  throw new Error(`${where} has no chain with this sequence (${sequence.length} residues; its chains have`
    + ` ${found || "no CA atoms"}) - ProteinEBM simulates a structure OF the sequence in the box`);
}
