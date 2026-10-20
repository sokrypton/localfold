/**
 * BioEmu2 on the page: its bundle, its network kept between folds, and its job.
 *
 * 🔴 THE JOB GOES THROUGH THE ARCHIVE'S OWN WRITER. The page's entity rows
 * become an AlphaFold 3 job with `jobInputJson` (web/job-json.js) - the same
 * function that writes a fold's request into its archive - and that job becomes
 * BioEmu2's features with shared/bioemu2/featurise.js, which is held field for
 * field to af3-any-model's port. A second conversion here would be a second
 * place for a ligand or a modification to be read differently.
 *
 * 🔴 AND THE NETWORK IS KEPT, WITH ITS WEIGHTS ON THE DEVICE, until the page
 * folds with another model (web/app.js calls releaseBioemu2 beside
 * releaseAllWeights). Re-uploading 815 MiB of float16 weights per fold is the
 * cost residency exists to avoid.
 */
import { HttpTensorStore } from "../shared/bundles/http-tensor-store.js";
import { readTensor } from "../shared/weights/dtype.js";
import { bundleBaseUrl, loadManifest } from "../shared/bundles/manifests/index.js";
import { bioemu2JobFromAf3Json, featuriseBioemu2, parseContacts } from "../shared/bioemu2/featurise.js";
import { fetchComponents, smilesTemplates, splitA3mByChains } from "../shared/bioemu2/inputs.js";
import { Bioemu2Gpu } from "../webgpu/bioemu2/network.js";
import { BIOEMU2_SAMPLER } from "../shared/bioemu2/weights.js";

/** The page's step dial for this model: the vendor's 200 first, and cheaper ones for a quick look. */
export const BIOEMU2_COUNTS = [200, 100, 50, 25];
export const BIOEMU2_DEFAULT_STEPS = BIOEMU2_SAMPLER.steps;

let weightsPromise;

/** The bundle as a reader webgpu/bioemu2 takes, memoised; `onProgress` reports the download. */
export function loadBioemu2Weights(onProgress) {
  weightsPromise ??= (async () => {
    const manifest = await loadManifest("bioemu2");
    const store = await HttpTensorStore.fromManifest(bundleBaseUrl("bioemu2"), manifest, onProgress);
    store.prefetch();
    const read = (name) => store.tensor(name);
    read.source = async (name) => {
      await store.open(name);
      return store.tensorSource(name);
    };
    read.decode = (source) => readTensor(source.record, source.buffer, source.byteOffset, true);
    read.has = async (name) => manifest.tensors[name] !== undefined;
    read.manifest = async () => manifest;
    return { read, manifest, megabytes: store.totalBytes / 1048576 };
  })();
  weightsPromise.catch(() => { weightsPromise = undefined; });
  return weightsPromise;
}

const networks = new WeakMap();

/** One network a device, its weights resident between folds. */
export function bioemu2Network(device, read) {
  let network = networks.get(device);
  if (network === undefined || network.read !== read) {
    network?.ops.destroyWeights();
    network = new Bioemu2Gpu(device, read);
    networks.set(device, network);
  }
  return network;
}

/** Give the device back: called when the page folds with another model. */
export function releaseBioemu2(device) {
  const network = networks.get(device);
  if (network === undefined) return 0;
  const bytes = network.ops.weightBytes + network.ops.bytes;
  network.ops.destroyWeights();
  networks.delete(device);
  return bytes;
}

/**
 * The page's rows -> BioEmu2's features.
 *
 * @param jobJson the job as `jobInputJson` writes it
 * @param alignment the fold's merged A3M over its protein chains (or null), and `proteinChains` their sequences in
 *   the order the merge used; each chain gets its own columns back
 * @param options `mode`, `contacts` (BioEmu2 Contact JSON, a string or a parsed list), `signal`
 */
export async function bioemu2Features(jobJson, { alignment = null, pairedAlignment = null, proteinChains = [],
                                                 mode = "structure", contacts = null, signal } = {}) {
  const job = bioemu2JobFromAf3Json(typeof jobJson === "string" ? JSON.parse(jobJson) : jobJson);
  // each protein chain's slice of the merged alignment, by sequence (a homomer's copies share one)
  if (alignment !== null && proteinChains.length > 0) {
    const lengths = proteinChains.map((chain) => chain.length);
    const unpaired = splitA3mByChains(alignment, lengths);
    const paired = pairedAlignment === null ? null : splitA3mByChains(pairedAlignment, lengths);
    const bySequence = new Map(proteinChains.map((chain, at) => [chain.toUpperCase(),
      { unpaired: unpaired[at], paired: paired?.[at] ?? null }]));
    for (const chain of job.chains) {
      if (chain.kind !== "protein") continue;
      const found = bySequence.get(chain.sequence.toUpperCase());
      if (found === undefined) continue;
      chain.unpairedMsa = found.unpaired;
      chain.pairedMsa = found.paired;
    }
  }
  const parsedContacts = contacts === null || contacts === "" ? null
    : parseContacts(typeof contacts === "string" ? JSON.parse(contacts) : contacts);
  const ccd = await fetchComponents(job, { signal });
  const templates = await smilesTemplates(job);
  return { job, ...featuriseBioemu2(job, { ccd, mode, contacts: parsedContacts, smilesTemplates: templates }) };
}

/**
 * A starting frame for propagation out of a PDB, in the featuriser's atom order: every atom key (chain, residue,
 * name) must be in the file - a structure from a different job is refused by name rather than read partially.
 */
export function x0FromPdb(text, atomKeys) {
  const where = new Map();
  for (const line of text.split(/\r?\n/)) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    const alt = line[16];
    if (alt !== " " && alt !== "A" && alt !== undefined) continue;
    const key = `${line[21]}|${Number(line.slice(22, 26))}|${line.slice(12, 16).trim()}`;
    if (!where.has(key)) where.set(key, [Number(line.slice(30, 38)), Number(line.slice(38, 46)), Number(line.slice(46, 54))]);
  }
  const x0 = new Float32Array(atomKeys.length * 3);
  const missing = [];
  atomKeys.forEach(([chain, residue, name], at) => {
    const xyz = where.get(`${chain}|${residue}|${name}`);
    if (xyz === undefined) { missing.push(`${chain}${residue}:${name}`); return; }
    x0.set(xyz, at * 3);
  });
  if (missing.length > 0) {
    throw new Error(`that structure lacks ${missing.length} atom${missing.length === 1 ? "" : "s"} of this job,`
      + ` e.g. ${missing.slice(0, 3).join(", ")} - propagate from a BioEmu2 fold of the same input`);
  }
  return x0;
}

/**
 * Which of BioEmu2's tokens the viewer draws as one position: a modified residue is one token per atom here and one
 * residue on screen, so its tokens collapse onto the one carrying CA (or C1'); a ligand stays one token per atom, as
 * the viewer draws it. The contact map is shown in that space (contactMapFor).
 */
export function bioemu2ViewerTokens(topology, tokens) {
  const byResidue = new Map();
  for (let a = 0; a < topology.names.length; a += 1) {
    const residue = topology.residueIdx[a];
    if (!byResidue.has(residue)) byResidue.set(residue, { tokens: new Set(), centre: -1, polymer: topology.molType[a] !== 3 });
    const entry = byResidue.get(residue);
    entry.tokens.add(topology.tokenIdxAtAtom[a]);
    if (topology.names[a] === "CA" || topology.names[a] === "C1'") entry.centre = topology.tokenIdxAtAtom[a];
  }
  const dropped = new Set();
  for (const entry of byResidue.values()) {
    if (!entry.polymer || entry.tokens.size < 2 || entry.centre < 0) continue;
    for (const token of entry.tokens) if (token !== entry.centre) dropped.add(token);
  }
  const keep = [];
  for (let token = 0; token < tokens; token += 1) if (!dropped.has(token)) keep.push(token);
  return keep;
}
