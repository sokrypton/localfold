// A whole BioEmu2 fold on the GPU: an AlphaFold 3 job in, samples out.
//
//     node tools/gpu-chrome.mjs tools/gpu/fold-bioemu2.js --job=/tools/fixtures/bioemu2/6mrr.json \
//       --crystal=/tools/fixtures/6mrr-crystal.pdb [--model=/model-bioemu2-int5] [--samples=1] [--steps=200]
//       [--seed=1] [--mode=structure|equilibrium] [--contacts=/tools/fixtures/bioemu2/contacts.json]
//       [--propagate=<lag ns>] [--propagation-steps=1] [--pdb] [--allow-broken-geometry]
//
// 🔴 THE PORT FOLDING, NOT A DIFFERENTIAL: the job goes through shared/bioemu2/featurise.js and every stage after
// it is this port's, so what this answers is whether they compose into a structure. The sampler's draws are ours, so
// the GATE is the geometry - consecutive CA spacing through shared/af3/chain-geometry.js, the same band every fold
// tool asserts - and the score is the RMSD to a crystal and the bond lengths against the CCD
// (tools/gpu/bond-geometry.js), never the vendor's coordinates.
//
// --propagate=<ns> is BioEmu2's propagation: the first sample, re-run as the starting frame of a trajectory of
// --propagation-steps lag-time steps (equilibrium mode, the companion through the sampler).
import { featuriseBioemu2, bioemu2JobFromAf3Json, parseContacts } from "../../shared/bioemu2/featurise.js";
import { bundleReader } from "../../shared/bioemu2/weights.js";
import { bioemu2Pdb, alphaCarbonSpacings } from "../../shared/bioemu2/structure.js";
import { foldBioemu2 } from "../../webgpu/bioemu2/fold.js";
import { fetchComponents, smilesTemplates } from "../../shared/bioemu2/inputs.js";
import { superpose } from "./superpose.js";
import { bondGeometry } from "./bond-geometry.js";
import { assertChainGeometry, chainGeometryOf } from "./chain-geometry.js";

const option = (args, name, fallback) => (args ?? []).find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

function crystalAlphaCarbons(text, chain) {
  const residues = new Map();
  for (const line of text.split("\n")) {
    if (!line.startsWith("ATOM") || line[21] !== chain || line.slice(12, 16).trim() !== "CA") continue;
    if (line[16] !== " " && line[16] !== "A") continue;
    const number = Number(line.slice(22, 26));
    if (!residues.has(number)) residues.set(number, [Number(line.slice(30, 38)), Number(line.slice(38, 46)), Number(line.slice(46, 54))]);
  }
  return [...residues.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
}

export async function main(device, args) {
  const jobUrl = option(args, "job", "/tools/fixtures/bioemu2/6mrr.json");
  const model = option(args, "model", "/model-bioemu2-int5");
  const crystalUrl = option(args, "crystal", null);
  const chain = option(args, "chain", "A");
  const contactsUrl = option(args, "contacts", null);
  const lag = option(args, "propagate", null);
  const job = bioemu2JobFromAf3Json(await (await fetch(jobUrl)).json());
  const ccd = await fetchComponents(job, { local: (code) => `/tools/fixtures/ccd/${code}.cif` });
  const contacts = contactsUrl ? parseContacts(await (await fetch(contactsUrl)).json()) : null;
  const mode = lag ? "equilibrium" : option(args, "mode", "structure");
  const t0 = performance.now();
  const { features, atomKeys, topology, tokens, atoms } = featuriseBioemu2(job, {
    ccd, mode, contacts, smilesTemplates: await smilesTemplates(job) });
  const featurisedMs = performance.now() - t0;
  const read = bundleReader(model);
  await read.prefetch();
  const options = { samples: Number(option(args, "samples", "1")), steps: Number(option(args, "steps", "200")),
                    seed: Number(option(args, "seed", "1")), mode };
  const t1 = performance.now();
  let result = await foldBioemu2(device, read, features, options);
  const foldMs = performance.now() - t1;
  let propagated = null;
  if (lag) {
    const x0 = result.samples[0].subarray(0, atomKeys.length * 3);
    propagated = await foldBioemu2(device, read, features, {
      ...options, samples: 1, propagation: { x0, lagNs: Number(lag), steps: Number(option(args, "propagation-steps", "1")) } });
  }
  const crystal = crystalUrl ? crystalAlphaCarbons(await (await fetch(crystalUrl)).text(), chain) : null;
  const conformers = await (await fetch("/tools/oracle/reference-conformers.json")).json();
  const score = (coords) => {
    const { spacing, centres } = alphaCarbonSpacings(atomKeys, topology, coords);
    const geometry = chainGeometryOf(spacing);
    const pdb = bioemu2Pdb(atomKeys, topology, coords);
    const bonds = bondGeometry(pdb, conformers);
    let fit;
    if (crystal) {
      const model = centres.filter((c) => c.chain === chain).map((c) => [coords[c.at * 3], coords[c.at * 3 + 1], coords[c.at * 3 + 2]]);
      if (model.length !== crystal.length) throw new Error(`${model.length} model CAs against ${crystal.length} in the crystal`);
      fit = superpose(model, crystal);
    }
    return { geometry, pdb, bonds, fit };
  };
  // 🔴 A NON-FINITE COORDINATE IS A FAILED FOLD, said here: chain-geometry's verdict skips a fold with no measurable
  // alpha carbons (a ligand has none), so a sample of NaN passed the geometry gate with its CA-CA reported null - and
  // with it check-stock-flags.mjs, whose signature is the atom checksum (a NaN folds to 0 there). That is how a stock
  // Chrome on Apple silicon folding NaN went unseen (webgpu/bioemu2/network.js, the trunk's accumulation)
  const allSamples = propagated ? [...result.samples, ...propagated.samples] : result.samples;
  allSamples.forEach((xs, k) => {
    const bad = xs.subarray(0, atomKeys.length * 3).findIndex((v) => !Number.isFinite(v));
    if (bad >= 0) throw new Error(`sample ${k}: a non-finite coordinate (atom ${Math.floor(bad / 3)}: ${xs[bad]}) - the fold failed`);
  });
  const scored = result.samples.map(score);
  for (const s of scored) {
    assertChainGeometry(s.geometry, { doc: "docs/BIOEMU2.md", allow: args.includes("--allow-broken-geometry") });
  }
  const out = {
    job: jobUrl, model, tokens, atoms, mode, steps: options.steps, samples: options.samples,
    featuriseMs: Math.round(featurisedMs), foldMs: Math.round(foldMs), timings: result.timings, deviceMiB: result.deviceMiB,
    perSample: scored.map((s) => ({
      rmsd: s.fit ? Number(s.fit.rmsd.toFixed(3)) : null, tm: s.fit ? Number(s.fit.tm.toFixed(4)) : null,
      caca: Number(s.geometry.caca.toFixed(3)), worstCaca: Number(s.geometry.worstCaca.toFixed(3)),
      bondRms: Object.fromEntries(["mainchain", "sidechain", "peptide", "nucleic", "ligand"]
        .filter((k) => s.bonds[k].bonds > 0).map((k) => [k, Number(s.bonds[k].rms.toFixed(3))])),
      bondRatios: s.bonds.ratios,
    })),
    geometry: scored[0].geometry,
    // 🔴 A CHECKSUM OF THE FIRST SAMPLE'S ATOMS, the signature tools/gate-folds.mjs records: an arm that moved the
    // structure says so where a geometry summary would not
    atomChecksum: (() => {
      let sum = 0;
      // (of |x|: a sample is centred on its atoms, so a plain sum is ~0 whatever the structure)
      for (let i = 0; i < atomKeys.length * 3; i += 1) sum = (sum + Math.round(Math.abs(result.samples[0][i]) * 1000)) | 0;
      return sum;
    })(),
  };
  if (propagated) {
    const p = score(propagated.samples[propagated.samples.length - 1]);
    assertChainGeometry(p.geometry, { doc: "docs/BIOEMU2.md", allow: args.includes("--allow-broken-geometry") });
    const first = alphaCarbonSpacings(atomKeys, topology, result.samples[0]).centres;
    const a = first.map((c) => [result.samples[0][c.at * 3], result.samples[0][c.at * 3 + 1], result.samples[0][c.at * 3 + 2]]);
    const last = propagated.samples[propagated.samples.length - 1];
    const b = first.map((c) => [last[c.at * 3], last[c.at * 3 + 1], last[c.at * 3 + 2]]);
    out.propagation = { lagNs: Number(lag), frames: propagated.samples.length, fromStart: Number(superpose(b, a).rmsd.toFixed(3)),
                        rmsd: p.fit ? Number(p.fit.rmsd.toFixed(3)) : null, caca: p.geometry.caca, timings: propagated.timings };
  }
  if (args.includes("--pdb")) out.pdb = scored[0].pdb;
  return out;
}
