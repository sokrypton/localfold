// A whole BioEmu2 fold in WebGL2: an AlphaFold 3 job in, a structure out.
//
//     LOCALFOLD_GPU_TIMEOUT_MS=2700000 node tools/gpu-chrome.mjs tools/gpu/fold-bioemu2-webgl.js \
//       --job=/tools/fixtures/bioemu2/6mrr-68.json --crystal=/tools/fixtures/6mrr-crystal.pdb \
//       [--model=/model-bioemu2-int5] [--steps=20] [--samples=1] [--seed=1] [--passes=4]
//       [--mixer-blocks=48] [--pdb] [--allow-broken-geometry]
//
// 🔴 SCORE `6mrr-68.json` AGAINST THE CRYSTAL, NOT `6mrr.json`. The two fixtures
// are the same protein at 68 and 71 residues and the crystal resolves 68 alpha
// carbons, so the 71-residue job folds fine and then dies in the scorer - "71
// model CAs against 68 in the crystal" - after the whole fold has been paid for.
// Which is why the crystal is now counted BEFORE the fold rather than after:
// fourteen minutes of GPU went into learning that the first time.
//
// 🔴 THE PORT FOLDING, NOT A DIFFERENTIAL. Every stage gate in this family
// compares one stage against the WGSL one on the same page; this composes all
// five - input embedder, sequence embedder, trunk, distogram, denoiser - and
// asks whether they make a structure. The AF3 WebGL2 driver found three bugs no
// stage gate could see (a clamp, a bin count, a weight table read from the wrong
// group), which is the whole argument for having one.
//
// 🔴 AND IT NEEDS `LOCALFOLD_GPU_TIMEOUT_MS` TO FINISH AT ALL: `gpu-chrome.mjs`
// gives a module ten minutes and a 20-step fold of 71 tokens is longer than
// that, so the run dies with "timed out after 600000 ms" and no JSON. Export
// `LOCALFOLD_GPU_TIMEOUT_MS=2700000` for a fold of this size. 🔴 And do not pipe
// it through `tail`: the pipe's exit status is tail's, so the timeout reads as a
// success with the output mysteriously missing - which cost one run here.
//
// 🔴 AND ITS DEFAULT STEP COUNT IS 20 WHERE THE MODEL'S IS 200. A WebGL2
// denoiser call is seconds, so 200 steps is an hour; 20 is a plumbing answer with
// a geometry floor and NOT a converged sample. Read the CA-CA spacing and the
// bond geometry, not the RMSD.
//
// 🔴 AND IT IS NOT A SPEED COMPARISON WITH `fold-bioemu2.js` EITHER: that one
// batches S samples through one compiled denoiser and halves its weights, and
// this one carries 1.6 GiB of f32 denoiser weights because WebGL2 has no f16.
import { featuriseBioemu2, bioemu2JobFromAf3Json, parseContacts }
  from "../../shared/bioemu2/featurise.js";
import { bundleReader } from "../../shared/bioemu2/weights.js";
import { bioemu2Pdb, alphaCarbonSpacings } from "../../shared/bioemu2/structure.js";
import { fetchComponents, smilesTemplates } from "../../shared/bioemu2/inputs.js";
import { foldBioemu2OnWebGl2 } from "../../webgl/bioemu2/model/fold.js";
import { webglExecution, webglProfile } from "./webgl-execution-options.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { superpose } from "./superpose.js";
import { bondGeometry } from "./bond-geometry.js";
import { assertChainGeometry, chainGeometryOf } from "./chain-geometry.js";

const option = (args, name, fallback) => (args ?? [])
  .find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

function crystalAlphaCarbons(text, chain) {
  const residues = new Map();
  for (const line of text.split("\n")) {
    if (!line.startsWith("ATOM") || line[21] !== chain
      || line.slice(12, 16).trim() !== "CA") continue;
    if (line[16] !== " " && line[16] !== "A") continue;
    const number = Number(line.slice(22, 26));
    if (!residues.has(number)) {
      residues.set(number, [Number(line.slice(30, 38)), Number(line.slice(38, 46)),
        Number(line.slice(46, 54))]);
    }
  }
  return [...residues.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
}

export async function main(device, args) {
  const jobUrl = option(args, "job", "/tools/fixtures/bioemu2/6mrr.json");
  const model = option(args, "model", "/model-bioemu2-int5").replace(/\/$/, "");
  const crystalUrl = option(args, "crystal", null);
  const chain = option(args, "chain", "A");
  const contactsUrl = option(args, "contacts", null);
  const job = bioemu2JobFromAf3Json(await (await fetch(jobUrl)).json());
  const ccd = await fetchComponents(job, {
    local: (code) => `/tools/fixtures/ccd/${code}.cif` });
  const contacts = contactsUrl
    ? parseContacts(await (await fetch(contactsUrl)).json()) : null;
  const mode = option(args, "mode", "structure");
  const t0 = performance.now();
  const { features, atomKeys, topology, tokens, atoms } = featuriseBioemu2(job, {
    ccd, mode, contacts, smilesTemplates: await smilesTemplates(job) });
  const featuriseMs = Math.round(performance.now() - t0);

  // 🔴 THE CRYSTAL IS COUNTED BEFORE THE FOLD, because the fold is minutes and
  // the mismatch is instant: `alphaCarbonSpacings`'s centres depend on the atom
  // keys and the topology, not on the coordinates, so the count is knowable now.
  const crystal = crystalUrl
    ? crystalAlphaCarbons(await (await fetch(crystalUrl)).text(), chain) : null;
  if (crystal) {
    const { centres } = alphaCarbonSpacings(atomKeys, topology,
      new Float32Array(atoms * 3));
    const mine = centres.filter((c) => c.chain === chain).length;
    if (mine !== crystal.length) {
      throw new Error(`${mine} model CAs in chain ${chain} against`
        + ` ${crystal.length} in the crystal - 6mrr.json is 71 residues and`
        + " 6mrr-68.json is the one this crystal resolves");
    }
  }

  const read = bundleReader(model);
  await read.prefetch();
  const gl = createWebGL2Context();
  // 🔴 `narrowAlignedX` IS ON FOR THIS MODEL AND OFF FOR AlphaFold 2 AND 3,
  // WHICH IS A MEASUREMENT AND NOT A PREFERENCE. Narrowing x on shapes whose
  // stride already divides the texture width is **1.18x (26.55 s to 22.45)** here, and on
  // AlphaFold 2 it is no faster at all while the worst atom against the WGSL
  // fold goes 0.0353 A to 0.0904 - past that comparison's own 0.0355 A dither
  // floor. `--no-narrow-aligned-x` is the arm.
  const execution = webglExecution(gl, args ?? [], { narrowAlignedX: true, narrowUnblockedX: true });
  const t1 = performance.now();
  const result = await foldBioemu2OnWebGl2(execution, read, features, {
    steps: Number(option(args, "steps", "20")),
    samples: Number(option(args, "samples", "1")),
    seed: Number(option(args, "seed", "1")),
    passes: Number(option(args, "passes", "4")),
    mixerBlocks: Number(option(args, "mixer-blocks", "48")),
    mode,
  });
  const foldMs = Math.round(performance.now() - t1);

  const conformers = await (await fetch("/tools/oracle/reference-conformers.json")).json();
  // 🔴 A NON-FINITE COORDINATE IS A FAILED FOLD, SAID HERE: chain-geometry's
  // verdict skips a fold with no measurable alpha carbons, so a sample of NaN
  // walks through the geometry gate with its CA-CA reported null. The WGSL
  // driver carries the same check for the same reason.
  result.samples.forEach((xs, index) => {
    const bad = xs.subarray(0, atomKeys.length * 3)
      .findIndex((value) => !Number.isFinite(value));
    if (bad >= 0) {
      throw new Error(`sample ${index}: a non-finite coordinate (atom`
        + ` ${Math.floor(bad / 3)}: ${xs[bad]}) - the fold failed`);
    }
  });
  const scored = result.samples.map((coords) => {
    const { spacing, centres } = alphaCarbonSpacings(atomKeys, topology, coords);
    const geometry = chainGeometryOf(spacing);
    const pdb = bioemu2Pdb(atomKeys, topology, coords);
    const bonds = bondGeometry(pdb, conformers);
    let fit;
    if (crystal) {
      const mine = centres.filter((c) => c.chain === chain)
        .map((c) => [coords[c.at * 3], coords[c.at * 3 + 1], coords[c.at * 3 + 2]]);
      if (mine.length !== crystal.length) {
        throw new Error(
          `${mine.length} model CAs against ${crystal.length} in the crystal`);
      }
      fit = superpose(mine, crystal);
    }
    return { geometry, pdb, bonds, fit };
  });
  for (const sample of scored) {
    assertChainGeometry(sample.geometry,
      { doc: "docs/BIOEMU2.md", allow: args.includes("--allow-broken-geometry") });
  }

  const out = {
    job: jobUrl,
    model,
    shape: { tokens, atoms, passes: Number(option(args, "passes", "4")) },
    mode: result.mode,
    stepScale: result.stepScale,
    steps: Number(option(args, "steps", "20")),
    samples: result.samples.length,
    denoiserCalls: result.calls,
    ms: { featurise: featuriseMs, fold: foldMs },
    stages: result.stages,
    peakMiB: result.peakMiB,
    perSample: scored.map((s) => ({
      rmsd: s.fit ? Number(s.fit.rmsd.toFixed(3)) : null,
      tm: s.fit ? Number(s.fit.tm.toFixed(4)) : null,
      caca: Number(s.geometry.caca.toFixed(3)),
      worstCaca: Number(s.geometry.worstCaca.toFixed(3)),
      bondRms: Object.fromEntries(
        ["mainchain", "sidechain", "peptide", "nucleic", "ligand"]
          .filter((k) => s.bonds[k].bonds > 0)
          .map((k) => [k, Number(s.bonds[k].rms.toFixed(3))])),
    })),
    geometry: scored[0].geometry,
    contacts: (() => {
      let count = 0;
      for (let i = 0; i < tokens; i += 1) {
        for (let j = i + 2; j < tokens; j += 1) {
          if (result.contact[i * tokens + j] > 0.5) count += 1;
        }
      }
      return count;
    })(),
    memory: execution.stats(),
    profile: webglProfile(execution, args ?? []),
  };
  if (args.includes("--pdb")) out.pdb = scored[0].pdb;
  out.verdict = `ok: ${tokens} tokens folded in WebGL2 at ${out.steps} steps`;
  return out;
}
