/**
 * Pocket and contact constraints on the GPU: boltz2's contact term at both of its sites, against the table the host
 * builds from the same weights (shared/af3/featurise/constraints.js boltz2ContactTable).
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-constraints.js [--case=pocket_contact] [--model=/model-boltz2-int5/manifest.json]
 *
 * The term is ADDED into the pair, so a fold with the job's constraints minus the same fold without them is the
 * term's change and nothing else: the trunk's `tap.z_init_generic` seam, and the confidence re-embed's pair
 * (`returnRepresentations`). Each must equal `table[index[pair]] - unspecified`, which the host computes. That table
 * is held against af3-any-model's own contact conditioning by test/constraints.test.js, at float32 rounding.
 *
 * The cases are af3-any-model's constraint harness cases (dev/oracles/vendor_native_parity.py there), as job files:
 * tools/fixtures/constraints/*.json.
 */
import { jobFromJson } from "../../web/job-json.js";
import { expandEntities } from "../../web/entities.js";
import { mergeJobAlignments } from "../../shared/input/chains.js";
import { smilesComponent } from "../../shared/chem/component.js";
import { parseCcdComponent, ligandChain } from "../../shared/af3/featurise/ccd-component.js";
import { af3BatchFromA3m } from "../../shared/af3/featurise/batch.js";
import { boltz2ContactTable } from "../../shared/af3/featurise/constraints.js";
import { dialectFor, featuriserDialect } from "../../shared/af3/dialect.js";
import { foldBatch } from "../../webgpu/af3/fold.js";
import { Af3ConfidenceHeadGpu } from "../../webgpu/af3/confidence/confidence.js";
import { confidenceWeights, openAf3Store, trunkDepths, trunkWeights } from "../../shared/af3/weights/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights } from "../../shared/af3/weights/diffusion-weights.js";

const option = (args, name, fallback) =>
  args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const text = async (path) => {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.text();
};
const ccd = async (code) => parseCcdComponent(await text(`/tools/fixtures/ccd/${code}.cif`));

async function batchFor(name, dialect) {
  const job = jobFromJson(await text(`/tools/fixtures/constraints/${name}.json`));
  const request = expandEntities(job.entities);
  const merged = job.alignments === undefined ? null
    : mergeJobAlignments(job.alignments, request.chains, request.chainKinds);
  const ligands = [];
  for (const entry of request.ligandCodes) {
    ligands.push(typeof entry === "string" ? await ccd(entry)
      : entry.codes ? ligandChain(await Promise.all(entry.codes.map(ccd)))
        : await smilesComponent(entry.smiles, { code: entry.code ?? "LIG" }));
  }
  const modifications = [];
  for (const m of request.modifications) modifications.push({ chain: m.chain, position: m.position, ...await ccd(m.code) });
  return af3BatchFromA3m(request.sequence, merged?.alignment ?? null, {
    chainKinds: request.chainKinds, ligands, modifications,
    ...(merged === null ? {} : { msaColumnKinds: merged.msaColumnKinds }),
    ...(request.bonds === undefined ? {} : { bonds: request.bonds }),
    ...featuriserDialect(dialect), constraints: request.constraints, modelName: "boltz2",
  }).batch;
}

/** relRMS of (constrained - unconstrained) against the host table's term minus the unspecified constant. */
function grade(withC, without, table, unspecified, C) {
  let err = 0;
  let scale = 0;
  for (let p = 0; p < table.index.length; p += 1) {
    const row = table.index[p] * C;
    for (let c = 0; c < C; c += 1) {
      const expected = table.table[row + c] - unspecified[c];
      const got = withC[p * C + c] - without[p * C + c];
      err += (got - expected) ** 2;
      scale += expected * expected;
    }
  }
  return Math.sqrt(err / Math.max(scale, 1e-30));
}

export async function main(device, args) {
  const cases = option(args, "case", "pocket,pocket_contact,contact_ligand,contact").split(",");
  const store = await openAf3Store(option(args, "model", "/model-boltz2-int5/manifest.json"));
  store.prefetch();
  const depths = trunkDepths(store);
  const weights = {
    trunk: await trunkWeights(store, depths.pairformerBlocks, depths.msaBlocks, { allowPrefix: true }),
    diffusion: await diffusionWeights(store),
    confidence: await confidenceWeights(store),
    atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };
  const dialect = weights.trunk.dialect ?? dialectFor("boltz2");
  const e = weights.trunk.embedder;
  const r = weights.confidence.reembed;
  const results = [];
  let worst = 0;
  for (const name of cases) {
    const batch = await batchFor(name, dialect);
    if (batch.contactConditioning === undefined) throw new Error(`${name}: no constraint features were built`);
    const plain = { ...batch };
    delete plain.contactConditioning;
    delete plain.contactThreshold;
    const zInit = async (b) => {
      let seam;
      await foldBatch(device, b, weights, { recycles: 0, steps: 2, stopAfter: 2, seed: 1,
                                            onSeam: (n, value) => { if (n === "tap.z_init_generic") seam = value; } });
      if (seam === undefined) throw new Error("no tap.z_init_generic seam");
      return Float32Array.from(seam);
    };
    const trunkTable = boltz2ContactTable(batch.contactConditioning, batch.contactThreshold, {
      fourierWeights: e.contactFourierWeights, fourierBias: e.contactFourierBias,
      encoderWeights: e.contactEncoderWeights, encoderBias: e.contactEncoderBias,
      unspecified: e.contactEncodingUnspecified, unselected: e.contactEncodingUnselected });
    const trunk = grade(await zInit(batch), await zInit(plain), trunkTable,
                        e.contactEncodingUnspecified, e.contactEncodingUnspecified.length);

    // the head's re-embedded pair, on any input: the contact term is added, so the difference is the term alone
    const tokens = batch.asymId.length;
    const C = weights.confidence.pairChannels;
    const dense = batch.refMask.length / tokens;
    const head = async (b) => (await new Af3ConfidenceHeadGpu(device).run({
      tokens, dense, seqMask: b.seqMask,
      pair: new Float32Array(tokens * tokens * C).map((_, k) => Math.sin(k)),
      single: new Float32Array(tokens * weights.confidence.singleChannels).map((_, k) => Math.cos(k)),
      targetFeat: new Float32Array(tokens * weights.confidence.targetFeatWidth),
      pseudoBeta: new Float32Array(tokens * 3).map((_, k) => k % 7),
      features: b.features, bondMatrix: b.bondMatrix, bondOrderMatrix: b.bondOrderMatrix,
      ...(b.contactConditioning === undefined ? {}
        : { contactConditioning: b.contactConditioning, contactThreshold: b.contactThreshold }),
    }, weights.confidence, dialect, { returnRepresentations: true })).embeddedPair;
    const confTable = boltz2ContactTable(batch.contactConditioning, batch.contactThreshold, {
      fourierWeights: r.contactFourierWeights, fourierBias: r.contactFourierBias,
      encoderWeights: r.contactEncoderWeights, encoderBias: r.contactEncoderBias,
      unspecified: r.contactEncodingUnspecified, unselected: r.contactEncodingUnselected });
    const conf = grade(await head(batch), await head(plain), confTable, r.contactEncodingUnspecified, C);
    worst = Math.max(worst, trunk, conf);
    results.push({ case: name, tokens, tableRows: trunkTable.entries, trunk, confidence: conf });
    console.log(`${name}: ${tokens} tokens, ${trunkTable.entries} table rows; relRMS trunk ${trunk.toExponential(2)}`
      + ` confidence ${conf.toExponential(2)}`);
  }
  const ok = worst < 1e-4;
  console.log(ok ? "OK" : `FAIL: worst ${worst.toExponential(2)}`);
  return { results, worst, ok };
}
