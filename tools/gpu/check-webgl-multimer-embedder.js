/**
 * AF2-MULTIMER's pair embedding in WebGL2: the 73-row relative encoding.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-multimer-embedder.js [--falsify]
 *
 * The monomer looks up ONE row - the clamped residue-index offset. The multimer
 * looks up three and sums them:
 *
 *     offset   row  clamp(i - j + 32, 0, 64)  when the two share a CHAIN,
 *                   else row 65, "a different chain entirely"
 *     entity   row  66, added only when the two share an ENTITY
 *     chain    row  67 + clamp(symmetry_i - symmetry_j + 2, 0, 4) when they
 *                   share an entity, else 67 + 5, "a different entity"
 *
 * 🔴 AND THE MULTIMER FORM IS A SUPERSET, WHICH THIS GATE ASSERTS RATHER THAN
 * QUOTES. `webgpu/af2/multimer/input-embedder.js` says "THE MULTIMER GRAPH IS
 * THE ONE GRAPH, and monomer is its single-chain case... zero rows there
 * reproduce the 65-row result EXACTLY, and the shader needs no monomer branch."
 * The `superset` arm runs a MONOMER's features through the chains kernel with a
 * widened weight table and holds the result **bit-identical** to the monomer
 * kernel - bar exactly 0, because a different lookup that merely rounds the same
 * is not the same lookup. That claim is what would justify deleting the monomer
 * form, so it is measured before anyone acts on it.
 *
 * 🔴 AND THE ARMS USE A REAL COMPLEX, because a monomer cannot reach any of the
 * eight new rows. Three chains, two of them copies of ONE entity with different
 * symmetry ids and one a second entity, so every branch runs: same chain, same
 * entity across chains, different entity, and symmetry deltas of 0, +1 and -1.
 * A gate whose input cannot reach a branch says nothing about it, and the eight
 * trailing rows are exactly the branches a monomer never lights.
 */
import { InputEmbedderGpu } from "../../webgpu/af2/multimer/input-embedder.js";
import { InputEmbedderGpu as MonomerInputEmbedderGpu } from "../../webgpu/af2/evoformer/input-embedder.js";
import { setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  MONOMER_RELATIVE_ROWS, MULTIMER_RELATIVE_ROWS, inputEmbedder, releaseInputEmbedderWeights,
  residueIdentity, uploadInputEmbedderWeights, widenRelativePositionWeight,
} from "../../webgl/af2/evoformer/input-embedder.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { generator } from "./af2-synthetic-block.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "1e-6"));
  const shape = {
    length: Number(option(args, "length", "12")),
    msaSequences: 2,
    extraSequences: 2,
    targetChannels: 22,
    msaFeatureChannels: 49,
    msaChannels: 32,
    pairChannels: 16,
    extraMsaChannels: 16,
  };
  const { length, msaChannels, pairChannels, targetChannels, msaFeatureChannels } = shape;

  const random = generator(1836);
  const noise = (count, spread, centre = 0) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  const integers = (count, limit) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = Math.floor(random() * limit);
    return out;
  };

  // Three chains: A and B are copies of entity 0 at symmetry 0 and 1, C is
  // entity 1 at symmetry 0. Residue indices restart per chain, as a real
  // complex's do, so the offset lookup sees small deltas within a chain and the
  // "different chain" row across them.
  const spans = [Math.ceil(length / 3), Math.ceil(length / 3)];
  spans.push(length - spans[0] - spans[1]);
  const asymId = new Float32Array(length);
  const entityId = new Float32Array(length);
  const symId = new Float32Array(length);
  const residueIndex = new Float32Array(length);
  {
    let residue = 0;
    const chains = [{ entity: 0, sym: 0 }, { entity: 0, sym: 1 }, { entity: 1, sym: 0 }];
    for (const [chain, span] of spans.entries()) {
      for (let offset = 0; offset < span; offset += 1, residue += 1) {
        asymId[residue] = chain;
        entityId[residue] = chains[chain].entity;
        symId[residue] = chains[chain].sym;
        residueIndex[residue] = offset;
      }
    }
  }
  const aatype = integers(length, 21);
  aatype[Math.floor(length / 2)] = 7;      // a glycine, for the CA pseudo-beta

  const monomerRelative = noise(MONOMER_RELATIVE_ROWS * pairChannels, 0.3);
  const weights = {
    preprocess1dWeight: noise(targetChannels * msaChannels, 1 / Math.sqrt(targetChannels)),
    preprocess1dBias: noise(msaChannels, 0.1),
    preprocessMsaWeight: noise(msaFeatureChannels * msaChannels,
      1 / Math.sqrt(msaFeatureChannels)),
    preprocessMsaBias: noise(msaChannels, 0.1),
    leftSingleWeight: noise(targetChannels * pairChannels, 1 / Math.sqrt(targetChannels)),
    leftSingleBias: noise(pairChannels, 0.1),
    rightSingleWeight: noise(targetChannels * pairChannels, 1 / Math.sqrt(targetChannels)),
    rightSingleBias: noise(pairChannels, 0.1),
    previousPositionWeight: noise(15 * pairChannels, 0.3),
    previousPositionBias: noise(pairChannels, 0.1),
    previousMsaNormScale: noise(msaChannels, 0.2, 1),
    previousMsaNormOffset: noise(msaChannels, 0.1),
    previousPairNormScale: noise(pairChannels, 0.2, 1),
    previousPairNormOffset: noise(pairChannels, 0.1),
    // 🔴 THE FULL 73 ROWS, WITH THE EIGHT NEW ONES NON-ZERO. Widening a monomer
    // table pads them with ZERO, which is exactly the case the `superset` arm
    // needs and exactly the case the multimer arms must NOT use: with zeros
    // there the three-row sum collapses to the one-row lookup and the gate would
    // pass while reading none of the new rows.
    relativePositionWeight: noise(MULTIMER_RELATIVE_ROWS * pairChannels, 0.3),
    relativePositionBias: noise(pairChannels, 0.1),
    extraMsaWeight: noise(25 * shape.extraMsaChannels, 0.3),
    extraMsaBias: noise(shape.extraMsaChannels, 0.1),
  };

  const input = {
    ...shape,
    targetFeatures: noise(length * targetChannels, 1),
    msaFeatures: noise(shape.msaSequences * length * msaFeatureChannels, 1),
    extraMsa: integers(shape.extraSequences * length, 23),
    extraHasDeletion: integers(shape.extraSequences * length, 2),
    extraDeletionValue: noise(shape.extraSequences * length, 0.5, 0.5),
    previousMsaFirstRow: noise(length * msaChannels, 0.5),
    previousPair: noise(length * length * pairChannels, 0.5),
    previousPositions: noise(length * 37 * 3, 12),
    aatype, residueIndex, asymId, entityId, symId,
    weights,
  };

  setHalfPrecision(device, false);
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const upload = (values, label) => execution.upload(values, label);
  const uploaded = uploadInputEmbedderWeights(execution, weights);
  const identity = upload(residueIdentity({ length, residueIndex, asymId, entityId, symId }),
    "identity");
  const shared = {
    msaFeatures: upload(input.msaFeatures, "msa-features"),
    extraMsa: upload(input.extraMsa, "extra-codes"),
    extraHasDeletion: upload(input.extraHasDeletion, "has-deletion"),
    // 🔴 THE EXTRA-MSA EMBEDDING READS NONE OF THE TARGET FEATURES, so the
    // control has to reach it separately - it takes a code, a deletion flag and
    // a deletion value and nothing else. The monomer embedder gate found this
    // exact vacuous arm and fixed it the same way; repeating the mistake here
    // would have been the fifth of this campaign.
    extraDeletionValue: upload(falsify
      ? Float32Array.from(input.extraDeletionValue, (v) => v * 1.01 + 0.01)
      : input.extraDeletionValue, "deletion-value"),
    previousMsaFirstRow: upload(input.previousMsaFirstRow, "previous-msa"),
    previousPair: upload(input.previousPair, "previous-pair"),
    previousPositions: upload(input.previousPositions, "previous-positions"),
    aatype: upload(aatype, "aatype"),
    residueIndex: upload(residueIndex, "residue-index"),
  };
  // The control skews half of each target row's channels: whole rows would be
  // absorbed by nothing here, but half a row is what the embedder gate already
  // uses and it reaches the left and right projections differently.
  const targetFeatures = upload(falsify
    ? Float32Array.from(input.targetFeatures,
      (v, i) => (i % targetChannels < targetChannels / 2 ? v * 1.01 + 0.01 : v))
    : input.targetFeatures, "target");

  const ours = inputEmbedder(execution, {
    ...shape, ...shared, targetFeatures,
    residueIdentity: identity, weights: uploaded,
  });
  const theirs = await new InputEmbedderGpu(device).run(input);

  const rows = [];
  const failures = [];
  /**
   * `falsifiable: false` says an arm CANNOT move under the control and why.
   *
   * 🔴 THE SUPERSET ARM IS ONE, STRUCTURALLY. It compares two of this port's own
   * kernels on the SAME input, so any perturbation of that input moves both
   * sides together - which is this repository's "comparing something with
   * itself" trap in the clothes of a control, and the fix is to declare it
   * rather than to invent a perturbation that only looks like it works. What
   * makes the arm worth having is its bar of exactly zero.
   */
  const arm = (name, got, want, armBar = bar, falsifiable = true) => {
    const relative = relativeRms(got, want, name);
    const ok = relative <= armBar;
    rows.push({
      arm: name, relativeRms: Number(relative.toPrecision(4)), bar: armBar, ok,
      ...(falsifiable ? {} : { falsifiable: false, reason: "compares two of this port's own kernels on one input" }),
    });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${armBar}`);
  };
  arm("msa", execution.read(ours.msa), theirs.msa);
  arm("extra msa", execution.read(ours.extraMsa), theirs.extraMsa);
  arm("pair without templates",
    execution.read(ours.pairWithoutTemplates), theirs.pairWithoutTemplates);
  for (const tensor of Object.values(ours)) execution.release(tensor);

  // --- the superset claim, at bar zero ------------------------------------
  //
  // A MONOMER's features - one chain, one entity, one symmetry copy, residue
  // indices running straight through - through both kernels, with the chains
  // one given the WIDENED monomer table. The two lookups are different code and
  // must agree to the last bit.
  const monomerIndex = Float32Array.from({ length }, (_, residue) => residue);
  const monomerWeights = { ...weights, relativePositionWeight: monomerRelative };
  const widenedWeights = {
    ...weights,
    relativePositionWeight: widenRelativePositionWeight(monomerRelative, pairChannels),
  };
  const monomerUploaded = uploadInputEmbedderWeights(execution, monomerWeights);
  const widenedUploaded = uploadInputEmbedderWeights(execution, widenedWeights);
  const monomerIndexTensor = upload(monomerIndex, "monomer-index");
  const monomerIdentity = upload(
    residueIdentity({ length, residueIndex: monomerIndex }), "monomer-identity");
  const viaMonomer = inputEmbedder(execution, {
    ...shape, ...shared, targetFeatures,
    residueIndex: monomerIndexTensor, weights: monomerUploaded,
  });
  const viaChains = inputEmbedder(execution, {
    ...shape, ...shared, targetFeatures,
    residueIdentity: monomerIdentity, weights: widenedUploaded,
  });
  arm("superset: one chain through the chains kernel",
    execution.read(viaChains.pairWithoutTemplates),
    Float32Array.from(execution.read(viaMonomer.pairWithoutTemplates)), 0, false);
  // ...and the same input through the WGSL monomer embedder, so "bit-identical"
  // is anchored to the reference and not only to this port's other kernel.
  const theirsMonomer = await new MonomerInputEmbedderGpu(device).run({
    ...input, targetFeatures: input.targetFeatures,
    residueIndex: monomerIndex, weights: monomerWeights,
  });
  arm("superset vs the WGSL monomer embedder",
    execution.read(viaChains.pairWithoutTemplates), theirsMonomer.pairWithoutTemplates);
  for (const tensor of [...Object.values(viaMonomer), ...Object.values(viaChains)]) {
    execution.release(tensor);
  }

  releaseInputEmbedderWeights(execution, uploaded);
  releaseInputEmbedderWeights(execution, monomerUploaded);
  releaseInputEmbedderWeights(execution, widenedUploaded);
  for (const tensor of [...Object.values(shared), targetFeatures, identity,
    monomerIndexTensor, monomerIdentity]) {
    execution.release(tensor);
  }
  const falsifiableRows = rows.filter((row) => row.falsifiable !== false);
  const red = falsifiableRows.filter((row) => !row.ok).length;
  return {
    shape,
    chains: spans,
    relativeRows: { monomer: MONOMER_RELATIVE_ROWS, multimer: MULTIMER_RELATIVE_ROWS },
    falsify,
    arms: rows,
    memory: execution.stats(),
    verdict: falsify
      ? (red === falsifiableRows.length
        ? `ok: all ${falsifiableRows.length} falsifiable arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${falsifiableRows.length} falsifiable arms noticed it`)
      : (failures.length === 0
        ? `ok: ${rows.length} arms, and one chain through the chains kernel is bit-identical`
        : `FAILED: ${failures.join("; ")}`),
  };
}
