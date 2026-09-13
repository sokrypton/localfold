/**
 * The confidence head against af3-any-model's own, on ITS inputs.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-confidence-oracle.js \
 *       --model=/model-boltz2-f32/manifest.json --name=boltz2
 *
 * 🔴 `check-af3-confidence.js` COMPARES THE GPU AGAINST THIS PORT'S OWN CPU, so
 * both can be wrong together - which is the whole failure this repository keeps
 * finding, and which the diffusion and trunk sides have both now been caught by.
 * boltz2's head is a DIFFERENT MODULE (it rebuilds z from nine terms, normalises
 * before no logit head, and splits both pair heads by chain) and was written
 * from a reading of the reference rather than from its numbers.
 *
 * The trunk activations are seeded and the atom LAYOUT is a real featurised
 * batch - both heads consume one, and it cannot be synthesised. See
 * tools/oracle/dump_af3_confidence.py.
 */
import { confidenceHead } from "../../src/af3/confidence-reference.js";
import { Af3ConfidenceHeadGpu } from "../../src/af3/confidence-webgpu.js";
import { pairformerBlock } from "../../src/af3/pairformer-reference.js";
import { af3Dialect, confidenceWeights, openAf3Store } from "../../src/af3/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function relativeRms(actual, expected) {
  let error = 0, scale = 0;
  for (let i = 0; i < expected.length; i += 1) {
    const d = actual[i] - expected[i];
    error += d * d; scale += expected[i] * expected[i];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
}

export async function main(device, args) {
  const name = option(args, "name", "boltz2");
  const dumpPath = option(args, "dump", `/oracle-dumps/af3-oracle-confidence-${name}.json`);
  const response = await fetch(dumpPath);
  if (!response.ok) throw new Error(`failed to load ${dumpPath}: ${response.status}`);
  const dump = await response.json();
  const raw = (key) => Float32Array.from(dump.stages[key].data);
  const ints = (key) => Int32Array.from(dump.stages[key].data, (v) => Math.round(v));

  const store = await openAf3Store(option(args, "model", undefined));
  const dialect = af3Dialect(store);
  const weights = await confidenceWeights(store);
  const tokens = dump.tokens;
  const dense = dump.slots;
  if (weights.targetFeatWidth !== dump.targetFeatWidth) {
    throw new Error(`this bundle's head takes target_feat at ${weights.targetFeatWidth} `
      + `channels and the dump was taken at ${dump.targetFeatWidth}`);
  }

  const input = {
    tokens, dense,
    seqMask: raw("in.seqMask"),
    pair: raw("in.pair"), single: raw("in.single"),
    targetFeat: raw("in.targetFeat"), pseudoBeta: raw("in.pseudoBeta"),
    bondMatrix: raw("in.bondMatrix"), bondOrderMatrix: raw("in.bondTypeMatrix"),
    features: {
      residueIndex: ints("in.residue_index"), tokenIndex: ints("in.token_index"),
      asymId: ints("in.asym_id"), entityId: ints("in.entity_id"),
      symId: ints("in.sym_id"),
    },
  };

  const results = {};
  const report = (label, ours, key) => {
    const want = raw(key);
    if (ours.length !== want.length) {
      console.log(`  ${label}\tLENGTH ${ours.length} vs ${want.length}`);
      results[label] = null;
      return;
    }
    const score = relativeRms(ours, want);
    const rms = (a) => Math.sqrt(a.reduce((t, v) => t + v * v, 0) / a.length);
    console.log(`  ${label}\t${score.toExponential(2)}`
      + `\tours rms ${rms(ours).toFixed(4)}\tnative rms ${rms(want).toFixed(4)}`);
    results[label] = score;
  };

  // 🔴 AND THE RE-EMBEDDING TERM BY TERM, where the dump carries them. Its
  // pair is a sum of NINE, and "z is 3.32 out" names none of them.
  const SCOPE_OF = {
    "reembed.sInputsNorm": "confidence_head/~_boltz2_reembed/s_inputs_norm",
    "reembed.sNorm": "confidence_head/~_boltz2_reembed/s_norm",
    "reembed.sInputToS": "confidence_head/~_boltz2_reembed/s_input_to_s",
    "reembed.zNorm": "confidence_head/~_boltz2_reembed/z_norm",
    "reembed.relPos": "confidence_head/~_boltz2_reembed/rel_pos_project",
    "reembed.left": "confidence_head/~_boltz2_reembed/left_target_feat_project",
    "reembed.right": "confidence_head/~_boltz2_reembed/right_target_feat_project",
    "reembed.prod1": "confidence_head/~_boltz2_reembed/s_to_z_prod_in1",
    "reembed.prod2": "confidence_head/~_boltz2_reembed/s_to_z_prod_in2",
    "reembed.dgram": "confidence_head/~_boltz2_reembed/distogram_feat_project",
    "reembed.prodOut": "confidence_head/~_boltz2_reembed/s_to_z_prod_out",
    "reembed.bondType": "confidence_head/~_boltz2_reembed/token_bonds_type_embed",
  };
  input.onStage = (label, value) => {
    const entry = dump.stages[`scope.${SCOPE_OF[label] ?? label}`];
    if (entry === undefined || value == null) return;
    const want = Float32Array.from(entry.data);
    if (want.length !== value.length) {
      console.log(`  ${label}\tLENGTH ${value.length} vs ${want.length}`);
      return;
    }
    const rms = (a) => Math.sqrt(a.reduce((t, v) => t + v * v, 0) / a.length);
    console.log(`  ${label}\t${relativeRms(value, want).toExponential(2)}`
      + `\tours rms ${rms(value).toFixed(4)}\tnative rms ${entry.rms.toFixed(4)}`);
  };

  if (option(args, "cpu", "on") !== "off") {
    const cpu = confidenceHead(input, weights, pairformerBlock, dialect);
    console.log("CPU");
    report("plddt", cpu.plddt, "out.predicted_lddt");
    report("pde", cpu.pde, "out.full_pde");
    report("pae", cpu.pae, "out.full_pae");
  }

  const gpu = await new Af3ConfidenceHeadGpu(device).run(input, weights, dialect);
  console.log("GPU");
  report("gpu.plddt", gpu.plddt, "out.predicted_lddt");
  report("gpu.pde", gpu.pde, "out.full_pde");
  report("gpu.pae", gpu.pae, "out.full_pae");

  // 🔴 THE BOUND IS ON pLDDT AND PAE, NOT ON PDE. All three are expectations
  // over softmaxed bins, so a logit difference at a bin boundary moves one
  // entry by O(1) while every other entry is exact - but pLDDT and PAE are what
  // a user reads, and a head that has a term wrong misses them together.
  const bound = Number(option(args, "bound", "2e-3"));
  for (const key of ["gpu.plddt", "gpu.pae"]) {
    if (!(results[key] < bound)) {
      throw new Error(`${key} differs from af3-any-model by `
        + `${results[key]?.toExponential(3)}, over ${bound.toExponential(0)}`);
    }
  }
  return { model: name, ...results };
}
