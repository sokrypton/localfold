/**
 * AF2-monomer's template embedder against AF2's own module.
 *
 *     python3 tools/oracle/dump_monomer_template.py \
 *       --out oracle-dumps/toy-template-monomer-jax.json
 *     python3 tools/oracle/dump_monomer_template.py --template tools/fixtures/1qys-crystal.pdb:A \
 *       --out oracle-dumps/toy-template-monomer-jax-real.json
 *     node tools/gpu-chrome.mjs tools/gpu/check-monomer-template.js
 *
 * 🔴 THE ORACLE IS AF2'S MODULE RUN ON ITS OWN, not a transcription and not a
 * whole-model capture. ColabDesign2 refuses monomer templates outright - it
 * puts the monomer on the multimer graph and the two embedders differ - so
 * tools/oracle/dump_monomer_template.py transforms `TemplateEmbedding` with
 * haiku and hands it the slice of the checkpoint whose names it owns.
 *
 * 🔴 AND THE f32 BUNDLE, `model.f32-backup`. The shipped `model/` is int8, and
 * comparing it against float32 parameters reports quantisation as a fault -
 * which cost an hour on the multimer side before its manifest was read.
 */
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";
import { AlphaFoldFixture } from "../../src/reference/alphafold-fixture.js";
import { QueryOnlyTemplateGpu } from "../../src/evoformer/template.js";
import { AF2_ATOM37_MONOMER } from "../../src/af3/template-features.js";

async function load(name) {
  const response = await fetch(`/${name}`);
  if (!response.ok) throw new Error(`${response.status} fetching ${name}`);
  return response.json();
}

function relativeRms(actual, expected) {
  let error = 0;
  let scale = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const difference = actual[index] - expected[index];
    error += difference * difference;
    scale += expected[index] * expected[index];
  }
  return Math.sqrt(error / (scale || 1));
}

export async function main(device) {
  const store = await HttpTensorStore.open("/model.f32-backup/manifest.json");
  const weights = await AlphaFoldFixture.fromStore(store).templateWeights();
  if (weights?.embeddingWeight === undefined) {
    throw new Error("this bundle carries no template embedding2d weight");
  }

  const results = [];
  for (const [name, file] of [["masked", "oracle-dumps/toy-template-monomer-jax.json"],
                              ["real", "oracle-dumps/toy-template-monomer-jax-real.json"]]) {
    let oracle;
    try {
      oracle = await load(file);
    } catch (error) {
      console.log(`${name}\tskipped: ${error.message}`);
      continue;
    }
    const length = oracle.length;
    const template = oracle.template === null || oracle.template === undefined
      ? undefined
      : {
        aatype: Int32Array.from(oracle.template.aatype),
        atomPositions: Float32Array.from(oracle.template.positions),
        atomMask: Float32Array.from(oracle.template.atomMask),
      };

    const result = await new QueryOnlyTemplateGpu(device).run({
      length,
      templateChannels: 64,
      pairChannels: 128,
      pairMask: Float32Array.from(oracle.pairMask),
      weights,
      template,
      // 🔴 FALSE IN EVERY SHIPPED MONOMER CONFIG, and read from the dump rather
      // than assumed: three of the six geometry features are deliberately
      // zeroed, and a checkpoint that turned them on would otherwise be
      // silently mis-embedded.
      useTemplateUnitVector: oracle.useTemplateUnitVector === true,
    });

    const expected = Float32Array.from(oracle.template_term);
    const relRms = relativeRms(result.pairUpdate, expected);
    let scale = 0;
    for (const value of expected) scale += value * value;
    console.log(`${name}\tlength=${length}`
      + `${template ? ` covered=${oracle.covered}` : ""}`
      + `\trelRMS ${relRms.toExponential(2)}`
      + `\treference RMS ${Math.sqrt(scale / expected.length).toFixed(3)}`);
    results.push({ name, relRms });
    if (!(relRms < 1e-3)) throw new Error(`${name}: relRMS ${relRms}`);
  }

  // 🔴 BOTH ARMS OR IT IS NOT A CHECK. The masked one is what every monomer
  // fold runs and reaches none of the geometry - with the whole concatenation
  // zeroed, `embedding2d` contributes its bias and nothing else, so an
  // implementation with no features at all passes it.
  for (const wanted of ["masked", "real"]) {
    if (!results.some((result) => result.name === wanted)) {
      throw new Error(`the ${wanted} arm did not run: write its dump with`
        + " tools/oracle/dump_monomer_template.py");
    }
  }
  return { layout: AF2_ATOM37_MONOMER.slots, results };
}
