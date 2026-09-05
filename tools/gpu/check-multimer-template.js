/**
 * AF2-multimer's template embedder against its numpy reference.
 *
 *     python3 tools/oracle/template_reference.py --out oracle-dumps/toy-template.json
 *     python3 tools/oracle/template_reference.py --template tools/fixtures/1qys-crystal.pdb:A \
 *       --out oracle-dumps/toy-template-real.json
 *     node tools/gpu-chrome.mjs tools/gpu/check-multimer-template.js
 *
 * 🔴 NOTHING CHECKED THIS MODULE AT ALL. `tools/oracle/template_reference.py`
 * computed the reference and wrote `oracle-dumps/toy-template.json`, and no JavaScript ever
 * read it - so AF2-multimer's template term, which runs on EVERY recycle of
 * every multimer fold and is not small (masking the templates off does not zero
 * it), was covered by nothing. That is how the aatype terms came to be folded
 * into a constant: correct while masked was the only case, and unexamined.
 *
 * 🔴 TWO ARMS, AND THE MASKED ONE CANNOT SEE THE GEOMETRY. With every template
 * masked, six of the nine input Linears reduce to their biases - so an
 * implementation with no geometry at all agrees to machine precision. The real
 * arm is the one that means anything about the six features, and it is new.
 */
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";
import { AlphaFoldFixture } from "../../src/reference/alphafold-fixture.js";
import { WebGpuExecution } from "../../src/runtime/execution.js";
import {
  TEMPLATE_CHANNELS, encodeTemplateEmbedding, templateAatypeTerms, templateConstantTerm,
} from "../../src/multimer/template.js";
import { AF2_ATOM37 } from "../../src/af3/template-features.js";

const PAIR_CHANNELS = 128;

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
  // 🔴 THE f32 BUNDLE, NOT THE SHIPPED ONE, AND THAT IS THE WHOLE DIFFERENCE
  // BETWEEN A BUG AND A TOLERANCE. `model-multimer` is int8 at block 64 -
  // `dtype: "int8"` in its manifest - and the numpy reference reads the
  // float32 parameters, so comparing the two reports relRMS 6e-3 on the input
  // term alone and 1.1e-2 at the output. Neither number is a fault in either
  // implementation; both are what int8 costs. Measured both ways below.
  const store = await HttpTensorStore.open("/model-multimer-f32/manifest.json");
  const weights = await AlphaFoldFixture.fromStore(store).templateEmbeddingWeights();
  if (weights === undefined) throw new Error("this bundle carries no template embedder");

  // The same pair the numpy reference reads, so a disagreement is in this
  // module and cannot be in the embedder before it.
  const stage = await load("oracle-dumps/toy-oracle-stages.json");
  const results = [];

  // 🔴 THE JAX ARM IS THE ONLY ORACLE HERE. The other two files are numpy
  // transcriptions - useful, and not ground truth. This one is AF2-multimer's
  // own module, captured through hk.intercept_methods by
  // tools/oracle/dump_multimer_template.py, with the pair it actually read.
  for (const [name, file] of [["jax masked", "oracle-dumps/toy-template-jax.json"],
                              ["jax real", "oracle-dumps/toy-template-jax-real.json"],
                              ["numpy masked", "oracle-dumps/toy-template.json"],
                              ["numpy real", "oracle-dumps/toy-template-real.json"]]) {
    let oracle;
    try {
      oracle = await load(file);
    } catch (error) {
      console.log(`${name}\tskipped: ${error.message}`);
      continue;
    }
    const length = oracle.length;
    // The JAX dump carries the pair the module actually read; the numpy ones
    // read the toy stage file, which is the same pair by construction.
    const pair = Float32Array.from(oracle.pair ?? stage.embedder_pair).subarray(
      0, length * length * PAIR_CHANNELS);
    // 🔴 THE MASKS THE MODULE WAS ACTUALLY GIVEN. Arg 2 is padding_mask_2d and
    // arg 3 is multichain_mask_2d, and both are all ones in these dumps -
    // ColabDesign2's featurisation gives one asym_id. Substituting our own
    // two-chain mask scored 7.3e-2 against a module that is right, which is a
    // check reporting a fault in its own setup.
    const pairMask = oracle.pairMask === undefined
      ? new Float32Array(length * length).fill(1)
      : Float32Array.from(oracle.pairMask);
    const multichainMask2d = oracle.multichainMask2d === undefined
      ? undefined : Float32Array.from(oracle.multichainMask2d);

    const template = oracle.template === undefined ? undefined : {
      aatype: Int32Array.from(oracle.template.aatype),
      atomPositions: Float32Array.from(oracle.template.positions),
      atomMask: Float32Array.from(oracle.template.atomMask),
      spanChains: oracle.spanChains === true,
    };
    // 🔴 THE CHAINS THE ORACLE USED, NOT CHAINS OF OUR OWN. Splitting these
    // into two for the spanning arm's benefit made the GPU mask cross-chain
    // pairs that the reference had left open, and the numpy real arm reported
    // 3.8e-1 against an implementation that was right. The spanning block
    // below builds its own ids, which is where a second chain belongs.
    const asymId = Int32Array.from(oracle.chains ?? new Array(length).fill(0));

    const execution = new WebGpuExecution(device);
    // COPY_SRC, because the term is read back off this buffer. The model
    // never reads the pair out mid-fold, so its usual allocation has no need
    // of it.
    const pairBuffer = execution.upload("pair", Float32Array.from(pair),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const maskBuffer = execution.upload("pair-mask", pairMask);
    const encoder = device.createCommandEncoder({ label: "multimer-template" });
    const captured = await encodeTemplateEmbedding(execution, encoder, {
      length, pairChannels: PAIR_CHANNELS, templates: 1, template, asymId,
      multichainMask2d,
    }, weights, pairBuffer, maskBuffer, { captureStages: true });
    // 🔴 THE INPUT TERM TOO, because the final number localises nothing. The
    // reference emits `act` after construct_input and after each block; this
    // reads the same buffer at the same points, so a disagreement names which
    // stage it started in.
    const readback = execution.createReadback("pair-out", pairBuffer, encoder);
    device.queue.submit([encoder.finish()]);

    // 🔴 THE MODULE ADDS TO THE PAIR IN PLACE, so the term is the difference.
    // Reading the buffer and calling it the answer would compare the pair plus
    // the term against the term, which is a large disagreement that says
    // nothing about the term.
    const after = await execution.mapFloat32(readback);
    const term = new Float32Array(after.length);
    for (let index = 0; index < term.length; index += 1) term[index] = after[index] - pair[index];

    // 🔴 THE PLAIN-JS ARM COMPUTES NO GEOMETRY, SO IT IS A MASKED-ONLY
    // DIAGNOSTIC. It exists to split "the shader is wrong" from "the weights
    // are wrong", which only needs the case where the geometry is zero -
    // and run against a real template it scored 5.4e-1 against an
    // implementation that is right, because it was missing six of the nine
    // features on purpose.
    // 🔴 AND THE INPUT TERM RECOMPUTED IN PLAIN JS, from the same weights.
    // The shader, the packing and the weights are three places a disagreement
    // can live and the GPU number cannot tell them apart: if this agrees with
    // the reference the shader is at fault, and if it does not the weights or
    // the reading of them are.
    if (oracle.stages !== undefined && template === undefined) {
      const cpu = new Float32Array(length * length * TEMPLATE_CHANNELS);
      const query = weights.pairEmbeddings[8];
      const constant = templateConstantTerm(weights);
      const terms = templateAatypeTerms(weights, template?.aatype
        ?? new Int32Array(length), length);
      for (let i = 0; i < length; i += 1) {
        for (let j = 0; j < length; j += 1) {
          const entry = i * length + j;
          const base = entry * PAIR_CHANNELS;
          let total = 0;
          for (let c = 0; c < PAIR_CHANNELS; c += 1) total += pair[base + c];
          const mean = total / PAIR_CHANNELS;
          let variance = 0;
          for (let c = 0; c < PAIR_CHANNELS; c += 1) {
            variance += (pair[base + c] - mean) ** 2;
          }
          const inverse = 1 / Math.sqrt(variance / PAIR_CHANNELS + 1e-5);
          for (let channel = 0; channel < TEMPLATE_CHANNELS; channel += 1) {
            let value = constant[channel] + query.bias[channel]
              + terms.row[j * TEMPLATE_CHANNELS + channel]
              + terms.column[i * TEMPLATE_CHANNELS + channel];
            for (let c = 0; c < PAIR_CHANNELS; c += 1) {
              const normalized = (pair[base + c] - mean) * inverse
                * weights.queryNormScale[c] + weights.queryNormOffset[c];
              value += normalized * query.weight[c * TEMPLATE_CHANNELS + channel];
            }
            cpu[entry * TEMPLATE_CHANNELS + channel] = value;
          }
        }
      }
      console.log(`  input term in plain JS\trelRMS `
        + `${relativeRms(cpu, Float32Array.from(oracle.stages[0])).toExponential(2)}`);
    }

    // Stage by stage first: a disagreement at the output says only that they
    // disagree, and this module is an input term plus two blocks plus a
    // projection.
    let inputRelRms = Number.NaN;
    if (oracle.stages !== undefined && captured?.stages !== undefined) {
      for (const [index, stageRead] of captured.stages.entries()) {
        const mine = await execution.mapFloat32(stageRead);
        const theirs = Float32Array.from(oracle.stages[index]);
        const stageRms = relativeRms(mine, theirs);
        if (index === 0) inputRelRms = stageRms;
        console.log(`  stage ${index}${index === 0 ? " (input term)" : ` (after block ${index})`}`
          + `\trelRMS ${stageRms.toExponential(2)}`);
      }
    }

    const expected = Float32Array.from(oracle.template_term);
    const relRms = relativeRms(term, expected);
    let scale = 0;
    for (const value of expected) scale += value * value;
    console.log(`${name}\tlength=${length}`
      + `${template ? ` covered=${oracle.template.aatype.filter((c) => c !== 21).length}` : ""}`
      + `\trelRMS ${relRms.toExponential(2)}`
      + `\treference RMS ${Math.sqrt(scale / expected.length).toFixed(3)}`);
    results.push({ name, relRms, inputRelRms });
    execution.release?.();

    // 🔴 THE numpy ARMS ASSERT ONLY THEIR INPUT TERM, BECAUSE THEIR PAIR
    // BLOCKS ARE WRONG AND THE JAX ARM SAYS SO. Both sides were unvalidated
    // when they were first compared - they agreed to 2e-7 on construct_input
    // and disagreed by 1.2e-1 after the first block - and asserting either way
    // would have been picking a winner. The capture settled it: the GPU
    // reproduces AF2 to 6.5e-5 and the numpy transcription misses by 1.0e-2,
    // so the transcription is at fault. It is kept because its INPUT term is a
    // second independent reading of construct_input, including the six
    // geometry features, and that is worth having.
    if (name.startsWith("jax")) {
      // 🔴 THE ORACLE ARMS ASSERT THE WHOLE MODULE, because they are the only
      // ones entitled to. Measured through two pair blocks of f32: 6.5e-5
      // masked and 3.0e-4 with a real template. The real arm is the larger of
      // the two by more than its magnitude accounts for (std 2.04 against
      // 1.75), so there may be a little left in the geometry path - but it is
      // a thirtieth of what the numpy transcription misses by, and the
      // featurisation feeding it is bit-identical to what AF2 read.
      if (!(relRms < 1e-3)) throw new Error(`${name}: relRMS ${relRms}`);
    } else if (!(inputRelRms < 2e-5)) {
      throw new Error(`${name} input term: relRMS ${inputRelRms}`);
    }
  }

  // 🔴 INTER-CHAIN TEMPLATES, WHICH AF2 DOES NOT DO AND SO HAS NO ORACLE FOR.
  // Its `multichain_mask_2d` closes every cross-chain pair because a complex's
  // chains are templated by SEPARATE searches - two structures that were never
  // in one frame. When both chains come from ONE file they are in one frame,
  // and those distances are the interface geometry a binder method wants. So
  // it is checked by construction: masking per chain and spanning must differ,
  // or the flag is decoration.
  const spanning = await load("oracle-dumps/toy-template-jax-real.json").catch(() => undefined);
  if (spanning !== undefined) {
    const length = spanning.length;
    const asymId = Int32Array.from(
      Array.from({ length }, (_, token) => (token < length / 2 ? 1 : 2)));
    const base = {
      aatype: Int32Array.from(spanning.template.aatype),
      atomPositions: Float32Array.from(spanning.template.positions),
      atomMask: Float32Array.from(spanning.template.atomMask),
    };
    const terms = [];
    for (const spanChains of [false, true]) {
      const execution = new WebGpuExecution(device);
      const pair = Float32Array.from(spanning.pair);
      const pairBuffer = execution.upload("pair", Float32Array.from(pair),
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
      const maskBuffer = execution.upload("pair-mask", Float32Array.from(spanning.pairMask));
      const encoder = device.createCommandEncoder({ label: `span-${spanChains}` });
      await encodeTemplateEmbedding(execution, encoder, {
        length, pairChannels: PAIR_CHANNELS, templates: 1, asymId,
        template: { ...base, spanChains },
      }, weights, pairBuffer, maskBuffer);
      const readback = execution.createReadback("out", pairBuffer, encoder);
      device.queue.submit([encoder.finish()]);
      const after = await execution.mapFloat32(readback);
      const term = new Float32Array(after.length);
      for (let index = 0; index < term.length; index += 1) term[index] = after[index] - pair[index];
      terms.push(term);
      execution.release?.();
    }
    const moved = relativeRms(terms[1], terms[0]);
    console.log(`spanning	moves the term by relRMS ${moved.toExponential(2)}`);
    if (!(moved > 1e-3)) throw new Error(`spanChains changed nothing (relRMS ${moved})`);
  }

  // 🔴 THE ORACLE ARMS ARE THE POINT AND THEIR ABSENCE IS A FAILURE. Without
  // them this file compares one unvalidated implementation against another and
  // reports agreement, which is the shape of a check that cannot fail.
  for (const wanted of ["jax masked", "jax real"]) {
    if (!results.some((result) => result.name === wanted)) {
      throw new Error(`the ${wanted} arm did not run: write its dump with`
        + " tools/oracle/dump_multimer_template.py");
    }
  }
  return { layout: AF2_ATOM37.slots, results };
}
