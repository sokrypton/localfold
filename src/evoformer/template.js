import { ATTENTION_NORMALIZE_SHADER, createAttentionNormParameters } from "./attention.js";
import { encodeTemplatePairBlock } from "./block.js";
import { WebGpuExecution } from "../runtime/execution.js";
// 🔴 THE GEOMETRY IS SHARED WITH AF3 AND AF2-MULTIMER. All three compute the
// same six features; the monomer differs in its atom layout and in masking the
// whole concatenation by the BACKBONE mask rather than each feature by its own.
// See AF2_ATOM37_MONOMER.
import {
  AF2_ATOM37_MONOMER, GEOMETRY_STRIDE, coverageOf, multichainMaskFor,
  packTemplateGeometry, templateGeometry,
} from "../af3/template-features.js";
import { GAP_AATYPE } from "../af3/template-input.js";

/** Channels in AF2-monomer's `to_concat`: 39 + 1 + 22 + 22 + 3 + 1. */
const FEATURE_COLUMNS = 88;

/**
 * 🔴 A MONOMER IS ONE CHAIN, AND THAT IS STILL NOT A LICENCE TO ASSUME IT.
 * This model folds a single sequence, so an all-ones mask is right - but the
 * same function is reachable with chain ids, and AF3 and AF2-multimer both
 * shipped a permissive default that was wrong for a complex. Stated once here
 * rather than defaulted silently.
 */
function chainMaskFor(input) {
  if (input.multichainMask2d !== undefined) return input.multichainMask2d;
  if (input.asymId !== undefined) {
    return multichainMaskFor(input.asymId, input.length, {
      coverage: coverageOf(input.template, input.length, AF2_ATOM37_MONOMER),
      spanChains: input.template.spanChains === true,
    });
  }
  return new Float32Array(input.length * input.length).fill(1);
}

/**
 * `embedding2d` over AF2-monomer's 88-channel concatenation.
 *
 * 🔴 ONE Linear OVER A CONCATENATION, where AF2-multimer and AF3 sum nine
 * separate projections. Same arithmetic, one [88, 64] weight, and the columns
 * are laid out in `to_concat`'s order:
 *
 *     0..38   the distogram, one-hot over 39 bins
 *     39      the pseudo-beta mask
 *     40..61  the template aatype along j, one-hot over 22
 *     62..83  ...and along i
 *     84..86  the unit vector
 *     87      the backbone mask
 *
 * 🔴 THE WHOLE 88 IS MULTIPLIED BY THE BACKBONE MASK, not each feature by its
 * own - `act *= template_mask_2d[..., None]` where that mask is N, CA and C.
 * So the monomer's distogram survives at a pair whose pseudo-beta is missing
 * and whose backbone is not, which is the opposite of what the other two do.
 * The BIAS is outside the mask, which is why an all-masked template leaves
 * exactly the bias - the value this shader used to write unconditionally.
 *
 * 🔴 AND `use_template_unit_vector` IS FALSE IN THE MONOMER CONFIG, so columns
 * 84..86 are deliberately zero for every shipped monomer model. The uniform
 * carries it rather than the shader assuming it: the flag is per model, and a
 * checkpoint that sets it true would otherwise be silently mis-embedded.
 */
const INIT_SHADER = `
const GRID_WIDTH: u32 = 32768u;
const GEOMETRY_STRIDE: u32 = 6u;
const DGRAM_BINS: u32 = 39u;
const COLUMNS: u32 = 88u;
struct Parameters {
  length: u32, template_channels: u32, unit_vectors: u32, padding: u32,
};
@group(0) @binding(0) var<storage, read> bias: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> geometry: array<f32>;
@group(0) @binding(4) var<storage, read> aatype: array<i32>;
@group(0) @binding(5) var<uniform> p: Parameters;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= arrayLength(&output)) { return; }
  let channel = index % p.template_channels;
  let entry = index / p.template_channels;
  let i = entry / p.length;
  let j = entry % p.length;

  let g = entry * GEOMETRY_STRIDE;
  let backbone = geometry[g + 5u];
  var value = bias[channel];
  if (backbone != 0.0) {
    var inner = 0.0;
    let bin = u32(geometry[g]);
    if (bin > 0u) { inner += weight[(bin - 1u) * p.template_channels + channel]; }
    inner += geometry[g + 1u] * weight[39u * p.template_channels + channel];
    let row_type = u32(clamp(aatype[j], 0, 21));
    let col_type = u32(clamp(aatype[i], 0, 21));
    inner += weight[(40u + row_type) * p.template_channels + channel];
    inner += weight[(62u + col_type) * p.template_channels + channel];
    if (p.unit_vectors != 0u) {
      inner += geometry[g + 2u] * weight[84u * p.template_channels + channel];
      inner += geometry[g + 3u] * weight[85u * p.template_channels + channel];
      inner += geometry[g + 4u] * weight[86u * p.template_channels + channel];
    }
    inner += weight[87u * p.template_channels + channel];
    value += backbone * inner;
  }
  output[index] = value;
}`;

const VALUE_SHADER = `
const GRID_WIDTH: u32 = 32768u;
struct Parameters { pairs: u32, template_channels: u32, projected: u32, pair_channels: u32 };
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.pairs * p.projected) { return; }
  let row = index / p.projected;
  let channel = index % p.projected;
  var result = 0.0;
  for (var c = 0u; c < p.template_channels; c += 1u) {
    result += source[row * p.template_channels + c] * weights[c * p.projected + channel];
  }
  output[index] = result;
}`;

const OUTPUT_SHADER = `
const GRID_WIDTH: u32 = 32768u;
struct Parameters { pairs: u32, template_channels: u32, projected: u32, pair_channels: u32 };
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> output_weight: array<f32>;
@group(0) @binding(2) var<storage, read> output_bias: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.pairs * p.pair_channels) { return; }
  let row = index / p.pair_channels;
  let channel = index % p.pair_channels;
  var result = output_bias[channel];
  for (var c = 0u; c < p.projected; c += 1u) {
    result += source[row * p.projected + c] * output_weight[c * p.pair_channels + channel];
  }
  output[index] = result;
}`;

export class QueryOnlyTemplateGpu {
  device;
  constructor(device) { this.device = device; }

  async run(input) {
    const execution = new WebGpuExecution(this.device);
    try {
      const pairs = input.length * input.length;
      const pair = execution.allocate(
        "template.pair", pairs * input.templateChannels, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      );
      const pairMask = execution.upload("template.pair-mask", input.pairMask);
      const bias = execution.upload("template.embedding-bias", input.weights.embeddingBias);
      // 🔴 A MASKED TEMPLATE IS EXPRESSED AS DATA, NOT AS A SECOND PATH. With
      // no template the geometry is zeros and the backbone mask is zero, so
      // the shader leaves exactly `bias` - which is the value it used to write
      // unconditionally, and is what AF2 computes when `act` is masked to
      // nothing and `embedding2d` adds its bias afterwards.
      const geometry = input.template === undefined
        ? new Float32Array(pairs * GEOMETRY_STRIDE)
        : packTemplateGeometry(
          templateGeometry(input.template, chainMaskFor(input), input.length,
                           AF2_ATOM37_MONOMER),
          input.length);
      const aatype = new Int32Array(input.length);
      // An uncovered position is the GAP type, not alanine - AF2's own
      // pipeline maps an alignment gap to `restypes_with_x_and_gap`'s 21.
      aatype.fill(GAP_AATYPE);
      if (input.template !== undefined) {
        for (let token = 0; token < input.length; token += 1) {
          aatype[token] = input.template.aatype[token];
        }
      }
      const geometryBuffer = execution.upload("template.geometry", geometry);
      const aatypeBuffer = execution.upload("template.aatype", aatype);
      const embeddingWeight = execution.upload(
        "template.embedding2d", input.weights.embeddingWeight
          ?? new Float32Array(FEATURE_COLUMNS * input.templateChannels));
      const initParameters = execution.upload("template.init-parameters",
        Uint32Array.from([input.length, input.templateChannels,
          // ...false in every shipped monomer config, and carried rather than
          // assumed because a checkpoint that set it would be mis-embedded.
          input.useTemplateUnitVector === true ? 1 : 0, 0]),
        GPUBufferUsage.UNIFORM);
      const init = await execution.pipelines.get("template:init", INIT_SHADER);
      let encoder = this.device.createCommandEncoder({ label: "template.initialize" });
      const initGrid = execution.linearGrid(pair.elements);
      execution.dispatch(encoder, init,
        [bias, pair, embeddingWeight, geometryBuffer, aatypeBuffer, initParameters],
        initGrid[0], initGrid[1], 1, "template.initialize");
      execution.endComputePass(encoder);
      this.device.queue.submit([encoder.finish()]);
      const persistentCheckpoint = execution.checkpoint();
      const start = performance.now();

      for (let block = 0; block < input.weights.blockWeights.length; block += 1) {
        encoder = this.device.createCommandEncoder({ label: `template.block-${block}` });
        this.device.pushErrorScope("validation");
        await encodeTemplatePairBlock(execution, encoder, {
          sequences: 1,
          length: input.length,
          cM: input.templateChannels,
          cZ: input.templateChannels,
          cOuter: 0,
          triangleHidden: input.weights.blockWeights[block] .triangleMultiplicationOutgoing.linearAPBias.length,
        }, input.weights.blockWeights[block], pair, pairMask);
        execution.endComputePass(encoder);
        this.device.queue.submit([encoder.finish()]);
        const error = await this.device.popErrorScope();
        if (error !== null) throw new Error(`WebGPU template block ${block} failed: ${error.message}`);
        execution.releaseSince(persistentCheckpoint);
      }

      const normWeights = new Float32Array(input.templateChannels * 2);
      normWeights.set(input.weights.outputNormScale);
      normWeights.set(input.weights.outputNormOffset, input.templateChannels);
      const normWeightBuffer = execution.upload("template.output-norm-weights", normWeights);
      const normParams = execution.upload("template.output-norm-parameters", createAttentionNormParameters(
        pairs, input.templateChannels, 0, input.templateChannels, false, 1, pairs, 1e-5,
      ), GPUBufferUsage.UNIFORM);
      const normalized = execution.allocate("template.normalized", pair.elements);
      const projected = input.weights.valueWeight.length / input.templateChannels;
      const value = execution.allocate("template.value", pairs * projected);
      const output = execution.allocate(
        "template.output", pairs * input.pairChannels, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      );
      const valueWeight = execution.upload("template.value-weight", input.weights.valueWeight);
      const outputWeight = execution.upload("template.output-weight", input.weights.outputWeight);
      const outputBias = execution.upload("template.output-bias", input.weights.outputBias);
      const params = execution.upload("template.pointwise-parameters", new Uint32Array([
        pairs, input.templateChannels, projected, input.pairChannels,
      ]), GPUBufferUsage.UNIFORM);
      const [normalize, valuePipeline, outputPipeline] = await Promise.all([
        execution.pipelines.get("template:normalize", ATTENTION_NORMALIZE_SHADER),
        execution.pipelines.get("template:value", VALUE_SHADER),
        execution.pipelines.get("template:output", OUTPUT_SHADER),
      ]);
      encoder = this.device.createCommandEncoder({ label: "template.pointwise-attention" });
      this.device.pushErrorScope("validation");
      const normGrid = execution.rowGrid(pairs);
      execution.dispatch(encoder, normalize, [pair, normWeightBuffer, normParams, normalized],
        normGrid[0], normGrid[1], 1, "template.output-normalize");
      const valueGrid = execution.linearGrid(value.elements);
      execution.dispatch(encoder, valuePipeline, [normalized, valueWeight, params, value],
        valueGrid[0], valueGrid[1], 1, "template.value");
      const outputGrid = execution.linearGrid(output.elements);
      execution.dispatch(encoder, outputPipeline, [value, outputWeight, outputBias, params, output],
        outputGrid[0], outputGrid[1], 1, "template.output");
      const readback = execution.createReadback("template.readback", output, encoder);
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU template output failed: ${error.message}`);
      return {
        pairUpdate: await execution.mapFloat32(readback),
        elapsedMilliseconds: performance.now() - start,
        memory: execution.snapshot(),
      };
    } finally {
      execution.release();
    }
  }
}
