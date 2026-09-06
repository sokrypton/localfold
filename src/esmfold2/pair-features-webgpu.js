/**
 * `z_init` and the recycle projection, on the device.
 *
 * 🔴 THESE ARE ON THE GPU BECAUSE OF n^2, NOT BECAUSE THEY ARE HARD. Every term
 * is a row lookup or a broadcast - `relativePositionEncoding` adds four weight
 * rows a pair and `zInitFromInputs` broadcasts two per-token projections - so
 * the arithmetic is trivial and there is 90,000 of it at 300 tokens, four times
 * over if the recycle runs on the host as well. src/esmfold2/featuriser-reference.js
 * is the specification and the oracle; this is the same arithmetic in a shader.
 *
 * 🔴 AND `rel_pos` IS NEEDED TWICE, WHICH IS WHY IT IS A BUFFER AND NOT A PASS.
 * `z_init` adds it, and the diffusion conditioning CONCATENATES the same tensor
 * onto the trunk's pair - so it outlives the trunk and is kept.
 */
import { GRID_WIDTH, LANES } from "../esmc/block-webgpu.js";
import { relativePositionBins } from "./featuriser-reference.js";

/**
 * The four relative-position blocks, gathered from their bins.
 *
 * 🔴 THE CHAIN BLOCK'S POLARITY IS THE OPPOSITE OF THE OTHER TWO - see the note
 * on `relativePositionBins`, which is where that is decided. This kernel is
 * handed the bins and cannot get it wrong; the host can.
 */
export function createRelativePositionShader({ pairs, channels, entityBase }) {
  return `
@group(0) @binding(0) var<storage, read> bins: array<i32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${pairs * channels}u) { return; }
  let pair = i / ${channels}u;
  let c = i % ${channels}u;
  let at = pair * 4u;
  var total = weights[u32(bins[at]) * ${channels}u + c]
            + weights[u32(bins[at + 1u]) * ${channels}u + c]
            + weights[u32(bins[at + 3u]) * ${channels}u + c];
  // ...same_entity is a value in [0, 1] rather than a one-hot, so it scales.
  total += f32(bins[at + 2u]) * weights[${entityBase}u * ${channels}u + c];
  output[i] = total;
}`;
}

/**
 * `z_init = rows[i] + columns[j] + relPos + bonds + lm`, in one pass.
 *
 * 🔴 PROJECT THEN BROADCAST, NEVER BROADCAST THEN PROJECT. The two per-token
 * projections are done on the host - 2n rows of 451 channels - and this only
 * places them, which is the difference between 2n projections and 2n^2.
 */
export function createZInitShader({ tokens, channels, hasBonds, hasLanguageModel }) {
  const pairs = tokens * tokens;
  return `
@group(0) @binding(0) var<storage, read> rows: array<f32>;
@group(0) @binding(1) var<storage, read> columns: array<f32>;
@group(0) @binding(2) var<storage, read> relative: array<f32>;
${hasBonds ? "@group(0) @binding(3) var<storage, read> bonds: array<f32>;" : ""}
${hasLanguageModel ? `@group(0) @binding(${hasBonds ? 4 : 3}) var<storage, read> language: array<f32>;` : ""}
@group(0) @binding(${3 + (hasBonds ? 1 : 0) + (hasLanguageModel ? 1 : 0)}) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${pairs * channels}u) { return; }
  let pair = i / ${channels}u;
  let c = i % ${channels}u;
  let row = pair / ${tokens}u;
  let column = pair % ${tokens}u;
  output[i] = rows[row * ${channels}u + c] + columns[column * ${channels}u + c]
    + relative[i]${hasBonds ? " + bonds[i]" : ""}${hasLanguageModel ? " + language[i]" : ""};
}`;
}

/** `token_bonds`: one input channel, so the projection is an outer product. */
export function createBondShader({ pairs, channels }) {
  return `
@group(0) @binding(0) var<storage, read> bonds: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${pairs * channels}u) { return; }
  output[i] = bonds[i / ${channels}u] * weights[i % ${channels}u];
}`;
}

/** The bins a caller uploads, so the shader has nothing to decide. */
export { relativePositionBins };

/**
 * The concatenation order of the 139 features, and where `same_entity` sits.
 *
 * residue (2r + 2) | token (2r + 2) | same_entity (1) | chain (2c + 2)
 */
export function relativeLayout(residxBins = 32, chainBins = 2) {
  const residueWidth = 2 * residxBins + 2;
  const tokenBase = residueWidth;
  const entityBase = tokenBase + residueWidth;
  const chainBase = entityBase + 1;
  return { residueWidth, tokenBase, entityBase, chainBase,
           features: chainBase + 2 * chainBins + 2 };
}

/**
 * The bins, shifted so each names a row of the concatenated matrix.
 *
 * 🔴 THE SHADER GATHERS ROWS AND THE BLOCKS ARE CONCATENATED, so a bin that
 * indexes its own block has to be offset by where that block starts. Doing it
 * here rather than in the shader is four additions a pair on the host instead
 * of four in every invocation, and it keeps `relativePositionBins` - which the
 * CPU reference and its checker share - untouched.
 */
export function relativeRows(features, tokens, residxBins = 32, chainBins = 2) {
  const bins = relativePositionBins(features, tokens, residxBins, chainBins);
  const { tokenBase, chainBase } = relativeLayout(residxBins, chainBins);
  const out = new Int32Array(bins.length);
  for (let pair = 0; pair < tokens * tokens; pair += 1) {
    const at = pair * 4;
    out[at] = bins[at];
    out[at + 1] = tokenBase + bins[at + 1];
    out[at + 2] = bins[at + 2];
    out[at + 3] = chainBase + bins[at + 3];
  }
  return out;
}
