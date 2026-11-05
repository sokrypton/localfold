/**
 * The whole attention block in WebGL2 against the WGSL block and a CPU
 * reference - the cross-API differential this backend's attention has lacked.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-attention-block.js [--falsify]
 *
 * 🔴 WHY THE BLOCK AND NOT THE CORE. `AttentionGpu` on the WGSL side is the
 * whole operation - LayerNorm, the q/k/v/gate projections, the attention, the
 * output projection - so the flash core alone has nothing on that side to be
 * compared against. Porting the block is what makes the comparison possible,
 * and that comparison is what caught the triangle contraction computing the
 * pair-transpose of itself: a convention error is invisible to a reference
 * transcribed from the same shader and obvious to a second implementation.
 *
 * Two conventions are being checked here as much as the arithmetic. The weights
 * are `[in][out]` - read off `let weight_row = weight_c * projected;` in the
 * WGSL shader, where the triangle's reference uses `[out][in]` - and the
 * `1/sqrt(head_dim)` scale is applied to the query there and to the logit here,
 * which is the same number and worth confirming rather than asserting.
 *
 * 🔴 THE PAIR-BIAS ARMS ARE AF2's MSA ROW ATTENTION, WHICH IS THE ONLY PLACE
 * THE TWO TRACKS MEET before the outer product mean sends information back. The
 * bias is projected out of a separate pair tensor with its own LayerNorm
 * (`pairBias: {source: "separate", ...}` on the WGSL side), lands head-major,
 * and is read by the flash core as `(head * Q + q) * Q + k`. Three things could
 * be independently wrong - the projection's `[c][heads]` weight, the
 * pair-major-to-head-major transpose, and which of the attention's two axes the
 * bias indexes - and an i/j transpose in the last of them is invisible to any
 * reference that builds the bias the way the kernel does, so the WGSL arm is
 * what covers it.
 *
 * 🔴 AND THE WGSL SIDE HAS FOUR PRECISION AXES, WHICH THIS GATE LEARNED BY
 * MEASURING 1e-3 TWICE WITH THE WRONG ONES PINNED. `flashPrecision`,
 * `projectPrecision`, `outputPrecision` and `storage` are separate, they all
 * default to "auto", and "auto" on a device with `shader-f16` is f16 - which is
 * a 1e-3 floor and exactly what the arms read. The one that mattered was
 * `projectPrecision`, the arithmetic of the q/k/v/gate GEMM, and it was the
 * last one tried.
 *
 * 🔴 AND THE OPTION NAMES ARE NOT THE REPORTED NAMES, WHICH COST THE FIRST
 * ATTEMPT. `storage` is read as `{input, value, output}`
 * (`effective.input ?? "f32"`) and reported back as
 * `{projected, value, weighted}`. Passing the reported spelling sets nothing
 * and raises no error: the run came back saying `projected: "f32"` - which it
 * already was - and the number did not move by a digit. A gate that cannot tell
 * "the option worked" from "the option was ignored and the default agreed" is
 * why the arms print the `storage` and `shaders` keys the run reports.
 *
 * The tell throughout was that the two references disagreed by four orders of
 * magnitude about the SAME tensor - 2.5e-7 against f64, 1e-3 against WGSL. When
 * that happens the instrument is wrong, not the kernel.
 */
import { AttentionGpu } from "../../webgpu/kernels/attention.js";
import {
  pairBias as computePairBias, releasePairBiasWeights, uploadPairBiasWeights,
} from "../../webgl/kernels/pair-bias.js";
import {
  attentionBlock, releaseAttentionBlockWeights, uploadAttentionBlockWeights,
} from "../../webgl/kernels/attention-block.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { transposeLeading } from "../../webgl/runtime/transpose.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function generator(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000;
  };
}
const noise = (count, spread, random, centre = 0) => {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
  return out;
};

/** The block, from the operation. Weights are `[in][out]`. */
function blockReference(input) {
  const { activations, mask, weights: w, batch, queryLength, channels, heads } = input;
  const headDim = channels / heads;
  const rows = batch * queryLength;
  const epsilon = input.epsilon ?? 1e-5;
  const scale = 1 / Math.sqrt(headDim);

  const normalised = new Float64Array(rows * channels);
  for (let row = 0; row < rows; row += 1) {
    const at = row * channels;
    let mean = 0;
    for (let c = 0; c < channels; c += 1) mean += activations[at + c];
    mean /= channels;
    let spread = 0;
    for (let c = 0; c < channels; c += 1) spread += (activations[at + c] - mean) ** 2;
    const inverseStd = 1 / Math.sqrt(spread / channels + epsilon);
    for (let c = 0; c < channels; c += 1) {
      normalised[at + c] = (activations[at + c] - mean) * inverseStd * w.queryNormScale[c]
        + w.queryNormOffset[c];
    }
  }

  const project = (weight, bias, sigmoid) => {
    const out = new Float64Array(rows * channels);
    for (let row = 0; row < rows; row += 1) {
      for (let o = 0; o < channels; o += 1) {
        let sum = bias === undefined ? 0 : bias[o];
        for (let i = 0; i < channels; i += 1) {
          sum += normalised[row * channels + i] * weight[i * channels + o];
        }
        out[row * channels + o] = sigmoid ? 1 / (1 + Math.exp(-sum)) : sum;
      }
    }
    return out;
  };
  const query = project(w.queryWeight, undefined, false);
  const key = project(w.keyWeight, undefined, false);
  const value = project(w.valueWeight, undefined, false);
  const gate = project(w.gatingWeight, w.gatingBias, true);

  // The triangle attentions project their bias off this same normalised tensor
  // with a [channels][heads] weight, head-major.
  let bias = input.pairBias;
  if (input.pairBiasProjection !== undefined) {
    bias = new Float64Array(heads * queryLength * queryLength);
    for (let row = 0; row < rows; row += 1) {
      for (let h = 0; h < heads; h += 1) {
        let sum = 0;
        for (let c = 0; c < channels; c += 1) {
          sum += normalised[row * channels + c] * input.pairBiasProjection[c * heads + h];
        }
        bias[h * rows + row] = sum;
      }
    }
  }

  const attended = new Float64Array(rows * channels);
  for (let b = 0; b < batch; b += 1) {
    for (let head = 0; head < heads; head += 1) {
      for (let q = 0; q < queryLength; q += 1) {
        const qAt = (b * queryLength + q) * channels + head * headDim;
        const logits = new Float64Array(queryLength);
        for (let k = 0; k < queryLength; k += 1) {
          const kAt = (b * queryLength + k) * channels + head * headDim;
          let score = 0;
          for (let d = 0; d < headDim; d += 1) score += query[qAt + d] * key[kAt + d];
          // ...AF2's column attention leaves the mask in `[key][batch]` order
          // and reads it through the transpose flag; with batch == queryLength
          // the two layouts coincide, which is why the column arm's shape has
          // sequences != length.
          const maskIndex = input.transpose ? k * batch + b : b * queryLength + k;
          let logit = score * scale + 1e9 * (mask[maskIndex] - 1);
          if (bias !== undefined) {
            logit += bias[(head * queryLength + q) * queryLength + k];
          }
          logits[k] = Math.min(Math.max(logit, -1e8), 1e8);
        }
        let top = -Infinity;
        for (const logit of logits) top = Math.max(top, logit);
        let total = 0;
        const weight = new Float64Array(queryLength);
        for (let k = 0; k < queryLength; k += 1) {
          weight[k] = Math.exp(logits[k] - top);
          total += weight[k];
        }
        for (let d = 0; d < headDim; d += 1) {
          let acc = 0;
          for (let k = 0; k < queryLength; k += 1) {
            acc += weight[k] * value[(b * queryLength + k) * channels + head * headDim + d];
          }
          attended[qAt + d] = (acc / total) * gate[qAt + d];
        }
      }
    }
  }

  const out = new Float32Array(rows * channels);
  for (let row = 0; row < rows; row += 1) {
    for (let o = 0; o < channels; o += 1) {
      let sum = w.outputBias[o];
      for (let i = 0; i < channels; i += 1) {
        sum += attended[row * channels + i] * w.outputWeight[i * channels + o];
      }
      out[row * channels + o] = sum;
    }
  }
  return out;
}

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "1e-5"));
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const rows = [];
  const failures = [];
  const arm = (name, got, want) => {
    const relative = relativeRms(got, want, name);
    const ok = relative <= bar;
    rows.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };

  const cases = [
    { batch: 2, queryLength: 10, channels: 64, heads: 2 },
    { batch: 3, queryLength: 8, channels: 32, heads: 4 },
    { batch: 1, queryLength: 12, channels: 128, heads: 4 },
    // ...with a pair bias: AF2's MSA row attention, 8 heads. The pair channel
    // width is independent of the MSA's, as it is in the model.
    { batch: 3, queryLength: 7, channels: 64, heads: 8, pairChannels: 32 },
    { batch: 2, queryLength: 9, channels: 32, heads: 4, pairChannels: 16 },
  ];
  const built = [];

  for (const shape of cases) {
    const { batch, queryLength, channels, heads } = shape;
    const random = generator(batch * 131 + queryLength * 17 + channels + heads);
    const rowCount = batch * queryLength;
    const input = {
      ...shape,
      activations: noise(rowCount * channels, 1.5, random),
      mask: new Float32Array(rowCount).fill(1),
      weights: {
        queryNormScale: noise(channels, 0.2, random, 1),
        queryNormOffset: noise(channels, 0.1, random),
        queryWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
        keyWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
        valueWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
        gatingWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
        gatingBias: noise(channels, 0.1, random, 1),
        outputWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
        outputBias: noise(channels, 0.05, random),
      },
    };
    for (let b = 0; b < batch; b += 1) input.mask[b * queryLength + queryLength - 1] = 0;

    const pairChannels = shape.pairChannels;
    const withPair = pairChannels !== undefined;
    const pairInput = withPair ? {
      activations: noise(queryLength * queryLength * pairChannels, 1.5, random),
      layerNormScale: noise(pairChannels, 0.2, random, 1),
      layerNormOffset: noise(pairChannels, 0.1, random),
      projectionWeight: noise(pairChannels * heads, 1 / Math.sqrt(pairChannels), random),
      channels: pairChannels,
      source: "separate",
    } : undefined;
    if (withPair) {
      // The CPU reference takes the bias pre-projected and head-major, so build
      // it the way the WGSL kernel stores it: normalise each pair row, project
      // with a [c][heads] weight, write [head][pair].
      const pairs = queryLength * queryLength;
      const bias = new Float32Array(heads * pairs);
      for (let pair = 0; pair < pairs; pair += 1) {
        const at = pair * pairChannels;
        let mean = 0;
        for (let c = 0; c < pairChannels; c += 1) mean += pairInput.activations[at + c];
        mean /= pairChannels;
        let spread = 0;
        for (let c = 0; c < pairChannels; c += 1) {
          spread += (pairInput.activations[at + c] - mean) ** 2;
        }
        const inverseStd = 1 / Math.sqrt(spread / pairChannels + 1e-5);
        for (let h = 0; h < heads; h += 1) {
          let sum = 0;
          for (let c = 0; c < pairChannels; c += 1) {
            const value = (pairInput.activations[at + c] - mean) * inverseStd
              * pairInput.layerNormScale[c] + pairInput.layerNormOffset[c];
            sum += value * pairInput.projectionWeight[c * heads + h];
          }
          bias[h * pairs + pair] = sum;
        }
      }
      input.pairBias = bias;
    }

    const label = `b=${batch} q=${queryLength} c=${channels} h=${heads}`
      + (withPair ? ` pairC=${pairChannels}` : "");
    const weights = uploadAttentionBlockWeights(execution, input.weights);
    // 🔴 THE PERTURBATION GOES TO THE WEBGL SIDE ONLY, or it moves both
    // references with the kernel and every difference stays where it was.
    const activations = execution.upload(falsify
      ? Float32Array.from(input.activations,
        (v, i) => (i === (input.activations.length >> 1) ? v * 1.01 + 0.01 : v))
      : input.activations, "activations");
    const mask = execution.upload(input.mask, "mask");

    let pairWeights;
    let pairTensor;
    let bias;
    if (withPair) {
      pairWeights = uploadPairBiasWeights(execution, pairInput);
      pairTensor = execution.upload(pairInput.activations, "pair");
      bias = computePairBias(execution, {
        pair: pairTensor, weights: pairWeights, length: queryLength, pairChannels, heads,
      });
    }

    const output = attentionBlock(execution, {
      ...shape, activations, mask, weights, pairBias: bias,
    });
    const ours = execution.read(output).slice();
    execution.release(output);

    arm(`vs CPU reference ${label}`, ours, blockReference(input));

    const wgsl = await new AttentionGpu(device, {
      flashPrecision: "f32",
      projectPrecision: "f32",
      outputPrecision: "f32",
      // ...the keys this option is READ with; it reports them back under three
      // different names, and the reported spelling sets nothing.
      storage: { input: "f32", value: "f32", output: "f32" },
    }).run({
      batch, queryLength, channels, heads,
      activations: input.activations, mask: input.mask, weights: input.weights,
      ...(withPair ? { pairBias: pairInput } : {}),
    });
    arm(`vs WGSL block ${label}`, ours, wgsl);
    built.push({ shape: label, storage: wgsl.storage, shaders: wgsl.shaders });

    releaseAttentionBlockWeights(execution, weights);
    execution.release(activations);
    execution.release(mask);
    if (withPair) {
      releasePairBiasWeights(execution, pairWeights);
      execution.release(pairTensor);
      execution.release(bias);
    }
  }

  // --- TRIANGLE attention: the bias projected off the block's own input ----
  //
  // 🔴 THE DIFFERENCE FROM THE MSA ROW ARMS IS WHERE THE BIAS COMES FROM, and it
  // is the whole reason the WGSL side has two pair-bias paths. Here the bias and
  // the q/k/v are projections of ONE normalised tensor - the pair - so a second
  // LayerNorm would compute the same statistics twice. `batch == queryLength`
  // because the activations are a square pair matrix, which is also the only
  // shape for which a `[head][query][key]` bias over `queryLength^2` has the
  // right length; the block refuses the rest.
  {
    const length = 7;
    const channels = 32;
    const heads = 4;
    const random = generator(31337);
    const rowCount = length * length;
    const pair = noise(rowCount * channels, 1.5, random);
    const mask = new Float32Array(rowCount).fill(1);
    mask[3] = 0;
    mask[rowCount - 2] = 0;
    const weights = {
      queryNormScale: noise(channels, 0.2, random, 1),
      queryNormOffset: noise(channels, 0.1, random),
      queryWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
      keyWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
      valueWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
      gatingWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
      gatingBias: noise(channels, 0.1, random, 1),
      outputWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
      outputBias: noise(channels, 0.05, random),
    };
    const projectionWeight = noise(channels * heads, 1 / Math.sqrt(channels), random);

    const uploaded = uploadAttentionBlockWeights(execution, weights);
    const activations = execution.upload(falsify
      ? Float32Array.from(pair, (v, i) => (i === (pair.length >> 1) ? v * 1.01 + 0.01 : v))
      : pair, "pair");
    const maskTensor = execution.upload(mask, "mask");
    const projection = execution.upload(projectionWeight, "w.pairProjection");
    const output = attentionBlock(execution, {
      activations, mask: maskTensor, weights: uploaded,
      batch: length, queryLength: length, channels, heads,
      pairBiasProjection: projection,
    });
    const ours = execution.read(output).slice();

    const label = `triangle L=${length} c=${channels} h=${heads}`;
    arm(`vs CPU reference ${label}`, ours, blockReference({
      activations: pair, mask, weights, batch: length, queryLength: length,
      channels, heads, pairBiasProjection: projectionWeight,
    }));
    const wgsl = await new AttentionGpu(device, {
      flashPrecision: "f32",
      projectPrecision: "f32",
      outputPrecision: "f32",
      storage: { input: "f32", value: "f32", output: "f32" },
    }).run({
      batch: length, queryLength: length, channels, heads,
      activations: pair, mask, weights,
      // ...no `source`, so the WGSL side projects it off its own normalised
      // input with `pair_channels` set to `channels`.
      pairBias: { projectionWeight },
    });
    arm(`vs WGSL block ${label}`, ours, wgsl);

    releaseAttentionBlockWeights(execution, uploaded);
    for (const tensor of [activations, maskTensor, projection, output]) {
      execution.release(tensor);
    }
  }

  // --- MSA COLUMN attention: the same block over the other axis ------------
  //
  // 🔴 THE THREE SIDES ARE DRIVEN THROUGH THREE DIFFERENT DOORS ON PURPOSE,
  // AND WORKING OUT WHICH TOOK A PROBE RATHER THAN A THIRD GUESS.
  //
  // This backend transposes the activations with a kernel and leaves the mask in
  // AF2's `[sequence][residue]` order, reading it through the flash core's
  // `transpose` flag as `k * BATCH + b`. That path agrees with the f64 reference
  // at 2.0e-7, so the flag is verified independently.
  //
  // `AttentionGpu` with `transpose: true` does not reproduce column attention at
  // all, which took a sweep to establish rather than a third reading. Its
  // `mask_index` becomes `key_index * p.batch + batch`, its `source_row(row)`
  // becomes `q * p.batch + b`, the norm shader does use `source_row` for its
  // input, and `run` does plumb the flag into both - so AF2's
  // `[sequence][residue]` activations with a `[sequence][residue]` mask ought to
  // be exactly it. `tools/gpu/probe-webgl-column-layout.js` enumerates all **32**
  // arrangements of activation order, mask order, the flag, and the output taken
  // as-given or swapped, against an f64 reference under both mask conventions:
  // **two agree and both have `transpose: false`**. Sixteen of the thirty-two
  // have the flag set and not one of them lands.
  //
  // That is a defect in the standalone class rather than in the fold, which
  // drives column attention through `encodeAttention` directly - and CLAUDE.md
  // already records this same class being checked in "a configuration nothing
  // runs" on its storage axis. It is reported, not worked around: the probe is
  // the evidence and this gate uses the arrangement that agrees.
  //
  // What the probe found agreeing, at 2.056e-7, is `transpose: false` with BOTH
  // inputs pre-transposed - residue-major activations and a `[residue][sequence]`
  // mask. So that is how the WGSL arm is driven here. The comparison still
  // covers the whole composition; what it does not cover is that class's
  // `transpose: true`, which nothing in this tree needs.
  //
  // The shape deliberately has sequences != length: with `batch == queryLength`
  // the two mask layouts coincide and the flag would be untested.
  {
    const sequences = 5;
    const length = 9;
    const channels = 32;
    const heads = 4;
    const random = generator(777);
    const rowCount = sequences * length;
    const bySequence = noise(rowCount * channels, 1.5, random);
    // [sequence][residue][channels] -> [residue][sequence][channels], in JS, for
    // the two references.
    const byResidue = new Float32Array(bySequence.length);
    for (let s = 0; s < sequences; s += 1) {
      for (let i = 0; i < length; i += 1) {
        for (let c = 0; c < channels; c += 1) {
          byResidue[(i * sequences + s) * channels + c]
            = bySequence[(s * length + i) * channels + c];
        }
      }
    }
    // AF2's mask order, [sequence][residue], for this backend's flag...
    const mask = new Float32Array(rowCount).fill(1);
    mask[(sequences - 1) * length + 2] = 0;
    mask[1 * length + (length - 1)] = 0;
    // ...and the same mask transposed, for the WGSL arm, which is driven
    // flagless.
    const maskByResidue = new Float32Array(rowCount).fill(1);
    for (let s = 0; s < sequences; s += 1) {
      for (let i = 0; i < length; i += 1) {
        maskByResidue[i * sequences + s] = mask[s * length + i];
      }
    }

    const weights = {
      queryNormScale: noise(channels, 0.2, random, 1),
      queryNormOffset: noise(channels, 0.1, random),
      queryWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
      keyWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
      valueWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
      gatingWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
      gatingBias: noise(channels, 0.1, random, 1),
      outputWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
      outputBias: noise(channels, 0.05, random),
    };

    const uploaded = uploadAttentionBlockWeights(execution, weights);
    const source = execution.upload(falsify
      ? Float32Array.from(bySequence,
        (v, i) => (i === (bySequence.length >> 1) ? v * 1.01 + 0.01 : v))
      : bySequence, "msa");
    const transposed = transposeLeading(execution, {
      x: source, outer: sequences, inner: length, channels,
    });
    const maskTensor = execution.upload(mask, "mask");
    const output = attentionBlock(execution, {
      activations: transposed, mask: maskTensor, weights: uploaded,
      batch: length, queryLength: sequences, channels, heads, transpose: true,
    });
    const ours = execution.read(output).slice();

    const label = `column s=${sequences} L=${length} c=${channels} h=${heads}`;
    arm(`vs CPU reference ${label}`, ours, blockReference({
      activations: byResidue, mask, weights, batch: length, queryLength: sequences,
      channels, heads, transpose: true,
    }));
    const wgsl = await new AttentionGpu(device, {
      flashPrecision: "f32",
      projectPrecision: "f32",
      outputPrecision: "f32",
      storage: { input: "f32", value: "f32", output: "f32" },
    }).run({
      batch: length, queryLength: sequences, channels, heads,
      // Both inputs pre-transposed and NO flag - the one configuration of the
      // eight probed that agrees. See the note above.
      activations: byResidue, mask: maskByResidue, weights,
    });
    arm(`vs WGSL block ${label}`, ours, wgsl);

    releaseAttentionBlockWeights(execution, uploaded);
    for (const tensor of [source, transposed, maskTensor, output]) execution.release(tensor);
  }

  const stats = execution.stats();
  const red = rows.filter((row) => !row.ok).length;
  return {
    bar,
    falsify,
    arms: rows,
    wgslPaths: built,
    liveTensorsAfter: stats.live,
    drawsPerBlock: Math.round(stats.draws / cases.length),
    verdict: falsify
      ? (red === rows.length
        ? `ok: all ${rows.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${rows.length} arms noticed a 1% perturbation`)
      : (failures.length === 0 && stats.live === 0
        ? `ok: ${rows.length} arms, no tensor left live`
        : `FAILED: ${[...failures, ...(stats.live ? [`${stats.live} tensors leaked`] : [])].join("; ")}`),
  };
}
