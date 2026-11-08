/**
 * ESM-C's tower in WebGL2 against `cpu/esmc/tower.js`, on a REAL checkpoint.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-esmc-tower.js [--falsify]
 *
 * 🔴 ON THE REAL 600M BUNDLE, NOT ON NOISE, because that is what is on this box
 * and because "a packing that moved the real model 4e-4 read 6e-7 on noise".
 * The f64 reference reads the same int3-decoded tensors, so what is compared is
 * two implementations of the arithmetic and not two decodes.
 *
 * 🔴 AND IT CHECKS ONE BLOCK'S INSIDES BEFORE IT CHECKS THE TOWER, which is
 * `tools/check-esmc-reference.js`'s own rule: "a tower that disagrees at state
 * 36 says nothing about where; the four matmul INPUTS of block 0 separate a
 * wrong LayerNorm from a wrong RoPE from a wrong SwiGLU split, and they are the
 * difference between a fix and a bisection". So the arms are the attention norm,
 * q and k AFTER their full-width norm and RoPE, v, the post-attention residual,
 * the FFN norm, and only then the block's output.
 *
 * 🔴 THE TWO SILENT CONVENTIONS ARE WHAT THOSE ARMS ARE FOR: the qk LayerNorm is
 * over the whole 1152 and not per head, and RoPE is the SPLIT-HALVES form rather
 * than the interleaved one. Both run, both fold, and neither changes a shape.
 *
 * 🔴 AND THE TOWER RUNS A PREFIX. 36 blocks of 1152 against an f64 reference in
 * JavaScript is minutes a side for an arm that a 2-block tower already answers -
 * what the depth would add is accumulation, and `--blocks=` is how to ask for
 * more. The mix is still normalised over every logit the bundle carries, on both
 * sides, because `layerMix` is over the whole tensor and the sum is over the
 * states that exist.
 */
import {
  esmcBlock as cpuBlock, layerNorm as cpuLayerNorm, linear as cpuLinear,
  applyRope, sequenceIds, shimPair as cpuShimPair, shimSingle as cpuShimSingle,
} from "../../cpu/esmc/tower.js";
import {
  esmcBlock, shimPair, towerSingle,
} from "../../webgl/esmc/tower.js";
import { HttpTensorStore } from "../../shared/bundles/http-tensor-store.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "2e-5"));
  const sequence = option(args, "sequence", "MKTAYIAKQR");
  const blocks = Number(option(args, "blocks", "2"));
  const bundle = option(args, "bundle", "/model-esmc-600m-int3").replace(/\/$/, "");

  const store = await HttpTensorStore.open(`${bundle}/manifest.json`);
  const shapeOf = (name) => store.manifest.tensors[name]?.shape;
  const model = shapeOf("embed/weights")[1];
  const ffn = shapeOf("blocks/0/fc2/weights")[0];
  const pair = shapeOf("lm/projection/weights")[1];
  const heads = Number(option(args, "heads", String(model / 64)));
  let layers = 0;
  while (shapeOf(`blocks/${layers}/qkv/weights`) !== undefined) layers += 1;
  const dims = { model, heads, ffn, layers, pair };

  const ids = sequenceIds(sequence);
  const rows = ids.length;
  const names = ["embed/weights", "final_norm/scale", "lm/combine", "lm/norm/scale",
    "lm/norm/offset", "lm/projection/weights", "lm/downproject/weights",
    "lm/downproject/bias", "lm/pair_mlp_1/weights", "lm/pair_mlp_1/bias",
    "lm/pair_mlp_2/weights", "lm/pair_mlp_2/bias", "lm/pair_norm/scale",
    "lm/pair_norm/offset"];
  for (let layer = 0; layer < blocks; layer += 1) {
    for (const leaf of ["attn_norm/scale", "attn_norm/offset", "qkv/weights",
      "q_norm/scale", "k_norm/scale", "attn_out/weights", "ffn_norm/scale",
      "ffn_norm/offset", "fc1/weights", "fc2/weights"]) {
      names.push(`blocks/${layer}/${leaf}`);
    }
  }
  /** @type {Record<string, Float32Array>} */
  const host = {};
  for (const name of names) host[name] = await store.tensor(name);

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  // 🔴 THE SKEW GOES TO THE DEVICE ALONE and it goes on the BLOCK's input where
  // there is one, because a control that feeds both sides is not a control. The
  // tower's own input is a token id, which cannot be perturbed by a percent - so
  // its arms take a skewed EMBEDDING TABLE instead.
  const skew = (values, period) => Float32Array.from(values,
    (v, i) => (i % period < period / 2 ? v * 1.01 + 0.01 : v));
  const weights = {};
  for (const name of names) {
    weights[name] = name === "lm/combine" ? host[name] : upload(
      falsify && name === "embed/weights" ? skew(host[name], model) : host[name],
      `w.esmc.${name}`);
  }

  // A block's input: the real embedding of this sequence, which is what block 0
  // sees in a fold.
  const embedded = new Float32Array(rows * model);
  for (let row = 0; row < rows; row += 1) {
    embedded.set(
      host["embed/weights"].subarray(ids[row] * model, (ids[row] + 1) * model),
      row * model);
  }
  const blockInput = upload(falsify ? skew(embedded, model) : embedded, "esmc.x");

  const arms = [];
  const failures = [];
  const ourStages = {};
  const out = esmcBlock(execution, {
    x: blockInput, rows, weights, dims, layer: 0,
    onStage: (stage, tensor) => {
      ourStages[stage] = Float32Array.from(execution.read(tensor));
    },
  });
  const ourBlock = Float32Array.from(execution.read(out));
  execution.release(out);

  // The reference's own four matmul inputs, step by step - the same spelling
  // `check-esmc-reference.js` uses against the oracle.
  const at = (leaf) => host[`blocks/0/${leaf}`];
  const theirStages = {};
  theirStages["attn-normed"] = cpuLayerNorm(embedded.slice(), rows, model,
    at("attn_norm/scale"), at("attn_norm/offset"));
  const projected = cpuLinear(theirStages["attn-normed"], rows, model,
    at("qkv/weights"), 3 * model);
  for (const [index, name] of [[0, "query"], [1, "key"], [2, "value"]]) {
    const slice = new Float32Array(rows * model);
    for (let row = 0; row < rows; row += 1) {
      slice.set(
        projected.subarray(row * 3 * model + index * model,
          row * 3 * model + (index + 1) * model),
        row * model);
    }
    theirStages[name] = slice;
  }
  cpuLayerNorm(theirStages.query, rows, model, at("q_norm/scale"), null);
  cpuLayerNorm(theirStages.key, rows, model, at("k_norm/scale"), null);
  applyRope(theirStages.query, rows, heads, model / heads);
  applyRope(theirStages.key, rows, heads, model / heads);
  const theirBlock = cpuBlock(embedded, rows, host, dims, 0);

  for (const stage of ["attn-normed", "query", "key", "value"]) {
    const relative = relativeRms(ourStages[stage], theirStages[stage], stage);
    const ok = relative <= bar;
    arms.push({ arm: `block 0, ${stage}`, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${stage}: relRMS ${relative.toExponential(3)} over ${bar}`);
  }
  {
    const relative = relativeRms(ourBlock, theirBlock, "block");
    const ok = relative <= bar;
    arms.push({ arm: "block 0's output", relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`block 0: relRMS ${relative.toExponential(3)} over ${bar}`);
  }

  // The tower prefix and the shim, which is where the mix and the final norm are.
  const tower = towerSingle(execution, {
    ids, weights, dims, combine: host["lm/combine"], blocks, stripTerminals: true,
  });
  const ourSingle = Float32Array.from(execution.read(tower.single));
  const ourPair = (() => {
    const tensor = shimPair(execution, {
      single: tower.single, rows: rows - 2, weights, pair,
    });
    const values = Float32Array.from(execution.read(tensor));
    execution.release(tensor);
    return values;
  })();
  execution.release(tower.single);
  execution.release(tower.lastState);

  // The reference's: the states, then the mix over the INNER rows.
  let state = embedded;
  const states = [state];
  for (let layer = 0; layer < blocks; layer += 1) {
    state = cpuBlock(state, rows, host, dims, layer);
    states.push(state);
  }
  states[states.length - 1] = cpuLayerNorm(states[states.length - 1].slice(),
    rows, model, host["final_norm/scale"], null);
  const inner = states.map((values) => values.subarray(model, values.length - model));
  const theirSingle = cpuShimSingle(inner, rows - 2, host, dims);
  const theirPair = cpuShimPair(theirSingle, rows - 2, host, dims);
  for (const [name, mine, theirs] of [["the mixed single", ourSingle, theirSingle],
    ["the pair representation", ourPair, theirPair]]) {
    const relative = relativeRms(mine, theirs, name);
    const ok = relative <= bar;
    arms.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  }

  for (const tensor of held) execution.release(tensor);
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const red = arms.filter((row) => !row.ok).length;
  return {
    bundle, sequence, rows, blocks,
    dims,
    residualScale: Number(Math.sqrt(layers / 36).toPrecision(6)),
    bar,
    falsify,
    arms,
    memory: execution.stats(),
    verdict: falsify
      ? (red === arms.length
        ? `ok: all ${arms.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${arms.length} arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms on a real checkpoint, ${blocks} of ${layers} blocks`
        : `FAILED: ${failures.join("; ")}`),
  };
}
