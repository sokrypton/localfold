/**
 * The AF3 pairformer reference's primitives, against values worked out by hand.
 *
 * The block as a whole is checked against AF3 itself by
 * tools/oracle/check_af3_block.js, which needs a 150 MiB export and a JAX run
 * and so cannot live here. What CAN live here is everything that has a right
 * answer independent of the model: the layout conventions, which are where the
 * wiring actually goes wrong, and which no amount of "it runs" would catch.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { layerNorm, linear, transition } from "../src/af3/pairformer-reference.js";

const close = (actual, expected, tolerance = 1e-5) => {
  assert.equal(actual.length, expected.length, "length");
  for (let index = 0; index < expected.length; index += 1) {
    assert.ok(Math.abs(actual[index] - expected[index]) <= tolerance,
              `[${index}]: ${actual[index]} != ${expected[index]}`);
  }
};

describe("AF3 layerNorm", () => {
  it("centres and scales each row independently", () => {
    const input = Float32Array.from([1, 2, 3, 4, 10, 20, 30, 40]);
    const scale = Float32Array.from([1, 1, 1, 1]);
    const offset = Float32Array.from([0, 0, 0, 0]);
    const output = layerNorm(input, 2, 4, scale, offset);
    // Both rows are the same shape scaled by ten, so a row-wise norm maps them
    // to the same values - which is the property that catches normalising over
    // the wrong axis.
    close(output.subarray(0, 4), output.subarray(4, 8), 1e-4);
    for (const row of [0, 1]) {
      let sum = 0;
      for (let c = 0; c < 4; c += 1) sum += output[row * 4 + c];
      assert.ok(Math.abs(sum) < 1e-4, `row ${row} should be centred, got ${sum}`);
    }
  });

  it("applies scale and offset per channel", () => {
    const input = Float32Array.from([1, 2, 3, 4]);
    const output = layerNorm(input, 1, 4, Float32Array.from([0, 0, 0, 0]),
                             Float32Array.from([5, 6, 7, 8]));
    close(output, Float32Array.from([5, 6, 7, 8]));
  });

  it("uses E[x^2] - E[x]^2, as AF3's use_fast_variance does", () => {
    // A constant row has zero variance either way, so the epsilon is what shows.
    const output = layerNorm(Float32Array.from([2, 2, 2, 2]), 1, 4,
                             Float32Array.from([1, 1, 1, 1]),
                             Float32Array.from([0, 0, 0, 0]));
    close(output, Float32Array.from([0, 0, 0, 0]));
  });
});

describe("AF3 linear", () => {
  it("reads (in, out) weights by default", () => {
    // input 1x2, weights 2x3 laid out row-major as (in, out)
    const output = linear(Float32Array.from([1, 2]), 1, 2, 3,
                          Float32Array.from([1, 2, 3, 10, 20, 30]));
    close(output, Float32Array.from([21, 42, 63]));
  });

  it("reads (out, in) weights when transposed, which is AF3's transpose_weights", () => {
    // The same map, stored the other way round. q and k projections use this
    // and v does not, in the same module.
    const output = linear(Float32Array.from([1, 2]), 1, 2, 3,
                          Float32Array.from([1, 10, 2, 20, 3, 30]), null, true);
    close(output, Float32Array.from([21, 42, 63]));
  });

  it("adds the bias when there is one", () => {
    const output = linear(Float32Array.from([1]), 1, 1, 2,
                          Float32Array.from([1, 1]), Float32Array.from([5, -5]));
    close(output, Float32Array.from([6, -4]));
  });
});

describe("AF3 transition", () => {
  it("splits transition1 BLOCKED, gate half first", () => {
    // 🔴 THE SPLIT THAT IS NOT THE TRIANGLE MULTIPLICATION'S. channels 1 means
    // intermediate 4 and transition1 is 1x8: outputs 0..3 are the swish half
    // and 4..7 the multiplicand. Interleaving them instead would pair output 0
    // with output 1 and return a different, entirely plausible number.
    const channels = 1;
    const intermediate = channels * 4;
    // LayerNorm of a single channel is exactly the offset, so pin it at 1 and
    // the transition becomes arithmetic we can do on paper.
    const weights = {
      inputLayerNormScale: Float32Array.from([1]),
      inputLayerNormOffset: Float32Array.from([1]),
      transition1: new Float32Array(intermediate * 2),
      transition2: new Float32Array(intermediate),
    };
    // gate half = [1,0,0,0], value half = [3,0,0,0]  ->  swish(1) * 3
    weights.transition1[0] = 1;
    weights.transition1[intermediate] = 3;
    weights.transition2[0] = 1;
    const output = transition(Float32Array.from([7]), 1, channels, weights);
    const swish1 = 1 / (1 + Math.exp(-1));
    close(output, Float32Array.from([swish1 * 3]), 1e-6);
  });
});
