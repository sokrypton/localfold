/**
 * The CPU fold's SIMD multiply against the reference loops it stands in for.
 *
 * The reference kernels are the specification, so every comparison runs the same
 * kernel twice on the same input, with the backend off and on. They differ in float
 * rounding order only, so the bound is relative and a few float32 ulps wide.
 */
import { strict as assert } from "node:assert";
import { after, before, describe, it } from "node:test";

import { linear as pairformerLinear, transition, triangleMultiplication }
  from "../cpu/af3/trunk/pairformer.js";
import { esmcAttention, linear as towerLinear } from "../cpu/esmc/tower.js";
import { CpuBackend } from "../cpu/runtime/backend.js";
import { gemmSupported } from "../cpu/runtime/gemm-wasm.js";
import { setLinearBackend } from "../cpu/runtime/linear-backend.js";

/** A small deterministic generator: these are shapes and arithmetic, not statistics. */
function randoms(length, seed) {
  const out = new Float32Array(length);
  let state = seed >>> 0;
  for (let index = 0; index < length; index += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    out[index] = state / 2147483648 - 1;
  }
  return out;
}

function relative(got, want) {
  assert.equal(got.length, want.length);
  let error = 0;
  let total = 0;
  for (let index = 0; index < want.length; index += 1) {
    const difference = got[index] - want[index];
    error += difference * difference;
    total += want[index] * want[index];
  }
  return Math.sqrt(error / Math.max(total, 1e-30));
}

const BOUND = 2e-6;

for (const threads of [1, 3]) {
  describe(`the SIMD multiply on ${threads} thread(s)`, () => {
    let backend;
    before(async () => { backend = await CpuBackend.create({ threads, gibibytes: 0.25 }); });
    after(() => { setLinearBackend(undefined); backend.close(); });

    it("is available here at all", () => assert.equal(gemmSupported(), true));

    for (const [rows, inner, outer] of [[40, 32, 48], [17, 33, 7], [64, 5, 130], [200, 64, 256], [3, 257, 513]]) {
      it(`matches the reference at ${rows} x ${inner} x ${outer}`, () => {
        const input = randoms(rows * inner, rows + 1);
        const weights = randoms(inner * outer, inner + 2);
        setLinearBackend(undefined);
        const want = pairformerLinear(input, rows, inner, outer, weights);
        setLinearBackend(backend);
        const got = pairformerLinear(input, rows, inner, outer, weights);
        setLinearBackend(undefined);
        assert.ok(relative(got, want) < BOUND, `relRMS ${relative(got, want)}`);
      });
    }

    it("adds a bias where a kernel has one", () => {
      const input = randoms(50 * 64, 3);
      const weights = randoms(64 * 96, 4);
      const bias = randoms(96, 5);
      setLinearBackend(undefined);
      const want = pairformerLinear(input, 50, 64, 96, weights, bias);
      setLinearBackend(backend);
      const got = pairformerLinear(input, 50, 64, 96, weights, bias);
      setLinearBackend(undefined);
      assert.ok(relative(got, want) < BOUND);
    });

    it("finishes a long input in chunks when its scratch is short", () => {
      // 96 MiB of input and output through a 256 MiB memory holding these weights.
      const rows = 16000;
      const input = randoms(rows * 256, 6);
      const weights = randoms(256 * 1024, 7);
      setLinearBackend(backend);
      const got = towerLinear(input, rows, 256, weights, 1024);
      setLinearBackend(undefined);
      const sample = [0, 1, rows - 1];
      for (const row of sample) {
        const want = towerLinear(input.subarray(row * 256, (row + 1) * 256), 1, 256, weights, 1024);
        assert.ok(relative(got.subarray(row * 1024, (row + 1) * 1024), want) < BOUND);
      }
    });

    for (const direction of ["outgoing", "incoming"]) {
      it(`gives the reference's ${direction} triangle`, () => {
        const n = 24;
        const channels = 12;
        const pairs = n * n;
        const weights = {
          leftNormInputScale: randoms(channels, 11), leftNormInputOffset: randoms(channels, 12),
          centerNormScale: randoms(channels, 13), centerNormOffset: randoms(channels, 14),
          projection: randoms(channels * channels * 2, 15), gate: randoms(channels * channels * 2, 16),
          outputProjection: randoms(channels * channels, 17), gatingLinear: randoms(channels * channels, 18),
        };
        const pair = randoms(pairs * channels, 19);
        const mask = new Float32Array(pairs).fill(1);
        setLinearBackend(undefined);
        const want = triangleMultiplication(pair, mask, n, channels, direction, weights);
        setLinearBackend(backend);
        const got = triangleMultiplication(pair, mask, n, channels, direction, weights);
        setLinearBackend(undefined);
        assert.ok(relative(got, want) < 2e-5, `relRMS ${relative(got, want)}`);
      });
    }

    it("gives the reference's transition", () => {
      const rows = 90;
      const channels = 32;
      const weights = {
        inputLayerNormScale: randoms(channels, 21), inputLayerNormOffset: randoms(channels, 22),
        transition1: randoms(channels * channels * 8, 23), transition2: randoms(channels * 4 * channels, 24),
      };
      const input = randoms(rows * channels, 25);
      setLinearBackend(undefined);
      const want = transition(input, rows, channels, weights);
      setLinearBackend(backend);
      const got = transition(input, rows, channels, weights);
      setLinearBackend(undefined);
      assert.ok(relative(got, want) < 2e-5, `relRMS ${relative(got, want)}`);
    });

    it("computes the logistic to float32 accuracy across its range", () => {
      const input = new Float32Array(20011);
      for (let at = 0; at < input.length; at += 1) input[at] = -60 + (120 * at) / input.length;
      const got = backend.logistic(input);
      assert.equal(got.length, input.length);
      for (let at = 0; at < input.length; at += 1) {
        const want = 1 / (1 + Math.exp(-input[at]));
        assert.ok(Math.abs(got[at] - want) <= 2.5e-7 * want + 1e-37,
                  `at ${input[at]}: ${got[at]} against ${want}`);
      }
    });

    it("saturates rather than overflowing far outside a gate's range", () => {
      const got = backend.logistic(Float32Array.from({ length: 8192 }, (_, at) => (at % 2 === 0 ? -1e6 : 1e6)));
      assert.ok(got[0] < 1e-30 && Number.isFinite(got[0]));
      assert.equal(got[1], 1);
    });

    it("keeps an adopted weight's values, and adopts it once", () => {
      const weights = randoms(64, 31);
      const adopted = backend.adopt(weights);
      assert.deepEqual(Array.from(adopted), Array.from(weights));
      assert.equal(backend.adopt(weights).byteOffset, adopted.byteOffset);
      assert.equal(backend.adopt(adopted), adopted);
    });
  });
}

describe("a backend whose weights pass 2 GiB", () => {
  // 600M's weights do, and 300M's do not, so nothing else here reaches these offsets.
  it("multiplies, contracts and gates at offsets above 2^31", { skip: !gemmSupported() }, async () => {
    const high = await CpuBackend.create({ threads: 2 });
    const low = await CpuBackend.create({ threads: 2 });
    try {
      high.reserve(2.5 * 2 ** 30);
      const n = 48;
      const channels = 32;
      const input = randoms(4096 * channels, 21);
      const weights = randoms(channels * 64, 22);
      relative(high.linear(input, 4096, channels, high.adopt(weights), 64),
               low.linear(input, 4096, channels, low.adopt(weights), 64));
      const a = randoms(channels * n * n, 23);
      const b = randoms(channels * n * n, 24);
      relative(high.triangle(a, b, n, channels, "outgoing"), low.triangle(a, b, n, channels, "outgoing"));
      relative(high.logistic(a), low.logistic(a));
    } finally {
      high.close();
      low.close();
    }
  });
});

describe("the language model's attention across chains", () => {
  it("is what each chain gets alone", () => {
    const heads = 2;
    const headDim = 8;
    const width = heads * headDim;
    const rows = 10;
    const query = randoms(rows * width, 41);
    const key = randoms(rows * width, 42);
    const value = randoms(rows * width, 43);
    const sequenceId = Int32Array.from([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
    const joint = esmcAttention(query, key, value, rows, heads, headDim, sequenceId);
    const alone = esmcAttention(query.subarray(4 * width), key.subarray(4 * width),
                                value.subarray(4 * width), 6, heads, headDim);
    assert.ok(relative(joint.subarray(4 * width), alone) < 1e-6);
  });
});
