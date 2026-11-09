/**
 * `linear` at given shapes against a float64 host product - in-major weights, with and without kept planes, a residual
 * and a multiplier - to place a shape-dependent fault in one projection.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-linear-shapes.js --shapes=576x320x1280,576x320x320
 */
import { linear } from "../../webgl/kernels/linear.js";
import { layerNorm } from "../../webgl/kernels/layer-norm.js";
import { webglExecution } from "./webgl-execution-options.js";
import { createWebGL2Context } from "../../webgl/index.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(_device, args) {
  const execution = webglExecution(createWebGL2Context(), args);
  let seed = 3;
  const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };
  const out = [];
  for (const shape of option(args, "shapes", "576x320x1280,576x320x640,576x320x320").split(",")) {
    const [rows, inC, outC] = shape.split("x").map(Number);
    const x = Float32Array.from({ length: rows * inC }, random);
    const w = Float32Array.from({ length: inC * outC }, random);
    const r = Float32Array.from({ length: rows * outC }, random);
    const m = Float32Array.from({ length: rows * outC }, random);
    const expected = new Float64Array(rows * outC);
    for (let i = 0; i < rows; i += 1) {
      for (let k = 0; k < inC; k += 1) {
        const xv = x[i * inC + k];
        for (let o = 0; o < outC; o += 1) expected[i * outC + o] += xv * w[k * outC + o];
      }
    }
    const xT = execution.upload(x, "x");
    const wT = execution.uploadWeight(w, "w");
    const relative = (got, want) => {
      let e = 0; let s = 0;
      for (let i = 0; i < want.length; i += 1) { e += (got[i] - want[i]) ** 2; s += want[i] ** 2; }
      return Number(Math.sqrt(e / s).toPrecision(3));
    };
    const plain = Float32Array.from(execution.read(linear(execution, {
      x: xT, weight: wT, rows, inChannels: inC, outChannels: outC, weightLayout: "in-major" })));
    const epilogue = Float32Array.from(execution.read(linear(execution, {
      x: xT, weight: wT, rows, inChannels: inC, outChannels: outC, weightLayout: "in-major",
      residual: execution.upload(r, "r"), multiplier: execution.upload(m, "m") })));
    // and a LayerNorm over the input width, unit scale and zero offset, against the host
    const ln = Float32Array.from(execution.read(layerNorm(execution, {
      x: xT, weight: execution.uploadWeight(new Float32Array(inC).fill(1), "s"),
      bias: execution.uploadWeight(new Float32Array(inC), "b"), rows, channels: inC, epsilon: 1e-5, variance: "fast" })));
    const lnWant = new Float64Array(rows * inC);
    for (let i = 0; i < rows; i += 1) {
      let mean = 0; for (let k = 0; k < inC; k += 1) mean += x[i * inC + k]; mean /= inC;
      let v = 0; for (let k = 0; k < inC; k += 1) v += (x[i * inC + k] - mean) ** 2; v /= inC;
      for (let k = 0; k < inC; k += 1) lnWant[i * inC + k] = (x[i * inC + k] - mean) / Math.sqrt(v + 1e-5);
    }
    out.push({
      shape, plain: relative(plain, expected),
      epilogue: relative(epilogue, Float64Array.from(expected, (v, i) => r[i] + v * m[i])),
      layerNorm: relative(ln, lnWant),
    });
  }
  return { rows: out };
}
