/**
 * Triangle multiplication in WebGL2 against the CPU reference AND against the
 * WGSL kernel, in one page - the first whole AlphaFold operation in this
 * backend.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-triangle.js [--falsify]
 *
 * 🔴 TWO REFERENCES, BECAUSE NEITHER COVERS WHAT THE OTHER DOES.
 * `triangleMultiplicationOutgoingReference` is f64 JavaScript and is the
 * operation's definition, but it exists for the OUTGOING direction only - so
 * the incoming kernel, which differs by two index expressions and is exactly
 * the kind of thing that returns a plausible tensor of the right shape and
 * magnitude, would have nothing to be checked against. The WGSL kernel has
 * both, and `tools/gpu-chrome.mjs` hands this module a WebGPU device, so both
 * backends run on one input in one process. That is also the comparison
 * docs/DEVELOPING.md's "computing the same numbers" has always asserted and
 * nothing has ever run.
 *
 * The WGSL side is driven at its own defaults - `precision: "f32"` and
 * `variance: "two-pass"`, which is AF2's formula - because those are what this
 * backend's LayerNorm implements. Comparing against its "fast" variance or its
 * f16 path would be comparing two different operations and would need its own
 * bar.
 *
 * 🔴 NINE DRAWS AGAINST A FUSED KERNEL, so the bar is accumulation and not
 * equality. Each of LayerNorm, the four projections, the gates, the mask, the
 * contraction and the output gate rounds to f32 at its own draw where the WGSL
 * side keeps more of the chain in registers; a reordering at f32 is ~1e-7 a
 * step, so 1e-5 over the whole operation has room and still separates it from a
 * wrong index, which reads 1e0.
 */
import { TriangleMultiplicationIncomingGpu, TriangleMultiplicationOutgoingGpu }
  from "../../webgpu/index.js";
import { triangleMultiplicationOutgoingReference }
  from "../../webgpu/kernels/triangle/cpu-reference.js";
import { createDeterministicTriangleInput } from "../../webgpu/testing/deterministic-input.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import {
  releaseTriangleWeights, triangleMultiplication, uploadTriangleWeights,
} from "../../webgl/kernels/triangle-multiplication.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

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

  const shapes = [
    { length: 8, cZ: 32, cHidden: 16 },
    { length: 12, cZ: 64, cHidden: 32 },
    { length: 16, cZ: 128, cHidden: 128 },
    { length: 24, cZ: 32, cHidden: 64 },
  ];

  for (const shape of shapes) {
    const input = createDeterministicTriangleInput(shape, shape.length * 31 + shape.cZ);
    const label = `L=${shape.length} cZ=${shape.cZ} cH=${shape.cHidden}`;
    // 🔴 THE PERTURBATION GOES TO THE WEBGL SIDE ONLY. Both arms below read the
    // same `input`, so skewing it would move every reference with the kernel and
    // leave the differences untouched - the vacuous control this repository has
    // now written twice.
    const z = falsify
      ? Float32Array.from(input.z, (v, i) => (i === (input.z.length >> 1) ? v * 1.01 + 0.01 : v))
      : input.z;

    const weights = uploadTriangleWeights(execution, input.weights);
    const zTensor = execution.upload(z, "z");
    const maskTensor = execution.upload(input.mask, "mask");

    const ours = {};
    for (const direction of ["outgoing", "incoming"]) {
      const output = triangleMultiplication(execution, {
        z: zTensor, mask: maskTensor, weights, shape, direction, epsilon: input.epsilon ?? 1e-5,
      });
      ours[direction] = execution.read(output).slice();
      execution.release(output);
    }

    arm(`outgoing vs CPU reference ${label}`,
      ours.outgoing, triangleMultiplicationOutgoingReference(input));

    for (const [direction, Kernel] of [
      ["outgoing", TriangleMultiplicationOutgoingGpu],
      ["incoming", TriangleMultiplicationIncomingGpu],
    ]) {
      const wgsl = await new Kernel(device).run(input);
      arm(`${direction} vs WGSL ${label}`, ours[direction], wgsl);
    }

    releaseTriangleWeights(execution, weights);
    execution.release(zTensor);
    execution.release(maskTensor);
  }

  const red = rows.filter((row) => !row.ok).length;
  const stats = execution.stats();
  return {
    falsify,
    bar,
    arms: rows,
    // Every intermediate is released as its last reader finishes, so a run that
    // leaks shows up here as live tensors rather than as a slow machine.
    liveTensorsAfter: stats.live,
    peakMegabytes: Number((stats.peakBytes / 2 ** 20).toFixed(1)),
    draws: stats.draws,
    programs: stats.programs,
    verdict: falsify
      ? (red === rows.length
        ? `ok: all ${rows.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${rows.length} arms noticed a 1% perturbation`)
      : (failures.length === 0 && stats.live === 0
        ? `ok: ${rows.length} arms, no tensor left live`
        : `FAILED: ${[...failures, ...(stats.live ? [`${stats.live} tensors leaked`] : [])].join("; ")}`),
  };
}
