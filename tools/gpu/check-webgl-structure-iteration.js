/**
 * One structure-module iteration against the WGSL one - and the frame update is
 * the point of it.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-structure-iteration.js [--falsify]
 *
 * 🔴 THE QUATERNION UPDATE IS THE ERROR THIS GATE EXISTS FOR, because it is the
 * one that degrades a fold rather than breaking it. The update's first three
 * floats are a VECTOR, and the increment is the product of `q` with `(1, v)`
 * less `q` - `(-dot(q.yzw, v), q.x * v + cross(q.yzw, v))` - added and then
 * NORMALISED. Reading those three as a rotation, or skipping the normalise,
 * gives frames that drift off the unit sphere over eight iterations: a structure
 * that gets worse with length rather than one that fails.
 *
 * And the translation is rotated by the OLD frame, not the new one, which is
 * what makes the update a displacement in the frame that produced it.
 *
 * Both outputs are compared, and the quaternion is additionally asserted to be a
 * UNIT - which the comparison alone would not catch if both sides drifted the
 * same way, and which is the property the eight iterations depend on.
 */
import { StructurePostAttentionGpu } from "../../webgpu/af2/structure/iteration.js";
import { setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  releaseIterationWeights, structureIteration, uploadIterationWeights,
} from "../../webgl/af2/structure/iteration.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { generator } from "./af2-synthetic-block.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "1e-6"));
  const length = Number(option(args, "length", "11"));
  const channels = Number(option(args, "channels", "32"));

  const random = generator(4242);
  const noise = (count, spread, centre = 0) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  const activations = noise(length * channels, 1.5);
  const attentionUpdate = noise(length * channels, 1);
  // A plausible frame: a unit quaternion and a translation, which is what the
  // previous iteration would hand over.
  const affine = new Float32Array(length * 7);
  for (let residue = 0; residue < length; residue += 1) {
    const q = [random() * 2 - 1, random() * 2 - 1, random() * 2 - 1, random() * 2 - 1];
    const norm = Math.hypot(...q);
    for (let c = 0; c < 4; c += 1) affine[residue * 7 + c] = q[c] / norm;
    for (let c = 4; c < 7; c += 1) affine[residue * 7 + c] = (random() * 2 - 1) * 2;
  }
  const weights = {
    attentionNormScale: noise(channels, 0.2, 1),
    attentionNormOffset: noise(channels, 0.1),
    transitionWeights: [0, 1, 2].map(
      () => noise(channels * channels, 1 / Math.sqrt(channels))),
    transitionBiases: [0, 1, 2].map(() => noise(channels, 0.1)),
    transitionNormScale: noise(channels, 0.2, 1),
    transitionNormOffset: noise(channels, 0.1),
    affineWeight: noise(channels * 6, 1 / Math.sqrt(channels)),
    affineBias: noise(6, 0.05),
  };

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const uploaded = uploadIterationWeights(execution, weights);
  const ours = structureIteration(execution, {
    activations: execution.upload(falsify
      ? Float32Array.from(activations, (v, i) => (i % channels < channels / 2 ? v * 1.01 + 0.01 : v))
      : activations, "activations"),
    update: execution.upload(attentionUpdate, "update"),
    affine: execution.upload(affine, "affine"),
    weights: uploaded, length, channels,
  });

  setHalfPrecision(device, false);
  const theirs = await new StructurePostAttentionGpu(device).run({
    activations, attentionUpdate, affine, length, channels, weights,
  });

  const rows = [];
  const failures = [];
  const arm = (name, got, want) => {
    const relative = relativeRms(got, want, name);
    const ok = relative <= bar;
    rows.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };
  arm("activations", execution.read(ours.activations), theirs.activations);
  arm("affine", execution.read(ours.affine), theirs.affine);

  // 🔴 AND THE QUATERNION MUST BE A UNIT, which a comparison would not catch if
  // both sides drifted the same way. Eight iterations depend on it.
  const nextAffine = execution.read(ours.affine);
  let offSphere = 0;
  for (let residue = 0; residue < length; residue += 1) {
    const base = residue * 7;
    const norm = Math.hypot(nextAffine[base], nextAffine[base + 1],
      nextAffine[base + 2], nextAffine[base + 3]);
    if (!(Math.abs(norm - 1) < 1e-5)) offSphere += 1;
  }
  if (offSphere !== 0) failures.push(`${offSphere} quaternions are not unit`);

  releaseIterationWeights(execution, uploaded);
  const red = rows.filter((row) => !row.ok).length;
  return {
    shape: { length, channels },
    bar,
    falsify,
    arms: rows,
    verdict: falsify
      ? (red === rows.length
        ? `ok: all ${rows.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${rows.length} arms noticed a 1% perturbation`)
      : (failures.length === 0
        ? `ok: ${rows.length} arms, every quaternion a unit`
        : `FAILED: ${failures.join("; ")}`),
  };
}
