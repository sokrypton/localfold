/**
 * The structure module's starting point, against the WGSL one.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-structure-initialize.js [--falsify]
 *
 * Three tensors and a frame, and all four are compared because all four are read
 * later: `single` by nothing yet, `initialRepresentation` by the SIDECHAIN head
 * (AF2's torsion head takes the initial single representation alongside the
 * final one), `activations` by the iteration core, and `affine` by everything
 * downstream.
 *
 * 🔴 THE AFFINE ARM IS A CONVENTION CHECK, NOT AN ARITHMETIC ONE. Seven floats
 * a residue, a quaternion then a translation, and the identity is
 * `(1, 0, 0, 0)` - not `(0, 0, 0, 1)`, which is the other convention in common
 * use and would start every residue rotated half a turn about an axis the data
 * never names. It cannot go red under a perturbation of the input, which it
 * declares.
 */
import { StructureInitializeGpu } from "../../webgpu/af2/structure/initialize.js";
import { setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  releaseStructureInitializeWeights, structureInitialize, uploadStructureInitializeWeights,
} from "../../webgl/af2/structure/initialize.js";
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
  const msaChannels = Number(option(args, "msa-channels", "64"));
  const structureChannels = Number(option(args, "structure-channels", "32"));

  const random = generator(606);
  const noise = (count, spread, centre = 0) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  const msaFirstRow = noise(length * msaChannels, 1.5);
  const weights = {
    singleProjectionWeight: noise(msaChannels * structureChannels, 1 / Math.sqrt(msaChannels)),
    singleProjectionBias: noise(structureChannels, 0.1),
    singleNormScale: noise(structureChannels, 0.2, 1),
    singleNormOffset: noise(structureChannels, 0.1),
    initialProjectionWeight: noise(structureChannels * structureChannels,
      1 / Math.sqrt(structureChannels)),
    initialProjectionBias: noise(structureChannels, 0.1),
  };

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const uploaded = uploadStructureInitializeWeights(execution, weights);
  const row = execution.upload(falsify
    ? Float32Array.from(msaFirstRow,
      (v, i) => (i % msaChannels < msaChannels / 2 ? v * 1.01 + 0.01 : v))
    : msaFirstRow, "msa-row");
  const ours = structureInitialize(execution, {
    msaFirstRow: row, weights: uploaded, length, msaChannels, structureChannels,
  });

  setHalfPrecision(device, false);
  const theirs = await new StructureInitializeGpu(device).run(
    msaFirstRow, length, msaChannels, structureChannels, weights);

  const rows = [];
  const failures = [];
  const arm = (name, got, want, falsifiable = true) => {
    const relative = relativeRms(got, want, name);
    const ok = relative <= bar;
    rows.push({
      arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok,
      ...(falsifiable ? {} : { falsifiable: false }),
    });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };
  arm("single", execution.read(ours.single), theirs.single);
  arm("initialRepresentation",
    execution.read(ours.initialRepresentation), theirs.initialRepresentation);
  arm("activations", execution.read(ours.activations), theirs.activations);
  // The identity frame: a convention, and no input can move it.
  arm("affine identity", execution.read(ours.affine), theirs.affine, false);

  // ...and that it IS the identity, which the comparison alone would not say if
  // both sides agreed on the wrong one.
  const affine = execution.read(ours.affine);
  let wrong = 0;
  for (let residue = 0; residue < length; residue += 1) {
    const base = residue * 7;
    if (affine[base] !== 1) wrong += 1;
    for (let c = 1; c < 7; c += 1) if (affine[base + c] !== 0) wrong += 1;
  }
  if (wrong !== 0) failures.push(`${wrong} affine components are not the identity`);

  releaseStructureInitializeWeights(execution, uploaded);
  for (const tensor of [row, ours.single, ours.initialRepresentation,
    ours.activations, ours.affine]) execution.release(tensor);

  const falsifiable = rows.filter((row2) => row2.falsifiable !== false);
  const red = falsifiable.filter((row2) => !row2.ok).length;
  return {
    shape: { length, msaChannels, structureChannels },
    bar,
    falsify,
    arms: rows,
    liveTensorsAfter: execution.storage.live.size,
    verdict: falsify
      ? (red === falsifiable.length
        ? `ok: all ${falsifiable.length} falsifiable arms go red on a 1% perturbation `
          + `(${rows.length - falsifiable.length} cannot by construction)`
        : `VACUOUS: only ${red} of ${falsifiable.length} falsifiable arms noticed`)
      : (failures.length === 0 && execution.storage.live.size === 0
        ? `ok: ${rows.length} arms, the frame is the identity, no tensor left live`
        : `FAILED: ${failures.join("; ")}`),
  };
}
