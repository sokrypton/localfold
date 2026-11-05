/**
 * Atom coordinates against the WGSL geometry stage.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-geometry.js [--falsify]
 *
 * 🔴 THIS STAGE IS ALMOST ALL CONVENTION AND ALMOST NO ARITHMETIC, which is what
 * makes a cross-check worth more here than anywhere else in the port. Five
 * choices decide whether the output is a structure or a plausible cloud: a
 * default frame is a 4x4 ROW-MAJOR matrix whose translation is the fourth
 * column; the quaternion is `(w, x, y, z)` and is used UNNORMALISED; the
 * backbone translation carries a factor of **10**; only groups 5, 6 and 7 chain;
 * and the torsion mixes columns 1 and 2 with `(sine, cosine)` in that order.
 * Every one of them produces finite coordinates when wrong.
 *
 * The arms compare atom14 AND atom37, because the second is a gather through two
 * index tables and an error there is invisible in the first.
 *
 * 🔴 AND `position_scale` IS SWEPT, BECAUSE IT IS A PER-MODEL FACT AND NOT A
 * CONSTANT. Ten for the monomer checkpoints and twenty for multimer - the
 * weights were trained against their own value - so both are compared, against
 * `webgpu/af2/multimer/geometry.js`, which generates its shader per scale for
 * the same reason. A monomer scale on multimer weights halves every coordinate
 * from the origin: a structure, just not the model's, and nothing downstream
 * would raise. `--position-scale=` runs one value alone.
 *
 * 🔴 AND THE AFFINE IS DELIBERATELY NOT A UNIT QUATERNION. The reference does
 * not normalise it, so a port that does would agree on normalised input and
 * differ on anything else - and the iteration core upstream does not guarantee
 * one either. A random quaternion is the arm that can see that.
 */
import { AtomGeometryGpu } from "../../webgpu/af2/multimer/geometry.js";
import { atomGeometry } from "../../webgl/af2/structure/geometry.js";
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
  const length = Number(option(args, "length", "13"));
  const random = generator(2718);
  const noise = (count, spread, centre = 0) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  const integers = (count, limit) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = Math.floor(random() * limit);
    return out;
  };

  // A quaternion that is NOT a unit, on purpose - see the header.
  const affine = noise(length * 7, 0.8, 0.2);
  // Torsions as normalised (sine, cosine) pairs, which is what the head emits.
  const angles = new Float32Array(length * 14);
  for (let pair = 0; pair < length * 7; pair += 1) {
    const theta = random() * Math.PI * 2;
    angles[pair * 2] = Math.sin(theta);
    angles[pair * 2 + 1] = Math.cos(theta);
  }
  const aatype = integers(length, 21);
  const tables = {
    defaultFrames: noise(21 * 8 * 16, 1),
    atom14ToGroup: integers(21 * 14, 8),
    atom14Positions: noise(21 * 14 * 3, 2),
    atom14Mask: integers(21 * 14, 2),
  };
  const atom37ToAtom14 = integers(length * 37, 14);
  const atom37Mask = integers(length * 37, 2);

  // Both of AF2's scales unless one is named. The multimer class is the
  // reference for both: it takes `positionScale` where the monomer's has 10.0
  // written into its shader, and at 10 the two generate the same kernel.
  const scaleArg = option(args, "position-scale", null);
  const scales = scaleArg === null ? [10, 20] : [Number(scaleArg)];

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const upload = (values, label) => execution.upload(values, label);
  const inputs = {
    affine: upload(falsify
      ? Float32Array.from(affine, (v, i) => (i % 7 < 3 ? v * 1.01 + 0.01 : v))
      : affine, "affine"),
    angles: upload(angles, "angles"),
    aatype: upload(aatype, "aatype"),
    atom37ToAtom14: upload(atom37ToAtom14, "map"),
    atom37Mask: upload(atom37Mask, "mask37"),
    tables: {
      defaultFrames: upload(tables.defaultFrames, "default-frames"),
      atom14ToGroup: upload(tables.atom14ToGroup, "atom-groups"),
      atom14Positions: upload(tables.atom14Positions, "literature"),
      atom14Mask: upload(tables.atom14Mask, "atom-mask"),
    },
  };

  const rows = [];
  const failures = [];
  const arm = (name, got, want) => {
    const relative = relativeRms(got, want, name);
    const ok = relative <= bar;
    rows.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };
  /** @type {Record<string, Float32Array>} */
  const byScale = {};
  let ours;
  for (const positionScale of scales) {
    if (ours !== undefined) {
      execution.release(ours.atom14);
      execution.release(ours.atom37);
      execution.release(ours.frames);
    }
    ours = atomGeometry(execution, { ...inputs, length, positionScale });
    const theirs = await new AtomGeometryGpu(device).run({
      affine, angles, aatype, atom37ToAtom14, atom37Mask, length, tables, positionScale,
    });
    arm(`atom14 at scale ${positionScale}`, execution.read(ours.atom14), theirs.atom14);
    arm(`atom37 at scale ${positionScale}`, execution.read(ours.atom37), theirs.atom37);
    byScale[positionScale] = Float32Array.from(execution.read(ours.atom37));
  }

  // 🔴 AND THE SCALE MUST MOVE THE ANSWER, or both sides could be ignoring it.
  // The two differ only in the backbone translation, so the coordinates shift
  // and the relRMS between them is large - measured, it is of order one.
  const scaleMoves = scales.length < 2 ? undefined : Number(
    relativeRms(byScale[scales[0]], byScale[scales[1]], "scale.moves").toPrecision(3));
  if (scaleMoves !== undefined && scaleMoves < 1e-3) {
    failures.push(`position_scale is INERT: 10 and 20 agree to ${scaleMoves}`);
  }

  // Every coordinate finite, and every masked-out slot exactly zero - two
  // readings a structure has to satisfy whatever the tables say.
  const atom37 = execution.read(ours.atom37);
  let nonFinite = 0;
  let leaked = 0;
  for (let slot = 0; slot < length * 37; slot += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      const value = atom37[slot * 3 + axis];
      if (!Number.isFinite(value)) nonFinite += 1;
      if (atom37Mask[slot] === 0 && value !== 0) leaked += 1;
    }
  }
  if (nonFinite !== 0) failures.push(`${nonFinite} atom37 coordinates are not finite`);
  if (leaked !== 0) failures.push(`${leaked} masked-out atom37 coordinates are non-zero`);

  const red = rows.filter((row) => !row.ok).length;
  return {
    shape: { length },
    bar,
    falsify,
    arms: rows,
    positionScales: scales,
    scaleMoves,
    draws: execution.stats().draws,
    verdict: falsify
      ? (red === rows.length
        ? `ok: all ${rows.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${rows.length} arms noticed a 1% perturbation`)
      : (failures.length === 0
        ? `ok: ${rows.length} arms, every coordinate finite, every masked slot zero`
        : `FAILED: ${failures.join("; ")}`),
  };
}
