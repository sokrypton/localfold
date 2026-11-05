/**
 * The sidechain torsion head against the WGSL one.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-sidechain.js [--falsify]
 *
 * 🔴 THE TWO INPUTS ARE THE SAME SHAPE AND ENTER THROUGH DIFFERENT WEIGHTS,
 * which is the error this gate exists for. The head reads the FINAL single
 * representation and the INITIAL one, projects each with its own weights and
 * adds them; passing the same tensor twice, or the two the other way round,
 * gives a plausible set of torsions. The arms use independent noise for the two
 * so a swap is visible.
 *
 * 🔴 AND BOTH OUTPUTS ARE COMPARED. `unnormalizedAngles` is what AF2's loss
 * reads and the page reports, and `angles` is the normalised pair - so an error
 * in the normalisation shows in one and not the other, and an error upstream
 * shows in both.
 *
 * The normalisation guards at 1e-12 rather than at zero, so a torsion whose two
 * components are both zero comes out (0, 0) rather than NaN. One arm sets a
 * residue's raw angle weights to zero to reach that branch, because a random
 * input never will.
 */
import { SidechainAnglesGpu } from "../../webgpu/af2/structure/sidechain.js";
import { setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  releaseSidechainWeights, sidechainAngles, uploadSidechainWeights,
} from "../../webgl/af2/structure/sidechain.js";
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
  const length = Number(option(args, "length", "9"));
  const inputChannels = Number(option(args, "input-channels", "32"));
  const hiddenChannels = Number(option(args, "hidden", "64"));

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const rows = [];
  const failures = [];
  // 🔴 THE ZEROED ARMS CANNOT GO RED AND SAY SO. With the angle projection
  // zeroed, the output is zero whatever the input is - which is the point of
  // that arm, since it is the only way to reach the 1e-12 guard - so no
  // perturbation can move it. Declaring that is better than tightening a bar it
  // would still not meet, or dropping the arm that covers the degenerate branch.
  const arm = (name, got, want, falsifiable = true) => {
    const relative = relativeRms(got, want, name);
    const ok = relative <= bar;
    rows.push({
      arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok,
      ...(falsifiable ? {} : { falsifiable: false }),
    });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };

  setHalfPrecision(device, false);

  for (const zeroAngles of [false, true]) {
    const random = generator(zeroAngles ? 909 : 707);
    const noise = (count, spread, centre = 0) => {
      const out = new Float32Array(count);
      for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
      return out;
    };
    const activations = noise(length * inputChannels, 1.5);
    // Independent noise, so feeding the same tensor twice would be visible.
    const initialActivations = noise(length * inputChannels, 1.5);
    const weights = {
      inputWeight: noise(inputChannels * hiddenChannels, 1 / Math.sqrt(inputChannels)),
      inputBias: noise(hiddenChannels, 0.1),
      initialInputWeight: noise(inputChannels * hiddenChannels, 1 / Math.sqrt(inputChannels)),
      initialInputBias: noise(hiddenChannels, 0.1),
      residual1Weights: [
        noise(hiddenChannels * hiddenChannels, 1 / Math.sqrt(hiddenChannels)),
        noise(hiddenChannels * hiddenChannels, 1 / Math.sqrt(hiddenChannels)),
      ],
      residual1Biases: [noise(hiddenChannels, 0.1), noise(hiddenChannels, 0.1)],
      residual2Weights: [
        noise(hiddenChannels * hiddenChannels, 1 / Math.sqrt(hiddenChannels)),
        noise(hiddenChannels * hiddenChannels, 1 / Math.sqrt(hiddenChannels)),
      ],
      residual2Biases: [noise(hiddenChannels, 0.1), noise(hiddenChannels, 0.1)],
      // ...zeroed for the degenerate arm, which is the only way to reach the
      // 1e-12 guard: a random projection never lands both components on zero.
      angleWeight: zeroAngles
        ? new Float32Array(hiddenChannels * 14)
        : noise(hiddenChannels * 14, 1 / Math.sqrt(hiddenChannels)),
      angleBias: zeroAngles ? new Float32Array(14) : noise(14, 0.1),
    };

    const uploaded = uploadSidechainWeights(execution, weights);
    const final = execution.upload(falsify
      ? Float32Array.from(activations,
        (v, i) => (i % inputChannels < inputChannels / 2 ? v * 1.01 + 0.01 : v))
      : activations, "final");
    const initial = execution.upload(initialActivations, "initial");
    const ours = sidechainAngles(execution, {
      activations: final, initialActivations: initial, weights: uploaded,
      length, inputChannels, hiddenChannels,
    });
    const theirs = await new SidechainAnglesGpu(device).run(
      activations, initialActivations, length, inputChannels, hiddenChannels, weights);

    const label = zeroAngles ? " (zeroed angle projection)" : "";
    arm(`unnormalizedAngles${label}`,
      execution.read(ours.unnormalizedAngles), theirs.unnormalizedAngles, !zeroAngles);
    arm(`angles${label}`, execution.read(ours.angles), theirs.angles, !zeroAngles);

    // Every normalised pair is a unit vector, or exactly zero where the raw pair
    // was - which is what the 1e-12 guard is for, and is not NaN either way.
    const angles = execution.read(ours.angles);
    let bad = 0;
    for (let a = 0; a < length * 7; a += 1) {
      const x = angles[a * 2];
      const y = angles[a * 2 + 1];
      const norm = Math.hypot(x, y);
      if (!Number.isFinite(norm)) bad += 1;
      else if (Math.abs(norm - 1) > 1e-5 && norm > 1e-6) bad += 1;
    }
    if (bad !== 0) failures.push(`${bad} normalised angles are neither unit nor zero${label}`);

    releaseSidechainWeights(execution, uploaded);
    for (const tensor of [final, initial, ours.angles, ours.unnormalizedAngles]) {
      execution.release(tensor);
    }
  }

  const falsifiable = rows.filter((row) => row.falsifiable !== false);
  const red = falsifiable.filter((row) => !row.ok).length;
  return {
    shape: { length, inputChannels, hiddenChannels },
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
        ? `ok: ${rows.length} arms, every angle unit or zero, no tensor left live`
        : `FAILED: ${failures.join("; ")}`),
  };
}
