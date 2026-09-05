/**
 * Where does a ligand's geometry actually resolve, and how few steps after it?
 *
 * The flow conditions the denoiser on a noise level that descends 2560 A to
 * about 0.2 A, and nothing is injected after the first draw - so the schedule is
 * a statement about how much the denoiser should trust its input, not about how
 * much noise is in it. This asks the empirical question that follows: starting
 * the walk at sigma0, how many calls does it take before the ligand's BONDS are
 * right, and which sigma0 needs the fewest?
 *
 * 🔴 BOND LENGTHS, NOT COORDINATES. A ligand is placed by the denoiser alone -
 * it has no backbone and no reference conformer holding it together - and AF3
 * resamples its torsions per instance anyway, so its coordinates are not
 * comparable to anything. Bond lengths are: they are the part of the chemistry
 * the dictionary fixes, and "all bonds ideal" is a check that the component came
 * out as itself rather than as 43 atoms in roughly the right place.
 *
 * 🔴 AND THEY ARE MEASURED AGAINST THE DICTIONARY'S OWN IDEAL CONFORMER, which
 * is the same file the featuriser read. A bond that is 0.05 A long here is 0.05
 * A off the chemistry AF3 was given, not off some other reference.
 *
 * Runs in a browser, because the native Dawn binding does not load on this
 * machine:
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-ligand-flow.js \
 *       --ligand=HEM --sigmas=2560,640,160,56,16 --steps=2,4,6,8,16
 *
 * WHAT IT FOUND, on HEM alone (50 bonds, one seed a cell, recycles 0), as
 * rms/max bond error in angstroms, AFTER the diffusion head's weight-name fix
 * (see docs/AF3.md, "Fixed: the side chains were compressed"):
 *
 *     sigma0 |  2 steps    4 steps    6 steps    8 steps   16 steps
 *       2560 | 0.401/.92  0.270/.78  0.198/.66  0.202/.72  0.168/.61
 *        640 | 0.324/.90  0.252/.73  0.210/.70  0.187/.65  0.069/.20
 *        160 | 0.311/.86  0.229/.73  0.188/.65  0.160/.59  0.047/.15
 *         56 | 0.342/.93  0.220/.74  0.164/.64  0.153/.60  0.043/.11
 *         16 | 0.233/.71  0.185/.66  0.115/.50  0.044/.12  0.043/.11
 *
 * The knee is still at sigma_data: sigma0 = 16 reaches in EIGHT steps what
 * AF3's own top of schedule does not reach in sixteen. What the fix changed is
 * the top of the table - 2560 used to get to 0.065 by sixteen steps and now
 * plateaus at 0.168, so the SPREAD between a high and a low start is wider
 * against the correct weights, not narrower.
 *
 * 🔴 AND IT IS STILL NOT A DEFAULT. probe-sigma0.js scores the same starts
 * against crystal structures: sigma0 = 16 costs 1QYS 0.26 A of backbone RMSD
 * and a whole seed's worth of variance (1.10 A mean, one seed at 1.70), because
 * the skip weight sigma_d^2/(sigma^2+sigma_d^2) is 1/2 at sigma = sigma_d -
 * above it the denoiser mostly ignores the coordinates it is given, below it
 * mostly trusts them. Starting at sigma_d spends every call refining and none
 * exploring: exactly right for a lone ligand, whose chemistry is local, and
 * wrong for a chain that has to find a fold first.
 *
 * sigma0 = 160 is the setting that is defensible for both: 0.047 against 0.168
 * on the ligand at sixteen steps, for 0.025 A of backbone RMSD on 1QYS.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { foldBatch } from "../../src/af3/fold.js";
import { ccdUrl, parseCcdComponent } from "../../src/af3/ccd-component.js";
import { confidenceWeights, openAf3Store, trunkWeights } from "../../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";

/** sigmaMax is in units of sigmaData, so the walk starts at sigmaData*sigmaMax. */
const SIGMA_DATA = 16;

/**
 * Bond-length error of one ligand against its dictionary ideal.
 *
 * @param {object} batch from featuriseProtein, carrying ligandSpans
 * @param {Float32Array} positions the dense atom layout the sampler returned
 * @param {{code: string, atoms: object[], bonds: object[]}[]} components
 * @returns {{rms: number, max: number, worst: string, bonds: number}}
 */
export function ligandBondError(batch, positions, components) {
  const { dense } = batch;
  let sum = 0;
  let count = 0;
  let max = 0;
  let worst = "";
  const spans = batch.ligandSpans ?? [];
  for (let index = 0; index < spans.length; index += 1) {
    const span = spans[index];
    const component = components[index];
    const at = (atom) => {
      // One token per heavy atom, and the atom sits in slot zero of it.
      const slot = (span.from + atom) * dense;
      return [positions[slot * 3], positions[slot * 3 + 1], positions[slot * 3 + 2]];
    };
    const ideal = (atom) => {
      const source = component.atoms[atom];
      return [source.x, source.y, source.z];
    };
    const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    for (const bond of component.bonds) {
      const predicted = distance(at(bond.from), at(bond.to));
      const reference = distance(ideal(bond.from), ideal(bond.to));
      const error = Math.abs(predicted - reference);
      sum += error * error;
      count += 1;
      if (error > max) {
        max = error;
        worst = `${component.atoms[bond.from].name}-${component.atoms[bond.to].name}`;
      }
    }
  }
  return { rms: count === 0 ? NaN : Math.sqrt(sum / count), max, worst, bonds: count };
}

/**
 * Fold one ligand repeatedly, varying where the walk starts and how long it is.
 *
 * @param {{device: GPUDevice, weights: object,
 *          components: object[], sequence?: string,
 *          startSigmas: number[], stepCounts: number[],
 *          seed?: number, recycles?: number,
 *          onResult?: (row: object) => void}} options
 *   `startSigmas` are absolute angstroms; AF3's own top of schedule is 2560.
 * @returns {Promise<object[]>} one row per (sigma0, steps) pair
 */
export async function sweepLigandFlow(options) {
  const { device, weights, components, startSigmas, stepCounts } = options;
  const sequence = options.sequence ?? "";
  const batch = featuriseProtein(sequence, { ligands: components });
  const rows = [];
  for (const sigma0 of startSigmas) {
    for (const steps of stepCounts) {
      const started = performance.now();
      const result = await foldBatch(device, batch, weights, {
        mode: "flow",
        steps,
        recycles: options.recycles ?? 0,
        seed: options.seed ?? 20260831,
        // 🔴 THE SAME SEED FOR EVERY CELL. The flow's only randomness is its
        // first draw, so holding the seed fixed makes the grid a comparison of
        // schedules rather than of draws - which at one sample per cell is the
        // difference between a measurement and a lottery.
        schedule: { sigmaMax: sigma0 / SIGMA_DATA },
      });
      const error = ligandBondError(batch, result.positions, components);
      const row = {
        sigma0, steps, ...error,
        plddt: result.meanPlddt,
        seconds: (performance.now() - started) / 1000,
      };
      rows.push(row);
      options.onResult?.(row);
    }
  }
  return rows;
}

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const codes = option(args, "ligand", "HEM").split(",");
  const components = [];
  for (const code of codes) {
    const response = await fetch(ccdUrl(code));
    if (!response.ok) throw new Error(`could not fetch ${code}: ${response.status}`);
    components.push(parseCcdComponent(await response.text()));
  }

  const store = await openAf3Store(option(args, "model", "/model-af3-full-f32/manifest.json"));
  const weights = {
    trunk: await trunkWeights(store), diffusion: await diffusionWeights(store),
    confidence: await confidenceWeights(store), atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };

  return sweepLigandFlow({
    device, weights, components,
    sequence: option(args, "sequence", ""),
    startSigmas: option(args, "sigmas", "2560,640,160,56,16").split(",").map(Number),
    stepCounts: option(args, "steps", "2,4,6,8,16").split(",").map(Number),
    seed: Number(option(args, "seed", "20260831")),
    recycles: Number(option(args, "recycles", "0")),
  });
}
