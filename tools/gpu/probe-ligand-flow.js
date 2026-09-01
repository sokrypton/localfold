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
 * machine. Import it from the page and call sweepLigandFlow.
 *
 * WHAT IT FOUND, on HEM alone (50 bonds, one seed a cell, recycles 0), as
 * rms/max bond error in angstroms:
 *
 *     sigma0 |    2 steps     4 steps     6 steps     8 steps
 *       2560 | 0.446/0.87  0.267/0.80  0.190/0.64  0.218/0.83
 *        640 | 0.368/1.00  0.198/0.67  0.218/0.79  0.187/0.69
 *        160 | 0.360/1.05  0.241/0.78  0.163/0.60  0.129/0.45
 *         56 | 0.312/0.92  0.199/0.70  0.138/0.54  0.102/0.38
 *         16 | 0.190/0.65  0.208/0.65  0.116/0.49  0.061/0.17
 *
 * AF3's own top of schedule needs 16 steps to reach 0.065/0.15 and does not
 * improve at 32. Starting at sigma0 = 16 gets there in EIGHT. Below 16 the
 * curve is flat - 8 and 4 both give 0.060 - so the knee is at sigma_data, and
 * across three seeds it is steady (0.071, 0.071, 0.059 against 0.218 thrice).
 *
 * 🔴 AND IT IS NOT A DEFAULT. The same sigma0 = 16 destroys a protein: the
 * 59-mer comes out with a CA-CA of 3.21 A instead of 3.88 and a radius of
 * gyration of 6.7 A instead of 12.6 - collapsed into a ball - at pLDDT 43.8
 * against 55.6, and sixteen steps does not rescue it (3.39 A, 8.0 A). The skip
 * weight is sigma_d^2/(sigma^2+sigma_d^2), which is 1/2 at sigma = sigma_d: above
 * it the denoiser mostly ignores the coordinates it is given, below it mostly
 * trusts them. Starting at sigma_d therefore spends every call refining and none
 * exploring - which is exactly right for a lone ligand, whose chemistry is
 * local, and fatal for a chain that has to find a fold first.
 *
 * sigma0 = 160 is the setting that is safe for both: the protein is unchanged
 * (CA-CA 3.88, pLDDT 54.7) and the ligand improves from 0.218 to 0.129 at eight
 * steps.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { foldBatch } from "../../src/af3/fold.js";

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
