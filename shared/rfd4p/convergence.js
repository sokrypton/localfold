/**
 * Early stopping for the flow-matching sampler: stop once the clean estimate `x1` has stopped moving and the
 * designed sequence has stopped changing.
 *
 * Measured on uncond_short / ppi_pdl1 (60 steps, 3 seeds each), the backbone x1 is within 0.15-0.22 Å of the final
 * sample by step 40, but 2-9% of residues still change until steps 45-50, so a backbone-only test stops too early.
 * Requiring both a still backbone and an unchanged sequence for `patience` consecutive steps fires around step 48-52.
 *
 * Stopping at step s and taking x1(s) as the sample is the same as one final Euler step from t(s) straight to t = 1.
 */
import { superimposeCoords3D } from './output.js';

export const DEFAULT_EARLY_STOP = {
  /** consecutive converged steps required */
  patience: 4,
  /** max Kabsch Cα RMSD (Å) between consecutive x1 estimates to count as converged */
  caTolA: 0.05,
  /** never stop before this fraction of the schedule */
  minFrac: 0.5,
};

/** Kabsch Cα RMSD (Å) between two `[n * 3]` nm coordinate arrays, over tokens with `mask[i]` set (all if null). */
export function caRmsdA(a, b, mask = null) {
  const n = a.length / 3;
  let m = 0;
  for (let i = 0; i < n; i += 1) if (!mask || mask[i]) m += 1;
  if (m < 3) return 0;
  const fa = new Float32Array(m * 3);
  const fb = new Float32Array(m * 3);
  for (let i = 0, k = 0; i < n; i += 1) {
    if (mask && !mask[i]) continue;
    for (let d = 0; d < 3; d += 1) {
      fa[k * 3 + d] = a[i * 3 + d];
      fb[k * 3 + d] = b[i * 3 + d];
    }
    k += 1;
  }
  const { aligned } = superimposeCoords3D(fa, fb);
  let se = 0;
  for (let i = 0; i < m * 3; i += 1) se += (aligned[i] - fa[i]) ** 2;
  return Math.sqrt(se / m) * 10;
}

/**
 * Tracks convergence across sampler steps. Call `observe` once per step with that step's x1 Cα (nm) and the
 * sequence the decoder would use; it returns the step's metrics and whether to stop after it.
 */
export class ConvergenceMonitor {
  /**
   * @param {number} nsteps
   * @param {Partial<typeof DEFAULT_EARLY_STOP>} [cfg]
   * @param {ArrayLike<number>|null} [mask] tokens to compare (designed / emitted ones)
   */
  constructor(nsteps, cfg = {}, mask = null) {
    this.cfg = { ...DEFAULT_EARLY_STOP, ...cfg };
    this.nsteps = nsteps;
    this.mask = mask;
    this.prevCa = null;
    this.prevSeq = null;
    this.streak = 0;
  }

  /** Whether a stop is possible right after `step` (so the caller can prepare its outputs, e.g. insertion logits). */
  mayStopAt(step) {
    return step + 1 >= Math.ceil(this.cfg.minFrac * this.nsteps) && this.streak + 1 >= this.cfg.patience &&
      step < this.nsteps - 1;
  }

  /**
   * @param {number} step
   * @param {Float32Array} caNm x1 Cα, `[n * 3]` nm
   * @param {ArrayLike<number>} seq designed residue type per token
   * @returns {{caDeltaA: number|null, seqChanged: number|null, streak: number, stop: boolean}}
   */
  observe(step, caNm, seq) {
    let caDeltaA = null;
    let seqChanged = null;
    if (this.prevCa) {
      caDeltaA = caRmsdA(this.prevCa, caNm, this.mask);
      seqChanged = 0;
      for (let i = 0; i < seq.length; i += 1) {
        if ((!this.mask || this.mask[i]) && seq[i] !== this.prevSeq[i]) seqChanged += 1;
      }
      this.streak = caDeltaA <= this.cfg.caTolA && seqChanged === 0 ? this.streak + 1 : 0;
    }
    this.prevCa = Float32Array.from(caNm);
    this.prevSeq = Int32Array.from(seq);
    const stop = step + 1 >= Math.ceil(this.cfg.minFrac * this.nsteps) && this.streak >= this.cfg.patience &&
      step < this.nsteps - 1;
    return { caDeltaA, seqChanged, streak: this.streak, stop };
  }
}
