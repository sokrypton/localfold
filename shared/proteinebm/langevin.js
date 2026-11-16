/**
 * ProteinEBM's Langevin dynamics at a fixed time level (ProteinEBM_Langevin_Dynamics.ipynb; run_dynamics.py's inner
 * update), as host arithmetic over a score a backend computed:
 *
 *   x <- x - (f(x, t) - g(t)^2 score T / s) dt + g(t) sqrt(dt) eps / s,     f = -b(t) x / 2, s = 0.1 (coordinate scaling)
 *   x <- x - mean(x) + delta,  delta ~ N(0, 1) per axis                    (center_random_augmentation, rotate=False)
 *
 * T is the temperature scaling (the notebook's 0.85714; 1 is the score's own temperature). The recentring keeps
 * the random translation the vendor's code adds: the network is not translation invariant (it embeds the coordinates
 * directly), so the translation is part of the sampler as trained and run. `translate: false` drops it.
 *
 * Self-conditioning: the vendor's scripts pass the coordinates under keys compute_energy never reads, so what they
 * RUN is a zero self-conditioning channel ("none"); the notebook passes the current coordinates ("current"). Both are
 * offered; "none" is the paper's.
 */
import { diffuser, PROTEINEBM } from "./model.js";
import { normalFrom } from "../af3/random.js";

export const LANGEVIN_DEFAULTS = Object.freeze({ t: 0.1, dt: 0.001, tempScaling: 0.85714, selfCondition: "none",
                                                 translate: true });

/** One update: returns the new coordinates (a new array). `noise` N*3 and `shift` 3 are standard normal draws. */
export function langevinStep(pos, score, { t, dt, tempScaling, noise, shift = null }) {
  const s = PROTEINEBM.coordinateScaling;
  const b = diffuser.b(t);
  const g = Math.sqrt(b);
  const n = pos.length / 3;
  const out = new Float32Array(pos.length);
  for (let k = 0; k < pos.length; k += 1) {
    const drift = -0.5 * b * pos[k];
    out[k] = pos[k] - (drift - g * g * score[k] * tempScaling / s) * dt + g * Math.sqrt(dt) * noise[k] / s;
  }
  for (let c = 0; c < 3; c += 1) {
    let m = 0;
    for (let i = 0; i < n; i += 1) m += out[i * 3 + c];
    m /= n;
    const d = shift ? shift[c] : 0;
    for (let i = 0; i < n; i += 1) out[i * 3 + c] = out[i * 3 + c] - m + d;
  }
  return out;
}

/** compute_score's pred_coords: x0 = (score (1 - e^-B) + s x) / e^(-B/2) / s, B the marginal integral at t. */
export function denoised(pos, score, t) {
  const s = PROTEINEBM.coordinateScaling;
  const beta = diffuser.marginal(t);
  const condVar = 1 - Math.exp(-beta);
  const keep = Math.exp(-0.5 * beta);
  return Float32Array.from(pos, (x, k) => (score[k] * condVar + s * x) / keep / s);
}

/**
 * Run the dynamics over `evaluate(pos, sc) -> {energy, score}` (a backend's), yielding a frame per step:
 * {step, pos, energy, rmsd (CA RMSD to the start after superposition-free recentring), denoised}.
 */
export async function* langevin(evaluate, start, options = {}) {
  const o = { ...LANGEVIN_DEFAULTS, steps: 100, seed: 0, ...options };
  const normal = normalFrom(o.seed);
  const draw = (count) => Float64Array.from({ length: count }, () => normal());
  let pos = Float32Array.from(start);
  for (let step = 0; step < o.steps; step += 1) {
    const sc = o.selfCondition === "current" ? pos : null;
    const { energy, score } = await evaluate(pos, sc);
    const next = langevinStep(pos, score, { ...o, noise: draw(pos.length), shift: o.translate ? draw(3) : null });
    yield { step, pos: next, energy, denoised: denoised(pos, score, o.t) };
    pos = next;
  }
}
