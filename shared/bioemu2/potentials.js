/**
 * BioEmu2's steering potentials: energies and their gradients, on the host.
 *
 * A transcription of af3-any-model's network/bioemu2_steering.py energy() and
 * its parameter schedules (bioemu2 @7846769's potentials.get_default_potentials
 * and denoising/steering), with the GRADIENT written out by hand where JAX
 * takes jax.grad - the samplers in sampler.js descend it. Held to that port's
 * own energies and jax.grad by tools/check-bioemu2-potentials.js.
 *
 * Coordinates are one particle at a time, Float64Array(nPad * 3), angstrom.
 * `tables` is steeringTables() then hostTables() for the active specs.
 */
import { geometryBounds } from "./steering-tables.js";

export const SIGMA_DATA = 16.0;
export const SIGMA_MIN = 0.0004;
export const SIGMA_MAX = 160.0;
export const RHO = 7;

export const exponential = (start, end, alpha) => (t) => (alpha !== 0
  ? start + ((end - start) * (Math.exp(alpha * t) - 1)) / (Math.exp(alpha) - 1)
  : start + (end - start) * t);
export const piecewise = (thresholds, values) => (t) => {
  let i = 0;
  while (i < thresholds.length && t > thresholds[i]) i += 1;
  return values[i];
};
export const linear = (startT, endT, startVal, endVal) => (t) => (t > startT || t < endT
  ? 0.0 : startVal - ((startVal - endVal) * (startT - t)) / (startT - endT));

/** get_default_potentials, as specs: [type, {parameter: value or schedule}, {constructor argument: value}]. */
export const DEFAULT_SPECS = Object.freeze([
  ["SymmetricChainCOMPotential", { guidance_interval: 4, guidance_weight: 0.5, resampling_weight: 0.5,
                                   buffer: exponential(1.0, 5.0, -2.0) }, {}],
  ["InterChainVDWOverlapPotential", { guidance_interval: 5, guidance_weight: piecewise([0.4], [0.125, 0.0]),
                                      resampling_weight: piecewise([0.6], [0.01, 0.0]), buffer: 0.225 }, {}],
  ["PoseBustersPotential", { guidance_interval: 1, guidance_weight: 0.10, resampling_weight: 0.1,
                             bond_buffer: 0.20, angle_buffer: 0.20, clash_buffer: 0.15 }, {}],
  ["ChiralAtomPotential", { guidance_interval: 1, guidance_weight: 0.20, resampling_weight: 1.0, buffer: 0.52360 }, {}],
  ["StereoBondPotential", { guidance_interval: 1, guidance_weight: 0.10, resampling_weight: 1.0, buffer: 0.52360 }, {}],
  ["PlanarBondPotential", { guidance_interval: 1, guidance_weight: 0.10, resampling_weight: 1.0, buffer: 0.26180 }, {}],
  ["IntraChainVDWOverlapPotential", { guidance_interval: 5, guidance_weight: piecewise([0.4], [0.125, 0.0]),
                                      resampling_weight: piecewise([0.6], [0.01, 0.0]), buffer: 0.225 }, {}],
  ["PolymerLocalClashPotential", { guidance_interval: 1, guidance_weight: piecewise([0.4], [0.125, 0.0]),
                                   resampling_weight: 0.1, minimum_distance: 2.0 }, {}],
  ["BondLengthPotential", { guidance_interval: 1, guidance_weight: 0.05, resampling_weight: 0.1 },
   { sigma_tolerance: 10.0 }],
  ["BondAnglePotential", { guidance_interval: 1, guidance_weight: 0.05, resampling_weight: 0.1 },
   { sigma_tolerance: 10.0 }],
]);
export const TYPES = Object.freeze(DEFAULT_SPECS.map(([kind]) => kind));
export const PARAMS = Object.freeze(["buffer", "bond_buffer", "angle_buffer", "clash_buffer", "minimum_distance"]);
const REQUIRED = {
  SymmetricChainCOMPotential: ["buffer"], InterChainVDWOverlapPotential: ["buffer"],
  IntraChainVDWOverlapPotential: ["buffer"], PoseBustersPotential: ["bond_buffer", "angle_buffer", "clash_buffer"],
  ChiralAtomPotential: ["buffer"], StereoBondPotential: ["buffer"], PlanarBondPotential: ["buffer"],
  PolymerLocalClashPotential: ["minimum_distance"],
};
const SCHEDULES = {
  ExponentialInterpolation: (a) => exponential(a.start, a.end, a.alpha),
  PiecewiseStepFunction: (a) => piecewise(a.thresholds, a.values),
  LinearInterpolation: (a) => linear(a.start_t, a.end_t, a.start_val, a.end_val),
};

/**
 * Potentials from a BioEmu2 `--potentials` file, already parsed (YAML or JSON):
 * {potentials: [{_target_: ...potentials.<Type>, sigma_tolerance?, parameters:
 * {name: value or {_target_: ...<Schedule>, ...}}}]}. Replaces the defaults.
 * Only BioEmu2's own potential and schedule classes can be named.
 */
export function potentialSpecs(raw) {
  if (!raw || typeof raw !== "object" || !("potentials" in raw)) throw new Error("no potentials: list");
  if (!raw.potentials.length) throw new Error("the potentials list is empty; use the edm sampler");
  return raw.potentials.map((entry) => {
    const { _target_: target, parameters = {}, ...rest } = entry;
    const kind = target.split(".").pop();
    if (!TYPES.includes(kind)) throw new Error(`${kind} is not one of BioEmu2's ported potentials ${TYPES}`);
    const params = {};
    for (const [k, v] of Object.entries(parameters ?? {})) {
      if (v && typeof v === "object") {
        const { _target_: schedule, ...args } = v;
        const name = schedule.split(".").pop();
        if (!(name in SCHEDULES)) throw new Error(`unknown schedule ${name}`);
        params[k] = SCHEDULES[name](args);
      } else {
        params[k] = v;
      }
    }
    for (const need of REQUIRED[kind] ?? []) if (!(need in params)) throw new Error(`${kind} needs parameters[${need}]`);
    if ((kind === "BondLengthPotential" || kind === "BondAnglePotential") && !("sigma_tolerance" in rest)) {
      throw new Error(`${kind} needs sigma_tolerance`);
    }
    const extra = Object.keys(rest).filter((k) => k !== "sigma_tolerance");
    if (extra.length) throw new Error(`${kind} takes no ${extra}`);
    return [kind, params, rest];
  });
}

const resolve = (v, t) => (typeof v === "function" ? v(t) : v);

/** Every spec's resolved parameters at normalised time t (absent ones NaN), and the sampler weights. */
export function stepParameters(t, specs = DEFAULT_SPECS) {
  const f = Math.fround;
  const out = {
    guidance_weight: specs.map(([, p]) => f(resolve(p.guidance_weight ?? 0.0, t))),
    resampling_weight: specs.map(([, p]) => f(resolve(p.resampling_weight ?? 1.0, t))),
  };
  for (const k of PARAMS) out[k] = specs.map(([, p]) => (k in p ? f(resolve(p[k], t)) : NaN));
  return out;
}

export const guidanceInterval = (t, specs = DEFAULT_SPECS) =>
  specs.map(([, p]) => Math.trunc(resolve(p.guidance_interval ?? 1, t)));

/**
 * host_tables: the planar impropers, the chain-pair filter, and each
 * Bond/Angle spec's bounds at its own sigma_tolerance.
 */
export function hostTables(tables, specs = DEFAULT_SPECS) {
  const t = { ...tables };
  const planar = tables.planar_bond_index;                 // [6][K] or []
  const K = planar.length ? planar[0].length : 0;
  // pb[:, imp].swapaxes(0, 1).reshape(4, -1), imp = [[1, 2, 3, 0], [4, 5, 0, 3]].T: row r, column 2k + m is
  // planar atom imp[r][m] of bond k - the two impropers of a bond side by side
  const improper = [[1, 4], [2, 5], [3, 0], [0, 3]];
  t.planar_improper_index = [0, 1, 2, 3].map((r) => {
    const out = new Int32Array(2 * K);
    for (let k = 0; k < K; k += 1) for (let m = 0; m < 2; m += 1) out[2 * k + m] = planar[improper[r][m]][k];
    return out;
  });
  const sym = tables.symmetric_chain_index;
  const big = Array.from(tables.chain_sizes, (s) => s > 1);
  const keep = sym[0] ? Array.from(sym[0], (a, k) => big[a] && big[sym[1][k]]) : [];
  t.symmetric_chain_pairs = sym[0] ? [Int32Array.from(sym[0].filter((_, k) => keep[k])),
                                      Int32Array.from(sym[1].filter((_, k) => keep[k]))] : [new Int32Array(0), new Int32Array(0)];
  specs.forEach(([kind, , args], i) => {
    if (kind === "BondLengthPotential") {
      const b = tables.bond_ref.map((r) => geometryBounds(r, args.sigma_tolerance));
      t[`bond_lower.${i}`] = Float32Array.from(b, (x) => x[0]);
      t[`bond_upper.${i}`] = Float32Array.from(b, (x) => x[1]);
    } else if (kind === "BondAnglePotential") {
      const b = tables.angle_ref.map((r) => geometryBounds(r, args.sigma_tolerance, true));
      t[`angle_lower.${i}`] = Float32Array.from(b, (x) => x[0]);
      t[`angle_upper.${i}`] = Float32Array.from(b, (x) => x[1]);
    }
  });
  for (const k of ["bond_lower", "bond_upper", "angle_lower", "angle_upper", "bond_ref", "angle_ref"]) delete t[k];
  return t;
}

// --- geometry with gradients ---------------------------------------------------------------

const sub = (c, i, j) => [c[i * 3] - c[j * 3], c[i * 3 + 1] - c[j * 3 + 1], c[i * 3 + 2] - c[j * 3 + 2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const addTo = (g, i, v, s = 1) => { if (!g) return; g[i * 3] += v[0] * s; g[i * 3 + 1] += v[1] * s; g[i * 3 + 2] += v[2] * s; };
/** d flat_bottom / dv: -1 below, +1 above, 0 inside (relu's zero subgradient). */
const flatBottom = (v, lo, hi) => [Math.max(lo - v, 0) + Math.max(v - hi, 0), v < lo ? -1 : v > hi ? 1 : 0];

/** |c_i - c_j| (torch.linalg.norm: zero subgradient at zero); gradient scaled by `g` into grad. */
function distance(c, i, j, grad, g) {
  const r = sub(c, i, j);
  const s = dot(r, r);
  const d = s > 0 ? Math.sqrt(s) : 0;
  if (grad && g !== 0 && d > 0) { addTo(grad, i, r, g / d); addTo(grad, j, r, -g / d); }
  return d;
}

/** The dihedral of (i, j, k, l) as the vendor's _dihedral, and its gradient times g. */
function dihedral(c, i, j, k, l, grad, gFn) {
  const rij = sub(c, i, j);
  const rkj = sub(c, k, j);
  const rkl = sub(c, k, l);
  const n1 = cross(rij, rkj);
  const n2 = cross(rkj, rkl);
  const m = cross(n1, n2);
  const L = Math.sqrt(dot(rkj, rkj));
  const sin = dot(rkj, m) / L;
  const cos = dot(n1, n2);
  const theta = Math.atan2(sin, cos);
  const g = gFn ? gFn(theta) : 0;
  if (grad && g !== 0) {
    const denom = sin * sin + cos * cos;
    const gs = (g * cos) / denom;
    const gc = (-g * sin) / denom;
    let gn1 = scale(n2, gc);
    let gn2 = scale(n1, gc);
    let grkj = [m[0] * gs / L, m[1] * gs / L, m[2] * gs / L];
    const proj = (gs * dot(rkj, m)) / (L * L * L);
    grkj = [grkj[0] - proj * rkj[0], grkj[1] - proj * rkj[1], grkj[2] - proj * rkj[2]];
    const gm = scale(rkj, gs / L);
    const a = cross(n2, gm);
    const b = cross(gm, n1);
    gn1 = [gn1[0] + a[0], gn1[1] + a[1], gn1[2] + a[2]];
    gn2 = [gn2[0] + b[0], gn2[1] + b[1], gn2[2] + b[2]];
    const grij = cross(rkj, gn1);
    const t1 = cross(gn1, rij);
    const t2 = cross(rkl, gn2);
    grkj = [grkj[0] + t1[0] + t2[0], grkj[1] + t1[1] + t2[1], grkj[2] + t1[2] + t2[2]];
    const grkl = cross(gn2, rkj);
    addTo(grad, i, grij); addTo(grad, j, grij, -1);
    addTo(grad, k, grkj); addTo(grad, j, grkj, -1);
    addTo(grad, k, grkl); addTo(grad, l, grkl, -1);
  }
  return theta;
}

/** The angle at the third index (vendor _angle, eps 1e-6), and its gradient times g. */
function angle(c, i, j, centre, grad, gFn, eps = 1e-6) {
  const a = sub(c, i, centre);
  const b = sub(c, j, centre);
  const la = Math.sqrt(dot(a, a));
  const lb = Math.sqrt(dot(b, b));
  const na = Math.max(la, eps);
  const nb = Math.max(lb, eps);
  const ua = scale(a, 1 / na);
  const ub = scale(b, 1 / nb);
  const raw = dot(ua, ub);
  const clipped = Math.min(Math.max(raw, -1 + eps), 1 - eps);
  const theta = Math.acos(clipped);
  const g = gFn ? gFn(theta) : 0;
  if (grad && g !== 0 && raw > -1 + eps && raw < 1 - eps) {
    const gd = -g / Math.sqrt(1 - clipped * clipped);
    const back = (u, other, len, n) => {
      const gu = scale(other, gd);
      if (len > eps) { const p = dot(u, gu); return scale([gu[0] - u[0] * p, gu[1] - u[1] * p, gu[2] - u[2] * p], 1 / n); }
      return scale(gu, 1 / n);
    };
    const ga = back(ua, ub, la, na);
    const gb = back(ub, ua, lb, nb);
    addTo(grad, i, ga); addTo(grad, centre, ga, -1);
    addTo(grad, j, gb); addTo(grad, centre, gb, -1);
  }
  return theta;
}

/** relu(minimum - sqrt(|d|^2 + 1e-12)) and its gradient times g. */
function overlapPair(c, i, j, minimum, grad, g) {
  const r = sub(c, i, j);
  const d = Math.sqrt(dot(r, r) + 1e-12);
  const e = minimum - d;
  if (e <= 0) return 0;
  if (grad && g !== 0) { addTo(grad, i, r, -g / d); addTo(grad, j, r, g / d); }
  return e;
}

/**
 * Spec `i` of type `kind`: one particle's energy at the step's parameters `p`
 * (this spec's resolved values), with its gradient times `weight` added into
 * `grad` when given.
 */
export function energy(kind, i, c, t, p, grad = null, weight = 1) {
  const nPad = t.atom_pad_mask.length;
  const radius = t.atom_vdw_radius;
  const chain = t.atom_chain_id;
  const f = Math.fround;
  let total = 0;
  if (kind === "SymmetricChainCOMPotential") {
    const [ia, ib] = t.symmetric_chain_pairs;
    if (!ia.length) return 0;
    const nChain = t.chain_sizes.length;
    const com = new Float64Array(nChain * 3);
    const count = new Float64Array(nChain);
    for (let a = 0; a < nPad; a += 1) {
      if (!t.atom_pad_mask[a]) continue;
      count[chain[a]] += 1;
      for (let x = 0; x < 3; x += 1) com[chain[a] * 3 + x] += c[a * 3 + x];
    }
    for (let k = 0; k < nChain; k += 1) for (let x = 0; x < 3; x += 1) com[k * 3 + x] /= Math.max(count[k], 1);
    const gcom = new Float64Array(nChain * 3);
    for (let k = 0; k < ia.length; k += 1) {
      const d = distance(com, ia[k], ib[k], gcom, 0);
      if (p.buffer - d > 0) {
        total += p.buffer - d;
        if (grad && d > 0) distance(com, ia[k], ib[k], gcom, -weight);
      }
    }
    if (grad) {
      for (let a = 0; a < nPad; a += 1) {
        if (!t.atom_pad_mask[a]) continue;
        const k = chain[a];
        for (let x = 0; x < 3; x += 1) grad[a * 3 + x] += gcom[k * 3 + x] / Math.max(count[k], 1);
      }
    }
    return total;
  }
  if (kind === "InterChainVDWOverlapPotential" || kind === "IntraChainVDWOverlapPotential") {
    const same = kind === "IntraChainVDWOverlapPotential";
    const eligible = same ? t.polymer_steric_atom_mask : t.interchain_steric_atom_mask;
    const keys = new Set(same ? t.polymer_bonded_pair_key : t.interchain_bonded_pair_key);
    const s = 1.0 - p.buffer;
    for (let a = 0; a < nPad; a += 1) {
      if (!eligible[a]) continue;
      for (let b = a + 1; b < nPad; b += 1) {
        if (!eligible[b] || (chain[a] === chain[b]) !== same || keys.has(a * nPad + b)) continue;
        total += overlapPair(c, a, b, (radius[a] + radius[b]) * s, grad, weight);
      }
    }
    return total;
  }
  if (kind === "PoseBustersPotential") {
    const idx = t.rdkit_bounds_index;
    if (!idx.length || !idx[0].length) return 0;
    const bb = f(p.bond_buffer);
    const ab = f(p.angle_buffer);
    const cb = f(p.clash_buffer);
    const mb = Math.min(bb, ab);
    for (let k = 0; k < idx[0].length; k += 1) {
      const bond = !!t.rdkit_bounds_bond_mask[k];
      const ang = !!t.rdkit_bounds_angle_mask[k];
      const lo0 = f(t.rdkit_lower_bounds[k]);
      const hi0 = f(t.rdkit_upper_bounds[k]);
      const buffer = bond && !ang ? bb : !bond && ang ? ab : bond && ang ? mb : cb;
      const lo = f(lo0 * f(1 - buffer));
      const hi = bond || ang ? f(hi0 * f(1 + buffer)) : Infinity;
      const d = distance(c, idx[0][k], idx[1][k], null, 0);
      const [e, de] = flatBottom(d, lo, hi);
      total += e;
      if (grad && de) distance(c, idx[0][k], idx[1][k], grad, de * weight);
    }
    return total;
  }
  if (kind === "ChiralAtomPotential" || kind === "StereoBondPotential" || kind === "PlanarBondPotential") {
    const idx = kind === "ChiralAtomPotential" ? t.chiral_atom_index
      : kind === "StereoBondPotential" ? t.stereo_bond_index : t.planar_improper_index;
    if (!idx.length || !idx[0].length) return 0;
    const b = f(p.buffer);
    for (let k = 0; k < idx[0].length; k += 1) {
      let lo;
      let hi;
      let absolute = false;
      if (kind === "ChiralAtomPotential") {
        const o = !!t.chiral_atom_orientations[k];
        lo = o ? b : -Infinity;
        hi = o ? Infinity : -b;
      } else if (kind === "StereoBondPotential") {
        const o = !!t.stereo_bond_orientations[k];
        lo = o ? f(Math.fround(Math.PI) - b) : -Infinity;
        hi = o ? Infinity : b;
        absolute = true;
      } else {
        lo = -Infinity;
        hi = b;
        absolute = true;
      }
      let energyHere = 0;
      dihedral(c, idx[0][k], idx[1][k], idx[2][k], idx[3][k], grad, (theta) => {
        const v = absolute ? Math.abs(theta) : theta;
        const [e, de] = flatBottom(v, lo, hi);
        energyHere = e;
        return de * weight * (absolute ? Math.sign(theta) : 1);
      });
      total += energyHere;
    }
    return total;
  }
  if (kind === "PolymerLocalClashPotential") {
    const idx = t.polymer_local_clash_index;
    if (!idx[0].length) return 0;
    for (let k = 0; k < idx[0].length; k += 1) total += overlapPair(c, idx[0][k], idx[1][k], p.minimum_distance, grad, weight);
    return total;
  }
  if (kind === "BondLengthPotential") {
    const idx = t.bond_index;
    const lo = t[`bond_lower.${i}`];
    const hi = t[`bond_upper.${i}`];
    for (let k = 0; k < idx[0].length; k += 1) {
      const d = distance(c, idx[0][k], idx[1][k], null, 0);
      const [e, de] = flatBottom(d, lo[k], hi[k]);
      total += e;
      if (grad && de) distance(c, idx[0][k], idx[1][k], grad, de * weight);
    }
    return total;
  }
  if (kind === "BondAnglePotential") {
    const idx = t.angle_index;
    const lo = t[`angle_lower.${i}`];
    const hi = t[`angle_upper.${i}`];
    for (let k = 0; k < idx[0].length; k += 1) {
      let e = 0;
      angle(c, idx[0][k], idx[1][k], idx[2][k], grad, (theta) => { const [v, de] = flatBottom(theta, lo[k], hi[k]); e = v; return de * weight; });
      total += e;
    }
    return total;
  }
  throw new Error(kind);
}

/** Every spec's energy of one particle; `weights` (per spec) scale the gradient added into `grad`. */
export function energies(c, t, p, specs = DEFAULT_SPECS, grad = null, weights = null) {
  return specs.map(([kind], i) => {
    const w = weights ? weights[i] : 1;
    const at = Object.fromEntries(PARAMS.map((k) => [k, p[k][i]]));
    return energy(kind, i, c, t, at, grad && w !== 0 ? grad : null, w);
  });
}
