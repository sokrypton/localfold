/**
 * Pocket and contact constraints as the two families trained on them read them: boltz2's contact conditioning and
 * chai-1's two distance restraints. af3-any-model's `constraints` input (docs/input.md there, model_features.py
 * _boltz2_constraints and _chai1_constraints), built on the FINISHED batch - after every featurisation rewrite - from
 * each token's chain, residue and atom names, as af3-any-model's `_constraint_tokens` reads its token-atom layout.
 *
 * A constraint here is RESOLVED: its chains are asym numbers in the featuriser's order (polymer chains first, then
 * each ligand, from 0 - the numbering `bondedAtomPairs` already reaches featuriseProtein in), not the job's letters.
 *
 *   { kind: "pocket", binder: <asym>, contacts: [{asym, residue, atom?}, ...], maxDistance }
 *   { kind: "contact", token1: {asym, residue, atom?}, token2: {...}, maxDistance }
 *
 * 🔴 NOTHING IS DROPPED. chai-lab logs a restraint it cannot place and then drops EVERY restraint of the job; boltz
 * raises. Both raise here, as af3-any-model does, with the constraint's index.
 */

/** boltz's contact classes (featurizerv2.py), the one-hot's order. */
export const CONTACT_CLASSES = Object.freeze({
  UNSPECIFIED: 0, UNSELECTED: 1, "POCKET>BINDER": 2, "BINDER>POCKET": 3, CONTACT: 4,
});

/** Which families read constraints, and as what. */
export const CONSTRAINT_FAMILIES = Object.freeze({ boltz2: "boltz2", chai1: "chai1" });

const PROTEIN_LETTERS = "ARNDCQEGHILKMFPSTWYVX";

/** Per token: [asym (0-based), residue, Set of atom names]; asym -1 for padding. */
function constraintTokens(batch) {
  const tokens = batch.asymId.length;
  const dense = batch.refMask.length / tokens;
  const out = [];
  for (let t = 0; t < tokens; t += 1) {
    const names = new Set();
    for (let slot = 0; slot < dense; slot += 1) {
      const flat = t * dense + slot;
      if (batch.refMask[flat] === 0) continue;
      let name = "";
      for (let c = 0; c < 4; c += 1) {
        const v = batch.refAtomNameChars[flat * 4 + c];
        if (v > 0 && v < 64) name += String.fromCharCode(v + 32);
      }
      names.add(name.trim());
    }
    out.push([names.size === 0 ? -1 : batch.asymId[t] - 1, batch.residueIndex[t], names]);
  }
  return out;
}

const polymerChains = (batch) => (batch.chains ?? batch.chainLengths).length;
const describe = (end) => `[${end.asym}, ${end.residue}${end.atom === undefined ? "" : `, ${end.atom}`}]`;

/** boltz's token_spec_to_ids: a polymer residue is every token of it, a ligand atom its one token. */
function boltzTokens(tokens, batch, end, where) {
  const ligand = end.asym >= polymerChains(batch);
  const hasAtom = end.atom !== undefined && end.atom !== null && end.atom !== "";
  if (ligand && !hasAtom) throw new Error(`${where} ${describe(end)}: a ligand token is named by its atom`);
  if (!ligand && hasAtom) {
    throw new Error(`${where} ${describe(end)}: boltz2 conditions polymer residues, not atoms`);
  }
  const hit = [];
  tokens.forEach(([asym, residue, names], t) => {
    if (asym === end.asym && residue === end.residue && (!ligand || names.has(end.atom))) hit.push(t);
  });
  if (hit.length === 0) throw new Error(`${where} ${describe(end)} names no token of the input`);
  return hit;
}

function checkCommon(constraint, where) {
  const d = constraint.maxDistance;
  if (typeof d !== "number" || !(d > 0)) {
    throw new Error(`${where}: maxDistance must be a positive number of Angstrom`);
  }
  if (constraint.force === true) {
    throw new Error(`${where}: force (steering the sampler toward the constraint) is not implemented;`
      + " drop it to condition the network only");
  }
}

/**
 * boltz-2's contact_conditioning and contact_threshold (featurizerv2.py process_token_features, inference branch):
 * every pair UNSELECTED; a pocket marks [binder, pocket] BINDER>POCKET and [pocket, binder] POCKET>BINDER at its
 * distance, for every token of each pocket residue; then a contact marks CONTACT both ways between the FIRST token
 * each end names. Later writes win. All UNSELECTED becomes all UNSPECIFIED.
 */
function boltz2Features(batch, constraints) {
  const tokens = constraintTokens(batch);
  const n = tokens.length;
  const classes = new Int32Array(n * n).fill(CONTACT_CLASSES.UNSELECTED);
  const threshold = new Float32Array(n * n);
  const set = (i, j, cls, d) => { classes[i * n + j] = cls; threshold[i * n + j] = d; };
  const pockets = constraints.map((c, k) => [k, c]).filter(([, c]) => c.kind === "pocket");
  const contacts = constraints.map((c, k) => [k, c]).filter(([, c]) => c.kind === "contact");
  for (const [k, c] of pockets) {
    const binder = [];
    tokens.forEach(([asym], t) => { if (asym === c.binder) binder.push(t); });
    if (binder.length === 0) throw new Error(`constraints[${k}] pocket: binder chain ${c.binder} has no token`);
    for (const end of c.contacts) {
      for (const t of boltzTokens(tokens, batch, end, `constraints[${k}] pocket`)) {
        for (const b of binder) {
          set(b, t, CONTACT_CLASSES["BINDER>POCKET"], c.maxDistance);
          set(t, b, CONTACT_CLASSES["POCKET>BINDER"], c.maxDistance);
        }
      }
    }
  }
  for (const [k, c] of contacts) {
    const i = boltzTokens(tokens, batch, c.token1, `constraints[${k}] token1`)[0];
    const j = boltzTokens(tokens, batch, c.token2, `constraints[${k}] token2`)[0];
    set(i, j, CONTACT_CLASSES.CONTACT, c.maxDistance);
    set(j, i, CONTACT_CLASSES.CONTACT, c.maxDistance);
  }
  if (classes.every((cls) => cls === CONTACT_CLASSES.UNSELECTED)) classes.fill(CONTACT_CLASSES.UNSPECIFIED);
  return { contactConditioning: classes, contactThreshold: threshold };
}

/**
 * chai-1's TokenDistanceRestraint and TokenPairPocketRestraint (generators/token_dist_restraint.py,
 * token_pair_pocket_restraint.py): distances, -1 where none. A contact sets [token1, token2] only - NOT symmetric -,
 * a pocket [pocket residue, every token of the binder chain]. Each end a standard protein residue of exactly one
 * token; a ligand only as a pocket's binder.
 */
function chai1Features(batch, constraints) {
  const tokens = constraintTokens(batch);
  const n = tokens.length;
  const kinds = batch.chainKinds ?? [];
  const sequences = batch.chains ?? [];
  const residue = (end, where) => {
    if (end.atom !== undefined && end.atom !== null && end.atom !== "") {
      throw new Error(`${where} ${describe(end)}: chai1 conditions residues, not atoms`);
    }
    if (end.asym >= polymerChains(batch) || (kinds[end.asym] ?? "protein") !== "protein") {
      throw new Error(`${where} ${describe(end)}: chai1 restrains protein residues only (a ligand only as a binder)`);
    }
    const letter = (sequences[end.asym] ?? "")[end.residue - 1];
    if (letter === undefined || !PROTEIN_LETTERS.includes(letter)) {
      throw new Error(`${where} ${describe(end)}: chai1 restrains standard residues only`);
    }
    const hit = [];
    tokens.forEach(([asym, res], t) => { if (asym === end.asym && res === end.residue) hit.push(t); });
    if (hit.length !== 1) {
      throw new Error(`${where} ${describe(end)}: chai1 wants a residue of exactly one token, found ${hit.length}`);
    }
    return hit[0];
  };
  const contact = new Float32Array(n * n).fill(-1);
  const pocket = new Float32Array(n * n).fill(-1);
  constraints.forEach((c, k) => {
    if (c.kind === "contact") {
      const i = residue(c.token1, `constraints[${k}] token1`);
      const j = residue(c.token2, `constraints[${k}] token2`);
      contact[i * n + j] = c.maxDistance;
    } else {
      const binder = [];
      tokens.forEach(([asym], t) => { if (asym === c.binder) binder.push(t); });
      if (binder.length === 0) throw new Error(`constraints[${k}] pocket: binder chain ${c.binder} has no token`);
      for (const end of c.contacts) {
        const t = residue(end, `constraints[${k}] pocket`);
        for (const b of binder) pocket[t * n + b] = c.maxDistance;
      }
    }
  });
  return { tokenDistanceRestraint: contact, tokenPairPocketRestraint: pocket };
}

/**
 * The constraint features `family` reads, written onto `batch`. A family that was not trained on constraints
 * refuses them rather than folding as if there were none.
 *
 * @param {object} batch featuriseProtein's batch
 * @param {Array} constraints resolved constraints (see the header)
 * @param {?string} family a CONSTRAINT_FAMILIES value, or null for a model trained with neither
 * @param {string} [model] the model's name, for the refusal
 */
export function applyConstraints(batch, constraints, family, model = "this model") {
  if (constraints === undefined || constraints === null || constraints.length === 0) return batch;
  if (family !== "boltz2" && family !== "chai1") {
    throw new Error(`${model} was not trained with pocket or contact conditioning and cannot honour the job's`
      + " constraints; boltz2 and chai1 take pocket and contact constraints");
  }
  constraints.forEach((c, k) => {
    if (c.kind !== "pocket" && c.kind !== "contact") throw new Error(`constraints[${k}]: kind ${c.kind}`);
    checkCommon(c, `constraints[${k}]`);
    if (c.kind === "pocket" && (!Array.isArray(c.contacts) || c.contacts.length === 0)) {
      throw new Error(`constraints[${k}]: a pocket needs at least one contact residue`);
    }
  });
  Object.assign(batch, family === "boltz2" ? boltz2Features(batch, constraints) : chai1Features(batch, constraints));
  return batch;
}

/**
 * The per-pair term a constraint adds, as a small table and an index into it: every pair's term depends only on its
 * (class, distance) - boltz2 - or (contact distance, pocket distance) - chai-1 -, so the distinct ones are a handful
 * of vectors and the device adds `table[index[pair]]`. Built on the host from the site's own weights; a 128-wide
 * linear over a few rows, which no profile here would see.
 *
 * boltz2 (af3-any-model evoformer.boltz2_contact_conditioning): UNSPECIFIED and UNSELECTED bypass the encoder for a
 * learned constant each; a restraint class runs it, `encoder([onehot(class - 2), tn, cos(2 pi (tn W_f + b_f))])` with
 * tn = (d - 4) / 16. The term REPLACES the unspecified constant every boltz2 pair carries today, so the table's
 * entries are whole terms, not differences.
 *
 * @param {Int32Array} classes n x n
 * @param {Float32Array} threshold n x n
 * @param {{fourierWeights, fourierBias, encoderWeights, encoderBias, unspecified, unselected}} w one site's weights:
 *   fourier (1, C) and (C), encoder (3 + 1 + C, C) row-major [in][out] and (C), the two constants (C)
 * @returns {{index: Float32Array, table: Float32Array, entries: number}}
 */
export function boltz2ContactTable(classes, threshold, w) {
  const C = w.unspecified.length;
  const keys = new Map();
  const index = new Float32Array(classes.length);
  const rows = [];
  for (let p = 0; p < classes.length; p += 1) {
    const cls = classes[p];
    const d = cls >= 2 ? threshold[p] : 0;
    const key = `${cls}:${d}`;
    let at = keys.get(key);
    if (at === undefined) { at = rows.length; keys.set(key, at); rows.push([cls, d]); }
    index[p] = at;
  }
  const table = new Float32Array(rows.length * C);
  rows.forEach(([cls, d], r) => {
    const out = table.subarray(r * C, (r + 1) * C);
    if (cls < 0) return;                                          // padding: a zero one-hot, no term
    if (cls === CONTACT_CLASSES.UNSPECIFIED) { out.set(w.unspecified); return; }
    if (cls === CONTACT_CLASSES.UNSELECTED) { out.set(w.unselected); return; }
    const tn = (d - 4) / 16;
    const x = new Float64Array(3 + 1 + C);
    x[cls - 2] = 1;
    x[3] = tn;
    for (let k = 0; k < C; k += 1) x[4 + k] = Math.cos(2 * Math.PI * (tn * w.fourierWeights[k] + w.fourierBias[k]));
    for (let c = 0; c < C; c += 1) {
      let sum = w.encoderBias[c];
      for (let k = 0; k < x.length; k += 1) if (x[k] !== 0) sum += x[k] * w.encoderWeights[k * C + c];
      out[c] = sum;
    }
  });
  return { index, table, entries: rows.length };
}

/**
 * chai-1's restraint columns (af3-any-model evoformer.chai1_restraint_feature) through `chai1_restraint_trunk`
 * (14, C): each restraint an RBF over 6 radii, exp(-min(((r - d) / sigma)^2, 16)) zeroed where it clamps, times
 * (1 - mask), then the mask column minus its no-restraint value - so a pair with neither restraint adds exactly zero
 * and is table row 0. Contact radii 6..30 A, pocket 6..20 A; sigma the spacing.
 *
 * @param {Float32Array} contact n x n, -1 where none
 * @param {Float32Array} pocket n x n, -1 where none
 * @param {Float32Array} weights (14, C) row-major
 */
export function chai1RestraintTable(contact, pocket, weights) {
  const C = weights.length / 14;
  const rbf = (d, lo, hi) => {
    const out = new Float64Array(7);
    if (d === -1) return out;                                     // mask 1: rbf zeroed, mask column 1 - 1 = 0
    const sigma = (hi - lo) / 5;
    for (let k = 0; k < 6; k += 1) {
      const x = Math.min(((lo + k * sigma - d) / sigma) ** 2, 16);
      out[k] = x === 16 ? 0 : Math.exp(-x);
    }
    out[6] = -1;                                                  // mask 0, minus the no-restraint 1
    return out;
  };
  const keys = new Map([["-1:-1", 0]]);
  const rows = [[-1, -1]];
  const index = new Float32Array(contact.length);
  for (let p = 0; p < contact.length; p += 1) {
    const key = `${contact[p]}:${pocket[p]}`;
    let at = keys.get(key);
    if (at === undefined) { at = rows.length; keys.set(key, at); rows.push([contact[p], pocket[p]]); }
    index[p] = at;
  }
  const table = new Float32Array(rows.length * C);
  rows.forEach(([dc, dp], r) => {
    if (r === 0) return;
    const f = [...rbf(dc, 6, 30), ...rbf(dp, 6, 20)];
    for (let c = 0; c < C; c += 1) {
      let sum = 0;
      for (let k = 0; k < 14; k += 1) sum += f[k] * weights[k * C + c];
      table[r * C + c] = sum;
    }
  });
  return { index, table, entries: rows.length };
}
