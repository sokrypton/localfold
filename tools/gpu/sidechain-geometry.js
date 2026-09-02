/**
 * How close is a predicted structure to the chemistry it was given?
 *
 * Shared by probe-sidechains.js and probe-af3-trunk-sample.js, because the two
 * only mean anything side by side: one measures what our whole pipeline
 * produces, the other what it produces from AF3's OWN trunk, and a difference
 * in how they were scored would be indistinguishable from a difference in what
 * they scored.
 *
 * 🔴 PAIRS ARE MATCHED BY ATOM NAME, NOT BY SLOT INDEX. If the question is
 * whether atoms are mislabelled, then indexing both sides by position assumes
 * the answer: two swapped labels would compare a bond against the ideal of the
 * bond it was swapped with, and could look perfect.
 */
import { REFERENCE_CONFORMERS } from "../../src/af3/reference-conformers.js";
import { atomName } from "../../src/af3/fold.js";

/** Every aromatic type, and the ring atoms whose closure is the question. */
export const RINGS = {
  F: ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
  Y: ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
  H: ["CG", "ND1", "CE1", "NE2", "CD2"],
  W: ["CG", "CD1", "NE1", "CE2", "CD2"],
  P: ["N", "CA", "CB", "CG", "CD"],
};

const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/**
 * Score one structure against the reference conformers' rigid tables.
 *
 * @param {{sequence: string, dense: number, refAtomNameChars: ArrayLike<number>,
 *          predDenseAtomMask: ArrayLike<number>}} batch
 * @param {ArrayLike<number>} positions  tokens * dense * 3, angstroms
 */
export function sidechainGeometry(batch, positions) {
  const { dense, sequence } = batch;
  const ratios = [];
  const perPair = new Map();
  const perType = new Map();
  const rings = [];

  for (let token = 0; token < sequence.length; token += 1) {
    const code = sequence[token];
    const conformer = REFERENCE_CONFORMERS[code];
    if (conformer === undefined) continue;
    const slotByName = new Map();
    for (let atom = 0; atom < dense; atom += 1) {
      if (!batch.predDenseAtomMask[token * dense + atom]) continue;
      slotByName.set(atomName(batch.refAtomNameChars, token * dense + atom), atom);
    }
    const coordOf = (atom) => {
      const slot = (token * dense + atom) * 3;
      return [positions[slot], positions[slot + 1], positions[slot + 2]];
    };
    const idealName = new Map(conformer.internal.map(([index, name]) => [index, name]));
    for (const [i, j, ideal] of conformer.rigid) {
      const a = slotByName.get(idealName.get(i));
      const b = slotByName.get(idealName.get(j));
      if (a === undefined || b === undefined) continue;
      const predicted = distance(coordOf(a), coordOf(b));
      // 🔴 THE RATIO, NOT ONLY THE ERROR. A uniform scale problem and a
      // placement problem both raise the error; only the ratio tells them apart.
      ratios.push({ kind: ideal < 1.8 ? "bond" : "1-3", ratio: predicted / ideal });
      const error = Math.abs(predicted - ideal);
      const key = `${code} ${idealName.get(i)}-${idealName.get(j)}`;
      if (!perPair.has(key)) perPair.set(key, []);
      perPair.get(key).push(error);
      if (!perType.has(code)) perType.set(code, []);
      perType.get(code).push(error);
    }
    const ring = RINGS[code];
    if (ring !== undefined && ring.every((name) => slotByName.has(name))) {
      rings.push({
        residue: `${code}${token + 1}`,
        bonds: ring.map((name, index) => {
          const next = ring[(index + 1) % ring.length];
          return {
            bond: `${name}-${next}`,
            length: Number(distance(coordOf(slotByName.get(name)),
                                    coordOf(slotByName.get(next))).toFixed(3)),
          };
        }),
      });
    }
  }

  const summarise = (errors) => ({
    n: errors.length,
    mean: Number((errors.reduce((a, b) => a + b, 0) / errors.length).toFixed(3)),
    max: Number(Math.max(...errors).toFixed(3)),
  });
  return {
    scale: ["bond", "1-3"].map((kind) => {
      const values = ratios.filter((r) => r.kind === kind).map((r) => r.ratio)
        .sort((a, b) => a - b);
      return {
        kind, n: values.length,
        median: Number(values[Math.floor(values.length / 2)].toFixed(3)),
        mean: Number((values.reduce((a, b) => a + b, 0) / values.length).toFixed(3)),
      };
    }),
    byType: Object.fromEntries([...perType].map(([k, v]) => [k, summarise(v)])),
    worstPairs: [...perPair]
      .map(([key, errors]) => ({ pair: key, ...summarise(errors) }))
      .sort((a, b) => b.max - a.max).slice(0, 12),
    rings,
  };
}
