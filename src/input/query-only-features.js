import { residueIndexWithChainBreaks } from "./chains.js";

const RESTYPES = "ARNDCQEGHILKMFPSTWYV";
const RESTYPE_INDEX = new Map([...RESTYPES].map((residue, index) => [residue, index]));

function randomGenerator(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}

function maskedCodes(aatype, recycle, seed) {
  const random = randomGenerator((seed ^ Math.imul(recycle + 1, 0x9e3779b9)) >>> 0);
  const result = aatype.slice();
  for (let residue = 0; residue < result.length; residue += 1) {
    if (random() >= 0.15) continue;
    const draw = random();
    // AlphaFold's masked-MSA mixture: 70% mask, 10% profile, 10% same,
    // and 10% uniform over the 20 standard residues.
    if (draw < 0.7) result[residue] = 22;
    else if (draw >= 0.9) result[residue] = Math.floor(random() * 20);
  }
  return result;
}

/**
 * Construct model-1 query-only tensors directly from an amino-acid sequence.
 * @param {string} sequenceValue concatenated chain sequences
 * @param {{atom37ToAtom14: Float32Array, atom37Mask: Float32Array}} tables
 * @param {{recycles?: number, randomSeed?: number, maskedMsaCodes?: readonly Float32Array[],
 *   chainLengths?: readonly number[], tolerance?: number}} [options]
 */
export function makeQueryOnlyFeatures(
  sequenceValue,
  tables,
  options = {},
) {
  const sequence = sequenceValue.trim().toUpperCase();
  if (sequence.length === 0 || [...sequence].some((residue) => residue !== "X" && !RESTYPE_INDEX.has(residue))) {
    throw new Error("query must contain only the 20 standard amino acids or X");
  }
  if (tables.atom37ToAtom14.length !== 21 * 37 || tables.atom37Mask.length !== 21 * 37) {
    throw new RangeError("residue feature tables must have shape [21, 37]");
  }
  const length = sequence.length;
  const aatype = Float32Array.from(sequence, (residue) => RESTYPE_INDEX.get(residue) ?? 20);
  const targetFeatures = new Float32Array(length * 22);
  const seqMask = new Float32Array(length).fill(1);
  const msaMask = new Float32Array(length).fill(1);
  const residueIndex = residueIndexWithChainBreaks(length, options.chainLengths);
  const atom37ToAtom14 = new Float32Array(length * 37);
  const atom37Mask = new Float32Array(length * 37);
  for (let residue = 0; residue < length; residue += 1) {
    const aa = aatype[residue];
    targetFeatures[residue * 22 + 1 + aa] = 1;
    atom37ToAtom14.set(tables.atom37ToAtom14.subarray(aa * 37, (aa + 1) * 37), residue * 37);
    atom37Mask.set(tables.atom37Mask.subarray(aa * 37, (aa + 1) * 37), residue * 37);
  }
  const recycles = options.recycles ?? 3;
  if (!Number.isSafeInteger(recycles) || recycles < 0) throw new RangeError("recycles must be a non-negative integer");
  if (options.maskedMsaCodes !== undefined && options.maskedMsaCodes.length !== recycles + 1) {
    throw new RangeError("maskedMsaCodes must contain one row per recycling iteration");
  }
  const result = [];
  for (let recycle = 0; recycle <= recycles; recycle += 1) {
    const codes = options.maskedMsaCodes?.[recycle] ?? maskedCodes(aatype, recycle, options.randomSeed ?? 0);
    if (codes.length !== length) throw new RangeError(`masked MSA row ${recycle} has the wrong length`);
    const msaFeatures = new Float32Array(length * 49);
    for (let residue = 0; residue < length; residue += 1) {
      const code = codes[residue];
      if (!Number.isSafeInteger(code) || code < 0 || code > 22) throw new RangeError("invalid masked MSA code");
      msaFeatures[residue * 49 + code] = 1;
      msaFeatures[residue * 49 + 25 + code] = 1 / (1 + 1e-6);
    }
    result.push({
      targetFeatures: targetFeatures.slice(), msaFeatures, msaMask: msaMask.slice(),
      extraMsa: new Float32Array(length), extraHasDeletion: new Float32Array(length),
      extraDeletionValue: new Float32Array(length), extraMsaMask: new Float32Array(length),
      residueIndex: residueIndex.slice(), aatype: aatype.slice(), seqMask: seqMask.slice(),
      atom37ToAtom14: atom37ToAtom14.slice(), atom37Mask: atom37Mask.slice(),
      targetChannels: 22, msaFeatureChannels: 49, extraSequences: 1,
    });
  }
  return result;
}
