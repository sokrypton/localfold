/**
 * Public entry point for the RFD4-Proteina pure-JS data pipeline (`shared/rfd4p/pipeline/`).
 *
 * Converts annotated CIFs (`.cif` or `.cif.zst`) or YAML/JSON condition specs (`input`, `contig`,
 * `conditions` with `sel(...)`) into the model batch dictionary (`batch.*`) ready for
 * `shared/rfd4p/features.js` (`trunkSeqFeatures`, `trunkCondFeatures`, `pairFeatures`, `pairCondFeatures`)
 * and the RFD4-Proteina trunk/decoder/encoder.
 */

import { parseJsonOrdered } from './ordered.js';
import { isZstdBytes, decompressZstd, decodeTextOrZstd } from './zstd.js';
import {
  CHAIN_TYPE,
  BOND_TYPE,
  SCALAR_1BODY_ANNOTATIONS,
  NONSCALAR_1BODY_ANNOTATIONS,
  parseCifBlock,
  parseCifToAtomArray,
  getResidueSegments,
  getPolymerizationAtoms,
  computeIdAndEntityAnnotations,
  buildBondsForAtomArray,
} from './cif.js';
import {
  parseConditionSpecYaml,
  evalSelection,
  applyContig,
  applyConditions,
  buildAtomArrayFromConditionSpec,
  getDefaultIndexGroups,
  sliceAtomArray,
  loadDirectInput,
  concatAtomArrays,
} from './spec.js';
import {
  MAX_OVERLAPPING_CENTERS,
  CHIRAL_TYPE,
  IDEAL_POLYHEDRA,
  DOUBLE_BOND_TEMPLATES,
  STEREOISOMER_TEMPLATES,
  alignVectors,
  ensureStereochemistryAnnotations,
  computeChiralityFeatures,
} from './chirality.js';
import {
  getTokenStarts,
  bondFlankingResidues,
  computeTokenPathLengths,
  computeTokenBondFeatures,
} from './bonds.js';
import {
  createSeededRng,
  sliceAtoms,
  cloneAtoms,
  concatAtoms,
  computeWithinChainSourceResIdx,
  ensureChemicalFlags,
  addSrcComponentAnnotation,
  realizeExpandableSegments,
  filterUnwantedAtoms,
  addGlobalAtomIdAnnotation,
  annotateCovalentModificationsAndAtomize,
  addGlobalTokenIdAnnotation,
  finalizeAndCanonicalizeSequenceConditioning,
  featurizeExpandedSegments,
  materializeInternalCoords,
  unindexFlaggedTokens,
  centerAndFeaturizeCentroid,
  addSupportsLocalLatentAnnotation,
  buildAtom37Slots,
  convertToProteinaBatch,
  inferSegmentLengthsFromOracleBatch,
  runValidationPipeline,
} from './transforms.js';

function isAtomArrayObject(x) {
  return (
    x !== null &&
    typeof x === 'object' &&
    !ArrayBuffer.isView(x) &&
    !(x instanceof ArrayBuffer) &&
    Array.isArray(x.res_name) &&
    Array.isArray(x.coord)
  );
}

function isCifString(str) {
  const trimmed = str.trimStart();
  return (
    trimmed.startsWith('data_') ||
    trimmed.includes('\ndata_') ||
    trimmed.includes('_atom_site.')
  );
}

function parseInputToAtomArray(input, opts = {}) {
  if (isAtomArrayObject(input)) {
    return input;
  }
  // A structure handed over by itself is upstream's direct path: its biological assembly 1.
  if (input instanceof ArrayBuffer || ArrayBuffer.isView(input)) {
    return loadDirectInput(input, opts.cifOptions);
  }
  if (typeof input === 'string') {
    if (isCifString(input)) {
      return loadDirectInput(input, opts.cifOptions);
    }
    const spec = input.trimStart().startsWith('{')
      ? parseJsonOrdered(input)
      : parseConditionSpecYaml(input);
    return buildAtomArrayFromConditionSpec(spec, opts);
  }
  if (input !== null && typeof input === 'object') {
    return buildAtomArrayFromConditionSpec(input, opts);
  }
  throw new Error('buildBatch: unsupported input type');
}

/**
 * Build the RFD4-Proteina model batch from an annotated CIF (`.cif` text or `.cif.zst` bytes),
 * a condition-spec YAML/JSON string or object, or a pre-parsed `AtomArray`.
 *
 * @param {string|Uint8Array|ArrayBuffer|object} cifTextOrSpec
 * @param {object} [opts]
 * @param {string|Uint8Array|ArrayBuffer|object} [opts.inputCif] Template CIF structure when `cifTextOrSpec` is a condition spec
 * @param {(inputPath: string) => string|Uint8Array|ArrayBuffer} [opts.loadInput] Loader callback for `spec.input`
 * @param {number} [opts.seed=0] Deterministic PRNG seed for segment lengths / permutations when not replayed
 * @param {number[]} [opts.segmentLengths] Explicit realized lengths for expandable segments (in sentinel order)
 * @param {ArrayLike<number>} [opts.chainPerm] Explicit permutation for `chain_iid` (`trace.perm.0`)
 * @param {ArrayLike<number>} [opts.moleculePerm] Explicit permutation for `molecule_iid` (`trace.perm.1`)
 * @param {Record<number, number[][]>} [opts.internalCoordRotations] Explicit 3x3 rotations per internal coord group ID
 * @param {object} [opts.oracleBatch] Optional oracle batch to replay segment lengths, permutations, and internal coord rotations
 * @param {number} [opts.pathLengthCutoff=64] BFS cutoff for `token_bond_path_length`
 * @param {boolean} [opts.isInference=true] Run inference/validation pipeline mode
 * @returns {{ batch: Record<string, {data: TypedArray, shape: number[], dtype: string}>, n: number, atoms: object, inputAtomArray: object }}
 */
export function buildBatch(cifTextOrSpec, opts = {}) {
  const inputAtomArray = parseInputToAtomArray(cifTextOrSpec, opts);
  const res = runValidationPipeline(inputAtomArray, opts);
  if (res.atoms?.annotations) {
    Object.assign(res.atoms, res.atoms.annotations);
  }
  if (res.batch && res.atoms) {
    Object.defineProperty(res.batch, '__atoms', {
      value: res.atoms,
      enumerable: false,
      configurable: true,
      writable: true,
    });
  }
  return res;
}

export {
  isZstdBytes,
  decompressZstd,
  decodeTextOrZstd,
  CHAIN_TYPE,
  BOND_TYPE,
  SCALAR_1BODY_ANNOTATIONS,
  NONSCALAR_1BODY_ANNOTATIONS,
  parseCifBlock,
  parseCifToAtomArray,
  getResidueSegments,
  getPolymerizationAtoms,
  computeIdAndEntityAnnotations,
  buildBondsForAtomArray,
  parseConditionSpecYaml,
  evalSelection,
  applyContig,
  applyConditions,
  buildAtomArrayFromConditionSpec,
  getDefaultIndexGroups,
  sliceAtomArray,
  concatAtomArrays,
  MAX_OVERLAPPING_CENTERS,
  CHIRAL_TYPE,
  IDEAL_POLYHEDRA,
  DOUBLE_BOND_TEMPLATES,
  STEREOISOMER_TEMPLATES,
  alignVectors,
  ensureStereochemistryAnnotations,
  computeChiralityFeatures,
  getTokenStarts,
  bondFlankingResidues,
  computeTokenPathLengths,
  computeTokenBondFeatures,
  createSeededRng,
  sliceAtoms,
  cloneAtoms,
  concatAtoms,
  computeWithinChainSourceResIdx,
  ensureChemicalFlags,
  addSrcComponentAnnotation,
  realizeExpandableSegments,
  filterUnwantedAtoms,
  addGlobalAtomIdAnnotation,
  annotateCovalentModificationsAndAtomize,
  addGlobalTokenIdAnnotation,
  finalizeAndCanonicalizeSequenceConditioning,
  featurizeExpandedSegments,
  materializeInternalCoords,
  unindexFlaggedTokens,
  centerAndFeaturizeCentroid,
  addSupportsLocalLatentAnnotation,
  buildAtom37Slots,
  convertToProteinaBatch,
  inferSegmentLengthsFromOracleBatch,
  runValidationPipeline,
};
