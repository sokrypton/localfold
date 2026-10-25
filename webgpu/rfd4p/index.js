/**
 * Public entry point for RFD4-Proteina WebGPU & CPU inference.
 *
 * 🔴 MOVED FROM `webgpu/index.js` ON THE WAY IN, because this repository
 * already has one and it is a barrel of fold kernels. This file is RFD4's own
 * entry point and belongs beside the engine it opens, so its relative paths
 * gained a level: `../../shared/rfd4p/...` where upstream wrote
 * `../shared/rfd4p/...`. See docs/DESIGN.md.
 */

import {
  ALLOWED_RESIDUE_TYPES,
  ATOM37_NAMES,
  STANDARD_POLYMER_RESIDUES,
  TokenType,
  UNIFIED_ATOM37_ENCODING,
  UNIFIED_ATOM37_TOKENS,
  UNIFIED_RESTYPE_ATOM37_MASK,
} from '../../shared/rfd4p/encoding.js';
import {
  COND,
  alignChirality,
  defaultRandperm,
  pairCondFeatures,
  pairFeatures,
  permuteCoordGroups,
  timeEmbedding,
  trunkCondFeatures,
  trunkSeqFeatures,
} from '../../shared/rfd4p/features.js';
import {
  DEFAULT_SAMPLING,
  RECYCLE_SCHEDULES,
  flowEmitMasks,
  noiseSchedule,
  recycleDepths,
  sampleInitialNoise,
  sampleSchedule,
  scheduleTsrSafe,
  seededRng,
  simulationStep,
  tsrRatio,
} from '../../shared/rfd4p/flow.js';
import {
  CCD_STANDARD_ATOM_ORDER,
  alignAtomsToInputFrame,
  buildPredictedInsertionMapping,
  getOrderedAtomNamesForResidue,
  insertInternalCoordConditionedGroups,
  linearSumAssignmentMaximize,
  linearSumAssignmentMinimize,
  parseAtomSiteCif,
  processSampleToAtoms,
  superimposeCoords3D,
  toCifString,
  toPdbString,
} from '../../shared/rfd4p/output.js';
import {
  RESAMPLED_FORWARD_MARGINAL,
  buildPartialSchedule,
  corruptBatch,
  resolveInverseFoldingMethod,
  resolvePartialT,
  sampleNoisedTargetBackbone,
} from '../../shared/rfd4p/partial.js';
import { RFD4P_PRESETS, getRfd4pPresetById } from '../../shared/rfd4p/presets.js';
import {
  buildAtomArrayFromConditionSpec,
  buildBatch,
  decodeTextOrZstd,
  decompressZstd,
  isZstdBytes,
  parseCifToAtomArray,
  parseConditionSpecYaml,
} from '../../shared/rfd4p/pipeline/index.js';
import {
  DEFAULT_HF_CONFIG_URL,
  DEFAULT_HF_WEIGHTS_URL,
  DEFAULT_LOCAL_CONFIG_URL,
  DEFAULT_LOCAL_SHARDED_DIR,
  DEFAULT_LOCAL_WEIGHTS_URL,
  fetchRfd4pConfig,
  fetchSafetensors,
  fetchShardedWeights,
  parseSafetensors,
  readShardedTensor,
  resolveRfd4pShardBase,
} from '../../shared/rfd4p/safetensors.js';
import {
  aeMasks,
  decodeLatentsCpu,
  decoderPostprocess,
  decoderSeqFeatures,
  formatSampleLocalLatents,
  sequenceForDecoder,
} from '../../cpu/rfd4p/decoder.js';
import {
  encodeLatentsCpu,
  encoderPostprocess,
  encoderSeqFeatures,
} from '../../cpu/rfd4p/encoder.js';
import {
  adaptiveLayerNorm,
  adaptiveOutputScale,
  applyNorm,
  forwardTrunkCpu,
  generateRfd4pCpu,
  layerNorm,
  linearLayer,
  multiheadAttnAndTransition,
  pairBiasAttention,
  pairReprUpdate,
  rmsNorm,
  runDecoderForwardCpu,
  runTrunkForwardCpu,
  swiGluTransition,
  triangleMultiplicativeUpdate,
} from '../../cpu/rfd4p/reference.js';
import { TokenBlocks, tokenBuffers, uploadTokenLayer } from './blocks.js';
import { Rfd4pDecoderGpu } from './decoder.js';
import { Rfd4pEncoderGpu } from './encoder.js';
import { Rfd4pEngine, fold2 } from './engine.js';
import { Rfd4pGeneratorGpu, generateRfd4p, yieldToBrowser } from './generate.js';
import { DEFAULT_CHUNK_BYTES, Rfd4pTrunkGpu, chunkPlan } from './trunk.js';

/**
 * Request a WebGPU device with generous storage buffer limits and optional `timestamp-query` + `shader-f16`.
 */
export async function requestWebGpuDevice() {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    return null;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  const requiredLimits = {};
  for (const name of [
    'maxStorageBufferBindingSize',
    'maxBufferSize',
    'maxComputeWorkgroupsPerDimension',
    'maxComputeWorkgroupStorageSize',
    'maxComputeWorkgroupSizeX',
    'maxComputeInvocationsPerWorkgroup',
  ]) {
    const v = adapter.limits?.[name];
    if (typeof v === 'number' && Number.isFinite(v)) {
      requiredLimits[name] = v;
    }
  }
  const requiredFeatures = [
    'timestamp-query',
    'shader-f16',
    'subgroups',
    'subgroup-size-control',
    'chromium-experimental-subgroup-matrix',
  ].filter((f) => adapter.features?.has(f));
  const device = await adapter.requestDevice({ requiredLimits, requiredFeatures });
  if (!device.adapterInfo && adapter.info) device.adapterInfo = adapter.info;
  return { adapter, device };
}

export {
  ALLOWED_RESIDUE_TYPES,
  ATOM37_NAMES,
  STANDARD_POLYMER_RESIDUES,
  TokenType,
  UNIFIED_ATOM37_ENCODING,
  UNIFIED_ATOM37_TOKENS,
  UNIFIED_RESTYPE_ATOM37_MASK,
  COND,
  alignChirality,
  defaultRandperm,
  pairCondFeatures,
  pairFeatures,
  permuteCoordGroups,
  timeEmbedding,
  trunkCondFeatures,
  trunkSeqFeatures,
  DEFAULT_SAMPLING,
  RECYCLE_SCHEDULES,
  flowEmitMasks,
  noiseSchedule,
  recycleDepths,
  sampleInitialNoise,
  sampleSchedule,
  scheduleTsrSafe,
  seededRng,
  simulationStep,
  tsrRatio,
  CCD_STANDARD_ATOM_ORDER,
  alignAtomsToInputFrame,
  buildPredictedInsertionMapping,
  getOrderedAtomNamesForResidue,
  insertInternalCoordConditionedGroups,
  linearSumAssignmentMaximize,
  linearSumAssignmentMinimize,
  parseAtomSiteCif,
  processSampleToAtoms,
  superimposeCoords3D,
  toCifString,
  toPdbString,
  RESAMPLED_FORWARD_MARGINAL,
  buildPartialSchedule,
  corruptBatch,
  resolveInverseFoldingMethod,
  resolvePartialT,
  sampleNoisedTargetBackbone,
  RFD4P_PRESETS,
  getRfd4pPresetById,
  buildAtomArrayFromConditionSpec,
  buildBatch,
  decodeTextOrZstd,
  decompressZstd,
  isZstdBytes,
  parseCifToAtomArray,
  parseConditionSpecYaml,
  DEFAULT_HF_CONFIG_URL,
  DEFAULT_HF_WEIGHTS_URL,
  DEFAULT_LOCAL_CONFIG_URL,
  DEFAULT_LOCAL_SHARDED_DIR,
  DEFAULT_LOCAL_WEIGHTS_URL,
  fetchRfd4pConfig,
  fetchSafetensors,
  fetchShardedWeights,
  parseSafetensors,
  readShardedTensor,
  resolveRfd4pShardBase,
  aeMasks,
  decodeLatentsCpu,
  decoderPostprocess,
  decoderSeqFeatures,
  formatSampleLocalLatents,
  sequenceForDecoder,
  encodeLatentsCpu,
  encoderPostprocess,
  encoderSeqFeatures,
  adaptiveLayerNorm,
  adaptiveOutputScale,
  applyNorm,
  forwardTrunkCpu,
  generateRfd4pCpu,
  layerNorm,
  linearLayer,
  multiheadAttnAndTransition,
  pairBiasAttention,
  pairReprUpdate,
  rmsNorm,
  runDecoderForwardCpu,
  runTrunkForwardCpu,
  swiGluTransition,
  triangleMultiplicativeUpdate,
  TokenBlocks,
  tokenBuffers,
  uploadTokenLayer,
  Rfd4pDecoderGpu,
  Rfd4pEncoderGpu,
  Rfd4pEngine,
  fold2,
  Rfd4pGeneratorGpu,
  generateRfd4p,
  yieldToBrowser,
  Rfd4pTrunkGpu,
  DEFAULT_CHUNK_BYTES,
  chunkPlan,
};
