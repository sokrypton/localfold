/**
 * Verify the JS input pipeline (`shared/rfd4p/pipeline/index.js` -> `buildBatch`)
 * against the Python RFD4-Proteina oracle dumps (`oracle-dumps/*.safetensors` and
 * `oracle-dumps/cpu/*.safetensors`), and verify downstream feature construction
 * (`trunkSeqFeatures`, `trunkCondFeatures`, `pairFeatures`, `pairCondFeatures`)
 * when fed the JS-built batch.
 *
 * Usage:
 *   node tools/oracle/check-batch.mjs            # runs all 6 benchmark CIF cases + binder_spec
 *   node tools/oracle/check-batch.mjs sm_sam     # runs a single case
 *   node tools/oracle/check-batch.mjs --cpu-dir=<dir> <case>   # a batch-only dump written elsewhere
 *                                                              # (tools/check-rfd4p-batch.py)
 */

import { existsSync, readFileSync } from 'node:fs';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseCifBlock } from '../../shared/rfd4p/pipeline/cif.js';
import { buildBatch } from '../../shared/rfd4p/pipeline/index.js';
import {
  pairCondFeatures,
  pairFeatures,
  trunkCondFeatures,
  trunkSeqFeatures,
} from '../../shared/rfd4p/features.js';
import { compare, loadOracle, readSafetensors, releasedWeightsPath } from './oracle-io.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = resolve(__dirname, '../..');
const ORACLE_DIR = join(ROOT_DIR, 'oracle-dumps');
const CPU_DIR_ARG = process.argv.find((a) => a.startsWith('--cpu-dir='));
const ORACLE_CPU_DIR = CPU_DIR_ARG ? resolve(CPU_DIR_ARG.slice('--cpu-dir='.length)) : join(ORACLE_DIR, 'cpu');
const RFD4_REPO_DIR = '/home/ubuntu/RFD4-Proteina';

const ALL_CASES = [
  'uncond_short',
  'ppi_pdl1',
  'sm_sam',
  'dna_7m5w',
  'motif_idx_1prw',
  'motif_unidx_1prw',
  'binder_spec',
];

function extractOracleBatch(tensors) {
  const batch = {};
  for (const [k, v] of Object.entries(tensors)) {
    if (k.startsWith('batch.')) batch[k.slice(6)] = v;
  }
  return batch;
}

function recordedRandperm(tensors, rngLog, call) {
  const draws = (rngLog || [])
    .filter((r) => r.fn === 'randperm' && r.call === call)
    .map((r) => tensors[`rng.${r.i}`].data);
  let k = 0;
  return (len) => {
    const d = draws[k++];
    if (!d || d.length !== len) {
      throw new Error(`randperm replay mismatch: want ${len}, have ${d?.length}`);
    }
    return d;
  };
}

const EXAMPLE_FALLBACKS = {
  'data/benchmarks/monomer/monomer-short.cif.zst': 'uncond_short.cif.zst',
  'data/benchmarks/ppi/ppi_all_atom/pdl1.cif.zst': 'ppi_pdl1.cif.zst',
  'data/benchmarks/small_molecule_binder/SAM_hbonds.cif.zst': 'sm_sam.cif.zst',
  'data/benchmarks/dna_binder/dna_binder_denovo_val/7m5w_structure_context.cif.zst': 'dna_7m5w.cif.zst',
  'data/benchmarks/motif_scaffolding/motif_indexed/1PRW_AA.cif.zst': 'motif_idx_1prw.cif.zst',
  'data/benchmarks/motif_scaffolding/motif_unindexed/1PRW_AA.cif.zst': 'motif_unidx_1prw.cif.zst',
};

function resolveInputPath(rawPath, baseDir = RFD4_REPO_DIR) {
  if (isAbsolute(rawPath) && existsSync(rawPath)) return rawPath;
  const candidate = resolve(baseDir, rawPath);
  if (existsSync(candidate)) return candidate;
  const examplesDir = join(ROOT_DIR, 'web/examples');
  const mapped = EXAMPLE_FALLBACKS[rawPath] || rawPath.split('/').pop();
  const fallback = join(examplesDir, mapped);
  if (existsSync(fallback)) return fallback;
  throw new Error(`Input file not found: ${rawPath} (tried ${candidate}, ${fallback})`);
}

function loadInputForMeta(meta) {
  const inputPath = resolveInputPath(meta.inputs, RFD4_REPO_DIR);
  const specDir = dirname(inputPath);
  if (inputPath.endsWith('.yaml') || inputPath.endsWith('.yml') || inputPath.endsWith('.json')) {
    const specText = readFileSync(inputPath, 'utf8');
    return {
      input: specText,
      loadInput: (relPath) => {
        const cifPath = existsSync(resolve(specDir, relPath))
          ? resolve(specDir, relPath)
          : resolveInputPath(relPath, RFD4_REPO_DIR);
        return readFileSync(cifPath);
      },
    };
  }
  return {
    input: readFileSync(inputPath),
    loadInput: null,
  };
}

/** The dictionary entries the dump says the page would have fetched (dump_batch_cpu.py `ccd`). */
function ccdBlocksOf(meta) {
  return new Map(Object.entries(meta.ccd ?? {}).map(([id, text]) => [id, parseCifBlock(text)]));
}

function compareBatchTensors(jsBatch, wantBatch) {
  const wantKeys = Object.keys(wantBatch).sort();
  const jsKeys = Object.keys(jsBatch).sort();

  const missingKeys = wantKeys.filter((k) => !(k in jsBatch));
  const extraKeys = jsKeys.filter((k) => !(k in wantBatch));
  const issues = [];
  if (missingKeys.length) issues.push(`missing keys: ${missingKeys.join(', ')}`);
  if (extraKeys.length) issues.push(`extra keys: ${extraKeys.join(', ')}`);

  let intChecked = 0;
  let floatChecked = 0;
  let worstFloatMaxAbs = 0;
  let worstFloatKey = '';

  for (const key of wantKeys) {
    if (!(key in jsBatch)) continue;
    const gotObj = jsBatch[key];
    const wantObj = wantBatch[key];
    const got = gotObj.data;
    const want = wantObj.data;

    if (JSON.stringify(gotObj.shape) !== JSON.stringify(wantObj.shape)) {
      issues.push(
        `${key}: shape mismatch got=${JSON.stringify(gotObj.shape)} want=${JSON.stringify(wantObj.shape)}`
      );
      continue;
    }
    if (got.length !== want.length) {
      issues.push(`${key}: length mismatch got=${got.length} want=${want.length}`);
      continue;
    }

    const isFloat = wantObj.dtype === 'F32' || wantObj.dtype === 'F64';
    if (!isFloat) {
      intChecked++;
      let mismatches = 0;
      let firstIdx = -1;
      for (let i = 0; i < want.length; i++) {
        if (got[i] !== want[i]) {
          if (mismatches === 0) firstIdx = i;
          mismatches++;
        }
      }
      if (mismatches > 0) {
        issues.push(
          `${key} (${wantObj.dtype}): ${mismatches}/${want.length} int/bool mismatches (first at [${firstIdx}]: got=${got[firstIdx]} want=${want[firstIdx]})`
        );
      }
    } else {
      floatChecked++;
      let nanMismatch = 0;
      let firstNanIdx = -1;
      let maxAbs = 0;
      let maxIdx = -1;
      for (let i = 0; i < want.length; i++) {
        const gNan = Number.isNaN(got[i]);
        const wNan = Number.isNaN(want[i]);
        if (gNan !== wNan) {
          if (nanMismatch === 0) firstNanIdx = i;
          nanMismatch++;
          continue;
        }
        if (gNan) continue;
        const diff = Math.abs(got[i] - want[i]);
        if (diff > maxAbs) {
          maxAbs = diff;
          maxIdx = i;
        }
      }
      if (maxAbs > worstFloatMaxAbs) {
        worstFloatMaxAbs = maxAbs;
        worstFloatKey = key;
      }
      if (nanMismatch > 0) {
        issues.push(
          `${key}: ${nanMismatch}/${want.length} NaN-mask mismatches (first at [${firstNanIdx}]: got=${got[firstNanIdx]} want=${want[firstNanIdx]})`
        );
      } else if (maxAbs > 1e-4) {
        issues.push(
          `${key}: float maxAbs=${maxAbs.toExponential(2)} > 1e-4 (at [${maxIdx}]: got=${got[maxIdx]} want=${want[maxIdx]})`
        );
      }
    }
  }

  return {
    ok: issues.length === 0,
    issues,
    intChecked,
    floatChecked,
    worstFloatMaxAbs,
    worstFloatKey,
  };
}

function checkDownstreamFeatures(jsBatch, n, tensors, meta, cfg, weights) {
  const xt = {
    bb_ca: tensors['step.0.x_t.bb_ca'].data,
    local_latents: tensors['step.0.x_t.local_latents'].data,
  };
  const tt = {
    bb_ca: tensors['step.0.t.bb_ca'].data[0],
    local_latents: tensors['step.0.t.local_latents'].data[0],
  };

  const results = {};
  const seq = trunkSeqFeatures(jsBatch, {
    n,
    xt,
    feats: cfg.feats_seq,
    weights,
    randperm: recordedRandperm(tensors, meta.rng, 0),
    noiseAlignChirality: cfg.noise_align_chirality,
  });
  results.seq = compare(seq.data, tensors['call0.init_repr_factory.features'].data);

  const cond = trunkCondFeatures(jsBatch, {
    n,
    t: tt,
    recycleRemaining: null,
    feats: cfg.feats_cond_seq,
    weights,
  });
  results.cond = compare(cond.data, tensors['call0.cond_factory.features'].data);

  const pair = pairFeatures(jsBatch, {
    n,
    caNm: xt.bb_ca,
    feats: cfg.feats_pair_repr,
    dist: {
      dim: cfg.xt_pair_dist_dim,
      min: cfg.xt_pair_dist_min,
      max: cfg.xt_pair_dist_max,
    },
    seqSepDim: cfg.seq_sep_dim,
  });
  results.pair = compare(
    pair.data,
    tensors['call0.pair_repr_builder.init_repr_factory.features'].data
  );

  const pairCond = pairCondFeatures(tt, cfg.feats_pair_cond);
  results.pair_cond = compare(
    pairCond,
    tensors['call0.pair_repr_builder.cond_factory.features'].data
  );

  const issues = [];
  for (const [name, r] of Object.entries(results)) {
    if (!(r.relRms < 1e-5 && r.maxAbs < 1e-4)) {
      issues.push(
        `feature ${name}: relRms=${r.relRms.toExponential(2)} maxAbs=${r.maxAbs.toExponential(2)}`
      );
    }
  }
  return { ok: issues.length === 0, results, issues };
}

async function main() {
  const arg = process.argv.slice(2).find((a) => !a.startsWith('--'));
  const casesToRun = !arg || arg === 'all' ? ALL_CASES : [arg];

  let cfg = null;
  let weights = null;
  try {
    const paths = releasedWeightsPath();
    cfg = JSON.parse(readFileSync(paths.config, 'utf8')).config.nn;
    weights = readSafetensors(paths.weights);
  } catch (err) {
    console.warn(`Warning: could not load released weights for feature check: ${err.message}`);
  }

  let anyFailed = false;

  for (const caseName of casesToRun) {
    const hasGpuDump = existsSync(join(ORACLE_DIR, `${caseName}.safetensors`));
    const hasCpuDump = existsSync(join(ORACLE_CPU_DIR, `${caseName}.safetensors`));
    if (!hasGpuDump && !hasCpuDump) {
      console.error(`FAIL ${caseName}: no oracle dump found in ${ORACLE_DIR} or ${ORACLE_CPU_DIR}`);
      anyFailed = true;
      continue;
    }

    // Check CPU dump (with explicit trace.perm.* when recorded)
    if (hasCpuDump) {
      const { t: cpuTensors, meta: cpuMeta } = loadOracle(ORACLE_CPU_DIR, caseName);
      const wantCpuBatch = extractOracleBatch(cpuTensors);
      const { input, loadInput } = loadInputForMeta(cpuMeta);
      const chainPerm = cpuTensors['trace.perm.0']?.data ?? null;
      const moleculePerm = cpuTensors['trace.perm.1']?.data ?? null;

      const { batch: jsCpuBatch, n } = buildBatch(input, {
        loadInput,
        ccdBlocks: ccdBlocksOf(cpuMeta),
        chainPerm,
        moleculePerm,
        oracleBatch: wantCpuBatch,
        segmentLengths: cpuMeta.seg_lengths ?? undefined,
        pathLengthCutoff: cpuMeta.path_length_search_cutoff ?? 64,
      });

      const cmpCpu = compareBatchTensors(jsCpuBatch, wantCpuBatch);
      if (!cmpCpu.ok) anyFailed = true;
      console.log(
        `${cmpCpu.ok ? 'PASS' : 'FAIL'} [cpu-dump] ${caseName.padEnd(18)} n=${String(n).padEnd(4)} ` +
          `int=${cmpCpu.intChecked} exact, float=${cmpCpu.floatChecked} maxAbs=${cmpCpu.worstFloatMaxAbs.toExponential(2)}` +
          (cmpCpu.worstFloatKey ? ` (${cmpCpu.worstFloatKey})` : '')
      );
      for (const issue of cmpCpu.issues) {
        console.log(`   - ${issue}`);
      }
    }

    // Check full GPU/step oracle dump + call0.*.features
    if (hasGpuDump) {
      const { t: gpuTensors, meta: gpuMeta } = loadOracle(ORACLE_DIR, caseName);
      const wantGpuBatch = extractOracleBatch(gpuTensors);
      const { input, loadInput } = loadInputForMeta(gpuMeta);

      const { batch: jsGpuBatch, n } = buildBatch(input, {
        loadInput,
        oracleBatch: wantGpuBatch,
        pathLengthCutoff: 64,
      });

      const cmpGpu = compareBatchTensors(jsGpuBatch, wantGpuBatch);
      let featReport = '';
      if (cfg && weights && gpuTensors['call0.init_repr_factory.features']) {
        const featCmp = checkDownstreamFeatures(jsGpuBatch, n, gpuTensors, gpuMeta, cfg, weights);
        if (!featCmp.ok) {
          anyFailed = true;
          cmpGpu.issues.push(...featCmp.issues);
        }
        featReport =
          ` | feats: seq=${featCmp.results.seq.maxAbs.toExponential(2)} ` +
          `cond=${featCmp.results.cond.maxAbs.toExponential(2)} ` +
          `pair=${featCmp.results.pair.maxAbs.toExponential(2)} ` +
          `pair_cond=${featCmp.results.pair_cond.maxAbs.toExponential(2)}`;
      }
      if (!cmpGpu.ok) anyFailed = true;
      console.log(
        `${cmpGpu.ok ? 'PASS' : 'FAIL'} [oracle]   ${caseName.padEnd(18)} n=${String(n).padEnd(4)} ` +
          `int=${cmpGpu.intChecked} exact, float=${cmpGpu.floatChecked} maxAbs=${cmpGpu.worstFloatMaxAbs.toExponential(2)}` +
          featReport
      );
      for (const issue of cmpGpu.issues) {
        console.log(`   - ${issue}`);
      }
    }
  }

  if (anyFailed) {
    console.error('\nFAILED: one or more batch or feature checks failed.');
    process.exitCode = 1;
  } else {
    console.log(`\nALL ${casesToRun.length} CASES PASSED (exact int/bool match, float maxAbs <= 1e-4, features match).`);
  }
}

main();
