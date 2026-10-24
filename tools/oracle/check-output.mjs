/**
 * Verify RFD4-Proteina output processing, Hungarian insertion mapping, internal-coord
 * group grafting, input-frame alignment, and mmCIF / PDB writers (`shared/rfd4p/output.js`)
 * against the upstream PyTorch oracle dumps (`oracle-dumps/<case>.cif`).
 *
 *   node tools/oracle/check-output.mjs [case=all]
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { formatSampleLocalLatents } from '../../cpu/rfd4p/decoder.js';
import {
  buildPredictedInsertionMapping,
  linearSumAssignmentMaximize,
  parseAtomSiteCif,
  processSampleToAtoms,
  toCifString,
  toPdbString,
} from '../../shared/rfd4p/output.js';
import { compare, loadOracle } from './oracle-io.mjs';

const ALL_CASES = [
  'uncond_short',
  'ppi_pdl1',
  'sm_sam',
  'dna_7m5w',
  'motif_idx_1prw',
  'motif_unidx_1prw',
];

const args = process.argv.slice(2);
const toRun = args.length === 0 || args[0] === 'all' ? ALL_CASES : args;
const dumpDir = new URL('../../oracle-dumps/', import.meta.url).pathname;

let failed = false;
function ok(label, cond, extra = '') {
  if (!cond) failed = true;
  console.log(`${cond ? 'ok  ' : 'FAIL'} ${label.padEnd(42)} ${extra}`);
}

// 1. Unit check for rectangular Hungarian solver (`linearSumAssignmentMaximize`)
{
  // 3x5 matrix where greedy argmax fails (row 0 and row 1 both prefer col 2)
  const score = new Float64Array([
    10,  1, 20,  2,  3,
     9, 15, 19,  4,  1,
     2,  3,  5, 12, 11,
  ]);
  const { rowInd, colInd } = linearSumAssignmentMaximize(score, 3, 5);
  // Optimal: row 0 -> col 2 (20), row 1 -> col 1 (15), row 2 -> col 3 (12), sum = 47
  const sum = score[0 * 5 + colInd[0]] + score[1 * 5 + colInd[1]] + score[2 * 5 + colInd[2]];
  ok(
    'hungarian rectangular 3x5 maximize',
    rowInd.length === 3 && colInd[0] === 2 && colInd[1] === 1 && colInd[2] === 3 && sum === 47,
    `cols=[${colInd.join(',')}] sum=${sum}`
  );
}

for (const caseName of toRun) {
  const { t } = loadOracle(dumpDir, caseName);
  const rawAtomsJson = readFileSync(join(dumpDir, 'cpu', `${caseName}.atoms.json`), 'utf8').replace(/\bNaN\b/g, 'null');
  const inputAtoms = JSON.parse(rawAtomsJson);
  const oracleCifText = readFileSync(join(dumpDir, `${caseName}.cif`), 'utf8');
  const wantAtoms = parseAtomSiteCif(oracleCifText);

  // Extract batch tensors
  const batch = {};
  for (const [k, v] of Object.entries(t)) {
    if (k.startsWith('batch.')) batch[k.slice(6)] = v;
  }
  const n = batch.padding_mask.shape[1];

  // Check `formatSampleLocalLatents` against `final.{coors,atom_mask,residue_type}`
  const formatted = formatSampleLocalLatents(batch, {
    n,
    caNm: t['final.x.bb_ca'].data,
    decoded: {
      coorsNm: t['decode.out.coors_nm'].data,
      atomMask: t['decode.out.atom_mask'].data,
      residueType: t['decode.out.residue_type'].data,
      emitMask: t['decode.out.emit_mask'].data,
    },
  });
  const coorsCmp = compare(formatted.coors, t['final.coors'].data);
  const maskMatch = formatted.atomMask.every((v, i) => v === t['final.atom_mask'].data[i]);
  const resMatch = formatted.residueType.every((v, i) => v === t['final.residue_type'].data[i]);
  ok(
    `[${caseName}] formatSampleLocalLatents`,
    coorsCmp.maxAbs < 1e-5 && maskMatch && resMatch,
    `coors maxAbs=${coorsCmp.maxAbs.toExponential(2)} Å`
  );

  // Run `processSampleToAtoms`
  const insLogits = t['step.0.out.insertion.logits']?.data ?? null;
  const result = processSampleToAtoms({
    coors: t['final.coors'].data,
    atomMask: t['final.atom_mask'].data,
    residueType: t['final.residue_type'].data,
    paddingMask: batch.padding_mask.data,
    insertionLogits: insLogits,
    inputAtoms,
    batch,
  });

  const gotAtoms = result.atoms;
  ok(
    `[${caseName}] atom count`,
    gotAtoms.length === wantAtoms.length,
    `got=${gotAtoms.length} want=${wantAtoms.length}`
  );

  let metaMismatches = 0;
  let firstMismatch = '';
  let maxCoordErr = 0;
  let sumSqErr = 0;
  const count = Math.min(gotAtoms.length, wantAtoms.length);

  for (let i = 0; i < count; i += 1) {
    const g = gotAtoms[i];
    const w = wantAtoms[i];
    if (
      g.groupPdb !== w.groupPdb ||
      g.chainId !== w.chainId ||
      g.resId !== w.resId ||
      g.resName !== w.resName ||
      g.atomName !== w.atomName ||
      g.element !== w.element
    ) {
      if (metaMismatches === 0) {
        firstMismatch = `at ${i}: got(${g.groupPdb} ${g.chainId}:${g.resId} ${g.resName} ${g.atomName} ${g.element}) vs want(${w.groupPdb} ${w.chainId}:${w.resId} ${w.resName} ${w.atomName} ${w.element})`;
      }
      metaMismatches += 1;
    }

    const dx = Math.abs(g.x - w.x);
    const dy = Math.abs(g.y - w.y);
    const dz = Math.abs(g.z - w.z);
    const dMax = Math.max(dx, dy, dz);
    if (dMax > maxCoordErr) maxCoordErr = dMax;
    sumSqErr += dx * dx + dy * dy + dz * dz;
  }

  const rmsd = Math.sqrt(sumSqErr / Math.max(1, count));
  ok(
    `[${caseName}] atom metadata & ordering`,
    metaMismatches === 0,
    metaMismatches === 0 ? `all ${count} atoms match` : `${metaMismatches} mismatches (${firstMismatch})`
  );
  ok(
    `[${caseName}] 3D coordinates (<= 1e-3 Å)`,
    maxCoordErr <= 1e-3,
    `maxAbs=${maxCoordErr.toExponential(2)} Å rmsd=${rmsd.toExponential(2)} Å`
  );

  // Check unindexed insertion mapping & provenance annotations on `motif_unidx_1prw`
  if (caseName === 'motif_unidx_1prw') {
    const step0Map = buildPredictedInsertionMapping(insLogits, n, inputAtoms);
    const step0Targets = new Set(Object.values(step0Map));
    ok(
      `[${caseName}] step.0 Hungarian bijection`,
      Object.keys(step0Map).length === 40 && step0Targets.size === 40,
      `mapped=${Object.keys(step0Map).length} uniqueTargets=${step0Targets.size}`
    );

    let unidxAnnotMismatches = 0;
    for (let i = 0; i < count; i += 1) {
      const g = gotAtoms[i];
      const w = wantAtoms[i];
      if (g.isUnindexedInsertion !== w.isUnindexedInsertion || g.crdSrcComponent !== w.crdSrcComponent) {
        unidxAnnotMismatches += 1;
      }
    }
    ok(
      `[${caseName}] unindexed insertion mapping`,
      Object.keys(result.predictedInsertionMapping).length === 40 && unidxAnnotMismatches === 0,
      `mapped=40/40 annotMismatches=${unidxAnnotMismatches}`
    );
  }

  // Verify `toCifString` round-trip and `toPdbString` formatting
  const cifOut = toCifString(result, { dataBlockName: caseName });
  const roundTripAtoms = parseAtomSiteCif(cifOut);
  let rtMaxErr = 0;
  let rtMetaDiff = 0;
  for (let i = 0; i < count; i += 1) {
    const r = roundTripAtoms[i];
    const w = wantAtoms[i];
    if (!r || r.chainId !== w.chainId || r.resId !== w.resId || r.resName !== w.resName || r.atomName !== w.atomName) {
      rtMetaDiff += 1;
    } else {
      const d = Math.max(Math.abs(r.x - w.x), Math.abs(r.y - w.y), Math.abs(r.z - w.z));
      if (d > rtMaxErr) rtMaxErr = d;
    }
  }
  const pdbOut = toPdbString(result, { title: caseName });
  const pdbAtomLines = pdbOut.split('\n').filter((l) => l.startsWith('ATOM  ') || l.startsWith('HETATM'));
  ok(
    `[${caseName}] toCifString & toPdbString`,
    roundTripAtoms.length === wantAtoms.length && rtMetaDiff === 0 && rtMaxErr <= 1e-3 + 1e-9 && pdbAtomLines.length === wantAtoms.length,
    `cifMaxDiff=${rtMaxErr.toExponential(2)} Å pdbLines=${pdbAtomLines.length}`
  );
}

console.log(failed ? 'FAILED' : 'all output checks passed');
process.exitCode = failed ? 1 : 0;
