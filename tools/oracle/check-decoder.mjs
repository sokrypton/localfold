/**
 * Check the CPU decoder (`cpu/rfd4p/decoder.js`) against the oracle's decode inputs/outputs,
 * including the sequence-readout conditioning of the last trunk step.
 *
 *   node tools/oracle/check-decoder.mjs [case=uncond_short]
 */

import { readFileSync } from 'node:fs';
import { decodeLatentsCpu, sequenceForDecoder } from '../../cpu/rfd4p/decoder.js';
import { compare, loadOracle, readSafetensors, releasedWeightsPath } from './oracle-io.mjs';

const cases = process.argv.slice(2);
const ALL = ['uncond_short', 'ppi_pdl1', 'sm_sam', 'dna_7m5w', 'motif_idx_1prw', 'motif_unidx_1prw'];
const toRun = cases.length === 0 ? ['uncond_short'] : cases[0] === 'all' ? ALL : cases;
const paths = releasedWeightsPath();
const cfg = JSON.parse(readFileSync(paths.config, 'utf8')).config.autoencoder_config.nn_ae.decoder;
const weights = readSafetensors(paths.weights);
const dumpDir = new URL('../../oracle-dumps/', import.meta.url).pathname;

let failed = false;
const ok = (label, cond, extra = '') => {
  if (!cond) failed = true;
  console.log(`${cond ? 'ok  ' : 'FAIL'} ${label} ${extra}`);
};

for (const caseName of toRun) {
  const { t, meta } = loadOracle(dumpDir, caseName);
  const batch = {};
  for (const [k, v] of Object.entries(t)) if (k.startsWith('batch.')) batch[k.slice(6)] = v;
  const n = batch.padding_mask.shape[1];
  const last = meta.steps.length - 1;

  const seq = sequenceForDecoder(batch, n, t[`step.${last}.out.sequence.logits`].data);
  const wantSeq = t['decode.in.C_SEQ'].data;
  ok(`[${caseName}] decoder C_SEQ from readout`, seq.every((v, i) => v === wantSeq[i]));

  const out = decodeLatentsCpu(batch, {
    n, caNm: t['decode.in.ca_coors_nm'].data, z: t['decode.in.z_latent'].data, seq: Int32Array.from(wantSeq), weights, cfg,
  });
  const rt = t['decode.out.residue_type'].data;
  ok(`[${caseName}] residue_type`, out.residueType.every((v, i) => v === rt[i]));
  const am = t['decode.out.atom_mask'].data;
  let maskDiff = 0;
  for (let i = 0; i < am.length; i += 1) if (am[i] !== out.atomMask[i]) maskDiff += 1;
  ok(`[${caseName}] atom_mask`, maskDiff === 0, `${maskDiff} slot(s) differ`);
  const r = compare(out.coorsNm, t['decode.out.coors_nm'].data);
  ok(`[${caseName}] coors_nm`, r.maxAbs < 1e-4, `relRms=${r.relRms.toExponential(2)} maxAbs=${r.maxAbs.toExponential(2)} nm`);
}
console.log(failed ? 'FAILED' : 'decoder matches');
process.exitCode = failed ? 1 : 0;
