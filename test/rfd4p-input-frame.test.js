/**
 * The two upstream fixes this repository was missing, each held by an arm that fails without it
 * (`sokrypton/rfd4-proteina-webgpu` 3a7cda9d and 4240ffc8, ported here 2026-11-03).
 *
 * 🔴 BOTH WERE INVISIBLE TO EVERY GATE HERE, for the same reason: nothing asked where a design ENDS
 * UP, or what an unresolved side chain does to the features. `test/rfd4p-core.test.js` drives
 * `processSampleToAtoms` and asserts the PDB parses - and a structure translated by its own centroid
 * parses perfectly - while no test had ever put a NaN coordinate into the encoder.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildBatch, processSampleToAtoms } from '../webgpu/rfd4p/index.js';
import { centerAndFeaturizeCentroid } from '../shared/rfd4p/pipeline/transforms.js';
import { encoderSeqFeatures } from '../cpu/rfd4p/encoder.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const EXAMPLES_DIR = path.resolve(__dirname, '../web/examples');

test('centerAndFeaturizeCentroid keeps the shift it centred by, and it is the CONDITIONED centroid', () => {
  const coord = [[10, 20, 30], [12, 22, 32], [100, 100, 100], [-100, -100, -100]];
  const nan = [NaN, NaN, NaN];
  const atoms = {
    nAtoms: 4,
    coord,
    // Two atoms are conditioned on coordinates; the other two are not and must not count towards it.
    annotations: { condition_coordinate_1_atom: [[10, 20, 30], [12, 22, 32], nan, nan] },
  };
  centerAndFeaturizeCentroid(atoms);
  assert.deepEqual(atoms.center_shift, [11, 21, 31]);
  // What the model is handed is centred by exactly that, which is why the output has to undo it.
  assert.deepEqual(atoms.coord_to_be_noised[0], [-1, -1, -1]);
  assert.deepEqual(atoms.annotations.condition_coordinate_1_atom[1], [1, 1, 1]);
  // Not a per-atom column: the atom table is walked by key in several places.
  assert.ok(!Object.keys(atoms).includes('center_shift'));
});

test('a design with anchors is put back in the input\'s frame, by exactly the shift it was centred by', () => {
  // 🔴 ppi_pdl1 BECAUSE IT IS THE ONE BUNDLED INPUT WITH BOTH ANCHORS AND AN OFF-ORIGIN CENTROID.
  // Measured over the examples: uncond_short, dna_7m5w and both 1prw motifs ship pre-centred, so the
  // restore is a no-op on them; 5o45_cropped and sm_sam have the offset and no coordinate anchors. A
  // reader's own upload is the real case, tens to hundreds of angstroms out - the third arm below.
  const bytes = fs.readFileSync(path.join(EXAMPLES_DIR, 'ppi_pdl1.cif.zst'));
  const { batch, n, atoms } = buildBatch(bytes, { seed: 42 });
  const shift = atoms.center_shift;
  assert.ok(Array.isArray(shift) && shift.length === 3, 'the atom table carries center_shift');
  assert.ok(Math.hypot(...shift) > 1e-2, `this input's conditioned centroid is off the origin (${shift})`);

  const coors = new Float32Array(n * 37 * 3);
  const atomMask = new Uint8Array(n * 37);
  const residueType = new Int32Array(n).fill(1); // ALA
  for (let i = 0; i < n; i += 1) {
    for (const s of [0, 1, 2, 3, 4]) {
      atomMask[i * 37 + s] = 1;
      coors[(i * 37 + s) * 3 + 0] = i * 3.8 + s * 0.5;
    }
  }
  // 🔴 A DIFFERENTIAL, NOT A POSITION, AND THAT IS NOT A WEAKER TEST - IT IS THE ONLY SOUND ONE HERE.
  // These coordinates are a straight line of fake atoms, so `alignAtomsToInputFrame` fits them onto
  // the conditioned coordinates as best it can and lands nowhere near them (168 A out, measured): an
  // assertion that anchors sit ON the input needs a real design, not a stand-in. What the fix changes
  // is a rigid translation, so comparing two runs isolates it exactly - and a zero shift IS the
  // behaviour before the fix, so the reference arm is the old code rather than a number copied down.
  const positions = (sh) => {
    atoms.center_shift = sh;
    return processSampleToAtoms({
      coors, atomMask, residueType, paddingMask: batch.padding_mask.data, inputAtoms: atoms, batch,
    }).atoms.map((a) => [a.x, a.y, a.z]);
  };

  const before = positions([0, 0, 0]);
  for (const want of [shift.slice(), [25, -40, 7.5]]) {
    const after = positions(want);
    assert.equal(after.length, before.length);
    let worst = 0;
    for (let i = 0; i < after.length; i += 1) {
      for (let k = 0; k < 3; k += 1) worst = Math.max(worst, Math.abs(after[i][k] - before[i][k] - want[k]));
    }
    assert.ok(worst < 1e-3, `every atom moves by the shift and nothing else (worst ${worst.toExponential(2)} A, shift ${want})`);
  }
  atoms.center_shift = shift;
});

test('a design held in place by nothing is left in the model\'s own frame', () => {
  // 🔴 THE GATE ON ANCHORS IS A DECISION, NOT AN OPTIMISATION: with no coordinate conditioning there
  // is no input frame to return to - the shift is then the centroid of EVERYTHING (measured 5.6 A on
  // this input, where a conditioned centroid would be its anchors') and translating by it would move
  // an unconditioned design for no reason. sm_sam is the case: an off-origin input with no anchors.
  const bytes = fs.readFileSync(path.join(EXAMPLES_DIR, 'sm_sam.cif.zst'));
  const { batch, n, atoms } = buildBatch(bytes, { seed: 42 });
  assert.ok(Math.hypot(...atoms.center_shift) > 3, 'sm_sam is off the origin');
  const cond = atoms.annotations.condition_coordinate_1_atom;
  let anchors = 0;
  for (let i = 0; i < atoms.nAtoms; i += 1) if (Number.isFinite(cond?.[i]?.[0])) anchors += 1;
  assert.equal(anchors, 0, 'and holds no coordinate anchors');

  const coors = new Float32Array(n * 37 * 3);
  const atomMask = new Uint8Array(n * 37);
  const residueType = new Int32Array(n).fill(1);
  for (let i = 0; i < n; i += 1) {
    for (const s of [0, 1, 2, 3, 4]) {
      atomMask[i * 37 + s] = 1;
      coors[(i * 37 + s) * 3 + 0] = i * 3.8 + s * 0.5;
    }
  }
  const positions = (sh) => {
    atoms.center_shift = sh;
    return processSampleToAtoms({
      coors, atomMask, residueType, paddingMask: batch.padding_mask.data, inputAtoms: atoms, batch,
    }).atoms.map((a) => [a.x, a.y, a.z]);
  };
  const withShift = positions(atoms.center_shift.slice());
  const withNone = positions([0, 0, 0]);
  assert.deepEqual(withShift, withNone, 'the shift is not applied without anchors');
});

test('an atom the structure leaves unmodelled is absent from the encoder features, not a NaN', () => {
  const n = 2;
  const slots = [0, 1, 2, 3, 4];
  const atom37Mask = new Uint8Array(n * 37);
  const coordsNm = new Float32Array(n * 37 * 3);
  for (let i = 0; i < n; i += 1) {
    for (const s of slots) {
      atom37Mask[i * 37 + s] = 1;
      coordsNm[(i * 37 + s) * 3 + 0] = 0.38 * i + 0.05 * s;
      coordsNm[(i * 37 + s) * 3 + 1] = 0.1 * s;
      coordsNm[(i * 37 + s) * 3 + 2] = -0.2 * i;
    }
  }
  // 🔴 PRESENT IN THE MASK, ABSENT FROM THE STRUCTURE: what an entry with an unresolved side chain
  // hands over. Token 1's slot 3 is flagged and has no coordinates.
  for (let k = 0; k < 3; k += 1) coordsNm[(1 * 37 + 3) * 3 + k] = NaN;

  const batch = {
    padding_mask: new Uint8Array(n).fill(1),
    token_type: new Int32Array(n),
    atom37_mask: atom37Mask,
    coords_nm: coordsNm,
    chain_iid: new Int32Array(n),
    incident_bond_type_counts: new Float32Array(n * 4),
    ground_truth_element: new Int32Array(n).fill(6),
    charge: new Float32Array(n),
  };
  // Only the two coordinate features, so the arm needs no weights: every other `feats_seq` entry
  // reads an embedding table out of the bundle.
  const cfg = { feats_seq: ['x1_a37coors_nm', 'x1_a37coors_nm_rel'], token_type_emb_dim: 32, charge_emb_dim: 16 };
  const { data: feats, dim: fdim } = encoderSeqFeatures(batch, { n, weights: {}, cfg });

  assert.equal(fdim, 2 * 37 * 4);
  const bad = [...feats].findIndex((v) => !Number.isFinite(v));
  assert.equal(bad, -1, `no feature is NaN (first at ${bad})`);
  // 🔴 AND THE NaN WOULD NOT HAVE STAYED IN ITS OWN COLUMN: the encoder's LayerNorms take a row's
  // mean, so one unmodelled atom reaches every latent of every token. Hence "a design with no atoms".
  for (const base of [0, 37 * 4]) {
    const o = 1 * fdim + base;
    assert.equal(feats[o + 111 + 3], 0, 'the unmodelled slot is masked off');
    assert.equal(feats[o + 3 * 3], 0);
    assert.equal(feats[o + 111 + 1], 1, 'the modelled slots are untouched');
  }
});
