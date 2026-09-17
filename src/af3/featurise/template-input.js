/**
 * A user's structure, turned into the template arrays AF3's embedder takes.
 *
 * The output is one slot: `{aatype, atomPositions, atomMask}` over the QUERY's
 * tokens, in AF3's dense-24 layout. src/af3/featurise/template-features.js turns that
 * into the six geometry features and src/af3/trunk/template-webgpu.js embeds them.
 *
 * 🔴 THERE IS NO ATOM TABLE HERE, AND THAT IS THE POINT. AF3's dense layout is
 * exactly the order `conformerFor` lists a residue's atoms in - N, CA, C, O,
 * CB, ... - and that is the same table src/af3/featurise/featurise.js builds the QUERY's
 * atoms from. Mapping a template's atoms by NAME through it means a template
 * and the query it is shown against cannot end up in different layouts. The
 * alternative, AF3's own PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37, is a second
 * table saying the same thing that would have to be kept in step by hand.
 *
 * 🔴 AN UNCOVERED QUERY POSITION IS THE GAP TOKEN, NOT ALANINE. Measured, not
 * assumed: an AF3 dump whose template covers 8 of 16 query residues carries
 * `template_aatype` 21 - "-" - at the other eight, with an all-zero atom mask.
 * Aatype 0 would be ALA, and the difference is not cosmetic: the aatype
 * features are two of the module's nine and they are read whether or not there
 * is geometry, so an uncovered position would contribute alanine's embedding
 * instead of the gap's at every pair it takes part in.
 *
 * 🔴 AND PARTIAL COVERAGE IS THE MECHANISM, NOT AN EDGE CASE. AF2 hid a
 * position by zeroing its row; AF3 hides it by leaving it out of the residue
 * map. So a crystal structure missing a loop, a construct that starts at
 * residue 17, and a template covering one domain of two are all the same
 * thing, and none of them is an error - which is exactly why the caller is
 * given `covered` to show rather than a boolean to ignore.
 */
import { ONE_LETTER } from "../fold.js";
import { aatypeFor, conformerFor } from "./reference-conformers.js";
import { ATOM37, NUM_DENSE } from "./template-features.js";
import { coordinateAtoms } from "../../heads/superpose-pdb.js";


/** AF3's gap residue type, which is what an uncovered query position gets. */
export const GAP_AATYPE = 21;

/**
 * 🔴 A WATER IS NOT A RESIDUE, AND A PDB PUTS IT ON A CHAIN. Reading every
 * HETATM as a template residue made 1qys chain A ninety-NINE residues instead
 * of ninety-one: eight waters, each an X with one atom, each contributing a
 * pseudo-beta position to the distogram and a row to the alignment.
 *
 * 🔴 BUT MSE IS A RESIDUE AND IS ALWAYS A HETATM. Selenomethionine is how a
 * great many crystal structures were phased, and dropping every HETATM would
 * take a real methionine out of the middle of a chain - leaving a hole the
 * alignment then has to guess across. It is the one exception worth carrying;
 * anything else heteroatomic is a ligand, an ion or a solvent molecule, and a
 * template is one protein chain.
 */
const HETERO_RESIDUES = { MSE: "M" };

/**
 * One chain of a structure, as residues in file order.
 *
 * 🔴 KEYED ON THE RESIDUE NUMBER, NEVER ON POSITION IN THE ATOM LIST. A PDB's
 * ATOM records are what was RESOLVED: a structure missing residues 45-52 has
 * no lines for them, so grouping by position closes the gap up and every
 * residue after it is numbered eight too low. The number plus its insertion
 * code is the only thing in the file that survives a hole.
 *
 * @param {string} text a PDB
 * @param {string} [chain] which chain; the first one found if absent
 * @returns {{chain: string, residues: {number: string, code: string,
 *            atoms: Map<string, number[]>}[], sequence: string}}
 */
export function chainResidues(text, chain) {
  const atoms = coordinateAtoms(text);
  // ...the first chain that has a polymer residue on it, not the first chain
  // of any kind: a structure whose first ATOM is a ligand would otherwise name
  // a chain with nothing in it.
  const wanted = chain ?? atoms.chains[atoms.hetero?.findIndex((h) => !h) ?? 0]
    ?? atoms.chains[0];
  const byNumber = new Map();
  const order = [];
  for (let index = 0; index < atoms.points.length; index += 1) {
    if (atoms.chains[index] !== wanted) continue;
    const name = atoms.residueNames[index];
    if (atoms.hetero?.[index] === true && HETERO_RESIDUES[name] === undefined) continue;
    const number = atoms.residues[index];
    if (!byNumber.has(number)) {
      const residue = {
        number,
        code: ONE_LETTER[name] ?? HETERO_RESIDUES[name] ?? "X",
        atoms: new Map(),
        // 🔴 AlphaFold DB PUTS pLDDT IN THE B-FACTOR, which is the only thing
        // in a predicted structure that says "I am not sure about this". A
        // crystal structure says it by having no atoms; a prediction has every
        // residue and no way to say it, so a disordered tail at pLDDT 30 would
        // otherwise be handed over as geometry. See filterByConfidence.
        confidence: 0,
        atomCount: 0,
      };
      byNumber.set(number, residue);
      order.push(residue);
    }
    // ...first occurrence wins, which is the A altLoc in a file that has them.
    const residue = byNumber.get(number);
    if (!residue.atoms.has(atoms.names[index])) {
      residue.atoms.set(atoms.names[index], atoms.points[index]);
      // ...a running mean, because a residue's atoms all carry the same pLDDT
      // in an AlphaFold DB file and an average is right either way.
      residue.atomCount += 1;
      residue.confidence += (atoms.bFactors[index] - residue.confidence)
        / residue.atomCount;
    }
  }
  return {
    chain: wanted,
    residues: order,
    sequence: order.map((residue) => residue.code).join(""),
  };
}

/**
 * One template slot over a query chain's tokens.
 *
 * @param {object} options
 * @param {{residues: object[]}} options.structure from chainResidues
 * @param {number} options.tokens the QUERY's token count
 * @param {Map<number, number>|number[][]} options.map query token index ->
 *   index into `structure.residues`. Everything not named is uncovered.
 * @param {(residue: number) => number} [options.tokenOf] which TOKEN a chain's
 *   residue occupies. Defaults to `residue + offset`, which is right only while
 *   one residue is one token.
 * @param {number} [options.offset] the first query token of this chain.
 *
 * 🔴 A RESIDUE IS NOT A TOKEN ONCE A CHAIN CARRIES A MODIFICATION. AF3 gives a
 * modified residue one token PER ATOM, so a chain with a phosphoserine in it is
 * longer in tokens than in residues, and every chain after it starts later than
 * a residue count says. `tokenOf` is how a caller that has featurised hands
 * over the real layout; the offset is the shortcut for when it has not.
 * @returns {{aatype: Int32Array, atomPositions: Float32Array,
 *            atomMask: Float32Array, covered: number, atoms: number}}
 */
export function templateSlot({ structure, tokens, map, offset = 0, tokenOf }) {
  const at = tokenOf ?? ((residue) => residue + offset);
  const aatype = new Int32Array(tokens).fill(GAP_AATYPE);
  const atomPositions = new Float32Array(tokens * NUM_DENSE * 3);
  const atomMask = new Float32Array(tokens * NUM_DENSE);
  let covered = 0;
  let atoms = 0;

  for (const [queryIndex, templateIndex] of map instanceof Map ? map : new Map(map)) {
    const token = at(queryIndex);
    const residue = structure.residues[templateIndex];
    if (token < 0 || token >= tokens || residue === undefined) continue;

    aatype[token] = aatypeFor(residue.code);
    // 🔴 THE DENSE SLOT IS THE ATOM'S POSITION IN ITS OWN CONFORMER, so a
    // residue's atoms land where the query's would. An atom the conformer does
    // not name - an OXT, a hydrogen the file kept, an alternate naming - is
    // DROPPED rather than given a slot, because a slot is a meaning and the
    // wrong one would put a side-chain atom where a backbone one belongs.
    const layout = conformerFor(residue.code, false);
    for (let slot = 0; slot < layout.length && slot < NUM_DENSE; slot += 1) {
      const point = residue.atoms.get(layout[slot][1]);
      if (point === undefined) continue;
      const base = (token * NUM_DENSE + slot) * 3;
      atomPositions[base] = point[0];
      atomPositions[base + 1] = point[1];
      atomPositions[base + 2] = point[2];
      atomMask[token * NUM_DENSE + slot] = 1;
      atoms += 1;
    }
    covered += 1;
  }
  return { aatype, atomPositions, atomMask, covered, atoms };
}

/**
 * The same slot, in AF2's **atom37** layout rather than AF3's dense 24.
 *
 * 🔴 THE TWO LAYOUTS ARE NOT INTERCHANGEABLE AND THE ARRAYS LOOK ALIKE. Dense
 * indexes an atom by its position in that RESIDUE's own conformer, so slot 4 is
 * whatever the fifth atom of this residue happens to be; atom37 indexes by
 * NAME, globally, so CB is slot 3 for everything that has one. Handing a dense
 * slot to `AF2_ATOM37_MONOMER` - whose `pseudoBeta` is 3 and whose `backbone`
 * is [2, 1, 0], meaning C, CA, N - would read the wrong atoms with no error and
 * produce a plausible, wrong distogram.
 *
 * 🔴 AND AN ATOM THE TABLE DOES NOT NAME IS DROPPED, for the reason the dense
 * builder gives: a slot is a meaning, and the wrong one puts a side-chain atom
 * where a backbone one belongs. A hydrogen the file kept, or an alternate
 * naming, is not given a home.
 *
 * @returns {{aatype: Int32Array, atomPositions: Float32Array,
 *            atomMask: Float32Array, covered: number, atoms: number}}
 */
export function templateSlotAtom37({ structure, tokens, map, offset = 0, tokenOf }) {
  const at = tokenOf ?? ((residue) => residue + offset);
  const slots = ATOM37.length;
  const aatype = new Int32Array(tokens).fill(GAP_AATYPE);
  const atomPositions = new Float32Array(tokens * slots * 3);
  const atomMask = new Float32Array(tokens * slots);
  let covered = 0;
  let atoms = 0;

  for (const [queryIndex, templateIndex] of map instanceof Map ? map : new Map(map)) {
    const token = at(queryIndex);
    const residue = structure.residues[templateIndex];
    if (token < 0 || token >= tokens || residue === undefined) continue;
    aatype[token] = aatypeFor(residue.code);
    for (let slot = 0; slot < slots; slot += 1) {
      const point = residue.atoms.get(ATOM37[slot]);
      if (point === undefined) continue;
      const base = (token * slots + slot) * 3;
      atomPositions[base] = point[0];
      atomPositions[base + 1] = point[1];
      atomPositions[base + 2] = point[2];
      atomMask[token * slots + slot] = 1;
      atoms += 1;
    }
    covered += 1;
  }
  return { aatype, atomPositions, atomMask, covered, atoms };
}

/**
 * The query-to-template map for a structure whose sequence IS the query's.
 *
 * 🔴 THE COMMON CASE, AND THE ONE WITH NOTHING TO GET WRONG. When the entity's
 * sequence was filled in FROM this structure the two agree residue for
 * residue, so the map is the identity over the residues the file resolved and
 * no alignment is involved at all. Everything else - a construct with a tag, a
 * homolog, a sequence the user typed themselves - needs a real alignment, and
 * that is a different function with a different failure mode.
 *
 * @param {{residues: object[]}} structure
 * @returns {Map<number, number>}
 */
export function identityMap(structure) {
  const map = new Map();
  for (let index = 0; index < structure.residues.length; index += 1) map.set(index, index);
  return map;
}

/**
 * Drop residues the structure is not confident about.
 *
 * 🔴 FOR AlphaFold DB, WHERE EVERY RESIDUE IS PRESENT AND SOME ARE GUESSES. A
 * crystal structure says "I did not see this" by having no atoms; a predicted
 * one has no way to say it and puts a low pLDDT in the B-factor instead. Fed
 * as geometry, a disordered tail at pLDDT 30 is noise the model is told to
 * believe. The mask expresses "not known" natively, so this is a filter on the
 * map rather than anything new.
 *
 * @param {Map<number, number>} map
 * @param {{residues: object[]}} structure
 * @param {number} minimum pLDDT below which a residue is dropped
 * @param {(residue: object) => number} confidenceOf
 * @returns {Map<number, number>}
 */
export function filterByConfidence(map, structure, minimum, confidenceOf) {
  if (!(minimum > 0)) return map;
  const kept = new Map();
  for (const [query, template] of map) {
    const residue = structure.residues[template];
    if (residue !== undefined && confidenceOf(residue) >= minimum) kept.set(query, template);
  }
  return kept;
}

/**
 * Fold several per-chain slots into ONE slot covering all of them.
 *
 * 🔴 BECAUSE A PER-CHAIN TEMPLATE CANNOT CARRY AN INTERFACE, AND AN INTERFACE
 * IS THE ONLY THING A COMPLEX HAS THAT A MONOMER DOES NOT. `multichainMaskFor`
 * opens a cross-chain pair only where `spanChains` is set AND the slot covers
 * BOTH ends, so two slots each holding one chain contribute nothing across the
 * boundary however the flag is set - each one covers one end and not the other.
 * A single slot holding both chains, in their deposited relative placement, is
 * the arm that tells the model where they sit.
 *
 * The slots must cover DISJOINT tokens, which is what per-chain slots built
 * from one batch do; an overlap is a caller error and raises rather than
 * letting the later slot win silently.
 *
 * @param {Array<{aatype: Int32Array, atomPositions: Float32Array,
 *                atomMask: Float32Array, covered: number, atoms: number}>} slots
 * @returns {object} one slot of the same shape
 */
export function mergeTemplateSlots(slots) {
  if (slots.length === 0) throw new Error("no template slots to merge");
  if (slots.length === 1) return slots[0];
  const tokens = slots[0].aatype.length;
  const merged = {
    aatype: new Int32Array(tokens).fill(GAP_AATYPE),
    atomPositions: new Float32Array(tokens * NUM_DENSE * 3),
    atomMask: new Float32Array(tokens * NUM_DENSE),
    covered: 0,
    atoms: 0,
  };
  const taken = new Uint8Array(tokens);
  for (const slot of slots) {
    if (slot.aatype.length !== tokens) {
      throw new Error(`template slots disagree on token count: ${slot.aatype.length}`
        + ` against ${tokens}`);
    }
    for (let token = 0; token < tokens; token += 1) {
      // A token this slot does not cover contributes nothing - and "covers" is
      // the ATOM MASK and not the aatype, because a residue whose atoms were
      // all dropped is uncovered however it is typed.
      let any = false;
      for (let slotIndex = 0; slotIndex < NUM_DENSE; slotIndex += 1) {
        if (slot.atomMask[token * NUM_DENSE + slotIndex] > 0) { any = true; break; }
      }
      if (!any) continue;
      if (taken[token]) throw new Error(`two template slots both cover token ${token}`);
      taken[token] = 1;
      merged.aatype[token] = slot.aatype[token];
      for (let slotIndex = 0; slotIndex < NUM_DENSE; slotIndex += 1) {
        const at = token * NUM_DENSE + slotIndex;
        merged.atomMask[at] = slot.atomMask[at];
        for (let axis = 0; axis < 3; axis += 1) {
          merged.atomPositions[at * 3 + axis] = slot.atomPositions[at * 3 + axis];
        }
      }
      merged.covered += 1;
      merged.atoms += slot.atomMask.slice(token * NUM_DENSE, (token + 1) * NUM_DENSE)
        .reduce((sum, v) => sum + (v > 0 ? 1 : 0), 0);
    }
  }
  return merged;
}
