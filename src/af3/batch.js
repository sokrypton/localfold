/**
 * One place where a sequence and an A3M become AF3's batch.
 *
 * Kept out of msa-features.js so that module stays a parser: this one reaches
 * the featuriser and the fold's seeded uniform, which msa-features.js must not
 * depend on.
 */
import { af3MsaFromA3m } from "./msa-features.js";
import { featuriseProtein } from "./featurise.js";
import { uniformFrom } from "./fold.js";

/**
 * A sequence plus an A3M, as the batch AF3's graph is actually fed.
 *
 * 🔴 THIS EXISTS BECAUSE THE PAGE AND THE CLI BUILT IT DIFFERENTLY, AND THE
 * DIFFERENCE WAS INVISIBLE. Both called `af3MsaFromA3m` and then
 * `featuriseProtein`, and each dropped something the other passed:
 *
 *   - the CLI never passed `profileMsa`/`profileDeletionMatrix`, so it profiled
 *     the 127 CROPPED rows where the page profiles all 8076. That is a
 *     different feature, not a different sample of one, and it fed `target_feat`
 *     as well as the profile itself.
 *   - the CLI never seeded the row subsample, so it took the alignment's PREFIX
 *     where the page takes a seeded random subset of the whole file.
 *
 * So every `--a3m` gate was measuring a fold the site does not run, and the one
 * bug that only the page had - boltz2's trunk truncated to 48 of its 64 blocks
 * by a hardcoded depth in web/af3-model.js - could not be reproduced by the
 * tool written to reproduce the page. Both call this now, so a divergence is a
 * change to this function rather than an omission in one caller.
 *
 * @param {string} sequence chains joined by ":"
 * @param {?(string|{paired?: ?string, unpaired?: ?string, unpairedProfile?: ?string})} alignment
 *   null for a single-sequence fold, which is what AF3 itself does with none.
 * @param {{maxSequences?: number, seed?: number, prefixRows?: boolean,
 *          ligands?: Array, modifications?: Array, symmetriseBonds?: boolean,
 *          chainKinds?: string[]}} [options]
 *   `prefixRows` is the control arm: the deterministic prefix the CLI used to
 *   take, for comparing against a recorded baseline. It is not a fold the page
 *   can produce.
 * @returns {{batch: object, rows: object}}
 */
export function af3BatchFromA3m(sequence, alignment, options = {}) {
  const rows = alignment === null || alignment === undefined
    ? { msa: [], deletionMatrix: [], depth: 1, unpairedFrom: 0 }
    : af3MsaFromA3m(alignment, {
      maxSequences: options.maxSequences,
      // 🔴 SEEDED FROM THE FOLD'S OWN SEED, so a subsample is part of what a
      // seed names. AF3 draws its shuffle from the same key that drives the
      // rest of the model; here two seeds are two alignments as well as two
      // starting draws, and one seed is reproducible.
      ...(options.prefixRows ? {} : { random: uniformFrom(options.seed ?? 0) }),
    });
  const batch = featuriseProtein(sequence, {
    ...(options.ligands === undefined ? {} : { ligands: options.ligands }),
    ...(options.modifications === undefined ? {} : { modifications: options.modifications }),
    ...(options.symmetriseBonds === undefined
      ? {} : { symmetriseBonds: options.symmetriseBonds }),
    ...(options.chainKinds === undefined ? {} : { chainKinds: options.chainKinds }),
    // 🔴 CENTRE_REF_CONFORMERS: every family but stock AlphaFold 3 expects the
    // reference conformers centred per ref_space_uid. See featurise.js.
    ...(options.centreRefConformers === undefined
      ? {} : { centreRefConformers: options.centreRefConformers }),
    // 🔴 PADDED_KEYS: rf3, opendde and protenix CLAMP the atom key window and
    // mask its out-of-range slots where AlphaFold 3 slides it in bounds. See
    // `paddedAtomKeys` in dialect.js.
    ...(options.paddedAtomKeys === undefined
      ? {} : { paddedAtomKeys: options.paddedAtomKeys }),
    ...(options.qblockAtomKeys === undefined
      ? {} : { qblockAtomKeys: options.qblockAtomKeys }),
    // 🔴 THE ATOMISED-TOKEN CONVENTIONS, all four inert until a batch has a
    // MODIFIED RESIDUE or a LIGAND in it - which is exactly why they went
    // unnoticed, and why tools/check-batch-fields.js grew a second target.
    // `atomizedElementNames` renames an atomised atom to its element symbol
    // (rf3); `atomizedUnknownRestype` gives an atomised token the UNKNOWN
    // restype (boltz2 and rf3); `atomizedUnknownMsa` carries that into the
    // alignment too (rf3 alone - boltz2 keeps the parent residue there);
    // `atomizedBackboneBonds` bonds an atomised residue back into the chain
    // (rf3 alone).
    ...(options.atomizedElementNames === undefined
      ? {} : { atomizedElementNames: options.atomizedElementNames }),
    ...(options.atomizedUnknownRestype === undefined
      ? {} : { atomizedUnknownRestype: options.atomizedUnknownRestype }),
    ...(options.atomizedUnknownMsa === undefined
      ? {} : { atomizedUnknownMsa: options.atomizedUnknownMsa }),
    ...(options.atomizedBackboneBonds === undefined
      ? {} : { atomizedBackboneBonds: options.atomizedBackboneBonds }),
    // 🔴 DROP_ATOMS: four families carry no terminal OXT and no 5' OP3. The
    // featuriser's switch is `terminalAtoms`, which ESMFold2 already used; this
    // is the same knob under the dialect's name. See dialect.js.
    ...(options.dropTerminalAtoms === true ? { terminalAtoms: false } : {}),
    // 🔴 AND THE QUERY TWICE, for the three families whose paired and unpaired
    // blocks each contribute it. Only where the alignment is empty; see
    // `duplicateQueryRow` in dialect.js.
    // 🔴 THE INVERSION HAPPENS HERE AND NOWHERE ELSE. The dialect names the
    // reference's convention (`dedupeSelfMsa`, which stock AlphaFold 3 does
    // NOT do) and the featuriser names the behaviour (`duplicateQueryRow`), so
    // one of them has to read the other way round; doing it once, at the
    // boundary, is what stops a caller getting it backwards.
    ...(options.dedupeSelfMsa === undefined
      ? {} : { duplicateQueryRow: options.dedupeSelfMsa === false }),
    msa: rows.msa,
    deletionMatrix: rows.deletionMatrix,
    unpairedFrom: rows.unpairedFrom,
    // The profile's rows, which are not the MSA's: AF3 computes the profile
    // before deduplicating the unpaired block against the paired one and before
    // cropping either. See af3MsaFromA3m.
    profileMsa: rows.profileMsa,
    profileDeletionMatrix: rows.profileDeletionMatrix,
  });
  return { batch, rows };
}
