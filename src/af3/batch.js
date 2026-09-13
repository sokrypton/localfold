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
