/**
 * Which MPNN the page designs with, and how it decides on its own.
 *
 * 🔴 THE AUTOMATIC RULE IS THE REFERENCE'S, NOT AN INVENTION.
 * `boltz_ph/pipeline.py` reads
 *
 *     model_type = "ligand_mpnn" if (ligand_smiles or ligand_ccd
 *                                    or nucleic_seq) else "soluble_mpnn"
 *
 * so "is there anything here that is not protein" is the whole of it. The one
 * thing added below is NA-MPNN: the reference sends a nucleic chain to
 * LigandMPNN, which reads it as loose heteroatoms, and NA-MPNN graphs it as a
 * POLYMER with its own backbone and its own node embedding. Where both would
 * work, the model that knows the chain is a chain is the better answer, and
 * the picker still offers LigandMPNN to anyone who disagrees.
 *
 * 🔴 AND EVERY FAMILY HERE DESIGNS A PROTEIN CHAIN. A ligand and a nucleic
 * chain are CONTEXT the designed chain is read against, never the thing being
 * designed - which is what makes NA-MPNN's 33-letter alphabet safe to use for
 * a protein binder. See designBias() in sample-sequence.js.
 */

/**
 * 🔴 THE FOUR CHECKPOINTS web/public/mpnn/ ACTUALLY CARRIES, at noise 0.2.
 * tools/sync-mpnn.py mirrors exactly this list and test/designers.test.js
 * holds the two together - a name here that has no file is a 404 the page
 * cannot explain, and a file with no name here is 4 MB nobody can reach.
 */
export const DESIGNERS = {
  soluble: {
    file: "solublempnn_v_48_020.mpnn",
    label: "SolubleMPNN",
    note: "ProteinMPNN retrained without membrane proteins. The default:"
      + " a designed binder should be soluble.",
  },
  protein: {
    file: "proteinmpnn_v_48_020.mpnn",
    label: "ProteinMPNN",
    note: "The original. Reach for it to compare against published designs.",
  },
  ligand: {
    file: "ligandmpnn_v_32_020_25.mpnn",
    label: "LigandMPNN",
    note: "Reads heteroatoms as context. The only family whose encoder sees a"
      + " small molecule at all.",
  },
  na: {
    file: "na_mpnn_design.mpnn",
    label: "NA-MPNN",
    note: "Graphs DNA and RNA as polymers rather than as loose atoms.",
  },
};

/** The order the picker lists them in. */
export const DESIGNER_NAMES = ["soluble", "protein", "ligand", "na"];

/**
 * The family to use for a job, given what is in the complex besides protein.
 *
 * @param {{ligands?: number, nucleic?: number}} input how many ligands and how
 *   many DNA or RNA chains the fold carries
 * @returns {{name: keyof typeof DESIGNERS, why: string}} `why` is shown on the
 *   page, because a picker that silently changes its own value is worse than
 *   one that cannot.
 */
export function chooseDesigner(input = {}) {
  const ligands = input.ligands ?? 0;
  const nucleic = input.nucleic ?? 0;
  // 🔴 NUCLEIC FIRST, AND THE ORDER MATTERS FOR A COMPLEX CARRYING BOTH.
  // NA-MPNN has no atom context, so a DNA chain AND a ligand together is the
  // one case where each choice loses something real: NA-MPNN drops the
  // ligand, LigandMPNN reads the DNA as unstructured atoms. It goes to
  // LigandMPNN, because a ligand seen badly is better than a ligand not seen -
  // its atoms are usually IN the site being designed, and a nucleic chain
  // usually is not.
  if (ligands > 0) {
    return {
      name: "ligand",
      why: nucleic > 0
        ? "a ligand is present, and only LigandMPNN reads heteroatoms"
        : "a ligand is present",
    };
  }
  if (nucleic > 0) return { name: "na", why: "a DNA or RNA chain is present" };
  return { name: "soluble", why: "the complex is protein only" };
}
