/**
 * 🔴 LOCALFOLD EDIT: THE RESIDUE LISTS IN THESE SELECTIONS ARE BRACKETED.
 *
 * `evalPathSelection` splits a selection on COMMAS before it splits a path on
 * slashes, so a selection whose residue field reads `16-35,52-71` is TWO tokens
 * - the path up to the first range, and then a stray `52-71` - of which only
 * the first is a path at all; and an unbracketed range reaches
 * `parseInt('16-35')` and becomes the single residue 16. Measured on
 * motif_idx_1prw: the bare form selects residue 16 alone where the bracketed
 * form selects 16-35. `ppi_pdl1`'s four hotspots were written the same way.
 *
 * 🔴 AND ONE QUESTION THIS DOES NOT ANSWER, for whoever owns these
 * upstream: bracketed, `[16-35,52-71]` still selects only 16-35, because the
 * example's chain A is renumbered and has no residue 52 to 71 (its given
 * segments are A1-20 and A22-41). So the numbering in these conditions may
 * belong to the original 1PRW rather than to the cropped example. Bracketing is
 * strictly closer to what the presets SAY they condition on; whether the numbers
 * themselves are right is not something this repository can settle.
 *
 * See test/rfd4p-selection.test.js, which pins the DSL behaviour.
 */
/**
 * RFD4-Proteina interactive design presets covering all upstream benchmark families
 * (Unconditional Monomer, PPI Binder w/ Hotspots, Indexed & Unindexed Motif Scaffolding,
 * Small-Molecule Pocket, DNA Binder, Partial Diffusion, Inverse Folding, and YAML/Contig specs).
 */

export const RFD4P_PRESETS = Object.freeze([
  {
    id: 'uncond_short',
    title: 'Unconditional Monomer (100 aa · monomer-short.cif.zst)',
    category: 'Unconditional',
    description:
      'De novo 100-residue monomer generation from pure Gaussian noise using the upstream annotated benchmark CIF.',
    cifUrl: './web/examples/uncond_short.cif.zst',
    contig: '100-100',
    nsteps: 60,
    recycleSchedule: 'low',
    conditions: {},
  },
  {
    id: 'ppi_pdl1',
    title: 'PD-L1 PPI Binder w/ Hotspots (208 tokens · pdl1.cif.zst)',
    category: 'Protein Binder',
    description:
      'Design a de novo binder against PD-L1 with target hotspot conditioning (C_HOT), internal coordinate group (C_CRDGRP), and locked target sequence (C_SEQ).',
    cifUrl: './web/examples/ppi_pdl1.cif.zst',
    contig: 'A18-132,/0,76-76',
    nsteps: 60,
    recycleSchedule: 'low',
    conditions: {
      C_CRD: { true: [{ select: "sel('A/*/*/*')" }] },
      C_SEQ: { true: [{ select: "sel('A/*/*/*')" }] },
      C_HOT: { true: [{ select: "sel('A/*/[54,56,66,115]/*')" }] },
    },
  },
  {
    id: 'motif_idx_1prw',
    title: 'Indexed Motif Scaffolding — 1PRW EF-Hand (76 aa)',
    category: 'Motif Scaffolding',
    description:
      'Scaffold two EF-hand calcium-binding loops (A16-35, A52-71) at indexed sequence positions with expandable variable-length linkers.',
    cifUrl: './web/examples/motif_idx_1prw.cif.zst',
    contig: '5-20,A16-35,10-25,A52-71,5-20',
    nsteps: 60,
    recycleSchedule: 'low',
    conditions: {
      C_CRDGRP: { 1: [{ select: "sel('A/*/[16-35,52-71]/*')" }] },
      C_SEQ: { true: [{ select: "sel('A/*/[16-35,52-71]/*')" }] },
    },
  },
  {
    id: 'motif_unidx_1prw',
    title: 'Unindexed Motif Scaffolding — 1PRW + Hungarian Insertion',
    category: 'Motif Scaffolding',
    description:
      'Scaffold two unindexed EF-hand motifs (C_IDX=false, is_in_context_conditioning=true) into a de novo scaffold and place them via the trunk insertion head + Hungarian linear sum assignment.',
    cifUrl: './web/examples/motif_unidx_1prw.cif.zst',
    contig: '60-105,A16-35,A52-71',
    nsteps: 60,
    recycleSchedule: 'low',
    conditions: {
      C_IDX: { false: [{ select: "sel('A/*/[16-35,52-71]/*')" }] },
      C_CRDGRP: { 1: [{ select: "sel('A/*/[16-35,52-71]/*')" }] },
      C_SEQ: { true: [{ select: "sel('A/*/[16-35,52-71]/*')" }] },
    },
  },
  {
    id: 'sm_sam',
    title: 'Small-Molecule Pocket — S-Adenosylmethionine (SAM_hbonds.cif.zst)',
    category: 'Small Molecule',
    description:
      'Design a binding pocket around atomized S-adenosylmethionine (33 <A> ligand tokens) conditioned on H-bond donors/acceptors (C_HB) and ligand coordinates.',
    cifUrl: './web/examples/sm_sam.cif.zst',
    contig: '110-110,/0,SAM',
    nsteps: 60,
    recycleSchedule: 'low',
    conditions: {
      C_CRD: { true: [{ select: "sel('B/*/*/*')" }] },
      C_HB: { ACCEPTOR: [{ select: "sel('B/SAM/*/*')" }] },
    },
  },
  {
    id: 'dna_7m5w',
    title: 'DNA-Binding Protein Complex (7m5w_structure_context.cif.zst)',
    category: 'Nucleic Acid',
    description:
      'Design a de novo DNA-binding protein in complex with double-stranded B-DNA (DA, DC, DG, DT nucleotide tokens and Watson-Crick base-pair features).',
    cifUrl: './web/examples/dna_7m5w.cif.zst',
    contig: '96-96,/0,B1-16,/0,C1-16',
    nsteps: 60,
    recycleSchedule: 'low',
    conditions: {
      C_CRD: { true: [{ select: "sel('B/*/*/*') | sel('C/*/*/*')" }] },
      C_SEQ: { true: [{ select: "sel('B/*/*/*') | sel('C/*/*/*')" }] },
    },
  },
  {
    id: 'yaml_5o45_binder',
    title: 'YAML Spec — Binder in 10 Lines (5o45_cropped.cif.zst)',
    category: 'YAML Spec',
    description:
      'Design a 50-120 aa binder against chain A17-131 of 5O45 using the declarative YAML condition_spec DSL (`sel(\'A/*/*/[N,CA,C,O]\')`).',
    cifUrl: './web/examples/5o45_cropped.cif.zst',
    specMode: true,
    contig: '70-70, /0, A17-131',
    nsteps: 60,
    recycleSchedule: 'low',
    conditions: {
      C_CRDGRP: { 1: [{ select: "sel('A/*/*/[N,CA,C,O]')" }] },
      C_SEQ: { true: [{ select: "sel('A')" }] },
    },
  },
  {
    id: 'partial_diffusion_1prw',
    title: 'Partial Diffusion — Diversify 1PRW Scaffold (frac=0.5, AE Encoder)',
    category: 'Partial Diffusion',
    description:
      'Encode clean all-atom coordinates into local latents with Rfd4pEncoderGpu, corrupt to the midpoint of the flow schedule (partialTrajectoryFractionSteps=0.5), and denoise.',
    cifUrl: './web/examples/motif_idx_1prw.cif.zst',
    contig: '5-20,A16-35,10-25,A52-71,5-20',
    nsteps: 60,
    partialTrajectoryFractionSteps: 0.5,
    recycleSchedule: 'low',
    conditions: {},
  },
  {
    id: 'inverse_folding_1prw',
    title: 'Inverse Folding — Resampled Forward Marginal (1PRW)',
    category: 'Inverse Folding',
    description:
      'Fixed-backbone sequence and rotamer co-design by resampling the target CA backbone from its forward marginal at every step while denoising sequence latents.',
    cifUrl: './web/examples/motif_idx_1prw.cif.zst',
    contig: '5-20,A16-35,10-25,A52-71,5-20',
    nsteps: 60,
    inverseFolding: true,
    recycleSchedule: 'low',
    conditions: {},
  },
  {
    id: 'custom_cath_monomer',
    title: 'Custom Contig — CATH 3.40.50 Rossmann + Secondary Structure',
    category: 'Fold-Conditioned',
    description:
      'Generate a de novo 54-residue α/β protein from a pure contig + condition_spec (CATH topology 3.40.50 and explicit helix-strand-helix secondary structure).',
    cifUrl: null,
    specMode: true,
    contig: 'h1:16-16P,l1:4-4P,s1:8-8P,l2:4-4P,h2:14-14P,s2:8-8P',
    nsteps: 60,
    recycleSchedule: 'low',
    conditions: {
      C_CATH: { '3.40.50': [{ select: "sel('*/*/*/*')" }] },
      C_SEC: {
        HELIX: [{ select: "sel('*/h1/*/*') | sel('*/h2/*/*')" }],
        LOOP: [{ select: "sel('*/l1/*/*') | sel('*/l2/*/*')" }],
        SHEET: [{ select: "sel('*/s1/*/*') | sel('*/s2/*/*')" }],
      },
    },
  },
]);

export function getRfd4pPresetById(presetId) {
  return RFD4P_PRESETS.find((p) => p.id === presetId) ?? RFD4P_PRESETS[0];
}
