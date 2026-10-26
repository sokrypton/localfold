/**
 * Write a shipped preset out as a plain condition spec, which is what the native featuriser takes.
 *
 *     node cuda/rfd4p/featurise/preset-spec.mjs custom_cath_monomer > job.json
 *
 * 🔴 A PRESET IS NOT AN INPUT FORMAT. `presets.js` is the page's menu; RFD4-Proteina's own jobs are
 * `{contig, conditions, nsteps}` objects, and that is what the C++ reads. This turns one into the
 * other so the gate can hold the native featuriser to the JavaScript on a CONDITIONED case without
 * either side learning about the other's menu.
 */
import { getRfd4pPresetById } from "../../../shared/rfd4p/presets.js";

const id = process.argv[2];
const preset = getRfd4pPresetById(id);
if (!preset) {
  console.error(`no preset ${id}`);
  process.exit(1);
}
// The WHOLE job, structure and run mode included: a preset with a `cifUrl` is a job on that
// structure, and partial diffusion and inverse folding are fields of the job like any other. This
// wrote the contig and conditions alone, so a structure preset came out as a different job.
//
// 🔴 AND A PRESET WITHOUT `specMode` IS ITS STRUCTURE, which upstream writes
// `{input, input_as_annotated_cif: true}` with no contig: its contig is in the original entry's
// numbering and names residues the cropped example does not have, and its conditions are already
// the file's annotations. Writing them out made a job that is not the preset.
const asStructure = Boolean(preset.cifUrl) && preset.specMode !== true;
process.stdout.write(JSON.stringify({
  // Absolute, because a spec's relative `input` is read against the SPEC's directory (upstream's
  // `_resolve_input_path`) and this is written wherever the caller puts it.
  ...(preset.cifUrl ? { input: new URL(`../../../${preset.cifUrl.replace(/^\.\//, "")}`, import.meta.url).pathname } : {}),
  ...(asStructure ? { input_as_annotated_cif: true }
    : { contig: preset.contig, conditions: preset.conditions }),
  nsteps: preset.nsteps,
  recycleSchedule: preset.recycleSchedule,
  partialTrajectoryFractionSteps: preset.partialTrajectoryFractionSteps,
  inverseFolding: preset.inverseFolding,
}) + "\n");
