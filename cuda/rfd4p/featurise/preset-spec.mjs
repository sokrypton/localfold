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
process.stdout.write(JSON.stringify({
  contig: preset.contig,
  conditions: preset.conditions,
  nsteps: preset.nsteps,
}) + "\n");
