/**
 * design.js
 * ---------
 * AI Context: THE PROTEIN HUNTER PAGE'S CONTROLS
 *
 * 🔴 THE LOOP IS NOT WIRED. This module makes the controls real and says so
 * when Hunt is pressed. What IS real here is everything that can be real
 * without a GPU run: the target list, the designer's automatic choice (through
 * `chooseDesigner`, the same rule the loop will use), the folder count advice,
 * and whether this browser can run a hunt at all.
 *
 * 🔴 WHY THE AUTOMATIC DESIGNER WRITES ITS REASON OUT. A picker that silently
 * changes its own value is worse than one that cannot - `cpu/design/designers.js`
 * says so in those words, and returns a `why` for exactly this. The note under
 * the picker is that `why`.
 *
 * See docs/DESIGN.md for the plan this page is step one of.
 */
import { createEntityList } from "./entity-ui.js";
import { chooseDesigner, DESIGNERS } from "../cpu/design/designers.js";

const element = (id) => {
  const value = document.getElementById(id);
  if (value === null) throw new Error(`missing element #${id}`);
  return value;
};

/** The two counts `chooseDesigner` reads, from the entity list's own rows. */
const contentsOf = (entities) => ({
  ligands: entities.filter((e) => e.type === "ligand" || e.type === "smiles").length,
  nucleic: entities.filter((e) => e.type === "dna" || e.type === "rna").length,
});

const designerSelect = element("designer");
const designerNote = element("designer-note");

/**
 * What the Designer picker says underneath itself.
 *
 * On Auto this is the rule's own reason. On an explicit choice it is the
 * checkpoint's description, plus a warning where that choice cannot see part of
 * the target - picking ProteinMPNN against a ligand is legal and loses the
 * ligand, and the page should say which.
 */
const paintDesignerNote = (entities) => {
  const contents = contentsOf(entities);
  if (designerSelect.value === "auto") {
    const { name, why } = chooseDesigner(contents);
    designerNote.textContent = `Auto → ${DESIGNERS[name].label}, because ${why}.`;
    designerNote.classList.remove("warn");
    return;
  }
  const chosen = DESIGNERS[designerSelect.value];
  const blind = [];
  // 🔴 NAMED, NOT IMPLIED. These are the cases where an explicit pick drops
  // something that is in the target: only LigandMPNN reads heteroatoms, and
  // only NA-MPNN has a nucleic alphabet.
  if (contents.ligands > 0 && designerSelect.value !== "ligand") blind.push("the ligand");
  if (contents.nucleic > 0 && designerSelect.value !== "na" && designerSelect.value !== "ligand") {
    blind.push("the DNA or RNA chain");
  }
  // The checkpoint's own description, authored beside the registry rather than
  // written a second time here, plus whatever this choice cannot see.
  designerNote.textContent = blind.length === 0
    ? `${chosen.label}. ${chosen.note}`
    : `${chosen.label} — it will not see ${blind.join(" or ")}. ${chosen.note}`;
  designerNote.classList.toggle("warn", blind.length > 0);
};

const folders = element("folders");
const folderNote = element("folder-note");

/**
 * What the Folders picker says underneath itself.
 *
 * 🔴 THE ADVICE IS SOMEBODY ELSE'S MEASUREMENT AND IS LABELLED AS SUCH. Yehlin
 * measured two or three folders beating one, and five or six doing worse than
 * two, on her stack with her judge (docs/DESIGN.md 3C). Nothing here has
 * reproduced it, so the page advises and does not enforce.
 */
const paintFolderNote = () => {
  const n = Array.from(folders.selectedOptions).length;
  if (n === 0) {
    folderNote.textContent = "Pick at least one model to fold with.";
    folderNote.classList.add("warn");
    return;
  }
  const rotation = n === 1 ? "One model folds every cycle." : `${n} models, rotating one per cycle.`;
  let advice = "";
  if (n === 1) advice = " Two or three usually beat one: a design has to satisfy all of them.";
  else if (n >= 5) advice = " Five or more did worse than two in the measurement this follows — weak members get an equal vote.";
  folderNote.textContent = rotation + advice;
  folderNote.classList.toggle("warn", n >= 5);
};

const entities = createEntityList(
  element("entity-rows"), element("add-entity"),
  {
    // Something to design against on arrival, as index.html ships a foldable
    // sequence: PD-L1's binding domain, the target Yehlin's ensemble numbers
    // were measured on.
    initial: [{ type: "protein", copies: 1,
      value: "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA" }],
    onChange: () => paintDesignerNote(entities.read()),
    // This page has no MSA control, so an automatic template has nothing to
    // draw a hit from. Answer honestly rather than letting the list assume.
    msaIsSearch: () => false,
  },
);

designerSelect.addEventListener("change", () => paintDesignerNote(entities.read()));
folders.addEventListener("change", paintFolderNote);
paintDesignerNote(entities.read());
paintFolderNote();

// 🔴 WHETHER A HUNT CAN RUN HERE AT ALL, SAID BEFORE IT IS TRIED. A hunt is
// many folds; finding out on cycle one that this browser has no WebGPU is the
// worst time to find out.
const gpuSummary = element("gpu-summary");
if (navigator.gpu === undefined) {
  gpuSummary.textContent = "WebGPU unavailable — this page needs Chrome or Edge with WebGPU to fold.";
  gpuSummary.classList.add("warn");
} else {
  navigator.gpu.requestAdapter().then((adapter) => {
    const info = adapter?.info;
    gpuSummary.textContent = adapter === null || adapter === undefined
      ? "WebGPU present but no adapter answered."
      : `WebGPU · ${[info?.vendor, info?.architecture].filter(Boolean).join(" / ") || "adapter available"}`;
  }).catch((error) => {
    gpuSummary.textContent = `WebGPU could not be queried: ${error.message}`;
  });
}

// 🔴 THE HONEST BUTTON. It is not disabled, because a disabled control with no
// explanation is the thing this page is supposed to avoid. It is pressable and
// it says exactly what is missing.
const status = element("status-message");
element("hunt").addEventListener("click", () => {
  status.textContent = "The design loop is not wired yet — this page is the shell. See docs/DESIGN.md.";
  status.classList.add("error");
});
