/**
 * The entity list, as rows on the page.
 *
 *     const list = createEntityList(element("entity-rows"), element("add-entity"));
 *     list.read();            // -> [{ type, value, copies }, ...]
 *
 * All of the model is in entities.js; this file is the DOM half and holds no
 * rules about what is foldable. It replaces the single sequence textarea, so it
 * has to keep everything that textarea could do: paste a sequence, paste colon
 * separated chains, paste FASTA, and be rewritten from an alignment's query.
 *
 * 🔴 ROWS ARE REBUILT ON STRUCTURE, NEVER ON TYPING. Re-rendering a row while
 * its field has focus moves the caret to the end, which throws a mid-sequence
 * correction away - the same reason the old box tidied on blur rather than on
 * input. So adding, removing and changing a type rebuild; typing and copies do
 * not, and update the model in place instead.
 */
import { COMMON_IONS, COMMON_LIGANDS, ENTITY_LABELS, ENTITY_TYPES, MENU_CODES, entitiesFromText, newEntity } from "./entities.js";
import { cleanSequence, extractFastaHeader } from "./sequence.js";

/**
 * Wire an entity list into a container.
 *
 * @param {HTMLElement} rowsContainer  where the rows go
 * @param {HTMLElement} addButton      the `+ Add entity` control
 * @param {{onChange?: () => void, initial?: object[]}} [options]
 * @returns {{read: () => object[], set: (entities: object[]) => void,
 *            setChains: (chains: string[]) => void, header: () => string | null}}
 */
export function createEntityList(rowsContainer, addButton, options = {}) {
  const notify = options.onChange ?? (() => {});
  /** @type {{type: string, value: string, copies: number}[]} */
  let entities = (options.initial ?? []).length > 0
    ? options.initial.map((entity) => ({ ...entity }))
    : [newEntity("protein")];
  // The description line of the last FASTA that was pasted, which is where the
  // job name used to come from when this was one textarea holding the record.
  let pastedHeader = null;

  const render = () => {
    rowsContainer.replaceChildren(...entities.map((entity, index) => row(entity, index)));
    notify();
  };

  const row = (entity, index) => {
    const wrapper = document.createElement("div");
    wrapper.className = "entity-row";

    const type = document.createElement("select");
    type.className = "entity-type";
    type.title = "What this entity is";
    for (const value of ENTITY_TYPES) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = ENTITY_LABELS[value];
      type.append(option);
    }
    type.value = entity.type;
    type.addEventListener("change", () => {
      entity.type = type.value;
      // The field itself changes shape with the type - a sequence wants a
      // monospace box that grows, a CCD code wants one small line - so this is
      // a structural change and the row is rebuilt.
      render();
    });

    // 🔴 A LIGAND GETS AN INPUT AND A PROTEIN GETS A TEXTAREA, because a CCD
    // code is five characters and a sequence is hundreds. A single control
    // sized for one is wrong for the other, and the wrong one invites a paste
    // that cannot be read back.
    // 🔴 A MENU IN FRONT OF THE BOX, NOT INSTEAD OF IT. The codes are the part
    // nobody remembers - "the code for heme", "is magnesium MG or MG2" - and
    // the ions are one or two letters, which is exactly what gets typed wrong.
    // But the fold fetches whatever code it is given from the PDB, so the menu
    // must not become the limit: "Custom" is the default, the box beside it
    // still takes anything, and picking from the menu only fills the box in.
    let picker = null;
    if (entity.type === "ligand") {
      picker = document.createElement("select");
      picker.className = "entity-picker";
      picker.title = "Common ligands and ions";
      const custom = document.createElement("option");
      custom.value = "";
      custom.textContent = "Custom…";
      picker.append(custom);
      for (const [label, entries] of [["Ligands", COMMON_LIGANDS], ["Ions", COMMON_IONS]]) {
        const group = document.createElement("optgroup");
        group.label = label;
        for (const entry of entries) {
          const option = document.createElement("option");
          option.value = entry.code;
          option.textContent = `${entry.code} · ${entry.name}`;
          group.append(option);
        }
        picker.append(group);
      }
      // A code typed by hand that happens to be on the menu shows as that entry
      // rather than as Custom, so the two controls never disagree.
      const current = entity.value.trim().toUpperCase();
      picker.value = MENU_CODES.has(current) ? current : "";
    }

    const value = entity.type === "ligand"
      ? document.createElement("input")
      : document.createElement("textarea");
    value.className = `entity-value entity-value-${entity.type}`;
    value.spellcheck = false;
    if (entity.type === "ligand") {
      value.type = "text";
      value.placeholder = "CCD code, e.g. HEM";
      value.setAttribute("aria-label", "Ligand CCD code");
    } else {
      // Two lines by default, and the user can drag it taller. One line was too
      // mean for the thing this box is actually for - a sequence of a few
      // hundred residues shows a sliver of itself - and a ligand, which really
      // is five characters, has its own one-line input above rather than
      // sharing this control.
      value.rows = 2;
      value.placeholder = "Paste a protein sequence";
      value.setAttribute("aria-label", "Protein sequence");
    }
    value.value = entity.value;
    value.addEventListener("input", () => {
      entity.value = value.value;
      // ...the menu follows the box WITHOUT a re-render, which would take the
      // focus and the caret away mid-word.
      if (picker !== null) {
        const typed = value.value.trim().toUpperCase();
        picker.value = MENU_CODES.has(typed) ? typed : "";
      }
      notify();
    });
    if (picker !== null) {
      picker.addEventListener("change", () => {
        if (picker.value === "") { value.focus(); return; }   // Custom: go type one
        entity.value = picker.value;
        value.value = picker.value;
        notify();
      });
    }
    // Tidied on blur, not on every keystroke: see the note at the top.
    value.addEventListener("blur", () => {
      if (entity.type === "ligand") {
        entity.value = value.value.trim().toUpperCase();
      } else {
        entity.value = cleanSequence(value.value);
      }
      if (value.value !== entity.value) value.value = entity.value;
      notify();
    });
    if (entity.type === "protein") {
      value.addEventListener("paste", (event) => {
        const text = event.clipboardData?.getData("text") ?? "";
        // 🔴 PASTING FASTA OR COLONS SPLITS INTO ROWS. This is the one thing
        // the textarea did that a plain field would lose: a user's clipboard
        // holds a multi-record FASTA or an "A:B" complex, and there is no
        // second box to put the rest in. Anything simpler than that - a bare
        // sequence - falls through to the browser's own paste.
        if (!text.includes(":") && !text.trim().startsWith(">")) return;
        event.preventDefault();
        const pasted = entitiesFromText(text);
        if (pasted.length === 0) return;
        if (text.trim().startsWith(">")) pastedHeader = extractFastaHeader(text);
        entities.splice(index, 1, ...pasted);
        render();
      });
    }

    const copiesLabel = document.createElement("label");
    copiesLabel.className = "entity-copies-label";
    copiesLabel.textContent = "copies";
    const copies = document.createElement("input");
    copies.className = "entity-copies";
    copies.type = "number";
    copies.min = "1";
    copies.max = "20";
    copies.step = "1";
    copies.value = String(entity.copies);
    copies.title = "How many identical copies of this entity to fold";
    copies.addEventListener("input", () => {
      // Left as typed while it is being typed; entitiesProblem reports a bad
      // count, and rewriting the field mid-edit fights the user for the caret.
      entity.copies = Number.parseInt(copies.value, 10);
      notify();
    });
    copiesLabel.prepend(copies);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "btn btn-grey btn-small entity-remove";
    remove.title = "Remove this entity";
    remove.setAttribute("aria-label", "Remove this entity");
    remove.innerHTML = '<i class="fa-solid fa-xmark"></i>';
    // The last row stays: an empty list has nothing to type into and no way
    // back except Add entity, which reads as the page having lost the input.
    remove.disabled = entities.length === 1;
    remove.addEventListener("click", () => {
      entities.splice(index, 1);
      if (entities.length === 0) entities.push(newEntity("protein"));
      render();
    });

    // The picker sits between the type and the box, so the row reads
    // "Ligand · [ATP ▾] [ATP] × 1" left to right.
    wrapper.append(type, ...(picker === null ? [] : [picker]), value, copiesLabel, remove);
    if (picker !== null) wrapper.classList.add("entity-row-ligand");
    return wrapper;
  };

  addButton.addEventListener("click", () => {
    entities.push(newEntity("protein"));
    render();
    // Focus the row that was just asked for, so it can be typed into at once.
    rowsContainer.lastElementChild?.querySelector(".entity-value")?.focus();
  });

  render();

  return {
    read: () => entities.map((entity) => ({ ...entity })),
    set: (next) => {
      entities = next.length === 0 ? [newEntity("protein")] : next.map((e) => ({ ...e }));
      render();
    },
    /**
     * Replace the protein rows with these chains, keeping every ligand.
     *
     * This is the alignment-query-wins path: an A3M carries its own first
     * record, and folding the box's sequence against somebody else's alignment
     * would be folding two different proteins at once.
     */
    setChains: (chains) => {
      const ligands = entities.filter((entity) => entity.type === "ligand");
      const proteins = [];
      for (const chain of chains) {
        const existing = proteins.find((row) => row.value === chain);
        if (existing === undefined) proteins.push({ type: "protein", value: chain, copies: 1 });
        else existing.copies += 1;
      }
      entities = [...proteins, ...ligands];
      render();
    },
    header: () => pastedHeader,
  };
}
