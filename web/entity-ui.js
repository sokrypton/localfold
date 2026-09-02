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
import { COMMON_IONS, COMMON_LIGANDS, COMMON_MODIFICATIONS, ENTITY_LABELS, ENTITY_TYPES,
  MENU_CODES, entitiesFromText, entityProblem, newEntity } from "./entities.js";
import { cleanSequence, cleanSequenceMap, extractFastaHeader } from "./sequence.js";

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

  /**
   * The modified-residue popup: one row per modification, a menu of the common
   * ones and a position, plus a free code for anything else.
   *
   * 🔴 ONE POPUP AT A TIME AND IT CLOSES ON ANYTHING ELSE. Left open it sits
   * over the row below and takes clicks meant for it, which reads as the page
   * having frozen rather than as a menu being open.
   */
  // Repainting the highlight of a row from outside it, when the popup changes
  // the modification list.
  const entityPaint = new WeakMap();
  let closePopup = null;
  const openModifications = (anchor, entity, index) => {
    if (closePopup !== null) { closePopup(); return; }
    const popup = document.createElement("div");
    popup.className = "entity-popup";
    popup.setAttribute("role", "dialog");
    popup.setAttribute("aria-label", "Modified residues");

    // 🔴 A CLICK ON THE SEQUENCE MEANS "THIS RESIDUE", AND IT HAS TO MEAN IT
    // FOR ONE ROW. Typing a number is the part people get wrong - counting to
    // 147 by eye - so the box itself becomes the position picker, and the row
    // it moves is the one last touched. Without a chosen row a click would
    // either do nothing or move all of them.
    let active = Math.max(0, (entity.modifications ?? []).length - 1);
    const draw = () => {
      // 🔴 NOT render(). The popup is a CHILD of the row, so rebuilding the row
      // deletes the popup out from under itself - clicking "Add modification"
      // closed the menu and left the modification behind, which reads as the
      // button having failed. The one thing outside the popup that has to keep
      // up is the badge on the button, so that is updated by hand.
      const badge = anchor.querySelector(".entity-options");
      const count = (entity.modifications ?? []).length;
      if (badge !== null) {
        if (count > 0) badge.dataset.count = String(count);
        else delete badge.dataset.count;
        badge.title = count === 0
          ? "Modified residues" : `${count} modified residue${count === 1 ? "" : "s"}`;
      }
      entityPaint.get(entity)?.();
      popup.replaceChildren();
      const head = document.createElement("div");
      head.className = "entity-popup-head";
      const title = document.createElement("div");
      title.className = "entity-popup-title";
      title.textContent = "Modified residues";
      // 🔴 CLOSED ON PURPOSE, NOT BY LOOKING AWAY. It used to dismiss on any
      // click outside itself, which fought the one thing it is for: clicking
      // the sequence to pick a position happens OUTSIDE the popup, so the
      // gesture that sets a residue was also the gesture that closed the menu.
      const shut = document.createElement("button");
      shut.type = "button";
      shut.className = "entity-popup-close";
      shut.title = "Close";
      shut.setAttribute("aria-label", "Close");
      shut.innerHTML = '<i class="fa-solid fa-xmark"></i>';
      shut.addEventListener("click", () => closePopup?.());
      head.append(title, shut);
      popup.append(head);

      const sequence = cleanSequence(entity.value);
      entity.modifications ??= [];
      if (entity.modifications.length === 0) {
        const empty = document.createElement("p");
        empty.className = "entity-popup-empty";
        empty.textContent = sequence.length === 0
          ? "Paste a sequence first, then add a modification to one of its residues."
          : "None. A modification replaces one residue with a PDB component.";
        popup.append(empty);
      }

      entity.modifications.forEach((modification, at) => {
        const line = document.createElement("div");
        line.className = "entity-popup-row";

        const menu = document.createElement("select");
        menu.className = "entity-popup-code";
        const custom = document.createElement("option");
        custom.value = "";
        custom.textContent = "Custom";
        menu.append(custom);
        for (const entry of COMMON_MODIFICATIONS) {
          const option = document.createElement("option");
          option.value = entry.code;
          option.textContent = `${entry.code} · ${entry.name}`;
          menu.append(option);
        }
        const code = (modification.code ?? "").toUpperCase();
        menu.value = COMMON_MODIFICATIONS.some((entry) => entry.code === code) ? code : "";

        const typed = document.createElement("input");
        typed.type = "text";
        typed.className = "entity-popup-typed";
        typed.value = modification.code ?? "";
        typed.placeholder = "CCD";
        typed.setAttribute("aria-label", "Modification CCD code");

        const at1 = document.createElement("span");
        at1.className = "entity-popup-at";
        at1.textContent = "at";

        const position = document.createElement("input");
        position.type = "number";
        position.className = "entity-popup-position";
        position.min = "1";
        if (sequence.length > 0) position.max = String(sequence.length);
        position.value = modification.position ? String(modification.position) : "";
        position.setAttribute("aria-label", "Residue position");

        // What is actually at that position, so a wrong number is visible
        // before the fold rather than as a validation message after it.
        const residue = document.createElement("span");
        residue.className = "entity-popup-residue";
        const letter = sequence[modification.position - 1];
        residue.textContent = letter === undefined ? "" : letter;

        const drop = document.createElement("button");
        drop.type = "button";
        drop.className = "btn btn-grey btn-small";
        drop.title = "Remove this modification";
        drop.setAttribute("aria-label", "Remove this modification");
        drop.innerHTML = '<i class="fa-solid fa-xmark"></i>';

        menu.addEventListener("change", () => {
          if (menu.value === "") { typed.focus(); return; }
          modification.code = menu.value;
          // ...and jump the position to the first residue this one can go on,
          // when the current one cannot take it.
          const entry = COMMON_MODIFICATIONS.find((e) => e.code === menu.value);
          if (entry && sequence[modification.position - 1] !== entry.parent) {
            const first = sequence.indexOf(entry.parent);
            modification.position = first < 0 ? modification.position : first + 1;
          }
          notify();
          draw();
        });
        typed.addEventListener("input", () => {
          modification.code = typed.value.trim().toUpperCase();
          notify();
        });
        typed.addEventListener("blur", () => { notify(); draw(); });
        position.addEventListener("input", () => {
          modification.position = Number.parseInt(position.value, 10);
          residue.textContent = cleanSequence(entity.value)[modification.position - 1] ?? "";
          notify();
        });
        drop.addEventListener("click", () => {
          entity.modifications.splice(at, 1);
          notify();
          draw();
        });

        if (at === active) line.classList.add("entity-popup-row-active");
        for (const control of [menu, typed, position]) {
          control.addEventListener("focus", () => {
            if (active === at) return;
            active = at;
            draw();
          });
        }
        line.addEventListener("mousedown", () => { active = at; });

        line.append(menu, typed, at1, position, residue, drop);
        popup.append(line);
      });

      if (entity.modifications.length > 0) {
        const hint = document.createElement("p");
        hint.className = "entity-popup-hint";
        hint.textContent = "Click a residue in the sequence to set the position.";
        popup.append(hint);
      }

      const add = document.createElement("button");
      add.type = "button";
      add.className = "btn btn-grey btn-small entity-popup-add";
      add.textContent = "+ Add modification";
      add.addEventListener("click", () => {
        // 🔴 THE FIRST FREE RESIDUE, NOT THE FIRST ONE. Defaulting to the first
        // match put every new row on the same residue, so adding a second
        // modification greeted the user with "Two modifications on residue 12"
        // before they had touched anything.
        const letters = cleanSequence(entity.value);
        const taken = new Set(entity.modifications.map((one) => one.position));
        let chosen = null;
        for (const entry of COMMON_MODIFICATIONS) {
          for (let at = 0; at < letters.length; at += 1) {
            if (letters[at] === entry.parent && !taken.has(at + 1)) {
              chosen = { code: entry.code, position: at + 1 };
              break;
            }
          }
          if (chosen !== null) break;
        }
        // ...and if every residue a listed modification fits is already taken,
        // an empty row is honest: the user picks a code and a position.
        entity.modifications.push(chosen ?? { code: "", position: 1 });
        notify();
        draw();
      });
      popup.append(add);

      const problem = entityProblem(entity);
      if (problem !== null) {
        const warning = document.createElement("p");
        warning.className = "entity-popup-problem";
        warning.textContent = problem;
        popup.append(warning);
      }
    };

    draw();
    anchor.append(popup);
    const box = anchor.querySelector("textarea.entity-value");
    const pick = () => {
      const { offsets } = cleanSequenceMap(box.value);
      const caret = box.selectionStart;
      // The residue the caret sits on, or the one before it when the click
      // landed on something this sequence does not count - a space, a header,
      // or the gap past the end of a line.
      let chosen = -1;
      for (let index = 0; index < offsets.length; index += 1) {
        if (offsets[index] <= caret) chosen = index; else break;
      }
      const modification = entity.modifications[active];
      if (chosen < 0 || modification === undefined) return;
      modification.position = chosen + 1;
      notify();
      draw();
    };
    if (box !== null) {
      box.addEventListener("click", pick);
      box.addEventListener("keyup", pick);
    }
    // 🔴 CAUGHT ON THE WAY DOWN, NOT ON THE WAY UP. A click inside the popup
    // redraws it, which REPLACES the element that was clicked - so by the time
    // the event bubbled to the document the target was detached and
    // `popup.contains(target)` was false, and the popup closed itself every
    // time "Add modification" was pressed. In the capture phase the target is
    // still where it was clicked.
    const onKey = (event) => { if (event.key === "Escape") closePopup?.(); };
    closePopup = () => {
      popup.remove();
      box?.removeEventListener("click", pick);
      box?.removeEventListener("keyup", pick);
      document.removeEventListener("keydown", onKey);
      closePopup = null;
      render();
    };
    document.addEventListener("keydown", onKey);
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
      const wasProtein = entity.type === "protein";
      entity.type = type.value;
      // 🔴 THE MODIFICATIONS GO WITH THE TYPE. They are amino-acid modifications
      // resolved through the amino-acid table, so carrying them onto a DNA row
      // leaves the row permanently invalid with its reason behind a button that
      // the row no longer has.
      if (wasProtein && entity.type !== "protein") entity.modifications = [];
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
      custom.textContent = "Custom";
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

    // 🔴 A TEXTAREA CANNOT COLOUR ONE CHARACTER, so the highlight is a mirror
    // BEHIND it: the same text in the same font at the same width, with the
    // modified residues wrapped in a mark, and the textarea itself made
    // transparent except for its caret. Every metric that affects wrapping has
    // to match or the two drift apart down the line - which is why the mirror
    // takes its font from the same rule the box does rather than restating it.
    let mirror = null;
    if (entity.type === "protein") {
      mirror = document.createElement("div");
      mirror.className = "entity-value entity-value-protein entity-mirror";
      mirror.setAttribute("aria-hidden", "true");
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
      const what = entity.type === "protein" ? "protein" : entity.type.toUpperCase();
      // "an RNA" and "a DNA": the article follows how the letters are SAID
      // (ar-en-ay, dee-en-ay), not how they are spelled.
      value.placeholder = `Paste ${entity.type === "rna" ? "an" : "a"} ${what} sequence`;
      value.setAttribute("aria-label", `${what} sequence`);
    }
    value.value = entity.value;
    /** Repaint the highlight from the text and the modification list. */
    const paint = () => {
      if (mirror === null) return;
      const { offsets } = cleanSequenceMap(value.value);
      const marked = new Set();
      for (const modification of entity.modifications ?? []) {
        const offset = offsets[modification.position - 1];
        if (offset !== undefined) marked.add(offset);
      }
      mirror.replaceChildren();
      const text = value.value;
      let run = "";
      const flush = () => { if (run !== "") { mirror.append(run); run = ""; } };
      for (let at = 0; at < text.length; at += 1) {
        if (!marked.has(at)) { run += text[at]; continue; }
        flush();
        const mark = document.createElement("mark");
        mark.textContent = text[at];
        mirror.append(mark);
      }
      flush();
      // A trailing newline collapses in a div and not in a textarea, so the
      // last line would sit one row higher than the text it is behind.
      mirror.append("\n");
      mirror.scrollTop = value.scrollTop;
    };
    entityPaint.set(entity, paint);
    value.addEventListener("scroll", () => { if (mirror !== null) mirror.scrollTop = value.scrollTop; });
    value.addEventListener("input", () => {
      entity.value = value.value;
      paint();
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

    // 🔴 A BARE BOX BESIDE THE TYPE, WITH NO WORD ON IT. It used to carry the
    // label "copies", which is what a number next to "Protein" already means -
    // and the label pushed the box to the far end of the row, away from the
    // thing it counts. AlphaFold Server puts it right after the type for the
    // same reason.
    const copies = document.createElement("input");
    copies.className = "entity-copies";
    copies.type = "number";
    copies.min = "1";
    copies.max = "20";
    copies.step = "1";
    copies.value = String(entity.copies);
    copies.title = "How many identical copies of this entity to fold";
    copies.setAttribute("aria-label", "Copies");
    copies.addEventListener("input", () => {
      // Left as typed while it is being typed; entitiesProblem reports a bad
      // count, and rewriting the field mid-edit fights the user for the caret.
      entity.copies = Number.parseInt(copies.value, 10);
      notify();
    });

    // 🔴 MODIFICATIONS GO BEHIND A BUTTON, NOT INTO THE SEQUENCE BOX. They were
    // going to be inline - "ACS[SEP]EFG" - which is compact and wrong for the
    // thing people actually do: paste a sequence from somewhere else, then say
    // that residue 3 is phosphorylated. Inline means editing the pasted text,
    // and a typo in it is a different sequence rather than a bad modification.
    // A row can carry several, so the button opens a list rather than a field.
    let options = null;
    if (entity.type === "protein") {
      options = document.createElement("button");
      options.type = "button";
      options.className = "btn btn-grey btn-small entity-options";
      const count = (entity.modifications ?? []).length;
      options.title = count === 0
        ? "Modified residues" : `${count} modified residue${count === 1 ? "" : "s"}`;
      options.setAttribute("aria-label", options.title);
      options.setAttribute("aria-haspopup", "dialog");
      options.innerHTML = '<i class="fa-solid fa-ellipsis-vertical"></i>';
      // ...and it says so without being opened, because a modification changes
      // what is folded and a closed popup hides it.
      if (count > 0) options.dataset.count = String(count);
      options.addEventListener("click", (event) => {
        event.stopPropagation();
        openModifications(wrapper, entity, index);
      });
    }

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
    // The row reads left to right: what it is, how many, what it holds, and
    // then the two buttons that act on it.
    // The mirror goes in a box with the textarea so the two share one grid
    // cell and one set of metrics.
    let field = value;
    if (mirror !== null) {
      field = document.createElement("div");
      field.className = "entity-field";
      field.append(mirror, value);
    }
    wrapper.append(type, copies, ...(picker === null ? [] : [picker]), field,
                   ...(options === null ? [] : [options]), remove);
    paint();
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
