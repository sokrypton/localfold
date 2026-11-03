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
import { COMMON_IONS, COMMON_LIGANDS, ENTITY_LABELS, ENTITY_TYPES,
  MENU_CODES, MSA_KINDS, POLYMER_TYPES, TEMPLATE_KINDS, entitiesFromText, entityProblem, foldModificationRows,
  modificationRows, newEntity, parseCrosslink, rowChains, templateAsked, templateKind, templateProblem }
  from "./entities.js";
import { cleanSequence, cleanSequenceMap, extractFastaHeader } from "./sequence.js";
import { fetchStructure, parseSource, structureChains } from "./template-source.js";
import { LINK_TYPES, buildLinkField } from "./link-builder.js";

/** What a fetched template says once fetched: the chain a fold will take, and the rest. */
function fetchedStatus(template) {
  const { id, chains } = template.fetched;
  const asked = parseSource(template.source ?? "", template.fetched.kind).chain;
  const used = chains.find(({ chain }) => chain === asked) ?? chains[0];
  const others = chains.length - 1;
  return `${id} chain ${used.chain}: ${used.residues} residues`
    + (others > 0 ? ` (${others} other protein chain${others === 1 ? "" : "s"})` : "")
    + ". Coverage appears after the fold.";
}

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
  // 🔴 ASKED, NOT READ. "From the MSA search" is the one template source whose
  // validity depends on a control this module does not own, and reaching for
  // `document.getElementById("msa-mode")` from here would tie the entity list
  // to one page's markup. The caller answers; a caller that does not answer
  // gets the old behaviour, which is to assume it is fine.
  const msaIsSearch = options.msaIsSearch ?? (() => true);
  // 🔴 WHAT THE CHOSEN MODEL CANNOT TAKE, ASKED THE SAME WAY: the page knows the model, so it answers - a sentence
  // saying why for a row type the model does not read, null for one it does. A type it refuses is left out of the
  // menu of a row that does not hold it, and a row that already holds it stays, greyed, with the sentence on it:
  // dropping the row would lose what was typed, and the fold still refuses it by name (web/app.js).
  const rowRefusal = options.rowRefusal ?? (() => null);

  /**
   * Everything the closed ⋮ button has to say about one entity.
   *
   * 🔴 ONE PAINTER, BECAUSE THERE WERE TWO AND THEY DISAGREED. `render()` set
   * the title and the count from the modifications alone, and the fuller
   * version - which counts the template too - lived inside the popup's
   * `refreshBadge` and therefore only ran once the popup had been OPENED. So a
   * template set through the list's own API, or restored into a fresh row,
   * left the button reading "Modified residues" with no count on it.
   */
  const paintBadge = (badge, entity) => {
    if (badge === null) return;
    // 🔴 THE BADGE COUNTS EVERYTHING BEHIND THE BUTTON. A template changes what
    // is folded, and a closed popup hides it. (Modifications were behind it too;
    // they are rows of their own now, in plain sight.)
    const count = 0;
    const templated = templateAsked(entity.template);
    // ...and an alignment turned OFF, which changes what is folded as much as
    // either: the chain goes to the model with its query row alone.
    const noMsa = entity.type === "protein" && entity.msa === "none";
    const total = count + (templated ? 1 : 0) + (noMsa ? 1 : 0);
    if (total > 0) badge.dataset.count = String(total);
    else delete badge.dataset.count;
    const said = [];
    if (count > 0) said.push(`${count} modified residue${count === 1 ? "" : "s"}`);
    if (templated) said.push("a template");
    if (noMsa) said.push("no alignment");
    // 🔴 AND THE ONE COMBINATION THAT CANNOT FOLD, ON THE CLOSED BUTTON. An
    // automatic template with the MSA not set to search has nothing to draw a
    // hit from, and the popup that says so is shut. The only way anyone found
    // out was to press Fold and be told.
    if (templateKind(entity.template) === "search" && !msaIsSearch()) {
      badge.dataset.warn = "template";
      badge.title = "This template needs the MSA set to search";
    } else {
      delete badge.dataset.warn;
      badge.title = said.length === 0 ? "Options" : said.join(" and ");
    }
    badge.setAttribute("aria-label", badge.title);
  };
  /** @type {{type: string, value: string, copies: number}[]} */
  // 🔴 A MODIFICATION IS A ROW HERE AND A FIELD OF ITS SEQUENCE EVERYWHERE ELSE: unfolded into rows coming in
  // (`set`, the initial list) and folded back going out (`read`), so the page, the job JSON and every tool see the
  // `modifications` they always did (entities.js, foldModificationRows).
  let entities = (options.initial ?? []).length > 0
    ? modificationRows(options.initial.map((entity) => ({ ...entity })))
    : [newEntity("protein")];
  /** Every polymer row's highlight painter, for a modification row to repaint the sequence it names. */
  let painters = [];
  const repaintAll = () => { for (const paint of painters) paint(); };
  // The description line of the last FASTA that was pasted, which is where the
  // job name used to come from when this was one textarea holding the record.
  let pastedHeader = null;

  const render = () => {
    painters = [];
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
    popup.setAttribute("aria-label", "Options");

    // 🔴 THE BADGE WITHOUT THE REBUILD, because a keystroke needs one and not
    // the other. `draw()` replaces every child of the popup, so calling it from
    // an input handler destroys the very box being typed into: the source field
    // took one character and lost the caret, which is a template that cannot be
    // named. python3 tools/entity-popup.py asserts the caret survives.
    const refreshBadge = () => {
      paintBadge(anchor.querySelector(".entity-options"), entity);
      entityPaint.get(entity)?.();
    };
    const draw = () => {
      // 🔴 NOT render(). The popup is a CHILD of the row, so rebuilding the row
      // deletes the popup out from under itself - clicking "Add modification"
      // closed the menu and left the modification behind, which reads as the
      // button having failed. The one thing outside the popup that has to keep
      // up is the badge on the button, so that is updated by hand.
      refreshBadge();
      popup.replaceChildren();
      const head = document.createElement("div");
      head.className = "entity-popup-head";
      const title = document.createElement("div");
      title.className = "entity-popup-title";
      title.textContent = "Options";
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

      // 🔴 THE TEMPLATE LIVES BEHIND THE SAME BUTTON, because it is the same
      // kind of thing: something set on ONE chain that changes what is folded
      // and is invisible on the row. It had a button of its own and that made
      // the entity row a six-track grid - a grid with fewer tracks than
      // children WRAPS rather than overflowing, so it read as "the desktop
      // entity row broke across 2 lines".
      if (entity.type === "protein") {
        // 🔴 WHETHER THIS CHAIN GETS AN ALIGNMENT, which is the same kind of
        // thing as the template below it: set on ONE chain, changes what is
        // folded, invisible on the row - so it lives behind the same button
        // and is counted by the same badge.
        //
        // Off means the chain is folded from its query row alone while the
        // rest of the complex keeps theirs. The page's own MSA dial still
        // decides what happens for the chains that DO want one; this cannot
        // turn an alignment on where the job has none, and says so.
        const msaSection = document.createElement("div");
        msaSection.className = "entity-popup-section entity-popup-section-msa";
        const msaHeading = document.createElement("div");
        msaHeading.className = "entity-popup-title";
        msaHeading.textContent = "Alignment";
        msaSection.append(msaHeading);
        const msaMenu = document.createElement("select");
        msaMenu.className = "entity-msa-kind";
        msaMenu.setAttribute("aria-label", "Alignment for this chain");
        for (const [value, label] of MSA_KINDS) {
          const option = document.createElement("option");
          option.value = value;
          option.textContent = label;
          msaMenu.append(option);
        }
        msaMenu.value = entity.msa === "none" ? "none" : "search";
        // ...and ONE LINE under it, in the template's own shape: the menu
        // says what is chosen and this says what it means for the fold.
        // Neither explains the job's own MSA dial, which is a control the
        // reader can see for themselves.
        const msaNote = document.createElement("p");
        msaNote.className = "entity-popup-empty entity-msa-status";
        const sayMsa = () => {
          msaNote.textContent = msaMenu.value === "none"
            ? "This chain folds from its sequence alone."
            : "This chain uses the alignment the job has.";
        };
        sayMsa();
        msaMenu.addEventListener("change", () => {
          // ...stored only when it is the unusual answer, so an entity that
          // says nothing still means "give me one" - which is what every job
          // written before this row existed says.
          if (msaMenu.value === "none") entity.msa = "none";
          else delete entity.msa;
          // ...the line under it follows, without rebuilding the popup: the
          // template's source box redraws on every keystroke and that is
          // what costs it the caret, so nothing here rebuilds either.
          sayMsa();
          refreshBadge();
          notify();
        });
        const msaLine = document.createElement("div");
        msaLine.className = "entity-popup-row";
        msaLine.append(msaMenu);
        msaSection.append(msaLine);
        msaSection.append(msaNote);
        popup.append(msaSection);

        entity.template ??= { source: "" };
        const template = entity.template;
        const section = document.createElement("div");
        section.className = "entity-popup-section entity-popup-section-template";
        const heading = document.createElement("div");
        heading.className = "entity-popup-title";
        heading.textContent = "Template";
        section.append(heading);

        // 🔴 THE SOURCE IS A MENU, BECAUSE "WHICH DATABASE" IS NOT A GUESS.
        // See TEMPLATE_KINDS in web/entities.js for why the single free-text
        // box that guessed by length was the wrong shape. Choosing the kind
        // first also means the field under it can say what it wants - a PDB
        // entry and a UniProt accession look nothing alike - and that the two
        // sources with no text at all can be named beside the two that have it.
        const kindMenu = document.createElement("select");
        kindMenu.className = "entity-template-kind";
        kindMenu.setAttribute("aria-label", "Template source");
        for (const [value, label] of TEMPLATE_KINDS) {
          const option = document.createElement("option");
          option.value = value;
          option.textContent = label;
          kindMenu.append(option);
        }
        const kind = templateKind(template);
        kindMenu.value = kind;
        kindMenu.addEventListener("change", () => {
          template.kind = kindMenu.value;
          // 🔴 CHANGING THE SOURCE CLEARS WHAT THE OLD ONE HELD. A PDB id left
          // behind under "AlphaFold DB" is a source the user cannot see and did
          // not choose, and an uploaded file left behind under "PDB entry" is
          // worse - it would fold, against a structure nothing on screen names.
          delete template.source;
          delete template.text;
          delete template.filename;
          delete template.status;
          delete template.statusError;
          delete template.fetched;
          draw();
          notify();
        });
        const kindLine = document.createElement("div");
        kindLine.className = "entity-popup-row";
        kindLine.append(kindMenu);
        section.append(kindLine);

        let chainLine;
        if (kind === "pdb" || kind === "afdb") {
          const source = document.createElement("input");
          source.type = "text";
          source.className = "entity-template-source";
          source.placeholder = kind === "pdb" ? "1abc, or 1abc_A" : "P00533";
          source.value = template.source ?? "";
          source.setAttribute("aria-label", kind === "pdb"
            ? "PDB entry" : "UniProt accession");
          source.addEventListener("input", () => {
            template.source = source.value;
            // A fetch answered a different id, so what it said no longer
            // describes this box. Written into the status line by hand rather
            // than through draw(), which would replace this input mid-word.
            const id = parseSource(source.value, kind).id;
            if (template.fetched !== undefined && template.fetched.id !== id) {
              delete template.fetched;
              delete template.status;
              delete template.statusError;
              status.className = "entity-popup-empty entity-template-status";
              status.textContent = waiting[kind];
              chainLine?.remove();
            }
            // ...the badge, not the popup: see refreshBadge. Rebuilding here
            // would replace this input between one keystroke and the next.
            refreshBadge();
            notify();
          });
          // 🔴 FETCH BEFORE FOLD, SO THE ID IS CHECKED WHERE IT IS TYPED. Until
          // this a PDB id was only fetched once Fold was pressed, so a typo, an
          // entry with five chains or one the RCSB has no .pdb for said nothing
          // here and surfaced as a failed or quietly different fold. The fetched
          // text is kept on the template and the fold reuses it (app.js), so the
          // structure checked here is the structure folded.
          const fetchButton = document.createElement("button");
          fetchButton.type = "button";
          fetchButton.className = "entity-template-fetch";
          fetchButton.textContent = template.fetched !== undefined ? "Fetched" : "Fetch";
          const fetchNow = async () => {
            const typed = source.value.trim();
            if (typed === "" || templateProblem(entity) !== null) return;
            const parsed = parseSource(typed, kind);
            template.status = `Fetching ${parsed.id}…`;
            delete template.statusError;
            delete template.fetched;
            draw();
            try {
              const structure = await fetchStructure(typed, { kind });
              const chains = structureChains(structure.text);
              if (chains.length === 0) throw new Error(`${parsed.id} holds no protein chain`);
              if (parsed.chain !== undefined && !chains.some(({ chain }) => chain === parsed.chain)) {
                throw new Error(`${parsed.id} has no protein chain ${parsed.chain}`
                  + ` (it has ${chains.map(({ chain }) => chain).join(", ")})`);
              }
              template.fetched = { id: parsed.id, kind, text: structure.text, chains };
              template.status = fetchedStatus(template);
            } catch (error) {
              template.status = `Could not fetch ${parsed.id}: ${error.message ?? error}`;
              template.statusError = true;
            }
            draw();
            notify();
          };
          fetchButton.addEventListener("click", fetchNow);
          source.addEventListener("keydown", (event) => {
            if (event.key === "Enter") { event.preventDefault(); void fetchNow(); }
          });
          const line = document.createElement("div");
          line.className = "entity-popup-row";
          line.append(source, fetchButton);
          section.append(line);

          // ...and once fetched, the chain is a MENU of what the file holds,
          // which is the part `1abc_A` asked the reader to know in advance.
          const fetched = template.fetched;
          if (fetched !== undefined && fetched.id === parseSource(source.value, kind).id
              && fetched.chains.length > 1) {
            const chainMenu = document.createElement("select");
            chainMenu.className = "entity-template-kind";
            chainMenu.setAttribute("aria-label", "Template chain");
            const current = parseSource(source.value, kind).chain ?? fetched.chains[0].chain;
            for (const { chain, residues } of fetched.chains) {
              const option = document.createElement("option");
              option.value = chain;
              option.textContent = `Chain ${chain} · ${residues} residues`;
              chainMenu.append(option);
            }
            chainMenu.value = current;
            chainMenu.addEventListener("change", () => {
              template.source = `${fetched.id}_${chainMenu.value}`;
              template.status = fetchedStatus(template);
              draw();
              notify();
            });
            chainLine = document.createElement("div");
            chainLine.className = "entity-popup-row";
            chainLine.append(chainMenu);
            section.append(chainLine);
          }
        }

        if (kind === "upload") {
          // 🔴 READ HERE AND CARRIED AS TEXT, because the fold happens long
          // after the popup has closed and a File handle is not something the
          // entity list can be serialised through. The chain box matters more
          // here than anywhere else: a downloaded entry can be asked for as
          // `1abc_A`, and an uploaded complex has no such spelling.
          const file = document.createElement("input");
          file.type = "file";
          file.className = "entity-template-file";
          file.accept = ".pdb,.cif,.mmcif,.ent,.txt";
          file.setAttribute("aria-label", "Template structure file");
          file.addEventListener("change", async () => {
            const chosen = file.files?.[0];
            if (chosen === undefined) return;
            template.filename = chosen.name;
            template.text = await chosen.text();
            delete template.status;
            draw();
            notify();
          });
          const chain = document.createElement("input");
          chain.type = "text";
          chain.className = "entity-template-chain";
          chain.placeholder = "chain";
          chain.value = template.source ?? "";
          chain.setAttribute("aria-label", "Which chain of the file");
          chain.addEventListener("input", () => {
            template.source = chain.value;
            notify();
          });
          const line = document.createElement("div");
          line.className = "entity-popup-row";
          line.append(file, chain);
          section.append(line);
        }

        const status = document.createElement("p");
        status.className = "entity-popup-empty entity-template-status";
        // Written by the page once it has fetched: coverage is a fact about a
        // file this list has never seen, and a template covering 17 of 120
        // residues folds perfectly well and says nothing about it.
        // 🔴 SHORT, BECAUSE THE CONTROL ABOVE ALREADY SAYS MOST OF IT. The
        // placeholders spell the format - "1abc, or 1abc_A", "P00533" - so
        // repeating it here was three lines saying what one already showed.
        // What is left is the one thing the control cannot say: where the
        // structure comes from, and that coverage is reported after the fetch.
        const waiting = {
          none: "A known structure to fold this chain against.",
          pdb: "A PDB entry. Fetch to see its chains; Fold fetches it otherwise.",
          afdb: "An accession from AlphaFold DB. Fetch to check it.",
          search: "The best hit the MSA search finds. Needs the MSA set to search.",
          upload: template.filename ?? "A PDB or mmCIF file from this machine.",
        };
        // 🔴 A HINT UNTIL IT IS WRONG, AND THEN A WARNING. This line has always
        // said "Needs the MSA set to search", and the way anyone found out it
        // was not was to press Fold and be told - the fold reaches the template
        // loop, finds no hits and throws. The page knows both halves before
        // anything runs, so it says so here, where the choice is made.
        const needsSearch = kind === "search" && !msaIsSearch();
        if (template.statusError === true) {
          status.className = "entity-popup-problem entity-template-status";
          status.textContent = template.status;
        } else if (needsSearch && template.status === undefined) {
          status.className = "entity-popup-problem entity-template-status";
          status.textContent = "The MSA is not set to search, so there are no hits"
            + " to take a template from. Set the MSA to search, or name a structure.";
        } else {
          status.textContent = template.status ?? waiting[kind];
        }
        section.append(status);
        popup.append(section);
      }

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
    // 🔴 CAUGHT ON THE WAY DOWN, NOT ON THE WAY UP. A click inside the popup
    // redraws it, which REPLACES the element that was clicked - so by the time
    // the event bubbled to the document the target was detached and
    // `popup.contains(target)` was false, and the popup closed itself every
    // time "Add modification" was pressed. In the capture phase the target is
    // still where it was clicked.
    const onKey = (event) => { if (event.key === "Escape") closePopup?.(); };
    closePopup = () => {
      popup.remove();
      document.removeEventListener("keydown", onKey);
      closePopup = null;
      render();
    };
    document.addEventListener("keydown", onKey);
  };

  /**
   * Which kind of thing a row holds, for deciding whether its VALUE survives a
   * type change: the three polymers share an alphabet, a CCD code is a name,
   * and a SMILES is a structure. Text means something different in each.
   */
  const categoryOf = (type) => (POLYMER_TYPES.includes(type) ? "polymer" : type);

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
      // (hidden where the browser hides an option, and disabled where it does not)
      if (value !== entity.type && rowRefusal(value) !== null) { option.hidden = true; option.disabled = true; }
      type.append(option);
    }
    type.value = entity.type;
    type.addEventListener("change", () => {
      const wasProtein = entity.type === "protein";
      const from = entity.type;
      entity.type = type.value;
      // 🔴 THE MODIFICATIONS GO WITH THE TYPE. They are amino-acid modifications
      // resolved through the amino-acid table, so carrying them onto a DNA row
      // leaves the row permanently invalid with its reason behind a button that
      // the row no longer has.
      if (wasProtein && entity.type !== "protein") entity.modifications = [];
      // 🔴 AND THE VALUE GOES WITH IT WHEN THE ROW CHANGES CATEGORY, BECAUSE
      // THE SAME TEXT IS A DIFFERENT MOLECULE ON THE OTHER SIDE. Switching a
      // CCD row holding `C` to SMILES gives methane where it meant cytidine
      // monophosphate; switching a SMILES `CCO` to CCD goes looking for a
      // dictionary entry called CCO. Both parse, both validate, and both are
      // silently not what was typed - which is the same failure as the
      // archive writing a SMILES as a code, one control earlier.
      //
      // 🔴 BUT NOT BETWEEN THE POLYMERS, WHERE CARRYING IT IS THE POINT. This
      // file's own note says `ACGT` is a valid protein AND a valid DNA chain
      // and that only the row's type says which - so someone pasting a
      // sequence and then correcting the type must not lose it. Clearing
      // within that group would make the commonest correction on the page
      // destructive.
      if (categoryOf(from) !== categoryOf(entity.type)) entity.value = "";
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

    // 🔴 A BOND IS ONE LINE TOO. It is a short spec, not a sequence, and a
    // textarea would give a bond three rows of empty box in the middle of the
    // list.
    const oneLine = entity.type === "ligand" || entity.type === "smiles"
      || entity.type === "bond" || entity.type === "crosslink" || entity.type === "pocket" || entity.type === "contact";
    const value = oneLine
      ? document.createElement("input")
      : document.createElement("textarea");
    value.className = `entity-value entity-value-${entity.type}`;
    value.spellcheck = false;
    if (entity.type === "ligand") {
      value.type = "text";
      value.placeholder = "CCD code, e.g. HEM";
      value.setAttribute("aria-label", "Ligand CCD code");
    } else if (entity.type === "bond") {
      // 🔴 A BOND BETWEEN TWO RESIDUES, WRITTEN AS A READER WOULD SAY IT.
      // Chains are the letters the viewer and the PDB use; the atoms are
      // optional, because `token_bonds` is token x token and a standard
      // residue has one token whichever atom is named. See parseBond.
      value.type = "text";
      value.placeholder = "A12:SG - B1:C25";
      value.setAttribute("aria-label", "Bond between two residues");
    } else if (entity.type === "crosslink") {
      // a crosslinker bonded to two residues as a ligand (af3x's crosslinks; shared/chem/crosslinkers.js)
      value.type = "text";
      value.placeholder = "DSSO: A2 - B30";
      value.setAttribute("aria-label", "Crosslink: a crosslinker and the two residues it joined");
    } else if (entity.type === "pocket") {
      // a binder chain within a distance of listed residues (shared/af3/featurise/constraints.js): boltz2 and chai-1
      value.type = "text";
      value.placeholder = "B: A9, A12, A16 @ 6";
      value.setAttribute("aria-label", "Pocket: a binder chain, the residues it sits by, and a distance in Angstrom");
    } else if (entity.type === "contact") {
      value.type = "text";
      value.placeholder = "A10 - B44 @ 8";
      value.setAttribute("aria-label", "Contact constraint: two residues within a distance, in Angstrom");
    } else if (entity.type === "smiles") {
      // 🔴 ITS OWN CLASS, BECAUSE `.entity-value-ligand` IS UPPERCASED IN CSS.
      // Borrowing the ligand's styling would DISPLAY a SMILES upper case -
      // `c1ccccc1` shown as `C1CCCCC1` - which is a different molecule on
      // screen from the one that would be folded.
      value.type = "text";
      value.placeholder = "SMILES, e.g. c1ccccc1 for benzene";
      value.setAttribute("aria-label", "Ligand SMILES");
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
    /** Repaint the highlight from the text, the modification list and the crosslink rows. */
    const paint = () => {
      if (mirror === null) return;
      const { offsets } = cleanSequenceMap(value.value);
      // text offset -> what marks it: a modified residue, or one a crosslink joins
      const marked = new Map();
      // (folded, the rows keep their order with the modification rows taken out)
      const mine = foldModificationRows(entities)[entities.filter((one) => one.type !== "modification").indexOf(entity)];
      // (none where the model takes no modifications: their rows are greyed, so the sequence is not marked either)
      for (const modification of rowRefusal("modification") === null ? mine?.modifications ?? [] : []) {
        const offset = offsets[modification.position - 1];
        if (offset !== undefined) marked.set(offset, "modification");
      }
      // ...and a crosslinked residue, named by the chain letter of any copy of this sequence (rowChains letters them
      // as the fold does) - its own colour, so a reader can tell the two apart; the same refusal rule
      if (rowRefusal("crosslink") === null && entities.some((one) => one.type === "crosslink")) {
        const index = entities.indexOf(entity);
        const letters = new Set(rowChains(entities).filter((chain) => chain.entity === index)
          .map((chain) => chain.letter));
        for (const one of entities) {
          if (one.type !== "crosslink") continue;
          const parsed = parseCrosslink(one.value);
          for (const end of parsed === null ? [] : [parsed.from, parsed.to]) {
            const offset = letters.has(end.chain) ? offsets[end.residue - 1] : undefined;
            if (offset !== undefined && !marked.has(offset)) marked.set(offset, "crosslink");
          }
        }
      }
      mirror.replaceChildren();
      const text = value.value;
      let run = "";
      const flush = () => { if (run !== "") { mirror.append(run); run = ""; } };
      for (let at = 0; at < text.length; at += 1) {
        if (!marked.has(at)) { run += text[at]; continue; }
        flush();
        const mark = document.createElement("mark");
        if (marked.get(at) === "crosslink") mark.className = "entity-mark-crosslink";
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
    if (mirror !== null) painters.push(paint);
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
      } else if (entity.type === "smiles") {
        // 🔴 TRIMMED AND NOTHING ELSE. Neither of the other two branches can
        // touch a SMILES: upper-casing turns `c1ccccc1` into cyclohexane, and
        // `cleanSequence` keeps only amino-acid letters, which turned benzene
        // into `CCCCCC` - HEXANE - and stripped biotin's ring-closure digits
        // so it no longer parsed. Both happened the moment the reader clicked
        // away from the box, silently rewriting what they had typed.
        entity.value = value.value.trim();
      } else if (entity.type === "bond" || entity.type === "crosslink" || entity.type === "pocket"
                 || entity.type === "contact") {
        // 🔴 TRIMMED AND NOTHING ELSE, FOR THE THIRD TIME AT THIS SEAM. (A constraint row too: the same spec syntax.) The
        // `else` below is `cleanSequence`, which keeps only amino-acid letters:
        // a contact typed as `A12:SG - B1:C25` came back `A:SGB:C` the moment
        // the reader clicked away - the residue numbers, the spaces and the
        // hyphen all gone, and what was left still looked like a sequence. It
        // is the same branch that turned benzene into hexane, and the comment
        // above it did not stop a third row type walking into it.
        //
        // 🔴 AND NOT UPPER-CASED HERE EITHER. `parseBond` upper-cases the
        // chain letter and the atom name when it reads them, which is where
        // that belongs; doing it here would also upper-case whatever a reader
        // is halfway through typing.
        entity.value = value.value.trim();
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
      options.setAttribute("aria-haspopup", "dialog");
      options.innerHTML = '<i class="fa-solid fa-ellipsis-vertical"></i>';
      // ...and it says so without being opened, because a modification and a
      // template each change what is folded and a closed popup hides both.
      paintBadge(options, entity);
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
    // 🔴 A BOND, CONTACT, POCKET OR CROSSLINK IS PICKED, NOT TYPED: chain menus, residue numbers, an atom and a
    // distance, which write the same spec text the box took (web/link-builder.js). Nobody remembers `A12:SG - B1:C25`.
    if (LINK_TYPES.includes(entity.type)) {
      field = buildLinkField(entity, () => entities, () => {
        notify();
        if (entity.type === "modification" || entity.type === "crosslink") repaintAll();
      });
      wrapper.classList.add("entity-row-link");
    } else if (mirror !== null) {
      field = document.createElement("div");
      field.className = "entity-field";
      field.append(mirror, value);
    }
    // 🔴 A ROW WITHOUT AN OPTIONS BUTTON STILL OCCUPIES ITS COLUMN. The grid
    // has five, and a DNA row - which has no modifications to open - was
    // appending four children, so its remove button slid up into the options
    // column and sat six pixels left of every other row's. An empty cell keeps
    // the column.
    //
    // 🔴 BUT NOT ON A LIGAND ROW, WHICH SPENDS THAT COLUMN ON ITS PICKER. Its
    // five are type, copies, picker, code, remove - so a placeholder there is a
    // SIXTH child in a five-column grid, and the remove button wraps onto a
    // second line 44 px below the row it belongs to. Every row has exactly five
    // cells; which five is what differs.
    const spacer = picker === null && options === null
      ? [document.createElement("span")] : [];
    wrapper.append(type, copies, ...(picker === null ? [] : [picker]), field,
                   ...(options === null ? [] : [options]), ...spacer, remove);
    paint();
    if (picker !== null) wrapper.classList.add("entity-row-ligand");
    const refused = rowRefusal(entity.type);
    if (refused !== null) {
      wrapper.classList.add("entity-row-unsupported");
      wrapper.title = refused;
      type.title = refused;
    }
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
    read: () => foldModificationRows(entities),
    set: (next) => {
      entities = next.length === 0 ? [newEntity("protein")] : modificationRows(next.map((e) => ({ ...e })));
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
      // 🔴 EVERY NON-POLYMER ROW SURVIVES, NOT JUST THE CCD ONES. This runs
      // when an alignment's own query replaces the chain list, and filtering
      // on `type === "ligand"` DELETED a SMILES row - so folding with an A3M
      // silently dropped the ligand and folded the protein alone.
      // ...and a bond, contact, pocket or crosslink row survives too, after the chains, for the same reason.
      const ligands = entities.filter((entity) => entity.type === "ligand" || entity.type === "smiles"
        || LINK_TYPES.includes(entity.type));
      const proteins = [];
      for (const chain of chains) {
        const existing = proteins.find((row) => row.value === chain);
        if (existing !== undefined) { existing.copies += 1; continue; }
        // 🔴 AND THE ROW KEEPS WHAT IT ALREADY SAID ABOUT THIS CHAIN. This
        // rebuilt every protein row as `{type, value, copies}`, which threw
        // away the per-chain settings that live nowhere else - the
        // ALIGNMENT override, the template, the modifications - and it runs
        // on every fold of a single-chain job that has an alignment at all.
        // So: turn a chain's alignment off, press Fold, and the setting was
        // gone from the row before the fold finished; press Fold again and
        // there was nothing left to honour, which is why it folded with the
        // alignment. Reported as both halves of exactly that.
        //
        // Carried by SEQUENCE, which is what makes it safe: where the
        // alignment's query really is a different protein there is no row
        // with that value and the new one is bare, which is right - those
        // settings belonged to the chain that is being replaced.
        const had = entities.find(
          (entity) => entity.type === "protein" && entity.value === chain);
        proteins.push(had === undefined
          ? { type: "protein", value: chain, copies: 1 }
          : { ...had, copies: 1 });
      }
      entities = [...proteins, ...ligands];
      render();
    },
    header: () => pastedHeader,
    /** Redraw, after the model changed: which types it takes is asked again (`options.rowRefusal`). */
    refresh: () => render(),
  };
}
