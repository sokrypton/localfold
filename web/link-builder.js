/**
 * The pickers that write a bond, contact, pocket or crosslink row, so nobody has to remember `A12:SG - B1:C25 @ 8`.
 *
 * WHAT IT OWNS: one row's controls - a chain menu per end listing the chains the rows will fold (lettered as
 * expandEntities letters them, via `rowChains`), a residue number with the residue it lands on beside it, an optional
 * atom, and a distance where the row has one. Every change writes the row's TEXT, the same spec the box took, so
 * `parseBond`/`parseContact`/`parsePocket`/`parseCrosslink`, the job JSON and every gate read exactly what they read
 * before: the controls are a way of writing the string, not a second representation of it.
 *
 * 🔴 THE MENUS ARE REBUILT WHEN OPENED, NOT WHEN DRAWN. Typing a sequence or changing a copy count does not re-render
 * the list, so a menu built at draw time would offer the chains as they were then. A letter the rows no longer have is
 * kept as "(missing)" rather than silently swapped for another chain, which would bond something else.
 */

import { CROSSLINKERS } from "../shared/chem/crosslinkers.js";
import { THREE_LETTER } from "../shared/af3/structure/pdb.js";
import { DEFAULT_CONSTRAINT_DISTANCE, POLYMER_TYPES, parseBond, parseContact, parseCrosslink, parsePocket, rowChains }
  from "./entities.js";

/** The row types written by pickers rather than typed. */
export const LINK_TYPES = ["bond", "crosslink", "pocket", "contact"];

const endText = (end) => (end.chain === "" || !(end.residue > 0) ? `${end.chain}`
  : `${end.chain}${end.residue}${end.atom ? `:${end.atom}` : ""}`);

/**
 * A row's text from its parts - the inverse of the parsers in entities.js. An unfinished end (no residue yet) is
 * written as far as it goes, so the row reads as invalid and says why rather than being dropped.
 */
export function linkText(type, parts) {
  if (type === "bond") return `${endText(parts.ends[0])} - ${endText(parts.ends[1])}`;
  if (type === "contact") return `${endText(parts.ends[0])} - ${endText(parts.ends[1])} @ ${parts.distance}`;
  if (type === "crosslink") {
    const bare = (end) => endText({ ...end, atom: null });
    return `${parts.linker}: ${bare(parts.ends[0])} - ${bare(parts.ends[1])}`;
  }
  return `${parts.binder}: ${parts.residues.trim()} @ ${parts.distance}`;
}

/** A row's text read back into parts; a row that does not parse starts from defaults on the first two chains. */
export function linkParts(type, value, chains) {
  const first = chains[0]?.letter ?? "A";
  const second = chains[1]?.letter ?? first;
  const blank = (chain) => ({ chain, residue: null, atom: null });
  const fromEnd = (end) => ({ chain: end.chain, residue: end.residue, atom: end.atom ?? null });
  const defaults = { ends: [blank(first), blank(second)], distance: DEFAULT_CONSTRAINT_DISTANCE,
                     linker: Object.keys(CROSSLINKERS)[0], binder: second, residues: "" };
  if (type === "bond") {
    const parsed = parseBond(value);
    return parsed === null ? defaults : { ...defaults, ends: [fromEnd(parsed.from), fromEnd(parsed.to)] };
  }
  if (type === "contact") {
    const parsed = parseContact(value);
    return parsed === null ? { ...defaults, distance: 8 }
      : { ...defaults, ends: [fromEnd(parsed.token1), fromEnd(parsed.token2)], distance: parsed.maxDistance };
  }
  if (type === "crosslink") {
    const parsed = parseCrosslink(value);
    return parsed === null ? defaults
      : { ...defaults, linker: parsed.name, ends: [fromEnd(parsed.from), fromEnd(parsed.to)] };
  }
  const parsed = parsePocket(value);
  return parsed === null ? defaults : {
    ...defaults, binder: parsed.binder, distance: parsed.maxDistance,
    residues: parsed.contacts.map(endText).join(", "),
  };
}

const element = (tag, className, props = {}) => Object.assign(document.createElement(tag), { className }, props);
const word = (text) => element("span", "link-word", { textContent: text });

/** What a residue number lands on, said the way a reader checks it: `LYS`, `G`, `HEM`. */
function residueName(chain, residue) {
  if (chain === undefined || !(residue >= 1) || residue > chain.residues.length) return "";
  const one = chain.residues[residue - 1];
  return chain.kind === "protein" ? THREE_LETTER[one] ?? one : one;
}

/**
 * The controls for one link row.
 * @param entity the row; its `value` is rewritten on every change
 * @param readEntities () => the list's rows as they are now, for the chain menus
 * @param changed () => void, after `entity.value` is written
 * @returns {HTMLElement} the field cell
 */
export function buildLinkField(entity, readEntities, changed) {
  const type = entity.type;
  const parts = linkParts(type, entity.value, rowChains(readEntities()));
  const field = element("div", `entity-link entity-link-${type}`);
  field.setAttribute("role", "group");
  field.setAttribute("aria-label", { bond: "Bond between two residues", contact: "Contact: two residues within a distance",
    pocket: "Pocket: a chain held near listed residues", crosslink: "Crosslink between two residues" }[type]);

  const write = () => {
    entity.value = linkText(type, parts);
    changed();
  };

  /** A menu of the chains, rebuilt when it is opened; `get`/`set` move its letter. */
  const chainMenu = (get, set, title) => {
    const menu = element("select", "entity-picker link-chain", { title });
    const fill = () => {
      const chains = rowChains(readEntities());
      const current = get();
      // the letter alone, as the viewer and the PDB name the chain; what it is shows on hover and beside the residue
      const options = chains.map((chain) => Object.assign(new Option(chain.letter, chain.letter), { title: chain.name }));
      if (current !== "" && !chains.some((chain) => chain.letter === current)) {
        options.push(new Option(`${current} (missing)`, current));
      }
      menu.replaceChildren(...options);
      menu.value = current;
    };
    fill();
    menu.addEventListener("focus", fill);
    menu.addEventListener("pointerdown", fill);
    menu.addEventListener("change", () => { set(menu.value); write(); });
    return { menu, fill };
  };

  /** One end: chain, residue (with what it lands on), and an atom where the row takes one. */
  const endControls = (end, label, withAtom) => {
    // text with a numeric keypad rather than type=number, whose spinner takes half of a box this narrow
    const residue = element("input", "entity-value link-residue", {
      type: "text", inputMode: "numeric", placeholder: "res", title: `${label}: residue number, counting from 1`,
    });
    residue.setAttribute("aria-label", `${label} residue number`);
    if (end.residue > 0) residue.value = String(end.residue);
    const named = element("span", "link-residue-name");
    const atom = withAtom ? element("input", "entity-value link-atom", {
      type: "text", placeholder: "atom", spellcheck: false, value: end.atom ?? "",
      title: `${label}: atom name, optional for a residue (e.g. SG), needed for a ligand atom (e.g. C1)`,
    }) : null;
    atom?.setAttribute("aria-label", `${label} atom (optional)`);
    const refresh = () => {
      const chain = rowChains(readEntities()).find((one) => one.letter === end.chain);
      const n = chain?.residues.length;
      named.textContent = residueName(chain, end.residue);
      named.title = end.residue > n ? `past the end: chain ${end.chain} has ${n}` : "";
      named.classList.toggle("link-residue-past", end.residue > n);
      if (named.textContent === "" && end.residue > n) named.textContent = "?";
      residue.title = `${label}: residue number, 1 to ${n ?? "?"}`;
    };
    // a ligand chain of one component has one residue: take it, so the row is complete without a number nobody knows
    const takeLoneResidue = () => {
      const chain = rowChains(readEntities()).find((one) => one.letter === end.chain);
      if (chain !== undefined && !POLYMER_TYPES.includes(chain.kind) && chain.residues.length === 1) {
        end.residue = 1;
        residue.value = "1";
      }
    };
    const { menu } = chainMenu(() => end.chain, (letter) => {
      end.chain = letter;
      takeLoneResidue();
      refresh();
    }, `${label}: chain`);
    if (!(end.residue > 0)) takeLoneResidue();
    menu.setAttribute("aria-label", `${label} chain`);
    residue.addEventListener("input", () => {
      const n = Number.parseInt(residue.value, 10);
      end.residue = Number.isInteger(n) && n > 0 ? n : null;
      refresh();
      write();
    });
    residue.addEventListener("focus", refresh);
    atom?.addEventListener("input", () => {
      const text = atom.value.trim().toUpperCase();
      end.atom = text === "" ? null : text;
      write();
    });
    refresh();
    const group = element("span", "link-end");
    group.append(menu, residue, named, ...(atom === null ? [] : [atom]));
    return group;
  };

  const distanceControl = () => {
    const box = element("input", "entity-value link-distance", {
      type: "text", inputMode: "decimal", value: String(parts.distance), title: "Distance bound, in Angstrom",
    });
    box.setAttribute("aria-label", "Distance in Angstrom");
    box.addEventListener("input", () => {
      const d = Number(box.value);
      parts.distance = box.value.trim() === "" ? "" : Number.isFinite(d) ? d : box.value;
      write();
    });
    const group = element("span", "link-end");
    group.append(word("within"), box, word("Å"));
    return group;
  };

  if (type === "bond") {
    field.append(endControls(parts.ends[0], "First end", true), word("–"),
                 endControls(parts.ends[1], "Second end", true));
  } else if (type === "contact") {
    field.append(endControls(parts.ends[0], "First residue", true), word("–"),
                 endControls(parts.ends[1], "Second residue", true), distanceControl());
  } else if (type === "crosslink") {
    const linker = element("select", "entity-picker link-linker", { title: "Crosslinker" });
    linker.setAttribute("aria-label", "Crosslinker");
    for (const [name, entry] of Object.entries(CROSSLINKERS)) {
      linker.append(Object.assign(new Option(name, name), { title: entry.description }));
    }
    if (!(parts.linker in CROSSLINKERS)) linker.append(new Option(`${parts.linker} (unknown)`, parts.linker));
    linker.value = parts.linker;
    linker.addEventListener("change", () => { parts.linker = linker.value; write(); });
    field.append(linker, endControls(parts.ends[0], "First residue", false), word("–"),
                 endControls(parts.ends[1], "Second residue", false));
  } else {
    const { menu } = chainMenu(() => parts.binder, (letter) => { parts.binder = letter; }, "The chain held in the pocket");
    menu.setAttribute("aria-label", "Pocket binder chain");
    const residues = element("input", "entity-value link-residues", {
      type: "text", placeholder: "near residues, e.g. A9, A12, A16", spellcheck: false, value: parts.residues,
      title: "The residues the chain sits by: chain letter and number, comma separated (a ligand atom as C1:O1)",
    });
    residues.setAttribute("aria-label", "Pocket residues");
    residues.addEventListener("input", () => { parts.residues = residues.value; write(); });
    const binder = element("span", "link-end");
    binder.append(menu, word("near"));
    field.append(binder, residues, distanceControl());
  }
  // a row that arrived empty is written once, so it shows its problem by what is missing rather than as blank
  if (entity.value.trim() === "") entity.value = linkText(type, parts);
  return field;
}
