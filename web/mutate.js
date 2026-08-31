// Click a residue in the viewer, pick a different amino acid, and the sequence
// box changes. The menu can then fold on the spot, or be left open and dragged
// out of the way while you look at what you changed.
import { hydropathyBand, hydropathyColor, RESIDUES_BY_HYDROPATHY } from "./hydrophobicity.js";

/** One-letter to three-letter, for the menu's title line. */
export const THREE_LETTER = {
  A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", Q: "GLN", E: "GLU",
  G: "GLY", H: "HIS", I: "ILE", L: "LEU", K: "LYS", M: "MET", F: "PHE",
  P: "PRO", S: "SER", T: "THR", W: "TRP", Y: "TYR", V: "VAL", X: "UNK",
};

function residueToCharIndex(sequence, residueIndex) {
  let count = 0;
  for (let i = 0; i < sequence.length; i += 1) {
    if (sequence[i] !== ":") {
      if (count === residueIndex) return i;
      count += 1;
    }
  }
  return -1;
}

function totalResidueCount(sequence) {
  let count = 0;
  for (let i = 0; i < sequence.length; i += 1) {
    if (sequence[i] !== ":") count += 1;
  }
  return count;
}

/**
 * The sequence with one or more positions replaced.
 *
 * THE INDEX IS INTO THE SEQUENCE THAT WAS FOLDED, which is not always what is
 * in the box: the box is editable while a structure is on screen, so a click on
 * residue 40 of a 59-mer could otherwise be applied to a sequence the reader
 * had since trimmed to 30. The caller passes the folded sequence for that
 * reason, and this refuses an index it cannot honour rather than silently
 * writing at the wrong place or off the end.
 *
 * @param {string} sequence the sequence the drawn structure was folded from
 * @param {number | Array<number> | Set<number>} index 0-based position(s)
 * @param {string} residue one-letter code to put there
 * @returns {string}
 */
export function substitute(sequence, index, residue) {
  if (typeof residue !== "string" || residue.length !== 1 || !(residue in THREE_LETTER)) {
    throw new RangeError(`${residue} is not one of the twenty amino acids`);
  }
  const total = totalResidueCount(sequence);
  if (Array.isArray(index) || index instanceof Set) {
    const indices = Array.from(index);
    if (indices.length === 0) return sequence;
    const charIndices = [];
    for (const i of indices) {
      if (!Number.isInteger(i) || i < 0 || i >= total) {
        throw new RangeError(`position ${i} is outside a ${total}-residue sequence`);
      }
      charIndices.push(residueToCharIndex(sequence, i));
    }
    const chars = [...sequence];
    for (const ci of charIndices) chars[ci] = residue;
    return chars.join("");
  }
  if (!Number.isInteger(index) || index < 0 || index >= total) {
    throw new RangeError(`position ${index} is outside a ${total}-residue sequence`);
  }
  const charIdx = residueToCharIndex(sequence, index);
  return sequence.slice(0, charIdx) + residue + sequence.slice(charIdx + 1);
}

/** How a mutation is written in the status line: the old residue, the position, the new one. */
export function mutationName(sequence, index, residue) {
  if (Array.isArray(index) || index instanceof Set) {
    const indices = Array.from(index).sort((a, b) => a - b);
    if (indices.length === 0) return "";
    if (indices.length === 1) {
      const i = indices[0];
      const ci = residueToCharIndex(sequence, i);
      const letter = ci >= 0 ? sequence[ci] : "UNK";
      return `${letter}${i + 1}${residue}`;
    }
    return indices.map((i) => {
      const ci = residueToCharIndex(sequence, i);
      const letter = ci >= 0 ? sequence[ci] : "UNK";
      return `${letter}${i + 1}${residue}`;
    }).join(", ");
  }
  const ci = residueToCharIndex(sequence, index);
  const letter = ci >= 0 ? sequence[ci] : "UNK";
  return `${letter}${index + 1}${residue}`;
}

/**
 * How far a pointer may travel between press and release and still be a click.
 *
 * FOUR, because that is py2Dmol's own number for the same decision - see the
 * `moved < 4` in its pointerup handler. The two have to agree: they are both
 * deciding whether one press was a pick or a rotation, and if they disagreed a
 * drag could open this panel without highlighting anything, or highlight
 * something without opening the panel.
 */
export const CLICK_SLOP_PX = 4;

/**
 * Was that press a click, or the end of a drag?
 *
 * A rotation is pointerdown, a lot of pointermove, and then a CLICK at the
 * release - so anything listening for clicks alone hears every rotation as a
 * pick on whatever the pointer happened to stop over.
 *
 * @param {{x: number, y: number}} [from] where the pointer went down
 * @param {{clientX: number, clientY: number}} to where it came up
 */
export function wasClick(from, to) {
  if (from === undefined || from === null) return false;
  return Math.hypot(to.clientX - from.x, to.clientY - from.y) < CLICK_SLOP_PX;
}

/**
 * Which residue of the sequence is under a pointer, or -1.
 *
 * TAKES THE VIEWER RATHER THAN READING ONE, so the rule can be tested without
 * a GPU, a fold and half a gigabyte of weights behind it.
 *
 * Two unmappings, and both matter. `pickResidueAt` answers with the ATOM it
 * hit, which for a drawn side chain is a position appended past the end of the
 * sequence - `sidechainMap` walks it back to the residue that owns it, which is
 * what makes clicking the tip of a tryptophan select the tryptophan. (py2Dmol's
 * own click-to-select does this walk internally, so `residueSelection` needs
 * none of it; picking directly is what buys the pointer coordinates a popup
 * has to be placed at, and this is the price.)
 *
 * And then the answer has to be a residue of the SEQUENCE: appended atoms sit
 * past its end, and a non-protein position is not something a substitution
 * means anything for. Both are reachable by an ordinary click.
 *
 * @param {object} viewer a py2Dmol embed viewer
 * @param {number} sequenceLength residues in the sequence that was folded
 * @param {number} clientX
 * @param {number} clientY
 * @returns {number} 0-based sequence index, or -1
 */
export function residueAt(viewer, sequenceLength, clientX, clientY) {
  const hit = viewer?.pickResidueAt?.(clientX, clientY) ?? -1;
  if (hit < 0) return -1;
  const atom = viewer.sidechainMap?.get(hit);
  const residue = atom ? atom.owner : hit;
  if (!Number.isInteger(residue) || residue < 0 || residue >= sequenceLength) return -1;
  if (viewer.positionTypes?.[residue] !== "P") return -1;
  return residue;
}

/**
 * The mutation panel: a strip docked under the viewer.
 *
 * IT WAS A POPUP AND THAT WAS WRONG. A floating menu placed at the pointer sat
 * on top of the thing it was about, and it closed on any press outside itself -
 * which is every drag that rotates the structure. So looking at the residue you
 * had just clicked dismissed the menu for choosing what to do with it.
 *
 * Docked, both problems go away without needing to be solved: it covers
 * nothing, so it never has to move or be moved, and nothing it does not own can
 * dismiss it, so rotating is just rotating.
 *
 * IT MOUNTS OUTSIDE THE VIEWER BOX, which is the other half of the fix. Every
 * fold calls container.replaceChildren() to build its viewer; a panel inside
 * that box was taken out with the old canvas and had to be put back after each
 * rebuild, silently failing when it was not. A sibling is never detached, so
 * there is no attach() any more and no rule to remember. It also keeps clear of
 * py2Dmol's own play bar, which lives along the bottom of the shell.
 *
 * @param {HTMLElement} host the element under the viewer to build into
 * @param {(indices: number | number[], residue: string) => void} onPick
 */
export function createMutationPanel(host, onPick) {
  const panel = document.createElement("div");
  panel.id = "mutate-panel";
  panel.hidden = true;
  panel.setAttribute("role", "group");

  const bar = document.createElement("div");
  bar.className = "mutate-bar";
  const title = document.createElement("p");
  title.className = "mutate-title";
  bar.appendChild(title);
  const hint = document.createElement("p");
  hint.className = "mutate-hint";
  bar.appendChild(hint);

  let positions = [];
  let staged = "";

  const hide = () => { panel.hidden = true; positions = []; staged = ""; };

  const close = document.createElement("button");
  close.type = "button";
  close.className = "mutate-close";
  close.setAttribute("aria-label", "Close");
  close.textContent = "\u00d7";
  close.addEventListener("click", hide);
  bar.appendChild(close);
  panel.appendChild(bar);

  const grid = document.createElement("div");
  grid.className = "mutate-grid";
  panel.appendChild(grid);
  host.appendChild(panel);

  /** Mark which cell the residue currently is - or has been staged as. */
  const markCurrent = (residueLetter) => {
    for (const cell of grid.querySelectorAll(".mutate-cell")) {
      cell.classList.toggle("is-current", cell.dataset.residue === residueLetter);
    }
  };

  // MOST HYDROPHOBIC FIRST, which makes the strip its own legend: the cells
  // are painted by hydropathy, so ordering them by it turns the row into the
  // scale itself - orange at one end, blue at the other, each band a run.
  for (const code of RESIDUES_BY_HYDROPATHY) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "mutate-cell";
    button.dataset.residue = code;
    button.textContent = code;
    button.title = `${THREE_LETTER[code]} \u00b7 ${hydropathyBand(code).label}`;
    // THE SAME COLOUR THE SIDE CHAIN WILL TAKE. The viewer paints side chains
    // by hydropathy, so painting the cells the same way lets the answer be
    // read before the click: swapping an orange residue for a blue one is a
    // bigger change than swapping it for another orange one, and that is
    // exactly what the reader is trying to judge.
    button.style.setProperty("--cell", hydropathyColor(code));
    button.addEventListener("click", () => {
      if (positions.length === 0) return;
      staged = code;
      markCurrent(code);
      hint.textContent = positions.length > 1
        ? `${THREE_LETTER[code]} staged for ${positions.length} residues`
        : `${THREE_LETTER[code]} staged`;
      onPick(positions.length === 1 ? positions[0] : positions, code);
    });
    grid.appendChild(button);
  }

  /** Open it against one or multiple residues. */
  const show = (indices, residueLetter, label) => {
    positions = Array.isArray(indices) || indices instanceof Set
      ? Array.from(indices).sort((a, b) => a - b)
      : [indices];
    staged = "";
    title.textContent = `${label} \u2192`;
    panel.setAttribute("aria-label", `Mutate ${label}`);
    hint.textContent = "pick a replacement";
    markCurrent(residueLetter);
    panel.hidden = false;
  };

  // ...Escape still closes it, because that is what Escape does. Nothing else
  // does: a press on the canvas is a rotation, and it used to be a dismissal.
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hide();
  });

  return {
    show,
    hide,
    element: panel,
    get position() { return positions[0] ?? -1; },
    get positions() { return positions; },
    get staged() { return staged; },
  };
}
