/**
 * A result that cannot be one: coordinates that are not numbers, lie further out than a structure file has columns for or are
 * all one point, or a confidence that is not a number.
 *
 * A driver's bug, a failing card or a kernel that overflowed gives these with no error anywhere, and nothing after them minds. The
 * status line read "pLDDT NaN", the file held NaNs, and the trunk behind them was kept to go on from, so the same fold read NaN
 * again a second later, and again, until a reload.
 */
export const IMPOSSIBLE_FOLD = "The numbers this fold came back with cannot be right, so it has no result. Press Fold to run it"
  + " again. If it happens again, the likely cause is a fault in the GPU or its driver, not the sequence.";

/** Further than this and a coordinate has run out of the columns a PDB file gives it. */
const FARTHEST = 9999;
const COORDINATE_COLUMNS = [[30, 38], [38, 46], [46, 54]];

const possible = (value) => Number.isFinite(value) && Math.abs(value) <= FARTHEST;

function coordinatesAreImpossible(pdb) {
  let first;
  let atoms = 0;
  let spread = false;
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    const here = COORDINATE_COLUMNS.map(([from, to]) => Number(line.slice(from, to)));
    if (!here.every(possible)) return true;
    first ??= here;
    atoms += 1;
    spread ||= here.some((value, axis) => value !== first[axis]);
  }
  // A card that hands back nothing but zeros has given a structure with every atom on one point.
  return atoms > 1 && !spread;
}

/** `prediction`: what a fold hands the page, whichever model made it. Where it has no confidence head it has nothing there to read. */
export function isImpossibleFold({ pdb, confidence }) {
  if (typeof pdb === "string" && coordinatesAreImpossible(pdb)) return true;
  const mean = confidence?.meanPlddt;
  return mean !== undefined && !Number.isFinite(mean);
}
