/**
 * Put one cycle's prediction in the previous one's reference frame.
 *
 * 🔴 WITHOUT THIS THE FRAMES DO NOT LINE UP, AND py2Dmol CANNOT FIX IT ITSELF.
 * Two reasons, and the second is why the viewer's own alignment is no help:
 *
 *   * AF3's sampler calls `randomAugmentation` at the top of every step, so
 *     consecutive FOLDS sit in unrelated reference frames. Two cycles of one
 *     design differ by a rotation before they differ by anything meaningful,
 *     and played back they read as a structure being thrown around a room.
 *   * `addFrame` superposes a frame onto the one before it only when the two
 *     have the SAME NUMBER OF POSITIONS - the note above superposeOnto() in
 *     web/morph.js says so - and a redesigned chain has different side chains,
 *     so the atom count changes every cycle. This is the same problem a point
 *     mutation has on the fold page, one step further along.
 *
 * 🔴 AND THE FIT IS ON THE TARGET, NOT ON EVERYTHING. The target is the one
 * thing that does not change across a run, so fitting on it holds it still and
 * lets the binder move against it - which is the thing worth watching. Fitting
 * on every atom instead splits the difference between a fixed target and a
 * moving binder, so BOTH drift, and a binder that found its site looks like a
 * binder that wandered. A monomer has no target and falls back to its own
 * alpha carbons, which are a constant count because the length is.
 *
 * The arithmetic is py2Dmol's `superpose`, injected rather than imported: this
 * file is a PDB transform, it is tested on the CPU where there is no viewer,
 * and the Kabsch fit is not the part that was ever going to be wrong.
 */

/** Fixed columns of a PDB coordinate line. */
const RECORD = [0, 6];
const NAME = [12, 16];
const RES_NAME = [17, 20];
const CHAIN = 21;
// The residue number AND its insertion code, together: two residues can share
// a number and be told apart only by the code, which is what a PDB uses when a
// structure is numbered against a reference it does not quite match.
const RES_SEQ = [22, 27];
// The B-factor, which a predicted structure uses to carry pLDDT.
const B_FACTOR = [60, 66];
const X = [30, 38];
const Y = [38, 46];
const Z = [46, 54];

/**
 * Every coordinate-bearing line of a PDB, as points.
 *
 * @param {string} pdb
 * @returns {{lines: string[], at: number[], points: number[][],
 *            names: string[], chains: string[], residueNames: string[],
 *            residues: string[]}} `at[i]` is the line index point `i` came
 *   from, so a rewrite can put it back without re-parsing. `residues[i]` is
 *   the residue NUMBER plus its insertion code, as written - the key a
 *   template's residue map has to use, because grouping by position in this
 *   list closes up every gap the structure has.
 */
export function coordinateAtoms(pdb) {
  const lines = pdb.split("\n");
  const at = [];
  const points = [];
  const names = [];
  const chains = [];
  const residueNames = [];
  const residues = [];
  const bFactors = [];
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const record = line.slice(...RECORD);
    if (record !== "ATOM  " && record !== "HETATM") continue;
    const x = Number(line.slice(...X));
    const y = Number(line.slice(...Y));
    const z = Number(line.slice(...Z));
    // A line whose coordinates do not parse is not one to move; leaving it
    // out of `at` leaves it untouched by the rewrite as well.
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    at.push(index);
    points.push([x, y, z]);
    names.push(line.slice(...NAME).trim());
    chains.push(line[CHAIN]);
    residueNames.push(line.slice(...RES_NAME).trim());
    residues.push(line.slice(...RES_SEQ).trim());
    // ...0 when the column is blank or unparseable, which reads as "no
    // confidence stated" rather than as a confident zero. Only AlphaFold DB
    // puts anything meaningful here.
    const bFactor = Number(line.slice(...B_FACTOR));
    bFactors.push(Number.isFinite(bFactor) ? bFactor : 0);
  }
  return { lines, at, points, names, chains, residueNames, residues, bFactors };
}

/**
 * Which of those atoms to fit on: the alpha carbons of the given chains.
 *
 * @param {{names: string[], chains: string[]}} atoms
 * @param {(chain: string) => boolean} wanted
 * @returns {number[]} indices into `atoms.points`
 */
export function alphaCarbons(atoms, wanted) {
  const indices = [];
  for (let index = 0; index < atoms.names.length; index += 1) {
    if (atoms.names[index] === "CA" && wanted(atoms.chains[index])) indices.push(index);
  }
  return indices;
}

/**
 * A PDB with every coordinate replaced, and nothing else touched.
 *
 * 🔴 THE COLUMNS ARE REBUILT, NOT THE LINE. A PDB is a fixed-column format and
 * everything after column 54 - occupancy, the B-factor the viewer colours by,
 * the element - has to survive exactly. So this splices the three coordinate
 * fields and keeps both ends of the line verbatim.
 *
 * @param {{lines: string[], at: number[]}} atoms
 * @param {number[][]} points one per entry of `atoms.at`
 * @returns {string}
 */
export function rewriteCoordinates(atoms, points) {
  if (points.length !== atoms.at.length) {
    throw new RangeError(`${points.length} points for ${atoms.at.length} atoms`);
  }
  const lines = [...atoms.lines];
  const field = (value) => {
    const text = value.toFixed(3);
    // 🔴 A COORDINATE THAT DOES NOT FIT IS TRUNCATED, NOT WIDENED. Eight
    // columns hold -9999.999 and no more, and a wider number would push every
    // field after it along - which is a PDB that parses as something else
    // rather than one that fails. It cannot happen to a real structure; it
    // could to a fit that went wrong, and that should look wrong.
    return text.length > 8 ? text.slice(0, 8) : text.padStart(8);
  };
  for (let index = 0; index < atoms.at.length; index += 1) {
    const line = lines[atoms.at[index]];
    const [x, y, z] = points[index];
    lines[atoms.at[index]] = line.slice(0, X[0])
      + field(x) + field(y) + field(z) + line.slice(Z[1]);
  }
  return lines.join("\n");
}

/**
 * `pdb`, moved onto `reference`.
 *
 * @param {(points: number[][], reference: number[][],
 *          pairs: {from: number[], to: number[]}) => number[][]} superpose
 *   py2Dmol's `superpose`: it fits on the named subset and moves everything.
 * @param {string} pdb
 * @param {string} reference
 * @param {{designed?: string}} [options] the designed chain's letter, which is
 *   the one chain NOT fitted on when there is anything else to fit on.
 * @returns {{pdb: string, fitted: number, on: "target"|"designed"|"none"}}
 *   `on` says what the fit used, and "none" means the pdb came back unchanged.
 */
export function superposePdb(superpose, pdb, reference, options = {}) {
  const designed = options.designed ?? "A";
  const atoms = coordinateAtoms(pdb);
  const target = coordinateAtoms(reference);

  // The target chains first; the designed chain only if there are none.
  let on = "target";
  let from = alphaCarbons(atoms, (chain) => chain !== designed);
  let to = alphaCarbons(target, (chain) => chain !== designed);
  if (from.length < 3 || to.length < 3) {
    on = "designed";
    from = alphaCarbons(atoms, (chain) => chain === designed);
    to = alphaCarbons(target, (chain) => chain === designed);
  }
  // 🔴 EQUAL COUNTS OR NO FIT, because these are paired index for index. The
  // chains being fitted on are the ones that do not change across a run, so
  // unequal counts mean the caller is comparing two different jobs - and a fit
  // that silently paired the first N of each would be a plausible-looking
  // structure in the wrong place.
  if (from.length < 3 || from.length !== to.length) {
    return { pdb, fitted: 0, on: "none" };
  }
  const moved = superpose(atoms.points, target.points, { from, to });
  return { pdb: rewriteCoordinates(atoms, moved), fitted: from.length, on };
}

/**
 * A whole cycle - its sampler trajectory and the structure it settled to -
 * moved by ONE transform, the one that fits the settled structure.
 *
 * 🔴 EVERY FRAME FITTED SEPARATELY WOULD BE WRONG, AND WOULD LOOK RIGHT. A
 * diffusion trajectory starts as noise: at step 1 the "target chain" is a
 * cloud, so fitting on its alpha carbons fits on nothing and lands the frame
 * somewhere arbitrary. The frames of one cycle are ALREADY in a common
 * reference frame - fittedPdb in web/af3-model.js puts them there before they
 * leave the fold - so the whole cycle needs one rigid move, derived from the
 * only frame that has a real structure in it.
 *
 * 🔴 AND THE TRANSFORM IS OBTAINED BY MOVING EVERYTHING AT ONCE, because
 * py2Dmol's `superpose` returns moved POINTS and not a matrix. Concatenating
 * the trajectory behind the settled structure and naming the settled
 * structure's alpha carbons as the fit set gives exactly that: one fit, one
 * transform, applied to every point handed in. No matrix arithmetic here to
 * get wrong.
 *
 * @param {Function} superpose py2Dmol's
 * @param {string[]} frames the trajectory, in order
 * @param {string} settled the structure the cycle finished at
 * @param {string} reference what to move onto
 * @param {{designed?: string}} [options]
 * @returns {{frames: string[], settled: string, fitted: number, on: string}}
 */
export function superposeCycle(superpose, frames, settled, reference, options = {}) {
  const designed = options.designed ?? "A";
  const anchor = coordinateAtoms(settled);
  const target = coordinateAtoms(reference);

  let on = "target";
  let from = alphaCarbons(anchor, (chain) => chain !== designed);
  let to = alphaCarbons(target, (chain) => chain !== designed);
  if (from.length < 3 || to.length < 3) {
    on = "designed";
    from = alphaCarbons(anchor, (chain) => chain === designed);
    to = alphaCarbons(target, (chain) => chain === designed);
  }
  if (from.length < 3 || from.length !== to.length) {
    return { frames, settled, fitted: 0, on: "none" };
  }

  // The settled structure FIRST, so `from` indexes straight into the
  // concatenation with no offset to get wrong.
  const parsed = frames.map((frame) => coordinateAtoms(frame));
  const points = [...anchor.points];
  for (const atoms of parsed) points.push(...atoms.points);

  const moved = superpose(points, target.points, { from, to });
  let at = anchor.points.length;
  const movedFrames = parsed.map((atoms) => {
    const slice = moved.slice(at, at + atoms.points.length);
    at += atoms.points.length;
    return rewriteCoordinates(atoms, slice);
  });
  return {
    frames: movedFrames,
    settled: rewriteCoordinates(anchor, moved.slice(0, anchor.points.length)),
    fitted: from.length,
    on,
  };
}
