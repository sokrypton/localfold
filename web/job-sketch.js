/**
 * THE JOB AS A PICTURE, BEFORE ANYTHING HAS RUN (design.html, 2026-10-27).
 *
 * The viewer used to stay empty until a structure was loaded, and then drew only that structure -
 * so a newcomer saw nothing of what step 3 says will be built. It now draws the job: the loaded
 * structure's real atoms, and for every piece still to be designed (a `Designed` or `Sequence`
 * span) a trace of its length - C-alphas, or C4's for DNA and RNA - faded, joining the pieces it sits
 * between.
 *
 * 🔴 A SKETCH, NOT A PREDICTION, and drawn so it cannot be read as one: its geometry is a curve or a
 * ball of the right number of 3.8 A steps, nothing a model computed, and the page fades it and says
 * so beside the viewer. What it carries is the arrangement - how long, attached where, beside which
 * hotspots - which is exactly what the span rows say in text.
 *
 * 🔴 EACH GHOST PIECE IS ITS OWN CHAIN LETTER, so the viewer never bonds one to a real residue, and
 * so a pick can tell a ghost from the structure: `ghosts` names those letters, and nothing a
 * condition can name lives on one.
 */
import { chainOfSpans } from "./contig-ui.js";

const STEP = 3.8;   // A, C-alpha to C-alpha
// 🔴 A DNA OR RNA PIECE IS SKETCHED AS NUCLEOTIDES (2026-10-28): it was GLY C-alphas whatever it was
// made of, so py2Dmol drew a nucleic piece as protein. Each residue is one C4' - the atom py2Dmol
// traces a nucleic chain through - a B-DNA step apart, named by its base (DA..DT, A..U) or, still
// to be designed, as the dictionary's unknown nucleotide (DN, N).
const NUCLEIC_STEP = 6.5;   // A, C4' to C4' along a strand
// (one letter, tested as one: "ACGT".includes("") is true, which named a designed piece "D")
const NUCLEIC_NAME = {
  dna: (letter) => (letter.length === 1 && "ACGT".includes(letter) ? `D${letter}` : "DN"),
  rna: (letter) => (letter.length === 1 && "ACGU".includes(letter) ? letter : "N"),
};

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);
function unit(a, fallback = [1, 0, 0]) {
  const n = norm(a);
  return n > 1e-6 ? scale(a, 1 / n) : fallback;
}
/** Any unit vector perpendicular to `a`. */
function perpendicular(a) {
  const trial = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  return unit(cross(a, trial));
}

/** How many residues a piece will be: a range's midpoint, a sequence's length. */
export function pieceLength(span) {
  if (span.kind === "generated") {
    const min = Number(span.min);
    const max = Number.isFinite(Number(span.max)) && String(span.max ?? "").trim() !== "" ? Number(span.max) : min;
    return Number.isFinite(min) && min > 0 ? Math.max(1, Math.round((min + Math.max(min, max)) / 2)) : 0;
  }
  if (span.kind === "sequence") return String(span.value ?? "").replace(/[^A-Za-z]/g, "").length;
  return 0;
}

/**
 * The radius of a ball of `n` residues: a real protein's outer radius (~17 A at 100 residues),
 * which is what leaves room between the turns for the sketch to read as a wire globe rather than
 * a helical spring.
 */
export function ballRadius(n, step = STEP) {
  return Math.max(step, 1.6 * step * Math.sqrt(n / (4 * Math.PI)));
}

/** Rotate `p` by the rotation that takes unit vector `from` onto unit vector `to`. */
function rotateOnto(p, from, to) {
  const axis = cross(from, to);
  const sin = norm(axis);
  const cos = dot(from, to);
  if (sin < 1e-9) return cos > 0 ? p : scale(p, -1);
  const k = scale(axis, 1 / sin);
  // Rodrigues
  return add(add(scale(p, cos), scale(cross(k, p), sin)), scale(k, dot(k, p) * (1 - cos)));
}

/**
 * `n` points 3.8 A apart on a spiral round a sphere about `centre`, the first on the axis `pole`.
 *
 * A protein-sized ball for a piece with at most one neighbour: a 100-residue span laid out straight
 * would be 380 A - a stick off the edge of the picture. The spiral stays clear of the sphere's poles,
 * where it would wind tight enough for the viewer to call it a helix.
 */
const PROLATE = 1.25;

export function ball(n, centre, pole, radius = ballRadius(n), step = STEP) {
  const want = Math.max(0, n - 1) * step;
  const CAP = 0.45;   // rad kept clear at each pole
  const SAMPLES = 2048;
  // ...a little longer along its axis than across, so the viewer's best view looks at it from the
  // side - a globe - rather than down the axis at a set of rings
  const at = (c, u) => {
    const theta = CAP + u * (Math.PI - 2 * CAP);
    const phi = c * theta;
    return [radius * Math.sin(theta) * Math.cos(phi), radius * Math.sin(theta) * Math.sin(phi),
      PROLATE * radius * Math.cos(theta)];
  };
  const lengthOf = (c) => {
    let total = 0;
    let last = at(c, 0);
    for (let s = 1; s <= SAMPLES; s += 1) {
      const p = at(c, s / SAMPLES);
      total += norm(sub(p, last));
      last = p;
    }
    return total;
  };
  // the winding: as many turns as the length needs (none, for a piece shorter than a meridian)
  let c = 0;
  if (lengthOf(0) < want) {
    let lo = 0;
    let hi = 1;
    while (lengthOf(hi) < want) hi *= 2;
    for (let i = 0; i < 40; i += 1) {
      const mid = (lo + hi) / 2;
      if (lengthOf(mid) < want) lo = mid; else hi = mid;
    }
    c = (lo + hi) / 2;
  }
  // ...sampled every 3.8 A of arc
  const local = [at(c, 0)];
  let travelled = 0;
  let last = local[0];
  for (let s = 1; s <= SAMPLES && local.length < n; s += 1) {
    const p = at(c, s / SAMPLES);
    travelled += norm(sub(p, last));
    last = p;
    if (travelled >= local.length * step) local.push(p);
  }
  while (local.length < n) local.push(at(c, 1));
  // turned so the first point sits on the pole axis, where the neighbour is
  const z = unit(pole);
  const first = unit(local[0]);
  return local.map((p) => add(centre, rotateOnto(p, first, z)));
}

/**
 * `n` points between `from` and `to`, each a 3.8 A step from the last: straight when the gap
 * already needs that, otherwise a circular arc bowed along `outward`.
 *
 * 🔴 AN ARC, NOT A PARABOLA: equal steps along a curve are equal CHORDS only where its curvature
 * is constant, and a long linker over a short gap is a tight bow - a parabola's apex put 2.4 A
 * steps there. A circle of the right size takes any length, up to nearly a full loop.
 */
export function bridge(n, from, to, outward, step = STEP) {
  const chord = sub(to, from);
  const gap = norm(chord);
  const along = unit(chord);
  const steps = n + 1;
  if (gap >= steps * step || gap < 1e-6) {
    return Array.from({ length: n }, (_, k) => add(from, scale(chord, (k + 1) / steps)));
  }
  let bow = sub(outward, scale(along, dot(outward, along)));
  bow = norm(bow) > 1e-6 ? unit(bow) : perpendicular(along);
  // the arc's angle: sin(theta/2) / sin(theta / 2steps) = gap / STEP, which falls from `steps`
  // at theta 0 to 0 at a full turn
  const ratio = (theta) => Math.sin(theta / 2) / Math.sin(theta / (2 * steps));
  let lo = 1e-9;
  let hi = 2 * Math.PI - 1e-9;
  for (let i = 0; i < 60; i += 1) {
    const mid = (lo + hi) / 2;
    if (ratio(mid) > gap / step) lo = mid; else hi = mid;
  }
  const theta = (lo + hi) / 2;
  const radius = step / (2 * Math.sin(theta / (2 * steps)));
  const centre = add(scale(add(from, to), 0.5), scale(bow, -radius * Math.cos(theta / 2)));
  const start = Math.PI / 2 + theta / 2;
  return Array.from({ length: n }, (_, k) => {
    const angle = start - ((k + 1) * theta) / steps;
    return add(centre, add(scale(along, radius * Math.cos(angle)), scale(bow, radius * Math.sin(angle))));
  });
}

/**
 * The job's atoms: the structure's, then a ghost chain per piece to be designed.
 *
 * @param {object[]} structure atoms as `inputStructureAtoms` gives them (`chainId` is the drawn
 *   letter; `chainOf` maps it back to the structure's own chain name)
 * @param {Map<string, string>} chainOf drawn letter to structure chain
 * @param {object[]} spans the span list's rows
 * @param {{hotspots?: {chain: string, resId: number}[]}} [options]
 * @returns {{atoms: object[], ghosts: Map<string, {span: number, chain: number}>}} `ghosts` maps
 *   each ghost letter to the span it sketches and the design chain it belongs to
 */
export function sketchJob(structure, chainOf, spans, { hotspots = [] } = {}) {
  // a residue's anchor: its C-alpha, or a nucleotide's C1'
  const ca = new Map();
  for (const atom of structure) {
    const name = String(atom.atomName).trim();
    if ((name !== "CA" && name !== "C1'") || atom.isHetero) continue;
    ca.set(`${chainOf.get(atom.chainId) ?? atom.chainId}:${Number(atom.resId)}`, [atom.x, atom.y, atom.z]);
  }
  const centreOf = (points) => (points.length === 0 ? [0, 0, 0]
    : scale(points.reduce((s, p) => add(s, p), [0, 0, 0]), 1 / points.length));
  // 🔴 OVER EVERY ATOM, not the C-alphas: a ligand-only or DNA-only structure has none, and its
  // centre was the origin, wherever the molecule actually is
  const all = structure.map((atom) => [atom.x, atom.y, atom.z]);
  const centroid = centreOf(all);
  const extent = all.reduce((m, p) => Math.max(m, norm(sub(p, centroid))), 0);
  const ligand = structure.filter((atom) => atom.isHetero).map((atom) => [atom.x, atom.y, atom.z]);
  const pocket = spans.some((span) => span.kind === "ligand") && ligand.length > 0;
  const ligandCentre = centreOf(ligand);
  const ligandExtent = ligand.reduce((m, p) => Math.max(m, norm(sub(p, ligandCentre))), 0);
  const hot = hotspots.map((h) => ca.get(`${h.chain}:${Number(h.resId)}`)).filter(Boolean);
  const hotCentre = hot.length === 0 ? null : scale(hot.reduce((s, p) => add(s, p), [0, 0, 0]), 1 / hot.length);

  const used = new Set(structure.map((atom) => atom.chainId));
  const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const nextLetter = () => {
    const letter = [...LETTERS].find((l) => !used.has(l));
    if (letter === undefined) throw new Error("too many chains to sketch the job");
    used.add(letter);
    return letter;
  };

  // the pieces of each design chain, in order
  const chainIndex = chainOfSpans(spans);
  const chains = new Map();
  spans.forEach((span, index) => {
    const chain = chainIndex[index];
    if (chain === null) return;
    if (!chains.has(chain)) chains.set(chain, []);
    chains.get(chain).push(index);
  });
  const ends = (span) => {
    const chain = String(span.chain ?? "").trim();
    return [ca.get(`${chain}:${Number(span.first)}`), ca.get(`${chain}:${Number(span.last)}`)];
  };

  const atoms = [];
  const ghosts = new Map();
  let free = 0;
  for (const [chain, indices] of chains) {
    let previous = null;
    indices.forEach((index, at) => {
      const span = spans[index];
      if (span.kind === "segment") {
        const [, last] = ends(span);
        previous = last ?? previous;
        return;
      }
      const n = pieceLength(span);
      if (n === 0) return;
      const polymer = span.polymer ?? "protein";
      const nucleic = NUCLEIC_NAME[polymer];
      const step = nucleic === undefined ? STEP : NUCLEIC_STEP;
      const nextSpan = spans[indices[at + 1]];
      const next = nextSpan?.kind === "segment" ? ends(nextSpan)[0] ?? null : null;
      let points;
      if (previous !== null && next !== null) {
        points = bridge(n, previous, next, sub(scale(add(previous, next), 0.5), centroid), step);
      } else if (previous !== null || next !== null) {
        // one neighbour: a ball touching it, on the side away from the structure
        const anchor = previous ?? next;
        const out = unit(sub(anchor, centroid));
        const radius = ballRadius(n, step);
        points = ball(n, add(anchor, scale(out, radius + step)), scale(out, -1), radius, step);
        if (previous === null) points.reverse();   // ...ending at the piece after it
      } else {
        // a chain of its own: beside the hotspots if there are any, round the ligand of a pocket
        // job, else beside the structure
        let radius = ballRadius(n, step);
        let centre;
        if (hotCentre === null && pocket && free === 0) {
          radius = Math.max(radius, ligandExtent + 5);
          centre = ligandCentre;
        } else if (all.length === 0) {
          centre = [free * 2 * (radius + 6), 0, 0];
        } else if (hotCentre !== null) {
          const out = unit(sub(hotCentre, centroid));
          centre = add(add(hotCentre, scale(out, radius + 8)), scale(perpendicular(out), free * 2 * (radius + 6)));
        } else {
          centre = add(centroid, [extent + radius + 8 + free * 2 * (radius + 6), 0, 0]);
        }
        points = ball(n, centre, [0, 0, 1], radius, step);
        free += 1;
      }
      const letter = nextLetter();
      ghosts.set(letter, { span: index, chain });
      // a Sequence piece's own letters; a Designed one has none yet
      const letters = span.kind === "sequence" ? String(span.value ?? "").replace(/[^A-Za-z]/g, "").toUpperCase() : "";
      points.forEach(([x, y, z], k) => {
        atoms.push({ chainId: letter, resId: k + 1, x, y, z, isHetero: false, element: "C",
          ...(nucleic === undefined ? { resName: "GLY", atomName: "CA" }
            : { resName: nucleic(letters[k] ?? ""), atomName: "C4'" }) });
      });
      previous = points[points.length - 1];
    });
  }
  return { atoms: [...structure, ...atoms], ghosts };
}
