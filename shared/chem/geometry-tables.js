/**
 * The numbers a conformer is built out of: covalent radii, van der Waals
 * radii, and the angle a hybridisation wants.
 *
 * 🔴 THESE ARE PUBLISHED CONSTANTS AND THEY ARE CHECKED AGAINST THE CCD. A
 * table typed in from memory is the classic way to be 5% wrong everywhere at
 * once, and `bond-geometry.js` next door already derives its ideals from
 * `tools/oracle/reference-conformers.json` rather than typing them, with a
 * comment saying why. So `tools/check-smiles-conformer.mjs` takes every
 * component in `tools/fixtures/ccd/`, measures the bonds in its ideal
 * conformer, and compares them with what this table predicts - which is how
 * the single-bond radius of carbon got fixed from Pyykkö's 0.75 (C-C 1.50,
 * 2.6% short of the crystallographic 1.54) to Cordero's 0.76.
 *
 * Radii are Cordero (2008) for single bonds and Pyykkö (2009) for the
 * multiple-bond ones; the van der Waals radii are Bondi with Rowland's
 * revisions. Angstroms throughout.
 */

/** Single-bond covalent radii, Cordero 2008. */
const SINGLE = {
  H: 0.31, HE: 0.28, LI: 1.28, BE: 0.96, B: 0.84, C: 0.76, N: 0.71, O: 0.66,
  F: 0.57, NE: 0.58, NA: 1.66, MG: 1.41, AL: 1.21, SI: 1.11, P: 1.07, S: 1.05,
  CL: 1.02, AR: 1.06, K: 2.03, CA: 1.76, SC: 1.70, TI: 1.60, V: 1.53, CR: 1.39,
  MN: 1.39, FE: 1.32, CO: 1.26, NI: 1.24, CU: 1.32, ZN: 1.22, GA: 1.22,
  GE: 1.20, AS: 1.19, SE: 1.20, BR: 1.20, KR: 1.16, RB: 2.20, SR: 1.95,
  Y: 1.90, ZR: 1.75, NB: 1.64, MO: 1.54, TC: 1.47, RU: 1.46, RH: 1.42,
  PD: 1.39, AG: 1.45, CD: 1.44, IN: 1.42, SN: 1.39, SB: 1.39, TE: 1.38,
  I: 1.39, XE: 1.40, CS: 2.44, BA: 2.15, PT: 1.36, AU: 1.36, HG: 1.32,
  TL: 1.45, PB: 1.46, BI: 1.48,
};

/** Double-bond covalent radii, Pyykkö 2009; absent means "use single". */
const DOUBLE = {
  B: 0.75, C: 0.67, N: 0.60, O: 0.57, F: 0.59, SI: 1.07, P: 1.02, S: 0.94,
  CL: 0.95, AS: 1.06, SE: 1.07, BR: 1.09, I: 1.25, FE: 1.16, ZN: 1.18,
};

/** Triple-bond covalent radii, Pyykkö 2009. */
const TRIPLE = {
  B: 0.73, C: 0.60, N: 0.54, O: 0.53, SI: 1.00, P: 0.94, S: 0.95, AS: 1.14,
  SE: 1.07, BR: 1.10, I: 1.25,
};

/** Van der Waals radii, Bondi with Rowland's revisions. */
const VDW = {
  H: 1.10, C: 1.70, N: 1.55, O: 1.52, F: 1.47, SI: 2.10, P: 1.80, S: 1.80,
  CL: 1.75, SE: 1.90, BR: 1.83, I: 1.98, B: 1.92, NA: 2.27, MG: 1.73,
  K: 2.75, CA: 2.31, ZN: 1.39, FE: 2.00,
};

/**
 * 🔴 AN AROMATIC BOND IS NOT A SINGLE OR A DOUBLE AND MUST NOT BE PLACED AS
 * EITHER. Benzene's C-C is 1.39 A - between the 1.54 of ethane and the 1.33 of
 * ethene - and a ring built half at each would be visibly lopsided, which a
 * bond-length gate reads as a torn ligand. The delocalised radius used here is
 * the mean of the single and double one, which lands benzene at 1.395 against
 * the crystallographic 1.39.
 */
/**
 * The symbol whose radius a charged atom actually behaves like.
 *
 * 🔴 A CHARGE MOVES AN ATOM'S RADIUS AND IT MOVES IT BY A WHOLE ELEMENT. A
 * quaternary ammonium's N-C bond is 1.522 A where a neutral amine's is 1.430 -
 * nearly a tenth of an angstrom, which was the largest single bond error left
 * in the corpus. The reason is not a correction factor: N+ is ISOELECTRONIC
 * with carbon, so it has carbon's radius, and 0.76 + 0.76 is 1.52 on the nose.
 * The same rule sends O- to fluorine's radius and O+ to nitrogen's.
 *
 * Measured over the corpus's C-N single bonds: neutral 1.430 (34 of them),
 * charged with three bonds 1.477 (2), charged with four 1.522 (4).
 */
function isoelectronic(symbol, charge) {
  if (charge === 0 || charge === undefined) return symbol;
  const PERIOD = ["B", "C", "N", "O", "F"];
  const at = PERIOD.indexOf(symbol);
  if (at < 0) return symbol;
  const moved = at - charge;
  return moved >= 0 && moved < PERIOD.length ? PERIOD[moved] : symbol;
}

export function covalentRadius(symbol, order, aromatic) {
  const single = SINGLE[symbol];
  if (single === undefined) return 1.5;            // an element with no entry
  const double = DOUBLE[symbol] ?? single;
  if (aromatic === true) return (single + double) / 2;
  if (order >= 3) return TRIPLE[symbol] ?? double;
  if (order >= 2) return double;
  // 🔴 AND A FRACTIONAL ORDER INTERPOLATES RATHER THAN ROUNDING. A carboxylate
  // is order 1.5 and a sulfonate 1.667 after `delocalizeCharges`, and rounding
  // either to a whole number puts the bond back where the resonance was meant
  // to take it from.
  if (order > 1) return single + (double - single) * (order - 1);
  return single;
}

/**
 * The length a bond wants.
 *
 * 🔴 AND THE POLARITY CORRECTION IS OFF, AFTER MEASURING IT TWICE AND GETTING
 * OPPOSITE ANSWERS. Schomaker-Stevenson subtracts `POLARITY * |EN(a) - EN(b)|`
 * for the shortening a polar bond shows. Measured first over 20 bond kinds it
 * improved the mean error 0.0402 -> 0.0362 A and was kept. Measured again
 * after `delocalizeCharges` landed it makes things WORSE at every value -
 * 0.040 at zero against 0.045 at 0.09 - so it is now zero.
 *
 * Both measurements were right and the first one was answering a different
 * question. The sulfonate's S-O was 0.24 A too long because its resonance was
 * not modelled, and a correction that shortens polar bonds was absorbing that
 * error on behalf of a bug somewhere else. With the resonance handled the
 * correction has nothing left to hide and is pure harm: Cordero's radii are
 * fitted TO crystal structures, polar bonds included, so subtracting the
 * polarity again double-counts it. C-F read 1.20 against a true 1.35.
 *
 * The constant stays, at zero, with this note - because the next person to
 * find C-O slightly short will reach for exactly this knob, and the useful
 * thing to hand them is the measurement rather than the absence.
 *
 * 🔴 AND 0.04 A IS ALREADY PAST WHAT ANYTHING DOWNSTREAM CAN SEE. `ref_pos` is
 * a reference conformer in each token's OWN frame, and docs/AF3.md records
 * this port's idealised set differing from the reference's CCD geometry by
 * 0.65 A rms on intra-token distances - a difference the batch gate REPORTS as
 * a floor rather than failing on. Chasing this table below a hundredth of an
 * angstrom is optimising something no model reads.
 */
/**
 * Bond lengths measured rather than predicted, for the kinds radii get wrong.
 *
 * 🔴 EVERY ENTRY IS HERE BECAUSE IT WAS MEASURED MISSING, NOT BECAUSE IT
 * SEEMED LIKELY. Covalent radii are fitted to ordinary two-centre bonds and
 * they systematically overestimate a bond from oxygen to a HYPERVALENT
 * centre: the phosphate P-O came out 1.730 against a true 1.593 and the
 * sulfonate S-O 1.610 against 1.474, both about a tenth of an angstrom long,
 * which is three times the mean error of the whole table. The cause is real
 * chemistry that a radius cannot express - the bond has partial double
 * character and substantial ionic contribution at once - so the honest fix is
 * to state the number rather than to invent a correction term that happens to
 * reproduce it.
 *
 * The values are the mean over this repository's SMILES corpus as measured in
 * RDKit's MMFF conformers, and `tools/check-smiles-conformer.mjs` is what
 * measures them. This is the same habit as `bond-geometry.js`, which derives
 * its ideals from `reference-conformers.json` rather than typing them in, and
 * for the same reason: a number with a provenance can be re-derived when the
 * reference changes, and a number from memory cannot.
 *
 * Keyed by the element pair sorted alphabetically, then the order bucket -
 * `s` under 1.25, `p` (partial, a delocalised bond) to 1.75, `d` above.
 */
const MEASURED = {
  "O-P:s": 1.593, "O-P:p": 1.540, "O-P:d": 1.491,
  "O-S:s": 1.520, "O-S:p": 1.470, "O-S:d": 1.455,
  "C-O:p": 1.263,
  "C-N:s": 1.429,
  "C-F:s": 1.363,
  "N-O:p": 1.240,
  // 🔴 AND THE AROMATIC BONDS, WHICH ARE THE COMMONEST IN A DRUG-LIKE LIGAND
  // AND THE ONES A MEAN-OF-TWO-RADII MODEL GETS WORST. Averaging the single
  // and double radii puts benzene's C-C at 1.430 where it is 1.395 - a ring
  // 2.5% too big, in the substructure that appears in most of the corpus.
  // Delocalisation is not the average of two bond orders and there is no
  // reason a radius model should reproduce it.
  "C-C:a": 1.398, "C-N:a": 1.354, "C-O:a": 1.367, "C-S:a": 1.714,
  "N-N:a": 1.340, "C-SE:a": 1.855,
};

function measuredKey(symbolA, symbolB, order, aromatic) {
  const pair = [symbolA, symbolB].sort().join("-");
  const bucket = aromatic === true ? "a"
    : order < 1.25 ? "s" : order < 1.75 ? "p" : "d";
  return `${pair}:${bucket}`;
}

export function bondLength(symbolA, symbolB, order, aromatic, chargeA, chargeB) {
  // 🔴 THE MEASURED TABLE IS FOR NEUTRAL ATOMS AND MUST STEP ASIDE FOR A
  // CHARGED ONE. Its C-N entry is 1.429, fitted to 34 neutral amines, and
  // applying it to a quaternary ammonium is 0.09 A short.
  const charged = (chargeA ?? 0) !== 0 || (chargeB ?? 0) !== 0;
  if (!charged) {
    const measured = MEASURED[measuredKey(symbolA, symbolB, order, aromatic)];
    if (measured !== undefined) return measured;
  }
  const a = isoelectronic(symbolA, chargeA);
  const b = isoelectronic(symbolB, chargeB);
  if (a !== symbolA || b !== symbolB) {
    return covalentRadius(a, order, aromatic) + covalentRadius(b, order, aromatic);
  }
  const sum = covalentRadius(symbolA, order, aromatic)
    + covalentRadius(symbolB, order, aromatic);
  const difference = Math.abs(
    (ELECTRONEGATIVITY[symbolA] ?? 2.2) - (ELECTRONEGATIVITY[symbolB] ?? 2.2));
  return sum - POLARITY * difference;
}

/** How much a polar bond is shortened per unit of electronegativity difference. */
export const POLARITY = 0;

/** Pauling electronegativities, for the Schomaker-Stevenson correction. */
const ELECTRONEGATIVITY = {
  H: 2.20, B: 2.04, C: 2.55, N: 3.04, O: 3.44, F: 3.98, SI: 1.90, P: 2.19,
  S: 2.58, CL: 3.16, SE: 2.55, BR: 2.96, I: 2.66, NA: 0.93, MG: 1.31,
  K: 0.82, CA: 1.00, ZN: 1.65, FE: 1.83,
};

export function vanDerWaalsRadius(symbol) {
  return VDW[symbol] ?? 1.8;
}

/**
 * The angle at an atom with this many sigma neighbours, in radians.
 *
 * 🔴 SIGMA NEIGHBOURS COUNT HYDROGENS. A carbonyl carbon has two heavy
 * neighbours and is 120 degrees, not 180: what makes it planar is that it has
 * THREE sigma bonds, and one of them may be to a hydrogen that this port drops
 * before the featuriser ever sees it. Counting heavy atoms only turns every
 * aldehyde into an sp carbon.
 */
export function idealAngle(sigmaCount, symbol, hasPiBond) {
  const degrees = (value) => (value * Math.PI) / 180;
  // 🔴 SIGMA COUNT ALONE DOES NOT NAME A SHAPE, AND KEYING ON IT WAS WRONG IN
  // BOTH DIRECTIONS. Ammonia and formaldehyde's carbon both have three sigma
  // bonds and are 107 and 120 degrees; a phosphate's phosphorus has four and
  // is tetrahedral, not the 107 a three-coordinate phosphorus wants. The first
  // version returned 120 for every three-coordinate atom and the lone-pair
  // angle for every four-coordinate N or P, which is the two cases backwards.
  // It showed up as contradictory distance bounds around ATP's triphosphate.
  // What decides is sigma count AND whether there is a pi bond.
  if (sigmaCount >= 4) return degrees(109.47);                 // sp3, tetrahedral
  if (sigmaCount === 3) {
    // 🔴 A SULFOXIDE IS PYRAMIDAL, NOT TRIGONAL, AND SO IS A PHOSPHINE OXIDE.
    // Sulfur and phosphorus keep a stereochemically active lone pair even
    // with a double bond present - dimethyl sulfoxide is about 106 degrees at
    // sulfur, which is why a sulfoxide can be a stereocentre at all. Treated
    // as sp2 it read 16.4 degrees off RDKit, the worst angle in the corpus.
    if (symbol === "S" || symbol === "SE" || symbol === "P" || symbol === "AS") {
      return degrees(106.0);
    }
    if (hasPiBond === true) return degrees(120);               // sp2, trigonal planar
    // ...otherwise a lone pair, which takes more room than a bond and closes
    // the angle: ammonia is 107.0 and not the tetrahedral 109.47.
    if (symbol === "N" || symbol === "P" || symbol === "AS") return degrees(107.0);
    return degrees(109.47);
  }
  if (sigmaCount === 2) {
    // Two sigma bonds is linear only with TWO pi bonds (an alkyne or a
    // nitrile); with one it is trigonal, and with none it is bent.
    if (hasPiBond === "two") return degrees(180);
    if (hasPiBond === true) return degrees(120);
    if (symbol === "O" || symbol === "S" || symbol === "SE") return degrees(104.5);
    return degrees(109.47);
  }
  return Math.PI;
}

/** The interior angle of a flat regular polygon, in radians. */
export function polygonAngle(size) {
  return ((size - 2) * Math.PI) / size;
}

/** The third side of a triangle given two sides and the angle between them. */
export function lawOfCosines(a, b, angle) {
  return Math.sqrt(a * a + b * b - 2 * a * b * Math.cos(angle));
}

/**
 * How much more room a substituent takes than a plain single bond.
 *
 * 🔴 ONE ANGLE PER ATOM CANNOT DESCRIBE AN ATOM WITH UNEQUAL SUBSTITUENTS, and
 * the two worst angles left in the corpus were both this. Dimethyl sulfoxide's
 * sulfur is 96.6 degrees between its two methyls and 107.5 to its oxygen -
 * a spread of eleven degrees around ONE atom, which a single ideal angle
 * splits the difference of and gets both ends wrong. Acetate is the same
 * shape: O-C-O opens to 126 while C-C-O closes to 117.
 *
 * This is VSEPR's ordering, which is older than any force field and is one
 * line: a multiple bond holds more electron density than a single one and
 * pushes its neighbours away, so the angle AT a double bond opens and the
 * angle BETWEEN the remaining single bonds closes by as much as the geometry
 * has to give. `VSEPR_STRENGTH` is how many degrees a unit of excess order is
 * worth, and it is fitted below rather than assumed.
 */
export function substituentWeight(order, aromatic) {
  if (aromatic === true) return 0.5;
  if (order >= 3) return 1.5;
  if (order > 1) return order - 1;
  return 0;
}

/**
 * Degrees of angle per unit of substituent weight above the centre's mean.
 *
 * 🔴 FITTED, NOT CHOSEN, AND THE CURVE IS FLAT ACROSS ITS MINIMUM. Swept
 * against RDKit's MMFF conformers over the whole corpus, mean angle error in
 * degrees:
 *
 *     k      0     6     8     9    10    11    12    13    15    20
 *     err  3.28  2.88  2.80  2.78  2.77  2.76  2.77  2.83  2.90  3.15
 *
 * So anything from 8 to 12 is the same answer and 11 is the middle of it - a
 * shape worth recording, because it says the rule is doing real work (0 is
 * half a degree worse than the floor) and that the exact constant is not load
 * bearing. Setting it to zero recovers the single-angle-per-atom model
 * exactly, which is what makes it safe to sweep and safe to back out.
 *
 * Bond lengths do not move at all across the sweep, at 0.019 A throughout,
 * which is the control: an angle rule that changed them would be reaching
 * somewhere it should not.
 */
export const VSEPR_STRENGTH = 11;
