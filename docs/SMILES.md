# SMILES: folding a ligand nobody has a code for

What it took to accept `OC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12` where the
page used to say *"`smiles` names a ligand by structure, and this page folds
ligands by CCD code"*. Newest material appended, as everywhere in `docs/`.

## The integration contract is one object, which is why this was tractable

A ligand reaches AlphaFold 3's featuriser as `parseCcdComponent`'s return
value and as nothing else:

```js
{ code, atoms: [{name, element, charge, x, y, z, leaving}],
         bonds: [{from, to, order}] }
```

The atomised tokens, `ref_pos`, the bond matrix, the second bond-order plane
boltz2 reads, the chirality centres and `bond-geometry.js`'s scoring all come
off that. So SMILES support is not a second featuriser, not a second code path
through the model, and not a dialect: it is **a second producer of that
object**, and nothing downstream can tell which one it got. `smilesComponent`
in `src/chem/component.js` is that producer, and the whole of `src/chem/` exists
to fill it in.

🔴 **AND ONLY AS MUCH CHEMISTRY AS THE MODELS READ.** No canonical SMILES, no
fingerprints, no descriptors, no substructure search, no conformer ensembles,
no force field. What a model reads is the element, the formal charge, the bonds
with their orders, one reference conformer and the right enantiomer.

## What was measured against what

Three gates, and they answer different questions - which matters, because two
of them cannot see the other's failure.

| gate | question | today |
|---|---|---|
| `check-smiles-vs-rdkit.mjs` | is the GRAPH right? | 74/74 |
| `check-stereo-vs-rdkit.mjs` | does `@` mean what RDKit means? | 43/43, 2/2 |
| `check-smiles-rewrites.mjs` | is it the same molecule however written? | 1148, 1622 |
| `check-smiles-batch.mjs` | does the MODEL see the same thing? | 20/20 |
| `test/job-json.test.js` | does a SAVED JOB describe the same fold? | 5 tests |
| `check-smiles-conformer.mjs` | is the CONFORMER a molecule? | see below |
| `check-smiles-path.mjs` | does it FOLD? | 3/3, both routes agree |
| `check-smiles-conformer-gpu.js` | does the device agree with the host? | 7/7 |

```
51 conformers built
  bond lengths   mean 0.022 A from RDKit   (mean bar 0.028)
  bond angles    mean 2.61 deg             (mean bar 3.3)
  aromatic rings worst 0.003 A out of plane
  local shape    mean 0.155 A rms to 4 bonds
  chiral centres 49/49 built with the right hand
```

over **74 molecules**, and the graph agrees 74/74.

Those read 0.055 A and 11.87 degrees when the conformer first ran end to end.
Six later fixes closed it, and each has its own note below: a collapsed
embedding axis, a flattened sulfoxide, constraint projection instead of
gradient descent alone, an adaptive attempt count, a VSEPR angle lean, and
rings solved as cyclic polygons rather than regular ones.

🔴 **THE BAR IS docs/AF3.md's OWN FLOOR, NOT MMFF.** `ref_pos` is a reference
conformer in each token's frame, and this repository already folds with
idealised amino-acid conformers that differ from the reference's CCD geometry
by **0.65 A rms** on intra-token distances - which `npm run test:batch`
REPORTS as a floor rather than failing on. At 0.160 A this is four times inside
that. Chasing MMFF molecule by molecule means shipping a force field.

## @rdkit/rdkit was evaluated and declined, and it is not about size

The obvious question is why any of this exists when RDKit has an official
WebAssembly build. Measured rather than argued:

| | raw | gzipped | 3D conformers? |
|---|---:|---:|---|
| `src/chem/`, all of it | 151 KB | **50 KB** | yes |
| ...without the GPU kernel | 129 KB | 43 KB | yes |
| ...code only, comments stripped | 60 KB | **18 KB** | yes |
| `@rdkit/rdkit` 2026.3.6 MinimalLib | 7.3 MB | **2.39 MB** | **no** |

🔴 **RE-MEASURED 2026-09-17 AND THE FIRST FIGURE HERE WAS STALE.** It read 34 KB
gzipped, taken before the device kernel existed and before three rounds of
fixes; it is 50. The ratio against RDKit went from 70x to 48x and the
conclusion does not move an inch, which is the point of writing the number
down rather than the ratio. Note also how much of this is prose: stripped of
comments the whole thing is 18 KB gzipped.

🔴 **MINIMALLIB IS 2D ONLY, AND THAT IS THE DECIDING FACT RATHER THAN THE
SEVENTYFOLD SIZE.** Run here on biotin: every z is `0.0000`, the molblock
header reads `RDKit          2D`, and the only coordinate entry points are
`set_new_coords` / `generate_aligned_coords`, which are CoordGen - a DEPICTION
library. There is no `EmbedMolecule`, no ETKDG, no distance geometry. So it
cannot produce `ref_pos`, which is the one thing the models need from a ligand
and the hard half of this work; it would replace the PARSER, which is already
51/51 against real RDKit, at seventy times the bytes.

A custom RDKit WASM build would go the wrong way: adding DistGeom for 3D makes
it bigger, not smaller. OpenChemLib (~400 KB gzipped) does have a conformer
generator and is the one real alternative; it is still twelve times the size
and would need every gate above pointed at it before its output could be
trusted, which is most of the work either way.

**RDKit itself is used, as an ORACLE.** `/home/ubuntu/.venv-rdkit` holds the
real Python package and `tools/oracle/dump_rdkit_smiles.py` writes the
reference that three of the gates above compare against. Nothing under `src/`
imports anything derived from it.

## The defects, each of which produced a plausible molecule

Every one of these passed every test that existed when it was written, because
the output was a chemically reasonable structure that was not the right one.

🔴 **KEKULISATION MUST RUN BEFORE THE IMPLICIT HYDROGENS.** Counted off
aromatic bonds worth 1.5, caffeine's three N-methyl nitrogens each sum to
1.5 + 1.5 + 1 = 4, past nitrogen's valence of 3, so the rule promotes them to 5
and hands each a hydrogen it does not have: **C8H13N4O2 against RDKit's
C8H10N4O2**, and ATP one too many. Kekulised, the same nitrogen has two single
ring bonds, sums to 3, and takes none.

🔴 **AN AROMATIC BOND'S ORDER IS NOT A FACT ABOUT THE MOLECULE.** Benzene has
two Kekulé structures and they are the same substance; which one a program
picks falls out of the order it searched in. Asserted bond by bond against
RDKit this failed **six of fifty-one** - benzene, ibuprofen, paracetamol, ATP,
porphine, NAD - and every difference was that arbitrary choice, with formula,
hydrogens and rings all agreeing. The gate asserts the structure chosen is a
LEGAL one instead (`valenceProblems`), which a wrong kekulisation breaks and a
differently-chosen one does not. Forcing every aromatic bond single: 31/51.

🔴 **A RING CLOSURE IS WRITTEN AT THE DIGIT AND CREATED WHEN THE RING CLOSES.**
`[C@@H]1SC...1` writes its ring neighbour SECOND and creates that bond LAST, so
ordering a chiral centre's neighbours by bond index puts it at the end and
inverts the centre. **9 of 29 corpus centres were the wrong enantiomer** -
biotin's two, ATP's ribose, glucose, three of cholesterol's, penicillin's,
NAD's - every one a ring opening, with every bond length and angle perfect.

🔴 **THE 1-4 TORSION PLACED BOTH OUTER ATOMS ON THE SAME SIDE.** `distanceAt`
used `PI - angle` for both, mirroring them through the axis: an sp3 1-4 pair
came out **0.51 A apart where it should be 2.53**, so every torsion's lower
bound was about a bond length. Alanine got `0C-4O: [1.19, 2.58]` - a lower
bound below any real bond - and the bounds as a SET had no three-dimensional
solution while no single PAIR was inconsistent, which the contradiction check
cannot see. It looked like an optimiser giving up. Fixing it took the corpus
mean angle 8.11 -> 4.40 degrees.

🔴 **THE VAN DER WAALS FLOOR CONTRADICTED EVERY BOND.** Two bonded carbons are
1.52 A apart and 0.8 times their vdW sum is 2.72, so applied to all pairs the
lower bound lands above the upper: glycerol 10 contradictions, ATP 86. A vdW
radius describes two atoms NOT bonded to each other.

🔴 **SIGMA COUNT DOES NOT NAME A SHAPE.** Ammonia and formaldehyde's carbon both
have three sigma bonds and are 107 and 120 degrees; a phosphate's phosphorus
has four and is tetrahedral, not the 107 a three-coordinate one wants. The
first version had both cases backwards. A sulfoxide keeps a lone pair and is
pyramidal at ~106 rather than trigonal - dimethyl sulfoxide was the worst angle
in the corpus at 16.4 degrees.

🔴 **A PAIR CAN HAVE TWO ANGLE ESTIMATES AND INTERSECTING THEM IS WRONG.**
Penicillin's beta-lactam diagonal is reachable through a carbon (two C-C bonds,
2.15 A) and through a nitrogen (C-N and an amide N-C, 2.01 A). Those are two
estimates of ONE distance and the bound must SPAN them; intersecting them, which
is what setting each bound as it is computed does, produced a lower above an
upper and a molecule with no valid geometry.

🔴 **DISTANCE BOUNDS CANNOT SAY "FLAT".** Benzene's bounds pin every bond at
1.43, every 1-3 at 2.48 and every para pair at 2.76-2.96 - and **a ring puckered
0.44 A satisfies all fifteen**, which is what came back, at a bounds error of
2.6e-9. The embedder was not failing to solve the problem; the problem did not
say what was wanted. Planarity is stated directly now, as a signed volume, the
way chirality is.

🔴 **AND THE PLANARITY WEIGHT WAS SWEPT, WHICH INVERTED THE OBVIOUS GUESS.**
0.5 beats 10 on planarity AND on bonds AND on angles - at 10 the line search's
step collapses before it converges, so pushing harder on flatness produces a
LESS flat ring. Swept: 0.2 / 0.5 / 1 / 2 / 5 / 10 gives worst-out-of-plane
0.051 / **0.009** / 0.026 / 0.018 / 0.010 / 0.010 and mean angle 7.85 / **7.92**
/ 8.53 / 9.70 / 11.12 / 11.87.

🔴 **A STALLED LINE SEARCH IS NOT A FINISHED ONE.** Alanine stopped after 44
steps at an error of 0.264 with EVERY bound violated and its bonds 8% short,
which reads exactly like a bad embedding rather than an optimiser giving up.
Restarting the step size four times is what finishes it.

🔴 **THE POLARITY CORRECTION WAS MEASURED TWICE WITH OPPOSITE ANSWERS, AND BOTH
MEASUREMENTS WERE RIGHT.** Schomaker-Stevenson improved the mean bond error
0.0402 -> 0.0362 A and was kept; after `delocalizeCharges` landed it makes
things worse at every value and is now zero. The sulfonate's S-O was 0.24 A too
long because its resonance was not modelled, and a term that shortens polar
bonds was absorbing that error on behalf of a bug somewhere else. With the
resonance handled it is pure harm - Cordero's radii are fitted TO crystal
structures, polar bonds included, so subtracting polarity again double-counts:
C-F read 1.20 against a true 1.35. **A knob that is paying for another
component's bug measures as valuable until that bug is fixed.**

## Four more, found by asking where the error still was

🔴 **AN ENTIRE EMBEDDING AXIS COLLAPSED AND THE MOLECULE CAME OUT FLAT.** Power
iteration converges to the largest eigenvalue **by magnitude**, and a metric
matrix built from distances drawn at random inside their bounds is not positive
semi-definite - those distances need not be realisable in three dimensions, or
in any number at all. So the iteration happily returned a large NEGATIVE
eigenvalue, `sqrt(max(value, 0))` made its scale zero, and that coordinate was
identically zero for every atom. Measured: **ATP's y spread was 0.000 and
biotin's z was 0.000.**

The tell was a contradiction that cannot be true. Asking whether a MIRRORED
copy had the right chirality, ATP scored **0 correct as-is and 0 correct
mirrored** - impossible, because mirroring flips every signed volume, so the
two must sum to the total. They summed to zero because every volume WAS zero:
the molecule was planar and no centre had a hand at all. Adding `shift * I`
before the iteration moves every eigenvalue up without moving a single
eigenVECTOR, and Gershgorin's bound is the cheapest shift certainly large
enough. ATP's best of 24 starts went **0.743 -> 0.0075**.

🔴 **A SULFOXIDE WAS BEING FORCED PLANAR, WHICH IS `idealAngle`'s MISTAKE IN A
SECOND PLACE.** The sp2 planarity rule fired on any three-coordinate atom with
a pi bond, so dimethyl sulfoxide's sulfur was flattened to 120 degrees against
RDKit's 95.8 and 107.5 - the worst angle in the corpus at 16.4 - and the angle
table's own 106 could not win against a planarity term pulling the other way.
Sulfur and phosphorus keep a stereochemically active lone pair, which is why a
sulfoxide can be a stereocentre at all. **The same wrong rule in two places,
fixed once and then again.**

🔴 **STEEPEST DESCENT IS THE WRONG SOLVER FOR A DISTANCE CONSTRAINT.** Over 24
random starts the gradient pass alone reached a satisfied set of bounds for
glycerol 23 times, for a CF3 group **3 times**, and for ATP **not once**. A CF3
carbon came out at F-C-F 161 degrees and F-C-C 180 - a planar carbon - with the
bounds pinning it at 109.47 perfectly correct and simply not met. A violated
distance has an exact local repair, and applying it pair by pair is
Gauss-Seidel on the constraint set: the same idea as SHAKE in molecular
dynamics, with no step size to choose and no line search to stall. It runs
first; the gradient pass polishes the volume terms, which are not pairwise and
which projection cannot touch. CF3 went 3 of 24 to 11 of 24, and with the
embedding fix beside it to 3.0 degrees from RDKit against 16.3.

🔴 **AND A FIXED ATTEMPT COUNT IS EITHER WASTEFUL OR WRONG AND CANNOT BE BOTH
RIGHT.** At four attempts ATP returned an error of 0.314 with a purine ring
bent to 176 degrees where it should be 118; six attempts reach 0.013, and the
extra two cost milliseconds. Glycerol satisfies its bounds on the first start.
So the loop stops when the bounds are met rather than after a number chosen for
the average molecule. Cost today: glycerol 21 ms, aspirin 61, biotin 100,
ATP 193.

## Two more, on the angles themselves

🔴 **ONE ANGLE PER ATOM CANNOT DESCRIBE AN ATOM WITH UNEQUAL SUBSTITUENTS.**
Dimethyl sulfoxide's sulfur is 96.6 degrees between its two methyls and 107.5
to its oxygen - eleven degrees of spread around ONE atom, which a single ideal
angle splits the difference of and gets both ends wrong. Acetate is the same
shape: O-C-O opens to 130 while C-C-O closes to 115. The rule is VSEPR's
ordering, which is older than any force field and is one line - a multiple bond
holds more electron density and pushes its neighbours away - and it is applied
as a shift against the MEAN substituent weight, so a centre whose substituents
are alike gets exactly the base angle and the rule vanishes.

Swept against RDKit, mean angle error in degrees:

| k | 0 | 6 | 8 | 9 | 10 | 11 | 12 | 13 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| err | 3.28 | 2.88 | 2.80 | 2.78 | 2.77 | **2.76** | 2.77 | 2.83 | 2.90 | 3.15 |

Flat from 8 to 12, so 11 is the middle of it - a shape worth recording, because
it says the rule does real work and that the constant is not load bearing. Bond
lengths are 0.019 A throughout, which is the control: an angle rule that moved
them would be reaching somewhere it should not. DMSO's C-S-C went 107 to 100.

🔴 **AND A REGULAR POLYGON IS THE WRONG MODEL FOR A HETEROAROMATIC RING.**
Thiophene's S-C bonds are 1.71 A and its C-C bonds 1.37, so its interior angles
are nothing like a pentagon's 108 - the real C-S-C is about 92 and the carbons
open past 111 to compensate. Using 108 for all five was the worst angle left in
the corpus, and it is wrong in the same direction for furan, pyrrole,
imidazole and most five-rings a ligand has. A planar ring with given sides is a
CYCLIC POLYGON and the circle is the only unknown: each side subtends
`2*asin(b/2R)` and those must sum to a turn, so one bisection solves it. Furan,
pyrrole and imidazole are now 2.0, 1.0 and 2.5 degrees from RDKit.

Thiophene itself is still 6.3, and that is the approximation showing rather
than a bug: a planar pentagon with fixed sides has two degrees of freedom and
inscribed-in-a-circle is one particular choice of them.

## Doubling the corpus, which found three more

🔴 **A CORPUS THAT AGREES EVERYWHERE HAS STOPPED BEING EVIDENCE.** Fifty-one
molecules all passing proves the cases somebody thought of are right. Twenty-
three more were chosen by reading the parser for branches nothing exercised and
the geometry for shapes nothing built - isotopes, atom maps, a quaternary
ammonium, an azide, an allene, a nitrile, boron, a tetrazole, an epoxide, a
coronene fragment, two macrolides, and a 62-atom taxol core against a previous
largest of 31. Three of them were bugs.

🔴 **AN EXPLICIT `[2H]` WAS BEING KEPT AS A HEAVY ATOM.** `[2H]C([2H])([2H])O`
is methanol with three deuteriums - two heavy atoms - and this port counted
FIVE, because the parser keeps every bracket atom and `[2H]` is written like
one. That reaches the featuriser as a ligand with three extra tokens, since AF3
tokenises one token per heavy atom. Folded into the host's count now, before
kekulisation, because it changes what valence every neighbour has left - and
everything that indexes an atom is renumbered with it, which is the half that
is easy to forget. `[H][H]` and `[H+]` stay as atoms: a hydrogen with nowhere
to go is not a substituent.

🔴 **AND THE SYMMETRIC-RING RULE WAS KEEPING RINGS THAT ARE NOT RINGS.** Taxol's
pool contains an eight-membered candidate that is exactly its six-ring PLUS its
four-ring - a lap around two fused faces - and "keep a dependent ring of the
same size as one already chosen" kept it, giving 8 rings where RDKit gives 7.
The test that separates it from adamantane's genuinely tied fourth six-ring is
whether the reduction needs anything SMALLER: adamantane's reduces using only
six-rings, taxol's collapses the moment its four-ring is allowed in. So each
candidate is reduced twice, once against smaller rings alone.

🔴 **AND THE CANDIDATE POOL WAS TOO SMALL FOR A CAGE.** Every bond of a cubane
lies on two of its faces and a breadth-first search returns whichever it
reaches first, so twelve bonds yielded four distinct faces where the circuit
rank is five. Horton's generation - for every atom and every bond, the two
shortest paths plus the bond - contains every ring that can be in the answer.
It is O(V*E) candidates against O(E) and costs nothing at ligand size.

## And two the bigger corpus exposed in the geometry

🔴 **DISTANCE BOUNDS CANNOT SAY "STRAIGHT", FOR A SHARPER REASON THAN THEY
CANNOT SAY "FLAT".** A 1-3 distance is `sqrt(a^2 + b^2 - 2ab cos(theta))` and
its sensitivity to theta goes as sin(theta), which is ZERO at 180 degrees. So
near a linear centre the distance carries almost no information about the
angle: a nitrile with a bound of +/-0.02 A still admits 166 degrees, and the
two bonds' own +/-0.01 A are alone enough to force a bend.

Three repairs from the distance side were tried and measured, and all three
failed: an angular tolerance converted through the law of cosines (nitrile 24.5
-> 14.3 degrees), inheriting the bonds' slack (**worse**, 14.3 -> 20.2, because
it loosens every other angle to fix one), and tracking the bond sum exactly
(20.0). Stating collinearity directly - the cross product of the two bond
vectors is zero - fixed it at once, and took the corpus mean from 3.54 to 2.61.
The angular tolerance was kept regardless, because it is what the slack always
meant.

🔴 **AND A CHARGE MOVES A RADIUS BY A WHOLE ELEMENT.** A quaternary ammonium's
N-C is 1.522 A where a neutral amine's is 1.430 - the largest single bond error
left - and the reason is not a correction factor: **N+ is isoelectronic with
carbon**, so it has carbon's radius, and 0.76 + 0.76 is 1.52 on the nose. The
same rule sends O- to fluorine's radius. Measured over the corpus: C-N neutral
1.430 (34 bonds), C-N+ with three bonds 1.477 (2), with four 1.522 (4).

## Fuzzing, which is where hand-picking runs out

🔴 **A HAND-PICKED CORPUS PROVES THE CASES SOMEBODY THOUGHT OF ARE RIGHT, AND
THAT METHOD HAS A FLOOR.** The second round of picking found three bugs in
twenty-three molecules and then ran out of conventions to name. What replaces
it needs no cases at all: a SMILES is one of enormous numbers of strings for
the same molecule - different starting atom, different branch nesting,
different ring-closure digits - and RDKit will emit them on demand. Every one
must parse to the SAME graph with the SAME handedness at every centre. The
invariant is the equality itself, so there are no reference values to be wrong
about, and the coverage is bounded only by how many strings are generated.

It aims squarely at the code that has actually been wrong here: re-writing a
molecule permutes exactly the ring-closure bookkeeping that once inverted nine
of twenty-nine stereocentres.

**1148 graph re-writings and 1622 centre checks, all agreeing.** Nothing new
was found, which is the useful part - that is the evidence the parser and the
stereo perception are done, in a way another twenty hand-picked molecules
could not have provided.

🔴 **AND THE FIRST VERSION OF THIS GATE COULD NOT FAIL, WHICH IS THE SAME TRAP
ARRIVED AT BY A NEW ROUTE.** It compared each conformer against the sign
`chiralCentres` had ASKED for - one source, two uses, agreement by
construction. Putting the original ring-closure bug back left it reporting
1622/1622. Taking the signed volume over the four neighbours in CANONICAL
order, sorted by their original atom number, makes it a property of the built
molecule alone; the same edit then produces 28 distinct failures. This is the
third time in this file that a chirality check has had to be rescued from
comparing something with itself.

## Bond lengths that are measured rather than predicted

Covalent radii are fitted to ordinary two-centre bonds and miss two classes
badly, both of them common. Stated as numbers with a provenance, the way
`bond-geometry.js` derives its ideals from `reference-conformers.json`:

| kind | radii predict | measured | why radii cannot |
|---|---:|---:|---|
| P-O single | 1.730 | 1.593 | hypervalent: partial double and ionic at once |
| S-O delocalised | 1.610 | 1.470 | the same |
| aromatic C-C | 1.430 | 1.398 | delocalisation is not the mean of two orders |
| aromatic C-N | 1.383 | 1.354 | the same |

The aromatic row matters most: averaging the single and double radii puts
benzene 2.5% too big, in the substructure that appears in most of the corpus.

## The device path, and where it actually pays

🔴 **THE MEASUREMENT MOVED THE TARGET.** Triangle smoothing is O(N^3) and looks
like the expensive step; it is **6%**. At 100 atoms the host spends 27.6 ms
refining, 7.9 embedding, 3.4 building bounds and 2.6 smoothing them.

🔴 **AND THE NATURAL PORT OF THE REFINEMENT IS SLOWER THAN THE HOST.** 400
steepest-descent steps, each with up to 20 line-search evaluations, is up to
8,000 dispatches for one ligand - at ~0.1 ms of launch overhead the overhead
alone is 800 ms against the host's 27.6. So the whole solve lives inside ONE
dispatch: a workgroup holds one conformer in workgroup memory and runs every
step and every reduction against `workgroupBarrier`, and the batch axis is
workgroups.

🔴 **AND WGSL WILL NOT LET IT STOP EARLY, WHICH IS WHY IT LOSES ON ONE LIGAND.**
A barrier may not sit inside control flow that depends on a workgroup
reduction, and the uniformity analysis does not consider a value read back out
of workgroup memory to be uniform even when it provably is. Dawn refuses the
module outright. So the step count is fixed, a `finished` flag makes later
iterations no-ops, and every acceptance is a `select` rather than a branch -
which means the device runs the full budget where the host breaks out. That is
strictly more arithmetic, not overhead to be tuned away.

Aspirin, one dispatch per batch, on the A100:

| conformers | host ms | device ms | speedup |
|---:|---:|---:|---:|
| 1 | 9.1 | 39.7 | 0.23 |
| 4 | 12.5 | 38.7 | 0.32 |
| 16 | 43.4 | 38.7 | **1.12** |
| 64 | 168.9 | 39.2 | 4.31 |
| 256 | 670.1 | 41.1 | **16.3** |

**The device's cost is flat from 1 to 256** and the host's is linear, so the
crossover is 16 conformers.

🔴 **AND THE PROJECTION SWEEP IS JACOBI ON THE DEVICE WHERE IT IS GAUSS-SEIDEL
ON THE HOST.** The host repairs one violated pair at a time and lets the next
pair see the result, which is inherently serial. On the device every lane owns
a row of atoms, sums the corrections that row's pairs want and applies the
AVERAGE - so no two lanes write the same atom and no atomic is needed, which
WGSL has no float version of anyway. Averaging converges more slowly per sweep,
so it gets more sweeps; both end at a satisfied set of bounds.

🔴 **AND THE DIFFERENTIAL TOOK TWO REFORMULATIONS THAT THE MEASUREMENTS
FORCED.** Comparing each side's BEST attempt compared two selections as much as
two solvers - the host stops at the first start that satisfies the bounds and
the device runs them all, so the two routinely returned different attempts.
Handing both the SAME start fixed that and exposed the real obstacle: this
objective is not convex, and from one start the two arithmetics reach DIFFERENT
local minima. ATP's two were **74 degrees apart with near-identical errors,
0.297 against 0.303** - a chaotic optimisation amplifying the last bit of an
f32 sum through four hundred descent steps, not a kernel defect.

So coordinates cannot be compared at all on a hard molecule, and where they CAN
be the agreement is exact: **glycerol and benzene, whose bounds admit
essentially one answer, come out bond 0.0000 and angle 0.00 apart.** Those two
are the arithmetic check. For the rest the assertion is quality and
correctness - the device must solve the bounds about as well as the host over
the same starts, and every chiral centre must have the hand the SMILES asked
for. On the corpus **the device is usually the better of the two**: caffeine
0.0043 against 0.0082, ATP 0.060 against 0.187, cholesterol 0.014 against
0.045, because it never stops early. A page folding one ligand should stay on the host,
which is why `smilesComponent` defaults to it and takes the device as a
parameter; a screen should not.

## What is refused rather than guessed

Each names itself, because a ligand read wrongly folds and scores and the
number is merely different:

- `*` (any atom) - no element, so no conformer and no mass
- the extended stereo classes `@TH` `@AL` `@SP` `@TB` `@OH`
- reaction SMILES (`>`), an unkekulisable ring, a duplicated bond
- every syntax error, with its position in the string
- a ligand with both `smiles` and `ccdCodes`, which names itself twice
- past `MAX_SMILES_ATOMS` (150) on the page, because smoothing is cubic and the
  tab is single-threaded

🔴 **AND `smiles` IS ITS OWN ENTITY TYPE, NOT A FLAG ON `ligand`.** A CCD code
is upper-cased and upper-casing a SMILES changes the molecule: `c1ccccc1` is
benzene and `C1CCCCC1` is cyclohexane. Nor can the two be told apart by
inspection - `C` is a valid SMILES and `CCO` looks like a three-letter code.
The page already learned this with the template database menu and stopped
guessing there for the same reason.

## The sharpest gate: the batch itself

🔴 **A FOLD GATE COMPARES STRUCTURES, AND A STRUCTURE IS NOISY, SAMPLED AND
MODEL-DEPENDENT.** `check-smiles-path.mjs` is a plumbing test with a geometry
floor and says so - it cannot tell a 15% rescaled conformer from a correct one.
The BATCH is what the model actually reads: token types, element numbers,
formal charges, reference positions, the bond matrix, the bond-order plane, the
atom windows, the chirality centres. If every field is identical then the model
cannot distinguish the two routes even in principle, and no sampling noise is
in the way.

🔴 **AND IT IS ELEMENTWISE, WHICH TOOK ONE TRICK.** `OCC(O)CO` is the obvious
glycerol and puts oxygen first; the CCD's GOL is C1 O1 C2 O2 C3 O3. Written as
`C(O)C(O)CO` it is the same molecule in the dictionary's own atom order, and
the permutation disappears - so the comparison can be value against value
rather than set against set, which is enormously stronger. Five ligands, four
models, **43 fields, 0 differ, 20 of 20**.

Sensitivity, measured by breaking it: a single formal charge flipped on one
atom is caught as `refCharge: 1 of 1776 differ`. Renaming every atom is caught
in two fields at once.

Three differences are REPORTED rather than asserted, each for a stated reason:

- **`refPos`**, which is the conformer. The dictionary ships an
  experimentally-derived ideal and this port builds one by distance geometry;
  two conformers of one molecule are both correct. docs/AF3.md already treats
  this exact field as a reported floor at 0.65 A rms between this port's
  idealised amino acids and the reference's CCD geometry.
- **`refAtomNameChars`**, for the two entries whose dictionary names are not
  element-plus-counter - ACE calls its methyl `CH3`, BEN calls its seventh
  carbon plain `C`. An atom name IS a model input, and for a ligand with no CCD
  entry - the whole point of this path - there is no dictionary name to match.
  Folding a known ligand by code rather than by structure gives the model the
  dictionary's names and by structure gives it these: a real difference, small,
  and not a bug in either direction.
- **`ligandSpans[n].bonds` ORDER**, which reaches nothing. 🔴 And the excuse is
  not "this field does not matter" - it is that `bondMatrix` and
  `bondOrderMatrix` are DERIVED from this list, are what the model reads, are
  compared elementwise, and come out identical. The gate refuses to grant the
  excuse if either of those two is missing from the batch.

🔴 **AND BEN's KEKULÉ CHOICE HAPPENED TO MATCH THE DICTIONARY'S.**
`bondOrderMatrix` is identical for a benzamidine written aromatic against one
written with explicit alternating bonds. That is worth knowing and is not worth
relying on: benzene has two Kekulé structures, this port picks one by search
order and the dictionary picked one by whoever deposited it, and a future
aromatic entry may disagree. It would show here as a `bondOrderMatrix`
difference, and it would be the resonance-form arbitrariness documented above
rather than a defect.

## 🔴 The archive recorded benzene as cyclohexane

The most serious thing found in this work, and it was found by asking what a
SAVED JOB describes rather than whether a fold succeeds.

`jobRequestJson`'s last branch was a catch-all `else` that turned every
non-polymer row into `{ligand: {ligand: value.toUpperCase()}}`. A benzene
folded as `c1ccccc1` was written into the archive's `job_request.json` as
**`C1CCCCC1` - cyclohexane** - labelled as a dictionary code.

🔴 **AND THE LOUD HALF IS THE SAFE HALF.** A long SMILES throws on read-back
("A CCD code is 1-5 letters or digits"), which is survivable. A SHORT one does
not: `C` is a valid SMILES for methane AND a valid CCD code for cytidine
monophosphate, so that job round-tripped **silently into a nucleotide**. The
archive is the file a reader hands back to reproduce a fold, and job-json.js's
own header says twice over that a request describing a different job is the
failure it exists to prevent.

The server dialect cannot express a SMILES at all - its ligand entry takes
`ligand`, `ion` and `count`, with no field for a structure - so a job carrying
one is written in the OPEN dialect, and only such a job. Three things that
dialect needs which the server's does not, each wrong in the first attempt:
copies are an `id` LIST whose length is the count and whose labels must be
unique; seeds are integers rather than strings; and 🔴 the `dialect` key must
be written EXPLICITLY, though this file's own header says the open dialect has
none - upstream's rule is both `dialect` and `version` or NEITHER, and neither
means the SERVER dialect at version 1, so a file with `version: 1` and no
dialect is a malformed server file that round-trips straight into a refusal.

Verified through the page with `--job-round-trip`, which WIPES the entity rows
before dropping the archive back: the SMILES returns byte for byte.

## Scale, which nothing had tested

The paired cases in `check-smiles-path.mjs` are all small, because a dictionary
twin has to be hand-written in the dictionary's atom order and that does not
scale. The page admits 150 heavy atoms and the largest paired case is 16, so
nothing was folding a ligand of the size people actually dock.

| solo ligand | atoms | bonds | pLDDT | bond rms |
|---|---:|---:|---:|---:|
| paclitaxel core | 62 | 68 | 84.8 | **0.052 A** |
| erythromycin fragment | 28 | 28 | 86.3 | 0.050 A |

Both better than the glycerol the CCD path folds at 0.057, and well inside the
0.20 A the ligand gate allows. They are not differentials - there is no
dictionary entry to compare against, which is the whole point of the SMILES
path - so what they assert is that a large ligand survives intact: every atom
present, every name unique, every bond the length its component asked for.

## 🔴 Two different SMILES in one job were the same molecule to the model

Found by asking what a job with MORE THAN ONE structural ligand does, which
nothing had. It was two bugs wearing one cause: every SMILES ligand was named
`LIG`.

**The visible half.** The output PDB wrote a benzene and a glycerol under one
residue name - a file a reader cannot tell apart, and which tooling that
filters by residue name silently mixes. `check-ligand-path.mjs` does exactly
that filtering.

🔴 **AND THE INVISIBLE HALF, WHICH IS THE SERIOUS ONE.** `featuriseProtein`
keys a ligand's ENTITY on its code - "identical codes are one entity, and each
occurrence is a copy of it" - so benzene and glycerol came out sharing an
`entity_id`, with the model told that six carbons and a glycerol are two copies
of one thing. That reaches the model as a token feature and reaches
chain-permutation scoring as a claim that the two are interchangeable.

The rule was right and its key was wrong. A CCD code identifies its contents,
so for every fold this port had ever done the code and the molecule agreed;
a SMILES ligand has no code and is GIVEN one. The featuriser keys on what the
component IS now - element, charge and bond list - which cannot change a
dictionary fold, and `npm run test:batch` and `test:ligand` confirm it does
not.

Both halves are fixed, deliberately: distinct structures get distinct names AND
the featuriser stops trusting the name. Either alone leaves the other failure
reachable from a different caller - which is not hypothetical, because there
were two callers and only one of them named anything.

🔴 **AND THE CLI HAD THE SAME BUG, WHICH IS WHY THE RULE NOW HAS ONE HOME.**
`tools/gpu/fold.js` took a single `--smiles-code` for every ligand, so
`--smiles='c1ccccc1|OCCO'` wrote ten atoms into ONE residue with `C1` and `C2`
appearing twice - a file no reader and no bond checker can make sense of.
`ligandName` and `nameSmilesLigands` live in `src/chem/component.js` and both
callers use them.

Names are three characters because `src/af3/fold.js` writes the residue name
with `.padEnd(3)` into a fixed-width column: `LIG`, `LG2` to `LG9`, then `L10`
upwards, and `LIG2` truncated back to `LIG` would have put the collision
straight back. Identical strings still share a name, because they are the same
molecule and genuinely one entity - verified in the fold, where two benzenes
and one ethylene glycol come back as `LIG`, `LG2`, `LIG` on three chains.

## The session, checked and clean

A restored session rebuilds the archive from `lastPrediction.entities`, so a
ligand that did not survive would come back as an archive describing a fold
WITHOUT it - and a SMILES is the one ligand whose value cannot be recovered
from a code. Measured through `fold-in-page.py --session`: saved, offered,
restored, and the rebuilt archive carries `requestLigands: ['c1ccccc1']`, lower
case and intact, at 74 tokens over two chains.

🔴 **AND `offered: false` IN AN EARLIER RUN WAS A MISREAD OF THE HARNESS, NOT A
BUG.** The session is saved when the reader LEAVES rather than when the fold
ends, and the probe that reports the offer is a different one from the probe
that had been grepped. A plain protein with no ligand showed the same thing,
which is what said the reading was wrong rather than the ligand path.

## 🔴 The control a reader actually touches, which nothing had driven

Every page check up to here set the row through `entityList.set()`. That is the
API, not the UI, and it cannot see a UI bug - which is how three of them
survived every `--smiles` run.

🔴 **THE BLUR HANDLER TURNED BENZENE INTO HEXANE.** A row that is not a
`ligand` fell to `entity.value = cleanSequence(value.value)`, which keeps only
amino-acid letters. Type `c1ccccc1`, click away, and the box says **`CCCCCC`** -
hexane. Biotin came back as
`OC(=O)CCCC[C@@H]SC[C@@H]NC(=O)N[C@H]` with every ring-closure digit stripped,
which no longer parses at all. Silently, on a click.

🔴 **AND `setChains` DELETED THE ROW.** It rebuilds the entity list keeping
`type === "ligand"`, and runs when an alignment's own query replaces the chain
list - so folding with an A3M dropped the SMILES ligand and folded the protein
alone.

🔴 **AND THE STYLING WOULD HAVE LIED.** `.entity-value-ligand` carries
`text-transform: uppercase`, so reusing the ligand's class would have DISPLAYED
`c1ccccc1` as `C1CCCCC1` - a different molecule on screen from the one folded.
Its own class says `text-transform: none` explicitly rather than by default,
because the neighbouring rule is two lines away and easy to extend by accident.

A SMILES row now gets a one-line input with its own placeholder, and
`fold-in-page.py --smiles-ui` drives the controls rather than the API:

```
offered   protein, dna, rna, ligand, smiles
label     "Ligand (SMILES)"     tag INPUT     transform none
typed     c1ccccc1              stored c1ccccc1      kept true
```

### Switching the type clears the box, but only across a category

The same text is a different molecule on the other side of a type change: a CCD
row holding `C` switched to SMILES is **methane** where it meant cytidine
monophosphate, and a SMILES `CCO` switched to CCD goes looking for a dictionary
entry called CCO. Both parse, both validate, and both are silently not what was
typed - the archive bug above, one control earlier.

🔴 **BUT NOT BETWEEN THE POLYMERS, WHERE CARRYING IT IS THE POINT.**
`entities.js` already records that `ACGT` is a valid protein AND a valid DNA
chain and that only the row's type says which. Correcting a mis-typed row is
the commonest edit on the page, and clearing there would make it destructive.
Verified in the page: `clearedCrossing: true`, `keptWithinPolymers: true`.

🔴 **AND THE PROBE WAS WRONG TWICE BEFORE IT WAS RIGHT**, both times reporting
a bug that was its own. It held an element across a re-render, which a type
change and a blur both cause; and it typed into the HIGHLIGHT LAYER, because a
polymer row carries two elements with class `entity-value` - localfold.css says
so - while a SMILES row has only the input, so exactly the half that was
already correct passed. A probe that disagrees with an isolated test of the
same logic is the probe's fault until shown otherwise.

## What this gate cannot see, stated so it is not trusted past it

`check-smiles-path.mjs` folds the same ligand from its code and from its
string and compares. Broken on purpose to find its reach:

- conformer replaced with noise: **fails all three loudly**, bond rms 0.22
  against 0.03, bonds 0.44 A apart
- every conformer shrunk 15%: **no change at all**, 0.035 A apart against 0.035

So `ref_pos` is a FEATURE and not a template - the diffusion head places the
atoms itself and reads the conformer for what the component IS rather than
where to put it. That gate is a plumbing test with a geometry floor; it proves
the component reaches the featuriser, survives atomisation, comes back with
unique names and holds together under the sampler, and it will not see a
conformer that is subtly rather than grossly wrong.
`check-smiles-conformer.mjs` is what resolves that, at 0.03 A against RDKit.

## Still open

- **Bonded chemistry**, which is `bondedAtomPairs` in the job format and three
  of AlphaFold 3's fourteen example jobs. A SMILES ligand covalently attached
  to a protein needs an inter-entity bond the featuriser does not take.
- **The `userCCD` field**, which is a whole component dictionary inline. The
  reader exists (`parseCcdComponent`); nothing routes it.
- **Larger ligands.** The page caps at 150 heavy atoms and the device kernel at
  256, both for workgroup storage. A polymer-sized ligand would want the
  smoothing tiled, which nothing needs yet.
- **More fuzzing, along axes this one does not reach.** The re-writings vary
  the STRING; nothing yet varies the MOLECULE. Enumerating small graphs and
  checking the same invariants would cover shapes no corpus contains.
- **A per-substituent angle rule.** The worst remaining angles are acetate at
  6.7 degrees and dimethyl sulfoxide's C-S-C at 107 against 96. One angle per
  ATOM cannot express that a lone pair closes more than a double bond does,
  which is VSEPR's ordering; this is the next real accuracy step and it is not
  a force field. Nothing downstream resolves 7 degrees, which is why it is
  here and not done.
- **ATP and cholesterol do not fully converge**, at 2.7e-2 and 1.4e-2 against
  glycerol's 1e-13. Both are geometrically fine - ATP is no longer the furthest
  molecule from RDKit on any axis - so the number says the bounds are slightly
  over-constrained for a large flexible molecule, not that the conformer is
  wrong.
