/**
 * The leaving groups protenix and OpenDDE drop at each DECLARED bond, before tokenising (af3-any-model b61bd1d,
 * structure_cleaning.declared_bond_leaving_atoms, as protenix's remove_leaving_atoms): at each atom a declared bond
 * forms on, the CCD's leaving-flagged heavy atoms bonded to it - each start atom's connected leaving group, in CCD
 * order, one group per bond declared at that atom. AlphaFold 3's own rule is narrower: a glycan's O1 (glycanO1Leaving).
 *
 * What it reaches here: a LIGAND's atoms (a glycan's O1 at each glycosidic C1), and the leaving atom a chain END
 * keeps - a protein's C-terminal OXT when the bond is at its C (a head-to-tail cyclic peptide), a nucleotide's 5' OP3
 * at its P. A modified residue mid-chain has lost its leaving atoms already (polymerResidue).
 *
 * cuda/featurise/featurise.h carries the twin (declaredBondLeaving); test:featurise holds the two to the same bytes.
 */

/** the atoms (indices into component.atoms) of the first `count` leaving groups at `centre`, within one part */
function leavingGroups(component, from, length, centre, count) {
  let at = -1;
  for (let k = from; k < from + length; k += 1) if (component.atoms[k].name === centre) { at = k; break; }
  if (at < 0) return [];
  const neighbours = new Map();
  for (const bond of component.bonds) {
    if (!neighbours.has(bond.from)) neighbours.set(bond.from, []);
    if (!neighbours.has(bond.to)) neighbours.set(bond.to, []);
    neighbours.get(bond.from).push(bond.to);
    neighbours.get(bond.to).push(bond.from);
  }
  const leaving = (k) => component.atoms[k].leaving === true;
  const groups = [];
  for (const start of (neighbours.get(at) ?? []).filter(leaving).sort((a, b) => a - b)) {
    const group = new Set(), todo = [start];
    while (todo.length > 0) {
      const k = todo.pop();
      if (group.has(k)) continue;
      group.add(k);
      for (const m of neighbours.get(k) ?? []) if (leaving(m) && m !== at) todo.push(m);
    }
    const key = [...group].sort((a, b) => a - b).join(",");
    if (!groups.some((g) => g.key === key)) groups.push({ key, atoms: group });
  }
  return groups.slice(0, count).flatMap((g) => [...g.atoms]);
}

/** a ligand without the given atom indices: its bonds renumbered and its residues' spans recounted */
function withoutAtoms(ligand, drop) {
  if (drop.size === 0) return ligand;
  const renumber = new Map();
  const atoms = [];
  ligand.atoms.forEach((atom, k) => { if (!drop.has(k)) { renumber.set(k, atoms.length); atoms.push(atom); } });
  const bonds = ligand.bonds.filter((b) => renumber.has(b.from) && renumber.has(b.to))
    .map((b) => ({ ...b, from: renumber.get(b.from), to: renumber.get(b.to) }));
  const out = { ...ligand, atoms, bonds };
  if (ligand.residues !== undefined) {
    out.residues = ligand.residues.map((part) => {
      let kept = 0, first = -1;
      for (let k = part.from; k < part.from + part.count; k += 1) {
        if (!renumber.has(k)) continue;
        if (first < 0) first = renumber.get(k);
        kept += 1;
      }
      return { ...part, from: first < 0 ? atoms.length : first, count: kept };
    });
  }
  return out;
}

/**
 * @param chains the polymer chains' sequences (featuriseProtein's, empty ones dropped)
 * @param chainKinds each chain's kind
 * @param modifications featuriseProtein's `modifications` ({chain, position, ...component})
 * @param ligands featuriseProtein's `ligands`
 * @param bonds the declared bonds ({from, to}: {asym, residue, atom})
 * @returns {{ligands: object[], terminalDropped: Set<string>}} the ligands without their leaving groups, and the
 *   "chain:index" (index from 0) of each chain-end residue whose terminal atom goes
 */
export function declaredBondLeaving(chains, chainKinds, modifications, ligands, bonds) {
  const count = new Map();
  for (const bond of bonds ?? []) {
    for (const end of [bond.from, bond.to]) {
      if (end.atom === undefined || end.atom === null || end.atom === "") continue;
      const key = `${end.asym}\u0000${end.residue}\u0000${end.atom}`;
      count.set(key, (count.get(key) ?? 0) + 1);
    }
  }
  const dropOf = ligands.map(() => new Set());
  const terminalDropped = new Set();
  for (const [key, n] of count) {
    const [asymText, residueText, atom] = key.split("\u0000");
    const asym = Number(asymText), residue = Number(residueText);
    if (asym >= chains.length) {
      const ligand = ligands[asym - chains.length];
      if (ligand === undefined) continue;
      const parts = ligand.residues ?? [{ from: 0, count: ligand.atoms.length }];
      const part = parts.length === 1 ? parts[0] : parts[residue - 1];
      if (part === undefined) continue;
      for (const k of leavingGroups(ligand, part.from, part.count, atom, n)) dropOf[asym - chains.length].add(k);
      continue;
    }
    const kind = chainKinds[asym] ?? "protein", length = chains[asym]?.length ?? 0;
    const at = residue - 1;
    const end = kind === "protein" ? at === length - 1 : at === 0;
    if (!end) continue;
    const terminalAtom = kind === "protein" ? "OXT" : "OP3";
    const modification = (modifications ?? []).find((m) => m.chain === asym && m.position === residue);
    let drops;
    if (modification !== undefined) {
      drops = leavingGroups(modification, 0, modification.atoms.length, atom, n)
        .some((k) => modification.atoms[k].name === terminalAtom);
    } else {
      // (a standard residue's only heavy leaving atom: OXT on its C, OP3 on its P)
      drops = atom === (kind === "protein" ? "C" : "P");
    }
    if (drops) terminalDropped.add(`${asym}:${at}`);
  }
  return { ligands: ligands.map((ligand, k) => withoutAtoms(ligand, dropOf[k])), terminalDropped };
}

/**
 * AlphaFold 3's own leaving-atom rule for a ligand (structure_cleaning.clean_structure, filter_leaving_atoms with
 * only_glycan_ligands_for_leaving_atoms, atom_layout.get_link_drop_atoms): every GLYCAN residue of a ligand chain
 * loses its O1 unless O1 is itself an end of a declared bond - standalone glycans included, the regime AF3 was trained
 * in (run_alphafold.py's --fix_standalone_glycans defaults off). Every family on AF3's pipeline applies it but the
 * three that keep a ligand whole (boltz2, intellifold2, chai1: `keepLigandLeavingAtoms`).
 *
 * 🔴 THIS FILE USED TO SAY "AlphaFold 3's RULE KEEPS EVERY ATOM OF A LIGAND", AND THAT WAS THE ORACLE'S CCD: the
 * reference featurised with a dictionary whose NAG carried no leaving flag, so its glycan kept O1 and so did this
 * port, exact against it. Given the full CCD AF3 drops all five O1s of the RNase B glycan, 190 tokens to 185.
 *
 * @param chainCount the polymer chains (a bond end's asym past them names a ligand)
 */
export function glycanO1Leaving(chainCount, ligands, bonds) {
  const bonded = new Set();
  for (const bond of bonds ?? []) {
    for (const end of [bond.from, bond.to]) if (end.atom === "O1") bonded.add(`${end.asym}\u0000${end.residue}`);
  }
  return ligands.map((ligand, k) => {
    const asym = chainCount + k;
    const parts = ligand.residues ?? [{ from: 0, count: ligand.atoms.length, glycan: ligand.glycan === true }];
    const drop = new Set();
    parts.forEach((part, p) => {
      if (part.glycan !== true) return;
      // (one component answers to any residue number, as declaredBondLeaving reads it)
      const isBonded = parts.length === 1
        ? [...bonded].some((key) => key.startsWith(`${asym}\u0000`)) : bonded.has(`${asym}\u0000${p + 1}`);
      if (isBonded) return;
      for (let a = part.from; a < part.from + part.count; a += 1) if (ligand.atoms[a].name === "O1") drop.add(a);
    });
    return withoutAtoms(ligand, drop);
  });
}

/**
 * chai-1's leaving groups at a declared bond on a LIGAND (`unresolvedLeavingAtoms`, af3-any-model's
 * _unresolved_leaving_atoms): stock chai keeps them as tokens with reference features - its atom_ref_mask - and does
 * not place them (atom_exists_mask false), so a NAG bonded to an Asn keeps O1 as a token the model never predicts.
 * The groups declaredBondLeaving would drop, as each ligand's atom indices.
 */
export function declaredLigandLeavingAtoms(chainCount, ligands, bonds) {
  const count = new Map();
  for (const bond of bonds ?? []) {
    for (const end of [bond.from, bond.to]) {
      if (end.atom === undefined || end.atom === null || end.atom === "") continue;
      const key = `${end.asym}\u0000${end.residue}\u0000${end.atom}`;
      count.set(key, (count.get(key) ?? 0) + 1);
    }
  }
  const out = ligands.map(() => new Set());
  for (const [key, n] of count) {
    const [asymText, residueText, atom] = key.split("\u0000");
    const k = Number(asymText) - chainCount;
    const ligand = ligands[k];
    if (k < 0 || ligand === undefined) continue;
    const parts = ligand.residues ?? [{ from: 0, count: ligand.atoms.length }];
    const part = parts.length === 1 ? parts[0] : parts[Number(residueText) - 1];
    if (part === undefined) continue;
    for (const a of leavingGroups(ligand, part.from, part.count, atom, n)) out[k].add(a);
  }
  return out;
}
