#!/usr/bin/env python3
"""What representative-atom distance means "in contact", by what the pair IS.

🔴 A RESIDUE'S REPRESENTATIVE IS AN APPROXIMATION AND A LIGAND'S IS NOT. The
distogram - AF3's, AF2's and ESMFold2's alike - predicts a distance between one
representative atom per TOKEN: a residue's pseudo-beta (CB, or CA for glycine),
and for a ligand the heavy atom that token IS. So the conventional 8 A contact
threshold is calibrated for a pair where BOTH ends carry a side chain's worth of
slack, and a ligand token carries none. Applying it unchanged to a
protein-ligand pair asks the wrong question, and to a ligand-ligand pair asks a
question with an exact answer available.

This measures the answer instead of arguing it. Ground truth is the real thing -
the minimum distance between the two tokens' actual heavy atoms, under 5 A - and
the sweep asks which representative-distance threshold reproduces it best, per
pair kind, over real deposited structures.

    python3 tools/calibrate-contact-cutoff.py
    python3 tools/calibrate-contact-cutoff.py --entries 3PTB,1STP --truth 4.5
"""
import argparse, gzip, itertools, math, os, sys, urllib.request

CACHE = "oracle-dumps/structures"
# Protein+ligand, protein+DNA, protein+RNA, and one of each with several kinds
# at once, so no conclusion rests on a single deposition's habits.
ENTRIES = ["3PTB", "1STP", "4HHB", "5P21", "1E66", "1HVR",
           "3HDD", "1LMB", "6MHT", "1A9N", "1B23", "2BTE",
           # ...and two with cofactors packed against each other, because a
           # set where every ligand sits alone reports inter-ligand contact as
           # a dozen pairs and concludes from noise.
           "1PRC", "1OCC"]
WATERS = {"HOH", "DOD", "WAT"}
# The residue alphabets, so a chain's kind comes from its contents rather than
# from a category the file may not carry.
PROTEIN = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER "
              "THR TRP TYR VAL MSE SEC PYL".split())
NUCLEIC = {"A", "C", "G", "U", "DA", "DC", "DG", "DT", "DU", "I", "DI"}
PURINES = {"A", "G", "DA", "DG", "I", "DI"}


def fetch(entry):
    path = os.path.join(CACHE, f"{entry.lower()}.cif")
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{entry.lower()}.cif.gz"
        with urllib.request.urlopen(url, timeout=60) as response:
            body = gzip.decompress(response.read()).decode("utf-8", "replace")
        os.makedirs(CACHE, exist_ok=True)
        open(path, "w").write(body)
    return open(path).read()


def atom_site(text):
    """The _atom_site loop, as dicts. Only the first model, no alternates."""
    lines = text.splitlines()
    for start, line in enumerate(lines):
        if line.startswith("_atom_site."):
            break
    else:
        return []
    columns, at = [], start
    while lines[at].startswith("_atom_site."):
        columns.append(lines[at].strip().split(".")[1])
        at += 1
    index = {name: i for i, name in enumerate(columns)}
    rows = []
    for line in lines[at:]:
        if line.startswith("#") or line.startswith("loop_"):
            break
        parts = line.split()
        if len(parts) != len(columns):
            continue
        rows.append(parts)
    out = []
    for parts in rows:
        get = lambda name, default="": parts[index[name]] if name in index else default
        if get("pdbx_PDB_model_num", "1") != "1":
            continue
        if get("label_alt_id", ".") not in (".", "?", "A"):
            continue
        if get("type_symbol") == "H":
            continue
        comp = get("label_comp_id")
        if comp in WATERS:
            continue
        out.append({
            "group": get("group_PDB"), "comp": comp,
            "chain": get("auth_asym_id") or get("label_asym_id"),
            "seq": get("auth_seq_id") or get("label_seq_id"),
            "name": get("label_atom_id").strip('"'),
            "xyz": (float(get("Cartn_x")), float(get("Cartn_y")), float(get("Cartn_z"))),
        })
    return out


def tokenise(atoms, rule="localfold"):
    """LocalFold's token layout: one per polymer residue, one per ligand atom.

    The representative is the same rule tools/gpu/fold-esmfold2.js uses - CB
    where there is one, else CA, else the token's first atom - which for a
    nucleotide lands on whatever the file writes first and for a ligand token is
    the atom itself.
    """
    residues = {}
    for atom in atoms:
        residues.setdefault((atom["chain"], atom["seq"], atom["comp"]), []).append(atom)
    tokens = []
    for (chain, seq, comp), group in residues.items():
        if comp in PROTEIN or comp in NUCLEIC:
            kind = "protein" if comp in PROTEIN else "nucleic"
            names = {a["name"]: a for a in group}
            if rule == "af3":
                # 🔴 AF3's OWN TABLE, from RESTYPE_PSEUDOBETA_INDEX in
                # protein_data_processing.py: CB for an amino acid and CA for
                # glycine, then C4 for a PURINE and C2 for a pyrimidine. Not
                # C1', which is the obvious guess and is a different atom.
                pick = (names.get("CA") if comp == "GLY"
                        else names.get("CB") if kind == "protein"
                        else names.get("C4") if comp in PURINES
                        else names.get("C2")) or group[0]
            else:
                pick = names.get("CB") or names.get("CA") or group[0]
            tokens.append({"kind": kind, "chain": chain, "seq": seq,
                           "molecule": (chain, seq),
                           "rep": pick["xyz"], "atoms": [a["xyz"] for a in group]})
        else:
            for atom in group:
                tokens.append({"kind": "ligand", "chain": chain, "seq": seq,
                               "molecule": (chain, seq),
                               "rep": atom["xyz"], "atoms": [atom["xyz"]]})
    return tokens


def distance(a, b):
    return math.dist(a, b)


def pair_kind(a, b):
    """🔴 INTRA-LIGAND AND INTER-LIGAND ARE DIFFERENT QUESTIONS, so they are
    different rows. Inside one molecule the representative IS the atom, so the
    "approximation" is exact and the connectivity is known from the component
    dictionary before anything is folded; between two molecules it is a real
    prediction. A single `ligand-ligand` row averages one of each."""
    if a["kind"] == b["kind"] == "ligand":
        return "ligand-intra" if a["molecule"] == b["molecule"] else "ligand-inter"
    return "-".join(sorted((a["kind"], b["kind"])))


def measure(entries, truth, thresholds, rule):
    counts = {}
    for entry in entries:
        try:
            tokens = tokenise(atom_site(fetch(entry)), rule)
        except Exception as error:                       # noqa: BLE001
            print(f"  {entry}: skipped ({error})", file=sys.stderr)
            continue
        print(f"  {entry}: {len(tokens)} tokens "
              f"({sum(t['kind'] == 'ligand' for t in tokens)} ligand)", file=sys.stderr)
        for i, j in itertools.combinations(range(len(tokens)), 2):
            a, b = tokens[i], tokens[j]
            # The same partner rule the certainty uses: a sequence neighbour is
            # the same chain within six residues, and a ligand has none.
            # The certainty's partner rule: a sequence neighbour is the same
            # chain within six residues, and one ligand molecule's atoms all
            # share a residue number - so its self-block goes whole. TWO
            # ligand molecules are kept, which is the pair kind a rule keyed on
            # the CHAIN silently reported as empty.
            same = a["molecule"] == b["molecule"]
            if not same and a["chain"] == b["chain"] and a["kind"] != "ligand" \
                    and b["kind"] != "ligand" \
                    and abs(int(a["seq"]) - int(b["seq"])) <= 6:
                continue
            if same and a["kind"] != "ligand":
                continue
            separation = distance(a["rep"], b["rep"])
            # ...cheap rejection: no atom pair can be closer than the
            # representatives are, minus each token's own radius. 20 A is far
            # beyond any residue's reach.
            if separation > 30:
                continue
            closest = min(distance(p, q) for p in a["atoms"] for q in b["atoms"])
            bucket = counts.setdefault(pair_kind(a, b),
                                       {t: [0, 0, 0] for t in thresholds})
            for t in thresholds:
                cell = bucket[t]
                if separation < t:
                    cell[0] += 1
                if closest < truth:
                    cell[1] += 1
                if separation < t and closest < truth:
                    cell[2] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", default=",".join(ENTRIES))
    parser.add_argument("--representative", default="localfold",
                        choices=["localfold", "af3"],
                        help="CB/CA/first, or AF3's CA and C1' token centres")
    parser.add_argument("--truth", type=float, default=5.0,
                        help="heavy-atom distance that IS a contact")
    args = parser.parse_args()
    thresholds = [4, 5, 6, 7, 8, 9, 10, 11, 12, 14]
    counts = measure(args.entries.split(","), args.truth, thresholds,
                     args.representative)
    print(f"\nground truth: any heavy atom within {args.truth} A\n")
    for kind in sorted(counts):
        rows = counts[kind]
        actual = max(cell[1] for cell in rows.values())
        print(f"{kind}  ({actual} real contacts)")
        print(f"  {'cut':>4} {'called':>8} {'both':>8} {'prec':>7} {'recall':>7} {'F1':>7}")
        best = None
        for t in thresholds:
            called, real, both = rows[t]
            precision = both / called if called else 0.0
            recall = both / real if real else 0.0
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            if best is None or f1 > best[1]:
                best = (t, f1)
            print(f"  {t:>4} {called:>8} {both:>8} {precision:>7.3f} {recall:>7.3f} {f1:>7.3f}")
        print(f"  best F1 at {best[0]} A\n")


if __name__ == "__main__":
    main()
