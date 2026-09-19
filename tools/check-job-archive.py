"""Does a fold archive describe the AlphaFold 3 job that was handed in?

    python3 tools/check-job-archive.py --job=<input>.json --archive=<fold>.zip

🔴 THIS IS DELIBERATELY NOT `fold-in-page.py --job-round-trip`. That one reads
the archive back through `web/job-json.js` - the same module that wrote it - and
compares entity lists. A writer and a reader that share a mistake agree
perfectly: the SMILES bug docs/SMILES.md records was exactly that shape, a
benzene written as a CCD code and read back as one, round-tripping in silence
into cyclohexane. So this process never imports the page's reader. It compares
the archive's `_job_request.json` against the ORIGINAL INPUT FILE, field by
field, in Python.

🔴 AND IT COMPARES ACROSS DIALECTS, WHICH IS THE POINT. AlphaFold 3's own
examples are the open-source dialect - `protein: {id: ["A","B"], sequence}`,
integer seeds, `ccdCodes`, `modificationType`/`basePosition`. This page writes
the SERVER dialect - `proteinChain: {sequence, count}`, string seeds, `ligand`,
`ptmType`/`ptmPosition` - except for a SMILES ligand, which the server dialect
cannot express at all and which goes out in the open one. So "the same job" can
never be a text diff; it is a comparison of MEANING, and the normalisation below
is where that meaning is written down.

🔴 AND THE REST OF THE ARCHIVE IS CHECKED AGAINST THE FOLD, not just the job.
A request that round-trips perfectly beside a `full_data_0.json` whose matrices
are the wrong size is still a broken save. What is asserted:

  - the job's `name`, which is the field the page used to overwrite
  - every member the layout promises is present
  - `pae` and `contact_probs` are square at the token count, and their values
    are in range - a decode with the wrong bounds fills the key with plausible
    nonsense, which is docs/WEB.md's own complaint about reading keys and not
    values
  - `token_chain_ids` and `token_res_ids` are as long as those matrices
  - `atom_plddts` has one entry per ATOM/HETATM record in the PDB, which is the
    one cross-file check here and the one that catches a ligand or a modified
    residue counted in one file and not the other
  - the PDB carries every ligand code and every modified-residue code the job
    asked for, which no confidence number can see
"""
import argparse
import json
import re
import sys
import zipfile
from collections import Counter

NUCLEIC = ("dna", "rna")


def norm_job(raw: str) -> dict:
    """A job, in either dialect, as the comparable set of things it asks for.

    🔴 SORTED, BECAUSE ORDER IS NOT MEANING HERE. `jobFromJson` puts polymers
    before ligands - expandEntities numbers chains in list order and the
    featuriser appends ligand tokens after every polymer token - so an input
    that lists its ligand first comes back with it last. That is a deliberate
    reordering of a set, not a change to the job.
    """
    job = json.loads(raw)
    job = job[0] if isinstance(job, list) else job
    out = {"name": job.get("name"), "seed": None, "chains": [], "ligands": []}
    seeds = job.get("modelSeeds") or []
    seeds = seeds if isinstance(seeds, list) else [seeds]
    # 🔴 STRING IN ONE DIALECT AND INTEGER IN THE OTHER, so compared as a number.
    if seeds:
        out["seed"] = int(seeds[0])
    for entry in job.get("sequences", []):
        for key, body in entry.items():
            kind = {"proteinChain": "protein", "dnaSequence": "dna",
                    "rnaSequence": "rna", "protein": "protein", "dna": "dna",
                    "rna": "rna", "ligand": "ligand", "ion": "ligand"}[key]
            # copies: a `count`, or the LENGTH of an id list.
            count = body.get("count")
            if count is None:
                ids = body.get("id")
                count = len(ids) if isinstance(ids, list) else 1
            if kind == "ligand":
                smiles = body.get("smiles")
                if smiles:
                    # 🔴 NOT UPPER-CASED: case is meaning in a SMILES.
                    out["ligands"].append((f"smiles:{smiles}", int(count)))
                    continue
                codes = body.get("ccdCodes")
                if codes is None:
                    named = body.get("ligand") or body.get("ion")
                    codes = [named] if named is not None else []
                codes = codes if isinstance(codes, list) else [codes]
                # 🔴 `CCD_ATP` IS ATP - upstream's own removeprefix.
                code = "+".join(str(c).strip().upper().removeprefix("CCD_")
                                for c in codes)
                out["ligands"].append((f"ccd:{code}", int(count)))
                continue
            mods = []
            for mod in body.get("modifications") or []:
                code = str(mod.get("ptmType") or mod.get("modificationType"))
                code = code.strip().upper().removeprefix("CCD_")
                where = mod.get("ptmPosition")
                if where is None:
                    where = mod.get("basePosition")
                mods.append(f"{code}@{where}")
            out["chains"].append((kind, body.get("sequence", "").strip().upper(),
                                  int(count), tuple(sorted(mods))))
    out["chains"].sort()
    out["ligands"].sort()
    return out


def pdb_counts(text: str):
    atoms = 0
    residues = set()
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atoms += 1
            residues.add(line[17:20].strip())
    return atoms, residues


def check(job_path: str, archive_path: str) -> list:
    problems = []
    note = []
    original = norm_job(open(job_path, encoding="utf-8").read())

    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        request = next((n for n in names if n.endswith("job_request.json")), None)
        if request is None:
            return ["no *_job_request.json in the archive"], names, note
        saved = norm_job(zf.read(request).decode("utf-8"))

        # 🔴 THE NAME IS PART OF WHAT WAS HANDED IN. Every AlphaFold 3 example
        # carries one and the page used to write the fold's stem over it, so
        # `calmodulin_4calcium` came back as `af3_1` - the same chemistry under
        # an identity its author would not recognise, and invisible to any
        # comparison of entity lists.
        if saved["name"] != original["name"]:
            problems.append(f"name {saved['name']!r} saved,"
                            f" {original['name']!r} asked")
        if saved["seed"] != original["seed"]:
            problems.append(f"seed {saved['seed']} saved, {original['seed']} asked")
        if saved["chains"] != original["chains"]:
            problems.append(f"chains differ:\n      asked {original['chains']}"
                            f"\n      saved {saved['chains']}")
        if saved["ligands"] != original["ligands"]:
            problems.append(f"ligands differ:\n      asked {original['ligands']}"
                            f"\n      saved {saved['ligands']}")

        full = next((n for n in names if n.endswith("full_data_0.json")), None)
        pdb = next((n for n in names if n.endswith(".pdb")), None)
        summary = next((n for n in names
                        if n.endswith("summary_confidences_0.json")), None)
        for label, found in (("full_data_0.json", full), ("a .pdb", pdb),
                             ("summary_confidences_0.json", summary),
                             ("README.md", "README.md" in names or None)):
            if found is None:
                problems.append(f"no {label} in the archive")

        tokens = None
        if full is not None:
            data = json.loads(zf.read(full))
            pae = data.get("pae")
            contact = data.get("contact_probs")
            if not pae:
                problems.append("full_data has no pae")
            else:
                tokens = len(pae)
                if any(len(row) != tokens for row in pae):
                    problems.append("pae is not square")
                flat = [v for row in pae for v in row]
                if min(flat) < 0 or max(flat) > 40:
                    problems.append(f"pae out of range [{min(flat)}, {max(flat)}]")
                if pae[0][0] > 5:
                    problems.append(f"pae diagonal {pae[0][0]}, expected near 0")
            if contact:
                if len(contact) != tokens or any(len(r) != tokens for r in contact):
                    problems.append("contact_probs is not square at the token count")
                flat = [v for row in contact for v in row]
                if min(flat) < 0 or max(flat) > 1.0001:
                    problems.append(
                        f"contact_probs out of [0,1]: [{min(flat)}, {max(flat)}]")
            for field in ("token_chain_ids", "token_res_ids"):
                got = data.get(field)
                if got is None:
                    problems.append(f"full_data has no {field}")
                elif tokens is not None and len(got) != tokens:
                    problems.append(
                        f"{field} is {len(got)} long, {tokens} tokens")
            note.append(f"tokens {tokens}")
            note.append("chains " + ",".join(
                sorted(set(data.get("token_chain_ids") or []))))

            if pdb is not None:
                atoms, residues = pdb_counts(zf.read(pdb).decode("utf-8"))
                plddts = data.get("atom_plddts")
                if plddts is None:
                    problems.append("full_data has no atom_plddts")
                elif len(plddts) != atoms:
                    problems.append(f"atom_plddts {len(plddts)},"
                                    f" PDB has {atoms} atoms")
                note.append(f"atoms {atoms}")
                # Every ligand code and every modified residue must be IN the
                # structure - the one thing a confidence number cannot show.
                for name, _count in original["ligands"]:
                    kind, _, value = name.partition(":")
                    want = "LIG" if kind == "smiles" else value.split("+")[0]
                    if want not in residues:
                        problems.append(f"ligand {want} is not in the PDB"
                                        f" (residues: {sorted(residues)[:12]}...)")
                for _kind, _seq, _count, mods in original["chains"]:
                    for mod in mods:
                        code = mod.split("@")[0]
                        if code not in residues:
                            problems.append(f"modified residue {code}"
                                            " is not in the PDB")

        if summary is not None:
            conf = json.loads(zf.read(summary))
            for field in ("ptm", "mean_plddt"):
                if conf.get(field) is None:
                    problems.append(f"summary has no {field}")
            note.append(f"ptm {conf.get('ptm')}")

    return problems, names, note


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="the input JSON that was folded")
    parser.add_argument("--archive", required=True, help="the .zip the page wrote")
    parser.add_argument("--members", action="store_true", help="list the members")
    args = parser.parse_args()

    problems, names, note = check(args.job, args.archive)
    if args.members:
        for name in names:
            print("   ", name)
    label = args.job.rsplit("/", 1)[-1]
    if problems:
        print(f"{label}: {len(problems)} PROBLEM(S)  [{'; '.join(note)}]")
        for one in problems:
            print("   -", one)
        return 1
    print(f"{label}: archive matches the job  [{'; '.join(note)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
