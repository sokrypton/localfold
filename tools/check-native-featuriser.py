"""The native featurisers against the JavaScript exporters they replace: byte for byte.

    python3 tools/check-native-featuriser.py                 # the whole corpus, every family it names
    python3 tools/check-native-featuriser.py --only=6mrr-sep # cases whose name contains it
    python3 tools/check-native-featuriser.py --keep          # leave both outputs in /tmp/claude-1000/nf

Each case runs the JavaScript exporter (cuda/af3/export-model.mjs --no-weights, cuda/af2/export_input.mjs,
cuda/esmfold2/export_input.mjs) and the native one (cuda/featurise/*-featurise) on the same arguments and holds
them to the SAME BYTES: model.idx's text, model.bin entry by entry, and the PDB records. The first entry that
differs is named with its first differing element, because "the files differ" says nothing about where.

🔴 A REFUSAL MUST BE THE SAME REFUSAL: a case either exporter refuses passes only when both refuse, in the same
words - the page's sentence is the answer a reader sees, so the native port may not say something else.
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NODE = ["node", "--js-float16array", "--max-old-space-size=24000"]
WORK = "/tmp/claude-1000/nf"
FIX = os.path.join(REPO, "tools", "fixtures")
AF3_EXAMPLES = os.path.join(FIX, "af3-jobs")

SEQ_6MRR = "GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK"
TEST_QUERY = open(os.path.join(FIX, "test.a3m")).read().split(">")[1].splitlines()[1].strip()
AF3_FAMILIES = ["af3", "openbind0", "opendde", "boltz2", "protenix2", "intellifold2", "rosettafold3", "chai1"]


def job(name, sequences, **extra):
    return {"name": name, "modelSeeds": [1], "dialect": "alphafold3", "version": 1, "sequences": sequences, **extra}


def protein(chain, sequence, **extra):
    return {"protein": {"id": chain, "sequence": sequence, **extra}}


def chain_sequence(name, chain):
    """A crystal's chain, one letter a residue, from its CA records"""
    three = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
             "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
             "TYR": "Y", "VAL": "V"}
    out, seen = [], set()
    for line in open(os.path.join(FIX, f"{name}-crystal.pdb")):
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == chain and line[22:27] not in seen:
            seen.add(line[22:27])
            out.append(three.get(line[17:20], "X"))
    return "".join(out)


def synthetic_a3m(name, query, rows, seed):
    """A deterministic alignment for query: substitutions, gaps, insertions and a duplicated row, written to WORK"""
    import random
    rng = random.Random(seed)
    letters = "ACDEFGHIKLMNPQRSTVWY"
    out = [f">query\n{query}"]
    previous = None
    for r in range(rows):
        if previous is not None and r % 7 == 3:
            out.append(f">dup{r}\n{previous}")      # (deduplication must drop it)
            continue
        row = []
        for c in query:
            x = rng.random()
            row.append("-" if x < 0.08 else rng.choice(letters) if x < 0.4 else c)
            if rng.random() < 0.03:
                row.append(rng.choice(letters).lower() * rng.randint(1, 3))
        previous = "".join(row)
        out.append(f">hit{r} desc {r}\n{previous}")
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, f"{name}.a3m")
    open(path, "w").write("\n".join(out) + "\n")
    return path


def cases():
    """(name, port, job dict or None, extra args, families)"""
    out = [
        ("6mrr", "af3", job("t", [protein("A", SEQ_6MRR)]), [], AF3_FAMILIES),
        ("6mrr-sep-gol", "af3", job("t", [protein("A", SEQ_6MRR, modifications=[{"ptmType": "SEP", "ptmPosition": 3}]),
                                          {"ligand": {"id": "B", "ccdCodes": ["GOL"]}}]), [], AF3_FAMILIES),
        ("sep-end", "af3", job("t", [protein("A", "MKTAYIAKQRS", modifications=[{"ptmType": "SEP", "ptmPosition": 11}])]),
         [], AF3_FAMILIES),
        ("dimer-dna", "af3", job("t", [protein(["A", "B"], "MKTAYIAKQRQISFVKSHFSRQ"),
                                       {"dna": {"id": "C", "sequence": "ACGTTGCA"}}, {"rna": {"id": "D", "sequence": "ACGUU"}}]),
         [], AF3_FAMILIES),
        ("msa-test", "af3", job("t", [protein("A", TEST_QUERY)]), ["--a3m=@test.a3m"], AF3_FAMILIES),
    ]
    caj, brs_a, brs_d = chain_sequence("5caj", "A"), chain_sequence("1brs", "A"), chain_sequence("1brs", "D")
    out += [
        ("tmpl-5caj", "af3", job("t", [protein("A", caj)]), ["--template=@F/5caj-crystal.pdb:A@0"], AF3_FAMILIES),
        # (a sequence the template does not match: the page's local alignment maps it; and a second, unrelated slot)
        ("tmpl-aligned", "af3", job("t", [protein("A", caj[:40] + caj[52:150].replace("L", "I") + caj[160:])]),
         ["--template=@F/5caj-crystal.pdb:A@0,@F/1qys-crystal.pdb:A@0"], AF3_FAMILIES),
        ("tmpl-1brs-merged", "af3", job("t", [protein("A", brs_a), protein("D", brs_d)]),
         ["--template=@F/1brs-crystal.pdb:A@0+@F/1brs-crystal.pdb:D@1"], AF3_FAMILIES),
        ("tmpl-1brs-nospan", "af3", job("t", [protein("A", brs_a), protein("D", brs_d)]),
         ["--template=@F/1brs-crystal.pdb:A@0+@F/1brs-crystal.pdb:D@1", "--no-span-chains"], AF3_FAMILIES),
        # (a modified residue and a ligand shift every token after them)
        ("tmpl-6mrr-sep-gol", "af3", job("t", [protein("A", SEQ_6MRR, modifications=[{"ptmType": "SEP", "ptmPosition": 3}]),
                                               {"ligand": {"id": "B", "ccdCodes": ["GOL"]}}]),
         ["--template=@F/6mrr-crystal.pdb:A@0"], AF3_FAMILIES),
        ("tmpl-four", "af3", job("t", [protein("A", caj)]),
         ["--template=" + ",".join(["@F/5caj-crystal.pdb:A@0"] * 4)], AF3_FAMILIES),
    ]
    # 🔴 THE NETWORK ARM (--network): both exporters search api.colabfold.com for the same queries - the server
    # answers one query the same way twice, so the alignments, the pairing, the merge and the template hits must agree
    out += [
        ("net-search-6mrr", "af3", job("t", [protein("A", SEQ_6MRR)]), ["--search"], ["af3", "boltz2"], "network"),
        ("net-search-1brs", "af3", job("t", [protein("A", brs_a), protein("D", brs_d)]), ["--search"], ["af3", "rosettafold3"], "network"),
        ("net-search-templates", "af3", job("t", [protein("A", SEQ_6MRR)]), ["--search-templates"], ["af3", "protenix2"], "network"),
        ("net-search-chain-template", "af3", job("t", [protein("A", brs_a), protein("D", brs_d)]),
         ["--search", "--template-search-chains=1"], ["af3", "boltz2"], "network"),
    ]
    # AlphaFold 2 (cuda/af2/export_input.mjs): "families" are the bundles, monomer and multimer
    sa, sd = synthetic_a3m("brs-a", brs_a, 300, 1), synthetic_a3m("brs-d", brs_d, 200, 2)
    pa, pd = synthetic_a3m("brs-pa", brs_a, 60, 3), synthetic_a3m("brs-pd", brs_d, 60, 4)
    m6 = synthetic_a3m("6mrr", SEQ_6MRR, 900, 5)
    deep = synthetic_a3m("deep-5caj", caj, 8000, 9)      # (past 1 MiB: the JavaScript exporter's worker-thread path)
    AF2 = ["monomer", "multimer"]
    out += [
        ("af2-6mrr", "af2", job("t", [protein("A", SEQ_6MRR)]), [], AF2),
        ("af2-6mrr-msa", "af2", job("t", [protein("A", SEQ_6MRR)]), [f"--a3m={m6}", "--seed=7", "--recycles=2"], AF2),
        ("af2-6mrr-shallow", "af2", job("t", [protein("A", SEQ_6MRR)]), [f"--a3m={m6}", "--max-msa=16", "--max-extra=40"], AF2),
        ("af2-deep", "af2", job("t", [protein("A", caj)]), [f"--a3m={deep}"], ["monomer"]),
        ("af2-test59", "af2", job("t", [protein("A", TEST_QUERY)]), ["--a3m=@test.a3m", "--max-msa=508"], AF2),
        ("af2-1brs", "af2", job("t", [protein("A", brs_a), protein("D", brs_d)]), [], AF2),
        ("af2-1brs-a3ms", "af2", job("t", [protein("A", brs_a), protein("D", brs_d)]), [f"--a3m={sa},{sd}"], AF2),
        ("af2-1brs-paired", "af2", job("t", [protein("A", brs_a), protein("D", brs_d)]),
         [f"--a3m={sa},{sd}", f"--paired-a3m={pa},{pd}"], AF2),
        ("af2-homodimer", "af2", job("t", [protein(["A", "B"], SEQ_6MRR)]), [f"--a3m={m6},{m6}"], AF2),
        ("af2-5caj-template", "af2", job("t", [protein("A", caj)]), ["--template=@F/5caj-crystal.pdb:A"], AF2),
        ("af2-1brs-template", "af2", job("t", [protein("A", brs_a), protein("D", brs_d)]),
         ["--template=@F/1brs-crystal.pdb:A@0+@F/1brs-crystal.pdb:D@1"], AF2),
        ("af2-ligand-refused", "af2", job("t", [protein("A", SEQ_6MRR), {"ligand": {"id": "B", "ccdCodes": ["GOL"]}}]), [], AF2),
    ]
    # ESMFold2 (cuda/esmfold2/export_input.mjs)
    EF = ["esmfold2"]
    out += [
        ("ef2-6mrr", "esmfold2", job("t", [protein("A", SEQ_6MRR)]), [], EF),
        ("ef2-6mrr-sep-gol", "esmfold2", job("t", [protein("A", SEQ_6MRR, modifications=[{"ptmType": "SEP", "ptmPosition": 3}]),
                                                   {"ligand": {"id": "B", "ccdCodes": ["GOL"]}}]), [], EF),
        ("ef2-dimer-dna", "esmfold2", job("t", [protein(["A", "B"], "MKTAYIAKQRQISFVKSHFSRQ"),
                                                {"dna": {"id": "C", "sequence": "ACGTTGCA"}}, {"rna": {"id": "D", "sequence": "ACGUU"}}]),
         [], EF),
        ("ef2-biotin-smiles", "esmfold2", job("t", [protein("A", SEQ_6MRR),
                                                    {"ligand": {"id": "B", "smiles": "OC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12"}}]), [], EF),
        ("ef2-1brs-a3m", "esmfold2", job("t", [protein("A", brs_a), protein("D", brs_d)]), [f"--a3m={sa},{sd}"], EF),
    ]
    if os.path.isdir(AF3_EXAMPLES):
        for name in sorted(os.listdir(AF3_EXAMPLES)):
            if name.endswith(".json"):
                out.append((f"example-{name[:-5]}", "af3", os.path.join(AF3_EXAMPLES, name), [], ["af3", "boltz2", "rosettafold3"]))
                out.append((f"ef2-example-{name[:-5]}", "esmfold2", os.path.join(AF3_EXAMPLES, name), [], ["esmfold2"]))
    return out


def resolve_cases():
    """(name, request, network) for resolve-templates against cuda/resolve_templates.mjs"""
    caj = open(os.path.join(FIX, "5caj-crystal.pdb")).read()
    upload = {"kind": "upload", "text": caj, "source": "A", "filename": "5caj.pdb"}
    return [
        ("resolve-upload", {"entities": [
            {"type": "protein", "value": "MKTAYIAKQR", "copies": 2, "template": upload},
            {"type": "dna", "value": "ACGT", "copies": 1},
            {"type": "protein", "value": "MKTAYIAKQR", "copies": 1, "template": {"kind": "search"}},
            {"type": "ligand", "value": "GOL", "copies": 1},
            {"type": "protein", "value": "GGGG", "copies": 1, "template": {"kind": "pdb", "source": ""}},
            {"type": "protein", "value": "GGGG", "copies": 1, "template": {"kind": "upload", "text": caj}}]}, False),
        ("resolve-pdb", {"entities": [{"type": "protein", "value": "MKTAYIAKQR", "copies": 1,
                                       "template": {"kind": "pdb", "source": "1QYS_A"}}]}, True),
        # 🔴 AlphaFold DB ANSWERS NODE'S fetch WITH 403 (and serves curl and a browser), so the JavaScript resolver
        # cannot be this case's reference from a command line: the native side is checked on its own output
        ("resolve-afdb", {"entities": [{"type": "protein", "value": "MKTAYIAKQR", "copies": 1,
                                        "template": {"kind": "afdb", "source": "P69905"}}]}, "native-only"),
    ]


def check_resolve(name, request, network):
    base = os.path.join(WORK, name)
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base)
    path = os.path.join(base, "request.json")
    json.dump(request, open(path, "w"))
    nv = run([os.path.join(REPO, "cuda", "featurise", "resolve-templates"), path, base + "/native"])
    if network == "native-only":
        if nv[0] != 0:
            return f"native exit {nv[0]} ({refusal(nv[1])})"
        [entry] = json.loads(nv[1].strip().splitlines()[-1])
        text = open(entry["file"]).read()
        atoms = sum(line.startswith("ATOM") for line in text.splitlines())
        if entry["kind"] != "afdb" or atoms < 100 or entry["source"] != request["entities"][0]["template"]["source"]:
            return f"{entry} with {atoms} ATOM records"
        return None
    js = run(["node", os.path.join(REPO, "cuda", "resolve_templates.mjs"), path, base + "/js"])
    if js[0] != 0 or nv[0] != 0:
        if js[0] != 0 and nv[0] != 0 and refusal(js[1]) == refusal(nv[1]):
            return None
        return f"JS exit {js[0]} ({refusal(js[1])}), native exit {nv[0]} ({refusal(nv[1])})"
    a = json.loads(js[1].strip().splitlines()[-1].replace(base + "/js", "@"))
    b = json.loads(nv[1].strip().splitlines()[-1].replace(base + "/native", "@"))
    if a != b:
        return f"{a} against native {b}"
    for entry in a:
        if "file" in entry:
            fa, fb = entry["file"].replace("@", base + "/js"), entry["file"].replace("@", base + "/native")
            if open(fa, "rb").read() != open(fb, "rb").read():
                return f"{entry['file']} differs"
    return None


def read_idx(path):
    entries = []
    for line in open(path).read().splitlines():
        parts = line.split(" ")
        if parts[0] == "m":
            entries.append(("m", parts[1], " ".join(parts[2:])))
        else:
            entries.append((parts[0], parts[1], int(parts[2]), int(parts[3])))
    return entries


def compare(js_dir, native_dir):
    """None when identical, else a sentence naming the first difference."""
    for name in ("template.pdb", "pdb.template"):
        a, b = os.path.join(js_dir, name), os.path.join(native_dir, name)
        if os.path.exists(a) != os.path.exists(b):
            return f"{name}: present on one side only"
        if os.path.exists(a) and open(a, "rb").read() != open(b, "rb").read():
            la, lb = open(a).read().splitlines(), open(b).read().splitlines()
            for i, (x, y) in enumerate(zip(la, lb)):
                if x != y:
                    return f"{name} line {i + 1}: {x!r} against native {y!r}"
            return f"{name}: {len(la)} lines against native {len(lb)}"
    ia, ib = read_idx(os.path.join(js_dir, "model.idx")), read_idx(os.path.join(native_dir, "model.idx"))
    ba, bb = open(os.path.join(js_dir, "model.bin"), "rb").read(), open(os.path.join(native_dir, "model.bin"), "rb").read()
    for k, (x, y) in enumerate(zip(ia, ib)):
        if x[0] != y[0] or x[1] != y[1]:
            return f"entry {k}: {x[0]} {x[1]} against native {y[0]} {y[1]}"
        if x[0] == "m":
            if x[2] != y[2]:
                return f"{x[1]}: {x[2]} against native {y[2]}"
            continue
        if x[3] != y[3]:
            return f"{x[1]}: length {x[3]} against native {y[3]}"
        ca, cb = ba[x[2] * 4:(x[2] + x[3]) * 4], bb[y[2] * 4:(y[2] + y[3]) * 4]
        if ca != cb:
            fmt = "f" if x[0] == "t" else "i"
            va, vb = struct.unpack(f"<{x[3]}{fmt}", ca), struct.unpack(f"<{y[3]}{fmt}", cb)
            diff = [i for i in range(x[3]) if struct.pack(f"<{fmt}", va[i]) != struct.pack(f"<{fmt}", vb[i])]
            i = diff[0]
            return f"{x[1]}: {len(diff)} of {x[3]} differ, first [{i}] {va[i]!r} against native {vb[i]!r}"
    if len(ia) != len(ib):
        extra = (ia if len(ia) > len(ib) else ib)[min(len(ia), len(ib))]
        return f"{len(ia)} entries against native {len(ib)} (first unmatched: {extra[1]} on the {'JS' if len(ia) > len(ib) else 'native'} side)"
    return None


def run(cmd, cwd=REPO):
    started = time.time()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr, time.time() - started


def refusal(said):
    for line in said.splitlines():
        if line.startswith("Error: ") or "Error: " in line[:40]:
            return line.split("Error: ", 1)[1].strip()
    return said.strip().splitlines()[-1] if said.strip() else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--family", default="")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--network", action="store_true", help="only the cases that search api.colabfold.com")
    a = parser.parse_args()
    native = os.path.join(REPO, "cuda", "featurise", "af3-featurise")
    if not os.access(native, os.X_OK):
        sys.exit(f"{native} is not built (cuda/featurise/build.sh)")
    os.makedirs(WORK, exist_ok=True)
    failed, passed, skipped = [], 0, 0
    for case in cases():
        name, port, spec, extra, families = case[:5]
        if a.only and a.only not in name:
            continue
        if (len(case) > 5 and case[5] == "network") != a.network:
            continue
        for family in families:
            if a.family and family != a.family:
                continue
            tag = f"{name}/{family}"
            base = os.path.join(WORK, name.replace("/", "_") + "-" + family)
            shutil.rmtree(base, ignore_errors=True)
            os.makedirs(base)
            if isinstance(spec, str):
                job_path = spec
            else:
                job_path = os.path.join(base, "job.json")
                json.dump(spec, open(job_path, "w"))
            args = [x.replace("@test.a3m", os.path.join(FIX, "test.a3m")).replace("@F/", FIX + "/") for x in extra]
            if port == "esmfold2":
                common = [f"--job={job_path}", *args]
                js_code, js_said, js_s = run([*NODE, os.path.join(REPO, "cuda", "esmfold2", "export_input.mjs"), base + "/js", *common])
                nv_code, nv_said, nv_s = run([native.replace("af3-featurise", "esmfold2-featurise"), base + "/native", *common])
            elif port == "af2":
                bundle = os.path.join(REPO, "model" if family == "monomer" else "model-multimer")
                common = [f"--bundle={bundle}", f"--job={job_path}", *args]
                js_code, js_said, js_s = run([*NODE, os.path.join(REPO, "cuda", "af2", "export_input.mjs"), base + "/js", *common])
                nv_code, nv_said, nv_s = run([native.replace("af3-featurise", "af2-featurise"), base + "/native", *common])
            else:
                common = ["--no-weights", f"--family={family}", f"--job={job_path}", "--max-msa=512", *args]
                js_code, js_said, js_s = run([*NODE, os.path.join(REPO, "cuda", "af3", "export-model.mjs"), base + "/js", *common],
                                             cwd=os.path.join(REPO, "cuda", "af3"))
                nv_code, nv_said, nv_s = run([native, base + "/native", *common])
            if js_code != 0 or nv_code != 0:
                if js_code != 0 and nv_code != 0 and refusal(js_said) == refusal(nv_said):
                    print(f"  ok   {tag}: both refuse - {refusal(js_said)}")
                    passed += 1
                elif js_code != 0 and nv_code != 0:
                    failed.append(tag)
                    print(f"  FAIL {tag}: both refuse, differently\n         JS:     {refusal(js_said)}\n         native: {refusal(nv_said)}")
                else:
                    failed.append(tag)
                    side, said = ("JS", js_said) if js_code != 0 else ("native", nv_said)
                    print(f"  FAIL {tag}: only {side} refuses - {refusal(said)}")
                continue
            problem = compare(base + "/js", base + "/native")
            if problem:
                failed.append(tag)
                print(f"  FAIL {tag}: {problem}")
            else:
                passed += 1
                print(f"  ok   {tag}: identical (JS {js_s:.2f} s, native {nv_s:.2f} s)")
            if not a.keep and not problem:
                shutil.rmtree(base, ignore_errors=True)
    for name, request, network in resolve_cases():
        if (a.only and a.only not in name) or bool(network) != a.network:
            continue
        problem = check_resolve(name, request, network)
        if problem:
            failed.append(name)
            print(f"  FAIL {name}: {problem}")
        else:
            passed += 1
            print(f"  ok   {name}: " + ("the native resolver fetched it (no JavaScript reference)" if network == "native-only" else "identical"))
    print(f"\n{passed} identical, {len(failed)} differ" + (f": {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
