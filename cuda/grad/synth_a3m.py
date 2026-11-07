# A synthetic 1024-row unpaired alignment per protein chain of a structure - the query and 1023 random variants, 25% of
# positions substituted - for timing, not for training (bench_train.sh's 1AON):
#   python3 cuda/grad/synth_a3m.py <entry.cif> <out dir> <job.json from make_of3_example.py>
import sys, os, random, json, gemmi
st = gemmi.read_structure(sys.argv[1]); st.setup_entities(); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
job = json.load(open(sys.argv[3])); seqs = [next(iter(s.values()))["sequence"] for s in job["sequences"] if "protein" in s]
subs = [sub.subchain_id() for ch in st[0] for sub in ch.subchains() if st.get_entity_of(sub) is not None and st.get_entity_of(sub).entity_type == gemmi.EntityType.Polymer]
AA = "ACDEFGHIKLMNPQRSTVWY"; rng = random.Random(0)
for lab, q in zip(subs, seqs):
    with open(f"{out}/{lab}.a3m", "w") as f:
        f.write(f">query\n{q}\n")
        for r in range(1023):
            f.write(f">r{r}\n" + "".join(rng.choice(AA) if rng.random() < 0.25 else c for c in q) + "\n")
print(len(subs), "chains")
