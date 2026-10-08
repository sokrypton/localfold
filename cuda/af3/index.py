import json, sys
d = json.load(open(sys.argv[1] + "/block.json"))
with open(sys.argv[1] + "/block.idx", "w") as f:
    for k, v in d["index"].items(): f.write(f"t {k} {v['offset']} {v['length']}\n")
    for k, v in d["meta"].items():
        if not isinstance(v, str): f.write(f"m {k} {float(v)}\n")
