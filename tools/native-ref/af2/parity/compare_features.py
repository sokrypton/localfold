"""Hold LocalFold's merged pre-sampling AF2 input to ColabFold's (vendor_features.py / ours_features.mjs).

    python3 cuda/af2/parity/compare_features.py <vendor.npz> <ours.json>

Rows are compared as SEQUENCES (a multiset), then by order, because the two merges may place the same row
differently and only the multiset is what the sampler draws from - while row 0 must be the query on both.
"""
import collections
import json
import sys

import numpy as np

ALPHA = "ARNDCQEGHILKMFPSTWYVX-"


def rows(msa):
    return ["".join(ALPHA[c] if c < 22 else "?" for c in r) for r in msa]


def main():
    v = np.load(sys.argv[1])
    o = json.load(open(sys.argv[2]))
    L = o["L"]
    om = np.asarray(o["msa"]).reshape(-1, L)
    od = np.asarray(o["deletion_matrix"]).reshape(-1, L)
    vm, vd = v["msa"], v["deletion_matrix"]
    print(f"depth  vendor {len(vm)}  ours {len(om)}")
    for key in ("aatype", "residue_index", "asym_id", "entity_id", "sym_id"):
        if key not in v.files:
            print(f"{key:14s} vendor has none; ours {None if o[key] is None else o[key][:3]}")
            continue
        a, b = np.asarray(v[key]).ravel(), (None if o[key] is None else np.asarray(o[key]).ravel())
        if b is None:
            print(f"{key:14s} ours has none; vendor {np.unique(a)[:5]}")
        else:
            ok = a.shape == b.shape and np.array_equal(a.astype(np.int64), b.astype(np.int64))
            # the chain ids are LABELS (ColabFold counts from 1, LocalFold from 0) and every consumer - the
            # relative encoding, ipTM's chain-pair mask - reads only equality and difference, so an offset is SAME
            shifted = (not ok and key.endswith("_id") and a.shape == b.shape
                       and np.array_equal(a.astype(np.int64) - int(a.min()), b.astype(np.int64) - int(b.min())))
            print(f"{key:14s} {'SAME' if ok else 'SAME up to a constant label offset' if shifted else 'DIFF'}"
                  + ("" if ok or shifted else f"  vendor {a[:4]}..{a[-4:]} ours {b[:4]}..{b[-4:]}"))
    vr, orr = rows(vm), rows(om)
    print("row 0 query:", "SAME" if vr[0] == orr[0] else f"DIFF\n  v {vr[0]}\n  o {orr[0]}")
    vc, oc = collections.Counter(zip(vr, map(tuple, vd.astype(int)))), collections.Counter(zip(orr, map(tuple, od.astype(int))))
    vs, os_ = collections.Counter(vr), collections.Counter(orr)
    print(f"distinct rows  vendor {len(vs)}  ours {len(os_)}  shared {len(set(vs) & set(os_))}")
    print(f"duplicates     vendor {sum(vs.values()) - len(vs)}  ours {sum(os_.values()) - len(os_)}")
    print(f"rows+deletions shared {sum((vc & oc).values())}")
    only_v = [r for r in vs if r not in os_]
    only_o = [r for r in os_ if r not in vs]
    def kind(r):
        ga = [i for i, c in enumerate(r) if c != "-"]
        return "empty" if not ga else (ga[0], ga[-1])
    asym = np.asarray(v["asym_id"]) if "asym_id" in v.files else None
    def span(r):
        if asym is None: return "?"
        ch = sorted({int(asym[i]) for i, c in enumerate(r) if c != "-"})
        return "+".join(map(str, ch)) or "none"
    print("only vendor:", len(only_v), collections.Counter(span(r) for r in only_v).most_common(4))
    print("only ours:  ", len(only_o), collections.Counter(span(r) for r in only_o).most_common(4))
    print("vendor spans:", collections.Counter(span(r) for r in vr).most_common(4))
    print("ours spans:  ", collections.Counter(span(r) for r in orr).most_common(4))


if __name__ == "__main__":
    main()
