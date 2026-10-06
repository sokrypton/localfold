#!/usr/bin/env bash
# The same folds on any GPU, to hold one machine's native ports against another's:
#
#   native/crosscheck.sh <out dir>
#
# Each input is featurised here by the repo's own exporters (deterministic), each port built for this
# GPU (into <out>), and every fold's log line and PDB kept: compare two machines' <out> dirs with
# native/af3/score.py (CA RMSD between the two PDBs of a case) and the pLDDT/pTM lines. Every port reads
# the page's published bundles as they are; a port whose bundle is not on disk is skipped.
set -uo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"; N="$repo/native"
out="$1"; shift; mkdir -p "$out"
arch="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader > "$out/gpu.txt"
log="$out/log.txt"; : > "$log"
run() {      # (shell timing: a Colab image has no /usr/bin/time)
  local name="$1" t0 rc; shift; t0=$(date +%s.%N)
  { echo "== $name"; "$@" 2>&1; rc=$?; echo "wall $(awk "BEGIN{printf \"%.2f\", $(date +%s.%N) - $t0}") s"; echo "exit $rc"; } >> "$log" 2>&1
}
build() { (cd "$N/$1" && nvcc -O1 -std=c++17 -arch=$arch --default-stream per-thread $2 src/$1.cu -lcublas -lcublasLt -lcupti -ldl \
          -o "$out/$1" 2>&1 | grep -E "error" >> "$log"); }
S6=GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE
seqof() { python3 - "$1" "$2" <<'EOF'
import sys
three = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K",
             MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V", MSE="M")
seen, s = set(), ""
for l in open(sys.argv[1]):
    if l.startswith(("ATOM", "HETATM")) and l[12:16] == " CA " and l[21] == sys.argv[2] and l[22:27] not in seen:
        seen.add(l[22:27]); s += three.get(l[17:20], "X")
print(s)
EOF
}
FX="$repo/tools/fixtures"; S5=$(seqof "$FX/5caj-crystal.pdb" A); SA=$(seqof "$FX/1brs-crystal.pdb" A); SD=$(seqof "$FX/1brs-crystal.pdb" D)
in="$out/inputs"; mkdir -p "$in"

EF=(--fold-bundle="$repo/model-esmfold2-int5" --esmc-bundle="$repo/model-esmc-600m-int3")    # (the bundles as they are)
if [ -f "$repo/model-esmfold2-int5/manifest.json" ]; then
  build ef2 ""
  ex() { local d="$in/ef2-$1"; shift; [ -f "$d/model.idx" ] || node --js-float16array "$N/ef2/export_input.mjs" "$d" "$@" > /dev/null; }
  ex 6mrr --sequence=$S6; ex 5caj --sequence=$S5; ex 1brs --sequence=$SA:$SD
  ex gol-sep --sequence=$S6 --ligands=GOL --modify=SEP@3; ex dna --sequence=GCGATCGATCGC:GCGATCGATCGC --kinds=dna,dna
  for c in 6mrr 5caj 1brs gol-sep dna; do run ef2-$c "$out/ef2" "$in/ef2-$c" "${EF[@]}" --fast --warm=96,800 --out="$out/ef2-$c.pdb"; done
  run ef2-6mrr-f32 "$out/ef2" "$in/ef2-6mrr" "${EF[@]}" --out="$out/ef2-6mrr-f32.pdb"
fi
AW=(--bundle="$repo/model-af3-int5" --map="$N/af3/maps/af3.map")    # (the bundle as it is, through its map)
if [ -f "$repo/model-af3-int5/manifest.json" ]; then
  build af3 "--use_fast_math"
  B="$repo/model-af3-int5/manifest.json"
  ex3() { local d="$in/af3-$1"; shift; [ -f "$d/model.idx" ] || (cd "$N/af3" && node --js-float16array --max-old-space-size=24000 \
          export-model.mjs "$d" --no-weights --bundle="$B" "$@" > /dev/null); }
  ex3 6mrr --sequence=$S6; ex3 5caj-tmpl --sequence=$S5 --template="$FX/5caj-crystal.pdb:A"
  ex3 1brs-tmpl --sequence=$SA:$SD --template="$FX/1brs-crystal.pdb:A@0+$FX/1brs-crystal.pdb:D@1"; ex3 gol --sequence=$S6 --ligands=GOL
  for c in 6mrr 5caj-tmpl 1brs-tmpl gol; do run af3-$c "$out/af3" "$in/af3-$c" "${AW[@]}" --fold --fast --out="$out/af3-$c.pdb"; done
  run af3-6mrr-f32 "$out/af3" "$in/af3-6mrr" "${AW[@]}" --fold --steps=20 --out="$out/af3-6mrr-f32.pdb"
fi
M1=(--bundle="$repo/model" --map="$N/af2/maps/model_1_ptm.map")
MM=(--bundle="$repo/model-multimer" --map="$N/af2/maps/model_1_multimer_v3.map")
if [ -f "$repo/model/manifest.json" ]; then
  build af2 "--use_fast_math"
  ex2() { local d="$in/af2-$1" b="$2"; shift 2; [ -f "$d/model.idx" ] || node --js-float16array "$N/af2/export_input.mjs" "$d" --bundle="$b" "$@" > /dev/null; }
  ex2 6mrr "$repo/model" --sequence=$S6; ex2 5caj-tmpl "$repo/model" --sequence=$S5 --template="$FX/5caj-crystal.pdb:A"
  run af2-6mrr "$out/af2" "$in/af2-6mrr" "${M1[@]}" --fast --out="$out/af2-6mrr.pdb"
  run af2-5caj-tmpl "$out/af2" "$in/af2-5caj-tmpl" "${M1[@]}" --fast --recycles=0 --out="$out/af2-5caj-tmpl.pdb"
  run af2-6mrr-f32 "$out/af2" "$in/af2-6mrr" "${M1[@]}" --out="$out/af2-6mrr-f32.pdb"
  if [ -f "$repo/model-multimer/manifest.json" ]; then
    ex2 1brs-tmpl "$repo/model-multimer" --sequence=$SA:$SD --template="$FX/1brs-crystal.pdb:A+D"
    run af2-1brs-tmpl "$out/af2" "$in/af2-1brs-tmpl" "${MM[@]}" --fast --out="$out/af2-1brs-tmpl.pdb"
  fi
fi
echo done > "$out/DONE"
