#!/bin/bash
# The README's training table, on this machine's card: train-of3 built for the card nvidia-smi reports (sm_80 an A100,
# sm_90 a GH200/H100, sm_100 a B200, sm_120 an RTX PRO 6000 - a binary built for an older card runs its hand-written
# kernels from JIT-compiled PTX), then one training step timed at each crop on 1AON (whole, a synthetic 1024-row
# alignment), 48 diffusion samples, no recycles - the third step's time and the card's peak memory:
#
#   bash cuda/grad/bench_train.sh [--dir=<work dir>] [--python=<python3>] [crop ...]   (default crops 256 384 512 640 768)
#
# Needs nvcc, NCCL's header and library (or pip's nvidia-nccl), and a python3 with gemmi and rdkit (make_of3_example.py's
# imports). The OpenBind-0 weights and
# wwPDB's component dictionary come by cuda/featurise/fetch-weights; 1AON from the RCSB. --offload-pairs is used only
# where a crop does not fit without it (it parks each block's pair in host memory: free over a GH200's C2C link, ~7 GB
# a step across PCIe at 512 tokens).
set -uo pipefail
cd "$(dirname "$0")/../.."
work=/tmp/of3-train-bench; py=python3; crops=()
for a in "$@"; do case "$a" in --dir=*) work="${a#--dir=}" ;; --python=*) py="${a#--python=}" ;; *) crops+=("$a") ;; esac; done
[ ${#crops[@]} -gt 0 ] || crops=(256 384 512 640 768)
mkdir -p "$work"
"$py" -c "import gemmi, rdkit" 2>/dev/null || {
  echo "missing python modules: pip install gemmi rdkit" >&2; exit 1; }
cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
git log --oneline -1 2>/dev/null
NI=""; NL=""
if [ ! -e /usr/include/nccl.h ]; then
  NCCL=$("$py" -c "import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__))" 2>/dev/null || true)
  [ -n "$NCCL" ] || { echo "no NCCL: install libnccl-dev, or pip install nvidia-nccl-cu12" >&2; exit 1; }
  NI="-I$NCCL/include"; NL="-L$NCCL/lib -Xlinker -rpath=$NCCL/lib"
  [ -e "$NCCL/lib/libnccl.so" ] || ln -sf "$(ls "$NCCL"/lib/libnccl.so.* | head -1)" "$NCCL/lib/libnccl.so"
fi
echo "building train-of3 for sm_$cc"
nvcc -std=c++17 -arch=sm_$cc -O2 --default-stream per-thread $NI cuda/grad/train_of3.cu -o "$work/train-of3" -lcublas -lcublasLt -ldl -lcurand $NL -lnccl > "$work/build.log" 2>&1 || { tail -20 "$work/build.log"; exit 1; }
bash cuda/build.sh featurise > "$work/build-featurise.log" 2>&1 || { tail -20 "$work/build-featurise.log"; exit 1; }
bundle=$(cuda/featurise/fetch-weights --weights-dir="$work/weights" openbind0 | sed -n 's/^openbind0: //p' | head -1)
cuda/featurise/fetch-weights --weights-dir="$work/weights" ccd > /dev/null
[ -n "$bundle" ] || { echo "fetching the OpenBind-0 weights failed" >&2; exit 1; }
mkdir -p ccd && [ -s ccd/components.cif ] || cp "$work/weights/ccd/components.cif" ccd/components.cif
ex="$work/1aon"
if [ ! -s "$ex/example.txt" ]; then
  curl -sfL https://files.rcsb.org/download/1AON.cif.gz | gunzip > "$work/1aon.cif"
  "$py" cuda/grad/make_of3_example.py "$work/1aon.cif" "$ex" --family=openbind0 --whole | tail -1
  "$py" cuda/grad/synth_a3m.py "$work/1aon.cif" "$work/a3m" "$ex/job.json"
  rm -rf "$ex"
  "$py" cuda/grad/make_of3_example.py "$work/1aon.cif" "$ex" --family=openbind0 --whole --a3m-dir="$work/a3m" | tail -1
fi
head -1 "$ex/example.txt" > "$work/big1.txt"
printf "| tokens | step (s) | peak (GiB) | pairs |\n|---:|---:|---:|---|\n"
for n in "${crops[@]}"; do
  ms=""
  for off in "" "--offload-pairs"; do
    (while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.2; done) > "$work/mem.txt" & W=$!
    timeout 3600 "$work/train-of3" --bundle="$bundle" --family=openbind0 --examples="$work/big1.txt" --steps=3 --samples=48 \
      --sample-batch=auto --max-recycles=0 --crop=$n --warmup=1 --tf32 --bf16 --bf16-confidence $off > "$work/t$n.txt" 2>&1
    kill $W; wait $W 2>/dev/null
    ms=$(grep -E "^step +2 " "$work/t$n.txt" | grep -oE "[0-9]+ ms$" | grep -oE "^[0-9]+")
    [ -n "$ms" ] && break
  done
  if [ -z "$ms" ]; then echo "| $n | out of memory | | |"; break; fi
  peak=$(sort -n "$work/mem.txt" | tail -1)
  awk -v n="$n" -v ms="$ms" -v p="$peak" -v o="${off:-on the card}" 'BEGIN { printf "| %d | %.2f | %.1f | %s |\n", n, ms / 1000, p / 1024, o }'
done
