#!/bin/bash
# xqueue.sh: when the Boltz-2 design runs are done, cross-evaluate C, NC and NH's designs with each other model in
# turn - one model on the GPU at a time - then resume the Chai-1 design runs.
cd "$(dirname "$0")"
D=/home/ubuntu/ppi-bench-2026-10-29
F=/home/ubuntu/localfold/cuda/af3/fold
while pgrep -f "config-NHM\.json" > /dev/null; do sleep 30; done
$F --stop --model=boltz2; $F --stop
for m in boltz2 chai1 protenix2 ef2; do
  for r in C NC NH; do python3 xeval.py $m $D/$r.jsonl $D/x-$m-$r.jsonl > xeval-$m-$r.log 2>&1 || echo "FAIL $m $r"; done
  echo "done $m"
done
$F --serve --model=chai1
(./run-native.sh NX config-NX.json $D/NX.jsonl 0 1 &); ./run-native.sh NXH config-NXH.json $D/NXH.jsonl 0
echo xqueue-done
