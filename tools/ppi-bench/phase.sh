#!/bin/bash
# phase.sh: one model server at a time - AF3 (NC's misses, NM1), then Boltz-2 (NH's misses, NHM), then Chai-1 (NX, NXH).
# Three resident servers and the MSA folds ran the 40 GB card out of memory (2026-10-29), so the phases do not overlap.
cd "$(dirname "$0")"
D=/home/ubuntu/ppi-bench-2026-10-29
F=/home/ubuntu/localfold/cuda/af3/fold
$F --serve
(./run-native.sh NC config-NC.json $D/NC.jsonl 0 1 2 3 &); ./run-native.sh NM1 config-NM1.json $D/NM1.jsonl 0 1 2 3
while pgrep -f "config-N(C|M1)\.json" > /dev/null; do sleep 30; done
$F --stop; $F --serve --model=boltz2
(./run-native.sh NH config-NH.json $D/NH.jsonl 0 1 &); ./run-native.sh NHM config-NHM.json $D/NHM.jsonl 0
while pgrep -f "config-NHM?\.json" > /dev/null; do sleep 30; done
$F --stop --model=boltz2; $F --serve --model=chai1
(./run-native.sh NX config-NX.json $D/NX.jsonl 0 1 &); ./run-native.sh NXH config-NXH.json $D/NXH.jsonl 0
while pgrep -f "config-NXH?\.json" > /dev/null; do sleep 30; done
echo phases-done
