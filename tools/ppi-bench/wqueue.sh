#!/bin/bash
# wqueue.sh: after the cross-evaluation and the Chai-1 design runs, the warm-start A/B in the browser - C's recipe
# with (W1) and without (W0) the warm start, same targets, lengths and seeds - alone on the GPU.
cd "$(dirname "$0")"
D=/home/ubuntu/ppi-bench-2026-10-29
while pgrep -f "xqueue\.sh|native-bench\.mjs" > /dev/null; do sleep 60; done
/home/ubuntu/localfold/cuda/af3/fold --stop --model=chai1
(./run2.sh W0 config-W0.json $D/W0.jsonl 0 1 2 &); ./run2.sh W1 config-W1.json $D/W1.jsonl 0 1 2
while pgrep -f "bench\.py" > /dev/null; do sleep 60; done
echo wqueue-done
