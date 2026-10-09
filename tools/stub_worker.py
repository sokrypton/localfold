"""A stand-in for python/localfold/worker.py that a gate drives, so the broker (python/localfold/server.py) and the reader's page are tested with no card.

    LOCALFOLD_WORKER=tools/stub_worker.py \
    LOCALFOLD_STUB_FEED=/tmp/feed.jsonl LOCALFOLD_STUB_JOBS=/tmp/jobs.jsonl \
        python3 python/localfold/server.py --native ...

It speaks the worker's protocol - one job a line on stdin, one bridge event a
line on stdout, `ready` first - and decides nothing itself:

  * every job it is handed is appended to `LOCALFOLD_STUB_JOBS`, which is how a
    gate reads what the reader's Fold actually sent;
  * from the moment a job arrives it emits each line a gate appends to
    `LOCALFOLD_STUB_FEED` (`{"kind", "payload"}`, stamped `at` here) and HOLDS
    the fold, as a real one on a card does, until it has emitted a `result`;
  * a line `{"kind": "__exit"}` makes it exit mid-fold, which is how a gate asks
    what the broker does with a worker that dies.

Used by tools/check-remote-bridge.py and tools/check-model-pending.py.
"""
import json
import os
import sys
import time


def say(kind, payload):
    print(json.dumps({"kind": kind, "payload": payload, "at": int(time.time() * 1000)}), flush=True)


def main():
    feed = os.environ["LOCALFOLD_STUB_FEED"]
    jobs = os.environ.get("LOCALFOLD_STUB_JOBS")
    say("ready", {})
    for line in sys.stdin:
        if not line.strip():
            continue
        # ...the feed's end as the job arrives: what a gate appended before it belongs to an earlier fold
        start = os.path.getsize(feed) if os.path.exists(feed) else 0
        if jobs:
            with open(jobs, "a") as handle:
                handle.write(line if line.endswith("\n") else line + "\n")
        done = False
        while not done:
            if os.path.exists(feed):
                with open(feed) as handle:
                    handle.seek(start)
                    text = handle.read()
                complete = text[:text.rfind("\n") + 1]
                start += len(complete.encode())
                for row in complete.splitlines():
                    if not row.strip():
                        continue
                    event = json.loads(row)
                    if event.get("kind") == "__exit":
                        sys.exit(3)
                    say(event["kind"], event.get("payload"))
                    done = done or event["kind"] == "result"
            if not done:
                time.sleep(0.02)


if __name__ == "__main__":
    main()
