"""Push, deploy, and prove the site is serving what was pushed.

    python3 tools/deploy.py              # push main, deploy, verify
    python3 tools/deploy.py --dry-run    # say what it would do
    python3 tools/deploy.py --verify     # just check what is live now

🔴 A PUSH MAY NOT DEPLOY, AND SAYS NOTHING WHEN IT DOES NOT. For most of this
repository's life it did not: it began as a fork, GitHub disables automatic
workflow triggers on forks, and the first 17 workflow runs here were every one
of them "manually run". `workflow_dispatch` kept working the whole time, because
that is an explicit request, so the Actions tab showed green runs while pushes
fired nothing.

That state is stored per repository and OUTLIVES ITS CAUSE. Detaching the fork
did not clear it - only "Enable Actions on this repository", on the Actions tab,
did. It is also invisible from the API that looks like it would say:
`/actions/permissions` reports `enabled: true` throughout, because that field is
the allowed-actions policy and not this switch.

Meanwhile the failure was silent and convincing: the push succeeded, a green run
from earlier sat at the top of the Actions tab, and the site served the previous
build. Worse, while GitHub Pages was still on its legacy branch build, a push DID
republish the site - without the weights, which are not in the repository - so
pushing appeared to work while quietly removing the model.

Dispatching explicitly costs nothing and cannot be silently switched off, so this
does it either way. What matters more is the last step: it ends by reading the
deployed site back. tools/build_site.py writes dist/build.json carrying the
commit it built, and this polls the live copy until that commit is the one being
served. "Live" is then a fact with a timestamp on it, not an impression - which
is the part worth keeping however GitHub is feeling about triggers.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

WORKFLOW = "Deploy WebGPU demo"
BRANCH = "main"
# Pages serves from a CDN that holds a build briefly after it is published.
POLL_SECONDS = 10
DEPLOY_TIMEOUT_SECONDS = 900


def run(command: list[str], **kwargs) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True, **kwargs)
    return result.stdout.strip()


def repository() -> str:
    """owner/name, from the origin remote rather than a constant.

    A hardcoded owner is wrong in every clone but one, and wrong quietly: it
    would push here and then poll someone else's site for the commit.
    """
    url = run(["git", "remote", "get-url", "origin"])
    path = url.split("github.com", 1)[-1].lstrip(":/")
    return path.removesuffix(".git")


def site(owner_and_name: str) -> str:
    owner, name = owner_and_name.split("/", 1)
    return f"https://{owner}.github.io/{name}"


def live_build(site_url: str) -> dict | None:
    """What the deployed site says it is, or None if it does not say."""
    request = urllib.request.Request(
        f"{site_url}/build.json?t={int(time.time())}",
        headers={"Cache-Control": "no-cache"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


def report_live(site_url: str) -> int:
    build = live_build(site_url)
    if build is None:
        print(f"{site_url}/build.json is not being served."
              " Either the site predates the build stamp, or nothing is deployed.")
        return 1
    print(f"live: {build['commit'][:8]}  built {build.get('builtAt', 'at an unknown time')}")
    try:
        head = run(["git", "rev-parse", "HEAD"])
        if build["commit"] == head:
            print("       ...which is this checkout's HEAD.")
        else:
            behind = run(["git", "rev-list", "--count", f"{build['commit']}..HEAD"])
            print(f"       ...which is {behind} commit(s) behind this checkout's {head[:8]}.")
    except subprocess.CalledProcessError:
        pass                    # the live commit may not exist locally
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="report what the site is serving and stop")
    arguments = parser.parse_args()

    name = repository()
    site_url = site(name)
    if arguments.verify:
        return report_live(site_url)

    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch != BRANCH:
        print(f"on {branch}, not {BRANCH}; deploying serves {BRANCH}", file=sys.stderr)
        return 1
    if run(["git", "status", "--porcelain"]):
        print("the working tree is dirty; commit or stash first", file=sys.stderr)
        return 1
    head = run(["git", "rev-parse", "HEAD"])

    if arguments.dry_run:
        print(f"would push {head[:8]} to origin/{BRANCH} ({name}),"
              f" dispatch {WORKFLOW!r}, and wait for {site_url} to serve it")
        return report_live(site_url)

    # ...checked BEFORE the push, because the push is the irreversible half. A
    # missing gh after a successful push leaves the commit on the remote with
    # nothing deploying it, which is the exact state this tool exists to prevent.
    if shutil.which("gh") is None:
        print("gh is not installed, so the workflow cannot be dispatched -"
              " and a push alone does not deploy. Install the GitHub CLI, or"
              f" dispatch {WORKFLOW!r} from the Actions tab after pushing.",
              file=sys.stderr)
        return 1

    print(f"pushing {head[:8]}…")
    subprocess.run(["git", "push", "origin", BRANCH], check=True)

    # ...dispatched EXPLICITLY. It may now be redundant - automatic triggers
    # were enabled after this tool was written - but a dispatch is idempotent
    # here (the deploy is a full rebuild) and it cannot be quietly turned off
    # the way the trigger was.
    print(f"dispatching {WORKFLOW!r}…")
    subprocess.run(["gh", "workflow", "run", WORKFLOW, "-R", name, "--ref", BRANCH],
                   check=True)

    print("waiting for the site to serve it…")
    deadline = time.monotonic() + DEPLOY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        build = live_build(site_url)
        if build is not None and build["commit"] == head:
            print(f"live: {head[:8]}  built {build.get('builtAt')}")
            return 0
        time.sleep(POLL_SECONDS)

    print(f"{DEPLOY_TIMEOUT_SECONDS}s passed and {site_url} is still not serving"
          f" {head[:8]}.", file=sys.stderr)
    report_live(site_url)
    print(f"check the run: gh run list -R {name}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
