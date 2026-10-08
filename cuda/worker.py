"""The CUDA backend: LocalFold's native ports folding a page's job, speaking the bridge.

    python3 cuda/worker.py            # run from the repository root

Started by tools/colab_backend.py (`--cuda`) and fed one job per line on stdin; every line it prints on
stdout is one bridge event, `{"kind", "payload", "at"}` - `status`, `progress`, `frame`, `contacts`,
`scores` and `result` - which the broker numbers and the reader's page follows (web/colab-bridge.js).

🔴 THE PAGE'S OWN INPUTS, THE PAGE'S OWN WEIGHTS - AND NO JAVASCRIPT. The job is the reader's AlphaFold 3 JSON
(web/job-json.js writes it) and each port's native featuriser (cuda/featurise: af3-featurise, af2-featurise,
esmfold2-featurise) reads it as the page's reader does and builds the page's input byte for byte
(tools/check-native-featuriser.py holds them to the JavaScript); the template rows are resolved as the page resolves
them (cuda/featurise/resolve-templates); the alignment search is the page's MMseqs2 client, natively; the weights are
the published bundles the page folds with, read as published through each port's weight walk (cuda/featurise/af3_weights.h,
af2_weights.h). What this file adds is the plumbing between
them - nothing about a molecule is decided here.

🔴 WHAT IT REFUSES, IT SAYS. A sampler, a model or an input a native port does not have is a refusal
naming it (Refused), never a nearby setting run instead: Flow on rosettafold3 (the page's own rule), a
template on AlphaFold 2's template-free models.

Ports: cuda/af3 (all seven AF3-lineage models), cuda/af2 (all five of each, monomer and multimer),
cuda/esmfold2 (ESMFold2, 600M and 300M). Each must be built (cuda/colab_setup.sh); a bundle not on disk is fetched.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CUDA = os.path.join(REPO, "cuda")
# ESMFold2's scheduled sampler steps: the checkpoint's own, and the floor for a job with per-atom tokens
# (web/esmfold2-model.js, ESMFOLD2_COUNTS and ESMFOLD2_ATOMISED_STEPS - the page's numbers, mirrored)
ESMFOLD2_DEFAULT_STEPS, ESMFOLD2_ATOMISED_STEPS = 15, 64


def atomised_job(text):
    """Whether an AlphaFold 3 job carries per-atom tokens: a ligand, or a modified residue or base."""
    spec = json.loads(text)
    spec = spec[0] if isinstance(spec, list) else spec
    for entry in spec.get("sequences", []):
        for kind, body in entry.items():
            if kind == "ligand" or (isinstance(body, dict) and body.get("modifications")):
                return True
    return False
WORK = os.environ.get("LOCALFOLD_CUDA_WORK", "/tmp/localfold-cuda")
AF3_FAMILIES = ("af3", "openbind0", "opendde", "boltz2", "protenix2", "intellifold2", "rosettafold3", "chai1")
# ...whose dialect has no working flow sampler (noFlowSampler, shared/af3/dialect.js)
NO_FLOW_FAMILIES = ("rosettafold3", "chai1")
FEATURISE = os.path.join(CUDA, "featurise")
OUT = sys.stdout


def emit(kind, payload, raw=None):
    """raw: {placeholder string: JSON text} spliced in for the placeholder after serialising (a result's matrices,
    passed through as the binary wrote them - see collect)"""
    line = json.dumps({"kind": kind, "payload": payload, "at": int(time.time() * 1000)})
    for placeholder, text in (raw or {}).items():
        line = line.replace(json.dumps(placeholder), text, 1)
    OUT.write(line + "\n")
    OUT.flush()


def flat_matrix(text, key):
    """The [[...]] matrix under `key` in a confidences file's text, as the text of one flat JSON list - and the
    file's text with it replaced by null. The binaries write it as rows of two-decimal numbers, so cutting the
    brackets out is the same list a parse, a flatten and a round produced, without the n^2 Python floats (95 ms of a
    261-token job's 167 ms of result handling: the load, the flatten and round, the dump)."""
    at = text.find(f'"{key}": [[')
    if at < 0:
        return text, None
    start = at + len(key) + 4
    end = text.index("]]", start) + 2
    # (the rows' newlines out too: the protocol is a JSON object a line)
    flat = "[" + text[start + 1:end - 1].replace("[", "").replace("]", "").replace("\n", "") + "]"
    return text[:start] + "null" + text[end:], flat


class Refused(Exception):
    """A job this backend does not run, said as such rather than approximated."""


def card_use():
    """(used, total) MiB of device memory on the card, from nvidia-smi."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    used, total = (int(v) for v in out[0].split(","))
    return used, total


def device_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return out[0] if out else "no GPU"
    except (OSError, subprocess.SubprocessError):
        return "no GPU"


def die_with_parent():
    """Linux: a child gets SIGKILL when this worker goes, however it goes - Stop kills the worker (the
    broker's way of ending a fold), and a binary left running would hold the card."""
    import ctypes
    import signal
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)       # PR_SET_PDEATHSIG


def model_weights(model, log):
    """A model's weights on disk, by its --model name - what its binary reads (cuda/featurise/fetch-weights: fetch.h's one
    table), downloaded the first time piece by piece, each said, so a slow download (one revision came at 0.9 MB/s on
    a Colab T4) reads as a download and not a hang. AlphaFold 3's are Google DeepMind's, for academic non-commercial
    use: the page folds only once its model-terms dialog has been accepted, which is the acceptance passed on."""
    env = dict(os.environ)
    if model == "af3":
        accepted = {n.strip() for n in env.get("LOCALFOLD_ACCEPT_MODEL_TERMS", "").split(",") if n.strip()}
        env["LOCALFOLD_ACCEPT_MODEL_TERMS"] = ",".join(sorted(accepted | {"alphafold3"}))
    fetcher = subprocess.Popen([featuriser("fetch-weights"), model], cwd=REPO, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1, preexec_fn=die_with_parent, env=env)
    said, dirs = [], []
    for line in fetcher.stdout:
        said.append(line)
        found = re.search(r"\((\d+)/(\d+)\)", line)
        if found:
            emit("status", f"fetching the {model} weights · part {found.group(1)} of {found.group(2)}")
        named = re.match(re.escape(model) + r": (/.+)$", line.rstrip("\n"))
        if named:
            dirs.append(named.group(1))
    if fetcher.wait() != 0:
        raise RuntimeError(f"fetching {model} failed: {''.join(said[-4:])}")
    log.append("".join(said))
    return dirs


BUILD_MARKER, BUILD_LOG = "/tmp/localfold-cuda-build", "/tmp/localfold-cuda-build.log"


def building():
    """Whether cuda/build.sh is running (its marker names a live pid)."""
    try:
        os.kill(int(open(BUILD_MARKER).read().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def binary(port):
    """A port's binary - waited for while cuda/build.sh is still compiling it (the notebook starts the
    build beside the service, so the first fold can arrive before it is done), refused if nothing is."""
    path = os.path.join(CUDA, port, port)
    started = time.time()
    while not os.access(path, os.X_OK) and building():
        emit("status", f"compiling the CUDA ports for this card (once a runtime) · {time.time() - started:.0f} s")
        time.sleep(3)
    if not os.access(path, os.X_OK):
        said = open(BUILD_LOG).read()[-600:] if os.path.exists(BUILD_LOG) else ""
        raise Refused(f"cuda/{port}/{port} is not built on this runtime - run cuda/build.sh"
                      + (f" (its last build said: {said.strip()})" if said.strip() else ""))
    return path


def featuriser(name):
    """A native featuriser (cuda/featurise/<name>) - waited for while cuda/build.sh compiles it, refused if nothing
    is building it."""
    path = os.path.join(FEATURISE, name)
    started = time.time()
    while not os.access(path, os.X_OK) and building():
        emit("status", f"compiling the native featurisers (once a runtime) · {time.time() - started:.0f} s")
        time.sleep(1)
    if not os.access(path, os.X_OK):
        raise Refused(f"cuda/featurise/{name} is not built on this runtime - run cuda/build.sh")
    return path


def polymer_chains(job_json):
    """The job's polymer chains in fold order, one per copy - the page's `chains`."""
    spec = json.loads(job_json)
    spec = spec[0] if isinstance(spec, list) else spec
    out = []
    for entry in spec.get("sequences", []):
        for kind in ("protein", "dna", "rna"):
            if kind in entry:
                body = entry[kind]
                copies = len(body["id"]) if isinstance(body.get("id"), list) else 1
                out += [body["sequence"]] * copies
    return out


def pdb_atoms(pdb):
    atoms = []
    for line in pdb.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 66:
            atoms.append({"chain": line[21].strip(), "res": line[22:26].strip(), "b": float(line[60:66])})
    return atoms


def token_plddt(atoms, chain_ids, res_ids):
    """One pLDDT a token from the atoms' own (the B factor column): a residue that is one token takes its
    atoms' mean, a ligand or atomised residue - one token an atom - each atom's own (keyed on
    (chain, residue) in token order)."""
    groups = {}
    for atom in atoms:
        groups.setdefault((atom["chain"], atom["res"]), []).append(atom["b"])
    counts = {}
    for chain, res in zip(chain_ids, res_ids):
        counts[(chain, str(res))] = counts.get((chain, str(res)), 0) + 1
    seen, out = {}, []
    for chain, res in zip(chain_ids, res_ids):
        key = (chain, str(res))
        values = groups.get(key)
        if not values:
            raise RuntimeError(f"the structure has no atom for token {chain}{res}")
        index = seen.get(key, 0)
        seen[key] = index + 1
        out.append(values[index] if counts[key] == len(values) and counts[key] > 1 else sum(values) / len(values))
    return [round(v, 2) for v in out]


def emit_scores(meta, pae_path, passes):
    """An AF2 pass's confidences as the page's `scores` event: its pLDDT (per residue and mean), pTM, ipTM
    and PAE (a byte a pair, PAE / 0.125, base64)."""
    import base64
    data = open(pae_path, "rb").read()
    emit("scores", {**meta, "n": int(round(len(data) ** 0.5)), "paeU8": base64.b64encode(data).decode(),
                    "paeScale": 0.125, "passes": passes})


def emit_contacts(path, index, passes):
    """A pass's contact map as the page's `contacts` event: one byte a pair (probability * 255), base64 -
    68 KB for 261 tokens where float32 JSON would be ten times that."""
    import base64
    data = open(path, "rb").read()
    n = int(round(len(data) ** 0.5))
    emit("contacts", {"n": n, "pass": index, "passes": passes, "u8": base64.b64encode(data).decode()})


class Server:
    """One model kept on the device between folds: the port's own --serve mode (common.cuh's serveJobs: a job
    is DIR/<id>.job - the input's directory, then a flag a line - its output <id>.log, its status <id>.done).
    A cold fold is mostly start-up - the CUDA context, the weight upload, the kernels' first launches: 6MRR
    is 0.92 s cold of which the fold is 0.18 on an A100 for AF3, ~0.75 s of start-up for AF2 and ESMFold2 -
    and this pays it once a model. `key` names the model (a port, its family, AF2's model number)."""

    def __init__(self, key, command):
        self.key = key
        self.dir = os.path.join(WORK + "-serve", "-".join(str(k) for k in key))
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir)
        self.log = open(os.path.join(self.dir, "server.log"), "w")
        self.proc = subprocess.Popen([*command, f"--serve={self.dir}"], cwd=REPO, stdout=self.log,
                                     stderr=subprocess.STDOUT, preexec_fn=die_with_parent)
        # 🔴 NOT WAITED FOR: the worker starts a model's server BEFORE it featurises the job, so the CUDA
        # context, the weights' upload and AF2's and ESMFold2's warm-up fold (--warm: every kernel, weight copy
        # and cuBLAS plan loaded) run beside the exporter or the MSA search; a job dropped meanwhile is taken
        # once the server is up, and a server that dies first says so through fold()'s wait
        self.count = 0

    def fold(self, inputs, flags, on_file=None):
        """...and with `on_file`, each streamed result handed over as it lands (--frames: written by a thread
        of the binary off the copy engine, so the fold does not wait for it - +0.1-0.4% of a fold, measured
        interleaved; the structure is byte-identical either way): on_file(name, path) in name order."""
        self.count += 1
        base = os.path.join(self.dir, f"{self.count:06d}")
        frames = base + ".frames"
        if on_file is not None:
            os.makedirs(frames, exist_ok=True)
            flags = [*flags, f"--frames={frames}"]
        with open(base + ".tmp", "w") as handle:
            handle.write("\n".join([inputs, *flags]) + "\n")
        os.rename(base + ".tmp", base + ".job")
        seen = set()

        def collect():
            if on_file is None:
                return
            for name in sorted(os.listdir(frames)):
                if name not in seen and not name.endswith(".tmp"):
                    seen.add(name)
                    on_file(name, os.path.join(frames, name))
        while not os.path.exists(base + ".done"):
            if self.proc.poll() is not None:
                raise RuntimeError(f"the {self.key[0]} server exited: " + open(self.log.name).read()[-400:])
            collect()
            time.sleep(0.005)
        collect()
        shutil.rmtree(frames, ignore_errors=True)
        said = open(base + ".log").read()
        if int(open(base + ".done").read().strip() or 1) != 0:
            thrown = [line for line in said.strip().splitlines() if line.strip()]
            raise RuntimeError("the fold failed: " + "\n".join(thrown[-4:]))
        return said

    def close(self):
        if self.proc.poll() is None:
            with open(os.path.join(self.dir, "quit.tmp"), "w") as handle:
                handle.write("quit\n")
            os.rename(os.path.join(self.dir, "quit.tmp"), os.path.join(self.dir, "~quit.job"))
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def featurise(cmd, what, log, on_line=None):
    """A native featuriser's run (cuda/featurise): its output kept, and each line handed to `on_line` as it comes (the
    MMseqs2 search says `search: queued` / `search: running` while it waits); a refusal - the page's own sentence,
    which every featuriser prints as `Error: <sentence>` - raised as one, anything else as the step that failed."""
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            preexec_fn=die_with_parent)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        if on_line is not None:
            on_line(line.rstrip("\n"))
    said = "".join(lines)
    log.append(f"$ {' '.join(cmd)}\n{said}")
    if proc.wait() != 0:
        first = re.search(r"^Error: (.+)$", said, re.M)
        if first:
            raise Refused(first.group(1))
        raise RuntimeError(f"{what} failed: " + "\n".join(said.strip().splitlines()[-4:]))
    return said


def streaming(job):
    """Whether a fold streams its intermediate results: the reader's own choice, the page's Live preview
    (web/colab-bridge.js), sent as `frames` - on unless it says false."""
    return job.get("frames", True) is not False


class Worker:
    # 🔴 SEVERAL MODELS MAY STAY ON THE CARD, WHILE THERE IS ROOM. A change of model paid its server's start
    # (0.3-0.56 s on an A100 - the context, the weights) and a first fold in a fresh process; idle, the three
    # ports hold 0.8-2.6 GB each. So the servers last used stay up while the card is at most half full after
    # a fold, the least recent going first, and every idle one is stopped before a job past LARGE residues -
    # a big fold's own buffers are what the card is for.
    LARGE = 400

    def __init__(self):
        self.device = device_name()
        self.servers = {}                # key -> Server, least recently used first

    def evict(self, keep, everything=False):
        """Idle servers stopped, least recently used first: all of them (`everything`), or until the card is
        at most half full. `keep` (the model about to fold, or just folded) is never one."""
        for key in [k for k in self.servers if k != keep]:
            if not everything:
                used, total = card_use()
                if used <= total // 2:
                    return
            self.servers.pop(key).close()

    def server_for(self, key, command, residues):
        """The model's server - kept, or started - moved to the most recently used end."""
        self.evict(key, everything=residues > self.LARGE)
        server = self.servers.pop(key, None)
        if server is not None and server.proc.poll() is not None:
            server = None
        if server is None:
            emit("status", f"{key[1]} on CUDA ({self.device}) · loading the weights onto the card")
            server = Server(key, command)
        self.servers[key] = server
        return server

    def fold(self, job):
        controls = job.get("controls", {})
        family = job.get("family") or controls.get("model-family", "af3")
        # 🔴 AF2's FAMILY CARRIES ITS MODEL NUMBER (the page resolves the number box into `monomer-2`,
        # `multimer-5`): models 2-5 are published as deltas on model 1 and read as the page reads them
        # (common.cuh's loadBundle --delta, bit-exact against shared/bundles/delta-tensor-store.js)
        base, _, number = family.partition("-")
        af2_model = 1
        if base in ("monomer", "multimer"):
            af2_model = int(number) if number.isdigit() else int(controls.get("af2Model") or 1)
            if not 1 <= af2_model <= 5:
                raise Refused(f"AlphaFold 2 has models 1 to 5, not {af2_model}")
            family = base
        if family in AF3_FAMILIES:
            port = "af3"
        elif family in ("monomer", "multimer"):
            port = "af2"
        elif family in ("ef2-fast-600m", "ef2-fast-300m"):
            port = "esmfold2"
        else:
            raise Refused(f"the CUDA backend has no port of {family!r} (it folds the AF3 lineage, AlphaFold 2"
                          " and ESMFold2)")
        sampler = controls.get("af3-mode", "diffusion")
        if port == "af3" and sampler not in ("diffusion", "flow"):
            raise Refused(f"the CUDA backend does not know the sampler {sampler!r}")
        if port == "af3" and sampler == "flow" and family in NO_FLOW_FAMILIES:
            # ...the page's own rule: rf3's walk collapses the backbone while pLDDT reads as if nothing were
            # wrong, and chai-1 samples with its own second-order step (noFlowSampler, shared/af3/dialect.js)
            raise Refused(f"{family} has no working flow sampler - set the sampler to Diffusion")
        emit("status", f"{family} on CUDA ({self.device}) · reading the job")
        emit("progress", 0.02)
        started = time.time()
        shutil.rmtree(WORK, ignore_errors=True)
        inputs = os.path.join(WORK, "in")
        os.makedirs(WORK)
        log = []
        job_path = os.path.join(WORK, "job.json")
        with open(job_path, "w") as handle:
            handle.write(job["job"])
        request_path = os.path.join(WORK, "request.json")
        with open(request_path, "w") as handle:
            json.dump({"entities": job.get("entities", [])}, handle)

        residues = sum(len(chain) for chain in polymer_chains(job["job"]))
        # the alignment, as the page's MSA row asked for it
        mode = "none" if port == "esmfold2" else controls.get("msa-mode", "none")
        flags, a3m = [], None
        if mode == "search":
            flags.append("--search")
        elif mode in ("paste", "upload"):
            msas = {"merged": controls.get("msa-text") or ""} if mode == "paste" else (job.get("msas") or {})
            if msas.get("merged"):
                path = os.path.join(WORK, "msa.a3m")
                with open(path, "w") as handle:
                    handle.write(msas["merged"])
                flags.append(f"--a3m={path}")
                a3m = msas["merged"]
            elif any(msas.get("unpaired") or []):
                paths = {"unpaired": [], "paired": []}       # (merged by each exporter as the page merges them)
                for side in paths:
                    for index, text in enumerate(msas.get(side) or []):
                        path = os.path.join(WORK, f"msa-{side}-{index}.a3m")
                        with open(path, "w") as handle:
                            handle.write(text or "")
                        paths[side].append(path)
                flags.append("--a3m=" + ",".join(paths["unpaired"]))
                if any(msas.get("paired") or []):
                    flags.append("--paired-a3m=" + ",".join(paths["paired"]))
            else:
                raise Refused(f"the MSA mode is {mode!r} but no alignment came with the job")
        elif mode != "none":
            raise Refused(f"the CUDA backend does not know the MSA mode {mode!r}")

        # the templates, resolved by the page's own code - where a row asks for one (templateKind: a row's
        # template with no kind, or "none", is no template; a Node start is 0.13 s of a 0.5 s warm fold)
        templates = []
        if any((entity.get("template") or {}).get("kind") not in (None, "none") for entity in job.get("entities", [])):
            templates = json.loads(featurise([featuriser("resolve-templates"), request_path, os.path.join(WORK, "templates")],
                                             "resolving the templates", log).strip().splitlines()[-1] or "[]")
        if templates and port == "esmfold2":
            raise Refused("ESMFold2 takes no template")
        searched = [t["chain"] for t in templates if t["kind"] == "search"]
        if searched and mode != "search":
            raise Refused("a template from the MSA search needs an MSA search: set the MSA to search, or name a"
                          " structure instead")
        named = [t for t in templates if t["kind"] != "search"]
        if named:
            emit("status", f"{family} on CUDA · templates {', '.join(t['source'] for t in named)}")
        if port == "af3":
            if named:
                flags.append("--template=" + ",".join(f"{t['file']}:{t['chainId'] or ''}@{t['chain']}" for t in named))
        elif named:
            flags.append("--template=" + "+".join(f"{t['file']}:{t['chainId'] or ''}@{t['chain']}" for t in named))
        if searched:
            flags.append("--template-search-chains=" + ",".join(str(c) for c in searched))

        recycles = controls.get("recycles")
        seed = int(controls.get("random-seed") or 1)
        depth = str(controls.get("max-msa") or "512:1024").split(":")
        requested = int(depth[0]) if depth[0].isdigit() else 512
        extra = int(depth[1]) if len(depth) > 1 and depth[1].isdigit() else 1024
        out_pdb = os.path.join(WORK, "fold.pdb")
        emit("status", f"{family} on CUDA ({self.device}) · featurising"
             + (" and searching the ColabFold MMseqs2 server" if mode == "search" else ""))
        def live(line):          # the featuriser's own word while it works: the MMseqs2 search waits on a queue
            if line.startswith("search: "):
                emit("status", f"{family} on CUDA ({self.device}) · MMseqs2 search · {line[8:]}")
        # each port's resident server (Server: the model's weights stay on the card between folds) and this
        # job's flags for it
        if port == "af3":
            # af3-any-model's own int8 blob, every family (AlphaFold 3's under DeepMind's academic terms)
            weights = model_weights(family, log)
            bundle, dialect = weights[0], f"--family={family}"
            key = ("af3", family)
            # chai-1's token features are ESM2 3B's, computed in the fold (cuda/plm/esm2.cuh)
            esm = [f"--esm-bundle={weights[1]}"] if len(weights) > 1 else []
            server = self.server_for(key, [binary("af3"), "-", f"--bundle={bundle}",
                                           f"--family={family}", "--fold", "--fast",
                                           *esm], residues)
            featurise([featuriser("af3-featurise"), inputs, "--no-weights", dialect, f"--job={job_path}",
                       f"--max-msa={requested}", *flags], "featurising", log, live)
            steps = int((job.get("schedule") or {}).get("steps") or controls.get("af3-count") or 0)
            fold = [f"--out={out_pdb}"]
            if sampler == "flow":
                fold.append("--flow")                    # (the page's Flow: cuda/af3/src/sampler.cuh)
            elif (job.get("schedule") or {}).get("sigmaMax"):
                # the page's short schedule, resolved there (diffusionScheduleFor), started where it starts
                fold.append(f"--sigma-max={float(job['schedule']['sigmaMax'])}")
            if steps:
                # ...the page's floor: a modified residue's atoms stay compressed below sixteen steps (app.js)
                spec = json.loads(job["job"])
                spec = spec[0] if isinstance(spec, list) else spec
                modified = any(body.get("modifications") for entry in spec.get("sequences", [])
                               for body in entry.values() if isinstance(body, dict))
                steps = max(steps, 16) if modified else steps
                fold.append(f"--steps={steps}")
            if recycles not in (None, ""):
                fold.append(f"--recycles={int(recycles)}")
            total = steps or 200                         # (af3's default)
        elif port == "af2":
            if templates and family == "monomer" and af2_model > 2:
                raise Refused(f"AlphaFold 2's model {af2_model} has no template embedder (models 3, 4 and 5 are"
                              " template-free) - pick model 1 or 2, or drop the template")
            model = f"model_{af2_model}_ptm" if family == "monomer" else f"model_{af2_model}_multimer_v3"
            weights = model_weights(model, log)       # (models 2-5: model 1's bundle and their delta on it)
            bundle, delta = weights[0], (weights[1] if len(weights) > 1 else None)
            key = ("af2", family, af2_model)
            server = self.server_for(key, [binary("af2"), "-", f"--bundle={bundle}",
                                           "--fast",
                                           *([f"--delta={delta}"] if delta else []), "--warm=64,8,8,0"], residues)
            export = [featuriser("af2-featurise"), inputs, f"--bundle={bundle}", f"--job={job_path}",
                      f"--max-msa={508 if requested == 512 else requested}", f"--max-extra={extra}", f"--seed={seed}", *flags]
            if recycles not in (None, ""):
                export.append(f"--recycles={int(recycles)}")
            featurise(export, "featurising", log, live)
            fold = [f"--out={out_pdb}", f"--tolerance={float(controls.get('tolerance') or 0)}"]   # (the page's early stop)
            total = 0
        else:
            small = family == "ef2-fast-300m"       # (the same port: it reads its widths off the bundle)
            trunk, tower = model_weights("esmfold2-fast-300m" if small else "esmfold2-fast-600m", log)
            key = ("esmfold2", family)
            server = self.server_for(key, [binary("esmfold2"), "-", f"--fold-bundle={trunk}", f"--esmc-bundle={tower}", "--fast",
                                           "--warm=96,800"], residues)
            featurise([featuriser("esmfold2-featurise"), inputs, f"--job={job_path}"], "featurising", log, live)
            fold = [f"--out={out_pdb}", f"--seed={seed}"]
            # the page's step count (scheduled, as the binary's --steps takes it: 15 runs 11), and the page's
            # floor for per-atom tokens - a ligand or a modified residue is torn at 11 steps and whole at 45
            # (web/esmfold2-model.js, ESMFOLD2_ATOMISED_STEPS) - applied here too, for a job no page sent
            steps = int((job.get("schedule") or {}).get("steps") or 0)
            if atomised_job(job["job"]):
                steps = max(steps or ESMFOLD2_DEFAULT_STEPS, ESMFOLD2_ATOMISED_STEPS)
            if steps:
                fold.append(f"--steps={steps}")
            total = 0
        emit("progress", 0.15)
        emit("status", f"{family} on CUDA ({self.device}) · folding")
        on_file = None
        if streaming(job):
            # 🔴 WHAT EACH PORT STREAMS, AS WebGPU AND JAX STREAM IT - unless the page's Live preview is off:
            # the AF3 lineage's trunk contacts and sampler frames, AF2's passes (structure, pLDDT, pTM/ipTM,
            # PAE, contacts), ESMFold2's trunk contacts and sampler frames - as the binary's tap writes them
            def on_file(name, path):
                tag = re.search(r"(\d+)-of-(\d+)", name)
                if name.startswith("pass-") and name.endswith(".pdb"):
                    index, passes = int(tag.group(1)), int(tag.group(2))
                    emit("frame", open(path).read())
                    emit_scores(json.load(open(path[:-4] + ".json")), os.path.join(os.path.dirname(path), f"pae-{tag.group(0)}.u8"), passes)
                    emit("progress", 0.15 + 0.85 * (index + 1) / passes)
                    emit("status", f"{family} on CUDA ({self.device}) · pass {index + 1}/{passes}")
                elif name.startswith("contacts-"):
                    index, passes = int(tag.group(1)), int(tag.group(2))
                    emit_contacts(path, index, passes)
                    if port != "af2":
                        emit("progress", 0.15 + 0.15 * (index + 1) / passes)
                        emit("status", f"{family} on CUDA ({self.device}) · trunk pass {index + 1}/{passes}")
                elif name.startswith("frame-"):
                    # frame-SSSS.pdb (af3, its step count the job's) or frame-SSSS-NNNN.pdb (esmfold2)
                    step = int(name[6:10])
                    steps = int(name[11:15]) if name[10] == "-" else total
                    emit("frame", open(path).read())     # (superposed onto the first by the binary's writer)
                    if steps:
                        emit("progress", 0.3 + 0.7 * step / steps)
                        emit("status", f"{family} on CUDA ({self.device}) · diffusion {step}/{steps}")
        said = server.fold(inputs, fold, on_file)
        log.append(said)
        self.evict(key)
        if not any("pLDDT" in line for line in said.splitlines()):
            raise RuntimeError("the fold printed no confidence line")
        emit("progress", 1.0)
        result = self.collect(out_pdb, family, port, job, time.time() - started)
        result["a3m"] = a3m
        if mode == "search" and os.path.exists(os.path.join(inputs, "search.a3m")):
            result["a3m"] = open(os.path.join(inputs, "search.a3m")).read()
        result["templates"] = [{"text": open(t["file"]).read(), "chainId": t["chainId"], "source": t["source"],
                                "chain": t["chain"]} for t in named]
        export_said = "\n".join(log)
        for found in re.findall(r"template[^\n]*?(\d+)/(\d+) residues", export_said):
            result["status"] += f" · template {found[0]}/{found[1]}"
        converged = re.search(r"converged at ([0-9.]+) A after (\d+) passes", said)
        if converged:
            # ...worded as the WebGPU fold words it
            result["status"] += f" · converged at {converged.group(1)} Å after {converged.group(2)} passes"
        return result

    def collect(self, pdb_path, family, port, job, seconds):
        stem = pdb_path[:-4]
        pdb = open(pdb_path).read()
        text = open(f"{stem}_confidences.json").read()
        text, pae = flat_matrix(text, "pae")
        text, contacts = flat_matrix(text, "contact_probs")
        confidences = json.loads(text)
        summary = json.load(open(f"{stem}_summary_confidences.json"))
        chain_ids = confidences["token_chain_ids"]
        res_ids = confidences["token_res_ids"]
        plddt = confidences.get("token_plddts") or token_plddt(pdb_atoms(pdb), chain_ids, res_ids)

        raw = {}

        def flat(text, name):                      # (a placeholder the emitter replaces with the matrix's text)
            if text is None:
                return None
            raw["\u0000" + name] = text
            return "\u0000" + name
        confidence = {
            "plddt": [round(float(v), 2) for v in plddt],
            "meanPlddt": round(sum(plddt) / max(1, len(plddt)), 2),
            "ptm": summary.get("ptm"),
            "predictedAlignedError": flat(pae, "pae"),
            "contactProbs": flat(contacts, "contacts"),
        }
        if summary.get("iptm") is not None:
            confidence["iptm"] = summary["iptm"]
        mean = confidence["meanPlddt"]
        return {
            "cuda": True, "model": f"cuda {port}", "family": family,
            "pdb": pdb, "confidence": confidence,
            "tokens": {"chainIds": chain_ids, "resIds": res_ids},
            "chains": polymer_chains(job["job"]), "msas": {},
            "atoms": len(pdb_atoms(pdb)),
            "status": f"{family} on CUDA ({self.device}) · done in {seconds:.1f} s · pLDDT {mean:.1f}",
            "_raw": raw,
        }


def main():
    emit("cuda-ready", {"at": int(time.time() * 1000)})
    worker = Worker()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            result = worker.fold(json.loads(line))
            emit("result", result, result.pop("_raw", None))
        except Exception as cause:                                # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            said = str(cause) if isinstance(cause, Refused) else f"{type(cause).__name__}: {cause}"
            emit("result", {"error": said})


if __name__ == "__main__":
    main()
