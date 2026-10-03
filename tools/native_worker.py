"""The CUDA backend: LocalFold's native ports folding a page's job, speaking the bridge.

    python3 tools/native_worker.py            # run from the repository root

Started by tools/colab_backend.py (`--native`) and fed one job per line on stdin, exactly as
tools/jax_worker.py is; every line it prints on stdout is one bridge event, `{"kind", "payload", "at"}` -
`status`, `progress` and `result` - so the reader's page follows a CUDA fold with the code that follows a
WebGPU or a JAX one.

🔴 THE PAGE'S OWN INPUTS, THE PAGE'S OWN WEIGHTS. The job is the reader's AlphaFold 3 JSON (web/job-json.js
writes it) and each port's exporter reads it with the page's reader; the template rows are resolved by the
page's expandEntities and fetchStructure (native/resolve_templates.mjs); the alignment search is the page's
MMseqs2 client; the weights are the published bundles the page folds with, read through native/*/maps.
What this file adds is the plumbing between them - nothing about a molecule is decided here.

🔴 WHAT IT REFUSES, IT SAYS. A sampler, a model or an input a native port does not have is a refusal
naming it (Refused), never a nearby setting run instead: Flow (native AF3 runs diffusion), AlphaFold 2's
models 2-5 (the page publishes them as deltas, which the native loader does not read), ESMFold2 300M,
and AF2's early stop, which is reported rather than applied.

Ports: native/af3 (all seven AF3-lineage models), native/af2 (model_1_ptm, model_1_multimer_v3),
native/ef2 (ESMFold2 600M). Each must be built (native/colab_setup.sh); a bundle not on disk is fetched.
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
NATIVE = os.path.join(REPO, "native")
WORK = os.environ.get("LOCALFOLD_NATIVE_WORK", "/tmp/localfold-native")
AF3_FAMILIES = ("af3", "openbind0", "opendde", "boltz2", "protenix2", "intellifold2", "rosettafold3")
NODE = ["node", "--js-float16array", "--max-old-space-size=24000"]
OUT = sys.stdout


def emit(kind, payload):
    OUT.write(json.dumps({"kind": kind, "payload": payload, "at": int(time.time() * 1000)}) + "\n")
    OUT.flush()


class Refused(Exception):
    """A job this backend does not run, said as such rather than approximated."""


def device_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return out[0] if out else "no GPU"
    except (OSError, subprocess.SubprocessError):
        return "no GPU"


def run(cmd, what, log, cwd=REPO):
    """A step, its output kept; a failure says the step and its last lines."""
    done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    log.append(f"$ {' '.join(cmd)}\n{done.stdout}{done.stderr}")
    if done.returncode != 0:
        # 🔴 A NODE STEP THAT THREW SAID WHY IN ONE SENTENCE, written for a person (the page's own readers
        # and featurisers - "AlphaFold 2 folds protein chains only"): that sentence is the answer, and
        # the stack under it is not
        thrown = re.findall(r"^(?:\w*Error): (.+)$", done.stderr or "", re.M)
        if thrown and cmd[0] == "node":
            raise Refused(thrown[-1])
        tail = "\n".join((done.stderr or done.stdout).strip().splitlines()[-4:])
        raise RuntimeError(f"{what} failed: {tail}")
    return done.stdout


def ensure_bundle(family, directory, log):
    if not os.path.exists(os.path.join(REPO, directory, "manifest.json")):
        emit("status", f"fetching the {family} weights")
        run([sys.executable, os.path.join(NATIVE, "fetch_bundles.py"), family], f"fetching {family}", log)
    return os.path.join(REPO, directory)


BUILD_MARKER, BUILD_LOG = "/tmp/localfold-native-build", "/tmp/localfold-native-build.log"


def building():
    """Whether native/build.sh is running (its marker names a live pid)."""
    try:
        os.kill(int(open(BUILD_MARKER).read().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def binary(port):
    """A port's binary - waited for while native/build.sh is still compiling it (the notebook starts the
    build beside the service, so the first fold can arrive before it is done), refused if nothing is."""
    path = os.path.join(NATIVE, port, port)
    started = time.time()
    while not os.access(path, os.X_OK) and building():
        emit("status", f"compiling the CUDA ports for this card (once a runtime) · {time.time() - started:.0f} s")
        time.sleep(3)
    if not os.access(path, os.X_OK):
        said = open(BUILD_LOG).read()[-600:] if os.path.exists(BUILD_LOG) else ""
        raise Refused(f"native/{port}/{port} is not built on this runtime - run native/build.sh"
                      + (f" (its last build said: {said.strip()})" if said.strip() else ""))
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
    atoms' mean, a ligand or atomised residue - one token an atom - each atom's own (tools/jax_worker.py's
    rule, keyed on (chain, residue) in token order)."""
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


class Worker:
    def __init__(self):
        self.device = device_name()

    def fold(self, job):
        controls = job.get("controls", {})
        family = job.get("family") or controls.get("model-family", "af3")
        base, _, number = family.partition("-")
        if base in ("monomer", "multimer") and number.isdigit():
            if number != "1":
                raise Refused(f"AlphaFold 2's model {number} is published as a delta on model 1, which the CUDA"
                              " backend does not read yet - pick model 1, or fold on WebGPU or JAX")
            family = base
        if base in ("monomer", "multimer") and controls.get("af2Model") not in (None, "", "1", 1):
            raise Refused(f"AlphaFold 2's model {controls.get('af2Model')} is published as a delta on model 1,"
                          " which the CUDA backend does not read yet - pick model 1, or fold on WebGPU or JAX")
        if family in AF3_FAMILIES:
            port = "af3"
        elif family in ("monomer", "multimer"):
            port = "af2"
        elif family == "ef2-fast-600m":
            port = "ef2"
        else:
            raise Refused(f"the CUDA backend has no port of {family!r} (it folds the AF3 lineage, AlphaFold 2"
                          " model 1 and ESMFold2 600M)")
        sampler = controls.get("af3-mode", "diffusion")
        if port == "af3" and sampler != "diffusion":
            raise Refused(f"the CUDA backend's AF3 samples by diffusion only - set the sampler to Diffusion"
                          f" (it was {sampler})")
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

        # the alignment, as the page's MSA row asked for it
        mode = "none" if port == "ef2" else controls.get("msa-mode", "none")
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
                if port != "af3":
                    raise Refused("an alignment per chain folds on WebGPU or JAX; the CUDA AF2 takes one merged"
                                  " alignment")
                paths = {"unpaired": [], "paired": []}
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

        # the templates, resolved by the page's own code
        templates = json.loads(run(["node", os.path.join(NATIVE, "resolve_templates.mjs"), request_path,
                                    os.path.join(WORK, "templates")], "resolving the templates", log)
                               .strip().splitlines()[-1] or "[]")
        if templates and port == "ef2":
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
        if port == "af3":
            bundle = ensure_bundle(family, f"model-{family}-int5", log)
            export = [*NODE, os.path.join(NATIVE, "af3", "export-model.mjs"), inputs, "--no-weights",
                      f"--bundle={bundle}/manifest.json", f"--job={job_path}", f"--max-msa={requested}", *flags]
            run(export, "featurising", log, cwd=os.path.join(NATIVE, "af3"))
            steps = int(controls.get("af3-count") or 0)
            fold = [binary("af3"), inputs, f"--bundle={bundle}", f"--map={os.path.join(NATIVE, 'af3', 'maps', family + '.map')}",
                    "--fold", "--fast", f"--out={out_pdb}"]
            if steps:
                # ...the page's floor: a modified residue's atoms stay compressed below sixteen steps (app.js)
                spec = json.loads(job["job"])
                spec = spec[0] if isinstance(spec, list) else spec
                modified = any(body.get("modifications") for entry in spec.get("sequences", [])
                               for body in entry.values() if isinstance(body, dict))
                fold.append(f"--steps={max(steps, 16) if modified else steps}")
            if recycles not in (None, ""):
                fold.append(f"--recycles={int(recycles)}")
        elif port == "af2":
            bundle = ensure_bundle(family, "model" if family == "monomer" else "model-multimer", log)
            export = [*NODE, os.path.join(NATIVE, "af2", "export_input.mjs"), inputs, f"--bundle={bundle}",
                      f"--job={job_path}", f"--max-msa={508 if requested == 512 else requested}",
                      f"--max-extra={extra}", f"--seed={seed}", *flags]
            if recycles not in (None, ""):
                export.append(f"--recycles={int(recycles)}")
            run(export, "featurising", log)
            model = "model_1_ptm" if family == "monomer" else "model_1_multimer_v3"
            fold = [binary("af2"), inputs, f"--bundle={bundle}",
                    f"--map={os.path.join(NATIVE, 'af2', 'maps', model + '.map')}", "--fast", f"--out={out_pdb}"]
        else:
            trunk = ensure_bundle("ef2-fast-600m", "model-esmfold2-int5", log)
            tower = ensure_bundle("esmc", "model-esmc-600m-int3", log)
            run([*NODE, os.path.join(NATIVE, "ef2", "export_input.mjs"), inputs, f"--job={job_path}"], "featurising", log)
            fold = [binary("ef2"), inputs, f"--fold-bundle={trunk}", f"--esmc-bundle={tower}", "--fast",
                    f"--seed={seed}", f"--out={out_pdb}"]
        emit("progress", 0.15)
        emit("status", f"{family} on CUDA ({self.device}) · folding")
        said = run(fold, "the fold", log)
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
        if port == "af2" and float(controls.get("tolerance") or 0) > 0:
            # 🔴 SAID, NOT APPLIED: native AF2 runs every pass, where the page stops on a settled structure
            result["status"] += " · every pass run (the CUDA AF2 has no early stop)"
        return result

    def collect(self, pdb_path, family, port, job, seconds):
        stem = pdb_path[:-4]
        pdb = open(pdb_path).read()
        confidences = json.load(open(f"{stem}_confidences.json"))
        summary = json.load(open(f"{stem}_summary_confidences.json"))
        chain_ids = confidences["token_chain_ids"]
        res_ids = confidences["token_res_ids"]
        plddt = confidences.get("token_plddts") or token_plddt(pdb_atoms(pdb), chain_ids, res_ids)

        def flat(matrix):
            return None if matrix is None else [round(float(v), 2) for row in matrix for v in row]
        confidence = {
            "plddt": [round(float(v), 2) for v in plddt],
            "meanPlddt": round(sum(plddt) / max(1, len(plddt)), 2),
            "ptm": summary.get("ptm"),
            "predictedAlignedError": flat(confidences.get("pae")),
            "contactProbs": flat(confidences.get("contact_probs")),
        }
        if summary.get("iptm") is not None:
            confidence["iptm"] = summary["iptm"]
        mean = confidence["meanPlddt"]
        return {
            "native": True, "model": f"native {port}", "family": family,
            "pdb": pdb, "confidence": confidence,
            "tokens": {"chainIds": chain_ids, "resIds": res_ids},
            "chains": polymer_chains(job["job"]), "msas": {},
            "atoms": len(pdb_atoms(pdb)),
            "status": f"{family} on CUDA ({self.device}) · done in {seconds:.1f} s · pLDDT {mean:.1f}",
        }


def main():
    emit("native-ready", {"at": int(time.time() * 1000)})
    worker = Worker()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            emit("result", worker.fold(json.loads(line)))
        except Exception as cause:                                # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            said = str(cause) if isinstance(cause, Refused) else f"{type(cause).__name__}: {cause}"
            emit("result", {"error": said})


if __name__ == "__main__":
    main()
