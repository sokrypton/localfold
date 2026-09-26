"""The JAX backend: af3-any-model folding a LocalFold job, speaking the bridge.

    python3 tools/jax_worker.py            # run from the ColabFold2 install dir

Started by tools/colab_backend.py (`--jax-dir`) and fed one job per line on
stdin; every line it prints on stdout is one bridge event,
`{"kind", "payload", "at"}`, the same kinds web/colab-bridge.js pushes from the
WebGPU runtime page - `status`, `progress`, `frame` (PDB text) and `result` -
so the reader's page follows a JAX fold with the code it follows a WebGPU one.

🔴 A SECOND IMPLEMENTATION ON PURPOSE, AND IT IS THE REFERENCE. The WebGPU fold
is this repository's port; this is sokrypton/alphafold3 (af3-any-model), the
implementation that port is checked against. It is here for what a browser
cannot do - a TPU, a complex past the browser's memory - not as a second
answer to the same question.

🔴 LONG-LIVED, BECAUSE A MODEL IS MINUTES OF COMPILE. One process holds each
model's weights and JAX's compile cache for the session; a fold is a line in,
a stream of lines out. Stopping a fold kills this process (JAX cannot be
interrupted mid-computation), and the broker starts a new one for the next.

The job is the reader's own AlphaFold 3 JSON (web/job-json.js writes it), so
the entity conversion is not re-implemented here either.
"""
import glob
import json
import os
import shutil
import sys
import time
import traceback

# The page's model row -> af3-any-model's --model.
MODELS = {
    "af3": "alphafold3", "openbind0": "openbind0", "opendde": "opendde",
    "boltz2": "boltz2", "protenix2": "protenix2", "intellifold2": "intellifold2",
    "rosettafold3": "rosettafold3", "monomer": "af2_ptm", "multimer": "af2_multimer",
    "ef2-fast-600m": "esmfold2_lm600m",
}
CACHE_DIR = os.environ.get("LOCALFOLD_JAX_CACHE", "/tmp/af3_cache")
# run_alphafold.py and live_frames.py are what the ColabFold2 install cell
# leaves in its working directory, which is where this is started.
sys.path.insert(0, os.getcwd())
# 🔴 EVENTS HAVE STDOUT TO THEMSELVES. The library prints its own progress, and
# a line of it among the events is a stream the broker cannot parse - so events
# go to a private copy of fd 1 and fd 1 itself (Python's prints and native
# code's alike) is pointed at stderr.
OUT = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def emit(kind, payload):
    OUT.write(json.dumps({"kind": kind, "payload": payload,
                          "at": int(time.time() * 1000)}) + "\n")
    OUT.flush()


def cif_to_pdb(cif):
    """The atom_site loop of one mmCIF model as PDB ATOM/HETATM records.

    The page ingests PDB (py2Dmol and every download path read it), and the
    B-factor column carries the atom's pLDDT, which is what the viewer colours
    by. Only the fields a PDB record has are kept.
    """
    lines = cif.splitlines()
    fields, rows, at = [], [], 0
    while at < len(lines):
        if lines[at].startswith("_atom_site."):
            while at < len(lines) and lines[at].startswith("_atom_site."):
                fields.append(lines[at].split(".", 1)[1].strip())
                at += 1
            while at < len(lines) and lines[at].strip() and not lines[at].startswith(("#", "_", "loop_")):
                rows.append(lines[at].split())
                at += 1
            break
        at += 1
    col = {name: index for index, name in enumerate(fields)}
    get = lambda row, *names: next((row[col[n]] for n in names if n in col), "")
    out = []
    for serial, row in enumerate(rows, start=1):
        name = get(row, "label_atom_id", "auth_atom_id").strip('"')
        element = get(row, "type_symbol")
        padded = name if len(name) == 4 or len(element) == 2 else f" {name}"
        out.append("%-6s%5d %-4s %3s %1s%4s    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s" % (
            get(row, "group_PDB") or "ATOM", serial % 100000, padded[:4],
            get(row, "label_comp_id", "auth_comp_id")[:3],
            get(row, "auth_asym_id", "label_asym_id")[:1],
            get(row, "auth_seq_id", "label_seq_id"),
            float(get(row, "Cartn_x")), float(get(row, "Cartn_y")), float(get(row, "Cartn_z")),
            1.0, float(get(row, "B_iso_or_equiv") or 0.0), element[:2]))
    return "\n".join(out) + "\nEND\n"


def residue_plddt(pdb):
    """Per residue, in structure order: the mean of its atoms' pLDDTs."""
    order, sums = [], {}
    for line in pdb.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        key = (line[21], line[22:26], line[17:20])
        if key not in sums:
            order.append(key)
            sums[key] = [0.0, 0]
        sums[key][0] += float(line[60:66])
        sums[key][1] += 1
    return [round(sums[k][0] / sums[k][1], 2) for k in order]


class Worker:
    """Imports once, builds a runner per model and settings, folds per job."""

    def __init__(self):
        from absl import flags
        import jax
        import run_alphafold as RA
        from alphafold3.model.components import platform
        self.jax, self.RA, self.flags = jax, RA, flags
        device = platform.attention_config()
        self.flash = device["attention"]
        flags_now = os.environ.get("XLA_FLAGS", "")
        for flag in device["xla_flags"]:
            if flag not in flags_now:
                flags_now = f"{flags_now} {flag}".strip()
        if flags_now:
            os.environ["XLA_FLAGS"] = flags_now
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["run_alphafold.py", "--norun_data_pipeline", f"--cache_dir={CACHE_DIR}"])
        flags.FLAGS.flash_attention_implementation = self.flash
        flags.FLAGS.stepwise_recycles = True
        flags.FLAGS.force_output_dir = True
        os.makedirs(os.path.join(CACHE_DIR, "jax"), exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", os.path.join(CACHE_DIR, "jax"))
        self.runners = {}
        self.device = str(jax.local_devices()[0].device_kind)

    def runner(self, model, recycles, steps, samples):
        key = (model, recycles, steps, samples)
        if key in self.runners:
            return self.runners[key]
        from alphafold3.model import weights
        self.flags.FLAGS.model = model
        extra = {"num_recycles": recycles} if recycles else {}
        config = self.RA.make_model_config(
            model_name=model, num_diffusion_samples=samples,
            flash_attention_implementation=self.flash, **extra)
        if steps:
            config.heads.diffusion.eval.steps = steps
        config.heads.diffusion.eval.stepwise = True
        precision = "fp32" if model == "alphafold3" else "int8"
        directory = "af3_native_weights" if model == "alphafold3" else weights.default_dir(model, precision)
        if model != "alphafold3":
            weights.ensure_weights(model, None, precision=precision)
        built = self.RA.ModelRunner(config=config, device=self.jax.local_devices()[0],
                                    model_dir=directory)
        self.runners = {key: (built, config)}      # one model's weights on the card at a time
        return built, config

    def fold(self, job):
        import numpy as np
        from alphafold3.common import folding_input
        import live_frames as LF
        family = job.get("controls", {}).get("model-family", "af3")
        model = MODELS.get(family)
        if model is None or model.startswith(("af2_", "esmfold2")):
            raise ValueError(f"the JAX backend does not fold {family} yet")
        controls = job.get("controls", {})
        recycles = int(controls.get("recycles") or 0)
        steps = int(controls.get("af3-count") or 0)
        emit("status", f"{model} on JAX ({self.device}) · reading the job")
        work = os.path.abspath("jax_jobs")
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        spec = json.loads(job["job"])
        search = controls.get("msa-mode", "none") == "search"
        # 🔴 NO TEMPLATE SEARCH EITHER WAY: the MMseqs2 fill supplies alignments
        # only, and a protein chain with neither templates nor an empty list is
        # refused ("Protein chain 1 is missing Templates").
        for entry in spec.get("sequences", []):
            if "protein" in entry:
                entry["protein"].setdefault("templates", [])
        if not search:
            # ...a single-sequence fold, stated in the job the way AlphaFold 3's
            # JSON states one: an empty alignment and no templates, rather than
            # left for a data pipeline that is not run.
            for entry in spec.get("sequences", []):
                for kind, body in entry.items():
                    if kind == "protein":
                        body.update(unpairedMsa="", pairedMsa="", templates=[])
                    elif kind == "rna":
                        body.update(unpairedMsa="")
        path = os.path.join(work, "job.json")
        with open(path, "w") as handle:
            json.dump(spec, handle)
        fold_input = next(iter(folding_input.load_fold_inputs_from_path(path)))
        if search:
            emit("status", f"{model} on JAX · searching the ColabFold MMseqs2 server")
            from alphafold3.data import msa_server
            fold_input = msa_server.fill_missing_msas(fold_input)
        emit("status", f"{model} on JAX · loading weights and compiling")
        runner, config = self.runner(model, recycles, steps, 1)
        passes = int(config.num_recycles) + 1
        steps = int(config.heads.diffusion.eval.steps)
        batch = [None]
        original = runner.run_inference

        def run_inference(featurised, *args, **kwargs):
            if batch[0] is None:
                batch[0] = LF.as_batch(featurised)
            return original(featurised, *args, **kwargs)
        runner.run_inference = run_inference

        # 🔴 THE BAR BY PHASE, NOT BY COUNTING CALLBACKS: the first fold of a
        # model reports its trunk passes twice (compile, then run), and a count
        # ran the bar past 100%. A trunk pass is a third of it, a denoise step
        # the rest.
        def on_frame(kind, index, data):
            if kind == "recycle":
                emit("progress", 0.3 * (index + 1) / passes)
                emit("status", f"{model} on JAX · trunk pass {index + 1}/{passes}")
            elif kind == "diffusion":
                emit("progress", 0.3 + 0.7 * (index + 1) / steps)
                emit("status", f"{model} on JAX · diffusion {index + 1}/{steps}")
                if batch[0] is not None:
                    positions = np.asarray(data[0] if getattr(data, "ndim", 0) == 4 else data)
                    emit("frame", cif_to_pdb(LF.frame_cif(positions, batch[0])))

        self.RA._FRAME_CALLBACK[0] = on_frame
        started = time.time()
        try:
            self.RA.process_fold_input(fold_input=fold_input, data_pipeline_config=None,
                                       model_runner=runner, output_dir=work, buckets=None,
                                       force_output_dir=True)
        finally:
            self.RA._FRAME_CALLBACK[0] = None
            runner.run_inference = original
        sequence = "".join(body.get("sequence", "") for entry in spec.get("sequences", [])
                           for kind, body in entry.items() if kind in ("protein", "dna", "rna")
                           for _ in (body.get("id") if isinstance(body.get("id"), list) else [0]))
        return self.collect(work, model, time.time() - started, sequence)

    def collect(self, work, model, seconds, sequence=""):
        """The top-ranked sample, as the page's result."""
        cif = sorted(glob.glob(f"{work}/**/*_model.cif", recursive=True), key=len)[0]
        stem = cif[:-len("_model.cif")]
        confidences = json.load(open(f"{stem}_confidences.json"))
        summary = json.load(open(f"{stem}_summary_confidences.json"))
        pdb = cif_to_pdb(open(cif).read())
        plddt = residue_plddt(pdb)
        pae = confidences.get("pae")
        scores = {"sequence": sequence, "plddt": plddt,
                  "mean_plddt": round(sum(plddt) / max(1, len(plddt)), 2),
                  "ptm": summary.get("ptm")}
        if summary.get("iptm") is not None:
            scores["iptm"] = summary["iptm"]
        if pae is not None and len(pae) == len(plddt):
            scores["pae"] = pae
        mean = scores["mean_plddt"]
        return {"pdb": pdb, "scores": scores, "length": len(plddt),
                "atoms": pdb.count("\nATOM") + pdb.count("\nHETATM") + int(pdb.startswith(("ATOM", "HETATM"))),
                "status": f"{model} on JAX ({self.device}) · done in {seconds:.0f} s"
                          f" · pLDDT {mean:.1f}"}


def codes_in(job_json):
    """Every CCD code a job names: ligands and modified residues, both dialects."""
    codes = set()
    parsed = json.loads(job_json)
    for fold in parsed if isinstance(parsed, list) else [parsed]:
        for entry in fold.get("sequences", []):
            for kind, body in entry.items():
                if kind == "ligand":
                    for code in ([body["ligand"]] if "ligand" in body else body.get("ccdCodes", [])):
                        codes.add(str(code).upper())
                for modification in body.get("modifications", []) if isinstance(body, dict) else []:
                    code = modification.get("ptmType") or modification.get("modificationType") or ""
                    if code:
                        codes.add(code.upper().removeprefix("CCD_"))
    return sorted(codes)


def fetch_ccd(codes):
    """The CCD tables alphafold3 reads AT IMPORT: the standard residues plus these.

    🔴 IN A CHILD PROCESS, because `alphafold3.constants` opens the pickles when
    it is first imported - so they must exist before this process imports it,
    and a code added after that is not seen until the process restarts (see
    main). It is the ColabFold2 input cell's own prefetch_ccd step.
    """
    script = (
        "import sys, os, importlib.metadata as md\n"
        "from alphafold3.constants import ccd_fetch\n"
        "root = os.path.dirname(md.distribution('alphafold3-colabfold').locate_file('alphafold3'))\n"
        "conv = os.path.join(root, 'alphafold3', 'constants', 'converters')\n"
        "os.makedirs(conv, exist_ok=True)\n"
        "ccd_fetch.write_pickles(ccd_fetch.codes_for_input(extra=sys.argv[1:]),\n"
        "  os.path.join(conv, 'ccd.pickle'), os.path.join(conv, 'chemical_component_sets.pickle'),\n"
        "  libcifpp_dir=os.path.join(root, 'share', 'libcifpp'))\n")
    import subprocess
    done = subprocess.run([sys.executable, "-c", script, *codes], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"could not fetch the CCD for {codes or 'the standard residues'}:"
                           f" {done.stderr[-400:]}")


def main():
    # 🔴 A JOB NAMING A NEW LIGAND RESTARTS THIS PROCESS WITH THAT JOB PENDING.
    # The CCD tables are read at import, so a code first seen after the import
    # cannot be added in place; `os.execv` keeps stdin and stdout, so the broker
    # sees one worker that took a little longer.
    pending = None
    if len(sys.argv) > 2 and sys.argv[1] == "--pending":
        pending = open(sys.argv[2]).read()
    fetched = set(json.loads(os.environ.get("LOCALFOLD_JAX_CODES", "null")) or [])
    worker = None
    emit("jax-ready", {"at": int(time.time() * 1000)})
    lines = iter(sys.stdin)
    while True:
        line = pending if pending is not None else next(lines, None)
        pending = None
        if line is None:
            break
        if not line.strip():
            continue
        try:
            job = json.loads(line)
            codes = set(codes_in(job["job"]))
            if worker is not None and not codes <= fetched:
                path = os.path.abspath("jax_pending.json")
                with open(path, "w") as handle:
                    handle.write(line)
                os.environ["LOCALFOLD_JAX_CODES"] = json.dumps(sorted(codes | fetched))
                emit("status", f"restarting JAX for {', '.join(sorted(codes - fetched))}")
                os.execv(sys.executable, [sys.executable, os.path.abspath(__file__),
                                          "--pending", path])
            if worker is None:
                emit("status", "fetching chemical definitions")
                fetched |= codes
                fetch_ccd(sorted(fetched))
                emit("status", "starting JAX")
                worker = Worker()
            emit("result", worker.fold(job))
        except Exception as cause:                            # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            emit("result", {"error": f"{type(cause).__name__}: {cause}"})


if __name__ == "__main__":
    main()
