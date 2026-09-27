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
    "ef2-fast-600m": "esmfold2_lm600m", "ef2-fast-300m": "esmfold2_lm300m",
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


AF2_NAMES = {"af2_ptm": "model_{}_ptm", "af2_multimer": "model_{}_multimer_v3"}
# 🔴 FOLDED WITHOUT LIVE FRAMES: af3-any-model's stepwise path fails on a
# structural-token model - `--stepwise_recycles` dies with KeyError 'init' in
# staged.py, and stepwise diffusion with a (68, 24) mask against (160, 24, 3)
# positions in random_augmentation - while its plain path folds the same job
# (run_alphafold.py --model=opendde, 109 s on an L4). Measured 2026-09-27 on the
# colab branch; the status line says so rather than leaving the bar still.
NO_LIVE = {"opendde"}
AF2_DIR = "af2_params"


def cif_atoms(cif):
    """The atom_site loop of one mmCIF model, one dict an atom."""
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

    def get(row, *names):
        return next((row[col[n]] for n in names if n in col and row[col[n]] != "?"), "")
    atoms = []
    for row in rows:
        seq = get(row, "label_seq_id")
        atoms.append({
            "group": get(row, "group_PDB") or "ATOM",
            "name": get(row, "label_atom_id", "auth_atom_id").strip('"'),
            "element": get(row, "type_symbol"),
            "comp": get(row, "label_comp_id", "auth_comp_id"),
            "chain": get(row, "label_asym_id", "auth_asym_id"),
            "authChain": get(row, "auth_asym_id", "label_asym_id"),
            "seq": seq if seq not in ("", ".") else get(row, "auth_seq_id"),
            "authSeq": get(row, "auth_seq_id", "label_seq_id"),
            "xyz": (float(get(row, "Cartn_x")), float(get(row, "Cartn_y")),
                    float(get(row, "Cartn_z"))),
            "b": float(get(row, "B_iso_or_equiv") or 0.0),
        })
    return atoms


def cif_to_pdb(cif):
    """One mmCIF model as PDB ATOM/HETATM records.

    The page ingests PDB (py2Dmol and every download path read it), and the
    B-factor column carries the atom's pLDDT, which is what the viewer colours
    by. Only the fields a PDB record has are kept.
    """
    out = []
    for serial, atom in enumerate(cif_atoms(cif), start=1):
        name, element = atom["name"], atom["element"]
        padded = name if len(name) == 4 or len(element) == 2 else f" {name}"
        out.append("%-6s%5d %-4s %3s %1s%4s    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s" % (
            atom["group"], serial % 100000, padded[:4], atom["comp"][:3],
            atom["authChain"][:1], atom["authSeq"][-4:], *atom["xyz"], 1.0, atom["b"],
            element[:2]))
    return "\n".join(out) + "\nEND\n"


def token_plddt(atoms, token_chain_ids, token_res_ids):
    """One pLDDT a TOKEN, which is what AlphaFold 3 scores and the page draws.

    AF3's own files give pLDDT per atom and the token layout, not a per-token
    pLDDT. A residue that is one token takes its atoms' mean; a ligand or an
    atomised residue is one token an atom and takes each atom's own. Keyed on
    (chain, residue) in the order the tokens name them.
    """
    groups = {}
    for atom in atoms:
        groups.setdefault((atom["chain"], str(atom["seq"])), []).append(atom["b"])
    counts = {}
    for chain, res in zip(token_chain_ids, token_res_ids):
        counts[(chain, str(res))] = counts.get((chain, str(res)), 0) + 1
    seen, out = {}, []
    for chain, res in zip(token_chain_ids, token_res_ids):
        key = (chain, str(res))
        values = groups.get(key, [0.0])
        index = seen.get(key, 0)
        seen[key] = index + 1
        if counts[key] == len(values) and counts[key] > 1:
            out.append(values[index])
        else:
            out.append(sum(values) / len(values))
    return [round(v, 2) for v in out]


class Refused(Exception):
    """A job this backend does not run, said as such rather than approximated."""


class Worker:
    """Imports once, keeps one model's runner, folds a job at a time."""

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
            flags.FLAGS(["run_alphafold.py", "--norun_data_pipeline",
                         f"--cache_dir={CACHE_DIR}"])
        flags.FLAGS.flash_attention_implementation = self.flash
        flags.FLAGS.stepwise_recycles = True
        flags.FLAGS.force_output_dir = True
        os.makedirs(os.path.join(CACHE_DIR, "jax"), exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", os.path.join(CACHE_DIR, "jax"))
        self.runners = {}
        self.device = str(jax.local_devices()[0].device_kind)

    def runner(self, model, settings):
        """One model's runner at a time: two models' weights do not share a T4."""
        key = (model, tuple(sorted(settings.items())))
        if key in self.runners:
            return self.runners[key]
        self.runners = {}
        from alphafold3.model import weights
        device = self.jax.local_devices()[0]
        self.flags.FLAGS.model = model
        self.flags.FLAGS.use_esm_embeddings = model.startswith("esmfold2")
        self.flags.FLAGS.stepwise_recycles = model not in NO_LIVE
        if model in AF2_NAMES:
            from alphafold3.af2 import inference as af2_inference
            from alphafold3.model import model_registry
            weights.ensure_af2_params(AF2_DIR)
            built = af2_inference.AF2ModelRunner(
                model_registry.get(model), device=device, model_dir=AF2_DIR,
                num_recycles=settings["recycles"] or 3,
                num_msa=settings["msa"], num_extra_msa=settings["msa"] * 2,
                model_names=[AF2_NAMES[model].format(settings["af2_model"])],
                use_templates=False)
            config = None
        else:
            extra = {"num_recycles": settings["recycles"]} if settings["recycles"] else {}
            config = self.RA.make_model_config(
                model_name=model, num_diffusion_samples=1,
                flash_attention_implementation=self.flash, **extra)
            if settings["steps"]:
                config.heads.diffusion.eval.steps = settings["steps"]
            if settings["msa"] and hasattr(config, "evoformer"):
                config.evoformer.num_msa = settings["msa"]
            config.heads.diffusion.eval.stepwise = model not in NO_LIVE
            if model == "alphafold3":
                directory = "af3_native_weights"
                if not glob.glob(f"{directory}/*.bin.zst"):
                    raise Refused(
                        "AlphaFold 3's parameters are not on this runtime - they come"
                        " from DeepMind under their own terms. Run ColabFold2's install"
                        " cell with model = alphafold3, or pick another model")
            else:
                directory = weights.ensure_weights(model, None, precision="int8")
            built = self.RA.ModelRunner(config=config, device=device, model_dir=directory)
        self.runners = {key: (built, config)}
        return built, config

    def fold(self, job):
        import numpy as np
        from alphafold3.common import folding_input
        import live_frames as LF
        controls = job.get("controls", {})
        family = job.get("family") or controls.get("model-family", "af3")
        # 🔴 AF2's FAMILY CARRIES ITS MODEL NUMBER: the page resolves the
        # number box into the name - `monomer-2`, `multimer-5` - because it is
        # which of five bundles loads. It is which of five parameter sets here.
        base, _, number = family.partition("-")
        if base in ("monomer", "multimer") and number.isdigit():
            family = base
            controls = {**controls, "af2Model": number}
        model = MODELS.get(family)
        if model is None:
            raise Refused(f"the JAX backend does not know the model {family!r}")
        af2 = model in AF2_NAMES
        single_sequence = model.startswith("esmfold2")
        if not af2 and controls.get("af3-mode", "diffusion") != "diffusion":
            raise Refused("the JAX backend samples with diffusion only - set the sampler"
                          " to Diffusion, or fold with WebGPU for Flow")
        if any((entity.get("template") or {}).get("kind", "none") != "none"
               for entity in job.get("entities", [])):
            raise Refused("templates are not wired to the JAX backend yet - remove the"
                          " template, or fold with WebGPU")
        depth = str(controls.get("max-msa") or "512:1024").split(":")[0]
        settings = {"recycles": int(controls.get("recycles") or 0),
                    "steps": int(controls.get("af3-count") or 0),
                    "msa": int(depth) if depth.isdigit() else 512,
                    "af2_model": int(controls.get("af2Model") or 1)}
        emit("status", f"{model} on JAX ({self.device}) · reading the job")
        work = os.path.abspath("jax_jobs")
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        spec = json.loads(job["job"])
        mode = "none" if single_sequence else controls.get("msa-mode", "none")
        self.apply_alignment(spec, mode, controls, job.get("msas"))
        path = os.path.join(work, "job.json")
        with open(path, "w") as handle:
            json.dump(spec, handle)
        fold_input = next(iter(folding_input.load_fold_inputs_from_path(path)))
        if mode == "search":
            emit("status", f"{model} on JAX · searching the ColabFold MMseqs2 server")
            from alphafold3.data import msa_server
            fold_input = msa_server.fill_missing_msas(fold_input)
        emit("status", f"{model} on JAX · loading weights and compiling")
        runner, config = self.runner(model, settings)
        batch = [None]
        original = runner.run_inference
        passes = ((settings["recycles"] or 3) + 1) if af2 else int(config.num_recycles) + 1
        steps = 0 if af2 else int(config.heads.diffusion.eval.steps)

        def frame(positions):
            emit("frame", cif_to_pdb(LF.frame_cif(np.asarray(positions), batch[0])))

        if af2:
            from alphafold3.af2.output import atom37_to_token_atoms

            def on_recycle(index, out):
                emit("progress", min(1.0, (index + 1) / passes))
                emit("status", f"{model} on JAX · recycle {index + 1}/{passes}")
                if batch[0] is not None:
                    atom37 = np.asarray(out["structure_module"]["final_atom_positions"])
                    frame(atom37_to_token_atoms(atom37, batch[0])[0])

            def run_inference(featurised, *args, **kwargs):
                if batch[0] is None:
                    batch[0] = LF.as_batch(featurised)
                kwargs.setdefault("on_recycle", on_recycle)
                return original(featurised, *args, **kwargs)
        else:
            # 🔴 THE BAR BY PHASE, NOT BY COUNTING CALLBACKS: the first fold of a
            # model reports its trunk passes twice (compile, then run), and a
            # count ran the bar past 100%. A trunk pass is a third of it, a
            # denoise step the rest.
            def on_frame(kind, index, data):
                if kind == "recycle":
                    emit("progress", 0.3 * (index + 1) / passes)
                    emit("status", f"{model} on JAX · trunk pass {index + 1}/{passes}")
                elif kind == "diffusion":
                    emit("progress", 0.3 + 0.7 * (index + 1) / steps)
                    emit("status", f"{model} on JAX · diffusion {index + 1}/{steps}")
                    if batch[0] is not None:
                        frame(data[0] if getattr(data, "ndim", 0) == 4 else data)

            def run_inference(featurised, *args, **kwargs):
                if batch[0] is None:
                    batch[0] = LF.as_batch(featurised)
                return original(featurised, *args, **kwargs)
            if model in NO_LIVE:
                emit("status", f"{model} on JAX · folding (this model has no live frames on"
                               " JAX - the structure arrives at the end)")
            else:
                self.RA._FRAME_CALLBACK[0] = on_frame
        runner.run_inference = run_inference
        started = time.time()
        try:
            self.RA.process_fold_input(fold_input=fold_input, data_pipeline_config=None,
                                       model_runner=runner, output_dir=work, buckets=None,
                                       force_output_dir=True)
        finally:
            self.RA._FRAME_CALLBACK[0] = None
            runner.run_inference = original
        return self.collect(work, model, time.time() - started, fold_input,
                            job.get("family") or controls.get("model-family"))

    @staticmethod
    def apply_alignment(spec, mode, controls, msas):
        """What the page's MSA row asked for, stated in the job's own fields.

        🔴 A PROTEIN CHAIN ALWAYS CARRIES `templates: []`: the MMseqs2 fill
        supplies alignments only, and a chain with neither templates nor an
        empty list is refused ("Protein chain 1 is missing Templates").
        """
        proteins = [entry["protein"] for entry in spec.get("sequences", []) if "protein" in entry]
        rnas = [entry["rna"] for entry in spec.get("sequences", []) if "rna" in entry]
        for protein in proteins:
            protein.setdefault("templates", [])
        if mode == "search":
            return
        if mode == "none":
            # ...a single-sequence fold, the way AlphaFold 3's JSON states one.
            for protein in proteins:
                protein.update(unpairedMsa="", pairedMsa="")
            for rna in rnas:
                rna.update(unpairedMsa="")
            return
        if mode == "paste":
            msas = {"merged": controls.get("msa-text") or ""}
        msas = msas or {}
        for rna in rnas:
            rna.setdefault("unpairedMsa", "")
        if msas.get("merged"):
            if len(proteins) != 1:
                raise Refused("one alignment for several protein chains cannot be split"
                              " for the JAX backend - upload per-chain alignments, search,"
                              " or fold with WebGPU")
            proteins[0].update(unpairedMsa=msas["merged"], pairedMsa="")
            return
        unpaired = msas.get("unpaired") or []
        paired = msas.get("paired") or []
        if not any(unpaired):
            raise Refused(f"the MSA mode is {mode!r} but no alignment came with the job")
        for index, protein in enumerate(proteins):
            protein.update(unpairedMsa=unpaired[index] if index < len(unpaired) else "",
                           pairedMsa=(paired[index] if index < len(paired) else "") or "")

    def collect(self, work, model, seconds, fold_input, job_family=None):
        """The top-ranked sample, as the page's own prediction fields."""
        from alphafold3.common import folding_input
        cif_path = sorted(glob.glob(f"{work}/**/*_model.cif", recursive=True), key=len)[0]
        stem = cif_path[:-len("_model.cif")]
        confidences = json.load(open(f"{stem}_confidences.json"))
        summary = json.load(open(f"{stem}_summary_confidences.json"))
        cif = open(cif_path).read()
        atoms = cif_atoms(cif)
        chain_ids = confidences.get("token_chain_ids") or []
        res_ids = confidences.get("token_res_ids") or []
        plddt = token_plddt(atoms, chain_ids, res_ids)

        def flat(matrix):
            return None if matrix is None else [round(float(v), 2) for row in matrix for v in row]
        confidence = {
            "plddt": plddt,
            "meanPlddt": round(sum(plddt) / max(1, len(plddt)), 2),
            "ptm": summary.get("ptm"),
            "predictedAlignedError": flat(confidences.get("pae")),
            "contactProbs": flat(confidences.get("contact_probs")),
        }
        if summary.get("iptm") is not None:
            confidence["iptm"] = summary["iptm"]
        # ...what the fold was actually given, per polymer chain in order - the
        # page's `chains`, and its archive's `msas/`.
        chains, unpaired, paired = [], [], []
        for chain in fold_input.chains:
            sequence = getattr(chain, "sequence", None)
            if sequence is None:
                continue
            chains.append(sequence)
            if isinstance(chain, folding_input.ProteinChain):
                unpaired.append(chain.unpaired_msa or "")
                paired.append(chain.paired_msa or "")
        msas = {"unpaired": unpaired, "paired": paired} if any(unpaired) else {}
        mean = confidence["meanPlddt"]
        return {
            "jax": True, "model": model, "family": job_family,
            "pdb": cif_to_pdb(cif), "confidence": confidence,
            "tokens": {"chainIds": chain_ids, "resIds": res_ids},
            "chains": chains, "msas": msas,
            "a3m": unpaired[0] if len(unpaired) == 1 and unpaired[0] else None,
            "atoms": len(atoms),
            "status": f"{model} on JAX ({self.device}) · done in {seconds:.0f} s"
                      f" · pLDDT {mean:.1f}",
        }


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
                # 🔴 fd 1 BACK ON THE PIPE FIRST. This process pointed it at
                # stderr and keeps events on a private copy, which does not
                # survive exec - so the new process would write its events to
                # stderr and the broker would see the pipe close.
                OUT.flush()
                os.dup2(OUT.fileno(), 1)
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
            said = str(cause) if isinstance(cause, Refused) else f"{type(cause).__name__}: {cause}"
            emit("result", {"error": said})


if __name__ == "__main__":
    main()
