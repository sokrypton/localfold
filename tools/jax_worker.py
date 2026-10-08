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
import subprocess
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
# 🔴 THE GPU IS SHARED, SO JAX MUST NOT TAKE THREE QUARTERS OF IT UP FRONT.
# XLA preallocates 75% of device memory at its first import; on a Colab runtime
# the page's own WebGPU Chrome is on the same card, and a head-to-head run had
# each starving the other - JAX out of memory on OpenDDE and ESMFold2, WebGPU's
# device lost on an L4. On demand, both fit.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
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


AF3_WEIGHTS_URL = "https://storage.googleapis.com/alphafold3/af3.bin.zst"
RCSB = "https://files.rcsb.org/download"
AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction"


def fetch_text(url):
    import urllib.request
    with urllib.request.urlopen(url, timeout=60) as answer:
        return answer.read().decode()


def split_merged(a3m, lengths):
    """One A3M over several chains, as one A3M a chain.

    The page's merged alignment runs the protein chains end to end in the
    query row; an uppercase letter or `-` is a column, a lowercase letter an
    insertion kept with the column before it. A row that has only gaps over a
    chain says nothing about it and is left out of that chain's alignment.
    """
    bounds = [sum(lengths[:i]) for i in range(len(lengths) + 1)]
    out = [[] for _ in lengths]
    header = None
    for line in a3m.splitlines():
        if line.startswith(">"):
            header = line
            continue
        if header is None or not line.strip():
            continue
        pieces = ["" for _ in lengths]
        column = 0
        for char in line.strip():
            if char.islower():
                chain = max(0, next(i for i in range(len(lengths)) if column <= bounds[i + 1]) )
                pieces[min(chain, len(lengths) - 1)] += char
                continue
            chain = next((i for i in range(len(lengths)) if column < bounds[i + 1]), None)
            if chain is None:
                break
            pieces[chain] += char
            column += 1
        for chain, piece in enumerate(pieces):
            if any(c.isupper() for c in piece) or not out[chain]:
                out[chain].append(f"{header}\n{piece}")
        header = None
    return ["\n".join(rows) + "\n" for rows in out]


def align(query, target):
    """A global alignment, 0-based query index -> target index, identity-scored,
    aligned mismatches included (they are template residues all the same).

    The page lines a template up with its chain the same way (web/align.js):
    a construct with a tag shifts every residue, and pairing index for index
    across the shift would swing the whole template.
    """
    n, m = len(query), len(target)
    gap = -2
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = i * gap
    for j in range(1, m + 1):
        score[0][j] = j * gap
    for i in range(1, n + 1):
        row, previous = score[i], score[i - 1]
        qi = query[i - 1]
        for j in range(1, m + 1):
            row[j] = max(previous[j - 1] + (2 if qi == target[j - 1] else -1),
                         previous[j] + gap, row[j - 1] + gap)
    pairs, i, j = {}, n, m
    while i > 0 and j > 0:
        if score[i][j] == score[i - 1][j - 1] + (2 if query[i - 1] == target[j - 1] else -1):
            pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif score[i][j] == score[i - 1][j] + gap:
            i -= 1
        else:
            j -= 1
    return dict(sorted(pairs.items()))


def pdb_to_cif(pdb, name, chain=None):
    """A PDB file's ATOM/HETATM records as the smallest mmCIF alphafold3 reads.

    🔴 ONE CONFORMATION: a crystal's alternate locations (5CAJ has fourteen)
    come through as duplicate atoms, which alphafold3's parser refused ("cannot
    assign 6 input values to the 2 output values"). The first location is kept,
    as the page's own reader keeps it.
    """
    lines = [f"data_{name}", "#",
             "_pdbx_audit_revision_history.revision_date 1970-01-01", "#",
             "loop_"]
    fields = ["group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
              "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
              "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
              "B_iso_or_equiv", "auth_seq_id", "auth_comp_id", "auth_asym_id",
              "auth_atom_id", "pdbx_PDB_model_num"]
    lines += [f"_atom_site.{field}" for field in fields]
    serial = 0
    for record in pdb.splitlines():
        if record.startswith("ENDMDL"):
            break
        if not record.startswith(("ATOM", "HETATM")):
            continue
        if record[16] not in (" ", "A"):
            continue
        # ...and one chain's POLYMER: a crystal's waters and ligands under the
        # same chain letter are further entities, which this minimal file cannot
        # describe and the parser refuses. MSE is the one HETATM that is chain.
        this = record[21].strip() or "A"
        chain = chain or this
        if this != chain or (record.startswith("HETATM") and record[17:20] != "MSE"):
            continue
        serial += 1
        element = record[76:78].strip() or record[12:16].strip()[0]
        lines.append(" ".join([
            record[:6].strip(), str(serial), element, record[12:16].strip(),
            ".", record[17:20].strip(), chain, "1",
            record[22:26].strip(), record[26].strip() or "?",
            record[30:38].strip(), record[38:46].strip(), record[46:54].strip(),
            record[54:60].strip() or "1.0", record[60:66].strip() or "0.0",
            record[22:26].strip(), record[17:20].strip(), chain,
            record[12:16].strip(), "1"]))
    return "\n".join(lines) + "\n#\n"


def template_entry(template, query):
    """One of the page's template rows as AlphaFold 3's own template input."""
    import datetime
    from alphafold3 import structure
    kind = template.get("kind")
    source = (template.get("source") or "").strip()
    if kind == "mmcif":
        # ...a structure already in hand: the MSA search's best hit.
        cif, chain, name = template["text"], template.get("chain") or "", template.get("name", "hit")
    elif kind == "pdb":
        entry, _, chain = source.replace(":", "_").partition("_")
        cif = fetch_text(f"{RCSB}/{entry.upper()}.cif")
        name = entry.upper()
    elif kind == "afdb":
        entry, _, chain = source.replace(":", "_").partition("_")
        listing = json.loads(fetch_text(f"{AFDB_API}/{entry.upper()}"))
        cif = fetch_text(listing[0]["cifUrl"])
        name = entry.upper()
    elif kind == "upload":
        text = template.get("text") or ""
        chain = source
        name = "upload"
        cif = text if ("_atom_site." in text) else pdb_to_cif(text, name, chain or None)
    else:
        raise Refused(f"the JAX backend does not know the template source {kind!r}")
    struc = structure.from_mmcif(cif, fix_mse_residues=True, fix_arginines=True,
                                 include_bonds=False, include_water=False)
    chains = list(struc.polymer_auth_asym_id_to_label_asym_id())
    if not chains:
        raise Refused(f"the template {name} has no polymer chain")
    chain = chain or chains[0]
    if chain not in chains:
        raise Refused(f"the template {name} has no chain {chain} (it has {', '.join(chains)})")
    struc = struc.filter(chain_auth_asym_id=chain)
    if struc.release_date is None or struc.name is None:
        struc = struc.copy_and_update_globals(
            name=struc.name or name, release_date=struc.release_date or datetime.date(1970, 1, 1))
    label = struc.polymer_auth_asym_id_to_label_asym_id()[chain]
    mapping = align(query, struc.chain_single_letter_sequence()[label])
    if not mapping:
        raise Refused(f"the template {name}_{chain} shares no residue with its chain")
    entry = {"mmcif": struc.to_mmcif(), "queryIndices": list(mapping),
             "templateIndices": list(mapping.values())}
    # ...and what the page's archive records, so a saved JAX fold reloads with
    # the structure it used rather than as a search: one chain, named.
    entry_used = {"text": entry["mmcif"], "chainId": chain, "source": name if name != "upload"
                  else template.get("filename") or "the uploaded structure"}
    return entry, len(mapping), entry_used


# 🔴 THE LIVE SAMPLER BAKED THE FIRST FOLD'S ROTARY TABLES INTO EVERY LATER FOLD. alphafold3's
# staged driver (src/alphafold3/model/staged.py, the `colab` branch the notebook installs) compiles its
# denoise step once and caches it, passing the atom conditioning's ARRAYS through jit and closing over
# the rest - and it sorted by `hasattr(v, 'shape')`, so ESMFold2's `rope_q`/`rope_k`, which are TUPLES
# of arrays, were closed over as constants, while the cache key looks only at 0-d values and so never
# changed between ESMFold2 folds. A second fold of another size died ("mul got incompatible shapes for
# broadcasting: (192, 32, 4, 32), (51, 32, 1, 32)"); one of the SAME padded size would have folded
# silently with the first molecule's conformer. A tuple or list of arrays is data. The fix belongs
# upstream; until it lands this applies it, and refuses if the line it fixes has changed shape.
STAGED_BUG = "(static if is_flag or not hasattr(v, 'shape') else arrays)[k] = v"
STAGED_FIX = ("(static if is_flag or not (hasattr(v, 'shape') or (isinstance(v, (tuple, list)) and len(v) > 0\n"
              "          and all(hasattr(x, 'shape') for x in v))) else arrays)[k] = v")


def patch_staged():
    import inspect
    from alphafold3.model import staged
    source = inspect.getsource(staged)
    if STAGED_BUG not in source:
        if "isinstance(v, (tuple, list))" in source:
            return "fixed upstream"
        raise RuntimeError("alphafold3.model.staged no longer has the line this worker patches; "
                           "check whether its tuple-of-arrays conditioning still crosses jit (tools/jax_worker.py)")
    exec(compile(source.replace(STAGED_BUG, STAGED_FIX), staged.__file__, "exec"), staged.__dict__)
    return "patched"


class Worker:
    """Imports once, keeps one model's runner, folds a job at a time."""

    def __init__(self):
        from absl import flags
        import jax
        import run_alphafold as RA
        from alphafold3.model.components import platform
        patch_staged()
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
        self.flags.FLAGS.stepwise_recycles = True
        if model in AF2_NAMES:
            from alphafold3.af2 import inference as af2_inference
            from alphafold3.model import model_registry
            weights.ensure_af2_params(AF2_DIR)
            built = af2_inference.AF2ModelRunner(
                model_registry.get(model), device=device, model_dir=AF2_DIR,
                num_recycles=settings["recycles"] or 3,
                num_msa=settings["msa"], num_extra_msa=settings["msa"] * 2,
                model_names=[AF2_NAMES[model].format(settings["af2_model"])],
                use_templates=settings["templates"])
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
            config.heads.diffusion.eval.stepwise = True
            # LocalFold's Flow, which sokrypton/alphafold3's sampler carries too.
            config.heads.diffusion.eval.flow = settings["flow"]
            if settings["sigma_max"] and not settings["flow"]:
                config.heads.diffusion.eval.sigma_max = settings["sigma_max"]
            if model == "alphafold3":
                directory = "af3_native_weights"
                if not glob.glob(f"{directory}/*.bin.zst"):
                    # 🔴 FETCHED HERE, AFTER THE PAGE'S OWN TERMS DIALOG. The reader
                    # cannot send an AlphaFold 3 fold without accepting DeepMind's
                    # terms (agreeModelTerms in web/app.js); this is the same file
                    # ColabFold2's install cell fetches for model = alphafold3.
                    self.download_af3(directory)
            else:
                directory = weights.ensure_weights(model, None, precision="int8")
            built = self.RA.ModelRunner(config=config, device=device, model_dir=directory)
        self.runners = {key: (built, config)}
        return built, config

    @staticmethod
    def download_af3(directory):
        import urllib.request
        os.makedirs(directory, exist_ok=True)
        partial = os.path.join(directory, "af3.bin.zst.part")
        with urllib.request.urlopen(AF3_WEIGHTS_URL, timeout=60) as answer, open(partial, "wb") as out:
            total = int(answer.headers.get("Content-Length") or 0)
            done, said = 0, 0
            while chunk := answer.read(1 << 22):
                out.write(chunk)
                done += len(chunk)
                if done - said > (64 << 20):
                    said = done
                    emit("status", "downloading AlphaFold 3's parameters"
                                   f" · {done >> 20} of {total >> 20} MiB")
        if done < 1_000_000:
            raise RuntimeError("the AlphaFold 3 download is incomplete")
        os.replace(partial, os.path.join(directory, "af3.bin.zst"))

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
        sampler = controls.get("af3-mode", "diffusion")
        if af2 or single_sequence:
            sampler = "diffusion"   # the row is hidden for these; their own samplers run
        if sampler not in ("diffusion", "flow"):
            raise Refused(f"the JAX backend does not know the sampler {sampler!r}")
        if sampler == "flow" and model == "rosettafold3":
            # ...the page's own rule: its walk collapses the backbone while pLDDT
            # reads as if nothing were wrong (noFlowSampler, shared/af3/dialect.js).
            raise Refused("rosettafold3 has no working flow sampler - set the sampler to"
                          " Diffusion")
        templated = [entity for entity in job.get("entities", [])
                     if entity.get("type") == "protein"
                     and (entity.get("template") or {}).get("kind", "none") != "none"]
        if templated and single_sequence:
            raise Refused("ESMFold2 takes no template")
        depth = str(controls.get("max-msa") or "512:1024").split(":")[0]
        settings = {"recycles": int(controls.get("recycles") or 0),
                    "steps": int((job.get("schedule") or {}).get("steps") or controls.get("af3-count") or 0),
                    "msa": int(depth) if depth.isdigit() else 512,
                    "af2_model": int(controls.get("af2Model") or 1),
                    "flow": sampler == "flow",
                    # the page's short schedule (web/af3-model.js diffusionScheduleFor), resolved there
                    "sigma_max": float((job.get("schedule") or {}).get("sigmaMax") or 0),
                    "templates": bool(templated)}
        # All five multimer models carry the multimer template embedder; only
        # the monomer's 3, 4 and 5 are template-free.
        if templated and model == "af2_ptm" and settings["af2_model"] not in (1, 2):
            raise Refused("AlphaFold 2's models 3, 4 and 5 have no template embedder -"
                          " pick model 1 or 2, or drop the template")
        emit("status", f"{model} on JAX ({self.device}) · reading the job")
        work = os.path.abspath("jax_jobs")
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        spec = json.loads(job["job"])
        mode = "none" if single_sequence else controls.get("msa-mode", "none")
        self.apply_alignment(spec, mode, controls, job.get("msas"))
        # ...and each protein row's template, onto the protein entry it wrote.
        proteins = [entry["protein"] for entry in spec.get("sequences", []) if "protein" in entry]
        rows = [entity for entity in job.get("entities", [])
                if entity.get("type") == "protein" and (entity.get("value") or "").strip()]
        searched = [protein for entity, protein in zip(rows, proteins)
                    if (entity.get("template") or {}).get("kind") == "search"]
        if searched and mode != "search":
            raise Refused("a template from the MSA search needs an MSA search: set the MSA"
                          " to search, or name a structure instead")
        # A fold chain is a polymer COPY, numbered over every polymer row - the
        # page's own numbering, which its archive writes templates under.
        first_chain, at = {}, 0
        for entity in job.get("entities", []):
            if entity.get("type") in ("protein", "dna", "rna") and (entity.get("value") or "").strip():
                first_chain[id(entity)] = at
                at += int(entity.get("copies") or 1)
        used, coverage = [], []

        def record(entity, found, covered, of):
            # ...and the coverage once an entity, for the final status line -
            # the WebGPU fold ends on it, and a template that arrived is
            # otherwise indistinguishable from one that did not.
            coverage.append(f" · template {found['source']} {covered}/{of}")
            for copy in range(int(entity.get("copies") or 1)):
                used.append({**found, "chain": first_chain[id(entity)] + copy})

        for entity, protein in zip(rows, proteins):
            template = entity.get("template") or {}
            if template.get("kind", "none") in ("none", "search"):
                continue
            named = (template.get("filename") or "the uploaded structure") if template.get("kind") == "upload" \
                else template.get("source")
            emit("status", f"{model} on JAX · template {named}")
            entry, covered, found = template_entry(template, protein["sequence"])
            protein["templates"] = [entry]
            record(entity, found, covered, len(protein["sequence"]))
            emit("status", f"{model} on JAX · template covers {covered} of"
                           f" {len(protein['sequence'])} residues")
        path = os.path.join(work, "job.json")
        with open(path, "w") as handle:
            json.dump(spec, handle)
        fold_input = next(iter(folding_input.load_fold_inputs_from_path(path)))
        if mode == "search":
            emit("status", f"{model} on JAX · searching the ColabFold MMseqs2 server")
            from alphafold3.data import msa_server
            hits = {}
            fold_input = msa_server.fill_missing_msas(fold_input, template_hits=hits)
            if searched:
                # 🔴 THE BEST HIT, AS THE PAGE TAKES IT - one template a chain,
                # from the search that produced the alignment. The chains are
                # rebuilt with it rather than edited, and the job re-read.
                searched_rows = [entity for entity in rows
                                 if (entity.get("template") or {}).get("kind") == "search"]
                for entity, protein in zip(searched_rows, searched):
                    best = (hits.get(protein["sequence"]) or [None])[0]
                    if best is None:
                        raise Refused(f"the search found no template for {protein['sequence'][:12]}…")
                    emit("status", f"{model} on JAX · template {best} from the search")
                    entry, covered, found = template_entry(
                        {"kind": "mmcif", "text": msa_server.fetch_template(best),
                         "chain": best.split("_")[1],
                         "name": best}, protein["sequence"])
                    protein["templates"] = [entry]
                    record(entity, found, covered, len(protein["sequence"]))
                    emit("status", f"{model} on JAX · template {best} covers {covered} of"
                                   f" {len(protein['sequence'])} residues")
                # The filled alignments go into the job with the templates.
                for chain, protein in zip([c for c in fold_input.chains
                                           if isinstance(c, folding_input.ProteinChain)], proteins):
                    protein["unpairedMsa"] = chain.unpaired_msa or ""
                    protein["pairedMsa"] = chain.paired_msa or ""
                with open(path, "w") as handle:
                    json.dump(spec, handle)
                fold_input = next(iter(folding_input.load_fold_inputs_from_path(path)))
        emit("status", f"{model} on JAX · loading weights and compiling")
        runner, config = self.runner(model, settings)
        batch = [None]
        converged = [None]
        original = runner.run_inference
        passes = ((settings["recycles"] or 3) + 1) if af2 else int(config.num_recycles) + 1
        steps = 0 if af2 else int(config.heads.diffusion.eval.steps)

        def frame(positions):
            emit("frame", cif_to_pdb(LF.frame_cif(np.asarray(positions), batch[0])))

        if af2:
            from alphafold3.af2.output import atom37_to_token_atoms

            from alphafold3.af2.common.confidence import compute_tol
            # 🔴 THE PAGE'S EARLY STOP, WHICH THIS BACKEND IGNORED. `tolerance`
            # was sent with every AF2 fold and read by nothing here, so JAX ran
            # every pass where WebGPU stopped on a settled structure. Same
            # metric (ColabFold's compute_tol - the RMS change of all C-alpha
            # pair distances), same rule (not before the second pass, and 0
            # means every pass), and af3-any-model's loop ends when this
            # returns True.
            tolerance = float(controls.get("tolerance") or 0)
            previous = [None]

            def on_recycle(index, out):
                emit("progress", min(1.0, (index + 1) / passes))
                emit("status", f"{model} on JAX · recycle {index + 1}/{passes}")
                atom37 = np.asarray(out["structure_module"]["final_atom_positions"])
                if batch[0] is not None:
                    frame(atom37_to_token_atoms(atom37, batch[0])[0])
                distance = None if previous[0] is None else float(
                    compute_tol(previous[0], atom37, np.ones(atom37.shape[0])))
                previous[0] = atom37
                if index > 0 and tolerance > 0 and distance is not None and distance < tolerance:
                    converged[0] = (distance, index + 1)
                    return True
                return False

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
            self.RA._FRAME_CALLBACK[0] = on_frame
        runner.run_inference = run_inference
        started = time.time()
        try:
            # 🔴 `use_esm` IS AN ARGUMENT HERE, NOT THE FLAG. The flag above is
            # what main() reads to decide this; process_fold_input defaults it
            # to False and never looks at the flag - so every ESMFold2 fold on
            # this backend ran WITHOUT its language model: 6MRR at 15.81 A on
            # the 600M where the CLI and the WebGPU port both give ~1.5.
            self.RA.process_fold_input(fold_input=fold_input, data_pipeline_config=None,
                                       model_runner=runner, output_dir=work, buckets=None,
                                       force_output_dir=True,
                                       use_esm=model.startswith("esmfold2"))
        finally:
            self.RA._FRAME_CALLBACK[0] = None
            runner.run_inference = original
        result = self.collect(work, model, time.time() - started, fold_input,
                              job.get("family") or controls.get("model-family"))
        result["templates"] = used
        # What the device holds after this fold, as JAX counts it - nvidia-smi
        # sees the allocator's pool, which only grows.
        stats = self.jax.local_devices()[0].memory_stats() or {}
        result["deviceBytesInUse"] = stats.get("bytes_in_use")
        result["status"] += "".join(coverage)
        if converged[0] is not None:
            # ...worded as the WebGPU fold words it.
            distance, ran = converged[0]
            result["status"] += f" · converged at {distance:.2f} Å after {ran} passes"
        return result

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
            pieces = split_merged(msas["merged"], [len(p["sequence"]) for p in proteins])
            for protein, piece in zip(proteins, pieces):
                protein.update(unpairedMsa=piece, pairedMsa="")
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
        # ...and ONE alignment for the page's MSA panel, which reads the chains
        # end to end (see loadIntoViewer): the paired rows side by side, then
        # each chain's unpaired rows with gaps over the others.
        a3m = None
        if len(unpaired) == 1 and unpaired[0]:
            a3m = unpaired[0]
        elif len(unpaired) > 1 and any(unpaired):
            proteins_only = [c.sequence for c in fold_input.chains
                             if isinstance(c, folding_input.ProteinChain)]
            rows = lambda text: [line for line in text.splitlines() if line and not line.startswith(">")]
            blocks = [">101", "".join(proteins_only)]
            paired_rows = [rows(text) for text in paired]
            depth = min((len(r) for r in paired_rows), default=0)
            for at in range(1, depth):
                blocks += [f">paired_{at}", "".join(r[at] for r in paired_rows)]
            for index, text in enumerate(unpaired):
                before = "-" * sum(len(q) for q in proteins_only[:index])
                after = "-" * sum(len(q) for q in proteins_only[index + 1:])
                for at, row in enumerate(rows(text)[1:], start=1):
                    blocks += [f">chain{index + 1}_{at}", before + row + after]
            a3m = "\n".join(blocks) + "\n"
        mean = confidence["meanPlddt"]
        return {
            "jax": True, "model": model, "family": job_family,
            "pdb": cif_to_pdb(cif), "confidence": confidence,
            "tokens": {"chainIds": chain_ids, "resIds": res_ids},
            "chains": chains, "msas": msas,
            "a3m": a3m,
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


def child_loop():
    """The worker proper: imports JAX once, folds a job a line.

    It never restarts itself - see supervise, which starts a fresh one when a
    job needs different chemistry tables or a different model.
    """
    fetched = set(json.loads(os.environ.get("LOCALFOLD_JAX_CODES", "null")) or [])
    written = set(json.loads(os.environ.get("LOCALFOLD_JAX_CCD_WRITTEN", "null")) or [])
    worker = None
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            job = json.loads(line)
            if worker is None:
                fetched |= set(codes_in(job["job"]))
                # ...and a restart for a MODEL, not a code, finds the tables it
                # needs already written.
                if not (written and fetched <= written):
                    emit("status", "fetching chemical definitions")
                    fetch_ccd(sorted(fetched))
                emit("status", "starting JAX")
                worker = Worker()
            emit("result", worker.fold(job))
        except Exception as cause:                            # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            said = str(cause) if isinstance(cause, Refused) else f"{type(cause).__name__}: {cause}"
            emit("result", {"error": said})


def die_with_parent():
    """Linux: the child gets SIGKILL when the supervisor goes, however it goes -
    so stopping a fold (which kills the supervisor) cannot orphan a process
    holding the device."""
    import ctypes
    import signal
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)       # PR_SET_PDEATHSIG


def supervise():
    """Relays jobs to a worker process and starts a NEW one when it must.

    🔴 A NEW LIGAND CODE OR A NEW MODEL IS A NEW PROCESS, AND NOT BY EXEC.
    The CCD tables are read at import, so a code first seen later cannot be
    added in place; and dropping a model's runner does not free its weights -
    a JAX trace kept alive through weakref.finalize's registry holds them as
    constants, so every model folded stayed on the device: 1.6 GB in use after
    one, 16 GB after eleven, and OpenDDE, ESMFold2 and the multimer ran out of
    memory on a 40 GB A100 (jax.clear_caches() and gc do not reach it). This
    used `os.execv`, which frees a GPU and NOT a TPU: exec keeps the process
    and its descriptors, libtpu still held the device, and every fold after
    the first on a TPU v5e died with "Unable to initialize backend 'tpu'". A
    child that EXITS gives both back. It takes its executables from the
    on-disk compile cache, which is what a change of model costs anyway.
    """
    emit("jax-ready", {"at": int(time.time() * 1000)})
    child, fetched, written, loaded = None, set(), set(), None
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            job = json.loads(line)
            codes = set(codes_in(job["job"]))
        except Exception as cause:                            # noqa: BLE001
            emit("result", {"error": f"{type(cause).__name__}: {cause}"})
            continue
        family = job.get("family") or job.get("controls", {}).get("model-family")
        if child is not None and (not codes <= fetched or family != loaded):
            emit("status", f"restarting JAX for {', '.join(sorted(codes - fetched))}"
                 if not codes <= fetched else f"restarting JAX for {family}")
            child.stdin.close()
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            written, child = set(fetched), None
        if child is None:
            fetched |= codes
            env = {**os.environ, "LOCALFOLD_JAX_CHILD": "1",
                   "LOCALFOLD_JAX_CODES": json.dumps(sorted(fetched)),
                   "LOCALFOLD_JAX_CCD_WRITTEN": json.dumps(sorted(written))}
            child = subprocess.Popen([sys.executable, os.path.abspath(__file__)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                     bufsize=1, env=env, preexec_fn=die_with_parent)
        loaded = family
        child.stdin.write(line if line.endswith("\n") else line + "\n")
        child.stdin.flush()
        for out in child.stdout:
            OUT.write(out)
            OUT.flush()
            try:
                if json.loads(out).get("kind") == "result":
                    break
            except ValueError:
                pass
        else:
            emit("result", {"error": "the JAX worker exited"})
            child = None


def main():
    if os.environ.get("LOCALFOLD_JAX_CHILD") == "1":
        child_loop()
    else:
        supervise()


if __name__ == "__main__":
    main()
