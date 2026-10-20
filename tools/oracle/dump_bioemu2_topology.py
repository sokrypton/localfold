"""BioEmu2's bonded graph, steering tables and physicality report, from af3-any-model's port.

    cd <sokrypton/alphafold3 colab checkout> && PYTHONPATH=src ~/.venv-lfjax/bin/python \
      ~/localfold/tools/oracle/dump_bioemu2_topology.py --job JOB.json --structure SAMPLE.cif [--out PATH]

What shared/bioemu2/{graph,steering-tables,physicality}.js are held to: the
colab branch's pipeline/bioemu2_steering.py (Graph, tables) and
pipeline/bioemu2_physicality.py (check), on the topology bioemu2_features
builds. The physicality report is taken on the sample and on copies of it with
Gaussian noise of 0.1, 0.3 and 1 A, so that the checks that only fire on a bad
structure fire. Plain JSON (no tensors here are large).
"""
import argparse
import json
import os

# 🔴 OR AN A100 RUNS THE REFERENCE'S float32 einsums IN TF32: the steering oracle's rigid augmentation read 2.7e-4
# off on its FIRST step until this was set
os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION', 'highest')

import numpy as np

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.model.pipeline import bioemu2_features as bf
from alphafold3.model.pipeline import bioemu2_physicality as phys
from alphafold3.model.pipeline import bioemu2_steering as st

NOISE = (0.0, 0.1, 0.3, 1.0)


def plain(x):
  if isinstance(x, np.ndarray):
    return x.tolist()
  if isinstance(x, (np.integer,)):
    return int(x)
  if isinstance(x, (np.floating,)):
    return float(x)
  if isinstance(x, dict):
    return {str(k): plain(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [plain(v) for v in x]
  return x


def _frame(path, keys, f):
  """The sample's atoms in BioEmu2's order. An atom the sample lacks - colab writes the AF3 layout, which
  drops a glycan's O1 that BioEmu2 predicts - is placed at its residue's sampled centroid plus its template
  offset, so every potential sees a whole molecule."""
  from alphafold3 import structure
  with open(path) as fh:
    s = structure.from_mmcif(fh.read())
  where = {(str(c), int(r), str(n)): (x, y, z) for c, r, n, x, y, z in zip(
      s.chain_id, s.res_id, s.atom_name, s.atom_x, s.atom_y, s.atom_z)}
  ref = np.asarray(f['template_coords'], np.float64)
  res = np.asarray(f['ref_space_uid'])
  x = np.full((len(keys), 3), np.nan)
  for i, k in enumerate(keys):
    if k in where:
      x[i] = where[k]
  for r in set(res[:len(keys)].tolist()):
    sel = np.flatnonzero(res[:len(keys)] == r)
    have = sel[np.isfinite(x[sel]).all(1)]
    miss = sel[~np.isfinite(x[sel]).all(1)]
    if len(miss):
      shift = x[have].mean(0) - ref[have].mean(0)
      x[miss] = ref[miss] + shift
      print(f'placed {len(miss)} atom(s) the sample lacks from the template:', [keys[m] for m in miss])
  return x


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--job', required=True)
  ap.add_argument('--structure', required=True)
  ap.add_argument('--out', default=None)
  a = ap.parse_args()
  with open(a.job) as fh:
    fold_input = folding_input.Input.from_json(fh.read())
  ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
  smiles = {ch.id: bf.smiles_template(ch.smiles) for ch in fold_input.chains if getattr(ch, 'smiles', None)}
  f, keys, topo = bf.featurise(fold_input, ccd, smiles_templates=smiles, return_topology=True)
  g = st.Graph(topo, ccd)
  topo = dict(topo, n_pad=f['template_coords'].shape[0])
  tables = st.tables(topo, ccd)
  x0 = _frame(a.structure, keys, f)
  rng = np.random.default_rng(0)
  frames = np.stack([x0 + s * rng.standard_normal(x0.shape) for s in NOISE])
  where = {k: i for i, k in enumerate(keys)}
  declared = [(where[tuple(p[0])], where[tuple(p[1])]) for p in (fold_input.bonded_atom_pairs or [])]
  report = {}
  for mode in (*phys.THRESHOLDS, 'no_filter'):
    ok, reasons, stats = phys.check(topo, g.bonds, ccd, frames, mode, declared)
    report[mode] = {'ok': ok, 'reasons': reasons, 'stats': stats}
  # the potentials on colab's own tables: energies per spec and jax.grad per spec, on the frames at three times
  import jax
  import jax.numpy as jnp
  from alphafold3.model.network import bioemu2_steering as ns
  host = ns.host_tables({k: v for k, v in tables.items() if k != 'graph_bonds'})
  n_pad = topo['n_pad']
  c = np.zeros((len(NOISE), n_pad, 3), np.float32)
  c[:, :len(keys)] = frames
  cj = jnp.asarray(c)
  tj = {k: jnp.asarray(v) for k, v in host.items()}
  steer = {'host': host, 'times': []}
  for t_norm in (0.9, 0.5, 0.1):
    prm = ns.step_parameters(t_norm)
    pj = {k: jnp.asarray(v) for k, v in prm.items()}
    en = np.asarray(ns.energies(cj, tj, pj))
    grads = []
    for i in range(len(ns.DEFAULT)):
      grads.append(np.asarray(jax.grad(lambda x: ns.energies(x, tj, pj)[i].sum())(cj)))
    steer['times'].append({'t': t_norm, 'params': prm, 'energies': en, 'grads': grads,
                           'interval': ns.guidance_interval(t_norm)})
  out = {
      'steer': steer, 'coords': c,
      'graph': {'bonds': g.bonds, 'angles': g.angles, 'exclusions': sorted(g.nonbonded_exclusions()),
                'labels': [g.labels[i] for i in range(len(keys))], 'ordinary_x': sorted(g.ordinary_x),
                'peptide_edges': sorted(g.peptide_edges)},
      'tables': tables, 'noise': NOISE, 'frames': frames, 'physicality': report,
  }
  path = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'oracle-dumps',
                               'bioemu2-topology-' + os.path.splitext(os.path.basename(a.job))[0] + '.json')
  with open(path, 'w') as fh:
    json.dump(plain(out), fh)
  print('wrote', path, f'{os.path.getsize(path) / 1e6:.1f} MB')
  for mode, r in report.items():
    print(f'{mode:26s} ok {r["ok"].tolist()}  reasons {r["reasons"]}')


if __name__ == '__main__':
  main()
