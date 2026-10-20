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
  x0 = bf.x0_from_mmcif(a.structure, keys).astype(np.float64)
  rng = np.random.default_rng(0)
  frames = np.stack([x0 + s * rng.standard_normal(x0.shape) for s in NOISE])
  where = {k: i for i, k in enumerate(keys)}
  declared = [(where[tuple(p[0])], where[tuple(p[1])]) for p in (fold_input.bonded_atom_pairs or [])]
  report = {}
  for mode in (*phys.THRESHOLDS, 'no_filter'):
    ok, reasons, stats = phys.check(topo, g.bonds, ccd, frames, mode, declared)
    report[mode] = {'ok': ok, 'reasons': reasons, 'stats': stats}
  out = {
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
