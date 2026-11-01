"""BioEmu2's steered samplers on a toy denoiser, from af3-any-model's port.

    cd <sokrypton/alphafold3 colab checkout> && PYTHONPATH=src ~/.venv-lfjax/bin/python \
      ~/localfold/tools/oracle/dump_bioemu2_steering.py --topology oracle-dumps/bioemu2-topology-<t>.json \
      [--steps 20] [--particles 4] [--out PATH]

What shared/bioemu2/sampler.js is held to, isolated from any network: colab's
network/bioemu2_steering.steered_sample (edm_gd and edm_fk) and fkc_sample
(edm_fkc) run with the SAME injected draws (initial positions, every step's
rotation, translation and noise, every resampling's uniform) and a toy
denoiser both sides can compute and differentiate exactly,

    D(x, sigma) = c_skip(sigma) x + (1 - c_skip(sigma)) ref,  c_skip = sigma_data^2 / (sigma^2 + sigma_data^2),

where ref is the topology dump's sample. The potentials are colab's on that
dump's own host tables. Every particle's final coordinates, and the FK trace
(weights, indices, energies per step), are written.
"""
import argparse
import json
import os

# 🔴 OR AN A100 RUNS THE REFERENCE'S float32 einsums IN TF32: the steering oracle's rigid augmentation read 2.7e-4
# off on its FIRST step until this was set
os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION', 'highest')

import jax
import jax.numpy as jnp
import numpy as np

from alphafold3.model.network import bioemu2_steering as ns

SD = 16.0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--topology', required=True)
  ap.add_argument('--steps', type=int, default=20)
  ap.add_argument('--particles', type=int, default=4)
  ap.add_argument('--gd-steps', type=int, default=20)
  ap.add_argument('--nudge', type=float, default=0.0,
                  help='scale the initial draw by (1 + nudge): the reference against itself, for the noise floor')
  ap.add_argument('--floor', type=float, default=1e-6)
  ap.add_argument('--out', default=None)
  a = ap.parse_args()
  text = open(a.topology).read().replace('NaN', 'null')
  dump = json.loads(text)
  host = {k: np.asarray([np.nan if x is None else x for x in v] if isinstance(v, list) and v and not isinstance(v[0], list)
                        else v) for k, v in dump['steer']['host'].items()}
  for k, v in host.items():
    if v.dtype == object:
      host[k] = np.asarray(v.tolist(), np.float32)
  for k in ('atom_pad_mask', 'interchain_steric_atom_mask', 'polymer_steric_atom_mask', 'rdkit_bounds_bond_mask',
            'rdkit_bounds_angle_mask', 'chiral_atom_orientations', 'stereo_bond_orientations'):
    host[k] = host[k].astype(bool)
  for k in ('atom_chain_id', 'chain_sizes', 'symmetric_chain_index', 'interchain_bonded_pair_key',
            'polymer_bonded_pair_key', 'polymer_local_clash_index', 'rdkit_bounds_index', 'chiral_atom_index',
            'stereo_bond_index', 'planar_improper_index', 'symmetric_chain_pairs', 'bond_index', 'angle_index'):
    host[k] = host[k].astype(np.int64)
  ref = np.asarray(dump['coords'][0], np.float32)
  mask = host['atom_pad_mask']
  P, S, N = a.particles, a.steps, ref.shape[0]
  rng = np.random.default_rng(1)
  rots = []
  for _ in range(S * P):
    q, _r = np.linalg.qr(rng.standard_normal((3, 3)))
    rots.append(q * np.sign(np.linalg.det(q)))
  inj = {'x0': rng.standard_normal((P, N, 3)).astype(np.float32),
         'rot': np.asarray(rots, np.float32).reshape(S, P, 3, 3),
         'trans': rng.standard_normal((S, P, 3)).astype(np.float32),
         'eps': rng.standard_normal((S, P, N, 3)).astype(np.float32),
         'uniform': rng.uniform(size=(S, 1)).astype(np.float32)}
  inj['x0'] = (inj['x0'] * (1 + a.nudge)).astype(np.float32)
  refj = jnp.asarray(ref)

  def toy(x, sigma, comp=None):
    sigma = jnp.asarray(sigma, jnp.float32)
    c = SD ** 2 / (sigma ** 2 + SD ** 2)
    c = c.reshape(c.shape + (1,) * (x.ndim - c.ndim)) if c.ndim else c
    return c * x + (1 - c) * refj[None]

  out = {'draws': {k: v.tolist() for k, v in inj.items()}, 'steps': S, 'particles': P, 'ref': ref.tolist(),
         'gd_steps': a.gd_steps}
  for name, interval in (('edm_gd', None), ('edm_fk', 3)):
    ns.TRACE = []
    x = jax.jit(lambda: ns.steered_sample(lambda xx, s, c: toy(xx, s), host, mask, P, jax.random.PRNGKey(0),
                                          num_steps=S, step_scale=1.5, resampling_interval=interval, injected=inj,
                                          num_gd_steps=a.gd_steps,
                                          scan=jax.lax.scan))()
    out[name] = {'x': np.asarray(x).tolist(),
                 'trace': [[np.asarray(v).tolist() for v in row] for row in ns.TRACE]}
    print(name, 'rms of the particles to ref', float(jnp.sqrt(((x - refj[None]) ** 2).sum(-1).mean())))
  ns.TRACE = None
  # each step's state and log-weights, recorded through the two functions every fkc step calls first and last
  fkc_trace = {'x': [], 'logw': []}
  augment, softmax = ns._augment, ns.grouped_softmax
  def traced_augment(x, mask_, rot, trans):
    if x.shape[0] == P:
      jax.debug.callback(lambda v: fkc_trace['x'].append(np.asarray(v).tolist()), x)
    return augment(x, mask_, rot, trans)
  def traced_softmax(logw, group):
    jax.debug.callback(lambda v: fkc_trace['logw'].append(np.asarray(v).tolist()), logw)
    return softmax(logw, group)
  ns._augment, ns.grouped_softmax = traced_augment, traced_softmax
  x, bad = jax.jit(lambda: ns.fkc_sample(lambda xx, s, c: toy(xx, s), host, mask, P, jax.random.PRNGKey(0),
                                         num_steps=S, injected=inj, scan=jax.lax.scan, grad=jax.value_and_grad))()
  ns._augment, ns.grouped_softmax = augment, softmax
  out['edm_fkc'] = {'x': np.asarray(x).tolist(), 'bad': int(bad), 'trace': fkc_trace}
  print('edm_fkc rms to ref', float(jnp.sqrt(((x - refj[None]) ** 2).sum(-1).mean())), 'non-finite step', int(bad))
  if a.floor:
    # the reference against ITSELF, the initial draw scaled by (1 + floor): how far float32 alone moves each sampler
    nudged = dict(inj, x0=(inj['x0'] * (1 + a.floor)).astype(np.float32))
    for name, interval in (('edm_gd', None), ('edm_fk', 3)):
      y = ns.steered_sample(lambda xx, s, c: toy(xx, s), host, mask, P, jax.random.PRNGKey(0), num_steps=S,
                            step_scale=1.5, resampling_interval=interval, injected=nudged, num_gd_steps=a.gd_steps,
                            scan=jax.lax.scan)
      x = np.asarray(out[name]['x'])
      out[name]['floor'] = float(np.sqrt(((np.asarray(y) - x) ** 2).sum() / (x ** 2).sum()))
      print(name, 'floor (itself, x0 nudged by %g): %.3e' % (a.floor, out[name]['floor']))
    y, _ = ns.fkc_sample(lambda xx, s, c: toy(xx, s), host, mask, P, jax.random.PRNGKey(0), num_steps=S,
                         injected=nudged, scan=jax.lax.scan, grad=jax.value_and_grad)
    x = np.asarray(out['edm_fkc']['x'])
    out['edm_fkc']['floor'] = float(np.sqrt(((np.asarray(y) - x) ** 2).sum() / (x ** 2).sum()))
    print('edm_fkc floor: %.3e' % out['edm_fkc']['floor'])
  path = a.out or a.topology.replace('bioemu2-topology-', 'bioemu2-steering-')
  with open(path, 'w') as fh:
    json.dump(out, fh)
  print('wrote', path, f'{os.path.getsize(path) / 1e6:.1f} MB')


if __name__ == '__main__':
  main()
