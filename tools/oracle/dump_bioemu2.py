"""BioEmu2's features and every stage of its network, from af3-any-model's own port.

    cd <sokrypton/alphafold3 colab checkout> && PYTHONPATH=src ~/.venv-lfjax/bin/python \
      ~/localfold/tools/oracle/dump_bioemu2.py --job JOB.json --model-dir ~/ported/bioemu2 \
      [--structure SAMPLE.cif] [--mode structure|equilibrium] [--contacts C.json] [--out PATH]

The reference is the colab branch's `network/bioemu2.py` and
`pipeline/bioemu2_features.py` (a transcription of microsoft/bioemu2 @7846769,
held to the vendor at 0.000 A by its own parity harness), so this dumps what
that port computes, stage by stage, in float32 at 'highest' matmul precision:

  feats.*        the whole feature dict bioemu2_features.featurise builds
  s_inputs, s_init, z_init, pass<k>.single / pass<k>.pair (k = 1..4), distogram
  ctx.z, ctx.q0, ctx.p   the denoiser's static context
  denoise<k>.{sigma, x, out}   one denoiser call at several noise levels

🔴 THE DENOISER IS FED A REAL STRUCTURE, re-noised. A denoise oracle on noise
alone is a wiring test - CLAUDE.md's row on dump_af3_denoise.py, which passed
while side chains were 6.7x too long. --structure names a sample (run_alphafold
--model bioemu2 writes one); its atoms are read in BioEmu2's own order through
x0_from_mmcif, so an atom the sample lacks is an error rather than a zero.

A propagation arm (`denoise_dyn`) conditions the same call on that structure at a
1 ns lag, which is the only path through `coords_cond`, `dt` and the
damped-harmonic time embedding.

Arrays are written {"shape", "dtype", "b64"} (little-endian float32 / int32),
which tools/check-bioemu2-*.js decode.
"""
import argparse
import base64
import json
import os

os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION', 'highest')

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.model import params as afp
from alphafold3.model.network import bioemu2 as net
from alphafold3.model.pipeline import bioemu2_features as bf

SIGMAS = (160.0, 16.0, 1.0, 0.05)


def enc(a):
  a = np.asarray(a)
  if a.dtype == np.bool_ or np.issubdtype(a.dtype, np.integer):
    a = a.astype('<i4')
    dtype = 'i32'
  else:
    a = a.astype('<f4')
    dtype = 'f32'
  return {'shape': list(a.shape), 'dtype': dtype, 'b64': base64.b64encode(a.tobytes()).decode()}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--job', required=True)
  ap.add_argument('--model-dir', default=os.path.expanduser('~/ported/bioemu2'))
  ap.add_argument('--structure', default=None)
  ap.add_argument('--mode', default='structure', choices=('structure', 'equilibrium'))
  ap.add_argument('--contacts', default=None)
  ap.add_argument('--passes', type=int, default=4)
  ap.add_argument('--out', default=None)
  a = ap.parse_args()

  with open(a.job) as fh:
    fold_input = folding_input.Input.from_json(fh.read())
  ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
  contacts = bf.load_contacts(a.contacts) if a.contacts else None
  # a SMILES ligand's template is colab's seeded RDKit draw; recorded, so the JS side is handed the same one
  smiles = {ch.id: bf.smiles_template(ch.smiles) for ch in fold_input.chains if getattr(ch, 'smiles', None)}
  f, keys = bf.featurise(fold_input, ccd, mode=a.mode, contacts=contacts, smiles_templates=smiles)
  record = {f'feats.{k}': enc(v) for k, v in f.items()}
  for cid, t in smiles.items():
    record[f'smiles.{cid}'] = {'names': list(t.names), 'element': enc(t.element), 'charge': enc(t.charge),
                               'coords': enc(t.coords), 'bonds': enc(t.bonds.T)}
  record['atom_keys'] = [list(k) for k in keys]

  params = afp.get_model_haiku_params(model_dir=a.model_dir)
  params = {k: v for k, v in params.items() if k.startswith('diffuser/~/bioemu2')}

  class Holder(hk.Module):
    def __init__(self):
      super().__init__(name='diffuser')
      self.b = net.BioEmu2()

  def run(fn):
    return jax.jit(hk.transform(lambda *x: fn(Holder().b, *x)).apply)

  fj = {k: jnp.asarray(v) for k, v in f.items()}
  for k in ('atom_pad_mask', 'token_pad_mask', 'atom_mask', 'is_dynamics_conditioned'):
    fj[k] = fj[k].astype(bool)
  key = jax.random.PRNGKey(0)

  s_inputs = run(lambda b, f: b.input_embedder(f))(params, key, fj)
  s_init, z_init = run(lambda b, f, si: b.trunk_init(f, si))(params, key, fj, s_inputs)
  record.update(s_inputs=enc(s_inputs), s_init=enc(s_init), z_init=enc(z_init))
  s, z = jnp.zeros_like(s_init), jnp.zeros_like(z_init)
  step = run(lambda b, f, si, s0, z0, s, z: b.trunk_pass(f, si, s0, z0, s, z))
  for k in range(a.passes):
    s, z = step(params, key, fj, s_inputs, s_init, z_init, s, z)
    record[f'pass{k + 1}.single'] = enc(s)
    record[f'pass{k + 1}.pair'] = enc(z)
    print(f'pass {k + 1}: single rms {float(jnp.sqrt((s ** 2).mean())):.4f}'
          f'  pair rms {float(jnp.sqrt((z ** 2).mean())):.4f}')
  record['distogram'] = enc(run(lambda b, z: b.distogram(z))(params, key, z))
  ctx = run(lambda b, f, s, z: b.static_context(f, s, z))(params, key, fj, s, z)
  record.update({f'ctx.{k}': enc(v) for k, v in ctx.items()})

  if a.structure:
    n_pad = f['template_coords'].shape[0]
    x0 = bf.x0_from_mmcif(a.structure, keys)
    clean = np.zeros((n_pad, 3), np.float32)
    clean[:len(x0)] = x0 - x0.mean(0)
    record['clean'] = enc(clean)
    rng = np.random.default_rng(0)
    den = run(lambda b, f, ctx, si, s, x, sg: b.denoise(f, ctx, si, s, x, sg))
    for k, sigma in enumerate(SIGMAS):
      x = clean + sigma * rng.standard_normal(clean.shape).astype(np.float32)
      x[len(x0):] = 0
      out = den(params, key, fj, ctx, s_inputs, s, jnp.asarray(x), jnp.float32(sigma))
      record[f'denoise{k}.sigma'] = sigma
      record[f'denoise{k}.x'] = enc(x)
      record[f'denoise{k}.out'] = enc(out)
      err = np.sqrt(((np.asarray(out)[:len(x0)] - clean[:len(x0)]) ** 2).sum(-1).mean())
      print(f'denoise sigma {sigma:7.2f}: rms to the clean structure {err:.3f} A')
    # propagation: conditioned on the clean frame at a 1 ns lag
    g = bf.add_propagation(f, x0, 1.0)
    gj = dict(fj, coords_cond=jnp.asarray(g['coords_cond']), dt=jnp.float32(g['dt']),
              is_dynamics_conditioned=jnp.bool_(True))
    sigma = 4.0
    x = clean + sigma * rng.standard_normal(clean.shape).astype(np.float32)
    x[len(x0):] = 0
    out = den(params, key, gj, ctx, s_inputs, s, jnp.asarray(x), jnp.float32(sigma))
    record.update({'denoise_dyn.sigma': sigma, 'denoise_dyn.x': enc(x), 'denoise_dyn.out': enc(out),
                   'denoise_dyn.coords_cond': enc(g['coords_cond']), 'denoise_dyn.dt': float(g['dt'])})

  out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'oracle-dumps',
                              'bioemu2-' + os.path.splitext(os.path.basename(a.job))[0] + '.json')
  with open(out, 'w') as fh:
    json.dump({'mode': a.mode, 'stages': record}, fh)
  print('wrote', out, f'{os.path.getsize(out) / 1e6:.1f} MB')


if __name__ == '__main__':
  main()
