"""Native AlphaFold 3 on this box, at settings LocalFold can actually match.

    .venv/bin/python tools/oracle/bench_af3_native.py --input <fold.json> \
        --steps 1 --recycles 0 --samples 1 --num-msa 1

Prints one JSON record: the featurisation cost, the first (JIT) call, and the
median of `--repeats` steady-state calls. The timed region is `run_inference`
alone - device_put, the forward pass, and the readback of every output to host.
It is NOT the CLI's wall clock, which additionally carries the data pipeline,
mmCIF writing and extraction, and which on this box is several seconds.

🔴 THE STOCK RUNNER CANNOT EXPRESS LocalFold's DEFAULTS, IN THREE SEPARATE
WAYS, AND EACH ONE FLATTERS JAX OR US IF LEFT ALONE.

  - `--num_recycles` is declared `lower_bound=1`, and `model.num_trunk_passes`
    returns `num_recycles + 1`, so the flag's FLOOR is two trunk passes.
    fold.js defaults to `recycles: 0`, meaning one. Comparing the flag's
    minimum against our default gives JAX twice the trunk work.
  - `steps` is not a flag at all. It lives at
    `config.heads.diffusion.eval.steps` and defaults to 200.
  - `evoformer.num_msa` defaults to **1024**. A single-sequence input still
    runs the MSA stack over 1024 rows, of which 1023 are padding, where
    LocalFold runs exactly one. The featuriser separately pads every MSA
    feature to `msa_crop_size` (16384) on the host; `--crop-msa` trims the
    batch to what the model will actually read, so the upload is not 16384
    rows either. Measured: cropping is worth nothing (55.0 vs 54.8 ms), which
    is itself the finding - the cost was never the upload.

🔴 AND DO NOT MEASURE THE DIFFUSION BY SWEEPING `--num_diffusion_samples`.
That is how this repository got the number wrong for a whole session.
`diffusion_head.py` is

    apply_denoising_step = hk.vmap(apply_denoising_step, in_axes=(0, None), ...)

so samples are a BATCH AXIS INSIDE one denoising step and the `scan` over
`config.steps` runs once no matter how many samples there are. Going from 1 to
5 samples widens each kernel's leading dimension; it does not add four more
200-step trajectories. At 68 tokens those kernels are tiny, so the marginal
sample is nearly free - 0.175 s - and dividing it by 200 gave "0.875 ms a
diffusion step", about a tenth of the truth. Sweep `--steps` instead: it is
linear, and the fit holds to 0.4% from 1 step out to 50.
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.expanduser('~/af3fork'))
os.environ.setdefault('JAX_PLATFORMS', 'cuda')

import numpy as np
from absl import flags
import run_alphafold as ra
flags.FLAGS(['run_alphafold'])

import jax
from alphafold3.common import folding_input
from alphafold3.constants import decoded_ccd
from alphafold3.data import featurisation

ap = argparse.ArgumentParser()
ap.add_argument('--input', required=True, help='an AF3 fold-input JSON')
ap.add_argument('--steps', type=int, default=1)
ap.add_argument('--recycles', type=int, default=0, help='trunk passes = recycles + 1')
ap.add_argument('--samples', type=int, default=1)
ap.add_argument('--num-msa', type=int, default=1)
ap.add_argument('--crop-msa', type=int, default=1,
                help='trim the host-side MSA padding to --num-msa rows')
ap.add_argument('--repeats', type=int, default=5)
ap.add_argument('--model-dir', default=os.path.expanduser('~/af3_official_weights'))
a = ap.parse_args()

fold_input = folding_input.Input.from_json(open(a.input).read())
ccd = decoded_ccd.get_ccd(user_ccd=fold_input.user_ccd)
t0 = time.time()
ex = featurisation.featurise_input(fold_input=fold_input, buckets=None, ccd=ccd,
                                   verbose=False)[0]
featurise_s = time.time() - t0

uploaded = None
if a.crop_msa:
    for key, value in list(ex.items()):
        if hasattr(value, 'shape') and value.ndim >= 1 and value.shape[0] == 16384:
            uploaded = a.num_msa
            ex[key] = value[:a.num_msa]

cfg = ra.make_model_config(num_diffusion_samples=a.samples, num_recycles=a.recycles,
                           model_name='alphafold3')
cfg.heads.diffusion.eval.steps = a.steps
cfg.evoformer.num_msa = a.num_msa

runner = ra.ModelRunner(config=cfg, device=jax.devices('cuda')[0], model_dir=a.model_dir)
_ = runner.model_params            # the weight load is not part of a call

t0 = time.time()
runner.run_inference(ex, jax.random.PRNGKey(0))
first_s = time.time() - t0

times = []
for i in range(a.repeats):
    t0 = time.time()
    runner.run_inference(ex, jax.random.PRNGKey(100 + i))
    times.append(time.time() - t0)
times.sort()

print(json.dumps({
    'tokens': int(ex['token_index'].shape[-1]),
    'steps': a.steps, 'trunk_passes': a.recycles + 1, 'samples': a.samples,
    'num_msa': a.num_msa,
    'msa_rows_uploaded': int(ex['msa'].shape[0]) if 'msa' in ex else uploaded,
    'featurise_s': round(featurise_s, 3),
    'first_call_s': round(first_s, 3),
    # 🔴 THE MEDIAN, NOT THE MEAN. One call in about twenty takes ~2.8 s on this
    # box, which pulls a 53 ms median to a 190 ms mean. See docs/A100.md.
    'median_s': round(times[len(times) // 2], 4),
    'min_s': round(times[0], 4),
    'all_s': [round(t, 4) for t in times],
}, indent=2))
