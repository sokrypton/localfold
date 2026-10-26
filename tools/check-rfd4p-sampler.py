#!/usr/bin/env python3
"""Hold the RFD4 sampler's schedules and steps to upstream's own, in every mode it offers.

    ~/venv_rfd4p/bin/python tools/check-rfd4p-sampler.py

The released sampler is `vf_tsr`, and that is all both ports ran; upstream also offers `vf` (the
flow's ODE) and `sc` (the SDE), a `gt` power and clamp, other schedule shapes, and re-centring at
every step. This calls upstream's own `sample_schedule` and `RDNFlowMatcher.simulation_step` - no
model, no checkpoint - and requires shared/rfd4p/flow.js to give the same numbers:

- the time and noise schedules of both modalities, over a spread of configurations;
- one step of each mode, on random states, at times inside and outside each mode's stochastic
  window, with and without `center_every_step`. The noise is the one thing two implementations
  cannot share, so the steps run with `sc_scale_noise = 0`, which leaves every mode's drift - the
  score terms, the TSR and endpoint rescalings - and removes only the draw.

Upstream computes in float32 and flow.js in double with float32 stores, so the bound is float32's
ACCUMULATED: `torch.logspace` and a power in float32 put the released schedule 6e-6 relative from
the double one, and a step inherits that - measured, 2e-5 at worst over these cases. A formula that
is wrong is wrong at order one.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT = {
    "bb_ca": {"schedule": {"mode": "log", "p": 2.0, "tail_steps": 10, "tail_step_multiplier": 2},
              "gt": {"mode": "1/t", "p": 1.0, "clamp_val": None},
              "simulation_step_params": {"sampling_mode": "vf_tsr", "sc_scale_noise": 1.0, "t_lim_ode": 0.95,
                                         "t_lim_ode_below": 0.02, "center_every_step": False, "tsr_k": 16.0,
                                         "tsr_sigma": 3.0}},
    "local_latents": {"schedule": {"mode": "power", "p": 2.0, "tail_steps": 10, "tail_step_multiplier": 2},
                      "gt": {"mode": "tan", "p": 1.0, "clamp_val": None},
                      "simulation_step_params": {"sampling_mode": "vf_tsr", "sc_scale_noise": 1.0, "t_lim_ode": 0.95,
                                                 "t_lim_ode_below": 0.02, "center_every_step": False, "tsr_k": 4.0,
                                                 "tsr_sigma": 12.0}},
}


def merged(override: dict) -> dict:
    out = json.loads(json.dumps(DEFAULT))
    for dm, parts in override.items():
        for part, values in parts.items():
            out[dm][part].update(values)
    return out


# (name, overrides, nsteps)
CASES = [
    ("released", {}, 60),
    ("released-25", {}, 25),
    ("sc", {"bb_ca": {"simulation_step_params": {"sampling_mode": "sc"}},
            "local_latents": {"simulation_step_params": {"sampling_mode": "sc"}}}, 60),
    ("vf", {"bb_ca": {"simulation_step_params": {"sampling_mode": "vf"}},
            "local_latents": {"simulation_step_params": {"sampling_mode": "vf"}}}, 40),
    ("gt-power", {"bb_ca": {"gt": {"p": 2.0, "clamp_val": 5.0}}, "local_latents": {"gt": {"p": 0.5}}}, 60),
    ("schedules", {"bb_ca": {"schedule": {"mode": "uniform", "tail_steps": 0}},
                   "local_latents": {"schedule": {"mode": "log", "p": 1.5, "tail_steps": 5, "tail_step_multiplier": 3}}}, 50),
    ("ode-gt", {"bb_ca": {"gt": {"mode": "ode"}}, "local_latents": {"gt": {"mode": "1-t/t"}}}, 30),
]


def upstream(cases):
    import torch
    from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
    from rfproteina.flow_matching.rdn_flow_matcher import RDNFlowMatcher

    class Fake:
        data_modes = ["bb_ca", "local_latents"]

    out = {}
    gen = torch.Generator().manual_seed(0)
    for name, override, nsteps in cases:
        cfg = merged(override)
        ts, gt = ProductSpaceFlowMatcher.sample_schedule(Fake(), nsteps, cfg)
        entry = {"ts": {k: v.double().tolist() for k, v in ts.items()},
                 "gt": {k: v.double().tolist() for k, v in gt.items()}, "steps": []}
        for dm, dim in (("bb_ca", 3), ("local_latents", 32)):
            fm = RDNFlowMatcher(dim=dim)
            for center in (False, True):
                params = dict(cfg[dm]["simulation_step_params"], sc_scale_noise=0.0, center_every_step=center)
                tsd = ts[dm]
                for i in (0, 1, nsteps // 2, nsteps - 3, nsteps - 1):
                    x = torch.randn(1, 7, dim, generator=gen)
                    v = torch.randn(1, 7, dim, generator=gen)
                    emit = torch.tensor([[1, 1, 0, 1, 1, 1, 0]], dtype=torch.bool)
                    t = tsd[i:i + 1]
                    dt = float(tsd[i + 1] - tsd[i])
                    nxt = fm.simulation_step(x.clone(), {"v": v}, t, dt, gt[dm][i], emit, params)
                    entry["steps"].append({"dm": dm, "center": center, "i": i, "x": x.flatten().tolist(),
                                           "v": v.flatten().tolist(), "emit": emit.flatten().int().tolist(),
                                           "t": float(t[0]), "dt": dt, "gt": float(gt[dm][i]),
                                           "next": nxt.flatten().tolist()})
        out[name] = entry
    return out


def page(cases, steps):
    script = """
import { readFileSync } from "node:fs";
import { resolveSampling, sampleSchedule, simulationStep } from "./shared/rfd4p/flow.js";
const { cases, steps } = JSON.parse(readFileSync(0, "utf8"));
const out = {};
for (const [name, override, nsteps] of cases) {
  const sampling = { ...resolveSampling(override), nsteps };
  const sched = sampleSchedule(sampling);
  const got = { ts: {}, gt: {}, steps: [] };
  for (const dm of ["bb_ca", "local_latents"]) { got.ts[dm] = Array.from(sched[dm].ts); got.gt[dm] = Array.from(sched[dm].gt); }
  for (const s of steps[name]) {
    const step = { ...sampling[s.dm].step, sc_scale_noise: 0, center_every_step: s.center };
    const x = Float32Array.from(s.x);
    const zero = { randn: (k) => new Float32Array(k) };
    simulationStep({ x, v: Float32Array.from(s.v), t: s.t, dt: s.dt, gt: s.gt, emit: s.emit,
      dim: s.dm === "bb_ca" ? 3 : 32, step, rng: zero });
    got.steps.push(Array.from(x));
  }
  out[name] = got;
}
process.stdout.write(JSON.stringify(out));
"""
    r = subprocess.run(["node", "--input-type=module", "-e", script], cwd=REPO, capture_output=True, text=True,
                       input=json.dumps({"cases": cases, "steps": steps}))
    if r.returncode != 0:
        sys.exit(f"node failed: {r.stderr[-2000:]}")
    return json.loads(r.stdout)


def main() -> int:
    cases = [[n, o, s] for n, o, s in CASES]
    up = upstream(cases)
    got = page(cases, {n: up[n]["steps"] for n, _, _ in cases})
    bad = 0
    for name, _, _ in cases:
        worst = 0.0
        for kind in ("ts", "gt"):
            for dm in ("bb_ca", "local_latents"):
                a, b = up[name][kind][dm], got[name][kind][dm]
                if len(a) != len(b):
                    worst = float("inf")
                    continue
                for x, y in zip(a, b, strict=True):
                    worst = max(worst, abs(x - y) / max(1.0, abs(x)))
        step_worst = 0.0
        for s, mine in zip(up[name]["steps"], got[name]["steps"], strict=True):
            for x, y in zip(s["next"], mine, strict=True):
                step_worst = max(step_worst, abs(x - y) / max(1.0, abs(x)))
        ok = worst < 2e-5 and step_worst < 1e-4
        bad += not ok
        print(f"  {name:12s} schedules {worst:.1e}  steps {step_worst:.1e}  {'ok' if ok else 'DIFFERS'}")
    print(f"{len(cases)} configurations, {bad} differing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
