"""PyTorch's cost for one AlphaFold 3 pairformer block trained - the baseline for cuda/grad/bench_af3_block.cu.

    ~/venv_ef2/bin/python cuda/grad/bench_torch_af3_block.py [--n 384] [--tf32] [--compile]

AF3's block at its widths (Cz 128, Cs 384; triangle attention 4 heads of 32; single attention 16 heads of 24 with a
pair bias; SwiGLU transitions x4), no biases where AF3 has none, float32, random weights at the CUDA bench's scale,
all masks one; a loss sum(z ct_z) + sum(s ct_s). Reports the forward, the backward (loss.backward()) and the peak.
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


def lin(i, o, bias=False):
  return nn.Linear(i, o, bias=bias)


class TriMul(nn.Module):
  def __init__(s, c, outgoing):
    super().__init__()
    s.out = outgoing
    s.ln, s.lnc = nn.LayerNorm(c), nn.LayerNorm(c)
    s.p, s.g, s.o, s.go = lin(c, 2 * c), lin(c, 2 * c), lin(c, c), lin(c, c)

  def forward(s, z, m):
    x = s.ln(z)
    ab = s.p(x) * torch.sigmoid(s.g(x)) * m[..., None]
    a, b = ab.chunk(2, -1)
    a, b = a.permute(2, 0, 1), b.permute(2, 0, 1)
    p = torch.matmul(a, b.transpose(-1, -2)) if s.out else torch.matmul(a.transpose(-1, -2), b)
    return s.o(s.lnc(p.permute(1, 2, 0))) * torch.sigmoid(s.go(x))


class GridAttn(nn.Module):
  def __init__(s, c, h, d):
    super().__init__()
    s.h, s.d = h, d
    s.ln, s.b = nn.LayerNorm(c), lin(c, h)
    s.q, s.k, s.v, s.g, s.o = lin(c, h * d), lin(c, h * d), lin(c, h * d), lin(c, h * d), lin(h * d, c)

  def forward(s, z, m):   # attention within each row of z [n, n, c]
    x = s.ln(z)
    n = z.shape[0]
    bias = s.b(x).permute(2, 0, 1)[None] + (1e9 * (m - 1))[:, None, None, :]
    q = s.q(x).view(n, n, s.h, s.d).transpose(1, 2)
    k = s.k(x).view(n, n, s.h, s.d).transpose(1, 2)
    v = s.v(x).view(n, n, s.h, s.d).transpose(1, 2)
    a = (torch.matmul(q * s.d ** -0.5, k.transpose(-1, -2)) + bias).softmax(-1)
    o = torch.matmul(a, v).transpose(1, 2).reshape(n, n, -1)
    return s.o(o * torch.sigmoid(s.g(x)))


class SwiGLU(nn.Module):
  def __init__(s, c, f=4):
    super().__init__()
    s.ln, s.a, s.b = nn.LayerNorm(c), lin(c, 2 * f * c), lin(f * c, c)

  def forward(s, x):
    u, v = s.a(s.ln(x)).chunk(2, -1)
    return s.b(F.silu(u) * v)


class SingleAttn(nn.Module):
  def __init__(s, c, cz, h, d):
    super().__init__()
    s.h, s.d = h, d
    s.ln, s.lnz, s.pz = nn.LayerNorm(c), nn.LayerNorm(cz), lin(cz, h)
    s.q, s.k, s.v, s.g, s.o = lin(c, h * d, True), lin(c, h * d), lin(c, h * d), lin(c, h * d), lin(h * d, c)

  def forward(s, x, z, m):
    x = s.ln(x)
    n = x.shape[0]
    bias = s.pz(s.lnz(z)).permute(2, 0, 1) + (1e9 * (m - 1))[None, None, :]
    q = s.q(x).view(n, s.h, s.d).transpose(0, 1)
    k = s.k(x).view(n, s.h, s.d).transpose(0, 1)
    v = s.v(x).view(n, s.h, s.d).transpose(0, 1)
    a = (torch.matmul(q * s.d ** -0.5, k.transpose(-1, -2)) + bias).softmax(-1)
    o = torch.matmul(a, v).transpose(0, 1).reshape(n, -1)
    return s.o(o * torch.sigmoid(s.g(x)))


class Pairformer(nn.Module):
  def __init__(s, cz=128, cs=384):
    super().__init__()
    s.tmo, s.tmi = TriMul(cz, True), TriMul(cz, False)
    s.a1, s.a2 = GridAttn(cz, 4, 32), GridAttn(cz, 4, 32)
    s.pt = SwiGLU(cz)
    s.sa, s.st = SingleAttn(cs, cz, 16, 24), SwiGLU(cs)

  def forward(s, z, x, zm, sm):
    z = z + s.tmo(z, zm)
    z = z + s.tmi(z, zm)
    z = z + s.a1(z, zm)
    z = z + s.a2(z.transpose(0, 1), zm.t()).transpose(0, 1)
    z = z + s.pt(z)
    x = x + s.sa(x, z, sm)
    x = x + s.st(x)
    return z, x


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--n", type=int, default=384)
  ap.add_argument("--tf32", action="store_true")
  ap.add_argument("--bf16", action="store_true", help="torch.autocast(bfloat16): float32 weights, bf16 matmuls")
  ap.add_argument("--pure-bf16", action="store_true", help="the block, its inputs and cotangents all bfloat16, no autocast")
  ap.add_argument("--compile", action="store_true")
  ap.add_argument("--reps", type=int, default=7)
  a = ap.parse_args()
  torch.backends.cuda.matmul.allow_tf32 = a.tf32
  torch.backends.cudnn.allow_tf32 = a.tf32
  if a.pure_bf16: torch.set_default_dtype(torch.bfloat16)   # parameters, inputs and cotangents
  torch.manual_seed(7)
  blk = Pairformer().cuda()
  with torch.no_grad():
    for p in blk.parameters():
      if p.dim() > 1: p.normal_(0, 0.05)
  z = torch.randn(a.n, a.n, 128, device="cuda", requires_grad=True)
  x = torch.randn(a.n, 384, device="cuda", requires_grad=True)
  zm, sm = torch.ones(a.n, a.n, device="cuda"), torch.ones(a.n, device="cuda")
  ctz, ctx = torch.randn_like(z), torch.randn_like(x)
  fn = torch.compile(blk) if a.compile else blk
  ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
  fw, bw = [], []
  for r in range(a.reps + 2):
    blk.zero_grad(set_to_none=False); z.grad = x.grad = None
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    ev[0].record()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
      zo, xo = fn(z, x, zm, sm)
    loss = (zo * ctz).sum() + (xo * ctx).sum()
    ev[1].record()
    loss.backward()
    ev[2].record()
    torch.cuda.synchronize()
    if r >= 2:
      fw.append(ev[0].elapsed_time(ev[1])); bw.append(ev[1].elapsed_time(ev[2]))
  med = lambda v: sorted(v)[len(v) // 2]
  print(f"torch n {a.n} (AF3 pairformer block{', bf16 autocast' if a.bf16 else ''}{', pure bf16' if a.pure_bf16 else ''}{', tf32' if a.tf32 else ''}{', compile' if a.compile else ''}): "
        f"forward {med(fw):.1f} ms, backward {med(bw):.1f} ms; peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
  main()
