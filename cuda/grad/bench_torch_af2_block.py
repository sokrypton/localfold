"""PyTorch's cost for one AlphaFold 2 Evoformer block trained - the baseline cuda/grad's backward is measured against.

    ~/venv_ef2/bin/python cuda/grad/bench_torch_af2_block.py [--L 256] [--S 128] [--extra] [--tf32] [--sdpa] [--compile]

The block as OpenFold writes it (multimer order: outer product mean first), float32, random weights at
bench-af2-block's scale, all masks one, a loss sum(m * ct_m) + sum(z * ct_z). It reports the forward, the backward
(loss.backward(), which is what cuda/grad's blockBackward is compared with) and the peak memory. Attention is the
plain softmax OpenFold uses; --sdpa routes the biased attentions through F.scaled_dot_product_attention instead,
and --compile wraps the block in torch.compile - the best PyTorch has to offer is the bar, not the worst.
"""
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
  def __init__(s, c, heads, hd, cb=None, glob=False):
    super().__init__()
    s.h, s.d, s.glob = heads, hd, glob
    s.ln = nn.LayerNorm(c)
    s.q = nn.Linear(c, heads * hd, bias=False)
    s.k = nn.Linear(c, (1 if glob else heads) * hd, bias=False)
    s.v = nn.Linear(c, (1 if glob else heads) * hd, bias=False)
    s.g = nn.Linear(c, heads * hd)
    s.o = nn.Linear(heads * hd, c)
    if cb is not None:
      s.lnb = nn.LayerNorm(cb)
      s.b = nn.Linear(cb, heads, bias=False)

  def forward(s, x, mask, bias_in=None, sdpa=False):
    # x: [B, N, c] attention over N within each of B rows; mask [B, N]
    x = s.ln(x)
    B, N, _ = x.shape
    g = torch.sigmoid(s.g(x)).view(B, N, s.h, s.d)
    if s.glob:  # AF2's global column attention: one mean query, a single key/value head
      q = (x * mask[..., None]).sum(1) / (mask.sum(1)[..., None] + 1e-10)
      q = s.q(q).view(B, s.h, s.d) * s.d ** -0.5
      k, v = s.k(x), s.v(x)  # [B, N, d]
      a = torch.einsum("bhd,bkd->bhk", q, k) + (1e9 * (mask[:, None, :] - 1))
      a = a.softmax(-1)
      o = torch.einsum("bhk,bkd->bhd", a, v)
      o = g * o[:, None]
      return s.o(o.reshape(B, N, -1))
    q = s.q(x).view(B, N, s.h, s.d).transpose(1, 2)
    k = s.k(x).view(B, N, s.h, s.d).transpose(1, 2)
    v = s.v(x).view(B, N, s.h, s.d).transpose(1, 2)
    bias = (1e9 * (mask - 1))[:, None, None, :]
    if bias_in is not None:
      bias = bias + s.b(s.lnb(bias_in)).permute(2, 0, 1)[None]  # [1, h, N, N]
    if sdpa:
      o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.expand(B, s.h, N, N))
    else:
      a = (torch.matmul(q * s.d ** -0.5, k.transpose(-1, -2)) + bias).softmax(-1)
      o = torch.matmul(a, v)
    o = g * o.transpose(1, 2)
    return s.o(o.reshape(B, N, -1))


class Transition(nn.Module):
  def __init__(s, c, n=4):
    super().__init__()
    s.ln, s.a, s.b = nn.LayerNorm(c), nn.Linear(c, n * c), nn.Linear(n * c, c)

  def forward(s, x, mask):
    return s.b(F.relu(s.a(s.ln(x)))) * mask[..., None]


class OuterProductMean(nn.Module):
  def __init__(s, cm, cz, c=32):
    super().__init__()
    s.ln, s.a, s.b, s.o = nn.LayerNorm(cm), nn.Linear(cm, c), nn.Linear(cm, c), nn.Linear(c * c, cz)

  def forward(s, m, mask):
    m = s.ln(m)
    a, b = s.a(m) * mask[..., None], s.b(m) * mask[..., None]
    o = torch.einsum("sic,sjd->ijcd", a, b).flatten(-2)
    n = torch.einsum("si,sj->ij", mask, mask)[..., None]
    return s.o(o) / (n + 1e-3)


class TriangleMultiplication(nn.Module):
  def __init__(s, cz, c=128, outgoing=True):
    super().__init__()
    s.out = outgoing
    s.ln, s.lno = nn.LayerNorm(cz), nn.LayerNorm(c)
    s.ap, s.ag, s.bp, s.bg = nn.Linear(cz, c), nn.Linear(cz, c), nn.Linear(cz, c), nn.Linear(cz, c)
    s.o, s.g = nn.Linear(c, cz), nn.Linear(cz, cz)

  def forward(s, z, mask):
    z = s.ln(z)
    mk = mask[..., None]
    a = s.ap(z) * torch.sigmoid(s.ag(z)) * mk
    b = s.bp(z) * torch.sigmoid(s.bg(z)) * mk
    a, b = a.permute(2, 0, 1), b.permute(2, 0, 1)
    p = torch.matmul(a, b.transpose(-1, -2)) if s.out else torch.matmul(a.transpose(-1, -2), b)
    p = p.permute(1, 2, 0)
    return s.o(s.lno(p)) * torch.sigmoid(s.g(z))


class EvoformerBlock(nn.Module):
  def __init__(s, cm=256, cz=128, extra=False):
    super().__init__()
    s.opm = OuterProductMean(cm, cz)
    s.row = Attention(cm, 8, cm // 8, cb=cz)
    s.col = Attention(cm, 8, 8 if extra else 32, glob=extra)
    s.mt = Transition(cm)
    s.tmo = TriangleMultiplication(cz, outgoing=True)
    s.tmi = TriangleMultiplication(cz, outgoing=False)
    s.tas = Attention(cz, 4, 32, cb=cz)
    s.tae = Attention(cz, 4, 32, cb=cz)
    s.pt = Transition(cz)

  def forward(s, m, z, mm, zm, sdpa=False):
    z = z + s.opm(m, mm)
    m = m + s.row(m, mm, z, sdpa)
    mt = m.transpose(0, 1)
    m = m + s.col(mt, mm.t(), sdpa=sdpa).transpose(0, 1)
    m = m + s.mt(m, mm)
    z = z + s.tmo(z, zm)
    z = z + s.tmi(z, zm)
    z = z + s.tas(z, zm, z, sdpa)
    zt = z.transpose(0, 1)
    z = z + s.tae(zt, zm.t(), zt, sdpa).transpose(0, 1)
    z = z + s.pt(z, zm)
    return m, z


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--L", type=int, default=256)
  ap.add_argument("--S", type=int, default=128)
  ap.add_argument("--extra", action="store_true")
  ap.add_argument("--tf32", action="store_true")
  ap.add_argument("--bf16", action="store_true", help="torch.autocast(bfloat16): float32 weights, bf16 matmuls")
  ap.add_argument("--sdpa", action="store_true")
  ap.add_argument("--pure-bf16", action="store_true", help="the block, its inputs and cotangents all bfloat16, no autocast")
  ap.add_argument("--compile", action="store_true")
  ap.add_argument("--reps", type=int, default=7)
  a = ap.parse_args()
  torch.backends.cuda.matmul.allow_tf32 = a.tf32
  torch.backends.cudnn.allow_tf32 = a.tf32
  if a.pure_bf16: torch.set_default_dtype(torch.bfloat16)   # parameters, inputs and cotangents
  torch.manual_seed(7)
  dev = "cuda"
  cm = 64 if a.extra else 256
  blk = EvoformerBlock(cm, 128, a.extra).to(dev)
  with torch.no_grad():
    for p in blk.parameters():
      if p.dim() > 1: p.normal_(0, 0.05)
  m = torch.randn(a.S, a.L, cm, device=dev, requires_grad=True)
  z = torch.randn(a.L, a.L, 128, device=dev, requires_grad=True)
  mm, zm = torch.ones(a.S, a.L, device=dev), torch.ones(a.L, a.L, device=dev)
  ctm, ctz = torch.randn_like(m), torch.randn_like(z)
  fn = torch.compile(blk) if a.compile else blk
  ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
  fw, bw = [], []
  for r in range(a.reps + 2):
    blk.zero_grad(set_to_none=False); m.grad = z.grad = None
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    ev[0].record()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
      mo, zo = fn(m, z, mm, zm, a.sdpa)
    loss = (mo * ctm).sum() + (zo * ctz).sum()
    ev[1].record()
    loss.backward()
    ev[2].record()
    torch.cuda.synchronize()
    if r >= 2:
      fw.append(ev[0].elapsed_time(ev[1])); bw.append(ev[1].elapsed_time(ev[2]))
  med = lambda x: sorted(x)[len(x) // 2]
  print(f"torch L {a.L} S {a.S} Cm {cm} ({'extra' if a.extra else 'main'}{', bf16 autocast' if a.bf16 else ''}{', pure bf16' if a.pure_bf16 else ''}{', tf32' if a.tf32 else ''}"
        f"{', sdpa' if a.sdpa else ''}{', compile' if a.compile else ''}): forward {med(fw):.1f} ms, "
        f"backward {med(bw):.1f} ms -> {med(fw) + med(bw):.1f} ms; peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
  main()
