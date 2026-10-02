# The same grid attention shape through PyTorch's fused attention, as a target:
# batch = n rows, heads 4, n queries x n keys, head dim 32, an additive [heads, n, n] bias.
import torch, time
from torch.nn.attention import sdpa_kernel, SDPBackend
for n in (256, 1024):
    for dt in (torch.float32, torch.bfloat16):
        q = torch.randn(n, 4, n, 32, device="cuda", dtype=dt)
        k, v = torch.randn_like(q), torch.randn_like(q)
        bias = torch.randn(1, 4, n, n, device="cuda", dtype=dt).expand(n, 4, n, n)
        for backend in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
            if backend == SDPBackend.MATH and n > 512: continue
            try:
                with sdpa_kernel(backend):
                    for _ in range(3): torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
                    torch.cuda.synchronize(); t = time.time()
                    for _ in range(5): torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
                    torch.cuda.synchronize()
                print(n, str(dt).split(".")[1], backend.name, f"{(time.time() - t) / 5 * 1000:.2f} ms per direction")
            except Exception as e:
                print(n, dt, backend.name, "failed:", str(e)[:100])
