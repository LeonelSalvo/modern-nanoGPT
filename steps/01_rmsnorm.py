"""
STEP 1 — RMSNorm (replaces LayerNorm).

Where: the norm before each sub-layer in every block (pre-norm scheme).
A token's embedding is a vector of n_embd numbers; their scales drift across
tokens/layers and training gets unstable. Normally LayerNorm centers (subtracts
mean) and scales (divides by std) + gamma/beta. RMSNorm only scales by the RMS
(vector size) + gamma — no centering, no beta. Simpler, nearly same effect,
fewer params; it's what LLaMA/Mistral/Qwen use.

Test: shape preserved, each token's RMS ≈ 1, and it does NOT center (shown vs LayerNorm).
Run:  python steps/01_rmsnorm.py
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # gamma (scale); no beta

    def forward(self, x):                             # x: (..., dim)
        sq      = x.pow(2)                             # 1) square each number
        msq     = sq.mean(dim=-1, keepdim=True)        # 2) mean of squares, per token
        inv_rms = torch.rsqrt(msq + self.eps)          # 3) 1/sqrt(mean) = scale factor
        x_norm  = x * inv_rms                          # 4) divide by token "size"
        return self.weight * x_norm                    # 5) reapply gamma


# ----------------------------- TEST (self-checking) -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C = 2, 4, 8                                  # batch, tokens, n_embd (small to inspect)
    # weird mean (+3) and scale (×5) on purpose, to see what the norm does
    x = torch.randn(B, T, C) * 5 + 3
    norm = RMSNorm(C)
    y = norm(x)

    print("=== Step 1: RMSNorm ===")
    print("input shape:", tuple(x.shape), " -> output shape:", tuple(y.shape))

    # (a) shape preserved
    assert y.shape == x.shape, "the shape changed, something is wrong"

    # (b) with gamma=1 (init), each token's RMS should be ~1
    sq_y    = y.pow(2)                                 # squares of output
    msq_y   = sq_y.mean(dim=-1)                        # mean per token → (B, T)
    rms_y   = msq_y.sqrt()                             # root = token RMS
    ones    = torch.ones_like(rms_y)                   # reference: all 1
    print("RMS per token (should be ~1):", [round(v, 3) for v in rms_y.flatten().tolist()])
    assert torch.allclose(rms_y, ones, atol=1e-3), "the RMS didn't land at 1"

    # (c) RMSNorm does NOT center: token mean is NOT forced to 0...
    mean_rms = y.mean(dim=-1)                         # mean per token after RMSNorm
    print("mean per token after RMSNorm (NOT forced to 0):",
          [round(v, 3) for v in mean_rms.flatten().tolist()])
    # ...whereas LayerNorm DOES force it to ~0:
    ln        = nn.LayerNorm(C, elementwise_affine=False)
    y_ln      = ln(x)                                  # same input via LayerNorm
    mean_ln  = y_ln.mean(dim=-1)                      # mean per token after LayerNorm
    print("mean per token after LayerNorm (IS ~0):  ",
          [round(v, 3) for v in mean_ln.flatten().tolist()])

    print("\nOK — RMSNorm works: it normalizes by SIZE (RMS~1) WITHOUT centering. On to step 2 (RoPE).")
