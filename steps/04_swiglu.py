"""
STEP 4 — SwiGLU (the gated feed-forward; replaces the GELU MLP).

Where: the MLP (2nd sub-layer of each block). Attention = communication; MLP = each token thinks alone.
Classic MLP (GPT-2): x → Linear(→4·n_embd) → GELU → Linear(→n_embd). Two matrices, one activation.
SwiGLU: gated MLP with 3 matrices — gate=SiLU(x·W_gate), up=x·W_up, out=(gate*up)·W_down.
The gate regulates per-feature how much signal passes (learned filter); better per param (LLaMA/PaLM/Mistral).
Hidden = 2/3·(4·n_embd) rounded to a multiple of 64: the 2/3 offsets the 3rd matrix so total params ≈ classic MLP.
SiLU ("swish") = x·sigmoid(x), a smooth activation.

Test: shape in=out, gating in action, the 2/3 trick keeps params on par with the classic MLP.
Run:  python steps/04_swiglu.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        # inner size: 2/3 of 4·n_embd (offsets the 3rd matrix), rounded to a multiple of 64
        hidden_raw = int(2 / 3 * 4 * n_embd)
        hidden       = 64 * ((hidden_raw + 63) // 64)   # round up to multiple of 64
        self.hidden  = hidden
        self.w_gate  = nn.Linear(n_embd, hidden, bias=False)  # gate
        self.w_up    = nn.Linear(n_embd, hidden, bias=False)  # signal
        self.w_down  = nn.Linear(hidden, n_embd, bias=False)  # compress back down
        self.drop    = nn.Dropout(dropout)

    def forward(self, x):                                 # x: (B, T, n_embd)
        gate  = F.silu(self.w_gate(x))                    # 1) gate:   SiLU(x·W_gate)
        up    = self.w_up(x)                              # 2) signal: x·W_up
        fused = gate * up                                 # 3) gating: the gate modulates the signal
        out   = self.w_down(fused)                        # 4) compress to n_embd
        return self.drop(out)


# ----------------------------- TEST (self-checking) -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, n_embd = 1, 4, 64
    mlp = SwiGLU(n_embd)

    print("=== Step 4: SwiGLU ===")
    print(f"n_embd={n_embd}  inner hidden={mlp.hidden}  (2/3·4·{n_embd} rounded to a multiple of 64)")

    # (a) shape in = out
    x = torch.randn(B, T, n_embd)
    y = mlp(x)
    print("input shape:", tuple(x.shape), " -> output shape:", tuple(y.shape))
    assert y.shape == x.shape, "the shape changed, something is wrong"

    # (b) gating in action: SiLU on sample values + verify out = w_down(gate*up)
    z = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    print("SiLU(z) for z=[-3,-1,0,1,3]:", [round(v, 3) for v in F.silu(z).tolist()],
          " (smooth: negatives ~0, positives ~z)")
    with torch.no_grad():
        gate  = F.silu(mlp.w_gate(x))
        up    = mlp.w_up(x)
        fused = gate * up
    print("out == w_down(gate*up)?:", torch.allclose(y, mlp.w_down(fused), atol=1e-6))

    # (c) the 2/3 trick: SwiGLU (3 matrices) ≈ classic MLP (2) in params
    swiglu_params = sum(p.numel() for p in (mlp.w_gate.weight, mlp.w_up.weight, mlp.w_down.weight))
    ffn_params    = 2 * n_embd * (4 * n_embd)             # classic MLP: 2 matrices n_embd×4n_embd
    print(f"SwiGLU params: {swiglu_params}   vs classic 4× MLP: {ffn_params}   (on par, thanks to the 2/3)")

    print("\nOK — SwiGLU works: gated MLP, params on par with the classic one. On to step 5 (block + model).")
