"""
STEP 3 — Causal attention + GQA (the core of the transformer).

Where: the sub-layer where tokens look at each other. Uses RMSNorm (step 1) before and RoPE (step 2) inside.
Q/K/V per token: compare Q·K (affinity) → softmax → weighted sum of V. Causal = no looking at the future.
GQA: fewer K/V heads shared across Q-head groups → smaller KV-cache, ~same quality (LLaMA/Mistral).
Here I compute attention by hand to see it; model.py uses F.scaled_dot_product_attention (Flash Attention), same math, fused.

Test: shape in=out, causality (changing a future token leaves earlier outputs untouched), GQA saves params.
Run:  python steps/03_attention.py
"""

import math
import torch
import torch.nn as nn


# --- RoPE from step 2 (re-included so this file is self-contained) ---
def build_rope_cache(head_dim, seq_len, base=10000.0, device=None):
    idx      = torch.arange(0, head_dim, 2, device=device).float()
    inv_freq = 1.0 / (base ** (idx / head_dim))
    t        = torch.arange(seq_len, device=device).float()
    freqs    = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    x1  = x[..., 0::2]
    x2  = x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)


# --- today's piece ---
class CausalSelfAttentionGQA(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must divide into n_head heads"
        assert n_head % n_kv_head == 0, "n_head must be a multiple of n_kv_head (to group)"
        self.n_head    = n_head
        self.n_kv_head = n_kv_head
        self.head_dim  = n_embd // n_head                 # size of each head
        self.rep       = n_head // n_kv_head              # q's per k/v
        # no bias (modern std); q has n_head heads, k/v only n_kv_head:
        self.q_proj = nn.Linear(n_embd, n_head    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)   # final head mix

    def forward(self, x, cos, sin):
        B, T, C = x.shape

        # 1) project x → q, k, v
        q = self.q_proj(x)                                # (B, T, n_head*head_dim)
        k = self.k_proj(x)                                # (B, T, n_kv_head*head_dim)
        v = self.v_proj(x)

        # 2) split heads, move head to dim 1: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_head,    self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # 3) RoPE: inject position by rotating q, k (step 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # 4) GQA: repeat each k/v head 'rep' times to match the q's
        k = k.repeat_interleave(self.rep, dim=1)          # (B, n_head, T, head_dim)
        v = v.repeat_interleave(self.rep, dim=1)

        # 5) scores = q·k affinity, scaled by 1/sqrt(head_dim)
        scale  = 1.0 / math.sqrt(self.head_dim)
        kt     = k.transpose(-2, -1)                      # (B, n_head, head_dim, T)
        scores = (q @ kt) * scale                         # (B, n_head, T, T)

        # 6) causal mask: future (j > i) → -inf before softmax
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))

        # 7) softmax → weights (sum 1) → weighted average of v's
        weights = torch.softmax(scores, dim=-1)           # (B, n_head, T, T)
        out     = weights @ v                             # (B, n_head, T, head_dim)

        # 8) recombine heads, mix with o_proj
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_embd)
        return self.o_proj(out)


# ----------------------------- TEST (self-checking) -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 1, 6
    n_embd, n_head, n_kv_head = 32, 4, 2                  # 4 q heads, 2 k/v (each k/v → 2 q)
    head_dim = n_embd // n_head

    attn = CausalSelfAttentionGQA(n_embd, n_head, n_kv_head)
    cos, sin = build_rope_cache(head_dim, T)

    print("=== Step 3: Causal attention + GQA ===")

    # (a) shape in = out
    x = torch.randn(B, T, n_embd)
    y = attn(x, cos, sin)
    print("input shape:", tuple(x.shape), " -> output shape:", tuple(y.shape))
    assert y.shape == x.shape, "the shape changed, something is wrong"

    # (b) causality: changing the last token must not affect earlier outputs
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(n_embd)                   # overwrite only the future token
    y2 = attn(x2, cos, sin)
    prev_unchanged = torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-6)   # positions 0..T-2
    last_changes = not torch.allclose(y[:, -1], y2[:, -1], atol=1e-6) # the last one does change
    print("previous outputs intact after changing the future?:", prev_unchanged)
    print("does the last token actually change?:", last_changes)
    assert prev_unchanged, "CAUSAL FAILURE: an earlier token saw the future"
    assert last_changes, "the last token should change"

    # (c) GQA saves k/v params vs MHA
    params_kv_gqa = sum(p.numel() for p in (attn.k_proj.weight, attn.v_proj.weight))
    params_kv_mha = 2 * n_embd * (n_head * head_dim)     # if k/v had n_head heads
    print(f"k/v params with GQA: {params_kv_gqa}   vs MHA: {params_kv_mha}   "
          f"(GQA uses {100*params_kv_gqa//params_kv_mha}% )")
    assert params_kv_gqa < params_kv_mha, "GQA should use fewer params than MHA"

    print("\nOK — causal attention + GQA works: looks only at the past and shares k/v. On to step 4 (SwiGLU).")
