"""
STEP 5 — The Block + the full Model (assemble everything).

Block (pre-norm + residual): x = x + Attention(RMSNorm(x)); x = x + SwiGLU(RMSNorm(x)).
Residual = add, not replace → gradient highway, lets you stack blocks (ResNet idea).
ModernGPT: token embedding (no position table — RoPE handles it) → N blocks → final RMSNorm →
lm_head (logits). Weight tying: lm_head shares tok_emb's matrix. forward returns (logits, loss).

Test: forward gives logits (B,T,vocab) + loss; initial loss ≈ ln(vocab); weight tying; generate works.
Run:  python steps/05_block_model.py
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== pieces from steps 1-4 (re-included so the file runs on its own) =====
class RMSNorm(nn.Module):                                 # step 1
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        msq = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(msq + self.eps))


def build_rope_cache(head_dim, seq_len, base=10000.0, device=None):  # step 2
    idx      = torch.arange(0, head_dim, 2, device=device).float()
    inv_freq = 1.0 / (base ** (idx / head_dim))
    t        = torch.arange(seq_len, device=device).float()
    freqs    = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):                              # step 2
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)


class CausalSelfAttentionGQA(nn.Module):                  # step 3
    def __init__(self, n_embd, n_head, n_kv_head, dropout=0.0):
        super().__init__()
        self.n_head, self.n_kv_head = n_head, n_kv_head
        self.head_dim = n_embd // n_head
        self.rep = n_head // n_kv_head
        self.q_proj = nn.Linear(n_embd, n_head    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.drop = dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.rep, dim=1)
        v = v.repeat_interleave(self.rep, dim=1)
        # fused Flash Attention — same math as step 3, faster
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.drop if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SwiGLU(nn.Module):                                  # step 4
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        hidden = 64 * ((int(2 / 3 * 4 * n_embd) + 63) // 64)
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)
        self.w_up   = nn.Linear(n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, n_embd, bias=False)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ===================== what's NEW today: Block and Model =====================
@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 64          # max context (T)
    n_layer:    int = 2
    n_head:     int = 4
    n_kv_head:  int = 2
    n_embd:     int = 32
    dropout:  float = 0.0
    rope_base: float = 10000.0


class Block(nn.Module):
    """Transformer block: two pre-normalized sub-layers, each with a residual."""
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)                  # norm before attention
        self.attn  = CausalSelfAttentionGQA(cfg.n_embd, cfg.n_head, cfg.n_kv_head, cfg.dropout)
        self.norm2 = RMSNorm(cfg.n_embd)                  # norm before the MLP
        self.mlp   = SwiGLU(cfg.n_embd, cfg.dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)        # residual: add, don't replace
        x = x + self.mlp(self.norm2(x))
        return x


class ModernGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token id → vector; no position (RoPE injects it)
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f  = RMSNorm(cfg.n_embd)                        # final norm
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # vector → logit per token
        self.lm_head.weight = self.tok_emb.weight                # weight tying: same matrix in/out

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = build_rope_cache(head_dim, cfg.block_size, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)  # RoPE cache (not a parameter)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)                                   # GPT-style init

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        # tied matrix shared by tok_emb/lm_head: count once
        return sum(p.numel() for p in self.parameters()) - self.lm_head.weight.numel()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x   = self.tok_emb(idx)                           # (B, T, n_embd)
        cos = self.rope_cos[:T]                            # trim RoPE to current length
        sin = self.rope_sin[:T]
        for block in self.blocks:                          # through the N blocks
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)                           # (B, T, vocab)

        loss = None
        if targets is not None:
            # each position's logit vs the real next token
            logits_flat  = logits.view(B * T, -1)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]       # trim to max context
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature        # last step, scaled by temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # sample next token
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ----------------------------- TEST (self-checking) -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=65, block_size=16, n_layer=2, n_head=4, n_kv_head=2, n_embd=32)
    model = ModernGPT(cfg)

    print("=== Step 5: Block + Model ===")
    print(f"params (not double-counting the tying): {model.num_params()/1e3:.1f}K")

    # (a) forward with targets → logits + loss
    B, T = 2, 16
    idx     = torch.randint(0, cfg.vocab_size, (B, T))    # random input tokens
    targets = torch.randint(0, cfg.vocab_size, (B, T))    # random "next" tokens
    logits, loss = model(idx, targets)
    print("logits shape:", tuple(logits.shape), " (expected (B, T, vocab) =", (B, T, cfg.vocab_size), ")")
    assert logits.shape == (B, T, cfg.vocab_size)

    # (b) initial loss ≈ ln(vocab): untrained model guesses at random
    expected = math.log(cfg.vocab_size)
    print(f"initial loss: {loss.item():.3f}   ln(vocab) = {expected:.3f}   (should be close)")
    assert abs(loss.item() - expected) < 0.5, "the initial loss looks off (bad init?)"

    # (c) weight tying: input and output matrices are the same tensor
    print("do tok_emb and lm_head share the matrix?:", model.tok_emb.weight is model.lm_head.weight)
    assert model.tok_emb.weight is model.lm_head.weight

    # (d) generate: from 1 token, generate 20 new ones
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=20)
    print("generate: started with 1 token and now I have", out.shape[1], "(1 + 20 new)")
    assert out.shape == (1, 21)

    print("\nOK — the whole model runs: forward, loss, and generation. On to step 6 (train it for real).")
