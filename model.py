"""
modern-nanoGPT / model.py — a GPT (decoder-only transformer) from scratch, with the
components used by modern open LLMs (Llama / Mistral / Qwen).

Skeleton (unchanged from GPT-2):  text → tokens → embeddings → [Block]×N → next-token logits
The 5 swaps vs GPT-2:
    LayerNorm → RMSNorm | learned pos-emb → RoPE | GELU FFN → SwiGLU
    MHA → GQA | biases → no biases + tied embeddings

Shapes:  B = batch, T = tokens (context), C = n_embd, hd = head_dim
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 256       # max context (T)
    n_layer: int = 6
    n_head: int = 8             # Query heads
    n_kv_head: int = 2          # Key/Value heads (GQA: < n_head shrinks the KV-cache)
    n_embd: int = 384           # model width (C)
    dropout: float = 0.0
    rope_base: float = 10000.0


# 1) RMSNorm — replaces LayerNorm. Normalizes each token by its RMS (size) only:
#    no mean-subtraction, no beta. Same stability, less compute.
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # gamma (learned scale)

    def forward(self, x):                             # (B, T, C)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x * rms)


# 2) RoPE — replaces learned positional embeddings. Rotates Q/K by an angle
#    proportional to position, so the Q·K product depends on RELATIVE distance.
#    No parameters; generalizes to longer sequences.
def build_rope_cache(head_dim: int, seq_len: int, base: float, device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                  # (T, hd/2): angle per (position, pair)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):                          # x: (B, n_head, T, hd)
    T = x.shape[-2]
    cos = cos[:T].view(1, 1, T, -1)                   # broadcast over B and heads
    sin = sin[:T].view(1, 1, T, -1)
    x1, x2 = x[..., 0::2], x[..., 1::2]               # even / odd coordinate pairs
    rx1 = x1 * cos - x2 * sin                          # 2D rotation
    rx2 = x1 * sin + x2 * cos
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)


# 3) Attention with GQA — replaces Multi-Head Attention. Many Query heads share
#    fewer Key/Value heads, so the KV-cache at inference is much smaller for almost
#    no quality loss. No biases; RoPE applied to Q/K only.
class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head, self.n_kv_head = cfg.n_head, cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.rep = cfg.n_head // cfg.n_kv_head         # Q heads per K/V head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):                   # (B, T, C)
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.rep, dim=1)       # GQA: expand K/V to n_head
        v = v.repeat_interleave(self.rep, dim=1)
        # fused causal attention (Flash Attention): softmax(Q·Kᵀ/√hd) masked + weighted V
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


# 4) SwiGLU — replaces the GELU feed-forward. Gated FFN: gate = SiLU(x·W_gate)
#    modulates content = x·W_up element-wise, then W_down. More expressive per param.
#    Hidden ≈ 2/3·(4·C) so the 3 matrices total roughly a classic FFN's params.
class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(2 / 3 * 4 * cfg.n_embd)
        hidden = 64 * ((hidden + 63) // 64)            # round to a multiple of 64
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up   = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):                             # (B, T, C)
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# Block: pre-norm + residual (x = x + sublayer(norm(x))). The residual "highway"
# lets gradients flow, so many blocks can be stacked.
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)     # tokens communicate
        x = x + self.mlp(self.norm2(x))                # each token computes
        return x


class ModernGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # no positional table (RoPE handles position)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight      # tied embeddings (shared in/out matrix)
        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = build_rope_cache(head_dim, cfg.block_size, cfg.rope_base, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):                                # GPT-style init: std 0.02
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):             # idx: (B, T) token ids
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than context (block_size)"
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos.to(x.device), self.rope_sin.to(x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)                        # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate token by token: predict → sample → append → repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]    # crop to the max context
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature     # last step, scaled by temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def num_params(self):
        # tok_emb and lm_head share the tied matrix → count it once
        return sum(p.numel() for p in self.parameters()) - self.lm_head.weight.numel()
