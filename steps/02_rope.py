"""
STEP 2 — RoPE (Rotary Position Embedding; replaces learned positions).

Where: inside attention, on q and k vectors right before they're compared.
Attention alone is permutation-invariant — it ignores token order — so I must
inject position. Normally (GPT-2) you ADD a learned per-position vector: absolute,
and it doesn't generalize past trained lengths. RoPE instead ROTATES q and k by an
angle proportional to position (each pair of numbers = a point spun on a plane;
different pairs spin at different frequencies). The magic: a dot product of two
rotated vectors depends ONLY on the position DIFFERENCE (m − n) — absolute on
rotation becomes RELATIVE on comparison. Generalizes to longer, adds no params;
it's what LLaMA/Mistral/Qwen use.

Terms: q = "what I'm looking for", k = "what I offer", dot product = similarity
(basis of attention), head_dim = vector size per attention head.

Test: shape preserved, vector norm preserved (rotation keeps size), and q·k
depends only on relative distance (m − n).
Run:  python steps/02_rope.py
"""

import torch


def build_rope_cache(head_dim: int, seq_len: int, base: float = 10000.0, device=None):
    """Precompute cos/sin of the rotation angles per position.
    inv_freq: spin speed per PAIR of dims (early pairs fast = fine position, late
    pairs slow = coarse). freqs[pos, i] = pos * inv_freq[i] = angle of pair i at pos.
    Returns cos, sin of shape (seq_len, head_dim/2)."""
    # inv_freq[i] = 1 / base^(2i/head_dim),  i = 0..head_dim/2-1
    idx      = torch.arange(0, head_dim, 2, device=device).float()  # 0,2,4,... one per pair
    exponent = idx / head_dim                             # normalize to [0,1)
    inv_freq = 1.0 / (base ** exponent)                   # spin speeds per pair
    t        = torch.arange(seq_len, device=device).float()  # positions 0..seq_len-1
    freqs    = torch.outer(t, inv_freq)                   # (seq_len, head_dim/2) angle per pair/pos
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """Rotate x with precomputed cos/sin.
    x: (..., T, head_dim), cos/sin: (T, head_dim/2). Take numbers in pairs
    (x1=even, x2=odd), apply 2D rotation (x1' = x1*cos − x2*sin, x2' = x1*sin +
    x2*cos), re-interleave."""
    x1 = x[..., 0::2]                                     # even indices (..., T, head_dim/2)
    x2 = x[..., 1::2]                                     # odd indices
    rx1 = x1 * cos - x2 * sin                            # rotated "x"
    rx2 = x1 * sin + x2 * cos                            # rotated "y"
    stacked = torch.stack((rx1, rx2), dim=-1)            # (..., T, head_dim/2, 2) pairs
    return stacked.flatten(-2)                           # re-interleave → (..., T, head_dim)


# ----------------------------- TEST (self-checking) -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    D, T = 8, 16                                          # head_dim, sequence length
    cos, sin = build_rope_cache(D, T)

    print("=== Step 2: RoPE ===")

    # (a) shape preserved
    x = torch.randn(2, T, D)                             # (B, T, head_dim)
    y = apply_rope(x, cos, sin)
    print("input shape:", tuple(x.shape), " -> output shape:", tuple(y.shape))
    assert y.shape == x.shape, "the shape changed, something is wrong"

    # (b) rotating keeps the vector's size: norm preserved per token
    norm_x = x.norm(dim=-1)                              # (B, T)
    norm_y = y.norm(dim=-1)
    print("norm preserved after rotating?:", torch.allclose(norm_x, norm_y, atol=1e-5))
    assert torch.allclose(norm_x, norm_y, atol=1e-5), "rotating changed the norm (wrong)"

    # (c) THE MAGIC: q·k depends ONLY on the relative distance (m - n)
    q = torch.randn(D)
    k = torch.randn(D)

    def rot_at(v, pos):                                   # rotate one vector at 'pos'
        v_row   = v.view(1, D)                           # (D,) -> (1, D): apply_rope wants (..., T, D)
        cos_pos  = cos[pos:pos+1]                         # cos row for pos
        sin_pos  = sin[pos:pos+1]                         # sin row for pos
        rotated   = apply_rope(v_row, cos_pos, sin_pos)   # (1, D)
        return rotated.view(D)                             # back to (D,)

    def score(m, n):                                     # dot of q@m with k@n
        q_rot = rot_at(q, m)                              # query rotated at m
        k_rot = rot_at(k, n)                              # key rotated at n
        return torch.dot(q_rot, k_rot).item()

    # same distance (m - n = 2) → same score
    s1, s2, s3 = score(5, 3), score(7, 5), score(10, 8)
    print(f"score with (m-n)=2:  {s1:.5f}, {s2:.5f}, {s3:.5f}  (should be equal)")
    assert abs(s1 - s2) < 1e-4 and abs(s1 - s3) < 1e-4, "doesn't depend only on distance (wrong)"

    # different distance → different score (so it DOES use position)
    print(f"score with (m-n)=0:  {score(4, 4):.5f}   (different, another distance)")

    print("\nOK — RoPE works: it rotates without changing size and encodes RELATIVE position. On to step 3 (attention + GQA).")
