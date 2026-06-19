# Building modern-nanoGPT step by step

Here I build the model **one piece at a time**, Karpathy-style: each step implements ONE
component and ships a **self-checking test** (you run the file and it tells you OK/FAIL).
You understand and verify each piece before moving to the next. Once they all pass, they
get composed into the `model.py` in the folder above.

Each step follows the same format: **zoom out** (where you are in the full architecture)
→ **zoom in** (what the piece is, with every term in plain language) → **implementation** → **test**.

## Stack (deliberately minimal, Karpathy-style)

- **Python 3.10+**
- **PyTorch 2.1+** — the only heavy dependency. It gives *autograd* (the `.backward()` I
  did by hand in micrograd) and *GPU*. Without it I'd be writing every gradient by hand.
- **numpy** — data utilities.
- Nothing else: no HuggingFace, no Lightning. The interesting parts I write myself.

## How to use

```bash
# (once) create the environment and install torch
# note: on Debian/Ubuntu/Pop!_OS the command is python3 (not python)
python3 -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt

# each step (run it, check it says OK, then move on):
python steps/00_setup.py       # is torch installed? which device?
python steps/01_rmsnorm.py     # does it print OK? → on to step 2
python steps/02_rope.py
...
```

## Roadmap

0. **Setup** — verify environment + device (cuda/mps/cpu).
1. **RMSNorm** — the normalization (vs LayerNorm).
2. **RoPE** — the rotary position.
3. **Attention + GQA** — the heart, with shared K/V.
4. **SwiGLU** — the gated feed-forward.
5. **Block + full model** — assemble everything.
6. **Mini-training** — make the loss drop (the real "gradient check").

The "why" of each component, in plain language, is in the docstring at the top of each step file.
