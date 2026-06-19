"""
STEP 6 — Train it for real (watch the loss drop = the real gradient-check).

Steps 1-5 are built and verified; the model runs but hasn't learned (random weights → garbage).
Now I train it: repeat forward (loss) → backward (gradients) → update (move weights).
Final check: if everything is wired right and gradients flow, the loss DROPS; a shape/sign bug
would leave it flat or blow up. Import model.py (= steps 1-5 assembled) and data.py.
Data defaults to small embedded text (no download); optionally grab TinyShakespeare (below).
Optimizer: AdamW with fixed lr (the real train.py adds warmup + cosine decay).

Run:  python steps/06_train_tiny.py

(Optional, to train on real Shakespeare:)
  mkdir -p data && curl -o data/input.txt \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""

import os
import sys
# model.py and data.py live in the repo root (folder above); add it to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import GPTConfig, ModernGPT       # ← steps 1-5 assembled
from data import load_text, CharData         # ← text → tokens


# --- config: small and fast, loss drops in under a minute on GPU ---
device      = "cuda" if torch.cuda.is_available() else "cpu"
block_size  = 128
batch_size  = 32
max_iters   = 800
eval_every  = 100
lr          = 3e-4
torch.manual_seed(1337)
print(f"[step6] device = {device}")

# --- data ---
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
text = load_text(os.path.join(repo_root, "data", "input.txt"))  # Shakespeare if present, else embedded text
data = CharData(text, block_size, device)
print(f"[step6] vocab = {data.vocab_size} characters,  text = {len(text)} chars")

# --- model (small) ---
cfg = GPTConfig(vocab_size=data.vocab_size, block_size=block_size,
                n_layer=4, n_head=4, n_kv_head=2, n_embd=128, dropout=0.0)
model = ModernGPT(cfg).to(device)
print(f"[step6] params = {model.num_params()/1e3:.1f}K")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)


@torch.no_grad()
def measure_loss(split, iters=20):
    """Average the loss over several batches (stable number). No gradients."""
    model.eval()
    total = 0.0
    for _ in range(iters):
        xb, yb = data.get_batch(split, batch_size)
        _, loss = model(xb, yb)
        total += loss.item()
    model.train()
    return total / iters


# --- loss before training (reference: ~ln(vocab)) ---
initial_loss = measure_loss("train")
print(f"\ninitial loss (untrained): {initial_loss:.3f}")

# ===================== THE TRAINING LOOP =====================
for it in range(1, max_iters + 1):
    xb, yb = data.get_batch("train", batch_size)          # 1) batch
    logits, loss = model(xb, yb)                          #    forward: predict + loss
    optimizer.zero_grad(set_to_none=True)                 #    clear old gradients
    loss.backward()                                       # 2) backward: gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  #    clip giant gradients
    optimizer.step()                                      # 3) update: move weights

    if it % eval_every == 0:
        l_train = measure_loss("train")
        print(f"  iter {it:4d}   train loss {l_train:.3f}")

final_loss = measure_loss("train")
print(f"\nfinal loss: {final_loss:.3f}   (started at {initial_loss:.3f})")

# --- generate a sample to see it learned to write like the text ---
print("\n--- generated sample ---")
start  = torch.zeros((1, 1), dtype=torch.long, device=device)
tokens = model.generate(start, max_new_tokens=300, temperature=0.8, top_k=40)[0].tolist()
print(data.decode(tokens))

# --- the check: the loss must have clearly dropped ---
assert final_loss < initial_loss - 0.5, "the loss didn't drop: there'd be a bug in some piece"
print("\nOK — the loss dropped and the model generates text resembling the training data.")
print("    That validates the WHOLE architecture end to end. 🎉  modern-nanoGPT works.")
