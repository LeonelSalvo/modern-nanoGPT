"""
train.py — train modern-nanoGPT.

Loop (same as micrograd/makemore): forward (loss) → backward (gradients) → update,
plus the standard engineering: AdamW, warmup→cosine LR schedule, gradient clipping.
Logs the loss curve to data/train_log.csv, keeps the best-val checkpoint, and writes a sample.

    python train.py     # uses data/input.txt if present, else an embedded text
"""

import math
import os
import time
import torch
from model import GPTConfig, ModernGPT
from data import load_text, CharData

# paths relative to this file (repo root) → runs from any cwd
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_TXT = os.path.join(HERE, "data", "input.txt")
LOG_CSV  = os.path.join(HERE, "data", "train_log.csv")
CKPT     = os.path.join(HERE, "ckpt.pt")
SAMPLES  = os.path.join(HERE, "samples.txt")

# --- hyperparameters (small by default; bump on GPU) ---
block_size    = 256     # context (T)
batch_size    = 32
max_iters     = 5000
eval_interval = 250
eval_iters    = 50      # batches averaged per eval (stable number)
lr            = 3e-4    # max learning rate
min_lr        = 3e-5
warmup_iters  = 100
weight_decay  = 0.1
grad_clip     = 1.0

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1337)
print(f"[train] device = {device}")

# --- data ---
text = load_text(DATA_TXT)
data = CharData(text, block_size, device)
# --- model ---
cfg = GPTConfig(vocab_size=data.vocab_size, block_size=block_size,
                n_layer=6, n_head=8, n_kv_head=2, n_embd=384, dropout=0.2)
model = ModernGPT(cfg).to(device)
print(f"[train] vocab={cfg.vocab_size}  params={model.num_params()/1e6:.2f}M")

# AdamW = Adam (per-weight step size) + weight decay (regularization)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))


def get_lr(it):
    """LR schedule: linear warmup, then cosine decay to min_lr."""
    if it < warmup_iters:
        return lr * (it + 1) / warmup_iters
    ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)   # 0 → 1
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * ratio))


@torch.no_grad()
def estimate_loss():
    """Average train/val loss. Val is what matters (generalization vs memorization)."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = data.get_batch(split, batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# loss-curve CSV (one row per eval) for plotting later
os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
log = open(LOG_CSV, "w")
log.write("iter,train,val,lr\n")

# ===================== TRAINING LOOP =====================
best_val = float("inf")
t0 = time.time()
for it in range(max_iters + 1):
    for g in optimizer.param_groups:                   # set this step's lr from the schedule
        g["lr"] = get_lr(it)

    if it % eval_interval == 0:                         # eval, log, save best checkpoint
        l = estimate_loss()
        lr_now = get_lr(it)
        print(f"  iter {it:5d}  train {l['train']:.4f}  val {l['val']:.4f}  lr {lr_now:.2e}")
        log.write(f"{it},{l['train']:.4f},{l['val']:.4f},{lr_now:.6f}\n")
        log.flush()
        if l["val"] < best_val:
            best_val = l["val"]
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "stoi": data.stoi, "itos": data.itos}, CKPT)

    x, y = data.get_batch("train", batch_size)         # 1) batch
    _, loss = model(x, y)                              #    forward: predict + loss
    optimizer.zero_grad(set_to_none=True)              #    clear old gradients
    loss.backward()                                    # 2) backward: gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # clip giant gradients
    optimizer.step()                                   # 3) update: move weights

log.close()
print(f"[train] done in {(time.time()-t0)/60:.1f} min. Best val loss: {best_val:.4f}. Checkpoint: {CKPT}")

# --- save a sample from the best checkpoint (for the README) ---
# weights_only=False because the checkpoint also stores GPTConfig (PyTorch 2.6+ default is True). Safe: it's mine.
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
start  = torch.zeros((1, 1), dtype=torch.long, device=device)
tokens = model.generate(start, max_new_tokens=1000, temperature=0.8, top_k=40)[0].tolist()
sample_text = data.decode(tokens)
with open(SAMPLES, "w") as f:
    f.write(sample_text)
print(f"[train] 1000-char sample saved to {SAMPLES}")
print("\n--- sample ---\n" + sample_text[:500])
