"""Generates text from a trained checkpoint:  python sample.py "Once upon a time" """

import sys
import torch
from model import ModernGPT

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
ckpt = torch.load("ckpt.pt", map_location=device, weights_only=False)
model = ModernGPT(ckpt["cfg"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
stoi, itos = ckpt["stoi"], ckpt["itos"]

prompt = sys.argv[1] if len(sys.argv) > 1 else "\n"
idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=500, temperature=0.8, top_k=40)[0].tolist()
print("".join(itos[int(i)] for i in out))
