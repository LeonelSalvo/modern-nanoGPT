"""Character-level data. If data/input.txt exists it uses that (e.g. TinyShakespeare);
otherwise it uses a small embedded text so the repo runs out-of-the-box."""

import os
import torch

_FALLBACK = (
    "The modern transformer shares its core with GPT-2: embeddings, attention and "
    "feed-forward blocks, and next-token prediction. What changes are five internal "
    "pieces. This tiny text exists only so that training runs without downloading "
    "anything; replace it with data/input.txt (TinyShakespeare) for something real.\n"
) * 200


def load_text(path: str = "data/input.txt") -> str:
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    print(f"[data] {path} does not exist → using the embedded example text. "
          f"Download TinyShakespeare to {path} to train for real.")
    return _FALLBACK


class CharData:
    def __init__(self, text: str, block_size: int, device: str, split: float = 0.9):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(data) * split)
        self.train, self.val = data[:n], data[n:]
        self.block_size, self.device = block_size, device

    def get_batch(self, split: str, batch_size: int):
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - self.block_size - 1, (batch_size,))
        x = torch.stack([d[i:i + self.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + self.block_size] for i in ix])
        return x.to(self.device), y.to(self.device)

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, t):
        return "".join(self.itos[int(i)] for i in t)
