"""
STEP 0 — SETUP (verify environment and stack).

No model yet: just check PyTorch is installed and pick the train device
(NVIDIA = cuda, Apple Silicon = mps, else cpu).

Stack is minimal (Karpathy style): Python 3.10+, PyTorch 2.1+ (the only heavy
dep — gives me autograd + GPU), numpy for data. Nothing else: no HuggingFace,
no Lightning — everything interesting I write myself.

Run:  python steps/00_setup.py
"""

import sys


def main():
    print("=== Step 0: setup ===")
    print("Python:", sys.version.split()[0])

    try:
        import torch
    except ImportError:
        print("\nPyTorch is MISSING. Set up the environment:")
        print("  python -m venv .venv && source .venv/bin/activate")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    print("PyTorch:", torch.__version__)

    # device: cuda > mps > cpu
    if torch.cuda.is_available():
        device = "cuda"
        print("NVIDIA GPU detected:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Apple Silicon (mps) detected")
    else:
        device = "cpu"
        print("No GPU: you'll use CPU (works, but slow for training)")

    # minimal test: tensor → op → device
    x = torch.randn(3, 3).to(device)
    y = (x @ x).sum()       # matmul + sum: a real computation
    print(f"test on {device}: x@x summed = {y.item():.4f}")

    print(f"\nOK — environment ready, device = '{device}'. On to step 1 (RMSNorm).")


if __name__ == "__main__":
    main()
