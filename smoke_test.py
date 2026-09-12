"""Smoke test: does this machine actually run the pipeline's heaviest ops?

Checks the two things that could stop the local run:
  1. DistilBERT training step at CONFIG batch 16 / max_len 192 inside 4 GB VRAM.
  2. Single-example ablated forward pass speed, which sets the COAR runtime.
"""
import os, time

PROJECT_DIR = r"D:/AI-Projects/COAR-Q1"
os.environ["HF_HOME"] = os.path.join(PROJECT_DIR, "hf_cache")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2).to(device)
print("model loaded")

MAX_LEN, BATCH = 192, 16
texts = ["This product was absolutely wonderful and arrived early. " * 12] * BATCH
enc = tok(texts, truncation=True, padding="max_length",
          max_length=MAX_LEN, return_tensors="pt").to(device)
labels = torch.ones(BATCH, dtype=torch.long, device=device)

# ---- 1. training step at the real batch size ----
opt = AdamW(model.parameters(), lr=2e-5)
model.train()
for i in range(6):
    if i == 1:
        torch.cuda.synchronize(); t0 = time.time()
    opt.zero_grad(set_to_none=True)
    out = model(**enc, labels=labels)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
torch.cuda.synchronize()
per_step = (time.time() - t0) / 5
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"\ntrain step  : {per_step*1000:.0f} ms   peak VRAM {peak:.2f} GB / 4.29 GB")
print(f"  -> 2500 steps (20k x 2 epochs) = {per_step*2500/60:.1f} min per dataset")

# ---- 2. single-example forward, as COAR does it ----
del opt
torch.cuda.empty_cache()
model.eval()
one = "This product was absolutely wonderful and arrived early. " * 12
with torch.no_grad():
    for i in range(60):
        if i == 10:
            torch.cuda.synchronize(); t0 = time.time()
        e = tok(one, truncation=True, padding="max_length",
                max_length=MAX_LEN, return_tensors="pt").to(device)
        model(**e).logits
torch.cuda.synchronize()
per_fwd = (time.time() - t0) / 50
print(f"\nsingle fwd  : {per_fwd*1000:.1f} ms")

# COAR/LOO/ref/budget/CI forward passes, both datasets
passes = 2 * (26_000 + 6_120 + 26_000 + 2_000 + 1_080 + 500) + 12_000 + 2_400
print(f"  -> {passes:,} ablation passes total = {passes*per_fwd/60:.0f} min")
print(f"\nESTIMATED GPU TIME: {(per_step*2500*2 + passes*per_fwd)/60:.0f} min "
      f"(+ ~12 min TF-IDF/RandomForest on CPU)")
