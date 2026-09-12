"""Download both datasets into the D: cache before the real run.

Separated out because this is the one step whose speed depends on the network
rather than the machine, and because it is fully resumable: re-running after a
dropped connection continues where it stopped.
"""
import os, time

PROJECT_DIR = r"D:/AI-Projects/COAR-Q1"
os.environ["HF_HOME"] = os.path.join(PROJECT_DIR, "hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(PROJECT_DIR, "hf_cache", "datasets")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset

for name in ["SetFit/amazon_polarity", "fancyzhx/yelp_polarity"]:
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    ds = load_dataset(name)
    print(f"  splits : { {k: len(v) for k, v in ds.items()} }")
    print(f"  columns: {ds['train'].column_names}")
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

print("\nBoth datasets cached. The notebook will load them from disk from now on.")
