"""Verify the resume machinery round-trips the exact object types the
pipeline stores, before committing to a multi-hour run."""
import os

PROJECT_DIR = r"D:/AI-Projects/COAR-Q1"
# Must be set BEFORE transformers is imported, and offline so this test never
# competes with the dataset download for bandwidth.
os.environ["HF_HOME"] = os.path.join(PROJECT_DIR, "hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np, joblib, torch
from sklearn.linear_model import Ridge
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints_selftest")
os.makedirs(CKPT_DIR, exist_ok=True)
RESUME = True

stage_file = lambda n: os.path.join(CKPT_DIR, n)
stage_done = lambda s: RESUME and os.path.exists(stage_file(s + ".pkl"))


def save_stage(stage, **objects):
    path = stage_file(stage + ".pkl")
    joblib.dump(objects, path, compress=3)
    print(f"[CHECKPOINT] {stage}: {os.path.getsize(path)/1e6:.1f} MB")


load_stage = lambda s: joblib.load(stage_file(s + ".pkl"))

NUM_COMPONENTS = 1224
rng = np.random.RandomState(0)

# A COAR_RESULTS entry, shaped exactly like the real one (Cell 6).
COAR_RESULTS = []
for ex in range(10):
    X = (rng.rand(300, NUM_COMPONENTS) > 0.05).astype(np.float32)
    y = rng.randn(300).astype(np.float32)
    reg = Ridge(alpha=10.0).fit(X, y)
    COAR_RESULTS.append({
        "Example": ex + 1, "Eval index": ex, "Label": ex % 2,
        "Model": reg,                                  # sklearn estimator
        "Weights": reg.coef_.astype(np.float32),
        "Intercept": float(reg.intercept_),
        "X_test": X, "y_test": y,
        "y_test_pred": reg.predict(X).astype(np.float32),
        "Text preview": "a review \u2014 with unicode \u0394",
    })

save_stage("selftest", COAR_RESULTS=COAR_RESULTS, coar_time=123.4)
assert stage_done("selftest"), "stage_done should see the checkpoint"

r = load_stage("selftest")["COAR_RESULTS"]
assert len(r) == 10
for a, b in zip(COAR_RESULTS, r):
    assert np.array_equal(a["X_test"], b["X_test"])
    assert np.array_equal(a["Weights"], b["Weights"])
    assert a["Text preview"] == b["Text preview"]
    # the restored estimator must still predict identically
    assert np.allclose(a["Model"].predict(a["X_test"]),
                       b["Model"].predict(b["X_test"]))
print("OK  COAR_RESULTS round-trip exact (arrays, Ridge models, unicode)")

# Model weights, as the d1_bert / d2_bert stages store them.
from transformers import AutoModelForSequenceClassification
m = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)
torch.save(m.state_dict(), stage_file("selftest_weights.pt"))
m2 = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)
m2.load_state_dict(torch.load(stage_file("selftest_weights.pt"), map_location="cpu"))
for k in m.state_dict():
    assert torch.equal(m.state_dict()[k], m2.state_dict()[k]), k
print("OK  DistilBERT state_dict round-trip bit-exact")

import shutil
shutil.rmtree(CKPT_DIR)
print("\nResume machinery verified.")
