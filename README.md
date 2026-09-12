# COAR / DistilBERT — local run

Local version of `COAR_DistilBERT_Annotated_TWO_DATASETS (3) (1).ipynb`.
The science is unchanged; what changed is that it runs on this laptop, saves
its work as it goes, and can redraw every figure without retraining.

## Run it

Open `COAR_DistilBERT_LOCAL.ipynb` and pick the kernel **COAR (venv)**, then
Run All. Or headless, which is better for a long unattended run:

```powershell
D:\AI-Projects\COAR-Q1\venv\Scripts\python.exe -m jupyter nbconvert `
  --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 `
  D:\AI-Projects\COAR-Q1\COAR_DistilBERT_LOCAL.ipynb
```

`--ExecutePreprocessor.timeout=-1` removes the per-cell time limit. There is no
session timeout of any kind locally — the only thing that can stop the run is
the machine sleeping, so set the power plan to never sleep first.

## Regenerating figures without retraining

This is what the checkpointing is for.

The twelve expensive stages each write their result to `checkpoints/` the first
time they run, and reload it on every run after:

| stage | what it costs the first time |
|---|---|
| `d1_baselines`, `d2_baselines` | TF-IDF + LogReg + SVM + SVD/RandomForest |
| `d1_bert`, `d2_bert` | fine-tuning DistilBERT (~11 min each) |
| `d1_coar`, `d2_coar` | 26,000 ablation forward passes each |
| `d1_loo`, `d2_loo` | 5 × 1,224 exhaustive leave-one-out ablations |
| `d1_refcoar`, `d2_refcoar` | 26,000 reference ablation passes each |
| `f11_budget` | Figure 7 mask-budget sweep |
| `f12_ci` | Figure 4 bootstrap intervals |

Both fine-tuned models are saved as `d1_bert_weights.pt` / `d2_bert_weights.pt`,
so cells F11 and F12 — which need a live model, not just stored numbers — work
on a re-run too.

After the first run, Run All reaches the figures in about ten minutes. Guarded
cells print `[RESUME] <stage>: restored from checkpoint` instead of computing.

To force a genuine recompute: set `RESUME = False` in Cell 0, or delete
`checkpoints/` (or just the one stage's `.pkl` you want to redo).

## Measured on this machine (RTX 3050 Laptop, 4.29 GB VRAM)

| | |
|---|---|
| Training step, batch 16 @ len 192 | 271 ms |
| Peak VRAM during training | 1.83 GB of 4.29 GB |
| Single ablation forward pass | 7.2 ms |
| Full first run | roughly 1–1.5 hours |

## Layout

```
COAR_DistilBERT_LOCAL.ipynb   the notebook to run
build_local_notebook.py       rebuilds it from the Colab original
checkpoints/                  stage results + both fine-tuned models
outputs/                      CSVs, diagnostic figures, figures_publication/
hf_cache/                     datasets and distilbert-base-uncased
venv/                         Python 3.11.9, torch 2.5.1+cu121
smoke_test.py                 GPU/VRAM/throughput check
test_checkpoint.py            verifies the resume machinery round-trips exactly
prefetch_data.py              downloads both datasets (resumable)
```

Everything lives on D: — the C: drive has under 9 GB free.

## If you revise the Colab notebook

`build_local_notebook.py` re-applies all the local changes to a fresh export.
It locates each guarded block by matching exact anchor lines and asserts if one
is missing, so it fails loudly rather than silently mispatching. Edit `SRC` at
the top if the filename changes, then:

```powershell
python D:\AI-Projects\COAR-Q1\build_local_notebook.py
```

The code inside every guard is copied byte-for-byte from the original and only
re-indented under `else:`. No result the pipeline produces can change.

## Notes

- The RandomForest baseline runs `n_jobs=-1` on 50k × 300 features with 7.6 GB
  of system RAM. If it thrashes, `n_jobs=4` is safe — sklearn seeds each tree
  individually, so it changes speed and nothing else.
- The network here drops large downloads regularly. Everything that downloads
  (datasets, model weights) resumes, so re-running a failed step continues
  rather than restarting.
