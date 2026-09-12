"""
Build a local, checkpointed version of the COAR notebook.

Reads the Colab notebook and writes COAR_DistilBERT_LOCAL.ipynb with:
  1. Colab-isms removed (!pip, /content paths, files.download).
  2. Caches redirected to D: (C: has little free space).
  3. RESUME GUARDS around the 10 expensive stages, so the notebook can be
     re-run to regenerate figures without recomputing anything.

The scientific code inside every guarded block is copied byte-for-byte from
the original (only re-indented one level to sit under `else:`).
"""
import json, io, os, sys

SRC = r"D:\Downloads\COAR_DistilBERT_Annotated_TWO_DATASETS (3) (1).ipynb"
DST = r"D:\AI-Projects\COAR-Q1\COAR_DistilBERT_LOCAL.ipynb"
PROJ = r"D:/AI-Projects/COAR-Q1"

nb = json.load(io.open(SRC, encoding="utf-8"))
cells = nb["cells"]


def src_of(i):
    return "".join(cells[i]["source"])


def set_src(i, text):
    cells[i]["source"] = text.splitlines(keepends=True)
    cells[i]["outputs"] = []
    cells[i]["execution_count"] = None


def replace_once(i, old, new):
    s = src_of(i)
    assert s.count(old) == 1, f"cell {i}: anchor found {s.count(old)}x, expected 1:\n{old[:120]}"
    set_src(i, s.replace(old, new))


def guard(cell_idx, stage, start_anchor, end_anchor, end_inclusive,
          restore_lines, save_lines, note, pre_lines=()):
    """Wrap [start_anchor .. end_anchor] of a cell in a resume guard.

    pre_lines run inside the `else:` branch *before* the original block. They
    are used only to free VRAM, never to change what the block computes.
    """
    s = src_of(cell_idx)
    a = s.find(start_anchor)
    assert a != -1, f"cell {cell_idx}: start anchor not found: {start_anchor[:80]}"
    b = s.find(end_anchor, a)
    assert b != -1, f"cell {cell_idx}: end anchor not found: {end_anchor[:80]}"
    if end_inclusive:
        b = b + len(end_anchor)
    block = s[a:b]

    indented = "\n".join(("    " + ln) if ln.strip() else ln
                         for ln in block.split("\n"))

    head = (
        f"# ---------------- RESUME GUARD ({stage}) ----------------\n"
        f"# {note}\n"
        f"# Skipped automatically when a checkpoint for this stage already exists.\n"
        f"if stage_done({stage!r}):\n"
        + "".join(f"    {ln}\n" for ln in restore_lines)
        + f"    print('[RESUME] {stage}: restored from checkpoint (compute skipped)')\n"
        f"else:\n"
        + "".join(f"    {ln}\n" for ln in pre_lines)
    )
    tail = (
        "\n"
        + "".join(f"    {ln}\n" for ln in save_lines)
        + f"# -------------- END RESUME GUARD ({stage}) --------------\n"
    )
    set_src(cell_idx, s[:a] + head + indented + tail + s[b:])


# =====================================================================
# 1. New first cell: environment (must run before any HF import)
# =====================================================================
ENV_CELL = f'''# ============================================================
# CELL 0 - LOCAL ENVIRONMENT (added for the local run)
# ============================================================
# Colab gave a fresh VM with everything preinstalled and a /content
# disk. Locally we instead: pin every cache onto the D: drive (the
# C: drive has very little free space), and turn on the resume
# machinery defined in Cell 1B.
#
# There is no session timeout locally. The run can take as long as
# it needs.
# ============================================================
import os

PROJECT_DIR = r"{PROJ}"

# Every large cache goes to D:, not C:.
os.environ["HF_HOME"]           = os.path.join(PROJECT_DIR, "hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(PROJECT_DIR, "hf_cache", "datasets")
os.environ["TORCH_HOME"]        = os.path.join(PROJECT_DIR, "torch_cache")
os.environ["MPLCONFIGDIR"]      = os.path.join(PROJECT_DIR, "mpl_cache")
for _p in ("hf_cache", "torch_cache", "mpl_cache", "outputs", "checkpoints"):
    os.makedirs(os.path.join(PROJECT_DIR, _p), exist_ok=True)

# Keep tokenizer threads quiet and avoid the fork warning on Windows.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# RESUME = True  -> expensive stages load from D:/AI-Projects/COAR-Q1/checkpoints
#                   instead of recomputing. This is what makes it possible to
#                   regenerate every figure without retraining.
# RESUME = False -> force a full recompute from scratch.
RESUME = True

print("Local environment ready.")
print("  project     :", PROJECT_DIR)
print("  HF cache    :", os.environ["HF_HOME"])
print("  RESUME mode :", RESUME)
'''

# =====================================================================
# 2. Checkpoint helper cell (inserted after the setup cell, index 2)
# =====================================================================
CKPT_CELL = '''# ============================================================
# CELL 1B - CHECKPOINT / RESUME MACHINERY (added for the local run)
# ============================================================
# Colab loses everything when the session times out. Locally the
# risk is different: a laptop reboot, a crash, or simply wanting
# to redraw a figure tomorrow without spending three hours
# retraining.
#
# Every expensive stage below is wrapped in a guard that writes its
# result here and reloads it on the next run. The figure cells
# (F1-F12) read those same variables, so re-running the notebook
# top-to-bottom with RESUME=True reaches the figures in a few
# minutes with zero training.
#
# Nothing here changes any number the pipeline produces - the
# guarded code is copied verbatim from the Colab version.
# ============================================================
import joblib

CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def stage_file(name: str) -> str:
    # Absolute path of an artefact belonging to a stage.
    return os.path.join(CKPT_DIR, name)


def stage_done(stage: str) -> bool:
    # True when this stage was already computed and RESUME is on.
    return RESUME and os.path.exists(stage_file(stage + ".pkl"))


def save_stage(stage: str, **objects):
    # Persist a stage's results so the next run can skip the compute.
    path = stage_file(stage + ".pkl")
    joblib.dump(objects, path, compress=3)
    size = os.path.getsize(path) / 1e6
    print(f"[CHECKPOINT] {stage}: saved {len(objects)} object(s), {size:.1f} MB -> {path}")


def load_stage(stage: str) -> dict:
    # Read a stage's results back.
    return joblib.load(stage_file(stage + ".pkl"))


# Report what is already on disk, so it is obvious what this run will skip.
_existing = sorted(f[:-4] for f in os.listdir(CKPT_DIR) if f.endswith(".pkl"))
if _existing:
    print("Checkpoints found (these stages will be skipped):")
    for _s in _existing:
        print("   -", _s)
else:
    print("No checkpoints yet - this run will compute every stage and save as it goes.")
'''

# =====================================================================
# 3. Patch the setup cell (index 2)
# =====================================================================
replace_once(
    2,
    "!pip install -q datasets transformers accelerate scikit-learn scipy pandas matplotlib tqdm joblib",
    "# (Colab installed packages here. Locally they live in the project venv:\n"
    "#  D:\\AI-Projects\\COAR-Q1\\venv - so no install step is needed.)",
)
replace_once(
    2,
    '"save_dir"                : "/content/amazon_distilbert_coar_outputs",  # where files are saved',
    '"save_dir"                : os.path.join(PROJECT_DIR, "outputs"),  # where files are saved (local)',
)

# =====================================================================
# 4. Resume guards
# =====================================================================
GUARDS = [
    # ---------------- dataset 1 ----------------
    dict(cell_idx=6, stage="d1_baselines",
         start_anchor="X_tr_tfidf = tfidf.fit_transform(X_train_base)",
         end_anchor="baseline_df = pd.DataFrame(baseline_results)    # put the three rows into a table",
         end_inclusive=True,
         note="TF-IDF + LogReg + LinearSVM + SVD/RandomForest on dataset 1 (~5 min, RAM-hungry).",
         restore_lines=[
             '_s = load_stage("d1_baselines")',
             'baseline_results, baseline_df = _s["baseline_results"], _s["baseline_df"]',
             'X_tr_tfidf, X_te_tfidf = _s["X_tr_tfidf"], _s["X_te_tfidf"]',
         ],
         save_lines=[
             'save_stage("d1_baselines", baseline_results=baseline_results, baseline_df=baseline_df,',
             '           X_tr_tfidf=X_tr_tfidf, X_te_tfidf=X_te_tfidf)',
         ]),
    dict(cell_idx=8, stage="d1_bert",
         start_anchor="# -- Training loop --------------------------------------------\ntrain_log, best_f1, best_state = [], -1.0, None",
         end_anchor="final_bert = eval_epoch(bert_model, eval_loader)   # final evaluation with the best model",
         end_inclusive=True,
         note="Fine-tuning DistilBERT on dataset 1 (~20 min on this GPU).",
         restore_lines=[
             '_s = load_stage("d1_bert")',
             'train_log, best_f1, final_bert = _s["train_log"], _s["best_f1"], _s["final_bert"]',
             'bert_model.load_state_dict(torch.load(stage_file("d1_bert_weights.pt"), map_location="cpu"))',
             'bert_model.to(device)',
         ],
         save_lines=[
             'torch.save(bert_model.state_dict(), stage_file("d1_bert_weights.pt"))',
             'save_stage("d1_bert", train_log=train_log, best_f1=best_f1, final_bert=final_bert)',
         ]),
    dict(cell_idx=12, stage="d1_coar",
         start_anchor="# -- Main COAR loop -------------------------------------------\nCOAR_RESULTS = []",
         end_anchor="coar_time = time.time() - t0   # how long the loop took",
         end_inclusive=True,
         note="COAR attribution on dataset 1: 10 examples x 2,600 masks = 26,000 forward passes.",
         restore_lines=[
             '_s = load_stage("d1_coar")',
             'COAR_RESULTS, coar_time = _s["COAR_RESULTS"], _s["coar_time"]',
         ],
         save_lines=['save_stage("d1_coar", COAR_RESULTS=COAR_RESULTS, coar_time=coar_time)']),
    dict(cell_idx=14, stage="d1_loo",
         start_anchor="LOO_RESULTS = []   # one result per example",
         end_anchor="# Show the per-example comparison (without the big arrays).",
         end_inclusive=False,
         note="Exhaustive leave-one-out sweep on dataset 1: 5 examples x 1,224 ablations.",
         restore_lines=[
             '_s = load_stage("d1_loo")',
             'LOO_RESULTS = _s["LOO_RESULTS"]',
         ],
         save_lines=['save_stage("d1_loo", LOO_RESULTS=LOO_RESULTS)']),
    dict(cell_idx=16, stage="d1_refcoar",
         start_anchor="REF_COAR = []",
         end_anchor="# Show the reference attribution quality.",
         end_inclusive=False,
         note="COAR attribution of the 10 reference examples on dataset 1 (another 26,000 passes).",
         restore_lines=[
             '_s = load_stage("d1_refcoar")',
             'REF_COAR = _s["REF_COAR"]',
         ],
         save_lines=['save_stage("d1_refcoar", REF_COAR=REF_COAR)']),
    # ---------------- dataset 2 ----------------
    dict(cell_idx=26, stage="d2_baselines",
         start_anchor="X_tr_tfidf_d2 = tfidf_d2.fit_transform(X_train_base_d2)",
         end_anchor="baseline_df_d2 = pd.DataFrame(baseline_results_d2)           # put the three rows into a table",
         end_inclusive=True,
         note="TF-IDF + LogReg + LinearSVM + SVD/RandomForest on dataset 2 (~5 min, RAM-hungry).",
         restore_lines=[
             '_s = load_stage("d2_baselines")',
             'baseline_results_d2, baseline_df_d2 = _s["baseline_results_d2"], _s["baseline_df_d2"]',
             'X_tr_tfidf_d2, X_te_tfidf_d2 = _s["X_tr_tfidf_d2"], _s["X_te_tfidf_d2"]',
         ],
         save_lines=[
             'save_stage("d2_baselines", baseline_results_d2=baseline_results_d2, baseline_df_d2=baseline_df_d2,',
             '           X_tr_tfidf_d2=X_tr_tfidf_d2, X_te_tfidf_d2=X_te_tfidf_d2)',
         ]),
    dict(cell_idx=28, stage="d2_bert",
         start_anchor="# -- Training loop (eval_epoch from Cell 4 is reused as-is) ---\ntrain_log_d2, best_f1_d2, best_state_d2 = [], -1.0, None",
         end_anchor="final_bert_d2 = eval_epoch(bert_model_d2, eval_loader_d2)   # final evaluation with the best model",
         end_inclusive=True,
         note="Fine-tuning DistilBERT on dataset 2 (~20 min on this GPU).",
         # VRAM only: this 4 GB card cannot hold the dataset-1 model, the
         # dataset-2 model, and a full AdamW training state at once. The
         # dataset-1 model is not read during dataset-2 training, so it is
         # parked on the CPU here and moved back immediately afterwards.
         # Moving a tensor between devices is a copy - every weight, and
         # therefore every number this pipeline reports, is unchanged.
         pre_lines=[
             'bert_model.to("cpu")',
             'gc.collect(); torch.cuda.empty_cache()',
             'print("[VRAM] dataset-1 model parked on CPU for the duration of this training")',
         ],
         restore_lines=[
             '_s = load_stage("d2_bert")',
             'train_log_d2, best_f1_d2, final_bert_d2 = _s["train_log_d2"], _s["best_f1_d2"], _s["final_bert_d2"]',
             'bert_model_d2.load_state_dict(torch.load(stage_file("d2_bert_weights.pt"), map_location="cpu"))',
             'bert_model_d2.to(device)',
         ],
         save_lines=[
             'torch.save(bert_model_d2.state_dict(), stage_file("d2_bert_weights.pt"))',
             'bert_model.to(device)   # dataset-1 model back on the GPU, unchanged',
             'print("[VRAM] dataset-1 model restored to", device)',
             'save_stage("d2_bert", train_log_d2=train_log_d2, best_f1_d2=best_f1_d2, final_bert_d2=final_bert_d2)',
         ]),
    dict(cell_idx=32, stage="d2_coar",
         start_anchor="# -- Main COAR loop -------------------------------------------\nCOAR_RESULTS_D2 = []",
         end_anchor="coar_time_d2 = time.time() - t0_d2   # how long the loop took",
         end_inclusive=True,
         note="COAR attribution on dataset 2: 10 examples x 2,600 masks = 26,000 forward passes.",
         restore_lines=[
             '_s = load_stage("d2_coar")',
             'COAR_RESULTS_D2, coar_time_d2 = _s["COAR_RESULTS_D2"], _s["coar_time_d2"]',
         ],
         save_lines=['save_stage("d2_coar", COAR_RESULTS_D2=COAR_RESULTS_D2, coar_time_d2=coar_time_d2)']),
    dict(cell_idx=34, stage="d2_loo",
         start_anchor="LOO_RESULTS_D2 = []   # one result per dataset-2 example",
         end_anchor="# Show the per-example comparison (without the big arrays).",
         end_inclusive=False,
         note="Exhaustive leave-one-out sweep on dataset 2: 5 examples x 1,224 ablations.",
         restore_lines=[
             '_s = load_stage("d2_loo")',
             'LOO_RESULTS_D2 = _s["LOO_RESULTS_D2"]',
         ],
         save_lines=['save_stage("d2_loo", LOO_RESULTS_D2=LOO_RESULTS_D2)']),
    dict(cell_idx=36, stage="d2_refcoar",
         start_anchor="REF_COAR_D2 = []",
         end_anchor='show_table("CELL D7B - Reference COAR Quality (Dataset 2)"',
         end_inclusive=False,
         note="COAR attribution of the 10 reference examples on dataset 2 (another 26,000 passes).",
         restore_lines=[
             '_s = load_stage("d2_refcoar")',
             'REF_COAR_D2 = _s["REF_COAR_D2"]',
         ],
         save_lines=['save_stage("d2_refcoar", REF_COAR_D2=REF_COAR_D2)']),
    # ---------------- figure cells that recompute ----------------
    dict(cell_idx=52, stage="f11_budget",
         start_anchor="budget_rows  = budget_curve(COAR_RESULTS,    build_coar_dataset,    D1_LABEL)",
         end_anchor="budget_df = pd.DataFrame(budget_rows)",
         end_inclusive=True,
         note="Figure 7 mask-budget sweep: ~13,800 forward passes per dataset.",
         restore_lines=[
             '_s = load_stage("f11_budget")',
             'budget_rows, budget_df = _s["budget_rows"], _s["budget_df"]',
         ],
         save_lines=['save_stage("f11_budget", budget_rows=budget_rows, budget_df=budget_df)']),
    dict(cell_idx=53, stage="f12_ci",
         start_anchor="ci_rows  = selectivity_with_ci(target_texts, target_labels, ref_texts, ref_labels,",
         end_anchor="selectivity_ci_df = pd.DataFrame(ci_rows)",
         end_inclusive=True,
         note="Figure 4 bootstrap confidence intervals: ~2,300 forward passes per dataset.",
         restore_lines=[
             '_s = load_stage("f12_ci")',
             'ci_rows, selectivity_ci_df = _s["ci_rows"], _s["selectivity_ci_df"]',
         ],
         save_lines=['save_stage("f12_ci", ci_rows=ci_rows, selectivity_ci_df=selectivity_ci_df)']),
]

for g in GUARDS:
    guard(**g)
    print(f"  guarded cell {g['cell_idx']:>3}  ->  stage '{g['stage']}'")

# =====================================================================
# 5. Export cell: drop the Colab download
# =====================================================================
replace_once(
    42,
    "# In Colab, start the download automatically; otherwise just print the path.\n"
    "try:\n"
    "    from google.colab import files\n"
    "    files.download(zip_path)\n"
    '    print("Download started.")\n'
    "except Exception as e:\n"
    '    print("Manual download path:", zip_path)',
    "# Local run: nothing to download, the files are already on disk.\n"
    'print("All outputs are here:", CONFIG["save_dir"])\n'
    'print("Zipped copy:", zip_path)',
)

# =====================================================================
# 6. Assemble and write
# =====================================================================
def code_cell(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


INTRO = f"""# Local run

This is the Colab notebook, unchanged in its science, prepared to run on this
machine. Three things are different:

1. **No session timeout.** Nothing here depends on a hosted runtime, so the run
   can take as long as it needs.
2. **Checkpointing.** Each of the ten expensive stages saves its result to
   `{PROJ}/checkpoints` the first time it runs, and reloads it on
   every later run. Re-running the notebook with `RESUME = True` therefore
   reaches the figure cells in minutes without retraining anything.
3. **Caches on D:.** The datasets, the model downloads and the outputs all live
   under `{PROJ}`, because C: is nearly full.

To redraw figures later: open this notebook, Run All. The guarded stages print
`[RESUME] ... restored from checkpoint` instead of computing.

To force a genuine full recompute: set `RESUME = False` in Cell 0, or delete the
`checkpoints` folder.
"""

new_cells = [md_cell(INTRO), code_cell(ENV_CELL)]
for i, c in enumerate(cells):
    new_cells.append(c)
    if i == 2:                       # right after the setup cell
        new_cells.append(md_cell(
            "## Cell 1B: Checkpoint and resume\n\n"
            "Added for the local run. Defines where each expensive stage stores its "
            "result so that figures can be regenerated later without retraining."))
        new_cells.append(code_cell(CKPT_CELL))

nb["cells"] = new_cells
nb["metadata"].pop("widgets", None)
nb["metadata"]["kernelspec"] = {"display_name": "COAR (venv)",
                                "language": "python", "name": "coar"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.11.9"}
nb["metadata"].pop("colab", None)
nb["metadata"].pop("accelerator", None)

with io.open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nWrote {DST}")
print(f"  cells: {len(cells)} -> {len(new_cells)}")
print(f"  size : {os.path.getsize(DST)/1e6:.2f} MB (outputs stripped from patched cells)")
