"""Trim figure_state.pkl down to what the figure cells actually read.

The full snapshot is ~618 MB, dominated by tokenised datasets and the raw text
splits that no figure touches. Loading it costs minutes per figure. This keeps
only the objects the drawing code uses, so each redraw loads in seconds.
"""
import os, pickle, joblib

PROJECT = r"D:\AI-Projects\COAR-Q1"
SRC = os.path.join(PROJECT, "figure_state.pkl")
DST = os.path.join(PROJECT, "figure_state_slim.pkl")

KEEP = [
    "COAR_RESULTS", "COAR_RESULTS_D2", "LOO_RESULTS", "LOO_RESULTS_D2",
    "coar_summary_df", "coar_summary_df_d2",
    "component_df", "component_importance_df", "component_importance_df_d2",
    "edit_df", "edit_df_d2", "edit_component_df", "edit_component_df_d2",
    "final5_df", "final5_df_d2", "baseline_df", "baseline_df_d2",
    "t_stat", "t_stat_d2", "LOO_DELTA_MATRIX", "LOO_DELTA_MATRIX_D2",
    "final_bert", "edited_bert", "final_bert_d2", "edited_bert_d2",
    "train_log_df", "train_log_df_d2",
    "FINAL_EDIT_IDS", "FINAL_EDIT_IDS_D2", "FINAL_EDIT_MASK", "FINAL_EDIT_MASK_D2",
    "TARGET_INDICES", "REF_INDICES", "TARGET_INDICES_D2", "REF_INDICES_D2",
    "target_labels", "ref_labels", "target_labels_d2", "ref_labels_d2",
    "CONFIG", "NUM_COMPONENTS", "SEED", "ATTN_COMPONENTS", "FFN_COMPONENTS",
    "N_LAYERS", "N_HEADS", "HIDDEN_DIM", "FFN_DIM", "GROUP_SIZE",
    "coar_time", "coar_time_d2", "audit_res", "audit_res_d2",
]

print("loading full state ...", flush=True)
with open(SRC, "rb") as f:
    S = pickle.load(f)

slim = {k: S[k] for k in KEEP if k in S}
missing = [k for k in KEEP if k not in S]

# budget_df / selectivity_ci_df live in their own checkpoints (cells F11/F12).
ck = os.path.join(PROJECT, "checkpoints")
for stage in ("f11_budget", "f12_ci"):
    p = os.path.join(ck, stage + ".pkl")
    if os.path.exists(p):
        slim.update(joblib.load(p))

with open(DST, "wb") as f:
    pickle.dump(slim, f, protocol=4)

print(f"kept {len(slim)} objects")
print(f"  {os.path.getsize(SRC)/1e6:8.1f} MB  ->  {os.path.getsize(DST)/1e6:.1f} MB")
if missing:
    print("  not present in snapshot:", ", ".join(missing))
