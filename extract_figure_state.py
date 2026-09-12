"""Run the notebook's compute cells from checkpoints, then cache everything
the figure cells need.

Nothing is retrained: every guarded stage restores from checkpoints/. This
exists so each figure can be redrawn in seconds instead of re-running the
whole notebook, which is what the one-figure-at-a-time review loop needs.
"""
import json, io, os, sys, pickle, warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")                      # never open a window

NB = r"D:\AI-Projects\COAR-Q1\COAR_DistilBERT_LOCAL.ipynb"
OUT = r"D:\AI-Projects\COAR-Q1\figure_state.pkl"
F1_INDEX = 47                              # first figure cell; we stop before it

nb = json.load(io.open(NB, encoding="utf-8"))
cells = nb["cells"]

ns = {"__name__": "__main__"}
for i, c in enumerate(cells):
    if i >= F1_INDEX:
        break
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if "CELL 12 - Export ZIP" in src:      # skip the zip step, it is not needed here
        continue
    print(f"\n########## executing cell {i} ##########", flush=True)
    exec(compile(src, f"<cell {i}>", "exec"), ns)

# ---- cache every picklable global ----
skip_types = (type(sys), type(lambda: 0), type(exec))
blob, dropped = {}, []
for k, v in list(ns.items()):
    if k.startswith("__") or isinstance(v, skip_types) or isinstance(v, type):
        continue
    try:
        pickle.dumps(v)
        blob[k] = v
    except Exception:
        dropped.append(k)

with open(OUT, "wb") as f:
    pickle.dump(blob, f, protocol=4)

print(f"\n\ncached {len(blob)} objects -> {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
print("dropped (not picklable, not needed for drawing):")
print("  ", ", ".join(sorted(dropped)))

need = ["COAR_RESULTS", "COAR_RESULTS_D2", "LOO_RESULTS", "LOO_RESULTS_D2",
        "coar_summary_df", "coar_summary_df_d2", "component_df",
        "component_importance_df", "component_importance_df_d2",
        "edit_df", "edit_df_d2", "final5_df", "final5_df_d2",
        "t_stat", "t_stat_d2", "LOO_DELTA_MATRIX", "LOO_DELTA_MATRIX_D2",
        "budget_df", "selectivity_ci_df", "CONFIG", "NUM_COMPONENTS",
        "target_texts", "ref_texts", "final_bert", "edited_bert"]
missing = [n for n in need if n not in blob]
print("\nfigure inputs missing:", missing if missing else "none - all present")
