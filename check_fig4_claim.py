"""Check the selectivity claim the Figure 4 caption makes, budget by budget."""
import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:\AI-Projects\COAR-Q1")
import matplotlib
matplotlib.use("Agg")

from elsevier_figures import load_state, _selectivity_table, D1_LABEL, D2_LABEL

S = load_state()
for df_key, label in [("edit_df", D1_LABEL), ("edit_df_d2", D2_LABEL)]:
    sel = _selectivity_table(S[df_key], label)
    print(f"\n=== {label} ===")
    print(f"{'k':>4} {'Random':>9} {'LOO-lite':>9} {'COAR':>9}   COAR beats")
    for k in sorted(sel["k"].unique()):
        g = {r["Strategy"]: r["Selectivity ratio"]
             for _, r in sel[sel["k"] == k].iterrows()}
        beats = []
        if g["COAR (t-stat)"] > g["Random"]:
            beats.append("Random")
        if g["COAR (t-stat)"] > g["LOO-lite"]:
            beats.append("LOO-lite")
        print(f"{k:>4} {g['Random']:>9.2f} {g['LOO-lite']:>9.2f} "
              f"{g['COAR (t-stat)']:>9.2f}   {', '.join(beats) if beats else 'NEITHER'}")
