import os, pandas as pd
os.chdir(r"D:\AI-Projects\COAR-Q1\outputs")
pd.set_option("display.width", 210)
pd.set_option("display.max_colwidth", 64)

print("===== FINAL SUMMARY (both datasets) =====")
print(pd.read_csv("FINAL_SUMMARY_both_datasets.csv").to_string(index=False))

print("\n===== COAR vs LOO =====")
print(pd.read_csv("coar_vs_loo_both_datasets.csv").to_string(index=False))

print("\n===== FINAL 5-MODEL COMPARISON (both datasets) =====")
print(pd.read_csv("final5_both_datasets.csv").to_string(index=False))
