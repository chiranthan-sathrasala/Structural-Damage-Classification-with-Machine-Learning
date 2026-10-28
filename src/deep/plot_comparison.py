# src/deep/plot_comparison.py  -  run from the repo root: python src/deep/plot_comparison.py
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from results_utils import load_all

rows = sorted(load_all("results"), key=lambda d: d["test_acc"])
pos = range(len(rows))
colors = ["#1f77b4" if d["track"] == "A" else "#ff7f0e" for d in rows]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.barh([i + 0.2 for i in pos], [d["test_acc"] for d in rows], height=0.4, color=colors)
ax.barh([i - 0.2 for i in pos], [d["val_acc"] for d in rows], height=0.4, color=colors, alpha=0.45)
ax.axvline(0.510, color="gray", linestyle=":")       # majority-class baseline on the test set
ax.set_yticks(list(pos)); ax.set_yticklabels([d["model"] for d in rows])
ax.set_xlim(0.45, 0.9); ax.set_xlabel("Accuracy (dark = test, light = validation; dotted = majority class)")
ax.legend(handles=[Patch(color="#1f77b4", label="Track A (classical / hybrid)"),
                   Patch(color="#ff7f0e", label="Track B (CNN)")], loc="lower right")
ax.set_title("Model comparison")
fig.tight_layout()
os.makedirs("docs", exist_ok=True)
fig.savefig("docs/model_comparison.png", dpi=150)
print("Saved docs/model_comparison.png")