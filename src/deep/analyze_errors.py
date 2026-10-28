"""Confusion matrix + misclassified-image gallery. Run from the repo root:
    python src/deep/analyze_errors.py <best_model> [other models ...]
The gallery is made for the FIRST model only. Labels: 1 = damaged."""
import os, sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from data_utils import load_data
from results_utils import confusion

NAMES = ["Undamaged", "Damaged"]

def gallery(X, y, p, name, k=6):
    pred = (p > 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)              # confidence in the predicted class
    fp = np.where((y == 0) & (pred == 1))[0]          # undamaged, predicted damaged
    fn = np.where((y == 1) & (pred == 0))[0]          # damaged, predicted undamaged
    fig, axes = plt.subplots(2, k, figsize=(2.2 * k, 5.6))
    for r, (idx, tag) in enumerate([(fp, "FP"), (fn, "FN")]):
        idx = idx[np.argsort(-conf[idx])][:k]         # most confident mistakes first
        for c in range(k):
            axes[r, c].axis("off")
            if c < len(idx):
                axes[r, c].imshow(X[idx[c]])
                axes[r, c].set_title(f"{tag} conf {conf[idx[c]]:.2f}", fontsize=9)
    fig.suptitle(f"{name}: top row = false positives (true Undamaged, predicted Damaged); "
                 "bottom row = false negatives (true Damaged, predicted Undamaged)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"docs/{name}_misclassified_examples.png", dpi=110)
    plt.close(fig)

def main():
    names = sys.argv[1:] or ["mobilenetv2_finetune30_aug"]
    _, _, _, _, X_te, y_te = load_data(size=224)
    os.makedirs("docs", exist_ok=True)
    for n, name in enumerate(names):
        model = tf.keras.models.load_model(f"models/{name}.keras", compile=False)
        p = model.predict(X_te, batch_size=64, verbose=1).ravel()
        cm = np.array(confusion(y_te, (p > 0.5).astype(int)))
        print(name, "[[TN, FP], [FN, TP]] =", cm.tolist(),
              "| true-damaged row sum:", int(cm[1].sum()), f"| acc {np.trace(cm) / cm.sum():.4f}")
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay(cm, display_labels=NAMES).plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title(f"{name} - test confusion matrix")
        fig.tight_layout()
        fig.savefig(f"docs/{name}_confusion_matrix.png", dpi=150)
        plt.close(fig)
        if n == 0:
            gallery(X_te, y_te, p, name)

if __name__ == "__main__":
    main()
