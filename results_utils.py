"""
Shared results helper (used by BOTH Person A and Person B).

Every model, classical or deep, records its results with save_results(), so all
JSON files have the exact same schema and the comparison table builds itself.

LABEL CONVENTION: 1 = damaged (positive class), 0 = undamaged.
Confusion matrix layout: rows = true class, columns = predicted class,
order [undamaged, damaged]  ->  [[TN, FP], [FN, TP]].

Usage in model code:
    from results_utils import accuracy, save_results

    train_acc = accuracy(y_tr, model.predict(X_tr))
    val_acc   = accuracy(y_val, model.predict(X_val))
    y_pred_te = model.predict(X_te)                  # hard 0/1 labels, NOT probabilities
    save_results("knn_k5", "A", train_acc, val_acc, y_te, y_pred_te,
                 hyperparameters={"k": 5, "input": "56x56 flattened"},
                 train_time_sec=12.3)

Print the comparison table (and save it to results/comparison.md):
    python results_utils.py --table
"""
import argparse
import glob
import json
import os

import numpy as np

LABEL_ORDER = ["undamaged", "damaged"]


def _as_labels(a, name):
    a = np.asarray(a).ravel()
    vals = set(np.unique(a).tolist())
    if not vals <= {0, 1}:
        raise ValueError(
            f"{name} must contain only 0/1 labels (1 = damaged), found {sorted(vals)[:5]}. "
            "If these are probabilities, threshold them first, e.g. (p > 0.5).astype(int); "
            "if one-hot, use argmax(axis=1).")
    return a.astype(int)


def accuracy(y_true, y_pred):
    y_true, y_pred = _as_labels(y_true, "y_true"), _as_labels(y_pred, "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    return float((y_true == y_pred).mean())


def confusion(y_true, y_pred):
    """[[TN, FP], [FN, TP]] with damaged (1) as the positive class."""
    y_true, y_pred = _as_labels(y_true, "y_true"), _as_labels(y_pred, "y_pred")
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return [[tn, fp], [fn, tp]]


def save_results(model, track, train_acc, val_acc, y_test_true, y_test_pred,
                 hyperparameters=None, train_time_sec=None, results_dir="results"):
    """Write results/<model>.json. `track` is "A" or "B". Returns the dict."""
    if track not in ("A", "B"):
        raise ValueError('track must be "A" or "B"')
    (tn, fp), (fn, tp) = confusion(y_test_true, y_test_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    res = {
        "model": str(model),
        "track": track,
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "test_acc": accuracy(y_test_true, y_test_pred),
        "confusion_matrix_test": [[tn, fp], [fn, tp]],
        "precision_damaged": float(precision),
        "recall_damaged": float(recall),
        "f1_damaged": float(f1),
        "hyperparameters": hyperparameters or {},
        "train_time_sec": None if train_time_sec is None else float(train_time_sec),
    }
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{model}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"Saved {path}  (val {res['val_acc']:.3f}, test {res['test_acc']:.3f}, "
          f"damaged recall {recall:.3f})")
    return res


def load_all(results_dir="results"):
    out = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(p) as f:
            d = json.load(f)
        if "model" in d and "test_acc" in d:     # skip anything that isn't a results file
            out.append(d)
    return sorted(out, key=lambda d: (d.get("track", ""), d["model"]))


def make_table(results_dir="results", save=True):
    rows = load_all(results_dir)
    if not rows:
        print("No results found in", results_dir)
        return ""
    head = "| Model | Track | Train acc | Val acc | Test acc | Precision (damaged) | Recall (damaged) | F1 (damaged) |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for d in rows:
        lines.append(f"| {d['model']} | {d.get('track','')} | {d['train_acc']:.3f} | {d['val_acc']:.3f} | "
                     f"{d['test_acc']:.3f} | {d['precision_damaged']:.3f} | {d['recall_damaged']:.3f} | "
                     f"{d['f1_damaged']:.3f} |")
    table = "\n".join(lines)
    print(table)
    if save:
        with open(os.path.join(results_dir, "comparison.md"), "w") as f:
            f.write(table + "\n")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true", help="print/save the comparison table")
    ap.add_argument("--dir", default="results")
    a = ap.parse_args()
    if a.table:
        make_table(a.dir)
    else:
        ap.print_help()
