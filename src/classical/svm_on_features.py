"""Phase 2, Step 2: SVM on InceptionV3 intermediate-layer features.

Run from the REPO ROOT (after extract_features.py produced data/features_mixed9/):
    python src/classical/svm_on_features.py
    python src/classical/svm_on_features.py --layer mixed9 --subset 800   # quick test

What it does:
  1. Loads the saved feature vectors.
  2. Standardises them (mean 0, std 1 per feature) - scaler fitted on TRAIN only.
  3. Trains an RBF-SVM (C=1) for several values of gamma and records train and
     validation accuracy -> the validation curve (like Figure 3 of the paper).
  4. Picks the gamma with the best VALIDATION accuracy, evaluates it on the
     test set ONCE, and saves results/hybrid_svm_<layer>.json.
"""
import argparse
import os
import sys
import threading
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from results_utils import accuracy, save_results


def fmt(seconds):
    m, s = divmod(int(max(seconds, 0)), 60)
    return f"{m:02d}:{s:02d}"


class Heartbeat:
    """Live 'still running' line (elapsed time) while one long call, e.g. model.fit(), runs.
    fit() reports no percentage, so we show that the program is alive instead."""

    def __init__(self, label):
        self.label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        t0 = time.perf_counter()
        while not self._stop.is_set():
            sys.stdout.write(f"\r      {self.label} ... running, elapsed {fmt(time.perf_counter() - t0)}   ")
            sys.stdout.flush()
            self._stop.wait(1.0)

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        sys.stdout.write(f"\r      {self.label} ... done in {fmt(time.perf_counter() - self._t0)}          \n")
        sys.stdout.flush()


def predict_in_chunks(model, X, label, chunk=500):
    """model.predict(X) in pieces, with a progress bar (percent + ETA)."""
    n, preds, t0 = len(X), [], time.perf_counter()
    for i in range(0, n, chunk):
        preds.append(model.predict(X[i:i + chunk]))
        done = min(i + chunk, n)
        frac = done / n
        el = time.perf_counter() - t0
        bar = "#" * int(frac * 30) + "-" * (30 - int(frac * 30))
        sys.stdout.write(f"\r      {label:<18} [{bar}] {100 * frac:5.1f}% ({done}/{n}) "
                         f"elapsed {fmt(el)} ETA {fmt(el / frac * (1 - frac))}   ")
        sys.stdout.flush()
    sys.stdout.write("\n")
    return np.concatenate(preds)


def load_features(layer, subset=0):
    d = os.path.join("data", f"features_{layer}")
    out = {}
    for split in ("train", "val", "test"):
        out[split] = (np.load(os.path.join(d, f"X_{split}.npy")),
                      np.load(os.path.join(d, f"y_{split}.npy")))
        assert len(out[split][0]) == len(out[split][1]), f"{split}: features/labels count differ"
    if subset:   # random subset (never "first N")
        rng = np.random.RandomState(42)
        for split in out:
            X, y = out[split]
            i = rng.choice(len(X), min(subset, len(X)), replace=False)
            out[split] = (X[i], y[i])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", default="mixed9")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--gammas", type=float, nargs="+",
                    default=[1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1],
                    help="gamma values to try (default: a log-spaced sweep)")
    ap.add_argument("--subset", type=int, default=0, help="quick test on N random samples per split")
    ap.add_argument("--results_dir", default="results")
    a = ap.parse_args()
    results_dir = a.results_dir
    if a.subset and results_dir == "results":
        results_dir = "results_smoke"
        print("Subset run: results go to results_smoke/ (not results/).")

    data = load_features(a.layer, a.subset)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = data["train"], data["val"], data["test"]
    print(f"Features: train={X_tr.shape}, val={X_val.shape}, test={X_te.shape}")

    # Standardise: fit on TRAIN only, then apply the same transform to val/test.
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_val, X_te = scaler.transform(X_tr), scaler.transform(X_val), scaler.transform(X_te)

    rows, best = [], None
    start = time.perf_counter()
    for k, g in enumerate(a.gammas, start=1):
        t0 = time.perf_counter()
        print(f"  [{k}/{len(a.gammas)}] gamma={g:g}")
        model = SVC(kernel="rbf", C=a.C, gamma=g, cache_size=1000, random_state=42)
        with Heartbeat("fitting SVM"):
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t0
        tr_acc = accuracy(y_tr, predict_in_chunks(model, X_tr, "predict train"))
        val_acc = accuracy(y_val, predict_in_chunks(model, X_val, "predict validation"))
        rows.append((g, tr_acc, val_acc))
        elapsed = time.perf_counter() - start
        eta = elapsed / k * (len(a.gammas) - k)
        print(f"      -> train={tr_acc:.3f} val={val_acc:.3f}  "
              f"(overall elapsed {fmt(elapsed)}, ETA for remaining gammas {fmt(eta)})")
        if best is None or val_acc > best["val_acc"]:
            best = dict(gamma=g, model=model, train_acc=tr_acc, val_acc=val_acc, fit_s=fit_s)

    print(f"\nBest gamma by VALIDATION accuracy: {best['gamma']:g} (val {best['val_acc']:.3f})")

    # Validation curve figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g, tr, va = zip(*rows)
    plt.figure(figsize=(6, 4))
    plt.semilogx(g, tr, "o-", label="Training accuracy")
    plt.semilogx(g, va, "s-", label="Validation accuracy")
    plt.axvline(best["gamma"], ls=":", color="gray")
    plt.xlabel("gamma (RBF kernel coefficient)"); plt.ylabel("Accuracy")
    plt.title(f"Validation curve: SVM on InceptionV3 '{a.layer}' features")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    os.makedirs("docs", exist_ok=True)
    fig_path = os.path.join("docs", f"gamma_curve_{a.layer}{'_smoke' if a.subset else ''}.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved figure: {fig_path}")

    # Test set: used ONCE, for the model chosen on validation.
    y_pred_te = predict_in_chunks(best["model"], X_te, "predict test")
    save_results(f"hybrid_svm_{a.layer}", "A", best["train_acc"], best["val_acc"], y_te, y_pred_te,
                 hyperparameters={"kernel": "rbf", "C": a.C, "gamma": best["gamma"],
                                  "gamma_grid": list(a.gammas), "features": f"InceptionV3 {a.layer} (GAP)",
                                  "scaling": "StandardScaler fitted on train"},
                 train_time_sec=best["fit_s"], results_dir=results_dir)


if __name__ == "__main__":
    main()