"""Phase 1 classical baselines on flattened, normalized 56x56 RGB images.

Run from the REPO ROOT:
    python src/classical/phase1_baselines.py --model all
Quick smoke test (writes to results_smoke/, not results/):
    python src/classical/phase1_baselines.py --model all --subset 500
"""

import argparse
import os
import random
import sys
import threading
import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# The script lives in src/classical/, two levels below the repo root. Python only
# searches the script's own folder by default, so we add the repo root to the
# search path to be able to import data_utils.py and results_utils.py from there.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_utils import load_data
from results_utils import accuracy, save_results


# ----------------------------------------------------------------------------
# Progress helpers
# ----------------------------------------------------------------------------
def fmt_time(seconds):
    """Turn a number of seconds into mm:ss (or h:mm:ss)."""
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Heartbeat:
    """Show a live 'still running' line with elapsed time while a long call runs.

    model.fit() is ONE long library call: it gives us no way to ask "how far along
    are you?", so a true percentage is impossible for the fit. What we CAN show is
    that the program is alive, by printing the elapsed time every second from a
    background thread. If this counter stops ticking, something is wrong.

    Usage:
        with Heartbeat("Fitting SVM"):
            model.fit(X, y)
    """

    def __init__(self, label):
        self.label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        start = time.perf_counter()
        while not self._stop.is_set():
            elapsed = fmt_time(time.perf_counter() - start)
            sys.stdout.write(f"\r    {self.label} ... running, elapsed {elapsed}   ")
            sys.stdout.flush()
            self._stop.wait(1.0)      # wakes up immediately when stop is set

    def __enter__(self):
        self._start = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        elapsed = fmt_time(time.perf_counter() - self._start)
        sys.stdout.write(f"\r    {self.label} ... done in {elapsed}          \n")
        sys.stdout.flush()


def predict_in_chunks(model, X, label, chunk=500):
    """model.predict(X), but in chunks, with a progress bar (percent + ETA).

    Predicting is the same work as predicting all at once; we just split X into
    pieces of `chunk` rows so we can update the bar between pieces.
    """
    n = len(X)
    preds = []
    start = time.perf_counter()
    for done in range(0, n, chunk):
        preds.append(model.predict(X[done:done + chunk]))
        finished = min(done + chunk, n)
        frac = finished / n
        elapsed = time.perf_counter() - start
        eta = elapsed / frac * (1 - frac)
        bar = "#" * int(frac * 30) + "-" * (30 - int(frac * 30))
        sys.stdout.write(
            f"\r    {label:<22} [{bar}] {100 * frac:5.1f}% "
            f"({finished}/{n})  elapsed {fmt_time(elapsed)}  ETA {fmt_time(eta)}   "
        )
        sys.stdout.flush()
    sys.stdout.write("\n")
    return np.concatenate(preds)


# ----------------------------------------------------------------------------
# Data and diagnostics
# ----------------------------------------------------------------------------
def preprocess(images):
    """Scale uint8 images to [-1, 1) and flatten them into feature vectors."""
    return (images.astype(np.float32) / 128.0 - 1.0).reshape(len(images), -1)


def random_subset(X, y, n, seed=42):
    """Return n randomly chosen samples (same choice every run, thanks to the seed)."""
    n = min(n, len(X))
    idx = np.random.RandomState(seed).choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def print_dataset_info(X_tr, y_tr, X_val, y_val, X_te, y_te):
    """Print the common dataset diagnostics used for all three models."""
    print(
        "Dataset: "
        f"train={X_tr.shape}, val={X_val.shape}, test={X_te.shape}; "
        f"dtype={X_tr.dtype}, scaled range=[{X_tr.min():.6f}, {X_tr.max():.6f}]"
    )
    print(
        "Percent damaged: "
        f"train={100 * y_tr.mean():.2f}%, "
        f"val={100 * y_val.mean():.2f}%, "
        f"test={100 * y_te.mean():.2f}%"
    )
    majority_class = int(np.bincount(y_val).argmax())
    majority_acc = accuracy(y_val, np.full(y_val.shape, majority_class))
    print(
        f"Validation majority-class baseline: {majority_acc:.3f} "
        f"(always predicts {majority_class}, 1 = damaged)"
    )


# ----------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------
def run_model(name, model, hyperparameters, X_tr, y_tr, X_val, y_val, X_te, y_te,
              results_dir):
    """Fit one model, evaluate it, and persist its shared-schema results."""
    # Step 1: fit on the TRAINING set only. This is the slow part for the SVM.
    with Heartbeat(f"Fitting {name}"):
        start = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_seconds = time.perf_counter() - start   # time of model.fit() only

    # Step 2: predictions (with progress bars). The TEST set is predicted once,
    # purely to report the final result - no decision is based on it.
    y_pred_tr = predict_in_chunks(model, X_tr, "predict train")
    y_pred_val = predict_in_chunks(model, X_val, "predict validation")
    y_pred_te = predict_in_chunks(model, X_te, "predict test")

    train_acc = accuracy(y_tr, y_pred_tr)
    val_acc = accuracy(y_val, y_pred_val)

    save_results(
        name,
        "A",
        train_acc,
        val_acc,
        y_te,
        y_pred_te,
        hyperparameters=hyperparameters,
        train_time_sec=fit_seconds,
        results_dir=results_dir,
    )
    print(
        f"{name}: train_acc={train_acc:.3f}, "
        f"val_acc={val_acc:.3f}, fit_time_sec={fit_seconds:.2f}"
    )


def run_knn(X_tr, y_tr, X_val, y_val, X_te, y_te, results_dir):
    # K-NN classifies by nearby training examples; k=5 follows the paper exactly.
    # fit() only stores the data; the real work happens at prediction time.
    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    run_model(
        "knn_k5",
        model,
        {
            "n_neighbors": 5,
            "n_jobs": -1,
            "input": "56x56x3 flattened, scaled to [-1,1)",
        },
        X_tr, y_tr, X_val, y_val, X_te, y_te, results_dir,
    )
    # NOTE: K-NN train accuracy is optimistic because each training point is
    # its own nearest neighbour when we predict on the training set.


def run_logreg(X_tr, y_tr, X_val, y_val, X_te, y_te, results_dir):
    # Logistic regression provides a linear baseline; these L2/lbfgs settings are
    # from the paper.
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=500,
        random_state=42,
    )
    run_model(
        "logreg_l2",
        model,
        {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 500,
            "random_state": 42,
            "input": "56x56x3 flattened, scaled to [-1,1)",
        },
        X_tr, y_tr, X_val, y_val, X_te, y_te, results_dir,
    )


def run_svm(X_tr, y_tr, X_val, y_val, X_te, y_te, results_dir):
    # RBF-SVM captures nonlinear pixel relationships using the paper's fixed
    # settings. This is the slowest model: fitting uses a single CPU core, which
    # is why the CPU % in Task Manager looks low.
    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma=0.001,
        cache_size=1000,
        random_state=42,
    )
    run_model(
        "svm_rbf",
        model,
        {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": 0.001,
            "cache_size": 1000,
            "random_state": 42,
            "input": "56x56x3 flattened, scaled to [-1,1)",
        },
        X_tr, y_tr, X_val, y_val, X_te, y_te, results_dir,
    )


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        choices=("knn", "logreg", "svm", "all"),
        default="all",
        help="baseline to run (default: all)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=0,
        help="use the first N training samples and max(N//4, 50) val/test samples",
    )
    parser.add_argument(
        "--results_dir",
        default="results",
        help="directory for result JSON files (default: results)",
    )
    args = parser.parse_args()
    if args.subset < 0:
        parser.error("--subset must be non-negative")
    return args


def main():
    args = parse_args()
    np.random.seed(42)
    random.seed(42)

    results_dir = args.results_dir
    if args.subset and results_dir == "results":
        results_dir = "results_smoke"
        print("Subset run detected; switching results directory to results_smoke.")

    X_tr, y_tr, X_val, y_val, X_te, y_te = load_data(size=56)
    if args.subset:
        # IMPORTANT: the images are stored SORTED BY CLASS, so taking the "first N"
        # would give only one class (and models like logistic regression refuse to
        # train on one class). We take a RANDOM sample instead (fixed seed, so the
        # same subset every time).
        n_small = max(args.subset // 4, 50)
        X_tr, y_tr = random_subset(X_tr, y_tr, args.subset)
        X_val, y_val = random_subset(X_val, y_val, n_small)
        X_te, y_te = random_subset(X_te, y_te, n_small)

    X_tr = preprocess(X_tr)
    X_val = preprocess(X_val)
    X_te = preprocess(X_te)
    print_dataset_info(X_tr, y_tr, X_val, y_val, X_te, y_te)

    runners = {
        "knn": run_knn,
        "logreg": run_logreg,
        "svm": run_svm,
    }
    selected = list(runners.keys()) if args.model == "all" else [args.model]
    overall_start = time.perf_counter()
    for i, model_name in enumerate(selected, start=1):
        print(f"\n=== Model {i}/{len(selected)}: {model_name} "
              f"(total elapsed {fmt_time(time.perf_counter() - overall_start)}) ===")
        runners[model_name](X_tr, y_tr, X_val, y_val, X_te, y_te, results_dir)
    print(f"\nAll done in {fmt_time(time.perf_counter() - overall_start)}. "
          f"Results are in ./{results_dir}/")


if __name__ == "__main__":
    main()