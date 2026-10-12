"""Phase 2, Step 1: extract InceptionV3 intermediate-layer features.

Run from the REPO ROOT:
    python src/classical/extract_features.py                 # default layer: mixed9
    python src/classical/extract_features.py --subset 200    # quick test

Output (in data/features_<layer>/, git-ignored because of *.npy):
    X_train.npy, X_val.npy, X_test.npy   float32, shape (N, n_features)
    y_train.npy, y_val.npy, y_test.npy   int labels, 1 = damaged
"""
import argparse
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_utils import load_data


def build_extractor(layer_name):
    """InceptionV3 (ImageNet weights) that stops at `layer_name`, then averages
    each feature map into one number -> one fixed-length vector per image."""
    import tensorflow as tf
    from tensorflow.keras.applications import InceptionV3

    base = InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    out = base.get_layer(layer_name).output
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    return tf.keras.Model(inputs=base.input, outputs=out)


def extract(model, X_uint8, batch_size=32):
    """uint8 RGB images -> features. Preprocessing must match what InceptionV3
    was trained with: pixels scaled to [-1, 1]."""
    from tensorflow.keras.applications.inception_v3 import preprocess_input

    feats = []
    n = len(X_uint8)
    start = time.perf_counter()
    for i in range(0, n, batch_size):
        batch = preprocess_input(X_uint8[i:i + batch_size].astype("float32"))
        feats.append(model.predict(batch, verbose=0))
        done = min(i + batch_size, n)
        frac = done / n
        elapsed = time.perf_counter() - start
        eta = elapsed / frac * (1 - frac)
        bar = "#" * int(frac * 30) + "-" * (30 - int(frac * 30))
        print(f"\r    [{bar}] {100 * frac:5.1f}% ({done}/{n})  "
              f"elapsed {elapsed:5.0f}s  ETA {eta:5.0f}s   ", end="", flush=True)
    print()
    return np.concatenate(feats).astype("float32")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", default="mixed9",
                    help="InceptionV3 layer name to read features from (default: mixed9)")
    ap.add_argument("--subset", type=int, default=0, help="use only N random images per split (test run)")
    a = ap.parse_args()

    out_dir = os.path.join("data", f"features_{a.layer}" + ("_smoke" if a.subset else ""))
    os.makedirs(out_dir, exist_ok=True)

    print("Loading 224x224 images (about 2 GB of RAM)...")
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_data(size=224)
    if a.subset:
        rng = np.random.RandomState(42)   # random, not "first N" (data is class-sorted in raw form)
        pick = lambda X, y: (lambda i: (X[i], y[i]))(rng.choice(len(X), min(a.subset, len(X)), replace=False))
        X_tr, y_tr = pick(X_tr, y_tr); X_val, y_val = pick(X_val, y_val); X_te, y_te = pick(X_te, y_te)

    print(f"Building InceptionV3 up to layer '{a.layer}' (downloads ImageNet weights the first time)...")
    model = build_extractor(a.layer)
    print(f"Feature vector length per image: {model.output_shape[-1]}")

    t0 = time.perf_counter()
    for name, X, y in (("train", X_tr, y_tr), ("val", X_val, y_val), ("test", X_te, y_te)):
        print(f"[{name}] extracting...")
        F = extract(model, X)
        np.save(os.path.join(out_dir, f"X_{name}.npy"), F)
        np.save(os.path.join(out_dir, f"y_{name}.npy"), y)
        print(f"    saved {F.shape}  (elapsed {time.perf_counter() - t0:.0f}s)")
    print(f"Done. Features are in ./{out_dir}/")


if __name__ == "__main__":
    main()