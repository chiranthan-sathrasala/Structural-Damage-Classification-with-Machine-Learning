"""
Shared data utilities for the PEER Phi-Net Task 2 (Damage State) project.

LABEL CONVENTION (used everywhere in this project):
    1 = damaged   (positive class)
    0 = undamaged
The raw PEER files use the opposite encoding (0 = damaged, 1 = undamaged) and
are one-hot; prepare_data() converts once, so nobody else has to think about it.

STEP 1 (run once, per machine):
    python data_utils.py --train_dir "C:/Users/you/Downloads/task2_damage_state_2" \
                         --test_dir  "C:/Users/you/Downloads/task2_damage_state_1" \
                         --out data

    This reads the huge float64 files with memory-mapping (never loads 7 GB
    into RAM) and writes compact uint8 copies:
        X_train_224.npy, X_test_224.npy   (N, 224, 224, 3) uint8   -> CNNs
        X_train_56.npy,  X_test_56.npy    (N, 56, 56, 3)   uint8   -> KNN/LR/SVM
        y_train.npy,     y_test.npy       (N,)             int64   (1 = damaged)

STEP 2 (in all model code):
    from data_utils import load_data
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_data(size=224)   # or size=56
"""
import argparse
import os
import numpy as np


# ----------------------------------------------------------------------------
# One-time preparation
# ----------------------------------------------------------------------------
def _labels(path):
    """One-hot (N,2) or 1-D -> 1-D int, with 1 = damaged (PEER raw: 0 = damaged)."""
    y = np.load(path)
    if y.ndim == 2:
        y = y.argmax(axis=1)
    y = y.astype(np.int64).ravel()
    return 1 - y


def _convert_images(src_path, out224, out56, chunk=128):
    X = np.load(src_path, mmap_mode="r")
    n = X.shape[0]
    assert X.shape[1:] == (224, 224, 3), f"unexpected image shape {X.shape}"
    # decide the scaling from the first chunk and apply the same rule throughout
    first = np.asarray(X[:min(chunk, n)], dtype=np.float32)
    lo, hi = float(first.min()), float(first.max())
    if lo < -50:        # ImageNet "caffe" preprocessing: BGR, per-channel mean subtracted
        mode = "caffe (BGR, mean-subtracted) -> undoing"
    elif lo < -0.01:
        mode = "[-1, 1] -> rescaling"
    elif hi <= 1.0 + 1e-6:
        mode = "[0, 1] -> rescaling"
    else:
        mode = "already 0..255"
    print(f"  {os.path.basename(src_path)}: shape={X.shape}, dtype={X.dtype}, "
          f"range of first chunk=[{lo:.3f}, {hi:.3f}]  => {mode}")
    caffe_mean_bgr = np.array([103.939, 116.779, 123.68], dtype=np.float32)

    big = np.lib.format.open_memmap(out224, mode="w+", dtype=np.uint8, shape=(n, 224, 224, 3))
    small = np.lib.format.open_memmap(out56, mode="w+", dtype=np.uint8, shape=(n, 56, 56, 3))
    for i in range(0, n, chunk):
        c = np.asarray(X[i:i + chunk], dtype=np.float32)
        if lo < -50:
            c = (c + caffe_mean_bgr)[..., ::-1]      # add means back, BGR -> RGB
        elif lo < -0.01:
            c = (c + 1.0) * 127.5
        elif hi <= 1.0 + 1e-6:
            c = c * 255.0
        c = np.clip(np.rint(c), 0, 255)
        big[i:i + chunk] = c.astype(np.uint8)
        # 4x4 block average: 224 -> 56 (no extra libraries needed)
        s = c.reshape(c.shape[0], 56, 4, 56, 4, 3).mean(axis=(2, 4))
        small[i:i + chunk] = np.rint(s).astype(np.uint8)
        big.flush(); small.flush()                   # keep RAM use low
        print(f"    {min(i + chunk, n)}/{n}", end="\r")
    big.flush(); small.flush()
    print()


def prepare_data(train_dir, test_dir, out="data"):
    os.makedirs(out, exist_ok=True)
    print("Converting train images...")
    _convert_images(os.path.join(train_dir, "task2_X_train.npy"),
                    os.path.join(out, "X_train_224.npy"), os.path.join(out, "X_train_56.npy"))
    print("Converting test images...")
    _convert_images(os.path.join(test_dir, "task2_X_test.npy"),
                    os.path.join(out, "X_test_224.npy"), os.path.join(out, "X_test_56.npy"))

    y_tr = _labels(os.path.join(test_dir, "task2_y_train.npy"))
    y_te = _labels(os.path.join(test_dir, "task2_y_test.npy"))
    np.save(os.path.join(out, "y_train.npy"), y_tr)
    np.save(os.path.join(out, "y_test.npy"), y_te)

    n_tr = np.load(os.path.join(out, "X_train_56.npy"), mmap_mode="r").shape[0]
    assert n_tr == len(y_tr), f"train images ({n_tr}) and labels ({len(y_tr)}) differ in count!"
    print(f"Train: {len(y_tr)} images, {y_tr.mean():.1%} damaged")
    print(f"Test:  {len(y_te)} images, {y_te.mean():.1%} damaged")
    print(f"Saved to ./{out}/")


# ----------------------------------------------------------------------------
# Loader used by all model code
# ----------------------------------------------------------------------------
def load_data(size=224, data_dir="data", val_frac=0.10, seed=42, mmap=False):
    """
    Returns X_train, y_train, X_val, y_val, X_test, y_test.
    Images are uint8 (N, size, size, 3); labels are 1 = damaged, 0 = undamaged.
    Validation is a stratified split of the TRAIN set (same split for everyone,
    because of the fixed seed). Test is PEER's official held-out test set.

    Scale at use time, e.g. for the paper's [-1, 1) normalisation:
        x = X.astype("float32") / 128.0 - 1.0
    """
    if size not in (224, 56):
        raise ValueError("size must be 224 or 56")
    mode = "r" if mmap else None
    X = np.load(os.path.join(data_dir, f"X_train_{size}.npy"), mmap_mode=mode)
    y = np.load(os.path.join(data_dir, "y_train.npy"))
    X_te = np.load(os.path.join(data_dir, f"X_test_{size}.npy"), mmap_mode=mode)
    y_te = np.load(os.path.join(data_dir, "y_test.npy"))

    # stratified split without needing sklearn
    rng = np.random.RandomState(seed)
    val_idx = []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        val_idx.extend(idx[: int(round(len(idx) * val_frac))])
    val_idx = np.sort(np.array(val_idx))
    mask = np.ones(len(y), dtype=bool)
    mask[val_idx] = False
    tr_idx = np.where(mask)[0]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], X_te, y_te


def preview_samples(data_dir="data", n=8, out="preview.png"):
    """Save a grid of n damaged + n undamaged training images, to eyeball colours and labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X = np.load(os.path.join(data_dir, "X_train_224.npy"), mmap_mode="r")
    y = np.load(os.path.join(data_dir, "y_train.npy"))
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(2, n, figsize=(2 * n, 4.4))
    for r, (cls, name) in enumerate([(1, "DAMAGED"), (0, "UNDAMAGED")]):
        idx = rng.choice(np.where(y == cls)[0], n, replace=False)
        for c, i in enumerate(idx):
            ax[r, c].imshow(X[i]); ax[r, c].axis("off")
        ax[r, 0].set_title(name, loc="left")
    plt.tight_layout(); plt.savefig(os.path.join(data_dir, out), dpi=80)
    print("Saved", os.path.join(data_dir, out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="folder containing task2_X_train.npy")
    ap.add_argument("--test_dir", required=True,
                    help="folder containing task2_X_test.npy, task2_y_train.npy, task2_y_test.npy")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    prepare_data(a.train_dir, a.test_dir, a.out)
    preview_samples(a.out)