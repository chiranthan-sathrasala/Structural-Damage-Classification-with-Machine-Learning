"""Learning curve by training-set size, frozen backbone, head trained on cached features.
Run from the repo root:  python src/deep/learning_curve.py --backbone mobilenetv2
Uses train and validation only (never the test set)."""
import argparse, json, os, sys, time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from data_utils import load_data

BACKBONES = {"inceptionv3": tf.keras.applications.InceptionV3,
             "mobilenetv2": tf.keras.applications.MobileNetV2,
             "mobilenet": tf.keras.applications.MobileNet}
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
REPEATS = 3

def extractor(backbone):
    base = BACKBONES[backbone](weights="imagenet", include_top=False, pooling="avg",
                               input_shape=(224, 224, 3))
    base.trainable = False
    inp = tf.keras.Input((224, 224, 3))
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1.0)(inp)     # same [-1, 1] scaling as the CNNs
    return tf.keras.Model(inp, base(x, training=False))

def stratified_subset(y, frac, rng):
    idx = []
    for c in (0, 1):
        ids = np.where(y == c)[0]
        k = max(1, int(round(frac * len(ids))))
        idx.extend(rng.choice(ids, size=k, replace=False))
    return np.sort(np.array(idx))

def train_head(F_tr, y_tr, F_val, y_val, seed):
    tf.keras.utils.set_random_seed(seed)
    m = tf.keras.Sequential([tf.keras.Input((F_tr.shape[1],)),
                             tf.keras.layers.Dense(1, activation="sigmoid")])
    m.compile(tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    m.fit(F_tr, y_tr, validation_data=(F_val, y_val), epochs=40, batch_size=32, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                                      restore_best_weights=True)])
    acc = lambda F, y: float(((m.predict(F, verbose=0).ravel() > 0.5).astype(int) == y).mean())
    return acc(F_tr, y_tr), acc(F_val, y_val)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=BACKBONES, default="mobilenetv2")
    a = ap.parse_args()

    X_tr, y_tr, X_val, y_val, _, _ = load_data(size=224)
    ex = extractor(a.backbone)
    print("Extracting features (train)...");      F_tr = ex.predict(X_tr, batch_size=64, verbose=1)
    print("Extracting features (validation)..."); F_val = ex.predict(X_val, batch_size=64, verbose=1)
    print("feature size:", F_tr.shape[1])

    train_m, train_s, val_m, val_s, sizes = [], [], [], [], []
    t0, total, done = time.perf_counter(), len(FRACTIONS) * REPEATS, 0
    for frac in FRACTIONS:
        tr, va, n = [], [], 0
        for rep in range(REPEATS):
            idx = stratified_subset(y_tr, frac, np.random.RandomState(100 + rep))
            n = len(idx)
            t_acc, v_acc = train_head(F_tr[idx], y_tr[idx], F_val, y_val, seed=rep)
            tr.append(t_acc); va.append(v_acc); done += 1
            print(f"[{done}/{total}] frac {frac:.2f} (n={n}) rep {rep + 1}: "
                  f"train {t_acc:.3f} val {v_acc:.3f} | elapsed {time.perf_counter() - t0:.0f}s")
        sizes.append(n)
        train_m.append(float(np.mean(tr))); train_s.append(float(np.std(tr)))
        val_m.append(float(np.mean(va)));   val_s.append(float(np.std(va)))

    os.makedirs("docs", exist_ok=True); os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(sizes, train_m, yerr=train_s, marker="o", capsize=3, label="Training accuracy")
    ax.errorbar(sizes, val_m, yerr=val_s, marker="s", capsize=3, label="Validation accuracy")
    ax.set_xlabel("Number of training images"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning curve: frozen {a.backbone}, head trained on cached features")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(f"docs/learning_curve_{a.backbone}.png", dpi=150)
    json.dump({"experiment": f"learning_curve_{a.backbone}", "backbone": a.backbone,
               "fractions": FRACTIONS, "n_train": sizes, "repeats_per_fraction": REPEATS,
               "train_acc_mean": train_m, "train_acc_std": train_s,
               "val_acc_mean": val_m, "val_acc_std": val_s,
               "note": "stratified random subsets; head = Dense(1, sigmoid) on frozen pooled features, no dropout"},
              open(f"results/learning_curve_{a.backbone}.json", "w"), indent=2)
    print("Saved docs/learning_curve_%s.png and results/learning_curve_%s.json" % (a.backbone, a.backbone))

if __name__ == "__main__":
    main()
