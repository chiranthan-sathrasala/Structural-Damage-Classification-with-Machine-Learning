import argparse, json, os, random, sys, time
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from data_utils import load_data
from results_utils import accuracy, save_results

SEED = 42
BACKBONES = {"inceptionv3": tf.keras.applications.InceptionV3,
             "mobilenetv2": tf.keras.applications.MobileNetV2,
             "mobilenet": tf.keras.applications.MobileNet}

def build(backbone, aug):
    base = BACKBONES[backbone](weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    inp = tf.keras.Input((224, 224, 3))            # raw RGB 0..255
    x = inp
    if aug:                                        # only active during training
        x = tf.keras.layers.RandomFlip("horizontal")(x)
        x = tf.keras.layers.RandomZoom(0.1)(x)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1.0, name="preprocess")(x)  # -> [-1, 1]
    x = base(x, training=False)                    # BatchNorm stays in inference mode
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="p_damaged")(x)   # P(damaged)
    return tf.keras.Model(inp, out), base

def labels(model, X, name):
    print(f"  predicting {name} ({len(X)})")
    return (model.predict(X, batch_size=64, verbose=1).ravel() > 0.5).astype(int)

def fit(model, X_tr, y_tr, X_val, y_val, epochs, lr, ckpt):
    model.compile(tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
           tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True)]
    return model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs,
                     batch_size=32, callbacks=cbs, verbose=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=BACKBONES, default="mobilenetv2")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--finetune", type=int, default=0, help="unfreeze top N backbone layers")
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_data(size=224)
    print("train/val/test:", len(X_tr), len(X_val), len(X_te), "| test damaged:", int(y_te.sum()))

    name = f"{a.backbone}_" + (f"finetune{a.finetune}" if a.finetune else "frozen") + ("_aug" if a.aug else "")
    os.makedirs("models", exist_ok=True)
    ckpt = f"models/{name}.keras"
    model, base = build(a.backbone, a.aug)

    t0 = time.perf_counter()
    h = fit(model, X_tr, y_tr, X_val, y_val, a.epochs, 1e-3, ckpt)          # phase 1: train the head
    epochs_run = len(h.history["loss"])
    if a.finetune:                                                           # phase 2: unfreeze top layers
        base.trainable = True
        for l in base.layers[:-a.finetune]:
            l.trainable = False
        for l in base.layers:
            if isinstance(l, tf.keras.layers.BatchNormalization):
                l.trainable = False
        h2 = fit(model, X_tr, y_tr, X_val, y_val, 8, 1e-5, ckpt)
        epochs_run += len(h2.history["loss"])
    secs = time.perf_counter() - t0

    train_acc = accuracy(y_tr, labels(model, X_tr, "train"))
    val_acc = accuracy(y_val, labels(model, X_val, "validation"))
    y_pred_te = labels(model, X_te, "test")                                  # test used once, for reporting
    save_results(name, "B", train_acc, val_acc, y_te, y_pred_te,
                 hyperparameters={"backbone": a.backbone, "optimizer": "adam", "lr": 1e-3,
                                  "finetune_top_layers": a.finetune, "finetune_lr": 1e-5 if a.finetune else None,
                                  "augmentation": "flip+zoom0.1" if a.aug else "none",
                                  "epochs_run": epochs_run, "batch_size": 32,
                                  "preprocessing": "Rescaling [-1,1] inside model"},
                 train_time_sec=secs, results_dir="results")

if __name__ == "__main__":
    main()
