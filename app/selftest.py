import os, sys
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "app"))
from data_utils import load_data
from app import predict_damage

_, _, _, _, X_te, y_te = load_data(size=224)
idx = list(np.where(y_te == 1)[0][:3]) + list(np.where(y_te == 0)[0][:3])
print("idx | true | predicted | P(damaged)")
for i in idx:
    out = predict_damage(Image.fromarray(X_te[i]))
    pred = "Damaged" if out["Damaged"] > 0.5 else "Undamaged"
    print(i, "|", "Damaged" if y_te[i] == 1 else "Undamaged", "|", pred, "|", round(out["Damaged"], 3))