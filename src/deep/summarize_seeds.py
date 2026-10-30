"""Mean/std over seeds. Seed 42 = the main run in results/, others are in results/seeds/.
Run from the repo root:  python src/deep/summarize_seeds.py"""
import glob, json
import numpy as np

BASES = ["mobilenetv2_frozen", "mobilenetv2_finetune30_aug"]
rows, out = [], {}
for base in BASES:
    files = [f"results/{base}.json"] + sorted(glob.glob(f"results/seeds/{base}_seed*.json"))
    runs = [json.load(open(f)) for f in files]
    val = np.array([r["val_acc"] for r in runs]); test = np.array([r["test_acc"] for r in runs])
    sd = lambda v: float(v.std(ddof=1)) if len(v) > 1 else 0.0
    out[base] = {"n_runs": len(runs), "val_mean": float(val.mean()), "val_std": sd(val),
                 "test_mean": float(test.mean()), "test_std": sd(test),
                 "test_min": float(test.min()), "test_max": float(test.max()),
                 "test_each": [round(float(t), 4) for t in test]}
    rows.append(f"| {base} | {len(runs)} | {val.mean():.3f} ± {sd(val):.3f} | "
                f"{test.mean():.3f} ± {sd(test):.3f} | {test.min():.3f} to {test.max():.3f} |")
table = ("| Model | Runs | Val acc (mean ± std) | Test acc (mean ± std) | Test min to max |\n"
         "|---|---|---|---|---|\n" + "\n".join(rows))
print(table)
json.dump(out, open("results/seed_summary.json", "w"), indent=2)
open("results/seed_summary.md", "w").write(table + "\n")
