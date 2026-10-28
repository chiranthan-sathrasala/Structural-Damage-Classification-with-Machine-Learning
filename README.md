# Structural Damage Classification with Machine Learning

UE24CS352A Machine Learning mini-project (team of two).
Binary classification of structure photos as **damaged** or **undamaged** on the PEER Hub ImageNet (Φ-Net) Task 2 "Damage State" dataset. We replicate the comparison in Ho & Troncoso (Stanford CS229) between classical models, transfer-learned CNNs and a hybrid CNN-features + SVM model, and extend it (see "What we changed or added").

**Team**
- `Chiranthan Sathrasala, PES1UG24CS565`: classical models, hybrid SVM, data pipeline, final CNN experiments, error analysis, demo, documentation
- `Chiranthan HG, PES1UG24CS564`: first CNN baselines (frozen InceptionV3 and MobileNetV2), first version of the Gradio demo app, initial evaluation plots

## Results at a glance

Test set = PEER's official held-out test set (1,460 images, 51.0% damaged). Validation = a stratified 10% of the training set. The best CNN was chosen on validation accuracy, and the test set was used once per model for reporting only.

| Model | Train acc | Val acc | Test acc | Precision (damaged) | Recall (damaged) | F1 (damaged) |
|---|---|---|---|---|---|---|
| Majority class (always "damaged") | n/a | 0.532 | 0.510 | n/a | n/a | n/a |
| K-NN (k=5), raw pixels | 0.711 | 0.592 | 0.547 | 0.550 | 0.621 | 0.583 |
| Logistic regression, raw pixels | 0.855 | 0.545 | 0.564 | 0.563 | 0.650 | 0.603 |
| RBF-SVM, raw pixels | 0.842 | 0.644 | 0.638 | 0.620 | 0.750 | 0.679 |
| Hybrid: RBF-SVM on InceptionV3 `mixed9` features | 0.929 | 0.847 | 0.855 | 0.873 | 0.838 | 0.855 |
| InceptionV3, frozen backbone | 0.850 | 0.804 | 0.816 | 0.807 | 0.840 | 0.823 |
| MobileNetV2, frozen backbone | 0.859 | 0.831 | 0.840 | 0.822 | 0.875 | 0.848 |
| MobileNetV2, frozen + augmentation | 0.838 | 0.814 | 0.832 | 0.818 | 0.862 | 0.839 |
| MobileNetV2, fine-tuned (top 30 layers) | 0.919 | 0.848 | 0.845 | 0.815 | 0.899 | 0.855 |
| **MobileNetV2, fine-tuned + augmentation (best)** | 0.924 | **0.858** | **0.864** | 0.845 | 0.899 | 0.871 |

The same table is generated from `results/*.json` by `python results_utils.py --table` (saved to `results/comparison.md`). Figure: `docs/model_comparison.png`.

**Main observations**
- Raw-pixel models are weak (best test accuracy 0.638) and overfit (train 0.84 to 0.86 against validation 0.54 to 0.64). Features from a pretrained CNN lift the same classifier (RBF-SVM) by about 21 points.
- All CNN variants reach 0.82 to 0.86 test accuracy. The best (fine-tuned MobileNetV2 with flip/zoom augmentation, 0.864) and the hybrid SVM (0.855) are a tie within noise: with 1,460 test images, each accuracy is uncertain by about ±1.8 points (95%).
- Fine-tuning lowers bias but widens the train/validation gap (0.07 against 0.03 for the frozen model), as in the paper. Augmentation alone did not help the frozen model (0.832 against 0.840) but helped together with fine-tuning. These are single runs, so treat the gains as modest.
- MobileNetV2 beat InceptionV3 in our runs (0.840 against 0.816), the opposite of the paper's finding.
- Recall for the damaged class (0.84 to 0.90 for the CNNs) is higher than precision, which is the safer direction for damage screening. Misclassified examples are mostly ambiguous cases (see `docs/mobilenetv2_finetune30_aug_misclassified_examples.png`).

## Dataset

- PEER Hub ImageNet (Φ-Net), Task 2 "Damage State". Download via the request form at https://apps.peer.berkeley.edu/phi-net/?page_id=1304 (license CC BY-NC-SA 4.0). It gives two zips: one with `task2_X_train.npy`, the other with `task2_X_test.npy`, `task2_y_train.npy` and `task2_y_test.npy`.
- 11,811 training images (53.2% damaged) and 1,460 test images (51.0% damaged), 224×224 RGB. This is a larger release than the ~5,900 images in the paper, so our numbers are not directly comparable with the paper's.
- The raw files are float32 images in ImageNet "caffe" preprocessing (BGR, mean-subtracted) with one-hot labels where 0 = damaged. `data_utils.py` undoes this once and stores RGB uint8 arrays.
- **Label convention everywhere in this project: 1 = damaged (positive class), 0 = undamaged.** Confusion matrices are `[[TN, FP], [FN, TP]]`, rows = true class in the order `[undamaged, damaged]`.
- Split used by all models: about 10,630 train / 1,181 validation (stratified 10% of the training set, seed 42) and the 1,460-image official test set. `load_data` returns every split in a fixed shuffled order.
- Some labels are noisy or ambiguous (for example, plain textured walls labelled "damaged"), which limits the achievable accuracy.

## Setup

Python 3.x with a virtual environment (developed locally with Python 3.14; the CNN training was run on Google Colab with a T4 GPU).

```
git clone https://github.com/chiranthan-sathrasala/Structural-Damage-Classification-with-Machine-Learning.git
cd Structural-Damage-Classification-with-Machine-Learning
python -m venv .venv
.venv\Scripts\activate            # Windows (on Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

The results (`results/`) and figures (`docs/`) are already committed. You only need the steps below to reproduce them.

### 1. Prepare the data (once, about 8 GB of disk during conversion)

Extract both PEER zips, then run from the repo root:
```
python data_utils.py --train_dir "<folder with task2_X_train.npy>" --test_dir "<folder with task2_X_test.npy, task2_y_train.npy, task2_y_test.npy>" --out data
```
(If both zips are extracted into one folder, use that folder for both arguments.) This writes `data/` (git-ignored): `X_train_224.npy`, `X_test_224.npy` (for the CNNs), `X_train_56.npy`, `X_test_56.npy` (for the classical models), `y_train.npy`, `y_test.npy`, and `preview.png`. Open `preview.png` to check: natural colours, damaged images in the top row.

Quick check:
```
python -c "from data_utils import load_data; a=load_data(size=56); print([x.shape for x in a])"
```
It should print about (10630, ...), (1181, ...) and (1460, ...).

### 2. Run everything from the repo root

All scripts must be run from the repo root, because `load_data` looks for `data/` in the current folder. Every model writes `results/<name>.json` through the shared helper `results_utils.py`.

## How to run

### Track A: classical models and hybrid (CPU is enough, except feature extraction)
```
python src/classical/phase1_baselines.py --model all     # K-NN, logistic regression, RBF-SVM on 56x56 pixels (about 30 min, the SVM dominates)
python src/classical/extract_features.py                 # InceptionV3 'mixed9' features -> data/features_mixed9/ (GPU recommended, a few minutes on a Colab T4)
python src/classical/svm_on_features.py                  # hybrid SVM + gamma validation curve (about 1 hour on CPU)
```
Add `--subset 500` to the first and third commands for a quick smoke test (results go to `results_smoke/`). The hybrid script also writes `docs/gamma_curve_mixed9.png`.

### Track B: CNNs (GPU needed, e.g. Google Colab T4)
```
python src/deep/train_cnn.py --backbone inceptionv3                          # InceptionV3, frozen
python src/deep/train_cnn.py --backbone mobilenetv2                          # MobileNetV2, frozen
python src/deep/train_cnn.py --backbone mobilenetv2 --aug                    # frozen + flip/zoom augmentation
python src/deep/train_cnn.py --backbone mobilenetv2 --finetune 30            # fine-tune top 30 layers
python src/deep/train_cnn.py --backbone mobilenetv2 --finetune 30 --aug      # best model
```
Each run takes about 2 to 6 minutes on a T4. Models are saved to `models/<name>.keras` (git-ignored), for example `models/mobilenetv2_finetune30_aug.keras`.

Error analysis and figures:
```
python src/deep/analyze_errors.py mobilenetv2_finetune30_aug mobilenetv2_frozen inceptionv3_frozen   # confusion matrices (+ misclassified gallery for the first model)
python src/deep/plot_comparison.py                                                                   # docs/model_comparison.png
python results_utils.py --table                                                                      # comparison table of all results
```

### Demo app
The trained model is not in Git (file size). Download `mobilenetv2_finetune30_aug.keras` from https://drive.google.com/file/d/1sZpXMuIz4JpsM_owq9O3DgwBcWilNWq6/view?usp=sharing and place it in `models/`. Then:
```
python app/app.py
```
Open http://127.0.0.1:7860 (it opens automatically), upload a photo of a structure, and press "Classify Image". The app shows Damaged / Undamaged with a confidence score. The saved model contains its own preprocessing, so it takes any RGB image.

Check that the app and the saved model agree with the evaluation (six test images through the same function the UI uses):
```
python app/selftest.py
```

## Repository layout

```
data_utils.py          data conversion (run once) and load_data()
results_utils.py       shared result format (save_results), comparison table
requirements.txt       dependencies
src/classical/         Track A: baselines, feature extraction, hybrid SVM
src/deep/              Track B: CNN training, error analysis, comparison figure
app/                   Gradio demo (app.py) and self-test (selftest.py)
results/               one JSON per model, plus comparison.md
docs/                  figures (confusion matrices, gamma curve, error gallery, model comparison)
data/, models/         generated locally, git-ignored
```

## Approach

**Classical models (raw pixels).** Images are downsampled to 56×56 (4×4 block average), scaled to [-1, 1) and flattened to 9,408 features. K-NN (k=5), logistic regression (L2, C=1) and RBF-SVM (C=1, γ=0.001) use the paper's hyperparameters and were not tuned.

**Hybrid model.** Features are extracted from the InceptionV3 layer `mixed9` (ImageNet weights) with global average pooling (2,048 features), standardized (scaler fitted on the training set only), and fed to an RBF-SVM. γ is chosen on validation accuracy from a sweep (1e-5 to 1e-1, best 3e-4); the validation curve is in `docs/gamma_curve_mixed9.png`.

**CNNs (transfer learning).** A frozen ImageNet backbone (InceptionV3 or MobileNetV2), global average pooling, dropout 0.2 and a single sigmoid output (P(damaged)). Pixel scaling to [-1, 1] is a layer inside the model. Adam (lr 1e-3), batch size 32, early stopping on validation loss. For fine-tuning, the top 30 backbone layers are unfrozen (BatchNorm layers kept frozen) and training continues with lr 1e-5. Augmentation is random horizontal flip and zoom (0.1), active only during training.

## What we changed or added compared with the paper

- Larger dataset release with PEER's official test set and a separate validation split (the paper had no held-out test set). All model selection uses validation only; the test set is used once per model.
- Hybrid model reads features from the named layer `mixed9` (the paper used "layer 288", a numbering that depends on the software version) and standardizes them.
- MobileNetV2 instead of MobileNet (V1) as the lightweight CNN.
- **Fine-tuning and augmentation experiments**, including their combination. This improved the frozen MobileNetV2 from 0.840 to 0.864 test accuracy (modest, within about one standard error of the hybrid model).
- Preprocessing built into the saved model, a reusable results format shared by both tracks, error analysis (confusion matrices and misclassified-image gallery), and a working demo app.

## Limitations

- Single training run per configuration, no repeated seeds, so differences of one or two points are within noise.
- The raw-pixel models and the hybrid SVM were not tuned beyond γ for the hybrid; the logistic regression stopped at its iteration limit (500) without fully converging.
- Frozen InceptionV3 scores below the SVM on its own `mixed9` features (0.816 against 0.855). We did not investigate this further. Likely factors are the layer used (the final layer for the CNN head, `mixed9` for the SVM) and the linear head, but this is untested.
- We did not produce a learning curve by training-set size (the paper did).
- Label noise and ambiguity in the dataset limit achievable accuracy.
- The demo is an academic prototype, not a substitute for professional structural inspection.

## Reference and data license

- Ho and Troncoso, Stanford CS229 project on structural damage classification with the PEER Hub ImageNet (Φ-Net) dataset.
- Dataset: PEER Hub ImageNet (Φ-Net), CC BY-NC-SA 4.0.