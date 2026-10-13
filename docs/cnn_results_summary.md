# Structural Damage Classification — CNN Results

## Models
- Frozen InceptionV3
- Frozen MobileNetV2

## Official test results

| Model | Test accuracy |
|---|---:|
| Frozen InceptionV3 | 77.2603% |
| Frozen MobileNetV2 | 83.0822% |

## Comparison
MobileNetV2 achieved 5.82
percentage points higher test accuracy than InceptionV3 on the same official test set.

## Saved artifacts
- `models/inceptionv3_frozen_best.keras`
- `models/mobilenetv2_frozen_best.keras`
- `results/inceptionv3_frozen.json`
- `results/mobilenetv2_frozen.json`
- `results/mobilenetv2_error_analysis.json`
- `docs/inceptionv3_frozen_confusion_matrix.png`
- `docs/mobilenetv2_frozen_confusion_matrix.png`
- `docs/mobilenetv2_misclassified_examples.png`
- `docs/cnn_model_comparison.png`
- `app/app.py`
- `app/requirements.txt`

## Limitations
The CNNs are frozen-backbone baselines. The learning curve and final integration
with the full project repository still need verification.


## MobileNetV2 training history
- `results/mobilenetv2_training_history.json`
- `docs/mobilenetv2_learning_curve_accuracy.png`
- `docs/mobilenetv2_learning_curve_loss.png`

The training and validation accuracy/loss curves are saved from the recorded
MobileNetV2 training run. Validation loss fluctuates in later epochs, so
generalization should be assessed using the held-out test metrics as well.
