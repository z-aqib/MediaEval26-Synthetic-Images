The **best experiment is the 2nd run**:

`20260514_105614_zuha_01_efficientnet_b0_constrained_ep5_bs32_lr0p0001_basic`
Training data: **Corvi + Wang CNN Synth Test + Wang Val**
Eval data: **DMImageDetect Test** 

## Ranking by main metrics

| Rank  | Experiment                                        |   Accuracy |         F1 |    ROC AUC |         AP |        EER | Comment                                   |
| ----- | ------------------------------------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ----------------------------------------- |
| **1** | **Exp 2: Corvi + Wang CNN Synth Test + Wang Val** | **0.7255** | **0.7403** | **0.7917** | **0.7978** | **0.2648** | Best overall and most balanced            |
| 2     | Exp 3: Exp 2 + RealRAISE                          |     0.6484 |     0.7107 |     0.7394 |     0.7352 |     0.3235 | Recall high, but too many false positives |
| 3     | Exp 1: Corvi + Wang Val                           |     0.5745 |     0.6866 |     0.6259 |     0.6068 |     0.4190 | Weak baseline, overpredicts synthetic     |

## Why Experiment 2 is clearly better

### 1. Best balanced performance

Experiment 2 has the highest:

* **Accuracy:** 72.55%
* **F1:** 74.03%
* **ROC AUC:** 79.17%
* **Average Precision:** 79.78%
* **Lowest EER:** 26.48%

Since this competition uses **F1 as the main ranking metric**, Exp 2 is your strongest candidate.

## Confusion matrix comparison

### Experiment 1

```text
TN = 867
FP = 3133
FN = 271
TP = 3729
```

This model is basically saying **almost everything is synthetic**.

Good recall, but terrible real-image detection.

### Experiment 2

```text
TN = 2674
FP = 1326
FN = 870
TP = 3130
```

This is much more balanced. It catches synthetic images well, but also correctly identifies many more real images.

### Experiment 3

```text
TN = 1732
FP = 2268
FN = 545
TP = 3455
```

Adding RealRAISE improved recall, but damaged real-image performance badly. It created **too many false positives**.

## Important observation

Experiment 3 added only **1000 extra real images from RealRAISE**, but performance dropped:

```text
Exp 2 accuracy: 0.7255
Exp 3 accuracy: 0.6484
```

That means RealRAISE is probably causing a **domain mismatch**. The real images in RealRAISE may not look like the real images inside DMImageDetect Test. So the model becomes more confused about what “real” means.

## Best experiment to tune next

Tune this one:

```text
Corvi + Wang CNN Synth Test + Wang Val
epochs = 5
batch_size = 32
learning_rate = 0.0001
optimizer = AdamW
weight_decay = 0.0001
augmentation = basic
```

This is your current best base.