# Model Ideas for Task A: Real vs Synthetic Image Detection

This file lists model notebooks the team can create and try after the first three base notebooks.

Current notebooks:
- `01_efficientnet_b0_experiments.ipynb`
- `02_convnext_tiny_experiments.ipynb`
- `03_clip_vit_experiments.ipynb`

Main goal for all models:
- Train on selected datasets.
- Evaluate on the same fixed evaluation dataset.
- Log every experiment into that model's summary CSV.
- Compare using F1 first, then ROC AUC, accuracy, precision, recall, AP, and EER.

Label convention:
- `0 = real`
- `1 = synthetic`

Important team rule:
- Do not change the evaluation dataset unless the whole team agrees.
- Change training datasets, learning rate, optimizer, batch size, augmentation, epochs, and freezing settings from the parameter block.
- After every experiment, commit the updated summary, predictions, history, config, and model file together so Git history can recover the exact best run later.

---

## 01. EfficientNet-B0

### What it is
EfficientNet-B0 is a lightweight CNN model. It is designed to balance accuracy and speed. It learns visual patterns through convolution layers, but it is more efficient than older CNNs like VGG or basic ResNet.

### Why it is useful
- Fast to train.
- Good first baseline.
- Lower GPU memory usage.
- Easy to test many hyperparameter combinations.
- Good for checking whether augmentations or dataset choices are helping.

### Benefits
- Stable and simple.
- Works well with small and medium datasets.
- Good starting point before trying heavier models.
- Useful for quick experiments with epochs, learning rate, optimizer, and augmentation.

### Limitations
- May not capture very complex global patterns.
- May perform worse than stronger models like ConvNeXt or ViT on harder in-the-wild images.
- Can overfit if training data is too generator-specific.

### Experiments to try
| Experiment | Values to test | Why |
|---|---|---|
| Learning rate | `1e-3`, `3e-4`, `1e-4`, `5e-5`, `1e-5` | Find the best fine-tuning speed |
| Batch size | `16`, `32`, `64` | Check stability and GPU memory |
| Epochs | `3`, `5`, `8`, `10` | See if model improves or overfits |
| Optimizer | `adam`, `adamw`, `sgd` | AdamW usually works well, but compare |
| Weight decay | `0`, `1e-5`, `1e-4`, `1e-3` | Control overfitting |
| Augmentation | `basic`, `light_aug`, `jpeg_like` | Check robustness |
| Freeze backbone | `True`, `False` | See if classifier-only training is enough |
| Training data | Wang only, Corvi only, Wang + Corvi | Check which data helps most |

### Suggested run order
1. `basic`, lr `1e-4`, batch `32`, epochs `5`
2. `light_aug`, lr `1e-4`, batch `32`, epochs `5`
3. `jpeg_like`, lr `1e-4`, batch `32`, epochs `5`
4. `jpeg_like`, lr `5e-5`, batch `32`, epochs `8`
5. `FREEZE_BACKBONE=True`, lr `1e-4`, batch `32`, epochs `5`
6. Wang only vs Corvi only vs Wang + Corvi

### What to analyze
- Does augmentation improve F1 or only accuracy?
- Does lower learning rate improve ROC AUC?
- Does the model produce more false positives or false negatives?
- Does training longer improve validation F1 or cause overfitting?
- Is the model better on GAN-generated images or diffusion-generated images?

---

## 02. ConvNeXt-Tiny

### What it is
ConvNeXt-Tiny is a modern CNN-style model inspired by transformer-era design choices. It still uses convolution, but it has a stronger architecture than older CNNs.

### Why it is useful
This is a strong second model after EfficientNet. It is still CNN-based but usually has stronger feature learning.

### Benefits
- Stronger visual representation than EfficientNet-B0.
- Good balance between modern architecture and practical training.
- Often performs well on image classification tasks.
- Good candidate for the final model or ensemble.

### Limitations
- Heavier than EfficientNet.
- Needs more GPU memory.
- Slower experiments.
- Can overfit if trained too long on limited generator types.

### Experiments to try
| Experiment | Values to test | Why |
|---|---|---|
| Learning rate | `3e-4`, `1e-4`, `5e-5`, `2e-5` | ConvNeXt can be sensitive during fine-tuning |
| Batch size | `8`, `16`, `24`, `32` | Avoid CUDA OOM |
| Epochs | `3`, `5`, `8`, `10` | Check learning curve |
| Optimizer | `adamw`, `adam` | AdamW should usually be first choice |
| Scheduler | `none`, `cosine`, `step` | See if learning-rate decay helps |
| Augmentation | `basic`, `light_aug`, `jpeg_like` | Test real-world robustness |
| Freeze backbone | `True`, `False` | Check if full fine-tuning is necessary |
| Weight decay | `1e-5`, `1e-4`, `5e-4` | Regularization check |

### Suggested run order
1. `basic`, lr `1e-4`, batch `24`, epochs `5`
2. `light_aug`, lr `1e-4`, batch `24`, epochs `5`
3. `jpeg_like`, lr `1e-4`, batch `24`, epochs `5`
4. `jpeg_like`, lr `5e-5`, batch `16`, epochs `8`
5. `SCHEDULER_NAME="cosine"`, lr `1e-4`, epochs `8`
6. `FREEZE_BACKBONE=True`, lr `1e-4`, epochs `5`

### What to analyze
- Does ConvNeXt beat EfficientNet on F1?
- Is the performance gain worth the extra training time?
- Does it reduce false negatives compared to EfficientNet?
- Is ConvNeXt more robust with `jpeg_like` augmentation?
- Does cosine scheduler help or not?

---

## 03. CLIP / ViT-B/16

### What it is
This notebook currently uses torchvision ViT-B/16. ViT means Vision Transformer. It splits images into patches and learns relationships between patches using transformer layers.

The notebook is named `clip_vit` because this model family is for transformer-based and CLIP-style experiments.

### Why it is useful
ViT can capture global image patterns differently from CNNs. It may detect synthetic images using different cues than EfficientNet or ConvNeXt.

### Benefits
- Learns global relationships between image patches.
- Useful comparison against CNN-based models.
- May generalize better in some real-world cases.
- Good candidate for ensemble diversity.

### Limitations
- Heavier and slower than EfficientNet.
- More sensitive to hyperparameters.
- May need more data to perform well.
- Can overfit if training data is not diverse.
- Batch size may need to be small on Kaggle.

### Experiments to try
| Experiment | Values to test | Why |
|---|---|---|
| Learning rate | `1e-4`, `5e-5`, `2e-5`, `1e-5` | ViT usually needs lower LR |
| Batch size | `4`, `8`, `16` | Manage GPU memory |
| Epochs | `3`, `5`, `8`, `10` | Check if it needs more training |
| Freeze backbone | `True`, `False` | Compare linear-head vs full fine-tuning |
| Augmentation | `basic`, `light_aug`, `jpeg_like` | Test robustness |
| Scheduler | `none`, `cosine` | ViT often benefits from smoother LR decay |
| Weight decay | `1e-5`, `1e-4`, `5e-4` | Regularization check |

### Suggested run order
1. `basic`, lr `5e-5`, batch `16`, epochs `5`
2. `light_aug`, lr `5e-5`, batch `16`, epochs `5`
3. `jpeg_like`, lr `5e-5`, batch `16`, epochs `5`
4. `FREEZE_BACKBONE=True`, lr `1e-4`, batch `16`, epochs `5`
5. `jpeg_like`, lr `2e-5`, batch `8`, epochs `8`
6. `SCHEDULER_NAME="cosine"`, lr `5e-5`, epochs `8`

### What to analyze
- Does ViT beat CNNs or just behave differently?
- Does it have higher ROC AUC even if F1 is lower?
- Does it need a different threshold than CNNs?
- Is it more sensitive to learning rate?
- Does it produce different false positive/false negative patterns?

---

# Additional Model Ideas

## 04. ResNet50

### What it is
ResNet50 is a classic CNN model that uses residual connections. These skip connections help train deeper networks more easily.

### Why it is useful
It is a standard baseline. Many papers compare against ResNet-style models, so it gives a familiar reference point.

### Benefits
- Easy to implement with torchvision.
- Stable and well-tested.
- Good baseline for comparison.
- Faster than many transformer models.

### Limitations
- Older architecture.
- May be weaker than ConvNeXt.
- Can miss subtle synthetic traces compared to newer architectures.

### Experiments to try
| Parameter | Values |
|---|---|
| Learning rate | `1e-4`, `5e-5`, `3e-4` |
| Batch size | `32`, `64` |
| Epochs | `5`, `8`, `10` |
| Augmentation | `basic`, `light_aug`, `jpeg_like` |
| Freeze backbone | `True`, `False` |
| Optimizer | `adamw`, `sgd` |

### What to analyze
- Is ResNet50 still competitive?
- Does it perform better than EfficientNet on certain datasets?
- Is it more stable across parameter changes?

---

## 05. ResNet101

### What it is
ResNet101 is a deeper version of ResNet50.

### Why it is useful
It checks whether a deeper classic CNN improves synthetic detection.

### Benefits
- Stronger than ResNet50 in many visual tasks.
- Still easy to implement.
- Good comparison for depth effect.

### Limitations
- Slower and heavier than ResNet50.
- More likely to overfit.
- May not improve enough to justify training cost.

### Experiments to try
| Parameter | Values |
|---|---|
| Learning rate | `1e-4`, `5e-5`, `2e-5` |
| Batch size | `8`, `16`, `24` |
| Epochs | `5`, `8` |
| Augmentation | `basic`, `jpeg_like` |
| Freeze backbone | `True`, `False` |

### What to analyze
- Does deeper ResNet improve F1?
- Does it overfit faster?
- Does it reduce false negatives?

---

## 06. DenseNet121

### What it is
DenseNet121 is a CNN where each layer connects to many later layers. It reuses features heavily.

### Why it is useful
DenseNet can be good at picking up fine-grained visual details, which may help with synthetic image traces.

### Benefits
- Good feature reuse.
- Often strong on limited data.
- May detect texture-level artifacts well.

### Limitations
- Can be memory-heavy.
- Slower than EfficientNet-B0.
- May not generalize if artifacts are dataset-specific.

### Experiments to try
| Parameter | Values |
|---|---|
| Learning rate | `1e-4`, `5e-5`, `3e-4` |
| Batch size | `16`, `32` |
| Epochs | `5`, `8`, `10` |
| Augmentation | `basic`, `light_aug`, `jpeg_like` |
| Freeze backbone | `True`, `False` |

### What to analyze
- Does DenseNet capture synthetic texture better?
- Does it improve precision or recall?
- Is it more stable than ResNet?

---

## 07. MobileNetV3-Large

### What it is
MobileNetV3-Large is a lightweight CNN designed for mobile and efficient inference.

### Why it is useful
It gives a speed-focused baseline. If it performs decently, it may be useful for practical deployment.

### Benefits
- Very fast.
- Low memory usage.
- Good for quick tests.
- Useful for lightweight final systems.

### Limitations
- Lower capacity than ConvNeXt or ViT.
- May miss subtle synthetic traces.
- Might underperform on difficult images.

### Experiments to try
| Parameter | Values |
|---|---|
| Learning rate | `1e-4`, `3e-4`, `5e-5` |
| Batch size | `32`, `64`, `96` |
| Epochs | `5`, `8`, `10` |
| Augmentation | `basic`, `jpeg_like` |
| Freeze backbone | `True`, `False` |

### What to analyze
- How close is it to EfficientNet?
- Is it much faster?
- Does it have poor recall on synthetic images?

---

## 08. EfficientNet-B3

### What it is
EfficientNet-B3 is a larger version of EfficientNet-B0.

### Why it is useful
It checks whether increasing EfficientNet capacity improves detection.

### Benefits
- Stronger than B0.
- Still efficient compared to many large models.
- Good upgrade if B0 performs well.

### Limitations
- More GPU memory needed.
- Slower experiments.
- Batch size may need to drop.

### Experiments to try
| Parameter | Values |
|---|---|
| Image size | `224`, `300` |
| Learning rate | `1e-4`, `5e-5`, `2e-5` |
| Batch size | `8`, `16`, `24` |
| Epochs | `5`, `8` |
| Augmentation | `basic`, `light_aug`, `jpeg_like` |
| Freeze backbone | `True`, `False` |

### What to analyze
- Does B3 improve over B0?
- Is bigger image size helpful?
- Is the training cost worth it?

---

## 09. ConvNeXt-Small

### What it is
ConvNeXt-Small is a larger version of ConvNeXt-Tiny.

### Why it is useful
If ConvNeXt-Tiny performs best, trying ConvNeXt-Small is a natural next step.

### Benefits
- Stronger model capacity.
- May improve F1 and ROC AUC.
- Good final candidate if GPU allows.

### Limitations
- Higher memory usage.
- Slower training.
- More risk of overfitting.

### Experiments to try
| Parameter | Values |
|---|---|
| Learning rate | `5e-5`, `2e-5`, `1e-4` |
| Batch size | `4`, `8`, `16` |
| Epochs | `5`, `8` |
| Augmentation | `light_aug`, `jpeg_like` |
| Scheduler | `none`, `cosine` |
| Freeze backbone | `True`, `False` |

### What to analyze
- Does it improve over ConvNeXt-Tiny?
- Does it overfit more?
- Can Kaggle GPU handle it comfortably?

---

## 10. Swin Transformer Tiny

### What it is
Swin Transformer is a vision transformer that works with local windows instead of full-image attention everywhere.

### Why it is useful
It combines some benefits of CNNs and transformers. It can capture local and global visual patterns efficiently.

### Benefits
- Strong image classification model.
- More image-friendly than plain ViT in many cases.
- May work well for subtle artifacts.

### Limitations
- Requires `timm` or compatible torchvision version.
- Heavier than EfficientNet.
- More implementation details to check.

### Experiments to try
| Parameter | Values |
|---|---|
| Learning rate | `5e-5`, `2e-5`, `1e-4` |
| Batch size | `8`, `16` |
| Epochs | `5`, `8` |
| Augmentation | `basic`, `light_aug`, `jpeg_like` |
| Scheduler | `cosine`, `none` |
| Freeze backbone | `True`, `False` |

### What to analyze
- Does Swin beat ViT-B/16?
- Does it handle augmentation better?
- Is it more stable than plain ViT?

---

## 11. DeiT-Tiny / DeiT-Small

### What it is
DeiT is a data-efficient vision transformer. It was designed to train transformers better on image classification tasks.

### Why it is useful
It may perform better than standard ViT when data is not huge.

### Benefits
- More data-efficient than plain ViT.
- Good transformer baseline.
- Tiny/Small variants are manageable on Kaggle.

### Limitations
- Usually needs `timm`.
- Can still be sensitive to learning rate.
- May not beat ConvNeXt.

### Experiments to try
| Parameter | Values |
|---|---|
| Model size | `deit_tiny`, `deit_small` |
| Learning rate | `5e-5`, `2e-5`, `1e-5` |
| Batch size | `8`, `16`, `24` |
| Epochs | `5`, `8` |
| Augmentation | `basic`, `jpeg_like` |
| Scheduler | `cosine` |

### What to analyze
- Does DeiT train more smoothly than ViT?
- Is Tiny enough or is Small needed?
- Does it improve ROC AUC?

---

## 12. CLIP ViT-B/32 Linear Probe

### What it is
CLIP is trained on image-text pairs. For this task, we can freeze CLIP and train only a small classifier on top of its image features.

### Why it is useful
CLIP features often generalize well to real-world images. A frozen CLIP linear probe can be fast and useful.

### Benefits
- Strong general-purpose features.
- Fast if backbone is frozen.
- Less overfitting than full fine-tuning.
- Good for open-run style experiments.

### Limitations
- Needs `open_clip_torch` or CLIP package.
- May not capture low-level forensic traces.
- Linear classifier may be too simple.

### Experiments to try
| Parameter | Values |
|---|---|
| Backbone | `ViT-B-32`, `ViT-B-16`, `RN50` |
| Freeze backbone | `True` first |
| Learning rate | `1e-3`, `3e-4`, `1e-4` for head only |
| Batch size | `32`, `64` |
| Epochs | `5`, `10` |
| Augmentation | `basic`, `jpeg_like` |

### What to analyze
- Does CLIP generalize better than CNNs?
- Does it have high ROC AUC but lower F1?
- Does threshold tuning help a lot?
- Does it reduce false positives on real in-the-wild images?

---

## 13. CLIP Partial Fine-Tuning

### What it is
Instead of freezing all CLIP layers, unfreeze the last few image encoder blocks and train them with a small learning rate.

### Why it is useful
This allows CLIP to adapt to synthetic image detection while still keeping general pretrained knowledge.

### Benefits
- More flexible than linear probe.
- May improve F1.
- Can still generalize better than training from scratch.

### Limitations
- More GPU memory.
- Easy to overfit.
- Learning rate must be low.
- More complex code.

### Experiments to try
| Parameter | Values |
|---|---|
| Unfrozen layers | last `1`, `2`, `4` blocks |
| Learning rate backbone | `1e-6`, `5e-6`, `1e-5` |
| Learning rate head | `1e-4`, `3e-4` |
| Batch size | `8`, `16` |
| Epochs | `5`, `8` |
| Augmentation | `jpeg_like` |

### What to analyze
- Does partial fine-tuning beat frozen CLIP?
- Does it overfit?
- Does it improve recall for synthetic images?

---

## 14. Xception

### What it is
Xception is a CNN architecture using depthwise separable convolutions. It has been commonly used in image forgery/deepfake-related tasks.

### Why it is useful
It is a strong forensic-style baseline because separable convolutions may capture texture and artifact patterns.

### Benefits
- Known in media forensics/deepfake detection contexts.
- Good for artifact-based learning.
- Can be implemented through `timm`.

### Limitations
- Not always available in torchvision.
- Needs `timm`.
- May be slower than EfficientNet.

### Experiments to try
| Parameter | Values |
|---|---|
| Learning rate | `1e-4`, `5e-5`, `2e-5` |
| Batch size | `16`, `32` |
| Epochs | `5`, `8` |
| Augmentation | `basic`, `jpeg_like` |
| Freeze backbone | `True`, `False` |

### What to analyze
- Does it detect synthetic artifacts better than general CNNs?
- Does it improve precision?
- Is it useful in an ensemble?

---

## 15. InceptionV3

### What it is
InceptionV3 is a CNN model that uses multiple filter sizes inside the network to capture patterns at different scales.

### Why it is useful
Synthetic artifacts can appear at different scales, so multi-scale features may help.

### Benefits
- Multi-scale feature extraction.
- Classic strong image classifier.
- Good baseline for comparison.

### Limitations
- Input size is usually `299`.
- Slightly more annoying to integrate than 224-size models.
- Older architecture.

### Experiments to try
| Parameter | Values |
|---|---|
| Image size | `299` |
| Learning rate | `1e-4`, `5e-5` |
| Batch size | `16`, `32` |
| Epochs | `5`, `8` |
| Augmentation | `basic`, `jpeg_like` |

### What to analyze
- Does image size 299 improve performance?
- Does multi-scale learning help?
- Does it beat ResNet/EfficientNet?

---

## 16. RegNet

### What it is
RegNet is a family of CNN models designed through a systematic architecture search approach.

### Why it is useful
It provides another modern CNN family that can be compared against EfficientNet and ConvNeXt.

### Benefits
- Strong CNN baseline.
- Available in torchvision.
- Several sizes to test.

### Limitations
- Less popular than ConvNeXt in recent experiments.
- Larger variants can be heavy.
- May not add much if ConvNeXt already performs well.

### Experiments to try
| Parameter | Values |
|---|---|
| Variant | `regnet_y_400mf`, `regnet_y_800mf`, `regnet_y_1_6gf` |
| Learning rate | `1e-4`, `5e-5` |
| Batch size | `16`, `32` |
| Epochs | `5`, `8` |
| Augmentation | `basic`, `jpeg_like` |

### What to analyze
- Which RegNet size is best?
- Does it beat EfficientNet at similar training cost?
- Is it stable across runs?

---

## 17. MaxViT-Tiny

### What it is
MaxViT combines convolution, local attention, and global attention. It tries to capture both local details and broader image structure.

### Why it is useful
Synthetic detection may need both local artifact detection and global consistency checks.

### Benefits
- Strong hybrid model.
- Captures multiple types of patterns.
- Good candidate if available in torchvision/timm.

### Limitations
- Heavy.
- Can be slow.
- May cause CUDA OOM.
- More difficult to tune.

### Experiments to try
| Parameter | Values |
|---|---|
| Learning rate | `2e-5`, `5e-5` |
| Batch size | `4`, `8` |
| Epochs | `5`, `8` |
| Augmentation | `light_aug`, `jpeg_like` |
| Scheduler | `cosine` |

### What to analyze
- Does hybrid attention improve F1?
- Is it worth the GPU cost?
- Does it overfit or generalize?

---

## 18. Frequency-Based CNN

### What it is
This is not one fixed pretrained model. It means training a model on frequency-domain inputs, such as FFT magnitude images, high-pass filtered images, or residual/noise maps.

### Why it is useful
Synthetic images can leave frequency artifacts that are not obvious in RGB space. This approach directly focuses on those signals.

### Benefits
- Very relevant to synthetic image detection.
- Can reveal artifacts missed by normal RGB models.
- Good for paper/insight analysis.
- Useful for ensemble with RGB models.

### Limitations
- More custom code.
- May be sensitive to compression.
- Needs careful preprocessing.
- Harder to compare directly with normal models.

### Experiments to try
| Experiment | Values |
|---|---|
| Input type | RGB, FFT magnitude, high-pass residual, noise residual |
| Base model | EfficientNet-B0, ResNet50, small CNN |
| Learning rate | `1e-4`, `5e-5` |
| Batch size | `16`, `32` |
| Augmentation | `basic`, `jpeg_like` |

### What to analyze
- Does frequency input improve detection?
- Does it help on compressed/social-media-like images?
- Does it reduce false negatives?
- Does RGB + frequency ensemble improve F1?

---

## 19. Texture Crop Model

### What it is
Instead of training on the full image only, this approach trains on random texture-heavy crops from the image. The idea is to force the model to look at local texture/forensic traces.

### Why it is useful
Synthetic traces may be easier to detect in local texture patches rather than full semantic image content.

### Benefits
- Very relevant to synthetic image detection.
- Can improve robustness.
- Good for insight experiments.
- Can be added to existing EfficientNet/ConvNeXt notebooks.

### Limitations
- Needs custom crop strategy.
- May lose global context.
- Too aggressive cropping can hurt performance.

### Experiments to try
| Parameter | Values |
|---|---|
| Crop size | `128`, `160`, `192`, `224` |
| Number of crops per image | `1`, `2`, `4` |
| Base model | EfficientNet-B0, ConvNeXt-Tiny |
| Learning rate | `1e-4`, `5e-5` |
| Aggregation | average probabilities over crops |

### What to analyze
- Does texture cropping improve F1?
- Does it improve robustness to resized/cropped images?
- Are false positives reduced or increased?
- Which crop size works best?

---

## 20. Ensemble Model

### What it is
An ensemble combines predictions from multiple models instead of relying on one model.

Example:
- EfficientNet probability
- ConvNeXt probability
- ViT probability

Final prediction:
- average probability
- weighted average probability
- majority vote

### Why it is useful
Different models may make different mistakes. Combining them can improve F1 and robustness.

### Benefits
- Often improves final performance.
- Reduces weakness of single model.
- Useful for final competition submission.
- No retraining needed if predictions already exist.

### Limitations
- Requires multiple trained models.
- Slower inference.
- Harder to manage final submission.
- Bad models can hurt ensemble if weighted incorrectly.

### Experiments to try
| Ensemble type | Values |
|---|---|
| Simple average | EfficientNet + ConvNeXt, EfficientNet + ViT, all three |
| Weighted average | `0.5 ConvNeXt + 0.3 EfficientNet + 0.2 ViT` |
| Majority vote | label vote across models |
| Best-two ensemble | top two by validation F1 |
| Threshold tuning | optimize final threshold on eval set |

### What to analyze
- Does ensemble beat the best single model?
- Which combination gives the best F1?
- Does ensemble improve recall or precision?
- Does weighted average beat simple average?
- Does threshold tuning matter?

---

# Suggested Priority Order

If the team has limited time, try models in this order:

1. EfficientNet-B0
2. ConvNeXt-Tiny
3. ViT-B/16
4. ResNet50
5. DenseNet121
6. CLIP ViT-B/32 linear probe
7. EfficientNet-B3
8. Swin Transformer Tiny
9. ConvNeXt-Small
10. Ensemble of best models

---

# Best Models to Try First

## Fast and practical
- EfficientNet-B0
- ResNet50
- MobileNetV3-Large
- DenseNet121

## Stronger but heavier
- ConvNeXt-Tiny
- EfficientNet-B3
- ResNet101
- Swin Tiny

## Transformer / generalization focused
- ViT-B/16
- DeiT-Small
- CLIP ViT-B/32
- CLIP partial fine-tuning

## Insight / paper-focused
- Frequency-Based CNN
- Texture Crop Model
- Ensemble Model

---

# General Hyperparameter Ranges

Use these when creating any new model notebook.

## Learning rate
Start with:
- `1e-4` for CNN full fine-tuning
- `5e-5` for ViT/transformers
- `1e-3` or `3e-4` for frozen-backbone classifier head only

Try:
- `3e-4`
- `1e-4`
- `5e-5`
- `2e-5`
- `1e-5`

## Batch size
Use based on GPU memory:
- Light CNNs: `32`, `64`
- Medium CNNs: `16`, `24`, `32`
- Transformers: `4`, `8`, `16`

## Epochs
Try:
- quick test: `2`
- baseline: `5`
- longer run: `8`
- possible overfit check: `10`

## Optimizer
Try:
- `adamw` first
- `adam` second
- `sgd` only for classic CNN comparison

## Scheduler
Try:
- `none` first
- `cosine` for longer runs
- `step` as a simple baseline

## Weight decay
Try:
- `0`
- `1e-5`
- `1e-4`
- `5e-4`
- `1e-3`

## Augmentation
Try:
- `basic`
- `light_aug`
- `jpeg_like`

For this competition, `jpeg_like` is especially important because the test data may include real-world resizing, cropping, and compression-like changes.

---

# What to Compare Across All Models

When using `analyze_experiments.ipynb`, compare:

## Main performance
- F1
- ROC AUC
- Accuracy
- Average Precision
- EER

## Error behavior
- False positives
- False negatives
- Precision vs recall
- Best threshold

## Training behavior
- Training loss curve
- Validation loss curve
- Best epoch
- Overfitting signs

## Dataset behavior
- Wang only vs Corvi only vs Wang + Corvi
- Does adding extra data improve or hurt?
- Does any model fail on one generator type?

## Robustness behavior
- Basic vs light augmentation vs jpeg-like augmentation
- Does augmentation help F1?
- Does it reduce false positives?
- Does it reduce false negatives?

---

# Notes for Making a New Model Notebook

To create a new model notebook:

1. Copy an existing notebook closest to your model type.
2. Rename it, for example:
   - `04_resnet50_experiments.ipynb`
   - `05_densenet121_experiments.ipynb`
   - `06_swin_tiny_experiments.ipynb`
3. Change `MODEL_NAME`.
4. Keep the same dataset loading and evaluation dataset.
5. Replace only:
   - imports
   - model creation function
   - classifier head replacement
   - recommended batch size / LR
6. Run a quick test first:
   - `MAX_TRAIN_IMAGES = 3000`
   - `MAX_EVAL_IMAGES = 1000`
   - `EPOCHS = 2`
7. If it works, run full:
   - `MAX_TRAIN_IMAGES = None`
   - `MAX_EVAL_IMAGES = None`
   - `EPOCHS = 5` or `8`
8. Download Kaggle outputs and commit the full model folder.
