# Model Experiment Plan for Task A

This file is for Zuha, Izma, and Fatima to use before running experiments. The goal is to avoid random parameter guessing. For every experiment, change only the parameter block at the top of the notebook, run the notebook on Kaggle GPU, download outputs, and commit the updated model output folder to GitHub.

## Common Rules for All Models

### Main goal
We are solving Task A: binary classification of images as real or synthetic.

Label convention:
- `0 = real`
- `1 = synthetic`

### Evaluation rule
Keep the evaluation dataset the same across all models.

Current setting:
```python
EVALUATION_DATASET_KEY = "dmimagedetect_train"
```

Do not change this unless the whole team agrees, because model comparison only makes sense when all models are tested on the same dataset.

### Training dataset rule
Training datasets can change depending on the experiment.

Recommended default:
```python
USE_WANG_TRAIN = True
USE_CORVI_TRAIN = True
USE_DMIMAGEDETECT_TRAIN = False
USE_REALRAISE_TRAIN = False
```

Keep `USE_DMIMAGEDETECT_TRAIN = False` while DMImageDetect is also being used as evaluation, otherwise the model may see the same images during training and evaluation.

### First check run for every model
Before any full run, do a small test run:
```python
EPOCHS = 2
MAX_TRAIN_IMAGES = 3000
MAX_EVAL_IMAGES = 1000
```

This checks whether the notebook runs correctly before spending GPU time.

### Full baseline run for every model
After the small test works:
```python
EPOCHS = 5
MAX_TRAIN_IMAGES = None
MAX_EVAL_IMAGES = None
AUGMENTATION_TYPE = "basic"
OPTIMIZER_NAME = "adamw"
SCHEDULER_NAME = "none"
```

### Metrics to watch
Primary metric:
- `f1`

Also check:
- `accuracy`
- `precision`
- `recall`
- `roc_auc`
- `average_precision`
- `eer`
- `fp` and `fn`

A model with slightly lower accuracy but higher F1 may still be better for this competition.

### What to write in EXPERIMENT_NOTES
Always write a useful note. Example:
```python
EXPERIMENT_NOTES = "EfficientNet-B0 full baseline, Wang + Corvi, basic augmentation, lr 1e-4"
```

Do not leave generic notes like `test` or `run 1` because later analysis will become confusing.

---

# EfficientNet-B0 Experiments

## What this model is
EfficientNet-B0 is a lightweight CNN model. It is fast, stable, and good for a first baseline. It is useful because it trains quicker than ConvNeXt and ViT, so we can use it to test ideas before spending more GPU time on heavier models.

## Best use case
Use EfficientNet-B0 for:
- quick baseline experiments
- learning rate testing
- augmentation testing
- checking whether dataset choices improve performance
- fast debugging

## Starting baseline
```python
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
OPTIMIZER_NAME = "adamw"
WEIGHT_DECAY = 1e-4
SCHEDULER_NAME = "none"
AUGMENTATION_TYPE = "basic"
FREEZE_BACKBONE = False
MAX_TRAIN_IMAGES = None
MAX_EVAL_IMAGES = None
```

## Experiment sequence

### EFF-01: Quick test
Purpose: check that notebook runs.
```python
EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
AUGMENTATION_TYPE = "basic"
MAX_TRAIN_IMAGES = 3000
MAX_EVAL_IMAGES = 1000
EXPERIMENT_NOTES = "EfficientNet-B0 quick test, basic augmentation, limited data"
```

### EFF-02: Full baseline
Purpose: first real score.
```python
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
AUGMENTATION_TYPE = "basic"
MAX_TRAIN_IMAGES = None
MAX_EVAL_IMAGES = None
EXPERIMENT_NOTES = "EfficientNet-B0 full baseline, Wang + Corvi, basic augmentation"
```

### EFF-03: Light augmentation
Purpose: check if mild augmentation improves generalization.
```python
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
AUGMENTATION_TYPE = "light_aug"
EXPERIMENT_NOTES = "EfficientNet-B0 with light augmentation"
```

### EFF-04: Real-world style augmentation
Purpose: test robustness against resizing, blur, and social-media-like changes.
```python
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
AUGMENTATION_TYPE = "jpeg_like"
EXPERIMENT_NOTES = "EfficientNet-B0 with jpeg_like augmentation for robustness"
```

### EFF-05: Lower learning rate
Purpose: see if training becomes more stable.
```python
EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
AUGMENTATION_TYPE = "jpeg_like"
EXPERIMENT_NOTES = "EfficientNet-B0 jpeg_like augmentation with lower lr 5e-5"
```

### EFF-06: Frozen backbone
Purpose: see if ImageNet features alone are enough.
```python
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
FREEZE_BACKBONE = True
AUGMENTATION_TYPE = "basic"
EXPERIMENT_NOTES = "EfficientNet-B0 frozen backbone, only classifier trained"
```

### EFF-07: Scheduler test
Purpose: test if cosine LR schedule improves F1.
```python
EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SCHEDULER_NAME = "cosine"
AUGMENTATION_TYPE = "jpeg_like"
EXPERIMENT_NOTES = "EfficientNet-B0 cosine scheduler with jpeg_like augmentation"
```

## EfficientNet things to analyze
Check:
- Does `light_aug` or `jpeg_like` improve F1?
- Does lower LR reduce false positives or false negatives?
- Is the frozen backbone much worse than full fine-tuning?
- Is EfficientNet fast enough to use for more dataset experiments?
- Which generator/source has most errors in `predictions.csv`?

---

# ConvNeXt-Tiny Experiments

## What this model is
ConvNeXt-Tiny is a modern CNN-style model. It is heavier than EfficientNet but often stronger. It keeps the useful CNN idea of local image features while using more modern architecture choices.

## Best use case
Use ConvNeXt-Tiny for:
- stronger CNN baseline
- robust visual feature learning
- comparing old-style CNN baseline vs modern CNN baseline
- testing whether extra augmentation helps heavier models

## Starting baseline
```python
EPOCHS = 5
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
OPTIMIZER_NAME = "adamw"
WEIGHT_DECAY = 1e-4
SCHEDULER_NAME = "none"
AUGMENTATION_TYPE = "basic"
FREEZE_BACKBONE = False
MAX_TRAIN_IMAGES = None
MAX_EVAL_IMAGES = None
```

If Kaggle gives CUDA out-of-memory:
```python
BATCH_SIZE = 16
```

If still out-of-memory:
```python
BATCH_SIZE = 8
```

## Experiment sequence

### CNX-01: Quick test
Purpose: check that notebook runs.
```python
EPOCHS = 2
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
AUGMENTATION_TYPE = "basic"
MAX_TRAIN_IMAGES = 3000
MAX_EVAL_IMAGES = 1000
EXPERIMENT_NOTES = "ConvNeXt-Tiny quick test, basic augmentation, limited data"
```

### CNX-02: Full baseline
Purpose: first full ConvNeXt result.
```python
EPOCHS = 5
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
AUGMENTATION_TYPE = "basic"
MAX_TRAIN_IMAGES = None
MAX_EVAL_IMAGES = None
EXPERIMENT_NOTES = "ConvNeXt-Tiny full baseline, Wang + Corvi, basic augmentation"
```

### CNX-03: Light augmentation
Purpose: compare against basic augmentation.
```python
EPOCHS = 5
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
AUGMENTATION_TYPE = "light_aug"
EXPERIMENT_NOTES = "ConvNeXt-Tiny with light augmentation"
```

### CNX-04: Real-world style augmentation
Purpose: train for social-media-like distortions.
```python
EPOCHS = 5
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
AUGMENTATION_TYPE = "jpeg_like"
EXPERIMENT_NOTES = "ConvNeXt-Tiny with jpeg_like augmentation for robustness"
```

### CNX-05: Lower learning rate
Purpose: safer fine-tuning for a stronger model.
```python
EPOCHS = 8
BATCH_SIZE = 24
LEARNING_RATE = 5e-5
AUGMENTATION_TYPE = "jpeg_like"
EXPERIMENT_NOTES = "ConvNeXt-Tiny jpeg_like augmentation with lower lr 5e-5"
```

### CNX-06: Frozen backbone
Purpose: compare full fine-tuning vs classifier-only training.
```python
EPOCHS = 5
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
FREEZE_BACKBONE = True
AUGMENTATION_TYPE = "basic"
EXPERIMENT_NOTES = "ConvNeXt-Tiny frozen backbone, only classifier trained"
```

### CNX-07: Cosine scheduler
Purpose: see if smoother LR decay improves final F1.
```python
EPOCHS = 8
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
SCHEDULER_NAME = "cosine"
AUGMENTATION_TYPE = "jpeg_like"
EXPERIMENT_NOTES = "ConvNeXt-Tiny cosine scheduler with jpeg_like augmentation"
```

## ConvNeXt things to analyze
Check:
- Does ConvNeXt beat EfficientNet on F1?
- Does it reduce false positives or false negatives compared to EfficientNet?
- Does it benefit more from `jpeg_like` augmentation?
- Is the extra training time worth the score gain?
- Does lower LR help because the model is heavier?

---

# ViT-B/16 / CLIP-ViT Folder Experiments

## What this model is
This notebook currently uses torchvision ViT-B/16 inside the `03_clip_vit` experiment folder. ViT means Vision Transformer. Instead of focusing mostly on local CNN patterns, it splits the image into patches and learns relationships between patches. It can capture global image structure better, but it is heavier and more sensitive to hyperparameters.

## Best use case
Use ViT-B/16 for:
- transformer comparison against CNN models
- testing global image features
- checking whether transformer-based models generalize better
- later expanding into actual CLIP/OpenCLIP experiments if needed

## Starting baseline
```python
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
OPTIMIZER_NAME = "adamw"
WEIGHT_DECAY = 1e-4
SCHEDULER_NAME = "none"
AUGMENTATION_TYPE = "basic"
FREEZE_BACKBONE = False
GRAD_CLIP_NORM = 1.0
MAX_TRAIN_IMAGES = None
MAX_EVAL_IMAGES = None
```

If Kaggle gives CUDA out-of-memory:
```python
BATCH_SIZE = 8
```

If still out-of-memory:
```python
BATCH_SIZE = 4
```

## Experiment sequence

### VIT-01: Quick test
Purpose: check that notebook runs.
```python
EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
AUGMENTATION_TYPE = "basic"
MAX_TRAIN_IMAGES = 3000
MAX_EVAL_IMAGES = 1000
EXPERIMENT_NOTES = "ViT-B/16 quick test, basic augmentation, limited data"
```

### VIT-02: Full baseline
Purpose: first full ViT result.
```python
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
AUGMENTATION_TYPE = "basic"
MAX_TRAIN_IMAGES = None
MAX_EVAL_IMAGES = None
EXPERIMENT_NOTES = "ViT-B/16 full baseline, Wang + Corvi, basic augmentation"
```

### VIT-03: Light augmentation
Purpose: see if ViT handles mild augmentation well.
```python
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
AUGMENTATION_TYPE = "light_aug"
EXPERIMENT_NOTES = "ViT-B/16 with light augmentation"
```

### VIT-04: Real-world style augmentation
Purpose: check robustness to social-media-like image changes.
```python
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
AUGMENTATION_TYPE = "jpeg_like"
EXPERIMENT_NOTES = "ViT-B/16 with jpeg_like augmentation for robustness"
```

### VIT-05: Lower learning rate
Purpose: safer transformer fine-tuning.
```python
EPOCHS = 8
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
AUGMENTATION_TYPE = "jpeg_like"
EXPERIMENT_NOTES = "ViT-B/16 jpeg_like augmentation with lower lr 2e-5"
```

### VIT-06: Frozen backbone
Purpose: test whether pretrained ViT features are enough.
```python
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
FREEZE_BACKBONE = True
AUGMENTATION_TYPE = "basic"
EXPERIMENT_NOTES = "ViT-B/16 frozen backbone, only classification head trained"
```

### VIT-07: Cosine scheduler
Purpose: smoother fine-tuning.
```python
EPOCHS = 8
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
SCHEDULER_NAME = "cosine"
AUGMENTATION_TYPE = "jpeg_like"
EXPERIMENT_NOTES = "ViT-B/16 cosine scheduler with jpeg_like augmentation"
```

## ViT things to analyze
Check:
- Does ViT beat CNN models on F1 or AUC?
- Does it overfit faster than EfficientNet/ConvNeXt?
- Does lower LR help more than 5e-5?
- Does frozen backbone perform reasonably or badly?
- Does ViT produce more false positives on real images?
- Does it need stronger augmentation or does augmentation hurt?

---

# Cross-Model Analysis Checklist

After several runs, use `analyze_experiments.ipynb` to check:

## Best model overall
Sort by:
1. F1
2. ROC AUC
3. Accuracy

## Best model per family
Compare:
- Best EfficientNet-B0
- Best ConvNeXt-Tiny
- Best ViT-B/16

## Parameter effects
Plot and discuss:
- learning rate vs F1
- batch size vs F1
- augmentation type vs F1
- optimizer vs F1
- scheduler vs F1
- epochs vs validation loss/F1

## Error patterns
Use `predictions.csv` to check:
- Which model has more false positives?
- Which model has more false negatives?
- Which generator/source is hardest?
- Are real images being wrongly marked synthetic?
- Are diffusion images harder than GAN images?

## Final selection rule
Pick the final model using:
1. Highest F1
2. If tied, higher ROC AUC
3. If tied, lower EER
4. If still tied, simpler/faster model

Do not pick a model only because it has the highest accuracy.
