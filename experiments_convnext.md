02 first experiment settings

For the first quick check:

EPOCHS = 2
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
OPTIMIZER_NAME = "adamw"
AUGMENTATION_TYPE = "basic"
MAX_TRAIN_IMAGES = 3000
MAX_EVAL_IMAGES = 1000
EXPERIMENT_NOTES = "ConvNeXt-Tiny quick test run with basic augmentation"

For the actual baseline:

EPOCHS = 5
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
OPTIMIZER_NAME = "adamw"
AUGMENTATION_TYPE = "basic"
MAX_TRAIN_IMAGES = None
MAX_EVAL_IMAGES = None
EXPERIMENT_NOTES = "ConvNeXt-Tiny full baseline with Wang + Corvi training and fixed DMImageDetect evaluation"
Good experiment sequence for ConvNeXt

Once the quick test works, run these in this order:

Run	Change	Why
convnext_001	basic, lr 1e-4, epochs 5	Baseline
convnext_002	light_aug, lr 1e-4, epochs 5	Check if stronger augmentation helps
convnext_003	jpeg_like, lr 1e-4, epochs 5	Real-world robustness check
convnext_004	jpeg_like, lr 5e-5, epochs 8	Slower fine-tuning
convnext_005	FREEZE_BACKBONE=True, epochs 5	See if feature extractor alone works
convnext_006	SCHEDULER_NAME="cosine"	Check scheduler effect

The competition explicitly mentions robustness under compression, resizing, and cropping, so the jpeg_like augmentation run is especially important for this task.

Small note before running

ConvNeXt may use more GPU memory than EfficientNet. If Kaggle gives OOM:

BATCH_SIZE = 16

If it still OOMs:

BATCH_SIZE = 8

The rest can stay the same.