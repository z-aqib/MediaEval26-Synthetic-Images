03 first test settings

Use this first:

EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
OPTIMIZER_NAME = "adamw"
AUGMENTATION_TYPE = "basic"
MAX_TRAIN_IMAGES = 3000
MAX_EVAL_IMAGES = 1000
EXPERIMENT_NOTES = "ViT-B/16 quick test run with basic augmentation"

If it gives OOM:

BATCH_SIZE = 8

For the full first baseline:

EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
OPTIMIZER_NAME = "adamw"
AUGMENTATION_TYPE = "basic"
MAX_TRAIN_IMAGES = None
MAX_EVAL_IMAGES = None
EXPERIMENT_NOTES = "ViT-B/16 full baseline with Wang + Corvi training and fixed DMImageDetect evaluation"
Good ViT experiment sequence
Run	Change	Why
vit_001	basic, lr 5e-5, epochs 5	Baseline
vit_002	light_aug, lr 5e-5, epochs 5	Augmentation comparison
vit_003	jpeg_like, lr 5e-5, epochs 5	Robustness experiment
vit_004	FREEZE_BACKBONE=True, lr 1e-4, epochs 5	Linear-head style test
vit_005	jpeg_like, lr 2e-5, epochs 8	Slower fine-tuning
vit_006	SCHEDULER_NAME="cosine"`	Scheduler test