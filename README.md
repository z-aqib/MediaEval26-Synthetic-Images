# Synthetic Images: Advancing detection and localization of generative AI used in real-world online images
[competition link](https://multimediaeval.github.io/editions/2026/tasks/synthim/)

## competition information
The MediaEval 2026 SynthIM competition focuses on detecting AI-generated images and localizing AI-manipulated regions. For our baseline milestone, we are focusing on Subtask A, which is binary classification of images into real or synthetic. We selected a simple but reliable baseline using EfficientNet-B0 fine-tuned on the official constrained training data because it is computationally efficient, well-established for image classification, and provides a strong reference point for future experiments. We will first evaluate it on the official validation set and then extend our work through augmentation, threshold tuning, model comparisons, and open-run experiments using additional public datasets.

## Baseline Model: EfficientNet-B0
We selected EfficientNet-B0 as our initial baseline for Subtask A because it is a compact and efficient convolutional model with strong performance on image classification tasks. Since this milestone is intended to ensure timely progress rather than final optimization, we prioritized a model that is easy to fine-tune, computationally manageable, and suitable for producing a reliable first benchmark on the official constrained data. This baseline will allow us to establish a reference point before exploring stronger architectures, more advanced augmentations, and open-run data extensions.

## Group Members
- Zuha Aqib, 26106
- Izma Khan, 
- Fatima Naeem, 