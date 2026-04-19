# Synthetic Images: Advancing detection and localization of generative AI used in real-world online images
[competition link](https://multimediaeval.github.io/editions/2026/tasks/synthim/)

## competition information
The MediaEval 2026 SynthIM competition focuses on detecting AI-generated images and localizing AI-manipulated regions. For our baseline milestone, we are focusing on Subtask A, which is binary classification of images into real or synthetic. We selected a simple but reliable baseline using EfficientNet-B0 fine-tuned on the official constrained training data because it is computationally efficient, well-established for image classification, and provides a strong reference point for future experiments. We will first evaluate it on the official validation set and then extend our work through augmentation, threshold tuning, model comparisons, and open-run experiments using additional public datasets.

## Competition Training data
Official task page:
https://multimediaeval.github.io/editions/2026/tasks/synthim/

Wang et al. dataset/code repo (CNNDetection):
https://github.com/peterwang512/CNNDetection

Wang dataset Google Drive folder (alternative link shown in repo):
https://drive.google.com/drive/u/2/folders/14E_R19lqIE9JgotGz09fLPQ4NVqlYbVc

Wang dataset Box mirror (alternative link shown in repo):
https://cmu.app.box.com/s/4syr4womrggfin0tsfhxohaec5dh6n48

Wang temporary/fixed dataset Google Drive folder mentioned in repo updates:
https://drive.google.com/drive/folders/1RwCSaraEUctIwFgoQXWMKFvW07gM80_3?usp=drive_link

Corvi et al. dataset/code repo (DMimageDetection):
https://github.com/grip-unina/DMimageDetection

Corvi latent diffusion training set zip:
https://www.grip.unina.it/download/prog/DMimageDetection/latent_diffusion_trainingset.zip

## ACCESIBLE DATASET FOR TRAINING
https://www.kaggle.com/datasets/zuhaaqib/wang-cnndetection-dataset

## Baseline Model: EfficientNet-B0
We selected EfficientNet-B0 as our initial baseline for Subtask A because it is a compact and efficient convolutional model with strong performance on image classification tasks. Since this milestone is intended to ensure timely progress rather than final optimization, we prioritized a model that is easy to fine-tune, computationally manageable, and suitable for producing a reliable first benchmark on the official constrained data. This baseline will allow us to establish a reference point before exploring stronger architectures, more advanced augmentations, and open-run data extensions.

## Group Members
- Zuha Aqib, 26106
- Izma Khan, 
- Fatima Naeem, 