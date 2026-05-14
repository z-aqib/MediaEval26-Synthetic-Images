In the current recommended setup, **Wang test and Wang cnn_synth_test are NOT being used yet**.

They are only added as optional switches so later you can turn them on without rewriting code.

## Current setup for model 01 run 01

Use this:

```python
USE_WANG_TRAIN = False
USE_WANG_VAL_AS_TRAIN = True
USE_WANG_TEST_AS_TRAIN = False
USE_WANG_CNN_SYNTH_TEST_AS_TRAIN = False

USE_CORVI_TRAIN = True
USE_DMIMAGEDETECT_TRAIN = False
USE_REALRAISE_TRAIN = False
```

So the actual data usage is:

| Dataset                                | Used for what right now?  | Why                                                                            |
| -------------------------------------- | ------------------------- | ------------------------------------------------------------------------------ |
| **Wang val/val**                       | **Training**              | Has both `0_real` and `1_fake`, so it gives real + synthetic training examples |
| **Corvi latent diffusion**             | **Training**              | Synthetic-only diffusion images, useful for learning diffusion artifacts       |
| **DMImageDetect-Test**                 | **Fixed evaluation only** | This is the fixed fair comparison eval set for all models                      |
| **Wang test/test**                     | Not used right now        | Keep aside for optional internal testing or future training if needed          |
| **Wang cnn_synth_test/cnn_synth_test** | Not used right now        | Optional bigger Wang source, but don’t use in first clean baseline             |
| **DMImageDetect-Train**                | Not used right now        | Synthetic-only extra training, useful later for open run                       |
| **RealRAISE**                          | Not used right now        | Real-only extra training, useful later for balancing/open run                  |
| **ClipDetWeights**                     | Not used for EfficientNet | Only useful if using pretrained ClipDet/Corvi-style weights                    |

## Why not use Wang test now?

Because we need some clean separation.

Since your Wang dataset has no train split, we are using:

```text
Wang val -> training
DMImageDetect-Test -> fixed evaluation
```

Then Wang test can stay as a backup internal check. If we use Wang val + Wang test + cnn_synth_test all at once from the start, we lose the ability to compare later whether adding more Wang data helped.

## Why not use Wang cnn_synth_test now?

It is bigger and useful, but for first run it can make the experiment less clean. First run should answer:

> “How does EfficientNet perform using Wang val + Corvi, evaluated on fixed DMImageDetect-Test?”

Then later run 2 can be:

```python
USE_WANG_CNN_SYNTH_TEST_AS_TRAIN = True
```

and you can compare whether adding it improved F1.

## Recommended experiment sequence

### Run 01: clean baseline

```python
USE_WANG_VAL_AS_TRAIN = True
USE_CORVI_TRAIN = True
USE_WANG_TEST_AS_TRAIN = False
USE_WANG_CNN_SYNTH_TEST_AS_TRAIN = False
USE_REALRAISE_TRAIN = False
USE_DMIMAGEDETECT_TRAIN = False
```

### Run 02: add more Wang data

```python
USE_WANG_VAL_AS_TRAIN = True
USE_WANG_CNN_SYNTH_TEST_AS_TRAIN = True
USE_CORVI_TRAIN = True
```

### Run 03: add RealRAISE for more real images

```python
USE_WANG_VAL_AS_TRAIN = True
USE_WANG_CNN_SYNTH_TEST_AS_TRAIN = True
USE_CORVI_TRAIN = True
USE_REALRAISE_TRAIN = True
```

### Run 04: open-style larger synthetic training

```python
USE_WANG_VAL_AS_TRAIN = True
USE_WANG_CNN_SYNTH_TEST_AS_TRAIN = True
USE_CORVI_TRAIN = True
USE_REALRAISE_TRAIN = True
USE_DMIMAGEDETECT_TRAIN = True
```

For now: **Wang val + Corvi for training, DMImageDetect-Test for eval.**
