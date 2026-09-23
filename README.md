# xdfdet

Code and trained models for the paper **"Augmentation and Cutout in Deepfake Detection: A Comparative Study of Accuracy, Calibration, and Attention"** (UBMK 2026) and the MSc thesis it comes from.

[Project site](https://xdfdet.mertkayacs.com) · [Models on Hugging Face](https://huggingface.co/mertkayacs/xdfdet) · [Thesis (DOI 10.5281/zenodo.18998566)](https://doi.org/10.5281/zenodo.18998566)

An EfficientNet-B4 scores 12 face-cropped frames per video, and the frame scores are averaged into one video score. We trained it under nine preprocessing configurations that vary augmentation and cutout, compared them with four metrics, and used Grad-CAM on 68 facial landmarks to see which face regions each model relies on.

## Quick start

Python 3.10 to 3.12.

```bash
pip install git+https://github.com/mertkayacs/xdfdet
xdfdet predict video.mp4 --gradcam cam.png
```

Output for a clip of a public-domain NASA portrait (a real person):

```json
{
  "video": "portrait.mp4",
  "model": "aug-cutout-black",
  "real_probability": 0.9996,
  "verdict": "real",
  "regions": {
    "jaw": 42.4, "left_eyebrow": 62.7, "right_eyebrow": 42.7, "nose": 62.8,
    "left_eye": 81.4, "right_eye": 57.2, "outer_mouth": 37.8, "inner_mouth": 37.3
  },
  "gradcam": "cam.png"
}
```

The model downloads from Hugging Face on first use. `predict` finds the face with MTCNN, crops 32 frames, scores 12 of them, and writes a Grad-CAM overlay with the mean activation per face region. Pick another configuration with `--model`, for example `--model baseline`.

From Python:

```python
import xdfdet
model = xdfdet.load_model("aug-cutout-black")   # Keras model, input (batch, 12, 224, 224, 3)
```

Or open [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) in Colab.

## Results

AUC, F1, Brier and LogLoss on the FaceForensics++ test split. **Paper** is the mean ± std over three runs (Table II). **Released** is the single run whose weights are published here, measured in its original notebook.

| Model | Paper AUC | Paper F1 | Paper Brier | Paper LogLoss | Released AUC | Released F1 | Released Brier | Released LogLoss |
|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.8678 ± 0.0044 | 0.7780 ± 0.0065 | 0.1524 ± 0.0013 | 0.4827 ± 0.0069 | 0.8684 | 0.7781 | 0.1523 | 0.4827 |
| `aug-standard` | 0.8610 ± 0.0051 | 0.7955 ± 0.0084 | 0.1572 ± 0.0010 | 0.5719 ± 0.0084 | 0.8616 | 0.8025 | 0.1576 | 0.5719 |
| `aug-intense` | 0.8718 ± 0.0043 | 0.7932 ± 0.0077 | 0.1431 ± 0.0008 | 0.5288 ± 0.0081 | not released | | | |
| `cutout-random` | 0.8711 ± 0.0051 | 0.7769 ± 0.0069 | 0.1524 ± 0.0012 | 0.5241 ± 0.0078 | 0.8642 | 0.7774 | 0.1526 | 0.5241 |
| `cutout-black` | 0.8666 ± 0.0043 | 0.7912 ± 0.0077 | 0.1537 ± 0.0009 | 0.4989 ± 0.0074 | 0.8669 | 0.7911 | 0.1537 | 0.4989 |
| `cutout-white` | 0.8639 ± 0.0047 | 0.7704 ± 0.0074 | 0.1462 ± 0.0011 | 0.5244 ± 0.0076 | 0.8700 | 0.7703 | 0.1463 | 0.5244 |
| `aug-cutout-random` | 0.8837 ± 0.0072 | 0.7950 ± 0.0098 | 0.1450 ± 0.0011 | **0.4656 ± 0.0067** | 0.8820 | 0.7950 | 0.1451 | 0.4656 |
| `aug-cutout-black` | **0.8971 ± 0.0064** | **0.8429 ± 0.0101** | **0.1242 ± 0.0009** | 0.4710 ± 0.0063 | 0.8981 | 0.8431 | 0.1247 | 0.4710 |
| `aug-cutout-white` | 0.8734 ± 0.0061 | 0.7951 ± 0.0095 | 0.1455 ± 0.0012 | 0.4761 ± 0.0071 | 0.8734 | 0.7883 | 0.1451 | 0.4761 |

What the paper found:

- Augmentation plus a black-filled cutout is the strongest setup on AUC, F1 and Brier. Standard augmentation on its own scores below the baseline on AUC.
- The four metrics disagree. `aug-standard` ranks last on AUC but second on F1, and the random-fill model has the lowest LogLoss. Judging a detector on one number hides this.
- The nose carries high Grad-CAM activation whether the model is right or wrong, so the region behind correct detections also dominates the errors. On missed fakes it rises from 49% at the baseline to 78% for the black-fill model.

The thesis adds one more result: models trained with augmentation spread their attention more evenly over the face, with a standard deviation of 11 to 27 across regions against 25 to 37 without augmentation.

## The nine configurations

| Model | Augmentation | Cutout fill |
|---|---|---|
| `baseline` | none | none |
| `aug-standard` | standard | none |
| `aug-intense` | intense | none |
| `cutout-random` / `cutout-black` / `cutout-white` | horizontal flip only | random / black / white |
| `aug-cutout-random` / `aug-cutout-black` / `aug-cutout-white` | standard | random / black / white |

Cutout follows Seferbekov's DFDC solution: an SSIM map between a real frame and its fake finds where the two look most alike, and a landmark polygon over that area is blanked on the fake frames. Real frames get a small star-shaped cutout instead. Exact probabilities are in [`src/xdfdet/config.py`](src/xdfdet/config.py).

## Reproduce

You need FaceForensics++ access from [its authors](https://github.com/ondyari/FaceForensics) (research use only) and a GPU for training. Evaluation and Grad-CAM run on a CPU, just slowly.

1. Put the raw videos in one folder: real clips as `000.mp4, 001.mp4, ...`, fakes as `<Manipulation>/000_003.mp4`. We used FaceSwap, Face2Face, FaceShifter and Deepfakes.
2. Crop faces (MTCNN, eye-aligned, 32 frames, 224×224):
   ```bash
   xdfdet crop raw/ faces/
   ```
3. Train a configuration. The split is 70/15/15 over 1,000 real/fake pairs, one manipulation per pair in rotation:
   ```bash
   xdfdet train --data faces/ --config aug-cutout-black --out aug-cutout-black.keras
   ```
4. Score the test split, then run the region analysis:
   ```bash
   xdfdet evaluate --data faces/ --model aug-cutout-black.keras
   xdfdet analyze  --data faces/ --model aug-cutout-black.keras --out regions.json
   ```

`analyze` reports, for each outcome (TP, TN, FP, FN with fake as the positive class), the mean and spread of Grad-CAM activation in the eight regions: jaw, both eyebrows, both eyes, nose, outer and inner mouth.

Tests run on a CPU without the dataset: `pip install -e ".[test]" && pytest`.

## Notes on the original runs

We rebuilt this code from the Colab notebooks behind the paper and checked it against them. Reading those notebooks surfaced a few details the paper does not state. We list them here so the published numbers can be read correctly.

- **Splits.** Each notebook drew its own random 70/15/15 split with no fixed seed, so every configuration was tested on a different set of 150 pairs. A published checkpoint may therefore have trained on videos that sit in another configuration's test set, and re-scoring the checkpoints on one common split would mix training and test data. This code seeds the split (`--seed 42`), so new runs are comparable.
- **Dropout.** The paper gives 0.55 for every model. The released `aug-cutout-random` checkpoint was trained with 0.25. Dropout is off at inference, so this changes how the model was trained and has no effect on how you use it.
- **Cutout-only models** kept Albumentations' `HorizontalFlip` at its default probability of 0.5, so they saw random flips.
- **Missing checkpoint.** The `aug-standard` and `aug-intense` runs saved to the same file name, and the later save overwrote `aug-intense`. Its row above comes from the paper; there are no weights for it.
- **Precision.** Training and the reported scores ran in mixed float16 on a T4 GPU. The released files are float32 copies with identical weights, and scores can differ from the float16 runs, most on inputs near the 0.5 boundary. `conversion.json` on Hugging Face lists both for two test images. To score in the original precision on a GPU, use `xdfdet evaluate --mixed-precision` or `xdfdet.load_model(name, mixed_precision=True)`.
- **Loading the Colab checkpoints.** In Keras 3.10, `keras.models.load_model` on the original `.h5` files leaves EfficientNet's normalization layer inactive. The released models rebuild the architecture and copy the weights, which avoids this.
- **Input scaling.** Frames are normalized with ImageNet statistics before the model, and Keras' EfficientNet normalizes again inside. The weights were trained this way, so keep both steps.
- **Metrics.** F1 treats real as the positive class; the Grad-CAM outcomes treat fake as positive. Both follow the thesis.
- **Crop margin.** The thesis says faces were cropped "with a fixed margin" without giving a value. `xdfdet crop` uses 30% of the face box.

[`scripts/convert_checkpoints.py`](scripts/convert_checkpoints.py) shows how each Colab checkpoint was matched to its configuration and converted to the released float32 `.keras` files with identical weights.

## Citation

```bibtex
@inproceedings{kaya2026augmentation,
  title     = {Augmentation and Cutout in Deepfake Detection: A Comparative Study of
               Accuracy, Calibration, and Attention},
  author    = {Kaya, Mert and Adanova, Venera},
  booktitle = {11th International Conference on Computer Science and Engineering (UBMK 2026)},
  year      = {2026},
  note      = {To appear}
}

@mastersthesis{kaya2025xdfdet,
  title  = {Explainable Deepfake Detection Using Frame Level CNN Models:
            A Comparative Study of Augmentation and Cutout Techniques},
  author = {Kaya, Mert},
  school = {TED University},
  year   = {2025},
  doi    = {10.5281/zenodo.18998566}
}
```

## License

Code: GPL-3.0 (see `LICENSE`). Model weights: CC BY-NC 4.0, because they were trained on FaceForensics++, which is licensed for non-commercial research only.
