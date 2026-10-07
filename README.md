# xdfdet: deepfake video detection with visual explanations

xdfdet checks whether a face video is real or manipulated and produces a heatmap of the regions behind its prediction. The study asks how training choices change both detection quality and the evidence a reviewer can inspect.

<img src="https://raw.githubusercontent.com/mertkayacs/xdfdet/main/docs/figures/strip-gradcam.webp" width="520" alt="Grad-CAM heatmaps of four training setups on the same real portrait: baseline, augmentation, cutout, and augmentation with cutout, the best setup">

[Read the study](https://xdfdet.mertkayacs.com) or try one of the eight released detectors below.

## Run a detector

Requires Python 3.10 to 3.12. Model weights download from Hugging Face on first use; a training dataset is not needed for prediction.

```sh
pip install git+https://github.com/mertkayacs/xdfdet && xdfdet predict video.mp4 --gradcam cam.png
```

The command returns a real-video probability and verdict, and saves a Grad-CAM heatmap to `cam.png`. Grad-CAM shows which image regions influence a prediction; it does not prove that the video is authentic or explain the cause of a manipulation.

Use the [quickstart notebook](notebooks/quickstart.ipynb) for Python and Colab examples. [Hugging Face](https://huggingface.co/mertkayacs/xdfdet) and [Kaggle](https://www.kaggle.com/models/mertilovski/xdfdet) host the eight EfficientNet-B4 models.

## Study and results

Nine training configurations vary image augmentation and cutout, which masks part of a face during training. The data is a paired subset of FaceForensics++, a widely used dataset of real and manipulated face videos. Each prediction averages frame scores; Grad-CAM attention is measured in eight facial regions.

<img src="https://raw.githubusercontent.com/mertkayacs/xdfdet/main/docs/figures/pair-data.webp" width="620" alt="A real FaceForensics++ video frame next to its manipulated version">

<img src="https://raw.githubusercontent.com/mertkayacs/xdfdet/main/docs/figures/pair-cutout.webp" width="620" alt="Training-time cutout: a black-filled region on a fake frame and a small star-shaped patch on a real frame">

<img src="https://raw.githubusercontent.com/mertkayacs/xdfdet/main/docs/figures/pair-explain.webp" width="620" alt="Grad-CAM heatmap of the best setup next to the eight facial regions used to measure attention">

In the paper, augmentation with black-fill cutout reached **0.8971 AUC**, compared with **0.8678 for the baseline EfficientNet-B4**. AUC measures how well a detector ranks real and fake videos; higher is better. These are means over three runs. The released checkpoint's AUC is **0.8981**, a separate single-run result.

The four metrics disagree on the best configuration. A ranking score alone does not describe confidence quality or the facial regions used for a prediction.

<details>
<summary><b>Full results table</b></summary>

AUC, F1, Brier and LogLoss on the FaceForensics++ test split. **Paper** is the mean ± std over three runs (Table II). **Released** is the single run whose weights are published, measured in its original notebook.

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

The nine settings:

| Augmentation ↓ · Cutout → | none | random fill | black fill | white fill |
|---|---|---|---|---|
| none | `baseline` | | | |
| flips only | | `cutout-random` | `cutout-black` | `cutout-white` |
| standard | `aug-standard` | `aug-cutout-random` | `aug-cutout-black` | `aug-cutout-white` |
| intense | `aug-intense` | | | |

Data, model, optimizer and training schedule are the same in every setting. The exact probabilities are in [`src/xdfdet/config.py`](src/xdfdet/config.py).

</details>

## Limits

On 398 labelled videos from the separate Deepfake Detection Challenge dataset (DFDC), the released models reached AUC 0.60 to 0.66, below their FaceForensics++ scores. They detected 15% to 36% of the fakes. See the [DFDC evaluation notebook](https://www.kaggle.com/code/mertilovski/xdfdet-on-dfdc) for that run.

The original configurations used different random train/test splits. Their results are not a paired comparison on one common test set. Published weights may have seen another configuration's test videos during training. Read the original-run notes below before comparing or re-scoring checkpoints.

These are research detectors. A verdict or heatmap is insufficient evidence for an authenticity decision on an unfamiliar video.

## Reproduce the study

Training and the full tables need FaceForensics++ from [its authors](https://github.com/ondyari/FaceForensics) (research use only) and a GPU; a Colab T4 is enough.

<details>
<summary><b>Step by step</b></summary>

1. Put the raw videos in one folder: real clips as `000.mp4, 001.mp4, ...`, fakes as `<Manipulation>/000_003.mp4`.
2. Crop the faces (MTCNN, eye-aligned, 32 frames, 224×224):
   ```bash
   xdfdet crop raw/ faces/
   ```
3. Train one setting. The split is 70/15/15 over 1,000 real/fake pairs, one manipulation per pair in rotation:
   ```bash
   xdfdet train --data faces/ --config aug-cutout-black --out aug-cutout-black.keras
   ```
4. Score the test split, then run the region analysis:
   ```bash
   xdfdet evaluate --data faces/ --model aug-cutout-black.keras && xdfdet analyze --data faces/ --model aug-cutout-black.keras --out regions.json
   ```

`analyze` reports, for each outcome (TP, TN, FP, FN, with fake as the positive class), the mean and spread of Grad-CAM activation in the eight regions.

Tests run on a CPU without the dataset: `pip install -e ".[test]" && pytest`. To redraw the figures: `python scripts/make_figures.py docs/figures`.

</details>

## Notes on the original runs

The code was rebuilt from the Colab notebooks behind the paper and checked against them. Reading those notebooks surfaced details the paper does not state.

<details>
<summary><b>Read these before comparing numbers</b></summary>

- **Splits.** Each notebook drew its own random 70/15/15 split with no fixed seed, so every configuration was tested on a different set of 150 pairs. A published checkpoint may have trained on videos that sit in another configuration's test set, so re-scoring the checkpoints on one common split would mix training and test data. This code seeds the split (`--seed 42`), so new runs are comparable.
- **Dropout.** The paper gives 0.55 for every model. The released `aug-cutout-random` checkpoint was trained with 0.25. Dropout is off at inference, so this affects how the model was trained and not how you use it.
- **Cutout-only models** kept Albumentations' `HorizontalFlip` at its default probability of 0.5, so they saw random flips.
- **Missing checkpoint.** The `aug-standard` and `aug-intense` runs saved to the same file name, and the later save overwrote `aug-intense`. Its row comes from the paper; there are no weights for it.
- **Jaw region.** The region masks are the landmark polygons filled as drawn. The jaw polygon (points 0 to 16) closes across the lower face, so its score overlaps the nose and mouth regions.
- **Precision.** Training and the reported scores ran in mixed float16 on a T4 GPU. The released files are float32 copies with identical weights, and scores can differ from the float16 runs, most near the 0.5 boundary. `conversion.json` on Hugging Face lists both for two test images. To score in the original precision on a GPU, use `xdfdet evaluate --mixed-precision` or `xdfdet.load_model(name, mixed_precision=True)`.
- **Loading the Colab checkpoints.** In Keras 3.10, `keras.models.load_model` on the original `.h5` files leaves EfficientNet's normalization layer inactive. The released models rebuild the architecture and copy the weights, which avoids this.
- **Input scaling.** Frames are normalized with ImageNet statistics before the model, and Keras' EfficientNet normalizes again inside. The weights were trained this way, so keep both steps.
- **Metrics.** F1 treats real as the positive class; the Grad-CAM outcomes treat fake as positive. Both follow the thesis.
- **Crop margin.** The thesis says faces were cropped "with a fixed margin" without giving a value. `xdfdet crop` uses 30% of the face box.

[`scripts/convert_checkpoints.py`](scripts/convert_checkpoints.py) shows how each Colab checkpoint was matched to its configuration and converted.

</details>

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

Code: [MIT](LICENSE). Model weights: CC BY-NC 4.0, under the non-commercial research terms of [FaceForensics++](https://github.com/ondyari/FaceForensics). Eight checkpoints are released; `aug-intense` is unavailable because its checkpoint was overwritten in the original runs.

The FaceForensics++ frames in `docs/figures/thesis-*` come from the thesis (CC BY 4.0). The portrait used for package figures is a [Pexels photo](https://www.pexels.com/photo/close-photo-of-a-woman-in-hoodie-sweater-10349430/) under the Pexels license.

This code accompanies Mert Kaya and Venera Adanova's UBMK 2026 paper, accepted and to appear in IEEE Xplore, and Mert Kaya's [MSc thesis](https://doi.org/10.5281/zenodo.18998566) at TED University. No paper DOI is available yet.

<a href="https://eschatialabs.com"><picture><source media="(min-resolution: 2dppx)" srcset="https://eschatialabs.com/brand/lockup-46@2x.png"><img src="https://eschatialabs.com/brand/lockup-46@1x.png" width="124" height="46" alt="Eschatia Labs"></picture></a><br>An [Eschatia Labs](https://eschatialabs.com) project by [Mert Kaya](https://mertkayacs.com).
