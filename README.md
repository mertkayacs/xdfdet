# xdfdet

**Where does a deepfake detector look?** We trained a computer to spot fake face videos, then made it show which part of the face it looked at before deciding.

[Project page](https://xdfdet.mertkayacs.com) ([Türkçe](https://xdfdet.mertkayacs.com/tr/), [Deutsch](https://xdfdet.mertkayacs.com/de/)) · [Models on Hugging Face](https://huggingface.co/mertkayacs/xdfdet) · [Thesis](https://doi.org/10.5281/zenodo.18998566)

<table>
  <tr>
    <td align="center"><img src="docs/figures/gradcam-baseline.webp" width="180" alt="Heat map, baseline detector"><br><sub>Baseline</sub></td>
    <td align="center"><img src="docs/figures/gradcam-aug-standard.webp" width="180" alt="Heat map, standard augmentation"><br><sub>Random edits</sub></td>
    <td align="center"><img src="docs/figures/gradcam-cutout-black.webp" width="180" alt="Heat map, black patch"><br><sub>Patch</sub></td>
    <td align="center"><img src="docs/figures/gradcam-aug-cutout-black.webp" width="180" alt="Heat map, best detector"><br><sub><b>Both (best)</b></sub></td>
  </tr>
</table>

*Four of our detectors look at the same real photo. All four say "real", but each one looks somewhere else.*

This repository has the code and eight trained models behind the paper *Augmentation and Cutout in Deepfake Detection: A Comparative Study of Accuracy, Calibration, and Attention* (UBMK 2026) and the MSc thesis it comes from.

## Try it

```bash
pip install git+https://github.com/mertkayacs/xdfdet
xdfdet predict video.mp4 --gradcam cam.png
```

You get the probability that the video is real, a verdict, and a heat map saved as `cam.png`. No dataset needed: the detector downloads itself the first time. Python 3.10 to 3.12.

<details>
<summary>Example output, Python usage and Colab</summary>

For a clip of the portrait above (a real person):

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

`regions` is the mean heat-map activation (0 to 100) inside each face region. Choose another detector with `--model`, for example `--model baseline`.

```python
import xdfdet
model = xdfdet.load_model("aug-cutout-black")   # Keras model, input (batch, 12, 224, 224, 3)
```

Or open [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) in Colab.

</details>

## What we did

**1. Collect faces.** We took 1,000 real videos and a fake version of each, and cropped out the faces.

<img src="docs/figures/thesis-ssim-real.webp" width="200" alt="Real frame"> <img src="docs/figures/thesis-ssim-fake.webp" width="200" alt="Fake frame">

<sub>Real and fake frame from FaceForensics++ (from the thesis). Four manipulation methods: FaceSwap, Face2Face, FaceShifter, Deepfakes.</sub>

**2. Train it nine ways.** We trained the same detector nine times. Each time we changed how the training faces were altered: small random edits, a patch over part of the fake face, both, or neither.

<img src="docs/figures/aug-standard.webp" width="200" alt="Random edits"> <img src="docs/figures/cutout-black.webp" width="200" alt="Patch on a fake face">

<sub>Left: random edits (augmentation). Right: a patch over the part of a fake face that already looks real (cutout).</sub>

**3. Ask it why.** For every video, a heat map shows which part of the face drove the decision. We measured it in eight regions: eyes, eyebrows, nose, mouth and jaw.

<img src="docs/figures/gradcam-aug-cutout-black.webp" width="200" alt="Heat map"> <img src="docs/figures/regions.webp" width="200" alt="Eight face regions">

<sub>Grad-CAM heat map and the eight region masks, drawn from 68 facial landmarks.</sub>

<details>
<summary><b>The full pipeline, step by step</b></summary>

Every image here except the FaceForensics++ frames is produced by [`scripts/make_figures.py`](scripts/make_figures.py), which runs the package's own functions on a public-domain portrait from scikit-image.

**Find the face.** MTCNN finds the face in each frame, levels the eyes and crops the face to 224×224 pixels. Each video keeps 32 frames, and the model sees 12 of them.

<img src="docs/figures/detect.webp" width="200" alt="Detected face and eye points"> <img src="docs/figures/crop.webp" width="200" alt="Aligned face crop">

**Map the face.** dlib places 68 landmarks that define the eight regions. The right image shows the masks exactly as scored; the jaw polygon closes across the lower face, so the jaw score also covers the cheeks, nose and mouth.

<img src="docs/figures/landmarks.webp" width="200" alt="68 landmarks"> <img src="docs/figures/regions.webp" width="200" alt="Region masks">

**Compare real and fake.** SSIM compares a real frame with its fake. Bright areas differ, dark areas look almost the same. Cutout aims at the dark areas, where the fake already passes for real.

<img src="docs/figures/thesis-ssim-real.webp" width="160" alt="Real frame"> <img src="docs/figures/thesis-ssim-map.webp" width="160" alt="SSIM difference map"> <img src="docs/figures/thesis-ssim-fake.webp" width="160" alt="Fake frame">

**Blank out a region.** A landmark polygon covering 2 to 5 percent of the frame is blanked on the fake frames with black, white or random pixels. Real frames get a small star-shaped cutout, so a blank patch alone never gives the answer away.

<img src="docs/figures/cutout-black.webp" width="150" alt="Black fill"> <img src="docs/figures/cutout-white.webp" width="150" alt="White fill"> <img src="docs/figures/cutout-random.webp" width="150" alt="Random fill"> <img src="docs/figures/star.webp" width="150" alt="Star cutout">

**Augment.** Albumentations adds noise, blur, slight color shifts, flips and small rotations. The intense setting applies them four to five times as often as the standard one. Below: input, one standard draw, one intense draw.

<img src="docs/figures/crop.webp" width="160" alt="Input"> <img src="docs/figures/aug-standard.webp" width="160" alt="Standard"> <img src="docs/figures/aug-intense.webp" width="160" alt="Intense">

**Score the video.** EfficientNet-B4 rates each of the 12 frames on its own, and the video score is the mean. Training uses binary cross-entropy on that mean. Below 0.5 means fake.

**Explain the decision.** Grad-CAM is computed on frames 0, 4 and 7, averaged, and measured inside each region. `xdfdet analyze` reports this per outcome (true and false positives and negatives).

</details>

## What we found

1. **Covering part of the fake face helps.** The best detector combined random edits with a black patch. It beat the plain detector on all four scores (AUC 0.8971 against 0.8678).
2. **Different scores, different rankings.** Standard augmentation comes last on AUC and second on F1. Judging a detector by one number hides this.
3. **The nose is a weak spot.** Detectors lean heavily on the nose, whether their answer is right or wrong.

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

| Random edits ↓ · Patch → | none | random fill | black fill | white fill |
|---|---|---|---|---|
| none | `baseline` | | | |
| flips only | | `cutout-random` | `cutout-black` | `cutout-white` |
| standard | `aug-standard` | `aug-cutout-random` | `aug-cutout-black` | `aug-cutout-white` |
| intense | `aug-intense` | | | |

Data, model, optimizer and training schedule are the same in every setting. The exact probabilities are in [`src/xdfdet/config.py`](src/xdfdet/config.py).

</details>

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
   xdfdet evaluate --data faces/ --model aug-cutout-black.keras
   xdfdet analyze  --data faces/ --model aug-cutout-black.keras --out regions.json
   ```

`analyze` reports, for each outcome (TP, TN, FP, FN, with fake as the positive class), the mean and spread of heat-map activation in the eight regions.

Tests run on a CPU without the dataset: `pip install -e ".[test]" && pytest`. To redraw the figures: `python scripts/make_figures.py docs/figures`.

</details>

## Models

Eight detectors are on [Hugging Face](https://huggingface.co/mertkayacs/xdfdet), one `.keras` file per setting, named as in the tables. `xdfdet.load_model(name)` downloads and loads one. A ninth, `aug-intense`, was lost.

## Notes on the original runs

The code was rebuilt from the Colab notebooks behind the paper and checked against them. Reading those notebooks surfaced details the paper does not state.

<details>
<summary><b>Read these before comparing numbers</b></summary>

- **Splits.** Each notebook drew its own random 70/15/15 split with no fixed seed, so every configuration was tested on a different set of 150 pairs. A published checkpoint may have trained on videos that sit in another configuration's test set, so re-scoring the checkpoints on one common split would mix training and test data. This code seeds the split (`--seed 42`), so new runs are comparable.
- **Dropout.** The paper gives 0.55 for every model. The released `aug-cutout-random` checkpoint was trained with 0.25. Dropout is off at inference, so this affects how the model was trained and not how you use it.
- **Patch-only models** kept Albumentations' `HorizontalFlip` at its default probability of 0.5, so they saw random flips.
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

Code: MIT (see [`LICENSE`](LICENSE)). Model weights: CC BY-NC 4.0, because they were trained on FaceForensics++, which is licensed for non-commercial research only. The FaceForensics++ frames in `docs/figures/thesis-*` come from the thesis (CC BY 4.0). The portrait is a public-domain NASA photo that ships with scikit-image.
