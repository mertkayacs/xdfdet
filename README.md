# xdfdet — Explainable Deepfake Detection

This repository contains the code and experiments for regional explainability analysis of CNN-based deepfake detectors. Rather than proposing a new detection architecture, this work investigates **what facial regions drive model decisions** under different preprocessing configurations, using Grad-CAM activation mapping across eight landmark-defined facial areas.

> **Explainable Deepfake Detection Using Frame Level CNN Models: A Comparative Study of Augmentation and Cutout Techniques**  
> Mert Kaya — MSc Computer Engineering, TED University, 2025  
> Supervisor: Asst. Prof. Dr. Venera Adanova  
> DOI: [10.5281/zenodo.18998566](https://doi.org/10.5281/zenodo.18998566)

## Motivation

Deepfake detection models regularly report AUC scores above 0.90, yet they remain largely unusable in contexts where decisions must be justified. A binary "real/fake" output with a confidence score is not sufficient when:

- **Legal proceedings** require evidence that can be examined and challenged. A judge or forensic analyst cannot cross-examine a black-box prediction. Detection outputs need to be traceable to specific visual cues in order to qualify as admissible evidence.
- **Content moderation at scale** flags millions of items. When a piece of content is removed, the creator is entitled to know why. "The model said so" is not an acceptable explanation under transparency requirements like the EU AI Act and DSA.
- **Model auditing and failure analysis** is impossible without understanding where the model focuses. If a detector achieves 90% AUC but relies on JPEG compression artifacts rather than facial manipulation cues, it will fail silently on re-compressed or high-quality deepfakes.

Existing benchmarks (DeepfakeBench, DeepFake-o-Meter) evaluate detection accuracy across multiple models and datasets, but none of them analyze the spatial attention patterns that underlie those predictions. This work addresses that gap.

## Approach

The detection pipeline is based on Seferbekov's DFDC competition solution, adapted for FaceForensics++:

1. **Frame extraction** — 12 frames per video at equal intervals, face-cropped to 224×224 using MTCNN
2. **SSIM-guided cutout** — structural similarity between real and fake frames identifies the most similar (hardest to detect) facial region; a polygon cutout is applied to fake frames with black, white, or random fill
3. **Star cutout on real frames** — small star-shaped masks prevent overfitting to pristine facial texture
4. **Augmentation** — Albumentations pipeline (noise, blur, geometric, color) at two intensity levels
5. **Classification** — EfficientNet-B4 backbone in a TimeDistributed wrapper, frame-level sigmoid predictions averaged for video-level classification
6. **Explainability** — Grad-CAM heatmaps computed per frame, mapped onto eight dlib 68-point landmark regions (left/right eyes, eyebrows, nose, inner/outer mouth, jaw), aggregated across TP/TN/FP/FN cases

Nine configurations are compared, varying augmentation (on/off, low/high probability) and cutout (on/off, black/white/random fill). All other hyperparameters are held constant.

## Findings

**Augmentation without cutout degrades performance.** The augmentation-only model scored lower than the no-augmentation baseline (AUC 0.8610 vs 0.8678). Combining augmentation with SSIM-guided cutout recovered and exceeded baseline performance (0.8971). This suggests that augmentation introduces noise that is only beneficial when paired with targeted regularization.

**Fill strategy has a measurable effect.** Black-filled cutout with augmentation achieved the highest AUC (0.8971), followed by random (0.8837) and white (0.8734). This contradicts Seferbekov's original finding that random fill performs best — likely due to the difference in dataset scale (1000 pairs here vs 100K+ in DFDC).

**Different metrics produce different rankings.** The augmentation-only model ranks last in AUC but second in F1-score. The random-filled model ranks second in AUC but first in LogLoss. Evaluating deepfake detectors on a single metric can be misleading.

**The nose region dominates model attention regardless of correctness.** Grad-CAM regional analysis shows that the nose has the highest mean activation across all four prediction categories (TP, TN, FP, FN) and across all nine configurations. The model over-relies on nose-region features for both correct and incorrect decisions.

**Augmentation produces more distributed attention patterns.** Models trained with augmentation show activation standard deviation of 11–27% across facial regions, compared to 25–37% without. More distributed attention correlates with lower false positive rates, while concentrated attention on a single region leads to overfitting.

## Results

| Model | AUC | F1 | Brier | LogLoss |
|---|---|---|---|---|
| **blackfilledwithaug** | **0.8971** ± 0.0064 | **0.8429** ± 0.0101 | **0.1242** ± 0.0009 | 0.4710 ± 0.0063 |
| randomfilledwithaug | 0.8837 ± 0.0072 | 0.7950 ± 0.0098 | 0.1450 ± 0.0011 | **0.4656** ± 0.0067 |
| whitefilledwithaug | 0.8734 ± 0.0061 | 0.7951 ± 0.0095 | 0.1455 ± 0.0012 | 0.4761 ± 0.0071 |
| noaugrandomfill | 0.8711 ± 0.0051 | 0.7769 ± 0.0069 | 0.1524 ± 0.0012 | 0.5241 ± 0.0078 |
| onlymoreaug | 0.8718 ± 0.0043 | 0.7932 ± 0.0077 | 0.1431 ± 0.0008 | 0.5288 ± 0.0081 |
| baseline | 0.8678 ± 0.0044 | 0.7780 ± 0.0065 | 0.1524 ± 0.0013 | 0.4827 ± 0.0069 |
| noaugblackfill | 0.8666 ± 0.0043 | 0.7912 ± 0.0077 | 0.1537 ± 0.0009 | 0.4989 ± 0.0074 |
| noaugwhitefill | 0.8639 ± 0.0047 | 0.7704 ± 0.0074 | 0.1462 ± 0.0011 | 0.5244 ± 0.0076 |
| onlyaug | 0.8610 ± 0.0051 | 0.7955 ± 0.0084 | 0.1572 ± 0.0010 | 0.5719 ± 0.0084 |

All models trained 3 times on FaceForensics++ (1000 video pairs, 70/15/15 split). Values are mean ± std.

## Limitations

- Dataset is limited to 1000 video pairs from FF++. Results may differ at larger scale.
- No cross-dataset evaluation (CelebDF, DFDC).
- Frame-level spatial analysis only — no temporal modeling (LSTM, attention over frames).
- Single backbone architecture (EfficientNet-B4).

## Repository Structure

```
notebooks/
  01_data_preparation.ipynb      # FF++ pairing, splitting
  02_preprocessing_pipeline.ipynb # SSIM, landmarks, cutout, augmentation
  03_training.ipynb               # 9 configs, model, training
  04_evaluation.ipynb             # AUC, F1, LogLoss, Brier, curves
  05_explainability.ipynb         # Grad-CAM, regional analysis, case comparison

app/
  app.py                          # Streamlit demo (in development)
```

## Setup

Python 3.9+, TensorFlow 2.10+, GPU recommended.

```
pip install tensorflow albumentations scikit-image scikit-learn opencv-python dlib seaborn tqdm
```

dlib requires `shape_predictor_68_face_landmarks.dat` — download from [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).

Notebooks are designed for Google Colab with GPU runtime. Run 01–04 sequentially to reproduce training. Notebook 05 requires a trained model.

## Citation

```bibtex
@mastersthesis{kaya2025xdfdet,
  title     = {Explainable Deepfake Detection Using Frame Level CNN Models:
               A Comparative Study of Augmentation and Cutout Techniques},
  author    = {Kaya, Mert},
  school    = {TED University},
  year      = {2025},
  doi       = {10.5281/zenodo.18998566}
}
```

## License

MIT
