---
license: cc-by-nc-sa-4.0   # TODO confirm (inherits PAXRay++)
library_name: pytorch
pipeline_tag: keypoint-detection
tags:
  - chest-x-ray
  - medical-imaging
  - anatomical-landmarks
  - segmentation
base_model: microsoft/rad-dino
---

# cxr-landmarks: joint dense model (v1)

Predicts 311 semantically named anatomical landmarks and masks of 177 anatomical structures on
frontal and lateral chest radiographs. Code, landmark definitions and training: <!-- GitHub URL -->.

> **Research use only.** Not a medical device; not for clinical decisions.

## Model

A RAD-DINO ViT-B/14 encoder (fine-tuned end-to-end) shared by two deconvolution decoders: landmark
heatmaps at stride 4 (one channel per view-specific class–landmark pair; decoded with DARK) and
class masks (one sigmoid channel per structure). Inputs: RGB, shortest edge 512 px.

## Usage

```python
from cxrlandmarks.predict import LandmarkPredictor
predictor = LandmarkPredictor.from_pretrained("<!-- hf repo id -->")
out = predictor.predict("chest_xray.png", view="frontal")
```

## Training data

PAXRay++ (images with masks of 177 structures projected from CT). Landmark labels are silver
labels produced by applying the paper's geometric rules to these masks (`cxrlm-make-labels`).
Trained for 18 epochs (checkpoint selected on validation OKS-AP: epoch 16).

## Evaluation (PAXRay++ test, silver labels, 95% bootstrap CI over images)

| OKS-AP | PCK@0.1 | Mask AP |
|---|---|---|
| 36.8 [36.2, 37.7] | 80.6 [79.7, 81.4] | 52.4 [51.7, 53.3] |

## Evaluation on real CXRs (gold standard, MIMIC-CXR)

50 images read by two readers (nine landmark groups); OKS with per-landmark inter-reader tolerances.

| OKS-AP | PCK@0.1 | MRE (px, 512²) |
|---|---|---|
| 51.1 [48.1, 56.7] | 76.7 [73.0, 80.1] | 14.2 [12.5, 16.4] |

## Limitations

- Trained on CT-projected (synthetic) radiographs; accuracy on real radiographs is lower (see above).
- The view (frontal/lateral) must be supplied.
- One instance per structure; no uncertainty calibration of landmark confidences.

## Citation

```bibtex
@inproceedings{seibold2027landmarks,
  title     = {Semantic landmark detection in chest radiographs},
  author    = {Seibold, Constantin and Havemann, Nils and Vinu, David and Fink, Matthias A.},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2027}
}
```
