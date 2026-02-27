# Pose6D: 6D Object Pose Estimation from RGB-D Images

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-00FFFF?style=flat-square&logo=yolo&logoColor=black"/>
  <img src="https://img.shields.io/badge/Python_3.x-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
</p>

This project estimates the full **6D pose** — 3D rotation and 3D translation — of objects from RGB and RGB-D images. The pipeline combines **YOLOv8-based object detection** with a **PoseNet-inspired pose regression network**, evaluated on the LineMOD benchmark. Potential applications include robotics manipulation, augmented reality, and object tracking.

> Full methodology and experimental analysis: [Report.pdf](Report.pdf)

---

## Overview

Estimating an object's 6D pose from a single image is a core challenge in computer vision with direct applications in robotic grasping, AR scene understanding, and autonomous navigation. This project implements a modular two-stage pipeline:

1. **Object Detection** — YOLOv8-nano fine-tuned on LineMOD performs real-time 2D localization, producing tight bounding boxes and class labels for each object in the scene.

2. **Pose Regression** — Cropped object regions are passed to one of two pose estimators:
   - **PoseNet6D** (RGB): A ResNet-18 backbone extracts visual features, combined with a learnable object ID embedding and normalized bounding box coordinates. The regression head predicts a unit quaternion for rotation and a scalar depth for translation, back-projected to 3D via the cropped intrinsic matrix `K_crop`.
   - **PoseNet6D_RGBD** (RGB + Depth): Extends the above with a parallel depth CNN branch. RGB and depth features are fused through channel-wise concatenation and a 1×1 convolution. Segmentation masks suppress background during training, and Gaussian blur is applied to depth inputs for noise reduction.

---

## Results

**Detection — YOLOv8 on LineMOD (test set)**

| Precision | Recall | mAP@50 | mAP@50-95 |
|:---------:|:------:|:------:|:---------:|
| 99.6% | 99.1% | 99.2% | 91.3% |

**Pose Estimation — ADD metric (lower is better)**

| Model | Modality | ADD (m) |
|-------|----------|:-------:|
| PoseNet6D | RGB | 0.095 |
| PoseNet6D_RGBD | RGB + Depth | **0.0795** |

Incorporating depth information yields a **~17% reduction in average pose error**, with the most notable gains on symmetric and low-texture objects such as glue and eggbox.

---

## Repository Structure

```
├── dataset/
│   └── yolo_conversion_steps.ipynb   # LineMOD → YOLO format conversion
├── models/
│   ├── YOLO_training.ipynb           # YOLOv8 fine-tuning
│   ├── training_RGB.ipynb            # PoseNet6D (RGB) training
│   ├── training_RGBD.ipynb           # PoseNet6D_RGBD training
│   ├── tryModel_RGB.ipynb            # RGB inference & evaluation
│   └── tryModel_RGBD.ipynb           # RGB-D inference & evaluation
├── checkpoints/                      # Model weights (see note below)
├── requirements.txt
└── Report.pdf
```

---

## Setup

```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm ultralytics scipy pillow scikit-image
```

Run notebooks in order: dataset conversion → detection training → pose training → evaluation.
Compatible with local Jupyter and Google Colab (mount Drive and update paths to `/content/`).

**Required files for inference:** `yolo.pt`, `posenet6d_RGB.pt`, `posenet6d_RGBD.pt`

> Trained weights exceed GitHub's file size limit and are not included. Open an Issue or contact the authors to request them.

---

## Authors

Hasti Azadnia · Elias Noorzad · Ayda Ghasemazar · Filip Nykvist  
*MSc Data Science & Engineering — Politecnico di Torino*
