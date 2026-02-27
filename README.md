
# Pose6D: 6D Object Pose Estimation via YOLOv8 and PoseNet
End-to-end 6D object pose estimation pipeline combining YOLOv8 (99.6% precision) with a PoseNet-inspired RGB-D regression network. Achieves 17% accuracy improvement over RGB-only baseline on the LineMOD dataset.


<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-00FFFF?style=flat-square&logo=yolo&logoColor=black"/>
  <img src="https://img.shields.io/badge/Python_3.x-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
</p>

This repository contains the implementation of an end-to-end pipeline for **6D object pose estimation** from RGB-D images, developed as part of a graduate course project at Politecnico di Torino. The pipeline integrates a fine-tuned YOLOv8 detector for 2D object localization with a PoseNet-inspired network for direct regression of 3D translation and rotation.

Two model variants are implemented and evaluated on the **LineMOD** dataset using the ADD metric: an RGB-only baseline and an RGB-D extension incorporating depth fusion, achieving a **17% improvement** in pose accuracy over the baseline.

A full technical description of the methodology is available in [Report.pdf](Report.pdf).

---

## Method

The pipeline consists of two sequential modules:

**1. Object Detection** — A YOLOv8-nano model fine-tuned on LineMOD detects and localizes objects in RGB frames, producing bounding boxes and class labels used to crop object regions for the downstream estimator.

**2. Pose Estimation** — Two regression networks predict the 6D pose (quaternion `q̂` + depth `Ẑ`) from cropped regions:
- `PoseNet6D`: ResNet-18 backbone on RGB crops, with object ID embedding and bounding box features fed into a pose regression head.
- `PoseNet6D_RGBD`: Extends the above with a parallel depth CNN branch; RGB and depth features are fused via channel-wise concatenation and a 1×1 convolution before regression. Ground-truth segmentation masks are applied externally during training to suppress background.

Translation is recovered by back-projecting the predicted depth through the cropped intrinsic matrix `K_crop`. Rotation is represented as an ℓ₂-normalized quaternion. The training objective combines weighted MSE for translation (with log-transform on the Z-axis) and a quaternion dot-product loss for rotation.

---

## Results

**Object Detection (YOLOv8 on LineMOD test set)**

| Precision | Recall | mAP@50 | mAP@50-95 |
|:---------:|:------:|:------:|:---------:|
| 99.6% | 99.1% | 99.2% | 91.3% |

**Pose Estimation (ADD metric, lower is better)**

| Model | Input | ADD (m) |
|-------|-------|:-------:|
| PoseNet6D | RGB | 0.095 |
| PoseNet6D_RGBD | RGB + Depth | **0.0795** |

---

## Repository Structure

```
├── dataset/
│   └── yolo_conversion_steps.ipynb   # LineMOD → YOLO format conversion
├── models/
│   ├── YOLO_training.ipynb           # YOLOv8 fine-tuning
│   ├── training_RGB.ipynb            # PoseNet6D training
│   ├── training_RGBD.ipynb           # PoseNet6D_RGBD training
│   ├── tryModel_RGB.ipynb            # RGB inference & evaluation
│   └── tryModel_RGBD.ipynb           # RGB-D inference & evaluation
├── checkpoints/                      # Model weights (see note below)
├── requirements.txt
└── Report.pdf
```

---

## Setup & Usage

```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm ultralytics scipy pillow scikit-image
```

Run the notebooks in order: dataset conversion → YOLO training → pose training → evaluation. Notebooks are compatible with both local Jupyter and Google Colab environments.

> **Model weights** are not included due to GitHub's file size limit. Request via Issues or email.

---

## Authors

Hasti Azadnia · Elias Noorzad · Ayda Ghasemazar · Filip Nykvist  
*MSc Data Science & Engineering, Politecnico di Torino*
