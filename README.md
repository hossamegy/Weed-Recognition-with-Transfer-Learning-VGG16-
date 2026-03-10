# 🌱 Weed Recognition via Transfer Learning (VGG16)

> **MSc AI Engineering — Coursework Implementation**  
> A modular PyTorch reproduction of Jiang (2019) with two-phase progressive fine-tuning and early stopping.

---

## 📄 Attribution & Copyright

This repository is an **educational reimplementation** of the following peer-reviewed paper:

> Jiang, Z. (2019). *A Novel Crop Weed Recognition Method Based on Transfer Learning from VGG16 Implemented by Keras.*  
> IOP Conference Series: Materials Science and Engineering, **677**, 032073.  
> DOI: [10.1088/1757-899X/677/3/032073](https://doi.org/10.1088/1757-899X/677/3/032073)  
> Published under **Creative Commons Attribution 3.0 (CC BY 3.0)** — © Zichao Jiang (2019)

All model design choices, experimental setup, and reported results belong to the original authors.  
This PyTorch implementation was produced solely for academic coursework and is **not** a commercial product.

**Dataset:**  
Plant Seedlings Classification — Aarhus University Signal Processing Group & University of Southern Denmark.  
Available on [Kaggle](https://www.kaggle.com/c/plant-seedlings-classification) under **CC0 Public Domain** license.

---

## 🎯 Overview

The paper uses VGG16 transfer learning to classify 12 weed species from field images. The key idea is to freeze the first 14 VGG16 layers (pre-trained on ImageNet) and train only a custom classification head (~330k parameters).

This implementation goes **beyond the original paper** by adding:
- ✅ Two-phase progressive fine-tuning (`AutoFinetuner`)
- ✅ Early stopping with best-model checkpointing (`Trainer`)
- ✅ Fully modular `src/` package structure
- ✅ Config-driven training via dataclass (`TrainingConfig`)
- ✅ `tqdm.auto` progress bars (works in both terminal and Jupyter/Colab)

| Metric | Paper Result |
|--------|-------------|
| Training Accuracy | **98.99%** |
| Validation Accuracy | **91.08%** |
| Trainable Parameters | **329,868** (~4.1% of total) |
| Optimizer | RMSprop |

---

## 📁 Repository Structure

```
weed-recognition-vgg16/
│
├── src/
│   ├── config.py               # TrainingConfig dataclass — all hyperparameters
│   ├── custom_dataset.py       # CustomDataSet — loads & filters images ≥300px
│   ├── data_augmentation.py    # DataAugmentation class + TransformSubset wrapper
│   ├── model_architecture.py   # WeedVGG16 — frozen VGG16 backbone + custom head
│   ├── trainer.py              # Trainer — training loop, early stopping, checkpointing
│   └── auto_finetuner.py       # AutoFinetuner — 2-phase progressive unfreezing
│
├── jupyter_notebooks/
│   └── Transfer_Learning_for_Weed_Recognition.ipynb   # Full Colab walkthrough
│
├── checkpoints/                # Auto-created at runtime
│   ├── phase1_best.pth         # Best model from Phase 1
│   └── phase2_best.pth         # Best model from Phase 2
│
├── pipline_training.py         # Main entry point — run full pipeline
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset Setup

### Option A — Kaggle API (recommended for Colab)

```bash
pip install kaggle
# Upload your kaggle.json API token, then:
kaggle competitions download -c plant-seedlings-classification
unzip plant-seedlings-classification.zip
```

### Option B — Manual download

Download from [Kaggle — Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification/data) and place the `train/` folder in the project root.

### Expected folder layout

```
train/
├── Black-grass/
├── Charlock/
├── Cleavers/
├── Common Chickweed/
├── Common wheat/
├── Fat Hen/
├── Loose Silky-bent/
├── Maize/
├── Scentless Mayweed/
├── Shepherds Purse/
├── Small-flowered Cranesbill/
└── Sugar beet/
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision tqdm pillow
```

### 2. Run the full pipeline

```bash
python pipline_training.py
```

### 3. Or open in Colab

```
jupyter_notebooks/Transfer_Learning_for_Weed_Recognition.ipynb
```

---

## ⚙️ Configuration

All hyperparameters live in `src/config.py` as a single dataclass — no scattered magic numbers:

```python
@dataclass
class TrainingConfig:
    data_dir:       str   = "/content/train"
    min_resolution: int   = 300          # filter images below 300×300 px (paper)
    batch_size:     int   = 32
    num_workers:    int   = 2
    mean:           tuple = (0.485, 0.456, 0.406)
    std:            tuple = (0.229, 0.224, 0.225)

    num_classes:    int   = 12
    dropout:        float = 0.2
    seed:           int   = 42
    save_dir:       str   = "checkpoints"

    # Phase 1 — train classifier head only
    phase1_epochs:   int   = 50
    phase1_lr:       float = 1e-3
    phase1_patience: int   = 10

    # Phase 2 — progressive unfreezing of VGG16 backbone
    phase2_epochs:          int   = 100
    phase2_lr:              float = 1e-4
    phase2_patience:        int   = 15
    phase2_unfreeze_blocks: int   = 2
```

---

## 🏗️ Model Architecture

```
VGG16 Features — FROZEN (block1 → block4_conv3)
  [7,635,264 parameters, requires_grad=False]
          ↓
GlobalAveragePooling2D  →  (B, 512)
          ↓
Linear(512 → 512)  + ReLU    [262,656 params]
          ↓
Linear(512 → 128)  + ReLU    [ 65,664 params]
          ↓
Dropout(p=0.2)
          ↓
Linear(128 → 12)  [Softmax]  [  1,548 params]
─────────────────────────────────────────────
Trainable:  329,868   ✓ matches paper (Fig. 9)
Frozen:   7,635,264
Total:    7,965,132
```

---

## 🔄 Two-Phase Fine-Tuning (`AutoFinetuner`)

This implementation extends the paper with a **progressive unfreezing** strategy:

```
Phase 1 — Head Only
──────────────────────────────────────────────────────
  VGG16 features : fully FROZEN
  Trains only    : 5-layer classifier head
  Optimizer      : RMSprop(lr=1e-3)
  Scheduler      : StepLR(step=20, γ=0.1)
  Epochs         : 50   |  Early stopping patience: 10
  Checkpoint     : checkpoints/phase1_best.pth

Phase 2 — Progressive Unfreezing
──────────────────────────────────────────────────────
  Unfreezes      : last 2 blocks of VGG16 features
  Lower lr       : preserves pretrained weights
  Optimizer      : RMSprop(lr=1e-4)
  Scheduler      : StepLR(step=30, γ=0.1)
  Epochs         : 100  |  Early stopping patience: 15
  Checkpoint     : checkpoints/phase2_best.pth
```

---

## 📊 Data Pipeline

```
CustomDataSet(transform=None)
  └── Scans 12 class subdirectories
  └── Filters images: w < 300px or h < 300px removed  ← paper preprocessing
  └── Builds sorted class→index mapping
          ↓
random_split(80% train / 20% val, seed=42)
          ↓
TransformSubset wrapper
  ├── training_data  →  train_transform  (with augmentation)
  └── eval_data      →  eval_transform   (clean, no augmentation)
          ↓
DataLoader(batch_size=32, shuffle=True/False, pin_memory=True)
```

### Augmentation details

| Split | Transforms applied |
|-------|--------------------|
| **Train** | Resize(256) → RandomResizedCrop(224) → RandomHFlip → RandomVFlip → RandomRotation(±45°) → ColorJitter → ToTensor → Normalize |
| **Val** | Resize(224) → ToTensor → Normalize |

Normalization: `mean=[0.485, 0.456, 0.406]`  `std=[0.229, 0.224, 0.225]`

---

## 🏋️ Trainer Features

The `Trainer` class (`src/trainer.py`) provides:

- **Early stopping** — halts training when val loss stalls for `patience` epochs
- **Best-model restoration** — loads best weights automatically at end of run
- **Auto-checkpointing** — saves `.pth` file every time val loss improves
- **tqdm.auto bars** — renders correctly in terminal and Jupyter/Colab

## 📚 References

1. Jiang, Z. (2019). A Novel Crop Weed Recognition Method Based on Transfer Learning from VGG16 Implemented by Keras. *IOP Conf. Ser.: Mater. Sci. Eng.* 677, 032073. https://doi.org/10.1088/1757-899X/677/3/032073

2. Simonyan, K. & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. *arXiv:1409.1556*

3. Giselsson, T.M. et al. (2017). A Public Image Database for Benchmark of Plant Seedling Classification Algorithms. *arXiv:1711.05458*

4. PyTorch Documentation — `torchvision.models.vgg16`. https://pytorch.org/vision/stable/models/vgg.html

---

## ⚖️ License

| Component | License |
|-----------|---------|
| Paper methodology | CC BY 3.0 — © Zichao Jiang (2019), IOP Publishing |
| Dataset | CC0 Public Domain — Kaggle Plant Seedlings |
| This implementation | Academic use only — MSc AI Engineering coursework, 2024 |
