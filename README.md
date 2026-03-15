# Signature Forge or Not

A deep learning project that classifies handwritten signatures as **real (authentic)** or **forged (fake)** using transfer learning with PyTorch. Built as a hands-on project to learn PyTorch fundamentals through a complete ML pipeline — from data loading to model evaluation and inference.

## Tech Stack

- **PyTorch** — deep learning framework for model building and training
- **torchvision** — pre-trained ResNet18 model and image transforms
- **scikit-learn** — evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- **Matplotlib & Seaborn** — training curves, confusion matrix, ROC curve, error analysis visualizations
- **PIL (Pillow)** — image loading and processing
- **NumPy** — numerical operations

## Project Structure

```
signature-forge-or-not/
├── src/
│   ├── dataset.py          # Custom PyTorch Dataset with augmentation pipeline
│   ├── dataloader.py       # Train/val/test DataLoader setup (80/20 split)
│   ├── model.py            # ResNet18 transfer learning, training loop, loss curves
│   ├── evaluate.py         # Metrics, confusion matrix, ROC curve, error analysis
│   └── predict.py          # Single-image inference CLI tool
├── data/
│   ├── train/
│   │   ├── real/           # Authentic training signatures
│   │   └── fake/           # Forged training signatures
│   └── test/
│       ├── real/           # Authentic test signatures
│       └── fake/           # Forged test signatures
├── models/                 # Saved model weights (best_model.pth)
├── results/                # Generated plots and visualizations
├── main.py                 # Entry point
├── requirements.txt
├── .gitignore
└── README.md
```

## How It Works

### Model Architecture

Uses **ResNet18** pre-trained on ImageNet with a transfer learning strategy:

- **Frozen layers**: `conv1`, `bn1`, `layer1`, `layer2`, `layer3` — these retain learned low/mid-level features (edges, textures, patterns)
- **Trainable layers**: `layer4` and a new fully connected head (`fc → 2 classes`) — these are fine-tuned for signature-specific features
- **Output**: 2 classes — `0` (real) and `1` (fake)

### Data Augmentation

Applied during training to improve generalization:

| Transform | Details |
|-----------|---------|
| Resize | 224 × 224 |
| Grayscale | Converted to 3-channel grayscale |
| Random Rotation | ±12° |
| Random Horizontal Flip | 50% probability |
| Random Affine | Translation (10%), scale (0.85–1.15), shear (±10°) |
| Gaussian Noise | σ = 0.1 |
| Normalize | mean=0.5, std=0.5 per channel |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.01 |
| LR Scheduler | StepLR (step=7, γ=0.1) |
| Loss Function | CrossEntropyLoss |
| Epochs | 40 |
| Batch Size | 32 |
| Train/Val Split | 80% / 20% |

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/signature-forge-or-not.git
cd signature-forge-or-not
```

### 2. Install dependencies

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn pillow
```

### 3. Prepare the dataset

Place your signature images in the following structure:

```
data/
├── train/
│   ├── real/    # authentic signatures
│   └── fake/    # forged signatures
└── test/
    ├── real/
    └── fake/
```

Any standard signature verification dataset will work (e.g., CEDAR, BHSig260, ICDAR). Images can be any format supported by PIL (PNG, JPG, BMP, etc.).

### 4. Train the model

```bash
cd src
python model.py
```

This will:
- Train for 40 epochs with validation after each epoch
- Save the best model weights to `best_model.pth`
- Generate `training_curves.png` with loss and accuracy plots

### 5. Evaluate the model

```bash
python evaluate.py
```

Outputs:
- **Metrics**: accuracy, precision, recall, F1-score printed to console
- **confusion_matrix.png** — heatmap with TP/TN/FP/FN breakdown
- **roc_curve.png** — ROC curve with AUC score
- **error_analysis.png** — grid of misclassified samples with true/predicted labels and confidence

### 6. Predict on a single image

```bash
python predict.py path/to/signature.png
```

```
Loading model...
Analyzing signature: path/to/signature.png

Prediction: This signature is REAL with 97.32% confidence
```

## What's NOT in Git

The following are excluded from version control (see [.gitignore](.gitignore)):

- **data/** — signature image dataset (~6,000 images)
- **models/best_model.pth** — trained model weights
- ***.jpg, *.png** — raw image files and generated plots
