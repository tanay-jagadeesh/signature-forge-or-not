# signature-forge-or-not

Project to learn PyTorch through building a signature verification system (real vs forged).

## Tech Stack

- **PyTorch**: Deep learning framework for building and training the signature verification model
- **Python**: Core programming language for ML development
- **torchvision**: Image processing and data augmentation
- **NumPy**: Numerical computing and array operations
- **Matplotlib**: Visualization of training metrics and results

## Project Structure

```
signature-forgery-detection/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── results/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── sample_predictions.png
└── main.py
```

## What's NOT in Git

The following are excluded from version control (see [.gitignore](.gitignore)):

- **data/** - Dataset (~6,000 images) - Download your own dataset
- **models/best_model.pth** - Trained model weights (50-200 MB)
- **\*.jpg, \*.png** - Raw image files
