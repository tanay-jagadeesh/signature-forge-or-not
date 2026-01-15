import torch
import torch.nn as nn
from torchvision import models
from dataloader import test_loader
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def load_model(model_path = 'best_model.pth'):
    weights = None
    model = models.resnet18(weights = None)

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, 2)

    model.load_state_dict(torch.load(model_path))

    model.eval()

    return model

def get_predictions(model, test_loader):
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)

            probabilities = torch.softmax(outputs, dim = 1)

            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())
    return np.array(all_predictions), np.array(all_labels), np.array(all_probs)
