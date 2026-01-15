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

            _,predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())
    return np.array(all_predictions), np.array(all_labels), np.array(all_probs)

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("EVALUATION METRICS")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")

    return accuracy, precision, recall, f1
#Visuals
def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])

    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {save_path}")

    plt.close()

    print("\nConfusion Matrix Breakdown:")
    print(f"True Negatives (Real as Real):   {cm[0, 0]}")
    print(f"False Positives (Real as Fake):  {cm[0, 1]}")
    print(f"False Negatives (Fake as Real):  {cm[1, 0]}")
    print(f"True Positives (Fake as Fake):   {cm[1, 1]}")

def plot_roc_curve(y_true, y_probabilities, save_path='roc_curve.png'):
    fpr, tpr, thresholds = roc_curve(y_true, y_probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nROC curve saved to {save_path}")
    print(f"AUC Score: {roc_auc:.4f}")
    plt.close()

    return roc_auc

def visualize_errors(model, test_loader, num_samples=10, save_path='error_analysis.png'):
    model.eval()
    errors = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    errors.append({
                        'image': images[i],
                        'true': labels[i].item(),
                        'pred': predicted[i].item(),
                        'conf': probabilities[i, predicted[i]].item()
                    })

    if len(errors) == 0:
        print("\nNo errors found - perfect accuracy!")
        return

    print(f"\nFound {len(errors)} misclassified images")

    num_to_show = min(num_samples, len(errors))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for idx in range(num_to_show):
        img = errors[idx]['image'].permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5) + 0.5
        img = np.clip(img, 0, 1)

        axes[idx].imshow(img, cmap='gray')
        axes[idx].axis('off')

        true_label = 'Real' if errors[idx]['true'] == 0 else 'Fake'
        pred_label = 'Real' if errors[idx]['pred'] == 0 else 'Fake'
        conf = errors[idx]['conf']

        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {conf:.2f}',
                          fontsize=9, color='red', weight='bold')

    for idx in range(num_to_show, 10):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Error analysis saved to {save_path}")
    plt.close()

def main(model_path='best_model.pth'):
    print("Starting evaluation")
    print(f"Loading model from: {model_path}\n")

    model = load_model(model_path)

    print("Running predictions on test set")
    y_pred, y_true, y_probs = get_predictions(model, test_loader)

    accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)

    plot_confusion_matrix(y_true, y_pred)

    roc_auc = plot_roc_curve(y_true, y_probs)

    visualize_errors(model, test_loader)

    print("Evaluation Results")
    print(f"Final Accuracy: {accuracy*100:.2f}%")
    print(f"AUC Score: {roc_auc:.4f}")
    print("\nGenerated files:")
    print("confusion_matrix.png")
    print("roc_curve.png")
    print("error_analysis.png")

if __name__ == "__main__":
    main(model_path='best_model.pth')