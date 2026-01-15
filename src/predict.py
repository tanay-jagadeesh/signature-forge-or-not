import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

def load_model(model_path='best_model.pth'):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_signature(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item() * 100

    label = 'REAL' if predicted_class == 0 else 'FAKE'

    return label, confidence

def main(image_path):
    print(f"Loading model...")
    model = load_model('best_model.pth')

    print(f"Analyzing signature: {image_path}")
    label, confidence = predict_signature(model, image_path)

    print(f"\nPrediction: This signature is {label} with {confidence:.2f}% confidence")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_signature_image>")
        print("Example: python predict.py data/test/real/signature_001.png")
    else:
        main(sys.argv[1])
