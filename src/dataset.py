import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os


# Create transform pipeline using Compose
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class SignatureDataset(Dataset):
    def __init__(self, folder_path):
        self.image_paths = []
        self.labels = []

        real_folder = os.path.join(folder_path, "real")

        for filename in os.listdir(real_folder):
            full_path = os.path.join(real_folder, filename)

            self.image_paths.append(full_path)
            self.labels.append(0) # 0 is real; 1 is fake
        
        fake_folder = os.path.join(folder_path, "fake")

        for filename in os.listdir(fake_folder):
            full_path = os.path.join(fake_folder, filename)

            self.image_paths.append(full_path)
            self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        img = Image.open(img_path)

        # Apply all transformations at once using pipeline
        img = transform_pipeline(img)

        return img, label
