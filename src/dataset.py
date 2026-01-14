import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


# Transform pipeline with data augmentation
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=(-90, 90), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-10, 10)),
    transforms.ToTensor(),
    transforms.GaussianNoise(mean=0.0, sigma=0.1),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
