import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import os


#Normalize (standardizing pixel values)
normalize_transform = transforms.Normalize(mean = [0.5], std = [0.5])

#Transform image into tensor
to_tensor = transforms.ToTensor()

# Create a transform that resizes images to 224x224
resize_transform = T.Resize((224, 224))

#Create a transform to make image grayscale
transform = transforms.Grayscale()

# Process train/real folder
filenames = os.listdir("data/train/real")
folder = "data/train/real"

for filename in filenames:
    full_path = os.path.join(folder, filename)
    print(full_path)

    img_loc = Image.open(full_path)
    resize_img = resize_transform(img_loc)
    transform_img = transform(resize_img)
    tensor_img = to_tensor(transform_img)
    normalized_img = normalize_transform(tensor_img)

# Process train/fake folder
filenames = os.listdir("data/train/fake")
folder = "data/train/fake"

for filename in filenames:
    full_path = os.path.join(folder, filename)
    print(full_path)

    img_loc = Image.open(full_path)
    resize_img = resize_transform(img_loc)
    transform_img = transform(resize_img)
    tensor_img = to_tensor(transform_img)
    normalized_img = normalize_transform(tensor_img)

# Process test/real folder
filenames = os.listdir("data/test/real")
folder = "data/test/real"

for filename in filenames:
    full_path = os.path.join(folder, filename)
    print(full_path)

    img_loc = Image.open(full_path)
    resize_img = resize_transform(img_loc)
    transform_img = transform(resize_img)
    tensor_img = to_tensor(transform_img)
    normalized_img = normalize_transform(tensor_img)

# Process test/fake folder
filenames = os.listdir("data/test/fake")
folder = "data/test/fake"

for filename in filenames:
    full_path = os.path.join(folder, filename)
    print(full_path)

    img_loc = Image.open(full_path)
    resize_img = resize_transform(img_loc)
    transform_img = transform(resize_img)
    tensor_img = to_tensor(transform_img)
    normalized_img = normalize_transform(tensor_img)

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

        # Apply transformations
        img = resize_transform(img)
        img = transform(img)
        img = to_tensor(img)
        img = normalize_transform(img)

        return img, label
