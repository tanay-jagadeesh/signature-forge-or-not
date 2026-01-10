import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

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
