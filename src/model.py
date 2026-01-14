import torch
import torch.nn as nn
from torchvision import models
from dataset import train_loader


model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

# Freeze early layers/residual blocks
for param in model.conv1.parameters():
    param.requires_grad = False
    
for param in model.bn1.parameters():
    param.requires_grad = False

for param in model.layer1.parameters():
    param.requires_grad = False
    
for param in model.layer2.parameters():
    param.requires_grad = False
    
for param in model.layer3.parameters():
    param.requires_grad = False
    
for param in model.layer4.parameters():
    param.requires_grad = False

# Get the number of input features for the original final layer
num_ftrs = model.fc.in_features

# Replace the final layer with a new one
model.fc = nn.Linear(num_ftrs, 2)

#built-in loss function
criterion = nn.CrossEntropyLoss()

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

num_epochs = 10

for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
