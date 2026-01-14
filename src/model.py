import torch
import torch.nn as nn
from torchvision import models
from dataloader import train_loader, val_loader


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

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

num_epochs = 20

model.train()

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    #Training Phase
    train_loss = 0.0
    for images, labels in train_loader:
        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    print(f"Training loss: {avg_train_loss}")

    #Validation Phase 
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
        
    val_accuracy = 100 * correct / total
    
    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
    print(f"Epoch {epoch+1} complete!\n")
    
    model.train()  
    scheduler.step()