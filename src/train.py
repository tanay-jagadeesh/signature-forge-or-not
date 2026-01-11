from dataset import SignatureDataset
from torch.utils.data import DataLoader, random_split
import torch

full_train_dataset = SignatureDataset("data/train")

train_split = int(0.8 * len(full_train_dataset))
val_split = int(0.2 * len(full_train_dataset))

train_dataset, val_dataset = random_split(full_train_dataset, [train_split, val_split])

#Create DataLoaders
train_loader = DataLoader(train_dataset, shuffle = True, batch_size = 32)
val_loader = DataLoader(val_dataset, shuffle = False, batch_size = 32)

test_dataset = SignatureDataset("data/test")
test_loader = DataLoader(test_dataset, shuffle = False, batch_size = 32)