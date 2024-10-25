import torch
from train import FruitClassifierCNN

model = torch.load('fruit_classifier.pth')
vars(model)
# Iterate through the model's state_dict and print the key-value pairs
for key, value in model.state_dict().items():
    print(f"Key: {key}")
    print(f"Shape: {value.shape}")
    print(f"Type: {value.dtype}")
    print()