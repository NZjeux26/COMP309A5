import torch
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from fruit_class import FruitClassifierCNN, FruitDataset, evaluate_model

def main():
    # Hyperparameters
    batch_size = 32  # Batch size for testing

    # GPU Setup
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Define image transformations (same as during training) ## This isn't set to the same as training, need to check
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
    ])

    # Load the separate test dataset
    test_data_path = os.path.join('..', 'split_data/test_data')  # Path to the test dataset
    test_dataset = FruitDataset(root_dir=test_data_path, transform=transform)

    # Create DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load the saved model
    model = FruitClassifierCNN(num_classes=3)
    checkpoint = torch.load('fruit_classifier.pth', map_location=device)  # Load the model checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print("\nEvaluating model on test set...")
    evaluate_model(model, test_loader, device, test_dataset.classes)

if __name__ == '__main__':
    main()
