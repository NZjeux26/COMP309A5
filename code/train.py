import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torchvision.transforms as transforms
from fruit_class import FruitClassifierCNN, FruitDataset, plot_training_history, train_model

def main():
    # Hyperparameters
    num_classes = 3  # Number of output classes (cherry, strawberry, tomato)
    num_epochs = 40  # Number of training epochs
    batch_size = 32  # Batch size for training
    learning_rate = 0.001  # Learning rate for optimizer

    # GPU Setup
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Define image transformations: resize to 299x299, convert to tensor, and normalize
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
    ])

    # Load the full training dataset (no splitting)
    train_data_path = os.path.join('..', 'split_data/train_data')  # Root directory for training dataset
    train_dataset = FruitDataset(root_dir=train_data_path, transform=transform)

    # Create DataLoader for the entire training dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = FruitClassifierCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print("\nStarting training...")
    train_losses, train_accuracies = train_model(
        model, train_loader, criterion, optimizer, num_epochs, device
    )

    # Plot training history (loss and accuracy)
    plot_training_history(train_losses, [], train_accuracies, [])
    print("\nTraining history plot saved as 'training_history.png'")

    # Save the trained model and optimizer state
    model_save_path = 'fruit_classifier.pth'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
    }, model_save_path)
    print(f"\nModel saved to {model_save_path}")

if __name__ == '__main__':
    main()
