import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torchvision.transforms as transforms
from fruit_class import FruitClassifierCNN, FruitDataset, plot_training_history, train_model
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Import the scheduler

def main():
    # Hyperparameters
    num_classes = 3  # Number of output classes (cherry, strawberry, tomato)
    num_epochs = 40  # Number of training epochs
    batch_size = 32  # Batch size for training
    learning_rate = 0.001  # Learning rate for optimizer
    val_split = 0.1  # Use 10% of the data for validation

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

    # Load the full dataset
    train_data_path = os.path.join('..', 'split_data/train_data')  # Root directory for dataset
    full_dataset = FruitDataset(root_dir=train_data_path, transform=transform)

    # Split dataset into training and validation sets
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create DataLoader for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = FruitClassifierCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

    # Train the model
    print("\nStarting training...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device
    )

    # Plot training and validation history (loss and accuracy)
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
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
