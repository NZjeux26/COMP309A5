import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Plot training and validation history (loss and accuracy) over epochs
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 4))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')  # Save the plot to a file
    plt.close()

# This function trains a given model using a specified optimiser, loss criterion, and learning rate scheduler.
# It tracks the training and validation losses, as well as accuracies, across a specified number of epochs. 
# After each epoch, it adjusts the learning rate based on the validation loss. The function returns lists of 
# training and validation losses and accuracies to enable further analysis.l
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    #Lists to store the losses and accuracies for each epoch
    train_losses = []
    val_losses = []  # To store validation losses
    train_accuracies = []
    val_accuracies = []  # To store validation accuracies

    for epoch in range(num_epochs):
        # Start timer for this epoch
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        #Iterate over batches in training set
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) #Move to device selected

            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(images)
            loss = criterion(outputs, labels) #compute losses
            loss.backward()
            optimizer.step() #update model
            
            # Track loss and accuracy
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate average training loss and accuracy for the epoch
        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        #Iterate over batches in validation set without gradient calculation
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        #Track validation loss and accuracy
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step(val_loss)  # Scheduler adjusts learning rate based on validation loss
        
        # End timer for this epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # Print progress for each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        print(f"Epoch {epoch+1} took {epoch_duration // 60:.0f} minutes {epoch_duration % 60:.0f} seconds")
        print('-' * 50)

    return train_losses, val_losses, train_accuracies, val_accuracies

# Custom dataset class to load images from directory
class FruitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # Root directory containing subfolders for each class
        self.transform = transform
        self.classes = ['cherry', 'strawberry', 'tomato']  # List of class labels
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}  # Mapping of class names to integer labels
        
        self.images = []  # List to hold file paths of images
        self.labels = []  # Corresponding list of labels for images
        
        # Traverse each class folder, and collect image file paths
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image file types
                    self.images.append(os.path.join(class_dir, img_name))  # Add image file path
                    self.labels.append(self.class_to_idx[class_name])  # Add corresponding label

    def __len__(self):
        return len(self.images)  # Return the total number of images

    def __getitem__(self, idx):
        img_path = self.images[idx]  # Get image path by index
        image = Image.open(img_path).convert('RGB')  # Open image and convert to RGB format
        label = self.labels[idx]  # Get corresponding label

        if self.transform:
            image = self.transform(image)  # Apply any transformations (e.g., resize, normalize)

        return image, label  # Return image and label pair

# This class defines a CNN architecture for classifying images of fruits into a specified number of classes.
# The model is designed to process 112x112 RGB images and includes convolutional, pooling, batch normalisation, 
# dropout, and fully connected layers to extract features and perform classification.

class FruitClassifierCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(FruitClassifierCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First convolutional block (3 x 112 x 112) -> output (32 x 56 x 56)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            # Second convolutional block (32 x 56 x 56) -> output (64 x 28 x 28)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
             
            # Third convolutional block (64 x 28 x 28) -> output (128 x 14 x 14)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
             
            # Fourth convolutional block (128 x 14 x 14) -> output (256 x 7 x 7)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            # Fully connected block 1: input (256 * 7 * 7) -> output (512)
            nn.Linear(256 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Fully connected block 2: 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # Output layer
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Pass through convolutional layers
        x = x.view(x.size(0), -1)  # Flatten the output of the convolutional layers
        x = self.fc_layers(x)  # Pass through fully connected layers
        return x

def main():
    # Hyperparameters
    num_classes = 3  # Number of output classes (cherry, strawberry, tomato)
    num_epochs = 40  # Number of training epochs
    batch_size = 32  # Batch size for training
    learning_rate = 0.001  # Learning rate for optimiser
    val_split = 0.1  # Use 10% of the data for validation

    # GPU Setup
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Define image transformations: resize to 112x112, convert to tensor, and normalise
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalise RGB channels
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

    # Initialise model, loss function, and optimizer
    model = FruitClassifierCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

    # Start timer for the total training time
    total_start_time = time.time()
    
    # Train the model
    print("\nStarting training...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device
    )
    
    # End timer for the total training time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\nTotal training time: {total_duration // 60:.0f} minutes {total_duration % 60:.0f} seconds")

    # Plot training and validation history (loss and accuracy)
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    print("\nTraining history plot saved as 'training_history.png'")

    # Save the trained model and optimizer state
    model_save_path = 'fruit_classifier.pth'
    
    #save the state_dict
    #torch.save(model, model_save_path)
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
    }, model_save_path)
    print(f"\nModel saved to {model_save_path}")

if __name__ == '__main__':
    main()
