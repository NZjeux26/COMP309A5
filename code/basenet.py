import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Check if CUDA is available and print GPU details if present
if torch.cuda.is_available():
    print("CUDA available:", torch.cuda.is_available())
    print("GPU Device:", torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()  # Clear any cached data to free up memory
else:
    print("CUDA not available.")

# Check if MPS (Metal Performance Shaders) is available (for Mac with Apple Silicon)
if torch.backends.mps.is_available():
    print("MPS (Metal Performance Shaders) available.")
    # You can set the default device to MPS if available
    device = torch.device("mps")
else:
    print("MPS not available.")

# Function to plot and save confusion matrix
def plot_confusion_matrix(true_labels, predictions, classes):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

# Define a simple feedforward neural network for fruit classification
class FruitClassifier(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=3):
        super(FruitClassifier, self).__init__()
        # Define layers: Flatten the input, followed by Linear -> ReLU -> Dropout -> Linear layers
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input (from 3D to 1D)
        return self.layers(x)  # Pass through the layers

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

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []  # To store training loss for each epoch
    val_losses = []  # To store validation loss for each epoch
    train_accuracies = []  # To store training accuracy for each epoch
    val_accuracies = []  # To store validation accuracy for each epoch
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set model to training mode
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the GPU if available
            
            optimizer.zero_grad()  # Clear the previous gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass to compute gradients
            optimizer.step()  # Update model parameters
            
            train_loss += loss.item()  # Accumulate training loss
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class
            train_total += labels.size(0)  # Total number of samples
            train_correct += (predicted == labels).sum().item()  # Count correct predictions
        
        train_loss = train_loss / len(train_loader)  # Average training loss
        train_accuracy = 100 * train_correct / train_total  # Training accuracy as a percentage
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase (no gradient computation)
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # Disable gradient calculations
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)  # Average validation loss
        val_accuracy = 100 * val_correct / val_total  # Validation accuracy as a percentage
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Print progress for each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        print('-' * 50)
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Evaluate the trained model on test data and generate performance metrics
def evaluate_model(model, test_loader, device, classes):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_labels = []
    
    # No gradient calculation required for evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())  # Move predictions to CPU
            all_labels.extend(labels.cpu().numpy())  # Move labels to CPU
    
    # Calculate overall accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    # Calculate metrics for each class
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
    
    # Plot the confusion matrix
    plot_confusion_matrix(all_labels, all_predictions, classes)
    
    # Print out the overall metrics
    print("\nOverall Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Print per-class metrics
    print("\nPer-class Metrics:")
    for i, class_name in enumerate(classes):
        print(f"\n{class_name}:")
        print(f"Precision: {per_class_precision[i]:.4f}")
        print(f"Recall: {per_class_recall[i]:.4f}")
        print(f"F1-score: {per_class_f1[i]:.4f}")

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

def main():
    # Hyperparameters
    input_size = 112 * 112 * 3  # Input size for 28x28 RGB images
    hidden_size = 512  # Number of neurons in the hidden layer
    num_classes = 3  # Number of output classes (cherry, strawberry, tomato)
    num_epochs = 25  # Number of training epochs
    batch_size = 32  # Batch size for training and evaluation
    learning_rate = 0.001  # Learning rate for optimizer
    train_split = 0.7  # Percentage of data to use for training
    val_split = 0.15  # Percentage of data to use for validation
    # Remaining 15% will be used for testing
    
     # GPU Setup
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # Explicitly use the first GPU
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Print PyTorch CUDA version for verification
    #print(f"PyTorch CUDA version: {torch.version.cuda}")

    # Define image transformations: resize to 28x28, convert to tensor, and normalize
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
    ])

    # Load the dataset
    train_data_path = os.path.join('..', 'train_data')  # Root directory for dataset
    full_dataset = FruitDataset(root_dir=train_data_path, transform=transform)
    
    # Calculate split sizes for training, validation, and testing
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split the dataset into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Set seed for reproducibility
    )
    
    # Create DataLoader for efficient data loading
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Shuffle data for better training
        num_workers=4,  # Use multiple workers for faster data loading
        pin_memory=True,  # Optimize data transfer to GPU
        persistent_workers=True  # Keep workers alive across epochs
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Print dataset sizes for training, validation, and testing
    print(f"\nDataset splits:")
    print(f"Training: {len(train_dataset)} images")
    print(f"Validation: {len(val_dataset)} images")
    print(f"Test: {len(test_dataset)} images")

    # Initialize the model, move it to the GPU (if available)
    model = FruitClassifier(input_size, hidden_size, num_classes)
    model = model.to(device)  # Move model to GPU or CPU as needed
    
    # Define loss function (CrossEntropy) and optimizer (Adam)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\nStarting training...")
    # Train the model and capture loss/accuracy history
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )
    
    # Plot training history (loss and accuracy)
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    print("\nTraining history plot saved as 'training_history.png'")

    # Evaluate the trained model on the test set
    print("\nEvaluating model on test set:")
    evaluate_model(model, test_loader, device, full_dataset.classes)
    print("\nConfusion matrix saved as 'confusion_matrix.png'")

    # # Save the trained model and optimizer state as a checkpoint
    # model_save_path = 'fruit_classifier.pth'
    # torch.save({
    #     'epoch': num_epochs,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'train_loss': train_losses[-1],
    #     'val_loss': val_losses[-1],
    # }, model_save_path)
    # print(f"\nModel checkpoint saved to {model_save_path}")

if __name__ == '__main__':
    main()
