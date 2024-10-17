import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

class FruitClassifierCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(FruitClassifierCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First convolutional block: (3 x 299 x 299) -> (32 x 149 x 149)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 299x299 -> 149x149
            nn.Dropout2d(0.1),
            
            # Second convolutional block: (32 x 149 x 149) -> (64 x 74 x 74)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 149x149 -> 74x74
            nn.Dropout2d(0.1),
             
            # Third convolutional block: (64 x 74 x 74) -> (128 x 37 x 37)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 74x74 -> 37x37
            nn.Dropout2d(0.2),
             
            # Fourth convolutional block: (128 x 37 x 37) -> (256 x 18 x 18)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 37x37 -> 18x18
            nn.Dropout2d(0.2),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            # The input to the fully connected layer is now 256 * 18 * 18
            nn.Linear(256 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
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
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    train_losses = []
    val_losses = []  # To store validation losses
    train_accuracies = []
    val_accuracies = []  # To store validation accuracies

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Call the learning rate scheduler
        scheduler.step(val_loss)  # Scheduler adjusts learning rate based on validation loss

        # Print progress for each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
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