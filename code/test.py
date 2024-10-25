import torch
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from train import FruitClassifierCNN, FruitDataset

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

    # Define image transformations (same as during training)
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
    ])

    # Load the separate test dataset
    test_data_path = os.path.join('..', 'test_data')  # Path to the test dataset
    test_dataset = FruitDataset(root_dir=test_data_path, transform=transform)

    # Create DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = FruitClassifierCNN(num_classes=3) # here is the issue
    checkpoint = torch.load('fruit_classifier.pth', map_location=device)  # Load the model checkpoint
    model.load_state_dict(checkpoint['model_state_dict']) # Load the model
    model = model.to(device)

    print("\nEvaluating model on test set...")
    evaluate_model(model, test_loader, device, test_dataset.classes)

if __name__ == '__main__':
    main()
