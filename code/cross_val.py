import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

from sklearn.model_selection import KFold
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from tqdm import tqdm
from PIL import Image

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
    
class HyperparameterOptimizer:
    def __init__(self, dataset, device, n_splits=5):
        self.dataset = dataset
        self.device = device
        self.n_splits = n_splits
        
        # Define hyperparameter search space
        self.param_grid = {
            'hidden_size': [128, 256, 512],
            'batch_size': [16, 32, 64],
            'learning_rate': [0.1, 0.01, 0.001],
            'dropout_rate': [0.1, 0.2, 0.3],
            'num_layers': [2, 3, 4],
            'optimizer': ['adam', 'sgd'],
            'weight_decay': [0.0, 0.0001, 0.001]
        }
        
        self.results = []
        
    def create_model(self, params):
        class DynamicMLP(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes, num_layers, dropout_rate):
                super(DynamicMLP, self).__init__()
                self.flatten = nn.Flatten()
                
                layers = []
                # Input layer
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                
                # Hidden layers
                for _ in range(num_layers - 1):
                    layers.append(nn.Linear(hidden_size, hidden_size))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))
                
                # Output layer
                layers.append(nn.Linear(hidden_size, num_classes))
                
                self.layers = nn.Sequential(*layers)
            
            def forward(self, x):
                x = self.flatten(x)
                return self.layers(x)
        
        return DynamicMLP(
            input_size=28 * 28 * 3,
            hidden_size=params['hidden_size'],
            num_classes=3,
            num_layers=params['num_layers'],
            dropout_rate=params['dropout_rate']
        ).to(self.device)
    
    def get_optimizer(self, model, params):
        if params['optimizer'] == 'adam':
            return optim.Adam(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        else:
            return optim.SGD(
                model.parameters(),
                lr=params['learning_rate'],
                momentum=0.9,
                weight_decay=params['weight_decay']
            )
    
    def train_evaluate_fold(self, model, train_loader, val_loader, optimizer, criterion, epochs=10):
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return best_val_loss, val_accuracy
    
    def run_optimization(self):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        criterion = nn.CrossEntropyLoss()
        
        # Generate all parameter combinations
        param_combinations = [dict(zip(self.param_grid.keys(), v)) 
                            for v in itertools.product(*self.param_grid.values())]
        
        print(f"Total parameter combinations to test: {len(param_combinations)}")
        
        for params in tqdm(param_combinations, desc="Testing parameter combinations"):
            fold_scores = []
            fold_accuracies = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
                # Create data loaders for this fold
                train_sampler = SubsetRandomSampler(train_idx)
                val_sampler = SubsetRandomSampler(val_idx)
                
                train_loader = DataLoader(
                    self.dataset,
                    batch_size=params['batch_size'],
                    sampler=train_sampler,
                    num_workers=4,
                    pin_memory=True
                )
                
                val_loader = DataLoader(
                    self.dataset,
                    batch_size=params['batch_size'],
                    sampler=val_sampler,
                    num_workers=4,
                    pin_memory=True
                )
                
                # Create and train model
                model = self.create_model(params)
                optimizer = self.get_optimizer(model, params)
                
                val_loss, val_accuracy = self.train_evaluate_fold(
                    model, train_loader, val_loader, optimizer, criterion
                )
                
                fold_scores.append(val_loss)
                fold_accuracies.append(val_accuracy)
            
            # Average scores across folds
            mean_score = np.mean(fold_scores)
            mean_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)
            
            # Store results
            result = {
                **params,
                'mean_val_loss': mean_score,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy
            }
            self.results.append(result)
            
            # Save intermediate results
            self.save_results()
    
    def save_results(self):
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Create results directory if it doesn't exist
        os.makedirs('optimization_results', exist_ok=True)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df.to_csv(f'optimization_results/hyperparameter_results_{timestamp}.csv', index=False)
        
        # Save best parameters
        best_idx = df['mean_accuracy'].idxmax()
        best_params = df.iloc[best_idx].to_dict()
        
        with open(f'optimization_results/best_params_{timestamp}.json', 'w') as f:
            json.dump(best_params, f, indent=4)
        
        # Print current best parameters
        print("\nCurrent Best Parameters:")
        print(f"Mean Accuracy: {best_params['mean_accuracy']:.2f}% (±{best_params['std_accuracy']:.2f})")
        for param, value in best_params.items():
            if param not in ['mean_val_loss', 'mean_accuracy', 'std_accuracy']:
                print(f"{param}: {value}")

def main():
    # GPU Setup
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load your dataset
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_data_path = os.path.join('..', 'train_data')
    dataset = FruitDataset(root_dir=train_data_path, transform=transform)
    
    # Create and run optimizer
    optimizer = HyperparameterOptimizer(dataset, device)
    optimizer.run_optimization()

if __name__ == '__main__':
    main()