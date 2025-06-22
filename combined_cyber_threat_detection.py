import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CombinedDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def detect_target_column(df):
    """Detect target column from common names"""
    possible_targets = ['Label', 'Prediction', 'Attack_type', 'attack', 'Type', 'phishing']
    for col in possible_targets:
        if col in df.columns:
            return col
    return None

def load_and_combine_data():
    """Load and combine all datasets"""
    try:
        data_dir = 'data'
        all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        if not all_files:
            raise ValueError("No CSV files found in the data directory")
            
        print(f'\nFound {len(all_files)} CSV files:')
        for filename in all_files:
            print(os.path.basename(filename))
            
        all_data = []
        all_targets = []
        
        # Process each file
        for filename in all_files:
            print(f"\nProcessing file: {filename}")
            df = pd.read_csv(filename, low_memory=False)
            
            # Detect target column
            target_col = detect_target_column(df)
            if not target_col:
                print(f"Warning: Could not detect target column in {filename}")
                continue
                
            # Get features and target
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            
            # Convert to numeric
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Add file identifier as a feature
            file_id = os.path.basename(filename).split('.')[0]
            X['file_id'] = file_id
            
            all_data.append(X)
            all_targets.append(y)
            
            print(f"Added {len(X)} samples from {filename}")
            
        # Combine all datasets
        combined_X = pd.concat(all_data, ignore_index=True)
        combined_y = pd.concat(all_targets, ignore_index=True)
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(combined_X.values, dtype=torch.float32)
        y_tensor = torch.tensor(combined_y.values, dtype=torch.long)
        
        print(f"\nCombined dataset shape:")
        print(f"Features: {X_tensor.shape}")
        print(f"Targets: {y_tensor.shape}")
        
        return X_tensor, y_tensor
        
    except Exception as e:
        print(f"Error combining data: {e}")
        return None, None

class CombinedCyberThreatModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CombinedCyberThreatModel, self).__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

def train_combined_model(X, y):
    """Train the combined model"""
    try:
        print("\nTraining combined model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Create dataset and dataloader
        train_dataset = CombinedDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        # Initialize model, loss, and optimizer
        model = CombinedCyberThreatModel(X.shape[1], len(torch.unique(y)))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create save directory
        save_dir = 'models_combined'
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        num_epochs = 50
        best_accuracy = 0
        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            history['train_loss'].append(running_loss)
            history['train_acc'].append(train_acc)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            test_acc = 100 * (predicted == y_test).sum().item() / y_test.size(0)
            print(f'\nTest Accuracy: {test_acc:.2f}%')
            
            # Save classification report
            report = classification_report(y_test.numpy(), predicted.numpy(), output_dict=True)
            report_file = os.path.join(save_dir, 'combined_report.json')
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=4)
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file = os.path.join(save_dir, f'combined_pytorch_model_{timestamp}.pth')
        torch.save(model.state_dict(), model_file)
        print(f"Combined model saved to: {model_file}")
        
        # Plot training history
        plt.figure(figsize=(12, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'training_loss.png'))
        plt.close()
        
        return model, test_acc
        
    except Exception as e:
        print(f"Error training combined model: {e}")
        return None, 0

def main():
    try:
        # Load and combine all data
        X, y = load_and_combine_data()
        if X is None or y is None:
            print("Failed to load and combine data")
            return
            
        # Train combined model
        model, test_acc = train_combined_model(X, y)
        if model:
            print(f"\nCombined model training complete. Test Accuracy: {test_acc:.2f}%")
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
