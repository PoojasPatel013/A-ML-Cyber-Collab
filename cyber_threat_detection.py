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
import json
import warnings
warnings.filterwarnings('ignore')

def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def detect_target_column(df):
    """Detect target column from common names"""
    possible_targets = ['Label', 'Prediction', 'Attack_type', 'attack', 'Type', 'phishing']
    for col in possible_targets:
        if col in df.columns:
            return col
    return None

def load_and_process_data(filename):
    """Load and preprocess data with flexible handling"""
    try:
        print(f"\nProcessing file: {filename}")
        df = pd.read_csv(filename, low_memory=False)
        
        # Detect target column
        target_col = detect_target_column(df)
        if not target_col:
            print(f"Warning: Could not detect target column in {filename}")
            return None, None
            
        # Get features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Convert to numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Convert to PyTorch tensors
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.long)
        
        print(f"Processed features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None, None

class CyberThreatModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CyberThreatModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
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
        return self.model(x)

def train_model(X, y, filename):
    """Train the PyTorch model with detailed logging"""
    try:
        print("\nTraining model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Create dataset and dataloader
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        # Initialize model, loss, and optimizer
        model = CyberThreatModel(X.shape[1], len(torch.unique(y)))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create save directory
        save_dir = 'models'
        create_directory(save_dir)
        
        # Training loop
        num_epochs = 50
        best_accuracy = 0
        
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
            report_file = os.path.join(save_dir, f'report_{os.path.basename(filename)}.json')
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=4)
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file = os.path.join(save_dir, f'pytorch_model_{os.path.basename(filename)}_{timestamp}.pth')
        torch.save(model.state_dict(), model_file)
        print(f"Model saved to: {model_file}")
        
        return model, test_acc
        
    except Exception as e:
        print(f"Error training model for {filename}: {e}")
        return None, 0

def plot_metrics(filename, history):
    """Plot training metrics"""
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss for {filename}')
    plt.legend()
    plt.savefig(f'metrics_{filename}.png')
    plt.close()

def main():
    try:
        # Create necessary directories
        create_directory('models')
        create_directory('metrics')
        
        # Find all CSV files
        data_dir = 'data'
        all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        if not all_files:
            print("No CSV files found in the data directory")
            return
            
        print(f'\nFound {len(all_files)} CSV files:')
        for filename in all_files:
            print(os.path.basename(filename))
        
        # Process each file
        results = []
        for filename in all_files:
            print("\n" + "="*80)
            print(f"Processing {filename}")
            print("="*80)
            
            # Load and process data
            X, y = load_and_process_data(filename)
            if X is None or y is None:
                continue
                
            # Train model
            model, test_acc = train_model(X, y, filename)
            
            if model:
                results.append({
                    'filename': filename,
                    'test_accuracy': test_acc
                })
            
            print("-"*80)
            
            # Clean up memory
            del X, y, model
            gc.collect()
            
        # Save overall results
        if results:
            with open('training_results.json', 'w') as f:
                json.dump(results, f, indent=4)
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
