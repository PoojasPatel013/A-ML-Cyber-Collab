import pandas as pd
import numpy as np
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import gc
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CombinedCyberThreatDataset(Dataset):
    def __init__(self, data_dir, batch_size=1024):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        self.current_file_idx = 0
        self.current_chunk = None
        self.feature_columns = None
        self.label_encoder = LabelEncoder()
        self._load_next_file()
        
    def _load_next_file(self):
        if self.current_file_idx >= len(self.all_files):
            self.current_file_idx = 0
        current_file = self.all_files[self.current_file_idx]
        print(f"Loading file: {current_file}")
        
        # Read file
        df = pd.read_csv(current_file, low_memory=False)
        
        # Detect target column
        target_col = self._detect_target_column(df)
        if not target_col:
            raise ValueError(f"Could not detect target column in {current_file}")
            
        # Process features
        X, y = self._process_file(df, target_col)
        self.current_chunk = {'X': X, 'y': y}
        self.current_file_idx += 1
        
    def _detect_target_column(self, df):
        patterns = ['Label', 'Prediction', 'Type', 'phishing', 'Attack_type', 'attack']
        for pattern in patterns:
            matching_cols = [col for col in df.columns if pattern.lower() in col.lower()]
            if matching_cols:
                return matching_cols[0]
        return None
        
    def _process_file(self, df, target_col):
        # Save feature columns if not already set
        if self.feature_columns is None:
            self.feature_columns = df.drop(target_col, axis=1).columns.tolist()
            print(f"Feature columns set: {len(self.feature_columns)}")
            
        # Align features with saved columns
        df_aligned = pd.DataFrame(columns=self.feature_columns)
        for col in self.feature_columns:
            if col in df.columns:
                df_aligned[col] = df[col]
            else:
                df_aligned[col] = 0  # Fill missing features with 0
                
        X = df_aligned
        y = df[target_col]
        
        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = pd.factorize(X[col])[0]
            
        # Convert to numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Encode labels
        y = self.label_encoder.fit_transform(y.astype(str))
        
        return X, y
        
    def __len__(self):
        if self.current_chunk is None:
            self._load_next_file()
        return len(self.current_chunk['X'])
        
    def __getitem__(self, idx):
        try:
            if self.current_chunk is None:
                self._load_next_file()
                
            X = self.current_chunk['X'].iloc[idx].values
            y = self.current_chunk['y'].iloc[idx]
            
            # Convert to tensors
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            
            return X, y
            
        except Exception as e:
            print(f"Error processing item: {e}")
            raise

class CombinedCyberThreatCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CombinedCyberThreatCNN, self).__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(2)
        )
        
        # Calculate output size
        self.output_size = self._get_output_size(input_size)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.output_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, num_classes)
        )
    
    def _get_output_size(self, input_size):
        x = torch.randn(1, 1, input_size)
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1).size(1)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_combined_model(data_dir, batch_size=1024):
    try:
        print("\nTraining combined CNN model...")
        
        # Create dataset and dataloader
        dataset = CombinedCyberThreatDataset(data_dir, batch_size=batch_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # Get first batch to determine input size and number of classes
        X_sample, y_sample = next(iter(dataloader))
        input_size = X_sample.shape[2]
        num_classes = len(torch.unique(y_sample))
        
        # Initialize model, loss, and optimizer
        model = CombinedCyberThreatCNN(input_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create save directory
        save_dir = 'models_combined'
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        num_epochs = 50
        best_accuracy = 0
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = model(X.squeeze(0))
                loss = criterion(outputs, y.squeeze(0))
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(1)
                correct += (predicted == y.squeeze(0)).sum().item()
                
                if (i + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {running_loss:.4f}')
            
            train_acc = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file = os.path.join(save_dir, f'combined_cnn_model_{timestamp}.pth')
        torch.save(model.state_dict(), model_file)
        print(f"Combined CNN model saved to: {model_file}")
        
        # Save label encoder
        le_file = os.path.join(save_dir, f'label_encoder_{timestamp}.pkl')
        import joblib
        joblib.dump(dataset.label_encoder, le_file)
        print(f"Label encoder saved to: {le_file}")
        
        return model
        
    except Exception as e:
        print(f"Error training combined model: {e}")
        raise

def main():
    try:
        # Create necessary directories
        os.makedirs('models_combined', exist_ok=True)
        
        # Train combined model
        data_dir = 'data'
        model = train_combined_model(data_dir)
        
        if model:
            print("\nSuccessfully trained combined CNN model")
            
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
