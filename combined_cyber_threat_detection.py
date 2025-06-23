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
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        self.X = None
        self.y = None
        self.num_classes = None
        self._load_and_combine_data()
        
    def get_num_classes(self):
        """Get the number of unique classes in the dataset"""
        if self.y is not None:
            self.num_classes = len(np.unique(self.y))
            return self.num_classes
        return None
        
    def _load_and_combine_data(self):
        """Load and combine data from files efficiently"""
        print("Loading and combining data from all files...")
        
        # First pass: Collect all unique features
        all_features = []
        chunk_size = 100000  # Process in chunks to handle large files
        
        # Process each file and combine immediately
        for current_file in self.all_files:
            print(f"\nProcessing file: {current_file}")
            try:
                # Process file in chunks
                first_chunk = True
                for chunk in pd.read_csv(current_file, chunksize=chunk_size):
                    if first_chunk:
                        # Detect target column
                        target_col = None
                        patterns = [
                            'label', 'prediction', 'type', 'phishing', 'attack_type', 'attack',
                            'class', 'result', 'target', 'y', 'outcome', 'category'
                        ]
                        
                        # Convert all column names to lowercase
                        lower_cols = [col.lower() for col in chunk.columns]
                        
                        # Try exact matches first
                        for pattern in patterns:
                            if pattern in lower_cols:
                                target_col = chunk.columns[lower_cols.index(pattern)]
                                break
                        
                        # If not found, try partial matches
                        if not target_col:
                            for col, lower_col in zip(chunk.columns, lower_cols):
                                if any(pattern in lower_col for pattern in patterns):
                                    target_col = col
                                    break
                        
                        # If still not found, try to find the column with the least unique values
                        if not target_col:
                            min_unique = float('inf')
                            for col in chunk.columns:
                                try:
                                    unique_count = chunk[col].nunique()
                                    if unique_count < min_unique:
                                        min_unique = unique_count
                                        target_col = col
                                except Exception as e:
                                    print(f"Warning: Error checking unique values for {col}: {e}")
                        
                        if not target_col:
                            print(f"Warning: No target column found in {current_file}")
                            break
                        
                        print(f"Found target column: {target_col} in {current_file}")
                        print(f"Column values: {chunk[target_col].unique()[:5]}...")
                        
                        # Add features from this file
                        if not all_features:  # Initialize with first file's features
                            all_features = chunk.columns.tolist()
                        else:
                            # Add new features from this file
                            new_features = set(chunk.columns) - set(all_features)
                            all_features.extend(sorted(new_features))
                        
                        first_chunk = False
                        
                        # Print feature stats
                        print(f"\nFeatures found in {current_file}:")
                        print(f"Total features: {len(chunk.columns)}")
                        print(f"First 5 features: {chunk.columns[:5].tolist()}")
                        print(f"Last 5 features: {chunk.columns[-5:].tolist()}")
                        
                    # Create aligned DataFrame with all features
                    df_aligned = pd.DataFrame(columns=all_features)
                    
                    # Fill with data where available, 0 where not
                    for col in all_features:
                        if col in chunk.columns:
                            df_aligned[col] = chunk[col]
                        else:
                            df_aligned[col] = 0
                    
                    # Process features
                    X = df_aligned.drop(target_col, axis=1)
                    y = chunk[target_col]
                    
                    # Handle categorical features
                    categorical_cols = X.select_dtypes(include=['object']).columns
                    for col in categorical_cols:
                        X[col] = pd.factorize(X[col])[0]
                    
                    # Convert to numeric and use float32
                    X = X.astype('float32').fillna(0)
                    
                    # Convert labels using LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    
                    # Validate feature shape
                    if X.shape[1] != len(all_features) - 1:
                        raise ValueError(f"Feature shape mismatch in {current_file}: Expected {len(all_features) - 1}, got {X.shape[1]}")
                    
                    # Validate label values
                    if y.min() < 0:  # Labels should always be non-negative
                        raise ValueError(f"Negative label values found: min={y.min()}")
                    
                    # Print label statistics
                    num_classes = len(np.unique(y))
                    print(f"Number of unique classes: {num_classes}")
                    print(f"Label distribution: {np.bincount(y)}")
                    
                    if num_classes > 1000:  # If too many classes, raise warning
                        print(f"Warning: Large number of classes found ({num_classes}). This might indicate a problem with the target column.")
                    
                    # Convert to numpy arrays immediately
                    X_array = X.to_numpy(dtype=np.float32)
                    y_array = np.array(y, dtype=np.int32)
                    
                    # Print label statistics
                    print(f"Label value range: min={y.min()}, max={y.max()}")
                    print(f"Unique labels: {len(np.unique(y))}")
                    
                    # If this is the first chunk, initialize arrays
                    if self.X is None:
                        self.X = X_array
                        self.y = y_array
                    else:
                        # Append to existing arrays
                        self.X = np.vstack([self.X, X_array])
                        self.y = np.concatenate([self.y, y_array])
                        
                    # Print array shapes for debugging
                    print(f"X array shape: {X_array.shape}")
                    print(f"y array shape: {y_array.shape}")
                    print(f"Combined X shape: {self.X.shape if self.X is not None else 'None'}")
                    print(f"Combined y shape: {self.y.shape if self.y is not None else 'None'}")
                    
                    print(f"Processed chunk with {len(X_array)} samples")
                    print(f"Current memory usage: {X_array.nbytes / 1e6:.2f} MB")
                    print(f"Total samples so far: {len(self.X)}")
                    
                    # Clean up
                    del X, y, X_array, y_array, df_aligned
                    gc.collect()
                
            except Exception as e:
                print(f"Warning: Error processing {current_file}: {e}")
                continue
        
        if self.X is not None:
            print(f"\nCombined dataset stats:")
            print(f"Total samples: {len(self.X)}")
            print(f"Feature shape: {self.X.shape}")
            print(f"Label shape: {self.y.shape}")
            print(f"Unique labels: {np.unique(self.y)}")
            print(f"Total memory usage: {self.X.nbytes / 1e6:.2f} MB")
        else:
            raise ValueError("No valid data loaded from files")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        try:
            X = torch.tensor(self.X[idx], dtype=torch.float32)
            y = torch.tensor(self.y[idx], dtype=torch.long)
            
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

def train_combined_model(data_dir):
    try:
        print("\nTraining combined CNN model...")
        
        # Create dataset and dataloader
        dataset = CombinedCyberThreatDataset(data_dir)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Get first batch to determine input size and number of classes
        X_sample, y_sample = next(iter(dataloader))
        input_size = X_sample.shape[1]  
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
        
        print(f"\nStarting training with input size: {input_size}, num_classes: {num_classes}")
        print(f"Total samples: {len(dataset)}")
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
                if (i + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {running_loss:.4f}')
            
            train_acc = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%')
            
            # Save best model
            if train_acc > best_accuracy:
                best_accuracy = train_acc
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_file = os.path.join(save_dir, f'combined_cnn_model_{timestamp}.pth')
                torch.save(model.state_dict(), model_file)
                print(f"Saved best model with accuracy: {train_acc:.2f}%")
        
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
        print(f"\nStarting training with data from: {data_dir}")
        model = train_combined_model(data_dir)
        
        if model:
            print("\nSuccessfully trained combined CNN model")
            
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
