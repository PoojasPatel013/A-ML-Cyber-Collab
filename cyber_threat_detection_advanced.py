import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from datetime import datetime

def load_and_process_data(filename):
    """Load and preprocess data"""
    try:
        print(f"\nProcessing file: {filename}")
        df = pd.read_csv(filename)
        
        # Handle different datasets
        if 'four.csv' in filename:
            X = df.drop(['SHA256', 'Type'], axis=1)
            y = df['Type']
        elif 'one.csv' in filename:
            X = df.drop(['phishing'], axis=1)
            y = df['phishing']
        elif 'two.csv' in filename:
            X = df.drop(['Attack_type'], axis=1)
            y = df['Attack_type']
        elif 'three.csv' in filename:
            X = df.drop(['attack'], axis=1)
            y = df['attack']
        elif 'cyberattack.csv' in filename:
            X = df.drop(['Label'], axis=1)
            y = df['Label']
        else:
            print(f"Unknown file type: {filename}")
            return None, None
            
        # Convert to numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        return X, y
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None, None

def build_model(input_shape, num_classes):
    """Build neural network model"""
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X, y, filename):
    """Train the model"""
    try:
        print("\nTraining model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Build model
        model = build_model(X_train_scaled.shape[1], len(np.unique(y)))
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=128,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model.save(f'nn_model_{filename}_{timestamp}')
        
        # Save scaler
        import joblib
        joblib.dump(scaler, f'nn_scaler_{filename}_{timestamp}.joblib')
        
        return model, scaler
        
    except Exception as e:
        print(f"Error training model for {filename}: {e}")
        return None, None

def main():
    try:
        data_dir = 'data'
        all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        if not all_files:
            print("No CSV files found in the data directory")
            return
            
        print(f'\nFound {len(all_files)} CSV files:')
        for filename in all_files:
            print(os.path.basename(filename))
            
        # Process each file
        for filename in all_files:
            print("\n" + "="*80)
            print(f"Processing {filename}")
            print("="*80)
            
            # Load and process data
            X, y = load_and_process_data(filename)
            if X is None or y is None:
                continue
                
            # Train model
            model, scaler = train_model(X, y, filename)
            
            print("-"*80)
            
            # Clean up memory
            del X, y, model, scaler
            gc.collect()
            
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
