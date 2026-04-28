import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

def preprocess_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)

    # Scaling Amount and Time
    print("Scaling Time and Amount...")
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    # Split into features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Train-test split (80/20)
    print("Splitting into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE only on training data
    print("Applying SMOTE on training data...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Save processed data for next steps
    print("Saving processed data...")
    joblib.dump((X_train_res, X_test, y_train_res, y_test), 'data/preprocessed_data.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("Preprocessing completed successfully!")
    return X_train_res, X_test, y_train_res, y_test

if __name__ == "__main__":
    dataset_path = 'dataset/creditcard.csv'
    if os.path.exists(dataset_path):
        preprocess_data(dataset_path)
    else:
        print(f"Error: Dataset not found at {dataset_path}")
