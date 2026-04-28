import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
import joblib
import os

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(16, activation='relu')(input_layer)
    encoded = Dense(8, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder():
    print("Loading preprocessed data for Autoencoder...")
    X_train, X_test, y_train, y_test = joblib.load('data/preprocessed_data.pkl')
    
    # Autoencoders for anomaly detection are typically trained on normal data only
    # But here we'll follow the requirement to provide reconstruction error as a feature
    # Using the normal samples to train the AE.
    X_train_normal = X_train[y_train == 0]
    
    print(f"Training Autoencoder on {len(X_train_normal)} normal samples...")
    input_dim = X_train_normal.shape[1]
    autoencoder = build_autoencoder(input_dim)
    
    # Training
    autoencoder.fit(X_train_normal, X_train_normal, 
                    epochs=20, 
                    batch_size=32, 
                    shuffle=True, 
                    validation_split=0.2, 
                    verbose=0)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    autoencoder.save('models/autoencoder.keras')
    print("Autoencoder model saved to models/autoencoder.keras")
    
    # Generate reconstruction error features
    print("Generating reconstruction error features...")
    train_pred = autoencoder.predict(X_train, verbose=0)
    mse_train = np.mean(np.power(X_train - train_pred, 2), axis=1)
    
    test_pred = autoencoder.predict(X_test, verbose=0)
    mse_test = np.mean(np.power(X_test - test_pred, 2), axis=1)
    
    # Add AE_Error as a new feature
    X_train['AE_Error'] = mse_train
    X_test['AE_Error'] = mse_test
    
    # Save augmented data
    joblib.dump((X_train, X_test, y_train, y_test), 'data/augmented_data.pkl')
    print("Data augmented with AE_Error saved to data/augmented_data.pkl")

if __name__ == "__main__":
    train_autoencoder()
