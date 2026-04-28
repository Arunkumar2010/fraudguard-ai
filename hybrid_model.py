import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import os

def train_hybrid_model():
    if not os.path.exists('data/augmented_data.pkl'):
        print("Error: Augmented data (with AE error) not found. Run autoencoder_model.py first.")
        return

    print("Loading augmented data for Hybrid Model...")
    X_train, X_test, y_train, y_test = joblib.load('data/augmented_data.pkl')
    
    # Base learners
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ]
    
    # Meta learner (XGBoost)
    meta_learner = XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    print("Training Hybrid Stacking Model...")
    # Stacking Classifier with probabilities as meta-features
    stack_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        passthrough=True, # Include original features (including AE_Error) in meta-learner
        cv=5
    )
    
    stack_model.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(stack_model, 'models/hybrid_model.pkl')
    print("Hybrid Model saved to models/hybrid_model.pkl")

if __name__ == "__main__":
    train_hybrid_model()
