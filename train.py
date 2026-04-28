import subprocess
import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
from src.models import HybridModel
warnings.filterwarnings('ignore')

def main():
    print("Step 1: Loading dataset...")
    if not os.path.exists('dataset/creditcard.csv'):
        print("Error: dataset/creditcard.csv not found.")
        return
    df = pd.read_csv('dataset/creditcard.csv')
    print(f"Dataset loaded: {df.shape}")

    print("Step 2: Preprocessing...")
    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    X['Time'] = scaler.fit_transform(X[['Time']])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("Step 3: Applying SMOTE (only on training data)...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_train_sm.shape}")

    os.makedirs('models', exist_ok=True)

    print("Step 4: Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_sm, y_train_sm)
    with open('models/random_forest.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("Random Forest saved!")

    print("Step 5: Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_model.fit(X_train_sm, y_train_sm)
    with open('models/xgboost.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print("XGBoost saved!")

    print("Step 6: Building Hybrid Stacking Model...")
    rf_train_proba = rf_model.predict_proba(X_train_sm)[:,1]
    xgb_train_proba = xgb_model.predict_proba(X_train_sm)[:,1]
    meta_X_train = np.column_stack([rf_train_proba, xgb_train_proba])

    meta_model = LogisticRegression()
    meta_model.fit(meta_X_train, y_train_sm)

    hybrid = HybridModel(rf_model, xgb_model, meta_model)

    with open('models/hybrid_model.pkl', 'wb') as f:
        pickle.dump(hybrid, f)
    print("Hybrid Model saved!")

    print("Step 7: Evaluating all models...")
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    hybrid_pred = hybrid.predict(X_test)

    print("\n=== RANDOM FOREST ===")
    print(classification_report(y_test, rf_pred))
    print("\n=== XGBOOST ===")
    print(classification_report(y_test, xgb_pred))
    print("\n=== HYBRID MODEL ===")
    print(classification_report(y_test, hybrid_pred))

    metrics = {
        'rf': classification_report(y_test, rf_pred, output_dict=True),
        'xgb': classification_report(y_test, xgb_pred, output_dict=True),
        'hybrid': classification_report(y_test, hybrid_pred, output_dict=True)
    }
    with open('models/metrics_modular.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("\nAll models trained and saved successfully!")

if __name__ == "__main__":
    main()
