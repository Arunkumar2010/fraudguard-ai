from xgboost import XGBClassifier
import joblib
import os

def train_xgb():
    print("Loading preprocessed data for XGBoost...")
    X_train, X_test, y_train, y_test = joblib.load('data/preprocessed_data.pkl')
    
    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb, 'models/xgboost.pkl')
    print("XGBoost model saved to models/xgboost.pkl")

if __name__ == "__main__":
    train_xgb()
