from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_rf():
    print("Loading preprocessed data for Random Forest...")
    X_train, X_test, y_train, y_test = joblib.load('data/preprocessed_data.pkl')
    
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, 'models/random_forest.pkl')
    print("Random Forest model saved to models/random_forest.pkl")

if __name__ == "__main__":
    train_rf()
