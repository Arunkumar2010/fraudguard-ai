import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, roc_curve
import os

def evaluate():
    print("Loading models and data for evaluation...")
    # Original data (no AE_Error)
    X_train_orig, X_test_orig, y_train, y_test = joblib.load('data/preprocessed_data.pkl')
    # Augmented data (with AE_Error)
    if os.path.exists('data/augmented_data.pkl'):
        X_train_aug, X_test_aug, _, _ = joblib.load('data/augmented_data.pkl')
    else:
        X_test_aug = None

    models = {}
    if os.path.exists('models/random_forest.pkl'):
        models['Random Forest'] = joblib.load('models/random_forest.pkl')
    if os.path.exists('models/xgboost.pkl'):
        models['XGBoost'] = joblib.load('models/xgboost.pkl')
    if os.path.exists('models/hybrid_model.pkl'):
        models['Hybrid'] = joblib.load('models/hybrid_model.pkl')

    results = []
    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        print(f"Evaluating {name}...")
        # Use augmented data for Hybrid, original for others
        X_eval = X_test_aug if name == 'Hybrid' else X_test_orig
        
        if X_eval is None:
            print(f"Skipping {name} as evaluation data is not available.")
            continue

        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1]

        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob),
            'MCC': matthews_corrcoef(y_test, y_pred)
        }
        results.append(metrics)

        # Plot ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {metrics["ROC-AUC"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    os.makedirs('static/js', exist_ok=True) # Place for charts maybe? Actually we'll save it to templates or static
    plt.savefig('static/roc_curve.png')
    print("ROC Curve saved to static/roc_curve.png")

    results_df = pd.DataFrame(results)
    print("\nModel Comparison Metrics:")
    print(results_df)

    joblib.dump(results_df, 'models/metrics_modular.pkl')
    print("Metrics saved to models/metrics_modular.pkl")

if __name__ == "__main__":
    evaluate()
