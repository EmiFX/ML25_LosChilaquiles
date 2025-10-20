import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from ml25.P01_customer_purchases.boilerplate.data_processing import read_test_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
from datetime import datetime
import sys
import pickle

from ml25.P01_customer_purchases.boilerplate import modelos as modelos_package
from ml25.P01_customer_purchases.boilerplate.modelos import xgboost as xgb_mod
from ml25.P01_customer_purchases.boilerplate.modelos import lightgbm as lgb_mod
from ml25.P01_customer_purchases.boilerplate.modelos import random_forest as rf_mod
from ml25.P01_customer_purchases.boilerplate.modelos import svm as svm_mod
from ml25.P01_customer_purchases.boilerplate.modelos import logistic_regression as lr_mod
from ml25.P01_customer_purchases.boilerplate.modelos import linear_regression as linreg_mod
from ml25.P01_customer_purchases.boilerplate.modelos.model import PurchaseModel

# Create aliases for old import paths
sys.modules['ml25.P01_customer_purchases.boilerplate.models'] = modelos_package
sys.modules['ml25.P01_customer_purchases.boilerplate.models.xgboost_model'] = xgb_mod
sys.modules['ml25.P01_customer_purchases.boilerplate.models.lightgbm_model'] = lgb_mod
sys.modules['ml25.P01_customer_purchases.boilerplate.models.random_forest_model'] = rf_mod
sys.modules['ml25.P01_customer_purchases.boilerplate.models.svm_model'] = svm_mod
sys.modules['ml25.P01_customer_purchases.boilerplate.models.logistic_regression_model'] = lr_mod
sys.modules['ml25.P01_customer_purchases.boilerplate.models.linear_regression_model'] = linreg_mod

# Old path (singular): ml25.P01_customer_purchases.boilerplate.model
sys.modules['ml25.P01_customer_purchases.boilerplate.model'] = modelos_package

# Create a custom unpickler to handle PurchaseModel class
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect old PurchaseModel references to the new location
        if name == 'PurchaseModel':
            return PurchaseModel
        return super().find_class(module, name)

CURRENT_FILE = Path(__file__).resolve()

RESULTS_DIR = CURRENT_FILE.parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

MODELS_DIR = CURRENT_FILE.parent / "trained_models"


def load_model(filename: str):
    filepath = Path(MODELS_DIR) / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")
    
    # Use custom unpickler to handle PurchaseModel class
    with open(filepath, 'rb') as f:
        model = CustomUnpickler(f).load()
    return model


def list_available_models():
    if not MODELS_DIR.exists():
        return []
    
    # Only get LightGBM models
    model_files = list(MODELS_DIR.glob("lightgbm*.pkl"))
    
    if not model_files:
        return []
    
    return [f.name for f in model_files]


def run_inference(model_name: str, X, save_results=True):

    # Load the model using custom unpickler
    full_path = MODELS_DIR / model_name
    with open(full_path, 'rb') as f:
        model = CustomUnpickler(f).load()
    
    # Check if model expects more features than provided
    # This handles old models trained with different preprocessing
    X_processed = X.copy()

    # Get predictions
    preds = model.predict(X_processed)
    probs = model.predict_proba(X_processed)[:, 1]  # Probability of class 1 (purchase)
    
    # Create results DataFrame
    results = pd.DataFrame({
        "ID": X.index,
        "pred": preds,
        "probability": probs
    })
    
    # Save results if requested
    if save_results:
        # Extract model type from filename
        model_type = model_name.split('_')[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{model_type}_{timestamp}.csv"
        filepath = RESULTS_DIR / filename
        
        # Save in Kaggle submission format (ID, pred)
        submission_df = results[["ID", "pred"]]
        submission_df.to_csv(filepath, index=False)
    
    return results


def plot_roc_curve(y_true, y_proba, model_name="Model", save_plot=True):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"roc_curve_{model_name}_{timestamp}.png"
        filepath = RESULTS_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to: {filepath}")
    
    plt.show()
    
    return roc_auc


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_plot=True):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                   display_labels=['No Purchase', 'Purchase'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.grid(False)
    
    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"confusion_matrix_{model_name}_{timestamp}.png"
        filepath = RESULTS_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {filepath}")
    
    plt.show()


def evaluate_model(model_name: str, X_test, y_test):
    print(f"\n{'='*80}")
    print(f"Evaluacion de modelo: {model_name}")
    print(f"{'='*80}\n")
    
    # Load model and make predictions
    model = load_model(model_name)
    
    # Check if model expects more features than provided
    X_processed = X_test.copy()
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'n_features_in_'):
            expected_features = model.model.n_features_in_
            current_features = X_processed.shape[1]
            
            if expected_features > current_features:
                print(f"⚠️  Model expects {expected_features} features, but data has {current_features}")
                print(f"   Adding {expected_features - current_features} zero-filled columns to match...")
                
                # Add missing features as zeros
                for i in range(current_features, expected_features):
                    X_processed[f'missing_feature_{i}'] = 0
                
                print(f"✓ Data shape adjusted from {X_test.shape} to {X_processed.shape}\n")
    except Exception as e:
        print(f"⚠️  Could not check feature count: {e}\n")
    
    y_pred = model.predict(X_processed)
    y_proba = model.predict_proba(X_processed)[:, 1]
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Print metrics
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    
    # Plot visualizations
    model_type = model_name.split('_')[0]
    plot_confusion_matrix(y_test, y_pred, model_type)
    plot_roc_curve(y_test, y_proba, model_type)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    return results


if __name__ == "__main__":
    print("="*80)
    print("LightGBM Inference")
    print("="*80)
    
    # List available LightGBM models
    available_models = list_available_models()
    
    # Load test data
    X_test = read_test_data()
    print(f"Datos de test: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Manually select a specific model
    model_name = "lightgbm_20251020_154304.pkl"  # Replace with your model name
    print(f"\nModelo usado: {model_name}\n")
    
    # Run inference
    results = run_inference(model_name, X_test, save_results=True)
    
    print("\n" + "="*80)
    print("Inferencia completa")
    print(f"Predictions: {len(results)} samples")
    print(f"Positive predictions: {results['pred'].sum()} ({results['pred'].mean()*100:.2f}%)")
    print("="*80)
    
    # Uncomment below if you have validation labels to evaluate
    from ml25.P01_customer_purchases.boilerplate.data_processing import read_train_data
    X_train, y_train = read_train_data()
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    eval_results = evaluate_model(model_name, X_val, y_val)
