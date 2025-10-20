"""
SVM Training Script
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from ml25.P01_customer_purchases.boilerplate.utils import setup_logger
from ml25.P01_customer_purchases.boilerplate.data_processing import read_train_data
from ml25.P01_customer_purchases.boilerplate.modelos.svm.svm_model import SVMModel


def train_svm(X, y, test_size=0.2, **model_params):
    """Train SVM model"""
    logger = setup_logger("svm_training")
    logger.info("Starting SVM training")
    
    # Split data
    logger.info(f"Splitting data: {len(X)} samples, test_size={test_size}")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    
    # Create and train model
    logger.info(f"Creating SVM model with params: {model_params}")
    model = SVMModel(**model_params)
    
    logger.info("Training model...")
    model.fit(X_train, y_train)
    logger.info("Training complete!")
    
    # Evaluate
    logger.info("Evaluating on validation set...")
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    logger.info(f"Validation Results:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  ROC AUC:   {roc_auc:.4f}")
    logger.info("\n" + classification_report(y_val, y_pred))
    
    # Save model
    filepath = model.save("svm")
    logger.info(f"Model saved to: {filepath}")
    
    return model, {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc
    }


if __name__ == "__main__":
    print("="*80)
    print("SVM Training")
    print("="*80)
    
    # Load data
    print("\nLoading training data...")
    X, y = read_train_data()
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Configure hyperparameters
    params = {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale'
    }
    
    # Train
    model, metrics = train_svm(X, y, test_size=0.2, **params)
    
    print("\n" + "="*80)
    print("SVM Training Complete!")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("="*80)
