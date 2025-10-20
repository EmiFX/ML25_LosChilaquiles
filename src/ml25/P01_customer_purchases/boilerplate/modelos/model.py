# Data management
from pathlib import Path
import joblib
from datetime import datetime
import os

# ML
import numpy as np


CURRENT_FILE = Path(__file__).resolve()
MODELS_DIR = CURRENT_FILE.parent.parent / "trained_models"

MODELS_DIR.mkdir(exist_ok=True, parents=True)


class PurchaseModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None
        self.model_name = self.__class__.__name__
    
    def fit(self, X, y):
        # Handle NaN values
        X_clean = X.fillna(0)
        
        # Convert to numpy array if needed
        if hasattr(X_clean, 'values'):
            X_clean = X_clean.values
        
        self.model.fit(X_clean, y)
        return self

    def predict(self, X):
        """Predict class labels"""
        X_clean = X.fillna(0)
        if hasattr(X_clean, 'values'):
            X_clean = X_clean.values
        return self.model.predict(X_clean)

    def predict_proba(self, X):
        """Predict class probabilities"""
        X_clean = X.fillna(0)
        if hasattr(X_clean, 'values'):
            X_clean = X_clean.values
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_clean)
        else:
            # For models without predict_proba (like LinearRegression)
            preds = self.model.predict(X_clean)
            # Convert to probabilities (clip to [0, 1])
            probs = np.clip(preds, 0, 1)
            return np.column_stack([1 - probs, probs])

    def get_config(self):
        """
        Return key hyperparameters of the model for logging.
        """
        config = {
            'model_name': self.model_name,
            'params': self.kwargs
        }
        return config
    
    def __repr__(self):
        return f"{self.model_name}(type={self.model_name.replace('Model', '')})"

    def save(self, prefix: str):
        """
        Save the model to disk in MODELS_DIR with filename:
        <prefix>_<timestamp>.pkl

        Try to use descriptive prefix that help you keep track of the paramteters used for training to distinguish between models.
        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{now}.pkl"
        filepath = Path(MODELS_DIR) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath = os.path.abspath(filepath)

        joblib.dump(self, filepath)
        print(f"{repr(self)} || Model saved to {filepath}")
        return filepath

    def load(self, filename: str):
        """
        Load the model from MODELS_DIR/filename
        """
        filepath = Path(MODELS_DIR) / filename
        model = joblib.load(filepath)
        print(f"{self.__repr__} || Model loaded from {filepath}")
        return model