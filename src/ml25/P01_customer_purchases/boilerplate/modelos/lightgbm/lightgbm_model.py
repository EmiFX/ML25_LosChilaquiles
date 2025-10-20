"""
LightGBM Model
"""
from lightgbm import LGBMClassifier
from ..model import PurchaseModel


class LightGBMModel(PurchaseModel):
    """LightGBM Classifier for Purchase Prediction"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        default_params.update(kwargs)
        
        self.model = LGBMClassifier(**default_params)
        self.kwargs = default_params
