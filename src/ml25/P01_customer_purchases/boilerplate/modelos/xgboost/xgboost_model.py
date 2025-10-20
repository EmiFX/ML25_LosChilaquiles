"""
XGBoost Model
"""
from xgboost import XGBClassifier
from ..model import PurchaseModel


class XGBoostModel(PurchaseModel):
    """XGBoost Classifier for Purchase Prediction"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        default_params = {
            'n_estimators': 200,
            'max_depth': 3,
            'learning_rate': 0.15,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        default_params.update(kwargs)
        
        self.model = XGBClassifier(**default_params)
        self.kwargs = default_params
