"""
Support Vector Machine Model
"""
from sklearn.svm import SVC
from ..model import PurchaseModel


class SVMModel(PurchaseModel):
    """Support Vector Machine Classifier for Purchase Prediction"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'probability': True,
            'random_state': 42,
            'max_iter': 1000
        }
        default_params.update(kwargs)
        
        self.model = SVC(**default_params)
        self.kwargs = default_params
