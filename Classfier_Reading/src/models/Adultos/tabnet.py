"""
TabNet implementation wrapper.
Usa pytorch-tabnet: pip install pytorch-tabnet
"""
from pytorch_tabnet.tab_model import TabNetClassifier
import torch


class TabNetWrapper:
    """Wrapper para TabNetClassifier compatible con pipeline de entrenamiento."""
    
    def __init__(self, input_dim: int, n_d: int = 64, n_a: int = 64, 
                 n_steps: int = 5, gamma: float = 1.5, 
                 n_independent: int = 2, n_shared: int = 2,
                 lambda_sparse: float = 0.001, momentum: float = 0.3,
                 mask_type: str = "entmax", num_classes: int = 3):
        
        self.model = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            lambda_sparse=lambda_sparse,
            momentum=momentum,
            mask_type=mask_type,
            optimizer_fn=torch.optim.Adam,
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=0
        )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            max_epochs=200, patience=20, batch_size=256):
        """Entrena el modelo."""
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=eval_set,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=128,
            eval_metric=['accuracy']
        )
    
    def predict(self, X):
        """Predice clases."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predice probabilidades."""
        return self.model.predict_proba(X)
    
    def save(self, path):
        """Guarda el modelo."""
        self.model.save_model(path)
    
    def load(self, path):
        """Carga el modelo."""
        self.model.load_model(path)
