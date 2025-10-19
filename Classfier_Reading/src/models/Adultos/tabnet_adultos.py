"""
TabNet Classifier - Modelo para clasificación de ADULTOS (18+ años)

TabNet con configuración optimizada para el grupo de adultos.

Características específicas para ADULTOS:
- 71 features de entrada
- Atención interpretable con máscaras
- CRÍTICO: Distribución MUY desbalanceada (20% / 0.02% / 80%)
- Requiere class weights obligatorio (clase 1 casi inexistente!)

Basado en: Arik & Pfister (2021) "TabNet: Attentive Interpretable Tabular Learning"
"""

import torch
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, Tuple
from pytorch_tabnet.tab_model import TabNetClassifier as PyTorchTabNetClassifier


class TabNetAdultos:
    """
    Wrapper para TabNet específico para ADULTOS (18+ años).
    
    Configuración optimizada para 71 features y manejo de
    distribución extremadamente desbalanceada (clase 1: 0.02%).
    """
    
    def __init__(
        self,
        num_features: int = 71,
        num_classes: int = 3,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_independent: int = 2,
        n_shared: int = 2,
        lambda_sparse: float = 1e-4,
        momentum: float = 0.3,
        mask_type: str = 'entmax'
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        
        self.model = PyTorchTabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            lambda_sparse=lambda_sparse,
            momentum=momentum,
            mask_type=mask_type,
            verbose=0
        )
        
        self.feature_importances_ = None
        self.history_ = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        max_epochs: int = 200,
        patience: int = 30,
        batch_size: int = 256,
        virtual_batch_size: int = 128
    ):
        """
        Entrenar TabNet.
        
        Args:
            weights: MUY CRÍTICO para adultos - pesos de clase (clase 1: 0.02%)
        """
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        self.history_ = self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=eval_set,
            eval_metric=['accuracy', 'logloss'],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            weights=weights  # ← MUY CRÍTICO
        )
        
        self.feature_importances_ = self.model.feature_importances_
        return self.history_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def explain(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Explicar predicciones con máscaras de atención."""
        explain_matrix, masks = self.model.explain(X)
        return masks, explain_matrix
    
    def get_feature_importance(self) -> np.ndarray:
        if self.feature_importances_ is None:
            raise ValueError("Modelo no entrenado")
        return self.feature_importances_
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(path)
        
        metadata = {
            'feature_importances': self.feature_importances_,
            'num_features': self.num_features,
            'num_classes': self.num_classes
        }
        joblib.dump(metadata, f"{path}_metadata.pkl")
        
        print(f"✅ TabNet ADULTOS guardado en: {path}")
    
    def load(self, path: str):
        self.model.load_model(path)
        metadata = joblib.load(f"{path}_metadata.pkl")
        self.feature_importances_ = metadata['feature_importances']
        print(f"✅ TabNet ADULTOS cargado desde: {path}")


def create_tabnet_for_adultos(num_features: int = 71) -> TabNetAdultos:
    """
    Función auxiliar para crear TabNet optimizado para Adultos.
    
    Args:
        num_features: Número de features (default=71 para adultos)
    
    Returns:
        Modelo TabNet configurado
    """
    model = TabNetAdultos(
        num_features=num_features,
        num_classes=3,
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        mask_type='entmax'
    )
    
    print("✅ TabNet para ADULTOS creado")
    print(f"   Features: {num_features}")
    print(f"   n_d=64, n_a=64, n_steps=5")
    print(f"   ⚠️  MUY CRÍTICO: Usar class weights (distribución 20/0.02/80%)")
    print(f"   ⚠️  Clase 1 casi inexistente (0.02%)!")
    
    return model
