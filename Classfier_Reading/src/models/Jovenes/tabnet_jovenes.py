"""
TabNet Classifier - Modelo para clasificación de JÓVENES (12-17 años)

TabNet es una arquitectura de atención secuencial interpretable diseñada
específicamente para datos tabulares.

Características específicas para JÓVENES:
- 139 features de entrada
- Atención interpretable con máscaras
- CRÍTICO: Distribución MUY desbalanceada (80% / 0.21% / 19%)
- Requiere class weights obligatorio

Basado en: Arik & Pfister (2021) "TabNet: Attentive Interpretable Tabular Learning"
"""

import torch
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, Tuple
from pytorch_tabnet.tab_model import TabNetClassifier as PyTorchTabNetClassifier


class TabNetJovenes:
    """
    Wrapper para TabNet específico para JÓVENES (12-17 años).
    
    Configuración optimizada para 139 features y manejo de
    distribución extremadamente desbalanceada.
    """
    
    def __init__(
        self,
        num_features: int = 139,
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
        """
        Args:
            num_features: Número de features de entrada (139 para jóvenes)
            num_classes: Número de clases (3)
            n_d: Dimensión de decision layer
            n_a: Dimensión de attention layer
            n_steps: Número de decision steps
            gamma: Factor de relajación para feature selection
            lambda_sparse: Penalización de sparsity
            mask_type: 'entmax' o 'sparsemax'
        """
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
            weights: CRÍTICO para jóvenes - pesos de clase para balanceo
        """
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # IMPORTANTE: weights para clase minoritaria (0.21%)
        self.history_ = self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=eval_set,
            eval_metric=['accuracy', 'logloss'],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            weights=weights  # ← CRÍTICO
        )
        
        # Guardar importancias
        self.feature_importances_ = self.model.feature_importances_
        
        return self.history_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def explain(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Explicar predicciones con máscaras de atención.
        
        Returns:
            masks: Máscaras de atención (n_samples, n_steps, n_features)
            importances: Importancias agregadas (n_samples, n_features)
        """
        explain_matrix, masks = self.model.explain(X)
        return masks, explain_matrix
    
    def get_feature_importance(self) -> np.ndarray:
        """Obtener importancia global de features."""
        if self.feature_importances_ is None:
            raise ValueError("Modelo no entrenado")
        return self.feature_importances_
    
    def save(self, path: str):
        """Guardar modelo."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # TabNet usa save_model
        self.model.save_model(path)
        
        # Guardar metadatos adicionales
        metadata = {
            'feature_importances': self.feature_importances_,
            'num_features': self.num_features,
            'num_classes': self.num_classes
        }
        joblib.dump(metadata, f"{path}_metadata.pkl")
        
        print(f"✅ TabNet JÓVENES guardado en: {path}")
    
    def load(self, path: str):
        """Cargar modelo."""
        self.model.load_model(path)
        
        # Cargar metadatos
        metadata = joblib.load(f"{path}_metadata.pkl")
        self.feature_importances_ = metadata['feature_importances']
        
        print(f"✅ TabNet JÓVENES cargado desde: {path}")


def create_tabnet_for_jovenes(num_features: int = 139) -> TabNetJovenes:
    """
    Función auxiliar para crear TabNet optimizado para Jóvenes.
    
    Args:
        num_features: Número de features (default=139 para jóvenes)
    
    Returns:
        Modelo TabNet configurado
    """
    model = TabNetJovenes(
        num_features=num_features,
        num_classes=3,
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        mask_type='entmax'
    )
    
    print("✅ TabNet para JÓVENES creado")
    print(f"   Features: {num_features}")
    print(f"   n_d=64, n_a=64, n_steps=5")
    print(f"   ⚠️  IMPORTANTE: Usar class weights (distribución 80/0.21/19%)")
    
    return model
