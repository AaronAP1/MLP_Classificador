"""
TabNet Classifier - Modelo para clasificación de NIÑOS (0-11 años)

TabNet es una arquitectura de atención interpretable secuencial diseñada
específicamente para datos tabulares. Utiliza mecanismos de atención
enmascarada para seleccionar features relevantes en cada paso de decisión.

Características:
- Atención interpretable (máscaras de importancia)
- Capacidad de aprendizaje incremental
- Regularización de dispersión (sparsity)
- No requiere ingeniería de features manual

Paper: "TabNet: Attentive Interpretable Tabular Learning"
       https://arxiv.org/abs/1908.07442
"""

import torch
import torch.nn as nn
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier as PyTorchTabNetClassifier
from typing import Dict, Tuple, Optional
import joblib
from pathlib import Path


class TabNetClassifier:
    """
    Wrapper para TabNetClassifier optimizado para clasificación de comportamiento lector.
    
    Diseñado específicamente para el grupo de NIÑOS (0-11 años).
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_independent: int = 2,
        n_shared: int = 2,
        lambda_sparse: float = 1e-3,
        momentum: float = 0.02,
        lr: float = 2e-2,
        mask_type: str = 'entmax',
        seed: int = 42
    ):
        """
        Inicializa el clasificador TabNet.
        
        Args:
            input_dim: Número de features de entrada
            num_classes: Número de clases (default: 3)
            n_d: Dimensión de la capa de decisión
            n_a: Dimensión de la capa de atención
            n_steps: Número de pasos de decisión secuencial
            gamma: Coeficiente de regularización de atención
            n_independent: Número de capas independientes por paso
            n_shared: Número de capas compartidas
            lambda_sparse: Peso de regularización de sparsity
            momentum: Momentum para Batch Normalization
            lr: Learning rate
            mask_type: Tipo de máscara ('sparsemax' o 'entmax')
            seed: Semilla para reproducibilidad
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.config = {
            'n_d': n_d,
            'n_a': n_a,
            'n_steps': n_steps,
            'gamma': gamma,
            'n_independent': n_independent,
            'n_shared': n_shared,
            'lambda_sparse': lambda_sparse,
            'momentum': momentum,
            'lr': lr,
            'mask_type': mask_type,
            'seed': seed
        }
        
        # Inicializar modelo
        self.model = PyTorchTabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            lambda_sparse=lambda_sparse,
            momentum=momentum,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=lr),
            mask_type=mask_type,
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            seed=seed,
            verbose=1
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        max_epochs: int = 200,
        batch_size: int = 256,
        virtual_batch_size: int = 128,
        patience: int = 30,
        weights: Optional[np.ndarray] = None
    ):
        """
        Entrena el modelo TabNet.
        
        Args:
            X_train: Features de entrenamiento (N, input_dim)
            y_train: Labels de entrenamiento (N,)
            X_val: Features de validación (M, input_dim)
            y_val: Labels de validación (M,)
            max_epochs: Número máximo de épocas
            batch_size: Tamaño de batch
            virtual_batch_size: Tamaño de batch virtual para Ghost Batch Normalization
            patience: Paciencia para early stopping
            weights: Pesos de clase para balanceo (opcional)
        """
        print(f"\n{'='*80}")
        print(f"🚀 ENTRENANDO TABNET - GRUPO: NIÑOS")
        print(f"{'='*80}\n")
        print(f"📊 Datos:")
        print(f"   Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        print(f"   Val:   {X_val.shape[0]:,} samples")
        print(f"\n⚙️  Hiperparámetros:")
        print(f"   n_d/n_a: {self.config['n_d']}/{self.config['n_a']}")
        print(f"   n_steps: {self.config['n_steps']}")
        print(f"   gamma: {self.config['gamma']}")
        print(f"   lambda_sparse: {self.config['lambda_sparse']}")
        print(f"   batch_size: {batch_size}")
        print(f"   max_epochs: {max_epochs}")
        print(f"   patience: {patience}\n")
        
        # Preparar pesos si se proporcionan
        sample_weights = None
        if weights is not None:
            # Mapear pesos de clase a pesos de muestra
            sample_weights = np.array([weights[int(label)] for label in y_train])
            print(f"⚖️  Usando pesos de clase para balanceo")
            print(f"   Pesos: {weights}\n")
        
        # Entrenar
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_name=['validation'],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=0,
            drop_last=False,
            weights=sample_weights if sample_weights is not None else 1
        )
        
        # Guardar historial
        # El objeto History de pytorch_tabnet tiene atributos, no es un dict
        self.history['train_loss'] = self.model.history['loss']
        
        # Buscar métrica de validación
        # Intentar diferentes nombres posibles
        val_loss = None
        possible_keys = ['val_0_loss', 'val_loss', 'validation_0_loss', 'valid_loss']
        
        for key in possible_keys:
            try:
                val_loss = self.model.history[key]
                break
            except (KeyError, AttributeError):
                continue
        
        if val_loss is not None:
            self.history['val_loss'] = val_loss
        else:
            # Si no hay validación, usar train loss como fallback
            self.history['val_loss'] = self.model.history['loss']
            print("   ⚠️  No se encontró historial de validación, usando train loss")
        
        print(f"\n✅ Entrenamiento completado")
        print(f"   Mejor época: {self.model.best_epoch}")
        if 'val_loss' in self.history and len(self.history['val_loss']) > 0:
            print(f"   Mejor val loss: {min(self.history['val_loss']):.4f}\n")
        else:
            print(f"   Train loss final: {self.history['train_loss'][-1]:.4f}\n")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice clases para nuevas muestras.
        
        Args:
            X: Features (N, input_dim)
            
        Returns:
            Predicciones de clase (N,)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predice probabilidades de clase.
        
        Args:
            X: Features (N, input_dim)
            
        Returns:
            Probabilidades de clase (N, num_classes)
        """
        return self.model.predict_proba(X)
    
    def explain(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtiene las máscaras de atención para interpretabilidad.
        
        Args:
            X: Features (N, input_dim)
            
        Returns:
            - masks: Máscaras de atención (N, n_steps, input_dim)
            - reduce_masks: Máscaras agregadas (N, input_dim)
        """
        masks, reduce_masks = self.model.explain(X)
        return masks, reduce_masks
    
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula importancia promedio de features en el dataset.
        
        Args:
            X: Features (N, input_dim)
            
        Returns:
            Importancia de features (input_dim,)
        """
        _, reduce_masks = self.explain(X)
        # Promediar sobre todas las muestras
        importance = reduce_masks.mean(axis=0)
        return importance
    
    def save(self, path: str):
        """
        Guarda el modelo entrenado.
        
        Args:
            path: Ruta del archivo (sin extensión)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar modelo TabNet
        self.model.save_model(str(path))
        
        # Guardar configuración e historial
        metadata = {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'config': self.config,
            'history': self.history
        }
        joblib.dump(metadata, f"{path}_metadata.pkl")
        
        print(f"✅ Modelo guardado en: {path}")
    
    def load(self, path: str):
        """
        Carga un modelo previamente entrenado.
        
        Args:
            path: Ruta del archivo (sin extensión)
        """
        path = Path(path)
        
        # Cargar modelo TabNet
        self.model.load_model(str(path) + ".zip")
        
        # Cargar configuración e historial
        metadata = joblib.load(f"{path}_metadata.pkl")
        self.input_dim = metadata['input_dim']
        self.num_classes = metadata['num_classes']
        self.config = metadata['config']
        self.history = metadata['history']
        
        print(f"✅ Modelo cargado desde: {path}")
    
    def get_params(self) -> Dict:
        """Retorna los hiperparámetros del modelo."""
        return self.config.copy()
    
    def summary(self):
        """Imprime un resumen del modelo."""
        print(f"\n{'='*80}")
        print(f"📊 TABNET CLASSIFIER - RESUMEN")
        print(f"{'='*80}\n")
        print(f"🎯 Arquitectura:")
        print(f"   Input dim: {self.input_dim}")
        print(f"   Num classes: {self.num_classes}")
        print(f"   Decision dim (n_d): {self.config['n_d']}")
        print(f"   Attention dim (n_a): {self.config['n_a']}")
        print(f"   Decision steps: {self.config['n_steps']}")
        print(f"\n⚙️  Configuración:")
        print(f"   Gamma: {self.config['gamma']}")
        print(f"   Lambda sparse: {self.config['lambda_sparse']}")
        print(f"   Independent layers: {self.config['n_independent']}")
        print(f"   Shared layers: {self.config['n_shared']}")
        print(f"   Mask type: {self.config['mask_type']}")
        print(f"\n📈 Entrenamiento:")
        if self.history['train_loss']:
            print(f"   Épocas entrenadas: {len(self.history['train_loss'])}")
            print(f"   Train loss final: {self.history['train_loss'][-1]:.4f}")
            print(f"   Val loss final: {self.history['val_loss'][-1]:.4f}")
            print(f"   Mejor val loss: {min(self.history['val_loss']):.4f}")
        else:
            print(f"   Modelo no entrenado")
        print(f"\n{'='*80}\n")


# Función auxiliar para crear TabNet con configuración por defecto
def create_tabnet_for_ninos(input_dim: int) -> TabNetClassifier:
    """
    Crea un TabNet con configuración optimizada para el grupo de NIÑOS.
    
    Args:
        input_dim: Número de features de entrada
        
    Returns:
        TabNetClassifier configurado
    """
    return TabNetClassifier(
        input_dim=input_dim,
        num_classes=3,
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-3,
        momentum=0.02,
        lr=2e-2,
        mask_type='entmax',
        seed=42
    )


if __name__ == "__main__":
    # Ejemplo de uso
    print("TabNet Classifier - Ejemplo de uso\n")
    
    # Crear modelo
    model = create_tabnet_for_ninos(input_dim=131)  # 131 features para niños
    model.summary()
    
    print("✅ Modelo TabNet creado correctamente")
    print("📋 Listo para entrenamiento con datos de NIÑOS")
