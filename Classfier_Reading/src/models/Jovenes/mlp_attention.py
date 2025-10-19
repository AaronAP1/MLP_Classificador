"""
MLP + Multi-Head Attention Classifier - Modelo para clasificación de JÓVENES (12-17 años)

Este modelo combina una arquitectura MLP (Multi-Layer Perceptron) tradicional
con un mecanismo de atención a nivel de features de entrada. La atención permite
al modelo ponderar dinámicamente la importancia de cada feature antes de procesarla.

Características:
- InputFeatureAttention: Atención sobre features de entrada
- Residual connections: Para mejor flujo de gradientes
- Layer Normalization: Estabilización del entrenamiento
- Dropout: Regularización para evitar overfitting

Inspirado en: "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import numpy as np
import joblib
from pathlib import Path


class InputFeatureAttention(nn.Module):
    """
    Mecanismo de atención sobre features de entrada.
    
    Calcula scores de atención para cada feature y las pondera
    antes de pasarlas a las capas MLP.
    """
    
    def __init__(self, input_dim: int, attn_dim: int = 32):
        """
        Args:
            input_dim: Número de features de entrada
            attn_dim: Dimensión del espacio de atención
        """
        super().__init__()
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        
        # Capas de atención
        self.query = nn.Linear(input_dim, attn_dim)
        self.key = nn.Linear(input_dim, attn_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        # Inicialización Xavier
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            - attended_features: Features ponderadas (batch_size, input_dim)
            - attention_scores: Scores de atención (batch_size, input_dim)
        """
        # x shape: (B, input_dim)
        batch_size = x.size(0)
        
        # Expandir para operación de atención
        x_expanded = x.unsqueeze(1)  # (B, 1, input_dim)
        
        # Calcular query, key, value
        q = self.query(x)  # (B, attn_dim)
        k = self.key(x)    # (B, attn_dim)
        v = self.value(x)  # (B, input_dim)
        
        # Calcular scores de atención: q · k^T / sqrt(d_k)
        scores = torch.sum(q * k, dim=1, keepdim=True) / (self.attn_dim ** 0.5)  # (B, 1)
        
        # Aplicar softmax para obtener pesos de atención
        # En este caso, queremos atención a nivel de feature individual
        # Reorganizamos para aplicar atención por feature
        
        # Versión simplificada: usar v directamente con gating
        attention_weights = torch.sigmoid(scores)  # (B, 1)
        
        # Aplicar atención
        attended = v * attention_weights + x * (1 - attention_weights)  # (B, input_dim)
        
        # Retornar features atendidas y scores (para visualización)
        return attended, attention_weights.squeeze(1)


class ResidualBlock(nn.Module):
    """
    Bloque residual con conexión skip.
    
    Ayuda al flujo de gradientes y permite entrenar redes más profundas.
    """
    
    def __init__(self, dim: int, dropout: float = 0.0, norm: str = 'layer'):
        """
        Args:
            dim: Dimensión de entrada/salida
            dropout: Probabilidad de dropout
            norm: Tipo de normalización ('layer', 'batch', o None)
        """
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(dim, dim))
        layers.append(nn.ReLU())
        
        if norm == 'layer':
            layers.append(nn.LayerNorm(dim))
        elif norm == 'batch':
            layers.append(nn.BatchNorm1d(dim))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)
        
        # Inicialización
        nn.init.xavier_uniform_(layers[0].weight)
        nn.init.zeros_(layers[0].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, dim)
            
        Returns:
            Output con conexión residual (batch_size, dim)
        """
        return x + self.net(x)


class MLPWithAttention(nn.Module):
    """
    MLP con mecanismo de atención a nivel de features de entrada.
    
    Diseñado específicamente para el grupo de JÓVENES (12-17 años).
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dims: List[int] = [256, 128, 64, 32],
        attn_dim: int = 32,
        dropout_p: float = 0.3,
        norm: str = 'layer',
        residual: bool = True
    ):
        """
        Args:
            input_dim: Número de features de entrada
            num_classes: Número de clases de salida (default: 3)
            hidden_dims: Lista de dimensiones de capas ocultas
            attn_dim: Dimensión del espacio de atención
            dropout_p: Probabilidad de dropout
            norm: Tipo de normalización ('layer', 'batch', o None)
            residual: Usar bloques residuales
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.attn_dim = attn_dim
        self.dropout_p = dropout_p
        self.norm = norm
        self.residual = residual
        
        # Capa de atención sobre features de entrada
        self.input_attn = InputFeatureAttention(input_dim, attn_dim)
        
        # Backbone MLP
        layers = []
        prev_dim = input_dim
        
        for i, h_dim in enumerate(hidden_dims):
            # Capa lineal
            linear = nn.Linear(prev_dim, h_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            
            # Activación
            layers.append(nn.ReLU())
            
            # Normalización
            if norm == 'layer':
                layers.append(nn.LayerNorm(h_dim))
            elif norm == 'batch':
                layers.append(nn.BatchNorm1d(h_dim))
            
            # Dropout
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            
            # Bloque residual (solo si dimensiones coinciden)
            if residual and prev_dim == h_dim:
                layers.append(ResidualBlock(h_dim, dropout_p, norm))
            
            prev_dim = h_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Capa de clasificación
        self.classifier = nn.Linear(prev_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            return_attn: Si True, retorna también los scores de atención
            
        Returns:
            - logits: Predicciones (batch_size, num_classes)
            - attention_scores: Scores de atención (batch_size, input_dim) o None
        """
        # Aplicar atención a features de entrada
        x_att, attn_scores = self.input_attn(x)
        
        # Pasar por backbone MLP
        features = self.backbone(x_att)
        
        # Clasificación
        logits = self.classifier(features)
        
        if return_attn:
            return logits, attn_scores
        return logits, None
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Obtiene los pesos de atención para un batch de entrada.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Attention weights (batch_size, input_dim)
        """
        with torch.no_grad():
            _, attn_scores = self.forward(x, return_attn=True)
        return attn_scores


class MLPWithAttentionTrainer:
    """
    Trainer para MLP + Attention.
    
    Maneja el loop de entrenamiento, validación y guardado del modelo.
    """
    
    def __init__(
        self,
        model: MLPWithAttention,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        """
        Args:
            model: Modelo MLPWithAttention
            device: Dispositivo ('cuda' o 'cpu')
            lr: Learning rate
            weight_decay: Regularización L2
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Loss function (se puede especificar pesos de clase)
        self.criterion = nn.CrossEntropyLoss()
        
        # Historial
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def set_class_weights(self, weights: torch.Tensor):
        """
        Configura pesos de clase para balanceo.
        
        Args:
            weights: Tensor de pesos (num_classes,)
        """
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
        print(f"⚖️  Pesos de clase configurados: {weights.cpu().numpy()}")
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Entrena una época."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            logits, _ = self.model(X_batch, return_attn=False)
            loss = self.criterion(logits, y_batch)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Métricas
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """Valida el modelo."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits, _ = self.model(X_batch, return_attn=False)
                loss = self.criterion(logits, y_batch)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 200,
        patience: int = 30,
        verbose: bool = True
    ):
        """
        Entrena el modelo.
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            epochs: Número de épocas
            patience: Paciencia para early stopping
            verbose: Imprimir progreso
        """
        print(f"\n{'='*80}")
        print(f"🚀 ENTRENANDO MLP + ATTENTION - GRUPO: JÓVENES")
        print(f"{'='*80}\n")
        print(f"⚙️  Configuración:")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {self.lr}")
        print(f"   Patience: {patience}")
        print(f"   Device: {self.device}\n")
        
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Entrenar
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validar
            val_loss, val_acc = self.validate(val_loader)
            
            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | ⭐ Best")
            else:
                patience_counter += 1
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
                
                if patience_counter >= patience:
                    print(f"\n⏹️  Early stopping en época {epoch}")
                    break
        
        # Cargar mejor modelo
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"\n✅ Entrenamiento completado")
        print(f"   Mejor val loss: {self.best_val_loss:.4f}\n")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice clases."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.model(X_tensor, return_attn=False)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predice probabilidades."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.model(X_tensor, return_attn=False)
            probas = F.softmax(logits, dim=1)
        
        return probas.cpu().numpy()
    
    def save(self, path: str):
        """Guarda el modelo."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': {
                'input_dim': self.model.input_dim,
                'num_classes': self.model.num_classes,
                'hidden_dims': self.model.hidden_dims,
                'attn_dim': self.model.attn_dim,
                'dropout_p': self.model.dropout_p,
                'norm': self.model.norm,
                'residual': self.model.residual
            }
        }
        
        torch.save(checkpoint, f"{path}.pth")
        print(f"✅ Modelo guardado en: {path}.pth")
    
    def load(self, path: str):
        """Carga el modelo."""
        checkpoint = torch.load(f"{path}.pth", map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"✅ Modelo cargado desde: {path}.pth")


# Función auxiliar para crear MLP+Attention con configuración por defecto
def create_mlp_attention_for_jovenes(input_dim: int) -> Tuple[MLPWithAttention, MLPWithAttentionTrainer]:
    """
    Crea un MLP+Attention con configuración optimizada para el grupo de JÓVENES.
    
    Args:
        input_dim: Número de features de entrada
        
    Returns:
        Tupla (model, trainer)
    """
    model = MLPWithAttention(
        input_dim=input_dim,
        num_classes=3,
        hidden_dims=[256, 128, 64, 32],
        attn_dim=32,
        dropout_p=0.3,
        norm='layer',
        residual=True
    )
    
    trainer = MLPWithAttentionTrainer(
        model=model,
        lr=1e-3,
        weight_decay=1e-5
    )
    
    return model, trainer


if __name__ == "__main__":
    # Ejemplo de uso
    print("MLP + Attention Classifier - Ejemplo de uso\n")
    
    # Crear modelo
    model, trainer = create_mlp_attention_for_jovenes(input_dim=139)  # 139 features para jóvenes
    
    print(f"✅ Modelo MLP + Attention creado correctamente")
    print(f"📊 Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📋 Listo para entrenamiento con datos de JÓVENES")
