"""
FT-Transformer Classifier - Modelo para clasificación de ADULTOS (18+ años)

FT-Transformer (Feature Tokenizer Transformer) es una arquitectura transformer
adaptada específicamente para datos tabulares. Convierte cada feature (numérica
o categórica) en un token embebido y aplica self-attention estándar.

Características:
- Feature Tokenization: Cada feature → embedding independiente
- CLS Token: Token especial para clasificación (como BERT)
- Self-Attention: Captura interacciones entre features
- Position-agnostic: No requiere orden específico de features

Paper: "Revisiting Deep Learning Models for Tabular Data"
       https://arxiv.org/abs/2106.11959
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np
from pathlib import Path


class FeatureTokenizer(nn.Module):
    """
    Convierte features tabulares en tokens embebidos.
    
    Cada feature (numérica) se convierte en un vector de dimensión emb_dim.
    """
    
    def __init__(self, num_features: int, emb_dim: int):
        """
        Args:
            num_features: Número de features numéricas
            emb_dim: Dimensión del embedding
        """
        super().__init__()
        self.num_features = num_features
        self.emb_dim = emb_dim
        
        # Una proyección lineal por cada feature
        self.linear_layers = nn.ModuleList([
            nn.Linear(1, emb_dim) for _ in range(num_features)
        ])
        
        # Inicialización
        for layer in self.linear_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, num_features)
            
        Returns:
            Tokens embebidos (batch_size, num_features, emb_dim)
        """
        # Procesar cada feature independientemente
        tokens = []
        for i, layer in enumerate(self.linear_layers):
            # Extraer feature i y expandir dimensión
            feature_i = x[:, i:i+1]  # (batch_size, 1)
            # Proyectar a emb_dim
            token_i = layer(feature_i)  # (batch_size, emb_dim)
            tokens.append(token_i)
        
        # Apilar tokens
        tokens = torch.stack(tokens, dim=1)  # (batch_size, num_features, emb_dim)
        return tokens


class FTTransformerClassifier(nn.Module):
    """
    FT-Transformer para clasificación tabular.
    
    Diseñado específicamente para el grupo de ADULTOS (18+ años).
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int = 3,
        emb_dim: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        """
        Args:
            num_features: Número de features de entrada
            num_classes: Número de clases de salida
            emb_dim: Dimensión de embeddings (debe ser divisible por n_heads)
            n_heads: Número de cabezas de atención
            n_layers: Número de capas transformer
            dim_feedforward: Dimensión de la capa feedforward
            dropout: Probabilidad de dropout
            activation: Función de activación ('relu' o 'gelu')
        """
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        assert emb_dim % n_heads == 0, f"emb_dim ({emb_dim}) debe ser divisible por n_heads ({n_heads})"
        
        # Feature Tokenizer
        self.tokenizer = FeatureTokenizer(num_features, emb_dim)
        
        # CLS Token (token especial para clasificación)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # Importante: (batch, seq, feature)
            norm_first=False
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Layer Normalization final
        self.norm = nn.LayerNorm(emb_dim)
        
        # Cabeza de clasificación
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, num_classes)
        )
        
        # Inicialización de la cabeza
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, num_features)
            return_attention: Si True, retorna mapas de atención
            
        Returns:
            - logits: Predicciones (batch_size, num_classes)
            - cls_embedding: Embedding del CLS token (batch_size, emb_dim) o None
        """
        batch_size = x.size(0)
        
        # Tokenizar features
        tokens = self.tokenizer(x)  # (batch_size, num_features, emb_dim)
        
        # Agregar CLS token al inicio
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, emb_dim)
        tokens = torch.cat([cls, tokens], dim=1)  # (batch_size, num_features+1, emb_dim)
        
        # Pasar por transformer
        out = self.transformer(tokens)  # (batch_size, num_features+1, emb_dim)
        
        # Extraer CLS token (primera posición)
        cls_out = out[:, 0, :]  # (batch_size, emb_dim)
        
        # Normalizar
        cls_out = self.norm(cls_out)
        
        # Clasificar
        logits = self.head(cls_out)  # (batch_size, num_classes)
        
        if return_attention:
            return logits, cls_out
        return logits, None
    
    def get_cls_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Obtiene el embedding del CLS token (representación aprendida).
        
        Args:
            x: Input tensor (batch_size, num_features)
            
        Returns:
            CLS embeddings (batch_size, emb_dim)
        """
        with torch.no_grad():
            _, cls_emb = self.forward(x, return_attention=True)
        return cls_emb


class FTTransformerTrainer:
    """
    Trainer para FT-Transformer.
    
    Maneja entrenamiento, validación, guardado y carga del modelo.
    """
    
    def __init__(
        self,
        model: FTTransformerClassifier,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10
    ):
        """
        Args:
            model: Modelo FTTransformerClassifier
            device: Dispositivo ('cuda' o 'cpu')
            lr: Learning rate
            weight_decay: Regularización L2
            warmup_epochs: Épocas de warmup para learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        
        # Optimizer (AdamW es mejor para transformers)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Scheduler con warmup
        self.scheduler = None  # Se inicializa en fit()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Historial
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
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
    
    def _get_warmup_lr(self, epoch: int) -> float:
        """Calcula learning rate con warmup."""
        if epoch < self.warmup_epochs:
            return self.lr * (epoch + 1) / self.warmup_epochs
        return self.lr
    
    def train_epoch(self, train_loader, epoch: int) -> Tuple[float, float]:
        """Entrena una época."""
        self.model.train()
        
        # Ajustar learning rate con warmup
        if epoch < self.warmup_epochs:
            lr = self._get_warmup_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            logits, _ = self.model(X_batch, return_attention=False)
            loss = self.criterion(logits, y_batch)
            
            # Backward
            loss.backward()
            
            # Gradient clipping (importante para transformers)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
                
                logits, _ = self.model(X_batch, return_attention=False)
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
        print(f"🚀 ENTRENANDO FT-TRANSFORMER - GRUPO: ADULTOS")
        print(f"{'='*80}\n")
        print(f"⚙️  Configuración:")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {self.lr}")
        print(f"   Warmup epochs: {self.warmup_epochs}")
        print(f"   Patience: {patience}")
        print(f"   Device: {self.device}")
        print(f"   Embedding dim: {self.model.emb_dim}")
        print(f"   Attention heads: {self.model.n_heads}")
        print(f"   Transformer layers: {self.model.n_layers}\n")
        
        # Inicializar scheduler (después del warmup)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Entrenar
            train_loss, train_acc = self.train_epoch(train_loader, epoch - 1)
            
            # Validar
            val_loss, val_acc = self.validate(val_loader)
            
            # Obtener learning rate actual
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Scheduler (solo después del warmup)
            if epoch > self.warmup_epochs:
                self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:3d} | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | ⭐ Best")
            else:
                patience_counter += 1
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:3d} | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
                
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
            logits, _ = self.model(X_tensor, return_attention=False)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predice probabilidades."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.model(X_tensor, return_attention=False)
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
                'num_features': self.model.num_features,
                'num_classes': self.model.num_classes,
                'emb_dim': self.model.emb_dim,
                'n_heads': self.model.n_heads,
                'n_layers': self.model.n_layers
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


# Función auxiliar para crear FT-Transformer con configuración por defecto
def create_ft_transformer_for_adultos(num_features: int) -> Tuple[FTTransformerClassifier, FTTransformerTrainer]:
    """
    Crea un FT-Transformer con configuración optimizada para el grupo de ADULTOS.
    
    Args:
        num_features: Número de features de entrada
        
    Returns:
        Tupla (model, trainer)
    """
    model = FTTransformerClassifier(
        num_features=num_features,
        num_classes=3,
        emb_dim=64,
        n_heads=8,
        n_layers=4,
        dim_feedforward=256,
        dropout=0.2,
        activation='relu'
    )
    
    trainer = FTTransformerTrainer(
        model=model,
        lr=1e-4,
        weight_decay=1e-5,
        warmup_epochs=10
    )
    
    return model, trainer


if __name__ == "__main__":
    # Ejemplo de uso
    print("FT-Transformer Classifier - Ejemplo de uso\n")
    
    # Crear modelo
    model, trainer = create_ft_transformer_for_adultos(num_features=71)  # 71 features para adultos
    
    print(f"✅ Modelo FT-Transformer creado correctamente")
    print(f"📊 Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📋 Listo para entrenamiento con datos de ADULTOS")
