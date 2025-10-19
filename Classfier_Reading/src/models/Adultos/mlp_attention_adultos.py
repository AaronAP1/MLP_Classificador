"""
MLP + Multi-Head Attention Classifier - Modelo para clasificación de ADULTOS (18+ años)

MLP con mecanismo de atención optimizado para el grupo de adultos.

Características específicas para ADULTOS:
- 71 features de entrada (menos que otros grupos)
- Arquitectura: 71 → [256, 128, 64, 32] → 3 clases
- CRÍTICO: Distribución MUY desbalanceada (20% / 0.02% / 80%)
- Requiere class weights obligatorio

Basado en: "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import numpy as np
import joblib
from pathlib import Path


class InputFeatureAttention(nn.Module):
    """Mecanismo de atención sobre features de entrada."""
    
    def __init__(self, input_dim: int, attn_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        
        self.query = nn.Linear(input_dim, attn_dim)
        self.key = nn.Linear(input_dim, attn_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        scores = torch.sum(q * k, dim=1, keepdim=True) / (self.attn_dim ** 0.5)
        attention_weights = torch.sigmoid(scores)
        
        attended = v * attention_weights + x * (1 - attention_weights)
        return attended, attention_weights.squeeze(1)


class ResidualBlock(nn.Module):
    """Bloque residual con conexión skip."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.linear(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out + identity


class MLPWithAttentionAdultos(nn.Module):
    """
    MLP + Attention para ADULTOS (18+ años).
    
    Arquitectura optimizada para 71 features.
    """
    
    def __init__(
        self,
        num_features: int = 71,
        num_classes: int = 3,
        hidden_dims: List[int] = [256, 128, 64, 32],
        attn_dim: int = 32,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Atención sobre features
        self.attention = InputFeatureAttention(num_features, attn_dim)
        
        # Capas MLP con residual connections
        layers = []
        in_dim = num_features
        
        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim
        
        self.mlp_layers = nn.ModuleList(layers)
        
        # Clasificación
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Atención
        x, attn_weights = self.attention(x)
        
        # MLP
        for layer in self.mlp_layers:
            x = layer(x)
        
        # Clasificar
        logits = self.classifier(x)
        return logits
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Obtener pesos de atención para interpretabilidad."""
        with torch.no_grad():
            _, attn_weights = self.attention(x)
        return attn_weights


class MLPAttentionAdultosTrainer:
    """Trainer para MLP+Attention de Adultos con soporte para class weights."""
    
    def __init__(
        self,
        model: MLPWithAttentionAdultos,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # CRÍTICO: class weights para adultos
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.best_val_loss = float('inf')
    
    def set_class_weights(self, class_weights: torch.Tensor):
        """Actualizar pesos de clase."""
        class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"✅ Class weights actualizados: {class_weights.cpu().numpy()}")
    
    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
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
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Best epoch: {best_epoch+1} | Best val loss: {self.best_val_loss:.4f}")
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probas = F.softmax(logits, dim=1)
        
        return probas.cpu().numpy()
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }, path)
        print(f"Modelo guardado en: {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Modelo cargado desde: {path}")


def create_mlp_attention_for_adultos(num_features: int = 71) -> Tuple[MLPWithAttentionAdultos, MLPAttentionAdultosTrainer]:
    """
    Función auxiliar para crear modelo MLP+Attention optimizado para Adultos.
    
    Args:
        num_features: Número de features (default=71 para adultos)
    
    Returns:
        Tupla (modelo, trainer)
    """
    model = MLPWithAttentionAdultos(
        num_features=num_features,
        num_classes=3,
        hidden_dims=[256, 128, 64, 32],
        attn_dim=32,
        dropout=0.3
    )
    
    trainer = MLPAttentionAdultosTrainer(
        model=model,
        lr=1e-3,
        weight_decay=1e-4
    )
    
    print("✅ MLP+Attention para ADULTOS creado")
    print(f"   Features: {num_features}")
    print(f"   Arquitectura: {num_features} → [256,128,64,32] → 3")
    print(f"   ⚠️  IMPORTANTE: Configurar class weights (distribución 20/0.02/80%)")
    print(f"   Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, trainer
