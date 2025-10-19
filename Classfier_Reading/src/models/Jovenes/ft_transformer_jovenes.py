"""
FT-Transformer Classifier - Modelo para clasificación de JÓVENES (12-17 años)

Feature Tokenizer Transformer adaptado para el grupo de jóvenes.

Características específicas para JÓVENES:
- 139 features de entrada
- Transformer con 4 capas, 8 heads
- CLS token para clasificación
- CRÍTICO: Distribución MUY desbalanceada (80% / 0.21% / 19%)
- Requiere class weights obligatorio

Basado en: Gorishniy et al. (2021) "Revisiting Deep Learning Models for Tabular Data"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
from pathlib import Path


class FeatureTokenizerJovenes(nn.Module):
    """Convierte cada feature tabular en un token embebido."""
    
    def __init__(self, num_features: int, emb_dim: int):
        super().__init__()
        self.num_features = num_features
        self.emb_dim = emb_dim
        
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, emb_dim) for _ in range(num_features)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        tokens = []
        for i in range(self.num_features):
            feature_val = x[:, i:i+1]
            token = self.feature_embeddings[i](feature_val)
            tokens.append(token)
        
        tokens = torch.stack(tokens, dim=1)
        return tokens


class FTTransformerJovenes(nn.Module):
    """
    FT-Transformer para JÓVENES (12-17 años).
    
    Arquitectura optimizada para 139 features con distribución desbalanceada.
    """
    
    def __init__(
        self,
        num_features: int = 139,
        num_classes: int = 3,
        emb_dim: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        super().__init__()
        
        assert emb_dim % n_heads == 0, "emb_dim debe ser divisible por n_heads"
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        
        self.tokenizer = FeatureTokenizerJovenes(num_features, emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.norm = nn.LayerNorm(emb_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        tokens = self.tokenizer(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        tokens = self.transformer(tokens)
        cls_output = tokens[:, 0, :]
        cls_output = self.norm(cls_output)
        
        logits = self.classifier(cls_output)
        return logits
    
    def get_cls_embedding(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        with torch.no_grad():
            tokens = self.tokenizer(x)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
            tokens = self.transformer(tokens)
            cls_output = self.norm(tokens[:, 0, :])
        
        return cls_output


class FTTransformerJovenesTrainer:
    """Trainer para FT-Transformer de Jóvenes con soporte para class weights."""
    
    def __init__(
        self,
        model: FTTransformerJovenes,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10,
        class_weights: Optional[torch.Tensor] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.base_lr = lr
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # CRÍTICO: class weights para jóvenes
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.best_val_loss = float('inf')
        self.current_epoch = 0
    
    def set_class_weights(self, class_weights: torch.Tensor):
        """Actualizar pesos de clase."""
        class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"✅ Class weights actualizados: {class_weights.cpu().numpy()}")
    
    def _get_warmup_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        return self.base_lr
    
    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0
        
        if self.current_epoch < self.warmup_epochs:
            lr = self._get_warmup_lr(self.current_epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
            self.current_epoch = epoch
            
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if epoch >= self.warmup_epochs:
                self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"LR: {current_lr:.6f} | "
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


def create_ft_transformer_for_jovenes(num_features: int = 139) -> Tuple[FTTransformerJovenes, FTTransformerJovenesTrainer]:
    """
    Función auxiliar para crear FT-Transformer optimizado para Jóvenes.
    
    Args:
        num_features: Número de features (default=139 para jóvenes)
    
    Returns:
        Tupla (modelo, trainer)
    """
    model = FTTransformerJovenes(
        num_features=num_features,
        num_classes=3,
        emb_dim=64,
        n_heads=8,
        n_layers=4,
        dim_feedforward=256,
        dropout=0.2
    )
    
    trainer = FTTransformerJovenesTrainer(
        model=model,
        lr=1e-4,
        weight_decay=1e-5,
        warmup_epochs=10
    )
    
    print("✅ FT-Transformer para JÓVENES creado")
    print(f"   Features: {num_features}")
    print(f"   Arquitectura: Transformer (emb_dim=64, heads=8, layers=4)")
    print(f"   ⚠️  IMPORTANTE: Configurar class weights (distribución 80/0.21/19%)")
    print(f"   Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, trainer
