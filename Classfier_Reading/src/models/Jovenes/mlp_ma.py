import torch
import torch.nn as nn


class MLPWithAttention(nn.Module):
    """MLP con Mecanismo de Atención a nivel de entrada."""
    
    def __init__(self, input_dim: int, hidden_dims: list, attn_dim: int, 
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        
        # Mecanismo de atención
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, input_dim),
            nn.Softmax(dim=1)
        )
        
        # MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Capa de salida
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        # Aplicar atención
        attn_weights = self.attention(x)
        x_attended = x * attn_weights
        
        # MLP
        logits = self.mlp(x_attended)
        
        return logits, attn_weights
