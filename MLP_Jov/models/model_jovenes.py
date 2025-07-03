import torch, torch.nn as nn

from utils.attention import FeatureAttention

class JovenesMLP(nn.Module):
    """
    MLP de 3 capas ocultas + ReLU → atención → capa de salida.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list = [128, 64, 32],
                 attn_dim: int = 16,
                 num_classes: int = 3,
                 dropout_p=0.2):        
        super().__init__()
        # 1) Bloque MLP
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p=dropout_p)]
            prev = h
        self.mlp = nn.Sequential(*layers)
        # 2) Atención
        self.attn = FeatureAttention(feature_dim=hidden_dims[-1],
                                     attn_dim=attn_dim)
        # 3) Salida
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp(x)            # → (B, hidden_dims[-1])
        h = self.attn(h)           # → (B, hidden_dims[-1])
        logits = self.classifier(h)
        return logits              # raw logits para CrossEntropyLoss
