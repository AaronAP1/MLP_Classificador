import torch, torch.nn as nn
from typing import Tuple
from utils.attention import InputFeatureAttention

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, norm: str = None):
        super().__init__()
        layers = [nn.Linear(dim, dim), nn.ReLU()]
        if norm == 'layer':
            layers.append(nn.LayerNorm(dim))
        elif norm == 'batch':
            layers.append(nn.BatchNorm1d(dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return x + self.net(x)

class MLPWithInputAttention(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 hidden_dims = [256,128,64,32],
                 attn_dim: int = 32,
                 dropout_p: float = 0.3,
                 norm: str = 'layer',
                 residual: bool = False):
        super().__init__()
        self.input_attn = InputFeatureAttention(input_dim, attn_dim)
        layers = []
        prev = input_dim
        for h in hidden_dims:
            lin = nn.Linear(prev, h)
            nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
            layers += [lin, nn.ReLU()]
            if norm == 'layer':
                layers += [nn.LayerNorm(h)]
            elif norm == 'batch':
                layers += [nn.BatchNorm1d(h)]
            if dropout_p > 0:
                layers += [nn.Dropout(dropout_p)]
            if residual and prev == h:
                layers += [ResidualBlock(h, dropout_p, norm)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight); nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x_att, scores = self.input_attn(x)
        h = self.backbone(x_att)
        logits = self.classifier(h)
        if return_attn:
            return logits, scores
        return logits, None
