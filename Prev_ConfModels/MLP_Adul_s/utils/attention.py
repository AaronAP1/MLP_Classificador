import torch, torch.nn as nn

class FeatureAttention(nn.Module):
    """
    Atención canal-a-canal:
      scores = softmax( V·tanh(W·h) )     → (B, F)
      out    = scores * h                  → (B, F)
    """
    def __init__(self, feature_dim: int, attn_dim: int):
        super().__init__()
        self.W = nn.Linear(feature_dim, attn_dim, bias=True)
        self.V = nn.Linear(attn_dim, feature_dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        e      = torch.tanh(self.W(h))            # (B, attn_dim)
        scores = torch.softmax(self.V(e), dim=1)  # (B, feature_dim)
        return scores * h                         # (B, feature_dim)
