import torch, torch.nn as nn

class InputFeatureAttention(nn.Module):
    """
    Atención a nivel de entrada (canal a canal) para interpretabilidad directa
    sobre las variables originales ya preprocesadas.
    out = softmax(V tanh(W x)) * x
    """
    def __init__(self, feature_dim: int, attn_dim: int):
        super().__init__()
        self.W = nn.Linear(feature_dim, attn_dim, bias=True)
        self.V = nn.Linear(attn_dim, feature_dim, bias=False)

    def forward(self, x: torch.Tensor):
        e = torch.tanh(self.W(x))
        scores = torch.softmax(self.V(e), dim=1)
        return scores * x, scores
