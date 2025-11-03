"""
FT-Transformer (Feature Tokenizer Transformer) para datos tabulares.
Basado en: https://arxiv.org/abs/2106.11959
"""
import torch
import torch.nn as nn
import math


class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer para clasificación tabular."""
    
    def __init__(self, input_dim: int, d_token: int = 192, n_blocks: int = 3,
                 attention_n_heads: int = 8, attention_dropout: float = 0.2,
                 ffn_d_hidden: int = 512, ffn_dropout: float = 0.1,
                 residual_dropout: float = 0.0, num_classes: int = 3):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_token = d_token
        
        # Feature Tokenizer: cada feature se convierte en un token
        self.feature_tokenizer = nn.Linear(1, d_token)
        
        # CLS token (para clasificación)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Positional embeddings
        self.pos_embeddings = nn.Parameter(torch.randn(1, input_dim + 1, d_token))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_token=d_token,
                n_heads=attention_n_heads,
                attention_dropout=attention_dropout,
                ffn_d_hidden=ffn_d_hidden,
                ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout
            )
            for _ in range(n_blocks)
        ])
        
        # Head de clasificación
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, num_classes)
        )
    
    def forward(self, x):
        # x: (batch_size, input_dim)
        batch_size = x.shape[0]
        
        # Tokenizar features: cada feature se procesa individualmente
        x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
        tokens = self.feature_tokenizer(x)  # (batch_size, input_dim, d_token)
        
        # Agregar CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (batch_size, input_dim+1, d_token)
        
        # Agregar positional embeddings
        tokens = tokens + self.pos_embeddings
        
        # Pasar por bloques transformer
        for block in self.blocks:
            tokens = block(tokens)
        
        # Tomar el CLS token para clasificación
        cls_output = tokens[:, 0]  # (batch_size, d_token)
        
        # Head de clasificación
        logits = self.head(cls_output)
        
        return logits


class TransformerBlock(nn.Module):
    """Bloque Transformer estándar."""
    
    def __init__(self, d_token: int, n_heads: int, attention_dropout: float,
                 ffn_d_hidden: int, ffn_dropout: float, residual_dropout: float):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        self.attention_dropout = nn.Dropout(residual_dropout)
        self.attention_norm = nn.LayerNorm(d_token)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_token, ffn_d_hidden),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_d_hidden, d_token),
            nn.Dropout(residual_dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_token)
    
    def forward(self, x):
        # Attention block
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.attention_dropout(attn_out)
        x = self.attention_norm(x + attn_out)
        
        # FFN block
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)
        
        return x
