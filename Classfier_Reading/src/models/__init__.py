"""
Módulo de modelos de clasificación.

Contiene las 3 arquitecturas principales:
- TabNet: Para clasificación de Niños
- MLP + Multi-Head Attention: Para clasificación de Jóvenes  
- FT-Transformer: Para clasificación de Adultos
"""

from .tabnet_classifier import TabNetClassifier
from .mlp_attention import MLPWithAttention
from .ft_transformer import FTTransformerClassifier

__all__ = [
    'TabNetClassifier',
    'MLPWithAttention',
    'FTTransformerClassifier'
]
