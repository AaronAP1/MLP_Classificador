"""
Modelos para clasificación de comportamiento lector - NIÑOS (0-11 años)

Este módulo contiene los 3 modelos especializados para el grupo etario de niños:
- TabNet Classifier: Modelo con atención interpretable
- MLP + Attention: Red neuronal profunda con atención
- FT-Transformer: Feature Tokenizer Transformer

Features: 131 dimensiones
Distribución: Balanceada ✅
"""

# TabNet
from .tabnet_classifier import (
    TabNetClassifier,
    create_tabnet_for_ninos
)

# MLP + Attention
from .mlp_attention_ninos import (
    MLPWithAttentionNinos,
    MLPAttentionNinosTrainer,
    create_mlp_attention_for_ninos
)

# FT-Transformer
from .ft_transformer_ninos import (
    FTTransformerNinos,
    FTTransformerNinosTrainer,
    create_ft_transformer_for_ninos
)

__all__ = [
    # TabNet
    'TabNetClassifier',
    'create_tabnet_for_ninos',
    
    # MLP + Attention
    'MLPWithAttentionNinos',
    'MLPAttentionNinosTrainer',
    'create_mlp_attention_for_ninos',
    
    # FT-Transformer
    'FTTransformerNinos',
    'FTTransformerNinosTrainer',
    'create_ft_transformer_for_ninos',
]
