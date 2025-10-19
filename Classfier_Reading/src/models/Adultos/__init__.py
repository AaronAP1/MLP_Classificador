"""
Modelos para clasificación de comportamiento lector - ADULTOS (18+ años)

Este módulo contiene los 3 modelos especializados para el grupo etario de adultos:
- TabNet Classifier: Modelo con atención interpretable
- MLP + Attention: Red neuronal profunda con atención
- FT-Transformer: Feature Tokenizer Transformer

Features: 71 dimensiones
Distribución: MUY desbalanceada (20/0.02/80%) ⚠️
Consideración especial: Clase 1 casi inexistente (0.02%)
"""

# TabNet
from .tabnet_adultos import (
    TabNetAdultos,
    create_tabnet_for_adultos
)

# MLP + Attention
from .mlp_attention_adultos import (
    MLPWithAttentionAdultos,
    MLPAttentionAdultosTrainer,
    create_mlp_attention_for_adultos
)

# FT-Transformer (original)
from .ft_transformer import (
    FeatureTokenizer,
    FTTransformerClassifier,
    FTTransformerTrainer,
    create_ft_transformer_for_adultos
)

__all__ = [
    # TabNet
    'TabNetAdultos',
    'create_tabnet_for_adultos',
    
    # MLP + Attention
    'MLPWithAttentionAdultos',
    'MLPAttentionAdultosTrainer',
    'create_mlp_attention_for_adultos',
    
    # FT-Transformer
    'FeatureTokenizer',
    'FTTransformerClassifier',
    'FTTransformerTrainer',
    'create_ft_transformer_for_adultos',
]
