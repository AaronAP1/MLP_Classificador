"""
Modelos para clasificación de comportamiento lector - JÓVENES (12-17 años)

Este módulo contiene los 3 modelos especializados para el grupo etario de jóvenes:
- TabNet Classifier: Modelo con atención interpretable
- MLP + Multi-Head Attention: Red neuronal profunda con atención
- FT-Transformer: Feature Tokenizer Transformer

Features: 139 dimensiones
Distribución: MUY desbalanceada (80/0.21/19%) ⚠️
Consideración especial: Requiere class weights obligatorio
"""

# TabNet
from .tabnet_jovenes import (
    TabNetJovenes,
    create_tabnet_for_jovenes
)

# MLP + Attention (original)
from .mlp_attention import (
    InputFeatureAttention,
    ResidualBlock,
    MLPWithAttention,
    MLPWithAttentionTrainer,
    create_mlp_attention_for_jovenes
)

# FT-Transformer
from .ft_transformer_jovenes import (
    FTTransformerJovenes,
    FTTransformerJovenesTrainer,
    create_ft_transformer_for_jovenes
)

__all__ = [
    # TabNet
    'TabNetJovenes',
    'create_tabnet_for_jovenes',
    
    # MLP + Attention
    'InputFeatureAttention',
    'ResidualBlock',
    'MLPWithAttention',
    'MLPWithAttentionTrainer',
    'create_mlp_attention_for_jovenes',
    
    # FT-Transformer
    'FTTransformerJovenes',
    'FTTransformerJovenesTrainer',
    'create_ft_transformer_for_jovenes',
]
