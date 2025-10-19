"""
Módulo de modelos de clasificación - Organizados por Grupo Etario

Estructura:
├── Ninos/     - Modelos para niños (0-11 años)
│   └── TabNet Classifier (131 features)
├── Jovenes/   - Modelos para jóvenes (12-17 años)
│   └── MLP + Multi-Head Attention (139 features)
└── Adultos/   - Modelos para adultos (18+ años)
    └── FT-Transformer (71 features)

Uso:
    # Importar modelos para niños
    from src.models.Ninos import TabNetClassifier, create_tabnet_for_ninos
    
    # Importar modelos para jóvenes
    from src.models.Jovenes import MLPWithAttention, create_mlp_attention_for_jovenes
    
    # Importar modelos para adultos
    from src.models.Adultos import FTTransformerClassifier, create_ft_transformer_for_adultos
"""

# Importar todos los modelos para acceso directo
from .Ninos import TabNetClassifier, create_tabnet_for_ninos
from .Jovenes import (
    MLPWithAttention, 
    MLPWithAttentionTrainer,
    create_mlp_attention_for_jovenes
)
from .Adultos import (
    FTTransformerClassifier,
    FTTransformerTrainer,
    create_ft_transformer_for_adultos
)

__all__ = [
    # Niños
    'TabNetClassifier',
    'create_tabnet_for_ninos',
    
    # Jóvenes
    'MLPWithAttention',
    'MLPWithAttentionTrainer',
    'create_mlp_attention_for_jovenes',
    
    # Adultos
    'FTTransformerClassifier',
    'FTTransformerTrainer',
    'create_ft_transformer_for_adultos',
]
