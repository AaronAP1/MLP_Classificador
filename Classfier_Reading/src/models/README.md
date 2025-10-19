# 📊 Modelos de Clasificación - Organizados por Grupo Etario

## 📁 Estructura

```
src/models/
├── __init__.py                    # Imports principales
├── Ninos/                         # 👶 NIÑOS (0-11 años)
│   ├── __init__.py
│   ├── tabnet_classifier.py       # 🎯 TabNet (400+ líneas)
│   └── README_TABNET.md
├── Jovenes/                       # 👦 JÓVENES (12-17 años)
│   ├── __init__.py
│   ├── mlp_attention.py           # 🧠 MLP + Attention (600+ líneas)
│   └── README_MLP_ATTENTION.md
└── Adultos/                       # 👨 ADULTOS (18+ años)
    ├── __init__.py
    ├── ft_transformer.py          # 🤖 FT-Transformer (700+ líneas)
    └── README_FT_TRANSFORMER.md
```

---

## 🎯 Modelos por Grupo Etario

### 👶 Niños (0-11 años) - `models/Ninos/`

**Modelo:** TabNet Classifier  
**Features:** 131 dimensiones  
**Registros:** 10,388

**Arquitectura:**
- Atención secuencial interpretable
- Decision steps: 5
- n_d = n_a = 64
- Masking con entmax

**Ventajas:**
- ✅ Interpretabilidad (feature importance)
- ✅ Eficiente con features categóricas
- ✅ State-of-the-art en tabular data

**Uso:**
```python
from src.models.Ninos import create_tabnet_for_ninos

model, trainer = create_tabnet_for_ninos(num_features=131)
```

📖 **Documentación:** [README_TABNET.md](Ninos/README_TABNET.md)

---

### 👦 Jóvenes (12-17 años) - `models/Jovenes/`

**Modelo:** MLP + Multi-Head Attention  
**Features:** 139 dimensiones  
**Registros:** 14,021

**Arquitectura:**
- Input Feature Attention
- Residual Blocks
- Hidden layers: [256, 128, 64, 32]
- Attention dimension: 32

**Desafío:**
- ⚠️ Distribución muy desbalanceada (80% / 0.21% / 19%)
- Requiere balanceo de clases (WeightedRandomSampler)

**Ventajas:**
- ✅ Attention sobre features de entrada
- ✅ Skip connections para gradientes
- ✅ Soporte robusto para class weights

**Uso:**
```python
from src.models.Jovenes import create_mlp_attention_for_jovenes

model, trainer = create_mlp_attention_for_jovenes(num_features=139)
```

📖 **Documentación:** [README_MLP_ATTENTION.md](Jovenes/README_MLP_ATTENTION.md)

---

### 👨 Adultos (18+ años) - `models/Adultos/`

**Modelo:** FT-Transformer (Feature Tokenizer Transformer)  
**Features:** 71 dimensiones  
**Registros:** 40,559

**Arquitectura:**
- Feature Tokenization (cada feature → embedding)
- CLS Token (como BERT)
- 4 Transformer Encoder Layers
- 8 Multi-Head Attention heads
- emb_dim = 64

**Desafío:**
- ⚠️ Distribución muy desbalanceada (20% / 0.02% / 80%)
- Requiere balanceo de clases

**Ventajas:**
- ✅ State-of-the-art en tabular data
- ✅ Captura interacciones complejas
- ✅ Warmup LR + Gradient Clipping
- ✅ Embeddings interpretables (CLS token)

**Uso:**
```python
from src.models.Adultos import create_ft_transformer_for_adultos

model, trainer = create_ft_transformer_for_adultos(num_features=71)
```

📖 **Documentación:** [README_FT_TRANSFORMER.md](Adultos/README_FT_TRANSFORMER.md)

---

## 🚀 Guía de Uso Rápida

### Opción 1: Import desde subcarpetas (RECOMENDADO)

```python
# Importar modelo específico para niños
from src.models.Ninos import TabNetClassifier, create_tabnet_for_ninos

# Importar modelo específico para jóvenes
from src.models.Jovenes import MLPWithAttention, create_mlp_attention_for_jovenes

# Importar modelo específico para adultos
from src.models.Adultos import FTTransformerClassifier, create_ft_transformer_for_adultos
```

### Opción 2: Import desde módulo principal

```python
# Todos los modelos accesibles desde src.models
from src.models import (
    # Niños
    TabNetClassifier,
    create_tabnet_for_ninos,
    
    # Jóvenes
    MLPWithAttention,
    MLPWithAttentionTrainer,
    create_mlp_attention_for_jovenes,
    
    # Adultos
    FTTransformerClassifier,
    FTTransformerTrainer,
    create_ft_transformer_for_adultos
)
```

---

## 📊 Comparación de Modelos

| Aspecto | Niños (TabNet) | Jóvenes (MLP+Attn) | Adultos (FT-Transformer) |
|---------|----------------|---------------------|--------------------------|
| **Features** | 131 | 139 | 71 |
| **Registros** | 10,388 | 14,021 | 40,559 |
| **Arquitectura** | Sequential Attention | MLP + Input Attention | Transformer |
| **Parámetros** | ~100K | ~150K | ~250K |
| **Interpretabilidad** | ⭐⭐⭐⭐⭐ Alta | ⭐⭐⭐ Media | ⭐⭐⭐ Media |
| **Complejidad** | ⭐⭐ Baja | ⭐⭐⭐ Media | ⭐⭐⭐⭐ Alta |
| **Balance de Clases** | ✅ Balanceado | ⚠️ Muy desbalanceado | ⚠️ Muy desbalanceado |
| **Ventaja Principal** | Feature Importance | Skip Connections | Self-Attention |

---

## 🎓 Para la Tesis

### Justificación de la Organización

> "Los modelos fueron organizados en carpetas separadas por grupo etario (Niños, Jóvenes, Adultos) 
> para facilitar el mantenimiento, la experimentación independiente y reflejar la estructura natural 
> de los datos. Cada grupo etario presenta características únicas en términos de:
> 
> - **Dimensionalidad:** Diferentes cantidades de features (131, 139, 71)
> - **Distribución de clases:** Desde balanceado (niños) hasta extremadamente desbalanceado (adultos)
> - **Complejidad de patrones:** Requieren arquitecturas especializadas
> 
> Esta organización modular permite entrenar, evaluar y optimizar cada modelo independientemente, 
> facilitando la comparación mediante el algoritmo de Copeland."

---

## 📈 Próximos Pasos

1. ✅ Modelos implementados y organizados
2. ⏭️ Crear scripts de entrenamiento en `experiments/`
3. ⏭️ Entrenar cada modelo con sus datos correspondientes
4. ⏭️ Evaluar con métricas completas
5. ⏭️ Comparar usando Copeland
6. ⏭️ Análisis de interpretabilidad

---

## 🔧 Desarrollo

### Agregar un nuevo modelo

Para agregar un modelo a un grupo existente:

1. Crear archivo en la carpeta correspondiente (e.g., `Ninos/nuevo_modelo.py`)
2. Implementar clase con métodos: `fit()`, `predict()`, `predict_proba()`, `save()`, `load()`
3. Agregar import en `Ninos/__init__.py`
4. Actualizar `models/__init__.py` si se quiere acceso directo
5. Documentar en README correspondiente

### Crear nuevo grupo etario

Si necesitas un nuevo grupo (e.g., "Ancianos"):

1. Crear carpeta `models/Ancianos/`
2. Crear `Ancianos/__init__.py`
3. Implementar modelo en `Ancianos/modelo.py`
4. Agregar import en `models/__init__.py`
5. Documentar en nuevo `README.md`

---

## 📚 Referencias

- **TabNet:** Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. AAAI.
- **MLP + Attention:** Vaswani et al. (2017). Attention Is All You Need. NeurIPS.
- **FT-Transformer:** Gorishniy et al. (2021). Revisiting Deep Learning Models for Tabular Data. NeurIPS.

---

**✅ 3/3 MODELOS ORGANIZADOS POR GRUPO ETARIO** 🎉

Total: ~1,700 líneas de código + documentación completa
