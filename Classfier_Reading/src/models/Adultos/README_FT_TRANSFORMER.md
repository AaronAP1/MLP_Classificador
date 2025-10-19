# 🤖 FT-Transformer Classifier - ADULTOS (18+ años)

## ✅ MODELO 3/3 COMPLETADO

**Archivo:** `ft_transformer.py`  
**Grupo objetivo:** Adultos (18+ años)  
**Features:** 71 dimensiones

---

## 🎉 ¡TODOS LOS MODELOS COMPLETADOS!

1. ✅ **TabNet** → Niños (131 features)
2. ✅ **MLP + Attention** → Jóvenes (139 features)
3. ✅ **FT-Transformer** → Adultos (71 features)

---

## 🎯 Descripción

**FT-Transformer** (Feature Tokenizer Transformer) es una arquitectura transformer adaptada para datos tabulares. A diferencia de transformers tradicionales para texto/imágenes, FT-Transformer convierte cada **feature tabular en un token embebido** y aplica self-attention para capturar interacciones entre features.

### Características Clave

✅ **Feature Tokenization**
   - Cada feature → embedding independiente de dimensión `emb_dim`
   - Permite al transformer tratar datos tabulares como secuencias

✅ **CLS Token (como BERT)**
   - Token especial de clasificación
   - Agregador de información de todas las features

✅ **Self-Attention Multi-Head**
   - Captura relaciones complejas entre features
   - 8 cabezas de atención

✅ **Position-Agnostic**
   - No depende del orden de features
   - Más robusto que modelos secuenciales

---

## 🏗️ Arquitectura

```
Input (71 features)
    ↓
[FeatureTokenizer]
    ↓
71 tokens × 64 dim
    ↓
[CLS Token prepended]
    ↓
(1 + 71) tokens × 64 dim
    ↓
[TransformerEncoder Layer 1]
  ├─ Multi-Head Attention (8 heads)
  ├─ Add & Norm
  ├─ FeedForward (64 → 256 → 64)
  └─ Add & Norm
    ↓
[TransformerEncoder Layer 2]
  └─ (same structure)
    ↓
[TransformerEncoder Layer 3]
  └─ (same structure)
    ↓
[TransformerEncoder Layer 4]
  └─ (same structure)
    ↓
[Extract CLS token]
    ↓
[LayerNorm]
    ↓
[Classifier Head: 64 → 32 → 3]
    ↓
Output (3 clases)
```

**Componentes:**
- **FeatureTokenizer**: 71 × Linear(1 → 64)
- **CLS Token**: Learnable parameter (1 × 64)
- **Transformer**: 4 layers, 8 heads, dim=64
- **FeedForward**: dim_ff=256
- **Classifier**: [64 → 32 → 3]

---

## 📝 Hiperparámetros

```python
FTTransformerClassifier(
    num_features=71,        # Features para adultos
    num_classes=3,          # 3 clases de comportamiento
    emb_dim=64,             # Dimensión de embeddings
    n_heads=8,              # 8 cabezas de atención
    n_layers=4,             # 4 capas transformer
    dim_feedforward=256,    # Dimensión feedforward
    dropout=0.2,            # Dropout
    activation='relu'       # Activación en FFN
)

FTTransformerTrainer(
    model=model,
    lr=1e-4,                # Learning rate bajo (típico de transformers)
    weight_decay=1e-5,      # L2 regularization
    warmup_epochs=10        # Warmup de LR
)
```

---

## 🚀 Uso

### Crear Modelo

```python
from src.models.ft_transformer import create_ft_transformer_for_adultos

# Crear modelo y trainer
model, trainer = create_ft_transformer_for_adultos(num_features=71)

# Ver parámetros
total_params = sum(p.numel() for p in model.parameters())
print(f"Parámetros totales: {total_params:,}")
```

### Entrenar

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Cargar datos
train_df = pd.read_csv("data/processed/adultos/adultos_train.csv")
val_df = pd.read_csv("data/processed/adultos/adultos_val.csv")

X_train = torch.FloatTensor(train_df.drop(columns=['target']).values)
y_train = torch.LongTensor(train_df['target'].values)
X_val = torch.FloatTensor(val_df.drop(columns=['target']).values)
y_val = torch.LongTensor(val_df['target'].values)

# Crear DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Entrenar
trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=200,
    patience=30,
    verbose=True
)
```

### Entrenar con Balanceo

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Calcular pesos
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train.numpy()),
    y=y_train.numpy()
)
class_weights_tensor = torch.FloatTensor(class_weights)

# Configurar pesos
trainer.set_class_weights(class_weights_tensor)

# Entrenar
trainer.fit(train_loader, val_loader, epochs=200)
```

### Predecir

```python
# Predecir clases
y_pred = trainer.predict(X_test.numpy())

# Predecir probabilidades
y_proba = trainer.predict_proba(X_test.numpy())
```

### Extraer Embeddings (CLS Token)

```python
# Obtener representaciones aprendidas
cls_embeddings = model.get_cls_embedding(X_test)

print(f"Embeddings shape: {cls_embeddings.shape}")  # (N, 64)

# Útil para:
# - Visualización con t-SNE/UMAP
# - Transfer learning
# - Análisis de similitud
```

### Guardar y Cargar

```python
# Guardar
trainer.save("results/final_models/ft_transformer_adultos")

# Cargar
model_new, trainer_new = create_ft_transformer_for_adultos(num_features=71)
trainer_new.load("results/final_models/ft_transformer_adultos")
```

---

## 📊 Ventajas para Clasificación de Adultos

1. **State-of-the-Art en Datos Tabulares** 🏆
   - FT-Transformer ha demostrado ser competitivo con GBDTs (XGBoost, CatBoost)
   - Ideal para capturar interacciones complejas

2. **Captura de Relaciones No Lineales** 🔗
   - Self-attention permite modelar dependencias entre features
   - No asume estructura específica de interacciones

3. **Embeddings Interpretables** 🔍
   - CLS token como representación aprendida
   - Puede visualizarse con dimensionality reduction

4. **Robusto a Feature Engineering** 🛠️
   - No requiere transformaciones manuales
   - Aprende representaciones óptimas automáticamente

5. **Escalabilidad** 📈
   - Eficiente con GPUs
   - Batch processing paralelo

---

## ⚠️ Consideraciones - Grupo Adultos

**Distribución de clases:**
- Clase 0 (No lector): 20%
- Clase 1 (Lector medio): 0.02% ← ¡MUY MINORITARIA!
- Clase 2 (Lector frecuente): 80%

**¡CRÍTICO!** Clase 1 casi no existe → **Considerar:**

1. ✅ Usar `WeightedRandomSampler`
2. ✅ O configurar pesos de clase muy altos para clase 1
3. ✅ Evaluar con **F1-Score macro**
4. ⚠️ Considerar colapsar a clasificación binaria (lector vs no lector)
5. ⚠️ O usar SMOTE para oversampli de clase minoritaria

---

## 🎓 Para la Tesis

**Capítulo de Metodología:**

> "Para el grupo de adultos, se implementó FT-Transformer (Gorishniy et al., 2021), 
> una arquitectura transformer adaptada para datos tabulares. El modelo convierte 
> cada feature en un token embebido de 64 dimensiones mediante proyecciones lineales 
> independientes. Se agregó un token especial CLS (classification) inspirado en BERT, 
> que actúa como agregador de información. El encoder transformer consta de 4 capas 
> con 8 cabezas de atención multi-head, permitiendo capturar interacciones complejas 
> entre features. Se aplicó warmup de learning rate durante las primeras 10 épocas, 
> seguido de ReduceLROnPlateau para ajuste dinámico. La cabeza de clasificación 
> proyecta el embedding del CLS token (64 dimensiones) a través de una capa oculta 
> de 32 dimensiones antes de la predicción final."

**Referencias:**
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. NeurIPS.

---

## 📈 Comparación con Modelos Previos

### Según Resultados V0.1 (Adultos):

| Modelo          | F1-Score | Accuracy | Copeland Score |
|-----------------|----------|----------|----------------|
| **FT-Transformer** | **0.925** | **0.925** | **+2.0** ⭐ |
| TabNet          | 0.903    | 0.903    | 0.0        |
| MLP_MA          | 0.874    | 0.874    | -2.0       |

**FT-Transformer fue el GANADOR** para adultos en experimentos previos.

---

## 🔧 Tips de Entrenamiento

### 1. Learning Rate Warmup
```python
# Importante para transformers
# Evita inestabilidad en épocas iniciales
trainer = FTTransformerTrainer(model, warmup_epochs=10)
```

### 2. Gradient Clipping
```python
# Implementado automáticamente en trainer
# max_norm=1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3. Batch Size
```python
# Transformers se benefician de batches más grandes
# Usar 128-256 si GPU lo permite
train_loader = DataLoader(dataset, batch_size=256)
```

### 4. AdamW Optimizer
```python
# Mejor que Adam para transformers
# Implementado automáticamente en trainer
self.optimizer = torch.optim.AdamW(...)
```

---

## ✅ Estado Final

### TODOS LOS MODELOS COMPLETADOS ✅

- [x] **TabNet** para Niños (400+ líneas)
- [x] **MLP + Attention** para Jóvenes (600+ líneas)
- [x] **FT-Transformer** para Adultos (600+ líneas)
- [x] Documentación completa (3 READMEs)
- [x] Trainers completos con early stopping
- [x] Funciones auxiliares de creación
- [x] Soporte para balanceo de clases
- [x] Métodos de guardado/carga

---

## 🚀 Próximos Pasos

1. **Crear scripts de entrenamiento**
   - `experiments/train_tabnet_ninos.py`
   - `experiments/train_mlp_jovenes.py`
   - `experiments/train_ft_adultos.py`

2. **Entrenar modelos**
   - Usar datos de `data/processed/`
   - Aplicar balanceo adecuado
   - Guardar en `results/final_models/`

3. **Evaluar y comparar**
   - Calcular métricas completas
   - Aplicar algoritmo de Copeland
   - Generar reportes

4. **Análisis e interpretabilidad**
   - Feature importance (TabNet)
   - Attention weights (MLP)
   - CLS embeddings (FT-Transformer)

---

**✅ 3/3 MODELOS IMPLEMENTADOS Y DOCUMENTADOS** 🎉

**📊 Total:** ~1,600 líneas de código + documentación

**🎓 Listo para entrenar y comparar los 9 modelos de la tesis** 🚀
