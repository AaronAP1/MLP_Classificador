# 🧠 MLP + Multi-Head Attention Classifier - JÓVENES (12-17 años)

## ✅ MODELO 2/3 COMPLETADO

**Archivo:** `mlp_attention.py`  
**Grupo objetivo:** Jóvenes (12-17 años)  
**Features:** 139 dimensiones

---

## 🎯 Descripción

MLP + Multi-Head Attention combina la capacidad de aprendizaje de capas

 completamente conectadas (MLP) con mecanismos de atención que permiten al modelo **ponderar dinámicamente la importancia de cada feature**. A diferencia de un MLP tradicional, este modelo "presta atención" a las features más relevantes antes de procesarlas.

### Ventajas

✅ **Atención sobre Features**
   - Pondera la importancia de cada variable de entrada
   - Aprende qué features son más relevantes para cada predicción

✅ **Residual Connections**
   - Mejora el flujo de gradientes
   - Permite entrenar redes más profundas sin degradación

✅ **Layer Normalization**
   - Estabiliza el entrenamiento
   - Reduce dependencia de inicialización

✅ **Regularización Robusta**
   - Dropout configurable
   - Weight decay en optimizer
   - Early stopping

---

## 🏗️ Arquitectura

```
Input (139 features)
    ↓
[InputFeatureAttention]  ← Pondera importancia de features
    ↓
[Linear 139 → 256] + ReLU + LayerNorm + Dropout
    ↓
[ResidualBlock(256)]  ← Conexión skip
    ↓
[Linear 256 → 128] + ReLU + LayerNorm + Dropout
    ↓
[ResidualBlock(128)]
    ↓
[Linear 128 → 64] + ReLU + LayerNorm + Dropout
    ↓
[ResidualBlock(64)]
    ↓
[Linear 64 → 32] + ReLU + LayerNorm + Dropout
    ↓
[ResidualBlock(32)]
    ↓
[Classifier 32 → 3]
    ↓
Output (3 clases)
```

**Componentes:**
- **InputFeatureAttention**: 139 → 32 (attn_dim) → 139
- **Hidden layers**: [256, 128, 64, 32]
- **Dropout**: 0.3
- **Normalization**: LayerNorm
- **Residual**: Sí (en cada bloque)

---

## 📝 Hiperparámetros

```python
MLPWithAttention(
    input_dim=139,              # Features para jóvenes
    num_classes=3,              # 3 clases de comportamiento
    hidden_dims=[256,128,64,32],# Capas ocultas decrecientes
    attn_dim=32,                # Dimensión de espacio de atención
    dropout_p=0.3,              # Dropout para regularización
    norm='layer',               # LayerNorm
    residual=True               # Conexiones residuales
)

MLPWithAttentionTrainer(
    model=model,
    lr=1e-3,                    # Learning rate
    weight_decay=1e-5           # Regularización L2
)
```

---

## 🚀 Uso

### Crear Modelo

```python
from src.models.mlp_attention import create_mlp_attention_for_jovenes

# Crear modelo y trainer
model, trainer = create_mlp_attention_for_jovenes(input_dim=139)

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
train_df = pd.read_csv("data/processed/jovenes/jovenes_train.csv")
val_df = pd.read_csv("data/processed/jovenes/jovenes_val.csv")

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

### Entrenar con Balanceo de Clases

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

# Configurar pesos en trainer
trainer.set_class_weights(class_weights_tensor)

# Entrenar con balanceo
trainer.fit(train_loader, val_loader, epochs=200)
```

### Entrenar con Sampler Balanceado

```python
from torch.utils.data import WeightedRandomSampler

# Calcular pesos de muestra
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
sample_weights = torch.FloatTensor([class_weights[y] for y in y_train])

# Crear sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# DataLoader con sampler
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    sampler=sampler  # ← Usar sampler en vez de shuffle
)
```

### Predecir

```python
# Predecir clases
y_pred = trainer.predict(X_test.numpy())

# Predecir probabilidades
y_proba = trainer.predict_proba(X_test.numpy())

print(f"Predicción: {y_pred[0]}")
print(f"Probabilidades: {y_proba[0]}")
```

### Analizar Atención

```python
# Obtener pesos de atención para un batch
X_sample = X_test[:10]  # 10 muestras
attn_weights = model.get_attention_weights(X_sample)

print(f"Atención shape: {attn_weights.shape}")  # (10, 139)
print(f"Atención muestra 1: {attn_weights[0]}")

# Visualizar features más atendidas
import matplotlib.pyplot as plt

feature_names = train_df.drop(columns=['target']).columns
attn_mean = attn_weights.mean(dim=0).cpu().numpy()

# Top 10 features más atendidas
top_indices = attn_mean.argsort()[-10:][::-1]
print("\nTop 10 features más atendidas:")
for idx in top_indices:
    print(f"  {feature_names[idx]}: {attn_mean[idx]:.4f}")
```

### Guardar y Cargar

```python
# Guardar modelo entrenado
trainer.save("results/final_models/mlp_attention_jovenes")

# Cargar modelo
model_new, trainer_new = create_mlp_attention_for_jovenes(input_dim=139)
trainer_new.load("results/final_models/mlp_attention_jovenes")
```

---

## 📊 Ventajas para Clasificación de Jóvenes

1. **Flexibilidad** 🔄
   - Se adapta a diferentes distribuciones de datos
   - Aprende representaciones jerárquicas

2. **Interpretabilidad Parcial** 🔍
   - Pesos de atención indican qué features son importantes
   - Útil para análisis exploratorio

3. **Rendimiento Robusto** 🎯
   - Funciona bien con datos desbalanceados (con balanceo)
   - Regularización múltiple evita overfitting

4. **Entrenamiento Estable** ⚖️
   - LayerNorm + Residual connections
   - Early stopping automático

---

## ⚠️ Consideraciones - Grupo Jóvenes

**Distribución de clases:**
- Clase 0 (No lector): 80%
- Clase 1 (Lector medio): 0.21%
- Clase 2 (Lector frecuente): 19%

**¡MUY DESBALANCEADO!** → **CRÍTICO usar balanceo**

**Recomendaciones:**
1. ✅ Usar `WeightedRandomSampler` en DataLoader
2. ✅ O configurar `class_weights` en loss function
3. ✅ Evaluar con **F1-Score macro** (no weighted)
4. ✅ Monitorear recall de clase minoritaria (clase 1)

---

## 🎓 Para la Tesis

**Capítulo de Metodología:**

> "Para el grupo de jóvenes, se implementó una arquitectura MLP con mecanismo 
> de atención a nivel de features de entrada. El modelo utiliza 4 capas ocultas 
> con dimensiones decrecientes [256, 128, 64, 32] y conexiones residuales para 
> facilitar el flujo de gradientes. El mecanismo de atención pondera dinámicamente 
> la importancia de cada feature antes del procesamiento, permitiendo al modelo 
> enfocarse en las variables más relevantes. Se aplicó LayerNormalization y 
> Dropout (p=0.3) para regularización. Dada la distribución desbalanceada de 
> clases (80% no lectores), se implementó WeightedRandomSampler para garantizar 
> la representación adecuada de todas las clases durante el entrenamiento."

---

## 📈 Métricas Esperadas

Con balanceo adecuado:
- **Accuracy**: 75-85%
- **F1-Score (macro)**: 60-70%
- **Recall clase 0**: >85%
- **Recall clase 1**: 20-40% (clase muy minoritaria)
- **Recall clase 2**: 50-70%

---

## ✅ Estado

- [x] Modelo implementado
- [x] Trainer completo
- [x] Documentación completa
- [ ] Entrenamiento con datos de jóvenes
- [ ] Evaluación y métricas
- [ ] Análisis de atención
- [ ] Comparación con TabNet y FT-Transformer

---

**Siguiente:** FT-Transformer para ADULTOS

**Archivo:** `ft_transformer.py` (próximo a crear)
