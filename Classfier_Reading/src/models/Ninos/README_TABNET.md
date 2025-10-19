# 📊 TabNet Classifier - NIÑOS (0-11 años)

## ✅ MODELO 1/3 COMPLETADO

**Archivo:** `tabnet_classifier.py`  
**Grupo objetivo:** Niños (0-11 años)  
**Features:** 131 dimensiones

---

## 🎯 Descripción

TabNet es una arquitectura de **atención interpretable secuencial** diseñada específicamente para datos tabulares. A diferencia de las redes neuronales tradicionales, TabNet utiliza mecanismos de atención enmascarada para seleccionar features relevantes en cada paso de decisión.

### Características Principales

✅ **Atención Interpretable**
   - Máscaras de importancia por feature
   - Visualización de qué variables influyen en cada predicción

✅ **Capacidad de Aprendizaje Incremental**
   - Puede aprender de nuevos datos sin reentrenar desde cero

✅ **Regularización de Dispersión (Sparsity)**
   - Selección automática de features relevantes
   - Reduce overfitting

✅ **No Requiere Feature Engineering**
   - Aprende representaciones óptimas automáticamente

---

## 🏗️ Arquitectura

```
Input (131 features)
    ↓
[Feature Transformer]
    ↓
[Attention Mask 1] → Decision Step 1
    ↓
[Attention Mask 2] → Decision Step 2
    ↓
    ...
    ↓
[Attention Mask 5] → Decision Step 5
    ↓
[Aggregation]
    ↓
Output (3 clases)
```

**Componentes:**
- **n_d**: 64 (dimensión de decisión)
- **n_a**: 64 (dimensión de atención)
- **n_steps**: 5 (pasos de decisión secuencial)
- **gamma**: 1.5 (coeficiente de regularización)
- **lambda_sparse**: 1e-3 (peso de regularización de sparsity)

---

## 📝 Hiperparámetros Configurados

```python
TabNetClassifier(
    input_dim=131,           # Features para niños
    num_classes=3,           # No lector, Lector medio, Lector frecuente
    n_d=64,                  # Dimensión de decisión
    n_a=64,                  # Dimensión de atención
    n_steps=5,               # 5 pasos de decisión
    gamma=1.5,               # Regularización de atención
    n_independent=2,         # Capas independientes por paso
    n_shared=2,              # Capas compartidas
    lambda_sparse=1e-3,      # Regularización de sparsity
    momentum=0.02,           # Batch Normalization momentum
    lr=2e-2,                 # Learning rate
    mask_type='entmax',      # Tipo de máscara (entmax > sparsemax)
    seed=42                  # Reproducibilidad
)
```

---

## 🚀 Uso

### Crear Modelo

```python
from src.models.tabnet_classifier import create_tabnet_for_ninos

# Crear modelo con configuración optimizada
model = create_tabnet_for_ninos(input_dim=131)

# Ver resumen
model.summary()
```

### Entrenar

```python
import pandas as pd
import numpy as np

# Cargar datos procesados
train_df = pd.read_csv("data/processed/ninos/ninos_train.csv")
val_df = pd.read_csv("data/processed/ninos/ninos_val.csv")

X_train = train_df.drop(columns=['target']).values
y_train = train_df['target'].values
X_val = val_df.drop(columns=['target']).values
y_val = val_df['target'].values

# Entrenar
model.fit(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    max_epochs=200,
    batch_size=256,
    patience=30
)
```

### Entrenar con Balanceo de Clases

```python
from sklearn.utils.class_weight import compute_class_weight

# Calcular pesos de clase
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Entrenar con pesos
model.fit(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    weights=class_weights,  # ← Balanceo
    max_epochs=200
)
```

### Predecir

```python
# Predecir clases
y_pred = model.predict(X_test)

# Predecir probabilidades
y_proba = model.predict_proba(X_test)

print(f"Predicción: {y_pred[0]}")  # 0, 1, o 2
print(f"Probabilidades: {y_proba[0]}")  # [p_class_0, p_class_1, p_class_2]
```

### Interpretar (Feature Importance)

```python
# Obtener importancia de features
importance = model.get_feature_importance(X_test)

# Ordenar por importancia
feature_names = train_df.drop(columns=['target']).columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print(importance_df.head(10))
```

### Guardar y Cargar

```python
# Guardar modelo entrenado
model.save("results/final_models/tabnet_ninos")

# Cargar modelo
model_loaded = TabNetClassifier(input_dim=131, num_classes=3)
model_loaded.load("results/final_models/tabnet_ninos")
```

---

## 📊 Ventajas para Clasificación de Niños

1. **Interpretabilidad** 🔍
   - Puedes ver qué variables (edad, acceso a libros, etc.) son más importantes
   - Ideal para justificar resultados en la tesis

2. **Manejo de Features Correlacionadas** 📈
   - TabNet maneja bien variables correlacionadas sin feature selection manual

3. **Rendimiento en Datos Tabulares** 🎯
   - Estado del arte en datasets tabulares
   - Supera a árboles de decisión y redes neuronales tradicionales

4. **Regularización Automática** ⚖️
   - Menos propenso a overfitting que MLP tradicionales

---

## 📈 Métricas Esperadas

Basado en la distribución del grupo de NIÑOS:
- Clase 0 (No lector): 7%
- Clase 1 (Lector medio): 81%
- Clase 2 (Lector frecuente): 12%

**Distribución más balanceada** que adultos/jóvenes → Buen desempeño esperado sin balanceo agresivo.

---

## 🎓 Para la Tesis

**Capítulo de Metodología:**

> "Para el grupo de niños, se implementó TabNet (Arik & Pfister, 2021), 
> una arquitectura de atención interpretable secuencial que realiza 
> selección de features en múltiples pasos de decisión. El modelo utiliza 
> 5 pasos de atención secuencial con dimensiones de decisión y atención 
> de 64. La regularización de sparsity (λ=1e-3) promueve la selección 
> automática de variables relevantes, facilitando la interpretabilidad 
> de las predicciones."

**Referencias:**
- Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. AAAI.

---

## ✅ Estado

- [x] Modelo implementado
- [x] Documentación completa
- [ ] Entrenamiento con datos de niños
- [ ] Evaluación y métricas
- [ ] Análisis de importancia de features
- [ ] Comparación con otros modelos

---

**Siguiente:** MLP + Multi-Head Attention para JÓVENES

**Archivo:** `mlp_attention.py` (próximo a crear)
