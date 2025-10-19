# 🎯 MATRIZ COMPLETA DE MODELOS - 3×3

## ✅ 9 MODELOS IMPLEMENTADOS

**Fecha:** Octubre 18, 2025  
**Proyecto:** Clasificador de Hábitos de Lectura  
**Universidad:** Universidad Peruana Unión

---

## 📊 MATRIZ DE COMPARACIÓN 3×3

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MATRIZ DE MODELOS                                │
├─────────────┬─────────────────┬─────────────────┬──────────────────────┤
│   GRUPO     │   TabNet        │   MLP+Attn      │   FT-Transformer     │
│             │   (Interpretable)│   (Deep)        │   (SOTA Tabular)     │
├─────────────┼─────────────────┼─────────────────┼──────────────────────┤
│             │                 │                 │                      │
│ 👶 NIÑOS    │ ✅ TabNet       │ ✅ MLP+Attn     │ ✅ FT-Trans          │
│ (0-11 años) │ 131 features    │ 131 features    │ 131 features         │
│ 10,388 rec  │ Balanceado      │ Balanceado      │ Balanceado           │
│             │                 │                 │                      │
├─────────────┼─────────────────┼─────────────────┼──────────────────────┤
│             │                 │                 │                      │
│ 👦 JÓVENES  │ ✅ TabNet       │ ✅ MLP+Attn     │ ✅ FT-Trans          │
│ (12-17 años)│ 139 features    │ 139 features    │ 139 features         │
│ 14,021 rec  │ ⚠️ 80/0.21/19%  │ ⚠️ 80/0.21/19%  │ ⚠️ 80/0.21/19%       │
│             │                 │                 │                      │
├─────────────┼─────────────────┼─────────────────┼──────────────────────┤
│             │                 │                 │                      │
│ 👨 ADULTOS  │ ✅ TabNet       │ ✅ MLP+Attn     │ ✅ FT-Trans          │
│ (18+ años)  │ 71 features     │ 71 features     │ 71 features          │
│ 40,559 rec  │ ⚠️ 20/0.02/80%  │ ⚠️ 20/0.02/80%  │ ⚠️ 20/0.02/80%       │
│             │                 │                 │                      │
└─────────────┴─────────────────┴─────────────────┴──────────────────────┘
```

---

## 📁 ESTRUCTURA DE ARCHIVOS

### 👶 Niños (131 features)

```
src/models/Ninos/
├── __init__.py
├── tabnet_classifier.py          # ✅ Modelo 1/3
├── mlp_attention_ninos.py         # ✅ Modelo 2/3 (NUEVO)
├── ft_transformer_ninos.py        # ✅ Modelo 3/3 (NUEVO)
└── README_TABNET.md
```

**Archivos:**
- `tabnet_classifier.py`: TabNet original (400+ líneas)
- `mlp_attention_ninos.py`: MLP+Attention nuevo (300+ líneas) ← NUEVO
- `ft_transformer_ninos.py`: FT-Transformer nuevo (350+ líneas) ← NUEVO

---

### 👦 Jóvenes (139 features)

```
src/models/Jovenes/
├── __init__.py
├── tabnet_jovenes.py              # ✅ Modelo 1/3 (NUEVO)
├── mlp_attention.py               # ✅ Modelo 2/3
├── ft_transformer_jovenes.py      # ✅ Modelo 3/3 (NUEVO)
├── README_MLP_ATTENTION.md
```

**Archivos:**
- `tabnet_jovenes.py`: TabNet nuevo (150+ líneas) ← NUEVO
- `mlp_attention.py`: MLP+Attention original (600+ líneas)
- `ft_transformer_jovenes.py`: FT-Transformer nuevo (350+ líneas) ← NUEVO

---

### 👨 Adultos (71 features)

```
src/models/Adultos/
├── __init__.py
├── tabnet_adultos.py              # ✅ Modelo 1/3 (NUEVO)
├── mlp_attention_adultos.py       # ✅ Modelo 2/3 (NUEVO)
├── ft_transformer.py              # ✅ Modelo 3/3
└── README_FT_TRANSFORMER.md
```

**Archivos:**
- `tabnet_adultos.py`: TabNet nuevo (150+ líneas) ← NUEVO
- `mlp_attention_adultos.py`: MLP+Attention nuevo (300+ líneas) ← NUEVO
- `ft_transformer.py`: FT-Transformer original (700+ líneas)

---

## 🚀 USO DE LOS 9 MODELOS

### Opción 1: Import Específico

```python
# NIÑOS - TabNet
from src.models.Ninos import create_tabnet_for_ninos
model = create_tabnet_for_ninos(num_features=131)

# NIÑOS - MLP+Attention
from src.models.Ninos import create_mlp_attention_for_ninos
model, trainer = create_mlp_attention_for_ninos(num_features=131)

# NIÑOS - FT-Transformer
from src.models.Ninos import create_ft_transformer_for_ninos
model, trainer = create_ft_transformer_for_ninos(num_features=131)

# ────────────────────────────────────────────────────────────

# JÓVENES - TabNet
from src.models.Jovenes import create_tabnet_for_jovenes
model = create_tabnet_for_jovenes(num_features=139)

# JÓVENES - MLP+Attention
from src.models.Jovenes import create_mlp_attention_for_jovenes
model, trainer = create_mlp_attention_for_jovenes(num_features=139)

# JÓVENES - FT-Transformer
from src.models.Jovenes import create_ft_transformer_for_jovenes
model, trainer = create_ft_transformer_for_jovenes(num_features=139)

# ────────────────────────────────────────────────────────────

# ADULTOS - TabNet
from src.models.Adultos import create_tabnet_for_adultos
model = create_tabnet_for_adultos(num_features=71)

# ADULTOS - MLP+Attention
from src.models.Adultos import create_mlp_attention_for_adultos
model, trainer = create_mlp_attention_for_adultos(num_features=71)

# ADULTOS - FT-Transformer
from src.models.Adultos import create_ft_transformer_for_adultos
model, trainer = create_ft_transformer_for_adultos(num_features=71)
```

---

## 📈 COMPARACIÓN POR MODELO

### 🎯 TabNet (3 versiones)

| Grupo | Features | Registros | Distribución | Archivo |
|-------|----------|-----------|--------------|---------|
| Niños | 131 | 10,388 | ✅ Balanceado | `Ninos/tabnet_classifier.py` |
| Jóvenes | 139 | 14,021 | ⚠️ 80/0.21/19% | `Jovenes/tabnet_jovenes.py` |
| Adultos | 71 | 40,559 | ⚠️ 20/0.02/80% | `Adultos/tabnet_adultos.py` |

**Características comunes:**
- n_d = 64, n_a = 64
- n_steps = 5
- mask_type = 'entmax'
- Interpretabilidad con feature importance

---

### 🧠 MLP + Attention (3 versiones)

| Grupo | Features | Registros | Distribución | Archivo |
|-------|----------|-----------|--------------|---------|
| Niños | 131 | 10,388 | ✅ Balanceado | `Ninos/mlp_attention_ninos.py` |
| Jóvenes | 139 | 14,021 | ⚠️ 80/0.21/19% | `Jovenes/mlp_attention.py` |
| Adultos | 71 | 40,559 | ⚠️ 20/0.02/80% | `Adultos/mlp_attention_adultos.py` |

**Características comunes:**
- Hidden layers: [256, 128, 64, 32]
- Attention dim: 32
- Residual blocks
- Layer Normalization

---

### 🤖 FT-Transformer (3 versiones)

| Grupo | Features | Registros | Distribución | Archivo |
|-------|----------|-----------|--------------|---------|
| Niños | 131 | 10,388 | ✅ Balanceado | `Ninos/ft_transformer_ninos.py` |
| Jóvenes | 139 | 14,021 | ⚠️ 80/0.21/19% | `Jovenes/ft_transformer_jovenes.py` |
| Adultos | 71 | 40,559 | ⚠️ 20/0.02/80% | `Adultos/ft_transformer.py` |

**Características comunes:**
- emb_dim = 64
- n_heads = 8
- n_layers = 4
- Warmup learning rate (10 epochs)
- Gradient clipping

---

## ⚠️ CONSIDERACIONES POR GRUPO

### 👶 Niños - SIN PROBLEMAS ✅
- Distribución balanceada
- No requiere class weights
- Entrenamientro directo

### 👦 Jóvenes - DESAFÍO MEDIO ⚠️
- Clase 0: 80%
- Clase 1: **0.21%** ← Muy minoritaria
- Clase 2: 19%
- **Acción:** Usar class weights o WeightedRandomSampler

### 👨 Adultos - DESAFÍO ALTO ⚠️⚠️
- Clase 0: 20%
- Clase 1: **0.02%** ← Casi inexistente!
- Clase 2: 80%
- **Acción:** Class weights OBLIGATORIO o considerar colapsar a binario

---

## 🎓 PARA LA TESIS

### Diseño Experimental

> "Se implementó un diseño factorial completo de 3×3 modelos, combinando 
> tres arquitecturas de aprendizaje profundo (TabNet, MLP+Attention, 
> FT-Transformer) con tres grupos etarios (Niños, Jóvenes, Adultos), 
> resultando en 9 modelos independientes.
> 
> Cada modelo fue optimizado específicamente para las características de 
> su grupo etario (dimensionalidad, distribución de clases, cantidad de 
> datos). Este diseño permite:
> 
> 1. **Comparación intra-grupo:** Determinar la mejor arquitectura para 
>    cada grupo etario.
> 
> 2. **Comparación inter-arquitectura:** Evaluar el desempeño de cada 
>    arquitectura across grupos etarios.
> 
> 3. **Análisis de robustez:** Validar la consistencia de cada arquitectura 
>    bajo diferentes condiciones de datos.
> 
> La comparación final se realizó mediante el algoritmo de Copeland, 
> permitiendo un ranking robusto considerando múltiples métricas de 
> evaluación (Accuracy, F1-score macro, Precision, Recall)."

---

## 📊 MÉTRICAS DE EVALUACIÓN

Para cada uno de los 9 modelos se calculará:

1. **Accuracy**: Precisión global
2. **F1-Score Macro**: Promedio de F1 por clase
3. **Precision Macro**: Promedio de precisión por clase
4. **Recall Macro**: Promedio de recall por clase
5. **Confusion Matrix**: Matriz de confusión
6. **Classification Report**: Reporte completo por clase

### Evaluación con Copeland

**Comparaciones a realizar:**

1. **Intra-grupo (3 comparaciones):**
   - Niños: TabNet vs MLP vs FT-Trans
   - Jóvenes: TabNet vs MLP vs FT-Trans
   - Adultos: TabNet vs MLP vs FT-Trans

2. **Inter-arquitectura (3 comparaciones):**
   - TabNet: Niños vs Jóvenes vs Adultos
   - MLP+Attn: Niños vs Jóvenes vs Adultos
   - FT-Trans: Niños vs Jóvenes vs Adultos

3. **Comparación global (9 modelos):**
   - Ranking final de todos los modelos
   - Identificación del mejor modelo general

---

## 🚀 PRÓXIMOS PASOS

### 1. Scripts de Entrenamiento

Crear 9 scripts (o 3 unificados):

```
experiments/
├── train_ninos_tabnet.py
├── train_ninos_mlp.py
├── train_ninos_ft.py
├── train_jovenes_tabnet.py
├── train_jovenes_mlp.py
├── train_jovenes_ft.py
├── train_adultos_tabnet.py
├── train_adultos_mlp.py
└── train_adultos_ft.py
```

O alternativamente:

```
experiments/
├── train_all_ninos.py      # 3 modelos para niños
├── train_all_jovenes.py    # 3 modelos para jóvenes
├── train_all_adultos.py    # 3 modelos para adultos
└── train_all_models.py     # Los 9 modelos
```

### 2. Configurar Balanceo

Para jóvenes y adultos:

```python
from sklearn.utils.class_weight import compute_class_weight

# Calcular class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
```

### 3. Entrenar y Evaluar

Para cada modelo:
1. Entrenar con datos correspondientes
2. Guardar modelo entrenado
3. Evaluar en conjunto de prueba
4. Guardar métricas

### 4. Comparación Copeland

```python
from src.evaluation.copeland import copeland_comparison

# Comparar los 9 modelos
results = copeland_comparison(
    models=all_9_models,
    metrics=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
)
```

---

## ✅ RESUMEN

```
┌─────────────────────────────────────────────────────────────┐
│  IMPLEMENTACIÓN COMPLETA                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✅ 9 modelos implementados                                │
│  ✅ 3 arquitecturas × 3 grupos etarios                      │
│  ✅ Configuración específica por grupo                      │
│  ✅ Soporte para class balancing                            │
│  ✅ APIs consistentes                                       │
│  ✅ Métodos save/load                                       │
│  ✅ Documentación completa                                  │
│                                                             │
│  Total líneas de código: ~3,500                             │
│  Archivos creados: 6 nuevos + 3 modificados                 │
│                                                             │
│  📊 LISTO PARA ENTRENAR Y COMPARAR                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

**✅ Estado:** COMPLETADO  
**📅 Fecha:** Octubre 18, 2025  
**🎯 Objetivo:** Matriz 3×3 de modelos implementada  
**🚀 Siguiente:** Scripts de entrenamiento
