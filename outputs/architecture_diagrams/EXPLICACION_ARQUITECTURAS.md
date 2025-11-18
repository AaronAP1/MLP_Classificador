# 🧠 Explicación de Arquitecturas de Modelos

## 📊 Resumen Visual

Se han generado 3 visualizaciones mejoradas de las arquitecturas neuronales utilizadas en este proyecto:

1. **FT-Transformer** (Feature Tokenizer Transformer)
2. **TabNet** (Attentive Interpretable Tabular Learning)
3. **MLP + Multi-head Attention**

---

## 🔷 1. FT-Transformer (Feature Tokenizer Transformer)

### **Concepto Principal**
Adapta la arquitectura Transformer (originalmente diseñada para NLP) a datos tabulares tratando cada característica como un "token".

### **Flujo de Procesamiento**

```
Input (71 features) 
    ↓
Feature Tokenization (convierte cada feature en embedding de 64-dim)
    ↓
[CLS] Token (token especial para clasificación)
    ↓
Transformer Encoder (4 capas)
    ├─ Layer 1: Multi-Head Attention → Feed Forward
    ├─ Layer 2: Multi-Head Attention → Feed Forward
    ├─ Layer 3: Multi-Head Attention → Feed Forward
    └─ Layer 4: Multi-Head Attention → Feed Forward
    ↓
[CLS] Embedding (extrae representación final)
    ↓
Output (3 clases: Lector Frecuente, Medio, No Lector)
```

### **Componentes Clave**

1. **Feature Tokenization**
   - Cada una de las 71 características se convierte en un vector de 64 dimensiones
   - Similar a cómo se procesan palabras en modelos de lenguaje
   - Permite capturar relaciones semánticas entre features

2. **[CLS] Token**
   - Token especial añadido al inicio de la secuencia
   - Acumula información global de todas las características
   - Se usa su embedding final para la clasificación

3. **Multi-Head Attention**
   - Permite que cada característica "observe" a todas las demás
   - 4 cabezas de atención capturan diferentes tipos de relaciones
   - Aprende qué features son importantes para cada predicción

4. **Feed Forward Networks**
   - Redes densas que procesan la información atendida
   - Aplican transformaciones no lineales
   - Refinan las representaciones en cada capa

### **Ventajas**
- ✅ Captura relaciones complejas entre características
- ✅ Excelente rendimiento en datos tabulares (92.5% accuracy)
- ✅ Aprende representaciones ricas de los datos
- ✅ Escalable a muchas características

### **Desventajas**
- ❌ Mayor costo computacional
- ❌ Requiere más memoria
- ❌ Menos interpretable que TabNet

### **Hiperparámetros Clave**
- **d_model**: 64 (dimensión de embeddings)
- **n_heads**: 4 (cabezas de atención)
- **n_layers**: 4 (capas del encoder)
- **Optimizer**: AdamW
- **Loss**: CrossEntropyLoss

---

## 🔶 2. TabNet (Attentive Interpretable Tabular Learning)

### **Concepto Principal**
Arquitectura diseñada específicamente para datos tabulares que usa **atención secuencial** para seleccionar características relevantes en cada paso de decisión.

### **Flujo de Procesamiento**

```
Input (71 features)
    ↓
Feature Transformer (procesamiento inicial - Batch Norm + FC)
    ↓
┌─────────────────────────────────────────┐
│  Decision Steps (3 pasos secuenciales) │
│                                         │
│  Step 1:                                │
│    ├─ Attention Mask 1 (selecciona features relevantes)
│    └─ Decision Step 1 (procesa features seleccionadas)
│         ↓ Output parcial                │
│  Step 2:                                │
│    ├─ Attention Mask 2 (nueva selección)│
│    └─ Decision Step 2                   │
│         ↓ Output parcial                │
│  Step 3:                                │
│    ├─ Attention Mask 3 (nueva selección)│
│    └─ Decision Step 3                   │
│         ↓ Output parcial                │
└─────────────────────────────────────────┘
    ↓
Aggregate (∑ - suma de outputs de todos los steps)
    ↓
Output (3 clases)
```

### **Componentes Clave**

1. **Feature Transformer**
   - Procesamiento inicial de todas las características
   - Batch Normalization para estabilidad
   - Fully Connected layers para transformación

2. **Attention Masks (Sparse)**
   - En cada step, selecciona un **subconjunto** de características relevantes
   - Usa función **entmax** (más sparse que softmax)
   - Permite interpretabilidad: puedes ver qué features se usaron
   - Máscaras complementarias: evita reusar las mismas features

3. **Decision Steps**
   - Procesan solo las features seleccionadas por la máscara
   - 3 pasos secuenciales de decisión
   - Cada step contribuye a la decisión final
   - Redes FC con ReLU

4. **Aggregation**
   - Suma las salidas de todos los decision steps
   - Combina información de diferentes "perspectivas" del dato
   - Permite que el modelo construya la decisión progresivamente

### **Ventajas**
- ✅ **Interpretable**: puedes ver qué features usa en cada paso
- ✅ Atención sparse → más eficiente
- ✅ Diseñado específicamente para tabular data
- ✅ Buen balance rendimiento/interpretabilidad (90.3% accuracy)

### **Desventajas**
- ❌ Menor accuracy que FT-Transformer
- ❌ Más complejo de configurar (muchos hiperparámetros)
- ❌ Puede ser inestable en entrenamiento

### **Hiperparámetros Clave**
- **n_d**: 64 (dimensión de decision step)
- **n_a**: 64 (dimensión de attention)
- **n_steps**: 3 (número de decision steps)
- **gamma**: 1.5 (control de sparsity)
- **Optimizer**: Adam
- **Loss**: CrossEntropyLoss

### **Interpretabilidad**
```python
# Puedes extraer qué features se usaron
attention_masks = model.explain(X_test)
# Cada máscara muestra importancia por step
# Step 1: [0.8, 0.1, 0.0, 0.4, ...]  ← Alta importancia en feature 0 y 3
# Step 2: [0.0, 0.6, 0.9, 0.0, ...]  ← Alta importancia en feature 1 y 2
# Step 3: [0.2, 0.0, 0.0, 0.7, ...]  ← Alta importancia en feature 0 y 3
```

---

## 🔵 3. MLP + Multi-head Attention

### **Concepto Principal**
Combina la simplicidad de redes densas (MLP) con el poder de atención multi-cabeza para capturar relaciones entre características.

### **Flujo de Procesamiento**

```
Input (71 features)
    ↓
Multi-head Attention (3 cabezas - captura relaciones globales)
    ├─ Head 1
    ├─ Head 2
    └─ Head 3
    ↓
Hidden Layer 1 (256 units)
    ├─ ReLU
    └─ Dropout (0.3)
    ↓
Hidden Layer 2 (128 units)
    ├─ ReLU
    └─ Dropout (0.3)
    ↓
Hidden Layer 3 (64 units)
    ├─ ReLU
    └─ Dropout (0.3)
    ↓
Hidden Layer 4 (32 units)
    ├─ ReLU
    └─ Dropout (0.3)
    ↓
Output Layer (3 clases - Softmax)
```

### **Componentes Clave**

1. **Multi-head Attention (Inicial)**
   - Aplicada **antes** de las capas densas
   - 3 cabezas de atención independientes
   - Cada cabeza aprende diferentes tipos de relaciones
   - Permite que el modelo "observe" todas las features simultáneamente

2. **Hidden Layers (Densas)**
   - 4 capas con reducción progresiva: 256 → 128 → 64 → 32
   - Función de activación: **ReLU** (Rectified Linear Unit)
   - Extracción jerárquica de características

3. **Regularización**
   - **Dropout (0.3)**: desactiva 30% de neuronas aleatoriamente
   - **Weight Decay**: penalización L2 en pesos
   - Previene overfitting

4. **Output Layer**
   - 3 neuronas (una por clase)
   - Softmax para probabilidades

### **Ventajas**
- ✅ Arquitectura más simple que FT-Transformer
- ✅ Combina atención con procesamiento denso tradicional
- ✅ Buena generalización (con regularización)
- ✅ Entrenamiento más rápido

### **Desventajas**
- ❌ Rendimiento ligeramente inferior (87.4% accuracy)
- ❌ Menos sofisticado que Transformers completos
- ❌ No tan interpretable como TabNet

### **Hiperparámetros Clave**
- **Attention heads**: 3
- **Hidden units**: [256, 128, 64, 32]
- **Dropout rate**: 0.3
- **Optimizer**: AdamW
- **Loss**: CrossEntropyLoss
- **Weight Decay**: aplicado

---

## 📊 Comparación de Arquitecturas

| Característica | FT-Transformer | TabNet | MLP + Attention |
|---------------|----------------|--------|-----------------|
| **Accuracy** | 🥇 92.5% | 🥈 90.3% | 🥉 87.4% |
| **Interpretabilidad** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Velocidad** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Memoria** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Complejidad** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Mejor para** | Max accuracy | Interpretabilidad | Rapidez/Simplicidad |

---

## 🎯 ¿Cuándo usar cada uno?

### **Usar FT-Transformer si:**
- Necesitas el **mejor rendimiento** posible
- Tienes suficientes recursos computacionales
- La interpretabilidad no es crítica
- Tienes muchos datos de entrenamiento

### **Usar TabNet si:**
- Necesitas **explicar** las predicciones
- Quieres ver qué features son importantes
- Buscas un balance rendimiento/interpretabilidad
- Trabajas en dominios regulados (salud, finanzas)

### **Usar MLP + Attention si:**
- Necesitas **entrenar rápido**
- Tienes recursos limitados
- Quieres una arquitectura más simple
- El rendimiento del 87% es aceptable

---

## 🔬 Detalles Técnicos Adicionales

### **Atención (Attention Mechanism)**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Donde:
- Q (Query): "¿Qué estoy buscando?"
- K (Key): "¿Qué información tengo?"
- V (Value): "¿Cuál es el contenido?"
- d_k: dimensión de las claves (para escalar)
```

### **Dropout**
- Desactiva aleatoriamente neuronas durante entrenamiento
- Fuerza al modelo a aprender representaciones redundantes
- Previene co-adaptación de neuronas

### **Batch Normalization**
- Normaliza las activaciones en cada batch
- Estabiliza el entrenamiento
- Permite usar learning rates más altos

### **CrossEntropyLoss**
```
L = -∑ y_true * log(y_pred)

Para clasificación multiclase
```

---

## 📚 Referencias

1. **FT-Transformer**: [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)
2. **TabNet**: [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
3. **Attention**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

## 📁 Ubicación de Archivos

- **Visualizaciones**: `outputs/architecture_diagrams/`
  - `ft_transformer_architecture.png`
  - `tabnet_architecture.png`
  - `mlp_attention_architecture.png`

- **Código de generación**: `tools/visualize_architectures.py`

---

**Fecha de generación**: 17 de Noviembre, 2025  
**Proyecto**: MLP Classificador - Comportamiento Lector
