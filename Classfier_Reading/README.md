# 📚 Clasificador de Hábitos de Lectura

> **⚡ ESTADO:** ETL Completado ✅ | Datos procesados y listos para entrenar

## Sistema de Clasificación Basado en Redes Neuronales Profundas para el Análisis del Comportamiento Lector en Perú

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-ETL%20Completado-success.svg)]()

---

## 📊 Estado Actual del Proyecto

| Componente | Estado | Detalles |
|------------|--------|----------|
| 🏗️ Estructura | ✅ Completado | 27 carpetas, 45+ elementos |
| 📊 ETL Pipeline | ✅ Completado | 64,968 registros procesados |
| 🧠 Algoritmo Copeland | ✅ Completado | Comparación científica funcionando |
| 🔧 Utils (seeds, config, logger) | ⏳ Pendiente | Base para entrenamiento |
| 📝 Configs YAML | ⏳ Pendiente | Definir hiperparámetros |
| 🧠 Arquitecturas | ⏳ Pendiente | MLP-MA, TabNet, FT-Transformer |
| 🚀 Scripts entrenamiento | ⏳ Pendiente | 9 modelos por entrenar |

**Próximo paso:** Ver `QUE_HACER_AHORA.md` para continuar

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un **sistema de clasificación** de hábitos de lectura utilizando tres arquitecturas de **Redes Neuronales Profundas**:

- **MLP + Mecanismo de Atención (MLP_MA)**: Red neuronal multicapa con atención a nivel de entrada
- **Feature Tokenizer Transformer (FT-Transformer)**: Arquitectura transformer adaptada para datos tabulares
- **TabNet**: Red neuronal con atención interpretable secuencial

### 🎓 Contexto Académico

Proyecto de tesis para clasificar el comportamiento lector de la población peruana según características demográficas y socioculturales, utilizando datos de la **Encuesta Nacional de Lectura (ENL - GdB-pe)**.

**Objetivo:** Desarrollar y comparar modelos de clasificación para tres grupos etarios:
- 👶 **Niños** (0-11 años)
- 🧑 **Jóvenes** (12-17 años)
- 👨 **Adultos** (18+ años)

**Clases de clasificación:**
1. `no_lector`: Personas con hábitos mínimos o nulos de lectura
2. `lector_medio`: Lectores ocasionales con frecuencia moderada
3. `lector_frecuente`: Lectores habituales con alta frecuencia

---

## 📁 Estructura del Proyecto

```
Classfier_Reading/
│
├─ 📊 data/                          # Gestión de datos
│  ├─ raw/                           # Datos originales ENL (inmutables)
│  ├─ processed/                     # Datos preprocesados por grupo
│  ├─ samples/                       # Muestras de 1k para desarrollo
│  ├─ metadata/                      # Diccionarios de mapeo de variables
│  └─ scripts/                       # Scripts de utilidad de datos
│
├─ 🧠 src/                           # Código fuente
│  ├─ data/                          # ETL y cargadores de datos
│  ├─ models/                        # Arquitecturas de clasificadores
│  ├─ training/                      # Lógica de entrenamiento
│  ├─ evaluation/                    # Métricas y comparación (Copeland)
│  └─ utils/                         # Utilidades generales
│
├─ 🔧 configs/                       # Configuraciones
│  ├─ data/                          # Configs por grupo etario
│  ├─ models/                        # Configs por arquitectura
│  ├─ experiments/                   # Configs de experimentos
│  └─ azure/                         # Configs de Azure ML
│
├─ 🚀 experiments/                   # Scripts de experimentación
│  ├─ 01_train_baseline.py           # Entrenar 9 modelos baseline
│  ├─ 02_compare_models.py           # Comparación con Copeland
│  ├─ 03_gridsearch_winners.py       # Optimización de hiperparámetros
│  ├─ 04_train_final.py              # Re-entrenamiento final
│  └─ 05_generate_report.py          # Informe de resultados
│
├─ 📈 results/                       # Resultados de experimentos
│  ├─ baseline/                      # Experimentos baseline (9 modelos)
│  ├─ comparisons/                   # Rankings de Copeland
│  ├─ gridsearch/                    # Búsqueda de hiperparámetros
│  └─ final_models/                  # Modelos finales optimizados
│
├─ ☁️ deployment/                    # Despliegue
│  ├─ azure/                         # Scripts para Azure ML
│  ├─ docker/                        # Contenedores
│  └─ inference/                     # API de clasificación
│
├─ 📚 notebooks/                     # Análisis exploratorio
├─ 🧪 tests/                         # Pruebas unitarias
└─ 📖 docs/                          # Documentación
```

---

## 🚀 Inicio Rápido

### 1. Requisitos

```bash
Python 3.12+
PyTorch 2.0+
CUDA 11.8+ (opcional, para GPU)
```

### 2. Instalación

```bash
# Clonar repositorio
git clone https://github.com/AaronAP1/MLP_Classificador.git
cd Classfier_Reading

# Crear entorno virtual
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Preparar Datos

```bash
# Copiar datos originales a data/raw/
# - adultos.csv
# - jovenes.csv
# - ninos.csv

# Copiar diccionarios de metadata a data/metadata/

# Crear muestras de 1000 registros para desarrollo rápido
python data/scripts/create_samples.py --size 1000
```

### 4. Entrenar Clasificadores Baseline (Desarrollo Rápido)

```bash
# Entrenar 9 modelos con muestras de 1k (~ 30 min)
python experiments/01_train_baseline.py \
  --config configs/experiments/sample_test.yaml \
  --epochs 5
```

### 5. Entrenar Clasificadores Completos

```bash
# Entrenar 9 modelos con datos completos (~74k registros, ~3-6 horas)
python experiments/01_train_baseline.py \
  --config configs/experiments/baseline_all.yaml
```

### 6. Comparar Modelos

```bash
# Aplicar algoritmo de Copeland para comparación
python experiments/02_compare_models.py \
  --results_dir results/baseline/

# Ver ranking: results/comparisons/overall_ranking.html
```

---

## 🎯 Metodología

### Fase 1: Preparación de Datos
- Extracción de ENL (Encuesta Nacional de Lectura)
- Limpieza y transformación
- Segmentación por grupo etario
- Split estratificado 70/15/15 (Train/Val/Test)
- Anti-leakage: Exclusión de variables objetivo

### Fase 2: Desarrollo de Clasificadores
- **9 modelos base** (3 arquitecturas × 3 grupos)
- Entrenamiento con:
  - Early stopping (patience=20)
  - ReduceLROnPlateau
  - Class weighting para desbalanceo
  - Mixed Precision Training (AMP)
  - Gradient clipping

### Fase 3: Evaluación Comparativa
- **Algoritmo de Copeland** para ranking
- Métricas: F1-score, Precision, Recall, Accuracy
- Selección del mejor clasificador por grupo

### Fase 4: Optimización (Opcional)
- GridSearchCV en clasificadores ganadores
- Re-entrenamiento con train+val (85%)
- Validación final en test (15%)

---

## 📊 Arquitecturas de Clasificadores

### 1. MLP + Mecanismo de Atención (MLP_MA)

```python
Input → Attention Layer → MLP(256→128→64→32) → Softmax → Classes
```

**Características:**
- Atención a nivel de entrada para ponderación de features
- 4 capas ocultas con ReLU
- LayerNorm + Dropout 0.3
- Bloques residuales opcionales

### 2. Feature Tokenizer Transformer (FT-Transformer)

```python
Input → Tokenization → [CLS] + Tokens → Transformer → Classification Head
```

**Características:**
- Tokenización individual por feature
- Multi-head self-attention
- CLS token para clasificación (estilo BERT)
- N capas de transformer encoder

### 3. TabNet

```python
Input → Sequential Attention Steps → Feature Selection → Decision → Classes
```

**Características:**
- Atención secuencial interpretable
- Máscaras esparsas (entmax)
- Feature selection adaptativa
- Alta interpretabilidad

---

## 📈 Resultados Esperados

### Estructura de Resultados

```
results/
├─ baseline/
│  ├─ adultos/
│  │  ├─ mlp_ma/20250106_143022/
│  │  │  ├─ model.pt                   # Clasificador entrenado
│  │  │  ├─ metrics.json               # Métricas de clasificación
│  │  │  ├─ confusion_matrix.png       # Matriz de confusión
│  │  │  └─ training_log.csv           # Log de entrenamiento
│  │  ├─ tabnet/
│  │  └─ ft_transformer/
│  └─ ... (jovenes, ninos)
│
├─ comparisons/
│  ├─ adultos_copeland.csv             # Ranking Copeland
│  └─ overall_ranking.html             # Dashboard interactivo
│
└─ final_models/
   └─ adultos_mlp_ma_final/
      ├─ model.pt                       # Mejor clasificador
      ├─ metrics_final.json
      └─ interpretability/
         └─ attention_weights.png       # Análisis de atención
```

---

## ☁️ Despliegue en Azure ML

```bash
# Enviar job de entrenamiento a Azure
python deployment/azure/submit_job.py \
  --experiment baseline_adultos \
  --compute gpu-cluster \
  --config configs/experiments/baseline_all.yaml

# Monitorear progreso
python deployment/azure/monitor_job.py --job-id <job_id>

# Descargar resultados
python deployment/azure/download_results.py \
  --job-id <job_id> \
  --output results/azure_runs/
```

---

## 🧪 Testing

```bash
# Ejecutar pruebas unitarias
pytest tests/

# Validar pipeline de datos
python tests/test_data_pipeline.py

# Validar modelos
python tests/test_models.py
```

---

## 📚 Documentación

- [Metodología Completa](docs/methodology.md)
- [Diccionario de Datos](docs/data_dictionary.md)
- [Arquitecturas de Modelos](docs/model_architectures.md)
- [Guía de Integración con Tesis](docs/thesis_integration.md)

---

## 🤝 Contribuciones

Este es un proyecto de tesis académica. Para consultas o colaboraciones:

**Autor:** Aaron (AaronAP1)  
**Institución:** Universidad Peruana Unión  
**Contacto:** [GitHub](https://github.com/AaronAP1)

---

## 📝 Licencia

MIT License - Ver [LICENSE](LICENSE) para más detalles

---

## 🙏 Agradecimientos

- Datos: Encuesta Nacional de Lectura (ENL - GdB-pe)
- Inspiración: Papers de TabNet, FT-Transformer
- Universidad Peruana Unión

---

## 📌 Notas Importantes

⚠️ **Este es un CLASIFICADOR, no un sistema de predicción:**
- Clasifica individuos en categorías de hábitos de lectura
- No predice valores continuos
- Salida: Clase discreta (no_lector, lector_medio, lector_frecuente)

✅ **Características clave:**
- 3 arquitecturas de clasificadores de última generación
- Evaluación rigurosa con Copeland Algorithm
- Reproducibilidad completa (seeds fijos)
- Anti-leakage en pipeline de datos
- Despliegue escalable en Azure ML

---

**Última actualización:** Octubre 2025
