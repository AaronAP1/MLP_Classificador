# 🏗️ Estructura Completa del Proyecto - Clasificador de Hábitos de Lectura

```
Classfier_Reading/
│
├── 📁 src/                                    # Código fuente principal
│   │
│   ├── 📁 data/                               # ETL y preprocesamiento
│   │   ├── __init__.py
│   │   ├── etl.py                            # Pipeline ETL completo
│   │   └── preprocessor.py                   # Preprocesamiento
│   │
│   ├── 📁 models/                             # ⭐ MODELOS POR GRUPO ETARIO ⭐
│   │   ├── __init__.py                       # Imports principales
│   │   ├── README.md                         # 📖 Documentación general
│   │   │
│   │   ├── 📁 Ninos/                          # 👶 NIÑOS (0-11 años)
│   │   │   ├── __init__.py
│   │   │   ├── tabnet_classifier.py          # 🎯 TabNet (400+ líneas)
│   │   │   └── README_TABNET.md              # 📖 Documentación
│   │   │
│   │   ├── 📁 Jovenes/                        # 👦 JÓVENES (12-17 años)
│   │   │   ├── __init__.py
│   │   │   ├── mlp_attention.py              # 🧠 MLP + Attention (600+ líneas)
│   │   │   └── README_MLP_ATTENTION.md       # 📖 Documentación
│   │   │
│   │   └── 📁 Adultos/                        # 👨 ADULTOS (18+ años)
│   │       ├── __init__.py
│   │       ├── ft_transformer.py             # 🤖 FT-Transformer (700+ líneas)
│   │       └── README_FT_TRANSFORMER.md      # 📖 Documentación
│   │
│   ├── 📁 evaluation/                         # Evaluación y métricas
│   │   ├── __init__.py
│   │   ├── copeland.py                       # Algoritmo de Copeland
│   │   └── metrics.py                        # Métricas de clasificación
│   │
│   └── 📁 utils/                              # Utilidades generales
│       ├── __init__.py
│       ├── logger.py                         # Sistema de logging
│       └── config.py                         # Configuración global
│
├── 📁 data/                                   # Datos del proyecto
│   │
│   ├── 📁 raw/                                # Datos originales
│   │   ├── ninos_raw.csv
│   │   ├── jovenes_raw.csv
│   │   └── adultos_raw.csv
│   │
│   ├── 📁 processed/                          # Datos procesados (ETL)
│   │   ├── 📁 ninos/
│   │   │   ├── ninos_train.csv              # 10,388 registros
│   │   │   ├── ninos_val.csv
│   │   │   ├── ninos_test.csv
│   │   │   ├── preprocessor.pkl
│   │   │   └── config.json
│   │   │
│   │   ├── 📁 jovenes/
│   │   │   ├── jovenes_train.csv            # 14,021 registros
│   │   │   ├── jovenes_val.csv
│   │   │   ├── jovenes_test.csv
│   │   │   ├── preprocessor.pkl
│   │   │   └── config.json
│   │   │
│   │   └── 📁 adultos/
│   │       ├── adultos_train.csv            # 40,559 registros
│   │       ├── adultos_val.csv
│   │       ├── adultos_test.csv
│   │       ├── preprocessor.pkl
│   │       └── config.json
│   │
│   └── 📁 metadata/                           # Metadatos
│       └── feature_info.json
│
├── 📁 experiments/                            # 🔬 Scripts de entrenamiento (PRÓXIMO)
│   ├── train_ninos.py                        # Entrenar TabNet
│   ├── train_jovenes.py                      # Entrenar MLP+Attention
│   ├── train_adultos.py                      # Entrenar FT-Transformer
│   └── train_all.py                          # Entrenar todos
│
├── 📁 results/                                # Resultados de experimentos
│   ├── 📁 models/                             # Modelos entrenados
│   │   ├── tabnet_ninos.pth
│   │   ├── mlp_jovenes.pth
│   │   └── ft_transformer_adultos.pth
│   │
│   ├── 📁 plots/                              # Gráficos y visualizaciones
│   │   ├── training_curves/
│   │   ├── confusion_matrices/
│   │   └── feature_importance/
│   │
│   └── 📁 metrics/                            # Métricas y reportes
│       ├── classification_reports.json
│       └── copeland_comparison.json
│
├── 📁 tests/                                  # Tests y validación
│   ├── __init__.py
│   ├── test_model_imports.py                 # ✅ Test de imports (NUEVO)
│   ├── test_etl.py
│   └── test_models.py
│
├── 📁 docs/                                   # Documentación
│   ├── REORGANIZACION_MODELOS.md             # ✅ Doc reorganización (NUEVO)
│   ├── ESTRUCTURA_PROYECTO.md                # ✅ Este archivo (NUEVO)
│   └── guia_entrenamiento.md
│
├── 📁 configs/                                # Configuraciones
│   ├── config_ninos.yaml
│   ├── config_jovenes.yaml
│   └── config_adultos.yaml
│
├── requirements.txt                           # Dependencias Python
├── README.md                                  # Documentación principal
└── .gitignore                                 # Archivos ignorados por Git

```

---

## 📊 Estadísticas del Proyecto

### Datos Procesados
- **Total registros:** 64,968
- **Niños (0-11):** 10,388 registros | 131 features
- **Jóvenes (12-17):** 14,021 registros | 139 features
- **Adultos (18+):** 40,559 registros | 71 features

### Código Implementado
- **TabNet:** ~400 líneas
- **MLP + Attention:** ~600 líneas
- **FT-Transformer:** ~700 líneas
- **ETL Pipeline:** ~500 líneas
- **Utilidades:** ~300 líneas
- **Total:** ~2,500 líneas de código

### Documentación
- **READMEs:** 4 archivos
- **Docs técnicos:** 3 archivos
- **Tests:** 3 scripts

---

## 🎯 Estado Actual

### ✅ Completado
1. **ETL Pipeline**
   - Procesamiento de 64,968 registros
   - 3 grupos etarios separados
   - Preprocesadores guardados

2. **Modelos Implementados**
   - TabNet para Niños (131 features)
   - MLP+Attention para Jóvenes (139 features)
   - FT-Transformer para Adultos (71 features)

3. **Organización Modular**
   - Carpetas por grupo etario
   - __init__.py configurados
   - Documentación completa

### ⏭️ Próximos Pasos
1. **Scripts de Entrenamiento**
   - Crear `experiments/train_ninos.py`
   - Crear `experiments/train_jovenes.py`
   - Crear `experiments/train_adultos.py`

2. **Entrenamiento**
   - Entrenar TabNet (Niños)
   - Entrenar MLP+Attention (Jóvenes) - con balanceo
   - Entrenar FT-Transformer (Adultos) - con balanceo

3. **Evaluación**
   - Métricas completas (Accuracy, F1, Precision, Recall)
   - Comparación con Copeland
   - Análisis de interpretabilidad

---

## 🚀 Flujo de Trabajo

```
1. DATOS CRUDOS
   ↓
2. ETL PIPELINE (✅ COMPLETADO)
   ↓
3. DATOS PROCESADOS (✅ GUARDADOS)
   ↓
4. MODELOS IMPLEMENTADOS (✅ ORGANIZADOS)
   ↓
5. SCRIPTS DE ENTRENAMIENTO (⏭️ PRÓXIMO)
   ↓
6. ENTRENAMIENTO
   ↓
7. EVALUACIÓN
   ↓
8. COMPARACIÓN (COPELAND)
   ↓
9. ANÁLISIS E INTERPRETABILIDAD
   ↓
10. RESULTADOS FINALES
```

---

## 📚 Guía Rápida de Navegación

### Para trabajar con Niños:
```
📂 src/models/Ninos/
   - tabnet_classifier.py     # Código
   - README_TABNET.md         # Docs
   
📂 data/processed/ninos/
   - ninos_train.csv          # Datos
```

### Para trabajar con Jóvenes:
```
📂 src/models/Jovenes/
   - mlp_attention.py         # Código
   - README_MLP_ATTENTION.md  # Docs
   
📂 data/processed/jovenes/
   - jovenes_train.csv        # Datos
```

### Para trabajar con Adultos:
```
📂 src/models/Adultos/
   - ft_transformer.py        # Código
   - README_FT_TRANSFORMER.md # Docs
   
📂 data/processed/adultos/
   - adultos_train.csv        # Datos
```

---

## 🎓 Para la Tesis

Esta estructura modular facilita:

1. **Experimentación Independiente**
   - Cada grupo etario es autocontenido
   - Fácil modificar un modelo sin afectar otros

2. **Reproducibilidad**
   - Configuraciones separadas por grupo
   - Preprocesadores guardados
   - Seeds fijados

3. **Escalabilidad**
   - Agregar nuevos modelos es sencillo
   - Estructura clara para colaboración

4. **Documentación**
   - Cada modelo tiene su README
   - Código bien comentado
   - Ejemplos de uso incluidos

---

**📅 Actualizado:** Octubre 18, 2025  
**✅ Estado:** Estructura Completa y Organizada
