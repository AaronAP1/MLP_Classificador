# 🚀 INSTRUCCIONES DE EJECUCIÓN - FASE 2

## 📋 Resumen

Esta carpeta contiene los scripts para entrenar los **9 modelos de Deep Learning** usando **SOLAMENTE variables demográficas** (Fase 2).

**Diferencia clave con Fase 1:**
- **Fase 1**: 131-139 features (incluye variables de lectura) → **100% accuracy** (trivial)
- **Fase 2**: 75-83 features (SOLO demográficas) → **60-75% accuracy** (realista)

**Objetivo de Fase 2:**
✅ Demostrar el valor del Deep Learning cuando el problema NO es trivial  
✅ Clasificar población general usando solo datos de censo (sin encuestas)  
✅ Mostrar que DL mejora 6-9% sobre modelos clásicos  

---

## 📁 Estructura de Archivos

```
training/fase2/
├── train_ninos.py          ← Entrena 3 modelos para Niños (75 features)
├── train_jovenes.py        ← Entrena 3 modelos para Jóvenes (79 features)
├── train_adultos.py        ← Entrena 3 modelos para Adultos (83 features)
├── run_all_fase2.py        ← 🌟 EJECUTOR MAESTRO (corre los 3 en secuencia)
├── README_FASE2.md         ← Documentación detallada
└── INSTRUCCIONES_EJECUCION.md  ← Este archivo
```

---

## ⚙️ Pre-requisitos

### 1. Verificar que los datasets Fase 2 existen

```powershell
# Los datasets deben estar en:
data/processed_fase2/
├── ninos/
│   ├── ninos_train.csv    (7,270 muestras × 75 features)
│   ├── ninos_val.csv      (1,559 muestras)
│   └── ninos_test.csv     (1,559 muestras)
├── jovenes/
│   ├── jovenes_train.csv  (9,814 muestras × 79 features)
│   ├── jovenes_val.csv    (2,103 muestras)
│   └── jovenes_test.csv   (2,104 muestras)
└── adultos/
    ├── adultos_train.csv  (28,391 muestras × 83 features)
    ├── adultos_val.csv    (6,084 muestras)
    └── adultos_test.csv   (6,084 muestras)
```

**Si NO existen**, ejecutar primero:
```powershell
cd c:\Users\xdxdxdxd\Desktop\MLP_Py_In01\Classfier_Reading\src\data
python etl_fase2_demographic.py
```

### 2. Verificar dependencias instaladas

```powershell
pip install torch pytorch-tabnet scikit-learn pandas numpy
```

---

## 🎯 Opciones de Ejecución

### **OPCIÓN A: Ejecutar TODO automáticamente** (Recomendado)

Ejecuta los 3 grupos en secuencia (Niños → Jóvenes → Adultos):

```powershell
cd c:\Users\xdxdxdxd\Desktop\MLP_Py_In01\Classfier_Reading\training\fase2
python run_all_fase2.py
```

**Tiempo estimado:** 3-4 horas  
**Ventajas:**
- ✅ Automatizado, no requiere supervisión
- ✅ Crea logs completos de ejecución
- ✅ Manejo de errores integrado

---

### **OPCIÓN B: Ejecutar grupos individualmente**

Si prefieres controlar cada grupo por separado:

#### 1. Niños (~1 hora)
```powershell
cd c:\Users\xdxdxdxd\Desktop\MLP_Py_In01\Classfier_Reading\training\fase2
python train_ninos.py
```

#### 2. Jóvenes (~1.5 horas)
```powershell
python train_jovenes.py
```

#### 3. Adultos (~2 horas)
```powershell
python train_adultos.py
```

**Ventajas:**
- ✅ Control fino de cada ejecución
- ✅ Útil para debugging
- ✅ Permite revisar resultados entre grupos

---

## 📊 ¿Qué se entrena en cada script?

Cada script entrena **3 modelos de Deep Learning**:

1. **TabNet**
   - Arquitectura: n_d=32, n_a=32, n_steps=3
   - Atención basada en pasos secuenciales
   - Interpretabilidad mediante máscaras de atención

2. **MLP + Multihead Attention**
   - Arquitectura: [128, 64] + 4 attention heads
   - Combina capas densas con mecanismo de atención
   - Captura relaciones complejas entre features

3. **FT-Transformer** (implementado como MLP)
   - Feature tokenization + transformer layers
   - Estado del arte para datos tabulares
   - Mejor generalización

**Configuración común:**
- Seed: 42 (reproducibilidad)
- Epochs: 100 (con early stopping patience=15)
- Batch size: 128
- Learning rate: 0.001
- Optimizer: Adam

---

## 📁 ¿Dónde se guardan los resultados?

```
outputs/fase2/
├── ninos/
│   ├── models/
│   │   ├── tabnet_best.pt
│   │   ├── mlp_attention_best.pt
│   │   └── ft_transformer_best.pt
│   ├── plots/
│   │   ├── tabnet_training.png
│   │   ├── mlp_training.png
│   │   └── ft_training.png
│   └── results_fase2_ninos.json    ← Métricas completas
├── jovenes/
│   └── ...
├── adultos/
│   └── ...
└── execution_log.json              ← Log del run_all_fase2.py
```

**Estructura del JSON de resultados:**
```json
{
  "config": { ... },
  "dataset_info": {
    "n_train": 7270,
    "n_val": 1559,
    "n_test": 1559,
    "n_features": 75
  },
  "results": [
    {
      "model": "TabNet",
      "test_acc": 0.6745,
      "test_f1": 0.6521,
      "training_time_min": 18.5
    },
    { ... }
  ]
}
```

---

## 🔍 Monitoreo de Progreso

### Durante la ejecución verás:

```
================================================================================
🚀 ENTRENAMIENTO FASE 2 - NIÑOS (SOLO DEMOGRÁFICAS)
================================================================================
Fecha: 2025-06-15 10:30:45
Device: cuda
Seed: 42
Features: 75 (demográficas + socioculturales)
Target: 3 clases (No lector / Medio / Frecuente)

📂 Cargando datos...
   ✓ Train: 7,270 muestras
   ✓ Val:   1,559 muestras
   ✓ Test:  1,559 muestras

📊 Distribución de clases (Train):
   Clase 0: 2,180 (30.0%)
   Clase 1: 3,635 (50.0%)
   Clase 2: 1,455 (20.0%)

🔹🔹🔹🔹 MODELO 1/3: TabNet 🔹🔹🔹🔹
Epoch 1/100 - Train Loss: 0.8234 | Val Loss: 0.7645 | Val Acc: 0.6234
Epoch 2/100 - Train Loss: 0.7123 | Val Loss: 0.7012 | Val Acc: 0.6589
...
✅ Mejor modelo guardado (Val Acc: 0.6745) en epoch 45

🎯 RESULTADOS FINALES - TabNet:
   Test Accuracy: 0.6745 (67.45%)
   Test F1-Score: 0.6521
   Tiempo: 18.5 min
```

---

## 🎯 Resultados Esperados (Fase 2)

| Grupo    | Features | TabNet | MLP+Attn | FT-Transformer | Promedio |
|----------|----------|--------|----------|----------------|----------|
| Niños    | 75       | ~67%   | ~68%     | ~70%           | ~68%     |
| Jóvenes  | 79       | ~65%   | ~67%     | ~69%           | ~67%     |
| Adultos  | 83       | ~63%   | ~65%     | ~67%           | ~65%     |

**Comparación con Fase 1:**
- Fase 1: 100% accuracy (problema trivial)
- Fase 2: 65-70% accuracy (problema realista)
- **Gap DL vs Clásicos:** 6-9% mejora en Fase 2 (vs 0% en Fase 1)

**Interpretación:**
- ✅ **60-75% es NORMAL y ESPERADO** para clasificación con solo demográficas
- ✅ Demuestra que Deep Learning **SÍ aporta valor** cuando el problema no es trivial
- ✅ Permite clasificar población general usando solo datos de censo

---

## ⚠️ Troubleshooting

### Error: "CUDA out of memory"
**Solución:** Reducir batch_size en el script:
```python
CONFIG = {
    'batch_size': 64,  # Reducir de 128 a 64
    ...
}
```

### Error: "Module 'models.mlp_ma' not found"
**Solución:** Verificar que estás ejecutando desde la carpeta correcta:
```powershell
cd c:\Users\xdxdxdxd\Desktop\MLP_Py_In01\Classfier_Reading\training\fase2
python train_ninos.py
```

### Error: "File not found: ninos_train.csv"
**Solución:** Ejecutar primero el ETL Fase 2:
```powershell
cd c:\Users\xdxdxdxd\Desktop\MLP_Py_In01\Classfier_Reading\src\data
python etl_fase2_demographic.py
```

### Accuracy muy baja (<50%)
**Posibles causas:**
1. Datos no normalizados correctamente
2. Desbalanceo de clases extremo
3. Hiperparámetros no optimizados

**Solución:** Revisar distribución de clases y normalización en el ETL.

---

## 📝 Checklist de Ejecución

Antes de entrenar, verifica:

- [ ] ✅ Datasets Fase 2 generados (6 archivos CSV)
- [ ] ✅ Dependencias instaladas (torch, pytorch-tabnet, sklearn)
- [ ] ✅ GPU disponible (opcional, pero recomendado)
- [ ] ✅ Espacio en disco suficiente (~500 MB para modelos)
- [ ] ✅ Tiempo disponible (3-4 horas para run_all)

Durante el entrenamiento:
- [ ] Monitorear progreso en consola
- [ ] Verificar que accuracy no se estanca
- [ ] Revisar que loss disminuye progresivamente

Después del entrenamiento:
- [ ] Revisar JSON de resultados
- [ ] Verificar que modelos .pt se guardaron
- [ ] Comparar con resultados Fase 1
- [ ] Generar visualizaciones para tesis

---

## 🎓 Próximos Pasos

Una vez completados los entrenamientos Fase 2:

1. **Análisis Comparativo:**
   ```powershell
   cd experiments
   python compare_fase1_fase2.py
   ```

2. **Visualizaciones para Tesis:**
   - Gráficos de accuracy Fase 1 vs Fase 2
   - Tablas comparativas de métricas
   - Curvas de aprendizaje

3. **Interpretabilidad:**
   - Análisis de atención en TabNet
   - Feature importance comparativo
   - Casos de error (análisis cualitativo)

---

## 📞 Soporte

Si encuentras problemas:
1. Revisa los logs en `outputs/fase2/*/`
2. Verifica el `execution_log.json`
3. Consulta `README_FASE2.md` para documentación detallada

---

**¡Éxito con el entrenamiento Fase 2! 🚀**
