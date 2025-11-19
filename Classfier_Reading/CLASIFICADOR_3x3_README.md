# 🔍 Clasificador 3x3 - Comportamiento Lector

Sistema de clasificación usando **9 modelos entrenados** (3 grupos × 3 arquitecturas).

## 📊 Matriz 3x3

|                    | **FT-Transformer** | **TabNet** | **MLP + Attention** |
|--------------------|--------------------|------------|---------------------|
| **Niños (0-11)**   | ✅ Clasificador 1  | ✅ Clasificador 2 | ✅ Clasificador 3 |
| **Jóvenes (12-17)**| ✅ Clasificador 4  | ✅ Clasificador 5 | ✅ Clasificador 6 |
| **Adultos (18+)**  | ✅ Clasificador 7  | ✅ Clasificador 8 | ✅ Clasificador 9 |

## 🎯 Clases de Salida

Cada clasificador predice una de tres categorías:
- **No Lector**: Comportamiento lector bajo o nulo
- **Lector Medio**: Comportamiento lector moderado
- **Lector Frecuente**: Comportamiento lector alto

## 🚀 Uso Rápido

### Clasificar todos los grupos con todos los modelos:

```bash
python run_classification_3x3.py
```

Esto ejecutará las 9 combinaciones automáticamente y generará:
- CSVs clasificados completos
- CSVs separados por clase
- Resumen de distribuciones

### Clasificar un dataset específico:

```bash
python classify_data_3x3.py --mode dataset \
    --csv data/mi_dataset.csv \
    --grupo adultos \
    --modelo FT-Transformer \
    --output outputs/mis_clasificaciones
```

### Clasificar un registro individual:

```python
from classify_data_3x3 import Classifier3x3

classifier = Classifier3x3()

# Datos de una persona
datos = {
    'P517': 4,      # Frecuencia de lectura
    'P311': 1,      # Género preferido
    'P210_A': 35,   # Edad
    # ... resto de features
}

# Clasificar con modelo específico
resultado = classifier.classify_single(
    data_dict=datos,
    grupo='adultos',
    modelo='FT-Transformer'
)

print(f"Predicción: {resultado['prediccion']}")
print(f"Probabilidades: {resultado['probabilidades']}")
```

## 📁 Estructura de Salida

```
outputs/classified_data_3x3/
├── resumen_clasificacion_3x3_YYYYMMDD_HHMMSS.csv  # Resumen general
├── ninos/
│   ├── FT-Transformer/
│   │   ├── clasificado_completo_YYYYMMDD_HHMMSS.csv
│   │   ├── no_lector_YYYYMMDD_HHMMSS.csv
│   │   ├── lector_medio_YYYYMMDD_HHMMSS.csv
│   │   └── lector_frecuente_YYYYMMDD_HHMMSS.csv
│   ├── TabNet/
│   │   └── ... (misma estructura)
│   └── MLP_MA/
│       └── ... (misma estructura)
├── jovenes/
│   └── ... (misma estructura para 3 modelos)
└── adultos/
    └── ... (misma estructura para 3 modelos)
```

### Archivos Generados por Combinación:

1. **`clasificado_completo_*.csv`**: Dataset original + predicciones + probabilidades
2. **`no_lector_*.csv`**: Solo registros clasificados como "No Lector"
3. **`lector_medio_*.csv`**: Solo registros clasificados como "Lector Medio"
4. **`lector_frecuente_*.csv`**: Solo registros clasificados como "Lector Frecuente"

## 📊 Columnas en CSVs Clasificados

### Columnas originales + nuevas columnas:

- `prediccion`: Clase predicha (texto)
- `prediccion_id`: ID de clase (0, 1, 2)
- `prob_no_lector`: Probabilidad de ser No Lector
- `prob_lector_medio`: Probabilidad de ser Lector Medio
- `prob_lector_frecuente`: Probabilidad de ser Lector Frecuente

## ⚙️ Requisitos

Los modelos entrenados deben estar en:
```
experiments/fase2_demografica/outputs/
├── ninos_FT-Transformer/
│   ├── config.json
│   └── best_model.pt
├── ninos_TabNet/
│   ├── config.json
│   └── tabnet_model.zip
├── ninos_MLP_MA/
│   ├── config.json
│   └── best_model.pt
... (9 combinaciones en total)
```

Los preprocesadores deben estar en:
```
data/processed_fase2/
├── ninos/ninos_preprocessor.pkl
├── jovenes/jovenes_preprocessor.pkl
└── adultos/adultos_preprocessor.pkl
```

## 🔧 Opciones Avanzadas

### Modo de Operación:

```bash
# Todas las combinaciones 3x3
python classify_data_3x3.py --mode all

# Dataset específico
python classify_data_3x3.py --mode dataset \
    --csv data/test.csv \
    --grupo jovenes \
    --modelo TabNet

# Registro individual
python classify_data_3x3.py --mode single \
    --grupo ninos \
    --modelo MLP_MA
```

### Parámetros:

- `--mode`: Modo de clasificación (`all`, `dataset`, `single`)
- `--grupo`: Grupo de edad (`ninos`, `jovenes`, `adultos`)
- `--modelo`: Arquitectura (`FT-Transformer`, `TabNet`, `MLP_MA`)
- `--csv`: Ruta al archivo CSV
- `--output`: Directorio de salida (default: `outputs/classified_data`)

## 📈 Ejemplo de Resumen

```
RESUMEN DE CLASIFICACIÓN:
============================================================
  No Lector          :   245 ( 24.5%)
  Lector Medio       :   489 ( 48.9%)
  Lector Frecuente   :   266 ( 26.6%)
============================================================
  Total: 1000 registros
============================================================
```

## 🐛 Troubleshooting

### Error: "No se encontró el modelo en..."
- Verifica que los modelos estén entrenados
- Revisa la estructura de directorios en `experiments/fase2_demografica/outputs/`

### Error: "No se encontró preprocessor en..."
- Asegúrate de que existan los archivos `.pkl` en `data/processed_fase2/`

### Error de importación de modelos
- Verifica que las rutas en `sys.path.append('src/models')` sean correctas
- Asegúrate de que los módulos existan en `src/models/Adultos/`, `src/models/Ninos/`, `src/models/Jovenes/`

## 📝 Notas

- La clasificación se realiza en **lote** para mayor eficiencia
- Los modelos se cargan automáticamente según el grupo y modelo especificado
- Las probabilidades se calculan con **softmax** sobre los logits
- Los timestamps previenen sobrescritura de archivos

## 🎓 Uso en Tesis

Este sistema permite:
1. **Evaluar modelos** en datos nuevos
2. **Comparar rendimiento** entre las 9 combinaciones
3. **Generar datasets** organizados por comportamiento lector
4. **Analizar distribuciones** demográficas por grupo

---

**Autor**: Aaron  
**Proyecto**: Clasificador de Comportamiento Lector - Tesis  
**Fecha**: Noviembre 2025
