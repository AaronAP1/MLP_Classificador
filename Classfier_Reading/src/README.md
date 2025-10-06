# 🧠 src/ - Código Fuente del Clasificador

Código fuente modular y reutilizable para el clasificador de hábitos de lectura.

## 📁 Estructura

```
src/
├─ data/           # ETL y cargadores de datos
├─ models/         # Arquitecturas de clasificadores
├─ training/       # Lógica de entrenamiento
├─ evaluation/     # Métricas y comparación
└─ utils/          # Utilidades generales
```

---

## 📂 Módulos

### `data/` - Procesamiento de Datos

**Archivos:**
- `__init__.py` - Inicialización del módulo
- `etl.py` - Pipeline ETL centralizado
- `loaders.py` - DataLoaders de PyTorch
- `augmentation.py` - Técnicas de augmentation (SMOTE, etc.)

**Uso:**
```python
from src.data.etl import preprocess_and_load
from src.data.loaders import create_dataloaders

# Cargar y procesar datos
train_loader, val_loader, test_loader = preprocess_and_load(
    config=config,
    grupo='adultos'
)
```

---

### `models/` - Arquitecturas de Clasificadores

**Archivos:**
- `__init__.py`
- `base_model.py` - Clase base para todos los clasificadores
- `mlp_ma.py` - MLP con Mecanismo de Atención
- `ft_transformer.py` - Feature Tokenizer Transformer
- `tabnet.py` - TabNet Wrapper

**Uso:**
```python
from src.models.mlp_ma import MLPWithInputAttention
from src.models.ft_transformer import FTTransformer
from src.models.tabnet import TabNetWrapper

# Instanciar clasificador
classifier = MLPWithInputAttention(
    input_dim=128,
    num_classes=3,  # no_lector, lector_medio, lector_frecuente
    hidden_dims=[256, 128, 64, 32],
    attn_dim=32,
    dropout_p=0.3
)
```

---

### `training/` - Entrenamiento

**Archivos:**
- `__init__.py`
- `trainer.py` - Clase Trainer genérica
- `callbacks.py` - Early stopping, checkpointing
- `optimizers.py` - Configuraciones de optimizadores

**Uso:**
```python
from src.training.trainer import Trainer

trainer = Trainer(
    model=classifier,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

# Entrenar
trainer.fit(epochs=200)

# Evaluar
metrics = trainer.evaluate(test_loader)
```

---

### `evaluation/` - Evaluación y Métricas

**Archivos:**
- `__init__.py`
- `metrics.py` - Cálculo de métricas de clasificación
- `copeland.py` - ⭐ Algoritmo de Copeland
- `visualization.py` - Plots y visualizaciones
- `reports.py` - Generación de informes

**Uso:**
```python
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.copeland import copeland_ranking

# Calcular métricas
metrics = compute_classification_metrics(y_true, y_pred, y_probs)

# Comparar clasificadores con Copeland
ranking = copeland_ranking(results_df)
```

---

### `utils/` - Utilidades

**Archivos:**
- `__init__.py`
- `seeds.py` - Reproducibilidad (seeds fijos)
- `config.py` - Manejo de configuraciones
- `logger.py` - Logging estructurado

**Uso:**
```python
from src.utils.seeds import set_global_seeds
from src.utils.config import load_config
from src.utils.logger import get_logger

# Reproducibilidad
set_global_seeds(42)

# Cargar config
config = load_config('configs/experiments/baseline_all.yaml')

# Logging
logger = get_logger(__name__)
logger.info("Entrenando clasificador...")
```

---

## 🏗️ Principios de Diseño

### 1. Modularidad
- Cada módulo tiene responsabilidad única
- Fácil reutilización de componentes
- Desacoplamiento entre capas

### 2. Extensibilidad
- Fácil agregar nuevas arquitecturas de clasificadores
- Fácil agregar nuevas métricas
- Fácil agregar nuevos datasets

### 3. Configurabilidad
- Todo parametrizable vía YAML
- Sin hardcoding de valores
- Configs versionables

### 4. Reproducibilidad
- Seeds fijos en todos los componentes
- Logs detallados de cada experimento
- Configs guardadas con cada run

---

## 🎯 Flujo de Ejecución

```python
# 1. Setup
from src.utils.seeds import set_global_seeds
from src.utils.config import load_config

set_global_seeds(42)
config = load_config('configs/experiments/baseline_all.yaml')

# 2. Cargar datos
from src.data.etl import preprocess_and_load

train_loader, val_loader, test_loader = preprocess_and_load(
    config=config,
    grupo='adultos'
)

# 3. Crear clasificador
from src.models.mlp_ma import MLPWithInputAttention

classifier = MLPWithInputAttention(**config['model'])

# 4. Entrenar
from src.training.trainer import Trainer

trainer = Trainer(classifier, train_loader, val_loader, config)
trainer.fit(epochs=config['training']['epochs'])

# 5. Evaluar
metrics = trainer.evaluate(test_loader)

# 6. Guardar resultados
from src.evaluation.reports import save_results

save_results(metrics, output_dir='results/baseline/adultos/mlp_ma/')
```

---

## 🔧 Convenciones de Código

### Naming
- Clases: `PascalCase` (ej: `MLPWithInputAttention`)
- Funciones: `snake_case` (ej: `compute_metrics`)
- Constantes: `UPPER_CASE` (ej: `DEFAULT_SEED`)
- Variables privadas: `_leading_underscore`

### Docstrings
```python
def compute_classification_metrics(y_true, y_pred, y_probs):
    """
    Calcula métricas de clasificación.
    
    Args:
        y_true (np.ndarray): Labels verdaderos
        y_pred (np.ndarray): Predicciones del clasificador
        y_probs (np.ndarray): Probabilidades por clase
    
    Returns:
        dict: Diccionario con métricas (accuracy, f1, precision, recall)
    """
    pass
```

### Type Hints
```python
from typing import Dict, List, Tuple

def preprocess_data(
    df: pd.DataFrame,
    config: Dict[str, any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Procesa datos para clasificación."""
    pass
```

---

## 🧪 Testing

```bash
# Test de módulo de datos
pytest tests/test_data_etl.py

# Test de modelos
pytest tests/test_models.py

# Test de métricas
pytest tests/test_evaluation.py

# Coverage
pytest --cov=src tests/
```

---

## 📚 Dependencias Principales

```python
# Core
torch >= 2.0.0
numpy >= 1.24.0
pandas >= 2.0.0

# Preprocesamiento
scikit-learn >= 1.3.0
imbalanced-learn >= 0.11.0  # Para SMOTE

# Modelos especializados
pytorch-tabnet >= 4.0

# Visualización
matplotlib >= 3.7.0
seaborn >= 0.12.0

# Configuración
pyyaml >= 6.0
```

---

**Última actualización:** Octubre 2025
