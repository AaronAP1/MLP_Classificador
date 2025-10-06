# 🔧 Configuraciones - Clasificador de Hábitos de Lectura

Configuraciones centralizadas en formato YAML para todos los experimentos.

## 📁 Estructura

```
configs/
├─ data/              # Configuraciones por grupo etario
├─ models/            # Configuraciones por arquitectura
├─ experiments/       # Configuraciones de experimentos
└─ azure/             # Configuraciones de Azure ML
```

---

## 📂 Tipos de Configuración

### `data/` - Configuración de Datos

**Archivos:**
- `adultos.yaml` - Config para grupo adultos
- `jovenes.yaml` - Config para grupo jóvenes
- `ninos.yaml` - Config para grupo niños

**Ejemplo: `data/adultos.yaml`**
```yaml
# Configuración de datos para ADULTOS
grupo: adultos
descripcion: "Población adulta (18+ años)"

# Archivos
data_path: "data/raw/adultos.csv"
sample_path: "data/samples/adultos_1000.csv"

# Split de datos
split:
  train: 0.70
  val: 0.15
  test: 0.15
  stratify: true
  random_state: 42

# Variables
features:
  categorical:
    - P505
    - P506
    - P507
    - P508
    - P509
    - P201
    - P203
    - P206
    - P207
    - P208
    - P311
    - P315_1
    - P315_2
    - P315_3
    - P315_4
    - P315_5
    - P315_6
  
  numerical:
    - FACTORFINAL_HOGAR
    - P210_A
    - P210_M
    - P310_N
    - P310_A

# Variable objetivo
target:
  column: "habito_lectura_completo"
  classes:
    - no_lector
    - lector_medio
    - lector_frecuente
  
  # Variables a excluir (anti-leakage)
  exclude:
    - P517
    - P311
    - P409
    - lectura_score
    - P315_1
    - P315_2
    - P315_3
    - P315_4
    - P315_5
    - P315_6
```

---

### `models/` - Configuración de Arquitecturas

**Archivos:**
- `mlp_ma.yaml` - MLP + Atención
- `ft_transformer.yaml` - FT-Transformer
- `tabnet.yaml` - TabNet

**Ejemplo: `models/mlp_ma.yaml`**
```yaml
# Configuración de MLP con Mecanismo de Atención
model_name: MLP_MA
descripcion: "Multi-Layer Perceptron con Atención a nivel de entrada"

# Arquitectura
architecture:
  hidden_dims: [256, 128, 64, 32]
  attn_dim: 32
  dropout_p: 0.3
  normalization: "layer"  # layer, batch, none
  residual: false

# Entrenamiento
training:
  optimizer: "AdamW"
  lr: 0.001
  weight_decay: 0.0001
  batch_size: 128
  epochs: 200
  
  # Regularización
  grad_clip: 1.0
  amp: true  # Mixed Precision Training
  
  # Callbacks
  early_stopping:
    patience: 20
    monitor: "val_f1"
    mode: "max"
  
  reduce_lr:
    patience: 10
    factor: 0.5
    monitor: "val_f1"
    mode: "max"

# Balanceo de clases
class_balancing:
  method: "weighted_loss"  # weighted_loss, weighted_sampler, smote
  
# Dispositivo
device: "cuda"  # cuda, cpu, auto
```

**Ejemplo: `models/tabnet.yaml`**
```yaml
# Configuración de TabNet
model_name: TabNet
descripcion: "TabNet con atención interpretable secuencial"

# Arquitectura TabNet
architecture:
  n_d: 32          # Decision dimensionality
  n_a: 32          # Attention dimensionality
  n_steps: 5       # Número de pasos secuenciales
  gamma: 1.5       # Factor de relajación
  n_independent: 2
  n_shared: 2
  lambda_sparse: 0.001
  mask_type: "entmax"  # entmax, sparsemax

# Entrenamiento
training:
  optimizer: "Adam"
  lr: 0.02
  batch_size: 128
  epochs: 200
  virtual_batch_size: 32
  
  early_stopping:
    patience: 20
    monitor: "val_f1"

device: "cuda"
```

---

### `experiments/` - Configuración de Experimentos

**Archivos:**
- `baseline_all.yaml` - Entrenar 9 modelos baseline
- `sample_test.yaml` - Experimento rápido con 1k registros
- `gridsearch_winners.yaml` - GridSearch en ganadores

**Ejemplo: `experiments/baseline_all.yaml`**
```yaml
# Experimento: Baseline Completo (9 modelos)
experiment_name: "baseline_all_groups"
descripcion: "Entrenar las 3 arquitecturas para los 3 grupos etarios"

# Configuración global
global:
  random_state: 42
  use_full_data: true  # false para usar samples
  
# Grupos etarios a entrenar
grupos:
  - adultos
  - jovenes
  - ninos

# Arquitecturas a entrenar
arquitecturas:
  - mlp_ma
  - tabnet
  - ft_transformer

# Importar configs
data_config: "configs/data/{grupo}.yaml"
model_config: "configs/models/{arquitectura}.yaml"

# Outputs
output:
  base_dir: "results/baseline"
  save_best_only: true
  save_interval: 10  # Guardar cada 10 épocas
  
# Logging
logging:
  level: "INFO"
  tensorboard: false
  file: "results/baseline/experiment.log"

# Métricas a trackear
metrics:
  - accuracy
  - precision_weighted
  - recall_weighted
  - f1_weighted
  - f1_macro
  - roc_auc_ovr
```

**Ejemplo: `experiments/sample_test.yaml`**
```yaml
# Experimento: Test Rápido con Muestras
experiment_name: "sample_test"
descripcion: "Prueba rápida con 1000 registros (desarrollo)"

global:
  random_state: 42
  use_full_data: false  # Usar samples
  
grupos:
  - adultos  # Solo un grupo para test rápido

arquitecturas:
  - mlp_ma  # Solo una arquitectura

# Override de epochs para test rápido
training_override:
  epochs: 5
  early_stopping:
    patience: 3

output:
  base_dir: "results/sample_test"
```

**Ejemplo: `experiments/gridsearch_winners.yaml`**
```yaml
# Experimento: GridSearch en Ganadores
experiment_name: "gridsearch_optimization"
descripcion: "Optimización de hiperparámetros en mejores modelos"

# Modelos a optimizar (output de Copeland)
models_to_optimize:
  adultos: mlp_ma
  jovenes: ft_transformer
  ninos: tabnet

# GridSearch parameters
gridsearch:
  cv: 5  # 5-fold cross-validation
  scoring: "f1_weighted"
  n_jobs: -1
  verbose: 2

# Grids por modelo
param_grids:
  mlp_ma:
    hidden_dims:
      - [256, 128, 64, 32]
      - [512, 256, 128, 64]
      - [128, 64, 32]
    attn_dim: [16, 32, 64]
    dropout_p: [0.2, 0.3, 0.4]
    lr: [0.001, 0.0001]
  
  tabnet:
    n_d: [16, 32, 64]
    n_a: [16, 32, 64]
    n_steps: [3, 5, 7]
    gamma: [1.0, 1.5, 2.0]
    lr: [0.01, 0.02, 0.05]
  
  ft_transformer:
    emb_dim: [64, 128, 256]
    n_heads: [4, 8]
    n_layers: [2, 3, 4]
    dropout: [0.1, 0.2, 0.3]
    lr: [0.0001, 0.00001]

output:
  base_dir: "results/gridsearch"
```

---

### `azure/` - Configuración de Azure ML

**Archivos:**
- `environment.yml` - Environment de Azure
- `compute.yml` - Configuración de compute
- `job_template.yml` - Template de jobs

**Ejemplo: `azure/environment.yml`**
```yaml
# Azure ML Environment
name: reading-classifier-env
version: 1.0

conda_file: |
  name: reading-classifier
  channels:
    - defaults
    - conda-forge
  dependencies:
    - python=3.12
    - pip
    - pip:
      - torch>=2.0.0
      - pandas>=2.0.0
      - scikit-learn>=1.3.0
      - pytorch-tabnet>=4.0
      - azureml-mlflow>=1.51.0
      - pyyaml>=6.0
      - tqdm>=4.65.0
```

**Ejemplo: `azure/compute.yml`**
```yaml
# Configuración de Compute para Azure ML
compute:
  name: "gpu-cluster"
  type: "AmlCompute"
  
  vm_size: "Standard_NC6"  # GPU: 1x K80
  # Alternativas:
  # - Standard_NC6s_v3  # GPU: 1x V100
  # - Standard_NC12     # GPU: 2x K80
  # - Standard_ND6s     # GPU: 1x P40
  
  min_nodes: 0
  max_nodes: 4
  idle_seconds_before_scaledown: 300
```

---

## 🎯 Uso de Configuraciones

### Cargar Config en Código

```python
from src.utils.config import load_config

# Cargar config de experimento
config = load_config('configs/experiments/baseline_all.yaml')

# Cargar config de modelo específico
model_config = load_config('configs/models/mlp_ma.yaml')

# Cargar config de datos
data_config = load_config('configs/data/adultos.yaml')
```

### Override de Valores

```python
# Desde línea de comandos
python experiments/01_train_baseline.py \
  --config configs/experiments/baseline_all.yaml \
  --override training.epochs=50 \
  --override training.lr=0.0001
```

---

## 📋 Mejores Prácticas

1. **Versionamiento:**
   - Guardar config usada con cada experimento
   - Incluir hash de config en nombre de run

2. **Documentación:**
   - Comentar propósito de cada parámetro
   - Incluir rangos válidos

3. **Defaults:**
   - Valores default basados en literatura
   - Indicar fuente de defaults en comentarios

4. **Validación:**
   - Validar configs al cargar
   - Fail fast si hay errores

---

**Última actualización:** Octubre 2025
