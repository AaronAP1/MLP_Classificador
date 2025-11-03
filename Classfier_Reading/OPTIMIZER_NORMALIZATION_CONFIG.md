# Configuración de Optimizadores y Normalización

## ✅ Optimizadores Usados

### Adam (6 modelos):
- **Adultos MLP-MA**: Adam con lr=0.001, weight_decay=0.0001
- **Adultos TabNet**: Adam (configurado internamente en TabNet)
- **Jóvenes MLP-MA**: Adam con lr=0.001, weight_decay=0.0001
- **Jóvenes TabNet**: Adam (configurado internamente en TabNet)
- **Niños MLP-MA**: Adam con lr=0.001, weight_decay=0.0001
- **Niños TabNet**: Adam (configurado internamente en TabNet)

### AdamW (3 modelos):
- **Adultos FT-Transformer**: AdamW con lr=0.0001, weight_decay variable
- **Jóvenes FT-Transformer**: AdamW con lr=0.0001, weight_decay variable
- **Niños FT-Transformer**: AdamW con lr=0.0001, weight_decay variable

## ✅ Normalización Usada

### Batch Normalization (MLP-MA - 3 modelos):
```python
# En src/models/Adultos/mlp_ma.py (línea 30)
nn.BatchNorm1d(hidden_dim)
```
- Aplica BatchNorm1d después de cada capa lineal
- Secuencia: Linear → BatchNorm → ReLU → Dropout
- Usado en: Adultos, Jóvenes y Niños MLP-MA

### Layer Normalization (FT-Transformer - 3 modelos):
```python
# En src/models/Adultos/ft_transformer.py (línea 138)
self.norm = nn.LayerNorm(emb_dim)
```
- Normaliza el CLS token antes de la cabeza de clasificación
- Usado en: Adultos, Jóvenes y Niños FT-Transformer

### Batch Normalization interna (TabNet - 3 modelos):
- TabNet usa batch normalization internamente (Ghost Batch Normalization)
- Virtual batch size: 128
- Momentum: 0.3 (parámetro específico de TabNet)

## Configuración detallada por modelo:

### MLP-MA (3 modelos):
- **Arquitectura**: [256, 128, 64, 32] capas ocultas
- **Normalización**: BatchNorm1d después de cada capa
- **Optimizador**: Adam (lr=0.001)
- **Regularización**: Dropout 0.3, Weight decay 0.0001
- **Scheduler**: CosineAnnealingLR

### FT-Transformer (3 modelos):
- **Arquitectura**: d_token=192, n_blocks=3, n_heads=8
- **Normalización**: LayerNorm en CLS token
- **Optimizador**: AdamW (lr=0.0001)
- **Regularización**: Dropout 0.2 (attention), 0.1 (ffn)
- **Scheduler**: CosineAnnealingLR

### TabNet (3 modelos):
- **Arquitectura**: n_d=64, n_a=64, n_steps=5
- **Normalización**: Ghost Batch Normalization (interna)
- **Optimizador**: Adam (configuración interna de pytorch-tabnet)
- **Regularización**: lambda_sparse=0.001, momentum=0.3
- **Scheduler**: StepLR (step_size=50, gamma=0.9)

## Resumen:

- **Todos usan Adam o AdamW** (variantes del optimizador Adam)
- **Todos usan normalización**:
  - MLP-MA: Batch Normalization
  - FT-Transformer: Layer Normalization
  - TabNet: Ghost Batch Normalization
- **No se usa SGD** en ningún modelo
- **Batch Normalization es la más común** (6/9 modelos si contamos TabNet)
