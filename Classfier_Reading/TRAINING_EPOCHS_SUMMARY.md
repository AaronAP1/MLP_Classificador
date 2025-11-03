# ÉPOCAS DE ENTRENAMIENTO - MODELOS BASELINE

## Resumen de Early Stopping por Modelo

Todos los modelos fueron configurados con **máximo 200 épocas** y **early stopping patience=20**.

### ✅ Adultos (3 modelos):

| Modelo | Épocas Entrenadas | Early Stop en Época | Mejor Época |
|--------|-------------------|---------------------|-------------|
| **MLP-MA** | 25 | Epoch 25 | ~5 (val acc 97.39%) |
| **TabNet** | 21 | Epoch 21 | Epoch 1 (val acc 97.68%) |
| **FT-Transformer** | _Entrenando..._ | - | - |

### ✅ Jóvenes (3 modelos):

| Modelo | Épocas Entrenadas | Early Stop en Época | Mejor Época |
|--------|-------------------|---------------------|-------------|
| **MLP-MA** | 28 | Epoch 28 | ~8 (val acc 91.06%) |
| **TabNet** | 56 | Epoch 56 | Epoch 36 (val acc 86.35%) |
| **FT-Transformer** | _Entrenando..._ | - | - |

### ✅ Niños (3 modelos):

| Modelo | Épocas Entrenadas | Early Stop en Época | Mejor Época |
|--------|-------------------|---------------------|-------------|
| **MLP-MA** | 23 | Epoch 23 | ~3 (val acc 47.20%) |
| **TabNet** | 23 | Epoch 23 | Epoch 3 (val acc 73.15%) |
| **FT-Transformer** | _Entrenando..._ | - | - |

---

## Detalles de Entrenamiento:

### Configuración Global:
- **Épocas máximas**: 200
- **Early stopping patience**: 20 épocas sin mejora
- **Métrica de early stopping**: Validation Accuracy
- **Logs cada**: 10 épocas

### Observaciones:

1. **Convergencia Rápida**:
   - Adultos MLP-MA: 25 épocas
   - Adultos TabNet: 21 épocas (mejor en época 1!)
   - Jóvenes MLP-MA: 28 épocas
   - Niños MLP-MA: 23 épocas

2. **Convergencia Media**:
   - Jóvenes TabNet: 56 épocas (mejor en época 36)

3. **TabNet más rápido**:
   - Adultos TabNet convergió en epoch 1 (overfitting temprano)
   - Niños TabNet convergió en epoch 3
   - Usa Ghost Batch Normalization que acelera convergencia

4. **MLP-MA consistente**:
   - Todos los MLP-MA convergieron entre 23-28 épocas
   - Más estable que TabNet

5. **Ningún modelo necesitó las 200 épocas**:
   - Máximo: 56 épocas (Jóvenes TabNet)
   - Mínimo: 21 épocas (Adultos TabNet)
   - Promedio: ~29 épocas

---

## Tiempo de Entrenamiento Aproximado:

Basado en logs (CPU):
- **Adultos MLP-MA**: ~3 minutos (25 épocas)
- **Adultos TabNet**: ~8 minutos (21 épocas) 
- **Jóvenes MLP-MA**: ~1.5 minutos (28 épocas)
- **Jóvenes TabNet**: ~6.5 minutos (56 épocas)
- **Niños MLP-MA**: ~1 minuto (23 épocas)
- **Niños TabNet**: ~4 minutos (23 épocas)

**Total para 6 modelos completados**: ~24 minutos
