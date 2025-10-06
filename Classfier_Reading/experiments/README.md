# 🚀 Experiments - Scripts de Experimentación

Scripts para entrenar, comparar y evaluar clasificadores de hábitos de lectura.

## 📁 Scripts Disponibles

### ✅ `compare_existing_results.py` - Comparar Modelos Existentes

**Descripción:** Compara los resultados de clasificadores ya entrenados usando el algoritmo de Copeland.

**Uso:**
```bash
# Comparar modelos para adultos
python experiments/compare_existing_results.py --grupo adultos

# Comparar con métricas específicas
python experiments/compare_existing_results.py \
  --grupo adultos \
  --metrics f1_weighted accuracy precision_weighted

# Especificar directorio de resultados
python experiments/compare_existing_results.py \
  --results_dir ../V0.1/Resultados_prev \
  --grupo adultos \
  --output_dir results/comparisons
```

**Parámetros:**
- `--results_dir`: Directorio con resultados de modelos (default: `../V0.1/Resultados_prev`)
- `--grupo`: Grupo etario a analizar (`adultos`, `jovenes`, `ninos`, `all`)
- `--output_dir`: Directorio de salida (default: `results/comparisons`)
- `--metrics`: Métricas específicas a comparar (opcional)

**Salidas:**
- `{grupo}_copeland_ranking.csv` - Ranking en CSV
- `{grupo}_copeland_ranking.json` - Ranking en JSON
- `{grupo}_copeland_ranking.md` - Tabla Markdown con ranking

**Ejemplo de salida:**
```
🏆 GANADOR
─────────────────────────────────────────────
   Clasificador: TabNet
   Puntuación Copeland: 2.8750
   F1-Score (weighted): 0.8712
   Accuracy: 0.8712
```

---

### 🔄 `01_train_baseline.py` - Entrenar Modelos Baseline (POR CREAR)

**Descripción:** Entrena las 3 arquitecturas para los 3 grupos etarios (9 modelos totales).

**Uso:**
```bash
python experiments/01_train_baseline.py \
  --config configs/experiments/baseline_all.yaml
```

---

### 📊 `02_compare_models.py` - Comparar Todos los Modelos (POR CREAR)

**Descripción:** Compara los 9 modelos baseline y genera rankings por grupo.

**Uso:**
```bash
python experiments/02_compare_models.py \
  --results_dir results/baseline/
```

---

### 🔍 `03_gridsearch_winners.py` - GridSearch en Ganadores (POR CREAR)

**Descripción:** Optimiza hiperparámetros de los mejores modelos por grupo.

**Uso:**
```bash
python experiments/03_gridsearch_winners.py \
  --config configs/experiments/gridsearch_winners.yaml
```

---

### 🎯 `04_train_final.py` - Entrenar Modelos Finales (POR CREAR)

**Descripción:** Re-entrena modelos ganadores con mejores hiperparámetros usando train+val.

**Uso:**
```bash
python experiments/04_train_final.py \
  --gridsearch_dir results/gridsearch/
```

---

### 📝 `05_generate_report.py` - Generar Informe Completo (POR CREAR)

**Descripción:** Genera un informe HTML interactivo con todos los resultados.

**Uso:**
```bash
python experiments/05_generate_report.py \
  --results_dir results/ \
  --output informe_clasificadores.html
```

---

## 🎯 Workflow Completo

### Paso 1: Comparar Modelos Existentes (DISPONIBLE AHORA)

```bash
# Desde Classfier_Reading/
cd experiments

# Comparar modelos para adultos
python compare_existing_results.py --grupo adultos

# Ver resultados
cat ../results/comparisons/adultos_copeland_ranking.md
```

### Paso 2: Entrenar Baseline (Futuro)

```bash
python 01_train_baseline.py --config ../configs/experiments/baseline_all.yaml
```

### Paso 3: Comparar Baseline (Futuro)

```bash
python 02_compare_models.py --results_dir ../results/baseline/
```

### Paso 4: Optimizar (Futuro)

```bash
python 03_gridsearch_winners.py
python 04_train_final.py
```

### Paso 5: Informe Final (Futuro)

```bash
python 05_generate_report.py --output ../docs/informe_final.html
```

---

## 📊 Algoritmo de Copeland

### ¿Qué es?

El algoritmo de Copeland compara clasificadores por pares en múltiples métricas:

- **+1 punto**: Si clasificador A > B en una métrica
- **-1 punto**: Si clasificador A < B en una métrica
- **0 puntos**: Si hay empate

El clasificador con mayor puntuación total es el ganador.

### ¿Por qué usarlo?

✅ **Comparación justa**: Considera múltiples métricas simultáneamente
✅ **Robusto**: No depende de una sola métrica
✅ **Transparente**: Fácil de explicar y justificar en tesis
✅ **Estándar**: Método reconocido en literatura científica

### Métricas Utilizadas

Por defecto, se comparan:
1. `f1_weighted` - F1-Score ponderado por clase
2. `f1_macro` - F1-Score macro (sin ponderar)
3. `accuracy` - Precisión global
4. `precision_weighted` - Precisión ponderada
5. `recall_weighted` - Recall ponderado

Puedes especificar tus propias métricas con `--metrics`.

---

## 📋 Estructura de Resultados Esperada

### Input (Resultados Previos)

```
V0.1/Resultados_prev/
├─ MLP/
│  ├─ classification_report.txt
│  ├─ metrics_by_epoch.csv
│  ├─ confusion_matrix.png
│  └─ ...
├─ TabNet/
│  └─ ...
└─ FT-Transformer/
   └─ ...
```

### Output (Comparaciones)

```
results/comparisons/
├─ adultos_copeland_ranking.csv
├─ adultos_copeland_ranking.json
├─ adultos_copeland_ranking.md
├─ jovenes_copeland_ranking.csv
├─ jovenes_copeland_ranking.json
└─ jovenes_copeland_ranking.md
```

---

## 🐛 Troubleshooting

### Error: "No se pudieron cargar resultados"

**Causa:** Archivos de resultados no encontrados

**Solución:**
```bash
# Verificar que existen los archivos
ls ../V0.1/Resultados_prev/MLP/
ls ../V0.1/Resultados_prev/TabNet/
ls ../V0.1/Resultados_prev/FT-Transformer/

# Ajustar ruta si es necesario
python compare_existing_results.py --results_dir /ruta/correcta
```

### Error: "Métricas faltantes"

**Causa:** Archivos de resultados incompletos

**Solución:**
```bash
# Especificar solo métricas disponibles
python compare_existing_results.py \
  --metrics f1_weighted accuracy
```

---

## 📚 Referencias

- Copeland, A. H. (1951). "A 'reasonable' social welfare function"
- Scikit-learn: Classification Metrics
- PyTorch: Model Evaluation

---

**Última actualización:** Octubre 2025
