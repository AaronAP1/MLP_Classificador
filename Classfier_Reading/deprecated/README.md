# Código en cuarentena

Los archivos de esta carpeta **no deben ejecutarse ni volver al pipeline**.
Se conservan solo como registro de lo que produjo los resultados anteriores.

## Por qué están aquí

### `generate_synthetic_classifications.py`
Fabrica predicciones con *accuracies* objetivo escritas a mano:

```python
model_acc = {'ft_transformer': 0.85, 'tabnet': 0.80, 'mlp_attention': 0.78}
```

Escribe en `outputs/classifications_3x3/`. El directorio
`outputs/classifications_3x3_real/` es **idéntico byte a byte** a esa salida
sintética (`diff -rq` no reporta diferencias), pese a que su nombre sugiere lo
contrario.

### `generate_evaluation_plots.py`
Contiene `generate_sample_predictions()`, que dibuja curvas ROC y matrices de
confusión a partir de `np.random` cuando no encuentra resultados reales, en
lugar de fallar. Es el patrón que convierte una figura ausente en una figura
falsa.

## Salidas contaminadas

Estos directorios se derivan de las predicciones sintéticas y **no deben citarse
en el manuscrito**:

- `outputs/classifications_3x3/`
- `outputs/classifications_3x3_real/`
- `outputs/comparison_traditional_vs_dl/`
- `outputs/embeddings_comparison/`
- `outputs/evaluation/resultsF/`

Los scripts que las consumen (`compare_traditional_vs_dl.py`,
`visualize_embedding_comparison.py`, `visualize_thesis_presentation.py`,
`visualize_comparison_heatmaps.py`) heredan el problema.

## Resultados anteriores en `results/baseline/`

Provienen de `data/processed/`, que contiene fuga del target: la etiqueta es un
mapeo determinista de `P517` / `P402`+`P403`, y esas mismas variables están
entre los predictores. Además, el FT-Transformer nunca completó un
entrenamiento en ninguno de los tres grupos.

## Reemplazo

El pipeline vigente es `src/aligned/`, con entrada única `run_pipeline.py`.
Ninguna de sus rutas genera datos, métricas ni predicciones simuladas: cuando
falta un resultado, lo declara como faltante y lo deja fuera de la tabla.

## Pipeline anterior (`training/`, `training/fase2/`)

Superseded por `src/aligned/`. Los scripts de `training/fase2/` importan
`models.Ninos.tabnet`, `models.Adultos.mlp_ma` y similares, que aparecen como
borrados en el árbol de trabajo (`git status`) y son duplicados de los módulos
que sí existen (`tabnet_classifier.py`, `mlp_attention_ninos.py`, …). Esa es la
causa del `ModuleNotFoundError` que dejó la última ejecución de fase 2 en
`successful: 0, failed: 3`.

No hace falta restaurarlos: el pipeline vigente no los usa. Queda a tu criterio
confirmar esas eliminaciones en git o revertirlas si quieres conservar el
historial ejecutable.
