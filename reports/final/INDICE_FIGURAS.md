# Índice de tablas y figuras

Generado por `python run_pipeline.py report` a partir de `results/final`.

| Archivo | Va en el artículo | Notas |
|---|---|---|
| `tabla_resumen_datos.csv` | Sección de datos (y reemplazo de la Table 1) | Registros, edades, clases y n.º de variables por grupo |
| `tabla2_hiperparametros.csv` | Table 2 | Configuración elegida por el Grid Search en cada combinación |
| `tabla3_media_std.csv/.tex` | **Table 3** | Media ± desviación en 5 semillas, incluido el ensemble. Una sola semilla puede invertir el orden de dos modelos |
| `tabla3_metricas.csv/.tex` | Material complementario | Métricas en test de la semilla principal (42); incluye el ensemble (promedio de probabilidades); es la que dibujan las matrices de confusión y las curvas ROC |
| `tabla4_auc.csv/.tex` | Table 4 | AUC one-vs-rest, promedio **macro** (semilla principal; la media está en la Table 3) |
| `tabla4_auc_por_clase.csv` | Texto de la sección ROC | AUC de cada clase |
| `tabla_ensemble.csv` | Texto del ensemble | F1-Macro de cada modelo y del ensemble, ganancia frente al mejor modelo individual (semilla principal y semilla a semilla) y desacuerdo entre modelos |
| `fig_comparacion_grupos.png` | Fig. 7 | Media entre semillas ± desviación |
| `fig_matrices_confusion.png` | Fig. 8 | Normalizadas por fila; entre paréntesis, conteos |
| `fig_curvas_roc.png` | Fig. 9 | Grupo × modelo. Variante 1×3: `fig_curvas_roc_por_grupo.png` |
| `fig_ranking_copeland.png` | Fig. 10 | Sobre la media de las semillas. Rango posible −9 a +9 (4 competidores, 3 grupos) |
| `fig_curvas_aprendizaje.png` | Fig. 11 | Etapa A: pérdida de train 70 % vs. validación 15 %, con la mejor época. En accuracy: `fig_curvas_aprendizaje_accuracy.png` |
| `fig_importancia_variables.png` | Sección de interpretabilidad | Atención (MLP-MA, FT) y máscaras (TabNet) |
| `fig_importancia_permutacion.png` | Sección de interpretabilidad | Caída de F1-Macro al permutar cada variable; misma vara para los 3 modelos |
| `fig_mascaras_tabnet.png` | Sección de interpretabilidad | Máscaras por decision step |
| `fig_atencion_ft.png` | Sección de interpretabilidad | Atención entre las variables más relevantes |
| `interpretabilidad_top.csv` | Sección de interpretabilidad | Top 15 por grupo, modelo y método, con descripción |
| `copeland_*.csv`, `estabilidad_multisemilla.csv` | Texto de resultados | |

Equipo de entrenamiento: NVIDIA GeForce RTX 4060 · PyTorch 2.14.1+cu126
Preset de Grid Search: standard · Corrida: definitiva
