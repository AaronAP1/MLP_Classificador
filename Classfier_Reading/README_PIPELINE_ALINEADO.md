# Pipeline alineado al manuscrito

Reemplaza al conjunto de scripts auditado. Cubre el ciclo completo: datos
crudos de la ENL 2022 → matriz de predictores → Grid Search con validación
cruzada → reentrenamiento → evaluación → tablas y figuras del artículo.

> **¿Solo quieres lanzar el entrenamiento?**
> Ve a [README_ENTRENAMIENTO.md](README_ENTRENAMIENTO.md): instalación con CUDA,
> qué configurar, qué se ve mientras corre y qué queda al final.
> Este documento explica el *diseño*; aquel explica la *operación*.

---

## Puesta en marcha en otra máquina

```bash
git clone <repo> && cd Classfier_Reading
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # primero, con CUDA
pip install -r requirements-train.txt

python run_pipeline.py check        # verifica dependencias, GPU y datos
python run_pipeline.py etl          # ~10 s
python run_pipeline.py smoke        # ~5 min: todo el camino con pocos datos
python run_pipeline.py benchmark    # estima el tiempo de entrenamiento AQUÍ
python run_pipeline.py train        # el paso largo
python run_pipeline.py report       # tablas y figuras
```

`data/unprocessed/{ninos,jovenes,adultos}.csv` deben estar presentes (no se
versionan por tamaño).

### Opciones

| Opción | Efecto |
|---|---|
| `--preset fast\|standard\|paper` | Tamaño del Grid Search (Tabla 2) |
| `--groups ninos jovenes adultos` | Subconjunto de cohortes |
| `--models mlp_ma tabnet ft_transformer` | Subconjunto de arquitecturas |
| `--no-gridsearch` | Salta la búsqueda (solo para pruebas) |
| `--no-multiseed` | Salta las 5 semillas de estabilidad |
| `--seeds 42 123` | Semillas de estabilidad (por defecto, las del YAML) |
| `--hp-from <results>` | Reutiliza los hiperparámetros de otra corrida y omite el Grid Search (para comparar versiones de los datos) |
| `--tag <nombre>` | Corrida separada en `results/aligned_<nombre>` y `outputs/aligned_<nombre>` |

**Reanudable:** `train` escribe un `results.json` por combinación en cuanto la
termina. Si el proceso se corta, se relanza acotando `--groups` / `--models` a
lo que falte; `report` usa lo que encuentre.

---

## Estructura

```
configs/pipeline.yaml         Fuente única de verdad. Ningún hiperparámetro
                              vive en el código.
run_pipeline.py               Entrada única (check/etl/benchmark/train/report)
src/aligned/
    variables.py              Registro ENL: qué es predictor y qué es fuga
    target.py                 Construcción de las 3 clases lectoras
    features.py               Indicadores derivados + contexto del hogar +
                              selección (varianza, correlación, ranking)
    etl.py                    ETL completo con armonización de CSV y
                              verificaciones duras
    balancing.py              Pesos de clase / SMOTE, solo dentro de cada ajuste
    models.py                 MLP+MA, TabNet, FT-Transformer
    trainer.py                AdamW, early stopping, curvas por época
    gridsearch.py             Grid Search + StratifiedKFold(5)
    pipeline.py               CV → etapa A (70/15) → etapa B (85%) → test 15%
    evaluate.py               Métricas, AUC OvR macro, Copeland
    report.py                 Tablas y figuras (solo desde artefactos reales)
    benchmark.py              Cronómetro para estimar coste
tests/test_aligned.py         pruebas con assert real (pytest)
```

---

## Correcciones de la revisión de septiembre de 2026

| Hallazgo | Efecto | Corrección |
|---|---|---|
| `jovenes.csv` codifica "no aplica" como vacío (los otros CSV usan 0) y tiene columnas `_x`/`_y` de un merge | El filtro de >20 % de faltantes descartaba el **100 %** de sus filas: nadie de 16 a 27 años llegaba al modelo (jóvenes = solo 14–15; adultos = solo 28+) | `etl.harmonize_source`: une `_x`/`_y` y convierte vacíos en 0. Verificado contra edades vecinas. Jóvenes: 2.899 → 5.642 registros (14–17); adultos: 30.617 → 40.530 (18–99) |
| SMOTE aplicado al 70 % antes de la CV | Folds de validación con muestras sintéticas hechas de otros folds: CV optimista | El ETL ya no balancea; `balancing.py` actúa dentro de cada ajuste. Por defecto, pérdida ponderada por clase |
| Reentrenamiento del 85 % con early stopping sobre datos incluidos en el propio entrenamiento | La curva de "validación" de la Fig. 11 no era validación | Etapa A (70 % vs. 15 %) para curvas y época óptima; etapa B (85 %) con esa época fija |
| TabNet sin curva de entrenamiento y con balanceo por muestreo | Fig. 11 incompleta; balanceo distinto al de los otros modelos | `eval_set` con train; pérdida ponderada igual que MLP+MA y FT |
| Target de adultos P402/P403 (cualquier lectura) vs. P506 (libros) en menores | Clases no comparables entre grupos; 67 % "frecuente" en adultos | Adultos por P408/P409 (lectura de libros), configurable |
| `FACTORFINAL_HOGAR`, `FACTOR200_FINAL`, `P201`, `P204` como predictores | Factores de expansión y números de orden dentro del modelo | Movidos a columnas administrativas |
| `P314` (monto de ingreso) tratado como categórico y `P314_1` (marca) como numérico | Montos no vistos en test mapeados al mínimo; ingreso del hogar calculado con la marca | Tipos corregidos |
| Interpretabilidad del MLP+MA casi uniforme si la atención no se especializa | Ranking poco informativo | Se añade importancia por permutación, común a los tres modelos |

## Correcciones de octubre de 2026 (ETL v2)

Detectadas al revisar la corrida definitiva del 1-oct-2026. Los datos
procesados llevan `etl_version: 2` y el entrenamiento se niega a usar datos
de la versión anterior.

| Hallazgo | Efecto | Corrección |
|---|---|---|
| Variables nominales (departamento, lengua materna, etnia, ocupación…) con Label Encoding entraban como **un número** de 0 a 1 | La red tenía que deshacer un orden arbitrario (Amazonas < Áncash < … < Ucayali). En una prueba con un MLP simple, tratarlas como categorías subió el F1-Macro en validación +0,05 (niños), +0,06 (jóvenes) y +0,01 (adultos), y el AUC +0,05 / +0,10 / +0,02 | **Embeddings**: cada categoría tiene un vector aprendido en las tres arquitecturas (`variables.NOMINAL_COLUMNS`; `models.CategoricalTokens`; en TabNet, `cat_idxs`/`cat_dims`). El Label Encoding se mantiene: da el índice del embedding |
| Ingreso (`P314`, `hh_ingreso_*`) escalado con Min-Max sin transformar | Mediana ~S/ 1 000 y máximo S/ 35 000: casi todo quedaba cerca de 0, con varianza 0,001, y el umbral de varianza lo **descartaba** en niños y adultos | `log1p` antes del escalado (`preprocessing.log_transform`) |
| `hh_alfabetismo` contaba como analfabeto a quien no respondió P307 | P307 solo se pregunta con primaria incompleta o menos; los demás (incluidos universitarios) quedaban como "no sabe leer" | Alfabeto si P306 ≥ primaria completa o P307 = sí; los "no aplica" no entran a los promedios del hogar (también en ocupación, lengua, pueblo y estudio) |
| `P305` (nombre del pueblo indígena, texto) | Al convertirlo a número quedaba "sin dato" (−1) en todo `jovenes.csv`: indicaba el archivo de origen de cada persona | Excluida (`variables.TEXT_COLUMNS`) y guarda en el ETL para cualquier otra columna con texto |
| Adultos declarados 18+ | Los capítulos 300-400 solo se aplican de 18 a 64 años: nadie de 65+ tiene la clase | `edad_max: 64`; tablas y figuras dicen 18–64 |
| `hh_etnia_jefe` tomaba la etnia del primer adulto, como número | No era necesariamente el jefe/a y su código (1–9) entraba como magnitud | Etnia del jefe/a, como nominal con embedding |
| Tabla 3, Fig. 7 y Copeland con la semilla 42 | El orden de dos modelos cambiaba según la suerte de esa semilla (TabNet adultos va de 0,417 a 0,480 de F1-Macro según la semilla) | Media ± desviación de las 5 semillas; Copeland sobre las medias |
| Ensemble solo con la semilla 42 | Sin desviación y comparado contra una sola semilla de cada modelo | Se guardan las probabilidades de cada semilla (`test_predictions_seeds.npz`) y el ensemble se calcula semilla a semilla |
| Fig. 11 en accuracy | Líneas casi planas: la mejor época se elige por la pérdida | Fig. 11 muestra la pérdida ponderada; la accuracy queda en `fig_curvas_aprendizaje_accuracy.png` |

---

## Qué corrige respecto a la auditoría

| Hallazgo | Corrección |
|---|---|
| Fuga del target (`target_related` declarada y nunca aplicada) | `variables.is_leaky()` bloquea por patrón; `assert_no_leakage()` se ejecuta tras el preprocesamiento, al cargar los datos y en las pruebas |
| MLP+MA sin atención multi-cabeza | `MultiHeadFeatureAttention`: `softmax(QKᵀ/√dₖ)V`, varias cabezas, concatenación y `W_O`. Una prueba exige que los pesos sumen 1 |
| TabNet con 5 steps y `entmax` | 3 decision steps y `sparsemax`, fijados en config y validados al arrancar |
| Optimizador Adam | `AdamW` en los tres modelos |
| `patience` 30 en código vs 20 en YAML | Un solo valor, 20, en `configs/pipeline.yaml` |
| Sin Grid Search ni validación cruzada | `gridsearch.py` con `StratifiedKFold(5)` y presets de la Tabla 2 |
| Sin reentrenamiento al 85 % | `pipeline.train_final()` ajusta sobre train+val |
| AUC en `except: return None` mudo | `roc_auc_ovr_error` registra siempre el motivo |
| Copeland con fórmula distinta a la descrita | +1 por par ganado, según III-F-3 |
| Sin curvas de aprendizaje reales | `TorchTrainer` registra `train_acc` y `val_acc` por época |
| Interpretabilidad FT-Transformer no extraíble | Encoder capa a capa; `attention_maps()` y `feature_importance()` |
| Esquema distinto entre splits (398 vs 387) | El ETL falla si train/val/test difieren |
| Figuras sintéticas | Ninguna ruta genera datos simulados; lo faltante se declara |
| Pruebas que aprobaban al fallar | 46 pruebas con `assert` |
| `StandardScaler` / `OneHotEncoder` | Min–Max Scaling + Label Encoding |
| Sin umbral de varianza ni correlación | Ambos aplicados, ajustados solo sobre train |

---

## Discrepancias que exigen corregir el manuscrito

El código no puede sostener estas tres afirmaciones. Son decisiones que
requieren tu criterio antes de enviar el artículo.

### 1. La matriz de 71 variables no es uniforme

Tras excluir las variables de comportamiento lector, el número real de
predictores es:

| Grupo | Universo | Final |
|---|---|---|
| Niños (3–13) | 99 | 32 |
| Jóvenes (14–17) | 99 | 32 |
| Adultos (18–64) | 99 | 56 |

Causa: la ENL solo administra el capítulo 300 —lengua materna, etnia, nivel
educativo alcanzado, ocupación, ingreso— a personas de 18 años o más. Para las
cohortes de 3 a 17 años esas columnas son constantes y el umbral de varianza
las elimina, correctamente.

El bloque `hh_*` (contexto del hogar) mitiga el problema dando a cada menor el
perfil sociocultural de los adultos de su hogar, con cobertura del 100 %. Aun
así, 71 no es alcanzable. `config.json` registra
`coincide_con_articulo: false` en cada grupo.

### 2. "Frecuencia Lectora Semanal" no puede ser un predictor

El artículo la lista entre los indicadores derivados (III-B). Se deriva de las
mismas variables que definen la etiqueta, así que usarla es exactamente la fuga
que produjo el *accuracy* 1.0000 del experimento multi-semilla anterior. La
función existe en `features.py`, marcada, y el pipeline nunca la incorpora
a **X**.

### 3. Los rangos etarios del artículo no coincidían con los datos

El texto define Niños 3–13, Jóvenes 14–17, Adultos 18+; la Figura 9 del mismo
artículo rotula 0–11 y 12–17; los CSV venían partidos en 0–15 / 16–27 / 28–99.
El pipeline recalcula la segmentación desde `edad_total` siguiendo el **texto**,
y las figuras nuevas rotulan los grupos con esos mismos rangos.

### Además

- **Índice de Acceso Digital**: el capítulo 600 (equipamiento del hogar) no
  está en `data/unprocessed/`. El índice usa estrato socioeconómico, grado de
  urbanidad y modalidad educativa como proxies, y se marca como aproximado en
  `config.json`. Si agregas las columnas `P602_*`, el cálculo exacto se activa
  solo.
- **Nivel educativo**: ahora sale de `P306` (último nivel aprobado, escala
  1–12). El ETL anterior usaba `P310_N`, que solo responde quien estudia
  actualmente — 4.263 adultos de 51.750.
- **Épocas en CV**: la validación cruzada usa 60 épocas por coste; el modelo
  final usa las 200 que declara el artículo. Queda registrado en
  `gridsearch_best.json`.

---

## Definición del target

| Cohorte | Fuente | Regla |
|---|---|---|
| 3–17 | `P506` frecuencia de lectura de libros | 1–2 → Frecuente · 3–5 → Medio · 6–8 → No lector · 9 (NS/NR) → descartado |
| 18+ | `P408` + `P409` lectura de libros (por defecto) | `P408=2` → No lector · `P409` 1–2 → Frecuente · 3–4 → Medio · 5–6 → No lector |
| 18+ (alternativa) | `P402` + `P403` cualquier contenido | misma regla; `target_source: chapter_300_400` |

Distribución resultante:

| Grupo | n | No lector | Medio | Frecuente |
|---|---|---|---|---|
| Niños | 15.333 | 2.632 | 4.095 | 8.606 |
| Jóvenes | 5.642 | 751 | 1.556 | 3.335 |
| Adultos (libros) | 40.530 | 23.216 | 6.539 | 10.775 |
| Adultos (cualquier contenido) | 40.530 | 3.782 | 9.552 | 27.196 |

> El ETL anterior construía el target de adultos con `P315_*` y `P311`, que en
> la ENL son **limitaciones permanentes (discapacidad)** y **si trabajó la
> semana pasada**, no tipos de lectura.

---

## Salidas

```
data/aligned/<grupo>/          train/val/test.parquet, preprocessor.pkl,
                               config.json (trazabilidad del ETL)
results/aligned/<grupo>_<modelo>/
                               results.json          métricas, ROC, curvas,
                                                     interpretabilidad, CV,
                                                     semillas, hardware
                               gridsearch_results.csv todos los candidatos
                               test_predictions.npz  y_true, y_pred, y_proba
                               interpretabilidad.npz máscaras y atención completas
                               best_model.pt | tabnet_model.zip
outputs/aligned/               INDICE_FIGURAS.md     qué archivo va a qué parte del artículo
                               tabla_resumen_datos.csv
                               tabla2_hiperparametros.csv
                               tabla3_metricas.{csv,tex}  tabla3_media_std.{csv,tex}
                               tabla4_auc.{csv,tex}       tabla4_auc_por_clase.csv
                               copeland_*.csv  estabilidad_multisemilla.csv
                               interpretabilidad_top.csv
                               fig_*.png             Figs. 7-11 e interpretabilidad
results/_smoke/, outputs/_smoke/   salida de la prueba de humo (marcada, no son resultados)
```

---

## Nota sobre reproducibilidad

`seed: 42` en la configuración se propaga a NumPy, PyTorch, los splits, el
balanceo y los folds. La estabilidad multi-semilla usa `[42, 123, 456, 789, 2024]`, las
cinco que declara el artículo. Cada `results.json` guarda la configuración
completa que lo produjo, de modo que una tabla siempre puede rastrearse hasta
sus hiperparámetros.
