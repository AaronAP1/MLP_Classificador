# Pipeline alineado al manuscrito

Reemplaza al conjunto de scripts auditado. Cubre el ciclo completo: datos
crudos de la ENL 2022 → matriz de predictores → Grid Search con validación
cruzada → reentrenamiento → evaluación → tablas y figuras del artículo.

---

## Puesta en marcha en otra máquina

```bash
git clone <repo> && cd Classfier_Reading
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py check        # verifica dependencias, GPU y datos
python run_pipeline.py etl          # ~10 s
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
    etl.py                    ETL completo con verificaciones duras
    models.py                 MLP+MA, TabNet, FT-Transformer
    trainer.py                AdamW, early stopping, curvas por época
    gridsearch.py             Grid Search + StratifiedKFold(5)
    pipeline.py               Orquestador CV → refit 85% → test 15%
    evaluate.py               Métricas, AUC OvR, Copeland
    report.py                 Tablas y figuras (solo desde artefactos reales)
    benchmark.py              Cronómetro para estimar coste
tests/test_aligned.py         46 pruebas con assert real
deprecated/                   Generadores sintéticos en cuarentena
```

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
| Niños (3–13) | 105 | 34 |
| Jóvenes (14–17) | 105 | 33 |
| Adultos (18+) | 121 | 59 |

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
El pipeline recalcula la segmentación desde `edad_total` siguiendo el **texto**.
Hay que corregir la Figura 9.

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
| 18+ | `P402` + `P403` | `P402=2` → No lector · `P403` 1–2 → Frecuente · 3–4 → Medio · 5–6 → No lector |

Distribución resultante:

| Grupo | n | No lector | Medio | Frecuente |
|---|---|---|---|---|
| Niños | 15.333 | 2.632 | 4.095 | 8.606 |
| Jóvenes | 5.642 | 751 | 1.556 | 3.335 |
| Adultos | 40.530 | 3.782 | 9.552 | 27.196 |

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
                                                     interpretabilidad, CV
                               gridsearch_results.csv todos los candidatos
                               test_predictions.npz
                               best_model.pt | tabnet_model.zip
outputs/aligned/               tabla3_metricas.{csv,tex}
                               tabla4_auc.{csv,tex}
                               copeland_global.csv
                               estabilidad_multisemilla.csv
                               fig_*.png
```

---

## Nota sobre reproducibilidad

`seed: 42` en la configuración se propaga a NumPy, PyTorch, los splits, SMOTE y
los folds. La estabilidad multi-semilla usa `[42, 123, 456, 789, 2024]`, las
cinco que declara el artículo. Cada `results.json` guarda la configuración
completa que lo produjo, de modo que una tabla siempre puede rastrearse hasta
sus hiperparámetros.
