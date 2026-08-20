# Cómo lanzar el entrenamiento

Guía operativa para correr el pipeline en la máquina con GPU.
Para el diseño del pipeline y las decisiones metodológicas, ver
[README_PIPELINE_ALINEADO.md](README_PIPELINE_ALINEADO.md).

Hardware de referencia: **RTX 4060 8 GB**. Los modelos usan menos de 2 GB de
VRAM, así que la tarjeta sobra; el límite es tiempo, no memoria.

---

## 1. Instalación (una sola vez)

```powershell
cd Classfier_Reading
python -m venv .venv
.venv\Scripts\activate

# PRIMERO PyTorch con CUDA. Si instalas requirements.txt antes,
# arrastra la build +cpu y todo va ~8x más lento sin avisarte.
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Los tres CSV de `data/unprocessed/` deben estar presentes (111 MB, no se
versionan por tamaño).

### Verificar

```powershell
python run_pipeline.py check
```

Tiene que decir:

```
  GPU: NVIDIA GeForce RTX 4060 (8.0 GB)
```

Si dice `no disponible`, el PyTorch instalado es el `+cpu`. Reinstala con el
índice de CUDA antes de seguir.

---

## 2. Qué configurar

Todo vive en [configs/pipeline.yaml](configs/pipeline.yaml). Solo hay **tres
decisiones** que tomar:

```yaml
gridsearch:
  preset: "standard"                   # fast | standard | paper

evaluation:
  stability:
    seeds: [42, 123, 456, 789, 2024]   # 5 semillas = 5 reentrenamientos extra

training:
  device: "auto"                       # déjalo así
  amp: "auto"                          # déjalo así
```

**`preset`** decide casi todo el tiempo de cómputo:

| Preset | Configuraciones por modelo | Ajustes de CV | RTX 4060 (estimado) |
|---|---|---|---|
| `fast` | 1 | 45 | ~8–12 h |
| `standard` | 16 | 720 | ~25–40 h |
| `paper` | 108–324 | 11.340 | inviable |

Usa **`standard`**. Es el equilibrio razonable: explora *learning rate*, tamaño
de lote, dropout y weight decay sin volverse impracticable.

**`seeds`** es la segunda palanca. Las 5 semillas son ~70 % del coste del preset
`fast`. Si vas justo de tiempo, déjalas en 3 — o quítalas del todo con
`--no-multiseed` y córrelas después solo para el modelo ganador.

El resto del archivo son valores anclados al manuscrito (200 épocas, paciencia
20, AdamW, 3 decision steps, 4 capas encoder, Min–Max, Label Encoding). Si
cambias uno de esos, **el pipeline se niega a arrancar** y te dice cuál
contradice al artículo. Es intencional: evita que el código y el texto se
separen sin que nadie lo note.

---

## 3. Los comandos, en orden

```powershell
python run_pipeline.py etl          # ~10 s
python run_pipeline.py benchmark    # ~10 min
python run_pipeline.py train        # el paso largo
python run_pipeline.py report       # ~1 min
```

### `etl`

Lee los CSV crudos, rehace la segmentación etaria, construye el target y las
variables, y escribe `data/aligned/`. Debe terminar con:

```
  universo -> final : 105 -> 34 features     (niños)
  universo -> final : 105 -> 33 features     (jóvenes)
  universo -> final : 121 -> 59 features     (adultos)
```

### `benchmark`

**Córrelo antes de comprometerte.** Cronometra 3 épocas de cada combinación en
*esa* GPU y extrapola el total por preset. Te dice si son 25 h o 60 h en tu
hardware, no en el mío. Deja el detalle en `results/aligned/benchmark.json`.

### `train`

Las 9 combinaciones (3 grupos × 3 arquitecturas). Para cada una: Grid Search con
validación cruzada 5-fold sobre el 70 %, reentrenamiento con la mejor
configuración sobre el 85 %, y evaluación única sobre el 15 % de prueba.

### `report`

Genera tablas y figuras desde los resultados reales.

### Opciones útiles

| Opción | Para qué |
|---|---|
| `--preset fast` | Prueba rápida antes de la corrida buena |
| `--groups jovenes` | Solo una cohorte |
| `--models tabnet mlp_ma` | Solo ciertas arquitecturas |
| `--no-multiseed` | Salta las 5 semillas |
| `--no-gridsearch` | Salta la búsqueda (solo para depurar) |

**Recomendación:** antes de lanzar 30 horas, haz una pasada de humo de ~15 min:

```powershell
python run_pipeline.py train --preset fast --groups jovenes --no-multiseed
python run_pipeline.py report
```

Si eso produce tablas y figuras, el camino completo funciona.

---

## 4. Lanzar en segundo plano

Para que no lo mate un cierre de sesión:

```powershell
Start-Process python -ArgumentList "-u","run_pipeline.py","train" `
  -RedirectStandardOutput train.log -RedirectStandardError train.err -NoNewWindow

Get-Content train.log -Wait -Tail 30      # seguir el avance
```

En Linux:

```bash
nohup python -u run_pipeline.py train > train.log 2>&1 &
tail -f train.log
```

Vigilar la GPU en otra terminal:

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu `
  --format=csv --loop=5
```

---

## 5. Qué ves mientras entrena

Por cada combinación:

```
==================================================================
ADULTOS · ft_transformer
==================================================================
  train (41565, 59) · val (4593, 59) · test (4593, 59)

  Grid Search · ft_transformer · preset 'standard' · 16 configuraciones x 5 folds = 80 ajustes
    [  1/16] lr=0.001   bs=64   do=0.2  wd=1e-05   d=4 -> f1_macro=0.5231 (±0.0182)
    [  2/16] lr=0.001   bs=64   do=0.3  wd=1e-05   d=4 -> f1_macro=0.5417 (±0.0143)
    ...
  mejor: {'lr': 0.0001, 'batch_size': 128, ...} -> f1_macro=0.5612

  Reentrenando sobre 85% (46,158 muestras) con {...}
    epoca  20/200 | train 0.7412/0.6891 | val 0.7893/0.6612
    epoca  40/200 | train 0.6234/0.7402 | val 0.7101/0.7014
    early stopping en la epoca 87 (mejor: 67)

  TEST · acc=0.7412 f1w=0.7301 f1m=0.5612 auc=0.8734

  Estabilidad multi-semilla: [42, 123, 456, 789, 2024]
    seed 42:  acc=0.7412 f1m=0.5612
    seed 123: acc=0.7389 f1m=0.5578
    ...
  artefactos -> results/aligned/adultos_ft_transformer  (312.4 min)
```

### Qué vigilar

- **`[ n/16]` es tu reloj.** Si la primera configuración tarda 20 min, el Grid
  Search de esa celda son ~5 h. Con eso decides si sigues o cambias de preset.
- **En las líneas de época**, `train` y `val` moviéndose juntos = sano. Si
  `train` sigue bajando y `val` empieza a subir, hay sobreajuste — pero el early
  stopping lo corta solo.
- **El `±`** es la desviación entre los 5 folds. Por encima de 0,05 esa
  configuración es inestable y no conviene aunque tenga buena media.
- **Utilización de GPU.** Si `nvidia-smi` marca 30 % o menos, es normal: los
  modelos son pequeños y buena parte del tiempo se va en lanzar kernels, no en
  calcular. No es un problema que puedas arreglar sin salirte del grid del
  artículo.

### Orden de coste

`adultos_ft_transformer` es ~2/3 del total. Le siguen `ninos_ft_transformer` y
`adultos_mlp_ma`. Las tres combinaciones de jóvenes juntas no llegan al 3 %.

---

## 6. Si se corta

Cada combinación escribe su `results.json` **en cuanto termina**. Si el proceso
muere en la séptima, no pierdes las seis anteriores.

Mira qué falta:

```powershell
Get-ChildItem results\aligned -Directory | ForEach-Object {
  $ok = Test-Path "$($_.FullName)\results.json"
  "{0,-30} {1}" -f $_.Name, $(if ($ok) {"OK"} else {"FALTA"})
}
```

Y relanza solo eso:

```powershell
python run_pipeline.py train --groups adultos --models ft_transformer
```

Una combinación que falla **no detiene las demás**: el pipeline la registra en
`results/aligned/run_summary.json` con su traza y sigue.

---

## 7. Qué queda al final

### Por combinación — `results/aligned/<grupo>_<modelo>/`

| Archivo | Contenido |
|---|---|
| `results.json` | Métricas de test, curvas ROC, curvas de aprendizaje por época, interpretabilidad, resultado de la CV, hiperparámetros ganadores |
| `gridsearch_results.csv` | Las 16 configuraciones con media y desviación por fold |
| `gridsearch_best.json` | La configuración ganadora y el coste de la búsqueda |
| `test_predictions.npz` | `y_true`, `y_pred`, `y_proba` para recalcular lo que sea |
| `best_model.pt` / `tabnet_model.zip` | El modelo entrenado |

### Para el artículo — `outputs/aligned/`

| Archivo | Va a |
|---|---|
| `tabla3_metricas.csv` / `.tex` | **Tabla 3** |
| `tabla4_auc.csv` / `.tex` | **Tabla 4** |
| `copeland_global.csv` | Ranking Copeland (III-F-3) |
| `estabilidad_multisemilla.csv` | Media ± desviación, 5 semillas (III-F-4) |
| `interpretabilidad_top.csv` | Variables más influyentes (III-G) |
| `fig_matrices_confusion.png` | **Fig. 8** |
| `fig_curvas_roc.png` | **Fig. 9** |
| `fig_ranking_copeland.png` | **Fig. 10** |
| `fig_curvas_aprendizaje.png` | **Fig. 11** |
| `fig_importancia_variables.png` | Interpretabilidad |

Los `.tex` se pegan directo en el manuscrito.

**Si falta alguna combinación**, `report` la nombra y la deja fuera de la tabla.
No rellena huecos con nada.

---

## 8. Problemas frecuentes

| Síntoma | Causa y arreglo |
|---|---|
| `check` dice `GPU: no disponible` | PyTorch `+cpu`. Reinstala con el índice de CUDA |
| `training.device='cuda' pero PyTorch no ve ninguna GPU` | Igual que arriba; el pipeline se niega a caer a CPU en silencio |
| `CUDA out of memory` | Improbable con 8 GB. Si pasa, baja `batch_size` en el preset o cierra lo que esté usando la GPU |
| `FileNotFoundError ... train.parquet` | Falta correr `python run_pipeline.py etl` |
| `contradice el manuscrito (esperado ...)` | Cambiaste un valor que el artículo fija. Revierte, o corrige también el artículo |
| `FUGA DEL TARGET detectada` | Entró una variable de comportamiento lector a la matriz. No lo ignores: invalida los resultados |
| Va mucho más lento de lo estimado | Confirma con `nvidia-smi` que el proceso está en la GPU y no en CPU |

---

## 9. Qué esperar de los números

Los resultados van a ser **más bajos** que los del artículo actual (97 % / 92 % /
73 %). Aquellos venían de predecir la frecuencia lectora usando la frecuencia
lectora como variable de entrada. Estos son los reales, prediciendo a partir de
características demográficas y socioculturales.

También es posible que **el ranking entre arquitecturas cambie**. El único
Copeland que llegó a calcularse con datos reales ponía primero al FT-Transformer,
no a TabNet. La discusión del artículo está escrita asumiendo lo contrario, así
que habrá que reescribirla según lo que salga.

Antes de dar los resultados por buenos, revisa tres cosas:

1. **`estabilidad_multisemilla.csv`** — desviaciones muy pequeñas (< 0,001) en
   accuracy son sospechosas: suelen indicar fuga.
2. **Ningún accuracy cercano a 1,000.** Si aparece, algo se filtró.
3. **`interpretabilidad_top.csv`** — las variables más influyentes deberían ser
   nivel educativo, lengua materna, acceso digital y contexto del hogar. Si
   sube algo que huela a lectura, hay que investigarlo.
