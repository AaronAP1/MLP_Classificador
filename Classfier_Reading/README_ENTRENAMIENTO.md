# Cómo lanzar el entrenamiento

Guía operativa para correr el pipeline en la PC con GPU (**RTX 4060, 8 GB**).
Para el diseño y las decisiones metodológicas, ver
[README_PIPELINE_ALINEADO.md](README_PIPELINE_ALINEADO.md).

Todo lo que produce este pipeline sale de entrenar modelos con los datos de
la ENL 2022. **Ninguna tabla ni figura se genera con valores simulados**, y
si falta una combinación, el reporte la omite en lugar de rellenarla.

---

## 1. Instalación (una sola vez)

Traer el código y los datos (los CSV de `data/unprocessed/` están en el
repositorio):

```powershell
git pull                      # o copiar la carpeta Classfier_Reading completa
cd Classfier_Reading
python -m venv .venv
.venv\Scripts\activate
```

**Primero PyTorch con CUDA**, después el resto. Si se invierte el orden, pip
instala la versión `+cpu` y todo corre en el procesador sin avisar:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-train.txt
```

`requirements-train.txt` tiene solo lo necesario. El `requirements.txt`
completo incluye Azure ML, Jupyter y mkdocs, que no hacen falta y pueden
fallar al instalarse.

## 2. Verificar

```powershell
python run_pipeline.py check
python -m pytest tests/test_aligned.py -q
```

`check` debe mostrar la GPU:

```
  GPU: NVIDIA GeForce RTX 4060 (8.6 GB) · CUDA 12.6
```

Si dice `no disponible · PyTorch +cpu`, reinstala torch con el comando del
paso 1. Las pruebas deben terminar en `passed`, sin `failed`.

## 3. Prueba de humo (~5 min)

```powershell
python run_pipeline.py etl
python run_pipeline.py smoke
```

Recorre todo el camino (ETL → Grid Search → etapas A y B → test → semillas →
tablas y figuras) con 600 filas por partición y 4 épocas. Debe terminar en:

```
  9/9 combinaciones completas en X min
  figuras generadas: 10/10
  TODO FUNCIONA
```

Escribe en `results/_smoke/` y `outputs/_smoke/`, y cada figura lleva la marca
**"PRUEBA DE HUMO · NO SON RESULTADOS"**. Sus números no significan nada: solo
confirman que el código funciona en esa PC.

## 4. Estimar el tiempo

```powershell
python run_pipeline.py benchmark
```

Cronometra unas épocas de cada combinación en esa GPU y estima el total por
preset (peor caso, sin early stopping). Deja el detalle en
`results/aligned/benchmark.json`.

## 5. Entrenar

Todo se configura en [configs/pipeline.yaml](configs/pipeline.yaml). Lo
normal es no tocar nada; la decisión principal es el tamaño del Grid Search:

| `gridsearch.preset` | Configuraciones por modelo | Uso |
|---|---|---|
| `fast` | 1 | Solo para probar |
| `standard` | 16 | **Recomendado** |
| `paper` | 108–324 | Tabla 2 completa; muy costoso |

```powershell
python run_pipeline.py train
```

En segundo plano, para que no lo corte un cierre de sesión:

```powershell
Start-Process python -ArgumentList "-u","run_pipeline.py","train" `
  -RedirectStandardOutput train.log -RedirectStandardError train.err -NoNewWindow
Get-Content train.log -Wait -Tail 30
```

Para cada una de las 9 combinaciones (3 grupos × 3 arquitecturas):

1. **Grid Search** con validación cruzada estratificada de 5 folds sobre el 70 %.
2. **Etapa A:** entrena con el 70 % y valida con el 15 % (hasta 200 épocas,
   early stopping con paciencia 20). Produce las curvas de la Fig. 11 y la
   mejor época.
3. **Etapa B:** reentrena con el 85 % durante esa cantidad de épocas.
4. **Test:** evalúa una sola vez sobre el 15 % que nunca se usó.
5. **Semillas:** repite la etapa B con 5 semillas (media ± desviación).

**Es reanudable.** Cada combinación guarda su `results.json` al terminar. Si el
proceso se corta, relanza solo lo que falte:

```powershell
python run_pipeline.py train --groups adultos --models ft_transformer
```

**Reentrenar solo una arquitectura.** Sobrescribe únicamente sus carpetas en
`results/aligned/`; las de los otros modelos se conservan y `report` vuelve a
armar todas las tablas y figuras con las 9 combinaciones:

```powershell
python run_pipeline.py train --models tabnet
python run_pipeline.py report
```

> **Octubre de 2026 · hay que repetir TODO (ETL v2).** Cambiaron los datos
> y la forma en que los modelos leen las variables nominales (embeddings), así
> que los resultados del 1-oct-2026 no se mezclan con los nuevos:
>
> ```powershell
> git pull
> python run_pipeline.py etl        # obligatorio: train se niega a usar datos v1
> python -m pytest tests/test_aligned.py -q
> python run_pipeline.py smoke
> python run_pipeline.py train      # las 9 combinaciones, ~13-15 h en la RTX 4060
> python run_pipeline.py report
> ```
>
> Antes de lanzar, conviene mover la carpeta `results/aligned` de la corrida
> anterior a otro nombre (por ejemplo `results/aligned_v1`) para conservarla.

## 6. Tablas y figuras

```powershell
python run_pipeline.py report
```

Todo queda en `outputs/aligned/`, con un `INDICE_FIGURAS.md` que indica a
qué parte del artículo va cada archivo:

| Archivo | En el artículo |
|---|---|
| `tabla_resumen_datos.csv` | Descripción de los datos (registros, edades, clases, n.º de variables) |
| `tabla2_hiperparametros.csv` | Table 2 (valores elegidos) |
| `tabla3_media_std.csv/.tex` | **Table 3**: media ± desviación en 5 semillas, con el ensemble |
| `tabla3_metricas.csv/.tex` | Complementaria: solo la semilla 42 (la de las matrices y las ROC) |
| `tabla4_auc.csv/.tex` | Table 4 (AUC macro one-vs-rest) |
| `tabla_ensemble.csv` | Texto del ensemble: ganancia frente al mejor modelo, semilla a semilla |
| `fig_comparacion_grupos.png` | Fig. 7 (media ± desviación) |
| `fig_matrices_confusion.png` | Fig. 8 |
| `fig_curvas_roc.png` | Fig. 9 |
| `fig_ranking_copeland.png` | Fig. 10 (sobre las medias de las semillas) |
| `fig_curvas_aprendizaje.png` | Fig. 11 (pérdida; en accuracy: `fig_curvas_aprendizaje_accuracy.png`) |
| `fig_importancia_variables.png`, `fig_importancia_permutacion.png`, `fig_mascaras_tabnet.png`, `fig_atencion_ft.png`, `interpretabilidad_top.csv` | Interpretabilidad |

Cada `results/aligned/<grupo>_<modelo>/results.json` guarda además el equipo
en que se entrenó (`hardware`), los hiperparámetros, las curvas, las
predicciones y el tiempo de cómputo. Conviene conservar esa carpeta junto al
manuscrito: permite demostrar de dónde sale cada número.

## 7. Qué hay que actualizar en el artículo

El código implementa lo que describe el artículo, con estas diferencias que
el texto debe recoger:

1. **Definición de las clases.** Menores (3–17): P506, frecuencia de lectura
   de libros. Adultos (18–64): P408 + P409, lectura de libros. Frecuente =
   diariamente o varias veces por semana; Medio = de una vez por semana a
   una vez al mes; No lector = cada tres meses o menos, o no leyó.
2. **Desbalance de clases.** Entropía cruzada ponderada por el inverso de la
   frecuencia de cada clase (sin SMOTE).
3. **Número de variables.** No son 71 en todos los grupos: niños 32, jóvenes
   32, adultos 56 (ETL v2). El capítulo 300 solo se aplica a mayores de 18, así que a
   los menores les llegan menos variables propias.
4. **Indicadores derivados.** Hay que quitar "Frecuencia Lectora Semanal",
   porque es la misma información que la clase a predecir y usarla sería
   fuga. El "Índice de Acceso Digital" se construye con estrato
   socioeconómico, tamaño del centro poblado y asistencia a educación como
   aproximación, porque el capítulo 600 no está en los datos: hay que
   decirlo así.
5. **Protocolo.** El número de épocas del reentrenamiento sobre el 85 % se
   fija con la mejor época de la corrida train/validación. Las tres
   arquitecturas detienen el entrenamiento con la misma pérdida ponderada.
   **Table 2:** TabNet se busca con learning rate 0,02 y 0,005 (rango que
   recomiendan sus autores, ref. [15]) y sin dropout, que no usa; MLP-MA y
   FT-Transformer con 0,001 y 0,0001.
6. **AUC.** Promedio macro one-vs-rest.
7. **Copeland.** Con el ensemble son 4 competidores: rango de −9 a +9
   (3 grupos, 3 rivales por grupo). Se calcula sobre la media de las 5
   semillas, no sobre una sola.
8. **Equipo.** El que registre `results.json` → `hardware`.
9. **Variables nominales (Métodos, III-A y III-C).** El Label Encoding se
   mantiene, pero las variables sin orden (departamento, dominio, estrato
   socioeconómico, lengua materna, etnia, parentesco, estado civil,
   ocupación, tipo de centro, nivel que cursa) no entran como número: cada
   categoría tiene un embedding aprendido. En FT-Transformer es su Feature
   Tokenizer original (Gorishniy et al., 2021); en MLP-MA, el token de
   atención de esa variable; en TabNet, sus embeddings categóricos nativos.
   Las ordinales y binarias siguen como escalar.
10. **Ingreso.** Se transforma con log(1 + x) antes del Min-Max.
11. **Adultos: 18 a 64 años**, no "18+": los capítulos 300 y 400 de la ENL
    solo se aplican en ese rango.
12. **Indicadores del hogar.** Promedian solo a los adultos que respondieron
    cada pregunta; el alfabetismo cuenta como alfabeta a quien terminó la
    primaria (la ENL no le pregunta P307).
13. **Table 3 y ensemble.** Media ± desviación en 5 semillas. El ensemble
    se calcula en cada semilla (promedio de las probabilidades de los tres
    modelos de esa semilla) y se compara semilla a semilla.
14. **Resultados, discusión y conclusiones:** reescribir con los números que
    salgan. Serán más bajos que los del manuscrito actual (97 / 92 / 73 %).

## 8. Antes de dar los resultados por buenos

- **Ningún accuracy cercano a 1,000.** Si aparece, algo se filtró.
- **`estabilidad_multisemilla.csv`:** desviaciones prácticamente nulas
  (< 0,001) también son sospechosas.
- **`interpretabilidad_top.csv`:** las variables más influyentes deberían ser
  nivel educativo, lengua materna, contexto del hogar o nivel
  socioeconómico. Si aparece algo que huela a lectura, hay que investigarlo.
- **`diagnostico` en cada `results.json`:** el modelo de la etapa A y el final
  deben tener métricas de test parecidas.

## 9. Problemas frecuentes

| Síntoma | Causa y arreglo |
|---|---|
| `check` dice `PyTorch +cpu` | Reinstalar torch con el índice de CUDA (paso 1) |
| `training.device='cuda' pero PyTorch no ve ninguna GPU` | Igual que arriba; el pipeline se niega a pasar a CPU en silencio |
| `CUDA out of memory` | Improbable con 8 GB. Bajar `batch_size` en el preset o cerrar lo que use la GPU |
| `FileNotFoundError ... train.parquet` | Falta `python run_pipeline.py etl` |
| `contradice el manuscrito (esperado ...)` | Se cambió un valor que el artículo fija. Revertir, o corregir también el artículo |
| `FUGA DEL TARGET detectada` | Entró una variable de comportamiento lector. No ignorarlo: invalida los resultados |
| Mucho más lento de lo estimado | Confirmar con `nvidia-smi` que el proceso usa la GPU |
