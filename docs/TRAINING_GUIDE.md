# Training guide

Step-by-step procedure to retrain the nine models on a GPU workstation. The reported run used an NVIDIA RTX 4060 (8 GB) and took about 15 hours. For the design of the pipeline, see [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md).

Every table and figure comes from models trained on the ENL 2022 data. No path of the code produces simulated values: if a combination is missing, the report leaves it out instead of filling it in.

---

## 1. Installation (once)

```bash
git clone https://github.com/AaronAP1/MLP_Classificador.git
cd MLP_Classificador
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
```

**Install PyTorch with CUDA first**, then the rest. In the opposite order pip installs the `+cpu` build and everything silently runs on the processor:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## 2. Check the installation

```bash
python run_pipeline.py check
python -m pytest tests/ -q
```

`check` lists the dependencies, the raw and processed data and the configuration checks, and must show the GPU, for example:

```
  GPU: NVIDIA GeForce RTX 4060 (8.6 GB) · CUDA 12.6
```

If it reports `PyTorch +cpu`, reinstall PyTorch with the command of step 1. The tests must end with `passed` and no `failed`.

## 3. Prepare the data and run the smoke test (~5 min)

```bash
python run_pipeline.py etl
python run_pipeline.py smoke
```

`etl` rebuilds `data/processed/` from `data/raw/`. `smoke` runs the whole path (ETL → Grid Search → stages A and B → test → seeds → tables and figures) with 600 rows per split and 4 epochs, and must end with:

```
  9/9 combinaciones completas en X min
  figuras generadas: 10/10
  TODO FUNCIONA
```

It writes to `results/_smoke/` and `reports/_smoke/`, and every figure carries the stamp **"PRUEBA DE HUMO · NO SON RESULTADOS"** (smoke test, not results). Its numbers mean nothing: they only confirm that the code runs on that machine.

## 4. Estimate the training time (optional)

```bash
python run_pipeline.py benchmark
```

It times a few epochs of each combination on the local GPU and estimates the total for each Grid Search preset (worst case, without early stopping). The details are saved in `results/latest/benchmark.json`.

## 5. Train

Everything is set in [`configs/pipeline.yaml`](../configs/pipeline.yaml). The only usual decision is the size of the Grid Search:

| `gridsearch.preset` | Configurations per model | Use |
|---|---|---|
| `fast` | 1 | Testing only |
| `standard` | 16 (TabNet: 8) | **Used in the article** |
| `paper` | 108–324 | Full grid of Table 2; very expensive |

```bash
python run_pipeline.py train
```

To run it in the background on Windows, so that closing the session does not stop it:

```powershell
Start-Process python -ArgumentList "-u","run_pipeline.py","train" `
  -RedirectStandardOutput train.log -RedirectStandardError train.err -NoNewWindow
Get-Content train.log -Wait -Tail 30
```

For each of the 9 combinations (3 groups × 3 architectures) the pipeline runs:

1. **Grid Search** with stratified 5-fold cross-validation on the 70% training split.
2. **Stage A:** training on 70% and validation on 15% (up to 200 epochs, early stopping with patience 20). It gives the learning curves and the best epoch.
3. **Stage B:** retraining on 85% for that number of epochs.
4. **Test:** a single evaluation on the 15% that was never used.
5. **Seeds:** stage B and the test repeated with five seeds (mean ± SD).

**Resuming.** Each combination saves its `results.json` as soon as it ends. If the process stops, run only what is missing:

```bash
python run_pipeline.py train --groups adultos --models ft_transformer
```

**Retraining one architecture.** Only its folders in `results/latest/` are overwritten, and `report` rebuilds every table and figure with the 9 combinations:

```bash
python run_pipeline.py train --models tabnet
python run_pipeline.py report
```

**Separate runs.** `--tag <name>` writes to `results/<name>/` and `reports/<name>/`, and stamps the figures with the name of the run. Do not use `--tag final` for a new training run: it would overwrite the reported results.

## 6. Tables and figures

```bash
python run_pipeline.py report               # from results/latest
python run_pipeline.py report --tag final   # from the reported run
python paper/make_article_figures.py        # Figs. 7-12 and Table 4 of the article, from results/final
```

`report` writes CSV and LaTeX tables, PNG figures and an index (`INDICE_FIGURAS.md`) that maps each file to the article. The main README (section 5) gives the English correspondence.

## 7. Sanity checks before trusting a new run

- **No accuracy close to 1.000.** It would mean that information about reading leaked into the features.
- **`estabilidad_multisemilla.csv`:** standard deviations of practically zero (< 0.001) are also suspicious.
- **`interpretabilidad_top.csv`:** the most influential variables should be educational, geographic, linguistic, household or socioeconomic. Anything related to reading must be investigated.
- **`diagnostico` in each `results.json`:** the stage-A model and the final model should have similar test metrics.

## 8. Troubleshooting

The pipeline's console messages are in Spanish; the left column quotes them.

| Symptom | Cause and fix |
|---|---|
| `check` reports `PyTorch +cpu` | Reinstall PyTorch from the CUDA index (step 1) |
| `training.device='cuda' pero PyTorch no ve ninguna GPU` | Same as above; the pipeline refuses to fall back to the CPU silently |
| `CUDA out of memory` | Unlikely with 8 GB. Lower `batch_size` in the preset or close other programs that use the GPU |
| `FileNotFoundError ... train.parquet` | Run `python run_pipeline.py etl` first |
| `train` refuses the processed data (ETL version) | The data were built by an older ETL; run `python run_pipeline.py etl` again |
| `contradice el manuscrito (esperado ...)` | A value fixed by the article was changed in `pipeline.yaml`. Revert it |
| `FUGA DEL TARGET detectada` | A reading-behavior variable entered the features. Do not ignore it: it invalidates the results |
| Much slower than estimated | Check with `nvidia-smi` that the process is using the GPU |
