# Data

All data come from Peru's **2022 National Reading Survey** (*Encuesta Nacional de Lectura 2022*, ENL 2022), carried out by the National Institute of Statistics and Informatics (INEI) with the Ministry of Culture and published as anonymized open data on the Peruvian National Open Data Platform (<https://www.datosabiertos.gob.pe>).

```
data/
├── raw/            Input of the pipeline (read-only)
├── dictionaries/   Official ENL data dictionaries
└── processed/      Output of `python run_pipeline.py etl`
```

## `raw/` · ENL 2022 extracts

One row per person. Each file joins, for each person, the survey chapters that apply to them, plus the household and sample-design identifiers (department, geographic domain, stratum, household and person numbers).

| File | Rows | Columns | Ages in the file | ENL chapters |
|---|---|---|---|---|
| `ninos.csv` | 21,652 | 422 | 0–15 | 200 (household members), 500 (reading practices, ages 0–17) |
| `jovenes.csv` | 14,021 | 433 | 16–27 | 200, 300–400 (ages 18–64) and 500 (ages 16–17) |
| `adultos.csv` | 40,559 | 422 | 28–99 | 200, 300–400 |

The age bands of the files do not match the study's groups. The ETL joins the three files and segments them again by exact age (`edad_total`): children 3–13, youths 14–17 and adults 18–64. It also harmonizes the coding: `jovenes.csv` codes "not applicable" as a blank cell, while the other two use 0. The question codes (`P2xx`, `P3xx`…) are those of the official questionnaire and are described in `dictionaries/`.

## `dictionaries/` · official data dictionaries

The ENL data dictionaries in Spanish, one per chapter (100–600 household characteristics, 200 household members, 300–400 reading practices of people aged 18–64, 500 reading practices of people aged 0–17). `datasets_info.json` lists the rows, columns, data types and missing-value counts of each raw file.

## `processed/` · ETL output (ETL version 2)

One folder per age group (`ninos`, `jovenes`, `adultos`):

| File | Content |
|---|---|
| `train.parquet`, `val.parquet`, `test.parquet` | Stratified 70 / 15 / 15 split (seed 42). One column per selected feature, already encoded and scaled to [0, 1], plus `target` (0 = non-reader, 1 = average reader, 2 = frequent reader) |
| `config.json` | Traceability of every ETL step: age segmentation, target rule and class counts, derived indicators, excluded columns, cleaning, feature selection (variance and correlation filters), encoding (number of categories and embedding size of each nominal variable), splits and the final feature list |

The fitted preprocessor (`preprocessor.pkl`) is not versioned. `python run_pipeline.py etl` rebuilds it, together with the three splits, from `raw/` in about ten seconds.

**Encoding.** A categorical variable with *k* categories is stored as `code / (k − 1)` in [0, 1]. The models turn the variables listed under `codificacion.embeddings` in `config.json` back into their category index and give each category a learned embedding.
