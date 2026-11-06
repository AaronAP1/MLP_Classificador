# Pipeline design

This document explains the design decisions behind the pipeline: how the target is built, how leakage is prevented, why the number of features differs between groups, and which data-quality problems were corrected during development. For the commands, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md); for an overview, the main [README](../README.md).

---

## 1. Principles

- **One source of truth.** Every hyperparameter, threshold, seed and path lives in `configs/pipeline.yaml`. `src/aligned/config.py` refuses to start if a value fixed by the article (AdamW, 200 epochs, patience 20, 5 folds, Label Encoding, Min–Max scaling, 3 TabNet steps with sparsemax…) is changed.
- **Nothing is fitted on data it will be evaluated on.** Imputation, encoding, scaling and feature selection are fitted on the training split only, class weights are computed inside each fit, and the test split is used once, at the end.
- **No simulated outputs.** Tables and figures are built only from saved results; a missing combination is reported as missing.
- **Traceability.** `data/processed/<group>/config.json` records every ETL step, and each `results.json` records the configuration, hyperparameters, hardware and time that produced it.

## 2. Target definition (`target.py`)

| Group | Source | Rule |
|---|---|---|
| 3–17 years | `P506`: frequency of reading books | 1–2 → frequent · 3–5 → average · 6–8 → non-reader · 9 (don't know) → discarded |
| 18–64 years | `P408` (read books in the last 12 months) + `P409` (frequency) | `P408` = no → non-reader · `P409` 1–2 → frequent · 3–4 → average · 5–6 → non-reader |

Both rules measure **book reading**, so the classes are comparable across groups. The configuration also offers an alternative for adults based on any reading content (`P402` + `P403`, `target_source: chapter_300_400`). It was not used: with it, 67% of adults would be frequent readers, and the class would no longer be comparable with that of minors.

| Group | n | Non-reader | Average | Frequent |
|---|---|---|---|---|
| Children (3–13) | 15,333 | 2,632 | 4,095 | 8,606 |
| Youths (14–17) | 5,642 | 751 | 1,556 | 3,335 |
| Adults (18–64) | 40,530 | 23,216 | 6,539 | 10,775 |

## 3. Leakage control (`variables.py`)

Every survey variable is classified as predictor, administrative or reading-related. Reading-related variables (frequency, type and amount of reading, library use, reading motives…) are blocked by `is_leaky()`, and `assert_no_leakage()` stops the pipeline if one of them reaches the feature matrix. The check runs after preprocessing, when the data are loaded for training and in the tests. Survey weights (`FACTORFINAL_HOGAR`, `FACTOR200_FINAL`) and person/household order numbers are treated as administrative columns, never as predictors. A "weekly reading frequency" indicator exists in `features.py` but is never added to the features, because it is derived from the same questions as the target.

## 4. Features (`features.py`, `etl.py`)

**Why the number of features differs.** After removing reading variables, every group starts from the same 99 candidate variables. The ENL asks the socioeconomic chapter (mother tongue, ethnicity, education reached, occupation, income) only to people aged 18 or older. For children and youths those columns are constant, and the variance threshold correctly removes them. The final matrices have **32 features for children and youths and 56 for adults**.

**Household context (`hh_*`).** To give minors sociocultural information, each person receives indicators computed from the adults of their household: highest, mean and household-head educational level, literacy, mother tongue, Indigenous membership, ethnicity of the household head, employment rate, mean and maximum income, household size and composition. The averages include only the adults who answered each question. Literacy counts as literate whoever completed primary school, since the survey does not ask question P307 to them.

**Derived indices (`idx_*`).** Educational level (from `P306`, last level passed), type of mother tongue, and a **Digital Access Index**. The survey's household-equipment chapter is not in the extracts, so the index is approximated from socioeconomic stratum, size of the populated center and school attendance. `config.json` marks it as approximate.

**Selection.** Variance threshold 0.005 and correlation filter |r| > 0.95, both fitted on the training split. A mutual-information ranking would trim the matrix to 71 features (the number stated in an earlier draft of the article), but no group reaches that number, so it never removes anything.

**Nominal variables.** Label Encoding gives each category an integer index. Variables without a natural order (listed in `variables.NOMINAL_COLUMNS`: department, geographic domain, socioeconomic stratum, kinship, marital status, mother tongue, level attended, ethnicity of the household head, type of mother tongue…) are not fed to the networks as magnitudes: each category has a **learned embedding** in the three architectures. Ordinal and binary variables enter as scalars. Income is transformed with `log(1 + x)` before Min–Max scaling because of its strong skew.

## 5. Models (`models.py`)

| Model | Implementation |
|---|---|
| MLP+MA | `MultiHeadFeatureAttention`: every feature becomes a token (embedding for nominal variables, linear projection for the rest); multi-head self-attention `softmax(QKᵀ/√d)V` with 4 heads and dimension 64; 4 values per attended token feed a 256 → 128 → 64 → 32 MLP with ReLU, layer normalization and dropout |
| TabNet | `pytorch-tabnet` with 3 decision steps, sparsemax masks, n_d = n_a = 32, γ = 1.3, ghost batch normalization (virtual batch 128) and native categorical embeddings of size min(4, k − 1) |
| FT-Transformer | Feature tokenizer (64-dimensional embedding per feature), `[CLS]` token, 4 encoder layers with 8 heads and feed-forward 256; the class is read from the `[CLS]` token |

The three models share the weighted cross-entropy, AdamW and early-stopping criterion. Training uses gradient clipping (1.0), ReduceLROnPlateau and automatic mixed precision on GPU.

## 6. Protocol (`gridsearch.py`, `pipeline.py`)

1. **Grid Search + stratified 5-fold CV** on the 70% training split, selecting by mean F1-Macro. CV runs at most 60 epochs per fit (patience 12) to keep the cost manageable; the final models use up to 200.
2. **Stage A** (70% train vs 15% validation) gives the learning curves and the best epoch.
3. **Stage B** retrains on 85% (train + validation) for that fixed number of epochs. No early stopping is possible here, because there is no held-out data left.
4. **Test** once on the remaining 15%.
5. **Five seeds** (42, 123, 456, 789, 2024) repeat stage B and the test. Tables, Fig. 7 and the Copeland ranking use their mean. The per-seed probabilities are saved, so the ensemble is computed and compared seed by seed.

Stage A and stage B are separate on purpose. Retraining on 85% while early-stopping on data contained in that same 85% would turn the "validation" curve into a training curve.

## 7. Corrections made during development

All of them are part of the reported run (processed data marked `etl_version: 2`; `train` refuses data built by older versions).

| Problem found | Effect | Correction |
|---|---|---|
| `jovenes.csv` coded "not applicable" as blank, and some of its columns were duplicated by a merge (`_x`/`_y`) | The >20% missing-values filter discarded almost every row of the file: nobody aged 16–27 reached the models | `etl.harmonize_source` joins the duplicated columns and codes blanks as "not applicable", as in the other files |
| Oversampling (SMOTE) applied before cross-validation | Validation folds contained synthetic records built from training folds, so CV was optimistic | Balancing moved inside each fit; the default is a class-weighted loss with no synthetic records |
| Nominal variables entered as a single number in [0, 1] | The networks had to undo an arbitrary order (e.g. alphabetical departments) | Learned embeddings in the three architectures |
| Income scaled without transformation | Most values ended up near 0 and the variance threshold removed income | `log(1 + x)` before scaling |
| Household literacy counted as illiterate everyone who was not asked P307 | Respondents with complete primary education or more appeared as unable to read | Literate if P306 ≥ complete primary or P307 = yes; "not applicable" answers excluded from household averages |
| A free-text column (name of the Indigenous people) converted to a number | It was "no data" for a whole file, so it identified the source file | Text columns excluded (`variables.TEXT_COLUMNS`) with a guard in the ETL |
| Adults defined as 18+ | Chapters 300–400 apply only to ages 18–64: nobody aged 65+ has the class | Adults restricted to 18–64 |
| Tables and ranking computed with a single seed | The order of two models could depend on that seed | Mean ± SD of five seeds; Copeland on the means |
