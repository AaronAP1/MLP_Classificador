# Results

`final/` is the run reported in the article: the 9 combinations of age group (`ninos`, `jovenes`, `adultos`) × architecture (`mlp_ma`, `tabnet`, `ft_transformer`), trained with the `standard` Grid Search preset and five seeds on an NVIDIA RTX 4060. New runs are written to `results/latest/` (or `results/<tag>/`) and are not versioned.

```
results/final/
├── run_summary.json                  Status of the 9 combinations and hardware of the run
└── <group>_<model>/                  e.g. adultos_ft_transformer
    ├── results.json                  Everything about the combination (see below)
    ├── gridsearch_results.csv        Every Grid Search configuration: CV mean and SD of
    │                                 Accuracy, F1-Macro and F1-Weighted over the 5 folds
    ├── gridsearch_best.json          Selected configuration and its CV score
    ├── test_predictions.npz          Seed-42 model on the test set: y_true, y_pred, y_proba (n × 3)
    ├── test_predictions_seeds.npz    The five seeds: y_true, seeds, y_proba (5 × n × 3)
    ├── interpretabilidad.npz         feature_names, permutation importance, attention / masks
    └── best_model.pt | tabnet_model.zip    Weights of the seed-42 model (PyTorch / pytorch-tabnet)
```

Class indices in every file: **0 = non-reader** (*No Lector*), **1 = average reader** (*Lector Medio*), **2 = frequent reader** (*Lector Frecuente*). The ensemble is not stored: it is the mean of the three models' `y_proba`, seed by seed, and `report` and `paper/make_article_figures.py` compute it.

## Keys of `results.json`

| Key | Content |
|---|---|
| `grupo`, `modelo`, `seed` | Age group, architecture and main seed |
| `hardware` | Device, GPU, CUDA, PyTorch and Python versions |
| `n_features`, `features`, `embeddings_categoricas` | Input features and the number of categories of each embedded nominal variable |
| `n_muestras` | Size of the train (70%), validation (15%), test (15%) and refit (85%) sets |
| `gridsearch` | Preset, folds, CV epochs, number of configurations and fits, best parameters and CV score |
| `hiperparametros_finales` | Hyperparameters used for the final model |
| `balanceo` | Class weights of the weighted loss and the training class distribution |
| `protocolo` | Description of the CV, stage A, stage B and test steps actually run |
| `mejor_epoca_etapa_a`, `epocas_ejecutadas_etapa_a` | Best epoch of stage A (= length of stage B) and epochs run before early stopping |
| `test_metrics` | Seed-42 model on the test set: accuracy, weighted/macro precision, recall and F1, macro and weighted one-vs-rest AUC, AUC per class, confusion matrix (raw and row-normalized), classification report |
| `roc_curves` | One-vs-rest ROC curve (FPR, TPR, AUC) of each class |
| `learning_curves` | Stage-A loss and accuracy per epoch, for training and validation |
| `diagnostico` | Test metrics of the stage-A model (trained only on 70%), to compare with the final model |
| `interpretabilidad` | Permutation importance (drop in test F1-Macro) and attention / mask importance, top 15 and full ranking |
| `estabilidad` | The five seeds: metrics of each run and their mean and standard deviation |
| `elapsed_sec` | Wall-clock time of the combination |
