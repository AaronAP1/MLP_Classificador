"""
Grid Search + validacion cruzada estratificada 5-Fold (seccion III-E-4).

Este modulo es la pieza que la auditoria encontro ausente: el articulo
dedica una subseccion y una figura a la busqueda de hiperparametros, y el
repositorio no contenia ninguna referencia a `StratifiedKFold`,
`GridSearchCV` ni nada equivalente; `results/gridsearch/` estaba vacio.

Procedimiento, tal como lo describe el articulo:
    1. Cada configuracion candidata se entrena sobre el 70% (train).
    2. Se evalua con 5-Fold estratificado dentro de ese 70%.
    3. Se elige la configuracion con mejor metrica promedio.
    4. El reentrenamiento final (en `pipeline.py`) usa 70% + 15% = 85%.

Durante la CV se usa `cv_epochs` (60 por defecto) en lugar de 200: son
5 folds por configuracion y el coste con 200 epocas es prohibitivo. El
modelo final si se entrena con las 200 epocas que declara el articulo.
Esta diferencia queda registrada en el JSON de resultados.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from .balancing import balance_training_set
from .models import IS_TORCH, build_model
from .pipeline import resolve_amp, resolve_device, set_seed
from .trainer import TorchTrainer, make_loader


def build_grid(cfg: dict, model_name: str) -> List[Dict[str, object]]:
    """
    Producto cartesiano del preset activo (Tabla 2 del articulo).

    `depth` significa distinto segun la arquitectura:
        mlp_ma          -> numero de capas densas (3-5)
        tabnet          -> numero de decision steps (2-4)
        ft_transformer  -> fijo en 4 encoders, no se explora
    """
    preset_name = cfg["gridsearch"]["preset"]
    preset = cfg["gridsearch"]["presets"][preset_name]

    lrs = list(preset["lr"])
    dropouts = list(preset["dropout"])
    depths = list(preset["depth"])
    if model_name == "ft_transformer":
        # III-C-3: 4 capas encoder, valor estructural que no se explora.
        depths = [cfg["models"]["ft_transformer"]["n_layers"]]
    elif model_name == "tabnet":
        # Tabla 2: "TabNet: 2-4 decision steps". Solo el preset "paper"
        # los mueve; el resto se queda en los 3 que fija el articulo.
        base = cfg["models"]["tabnet"]["n_steps"]
        depths = (sorted({min(max(int(d) - 1, 2), 4) for d in depths})
                  if preset_name == "paper" else [base])
        # Rango de learning rate propio (ver configs/pipeline.yaml) y sin
        # explorar dropout, que TabNet no usa: evita configuraciones repetidas.
        lrs = list(preset.get("lr_tabnet", lrs))
        dropouts = dropouts[:1]

    combos = itertools.product(
        lrs, preset["batch_size"], dropouts,
        preset["weight_decay"], depths,
    )
    grid = []
    for lr, bs, dr, wd, dp in combos:
        hp = {"lr": lr, "batch_size": bs, "dropout": dr,
              "weight_decay": wd, "depth": dp}
        if model_name == "tabnet":
            hp["n_steps"] = dp
        grid.append(hp)
    return grid


def _fit_and_score(
    cfg: dict, model_name: str, hp: dict,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    epochs: int, patience: int, seed: int,
) -> Dict[str, float]:
    """Entrena una configuracion en un fold y devuelve sus metricas."""
    # El balanceo se aplica aqui, SOLO a la parte de entrenamiento del fold.
    # La parte de validacion conserva su distribucion natural.
    X_tr, y_tr, weights, _ = balance_training_set(X_tr, y_tr, cfg, seed)

    set_seed(seed)
    model = build_model(model_name, X_tr.shape[1], cfg, hp, seed)

    if IS_TORCH[model_name]:
        trainer = TorchTrainer(
            model, lr=hp["lr"], weight_decay=hp["weight_decay"],
            class_weights=weights,
            gradient_clip=cfg["training"]["gradient_clip"],
            scheduler_patience=cfg["training"]["scheduler_patience"],
            scheduler_factor=cfg["training"]["scheduler_factor"],
            device=resolve_device(cfg), amp=resolve_amp(cfg),
        )
        trainer.fit(
            make_loader(X_tr, y_tr, hp["batch_size"], True, seed),
            make_loader(X_va, y_va, hp["batch_size"], False, seed),
            epochs=epochs, patience=patience, verbose=False,
        )
        y_pred = trainer.predict(X_va)
    else:
        model.fit(
            X_tr, y_tr, X_va, y_va, epochs=epochs, patience=patience,
            batch_size=hp["batch_size"],
            virtual_batch_size=cfg["models"]["tabnet"]["virtual_batch_size"],
            class_weights=weights,
        )
        y_pred = model.predict(X_va)

    return {
        "accuracy": float(accuracy_score(y_va, y_pred)),
        "f1_macro": float(f1_score(y_va, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_va, y_pred, average="weighted", zero_division=0)),
    }


def grid_search_cv(
    cfg: dict, model_name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    out_dir: Optional[Path] = None, verbose: bool = True,
) -> Dict[str, object]:
    """
    Ejecuta la busqueda completa y devuelve la mejor configuracion junto
    con la tabla de todos los candidatos evaluados.
    """
    gs = cfg["gridsearch"]
    seed = cfg["seed"]
    grid = build_grid(cfg, model_name)
    scoring = gs["scoring"]

    skf = StratifiedKFold(n_splits=gs["cv_folds"], shuffle=True, random_state=seed)
    rows: List[Dict[str, object]] = []
    started = time.time()

    if verbose:
        print(f"  Grid Search · {model_name} · preset '{gs['preset']}' · "
              f"{len(grid)} configuraciones x {gs['cv_folds']} folds "
              f"= {len(grid) * gs['cv_folds']} ajustes")

    for i, hp in enumerate(grid, 1):
        fold_scores: List[Dict[str, float]] = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            fold_scores.append(_fit_and_score(
                cfg, model_name, hp,
                X_train[tr_idx], y_train[tr_idx],
                X_train[va_idx], y_train[va_idx],
                epochs=gs["cv_epochs"], patience=gs["cv_patience"], seed=seed,
            ))

        row: Dict[str, object] = dict(hp)
        for key in ("accuracy", "f1_macro", "f1_weighted"):
            vals = [f[key] for f in fold_scores]
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_std"] = float(np.std(vals, ddof=0))
        rows.append(row)

        if verbose:
            print(f"    [{i:>3}/{len(grid)}] lr={hp['lr']:<8} bs={hp['batch_size']:<4} "
                  f"do={hp['dropout']:<4} wd={hp['weight_decay']:<9} d={hp['depth']} "
                  f"-> {scoring}={row[f'{scoring}_mean']:.4f} "
                  f"(±{row[f'{scoring}_std']:.4f})")

    table = pd.DataFrame(rows).sort_values(f"{scoring}_mean", ascending=False)
    best_row = table.iloc[0].to_dict()
    keys = ["lr", "batch_size", "dropout", "weight_decay", "depth"]
    if model_name == "tabnet":
        keys.append("n_steps")
    best_hp = {k: best_row[k] for k in keys if k in best_row}
    best_hp["batch_size"] = int(best_hp["batch_size"])
    best_hp["depth"] = int(best_hp["depth"])
    if "n_steps" in best_hp:
        best_hp["n_steps"] = int(best_hp["n_steps"])

    result: Dict[str, object] = {
        "model": model_name,
        "preset": gs["preset"],
        "cv_folds": gs["cv_folds"],
        "cv_epochs": gs["cv_epochs"],
        "final_epochs": cfg["training"]["final_epochs"],
        "nota_epocas": (
            f"La CV usa {gs['cv_epochs']} epocas por coste; el modelo final "
            f"se entrena con {cfg['training']['final_epochs']} epocas."
        ),
        "scoring": scoring,
        "n_configs": len(grid),
        "n_fits": len(grid) * gs["cv_folds"],
        "elapsed_sec": round(time.time() - started, 2),
        "best_params": best_hp,
        "best_cv_score": float(best_row[f"{scoring}_mean"]),
        "best_cv_std": float(best_row[f"{scoring}_std"]),
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / "gridsearch_results.csv", index=False)
        with open(out_dir / "gridsearch_best.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)

    if verbose:
        print(f"  mejor: {best_hp} -> {scoring}={result['best_cv_score']:.4f}")
    return result
