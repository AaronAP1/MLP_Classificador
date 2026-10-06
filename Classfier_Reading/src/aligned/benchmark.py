"""
Estimacion del coste de entrenamiento.

Mide el tiempo real de unas pocas epocas por combinacion (grupo, modelo)
sobre los datos ya procesados y extrapola el coste total del pipeline
(peor caso: sin early stopping):

    Grid Search   = n_configs x cv_folds x cv_epochs x t_epoca(80% del train)
    Etapa A       = final_epochs x t_epoca(70%)            train vs. validacion
    Etapa B       = final_epochs x t_epoca(85%)            reentrenamiento
    Multi-semilla = n_seeds x Etapa B

No guarda modelos ni metricas: es solo un cronometro. Los pesos que se
crean aqui se descartan.

Uso:
    python run_pipeline.py benchmark
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .gridsearch import build_grid
from .models import IS_TORCH, MODEL_NAMES, build_model
from .pipeline import load_categorical_info, load_group, set_seed
from .trainer import TorchTrainer, compute_class_weights, make_loader

PROBE_EPOCHS = 3        # epocas cronometradas por combinacion


def _fmt(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _time_epochs(cfg: dict, model_name: str, X: np.ndarray, y: np.ndarray,
                 hp: dict, epochs: int = PROBE_EPOCHS, cat_info=None) -> float:
    """Segundos por epoca, medidos sobre datos reales."""
    set_seed(cfg["seed"])
    weights = compute_class_weights(y, cfg["n_classes"])

    # Particion estratificada para que el "val" del cronometro tenga las
    # tres clases, como en el entrenamiento real.
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.15, random_state=cfg["seed"], stratify=y)

    model = build_model(model_name, X.shape[1], cfg, hp, cfg["seed"], cat_info=cat_info)

    if IS_TORCH[model_name]:
        trainer = TorchTrainer(
            model, lr=hp["lr"], weight_decay=hp["weight_decay"],
            class_weights=weights, gradient_clip=cfg["training"]["gradient_clip"],
        )
        tl = make_loader(X_tr, y_tr, hp["batch_size"], True, cfg["seed"])
        vl = make_loader(X_va, y_va, hp["batch_size"], False, cfg["seed"])
        trainer._run_epoch(tl, train=True)                # calentamiento
        started = time.time()
        for _ in range(epochs):
            trainer._run_epoch(tl, train=True)
            trainer._run_epoch(vl, train=False)
        return (time.time() - started) / epochs

    started = time.time()
    model.fit(X_tr, y_tr, X_va, y_va, epochs=epochs, patience=epochs + 1,
              batch_size=hp["batch_size"],
              virtual_batch_size=cfg["models"]["tabnet"]["virtual_batch_size"],
              class_weights=weights)
    return (time.time() - started) / epochs


def run_benchmark(cfg: dict, root: Path, groups: Optional[List[str]] = None,
                  models: Optional[List[str]] = None) -> Dict[str, object]:
    groups = groups or list(cfg["groups"])
    models = models or MODEL_NAMES
    gs_cfg, tr_cfg = cfg["gridsearch"], cfg["training"]

    device = "GPU " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Hardware medido        : {device}")
    print(f"  Epocas cronometradas   : {PROBE_EPOCHS} por combinacion")
    print(f"  Multi-semilla activa   : {cfg['evaluation']['stability']['multiseed']}"
          f" ({len(cfg['evaluation']['stability']['seeds'])} semillas)\n")

    hp_probe = {"lr": 1e-3, "batch_size": 128, "dropout": 0.3,
                "weight_decay": 1e-4, "depth": 4}
    n_seeds = (len(cfg["evaluation"]["stability"]["seeds"])
               if cfg["evaluation"]["stability"]["multiseed"] else 0)

    per_epoch: Dict[str, float] = {}
    sizes: Dict[str, int] = {}
    n_train: Dict[str, int] = {}

    for group in groups:
        try:
            data, feats = load_group(cfg, group, root)
        except FileNotFoundError as exc:
            print(f"  {group}: {exc}")
            continue
        X_tr, y_tr = data["train"]
        X_val, _ = data["val"]
        sizes[group] = len(X_tr) + len(X_val)
        n_train[group] = len(X_tr)

        for model_name in models:
            key = f"{group}_{model_name}"
            print(f"  midiendo {key} ...", end=" ", flush=True)
            try:
                t = _time_epochs(cfg, model_name, X_tr, y_tr, hp_probe,
                                 cat_info=load_categorical_info(cfg, group, root, feats))
                per_epoch[key] = t
                print(f"{t:.2f} s/epoca  (train n={len(X_tr):,}, {len(feats)} features)")
            except Exception as exc:                      # noqa: BLE001
                print(f"ERROR {type(exc).__name__}: {exc}")

    if not per_epoch:
        print("\n  No se pudo medir nada. Ejecuta antes: python run_pipeline.py etl")
        return {}

    # ----------------------------------------------------------------
    print(f"\n{'-' * 70}\n  ESTIMACION POR PRESET DE GRID SEARCH\n{'-' * 70}")
    estimates: Dict[str, object] = {}

    for preset in ("fast", "standard", "paper"):
        cfg_preset = dict(cfg)
        cfg_preset["gridsearch"] = {**gs_cfg, "preset": preset}
        total = 0.0
        detail = {}

        for key, t_epoch in per_epoch.items():
            group, model_name = key.split("_", 1)
            n_configs = len(build_grid(cfg_preset, model_name))

            # CV: cada fold entrena sobre 4/5 del train.
            t_cv = n_configs * gs_cfg["cv_folds"] * gs_cfg["cv_epochs"] * t_epoch * 0.8
            # Etapa A sobre el 70% y etapa B sobre el 85%, escalada por tamaño.
            scale = sizes[group] / max(n_train[group], 1)
            t_stage_a = tr_cfg["final_epochs"] * t_epoch
            t_refit = tr_cfg["final_epochs"] * t_epoch * scale
            t_seeds = n_seeds * t_refit
            sub = t_cv + t_stage_a + t_refit + t_seeds
            detail[key] = {"n_configs": n_configs, "grid_search_s": t_cv,
                           "stage_a_s": t_stage_a, "refit_s": t_refit,
                           "multiseed_s": t_seeds, "total_s": sub}
            total += sub

        estimates[preset] = {"total_s": total, "detail": detail}
        n_fits = sum(d["n_configs"] for d in detail.values()) * gs_cfg["cv_folds"]
        print(f"\n  preset '{preset}'  ({n_fits:,} ajustes de CV en total)")
        print(f"    Grid Search + CV : {_fmt(sum(d['grid_search_s'] for d in detail.values()))}")
        print(f"    Etapa A (70/15)  : {_fmt(sum(d['stage_a_s'] for d in detail.values()))}")
        print(f"    Etapa B (85%)    : {_fmt(sum(d['refit_s'] for d in detail.values()))}")
        print(f"    Multi-semilla    : {_fmt(sum(d['multiseed_s'] for d in detail.values()))}")
        print(f"    TOTAL            : {_fmt(total)}")

    print(f"\n{'-' * 70}")
    print("  Nota: estimacion de peor caso. El early stopping (paciencia 20) de la")
    print("  etapa A fija cuantas epocas usan la etapa B y las semillas; en la")
    print("  practica suele recortar entre un 30% y un 60% del tiempo estimado.")
    print("  Una GPU reduce el tiempo de MLP+MA y FT-Transformer entre 5x y 15x")
    print("  frente a CPU; TabNet mejora menos porque su cuello de botella es el")
    print("  preprocesamiento por lotes.")

    out = root / cfg["paths"]["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "hardware": device,
        "probe_epochs": PROBE_EPOCHS,
        "segundos_por_epoca": per_epoch,
        "n_muestras_85pct": sizes,
        "estimaciones": estimates,
    }
    with open(out / "benchmark.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Detalle guardado en {out / 'benchmark.json'}")
    return payload
