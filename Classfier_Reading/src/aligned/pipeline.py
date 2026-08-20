"""
Orquestador de entrenamiento y evaluacion (secciones III-D, III-E, III-G).

Para cada par (grupo, modelo) ejecuta:

    1. Grid Search + 5-Fold CV estratificado sobre el 70% (train).
    2. Reentrenamiento con la mejor configuracion sobre el 85%
       (train + val), 200 epocas y early stopping de paciencia 20.
    3. Evaluacion unica sobre el 15% de test, jamas visto en 1 ni 2.
    4. Interpretabilidad: mascaras de TabNet, atencion de FT-Transformer
       y atencion multi-cabeza del MLP+MA.
    5. Estabilidad multi-semilla (5 semillas) si esta habilitada.

Todo artefacto que el articulo reporta debe salir de aqui. No existe
ninguna ruta que fabrique metricas ni predicciones sinteticas.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from . import variables as V
from .evaluate import compute_metrics, roc_curves, aggregate_seed_runs
from .models import IS_TORCH, MODEL_NAMES, build_model
from .trainer import TorchTrainer, compute_class_weights, make_loader


# =====================================================================
# Datos
# =====================================================================
def load_group(cfg: dict, group: str, root: Path):
    """Carga los tres splits ya procesados y verifica que no haya fuga."""
    d = root / cfg["paths"]["processed_dir"] / group
    parts = {}
    for name in ("train", "val", "test"):
        path = d / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"No existe {path}. Ejecuta primero:  python run_pipeline.py etl"
            )
        parts[name] = pd.read_parquet(path)

    feature_cols = [c for c in parts["train"].columns if c != "target"]
    V.assert_no_leakage(feature_cols)

    for name in ("val", "test"):
        other = [c for c in parts[name].columns if c != "target"]
        if other != feature_cols:
            raise AssertionError(
                f"Esquema inconsistente entre train y {name} en el grupo {group}."
            )

    out = {}
    for name, frame in parts.items():
        out[name] = (
            frame[feature_cols].to_numpy(dtype=np.float32),
            frame["target"].to_numpy().astype(int),
        )
    return out, feature_cols


def resolve_device(cfg: dict) -> str:
    """Dispositivo efectivo: `auto` usa GPU si la hay."""
    want = cfg["training"].get("device", "auto")
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if want == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "training.device='cuda' pero PyTorch no ve ninguna GPU. "
            "Instala la build con CUDA o usa device: auto."
        )
    return want


def resolve_amp(cfg: dict) -> bool:
    """Precision mixta: solo tiene sentido en GPU."""
    want = cfg["training"].get("amp", "auto")
    if want == "auto":
        return resolve_device(cfg).startswith("cuda")
    return bool(want)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# Entrenamiento final
# =====================================================================
def train_final(
    cfg: dict, model_name: str, hp: dict,
    X_fit: np.ndarray, y_fit: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    seed: int, verbose: bool = True,
) -> Tuple[object, Dict[str, list]]:
    """
    Reentrena sobre el 85% con la configuracion ganadora.

    `X_val` se usa unicamente como senal de early stopping. Es un
    subconjunto del propio 85%, de modo que el 15% de test permanece
    intacto (III-E-5).
    """
    set_seed(seed)
    tcfg = cfg["training"]
    weights = (compute_class_weights(y_fit, cfg["n_classes"])
               if tcfg["use_class_weights"] else None)

    model = build_model(model_name, X_fit.shape[1], cfg, hp, seed)

    if IS_TORCH[model_name]:
        trainer = TorchTrainer(
            model, lr=hp["lr"], weight_decay=hp["weight_decay"],
            class_weights=weights,
            gradient_clip=tcfg["gradient_clip"],
            scheduler_patience=tcfg["scheduler_patience"],
            scheduler_factor=tcfg["scheduler_factor"],
            device=resolve_device(cfg), amp=resolve_amp(cfg),
        )
        history = trainer.fit(
            make_loader(X_fit, y_fit, hp["batch_size"], True, seed),
            make_loader(X_val, y_val, hp["batch_size"], False, seed),
            epochs=tcfg["final_epochs"],
            patience=tcfg["early_stopping_patience"],
            verbose=verbose,
        )
        return trainer, history

    history = model.fit(
        X_fit, y_fit, X_val, y_val,
        epochs=tcfg["final_epochs"],
        patience=tcfg["early_stopping_patience"],
        batch_size=hp["batch_size"],
        virtual_batch_size=cfg["models"]["tabnet"]["virtual_batch_size"],
        class_weights=weights,
    )
    return model, history


def predict(estimator, model_name: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    proba = estimator.predict_proba(X)
    return proba.argmax(axis=1), proba


# =====================================================================
# Interpretabilidad (III-G)
# =====================================================================
def interpretability(
    estimator, model_name: str, X: np.ndarray, feature_names: List[str],
    max_samples: int = 2048,
) -> Dict[str, object]:
    Xs = X[:max_samples]
    out: Dict[str, object] = {"model": model_name, "n_samples": int(len(Xs))}

    try:
        if model_name == "tabnet":
            imp = estimator.feature_importance()
            out["step_masks"] = {
                k: dict(sorted(
                    zip(feature_names, np.asarray(v).tolist()),
                    key=lambda kv: kv[1], reverse=True)[:15])
                for k, v in estimator.step_masks(Xs).items()
            }
        else:
            xb = torch.as_tensor(Xs, dtype=torch.float32).to(estimator.device)
            net = estimator.model
            imp = net.feature_importance(xb)
            if model_name == "ft_transformer":
                out["attention_shape"] = list(np.asarray(net.attention_maps(xb[:256])).shape)

        imp = np.asarray(imp, dtype=float)
        if imp.sum() > 0:
            imp = imp / imp.sum()
        ranking = sorted(zip(feature_names, imp.tolist()),
                         key=lambda kv: kv[1], reverse=True)
        out["feature_importance"] = dict(ranking)
        out["top_15"] = dict(ranking[:15])
        out["por_dimension"] = (
            pd.DataFrame({"f": feature_names, "w": imp})
            .assign(dim=lambda d: d.f.map(V.dimension_of))
            .groupby("dim")["w"].sum().sort_values(ascending=False).to_dict()
        )
    except Exception as exc:                              # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# =====================================================================
# Una combinacion (grupo, modelo)
# =====================================================================
def run_one(
    cfg: dict, group: str, model_name: str, root: Path,
    do_gridsearch: Optional[bool] = None, verbose: bool = True,
) -> Dict[str, object]:
    from .gridsearch import grid_search_cv

    started = time.time()
    seed = cfg["seed"]
    out_dir = root / cfg["paths"]["results_dir"] / f"{group}_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data, feature_names = load_group(cfg, group, root)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = data["train"], data["val"], data["test"]

    if verbose:
        print(f"\n{'=' * 66}\n{group.upper()} · {model_name}\n{'=' * 66}")
        print(f"  train {X_tr.shape} · val {X_val.shape} · test {X_te.shape}")

    # --- 1. Grid Search + 5-Fold CV --------------------------------
    use_gs = cfg["gridsearch"]["enabled"] if do_gridsearch is None else do_gridsearch
    if use_gs:
        gs = grid_search_cv(cfg, model_name, X_tr, y_tr, out_dir, verbose)
        best_hp = gs["best_params"]
    else:
        preset = cfg["gridsearch"]["presets"]["fast"]
        best_hp = {"lr": preset["lr"][0], "batch_size": preset["batch_size"][0],
                   "dropout": preset["dropout"][0],
                   "weight_decay": preset["weight_decay"][0],
                   "depth": preset["depth"][0]}
        if model_name == "tabnet":
            best_hp["n_steps"] = cfg["models"]["tabnet"]["n_steps"]
        gs = {"skipped": True, "best_params": best_hp}

    # --- 2. Reentrenamiento sobre el 85% ----------------------------
    X_fit = np.vstack([X_tr, X_val])
    y_fit = np.concatenate([y_tr, y_val])
    if verbose:
        print(f"  Reentrenando sobre 85% ({len(X_fit):,} muestras) con {best_hp}")

    estimator, history = train_final(
        cfg, model_name, best_hp, X_fit, y_fit, X_val, y_val, seed, verbose)

    # --- 3. Evaluacion sobre el 15% de test -------------------------
    y_pred, y_proba = predict(estimator, model_name, X_te)
    metrics = compute_metrics(y_te, y_pred, y_proba,
                              cfg["n_classes"], cfg["class_names"])

    if verbose:
        auc = metrics["roc_auc_ovr"]
        print(f"  TEST · acc={metrics['accuracy']:.4f} "
              f"f1w={metrics['f1_weighted']:.4f} f1m={metrics['f1_macro']:.4f} "
              f"auc={auc:.4f}" if auc is not None else
              f"  TEST · acc={metrics['accuracy']:.4f} (AUC no disponible)")

    # --- 4. Interpretabilidad ---------------------------------------
    interp = (interpretability(estimator, model_name, X_te, feature_names)
              if cfg["evaluation"]["interpretability"] else {})

    # --- 5. Estabilidad multi-semilla -------------------------------
    stability: Dict[str, object] = {}
    if cfg["evaluation"]["stability"]["multiseed"]:
        runs = []
        seeds = cfg["evaluation"]["stability"]["seeds"]
        if verbose:
            print(f"  Estabilidad multi-semilla: {seeds}")
        for s in seeds:
            est_s, _ = train_final(cfg, model_name, best_hp,
                                   X_fit, y_fit, X_val, y_val, s, verbose=False)
            yp, pp = predict(est_s, model_name, X_te)
            m = compute_metrics(y_te, yp, pp, cfg["n_classes"], cfg["class_names"])
            runs.append({k: m[k] for k in
                         ("accuracy", "f1_weighted", "f1_macro", "roc_auc_ovr")})
            if verbose:
                print(f"    seed {s}: acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f}")
        stability = {
            "seeds": seeds, "runs": runs,
            **aggregate_seed_runs(
                runs, ["accuracy", "f1_weighted", "f1_macro", "roc_auc_ovr"]),
        }

    # --- Persistencia -----------------------------------------------
    payload = {
        "grupo": group, "modelo": model_name, "seed": seed,
        "n_features": len(feature_names),
        "features": feature_names,
        "gridsearch": gs,
        "hiperparametros_finales": best_hp,
        "protocolo": {
            "cv": f"{cfg['gridsearch']['cv_folds']}-Fold estratificado sobre el 70%",
            "refit": "85% (train + val)",
            "test": "15% nunca visto",
            "epocas": cfg["training"]["final_epochs"],
            "early_stopping_patience": cfg["training"]["early_stopping_patience"],
            "optimizador": cfg["training"]["optimizer"],
        },
        "test_metrics": metrics,
        "roc_curves": roc_curves(y_te, y_proba, cfg["n_classes"]),
        "learning_curves": history,
        "interpretabilidad": interp,
        "estabilidad": stability,
        "elapsed_sec": round(time.time() - started, 2),
    }

    with open(out_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    np.savez_compressed(out_dir / "test_predictions.npz",
                        y_true=y_te, y_pred=y_pred, y_proba=y_proba)

    if IS_TORCH[model_name]:
        estimator.save(out_dir / "best_model.pt")
    else:
        estimator.save(out_dir / "tabnet_model")

    if verbose:
        print(f"  artefactos -> {out_dir}  ({payload['elapsed_sec'] / 60:.1f} min)")
    return payload


def run_all(
    cfg: dict, root: Path, groups: Optional[List[str]] = None,
    models: Optional[List[str]] = None, do_gridsearch: Optional[bool] = None,
) -> Dict[str, dict]:
    """Las nueve combinaciones grupo x modelo."""
    groups = groups or list(cfg["groups"])
    models = models or MODEL_NAMES
    results: Dict[str, dict] = {}

    for group in groups:
        for model_name in models:
            key = f"{group}_{model_name}"
            try:
                results[key] = run_one(cfg, group, model_name, root, do_gridsearch)
            except Exception as exc:                      # noqa: BLE001
                import traceback
                print(f"  ERROR en {key}: {type(exc).__name__}: {exc}")
                results[key] = {"error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc()}

    summary_dir = root / cfg["paths"]["results_dir"]
    summary_dir.mkdir(parents=True, exist_ok=True)
    ok = {k: v for k, v in results.items() if "error" not in v}
    with open(summary_dir / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"total": len(results), "exitosos": len(ok),
                   "fallidos": len(results) - len(ok),
                   "combinaciones": {k: ("ok" if "error" not in v else v["error"])
                                     for k, v in results.items()}},
                  fh, indent=2, ensure_ascii=False)
    return results
