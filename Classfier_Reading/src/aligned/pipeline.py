"""
Orquestador de entrenamiento y evaluacion (secciones III-D, III-E, III-G).

Para cada par (grupo, modelo) ejecuta:

    1. Grid Search + 5-Fold CV estratificado sobre el 70% (train).
    2. Etapa A · train (70%) vs. validacion (15%): hasta 200 epocas con
       early stopping de paciencia 20. De aqui salen las curvas de
       aprendizaje (Fig. 11) y la epoca optima.
    3. Etapa B · reentrenamiento sobre el 85% (train + val) durante ese
       numero de epocas, sin early stopping.
    4. Evaluacion unica sobre el 15% de test, jamas visto en 1, 2 ni 3.
    5. Interpretabilidad: mascaras de TabNet, atencion de FT-Transformer
       y atencion multi-cabeza del MLP+MA.
    6. Estabilidad multi-semilla: la etapa B repetida con 5 semillas.

Correccion respecto a la version anterior: el reentrenamiento sobre el
85% usaba como "validacion" para el early stopping un subconjunto de sus
propios datos de entrenamiento. Esa senal no detiene nada y la curva de
"validacion" resultante no era una validacion. Ahora la validacion es
siempre un conjunto que el modelo no ve al ajustarse.

Todo artefacto que el articulo reporta debe salir de aqui. No existe
ninguna ruta que fabrique metricas ni predicciones sinteticas.
"""

from __future__ import annotations

import json
import platform
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from . import variables as V
from .balancing import balance_training_set
from .evaluate import compute_metrics, roc_curves, aggregate_seed_runs
from .models import IS_TORCH, MODEL_NAMES, build_model
from .trainer import TorchTrainer, make_loader

# pytorch-tabnet avisa en cada ajuste que restaura la mejor epoca; es lo
# que se le pide, no un problema.
warnings.filterwarnings("ignore", message="Best weights from best epoch")


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

    # Prueba de humo: submuestra estratificada para que todo corra en
    # minutos. Nunca se activa en el entrenamiento real.
    limit = (cfg.get("smoke") or {}).get("max_rows_per_split")
    if limit:
        out = {name: _stratified_head(X, y, limit, cfg["seed"])
               for name, (X, y) in out.items()}
    return out, feature_cols


def _stratified_head(X: np.ndarray, y: np.ndarray, n: int, seed: int):
    if len(y) <= n:
        return X, y
    from sklearn.model_selection import train_test_split
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, random_state=seed, stratify=y)
    return Xs, ys


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


def hardware_info(cfg: dict) -> Dict[str, object]:
    """Equipo en el que se entreno, para declararlo en el articulo."""
    device = resolve_device(cfg)
    info: Dict[str, object] = {
        "dispositivo": device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "procesador": platform.processor() or platform.machine(),
    }
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memoria_gb"] = round(props.total_memory / 1024 ** 3, 1)
        info["cuda"] = torch.version.cuda
    return info


# =====================================================================
# Entrenamiento
# =====================================================================
def fit_estimator(
    cfg: dict, model_name: str, hp: dict,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
    epochs: int, patience: int, seed: int,
    verbose: bool = True, track_train: bool = False,
) -> Tuple[object, Dict[str, list], int, Dict[str, object]]:
    """
    Entrena una arquitectura con la configuracion `hp`.

    Con `X_val`: early stopping sobre `X_val` y pesos de la mejor epoca.
    Con `X_val=None`: exactamente `epochs` epocas (reentrenamiento 85%).

    Returns:
        (estimador, historia, mejor_epoca_base0, informe_de_balanceo)
    """
    X_bal, y_bal, weights, bal_info = balance_training_set(X_tr, y_tr, cfg, seed)
    set_seed(seed)
    model = build_model(model_name, X_bal.shape[1], cfg, hp, seed)
    tcfg = cfg["training"]

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
            make_loader(X_bal, y_bal, hp["batch_size"], True, seed,
                        num_workers=tcfg.get("num_workers", 0)),
            (make_loader(X_val, y_val, hp["batch_size"], False, seed)
             if X_val is not None else None),
            epochs=epochs, patience=patience, verbose=verbose,
        )
        return trainer, history, int(trainer.best_epoch), bal_info

    history = model.fit(
        X_bal, y_bal, X_val, y_val,
        epochs=epochs, patience=patience,
        batch_size=hp["batch_size"],
        virtual_batch_size=cfg["models"]["tabnet"]["virtual_batch_size"],
        class_weights=weights, track_train=track_train,
    )
    return model, history, int(model.best_epoch), bal_info


def predict(estimator, model_name: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    proba = estimator.predict_proba(X)
    return proba.argmax(axis=1), proba


# =====================================================================
# Interpretabilidad (III-G)
# =====================================================================
def permutation_importance(estimator, X: np.ndarray, y: np.ndarray,
                           n_repeats: int = 3, seed: int = 42) -> Tuple[float, np.ndarray]:
    """
    Caida del F1-Macro en test al permutar cada variable. Es independiente
    de la arquitectura, asi que permite comparar los tres modelos con la
    misma vara, y no depende de que la atencion se haya especializado.
    """
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    base = f1_score(y, estimator.predict_proba(X).argmax(1), average="macro", zero_division=0)
    drops = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        vals = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            pred = estimator.predict_proba(Xp).argmax(1)
            vals.append(base - f1_score(y, pred, average="macro", zero_division=0))
        drops[j] = float(np.mean(vals))
    return float(base), drops


def interpretability(
    estimator, model_name: str, X: np.ndarray, feature_names: List[str],
    max_samples: int = 2048, out_npz: Optional[Path] = None,
    y: Optional[np.ndarray] = None, seed: int = 42,
) -> Dict[str, object]:
    """
    Importancia por variable sobre el conjunto de prueba.

    1) Propia de cada arquitectura (lo que describe la seccion III-G):
        MLP+MA          atencion recibida por cada variable
        TabNet          mascaras agregadas de los decision steps
        FT-Transformer  atencion recibida, promediada en cabezas y capas
    2) Por permutacion (caida de F1-Macro), comun a las tres, si se pasa `y`.

    Los mapas completos (mascaras por paso, matrices de atencion) se
    guardan en `out_npz` para los heatmaps del reporte.
    """
    Xs = X[:max_samples]
    out: Dict[str, object] = {"model": model_name, "n_samples": int(len(Xs)),
                              "conjunto": "prueba (15%)"}
    arrays: Dict[str, np.ndarray] = {}

    if y is not None:
        try:
            base, drops = permutation_importance(estimator, Xs, np.asarray(y)[:max_samples],
                                                 seed=seed)
            ranking = sorted(zip(feature_names, drops.tolist()),
                             key=lambda kv: kv[1], reverse=True)
            out["permutacion"] = {
                "metrica": "caida de F1-Macro en test al permutar la variable",
                "f1_macro_base": base,
                "importancia": dict(ranking),
                "top_15": dict(ranking[:15]),
            }
            arrays["permutation"] = drops
        except Exception as exc:                          # noqa: BLE001
            out["permutacion"] = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        if model_name == "tabnet":
            M_explain, masks = estimator.model.explain(np.asarray(Xs, dtype=np.float32))
            imp = np.asarray(M_explain).mean(axis=0)
            steps = np.stack([np.asarray(masks[k]).mean(axis=0) for k in sorted(masks)])
            arrays["tabnet_step_masks"] = steps                     # (pasos, F)
            out["step_masks"] = {
                f"step_{i + 1}": dict(sorted(zip(feature_names, steps[i].tolist()),
                                             key=lambda kv: kv[1], reverse=True)[:15])
                for i in range(len(steps))
            }
        else:
            xb = torch.as_tensor(Xs, dtype=torch.float32).to(estimator.device)
            net = estimator.model
            imp = net.feature_importance(xb)
            maps = np.asarray(net.attention_maps(xb[:512]))
            if model_name == "ft_transformer":
                arrays["ft_attention"] = maps.mean(axis=(0, 1))        # (F+1, F+1)
            else:
                arrays["mlp_attention"] = maps.mean(axis=0)            # (F, F)
            out["attention_shape"] = list(maps.shape)

        imp = np.asarray(imp, dtype=float)
        if imp.sum() > 0:
            imp = imp / imp.sum()
        arrays["importance"] = imp
        ranking = sorted(zip(feature_names, imp.tolist()),
                         key=lambda kv: kv[1], reverse=True)
        out["feature_importance"] = dict(ranking)
        out["top_15"] = dict(ranking[:15])
        out["por_dimension"] = (
            pd.DataFrame({"f": feature_names, "w": imp})
            .assign(dim=lambda d: d.f.map(V.dimension_of))
            .groupby("dim")["w"].sum().sort_values(ascending=False).to_dict()
        )
        if out_npz is not None:
            np.savez_compressed(out_npz, feature_names=np.array(feature_names), **arrays)
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
    tcfg = cfg["training"]
    out_dir = root / cfg["paths"]["results_dir"] / f"{group}_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data, feature_names = load_group(cfg, group, root)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = data["train"], data["val"], data["test"]

    if verbose:
        print(f"\n{'=' * 66}\n{group.upper()} · {model_name}\n{'=' * 66}")
        print(f"  train {X_tr.shape} · val {X_val.shape} · test {X_te.shape}")

    # --- 1. Grid Search + 5-Fold CV sobre el 70% ---------------------
    use_gs = cfg["gridsearch"]["enabled"] if do_gridsearch is None else do_gridsearch
    if use_gs:
        gs = grid_search_cv(cfg, model_name, X_tr, y_tr, out_dir, verbose)
        best_hp = gs["best_params"]
    else:
        preset = cfg["gridsearch"]["presets"]["fast"]
        lr0 = (preset.get("lr_tabnet", preset["lr"]) if model_name == "tabnet"
               else preset["lr"])[0]
        best_hp = {"lr": lr0, "batch_size": preset["batch_size"][0],
                   "dropout": preset["dropout"][0],
                   "weight_decay": preset["weight_decay"][0],
                   "depth": preset["depth"][0]}
        if model_name == "tabnet":
            best_hp["n_steps"] = cfg["models"]["tabnet"]["n_steps"]
        gs = {"skipped": True, "best_params": best_hp}

    # --- 2. Etapa A: train (70%) vs. validacion (15%) ----------------
    if verbose:
        print(f"  Etapa A · train {len(X_tr):,} vs. validacion {len(X_val):,} "
              f"(max {tcfg['final_epochs']} epocas, paciencia {tcfg['early_stopping_patience']})")
    est_a, curves, best_epoch, bal_info = fit_estimator(
        cfg, model_name, best_hp, X_tr, y_tr, X_val, y_val,
        epochs=tcfg["final_epochs"], patience=tcfg["early_stopping_patience"],
        seed=seed, verbose=verbose, track_train=True,
    )
    n_refit = best_epoch + 1
    yp_a, pp_a = predict(est_a, model_name, X_te)
    diag_a = compute_metrics(y_te, yp_a, pp_a, cfg["n_classes"], cfg["class_names"])

    # --- 3. Etapa B: reentrenamiento sobre el 85% ---------------------
    X_fit = np.vstack([X_tr, X_val])
    y_fit = np.concatenate([y_tr, y_val])
    if verbose:
        print(f"  Etapa B · reentrenando sobre 85% ({len(X_fit):,} muestras) "
              f"durante {n_refit} epocas (mejor epoca de la etapa A)")
    estimator, hist_b, _, _ = fit_estimator(
        cfg, model_name, best_hp, X_fit, y_fit, None, None,
        epochs=n_refit, patience=0, seed=seed, verbose=False,
    )

    # --- 4. Evaluacion sobre el 15% de test --------------------------
    y_pred, y_proba = predict(estimator, model_name, X_te)
    metrics = compute_metrics(y_te, y_pred, y_proba,
                              cfg["n_classes"], cfg["class_names"])

    if verbose:
        auc = metrics["roc_auc_ovr"]
        print(f"  TEST · acc={metrics['accuracy']:.4f} "
              f"f1w={metrics['f1_weighted']:.4f} f1m={metrics['f1_macro']:.4f} "
              + (f"auc={auc:.4f}" if auc is not None else "(AUC no disponible)"))

    # --- 5. Interpretabilidad -----------------------------------------
    interp = (interpretability(estimator, model_name, X_te, feature_names,
                               max_samples=cfg["evaluation"].get("interpretability_samples", 2048),
                               out_npz=out_dir / "interpretabilidad.npz",
                               y=y_te, seed=seed)
              if cfg["evaluation"]["interpretability"] else {})

    # --- 6. Estabilidad multi-semilla (etapa B con otras semillas) -----
    stability: Dict[str, object] = {}
    if cfg["evaluation"]["stability"]["multiseed"]:
        runs = []
        seeds = cfg["evaluation"]["stability"]["seeds"]
        if verbose:
            print(f"  Estabilidad multi-semilla: {seeds}")
        for s in seeds:
            est_s, _, _, _ = fit_estimator(
                cfg, model_name, best_hp, X_fit, y_fit, None, None,
                epochs=n_refit, patience=0, seed=s, verbose=False)
            yp, pp = predict(est_s, model_name, X_te)
            m = compute_metrics(y_te, yp, pp, cfg["n_classes"], cfg["class_names"])
            runs.append({"seed": s, **{k: m[k] for k in
                         ("accuracy", "f1_weighted", "f1_macro",
                          "precision_weighted", "recall_weighted", "roc_auc_ovr")}})
            if verbose:
                print(f"    seed {s}: acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f}")
        stability = {
            "seeds": seeds, "runs": runs,
            "protocolo": f"etapa B ({n_refit} epocas sobre el 85%) con cada semilla",
            **aggregate_seed_runs(
                runs, ["accuracy", "f1_weighted", "f1_macro",
                       "precision_weighted", "recall_weighted", "roc_auc_ovr"]),
        }

    # --- Persistencia ---------------------------------------------------
    payload = {
        "grupo": group, "modelo": model_name, "seed": seed,
        "prueba_de_humo": bool(cfg.get("smoke")),
        "hardware": hardware_info(cfg),
        "n_features": len(feature_names),
        "features": feature_names,
        "n_muestras": {"train": int(len(X_tr)), "val": int(len(X_val)),
                       "test": int(len(X_te)), "refit_85": int(len(X_fit))},
        "gridsearch": gs,
        "hiperparametros_finales": best_hp,
        "balanceo": bal_info,
        "protocolo": {
            "cv": f"{cfg['gridsearch']['cv_folds']}-Fold estratificado sobre el 70%",
            "etapa_a": (f"train 70% vs. validacion 15%, max {tcfg['final_epochs']} epocas, "
                        f"early stopping paciencia {tcfg['early_stopping_patience']}"),
            "etapa_b": f"reentrenamiento sobre el 85% durante {n_refit} epocas",
            "test": "15% nunca visto",
            "optimizador": tcfg["optimizer"],
        },
        "mejor_epoca_etapa_a": int(n_refit),
        "epocas_ejecutadas_etapa_a": int(len(curves.get("val_loss") or curves.get("train_loss") or [])),
        "test_metrics": metrics,
        "roc_curves": roc_curves(y_te, y_proba, cfg["n_classes"]),
        "learning_curves": curves,
        "diagnostico": {
            "test_modelo_etapa_a": {k: diag_a[k] for k in
                                    ("accuracy", "f1_weighted", "f1_macro", "roc_auc_ovr")},
            "nota": ("Metricas en test del modelo de la etapa A (entrenado solo con el "
                     "70%). Deben ser cercanas a las del modelo final; una diferencia "
                     "grande indica un problema en el reentrenamiento."),
        },
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
                   "hardware": hardware_info(cfg),
                   "combinaciones": {k: ("ok" if "error" not in v else v["error"])
                                     for k, v in results.items()},
                   "trazas": {k: v["traceback"] for k, v in results.items()
                              if "traceback" in v}},
                  fh, indent=2, ensure_ascii=False)
    return results
