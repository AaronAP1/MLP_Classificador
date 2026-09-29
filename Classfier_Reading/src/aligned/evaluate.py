"""
Metricas y ranking comparativo (secciones III-E-5, III-F-1 y III-F-3).

Dos correcciones respecto al codigo auditado:

  * El AUC ya no se calcula dentro de un `except: return None` mudo. Si
    falla, se registra el motivo, de modo que un AUC ausente nunca puede
    confundirse con un AUC no solicitado. La Tabla 4 del articulo existia
    sin ningun artefacto de AUC que la respaldara.

  * Copeland implementa la regla que describe el articulo (III-F-3):
    "+1 punto si supera a B en al menos dos metricas", es decir un unico
    punto por par comparado. La version anterior sumaba +/-1 por cada
    metrica, lo que produce una escala distinta de la reportada.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, roc_curve, classification_report,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true, y_pred, y_proba: Optional[np.ndarray] = None,
    n_classes: int = 3, class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Conjunto completo declarado en la seccion III-E-5."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    pw, rw, f1w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    pm, rm, f1m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)

    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1w),
        "f1_macro": float(f1m),
        "precision_weighted": float(pw),
        "precision_macro": float(pm),
        "recall_weighted": float(rw),
        "recall_macro": float(rm),
    }

    # AUC One-vs-Rest (III-F-1-e), con motivo explicito si no se puede.
    #   roc_auc_ovr        -> promedio MACRO (media simple de las 3 clases);
    #                         es el que va a la Tabla 4.
    #   roc_auc_ovr_weighted -> promedio ponderado por soporte, de referencia.
    metrics["roc_auc_ovr"] = None
    metrics["roc_auc_ovr_weighted"] = None
    metrics["roc_auc_ovr_error"] = None
    metrics["roc_auc_per_class"] = None
    if y_proba is None:
        metrics["roc_auc_ovr_error"] = "no se entregaron probabilidades"
    elif len(np.unique(y_true)) < 2:
        metrics["roc_auc_ovr_error"] = "el conjunto tiene una sola clase presente"
    else:
        try:
            metrics["roc_auc_ovr"] = float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro",
                labels=list(range(n_classes))))
            metrics["roc_auc_ovr_weighted"] = float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted",
                labels=list(range(n_classes))))
            per_class = {}
            yb = label_binarize(y_true, classes=list(range(n_classes)))
            for c in range(n_classes):
                name = class_names[c] if class_names else str(c)
                if yb[:, c].sum() in (0, len(yb)):
                    per_class[name] = None
                else:
                    per_class[name] = float(roc_auc_score(yb[:, c], y_proba[:, c]))
            metrics["roc_auc_per_class"] = per_class
        except Exception as exc:                          # noqa: BLE001
            metrics["roc_auc_ovr_error"] = f"{type(exc).__name__}: {exc}"

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    metrics["confusion_matrix"] = cm.tolist()
    with np.errstate(divide="ignore", invalid="ignore"):
        cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    metrics["confusion_matrix_normalized"] = np.nan_to_num(cmn).tolist()
    metrics["classification_report"] = classification_report(
        y_true, y_pred, labels=list(range(n_classes)),
        target_names=list(class_names) if class_names else None,
        zero_division=0, output_dict=True,
    )
    metrics["support_per_class"] = np.bincount(
        y_true, minlength=n_classes).tolist()
    return metrics


def roc_curves(y_true, y_proba, n_classes: int = 3) -> Dict[str, dict]:
    """Puntos de la curva ROC one-vs-rest, para la figura de resultados."""
    y_true = np.asarray(y_true).astype(int)
    yb = label_binarize(y_true, classes=list(range(n_classes)))
    curves = {}
    for c in range(n_classes):
        if yb[:, c].sum() in (0, len(yb)):
            continue
        fpr, tpr, _ = roc_curve(yb[:, c], y_proba[:, c])
        curves[str(c)] = {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "auc": float(roc_auc_score(yb[:, c], y_proba[:, c])),
        }
    return curves


# =====================================================================
# Copeland (III-F-3)
# =====================================================================
def copeland_ranking(
    results: pd.DataFrame, metrics: Sequence[str], tol: float = 1e-9
) -> pd.DataFrame:
    """
    Regla del articulo, por par de modelos:
        +1  si A supera a B en al menos dos metricas
         0  si empatan
        -1  si B supera a A

    Args:
        results: DataFrame indexado por nombre de modelo, con una columna
                 por metrica.
    Returns:
        DataFrame con copeland_score, wins, losses, ties y rank.
    """
    models = list(results.index)
    missing = set(metrics) - set(results.columns)
    if missing:
        raise ValueError(f"Faltan metricas en el DataFrame: {sorted(missing)}")

    score = {m: 0 for m in models}
    wins = {m: 0 for m in models}
    losses = {m: 0 for m in models}
    ties = {m: 0 for m in models}
    majority = len(metrics) // 2 + 1          # "al menos dos" de tres

    for a, b in itertools.combinations(models, 2):
        a_better = sum(
            1 for k in metrics
            if float(results.loc[a, k]) > float(results.loc[b, k]) + tol
        )
        b_better = sum(
            1 for k in metrics
            if float(results.loc[b, k]) > float(results.loc[a, k]) + tol
        )
        if a_better >= majority and a_better > b_better:
            score[a] += 1; score[b] -= 1; wins[a] += 1; losses[b] += 1
        elif b_better >= majority and b_better > a_better:
            score[b] += 1; score[a] -= 1; wins[b] += 1; losses[a] += 1
        else:
            ties[a] += 1; ties[b] += 1

    out = pd.DataFrame({
        "classifier": models,
        "copeland_score": [score[m] for m in models],
        "wins": [wins[m] for m in models],
        "losses": [losses[m] for m in models],
        "ties": [ties[m] for m in models],
    })
    for k in metrics:
        out[k] = [float(results.loc[m, k]) for m in models]

    out = out.sort_values("copeland_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def copeland_across_groups(
    per_group: Dict[str, pd.DataFrame], metrics: Sequence[str]
) -> pd.DataFrame:
    """
    Ranking global: se acumula el puntaje Copeland de cada grupo etario.
    Con 3 modelos y 3 grupos el rango teorico es [-6, +6].
    """
    total: Dict[str, int] = {}
    detail = {}
    for group, frame in per_group.items():
        r = copeland_ranking(frame, metrics)
        detail[group] = r.set_index("classifier")["copeland_score"].to_dict()
        for m, s in detail[group].items():
            total[m] = total.get(m, 0) + int(s)

    out = pd.DataFrame({
        "classifier": list(total),
        "copeland_score_global": [total[m] for m in total],
    })
    for group in per_group:
        out[f"score_{group}"] = [detail[group].get(m, 0) for m in out["classifier"]]
    out = out.sort_values("copeland_score_global", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def aggregate_seed_runs(runs: List[Dict[str, float]], metrics: Sequence[str]) -> Dict[str, float]:
    """Media y desviacion estandar multi-semilla (III-F-4)."""
    out: Dict[str, float] = {}
    for k in metrics:
        vals = [float(r[k]) for r in runs if r.get(k) is not None]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals, ddof=0))
    out["n_runs"] = len(runs)
    return out
