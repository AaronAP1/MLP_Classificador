"""
Tablas y figuras del articulo, generadas exclusivamente desde artefactos
reales de entrenamiento.

Regla central: si falta el `results.json` de una combinacion, la funcion
lo reporta como faltante y lo deja fuera. NUNCA rellena con datos
simulados. El generador anterior producia curvas ROC y matrices de
confusion con `np.random` cuando no encontraba resultados, que es como
una figura ausente se convierte en una figura falsa.

Produce:
    tabla3_metricas.{csv,tex}    Tabla 3  metricas por grupo y modelo
    tabla4_auc.{csv,tex}         Tabla 4  AUC por grupo y modelo
    copeland_global.csv          Ranking Copeland (III-F-3)
    estabilidad_multisemilla.csv Media y desviacion por semilla (III-F-4)
    interpretabilidad_top.csv    Variables mas influyentes (III-G)
    fig_matrices_confusion.png   Fig. 8
    fig_curvas_roc.png           Fig. 9
    fig_ranking_copeland.png     Fig. 10
    fig_curvas_aprendizaje.png   Fig. 11
    fig_importancia_variables.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .evaluate import copeland_across_groups, copeland_ranking
from .models import MODEL_NAMES

DISPLAY = {"mlp_ma": "MLP-MA", "tabnet": "TabNet", "ft_transformer": "FT-Transformer"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    })
    return plt


def load_results(cfg: dict, root: Path) -> Tuple[Dict[str, dict], List[str]]:
    """Carga los results.json disponibles y lista los que faltan."""
    base = root / cfg["paths"]["results_dir"]
    found, missing = {}, []
    for group in cfg["groups"]:
        for model in MODEL_NAMES:
            key = f"{group}_{model}"
            path = base / key / "results.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as fh:
                    found[key] = json.load(fh)
            else:
                missing.append(key)
    return found, missing


# =====================================================================
# Tablas
# =====================================================================
def tabla3(results: Dict[str, dict], cfg: dict) -> pd.DataFrame:
    rows = []
    for group in cfg["groups"]:
        for model in MODEL_NAMES:
            r = results.get(f"{group}_{model}")
            if not r:
                continue
            m = r["test_metrics"]
            rows.append({
                "Grupo": group.capitalize(), "Modelo": DISPLAY[model],
                "Accuracy": m["accuracy"], "F1-Weighted": m["f1_weighted"],
                "F1-Macro": m["f1_macro"],
                "Precision-Weighted": m["precision_weighted"],
                "Recall-Weighted": m["recall_weighted"],
                "n_test": int(sum(m["support_per_class"])),
            })
    return pd.DataFrame(rows)


def tabla4(results: Dict[str, dict], cfg: dict) -> pd.DataFrame:
    rows = []
    for group in cfg["groups"]:
        row = {"Grupo": group.capitalize()}
        for model in MODEL_NAMES:
            r = results.get(f"{group}_{model}")
            auc = r["test_metrics"]["roc_auc_ovr"] if r else None
            row[DISPLAY[model]] = auc
        rows.append(row)
    df = pd.DataFrame(rows)
    numeric = [DISPLAY[m] for m in MODEL_NAMES]
    if not df.empty:
        promedio = {"Grupo": "Promedio"}
        for c in numeric:
            vals = df[c].dropna()
            promedio[c] = float(vals.mean()) if len(vals) else None
        df = pd.concat([df, pd.DataFrame([promedio])], ignore_index=True)
    return df


def tabla_estabilidad(results: Dict[str, dict], cfg: dict) -> pd.DataFrame:
    rows = []
    for r in results.values():
        est = r.get("estabilidad") or {}
        if not est.get("runs"):
            continue
        rows.append({
            "Grupo": r["grupo"].capitalize(), "Modelo": DISPLAY[r["modelo"]],
            "n_semillas": est.get("n_runs"),
            "Accuracy_mean": est.get("accuracy_mean"),
            "Accuracy_std": est.get("accuracy_std"),
            "F1-Macro_mean": est.get("f1_macro_mean"),
            "F1-Macro_std": est.get("f1_macro_std"),
        })
    return pd.DataFrame(rows)


def tabla_interpretabilidad(results: Dict[str, dict], top_n: int = 15) -> pd.DataFrame:
    rows = []
    for r in results.values():
        interp = r.get("interpretabilidad") or {}
        for feat, weight in list((interp.get("top_15") or {}).items())[:top_n]:
            rows.append({
                "Grupo": r["grupo"].capitalize(), "Modelo": DISPLAY[r["modelo"]],
                "Variable": feat, "Importancia": weight,
            })
    return pd.DataFrame(rows)


def ranking_copeland(results: Dict[str, dict], cfg: dict):
    metrics = cfg["evaluation"]["copeland_metrics"]
    per_group, tables = {}, {}
    for group in cfg["groups"]:
        rows = {}
        for model in MODEL_NAMES:
            r = results.get(f"{group}_{model}")
            if r:
                rows[DISPLAY[model]] = {k: r["test_metrics"][k] for k in metrics}
        if len(rows) >= 2:
            frame = pd.DataFrame(rows).T
            per_group[group] = frame
            tables[group] = copeland_ranking(frame, metrics)
    global_rank = copeland_across_groups(per_group, metrics) if per_group else pd.DataFrame()
    return tables, global_rank


# =====================================================================
# Figuras
# =====================================================================
def fig_confusion(results, cfg, out: Path) -> Optional[Path]:
    plt = _mpl()
    groups, names = list(cfg["groups"]), cfg["class_names"]
    fig, axes = plt.subplots(len(groups), len(MODEL_NAMES),
                             figsize=(4.2 * len(MODEL_NAMES), 3.6 * len(groups)),
                             squeeze=False)
    drew = False
    for i, group in enumerate(groups):
        for j, model in enumerate(MODEL_NAMES):
            ax = axes[i][j]
            r = results.get(f"{group}_{model}")
            if not r:
                ax.text(.5, .5, "sin resultados", ha="center", va="center",
                        transform=ax.transAxes, color="#999")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{group.capitalize()} · {DISPLAY[model]}")
                continue
            cm = np.array(r["test_metrics"]["confusion_matrix_normalized"])
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            for a in range(cm.shape[0]):
                for b in range(cm.shape[1]):
                    ax.text(b, a, f"{cm[a, b]:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if cm[a, b] > .5 else "black")
            ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
            ax.set_xticklabels([n[:12] for n in names], rotation=30, ha="right", fontsize=7)
            ax.set_yticklabels([n[:12] for n in names], fontsize=7)
            ax.set_title(f"{group.capitalize()} · {DISPLAY[model]}")
            ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
            fig.colorbar(im, ax=ax, fraction=.046)
            drew = True
    if not drew:
        plt.close(fig); return None
    fig.suptitle("Matrices de confusion normalizadas (conjunto de prueba)", fontweight="bold")
    fig.tight_layout()
    path = out / "fig_matrices_confusion.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def fig_roc(results, cfg, out: Path) -> Optional[Path]:
    plt = _mpl()
    groups, names = list(cfg["groups"]), cfg["class_names"]
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 4.4), squeeze=False)
    drew = False
    for i, group in enumerate(groups):
        ax = axes[0][i]
        for model in MODEL_NAMES:
            r = results.get(f"{group}_{model}")
            if not r or not r.get("roc_curves"):
                continue
            for cls, curve in r["roc_curves"].items():
                ax.plot(curve["fpr"], curve["tpr"], lw=1.2,
                        label=f"{DISPLAY[model]} · {names[int(cls)][:10]} "
                              f"(AUC={curve['auc']:.3f})")
                drew = True
        ax.plot([0, 1], [0, 1], "k--", lw=.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_title(f"{group.capitalize()} "
                     f"({cfg['groups'][group]['edad_min']}-{cfg['groups'][group]['edad_max']} años)")
        ax.legend(fontsize=6, loc="lower right"); ax.grid(alpha=.3)
    if not drew:
        plt.close(fig); return None
    fig.suptitle("Curvas ROC (one-vs-rest) por grupo etario", fontweight="bold")
    fig.tight_layout()
    path = out / "fig_curvas_roc.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def fig_learning(results, cfg, out: Path) -> Optional[Path]:
    plt = _mpl()
    groups = list(cfg["groups"])
    fig, axes = plt.subplots(len(groups), len(MODEL_NAMES),
                             figsize=(4.2 * len(MODEL_NAMES), 3.1 * len(groups)),
                             squeeze=False)
    drew = False
    for i, group in enumerate(groups):
        for j, model in enumerate(MODEL_NAMES):
            ax = axes[i][j]
            r = results.get(f"{group}_{model}")
            hist = (r or {}).get("learning_curves") or {}
            tr, va = hist.get("train_acc") or [], hist.get("val_acc") or []
            if not tr and not va:
                ax.text(.5, .5, "sin curvas", ha="center", va="center",
                        transform=ax.transAxes, color="#999")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                if tr: ax.plot(tr, lw=1, label="Train")
                if va: ax.plot(va, lw=1, label="Validacion")
                ax.set_ylim(0, 1.02); ax.legend(fontsize=7); ax.grid(alpha=.3)
                ax.set_xlabel("Epoca"); ax.set_ylabel("Accuracy")
                drew = True
            ax.set_title(f"{group.capitalize()} · {DISPLAY[model]}")
    if not drew:
        plt.close(fig); return None
    fig.suptitle("Curvas de aprendizaje (train vs. validacion)", fontweight="bold")
    fig.tight_layout()
    path = out / "fig_curvas_aprendizaje.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def fig_copeland(global_rank: pd.DataFrame, out: Path) -> Optional[Path]:
    if global_rank.empty:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    d = global_rank.sort_values("copeland_score_global")
    colors = ["#c0392b" if v < 0 else "#7f8c8d" if v == 0 else "#27ae60"
              for v in d["copeland_score_global"]]
    ax.barh(d["classifier"], d["copeland_score_global"], color=colors)
    for y, v in zip(d["classifier"], d["copeland_score_global"]):
        ax.text(v + (.06 if v >= 0 else -.06), y, f"{v:+d}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", lw=.8)
    ax.set_xlabel("Puntaje Copeland (victorias - derrotas)")
    ax.set_title("Ranking de modelos · Algoritmo de Copeland", fontweight="bold")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    path = out / "fig_ranking_copeland.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def fig_importance(results, cfg, out: Path, top_n: int = 12) -> Optional[Path]:
    plt = _mpl()
    groups = list(cfg["groups"])
    fig, axes = plt.subplots(len(groups), len(MODEL_NAMES),
                             figsize=(4.6 * len(MODEL_NAMES), 3.4 * len(groups)),
                             squeeze=False)
    drew = False
    for i, group in enumerate(groups):
        for j, model in enumerate(MODEL_NAMES):
            ax = axes[i][j]
            r = results.get(f"{group}_{model}")
            top = ((r or {}).get("interpretabilidad") or {}).get("top_15") or {}
            if not top:
                ax.text(.5, .5, "sin datos", ha="center", va="center",
                        transform=ax.transAxes, color="#999")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                items = list(top.items())[:top_n][::-1]
                ax.barh([k[:22] for k, _ in items], [v for _, v in items],
                        color="#2c5c4d")
                ax.tick_params(labelsize=7)
                ax.set_xlabel("Importancia normalizada")
                drew = True
            ax.set_title(f"{group.capitalize()} · {DISPLAY[model]}", fontsize=9)
    if not drew:
        plt.close(fig); return None
    fig.suptitle("Importancia de variables por arquitectura", fontweight="bold")
    fig.tight_layout()
    path = out / "fig_importancia_variables.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


# =====================================================================
# Entrada
# =====================================================================
def build_report(cfg: dict, root: Path) -> Dict[str, object]:
    out = root / cfg["paths"]["figures_dir"]
    out.mkdir(parents=True, exist_ok=True)

    results, missing = load_results(cfg, root)
    if not results:
        raise RuntimeError(
            "No hay ningun results.json en "
            f"{root / cfg['paths']['results_dir']}. Ejecuta antes:  "
            "python run_pipeline.py train"
        )
    if missing:
        print(f"  AVISO · combinaciones sin resultados ({len(missing)}): "
              f"{', '.join(missing)}")
        print("  Las tablas y figuras las omiten. No se rellena nada.")

    t3, t4 = tabla3(results, cfg), tabla4(results, cfg)
    t3.to_csv(out / "tabla3_metricas.csv", index=False)
    t4.to_csv(out / "tabla4_auc.csv", index=False)
    (out / "tabla3_metricas.tex").write_text(
        t3.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
    (out / "tabla4_auc.tex").write_text(
        t4.to_latex(index=False, float_format="%.4f"), encoding="utf-8")

    per_group, global_rank = ranking_copeland(results, cfg)
    for group, table in per_group.items():
        table.to_csv(out / f"copeland_{group}.csv", index=False)
    if not global_rank.empty:
        global_rank.to_csv(out / "copeland_global.csv", index=False)

    est = tabla_estabilidad(results, cfg)
    if not est.empty:
        est.to_csv(out / "estabilidad_multisemilla.csv", index=False)
    interp = tabla_interpretabilidad(results)
    if not interp.empty:
        interp.to_csv(out / "interpretabilidad_top.csv", index=False)

    figures = {
        "confusion": fig_confusion(results, cfg, out),
        "roc": fig_roc(results, cfg, out),
        "learning": fig_learning(results, cfg, out),
        "copeland": fig_copeland(global_rank, out),
        "importance": fig_importance(results, cfg, out),
    }

    print(f"\n  Tabla 3 ({len(t3)} filas):")
    print(t3.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not global_rank.empty:
        print("\n  Ranking Copeland global:")
        print(global_rank.to_string(index=False))

    return {
        "output_dir": str(out),
        "combinaciones_con_resultados": sorted(results),
        "combinaciones_faltantes": missing,
        "figuras": {k: (str(v) if v else None) for k, v in figures.items()},
    }
