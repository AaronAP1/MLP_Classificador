"""
Tablas y figuras del articulo, generadas exclusivamente desde artefactos
reales de entrenamiento.

Regla central: si falta el `results.json` de una combinacion, la funcion
lo reporta como faltante y lo deja fuera. NUNCA rellena con datos
simulados. El generador anterior producia curvas ROC y matrices de
confusion con `np.random` cuando no encontraba resultados, que es como
una figura ausente se convierte en una figura falsa.

Salida (outputs/aligned/). La correspondencia con el manuscrito queda
escrita en INDICE_FIGURAS.md dentro de la misma carpeta.

    tabla_resumen_datos.csv          registros, edades, clases y variables por grupo
    tabla2_hiperparametros.csv       configuracion elegida por el Grid Search
    tabla3_metricas.{csv,tex}        Tabla 3 · metricas en test (semilla 42)
    tabla3_media_std.{csv,tex}       Tabla 3 · media ± desviacion (5 semillas)
    tabla4_auc.{csv,tex}             Tabla 4 · AUC macro one-vs-rest
    tabla4_auc_por_clase.csv         AUC de cada clase
    copeland_<grupo>.csv / copeland_global.csv
    estabilidad_multisemilla.csv     III-F-4
    interpretabilidad_top.csv        III-G
    fig_comparacion_grupos.png       Fig. 7
    fig_matrices_confusion.png       Fig. 8
    fig_curvas_roc.png               Fig. 9  (3x3: grupo x modelo)
    fig_curvas_roc_por_grupo.png     Fig. 9  (variante 1x3)
    fig_ranking_copeland.png         Fig. 10
    fig_curvas_aprendizaje.png       Fig. 11
    fig_importancia_variables.png    III-G · importancia propia de cada arquitectura
    fig_importancia_permutacion.png  III-G · importancia por permutacion (comun)
    fig_mascaras_tabnet.png          III-G · mascaras de seleccion de TabNet
    fig_atencion_ft.png              III-G · mapas de atencion del FT-Transformer
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import variables as V
from .evaluate import copeland_across_groups, copeland_ranking
from .models import MODEL_NAMES

DISPLAY = {"mlp_ma": "MLP-MA", "tabnet": "TabNet", "ft_transformer": "FT-Transformer"}
GROUP_DISPLAY = {"ninos": "Niños", "jovenes": "Jóvenes", "adultos": "Adultos"}
MODEL_COLORS = {"mlp_ma": "#3b82c4", "tabnet": "#2e9e6b", "ft_transformer": "#d9654b"}
CLASS_STYLES = ["-", "--", ":"]
SMOKE_MARK = "PRUEBA DE HUMO · NO SON RESULTADOS"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "font.family": "serif",
    })
    return plt


def _group_title(cfg: dict, group: str) -> str:
    g = cfg["groups"][group]
    hi = "+" if g["edad_max"] >= 99 else f"-{g['edad_max']}"
    return f"{GROUP_DISPLAY.get(group, group)} ({g['edad_min']}{hi} años)"


def _save(fig, plt, path: Path, cfg: dict) -> Path:
    if cfg.get("smoke"):
        fig.text(0.5, 0.5, SMOKE_MARK, ha="center", va="center", rotation=25,
                 fontsize=26, color="#c0392b", alpha=0.25, fontweight="bold",
                 transform=fig.transFigure, zorder=100)
    if cfg.get("run_label"):
        fig.text(0.995, 0.002, f"Corrida: {cfg['run_label']}", ha="right", va="bottom",
                 fontsize=7, color="#888", transform=fig.transFigure)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


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


def _load_npz(cfg: dict, root: Path, key: str) -> Optional[dict]:
    path = root / cfg["paths"]["results_dir"] / key / "interpretabilidad.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# =====================================================================
# Tablas
# =====================================================================
def tabla_resumen_datos(cfg: dict, root: Path) -> pd.DataFrame:
    """Lo que el articulo necesita para describir los datos de cada grupo."""
    rows = []
    for group in cfg["groups"]:
        path = root / cfg["paths"]["processed_dir"] / group / "config.json"
        if not path.exists():
            continue
        c = json.loads(path.read_text(encoding="utf-8"))
        counts = c["target"]["counts"]
        total = sum(counts.values())
        row = {
            "Grupo": GROUP_DISPLAY.get(group, group),
            "Edades": f"{c['segmentacion']['edad_observada'][0]}-{c['segmentacion']['edad_observada'][1]}",
            "Registros con clase": c["target"]["n_valid"],
            "Eliminados >20% faltantes": c["limpieza"]["n_removidos"],
            "Registros finales": sum(v["n"] for v in c["splits"].values()),
        }
        for name, n in counts.items():
            row[name] = f"{n} ({100 * n / total:.1f} %)"
        row["Variables finales"] = c["n_features"]
        row["Train/Val/Test"] = " / ".join(str(c["splits"][s]["n"]) for s in ("train", "val", "test"))
        rows.append(row)
    return pd.DataFrame(rows)


def tabla_hiperparametros(results: Dict[str, dict], cfg: dict) -> pd.DataFrame:
    rows = []
    for group in cfg["groups"]:
        for model in MODEL_NAMES:
            r = results.get(f"{group}_{model}")
            if not r:
                continue
            hp = r.get("hiperparametros_finales", {})
            gs = r.get("gridsearch", {})
            rows.append({
                "Grupo": GROUP_DISPLAY.get(group, group), "Modelo": DISPLAY[model],
                "Learning rate": hp.get("lr"), "Batch size": hp.get("batch_size"),
                # TabNet no usa dropout: se deja vacio en lugar de un valor inerte.
                "Dropout": None if model == "tabnet" else hp.get("dropout"),
                "Weight decay": hp.get("weight_decay"),
                "Profundidad": hp.get("n_steps", hp.get("depth")),
                "F1-Macro CV (media)": gs.get("best_cv_score"),
                "F1-Macro CV (desv.)": gs.get("best_cv_std"),
                "Epocas (etapa A)": r.get("mejor_epoca_etapa_a"),
            })
    return pd.DataFrame(rows)


def tabla3(results: Dict[str, dict], cfg: dict) -> pd.DataFrame:
    rows = []
    for group in cfg["groups"]:
        for model in MODEL_NAMES:
            r = results.get(f"{group}_{model}")
            if not r:
                continue
            m = r["test_metrics"]
            rows.append({
                "Grupo": GROUP_DISPLAY.get(group, group), "Modelo": DISPLAY[model],
                "Accuracy": m["accuracy"], "F1-Weighted": m["f1_weighted"],
                "F1-Macro": m["f1_macro"],
                "Precision-Weighted": m["precision_weighted"],
                "Recall-Weighted": m["recall_weighted"],
                "n_test": int(sum(m["support_per_class"])),
            })
    return pd.DataFrame(rows)


def tabla3_media_std(results: Dict[str, dict], cfg: dict) -> pd.DataFrame:
    """Tabla 3 en formato media ± desviacion sobre las semillas."""
    rows = []
    keys = [("accuracy", "Accuracy"), ("f1_weighted", "F1-Weighted"),
            ("f1_macro", "F1-Macro"), ("precision_weighted", "Precision-Weighted"),
            ("recall_weighted", "Recall-Weighted"), ("roc_auc_ovr", "AUC macro")]
    for group in cfg["groups"]:
        for model in MODEL_NAMES:
            r = results.get(f"{group}_{model}")
            est = (r or {}).get("estabilidad") or {}
            if not est.get("runs"):
                continue
            row = {"Grupo": GROUP_DISPLAY.get(group, group), "Modelo": DISPLAY[model],
                   "Semillas": est.get("n_runs")}
            for k, label in keys:
                mu, sd = est.get(f"{k}_mean"), est.get(f"{k}_std")
                row[label] = f"{mu:.4f} ± {sd:.4f}" if mu is not None else ""
            rows.append(row)
    return pd.DataFrame(rows)


def tabla4(results: Dict[str, dict], cfg: dict) -> pd.DataFrame:
    rows = []
    for group in cfg["groups"]:
        row = {"Grupo": GROUP_DISPLAY.get(group, group)}
        for model in MODEL_NAMES:
            r = results.get(f"{group}_{model}")
            row[DISPLAY[model]] = r["test_metrics"]["roc_auc_ovr"] if r else None
        rows.append(row)
    df = pd.DataFrame(rows)
    numeric = [DISPLAY[m] for m in MODEL_NAMES]
    if not df.empty:
        promedio = {"Grupo": "Promedio"}
        for c in numeric:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            promedio[c] = float(vals.mean()) if len(vals) else None
        df = pd.concat([df, pd.DataFrame([promedio])], ignore_index=True)
    return df


def tabla4_por_clase(results: Dict[str, dict], cfg: dict) -> pd.DataFrame:
    rows = []
    for group in cfg["groups"]:
        for model in MODEL_NAMES:
            r = results.get(f"{group}_{model}")
            if not r:
                continue
            per = r["test_metrics"].get("roc_auc_per_class") or {}
            rows.append({"Grupo": GROUP_DISPLAY.get(group, group), "Modelo": DISPLAY[model],
                         **{f"AUC {k}": v for k, v in per.items()},
                         "AUC macro": r["test_metrics"].get("roc_auc_ovr")})
    return pd.DataFrame(rows)


def tabla_estabilidad(results: Dict[str, dict], cfg: dict) -> pd.DataFrame:
    rows = []
    for r in results.values():
        est = r.get("estabilidad") or {}
        if not est.get("runs"):
            continue
        row = {"Grupo": GROUP_DISPLAY.get(r["grupo"], r["grupo"]),
               "Modelo": DISPLAY[r["modelo"]], "n_semillas": est.get("n_runs")}
        for k in ("accuracy", "f1_weighted", "f1_macro", "roc_auc_ovr"):
            row[f"{k}_mean"] = est.get(f"{k}_mean")
            row[f"{k}_std"] = est.get(f"{k}_std")
        rows.append(row)
    return pd.DataFrame(rows)


INTERP_METHODS = {
    "propia": ("Atención / máscaras de la arquitectura", "Importancia normalizada"),
    "permutacion": ("Permutación (caída de F1-Macro)", "Caída de F1-Macro al permutar"),
}


def _top(r: dict, method: str) -> Dict[str, float]:
    interp = (r or {}).get("interpretabilidad") or {}
    if method == "permutacion":
        return (interp.get("permutacion") or {}).get("top_15") or {}
    return interp.get("top_15") or {}


def tabla_interpretabilidad(results: Dict[str, dict], top_n: int = 15) -> pd.DataFrame:
    rows = []
    for r in results.values():
        for method, (label, _) in INTERP_METHODS.items():
            for rank, (feat, weight) in enumerate(list(_top(r, method).items())[:top_n], 1):
                rows.append({
                    "Grupo": GROUP_DISPLAY.get(r["grupo"], r["grupo"]),
                    "Modelo": DISPLAY[r["modelo"]], "Metodo": label, "Puesto": rank,
                    "Variable": feat, "Descripcion": V.feature_label(feat),
                    "Dimension": V.dimension_of(feat), "Importancia": weight,
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
def fig_comparacion(results, cfg, out: Path) -> Optional[Path]:
    """Fig. 7 · metricas por grupo y arquitectura (barras ± desv. multi-semilla)."""
    plt = _mpl()
    groups = [g for g in cfg["groups"] if any(f"{g}_{m}" in results for m in MODEL_NAMES)]
    if not groups:
        return None
    panels = [("accuracy", "Accuracy"), ("f1_weighted", "F1-Weighted"),
              ("f1_macro", "F1-Macro"), ("precision_weighted", "Precision-Weighted")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8))
    width = 0.8 / len(MODEL_NAMES)
    x = np.arange(len(groups))
    for ax, (key, label) in zip(axes.ravel(), panels):
        for j, model in enumerate(MODEL_NAMES):
            vals, errs = [], []
            for g in groups:
                r = results.get(f"{g}_{model}")
                vals.append(r["test_metrics"][key] if r else np.nan)
                est = (r or {}).get("estabilidad") or {}
                errs.append(est.get(f"{key}_std") or 0.0)
            ax.bar(x + (j - 1) * width, vals, width, yerr=errs, capsize=3,
                   color=MODEL_COLORS[model], label=DISPLAY[model], alpha=.9,
                   edgecolor="white", error_kw={"elinewidth": .8})
        ax.set_xticks(x)
        ax.set_xticklabels([_group_title(cfg, g) for g in groups], fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_title(label, fontweight="bold")
        ax.set_ylabel(label); ax.grid(axis="y", alpha=.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MODEL_NAMES),
               bbox_to_anchor=(0.5, 0.955), frameon=False)
    fig.suptitle("Comparación del rendimiento entre grupos (conjunto de prueba)",
                 fontweight="bold", y=0.995)
    fig.text(0.5, 0.005, "Barras de error: desviación estándar entre semillas.",
             ha="center", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    return _save(fig, plt, out / "fig_comparacion_grupos.png", cfg)


def fig_confusion(results, cfg, out: Path) -> Optional[Path]:
    """Fig. 8 · matrices de confusion normalizadas por fila (recall por clase)."""
    plt = _mpl()
    groups, names = list(cfg["groups"]), cfg["class_names"]
    fig, axes = plt.subplots(len(groups), len(MODEL_NAMES),
                             figsize=(4.8 * len(MODEL_NAMES), 3.7 * len(groups)),
                             squeeze=False)
    drew = False
    for i, group in enumerate(groups):
        for j, model in enumerate(MODEL_NAMES):
            ax = axes[i][j]
            r = results.get(f"{group}_{model}")
            title = f"{_group_title(cfg, group)} · {DISPLAY[model]}"
            if not r:
                ax.text(.5, .5, "sin resultados", ha="center", va="center",
                        transform=ax.transAxes, color="#999")
                ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
                continue
            cm = np.array(r["test_metrics"]["confusion_matrix_normalized"])
            raw = np.array(r["test_metrics"]["confusion_matrix"])
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            for a in range(cm.shape[0]):
                for b in range(cm.shape[1]):
                    ax.text(b, a, f"{cm[a, b]:.2f}\n(n={raw[a, b]})", ha="center",
                            va="center", fontsize=7,
                            color="white" if cm[a, b] > .5 else "black")
            ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
            ax.set_yticklabels(names if j == 0 else [""] * len(names), fontsize=7)
            ax.set_title(title, fontsize=9)
            if i == len(groups) - 1:
                ax.set_xlabel("Clase predicha")
            if j == 0:
                ax.set_ylabel("Clase real")
            fig.colorbar(im, ax=ax, fraction=.046)
            drew = True
    if not drew:
        plt.close(fig); return None
    fig.suptitle("Matrices de confusión normalizadas por fila (conjunto de prueba)",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, plt, out / "fig_matrices_confusion.png", cfg)


def fig_roc(results, cfg, out: Path) -> Optional[Path]:
    """Fig. 9 · ROC one-vs-rest, una celda por grupo y arquitectura."""
    plt = _mpl()
    groups, names = list(cfg["groups"]), cfg["class_names"]
    colors = ["#c0392b", "#2c7fb8", "#27ae60"]
    fig, axes = plt.subplots(len(groups), len(MODEL_NAMES),
                             figsize=(4.2 * len(MODEL_NAMES), 3.9 * len(groups)),
                             squeeze=False)
    drew = False
    for i, group in enumerate(groups):
        for j, model in enumerate(MODEL_NAMES):
            ax = axes[i][j]
            r = results.get(f"{group}_{model}")
            ax.set_title(f"{_group_title(cfg, group)} · {DISPLAY[model]}", fontsize=9)
            if not r or not r.get("roc_curves"):
                ax.text(.5, .5, "sin resultados", ha="center", va="center",
                        transform=ax.transAxes, color="#999")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            for cls, curve in sorted(r["roc_curves"].items()):
                c = int(cls)
                ax.plot(curve["fpr"], curve["tpr"], lw=1.5, color=colors[c % 3],
                        label=f"{names[c]} (AUC={curve['auc']:.3f})")
            ax.plot([0, 1], [0, 1], "--", color="#999", lw=.8)
            macro = r["test_metrics"].get("roc_auc_ovr")
            if macro is not None:
                ax.text(.97, .03, f"AUC macro = {macro:.3f}", ha="right", va="bottom",
                        transform=ax.transAxes, fontsize=7,
                        bbox={"fc": "white", "ec": "#ccc", "pad": 2})
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
            ax.legend(fontsize=6.5, loc="center right"); ax.grid(alpha=.3)
            drew = True
    if not drew:
        plt.close(fig); return None
    fig.suptitle("Curvas ROC one-vs-rest por grupo etario y arquitectura (conjunto de prueba)",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, plt, out / "fig_curvas_roc.png", cfg)


def fig_roc_por_grupo(results, cfg, out: Path) -> Optional[Path]:
    """Fig. 9 (variante) · las tres arquitecturas superpuestas en cada grupo."""
    plt = _mpl()
    groups, names = list(cfg["groups"]), cfg["class_names"]
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 4.6), squeeze=False)
    drew = False
    for i, group in enumerate(groups):
        ax = axes[0][i]
        for model in MODEL_NAMES:
            r = results.get(f"{group}_{model}")
            if not r or not r.get("roc_curves"):
                continue
            for cls, curve in sorted(r["roc_curves"].items()):
                c = int(cls)
                ax.plot(curve["fpr"], curve["tpr"], lw=1.2, color=MODEL_COLORS[model],
                        ls=CLASS_STYLES[c % 3],
                        label=f"{DISPLAY[model]} · {names[c]} ({curve['auc']:.3f})")
                drew = True
        ax.plot([0, 1], [0, 1], "--", color="#999", lw=.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_title(_group_title(cfg, group)); ax.legend(fontsize=5.8, loc="lower right")
        ax.grid(alpha=.3)
    if not drew:
        plt.close(fig); return None
    fig.suptitle("Curvas ROC one-vs-rest por grupo etario", fontweight="bold")
    fig.tight_layout()
    return _save(fig, plt, out / "fig_curvas_roc_por_grupo.png", cfg)


def fig_learning(results, cfg, out: Path) -> Optional[Path]:
    """Fig. 11 · accuracy de train y de validacion por epoca (etapa A)."""
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
                from matplotlib.ticker import MaxNLocator
                ep = np.arange(1, max(len(tr), len(va)) + 1)
                if tr: ax.plot(ep[:len(tr)], tr, lw=1, color="#2c7fb8", label="Entrenamiento")
                if va: ax.plot(ep[:len(va)], va, lw=1, color="#d9654b", label="Validación")
                best = (r or {}).get("mejor_epoca_etapa_a")
                if best:
                    ax.axvline(best, color="#555", ls=":", lw=.9,
                               label=f"Mejor época: {best} (mín. pérdida val.)")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_ylim(0, 1.02); ax.legend(fontsize=6.5, loc="lower right")
                ax.grid(alpha=.3); ax.set_xlabel("Época"); ax.set_ylabel("Accuracy")
                drew = True
            ax.set_title(f"{_group_title(cfg, group)} · {DISPLAY[model]}", fontsize=9)
    if not drew:
        plt.close(fig); return None
    fig.suptitle("Curvas de aprendizaje: entrenamiento (70 %) vs. validación (15 %)",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, plt, out / "fig_curvas_aprendizaje.png", cfg)


def fig_copeland(global_rank: pd.DataFrame, cfg: dict, out: Path) -> Optional[Path]:
    """Fig. 10 · puntaje Copeland acumulado en los tres grupos."""
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
    n_groups = len([c for c in global_rank.columns if c.startswith("score_")])
    lim = 2 * max(n_groups, 1)
    ax.set_xlim(-lim - .8, lim + .8)
    ax.axvline(0, color="black", lw=.8)
    ax.set_xlabel(f"Puntaje Copeland acumulado en {n_groups} grupos (rango −{lim} a +{lim})")
    ax.set_title("Ranking de modelos · Algoritmo de Copeland", fontweight="bold")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    return _save(fig, plt, out / "fig_ranking_copeland.png", cfg)


def fig_importance(results, cfg, out: Path, method: str = "propia",
                   top_n: int = 12) -> Optional[Path]:
    plt = _mpl()
    groups = list(cfg["groups"])
    title, xlabel = INTERP_METHODS[method]
    fig, axes = plt.subplots(len(groups), len(MODEL_NAMES),
                             figsize=(5.4 * len(MODEL_NAMES), 3.6 * len(groups)),
                             squeeze=False)
    drew = False
    for i, group in enumerate(groups):
        for j, model in enumerate(MODEL_NAMES):
            ax = axes[i][j]
            top = _top(results.get(f"{group}_{model}"), method)
            if not top:
                ax.text(.5, .5, "sin datos", ha="center", va="center",
                        transform=ax.transAxes, color="#999")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                items = list(top.items())[:top_n][::-1]
                ax.barh([V.feature_label(k)[:38] for k, _ in items], [v for _, v in items],
                        color=MODEL_COLORS[model])
                ax.axvline(0, color="#333", lw=.6)
                ax.tick_params(labelsize=6.5)
                ax.set_xlabel(xlabel)
                drew = True
            ax.set_title(f"{_group_title(cfg, group)} · {DISPLAY[model]}", fontsize=9)
    if not drew:
        plt.close(fig); return None
    fig.suptitle(f"Importancia de variables · {title} (conjunto de prueba)",
                 fontweight="bold")
    fig.tight_layout()
    name = "fig_importancia_variables.png" if method == "propia" else "fig_importancia_permutacion.png"
    return _save(fig, plt, out / name, cfg)


def fig_mascaras_tabnet(results, cfg, root: Path, out: Path, top_n: int = 12) -> Optional[Path]:
    """Mascaras sparsemax medias de cada decision step (III-G-1)."""
    plt = _mpl()
    groups = [g for g in cfg["groups"] if f"{g}_tabnet" in results]
    panels = []
    for g in groups:
        z = _load_npz(cfg, root, f"{g}_tabnet")
        if z is not None and "tabnet_step_masks" in z:
            panels.append((g, z))
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.2), squeeze=False)
    for ax, (g, z) in zip(axes[0], panels):
        names = [str(n) for n in z["feature_names"]]
        order = np.argsort(z["importance"])[::-1][:top_n]
        masks = z["tabnet_step_masks"][:, order]
        im = ax.imshow(masks, aspect="auto", cmap="viridis")
        ax.set_yticks(range(masks.shape[0]))
        ax.set_yticklabels([f"Paso {k + 1}" for k in range(masks.shape[0])])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([V.feature_label(names[k])[:30] for k in order],
                           rotation=60, ha="right", fontsize=6.5)
        ax.set_title(_group_title(cfg, g), fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.046, label="Peso medio de la máscara")
    fig.suptitle("TabNet · máscaras de selección por decision step", fontweight="bold")
    fig.tight_layout()
    return _save(fig, plt, out / "fig_mascaras_tabnet.png", cfg)


def fig_atencion_ft(results, cfg, root: Path, out: Path, top_n: int = 10) -> Optional[Path]:
    """Atencion media entre las variables mas relevantes (III-G-2)."""
    plt = _mpl()
    panels = []
    for g in cfg["groups"]:
        z = _load_npz(cfg, root, f"{g}_ft_transformer")
        if z is not None and "ft_attention" in z:
            panels.append((g, z))
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.0), squeeze=False)
    for ax, (g, z) in zip(axes[0], panels):
        names = [str(n) for n in z["feature_names"]]
        order = np.argsort(z["importance"])[::-1][:top_n]
        att = z["ft_attention"][1:, 1:]                      # sin el token [CLS]
        sub = att[np.ix_(order, order)]
        im = ax.imshow(sub, cmap="magma")
        labels = [V.feature_label(names[k])[:26] for k in order]
        ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel("Variable atendida (j)"); ax.set_ylabel("Variable que atiende (i)")
        ax.set_title(_group_title(cfg, g), fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.046, label="α_ij media (capas y cabezas)")
    fig.suptitle("FT-Transformer · atención entre las variables más relevantes",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, plt, out / "fig_atencion_ft.png", cfg)


# =====================================================================
# Indice de figuras
# =====================================================================
INDEX_TEXT = """# Índice de tablas y figuras

Generado por `python run_pipeline.py report` a partir de `{results_dir}`.
{smoke_note}
| Archivo | Va en el artículo | Notas |
|---|---|---|
| `tabla_resumen_datos.csv` | Sección de datos (y reemplazo de la Table 1) | Registros, edades, clases y n.º de variables por grupo |
| `tabla2_hiperparametros.csv` | Table 2 | Configuración elegida por el Grid Search en cada combinación |
| `tabla3_metricas.csv/.tex` | Table 3 | Métricas en test del modelo final (semilla 42) |
| `tabla3_media_std.csv/.tex` | Table 3 (alternativa) o texto de estabilidad | Media ± desviación en {n_seeds} semillas |
| `tabla4_auc.csv/.tex` | Table 4 | AUC one-vs-rest, promedio **macro** |
| `tabla4_auc_por_clase.csv` | Texto de la sección ROC | AUC de cada clase |
| `fig_comparacion_grupos.png` | Fig. 7 | |
| `fig_matrices_confusion.png` | Fig. 8 | Normalizadas por fila; entre paréntesis, conteos |
| `fig_curvas_roc.png` | Fig. 9 | 3×3 (grupo × modelo). Variante 1×3: `fig_curvas_roc_por_grupo.png` |
| `fig_ranking_copeland.png` | Fig. 10 | Rango posible −6 a +6 (3 grupos, 2 rivales por grupo) |
| `fig_curvas_aprendizaje.png` | Fig. 11 | Etapa A: train 70 % vs. validación 15 % |
| `fig_importancia_variables.png` | Sección de interpretabilidad | Atención (MLP-MA, FT) y máscaras (TabNet) |
| `fig_importancia_permutacion.png` | Sección de interpretabilidad | Caída de F1-Macro al permutar cada variable; misma vara para los 3 modelos |
| `fig_mascaras_tabnet.png` | Sección de interpretabilidad | Máscaras por decision step |
| `fig_atencion_ft.png` | Sección de interpretabilidad | Atención entre las variables más relevantes |
| `interpretabilidad_top.csv` | Sección de interpretabilidad | Top 15 por grupo, modelo y método, con descripción |
| `copeland_*.csv`, `estabilidad_multisemilla.csv` | Texto de resultados | |

Equipo de entrenamiento: {hardware}
Preset de Grid Search: {preset} · Corrida: {run_label}
"""


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

    resumen = tabla_resumen_datos(cfg, root)
    if not resumen.empty:
        resumen.to_csv(out / "tabla_resumen_datos.csv", index=False)
    tabla_hiperparametros(results, cfg).to_csv(out / "tabla2_hiperparametros.csv", index=False)

    t3, t4 = tabla3(results, cfg), tabla4(results, cfg)
    t3.to_csv(out / "tabla3_metricas.csv", index=False)
    t4.to_csv(out / "tabla4_auc.csv", index=False)
    (out / "tabla3_metricas.tex").write_text(
        t3.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
    (out / "tabla4_auc.tex").write_text(
        t4.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
    tabla4_por_clase(results, cfg).to_csv(out / "tabla4_auc_por_clase.csv", index=False)

    t3s = tabla3_media_std(results, cfg)
    if not t3s.empty:
        t3s.to_csv(out / "tabla3_media_std.csv", index=False)
        (out / "tabla3_media_std.tex").write_text(t3s.to_latex(index=False), encoding="utf-8")

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
        "fig7_comparacion": fig_comparacion(results, cfg, out),
        "fig8_confusion": fig_confusion(results, cfg, out),
        "fig9_roc": fig_roc(results, cfg, out),
        "fig9b_roc_por_grupo": fig_roc_por_grupo(results, cfg, out),
        "fig10_copeland": fig_copeland(global_rank, cfg, out),
        "fig11_aprendizaje": fig_learning(results, cfg, out),
        "importancia": fig_importance(results, cfg, out, "propia"),
        "importancia_permutacion": fig_importance(results, cfg, out, "permutacion"),
        "mascaras_tabnet": fig_mascaras_tabnet(results, cfg, root, out),
        "atencion_ft": fig_atencion_ft(results, cfg, root, out),
    }

    hw = next(iter(results.values())).get("hardware", {})
    hw_text = hw.get("gpu") or hw.get("procesador") or hw.get("dispositivo", "?")
    (out / "INDICE_FIGURAS.md").write_text(INDEX_TEXT.format(
        results_dir=cfg["paths"]["results_dir"],
        smoke_note=("\n> **PRUEBA DE HUMO.** Pocos datos y pocas épocas: estos números "
                    "solo verifican que el código funciona. No son resultados.\n"
                    if cfg.get("smoke") else ""),
        n_seeds=len(cfg["evaluation"]["stability"]["seeds"]),
        hardware=f"{hw_text} · PyTorch {hw.get('torch', '?')}",
        preset=next(iter(results.values())).get("gridsearch", {}).get("preset", "sin busqueda"),
        run_label=cfg.get("run_label") or "definitiva",
    ), encoding="utf-8")

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
