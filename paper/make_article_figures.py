"""
Figures 7-12 and Table 4 of the article, rebuilt from the saved run in results/final.

    python paper/make_article_figures.py

Writes paper/figures/fig07_comparison.png ... fig12_importance.png and
paper/tables/table4_auc_per_class.csv. Nothing is retrained: every value comes
from the per-seed test probabilities (test_predictions_seeds.npz), the stage-A
learning curves (results.json) and the permutation importance
(interpretabilidad.npz) of each group x model folder.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "final"
REP = ROOT / "reports" / "final"
OUT = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

GROUPS = ["ninos", "jovenes", "adultos"]
GROUP_EN = {"ninos": "Children", "jovenes": "Youths", "adultos": "Adults"}
AGES = {"ninos": "3–13", "jovenes": "14–17", "adultos": "18–64"}
MODELS = ["mlp_ma", "tabnet", "ft_transformer"]
KEYS = MODELS + ["ensemble"]
NAMES = {"mlp_ma": "MLP+MA", "tabnet": "TabNet", "ft_transformer": "FT-Transformer", "ensemble": "Ensemble"}
COLORS = {"mlp_ma": "#3B7EC1", "tabnet": "#2D9E69", "ft_transformer": "#DB6449", "ensemble": "#7B5DA6"}
CLASSES = ["Non-reader", "Average reader", "Frequent reader"]     # class index 0, 1, 2
CLASSES_SHORT = ["Non", "Avg.", "Freq."]

FEATURES_EN = {
    "CCDD": "Department", "DOMINIO": "Geographic domain", "ESTRATO": "Size of populated center",
    "ESTRATOSOCIO": "Socioeconomic stratum", "P209": "Sex", "P210_A": "Age (years)", "P212": "Marital status",
    "P309": "Currently in education", "P310_C": "Type of school attended",
    "P313": "Occupational category", "P513A": "Mother tongue", "P514_1": "Current educational level",
    "P514_3": "School grade", "idx_nivel_educativo": "Educational level",
    "hh_nivel_educ_mean": "Household: mean education",
    "hh_pueblo_indigena": "Household: Indigenous adults", "hh_etnia_jefe": "Household: head's ethnicity",
    "hh_ingreso_mean": "Household: average income", "hh_tamano": "Household: size",
}

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "savefig.dpi": 400,
    "legend.frameon": False, "mathtext.fontset": "stix",
})


def save(fig, name, tight=True):
    fig.savefig(OUT / name, bbox_inches="tight" if tight else None, pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print(f"  {OUT.relative_to(ROOT) / name}")


# ---------------------------------------------------------------- data
def load_probs(group):
    probs, y = {}, None
    for m in MODELS:
        z = np.load(RES / f"{group}_{m}" / "test_predictions_seeds.npz")
        probs[m], y = z["y_proba"].astype(float), z["y_true"]
    probs["ensemble"] = np.mean([probs[m] for m in MODELS], axis=0)   # soft voting, seed by seed
    return probs, y


def seed_metrics(p, y):
    rows = []
    for s in range(p.shape[0]):
        pred = p[s].argmax(1)
        rows.append({"Accuracy": accuracy_score(y, pred),
                     "F1-Weighted": f1_score(y, pred, average="weighted"),
                     "F1-Macro": f1_score(y, pred, average="macro"),
                     "AUC": roc_auc_score(y, p[s], multi_class="ovr", average="macro")})
    return pd.DataFrame(rows)


DATA = {g: load_probs(g) for g in GROUPS}
MET = {(g, k): seed_metrics(DATA[g][0][k], DATA[g][1]) for g in GROUPS for k in KEYS}

# consistency check against the pipeline report (population std, ddof=0)
ref = pd.read_csv(REP / "estabilidad_multisemilla.csv")
r = ref[(ref.Grupo == "Niños") & (ref.Modelo == "MLP-MA")].iloc[0]
assert abs(MET["ninos", "mlp_ma"]["F1-Macro"].mean() - r.f1_macro_mean) < 1e-9
assert abs(MET["ninos", "mlp_ma"]["F1-Macro"].std(ddof=0) - r.f1_macro_std) < 1e-9

# ---------------------------------------------------------------- Table 4 · AUC per class (mean of seeds)
rows = []
for g in GROUPS:
    p, y = DATA[g]
    for k in KEYS:
        per_class = np.array([[roc_auc_score(y == c, p[k][s][:, c]) for c in range(3)]
                              for s in range(p[k].shape[0])])
        macro = per_class.mean(axis=1)
        rows.append({"Group": GROUP_EN[g], "Model": NAMES[k],
                     **{f"AUC {c}": round(v, 3) for c, v in zip(CLASSES, per_class.mean(axis=0))},
                     "Macro AUC (mean)": round(macro.mean(), 3), "Macro AUC (SD)": round(macro.std(), 3)})
pd.DataFrame(rows).to_csv(TAB / "table4_auc_per_class.csv", index=False)
print(f"  {(TAB / 'table4_auc_per_class.csv').relative_to(ROOT)}")

# ---------------------------------------------------------------- Fig. 7 · performance comparison
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
panels = [("Accuracy", "Accuracy", (0.2, 0.8)), ("F1-Weighted", "F1-Weighted", (0.2, 0.8)),
          ("F1-Macro", "F1-Macro", (0.2, 0.8)), ("AUC", "Macro AUC (OvR)", (0.45, 0.78))]
w = 0.2
for ax, (col, title, ylim) in zip(axes.flat, panels):
    for i, k in enumerate(KEYS):
        mu = [MET[g, k][col].mean() for g in GROUPS]
        sd = [MET[g, k][col].std(ddof=0) for g in GROUPS]
        ax.bar(np.arange(3) + (i - 1.5) * w, mu, w, yerr=sd, color=COLORS[k], edgecolor="white", lw=0.6,
               capsize=3, error_kw=dict(lw=0.9, capthick=0.9), zorder=2)
    ax.set_xticks(range(3), [GROUP_EN[g] for g in GROUPS])
    ax.set_ylim(*ylim); ax.set_title(title)
    ax.grid(axis="y", color="#E5E5E5", lw=0.6, zorder=0)
    if col == "F1-Macro":
        ax.axhline(1 / 3, ls="--", color="#555555", lw=1.0, zorder=3)
        ax.text(-0.48, 1 / 3 + 0.007, "stratified random", fontsize=7.5, color="#555555", va="bottom")
    if col == "AUC":
        ax.axhline(0.5, ls="--", color="#555555", lw=1.0, zorder=3)
        ax.text(-0.48, 0.5 + 0.003, "chance", fontsize=7.5, color="#555555", va="bottom")
fig.legend([Patch(color=COLORS[k]) for k in KEYS], [NAMES[k] for k in KEYS], loc="upper center", ncol=4,
           bbox_to_anchor=(0.5, 1.03), fontsize=9, handlelength=2.2, columnspacing=2.5)
fig.tight_layout(h_pad=1.6, w_pad=1.6, rect=(0, 0, 1, 0.96))
save(fig, "fig07_comparison.png")

# ---------------------------------------------------------------- Fig. 8 · confusion matrices (mean of seeds)
fig, axes = plt.subplots(3, 4, figsize=(6.85, 5.15), sharex=True, sharey=True)
for r_, g in enumerate(GROUPS):
    p, y = DATA[g]
    for c_, k in enumerate(KEYS):
        cm = np.mean([confusion_matrix(y, p[k][s].argmax(1), labels=[0, 1, 2], normalize="true")
                      for s in range(p[k].shape[0])], axis=0)
        ax = axes[r_, c_]
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if cm[i, j] > 0.56 else "black")
        ax.set_xticks(range(3), CLASSES_SHORT); ax.set_yticks(range(3), CLASSES_SHORT)
        if r_ == 0:
            ax.set_title(NAMES[k], fontsize=10)
        if c_ == 0:
            ax.set_ylabel(f"{GROUP_EN[g]} ({AGES[g]} years)\nTrue class", fontsize=8.5)
        if r_ == 2:
            ax.set_xlabel("Predicted class", fontsize=8.5)
fig.subplots_adjust(wspace=0.08, hspace=0.18, right=0.9)
cax = fig.add_axes([0.925, 0.3, 0.015, 0.42])
fig.colorbar(im, cax=cax).ax.tick_params(labelsize=7.5)
save(fig, "fig08_confusion.png")

# ---------------------------------------------------------------- Fig. 9 · macro ROC curves
grid = np.linspace(0, 1, 1001)
fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.75), sharey=True)
for ax, g in zip(axes, GROUPS):
    p, y = DATA[g]
    for k in KEYS:
        curves = []
        for s in range(p[k].shape[0]):
            for c in range(3):
                fpr, tpr, _ = roc_curve(y == c, p[k][s][:, c])
                curves.append(np.interp(grid, fpr, tpr))
        tpr = np.mean(curves, axis=0); tpr[0] = 0
        ax.plot(grid, tpr, color=COLORS[k], lw=1.3, label=f"{NAMES[k]} ({MET[g, k]['AUC'].mean():.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=0.8)
    ax.set_title(f"{GROUP_EN[g]} ({AGES[g]} years)")
    ax.set_xlabel("False positive rate")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.grid(color="#E5E5E5", lw=0.6)
    ax.legend(title="Macro AUC", loc="lower right", fontsize=7.5, title_fontsize=7.5, handlelength=1.6)
axes[0].set_ylabel("True positive rate")
fig.tight_layout(w_pad=1.0)
save(fig, "fig09_roc.png")

# ---------------------------------------------------------------- Fig. 10 · Copeland ranking
cop = pd.read_csv(REP / "copeland_global.csv").sort_values("copeland_score_global")
key_of = {"Ensemble": "ensemble", "FT-Transformer": "ft_transformer", "MLP-MA": "mlp_ma", "TabNet": "tabnet"}
ks = [key_of[c] for c in cop.classifier]
vals = cop.copeland_score_global.to_numpy()
fig, ax = plt.subplots(figsize=(3.3, 1.85))
ax.barh([NAMES[k] for k in ks], vals, color=[COLORS[k] for k in ks], height=0.6, zorder=2)
for i, v in enumerate(vals):
    ax.text(v + (0.25 if v > 0 else -0.25), i, f"{v:+d}".replace("-", "−"), va="center",
            ha="left" if v > 0 else "right", fontsize=9)
ax.axvline(0, color="black", lw=0.8, zorder=3)
ax.set_xlim(-10.5, 10.5)
ax.set_xticks([-10, -5, 0, 5, 10], ["−10", "−5", "0", "5", "10"])
ax.grid(axis="x", color="#E5E5E5", lw=0.6, zorder=0)
ax.set_xlabel("Cumulative Copeland score (range −9 to +9)")
save(fig, "fig10_copeland.png")

# ---------------------------------------------------------------- Fig. 11 · learning curves (stage A)
fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.0))
for r_, g in enumerate(GROUPS):
    for c_, m in enumerate(MODELS):
        res = json.loads((RES / f"{g}_{m}" / "results.json").read_text(encoding="utf-8"))
        lc, best = res["learning_curves"], res["mejor_epoca_etapa_a"]
        ep = np.arange(1, len(lc["train_loss"]) + 1)
        ax = axes[r_, c_]
        ax.plot(ep, lc["train_loss"], color="#2C76B5", lw=1.1)
        ax.plot(ep, lc["val_loss"], color="#D6604D", lw=1.1)
        ax.axvline(best, color="#333333", ls=":", lw=1.1)
        ax.text(best + 0.4, 1.0, str(best), transform=ax.get_xaxis_transform(), va="top", fontsize=8)
        ax.grid(color="#E5E5E5", lw=0.6)
        ax.tick_params(labelsize=8)
        if r_ == 0:
            ax.set_title(NAMES[m])
        if c_ == 0:
            ax.set_ylabel(f"{GROUP_EN[g]}\nWeighted loss", fontsize=8.5)
        if r_ == 2:
            ax.set_xlabel("Epoch")
fig.legend([Line2D([], [], color="#2C76B5"), Line2D([], [], color="#D6604D"),
            Line2D([], [], color="#333333", ls=":")],
           ["Training (70%)", "Validation (15%)", "Best epoch"], loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.02), fontsize=8.5, columnspacing=2.5)
fig.tight_layout(h_pad=1.2, w_pad=1.0, rect=(0, 0, 1, 0.965))
save(fig, "fig11_learning_curves.png")

# ---------------------------------------------------------------- Fig. 12 · permutation importance
MARK = {"mlp_ma": "o", "tabnet": "s", "ft_transformer": "^"}
XTICKS = {"ninos": [0, 0.05, 0.10], "jovenes": [0, 0.05, 0.10], "adultos": [0, 0.02, 0.04]}
LABEL_W, PLOT_W = 0.205, 0.11                           # fractions of the width: labels / bars of each panel
fig = plt.figure(figsize=(7.1, 3.3))
axes = [fig.add_axes([i * 0.322 + LABEL_W, 0.14, PLOT_W, 0.69]) for i in range(3)]
for ax, g in zip(axes, GROUPS):
    imp = {}
    for m in MODELS:
        z = np.load(RES / f"{g}_{m}" / "interpretabilidad.npz")
        imp[m] = pd.Series(z["permutation"], index=z["feature_names"])
    df = pd.DataFrame(imp)
    df["mean"] = df[MODELS].mean(axis=1)
    top = df.sort_values("mean", ascending=False).head(10)
    yy = np.arange(len(top))
    ax.barh(yy, top["mean"], color="#9BA8BA", height=0.62, zorder=1)
    for m in MODELS:
        ax.scatter(top[m], yy, marker=MARK[m], s=16, color=COLORS[m], zorder=3, lw=0)
    ax.set_yticks(yy, [FEATURES_EN[f] for f in top.index], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xticks(XTICKS[g], [f"{t:g}" for t in XTICKS[g]])
    ax.set_xlim(min(-0.004, top[MODELS].min().min() - 0.003), max(XTICKS[g][-1], top[MODELS].max().max()) * 1.06)
    ax.grid(axis="x", color="#E5E5E5", lw=0.6, zorder=0)
    ax.set_title(f"{GROUP_EN[g]} ({AGES[g]} years)", x=0.5 - LABEL_W / PLOT_W / 2.6)
    ax.set_xlabel("Drop in F1-Macro")
    ax.tick_params(axis="x", labelsize=8)
fig.legend([Line2D([], [], marker=MARK[m], color=COLORS[m], ls="", markersize=5) for m in MODELS],
           [NAMES[m] for m in MODELS], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=9,
           columnspacing=3)
save(fig, "fig12_importance.png", tight=False)           # a tight box cut the last x label
