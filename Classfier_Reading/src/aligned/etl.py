"""
ETL alineado al manuscrito (seccion III-A y III-B).

Flujo:
    1. Consolidacion de los tres CSV crudos en un unico marco.
    2. Segmentacion etaria segun el articulo (Ninos 3-13, Jovenes 14-17,
       Adultos 18+), recalculada desde `edad_total`.
    3. Construccion del target y descarte de filas sin target.
    4. Limpieza: filas con mas del 20% de valores faltantes.
    5. Indicadores derivados.
    6. Seleccion de predictores libres de fuga.
    7. Split estratificado 70 / 15 / 15.
    8. Ajuste SOLO sobre train de: imputacion, Label Encoding,
       Min-Max Scaling, umbral de varianza, correlacion y recorte a 71.
    9. SMOTE unicamente sobre train.
   10. Verificaciones duras: sin fuga, y esquema identico en los tres
       splits (el bug de 398 vs 387 columnas que detecto la auditoria).

Todos los artefactos quedan en `data/aligned/<grupo>/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import features as F
from . import variables as V
from .target import build_target, target_report


# =====================================================================
# Carga y consolidacion
# =====================================================================
def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Concatena los CSV crudos y normaliza los nombres de columna."""
    frames = []
    for name in ("ninos", "jovenes", "adultos"):
        path = raw_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No se encontro {path}")
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.replace("﻿", "").strip() for c in df.columns]
        df["_src"] = name
        frames.append(df)

    consolidated = pd.concat(frames, ignore_index=True, sort=False)
    if "edad_total" not in consolidated.columns:
        raise KeyError("Falta `edad_total`; es necesaria para la segmentacion etaria.")
    return consolidated


def segment_by_age(df: pd.DataFrame, edad_min: int, edad_max: int) -> pd.DataFrame:
    edad = pd.to_numeric(df["edad_total"], errors="coerce")
    mask = edad.between(edad_min, edad_max, inclusive="both")
    return df.loc[mask].copy()


# =====================================================================
# Transformadores ajustados sobre train
# =====================================================================
class AlignedPreprocessor:
    """
    Label Encoding para categoricas + Min-Max Scaling para numericas,
    tal como declara la seccion III-A. Ambos se ajustan exclusivamente
    con el conjunto de entrenamiento.

    El resultado deja TODAS las columnas en [0, 1], lo que permite
    aplicar despues el umbral de varianza sobre una escala comun.
    """

    def __init__(self, numeric: List[str], categorical: List[str], fill_cat: int = -1):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.fill_cat = fill_cat
        self.medians_: Dict[str, float] = {}
        self.categories_: Dict[str, Dict[float, int]] = {}
        self.ranges_: Dict[str, Tuple[float, float]] = {}
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame) -> "AlignedPreprocessor":
        for col in self.numeric:
            s = pd.to_numeric(X[col], errors="coerce")
            self.medians_[col] = float(s.median()) if s.notna().any() else 0.0
            filled = s.fillna(self.medians_[col])
            self.ranges_[col] = (float(filled.min()), float(filled.max()))

        for col in self.categorical:
            s = pd.to_numeric(X[col], errors="coerce").fillna(self.fill_cat)
            cats = sorted(s.unique().tolist())
            self.categories_[col] = {v: i for i, v in enumerate(cats)}
            self.ranges_[col] = (0.0, float(max(len(cats) - 1, 1)))

        self.columns_ = self.numeric + self.categorical
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in self.numeric:
            s = pd.to_numeric(X[col], errors="coerce").fillna(self.medians_[col])
            lo, hi = self.ranges_[col]
            out[col] = (s - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=X.index)

        for col in self.categorical:
            s = pd.to_numeric(X[col], errors="coerce").fillna(self.fill_cat)
            mapping = self.categories_[col]
            # Categoria no vista en train -> 0 (primera categoria conocida).
            codes = s.map(mapping).fillna(0).astype("float64")
            lo, hi = self.ranges_[col]
            out[col] = codes / hi if hi > 0 else codes

        return pd.DataFrame(out, index=X.index)[self.columns_].clip(0.0, 1.0)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


# =====================================================================
# Balanceo
# =====================================================================
def apply_smote(
    X: pd.DataFrame, y: pd.Series, k_neighbors: int, min_class_count: int, seed: int
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """SMOTE solo sobre train. Degrada con elegancia si una clase es minuscula."""
    counts = y.value_counts()
    smallest = int(counts.min())

    if smallest < min_class_count:
        return X, y, {
            "applied": False,
            "reason": (
                f"La clase minoritaria tiene {smallest} muestras "
                f"(< min_class_count={min_class_count}). Se omite SMOTE y "
                f"el entrenamiento se apoya en class_weights."
            ),
            "before": counts.to_dict(),
        }

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        return X, y, {"applied": False, "reason": "imbalanced-learn no instalado."}

    k = min(k_neighbors, smallest - 1)
    sm = SMOTE(random_state=seed, k_neighbors=k)
    Xr, yr = sm.fit_resample(X, y)
    Xr = pd.DataFrame(Xr, columns=X.columns)
    yr = pd.Series(yr, name=y.name)
    return Xr, yr, {
        "applied": True,
        "k_neighbors": int(k),
        "before": counts.to_dict(),
        "after": yr.value_counts().to_dict(),
    }


# =====================================================================
# Pipeline por grupo
# =====================================================================
def run_group(cfg: dict, group: str, raw: pd.DataFrame, root: Path,
              raw_all: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    gcfg = cfg["groups"][group]
    pre = cfg["preprocessing"]
    fcfg = cfg["features"]
    scfg = cfg["split"]
    bcfg = cfg["balancing"]
    seed = cfg["seed"]
    class_names = cfg["class_names"]

    out_dir = root / cfg["paths"]["processed_dir"] / group
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, object] = {"grupo": group, "seed": seed}

    # --- 2. Segmentacion etaria ------------------------------------
    df = segment_by_age(raw, gcfg["edad_min"], gcfg["edad_max"])
    report["segmentacion"] = {
        "rango_edad": [gcfg["edad_min"], gcfg["edad_max"]],
        "n_registros": int(len(df)),
        "procedencia_csv": df["_src"].value_counts().to_dict(),
    }

    # --- 3. Target --------------------------------------------------
    y_raw = build_target(df, gcfg["target_source"])
    report["target"] = {"fuente": gcfg["target_source"], **target_report(y_raw, class_names)}

    valid = y_raw.notna()
    df, y_raw = df.loc[valid].copy(), y_raw.loc[valid].astype(int)

    # --- 5. Indicadores derivados (antes de descartar administrativas)
    derived, derived_report = F.build_derived(df)

    # Contexto del hogar: agregados de los adultos del mismo hogar. Se
    # calcula sobre TODA la encuesta, no solo sobre el grupo etario, para
    # que un niño herede el perfil sociocultural de sus adultos.
    source_all = raw_all if raw_all is not None else raw
    hh = F.household_context(source_all, df.index)
    derived_report["contexto_hogar"] = {
        "columnas": list(hh.columns),
        "clave_hogar": F.HOUSEHOLD_KEY,
        "motivo": (
            "La ENL solo aplica el capitulo 300 (lengua materna, etnia, "
            "nivel educativo alcanzado, ocupacion, ingreso) a personas de "
            "18+. Sin este bloque los grupos de 3-17 quedan sin variables "
            "socioculturales propias."
        ),
        "cobertura": (
            round(float(hh.notna().any(axis=1).mean()), 4) if len(hh) else 0.0
        ),
    }
    report["indicadores_derivados"] = derived_report

    # --- 6. Universo de predictores sin fuga ------------------------
    base_features = V.build_feature_list(df.columns)
    V.assert_no_leakage(base_features)

    X_raw = pd.concat([df[base_features], derived, hh], axis=1)
    X_raw = X_raw.loc[:, ~X_raw.columns.duplicated()]

    # --- 4. Limpieza de filas con >20% de faltantes -----------------
    missing_ratio = X_raw.isna().mean(axis=1)
    keep_rows = missing_ratio <= pre["max_missing_row_ratio"]
    report["limpieza"] = {
        "umbral_faltantes_fila": pre["max_missing_row_ratio"],
        "n_antes": int(len(X_raw)),
        "n_despues": int(keep_rows.sum()),
        "n_removidos": int((~keep_rows).sum()),
    }
    X_raw, y_raw = X_raw.loc[keep_rows], y_raw.loc[keep_rows]

    # Columnas totalmente vacias: no aportan y rompen el escalado.
    all_nan = [c for c in X_raw.columns if X_raw[c].isna().all()]
    X_raw = X_raw.drop(columns=all_nan)

    # --- 7. Split estratificado 70 / 15 / 15 ------------------------
    strat = y_raw if scfg["stratify"] else None
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_raw, y_raw,
        test_size=scfg["val_size"] + scfg["test_size"],
        random_state=seed, stratify=strat,
    )
    rel = scfg["test_size"] / (scfg["val_size"] + scfg["test_size"])
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=rel, random_state=seed,
        stratify=y_tmp if scfg["stratify"] else None,
    )

    # --- 8. Ajuste de transformadores SOLO con train ----------------
    split_cols = V.classify_columns(list(X_raw.columns))
    prep = AlignedPreprocessor(
        numeric=[c for c in split_cols["numeric"] if c in X_raw.columns]
                + [c for c in X_raw.columns
                   if (c.startswith("idx_") or c.startswith("hh_"))
                   and c not in split_cols["numeric"]],
        categorical=[c for c in split_cols["categorical"]
                     if not (c.startswith("idx_") or c.startswith("hh_"))],
        fill_cat=pre["missing_categorical_fill"],
    )
    Xtr = prep.fit_transform(X_tr)
    Xval, Xte = prep.transform(X_val), prep.transform(X_te)

    # Umbral de varianza -> correlacion -> recorte a 71, ajustado en train.
    Xtr, dropped_var = F.drop_low_variance(Xtr, fcfg["variance_threshold"])
    Xtr, dropped_corr = F.drop_correlated(Xtr, fcfg["correlation_threshold"])
    Xtr, selected = F.rank_to_n(
        Xtr, y_tr, fcfg["target_n_features"], seed, fcfg["ranking_method"]
    )
    Xval, Xte = Xval[selected], Xte[selected]

    report["seleccion_features"] = {
        "universo_inicial": int(len(X_raw.columns)),
        "descartadas_varianza": len(dropped_var),
        "descartadas_correlacion": len(dropped_corr),
        "objetivo_articulo": fcfg["target_n_features"],
        "n_final": len(selected),
        "coincide_con_articulo": len(selected) == fcfg["target_n_features"],
        "features": selected,
        "dimensiones": pd.Series([V.dimension_of(c) for c in selected])
                         .value_counts().to_dict(),
    }

    # --- Verificaciones duras ---------------------------------------
    V.assert_no_leakage(selected)
    if not (list(Xtr.columns) == list(Xval.columns) == list(Xte.columns)):
        raise AssertionError(
            "Esquema inconsistente entre splits: "
            f"train={Xtr.shape[1]}, val={Xval.shape[1]}, test={Xte.shape[1]}"
        )
    for name, part in (("train", Xtr), ("val", Xval), ("test", Xte)):
        if part.isna().any().any():
            raise AssertionError(f"Quedaron NaN en {name} tras el preprocesamiento.")

    # --- 9. SMOTE solo en train -------------------------------------
    Xtr_bal, ytr_bal, smote_report = apply_smote(
        Xtr, y_tr, bcfg["k_neighbors"], bcfg["min_class_count"], seed
    )
    report["balanceo"] = smote_report

    # --- 10. Persistencia -------------------------------------------
    for name, Xp, yp in (
        ("train", Xtr_bal, ytr_bal),
        ("val", Xval, y_val),
        ("test", Xte, y_te),
    ):
        frame = Xp.copy()
        frame["target"] = np.asarray(yp)
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)

    import joblib
    joblib.dump(
        {"preprocessor": prep, "selected_features": selected},
        out_dir / "preprocessor.pkl",
    )

    report["splits"] = {
        "train": {"n": int(len(Xtr_bal)), "dist": pd.Series(ytr_bal).value_counts().sort_index().to_dict()},
        "val": {"n": int(len(Xval)), "dist": y_val.value_counts().sort_index().to_dict()},
        "test": {"n": int(len(Xte)), "dist": y_te.value_counts().sort_index().to_dict()},
    }
    report["n_features"] = len(selected)

    with open(out_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    return report


def run_etl(cfg: dict, root: Path) -> Dict[str, dict]:
    """Ejecuta el ETL para los tres grupos etarios."""
    raw = load_raw(root / cfg["paths"]["raw_dir"])
    reports = {}
    for group in cfg["groups"]:
        print(f"\n{'=' * 66}\nETL · {group.upper()}\n{'=' * 66}")
        rep = run_group(cfg, group, raw, root, raw_all=raw)
        reports[group] = rep
        sel = rep["seleccion_features"]
        print(f"  registros con target valido : {rep['target']['n_valid']:,}")
        print(f"  distribucion                : {rep['target']['counts']}")
        print(f"  universo -> final           : {sel['universo_inicial']} -> {sel['n_final']} features")
        print(f"  train/val/test              : "
              f"{rep['splits']['train']['n']:,} / {rep['splits']['val']['n']:,} / {rep['splits']['test']['n']:,}")
        print(f"  SMOTE                       : {rep['balanceo'].get('applied')}")
    return reports
