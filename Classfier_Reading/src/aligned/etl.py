"""
ETL alineado al manuscrito (seccion III-A y III-B).

Flujo:
    1. Consolidacion de los tres CSV crudos en un unico marco, con
       armonizacion de su codificacion (ver `harmonize_source`).
    2. Segmentacion etaria segun el articulo (Ninos 3-13, Jovenes 14-17,
       Adultos 18-64), recalculada desde `edad_total`. Los capitulos 300 y
       400 de la ENL solo se aplican de 18 a 64 años: nadie mayor tiene la
       clase de adultos.
    3. Construccion del target y descarte de filas sin target.
    4. Limpieza: filas con mas del 20% de valores faltantes.
    5. Indicadores derivados.
    6. Seleccion de predictores libres de fuga.
    7. Split estratificado 70 / 15 / 15.
    8. Ajuste SOLO sobre train de: imputacion, Label Encoding,
       Min-Max Scaling, umbral de varianza, correlacion y recorte a 71.
    9. Verificaciones duras: sin fuga, y esquema identico en los tres
       splits (el bug de 398 vs 387 columnas que detecto la auditoria).

El balanceo de clases YA NO se aplica aqui. Aplicar SMOTE al 70% antes de
la validacion cruzada deja muestras sinteticas en los folds de
validacion, fabricadas a partir de los folds de entrenamiento, y la CV
sale optimista. Ahora se hace dentro de cada ajuste, solo sobre la parte
de entrenamiento (`pipeline.balance_training_set`).

Todos los artefactos quedan en `data/aligned/<grupo>/`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import features as F
from . import variables as V
from .target import build_target, target_report

# Version de los datos procesados. El entrenamiento se niega a usar datos
# de una version anterior (ver pipeline.load_group).
#   2 · ingreso con log1p, alfabetismo del hogar corregido, sin P305,
#       adultos 18-64 y codificacion para los embeddings de las nominales.
ETL_VERSION = 2


# =====================================================================
# Carga y consolidacion
# =====================================================================
def harmonize_source(df: pd.DataFrame, blank_means_not_applicable: bool
                     ) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Unifica la codificacion de un CSV crudo con la de los demas.

    Dos defectos de `jovenes.csv` (16-27 años) que, sin esta correccion,
    hacian que el filtro de ">20 % de faltantes" descartara el 100 % de
    sus filas, de modo que ningun joven de 16-17 ni adulto de 18-27
    llegaba al modelo:

      a) Columnas duplicadas por un merge de capitulos (`P201_x`,
         `P201_y`, ...): el valor real quedo repartido entre la columna
         base y sus sufijos. Se unifican en la columna base.
      b) "No aplica" codificado como celda vacia, mientras `ninos.csv` y
         `adultos.csv` usan 0, la convencion de la ENL. Se verifico sobre
         edades vecinas (16-17 vs 14-15; 18-27 vs 28-35): la proporcion de
         vacios en jovenes.csv coincide con la de ceros en los otros
         archivos (P208, P506, P513: 100 % vs 100 %; P402/P408/P301/P306:
         10-11 % vs 9-10 %). Los vacios se convierten en 0.
    """
    info: Dict[str, object] = {}

    suffixed = [c for c in df.columns if re.search(r"_(x|y)$", c)]
    bases = sorted({c[:-2] for c in suffixed})
    for base in bases:
        parts = [c for c in (base, f"{base}_x", f"{base}_y") if c in df.columns]
        merged = df[parts[0]]
        for extra in parts[1:]:
            merged = merged.combine_first(df[extra])
        df[base] = merged
    df = df.drop(columns=suffixed)
    info["columnas_merge_unificadas"] = bases

    if blank_means_not_applicable:
        numeric = [c for c in df.columns
                   if c not in ("edad_total", "_src")
                   and pd.api.types.is_numeric_dtype(df[c])]
        info["celdas_vacias_convertidas_a_0"] = int(df[numeric].isna().sum().sum())
        df[numeric] = df[numeric].fillna(0)
    return df, info


def load_raw(raw_dir: Path, blank_sources: Optional[List[str]] = None,
             report: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    """Concatena los CSV crudos, armoniza su codificacion y normaliza nombres."""
    blank_sources = list(blank_sources or [])
    frames = []
    for name in ("ninos", "jovenes", "adultos"):
        path = raw_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No se encontro {path}")
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.replace("﻿", "").strip() for c in df.columns]
        df, info = harmonize_source(df, name in blank_sources)
        if report is not None:
            report[name] = {"filas": int(len(df)), **info}
        frames.append(df.assign(_src=name))

    consolidated = pd.concat(frames, ignore_index=True, sort=False)
    if "edad_total" not in consolidated.columns:
        raise KeyError("Falta `edad_total`; es necesaria para la segmentacion etaria.")
    return consolidated


def _has_text(s: pd.Series) -> bool:
    """True si la columna trae respuestas alfanumericas en lugar de codigos."""
    if s.dtype != object:
        return False
    t = s.dropna().astype(str).str.strip()
    t = t[t != ""]
    return bool(pd.to_numeric(t, errors="coerce").isna().any())


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
    aplicar despues el umbral de varianza sobre una escala comun. Una
    categorica con k categorias queda en {0, 1/(k-1), ..., 1}; los modelos
    recuperan el codigo entero (round(x * (k-1))) para sus embeddings.

    `log_columns`: numericas muy asimetricas (montos en soles) que pasan
    por log1p antes del escalado. Sin eso, el ingreso (mediana ~S/ 1 000,
    maximo S/ 35 000) quedaba casi entero cerca de 0, con varianza 0,001,
    y el umbral de varianza lo descartaba: nunca llegaba al modelo.
    """

    def __init__(self, numeric: List[str], categorical: List[str], fill_cat: int = -1,
                 log_columns: Optional[List[str]] = None):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.fill_cat = fill_cat
        self.log_columns = [c for c in (log_columns or []) if c in self.numeric]
        self.medians_: Dict[str, float] = {}
        self.categories_: Dict[str, Dict[float, int]] = {}
        self.ranges_: Dict[str, Tuple[float, float]] = {}
        self.columns_: List[str] = []

    def _numeric(self, X: pd.DataFrame, col: str) -> pd.Series:
        s = pd.to_numeric(X[col], errors="coerce")
        if col in getattr(self, "log_columns", []):
            s = np.log1p(s.clip(lower=0))
        return s

    def n_categories(self, col: str) -> int:
        return len(self.categories_[col])

    def fit(self, X: pd.DataFrame) -> "AlignedPreprocessor":
        for col in self.numeric:
            s = self._numeric(X, col)
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
            s = self._numeric(X, col).fillna(self.medians_[col])
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
# Pipeline por grupo
# =====================================================================
def run_group(cfg: dict, group: str, raw: pd.DataFrame, root: Path,
              raw_all: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    gcfg = cfg["groups"][group]
    pre = cfg["preprocessing"]
    fcfg = cfg["features"]
    scfg = cfg["split"]
    seed = cfg["seed"]
    class_names = cfg["class_names"]

    out_dir = root / cfg["paths"]["processed_dir"] / group
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, object] = {"grupo": group, "seed": seed, "etl_version": ETL_VERSION}

    # --- 2. Segmentacion etaria ------------------------------------
    df = segment_by_age(raw, gcfg["edad_min"], gcfg["edad_max"])
    edades = pd.to_numeric(df["edad_total"], errors="coerce")
    report["segmentacion"] = {
        "rango_edad": [gcfg["edad_min"], gcfg["edad_max"]],
        "edad_observada": [int(edades.min()), int(edades.max())],
        "n_registros": int(len(df)),
        "procedencia_csv": df["_src"].value_counts().to_dict(),
    }

    # --- 3. Target --------------------------------------------------
    y_raw = build_target(df, gcfg["target_source"])
    report["target"] = {"fuente": gcfg["target_source"], **target_report(y_raw, class_names)}

    valid = y_raw.notna()
    df, y_raw = df.loc[valid].copy(), y_raw.loc[valid].astype(int)
    edades_validas = pd.to_numeric(df["edad_total"], errors="coerce")
    report["segmentacion"]["edad_con_clase"] = (
        [int(edades_validas.min()), int(edades_validas.max())] if len(df) else None)

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

    # Guarda: una columna con respuestas de texto se volveria "sin dato"
    # (-1) al codificarla, como paso con P305. Se excluye y se informa.
    text_cols = [c for c in base_features if _has_text(df[c])]
    if text_cols:
        print(f"  AVISO · columnas con texto excluidas: {text_cols}")
        base_features = [c for c in base_features if c not in text_cols]
    report["columnas_texto_excluidas"] = text_cols

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
        numeric=split_cols["numeric"],
        categorical=split_cols["categorical"],
        fill_cat=pre["missing_categorical_fill"],
        log_columns=pre.get("log_transform", []),
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

    # Codificacion que necesitan los modelos: n.º de categorias de cada
    # categorica (para recuperar el codigo entero) y cuales son nominales,
    # que reciben un embedding en lugar de entrar como escalar.
    n_cat = {c: prep.n_categories(c) for c in selected if c in prep.categorical}
    report["codificacion"] = {
        "categorias_por_variable": n_cat,
        "embeddings": {c: n for c, n in n_cat.items() if c in V.NOMINAL_COLUMNS and n >= 3},
        "log1p": [c for c in prep.log_columns if c in selected],
        "nota": ("Label Encoding: una categorica con k categorias se guarda como "
                 "codigo/(k-1) en [0, 1]. Las nominales de `embeddings` entran a los "
                 "modelos como embedding (round(x*(k-1)) -> vector aprendido)."),
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

    # --- 9. Balanceo: se difiere al entrenamiento ------------------
    report["balanceo"] = {
        "metodo": cfg["balancing"]["method"],
        "donde": ("Dentro de cada ajuste, solo sobre la parte de entrenamiento "
                  "(cada fold de la CV y el reentrenamiento). Los splits "
                  "guardados conservan la distribucion natural de clases."),
    }

    # --- 10. Persistencia -------------------------------------------
    for name, Xp, yp in (
        ("train", Xtr, y_tr),
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
        "train": {"n": int(len(Xtr)), "dist": y_tr.value_counts().sort_index().to_dict()},
        "val": {"n": int(len(Xval)), "dist": y_val.value_counts().sort_index().to_dict()},
        "test": {"n": int(len(Xte)), "dist": y_te.value_counts().sort_index().to_dict()},
    }
    report["n_features"] = len(selected)

    with open(out_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    return report


def run_etl(cfg: dict, root: Path) -> Dict[str, dict]:
    """Ejecuta el ETL para los tres grupos etarios."""
    harm = cfg.get("raw_harmonization", {})
    raw_report: Dict[str, object] = {}
    raw = load_raw(root / cfg["paths"]["raw_dir"],
                   blank_sources=harm.get("blank_means_not_applicable", []),
                   report=raw_report)
    for name, info in raw_report.items():
        extra = []
        if info.get("columnas_merge_unificadas"):
            extra.append(f"{len(info['columnas_merge_unificadas'])} columnas _x/_y unificadas")
        if info.get("celdas_vacias_convertidas_a_0"):
            extra.append(f"{info['celdas_vacias_convertidas_a_0']:,} vacios -> 0 (no aplica)")
        print(f"  {name}.csv: {info['filas']:,} filas" + (f" · {' · '.join(extra)}" if extra else ""))

    reports = {}
    for group in cfg["groups"]:
        print(f"\n{'=' * 66}\nETL · {group.upper()}\n{'=' * 66}")
        rep = run_group(cfg, group, raw, root, raw_all=raw)
        rep["armonizacion_csv"] = raw_report
        with open(root / cfg["paths"]["processed_dir"] / group / "config.json",
                  "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2, ensure_ascii=False, default=str)
        reports[group] = rep
        sel, seg = rep["seleccion_features"], rep["segmentacion"]
        edades = seg.get("edad_con_clase") or seg["edad_observada"]
        print(f"  edades con clase            : {edades[0]}-{edades[1]} "
              f"· origen {seg['procedencia_csv']}")
        print(f"  registros con target valido : {rep['target']['n_valid']:,}")
        print(f"  distribucion                : {rep['target']['counts']}")
        print(f"  eliminados por >20% faltante: {rep['limpieza']['n_removidos']:,}")
        print(f"  universo -> final           : {sel['universo_inicial']} -> {sel['n_final']} features")
        emb = rep["codificacion"]["embeddings"]
        print(f"  nominales con embedding     : {len(emb)} "
              f"({', '.join(f'{c}:{n}' for c, n in emb.items())})")
        print(f"  train/val/test              : "
              f"{rep['splits']['train']['n']:,} / {rep['splits']['val']['n']:,} / {rep['splits']['test']['n']:,}")
    return reports
