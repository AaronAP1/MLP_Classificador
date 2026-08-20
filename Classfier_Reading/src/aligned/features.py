"""
Feature engineering y seleccion de variables (seccion III-B del articulo).

Cubre las tres etapas declaradas:
    (i)   seleccion de variables relevantes
    (ii)  creacion de indicadores derivados
    (iii) consolidacion de la matriz final de 71 variables

--------------------------------------------------------------------
DESVIACION DOCUMENTADA respecto al articulo
--------------------------------------------------------------------
El articulo lista cuatro indicadores derivados. Tres se construyen aqui.
El cuarto, "Frecuencia Lectora Semanal (numero de dias dedicados a
lectura)", NO se construye como predictor y no puede construirse: la
variable objetivo de este estudio ES la frecuencia lectora. Usarla como
feature es precisamente la fuga que la auditoria encontro en el ETL
anterior y la causa del accuracy 1.0000 en el experimento multi-semilla.
Se deja la funcion `frecuencia_lectora_semanal()` implementada y
claramente marcada, para analisis descriptivo, pero el pipeline nunca la
incorpora a X. Esta decision debe reflejarse en el manuscrito.

"Indice de Acceso Digital": el capitulo 600 de la ENL (equipamiento del
hogar: computador, conexion a internet, dispositivo de lectura digital)
no esta presente en los CSV de `data/unprocessed/`. El indice se
construye con los mejores proxies disponibles (estrato socioeconomico,
grado de urbanidad, modalidad educativa) y se marca como aproximado en
`derived_indicators_report`. Si en el futuro se agregan las columnas
P602_*, `indice_acceso_digital()` las usa automaticamente.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Escalera ordinal de nivel educativo, normalizada a 0-12.
#
# La fuente principal es P306, "cual es el ultimo año o grado de estudios
# y nivel que aprobo" (escala 1-12), que es el nivel educativo alcanzado.
# El ETL auditado nunca la uso: se apoyaba en P310_N, que solo responde
# quien esta estudiando actualmente (4.263 adultos de 51.750).
NIVEL_P306 = {                       # ultimo nivel aprobado (18+)
    1.0: 0,   # sin nivel
    2.0: 1,   # inicial
    3.0: 2,   # primaria incompleta
    4.0: 3,   # primaria completa
    5.0: 4,   # secundaria incompleta
    6.0: 5,   # secundaria completa
    7.0: 3,   # basica especial
    8.0: 6,   # superior no universitaria incompleta
    9.0: 7,   # superior no universitaria completa
    10.0: 8,  # superior universitaria incompleta
    11.0: 9,  # superior universitaria completa
    12.0: 10,  # maestria / doctorado
}
NIVEL_P310 = {1.0: 2, 2.0: 4, 3.0: 2, 4.0: 6, 5.0: 8, 6.0: 10}   # nivel en curso
NIVEL_P514 = {1.0: 1, 2.0: 2, 3.0: 4, 4.0: 2, 5.0: 6, 6.0: 8, 8.0: 0}  # 0-17

LENGUA_INDIGENA = set(range(1, 10))     # codigos 1-9 en P301 / P513A
LENGUA_CASTELLANO = {10}
LENGUA_EXTRANJERA = {11, 12, 13}

# CONG (conglomerado) es imprescindible: sin el, VIVIENDA se repite entre
# conglomerados y los "hogares" resultantes llegan a 244 miembros.
HOUSEHOLD_KEY = ["UBIGEO", "CONG", "VIVIENDA", "HOGAR"]


# =====================================================================
# (ii) Indicadores derivados
# =====================================================================
def nivel_educativo(df: pd.DataFrame) -> pd.Series:
    """
    Nivel educativo individual en escala ordinal 0-10.

    Prioridad: P306 (ultimo nivel aprobado) > P310_N (nivel en curso,
    adultos) > P514_1 (nivel en curso, 0-17).
    """
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col, mapping in (("P306", NIVEL_P306),
                         ("P310_N", NIVEL_P310),
                         ("P514_1", NIVEL_P514)):
        if col in df.columns:
            out = out.fillna(df[col].replace(0, np.nan).map(mapping))
    return out


def indice_acceso_digital(df: pd.DataFrame) -> Tuple[pd.Series, bool]:
    """
    "Indice de Acceso Digital (uso de dispositivos, conectividad, acceso
    a internet)", normalizado a [0, 1].

    Returns:
        (indice, exacto) donde `exacto` indica si se pudo usar el
        capitulo 600 real en lugar de proxies.
    """
    equip = [c for c in ("P602_3", "P602_5", "P602_7", "P602_10") if c in df.columns]
    if equip:
        # Ruta exacta: 1 = si, 2 = no en la ENL.
        score = sum((df[c].replace(0, np.nan) == 1).astype("float64") for c in equip)
        return (score / len(equip)).astype("float64"), True

    # Ruta aproximada.
    parts: List[pd.Series] = []
    if "ESTRATOSOCIO" in df.columns:
        # 1 alto ... 5 bajo  ->  invertido y normalizado
        s = df["ESTRATOSOCIO"].replace(0, np.nan).clip(1, 5)
        parts.append((5 - s) / 4)
    if "ESTRATO" in df.columns:
        # 1 = mayor concentracion urbana ... 8 = rural disperso
        s = df["ESTRATO"].replace(0, np.nan).clip(1, 8)
        parts.append((8 - s) / 7)
    if "P309" in df.columns:
        # asiste a educacion presencial o virtual
        parts.append((df["P309"].replace(0, np.nan) == 1).astype("float64"))

    if not parts:
        return pd.Series(np.nan, index=df.index, dtype="float64"), False
    return pd.concat(parts, axis=1).mean(axis=1).astype("float64"), False


def entorno_linguistico(df: pd.DataFrame) -> pd.DataFrame:
    """
    "Entorno Linguistico (idioma materno y lenguas secundarias)".
    Devuelve dos columnas: tipo de lengua materna y numero de lenguas
    adicionales declaradas.
    """
    materna = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in ("P301", "P513A"):
        if col in df.columns:
            materna = materna.fillna(df[col].replace(0, np.nan))

    def tipo(v: float) -> float:
        if pd.isna(v):
            return np.nan
        v = int(v)
        if v in LENGUA_CASTELLANO:
            return 0.0        # castellano
        if v in LENGUA_INDIGENA:
            return 1.0        # indigena u originaria
        if v in LENGUA_EXTRANJERA:
            return 2.0        # extranjera
        return 3.0            # otra situacion (no escucha/habla, señas)

    habladas = [c for c in df.columns if c.startswith("P302_")]
    n_lenguas = (
        df[habladas].apply(pd.to_numeric, errors="coerce")
        .eq(1).sum(axis=1).astype("float64")
        if habladas else pd.Series(0.0, index=df.index)
    )

    return pd.DataFrame(
        {
            "idx_lengua_materna_tipo": materna.map(tipo),
            "idx_n_lenguas_habladas": n_lenguas,
        },
        index=df.index,
    )


def frecuencia_lectora_semanal(df: pd.DataFrame) -> pd.Series:
    """
    NO USAR COMO PREDICTOR. Se conserva solo para analisis descriptivo.

    Este indicador se deriva de las mismas variables que definen el
    target (P403 / P506) y por lo tanto lo filtra por completo.
    `build_derived()` no lo incorpora nunca a la matriz X.
    """
    dias = pd.Series(np.nan, index=df.index, dtype="float64")
    escala = {1.0: 7.0, 2.0: 3.0, 3.0: 1.0, 4.0: 0.5, 5.0: 0.25, 6.0: 0.1}
    for col in ("P403", "P506"):
        if col in df.columns:
            dias = dias.fillna(df[col].replace(0, np.nan).map(escala))
    return dias


def household_context(df_all: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """
    Contexto sociocultural del hogar, agregado desde los miembros adultos.

    Motivacion: la ENL solo administra el capitulo 300 (lengua materna,
    autoidentificacion etnica, nivel educativo alcanzado, ocupacion,
    ingreso) a las personas de 18 años o mas. Sin este bloque, un niño de
    3-13 años se queda practicamente sin variables socioculturales
    propias y la matriz de predictores colapsa a una docena de columnas
    geograficas y demograficas.

    Aqui cada persona recibe el perfil de los adultos de su hogar, que es
    exactamente la nocion de "Nivel Educativo Familiar" y "Entorno
    Linguistico" que describe la seccion III-B del articulo, extendida a
    las demas dimensiones del capitulo 300.

    Ninguna de estas variables toca el comportamiento lector: se calculan
    sobre columnas que `variables.is_leaky()` considera admisibles.

    Args:
        df_all: TODAS las personas encuestadas (no solo el grupo etario).
        index:  indice de las filas del grupo para el que se construye.
    """
    key = [c for c in HOUSEHOLD_KEY if c in df_all.columns]
    if not key or "edad_total" not in df_all.columns:
        return pd.DataFrame(index=index)

    base = df_all[key].copy()
    base["_edad"] = pd.to_numeric(df_all["edad_total"], errors="coerce")
    base["_nivel"] = nivel_educativo(df_all)
    base["_es_jefe"] = (df_all["P203"].replace(0, np.nan) == 1) if "P203" in df_all else False

    def col(name: str, transform=None) -> pd.Series:
        if name not in df_all.columns:
            return pd.Series(np.nan, index=df_all.index)
        s = df_all[name].replace(0, np.nan)
        return transform(s) if transform else s

    base["_alfabeto"] = col("P307_1", lambda s: (s == 1).astype("float64"))
    base["_lengua_indigena"] = col("P301", lambda s: s.isin(LENGUA_INDIGENA).astype("float64"))
    base["_pueblo_indigena"] = col("P304", lambda s: (s == 1).astype("float64"))
    base["_etnia"] = col("P303")
    base["_trabaja"] = col("P311", lambda s: (s == 1).astype("float64"))
    base["_ingreso"] = pd.to_numeric(col("P314_1"), errors="coerce")
    base["_estudia"] = col("P309", lambda s: (s == 1).astype("float64"))
    lim = [c for c in ("P315_1", "P315_2", "P315_3", "P315_4", "P315_5", "P315_6")
           if c in df_all.columns]
    base["_limitacion"] = (
        (df_all[lim].replace(0, np.nan) == 1).any(axis=1).astype("float64")
        if lim else np.nan
    )

    adultos = base[base["_edad"] >= 18]
    grp = adultos.groupby(key, dropna=False)

    agg = pd.DataFrame({
        "hh_nivel_educ_max": grp["_nivel"].max(),
        "hh_nivel_educ_mean": grp["_nivel"].mean(),
        "hh_alfabetismo": grp["_alfabeto"].mean(),
        "hh_lengua_indigena": grp["_lengua_indigena"].mean(),
        "hh_pueblo_indigena": grp["_pueblo_indigena"].mean(),
        "hh_etnia_jefe": grp["_etnia"].first(),
        "hh_tasa_ocupacion": grp["_trabaja"].mean(),
        "hh_ingreso_mean": grp["_ingreso"].mean(),
        "hh_ingreso_max": grp["_ingreso"].max(),
        "hh_estudia": grp["_estudia"].mean(),
        "hh_limitacion": grp["_limitacion"].max(),
        "hh_n_adultos": grp.size(),
    })

    # Nivel educativo del jefe de hogar, tomado aparte.
    jefes = base[base["_es_jefe"].fillna(False).astype(bool)]
    if len(jefes):
        agg["hh_nivel_educ_jefe"] = jefes.groupby(key, dropna=False)["_nivel"].max()

    # Composicion del hogar, sobre todos los miembros.
    total = base.groupby(key, dropna=False)
    agg["hh_tamano"] = total.size()
    agg["hh_n_menores"] = base.assign(
        _m=(base["_edad"] < 18).astype(float)).groupby(key, dropna=False)["_m"].sum()
    agg["hh_ratio_menores"] = agg["hh_n_menores"] / agg["hh_tamano"].replace(0, np.nan)

    # Proyeccion sobre las filas del grupo.
    rows = df_all.loc[index, key]
    joined = rows.merge(agg, left_on=key, right_index=True, how="left")
    return joined.drop(columns=key).set_index(index)


def build_derived(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Construye el bloque de indicadores derivados que si entra a X.

    Returns:
        (DataFrame de indicadores, reporte para el config JSON)
    """
    acceso, exacto = indice_acceso_digital(df)
    ling = entorno_linguistico(df)

    derived = pd.DataFrame(
        {
            "idx_acceso_digital": acceso,
            "idx_nivel_educativo": nivel_educativo(df),
        },
        index=df.index,
    ).join(ling)

    report = {
        "indicadores_construidos": list(derived.columns),
        "nivel_educativo_fuente": (
            "P306 (ultimo nivel aprobado, escala 1-12) con respaldo en "
            "P310_N y P514_1."
        ),
        "nivel_educativo_familiar": (
            "Se construye en el bloque de contexto del hogar como "
            "hh_nivel_educ_mean / _max / _jefe."
        ),
        "acceso_digital_exacto": bool(exacto),
        "acceso_digital_nota": (
            "Construido con el capitulo 600 (equipamiento del hogar)."
            if exacto else
            "APROXIMADO: el capitulo 600 no esta en data/unprocessed/. "
            "Se usan estrato socioeconomico, grado de urbanidad y "
            "modalidad educativa como proxies."
        ),
        "frecuencia_lectora_semanal": (
            "NO incorporada a X: se deriva del target y provocaria fuga. "
            "Ver docstring de features.py."
        ),
    }
    return derived, report


# =====================================================================
# (i) + (iii) Seleccion y consolidacion a 71 variables
# =====================================================================
def drop_low_variance(X: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    """Umbral de varianza. Se aplica sobre datos ya escalados a [0,1]."""
    variances = X.var(axis=0, ddof=0)
    keep = variances[variances > threshold].index.tolist()
    dropped = [c for c in X.columns if c not in keep]
    return X[keep], dropped


def drop_correlated(X: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    """
    Inspeccion de correlacion: de cada par con |r| por encima del umbral
    se descarta la segunda variable en orden de aparicion.
    """
    if X.shape[1] < 2:
        return X, []
    corr = X.corr().abs().to_numpy()
    cols = list(X.columns)
    dropped: List[str] = []
    keep_mask = np.ones(len(cols), dtype=bool)
    for i in range(len(cols)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(cols)):
            if keep_mask[j] and corr[i, j] > threshold:
                keep_mask[j] = False
                dropped.append(cols[j])
    keep = [c for c, m in zip(cols, keep_mask) if m]
    return X[keep], dropped


def rank_to_n(
    X: pd.DataFrame,
    y: pd.Series,
    n: int,
    seed: int,
    method: str = "mutual_info",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Recorta a exactamente `n` variables (71 segun el articulo).

    Si tras los filtros quedan menos de `n`, se devuelve lo que haya y el
    llamador lo registra: es preferible reportar 64 variables reales que
    inventar 71.
    """
    if X.shape[1] <= n:
        return X, list(X.columns)

    if method == "mutual_info":
        scores = mutual_info_classif(
            X.to_numpy(), y.to_numpy(), random_state=seed, discrete_features=False
        )
    else:
        scores = X.var(axis=0, ddof=0).to_numpy()

    order = np.argsort(scores)[::-1][:n]
    selected = [X.columns[i] for i in sorted(order)]
    return X[selected], selected
